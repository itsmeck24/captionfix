"""Which stretches of the captions a person should check, and which words to mark as doubtful.

The review list ranks words the two engines disagree on, course terms first: a course term is a
slide term or a rare word, in the draft or heard by the second engine at the same moment. The
riskiest words are grouped into stretches until the review budget (a share of the lecture's
length) is used up. A small logistic model over more signals (confidence, close calls) is kept
for comparison; on the dev and test lectures it did no better than disagreement alone."""

from __future__ import annotations

import difflib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .captions import Word
from .correct import Candidate, plain
from .metrics import _norm_word, is_rare

HERE = Path(__file__).parent
FEATURES = ["neg_log_prob", "min_neighbour_neg_log_prob", "disagrees", "second_neg_log_prob",
            "close_call", "corrected", "short_word"]


def model_path(engine: str) -> Path:
    return HERE / f"flag_model_{engine}.json"


def align(primary: list[Word], second: list[Word]) -> list[int | None]:
    """For each word in `primary`, the index of the same word in `second`, or None."""
    a = [plain(w.text) for w in primary]
    b = [plain(w.text) for w in second]
    out: list[int | None] = [None] * len(a)
    for block in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(block.size):
            out[block.a + k] = block.b + k
    return out


def disagreement(primary: list[Word], second: list[Word]) -> list[bool]:
    """True for words in `primary` that the second transcript does not have at that point."""
    return [j is None for j in align(primary, second)]


def features(words: list[Word], second: list[Word], candidates: list[Candidate], margin: float,
             band: float = 2.0) -> np.ndarray:
    close = [False] * len(words)
    for c in candidates:
        if c.delta is not None and abs(c.delta - margin) < band:
            for i, w in enumerate(words):
                if w.start < c.end and w.end > c.start:
                    close[i] = True
    matched = align(words, second)
    nlp = [-math.log(max(w.prob, 1e-4)) for w in words]
    rows = []
    for i, w in enumerate(words):
        neighbours = nlp[max(0, i - 2) : i] + nlp[i + 1 : i + 3]
        j = matched[i]
        rows.append([
            nlp[i],
            max(neighbours) if neighbours else 0.0,
            float(j is None),
            -math.log(max(second[j].prob, 1e-4)) if j is not None else 0.0,
            float(close[i]),
            float("corrected" in w.flags),
            float(len(plain(w.text)) <= 2),
        ])
    return np.array(rows, dtype=float).reshape(-1, len(FEATURES))


def is_course(w: Word, term_keys: set[str]) -> bool:
    return any(t in term_keys or is_rare(t) for t in _norm_word(w.text))


def course_nearby(words: list[Word], second: list[Word], term_keys: set[str], pad: float = 0.3) -> np.ndarray:
    """1 where the draft word, or a second-opinion word heard at the same time, is a course term."""
    marks = [(w.start - pad, w.end + pad) for w in second if is_course(w, term_keys)]
    out = np.array([is_course(w, term_keys) for w in words], dtype=float)
    k = 0
    for i, w in enumerate(words):
        while k < len(marks) and marks[k][1] < w.start:
            k += 1
        if k < len(marks) and marks[k][0] <= w.end:
            out[i] = 1.0
    return out


def course_first(x: np.ndarray, course: np.ndarray) -> np.ndarray:
    """Disagreements first, disagreements about a course term before the rest, then the second
    engine's doubt as a tie-break."""
    disagrees = x[:, FEATURES.index("disagrees")]
    return disagrees * (1 + course) + 0.01 * x[:, FEATURES.index("second_neg_log_prob")]


def doubts(words: list[Word], second: list[Word], term_keys: set[str]) -> list[tuple[int, str]]:
    """Course terms in the captions that the second engine heard differently: (word index, what
    it heard instead). These are the words a student sees marked."""
    matched = align(words, second)
    out = []
    for i, w in enumerate(words):
        if matched[i] is None and is_course(w, term_keys):
            alt = " ".join(v.text for v in second if v.start < w.end + 0.2 and v.end > w.start - 0.2)
            if plain(alt) != plain(w.text):
                out.append((i, alt))
    return out


@dataclass
class RiskModel:
    weights: list[float]
    bias: float

    def risk(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(x @ np.array(self.weights) + self.bias)))

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> "RiskModel":
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(x, y)
        return cls([round(float(v), 4) for v in m.coef_[0]], round(float(m.intercept_[0]), 4))

    def save(self, engine: str) -> None:
        model_path(engine).write_text(json.dumps(
            {"features": FEATURES, "weights": self.weights, "bias": self.bias}, indent=2) + "\n")

    @classmethod
    def load(cls, engine: str) -> "RiskModel":
        path = model_path(engine)
        if not path.exists():
            # before fitting: low confidence and disagreement
            return cls([1.0, 0.3, 1.5, 0.5, 1.0, 0.5, 0.0], -3.0)
        d = json.loads(path.read_text())
        assert d["features"] == FEATURES, f"{path.name} was fitted on different features"
        return cls(d["weights"], d["bias"])


def regions(words: list[Word], risk: np.ndarray, budget: float, duration: float,
            pad: float = 0.4, join: float = 1.0) -> list[tuple[float, float]]:
    """Take words from riskiest down, growing review stretches, until they cover `budget` of `duration`."""
    spans: list[list[float]] = []
    for i in np.argsort(-risk):
        w = words[i]
        spans.append([max(0.0, w.start - pad), w.end + pad])
        merged = _merge(spans, join)
        if sum(b - a for a, b in merged) > budget * duration:
            spans.pop()
            break
    return [(round(a, 2), round(b, 2)) for a, b in _merge(spans, join)]


def _merge(spans: list[list[float]], join: float) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= join:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def mark(words: list[Word], spans: list[tuple[float, float]]) -> None:
    for w in words:
        mid = (w.start + w.end) / 2
        if any(a <= mid <= b for a, b in spans) and "review" not in w.flags:
            w.flags.append("review")
