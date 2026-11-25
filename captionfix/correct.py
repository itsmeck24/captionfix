"""Fix misheard course terms, keeping a fix only when the audio agrees.

Fixes come from four places:
  - sound: 1 to 3 words that sound like a slide term ("I can value" -> "eigenvalue")
  - second: where the second transcript (Whisper, when Zoom wrote the draft) heard a slide term
    or a rare word that the draft does not have
  - spelling: words that join into a slide term exactly ("page rank" -> "PageRank")
  - loop: a stretch where the draft got stuck repeating one phrase, replaced by what the second
    transcript heard over the same seconds

Sound and second-opinion fixes are scored by Whisper against the ~25 s of audio around them and kept only
if the corrected text is a better match than the original by a set margin. Spelling fixes sound
the same by definition, and a loop is never what was said, so those two are not scored. Nothing
that comes from none of these sources is ever written."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, asdict

import jellyfish
import numpy as np
from wordfreq import zipf_frequency

from .asr import SR, quiet_cuts
from .captions import Word
from .slides import Term

PUNCT = ".,?!;:\"'()"
COMMON_ZIPF = 5.0  # "have", "going", "first": too common to be a misheard term on their own
RARE_ZIPF = 3.0
SCORED = ("sound", "second")  # sources checked against the audio


def plain(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def sound(text: str) -> str:
    """Metaphone code, loosened so that "I can value" and "eigenvalue" both give AKNFL:
    a leading vowel is just A, and soft and hard G (J and K) count as the same sound."""
    letters = re.sub(r"[^a-z]", "", text.lower())
    if not letters:
        return ""
    code = jellyfish.metaphone(letters).replace("J", "K")
    return "A" + code[1:] if code[:1] in "AEIOU" else code


def sound_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return 1 - jellyfish.levenshtein_distance(a, b) / max(len(a), len(b))


def same_stem(a: str, b: str) -> bool:
    """"optimization" vs "optimizations", "unroll" vs "unrolled": leave grammar to Whisper."""
    short, long = sorted((a, b), key=len)
    return long.startswith(short[: max(4, len(short) - 2)])


@dataclass
class Candidate:
    i: int  # first word replaced
    n: int  # number of words replaced
    term: str  # replacement text, punctuation included
    original: str
    similarity: float  # how alike they sound (1.0 for the other sources)
    start: float = 0.0
    end: float = 0.0
    source: str = "sound"  # "sound", "second", "spelling" or "loop"
    delta: float | None = None  # log P(audio says the fix) - log P(audio says the original)
    accepted: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _dress(words: list[Word], i: int, n: int, term: str) -> str:
    """Carry the replaced words' punctuation over, and capitalize at a sentence start."""
    first, last = words[i].text, words[i + n - 1].text
    lead = first[: len(first) - len(first.lstrip(PUNCT))]
    trail = last[len(last.rstrip(PUNCT)) :]
    sentence_start = i == 0 or words[i - 1].text.endswith((".", "?", "!"))
    if sentence_start and term[:1].islower():
        term = term[:1].upper() + term[1:]
    return lead + term + trail


def by_sound(words: list[Word], terms: list[Term], min_similarity: float = 0.8,
             max_words: int = 3) -> list[Candidate]:
    """Spans of 1-3 words that sound like a slide term but are not that term."""
    known = {plain(t.text) for t in terms}
    index = [(t, plain(t.text), sound(t.text)) for t in terms]
    index = [(t, p, s) for t, p, s in index if len(s) >= 2 and len(p) >= 4]
    keys = [plain(w.text) for w in words]
    out = []
    for i in range(len(words)):
        for n in range(1, max_words + 1):
            span = keys[i : i + n]
            if len(span) < n or not all(span) or any(k in known for k in span):
                continue
            if any(words[j].text.endswith((".", "?", "!")) for j in range(i, i + n - 1)):
                continue
            joined = "".join(span)
            s = sound(joined)
            common = n == 1 and zipf_frequency(span[0], "en") >= COMMON_ZIPF
            for term, p, ts in index:
                if p == joined:
                    if n > 1:
                        out.append(Candidate(i, n, _dress(words, i, n, term.text),
                                             " ".join(w.text for w in words[i : i + n]), 1.0,
                                             words[i].start, words[i + n - 1].end, "spelling"))
                    continue
                if not 0.6 <= len(joined) / len(p) <= 1.6:
                    continue
                if any(len(k) >= 4 and same_stem(p, k) for k in span + [joined]):
                    continue
                sim = sound_similarity(s, ts)
                # a very common word ("calls") is only a candidate when it sounds exactly like the term ("cols")
                if sim >= min_similarity and (not common or sim == 1.0):
                    out.append(Candidate(i, n, _dress(words, i, n, term.text),
                                         " ".join(w.text for w in words[i : i + n]), round(sim, 3),
                                         words[i].start, words[i + n - 1].end, "sound"))
    return out


def by_second(words: list[Word], second: list[Word], terms: list[Term], max_words: int = 4) -> list[Candidate]:
    """Places where the second transcript has a slide term or a rare word the draft lacks."""
    known = {plain(t.text) for t in terms}
    a = [plain(w.text) for w in words]
    b = [plain(w.text) for w in second]

    def notable(k: str) -> bool:
        return k in known or (k.isalpha() and len(k) >= 4 and zipf_frequency(k, "en") < RARE_ZIPF)

    out = []
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "replace" or i2 - i1 > max_words or j2 - j1 > max_words:
            continue
        new = [k for k in b[j1:j2] if notable(k)]
        if not new or any(k in a[i1:i2] for k in new):
            continue
        text = " ".join(w.text for w in second[j1:j2])
        out.append(Candidate(i1, i2 - i1, text, " ".join(w.text for w in words[i1:i2]), 1.0,
                             words[i1].start, words[i2 - 1].end, "second"))
    return out


def find_loops(words: list[Word], min_repeats: int = 3, min_words: int = 8) -> list[tuple[int, int]]:
    """Index ranges where one phrase of 1-10 words repeats back to back, as in a Whisper loop."""
    keys = [plain(w.text) for w in words]
    loops, i = [], 0
    while i < len(keys):
        best = None
        for k in range(1, 11):
            reps = 1
            while keys[i + reps * k : i + (reps + 1) * k] == keys[i : i + k]:
                reps += 1
            if reps >= min_repeats and reps * k >= min_words and (best is None or reps * k > best):
                best = reps * k
        if best:
            loops.append((i, i + best))
            i += best
        else:
            i += 1
    return loops


def by_loop(words: list[Word], second: list[Word]) -> list[Candidate]:
    out = []
    second_loops = {j for a, b in find_loops(second) for j in range(a, b)}
    for a, b in find_loops(words):
        t0, t1 = words[a].start, words[b - 1].end
        inside = [j for j, w in enumerate(second) if t0 <= (w.start + w.end) / 2 <= t1]
        if not inside or any(j in second_loops for j in inside):
            continue
        out.append(Candidate(a, b - a, " ".join(second[j].text for j in inside),
                             " ".join(w.text for w in words[a:b]), 1.0, t0, t1, "loop"))
    return out


def verify(words: list[Word], candidates: list[Candidate], audio: np.ndarray, scorer,
           prompt: str = "") -> None:
    """Score every candidate against the audio of the ~25 s piece it falls in (fills in `delta`)."""
    mids = np.array([(w.start + w.end) / 2 for w in words])
    for a, b in quiet_cuts(audio):
        t0, t1 = a / SR, b / SR
        idx = np.nonzero((mids >= t0) & (mids < t1))[0]
        if not len(idx):
            continue
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        inside = [c for c in candidates if lo <= c.i and c.i + c.n <= hi and c.source in SCORED]
        if not inside:
            continue
        base = [w.text for w in words[lo:hi]]
        texts = [" ".join(base)]
        for c in inside:
            k = c.i - lo
            texts.append(" ".join(base[:k] + [c.term] + base[k + c.n :]))
        scores = scorer.logprob(audio, t0, t1, texts, prompt=prompt)
        for c, s in zip(inside, scores[1:]):
            c.delta = round(s - scores[0], 3)


def apply(words: list[Word], candidates: list[Candidate], margin: float = 1.0) -> list[Word]:
    """Keep candidates whose audio score beats the original by `margin`, best first, no overlaps."""
    taken: set[int] = set()
    chosen = []
    # loops first (they are certain), then by how strongly the audio backs the fix
    rank = {"loop": 2, "spelling": 1}
    for c in sorted(candidates, key=lambda c: (-rank.get(c.source, 0), -(c.delta if c.delta is not None else -1e9))):
        span = set(range(c.i, c.i + c.n))
        backed = c.source not in SCORED or (c.delta is not None and c.delta > margin)
        if backed and not span & taken:
            c.accepted = True
            taken |= span
            chosen.append(c)
    starts = {c.i: c for c in chosen}
    out, i = [], 0
    while i < len(words):
        if i in starts:
            c = starts[i]
            span = words[i : i + c.n]
            tokens = c.term.split()
            a, b = span[0].start, span[-1].end
            step = (b - a) / len(tokens)
            prob = min(w.prob for w in span)
            for k, t in enumerate(tokens):
                out.append(Word(t, round(a + k * step, 3), round(a + (k + 1) * step, 3), prob,
                                ["corrected"], c.original if k == 0 else None))
            i += c.n
        else:
            w = words[i]
            out.append(Word(w.text, w.start, w.end, w.prob, list(w.flags), w.original))
            i += 1
    return out
