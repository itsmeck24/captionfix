"""Scoring a transcript against human captions: WER, rare and slide-term recall, and which
transcribed words are wrong (used to check the review list)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer
from wordfreq import zipf_frequency

from .captions import Word

_norm = EnglishTextNormalizer()
RARE_ZIPF = 3.0


def _tokens(text: str) -> list[str]:
    # the normalizer can leave a lone "." (e.g. from "...") that is not a word
    return [t for t in _norm(text).split() if any(c.isalnum() for c in t)]


def normalize(text: str) -> list[str]:
    return _tokens(text)


@lru_cache(maxsize=None)
def _norm_word(text: str) -> tuple[str, ...]:
    return tuple(_tokens(text))


def is_rare(token: str) -> bool:
    return token.isalpha() and len(token) >= 3 and zipf_frequency(token, "en") < RARE_ZIPF


@dataclass
class Score:
    wer: float
    ref_words: int
    substitutions: int
    deletions: int
    insertions: int
    rare_total: int
    rare_correct: int
    term_total: int
    term_correct: int
    term_false: int  # slide terms in the transcript where the captions have something else
    # course terms: rare words (zipf < 3) or slide terms. A mistake is a course term in the
    # captions that the transcript got wrong, or one the transcript wrote that was not said.
    course_total: int = 0
    course_correct: int = 0
    course_false: int = 0
    word_errors: list[bool] = field(repr=False, default_factory=list)  # per transcript word
    course_errors: list[bool] = field(repr=False, default_factory=list)  # per transcript word

    @property
    def course_mistakes(self) -> int:
        return self.course_total - self.course_correct + self.course_false

    @property
    def rare_recall(self) -> float:
        return self.rare_correct / self.rare_total if self.rare_total else float("nan")

    @property
    def term_recall(self) -> float:
        return self.term_correct / self.term_total if self.term_total else float("nan")

    def summary(self) -> dict:
        return {
            "wer": round(self.wer, 4), "ref_words": self.ref_words, "substitutions": self.substitutions,
            "deletions": self.deletions, "insertions": self.insertions,
            "rare_total": self.rare_total, "rare_correct": self.rare_correct,
            "term_total": self.term_total, "term_correct": self.term_correct, "term_false": self.term_false,
            "course_total": self.course_total, "course_correct": self.course_correct,
            "course_false": self.course_false,
        }


def score(hyp: list[Word], reference: str, term_keys: set[str]) -> Score:
    """Align the transcript's words with the reference captions after Whisper's English normalizer.

    Each transcript word is normalized on its own so every token maps back to the word (and its
    time) it came from."""
    ref = normalize(reference)
    tokens, owner = [], []
    for i, w in enumerate(hyp):
        for t in _norm_word(w.text):
            tokens.append(t)
            owner.append(i)
    terms = {t for k in term_keys for t in _norm_word(k)}

    out = jiwer.process_words(" ".join(ref), " ".join(tokens))
    def course(t: str) -> bool:
        return t in terms or is_rare(t)

    word_errors = [False] * len(hyp)
    course_errors = [False] * len(hyp)
    ref_ok = [False] * len(ref)
    term_false = course_false = 0
    for chunk in out.alignments[0]:
        if chunk.type == "equal":
            for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                ref_ok[chunk.ref_start_idx + k] = True
            continue
        hyp_js = list(range(chunk.hyp_start_idx, chunk.hyp_end_idx))
        if chunk.type == "delete" and owner:
            # a dropped word is charged to the transcript word just before it
            hyp_js = [min(max(chunk.hyp_start_idx - 1, 0), len(owner) - 1)]
        about_course = (any(course(t) for t in ref[chunk.ref_start_idx:chunk.ref_end_idx])
                        or any(course(tokens[j]) for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx)))
        for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
            term_false += tokens[j] in terms
            course_false += course(tokens[j])
        for j in hyp_js:
            word_errors[owner[j]] = True
            course_errors[owner[j]] |= about_course

    rare = [i for i, t in enumerate(ref) if is_rare(t)]
    in_terms = [i for i, t in enumerate(ref) if t in terms]
    in_course = [i for i, t in enumerate(ref) if course(t)]
    return Score(
        wer=out.wer, ref_words=len(ref), substitutions=out.substitutions, deletions=out.deletions,
        insertions=out.insertions,
        rare_total=len(rare), rare_correct=sum(ref_ok[i] for i in rare),
        term_total=len(in_terms), term_correct=sum(ref_ok[i] for i in in_terms),
        term_false=term_false, course_total=len(in_course),
        course_correct=sum(ref_ok[i] for i in in_course), course_false=course_false,
        word_errors=word_errors, course_errors=course_errors,
    )
