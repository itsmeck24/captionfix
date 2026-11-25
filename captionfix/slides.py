"""Course vocabulary from lecture slides: the rare words an instructor is likely to say."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from wordfreq import zipf_frequency

TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+#'_.-]*[A-Za-z0-9+#]|[A-Za-z]")
# Rare enough that a general speech model may not expect it. Zipf 3 is about once per million words.
RARE_ZIPF = 3.0
SKIP = {"mit", "opencourseware", "ocw", "creative", "commons", "licensed", "license"}


@dataclass(frozen=True)
class Term:
    text: str  # as written on the slides, e.g. "Cilk" or "SIMD"
    count: int  # times it appears in the slides
    zipf: float  # how common the word is in general English

    @property
    def key(self) -> str:
        return self.text.lower()


def slide_text(path: str | Path) -> list[str]:
    """Text of each page of a PDF."""
    with pymupdf.open(path) as doc:
        return [page.get_text() for page in doc]


def is_speakable(tok: str) -> bool:
    """Something a person could say: no code identifiers, URLs or diagram labels like "B9B"."""
    if len(tok) < 3 or len(tok) > 24:
        return False
    if any(c in tok for c in "_.@/\\") or tok.isdigit():
        return False
    core = tok[:-2] if tok.endswith("++") else tok
    if "+" in core or "#" in core:
        return False
    # letters, then at most two digits at the end: x86, AVX2
    return re.fullmatch(r"[A-Za-z][A-Za-z'-]*\d{0,2}", core) is not None


def extract_terms(pages: list[str], rare_zipf: float = RARE_ZIPF) -> list[Term]:
    """Rare, speakable words from the slides, most frequent first."""
    counts: Counter[str] = Counter()
    forms: dict[str, Counter[str]] = {}
    for page in pages:
        for tok in TOKEN.findall(page):
            tok = tok.strip("'-")
            if not is_speakable(tok):
                continue
            key = tok.lower()
            counts[key] += 1
            forms.setdefault(key, Counter())[tok] += 1

    terms = []
    for key, n in counts.items():
        if key in SKIP:
            continue
        z = zipf_frequency(key, "en")
        written = forms[key].most_common(1)[0][0]
        is_acronym = written.isupper() and len(written) <= 5
        if (z < rare_zipf and not written.isupper()) or (is_acronym and n >= 2 and z < 4.0):
            terms.append(Term(written, n, z))
    terms.sort(key=lambda t: (-t.count, t.zipf))
    return terms


def terms_from_pdf(path: str | Path) -> list[Term]:
    return extract_terms(slide_text(path))
