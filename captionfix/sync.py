"""Slide-synced vocabulary: give Zoom only the terms of the slides being talked about.

Handing Scribe every rare word in the deck (up to 100) made it write course terms that were
never said. A lecturer talks about the slide on screen, so for each piece of audio we find the
slides whose words match a quick local transcript of that piece and pass only their terms."""

from __future__ import annotations

import math
import re
from collections import Counter

from wordfreq import zipf_frequency

from .captions import Word
from .slides import TOKEN, Term

COMMON_ZIPF = 5.5  # words like "the", "and" say nothing about which slide is up
PAD = 30.0  # seconds of transcript on each side of a piece, for slides shown across a cut


def _content(text: str) -> set[str]:
    out = set()
    for tok in TOKEN.findall(text):
        tok = tok.strip("'-").lower()
        if len(tok) >= 3 and re.fullmatch(r"[a-z][a-z'-]*\d{0,2}", tok) and zipf_frequency(tok, "en") < COMMON_ZIPF:
            out.add(tok)
    return out


def _speakable(term: str) -> bool:
    # "NeuralNetworks" is two words run together by the PDF; Zoom would only learn to write it
    return re.search(r"[a-z][A-Z]", term) is None


class SlideIndex:
    def __init__(self, pages: list[str], terms: list[Term]):
        # join words the PDF broke across lines ("auto-\nmatically") so the halves are not terms
        pages = [re.sub(r"(\w)[-\xad]\s*\n\s*(\w)", r"\1\2", p) for p in pages]
        self.pages = [_content(p) for p in pages]
        df = Counter(w for p in self.pages for w in p)
        n = len(self.pages)
        self.idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}
        keys = {t.key: t for t in terms if _speakable(t.text)}
        self.page_terms = [[keys[w] for w in p if w in keys] for p in self.pages]
        self.order = {t.key: i for i, t in enumerate(terms)}

    def match(self, text: str) -> list[float]:
        heard = _content(text)
        return [sum(self.idf[w] for w in p & heard) for p in self.pages]

    def phrases(self, words: list[Word], start: float, end: float, k: int) -> list[str]:
        """Terms of the k slides that best match what was said between start and end."""
        text = " ".join(w.text for w in words if start - PAD <= w.start <= end + PAD)
        scores = self.match(text)
        best = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: -scores[i])[:k]
        terms = {t.key: t for i in best for t in self.page_terms[i]}
        return [t.text for t in sorted(terms.values(), key=lambda t: self.order[t.key])]
