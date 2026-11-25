from captionfix.captions import Word
from captionfix.slides import Term
from captionfix.sync import SlideIndex

PAGES = [
    "Bit hacks\nSwap two integers with XOR, no temporary",
    "Branchless minimum\nThe compiler emits a CMOV instead of a branch",
    "Popcount\nCount the ones with a lookup table; the builtin popcount instruction",
]
TERMS = [Term("XOR", 3, 2.5), Term("CMOV", 2, 1.0), Term("Popcount", 2, 1.0), Term("NeuralNetworks", 1, 0.0)]


def said(text, start=0.0):
    return [Word(t, start + i * 0.4, start + i * 0.4 + 0.3) for i, t in enumerate(text.split())]


def test_terms_come_from_the_slide_being_discussed():
    index = SlideIndex(PAGES, TERMS)
    heard = said("so the compiler can emit a conditional move instead of a branch for the minimum")
    assert index.phrases(heard, 0, 10, k=1) == ["CMOV"]


def test_nothing_matched_gives_no_vocabulary():
    index = SlideIndex(PAGES, TERMS)
    assert index.phrases(said("good morning everyone"), 0, 10, k=2) == []


def test_line_broken_and_run_together_words_are_not_terms():
    pages = ["does not auto\xad\nmatically NeuralNetworks", "other slide"]
    index = SlideIndex(pages, [Term("matically", 1, 0.0), Term("NeuralNetworks", 1, 0.0)])
    assert index.page_terms[0] == []
