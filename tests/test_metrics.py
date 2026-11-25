from captionfix.captions import Word
from captionfix.metrics import score


def ws(text):
    return [Word(t, i, i + 0.5) for i, t in enumerate(text.split())]


def test_perfect_transcript():
    s = score(ws("We use Cilk for parallel loops."), "we use cilk for parallel loops", {"cilk"})
    assert s.wer == 0 and s.term_correct == s.term_total == 1 and not any(s.word_errors)


def test_counts_term_misses_and_false_terms():
    s = score(ws("we use silk for Cilk loops"), "we use cilk for parallel loops", {"cilk"})
    assert s.term_total == 1 and s.term_correct == 0
    assert s.term_false == 1  # "Cilk" written where the captions say "parallel"
    assert s.word_errors == [False, False, True, False, True, False]


def test_deletion_is_charged_to_previous_word():
    s = score(ws("we use for loops"), "we use cilk for loops", set())
    assert s.deletions == 1 and s.word_errors == [False, True, False, False]


def test_rare_words_come_from_the_reference_not_the_slides():
    s = score(ws("the memoization table"), "the memoization table", set())
    assert s.rare_total == 1 and s.rare_correct == 1
