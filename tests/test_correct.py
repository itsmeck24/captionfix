import numpy as np

from captionfix import correct
from captionfix.captions import Word
from captionfix.slides import Term

TERMS = [Term("Cilk", 5, 1.0), Term("eigenvalue", 3, 2.0), Term("memoization", 2, 1.5)]


def words(text: str) -> list[Word]:
    return [Word(t, i * 0.5, i * 0.5 + 0.4, 0.9) for i, t in enumerate(text.split())]


class FakeScorer:
    """Pretends the audio says `truth`: scores a text by how many of truth's words it contains."""

    def __init__(self, truth: str):
        self.truth = set(truth.lower().replace(".", "").split())

    def logprob(self, audio, start, end, texts, prompt=""):
        return [float(sum(t.strip(".,").lower() in self.truth for t in text.split()) * 3
                      - len(text.split())) for text in texts]


def test_finds_sound_alike_spans():
    ws = words("we use silk to spawn and the I can value of the matrix")
    found = {(c.original, c.term) for c in correct.by_sound(ws, TERMS, 0.75)}
    assert ("silk", "Cilk") in found
    assert ("I can value", "eigenvalue") in found


def test_never_touches_words_already_on_the_slides():
    ws = words("Cilk is fine here")
    assert correct.by_sound(ws, TERMS, 0.5) == []


def test_keeps_only_what_the_audio_supports():
    ws = words("we use silk to spawn and the I can value of the matrix")
    cands = correct.by_sound(ws, TERMS, 0.75)
    audio = np.zeros(16000 * 10, dtype=np.float32)
    correct.verify(ws, cands, audio, FakeScorer("we use Cilk to spawn and the I can value of the matrix"))
    out = correct.apply(ws, cands, margin=0.5)
    text = " ".join(w.text for w in out)
    assert "Cilk" in text and "I can value" in text  # the audio "said" I can value here
    fixed = next(w for w in out if w.text == "Cilk")
    assert fixed.original == "silk" and fixed.flags == ["corrected"]


def test_correction_keeps_punctuation_and_capital():
    ws = words("Silk, then more")
    cands = correct.by_sound(ws, TERMS, 0.75)
    for c in cands:
        c.delta = 5.0
    out = correct.apply(ws, cands, margin=1.0)
    assert out[0].text == "Cilk,"


def test_overlapping_candidates_take_the_best_one():
    ws = words("the I can value here")
    cands = [correct.Candidate(1, 3, "eigenvalue", "I can value", 0.9, delta=4.0),
             correct.Candidate(2, 2, "eigenvalue", "can value", 0.8, delta=2.0)]
    out = correct.apply(ws, cands, margin=1.0)
    assert [w.text for w in out] == ["the", "eigenvalue", "here"]
    assert out[1].start == ws[1].start and out[1].end == ws[3].end


def test_nothing_changes_below_margin():
    ws = words("we use silk")
    cands = [correct.Candidate(2, 1, "Cilk", "silk", 1.0, delta=0.4)]
    assert [w.text for w in correct.apply(ws, cands, margin=1.0)] == ["we", "use", "silk"]


def test_common_words_and_grammar_are_left_alone():
    ws = words("I have to unroll the optimization")
    terms = [Term("HEIF", 2, 1.0), Term("unrolled", 2, 2.0), Term("optimizations", 3, 2.5)]
    found = correct.by_sound(ws, terms, 0.6)
    assert not [c for c in found if c.term in ("unrolled", "optimizations")]
    assert not [c for c in correct.by_sound(words("we are going there"), [Term("Caching", 3, 2.0)], 0.6)]


def test_second_transcript_suggests_rare_words_it_heard():
    main = words("we spawn with silk plus and then sync")
    second = words("we spawn with Cilk Plus and then sync")
    cands = correct.by_second(main, second, TERMS)
    assert [(c.original, c.term, c.source) for c in cands] == [("silk", "Cilk", "second")]


def test_second_transcript_ignores_ordinary_differences():
    main = words("so we go to the store")
    second = words("so we went to a store")
    assert correct.by_second(main, second, TERMS) == []


def test_multi_word_fix_spreads_over_the_old_words_time():
    ws = words("we spawn with silk plus now")
    cands = [correct.Candidate(3, 2, "Cilk Plus", "silk plus", 1.0, source="second", delta=3.0)]
    out = correct.apply(ws, cands, margin=1.0)
    assert [w.text for w in out] == ["we", "spawn", "with", "Cilk", "Plus", "now"]
    assert out[3].start == ws[3].start and out[4].end == ws[4].end and out[3].original == "silk plus"


def test_spelling_follows_the_slides():
    ws = words("rank pages with page rank today")
    cands = [c for c in correct.by_sound(ws, [Term("PageRank", 3, 1.0)], 0.8) if c.source == "spelling"]
    assert [(c.original, c.term) for c in cands] == [("page rank", "PageRank")]
    out = correct.apply(ws, cands, margin=100.0)  # not scored, so the margin does not apply
    assert [w.text for w in out] == ["rank", "pages", "with", "PageRank", "today"]


def test_common_word_needs_an_exact_sound_match():
    ws = words("the row and calls of the matrix")
    found = {(c.original, c.term) for c in correct.by_sound(ws, [Term("cols", 4, 2.0)], 0.7)}
    assert ("calls", "cols") in found


def test_loop_is_found_and_replaced_by_the_second_transcript():
    loop = "how will we help out " * 4
    ws = words("so the question is " + loop + "and then")
    second = words("so the question is how will we help out the internist who is understaffed and then")
    assert correct.find_loops(ws) == [(4, 24)]
    cands = correct.by_loop(ws, second)
    assert len(cands) == 1 and cands[0].source == "loop"
    out = correct.apply(ws, cands, margin=1.0)
    assert "internist" in " ".join(w.text for w in out)
    assert " ".join(w.text for w in out).count("help out") == 1


def test_real_repetition_is_not_a_loop():
    assert correct.find_loops(words("no no no I said no")) == []
