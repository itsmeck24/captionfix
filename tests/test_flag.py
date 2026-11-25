import numpy as np

from captionfix import flag
from captionfix.captions import Word


def ws(text, probs=None):
    probs = probs or [0.9] * len(text.split())
    return [Word(t, i, i + 0.5, p) for i, (t, p) in enumerate(zip(text.split(), probs))]


def test_disagreement_marks_words_the_other_transcript_lacks():
    a = ws("we use Cilk to spawn")
    b = ws("we use silk to spawn")
    assert flag.disagreement(a, b) == [False, False, True, False, False]


def test_features_shape_and_low_confidence():
    words = ws("one two three", [0.9, 0.05, 0.9])
    second = ws("one too three", [0.8, 0.5, 0.2])
    x = flag.features(words, second, [], margin=1.0)
    assert x.shape == (3, len(flag.FEATURES))
    assert x[1, 0] > x[0, 0] and x[0, 1] == x[1, 0]
    assert list(x[:, 2]) == [0.0, 1.0, 0.0]  # "two" is not in the second transcript
    assert x[2, 3] > x[0, 3] > 0 and x[1, 3] == 0.0  # the second engine's doubt about matched words


def test_regions_stay_within_budget_and_take_riskiest_first():
    words = [Word(str(i), i * 1.0, i * 1.0 + 0.5) for i in range(100)]
    risk = np.zeros(100)
    risk[[10, 11, 50]] = [0.9, 0.8, 0.7]
    spans = flag.regions(words, risk, budget=0.04, duration=100.0, pad=0.2, join=1.0)
    assert sum(b - a for a, b in spans) <= 4.0
    assert any(a <= 10.25 <= b for a, b in spans)


def test_fitted_model_round_trips(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    x = rng.random((200, len(flag.FEATURES)))
    y = (x[:, 0] > 0.5).astype(int)
    m = flag.RiskModel.fit(x, y)
    monkeypatch.setattr(flag, "HERE", tmp_path)
    m.save("zoom")
    m2 = flag.RiskModel.load("zoom")
    assert np.allclose(m.risk(x), m2.risk(x))
    assert m.weights[0] > 0
