import json

import numpy as np
import pytest

from captionfix import pipeline
from captionfix.captions import Word, read_cues
from captionfix.slides import Term

SAID = "today we use Cilk to spawn work and the scheduler steals it".split()
HEARD = "today we use silk to spawn work and the scheduler steals it".split()


def words(tokens, prob=0.95):
    return [Word(t, i * 0.5, i * 0.5 + 0.4, 0.3 if t == "silk" else prob) for i, t in enumerate(tokens)]


class Calls:
    n = 0


class FakeASR:
    def plain(self, audio):
        Calls.n += 1
        return words(HEARD)

    def chunked(self, audio, terms=None):
        Calls.n += 1
        return words(SAID)


class FakeScribe:
    def transcribe(self, audio, cache, phrases=None):
        Calls.n += 1
        return words(HEARD, prob=1.0)  # Zoom reports no confidence


class FakeScorer:
    def logprob(self, audio, start, end, texts, prompt=""):
        return [float(sum(t in SAID for t in text.split())) * 2 for text in texts]


def make(engine, monkeypatch):
    monkeypatch.setattr(pipeline, "load_audio", lambda p: np.zeros(16000 * 7, dtype=np.float32))
    monkeypatch.setattr(pipeline, "terms_from_pdf", lambda p: [Term("Cilk", 4, 1.0)])
    monkeypatch.setattr(pipeline, "slide_text", lambda p: ["Cilk spawn and the work-stealing scheduler"])
    p = pipeline.Pipeline(engine, pipeline.Settings(margin=1.0, min_similarity=0.8, budget=0.2))
    p.asr, p._scribe, p._scorer = FakeASR(), FakeScribe(), FakeScorer()
    Calls.n = 0
    return p


@pytest.mark.parametrize("engine", ["zoom", "whisper"])
def test_run_writes_captions_and_review_list(engine, monkeypatch, tmp_path):
    out = tmp_path / "out"
    summary = make(engine, monkeypatch).run(tmp_path / "lecture.m4a", tmp_path / "slides.pdf", out)
    text = " ".join(c.text.replace("\n", " ") for c in read_cues(out / "captions.vtt"))
    assert text.split() == SAID
    assert summary["corrections"] == 1 and summary["engine"] == engine
    review = json.loads((out / "review.json").read_text())
    assert review and sum(r["end"] - r["start"] for r in review) <= 0.2 * 7 + 1e-6


def test_rerun_after_a_crash_does_not_transcribe_or_pay_again(monkeypatch, tmp_path):
    out = tmp_path / "out"
    p = make("zoom", monkeypatch)
    monkeypatch.setattr(pipeline.correct, "verify", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        p.run(tmp_path / "lecture.m4a", tmp_path / "slides.pdf", out)
    assert Calls.n == 2 and not (out / "captions.vtt").exists()
    assert not list(out.glob("*.tmp"))
    monkeypatch.undo()
    p = make("zoom", monkeypatch)
    p.run(tmp_path / "lecture.m4a", tmp_path / "slides.pdf", out)
    assert Calls.n == 0  # Zoom and Whisper transcripts both came from the cache
    assert (out / "captions.vtt").exists()
