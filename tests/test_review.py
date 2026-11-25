import json

from fastapi.testclient import TestClient

from captionfix.captions import Word, read_cues
from captionfix.review import apply_edits, make_app


def setup(tmp_path):
    words = [Word("we", 0, 0.4), Word("use", 0.5, 0.9), Word("silk", 1.0, 1.4, 0.2), Word("today.", 1.5, 2.0)]
    (tmp_path / "words.json").write_text(json.dumps([w.to_dict() for w in words]))
    (tmp_path / "review.json").write_text(json.dumps([{"start": 0.9, "end": 1.5, "text": "silk"}]))
    (tmp_path / "run.json").write_text(json.dumps({"audio": str(tmp_path / "a.wav"), "minutes": 0.1,
                                                   "review_minutes": 0.01, "corrections": 0}))
    (tmp_path / "a.wav").write_bytes(b"RIFF")
    return words


def test_edit_replaces_only_that_stretch(tmp_path):
    words = setup(tmp_path)
    out = apply_edits(words, [{"start": 0.9, "end": 1.5}], {"0": {"text": "Cilk", "status": "edited"}})
    assert [w.text for w in out] == ["we", "use", "Cilk", "today."]


def test_review_flow(tmp_path):
    setup(tmp_path)
    client = TestClient(make_app(tmp_path))
    data = client.get("/api/review").json()
    assert data["stretches"][0]["words"][0]["text"] == "silk"
    assert client.put("/api/review/0", json={"text": "Cilk", "status": "edited"}).status_code == 200
    assert client.put("/api/review/5", json={"text": "x", "status": "edited"}).status_code == 400
    client.post("/api/export")
    text = " ".join(c.text for c in read_cues(tmp_path / "captions.reviewed.vtt"))
    assert text == "we use Cilk today."
