"""A local page for checking the flagged stretches: play each one, fix the text, export the captions."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .captions import Word, words_to_cues, write_vtt
from .pipeline import save_json

STATIC = Path(__file__).with_name("static")


class Edit(BaseModel):
    text: str
    status: str  # "ok" (left as is) or "edited"


def apply_edits(words: list[Word], stretches: list[dict], edits: dict[str, dict]) -> list[Word]:
    """Swap each edited stretch's words for the reviewer's text, spread over the stretch's time."""
    out, used = [], set()
    for w in words:
        mid = (w.start + w.end) / 2
        hit = next((i for i, s in enumerate(stretches) if s["start"] <= mid <= s["end"]), None)
        edit = edits.get(str(hit)) if hit is not None else None
        if edit is None or edit["status"] != "edited":
            out.append(w)
            continue
        if hit in used:
            continue
        used.add(hit)
        inside = [x for x in words if stretches[hit]["start"] <= (x.start + x.end) / 2 <= stretches[hit]["end"]]
        tokens = edit["text"].split()
        if not tokens:
            continue
        a, b = inside[0].start, inside[-1].end
        step = (b - a) / len(tokens)
        out += [Word(t, round(a + k * step, 3), round(a + (k + 1) * step, 3), 1.0, ["reviewed"])
                for k, t in enumerate(tokens)]
    return out


def make_app(folder: Path, recording: Path | None = None) -> FastAPI:
    folder = Path(folder)
    run = json.loads((folder / "run.json").read_text())
    recording = Path(recording or run["audio"])
    edits_path = folder / "edits.json"

    def load():
        words = [Word.from_dict(d) for d in json.loads((folder / "words.json").read_text())]
        stretches = json.loads((folder / "review.json").read_text())
        edits = json.loads(edits_path.read_text()) if edits_path.exists() else {}
        return words, stretches, edits

    app = FastAPI(title="CaptionFix review")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "review.html")

    @app.get("/recording")
    def audio():
        return FileResponse(recording)

    @app.get("/api/review")
    def review():
        words, stretches, edits = load()
        doubts_path = folder / "doubts.json"
        heard = {d["start"]: d["heard"] for d in json.loads(doubts_path.read_text())} if doubts_path.exists() else {}
        items = []
        for i, s in enumerate(stretches):
            inside = [{**w.to_dict(), "heard": heard.get(w.start)} for w in words
                      if s["start"] <= (w.start + w.end) / 2 <= s["end"]]
            items.append({**s, "id": i, "words": inside, "edit": edits.get(str(i))})
        return {"run": run, "stretches": items}

    @app.put("/api/review/{i}")
    def save(i: int, edit: Edit):
        _, stretches, edits = load()
        if not 0 <= i < len(stretches) or edit.status not in ("ok", "edited"):
            raise HTTPException(400)
        edits[str(i)] = edit.model_dump()
        save_json(edits_path, edits)
        return {"saved": i}

    @app.post("/api/export")
    def export():
        words, stretches, edits = load()
        final = apply_edits(words, stretches, edits)
        write_vtt(words_to_cues(final), folder / "captions.reviewed.vtt")
        return {"file": str(folder / "captions.reviewed.vtt"), "checked": len(edits), "total": len(stretches)}

    return app
