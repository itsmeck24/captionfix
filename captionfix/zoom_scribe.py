"""Zoom's speech-to-text (Zoom AI Services, Scribe fast mode).

A lecture is cut at quiet points into pieces of about 4.5 minutes, each sent as a small MP3 in the
request body, so nothing has to be hosted anywhere. Each piece's response is cached, so a rerun
or a crash halfway through never pays for the same audio twice.

Needs ZOOM_API_KEY in .env (from zoom.ai, under the project's API keys)."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .asr import SR, quiet_cuts
from .captions import Word

URL = "https://api.zoom.us/v2/aiservices/scribe/transcribe"
MAX_PHRASES = 100  # Scribe's limit for inline vocabulary
PIECE = (240.0, 280.0)  # seconds: a 280 s piece at 32 kbps is ~1.1 MB, under the 2 MB body limit


def api_key(env: Path = Path(".env")) -> str:
    key = os.environ.get("ZOOM_API_KEY", "")
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ZOOM_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("ZOOM_API_KEY is not set. Put it in .env (see .env.example).")
    return key


def mp3_data_url(audio: np.ndarray) -> str:
    with tempfile.TemporaryDirectory() as d:
        wav, mp3 = Path(d) / "a.raw", Path(d) / "a.mp3"
        wav.write_bytes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", str(wav),
                        "-b:a", "32k", "-map_metadata", "-1", str(mp3)], check=True)
        return "data:audio/mpeg;base64," + base64.b64encode(mp3.read_bytes()).decode()


class Scribe:
    def __init__(self, key: str | None = None, model: str = "zoom-scribe-en"):
        self.key = key or api_key()
        self.model = model

    def request(self, audio: np.ndarray, phrases: list[str] | None = None) -> dict:
        config = {"language": "en-US", "model": self.model, "word_time_offsets": True, "diarization": False}
        if phrases:
            config["vocabulary"] = {"phrases": phrases[:MAX_PHRASES]}
        body = json.dumps({"file": mp3_data_url(audio), "config": config}).encode()
        for attempt in range(5):
            req = urllib.request.Request(URL, data=body, method="POST", headers={
                "x-api-key": self.key, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code in (429, 500, 502, 503) and attempt < 4:
                    time.sleep(2 ** attempt * 3)
                    continue
                raise RuntimeError(f"Scribe returned {e.code}: {detail}") from None
        raise RuntimeError("Scribe kept failing")

    def transcribe(self, audio: np.ndarray, cache: Path,
                   phrases: list[str] | Callable[[float, float], list[str]] | None = None) -> list[Word]:
        """Whole lecture, piece by piece. `cache` is a folder for the raw responses.

        `phrases` is one vocabulary for every piece, or a function of the piece's start and end
        (in seconds) that picks a vocabulary for that piece."""
        cache.mkdir(parents=True, exist_ok=True)
        words: list[Word] = []
        for k, (a, b) in enumerate(quiet_cuts(audio, *PIECE)):
            path = cache / f"{k:03d}.json"
            if path.exists():
                resp = json.loads(path.read_text())
            else:
                vocab = phrases(a / SR, b / SR) if callable(phrases) else phrases
                resp = self.request(audio[a:b], vocab)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(resp))
                os.replace(tmp, path)
            words += response_words(resp, a / SR)
        return words


def response_words(resp: dict, offset: float) -> list[Word]:
    """Scribe gives no per-word confidence, so prob is left at 1.0."""
    out = []
    for seg in resp.get("result", {}).get("segments", []):
        for w in seg.get("words") or []:
            text = w["word"].strip()
            if text:
                out.append(Word(text, round(w["start"] + offset, 3), round(w["end"] + offset, 3)))
    return out
