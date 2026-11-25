"""Words with times, reading and writing SRT/WebVTT, and splitting words into caption cues."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

TIME = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})")
# Zoom transcripts start each cue with "Speaker Name: ". OCW captions use "PROFESSOR:" or "AUDIENCE:".
SPEAKER = re.compile(r"^\s*(?:[A-Z][\w.'-]*(?: [A-Z][\w.'-]*){1,3}|[A-Z][A-Z .'-]+):\s+")
NON_SPEECH = re.compile(r"\[[^\]]*\]|\([A-Z ]+\)")


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0
    flags: list[str] = field(default_factory=list)
    original: str | None = None  # set when a correction replaced this word

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d["flags"]:
            del d["flags"]
        if d["original"] is None:
            del d["original"]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Word":
        return cls(**d)


@dataclass
class Cue:
    start: float
    end: float
    text: str


def parse_time(s: str) -> float:
    m = TIME.search(s)
    if not m:
        raise ValueError(f"not a timestamp: {s!r}")
    h, mnt, sec, ms = m.groups()
    return int(h or 0) * 3600 + int(mnt) * 60 + int(sec) + int(ms) / 1000


def fmt_time(t: float) -> str:
    ms = round(max(t, 0) * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def read_cues(path: str | Path) -> list[Cue]:
    """Read an .srt or .vtt file (including Zoom's transcript export) into cues."""
    text = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    cues = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln for ln in block.strip().split("\n") if ln.strip()]
        for i, ln in enumerate(lines):
            if "-->" in ln:
                a, b = ln.split("-->")
                body = " ".join(lines[i + 1 :])
                body = re.sub(r"<[^>]+>", "", body)
                cues.append(Cue(parse_time(a), parse_time(b), body.strip()))
                break
    return cues


def clean_reference(text: str) -> str:
    """Drop speaker labels and bracketed sound notes like [LAUGHTER] from caption text."""
    text = SPEAKER.sub("", text)
    return NON_SPEECH.sub(" ", text).strip()


def cues_to_words(cues: list[Cue]) -> list[Word]:
    """Spread each cue's words evenly over its time span (for captions without word times)."""
    words = []
    for cue in cues:
        toks = clean_reference(cue.text).split()
        if not toks:
            continue
        step = (cue.end - cue.start) / len(toks)
        for k, tok in enumerate(toks):
            words.append(Word(tok, cue.start + k * step, cue.start + (k + 1) * step))
    return words


def words_to_cues(
    words: list[Word],
    max_chars: int = 42,
    max_lines: int = 2,
    max_duration: float = 6.0,
    max_gap: float = 1.5,
) -> list[Cue]:
    """Group words into cues of at most two 42-character lines, breaking at pauses and sentence ends."""
    cues: list[Cue] = []
    cur: list[Word] = []

    def flush():
        if cur:
            cues.append(Cue(cur[0].start, cur[-1].end, _wrap([w.text for w in cur], max_chars)))
            cur.clear()

    for w in words:
        if cur:
            candidate = " ".join(x.text for x in cur + [w])
            too_long = len(_wrap(candidate.split(), max_chars).split("\n")) > max_lines
            if too_long or w.start - cur[-1].end > max_gap or w.end - cur[0].start > max_duration:
                flush()
        cur.append(w)
        if w.text.endswith((".", "?", "!")) and sum(len(x.text) + 1 for x in cur) > max_chars:
            flush()
    flush()
    # captions should stay up long enough to read
    for a, b in zip(cues, cues[1:]):
        a.end = min(max(a.end, a.start + 1.0), b.start)
    if cues:
        cues[-1].end = max(cues[-1].end, cues[-1].start + 1.0)
    return cues


def _wrap(tokens: list[str], width: int) -> str:
    lines, line = [], ""
    for t in tokens:
        if line and len(line) + 1 + len(t) > width:
            lines.append(line)
            line = t
        else:
            line = f"{line} {t}".strip()
    if line:
        lines.append(line)
    return "\n".join(lines)


def write_vtt(cues: list[Cue], path: str | Path) -> None:
    out = ["WEBVTT", ""]
    for c in cues:
        out += [f"{fmt_time(c.start)} --> {fmt_time(c.end)}", c.text, ""]
    Path(path).write_text("\n".join(out), encoding="utf-8")
