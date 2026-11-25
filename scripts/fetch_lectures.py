"""Download the evaluation lectures: audio (from the archive.org video), human captions and slides.

Videos are converted to 16 kHz mono FLAC and deleted. Everything lands in data/lectures/<id>/,
which git ignores. Safe to rerun: finished files are skipped."""

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "captionfix-eval/0.1 (research; contact via GitHub)"}


def get(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)


def slides_url(url: str) -> str:
    if url.endswith(".pdf"):
        return url
    html = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode()
    links = re.findall(r'href="([^"]+\.pdf)"', html)
    if not links:
        raise RuntimeError(f"no PDF on {url}")
    return links[0] if links[0].startswith("http") else "https://ocw.mit.edu" + links[0]


def fetch(lec: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    base = f"https://archive.org/download/{lec['archive']}/{lec['stem']}"
    get(base + ".srt", out / "reference.srt")
    get(slides_url(lec["slides"]), out / "slides.pdf")
    audio = out / "audio.flac"
    if not audio.exists():
        video = out / "video.mp4"
        get(base + ".mp4", video)
        tmp = out / "audio.part.flac"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vn", "-ac", "1",
                        "-ar", "16000", str(tmp)], check=True)
        tmp.rename(audio)
        video.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="lecture ids")
    args = ap.parse_args()
    lectures = json.loads((ROOT / "data/lectures.json").read_text())["lectures"]
    for lec in lectures:
        if args.only and lec["id"] not in args.only:
            continue
        print(lec["id"], flush=True)
        try:
            fetch(lec, ROOT / "data/lectures" / lec["id"])
        except Exception as e:  # keep going, report at the end
            print(f"  failed: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
