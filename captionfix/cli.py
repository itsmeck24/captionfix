"""captionfix run   recording [--slides deck.pdf] [--engine zoom|whisper] --out folder
captionfix review folder [--port 8000]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(prog="captionfix", description="Accurate captions for recorded lectures.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="caption a recording")
    run.add_argument("recording", type=Path, help="audio or video file (Zoom's .mp4 or .m4a works)")
    run.add_argument("--slides", type=Path, help="the lecture's slides as a PDF")
    run.add_argument("--engine", choices=["zoom", "whisper"], default="zoom",
                     help="zoom: Zoom Scribe writes the draft, Whisper checks it (needs ZOOM_API_KEY); "
                          "whisper: everything local")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--budget", type=float, help="share of the lecture to send to review (default 0.10)")

    review = sub.add_parser("review", help="check the flagged stretches in a browser")
    review.add_argument("folder", type=Path)
    review.add_argument("--recording", type=Path, help="defaults to the one the run used")
    review.add_argument("--port", type=int, default=8000)

    args = ap.parse_args()
    if args.cmd == "run":
        from .pipeline import Pipeline, Settings

        settings = Settings.load(args.engine)
        if args.budget is not None:
            settings.budget = args.budget
        summary = Pipeline(args.engine, settings).run(args.recording, args.slides, args.out)
        print(json.dumps(summary, indent=2))
        print(f"\ncaptions: {args.out / 'captions.vtt'}\nreview:   captionfix review {args.out}")
    else:
        import uvicorn

        from .review import make_app

        print(f"open http://localhost:{args.port}")
        uvicorn.run(make_app(args.folder, args.recording), port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
