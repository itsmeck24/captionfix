"""Evaluate CaptionFix on the OCW lectures against MIT's human captions.

    uv run python scripts/evaluate.py tune --engine zoom      # dev lectures: pick the vocabulary and settings,
    uv run python scripts/evaluate.py tune --engine whisper   #   fit the review model
    uv run python scripts/evaluate.py test                    # test lectures: score once, both engines
    uv run python scripts/evaluate.py fresh                   # lectures from a course never looked at
    uv run python scripts/evaluate.py doubts                  # how good the doubt marks are

Transcripts, Zoom responses and audio scores are cached in runs/<lecture>/, so reruns only redo
the scoring and never pay Zoom twice."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from captionfix import flag  # noqa: E402
from captionfix.asr import SR, load_audio  # noqa: E402
from captionfix.captions import clean_reference, read_cues  # noqa: E402
from captionfix.metrics import score  # noqa: E402
from captionfix.pipeline import ENGINES, SECOND, Pipeline, Settings, save_json, settings_path  # noqa: E402
from captionfix.slides import terms_from_pdf  # noqa: E402

LECTURES = json.loads((ROOT / "data/lectures.json").read_text())["lectures"]
RESULTS = ROOT / "results"
TRANSCRIPTS = ("zoom", "zoom_vocab", "whisper", "whisper_primed")
SYNC_SLIDES = [2, 4]  # slides per piece tried on the dev lectures
MARGINS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
SIMILARITIES = [0.7, 0.75, 0.8, 0.85, 0.9]
ALL = ("sound", "second", "spelling", "loop")
SOURCES = [ALL] + [tuple(s for s in ALL if s != drop) for drop in ALL] + [("second",), ("sound",)]
BUDGETS = [0.05, 0.10, 0.15, 0.20]


class Lecture:
    def __init__(self, meta: dict, pipes: dict[str, Pipeline], extra: tuple[str, ...] = ()):
        self.meta, self.id, self.pipes = meta, meta["id"], pipes
        src = ROOT / "data/lectures" / self.id
        self.out = ROOT / "runs" / self.id
        self.audio_path, self.slides = src / "audio.flac", src / "slides.pdf"
        self.reference = " ".join(clean_reference(c.text) for c in read_cues(src / "reference.srt"))
        self.terms = terms_from_pdf(self.slides)
        self.term_keys = {t.key for t in self.terms}
        self.hyps = pipes["zoom"].transcripts(self.audio_path, self.slides, self.out, need=TRANSCRIPTS + extra)
        self.audio = load_audio(str(self.audio_path))
        self.minutes = round(len(self.audio) / SR / 60, 1)
        self._cands = {}
        self.timing = {}

    def candidates(self, engine: str, with_prompt: bool):
        key = (engine, with_prompt)
        if key not in self._cands:
            pipe = self.pipes[engine]
            pipe.settings = replace(pipe.settings, score_with_prompt=with_prompt)
            path = self.out / f"candidates_{pipe.draft}_{'prompt' if with_prompt else 'noprompt'}.json"
            t = time.time()
            self._cands[key] = pipe.candidates(self.hyps, self.terms, self.audio, path)
            self.timing[f"check_{engine}"] = round(time.time() - t, 1)
        return self._cands[key]

    def corrected(self, engine: str, s: Settings):
        pipe = self.pipes[engine]
        cands = self.candidates(engine, s.score_with_prompt)
        pipe.settings = s
        return pipe.correct(self.hyps, cands)

    def score(self, words):
        return score(words, self.reference, self.term_keys)


def totals(scores: list) -> dict:
    keys = ["ref_words", "substitutions", "deletions", "insertions", "rare_total", "rare_correct",
            "term_total", "term_correct", "term_false", "course_total", "course_correct", "course_false"]
    t = {k: sum(getattr(s, k) for s in scores) for k in keys}
    t["errors"] = t["substitutions"] + t["deletions"] + t["insertions"]
    t["course_mistakes"] = t["course_total"] - t["course_correct"] + t["course_false"]
    t["course_mistake_rate"] = round(t["course_mistakes"] / t["course_total"], 4)
    t["wer"] = round(t["errors"] / t["ref_words"], 4)
    t["rare_recall"] = round(t["rare_correct"] / t["rare_total"], 4)
    t["term_recall"] = round(t["term_correct"] / t["term_total"], 4)
    return t


def correction_outcomes(lec: Lecture, engine: str, s: Settings) -> dict:
    """For each kept fix: did it fix a wrong word, break a right one, or neither?"""
    draft = lec.hyps[lec.pipes[engine].draft]
    before = lec.score(draft).word_errors
    words, cands = lec.corrected(engine, s)
    after = lec.score(words).word_errors
    first = {w.start: i for i, w in enumerate(words) if "corrected" in w.flags and w.original is not None}
    out = {"kept": 0, "fixed": 0, "broke": 0, "still_wrong": 0, "by_source": {}, "examples": []}
    for c in cands:
        if not c.accepted:
            continue
        i = first[draft[c.i].start]
        n = len(c.term.split())
        was_wrong, now_wrong = any(before[c.i : c.i + c.n]), any(after[i : i + n])
        kind = ("fixed" if was_wrong and not now_wrong else "broke" if now_wrong and not was_wrong
                else "still_wrong" if now_wrong else "neither")
        out["kept"] += 1
        src = out["by_source"].setdefault(c.source, {"kept": 0, "fixed": 0, "broke": 0, "still_wrong": 0})
        src["kept"] += 1
        if kind != "neither":
            out[kind] += 1
            src[kind] += 1
        out["examples"].append({"t": round(c.start, 2), "end": round(c.end, 2), "from": c.original, "to": c.term,
                                "source": c.source, "delta": c.delta, "outcome": kind})
    return out


def review_eval(lec: Lecture, engine: str, s: Settings, model: flag.RiskModel, seed: int) -> dict:
    """How many of the wrong words land inside the review stretches, at each budget."""
    words, cands = lec.corrected(engine, s)
    sc = lec.score(words)
    errors, course_err = np.array(sc.word_errors), np.array(sc.course_errors)
    duration = len(lec.audio) / SR
    second = lec.hyps[SECOND[engine]]
    x = flag.features(words, second, cands, s.margin)
    rng = np.random.default_rng(seed)
    disagree = x[:, 2] + 0.01 * x[:, 3]
    rankers = {
        "model": model.risk(x),
        # the simple rule: words the two engines disagree on, then the second engine's doubt
        "disagreement_only": disagree,
        # what CaptionFix ships (fixed before scoring): disagreements about a course term first
        "course_first": flag.course_first(x, flag.course_nearby(words, second, lec.term_keys)),
        "random": rng.random(len(words)),
    }
    mids = np.array([(w.start + w.end) / 2 for w in words])
    out = {"errors": int(errors.sum()), "course_errors": int(course_err.sum()), "words": len(words)}
    for name, risk in rankers.items():
        for b in BUDGETS:
            spans = flag.regions(words, risk, b, duration)
            inside = np.zeros(len(words), bool)
            for a, e in spans:
                inside |= (mids >= a) & (mids <= e)
            out[f"{name}@{b}"] = {"caught": int((errors & inside).sum()),
                                  "course_caught": int((course_err & inside).sum()),
                                  "review_minutes": round(sum(e - a for a, e in spans) / 60, 2)}
    return out


def lectures(split: str, pipes, extra: tuple[str, ...] = ()) -> list[Lecture]:
    return [Lecture(m, pipes, extra) for m in LECTURES if m["split"] == split]


def choose_vocabulary(dev: list[Lecture]) -> tuple[Settings, dict]:
    """Which vocabulary Zoom gets: none, every slide term, or the terms of the slides being discussed.

    Rule written before the dev lectures were transcribed: fewest course-term mistakes, among the
    options with no more total errors than Zoom without any vocabulary."""
    options = {"none": ("zoom", Settings(vocabulary="none")), "deck": ("zoom_vocab", Settings(vocabulary="deck"))}
    for k in SYNC_SLIDES:
        options[f"synced{k}"] = (f"zoom_sync{k}", Settings(vocabulary="synced", slides=k))
    scores = {name: totals([lec.score(lec.hyps[key]) for lec in dev]) for name, (key, _) in options.items()}
    allowed = [n for n in options if scores[n]["errors"] <= scores["none"]["errors"]]
    best = min(allowed, key=lambda n: (scores[n]["course_mistakes"], scores[n]["errors"]))
    return options[best][1], scores


def tune(engine: str, pipes) -> None:
    dev = lectures("dev", pipes, tuple(f"zoom_sync{k}" for k in SYNC_SLIDES) if engine == "zoom" else ())
    base, vocab_scores = Settings(), None
    if engine == "zoom":
        base, vocab_scores = choose_vocabulary(dev)
        print("vocabulary:", base.vocabulary, base.slides,
              {n: (v["errors"], v["course_mistakes"]) for n, v in vocab_scores.items()}, flush=True)
    pipes[engine].settings = base
    draft = totals([lec.score(lec.hyps[pipes[engine].draft]) for lec in dev])
    grid = []
    for with_prompt, sources, sim, margin in itertools.product([True, False], SOURCES, SIMILARITIES, MARGINS):
        if "sound" not in sources and sim != SIMILARITIES[0]:
            continue  # similarity only filters fixes found by sound
        s = replace(base, margin=margin, min_similarity=sim, sources=sources, score_with_prompt=with_prompt)
        t = totals([lec.score(lec.corrected(engine, s)[0]) for lec in dev])
        grid.append({**s.to_dict(), **t})

    # Rule written before any test lecture was scored: fewest total errors on the dev lectures,
    # then fewest false slide terms, then the larger (more cautious) margin.
    best = min(grid, key=lambda g: (g["errors"], g["term_false"], -g["margin"]))
    s = replace(base, margin=best["margin"], min_similarity=best["min_similarity"],
                sources=tuple(best["sources"]), score_with_prompt=best["score_with_prompt"])
    settings_path(engine).write_text(json.dumps(s.to_dict(), indent=2) + "\n")

    xs, ys = [], []
    for lec in dev:
        words, cands = lec.corrected(engine, s)
        xs.append(flag.features(words, lec.hyps[SECOND[engine]], cands, s.margin))
        ys.append(np.array(lec.score(words).word_errors))
    model = flag.RiskModel.fit(np.vstack(xs), np.concatenate(ys))
    model.save(engine)

    RESULTS.mkdir(exist_ok=True)
    save_json(RESULTS / f"dev_tuning_{engine}.json", {
        "engine": engine, "lectures": [lec.id for lec in dev], "vocabulary": vocab_scores,
        "draft": draft, "chosen": s.to_dict(),
        "chosen_scores": best, "grid": grid,
        "review_model": {"features": flag.FEATURES, "weights": model.weights, "bias": model.bias},
    })
    print(f"{engine}: draft wer {draft['wer']} -> {best['wer']} with {s.to_dict()}")


def evaluate(split: str, pipes) -> None:
    settings = {e: Settings.load(e) for e in ENGINES}
    models = {e: flag.RiskModel.load(e) for e in ENGINES}
    for e in ENGINES:
        pipes[e].settings = settings[e]
    synced = pipes["zoom"].draft if pipes["zoom"].draft.startswith("zoom_sync") else None
    lecs = lectures(split, pipes, (synced,) if synced else ())
    systems = list(TRANSCRIPTS) + (["zoom_synced"] if synced else []) + [f"captionfix_{e}" for e in ENGINES]
    rows, by_system = [], {k: [] for k in systems}
    for k, lec in enumerate(lecs):
        row = {"id": lec.id, "course": lec.meta["course"], "title": lec.meta["title"],
               "minutes": lec.minutes, "slide_terms": len(lec.terms)}
        for name in systems:
            if name.startswith("captionfix_"):
                engine = name.removeprefix("captionfix_")
                words = lec.corrected(engine, settings[engine])[0]
                save_json(lec.out / f"{name}_words.json", [w.to_dict() for w in words])
            else:
                words = lec.hyps[synced if name == "zoom_synced" else name]
            sc = lec.score(words)
            by_system[name].append(sc)
            row[name] = sc.summary()
        for e in ENGINES:
            row[f"corrections_{e}"] = correction_outcomes(lec, e, settings[e])
            row[f"review_{e}"] = review_eval(lec, e, settings[e], models[e], seed=k)
        row["timing"] = lec.timing
        rows.append(row)
        print(lec.id, {n: row[n]["wer"] for n in systems}, flush=True)

    summary = {"split": split, "lectures": len(rows), "minutes": round(sum(r["minutes"] for r in rows), 1),
               "settings": {e: settings[e].to_dict() for e in ENGINES},
               "systems": {k: totals(v) for k, v in by_system.items()}}
    for e in ENGINES:
        errors = sum(r[f"review_{e}"]["errors"] for r in rows)
        course = sum(r[f"review_{e}"]["course_errors"] for r in rows)
        review = {}
        for key in rows[0][f"review_{e}"]:
            if "@" in key:
                caught = sum(r[f"review_{e}"][key]["caught"] for r in rows)
                course_caught = sum(r[f"review_{e}"][key]["course_caught"] for r in rows)
                review[key] = {"caught": caught, "share_caught": round(caught / errors, 4),
                               "course_caught": course_caught, "course_share_caught": round(course_caught / course, 4),
                               "review_minutes": round(sum(r[f"review_{e}"][key]["review_minutes"] for r in rows), 1)}
        corr = {k: sum(r[f"corrections_{e}"][k] for r in rows) for k in ("kept", "fixed", "broke", "still_wrong")}
        by_source = {}
        for r in rows:
            for src, v in r[f"corrections_{e}"]["by_source"].items():
                acc = by_source.setdefault(src, {"kept": 0, "fixed": 0, "broke": 0, "still_wrong": 0})
                for kk in acc:
                    acc[kk] += v[kk]
        summary[f"corrections_{e}"] = {**corr, "by_source": by_source}
        summary[f"review_{e}"] = {"errors_after_correction": errors, "course_errors_after_correction": course, **review}
    summary["per_lecture"] = rows
    RESULTS.mkdir(exist_ok=True)
    save_json(RESULTS / f"{split}.json", summary)
    for k, v in summary["systems"].items():
        print(f"{k:20s} wer={v['wer']} errors={v['errors']} course_mistakes={v['course_mistakes']} "
              f"({v['course_mistake_rate']}) terms={v['term_recall']} false={v['term_false']}")
    for e in ENGINES:
        print(e, summary[f"corrections_{e}"])
        for b in BUDGETS:
            r = summary[f"review_{e}"]
            print(" ", b, {n: (r[f"{n}@{b}"]["share_caught"], r[f"{n}@{b}"]["course_share_caught"])
                           for n in ("model", "disagreement_only", "course_first", "random")})


def doubt_marks(pipes) -> None:
    """How good the doubt marks on Zoom's captions are: how many marked words are really wrong,
    and how many of the course-term mistakes get a mark."""
    out = {}
    for split in ("dev", "test", "fresh"):
        marks = right = errors = 0
        for lec in lectures(split, pipes):
            words = lec.hyps["zoom"]
            course_err = lec.score(words).course_errors
            found = flag.doubts(words, lec.hyps["whisper"], lec.term_keys)
            marks += len(found)
            right += sum(course_err[i] for i, _ in found)
            errors += sum(course_err)
        out[split] = {"marks": marks, "on_mistakes": right, "course_error_words": errors,
                      "precision": round(right / max(marks, 1), 3), "recall": round(right / max(errors, 1), 3)}
        print(split, out[split], flush=True)
    save_json(RESULTS / "doubts.json", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["tune", "test", "fresh", "dev", "doubts"])
    ap.add_argument("--engine", choices=ENGINES, default="zoom")
    args = ap.parse_args()
    pipes = {e: Pipeline(e) for e in ENGINES}
    if args.step == "tune":
        tune(args.engine, pipes)
    elif args.step == "doubts":
        doubt_marks(pipes)
    else:
        evaluate(args.step, pipes)


if __name__ == "__main__":
    main()
