"""Recording + slides -> corrected captions and a review list.

Two engines can write the first draft:
  - zoom: Zoom Scribe (Zoom AI Services), given the terms of the slides being talked about in each
    piece of audio as its vocabulary, cross-checked against Whisper running locally
  - whisper: Whisper large-v3-turbo locally, cross-checked against a second Whisper pass that is
    prompted with the slide terms

Each step writes its result to the output folder and is skipped on a rerun, so a crash halfway
through a semester of recordings only repeats the lecture it was on. Files are written to a
temporary name and renamed, so a half-written file is never read back."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from . import correct, flag
from .asr import SR, Transcriber, load_audio
from .captions import Word, words_to_cues, write_vtt
from .slides import Term, slide_text, terms_from_pdf
from .sync import SlideIndex

HERE = Path(__file__).parent
ENGINES = ("zoom", "whisper")
MIN_SIMILARITY = 0.7  # candidates are found and scored at this, then filtered by the setting
# the second opinion each engine's draft is checked against
SECOND = {"zoom": "whisper", "whisper": "whisper_primed"}
FILES = {"whisper": "plain.json", "whisper_primed": "primed.json", "zoom": "zoom.json", "zoom_vocab": "zoom_vocab.json"}


def settings_path(engine: str) -> Path:
    return HERE / f"settings_{engine}.json"


@dataclass
class Settings:
    margin: float = 1.0  # nats; the value chosen on the dev lectures is in settings_<engine>.json
    min_similarity: float = 0.8
    sources: tuple[str, ...] = ("sound", "second", "spelling", "loop")
    score_with_prompt: bool = True
    budget: float = 0.10  # share of the lecture to send to review
    vocabulary: str = "synced"  # zoom only: "none", "deck" (every slide term) or "synced"
    slides: int = 3  # with "synced": how many best-matching slides give terms to each piece

    @classmethod
    def load(cls, engine: str) -> "Settings":
        path = settings_path(engine)
        if path.exists():
            d = json.loads(path.read_text())
            d["sources"] = tuple(d.get("sources", cls.sources))
            return cls(**d)
        return cls()

    def to_dict(self) -> dict:
        return {**self.__dict__, "sources": list(self.sources)}


def save_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def cached_words(path: Path, make) -> list[Word]:
    if path.exists():
        return [Word.from_dict(d) for d in json.loads(path.read_text())]
    words = make()
    save_json(path, [w.to_dict() for w in words])
    return words


def terms_prompt(terms: list[Term], n: int = 60) -> str:
    return "Lecture terms: " + ", ".join(t.text for t in terms[:n]) + "." if terms else ""


class Pipeline:
    def __init__(self, engine: str = "zoom", settings: Settings | None = None):
        assert engine in ENGINES, engine
        self.engine = engine
        self.settings = settings or Settings.load(engine)
        self.asr = Transcriber()
        self._scorer = None
        self._scribe = None

    @property
    def scorer(self):
        if self._scorer is None:
            from .score import AudioScorer

            self._scorer = AudioScorer()
        return self._scorer

    @property
    def scribe(self):
        if self._scribe is None:
            from .zoom_scribe import Scribe

            self._scribe = Scribe()
        return self._scribe

    @property
    def draft(self) -> str:
        """Name of the transcript this engine corrects."""
        if self.engine == "whisper":
            return "whisper"
        s = self.settings
        return {"none": "zoom", "deck": "zoom_vocab"}.get(s.vocabulary, f"zoom_sync{s.slides}")

    def transcripts(self, audio_path: Path, slides: Path | None, out: Path,
                    need: tuple[str, ...] | None = None) -> dict[str, list[Word]]:
        """The draft and second-opinion transcripts this engine needs (or `need`), cached in `out`.

        Slide-synced Zoom transcripts are named zoom_sync<k>, k being the slides per piece; they
        use the local Whisper transcript to tell which slides are up, so that one comes first."""
        out.mkdir(parents=True, exist_ok=True)
        need = need or (self.draft, SECOND[self.engine])
        need = sorted(need, key=lambda k: k != "whisper")
        audio = load_audio(str(audio_path))
        terms = terms_from_pdf(slides) if slides else []
        hyps: dict[str, list[Word]] = {}

        def synced(k: int):
            if "whisper" not in hyps:
                hyps["whisper"] = cached_words(out / FILES["whisper"], lambda: self.asr.plain(audio))
            index = SlideIndex(slide_text(slides), terms) if slides else None
            vocab = (lambda a, b: index.phrases(hyps["whisper"], a, b, k)) if index else None
            return self.scribe.transcribe(audio, out / f"scribe_sync{k}", vocab)

        makers = {
            "whisper": lambda: self.asr.plain(audio),
            "whisper_primed": lambda: self.asr.chunked(audio, terms),
            "zoom": lambda: self.scribe.transcribe(audio, out / "scribe"),
            "zoom_vocab": lambda: self.scribe.transcribe(audio, out / "scribe_vocab", [t.text for t in terms]),
        }
        for k in need:
            if k.startswith("zoom_sync"):
                n = int(k.removeprefix("zoom_sync"))
                hyps[k] = cached_words(out / f"{k}.json", lambda n=n: synced(n))
            else:
                hyps[k] = cached_words(out / FILES[k], makers[k])
        return {k: hyps[k] for k in need}

    def candidates(self, hyps: dict[str, list[Word]], terms: list[Term], audio, path: Path) -> list[correct.Candidate]:
        """Every candidate fix, scored against the audio once and cached."""
        if path.exists():
            return [correct.Candidate(**d) for d in json.loads(path.read_text())]
        draft, second = hyps[self.draft], hyps[SECOND[self.engine]]
        cands = (correct.by_sound(draft, terms, MIN_SIMILARITY)
                 + correct.by_second(draft, second, terms)
                 + correct.by_loop(draft, second))
        prompt = terms_prompt(terms) if self.settings.score_with_prompt else ""
        correct.verify(draft, cands, audio, self.scorer, prompt)
        save_json(path, [c.to_dict() for c in cands])
        return cands

    def correct(self, hyps, cands: list[correct.Candidate]) -> tuple[list[Word], list[correct.Candidate]]:
        s = self.settings
        kept = [c for c in cands if c.source in s.sources and c.similarity >= s.min_similarity]
        for c in kept:
            c.accepted = False
        return correct.apply(hyps[self.draft], kept, s.margin), kept

    def features(self, words: list[Word], hyps, kept):
        return flag.features(words, hyps[SECOND[self.engine]], kept, self.settings.margin)

    def run(self, audio_path: Path, slides: Path | None, out: Path) -> dict:
        s = self.settings
        t0 = time.time()
        hyps = self.transcripts(audio_path, slides, out)
        audio = load_audio(str(audio_path))
        duration = len(audio) / SR
        terms = terms_from_pdf(slides) if slides else []
        cands = self.candidates(hyps, terms, audio, out / f"candidates_{self.draft}.json")
        words, kept = self.correct(hyps, cands)

        second = hyps[SECOND[self.engine]]
        keys = {t.key for t in terms}
        risk = flag.course_first(self.features(words, hyps, kept), flag.course_nearby(words, second, keys))
        spans = flag.regions(words, risk, s.budget, duration)
        flag.mark(words, spans)
        doubts = flag.doubts(words, second, keys)
        for i, _ in doubts:
            words[i].flags.append("doubt")

        write_vtt(words_to_cues(words), out / "captions.vtt")
        review = [{"start": a, "end": b,
                   "text": " ".join(w.text for w in words if a <= (w.start + w.end) / 2 <= b)} for a, b in spans]
        save_json(out / "words.json", [w.to_dict() for w in words])
        save_json(out / "review.json", review)
        save_json(out / "doubts.json", [{"i": i, "word": words[i].text, "heard": alt, "start": words[i].start}
                                        for i, alt in doubts])
        summary = {
            "audio": str(audio_path), "slides": str(slides) if slides else None, "engine": self.engine,
            "draft": self.draft,
            "minutes": round(duration / 60, 1), "slide_terms": len(terms),
            "candidates": len(kept), "corrections": sum(c.accepted for c in kept),
            "review_stretches": len(spans), "doubts": len(doubts),
            "review_minutes": round(sum(b - a for a, b in spans) / 60, 1),
            "seconds": round(time.time() - t0, 1),
        }
        save_json(out / "run.json", summary)
        return summary
