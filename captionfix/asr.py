"""Speech to text with Whisper on Apple silicon (mlx-whisper), with or without slide vocabulary."""

from __future__ import annotations

import numpy as np

from .captions import Word
from .slides import Term

MODEL = "mlx-community/whisper-large-v3-turbo"
SR = 16_000
MAX_PROMPT_TOKENS = 200  # Whisper accepts up to 223 prompt tokens


def load_audio(path: str) -> np.ndarray:
    from mlx_whisper.audio import load_audio as _load

    return np.asarray(_load(path), dtype=np.float32)


def quiet_cuts(audio: np.ndarray, target: float = 24.0, longest: float = 28.0) -> list[tuple[int, int]]:
    """Split audio into pieces under `longest` seconds, cutting at the quietest 20 ms
    between `target` and `longest` so a cut rarely lands inside a word."""
    hop = SR // 50
    n = len(audio) // hop
    energy = np.sqrt(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    cuts, start = [], 0
    while start < n:
        if n - start <= longest * 50:
            cuts.append((start * hop, len(audio)))
            break
        lo, hi = start + int(target * 50), start + int(longest * 50)
        # smooth over 100 ms so a single quiet frame inside a word does not win
        window = np.convolve(energy[lo:hi], np.ones(5) / 5, mode="same")
        cut = lo + int(np.argmin(window))
        cuts.append((start * hop, cut * hop))
        start = cut
    return cuts


class Transcriber:
    def __init__(self, model: str = MODEL):
        self.model = model
        self._tokenizer = None

    def _tokens(self, text: str) -> int:
        if self._tokenizer is None:
            from mlx_whisper.tokenizer import get_tokenizer

            self._tokenizer = get_tokenizer(multilingual=True, language="en", task="transcribe")
        return len(self._tokenizer.encode(" " + text))

    def _run(self, audio: np.ndarray, offset: float, **opts) -> list[Word]:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=self.model, language="en", word_timestamps=True, verbose=None, **opts
        )
        words = []
        for seg in result["segments"]:
            for w in seg.get("words", []):
                raw, text = w["word"], w["word"].strip()
                if not text:
                    continue
                start, end, prob = round(w["start"] + offset, 3), round(w["end"] + offset, 3), float(w["probability"])
                if words and not raw[:1].isspace():
                    # a piece of the previous word, as in "ocw" ".mit" ".edu"
                    prev = words[-1]
                    prev.text, prev.end, prev.prob = prev.text + text, end, min(prev.prob, round(prob, 4))
                else:
                    words.append(Word(text, start, end, round(prob, 4)))
        return words

    def plain(self, audio: np.ndarray) -> list[Word]:
        """Whisper as it is usually run: the whole file, default settings."""
        return self._run(audio, 0.0)

    def chunked(self, audio: np.ndarray, terms: list[Term] | None = None, n_terms: int = 60) -> list[Word]:
        """Transcribe ~25 s pieces. Each piece is prompted with the end of the previous piece and,
        if `terms` is given, slide vocabulary: the most frequent terms plus terms heard recently."""
        words: list[Word] = []
        heard: dict[str, int] = {}  # term key -> index of the piece it was last heard in
        by_key = {t.key: t for t in terms or []}
        for i, (a, b) in enumerate(quiet_cuts(audio)):
            prompt = self._prompt(words, terms, heard, i, n_terms)
            piece = self._run(audio[a:b], a / SR, initial_prompt=prompt or None, condition_on_previous_text=False)
            words += piece
            for w in piece:
                key = w.text.strip(".,?!;:\"'()").lower()
                if key in by_key:
                    heard[key] = i
        return words

    def _prompt(self, words, terms, heard, i, n_terms) -> str:
        previous = " ".join(w.text for w in words[-40:])
        if not terms:
            return previous
        written = {t.key: t.text for t in terms}
        recent = sorted((k for k, j in heard.items() if i - j <= 6), key=lambda k: -heard[k])
        chosen, seen = [], set()
        for key in recent + [t.key for t in terms]:
            if key not in seen:
                seen.add(key)
                chosen.append(written[key])
            if len(chosen) >= n_terms:
                break
        while chosen:
            text = f"Lecture terms: {', '.join(chosen)}. {previous}".strip()
            if self._tokens(text) <= MAX_PROMPT_TOKENS:
                return text
            if len(chosen) > 20:
                chosen.pop()
            else:
                previous = " ".join(previous.split()[10:])
                if not previous:
                    chosen.pop()
        return previous
