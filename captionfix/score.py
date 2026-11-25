"""How well a piece of text matches a stretch of audio, by Whisper's own log probability.

Used to check a correction: "I can value" -> "eigenvalue" is kept only if the audio supports
"eigenvalue" more than it supports what was transcribed."""

from __future__ import annotations

import numpy as np

from .asr import MODEL, SR


class AudioScorer:
    def __init__(self, model: str = MODEL):
        import mlx.core as mx
        from mlx_whisper.load_models import load_model
        from mlx_whisper.tokenizer import get_tokenizer

        self.mx = mx
        self.model = load_model(model, dtype=mx.float16)
        self.tok = get_tokenizer(self.model.is_multilingual, num_languages=self.model.num_languages,
                                 language="en", task="transcribe")
        self._cache_key = None
        self._features = None

    def _features_for(self, audio: np.ndarray, start: float, end: float):
        from mlx_whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim

        key = (id(audio), round(start, 3), round(end, 3))
        if key != self._cache_key:
            clip = audio[int(start * SR) : int(end * SR)]
            mel = log_mel_spectrogram(pad_or_trim(self.mx.array(clip), N_SAMPLES), n_mels=self.model.dims.n_mels)
            self._features = self.model.embed_audio(mel[None].astype(self.mx.float16))
            self._cache_key = key
        return self._features

    def logprob(self, audio: np.ndarray, start: float, end: float, texts: list[str], prompt: str = "") -> list[float]:
        """Total log probability of each text being what is said between `start` and `end`."""
        mx = self.mx
        features = self._features_for(audio, start, end)
        prefix = []
        if prompt:
            prefix = [self.tok.sot_prev] + self.tok.encode(" " + prompt.strip())[-150:]
        prefix += list(self.tok.sot_sequence_including_notimestamps)
        scores = []
        for text in texts:
            body = self.tok.encode(" " + text.strip()) + [self.tok.eot]
            tokens = mx.array([prefix + body])
            logits = self.model.logits(tokens, features)[0].astype(mx.float32)
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            # the logit at position p predicts token p + 1
            pos = mx.arange(len(prefix) - 1, len(prefix) + len(body) - 1)
            picked = logp[pos, mx.array(body)]
            scores.append(float(picked.sum()))
        return scores
