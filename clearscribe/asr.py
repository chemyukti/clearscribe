"""Acoustic scoring with the existing wav2vec 2.0 CTC model.

The model is run once per recording; both the standard and the glossary-assisted transcripts are
decoded from the same frame log-probabilities, so any difference between them comes only from the
glossary boost.
"""

import threading
from dataclasses import dataclass

import numpy as np

from .config import CHUNK_OVERLAP_SECONDS, CHUNK_SECONDS, MODEL_NAME, SAMPLE_RATE


@dataclass
class Vocabulary:
    labels: list          # token id -> character ("|" is the word delimiter)
    blank_id: int
    delimiter_id: int
    ignored_ids: list     # special tokens that never appear in a transcript (<s>, </s>, <unk>)


@dataclass
class AcousticScores:
    log_probs: np.ndarray  # (frames, vocab) natural-log CTC posteriors
    frame_seconds: float   # audio duration covered by one frame
    vocab: Vocabulary


class SpeechModel:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self._torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device).eval()

        tok = self.processor.tokenizer
        vocab = tok.get_vocab()
        labels = [""] * len(vocab)
        for token, idx in vocab.items():
            labels[idx] = token
        delimiter = tok.word_delimiter_token or "|"
        ignored = [vocab[t] for t in (tok.bos_token, tok.eos_token, tok.unk_token) if t in vocab]
        self.vocab = Vocabulary(labels, tok.pad_token_id, vocab[delimiter], ignored)
        # Samples per output frame (320 for wav2vec 2.0 base).
        self.samples_per_frame = int(np.prod(self.model.config.conv_stride))

    def _score(self, wave: np.ndarray) -> np.ndarray:
        torch = self._torch
        inputs = self.processor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits[0]
        return torch.log_softmax(logits.float(), dim=-1).cpu().numpy()

    def score(self, wave: np.ndarray) -> AcousticScores:
        """Frame log-probabilities for a whole recording, in overlapping windows."""
        chunk = int(CHUNK_SECONDS * SAMPLE_RATE)
        context = int(CHUNK_OVERLAP_SECONDS * SAMPLE_RATE)
        spf = self.samples_per_frame

        if len(wave) <= chunk + 2 * context:
            log_probs = self._score(wave)
        else:
            pieces = []
            for start in range(0, len(wave), chunk):
                end = min(start + chunk, len(wave))
                a, b = max(0, start - context), min(len(wave), end + context)
                lp = self._score(wave[a:b])
                first = int(round((start - a) / spf))
                last = len(lp) if end == len(wave) else int(round((end - a) / spf))
                pieces.append(lp[first:last])
            log_probs = np.concatenate(pieces, axis=0)

        return AcousticScores(log_probs, spf / SAMPLE_RATE, self.vocab)


_model = None
_model_lock = threading.Lock()


def get_model() -> SpeechModel:
    """Load the model once per process (first request pays the start-up cost)."""
    global _model
    with _model_lock:
        if _model is None:
            _model = SpeechModel()
        return _model
