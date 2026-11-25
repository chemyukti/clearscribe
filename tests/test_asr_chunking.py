import numpy as np

from clearscribe import asr
from conftest import VOCAB


class StubModel(asr.SpeechModel):
    """SpeechModel without the network: each frame's 'log-prob' row is its absolute sample offset."""

    def __init__(self):
        self.vocab = VOCAB
        self.samples_per_frame = 320
        self.offset = 0

    def _score(self, wave):
        n = len(wave) // 320
        return np.repeat(wave[::320][:n, None], len(VOCAB.labels), axis=1)


def test_long_audio_is_stitched_without_gaps_or_duplicates():
    seconds = 95
    wave = np.arange(seconds * 16000, dtype=np.float32)  # sample value == sample index
    scores = StubModel().score(wave)
    starts = scores.log_probs[:, 0]
    assert abs(len(starts) - len(wave) / 320) <= 2
    steps = np.diff(starts)
    assert (steps > 0).all()            # in order, no duplicated frames at chunk joins
    assert steps.max() <= 2 * 320       # at most one frame of drift at a join
    assert scores.frame_seconds == 0.02


def test_short_audio_is_scored_in_one_pass():
    wave = np.arange(5 * 16000, dtype=np.float32)
    assert len(StubModel().score(wave).log_probs) == len(wave) // 320
