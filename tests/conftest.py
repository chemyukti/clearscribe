import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clearscribe.asr import AcousticScores, Vocabulary  # noqa: E402

# Same layout as facebook/wav2vec2-base-960h.
LABELS = ["<pad>", "<s>", "</s>", "<unk>", "|", "E", "T", "A", "O", "N", "I", "H", "S", "R", "D",
          "L", "U", "M", "W", "C", "F", "G", "Y", "P", "B", "V", "K", "'", "X", "J", "Q", "Z"]
VOCAB = Vocabulary(LABELS, blank_id=0, delimiter_id=4, ignored_ids=[1, 2, 3])
FRAME_SECONDS = 0.02


def frames_to_log_probs(frames, floor=1e-6):
    """Each frame is a char (prob 0.95), "-" for blank, or a {char: prob} dict (rest goes to blank)."""
    out = np.full((len(frames), len(LABELS)), floor)
    for t, f in enumerate(frames):
        if isinstance(f, str):
            f = {"<pad>" if f == "-" else f: 0.95}
        for ch, p in f.items():
            out[t, LABELS.index(ch)] = p
        if "<pad>" not in f:
            out[t, 0] = max(floor, 1 - sum(f.values()))
        out[t] /= out[t].sum()
    return np.log(out)


def spell(text, gap=1):
    """Frames that clearly spell ``text`` (spaces become the word delimiter)."""
    frames = []
    for ch in text.replace(" ", "|"):
        frames += [ch] + ["-"] * gap
    return frames


# "WE MET NEAR NEELA ROAD", where D, R, I are faintly present after NEELA (blank wins without help).
NEELADRI_FRAMES = (
    spell("WE MET NEAR NEELA")
    + [{"D": 0.3}, "-", {"R": 0.3}, "-", {"I": 0.3}, "-"]
    + spell(" ROAD")
)


class FakeModel:
    model_name = "fake"

    def __init__(self, frames):
        self.log_probs = frames_to_log_probs(frames)

    def score(self, wave):
        return AcousticScores(self.log_probs, FRAME_SECONDS, VOCAB)


@pytest.fixture
def neeladri_scores():
    return FakeModel(NEELADRI_FRAMES).score(None)
