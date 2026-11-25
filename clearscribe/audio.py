"""Audio loading: any common file (or a saved .pt tensor from the dataset pipeline) -> 16 kHz mono float32."""

from math import gcd
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE


def _resample(wave: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == SAMPLE_RATE:
        return wave
    from scipy.signal import resample_poly

    g = gcd(source_rate, SAMPLE_RATE)
    return resample_poly(wave, SAMPLE_RATE // g, source_rate // g).astype(np.float32)


def _load_tensor(path: Path) -> np.ndarray:
    """Load a waveform saved with torch.save (same handling as the training notebook)."""
    import torch

    try:
        wave = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        wave = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(wave, dict):
        for key in ("waveform", "audio", "input_values", "samples"):
            if key in wave:
                wave = wave[key]
                break
    wave = torch.as_tensor(wave).float().squeeze()
    if wave.ndim == 2:
        wave = wave.mean(dim=0) if wave.shape[0] <= 8 else wave.mean(dim=1)
    if wave.ndim != 1:
        raise ValueError(f"Expected a mono waveform, got shape {tuple(wave.shape)}")
    return torch.nan_to_num(wave).numpy().astype(np.float32)


def load_audio(path) -> np.ndarray:
    """Return the recording as a 16 kHz mono float32 array."""
    path = Path(path)
    if path.suffix.lower() == ".pt":
        return _load_tensor(path)  # dataset tensors are already 16 kHz

    import soundfile as sf

    wave, rate = sf.read(str(path), dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    wave = np.nan_to_num(_resample(wave, rate)).astype(np.float32)
    if wave.size == 0:
        raise ValueError("The audio file is empty.")
    return wave
