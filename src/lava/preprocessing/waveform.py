"""Native-waveform adapter with explicit sample-rate and length contracts."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def normalize_waveform_length(audio: np.ndarray, target_samples: int) -> np.ndarray:
    """Apply the LAVA common deterministic prefix/zero-pad duration policy."""
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("Audio contains NaN or infinite samples")
    if values.size >= target_samples:
        return values[:target_samples].astype(np.float32, copy=False)
    return np.pad(values, (0, target_samples - values.size)).astype(np.float32, copy=False)


def load_waveform(
    path: str | os.PathLike[str],
    *,
    sample_rate: int = 16_000,
    target_samples: int = 48_000,
) -> np.ndarray:
    file_path = os.fspath(path)
    try:
        with sf.SoundFile(file_path) as audio_file:
            source_rate = int(audio_file.samplerate)
            source_limit = int(round(source_rate * target_samples / sample_rate))
            channels = audio_file.read(source_limit, dtype="float32", always_2d=True)
        audio = channels.mean(axis=1, dtype=np.float32)
        if source_rate != sample_rate:
            divisor = int(np.gcd(source_rate, sample_rate))
            audio = resample_poly(audio, sample_rate // divisor, source_rate // divisor).astype(np.float32)
    except (RuntimeError, OSError, sf.LibsndfileError):
        import librosa

        audio, _ = librosa.load(file_path, sr=sample_rate, mono=True, duration=target_samples / sample_rate)
    return normalize_waveform_length(audio, target_samples)

