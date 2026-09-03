"""Quality-gated microphone capture preparation for inference."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf


MIN_RECORDING_SECONDS = 2.0
MAX_RECORDING_SECONDS = 15.0
MIN_RMS_DBFS = -45.0
MAX_CLIPPING_RATIO = 0.05
MIN_ACTIVE_RATIO = 0.20
FRAME_SECONDS = 0.02
CONTEXT_SECONDS = 0.10


class MicrophoneQualityError(ValueError):
    """Raised when a capture is unsafe to classify."""


@dataclass(frozen=True)
class MicrophoneQuality:
    source_sample_rate: int
    original_duration: float
    retained_duration: float
    rms_dbfs: float
    clipping_ratio: float
    active_ratio: float


@dataclass(frozen=True)
class PreparedMicrophoneAudio:
    wav_bytes: bytes
    quality: MicrophoneQuality


def _frame_rms(audio: np.ndarray, frame_samples: int) -> np.ndarray:
    frame_count = int(np.ceil(audio.size / frame_samples))
    padded = np.pad(audio, (0, frame_count * frame_samples - audio.size))
    frames = padded.reshape(frame_count, frame_samples)
    return np.sqrt(np.mean(np.square(frames), axis=1))


def prepare_microphone_recording(audio_bytes: bytes) -> PreparedMicrophoneAudio:
    """Validate and trim a browser WAV without altering speech amplitude/content."""
    if not audio_bytes:
        raise MicrophoneQualityError("The microphone recording is empty.")
    try:
        with sf.SoundFile(io.BytesIO(audio_bytes)) as audio_file:
            sample_rate = int(audio_file.samplerate)
            channels = audio_file.read(dtype="float32", always_2d=True)
    except (RuntimeError, OSError, sf.LibsndfileError) as exc:
        raise MicrophoneQualityError("The browser recording could not be decoded as audio.") from exc

    if sample_rate <= 0 or channels.size == 0:
        raise MicrophoneQualityError("The microphone returned no usable audio samples.")
    audio = channels.mean(axis=1, dtype=np.float32)
    if not np.all(np.isfinite(audio)):
        raise MicrophoneQualityError("The microphone recording contains invalid samples.")

    duration = audio.size / sample_rate
    if duration < MIN_RECORDING_SECONDS:
        raise MicrophoneQualityError(
            f"Recording is too short ({duration:.1f} s). Speak continuously for 3–5 seconds."
        )
    if duration > MAX_RECORDING_SECONDS:
        raise MicrophoneQualityError(
            f"Recording is too long ({duration:.1f} s). Keep the capture below 15 seconds."
        )

    rms = float(np.sqrt(np.mean(np.square(audio))))
    rms_dbfs = float(20.0 * np.log10(max(rms, 1e-8)))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.99))
    if rms_dbfs < MIN_RMS_DBFS:
        raise MicrophoneQualityError(
            f"Recording is too quiet ({rms_dbfs:.1f} dBFS). Move closer to the microphone and retry."
        )
    if clipping_ratio > MAX_CLIPPING_RATIO:
        raise MicrophoneQualityError(
            f"Recording is clipped ({clipping_ratio * 100:.1f}%). Lower the input level and retry."
        )

    frame_samples = max(1, int(round(sample_rate * FRAME_SECONDS)))
    energies = _frame_rms(audio, frame_samples)
    activity_threshold = max(0.008, float(np.percentile(energies, 90)) * 0.18)
    active = energies >= activity_threshold
    active_ratio = float(np.mean(active))
    if not np.any(active) or active_ratio < MIN_ACTIVE_RATIO:
        raise MicrophoneQualityError(
            "Too little active speech was detected. Record one continuous sentence for 3–5 seconds."
        )

    active_indices = np.flatnonzero(active)
    context_frames = int(np.ceil(CONTEXT_SECONDS / FRAME_SECONDS))
    first_frame = max(0, int(active_indices[0]) - context_frames)
    last_frame = min(len(energies), int(active_indices[-1]) + context_frames + 1)
    trimmed = audio[first_frame * frame_samples : min(audio.size, last_frame * frame_samples)]
    retained_duration = trimmed.size / sample_rate
    if retained_duration < MIN_RECORDING_SECONDS:
        raise MicrophoneQualityError(
            f"Only {retained_duration:.1f} s of active speech remained. Record a longer sentence."
        )

    output = io.BytesIO()
    sf.write(output, trimmed, sample_rate, format="WAV", subtype="PCM_16")
    quality = MicrophoneQuality(
        source_sample_rate=sample_rate,
        original_duration=duration,
        retained_duration=retained_duration,
        rms_dbfs=rms_dbfs,
        clipping_ratio=clipping_ratio,
        active_ratio=active_ratio,
    )
    return PreparedMicrophoneAudio(wav_bytes=output.getvalue(), quality=quality)
