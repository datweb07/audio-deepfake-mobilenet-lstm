"""Phase 2 scientific controls, separate from the sealed Phase 1 protocol."""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
from math import gcd
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.signal import correlate, correlation_lags, resample_poly
import soundfile as sf

from .transforms import Audio, Transform, diagnostics, measured_snr


def prepare_headroom(audio: Audio, peak_limit: float | None) -> tuple[Audio, dict]:
    if peak_limit is not None and not 0 < peak_limit <= 1:
        raise ValueError("Headroom peak must be in (0,1]")
    peak = float(np.max(np.abs(audio.samples)))
    if peak == 0:
        raise ValueError("Silent source has no noise-reference power")
    gain = min(1.0, peak_limit / peak) if peak_limit else 1.0
    return Audio(audio.samples * gain, audio.sample_rate), {"peak_limit": peak_limit, "gain": gain,
                                                           "original_peak": peak}


@dataclass(frozen=True)
class NoisePlan:
    """One realization, calibrated once on prepared clean, reused across orders."""
    unit_noise: np.ndarray
    reference_noise_rms: float
    target_snr_db: float
    seed: int

    @classmethod
    def build(cls, reference: Audio, *, target_snr_db: float, seed: int) -> "NoisePlan":
        power = float(np.mean(reference.samples.astype(np.float64) ** 2))
        if power <= 0 or not np.isfinite(target_snr_db):
            raise ValueError("Noise reference must have finite positive power/SNR")
        noise = np.random.default_rng(seed).standard_normal(len(reference.samples))
        noise /= np.sqrt(np.mean(noise ** 2))
        return cls(noise, float(np.sqrt(power / 10 ** (target_snr_db / 10))), target_snr_db, seed)

    def insert(self, audio: Audio, policy: str) -> tuple[Audio, dict]:
        if len(audio.samples) != len(self.unit_noise):
            raise ValueError("Paired noise requires aligned lengths")
        signal_rms = float(np.sqrt(np.mean(audio.samples.astype(np.float64) ** 2)))
        if signal_rms <= 0:
            raise ValueError("Silent insertion point")
        if policy == "local_target_snr":
            noise_rms = signal_rms / 10 ** (self.target_snr_db / 20)
        elif policy == "fixed_absolute":
            noise_rms = self.reference_noise_rms
        else:
            raise ValueError("Unknown noise policy")
        noise = self.unit_noise * noise_rms
        changed = Audio(audio.samples.astype(np.float64) + noise, audio.sample_rate)
        return changed, {"noise_policy": policy, "noise_seed": self.seed, "noise_rms": noise_rms,
                         "signal_rms_before_insertion": signal_rms, "absolute_noise_power": noise_rms ** 2,
                         "measured_snr_db": measured_snr(audio.samples, changed.samples),
                         "target_snr_db": self.target_snr_db}


def pcm16_control(audio: Audio) -> tuple[Audio, dict]:
    if np.max(np.abs(audio.samples)) > 1:
        raise ValueError("PCM16 control would clip")
    buffer = io.BytesIO()
    sf.write(buffer, audio.samples, audio.sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    samples, rate = sf.read(buffer, dtype="float32")
    changed = Audio(samples, rate)
    return changed, {"policy": "libsndfile FLOAT->PCM16->FLOAT, no resampling", "quantization_step": 1 / 32768,
                     "maximum_abs_error": float(np.max(np.abs(changed.samples - audio.samples)))}


def resample_roundtrip(audio: Audio, intermediate_rate: int = 24000) -> tuple[Audio, dict]:
    if intermediate_rate <= 0:
        raise ValueError("Intermediate rate must be positive")
    divisor = gcd(audio.sample_rate, intermediate_rate)
    values = resample_poly(audio.samples, intermediate_rate // divisor, audio.sample_rate // divisor)
    values = resample_poly(values, audio.sample_rate // divisor, intermediate_rate // divisor)
    values = np.pad(values, (0, max(0, len(audio.samples) - len(values))))[:len(audio.samples)]
    return Audio(values, audio.sample_rate), {"intermediate_rate": intermediate_rate,
                                            "policy": "SciPy polyphase two-way; prefix trim / zero-pad"}


def ffmpeg_identity(audio: Audio, executable: str) -> tuple[Audio, dict]:
    """Lossless PCM transport control including exactly one PCM16 quantizer."""
    quantized, quantization = pcm16_control(audio)
    with tempfile.TemporaryDirectory(prefix="jaes-identity-") as directory:
        source, target = Path(directory) / "source.wav", Path(directory) / "decoded.wav"
        sf.write(source, quantized.samples, audio.sample_rate, subtype="PCM_16")
        subprocess.run([executable, "-nostdin", "-v", "error", "-i", str(source), "-c:a", "pcm_f32le", str(target)],
                       capture_output=True, check=True, timeout=60)
        samples, rate = sf.read(target, dtype="float32")
    return Audio(samples, rate), {"policy": "PCM16 WAV -> FFmpeg PCM_FLOAT WAV; no lossy compression", **quantization}


def estimate_delay(reference: np.ndarray, observed: np.ndarray, *, max_lag: int = 1024) -> dict:
    """Positive lag means observed is delayed. Do not infer separate encoder/decoder delays."""
    first, second = np.asarray(reference, np.float64), np.asarray(observed, np.float64)
    if first.ndim != 1 or second.ndim != 1 or max_lag < 0 or not len(first) or not len(second):
        raise ValueError("Invalid delay inputs")
    if not np.isfinite(first).all() or not np.isfinite(second).all() or not np.any(first) or not np.any(second):
        raise ValueError("Delay undefined for silence/nonfinite audio")
    cross = correlate(second, first, mode="full", method="fft")
    lags = correlation_lags(len(second), len(first), mode="full")
    keep = np.abs(lags) <= max_lag
    index = int(np.argmax(np.abs(cross[keep])))
    lag = int(lags[keep][index])
    similarity = float(np.abs(cross[keep][index]) / np.sqrt(np.dot(first, first) * np.dot(second, second)))
    return {"lag_samples": lag, "normalized_cross_correlation": similarity,
            "search_limit_samples": max_lag, "boundary_peak": abs(lag) == max_lag,
            "interpretation": "net waveform residual delay after container handling; codec distortion may bias peak"}


def align_control(audio: Audio, lag_samples: int) -> Audio:
    """Explicit sensitivity treatment only; never silently applied to native codec results."""
    size = len(audio.samples)
    if abs(lag_samples) >= size:
        raise ValueError("Alignment shift outside waveform")
    if lag_samples > 0:
        values = np.pad(audio.samples[lag_samples:], (0, lag_samples))
    elif lag_samples < 0:
        values = np.pad(audio.samples[:lag_samples], (-lag_samples, 0))
    else:
        values = audio.samples.copy()
    return Audio(values, audio.sample_rate)


def codec_delay_probe(executable: str, *, sample_rate: int = 22050) -> dict:
    impulse = np.zeros(sample_rate, np.float32)
    impulse[sample_rate // 3] = .25
    reference = Audio(impulse, sample_rate)
    changed, evidence = Transform("probe-opus", "codec", {"encoder": "libopus", "bitrate": "24k", "extension": ".ogg"}).apply(reference, seed=0, ffmpeg=executable)
    result = estimate_delay(reference.samples, changed.samples, max_lag=1024)
    packet_metadata = "NOT_AVAILABLE"
    probe = shutil.which("ffprobe")
    if probe:
        with tempfile.TemporaryDirectory(prefix="jaes-delay-") as directory:
            source, target = Path(directory) / "pulse.wav", Path(directory) / "pulse.ogg"
            sf.write(source, reference.samples, sample_rate, subtype="PCM_16")
            subprocess.run([executable, "-nostdin", "-v", "error", "-i", str(source), "-c:a", "libopus", "-b:a", "24k", str(target)],
                           check=True, capture_output=True, timeout=60)
            raw = subprocess.check_output([probe, "-v", "error", "-select_streams", "a:0", "-read_intervals", "%+0.05",
                                           "-show_packets", "-show_streams", "-of", "json", str(target)], text=True, timeout=30)
            packet_metadata = json.loads(raw)
    return {"probe": "one-second impulse, amplitude .25, position one-third", "sample_rate": sample_rate,
            "net_residual_delay": result, "stream_evidence": evidence, "packet_and_skip_metadata": packet_metadata,
            "encoder_component_delay": "NOT_SEPARATELY_IDENTIFIABLE; inspect codec preskip at encoded rate",
            "decoder_component_delay": "NOT_SEPARATELY_IDENTIFIABLE; container compensation and codec delay are coupled",
            "limitations": "Impulse is not speech. Preserve native codec path; speech correlation and explicit aligned control are additional diagnostics."}
