"""Versioned ordered audio transforms with measured evidence and no silent clipping."""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt
from scipy.io import wavfile
import soundfile as sf

from .evidence import file_hash, object_hash

TRANSFORM_VERSION = "jaes-audio-chain-1"


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        values = np.asarray(self.samples, dtype=np.float32)
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError("Expected finite nonempty mono waveform")
        if self.sample_rate <= 0:
            raise ValueError("Expected positive sample rate")
        object.__setattr__(self, "samples", values)


def load_audio(path: Path, *, sample_rate: int, max_seconds: float) -> Audio:
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    with sf.SoundFile(path) as source:
        original_rate = source.samplerate
        values = source.read(round(max_seconds * original_rate), dtype="float32", always_2d=True).mean(axis=1)
    if original_rate != sample_rate:
        divisor = gcd(original_rate, sample_rate)
        values = resample_poly(values, sample_rate // divisor, original_rate // divisor)
    return Audio(values[:round(max_seconds * sample_rate)], sample_rate)


def diagnostics(audio: Audio) -> dict:
    values = audio.samples.astype(np.float64)
    return {"sample_rate": audio.sample_rate, "samples": len(values),
            "duration_seconds": len(values) / audio.sample_rate,
            "rms": float(np.sqrt(np.mean(values ** 2))), "peak_abs": float(np.max(np.abs(values))),
            "outside_unit_fraction": float(np.mean(np.abs(values) > 1.0))}


def write_audio(path: Path, audio: Audio) -> None:
    """Deterministic IEEE float32 WAV; no wall-clock PEAK chunk in the container."""
    wavfile.write(path, audio.sample_rate, audio.samples.astype("<f4"))


def measured_snr(reference: np.ndarray, changed: np.ndarray) -> float:
    signal_power = float(np.mean(np.asarray(reference, np.float64) ** 2))
    noise_power = float(np.mean((np.asarray(changed, np.float64) - reference) ** 2))
    if signal_power <= 0 or noise_power <= 0:
        raise ValueError("SNR undefined for silent source or zero noise")
    return float(10 * np.log10(signal_power / noise_power))


@dataclass(frozen=True)
class Transform:
    key: str
    kind: str
    parameters: dict[str, Any]

    def __post_init__(self) -> None:
        schemas = {"awgn": {"snr_db"}, "codec": {"encoder", "bitrate", "extension"},
                   "bandlimit": {"low_hz", "high_hz", "order"}, "resample": {"target_rate"},
                   "rir": {"path"}, "environmental_noise": {"path", "snr_db"}}
        if not self.key or self.kind not in schemas or set(self.parameters) != schemas.get(self.kind):
            raise ValueError(f"Invalid transform/schema: {self.kind}")
        object_hash(self.parameters)  # finite, JSON-serializable settings
        if self.kind == "codec":
            allowed = {"libopus": ".ogg", "libmp3lame": ".mp3", "aac": ".m4a"}
            if allowed.get(self.parameters["encoder"]) != self.parameters["extension"]:
                raise ValueError("Unsupported conventional codec/container")
            bitrate = str(self.parameters["bitrate"])
            if not bitrate.rstrip("k").isdigit() or int(bitrate.rstrip("k")) <= 0:
                raise ValueError("Invalid codec bitrate")
        if self.kind == "resample" and int(self.parameters["target_rate"]) <= 0:
            raise ValueError("Invalid resampling rate")

    def description(self) -> dict:
        parameters = dict(self.parameters)
        if self.kind in {"rir", "environmental_noise"}:
            parameters["asset_sha256"] = file_hash(Path(parameters["path"]))
        return {"key": self.key, "kind": self.kind, "parameters": parameters, "version": TRANSFORM_VERSION}

    def apply(self, audio: Audio, *, seed: int, ffmpeg: str | None = None) -> tuple[Audio, dict]:
        values = audio.samples.astype(np.float64)
        rate = audio.sample_rate
        parameters = self.parameters
        evidence: dict[str, Any] = {"seed": seed}
        if self.kind in {"awgn", "environmental_noise"}:
            if np.mean(values ** 2) <= 0:
                raise ValueError("Noise SNR undefined for silent source")
            generator = np.random.default_rng(seed)
            if self.kind == "awgn":
                noise = generator.standard_normal(len(values))
            else:
                source = Path(parameters["path"])
                info = sf.info(source)
                noise_audio = load_audio(source, sample_rate=rate, max_seconds=info.duration)
                tiled = np.tile(noise_audio.samples, int(np.ceil(len(values) / len(noise_audio.samples))) + 1)
                offset = int(generator.integers(0, len(tiled) - len(values) + 1))
                noise = tiled[offset:offset + len(values)].astype(np.float64)
                evidence.update(asset_sha256=file_hash(source), offset_samples=offset)
            noise_power = float(np.mean(noise ** 2))
            if noise_power <= 0:
                raise ValueError("Silent noise asset")
            noise *= np.sqrt(np.mean(values ** 2) / (10 ** (float(parameters["snr_db"]) / 10) * noise_power))
            output = Audio(values + noise, rate)
            evidence["measured_snr_db"] = measured_snr(audio.samples, output.samples)
        elif self.kind == "bandlimit":
            low, high = float(parameters["low_hz"]), float(parameters["high_hz"])
            order = int(parameters["order"])
            if not 0 < low < high < rate / 2 or not 1 <= order <= 16:
                raise ValueError("Invalid bandpass cutoffs/order")
            output = Audio(sosfilt(butter(order, [low, high], btype="bandpass", fs=rate, output="sos"), values), rate)
            evidence["phase_policy"] = "causal SOS; no delay compensation"
        elif self.kind == "resample":
            target = int(parameters["target_rate"])
            divisor = gcd(rate, target)
            output = Audio(resample_poly(values, target // divisor, rate // divisor), target)
        elif self.kind == "rir":
            path = Path(parameters["path"])
            impulse, impulse_rate = sf.read(path, dtype="float32", always_2d=True)
            if impulse_rate != rate:
                raise ValueError("RIR rate must match; resample asset explicitly")
            impulse = impulse.mean(axis=1)
            if not np.any(impulse):
                raise ValueError("Silent impulse response")
            from scipy.signal import fftconvolve
            output = Audio(fftconvolve(values, impulse, mode="full")[:len(values)], rate)
            evidence.update(asset_sha256=file_hash(path), tail_policy="truncate to input length; no gain normalization")
        elif self.kind == "codec":
            executable = ffmpeg or shutil.which("ffmpeg")
            if not executable:
                raise FileNotFoundError("ffmpeg unavailable; codec condition cannot run")
            if np.max(np.abs(values)) > 1.0:
                raise ValueError("Codec PCM16 input would clip; select documented common headroom before all chains")
            with tempfile.TemporaryDirectory(prefix="jaes-codec-") as directory:
                temporary = Path(directory)
                source, encoded, decoded = temporary / "in.wav", temporary / ("encoded" + parameters["extension"]), temporary / "decoded.wav"
                sf.write(source, values, rate, subtype="PCM_16")
                command = [executable, "-nostdin", "-v", "error", "-threads", "1", "-i", str(source),
                           "-c:a", parameters["encoder"], "-b:a", str(parameters["bitrate"]), str(encoded)]
                subprocess.run(command, check=True, capture_output=True, timeout=60)
                subprocess.run([executable, "-nostdin", "-v", "error", "-i", str(encoded), "-ar", str(rate),
                                "-ac", "1", "-c:a", "pcm_f32le", str(decoded)], check=True, capture_output=True, timeout=60)
                decoded_values, decoded_rate = sf.read(decoded, dtype="float32")
                decoded_length = len(decoded_values)
                if decoded_rate != rate:
                    raise ValueError("Codec decoded rate mismatch")
                evidence.update(encoded_sha256=file_hash(encoded), encoded_bytes=encoded.stat().st_size,
                                decoded_samples_before_alignment=decoded_length,
                                input_samples=len(values), sample_delta=decoded_length - len(values),
                                alignment_policy="FFmpeg container timestamps; prefix trim / zero-pad; residual delay NOT_MEASURED",
                                input_quantization="PCM16, common to codec paths only; included in channel effect")
                probe = shutil.which("ffprobe")
                if probe:
                    import json
                    probe_result = subprocess.check_output(
                        [probe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                         "stream=codec_name,sample_rate,channels,start_time,duration:format=duration", "-of", "json", str(encoded)],
                        text=True, timeout=30)
                    evidence["encoded_stream"] = json.loads(probe_result)
                else:
                    evidence["encoded_stream"] = "NOT_MEASURED; ffprobe unavailable"
                aligned = np.pad(decoded_values, (0, max(0, len(values) - decoded_length)))[:len(values)]
                output = Audio(aligned, rate)
        else:  # schemas guard above; defensive against future additions
            raise ValueError(self.kind)
        return output, {**self.description(), **evidence, "input": diagnostics(audio), "output": diagnostics(output)}


@dataclass(frozen=True)
class AudioTransformChain:
    transforms: tuple[Transform, ...]

    def __post_init__(self) -> None:
        keys = [transform.key for transform in self.transforms]
        if len(keys) != len(set(keys)):
            raise ValueError("Transform keys must be unique within a chain")

    @property
    def fingerprint(self) -> str:
        return object_hash({"version": TRANSFORM_VERSION, "ordered": [t.description() for t in self.transforms]})

    def apply(self, audio: Audio, *, sample_id: str, seed: int, ffmpeg: str | None = None) -> tuple[Audio, list[dict]]:
        evidence = []
        for transform in self.transforms:
            # Same named noise realization across orders, not a new draw per condition.
            derived = int(object_hash({"sample": sample_id, "seed": seed, "transform": transform.key})[:16], 16)
            audio, step = transform.apply(audio, seed=derived, ffmpeg=ffmpeg)
            evidence.append(step)
        return audio, evidence
