"""Deterministic shared test perturbations, not training augmentation.

Noise is synthetic white Gaussian noise, NOT environmental background recordings.
Replay is a fixed synthetic channel, NOT measured room/physical replay evidence.
"""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from benchmark.lava5 import ROOT, read_csv, sha256, verify_protocol, write_csv
from src.lava.artifacts import write_json_atomic

NOISE = {f"snr_{snr}": snr for snr in (20, 10, 5, 0)}
CODECS = {"mp3_128k": ("libmp3lame", "128k", ".mp3"),
          "mp3_64k": ("libmp3lame", "64k", ".mp3"),
          "opus_64k": ("libopus", "64k", ".ogg"),
          "aac_96k": ("aac", "96k", ".m4a")}


def codec_roundtrip(audio, rate, target, setting, ffmpeg):
    import tempfile
    codec, bitrate, suffix = setting
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        pcm, encoded = Path(temporary) / "input.wav", Path(temporary) / ("encoded" + suffix)
        sf.write(pcm, audio, rate, subtype="PCM_16")
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(pcm), "-c:a", codec, "-b:a", bitrate, str(encoded)], check=True, capture_output=True)
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(encoded), "-ar", str(rate), "-ac", "1", "-c:a", "pcm_f32le", str(target)], check=True, capture_output=True)


def add_noise(audio, snr_db, seed):
    x = np.asarray(audio, dtype=np.float64)
    power = np.mean(x * x)
    if power == 0:
        raise ValueError("SNR undefined for silent source; do not fabricate a noise condition")
    noise = np.random.default_rng(seed).standard_normal(x.shape)
    noise *= np.sqrt(power / (10 ** (snr_db / 10) * np.mean(noise * noise)))
    return (x + noise).astype(np.float32)


def simulated_replay(audio, rate):
    # One direct path plus three decaying reflections. Fixed gain normalization;
    # no per-recording peak normalization and no extra random additive noise.
    taps = [(0.0, 1.0), (0.017, 0.45), (0.043, 0.25), (0.089, 0.12)]
    output = np.zeros_like(audio)
    for seconds, gain in taps:
        offset = round(seconds * rate)
        if offset < len(audio):
            output[offset:] += gain * audio[:len(audio) - offset]
    output /= sum(gain for _, gain in taps)
    cutoff = min(3800, rate * 0.45)
    return sosfilt(butter(4, [100, cutoff], btype="bandpass", fs=rate, output="sos"), output).astype(np.float32)


def generate(output, suite):
    if suite not in {"noise", "compression", "replay"}:
        raise ValueError("Choose noise, compression, or replay")
    verify_protocol(output)
    rows = read_csv(output / "protocol/test_samples.csv")
    ffmpeg = shutil.which("ffmpeg")
    if suite == "compression" and ffmpeg is None:
        write_json_atomic(output / "protocol/compression_status.json", {"status": "NOT_AVAILABLE", "reason": "ffmpeg missing"})
        return
    conditions = NOISE if suite == "noise" else CODECS if suite == "compression" else {"synthetic_channel": None}
    # Float32 mono at original rate, <=4.1 s; all model prefixes remain available.
    estimate = int(sum(min(float(r["duration_seconds"]), 4.1) * int(r["sample_rate"]) * 4 for r in rows) * len(conditions))
    if shutil.disk_usage(output).free < estimate + 1024**3:
        raise OSError(f"Insufficient disk: conservative requirement {estimate / 1024**3:.2f} GiB plus 1 GiB reserve")
    settings = dict(suite=suite, version="lava5-stress-v1", conditions=conditions,
        seed="first 16 hex digits of source SHA256", sample_rate="original", mono="channel mean",
        duration="preserve original length up to 4.1 seconds; detector retains its own prefix/padding policy",
        storage="WAV FLOAT (no clipping/quantization for noise/replay)",
        noise="seeded AWGN, exact whole-prefix RMS SNR, not recorded environmental noise",
        replay="synthetic direct+echo taps (0,1),(.017,.45),(.043,.25),(.089,.12), divide gain by1.82; causal 4th-order 100-3800Hz Butterworth bandpass; not physical replay",
        compression="FFmpeg encode/decode, PCM16 codec input; no custom loudness normalization; codec round-trip includes quantization/channel effects",
        estimated_bytes=estimate,
        ffmpeg_version=subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0] if ffmpeg else None)
    settings_path = output / "protocol" / f"{suite}_generation.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Stress protocol changed: use a new output directory")
    write_json_atomic(settings_path, settings)
    settings_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    for condition, setting in conditions.items():
        manifest_path = output / "protocol/conditions" / suite / f"{condition}.csv"
        if manifest_path.exists():
            existing = read_csv(manifest_path)
            if len(existing) != len(rows) or any(sha256(output / r["path"]) != r["sha256"] for r in existing):
                raise ValueError(f"Corrupted condition: {condition}")
            print(f"Verified existing condition {condition}", flush=True)
            continue
        manifest = []
        directory = output / "stress_audio" / suite / condition
        directory.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(rows):
            source = ROOT / row["path"]
            if sha256(source) != row["sha256"]:
                raise ValueError(f"Changed source: {source}")
            with sf.SoundFile(source) as stream:
                rate = stream.samplerate
                audio = stream.read(round(rate * 4.1), dtype="float32", always_2d=True).mean(axis=1)
            target = directory / f"{row['sample_id']}.wav"
            receipt_path = target.with_suffix(".json")
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
            reused = False
            if receipt is not None:
                if (receipt["settings_sha256"] != settings_hash or receipt["source_sha256"] != row["sha256"]
                        or not target.exists() or sha256(target) != receipt["output_sha256"]):
                    raise ValueError(f"Invalid cached stress audio: {target}")
                reused = True
            if not reused:
                pending = target.with_suffix(".pending.wav")
                # Only this runner's private incomplete temporary output is replaced.
                if pending.exists():
                    pending.unlink()
                if suite == "noise":
                    values = add_noise(audio, setting, int(row["sha256"][:16], 16))
                    sf.write(pending, values, rate, subtype="FLOAT")
                elif suite == "replay":
                    sf.write(pending, simulated_replay(audio, rate), rate, subtype="FLOAT")
                else:
                    codec_roundtrip(audio, rate, pending, setting, ffmpeg)
                if target.exists():
                    if sha256(target) != sha256(pending):
                        raise ValueError(f"Existing stress file differs from deterministic regeneration: {target}")
                    pending.unlink()
                else:
                    pending.rename(target)
                write_json_atomic(receipt_path, dict(settings_sha256=settings_hash, source_sha256=row["sha256"], output_sha256=sha256(target)))
            manifest.append(dict(sample_id=row["sample_id"], label=int(row["label"]), condition=condition,
                source_sha256=row["sha256"], path=target.relative_to(output).as_posix(), sha256=sha256(target)))
            if i % 250 == 0:
                print(f"Generate {suite}/{condition}: {i+1}/{len(rows)}", flush=True)
        write_csv(manifest_path, manifest)
    write_json_atomic(output / "protocol" / f"{suite}_status.json", {"status": "GENERATED", "samples_per_condition": len(rows)})
