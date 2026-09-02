"""Scan the production dataset without inferring unavailable provenance metadata."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

import soundfile as sf

import config


UNKNOWN = "UNKNOWN"
INTEGRITY_VERSION = "lava-integrity-v1"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audio_metadata(path: Path) -> tuple[str, str, str]:
    try:
        with sf.SoundFile(str(path)) as audio:
            sample_rate = int(audio.samplerate)
            channels = int(audio.channels)
            duration = float(len(audio)) / sample_rate if sample_rate else 0.0
        return str(sample_rate), f"{duration:.9f}", str(channels)
    except (RuntimeError, OSError, ValueError, sf.LibsndfileError):
        return UNKNOWN, UNKNOWN, UNKNOWN


def _sample_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]


def scan_inventory(
    *,
    real_dir: str | os.PathLike[str] = config.REAL_DIR,
    fake_dir: str | os.PathLike[str] = config.FAKE_DIR,
    base_dir: str | os.PathLike[str] = config.BASE_DIR,
    extensions: Iterable[str] = config.SUPPORTED_AUDIO_EXTENSIONS,
) -> list[dict[str, object]]:
    """Return a deterministic inventory of supported production audio files."""
    allowed = {extension.lower() for extension in extensions}
    base = Path(base_dir).resolve()
    records: list[dict[str, object]] = []
    for label_name, label, directory in (
        (config.REAL_NAME, config.REAL_LABEL, Path(real_dir)),
        (config.FAKE_NAME, config.FAKE_LABEL, Path(fake_dir)),
    ):
        if not directory.exists():
            continue
        paths = sorted(
            (path.resolve() for path in directory.iterdir() if path.is_file() and path.suffix.lower() in allowed),
            key=lambda path: path.as_posix().casefold(),
        )
        for path in paths:
            try:
                relative_path = path.relative_to(base).as_posix()
            except ValueError:
                relative_path = path.as_posix()
            sample_rate, duration, channels = _audio_metadata(path)
            records.append(
                {
                    "sample_id": _sample_id(relative_path),
                    "path": relative_path,
                    "basename": path.stem,
                    "extension": path.suffix.lower(),
                    "label": int(label),
                    "label_name": label_name,
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                    "sample_rate": sample_rate,
                    "duration_seconds": duration,
                    "channels": channels,
                    "speaker_id": UNKNOWN,
                    "source_id": UNKNOWN,
                    "generator_id": UNKNOWN,
                    "dataset_id": UNKNOWN,
                    "parent_recording_id": UNKNOWN,
                }
            )
    return records

