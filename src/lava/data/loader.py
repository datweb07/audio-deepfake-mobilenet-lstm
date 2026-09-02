"""Pure-Python canonical manifest reader safe in TensorFlow and PyTorch processes."""

from __future__ import annotations

import csv
from pathlib import Path

import config
from src.lava.data.manifest import SPLIT_MANIFEST


def load_split(split: str) -> tuple[list[str], list[int]]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if not SPLIT_MANIFEST.is_file():
        raise FileNotFoundError("Canonical split manifest not found. Run: python -m src.lava.data.manifest build")
    paths: list[str] = []
    labels: list[int] = []
    with SPLIT_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            path = Path(row["path"])
            if not path.is_absolute():
                path = Path(config.BASE_DIR) / path
            paths.append(str(path.resolve()))
            labels.append(int(row["label"]))
    if not paths:
        raise RuntimeError(f"Canonical manifest contains no samples for split '{split}'")
    return paths, labels

