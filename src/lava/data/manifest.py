"""Build, persist, and verify canonical LAVA dataset manifests."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import config
from src.lava.data.integrity import (
    DUPLICATE_EXCLUDED,
    LABEL_CONFLICT,
    apply_integrity_policy,
    integrity_summary,
)
from src.lava.data.inventory import INTEGRITY_VERSION, scan_inventory
from src.lava.data.split import assign_splits, stable_manifest_hash, validate_split_records


MANIFEST_DIR = Path(config.DATA_DIR) / "manifests"
DATASET_MANIFEST = MANIFEST_DIR / "dataset_manifest.csv"
SPLIT_MANIFEST = MANIFEST_DIR / "split_manifest.csv"
DUPLICATE_REPORT = MANIFEST_DIR / "duplicate_report.csv"
LABEL_CONFLICTS = MANIFEST_DIR / "label_conflicts.csv"
MANIFEST_METADATA = MANIFEST_DIR / "manifest_metadata.json"

DATASET_FIELDS = [
    "sample_id", "path", "basename", "extension", "label", "label_name", "size_bytes", "sha256",
    "sample_rate", "duration_seconds", "channels", "speaker_id", "source_id", "generator_id",
    "dataset_id", "parent_recording_id", "duplicate_group_id", "canonical_sample_id",
    "integrity_status", "included", "exclusion_reason",
]
SPLIT_FIELDS = DATASET_FIELDS + ["split"]


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_manifests(
    *,
    real_dir: str | os.PathLike[str] = config.REAL_DIR,
    fake_dir: str | os.PathLike[str] = config.FAKE_DIR,
    output_dir: str | os.PathLike[str] = MANIFEST_DIR,
    base_dir: str | os.PathLike[str] = config.BASE_DIR,
    seed: int = config.RANDOM_SEED,
) -> dict[str, object]:
    output = Path(output_dir)
    inventory = scan_inventory(real_dir=real_dir, fake_dir=fake_dir, base_dir=base_dir)
    if not inventory:
        raise RuntimeError("No supported audio files found in production dataset")
    annotated = apply_integrity_policy(inventory)
    splits = assign_splits(annotated, seed=seed)
    validate_split_records(annotated, splits)
    manifest_hash = stable_manifest_hash(splits)

    dataset_path = output / DATASET_MANIFEST.name
    split_path = output / SPLIT_MANIFEST.name
    duplicate_path = output / DUPLICATE_REPORT.name
    conflict_path = output / LABEL_CONFLICTS.name
    metadata_path = output / MANIFEST_METADATA.name
    duplicates = [row for row in annotated if row["integrity_status"] == DUPLICATE_EXCLUDED]
    conflicts = [row for row in annotated if row["integrity_status"] == LABEL_CONFLICT]
    _write_csv(dataset_path, annotated, DATASET_FIELDS)
    _write_csv(split_path, splits, SPLIT_FIELDS)
    _write_csv(duplicate_path, duplicates, DATASET_FIELDS)
    _write_csv(conflict_path, conflicts, DATASET_FIELDS)

    summary = integrity_summary(annotated)
    class_counts = Counter(str(row["label_name"]) for row in annotated)
    included_counts = Counter(str(row["label_name"]) for row in annotated if bool(row["included"]))
    split_counts = Counter(str(row["split"]) for row in splits)
    metadata: dict[str, object] = {
        "creation_time": datetime.now(timezone.utc).isoformat(),
        "integrity_version": INTEGRITY_VERSION,
        "random_seed": seed,
        "manifest_hash": manifest_hash,
        "class_counts": dict(sorted(class_counts.items())),
        "included_class_counts": dict(sorted(included_counts.items())),
        **summary,
        "split_ratios": {"train": config.TRAIN_RATIO, "validation": config.VAL_RATIO, "test": config.TEST_RATIO},
        "split_counts": dict(sorted(split_counts.items())),
        "split_claim": "checksum-group-disjoint only",
        "metadata_availability": {
            "speaker_id": "UNKNOWN", "source_id": "UNKNOWN", "generator_id": "UNKNOWN",
            "dataset_id": "UNKNOWN", "parent_recording_id": "UNKNOWN",
        },
        "canonical_duplicate_policy": "retain lexicographically first same-label path; exclude other members",
        "label_conflict_policy": "exclude every member from all splits",
    }
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.parent.mkdir(parents=True, exist_ok=True)
    with temporary_metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    os.replace(temporary_metadata, metadata_path)
    return metadata


def validate_manifest_files(manifest_dir: str | os.PathLike[str] = MANIFEST_DIR) -> dict[str, object]:
    directory = Path(manifest_dir)
    dataset_rows = _read_csv(directory / DATASET_MANIFEST.name)
    split_rows = _read_csv(directory / SPLIT_MANIFEST.name)
    for row in dataset_rows:
        row["included"] = str(row["included"]).lower() == "true"
    validate_split_records(dataset_rows, split_rows)
    actual_hash = stable_manifest_hash(split_rows)
    with (directory / MANIFEST_METADATA.name).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("manifest_hash") != actual_hash:
        raise ValueError("Manifest hash mismatch")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify canonical LAVA dataset manifests")
    parser.add_argument("command", choices=("build", "check"), nargs="?", default="build")
    arguments = parser.parse_args()
    metadata = build_manifests() if arguments.command == "build" else validate_manifest_files()
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

