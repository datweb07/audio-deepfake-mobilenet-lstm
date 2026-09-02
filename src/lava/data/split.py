"""Deterministic checksum-group-disjoint train/validation/test assignment."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from typing import Iterable

import config
from src.lava.data.integrity import INCLUDED


SPLIT_NAMES = ("train", "validation", "test")


def _class_split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError("Each class requires at least three included checksum groups")
    train_count = int(total * train_ratio)
    validation_count = int(total * val_ratio)
    train_count = max(1, train_count)
    validation_count = max(1, validation_count)
    test_count = total - train_count - validation_count
    if test_count < 1:
        train_count -= 1
        test_count += 1
    return train_count, validation_count, test_count


def assign_splits(
    records: Iterable[dict[str, object]],
    *,
    seed: int = config.RANDOM_SEED,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
) -> list[dict[str, object]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to one")
    included = [dict(row) for row in records if row["integrity_status"] == INCLUDED]
    output: list[dict[str, object]] = []
    for label in (config.REAL_LABEL, config.FAKE_LABEL):
        class_rows = sorted(
            (row for row in included if int(row["label"]) == label),
            key=lambda row: (str(row["duplicate_group_id"]), str(row["sample_id"])),
        )
        random.Random(seed + int(label)).shuffle(class_rows)
        train_count, validation_count, _ = _class_split_counts(len(class_rows), train_ratio, val_ratio)
        for index, row in enumerate(class_rows):
            if index < train_count:
                split = "train"
            elif index < train_count + validation_count:
                split = "validation"
            else:
                split = "test"
            row["split"] = split
            output.append(row)
    return sorted(output, key=lambda row: (str(row["split"]), int(row["label"]), str(row["sample_id"])))


def validate_split_records(
    inventory_records: Iterable[dict[str, object]], split_records: Iterable[dict[str, object]]
) -> None:
    inventory = list(inventory_records)
    splits = list(split_records)
    included_ids = {str(row["sample_id"]) for row in inventory if bool(row["included"])}
    split_ids = [str(row["sample_id"]) for row in splits]
    if len(split_ids) != len(set(split_ids)):
        raise ValueError("An included sample appears more than once in split manifest")
    if set(split_ids) != included_ids:
        raise ValueError("Every included sample must appear in exactly one split")
    if any(str(row.get("split")) not in SPLIT_NAMES for row in splits):
        raise ValueError("Invalid split name")
    if any(str(row["integrity_status"]) != INCLUDED for row in splits):
        raise ValueError("Excluded integrity record appears in split manifest")
    checksum_splits: dict[str, set[str]] = {}
    group_splits: dict[str, set[str]] = {}
    for row in splits:
        checksum_splits.setdefault(str(row["sha256"]), set()).add(str(row["split"]))
        group_splits.setdefault(str(row["duplicate_group_id"]), set()).add(str(row["split"]))
    if any(len(values) > 1 for values in checksum_splits.values()):
        raise ValueError("A checksum occurs across multiple splits")
    if any(len(values) > 1 for values in group_splits.values()):
        raise ValueError("A duplicate group occurs across multiple splits")
    labels = Counter(int(row["label"]) for row in splits)
    if set(labels) != {config.REAL_LABEL, config.FAKE_LABEL}:
        raise ValueError("Split manifest must contain REAL=0 and FAKE=1")


def stable_manifest_hash(split_records: Iterable[dict[str, object]]) -> str:
    lines = []
    for row in sorted(split_records, key=lambda value: str(value["sample_id"])):
        lines.append(
            "|".join(
                str(row[key])
                for key in ("sample_id", "path", "label", "sha256", "duplicate_group_id", "split")
            )
        )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()

