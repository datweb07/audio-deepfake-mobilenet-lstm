"""Explicit source lineage and partition checks, independent of model frameworks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

from .evidence import file_hash, object_hash, read_csv

UNKNOWN = "UNKNOWN"
PARTITIONS = {"train", "development", "calibration", "test", "external_confirmation"}


def known(value: str) -> bool:
    return bool(value.strip()) and value.strip().upper() not in {UNKNOWN, "NONE", "N/A"}


@dataclass(frozen=True)
class SourceRecord:
    sample_id: str
    path: str
    parent_recording_id: str
    dataset_id: str
    dataset_version: str
    split: str
    label: int
    checksum: str
    sample_rate: int
    duration: float
    speaker_id: str = UNKNOWN
    source_speaker_id: str = UNKNOWN
    target_speaker_id: str = UNKNOWN
    generator_id: str = UNKNOWN
    attack_id: str = UNKNOWN
    language: str = UNKNOWN
    source_url_or_record: str = UNKNOWN
    license: str = UNKNOWN
    provenance_status: str = "INCOMPLETE"
    parent_scope: str = "UNKNOWN"

    def __post_init__(self) -> None:
        if self.split not in PARTITIONS or self.label not in {0, 1}:
            raise ValueError("Invalid partition or REAL=0/FAKE=1 label")
        if not self.sample_id or not known(self.parent_recording_id):
            raise ValueError("Every record needs a local parent grouping key")
        if len(self.checksum) != 64 or any(c not in "0123456789abcdef" for c in self.checksum):
            raise ValueError("Expected lowercase SHA-256")
        if self.sample_rate <= 0 or not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("Invalid source rate/duration")

    def to_dict(self) -> dict:
        return asdict(self)


def load_manifest(path: Path) -> list[SourceRecord]:
    rows = read_csv(path)
    records = [SourceRecord(**{**row, "label": int(row["label"]),
                             "sample_rate": int(row["sample_rate"]), "duration": float(row["duration"])}) for row in rows]
    audit_partitions(records)
    return records


def audit_partitions(records: Iterable[SourceRecord], *, require_speaker_disjoint: bool = False,
                     require_generator_disjoint: bool = False) -> dict:
    records = list(records)
    identifiers: set[str] = set()
    indexes: dict[str, dict[str, set[str]]] = {key: {} for key in ("checksum", "parent_recording_id", "speaker")}
    generator_splits: dict[str, set[str]] = {}
    label_groups: dict[str, set[int]] = {}
    for record in records:
        if record.sample_id in identifiers:
            raise ValueError(f"Duplicate sample ID: {record.sample_id}")
        identifiers.add(record.sample_id)
        for key in ("checksum", "parent_recording_id"):
            value = getattr(record, key)
            indexes[key].setdefault(value, set()).add(record.split)
        label_groups.setdefault(record.checksum, set()).add(record.label)
        speakers = (record.speaker_id, record.source_speaker_id, record.target_speaker_id)
        if require_speaker_disjoint and not any(known(value) for value in speakers):
            raise ValueError("Cannot certify speaker hold-out with UNKNOWN identity")
        for speaker in speakers:
            if known(speaker):
                indexes["speaker"].setdefault(speaker, set()).add(record.split)
        if record.label == 1:
            if require_generator_disjoint and not known(record.generator_id):
                raise ValueError("Cannot certify generator hold-out with UNKNOWN generator")
            if known(record.generator_id):
                generator_splits.setdefault(record.generator_id, set()).add(record.split)
    if any(len(labels) > 1 for labels in label_groups.values()):
        raise ValueError("Cross-label checksum conflict; quarantine required")
    for key in ("checksum", "parent_recording_id"):
        if any(len(splits) > 1 for splits in indexes[key].values()):
            raise ValueError(f"Cross-partition {key} leakage")
    if require_speaker_disjoint and any(len(splits) > 1 for splits in indexes["speaker"].values()):
        raise ValueError("Cross-partition speaker leakage")
    if require_generator_disjoint and any(len(splits) > 1 for splits in generator_splits.values()):
        raise ValueError("Cross-partition generator leakage")
    return {"records": len(records), "manifest_hash": object_hash([r.to_dict() for r in records]),
            "speaker_disjoint_checked": require_speaker_disjoint,
            "generator_disjoint_checked": require_generator_disjoint,
            "speaker_overlap_ids": sum(len(s) > 1 for s in indexes["speaker"].values()),
            "generator_overlap_ids": sum(len(s) > 1 for s in generator_splits.values()),
            "global_parent_lineage_verified": bool(records) and all(r.parent_scope == "VERIFIED" for r in records)}


def select_legacy_development(root: Path, manifest: Path, *, per_class: int, seed: int) -> list[SourceRecord]:
    """Use validation only. A checksum parent is a LOCAL variant key, not true source lineage."""
    if per_class < 1:
        raise ValueError("per_class must be positive")
    rows = [r for r in read_csv(manifest) if r["split"] == "validation" and r["included"].lower() == "true"]
    rng = np.random.default_rng(seed)
    selected = []
    for label in (0, 1):
        candidates = sorted((r for r in rows if int(r["label"]) == label), key=lambda r: r["sample_id"])
        if len(candidates) < per_class:
            raise ValueError("Insufficient validation samples; never fall back to test")
        for index in sorted(rng.choice(len(candidates), per_class, replace=False).tolist()):
            row = candidates[index]
            path = root / row["path"]
            if file_hash(path) != row["sha256"]:
                raise ValueError(f"Source checksum mismatch: {path}")
            info = sf.info(path)
            selected.append(SourceRecord(
                sample_id=row["sample_id"], path=row["path"], parent_recording_id="local-sha256:" + row["sha256"],
                dataset_id="LAVA_LEGACY_INTERNAL", dataset_version="legacy-manifest-v1",
                split="development", label=label, checksum=row["sha256"], sample_rate=info.samplerate,
                duration=info.duration, source_url_or_record=f"{manifest.relative_to(root).as_posix()}#{row['sample_id']}",
                provenance_status="LEGACY_UNKNOWN_SOURCE", parent_scope="LOCAL_BYTES_ONLY"))
    audit_partitions(selected)
    return selected
