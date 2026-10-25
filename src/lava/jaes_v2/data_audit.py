"""Decoded identity audit and deliberately simple metadata shortcut diagnostics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .evidence import file_hash, write_csv, write_json
from .provenance import SourceRecord, audit_partitions, load_manifest


def audit_sources(root: Path, records: list[SourceRecord]) -> tuple[dict, list[dict]]:
    partition_audit = audit_partitions(records)
    decoded_groups: dict[str, list[SourceRecord]] = defaultdict(list)
    features = []
    for record in records:
        path = root / record.path
        if file_hash(path) != record.checksum:
            raise ValueError(f"Checksum mismatch: {record.sample_id}")
        values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if not np.isfinite(values).all() or not values.size:
            raise ValueError("Invalid decoded waveform")
        # Exact decoded identity only; neither a perceptual nor speaker fingerprint.
        digest = hashlib.sha256()
        digest.update(f"{sample_rate}:{values.shape}".encode("ascii"))
        digest.update(values.astype("<f4").tobytes())
        decoded_groups[digest.hexdigest()].append(record)
        mono = values.mean(axis=1).astype(np.float64)
        features.append({"sample_id": record.sample_id, "parent_recording_id": record.parent_recording_id,
                         "dataset_id": record.dataset_id, "label": record.label, "split": record.split,
                         "sample_rate": sample_rate, "duration": len(mono) / sample_rate,
                         "rms": float(np.sqrt(np.mean(mono ** 2))), "peak": float(np.max(np.abs(mono))),
                         "silence_fraction_below_1e-4": float(np.mean(np.abs(mono) < 1e-4)),
                         "extension": path.suffix.lower(), "decoded_sha256": digest.hexdigest()})
    conflicts = [group for group in decoded_groups.values() if len({r.split for r in group}) > 1 or len({r.label for r in group}) > 1]
    if conflicts:
        raise ValueError("Decoded-waveform cross-partition or cross-label conflict: " + str([[r.sample_id for r in group] for group in conflicts]))
    summary = {"partition_audit": partition_audit, "decoded_duplicate_groups": sum(len(g) > 1 for g in decoded_groups.values()),
               "decoded_identity_policy": "exact float32 channel-preserving PCM plus sample rate; no resampling",
               "near_duplicate_check": "NOT_IMPLEMENTED; no perceptual-disjointness claim",
               "shortcut_baseline": "NOT_RUN", "class_feature_means": {}}
    keys = ("sample_rate", "duration", "rms", "peak", "silence_fraction_below_1e-4")
    for label in (0, 1):
        selected = [row for row in features if row["label"] == label]
        summary["class_feature_means"][str(label)] = {key: float(np.mean([row[key] for row in selected])) for key in keys} if selected else {}
    # Train-only/DEVELOPMENT diagnostic, explicitly blocked for final or calibration data.
    if features and all(row["split"] in {"train", "development"} for row in features):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        labels = np.array([row["label"] for row in features])
        groups = np.array([row["parent_recording_id"] for row in features])
        if all(len({row["parent_recording_id"] for row in features if row["label"] == label}) >= 3 for label in (0, 1)):
            matrix = np.array([[row[key] for key in keys] for row in features])
            predicted = np.empty(len(labels), dtype=int)
            for train, validation in LeaveOneGroupOut().split(matrix, labels, groups):
                model = make_pipeline(StandardScaler(), LogisticRegression(C=1, solver="liblinear", random_state=0))
                model.fit(matrix[train], labels[train])
                predicted[validation] = model.predict(matrix[validation])
            summary["shortcut_baseline"] = {"scope": "DEVELOPMENT_DIAGNOSTIC_ONLY", "features": list(keys),
                                            "validation": "leave-one-local-parent-out; 5 numeric features; no tuning",
                                            "accuracy": float(np.mean(predicted == labels)), "samples": len(labels)}
    return summary, features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary, features = audit_sources(args.root, load_manifest(args.manifest))
    write_json(args.output / "source_audit.json", summary)
    write_csv(args.output / "source_features.csv", features)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
