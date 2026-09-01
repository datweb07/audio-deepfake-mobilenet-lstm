"""Canonical per-detector clean evaluation with per-sample P(FAKE) evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

import config
from src.lava.data.loader import load_split
from src.lava.artifacts import artifact_readiness, load_json, load_threshold, write_json_atomic
from src.lava.data.manifest import MANIFEST_METADATA
from src.lava.registry import create, get_spec
from src.lava.score_semantics import decisions_from_p_fake
from src.lava.evaluation_metrics import compute_eer


def _manifest_hash() -> str:
    with MANIFEST_METADATA.open("r", encoding="utf-8") as handle:
        return str(json.load(handle)["manifest_hash"])


def run_clean(model_name: str, *, limit: int | None = None) -> dict[str, object]:
    spec = get_spec(model_name)
    ready, missing = artifact_readiness(spec)
    if not ready:
        raise FileNotFoundError(f"Detector artifacts are not ready for {model_name}: {missing}")
    metadata = load_json(spec.metadata_artifact)
    current_hash = _manifest_hash()
    if metadata.get("training_manifest_hash") != current_hash:
        raise RuntimeError(
            f"{model_name} was not trained on current canonical manifest {current_hash}. Retrain before benchmarking."
        )
    paths, labels = load_split("test")
    selected_indices = list(range(len(paths)))
    if limit is not None:
        per_class = max(1, limit // 2)
        selected_indices = (
            [index for index, label in enumerate(labels) if label == config.REAL_LABEL][:per_class]
            + [index for index, label in enumerate(labels) if label == config.FAKE_LABEL][:per_class]
        )
        paths, labels = [paths[index] for index in selected_indices], [labels[index] for index in selected_indices]
    detector = create(model_name)
    detector.load()
    scores = detector.predict_scores(paths)
    y_true = np.asarray(labels, dtype=np.int32)
    if np.unique(y_true).size != 2:
        raise ValueError("Clean benchmark requires both classes")
    threshold = load_threshold(spec)
    predictions = decisions_from_p_fake(scores, threshold)
    output_dir = Path(config.OUTPUTS_DIR) / "benchmark" / "clean" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "scores.csv"
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "p_fake", "threshold", "prediction", "correct"])
        with (Path(config.DATA_DIR) / "manifests" / "split_manifest.csv").open("r", encoding="utf-8", newline="") as source:
            test_rows = [row for row in csv.DictReader(source) if row["split"] == "test"]
        if limit is not None:
            test_rows = [test_rows[index] for index in selected_indices]
        for row, label, score, prediction in zip(test_rows, y_true, scores, predictions):
            writer.writerow([row["sample_id"], int(label), float(score), threshold, int(prediction), int(label == prediction)])
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    eer, eer_threshold = compute_eer(y_true, scores)
    summary = {
        "status": "BENCHMARKED" if limit is None else "SMOKE_TESTED",
        "detector": model_name,
        "framework": spec.framework,
        "manifest_hash": current_hash,
        "pretraining_stratum": metadata.get("pretraining_stratum", "UNKNOWN"),
        "samples": len(y_true),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "scores_path": str(score_path.resolve()),
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary
