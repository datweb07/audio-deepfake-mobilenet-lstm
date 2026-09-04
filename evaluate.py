"""Unified independent canonical-test evaluation for any trained LAVA detector."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from src.lava.decision_display import threshold_description
from src.lava.data.loader import load_split
from src.lava.artifacts import load_threshold
from src.lava.registry import create, get_spec, names
from src.lava.score_semantics import decisions_from_p_fake
from src.lava.evaluation_metrics import compute_eer


def evaluate_detector(model_name: str = "mobilenetv3_lstm", *, limit: int | None = None) -> dict[str, object]:
    spec = get_spec(model_name)
    paths, labels = load_split("test")
    if limit is not None:
        per_class = max(1, limit // 2)
        indices = (
            [index for index, label in enumerate(labels) if label == config.REAL_LABEL][:per_class]
            + [index for index, label in enumerate(labels) if label == config.FAKE_LABEL][:per_class]
        )
        paths, labels = [paths[index] for index in indices], [labels[index] for index in indices]
    detector = create(model_name)
    detector.load()
    probabilities = detector.predict_scores(paths)
    y_true = np.asarray(labels, dtype=np.int32)
    if np.unique(y_true).size != 2:
        raise ValueError("Evaluation requires both REAL and FAKE; increase --limit")
    threshold = load_threshold(spec)
    predictions = decisions_from_p_fake(probabilities, threshold)
    matrix = confusion_matrix(y_true, predictions, labels=[config.REAL_LABEL, config.FAKE_LABEL])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    eer, eer_threshold = compute_eer(y_true, probabilities)
    result = {
        "model": model_name,
        "framework": spec.framework,
        "threshold": threshold,
        "samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, pos_label=config.FAKE_LABEL, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, pos_label=config.FAKE_LABEL, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, pos_label=config.FAKE_LABEL, zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "classification_report": classification_report(
            y_true, predictions, labels=[0, 1], target_names=["REAL", "FAKE"], zero_division=0
        ),
    }
    return result


def main(model_name: str = "mobilenetv3_lstm", *, limit: int | None = None) -> None:
    result = evaluate_detector(model_name, limit=limit)
    print(f"Model:      {result['model']} ({result['framework']})")
    print(f"Threshold:  {result['threshold']:.4f}")
    print(threshold_description(get_spec(model_name)))
    print(f"Samples:    {result['samples']}")
    for label, key in (
        ("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
        ("F1", "f1"), ("Macro F1", "macro_f1"), ("ROC-AUC", "roc_auc"), ("EER", "eer"),
    ):
        print(f"{label:10s}: {result[key]:.4f}")
    matrix = result["confusion_matrix"]
    print(f"Confusion Matrix: TN={matrix['tn']} FP={matrix['fp']} FN={matrix['fn']} TP={matrix['tp']}")
    print("\nClassification Report")
    print(result["classification_report"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=names(), default="mobilenetv3_lstm")
    parser.add_argument("--limit", type=int, help="Diagnostic subset only; omit for final evaluation")
    arguments = parser.parse_args()
    main(arguments.model, limit=arguments.limit)
