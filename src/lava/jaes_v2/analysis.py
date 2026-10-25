"""Paired parent-level risk contrasts; descriptive, never a causal certificate."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.lava.evaluation_metrics import compute_eer

CONDITIONS = ("clean", "A", "B", "A_then_B", "B_then_A")
SCORE_FIELDS = [
    "run_id", "sample_id", "parent_recording_id", "dataset_id", "generator_id", "speaker_id",
    "label", "model_id", "condition_id", "transform_order", "seed", "raw_score", "calibrated_score",
    "threshold", "predicted_label", "correct", "abstain", "stage_used", "preprocessing_latency_ms",
    "model_latency_ms", "end_to_end_latency_ms", "status", "error",
]


def paired_index(rows: Iterable[dict]) -> dict[str, dict[str, dict]]:
    index: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        sample, condition = row["sample_id"], row["condition_id"]
        if condition not in CONDITIONS or condition in index[sample]:
            raise ValueError("Unknown or duplicate paired condition")
        if row["status"] != "OK":
            raise ValueError("Failed inference: complete-case deletion prohibited; analysis unavailable")
        score = float(row["raw_score"])
        label, threshold = int(row["label"]), float(row["threshold"])
        if not np.isfinite(score) or not 0 <= score <= 1 or label not in {0, 1} or not 0 <= threshold <= 1:
            raise ValueError("Invalid score/label/threshold")
        if int(row["predicted_label"]) != int(score >= threshold):
            raise ValueError("Decision inconsistent with fixed threshold")
        index[sample][condition] = row
    if not index:
        raise ValueError("No scores")
    for conditions in index.values():
        if set(conditions) != set(CONDITIONS):
            raise ValueError("Incomplete paired cohort")
        for key in ("parent_recording_id", "label", "threshold", "model_id", "seed", "run_id"):
            if len({str(row[key]) for row in conditions.values()}) != 1:
                raise ValueError(f"Paired identity mismatch: {key}")
    for key in ("model_id", "seed", "run_id", "threshold"):
        if len({str(conditions["clean"][key]) for conditions in index.values()}) != 1:
            raise ValueError(f"Mixed {key}; analyze separate strata")
    return dict(index)


def loss(row: dict, metric: str) -> float:
    label, score = int(row["label"]), float(row["raw_score"])
    if metric == "error":
        return float(int(row["predicted_label"]) != label)
    if metric == "squared_score_error":
        return (score - label) ** 2
    if metric == "log_loss":
        bounded = np.clip(score, 1e-7, 1 - 1e-7)
        return float(-label * np.log(bounded) - (1 - label) * np.log1p(-bounded))
    raise ValueError(metric)


def parent_bootstrap(values: dict[str, list[float]], *, replicates: int, seed: int) -> dict:
    """Uniform parent estimand; each parent's utterance mean receives equal weight."""
    if not values or replicates < 100:
        raise ValueError("Require parents and at least 100 replicates")
    means = np.array([np.mean(values[parent]) for parent in sorted(values)], dtype=np.float64)
    if not np.isfinite(means).all():
        raise ValueError("Nonfinite contrast")
    result = {"estimate": float(means.mean()), "parents": len(means), "ci95": None,
              "ci_status": "INSUFFICIENT_PARENTS" if len(means) < 2 else "DESCRIPTIVE_PERCENTILE"}
    if len(means) >= 2:
        generator = np.random.default_rng(seed)
        draws = means[generator.integers(0, len(means), (replicates, len(means)))].mean(axis=1)
        result["ci95"] = np.quantile(draws, [0.025, 0.975]).tolist()
    return result


def analyze(rows: list[dict], *, replicates: int, seed: int) -> dict:
    paired = paired_index(rows)
    metrics, flips = {}, {}
    for condition in CONDITIONS:
        records = [values[condition] for values in paired.values()]
        labels = np.array([int(row["label"]) for row in records])
        predictions = np.array([int(row["predicted_label"]) for row in records])
        scores = np.array([float(row["raw_score"]) for row in records])
        real, fake = labels == 0, labels == 1
        metrics[condition] = {
            "samples": len(records), "real": int(real.sum()), "fake": int(fake.sum()),
            "error_rate": float(np.mean(labels != predictions)),
            "false_alarm_rate": float(predictions[real].mean()) if real.any() else None,
            "miss_rate": float((1 - predictions[fake]).mean()) if fake.any() else None,
            "squared_score_error": float(np.mean((scores - labels) ** 2)),
            "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
            "pr_auc_average_precision": float(average_precision_score(labels, scores)) if len(set(labels)) == 2 else None,
            "diagnostic_eer": compute_eer(labels, scores)[0] if len(set(labels)) == 2 else None,
        }
        flips[condition] = {"clean_correct_to_error": 0, "clean_error_to_correct": 0,
                            "real_correct_to_false_alarm": 0, "fake_correct_to_miss": 0}
        for values in paired.values():
            clean_error, stress_error = loss(values["clean"], "error"), loss(values[condition], "error")
            if not clean_error and stress_error:
                flips[condition]["clean_correct_to_error"] += 1
                key = "real_correct_to_false_alarm" if int(values["clean"]["label"]) == 0 else "fake_correct_to_miss"
                flips[condition][key] += 1
            if clean_error and not stress_error:
                flips[condition]["clean_error_to_correct"] += 1
    contrasts = []
    for metric in ("error", "squared_score_error", "log_loss"):
        for label in (None, 0, 1):
            for name in ("interaction_A_then_B", "interaction_B_then_A", "order_A_then_B_minus_B_then_A"):
                groups: dict[str, list[float]] = defaultdict(list)
                for values in paired.values():
                    clean = values["clean"]
                    if label is not None and int(clean["label"]) != label:
                        continue
                    risks = {condition: loss(row, metric) for condition, row in values.items()}
                    if name.startswith("order"):
                        contrast = risks["A_then_B"] - risks["B_then_A"]
                    else:
                        joint = "A_then_B" if name == "interaction_A_then_B" else "B_then_A"
                        contrast = risks[joint] - risks["A"] - risks["B"] + risks["clean"]
                    groups[clean["parent_recording_id"]].append(contrast)
                if groups:
                    contrasts.append({"risk": metric, "class": "ALL" if label is None else ("REAL" if label == 0 else "FAKE"),
                                      "contrast": name, **parent_bootstrap(groups, replicates=replicates, seed=seed)})
    return {"scope": "DESCRIPTIVE_PILOT", "conditions": metrics, "prediction_flips": flips, "contrasts": contrasts,
            "bootstrap": {"unit": "parent_recording_id", "estimand": "equal-weight parent mean", "replicates": replicates,
                          "seed": seed, "interval": "percentile; no significance or multiplicity claims"},
            "limitations": ["No generator-shift contrast: generator IDs UNKNOWN", "One legacy checkpoint; no training-seed uncertainty",
                            "No probability calibration fitted", "No causal attribution", "Legacy development data; no independent confirmation"]}
