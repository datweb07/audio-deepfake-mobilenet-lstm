"""Complete-cohort Phase 2 control summaries and prespecified paired risk differences."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.lava.evaluation_metrics import compute_eer
from .analysis import loss, parent_bootstrap

CONTROL_CONDITIONS = ("clean_native", "clean", "headroom_low", "pcm16", "identity", "resample", "codec", "codec_aligned",
                      "local_noise", "fixed_noise", "local_A_then_B", "local_B_then_A", "fixed_A_then_B", "fixed_B_then_A")
EFFECTS = {
    "headroom_vs_native": {"clean": 1, "clean_native": -1},
    "headroom_low_vs_prepared": {"headroom_low": 1, "clean": -1},
    "pcm_vs_prepared": {"pcm16": 1, "clean": -1},
    "identity_vs_pcm": {"identity": 1, "pcm16": -1},
    "resample_vs_prepared": {"resample": 1, "clean": -1},
    "codec_vs_pcm": {"codec": 1, "pcm16": -1},
    "alignment_vs_native_codec": {"codec_aligned": 1, "codec": -1},
    "local_order": {"local_A_then_B": 1, "local_B_then_A": -1},
    "fixed_order": {"fixed_A_then_B": 1, "fixed_B_then_A": -1},
    "noise_policy_order_difference": {"local_A_then_B": 1, "local_B_then_A": -1, "fixed_A_then_B": -1, "fixed_B_then_A": 1},
}


def validate_scores(rows: list[dict], manifest: list[dict]) -> dict:
    expected = {(row["sample_id"], condition) for row in manifest for condition in CONTROL_CONDITIONS}
    if len(rows) != len(expected) or {(row["sample_id"], row["condition_id"]) for row in rows} != expected:
        raise ValueError("Missing/duplicate Phase 2 score records")
    sources = {row["sample_id"]: row for row in manifest}
    index = defaultdict(dict)
    for row in rows:
        if row["status"] != "OK":
            raise ValueError("Failed inference cannot be silently excluded")
        original = sources[row["sample_id"]]
        for field in ("label", "dataset_id", "parent_recording_id"):
            if str(row[field]) != str(original[field]):
                raise ValueError(f"Source/score identity mismatch: {field}")
        score, threshold = float(row["raw_score"]), float(row["threshold"])
        if not np.isfinite(score) or not 0 <= score <= 1 or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid raw score/threshold")
        if int(row["predicted_label"]) != int(score >= threshold):
            raise ValueError("Frozen decision rule violated")
        index[row["sample_id"]][row["condition_id"]] = row
    for field in ("run_id", "model_id", "seed", "threshold"):
        if len({str(row[field]) for row in rows}) != 1:
            raise ValueError(f"Mixed model stratum: {field}")
    return dict(index)


def summarize(rows: list[dict], manifest: list[dict], *, replicates: int, seed: int) -> dict:
    index = validate_scores(rows, manifest)
    metrics = {}
    for condition in CONTROL_CONDITIONS:
        selected = [sample[condition] for sample in index.values()]
        labels = np.array([int(row["label"]) for row in selected])
        predictions = np.array([int(row["predicted_label"]) for row in selected])
        scores = np.array([float(row["raw_score"]) for row in selected])
        real, fake = labels == 0, labels == 1
        metrics[condition] = {"recordings": len(labels), "real": int(real.sum()), "fake": int(fake.sum()),
                              "error_rate": float(np.mean(labels != predictions)),
                              "false_alarm_rate": float(predictions[real].mean()) if real.any() else None,
                              "miss_rate": float((1 - predictions[fake]).mean()) if fake.any() else None,
                              "mean_raw_score": float(scores.mean()),
                              "roc_auc": float(roc_auc_score(labels, scores)) if real.any() and fake.any() else None,
                              "diagnostic_eer": compute_eer(labels, scores)[0] if real.any() and fake.any() else None,
                              "pr_auc_average_precision": float(average_precision_score(labels, scores)) if real.any() and fake.any() else None}
    contrasts = []
    for risk in ("error", "squared_score_error"):
        for label in (None, 0, 1):
            for effect, coefficients in EFFECTS.items():
                clusters = defaultdict(list)
                for sample in index.values():
                    reference = sample["clean"]
                    if label is not None and int(reference["label"]) != label:
                        continue
                    value = sum(coefficient * loss(sample[condition], risk) for condition, coefficient in coefficients.items())
                    clusters[reference["parent_recording_id"]].append(value)
                if clusters:
                    contrasts.append({"risk": risk, "class": "ALL" if label is None else str(label), "effect": effect,
                                      **parent_bootstrap(clusters, replicates=replicates, seed=seed)})
    return {"scope": "EXPLORATORY_LEGACY_DEVELOPMENT", "primary_estimand": "fixed_absolute order error difference; equal local-parent weight",
            "metrics": metrics, "contrasts": contrasts, "bootstrap_replicates": replicates,
            "limitations": ["No verified independent original-parent or speaker lineage", "No model-relative unseen-generator claim",
                            "Intervals descriptive; no multiplicity-adjusted significance", "Squared raw-score error is not evidence of probability calibration",
                            "Encoder/decoder component delays are not separately identified"]}
