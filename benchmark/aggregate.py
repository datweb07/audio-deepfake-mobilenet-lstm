"""Aggregate only completed clean/efficiency artifacts; never invent missing metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import config


FIELDS = [
    "Model", "EER", "F1", "MacroF1", "AUC", "NoiseDeg", "CodecDeg", "ReplayDeg", "Unseen",
    "Params", "ModelSize", "Memory", "Latency", "RTF", "PretrainingStratum",
]


def aggregate(model_names: Iterable[str]) -> Path:
    manifest_metadata = Path(config.DATA_DIR) / "manifests" / "manifest_metadata.json"
    current_manifest_hash = None
    if manifest_metadata.is_file():
        current_manifest_hash = json.loads(manifest_metadata.read_text(encoding="utf-8")).get("manifest_hash")
    rows = []
    for name in model_names:
        clean_path = Path(config.OUTPUTS_DIR) / "benchmark" / "clean" / name / "summary.json"
        efficiency_path = Path(config.OUTPUTS_DIR) / "benchmark" / "efficiency" / name / "summary.json"
        if not clean_path.is_file():
            continue
        clean = json.loads(clean_path.read_text(encoding="utf-8"))
        if clean.get("status") != "BENCHMARKED" or clean.get("manifest_hash") != current_manifest_hash:
            continue
        efficiency = json.loads(efficiency_path.read_text(encoding="utf-8")) if efficiency_path.is_file() else {}
        if efficiency.get("status") != "BENCHMARKED":
            efficiency = {}
        rows.append({
            "Model": name, "EER": clean["eer"], "F1": clean["f1"], "MacroF1": clean["macro_f1"],
            "AUC": clean["roc_auc"], "NoiseDeg": "NOT_RUN", "CodecDeg": "NOT_RUN",
            "ReplayDeg": "NOT_RUN", "Unseen": "NOT_RUN", "Params": efficiency.get("parameter_count", "NOT_RUN"),
            "ModelSize": efficiency.get("serialized_size_bytes", "NOT_RUN"),
            "Memory": efficiency.get("process_rss_mib", "NOT_RUN"),
            "Latency": efficiency.get("model_only", {}).get("mean_seconds", "NOT_RUN"),
            "RTF": efficiency.get("model_only_rtf", "NOT_RUN"),
            "PretrainingStratum": clean.get("pretraining_stratum", "UNKNOWN"),
        })
    destination = Path(config.OUTPUTS_DIR) / "benchmark" / "results.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    return destination
