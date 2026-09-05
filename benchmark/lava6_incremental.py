"""Incrementally benchmark only ShuffleNetV2 and preserve authoritative LAVA-5 runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

import config
from benchmark.lava5 import metrics, read_csv, sha256, verify_protocol, write_csv
from benchmark.lava5_runtime import Runtime, efficiency
from src.lava.artifacts import artifact_diagnostics, load_threshold, write_json_atomic
from src.lava.data.manifest import validate_manifest_files
from src.lava.registry import get_spec
from src.lava.score_semantics import validate_p_fake

ROOT = Path(config.BASE_DIR)
LAVA5 = ROOT / "outputs/lava_5"
OUTPUT = ROOT / "outputs/lava_6"
DIAGNOSTIC = "diagnostic_100"
MODEL = "shufflenetv2_lstm"
VERSION = "lava6-incremental-v1"


def _audit() -> dict:
    manifest = validate_manifest_files()
    spec = get_spec(MODEL)
    issues = artifact_diagnostics(spec)
    if issues:
        raise ValueError("; ".join(issues))
    metadata = json.loads(spec.metadata_artifact.read_text(encoding="utf-8"))
    training_hash = metadata.get("training_manifest_hash", metadata.get("manifest_hash", "UNKNOWN"))
    paths = (spec.model_artifact, spec.threshold_artifact, spec.metadata_artifact)
    return {
        "model": MODEL,
        "registry_name": MODEL,
        "group": spec.group,
        "framework": spec.framework,
        "inference_framework": "tensorflow",
        "artifact": str(spec.model_artifact.relative_to(ROOT)),
        "artifact_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
        "size_bytes": spec.model_artifact.stat().st_size,
        "threshold": load_threshold(spec),
        "threshold_path": str(spec.threshold_artifact.relative_to(ROOT)),
        "threshold_source": metadata.get("threshold_source", "UNKNOWN"),
        "metadata_path": str(spec.metadata_artifact.relative_to(ROOT)),
        "checkpoint_origin": "LAVA_training",
        "trained_with_lava_pipeline": True,
        "original_training_dataset": "canonical LAVA dataset",
        "original_checkpoint_source": "Kaggle LAVA scratch training run",
        "training_manifest_hash": training_hash,
        "manifest_compatibility": "MANIFEST_MATCHED" if training_hash == manifest["manifest_hash"] else "MANIFEST_MISMATCH",
        "duration": spec.audio_duration,
        "sample_rate": spec.sample_rate,
        "input_type": spec.input_type,
        "label_mapping": metadata.get("label_mapping"),
        "score_semantics": metadata.get("score_semantics"),
        "initialization": metadata.get("initialization"),
        "training_policy": metadata.get("training_policy"),
        "source_metadata": metadata,
        "load_status": "NOT_RUN",
    }


def prepare(output: Path) -> None:
    old = verify_protocol(LAVA5)
    old_diagnostic = verify_protocol(LAVA5 / DIAGNOSTIC)
    audit = _audit()
    if audit["manifest_compatibility"] != "MANIFEST_MATCHED":
        raise ValueError("ShuffleNet training manifest does not match the canonical manifest")
    protocol = {
        "version": VERSION,
        "title": "LAVA SIX-DETECTOR INCREMENTAL BENCHMARK",
        "manifest": old["manifest"],
        "manifest_file_hashes": old["manifest_file_hashes"],
        "models": old["models"] + [audit],
        "model_order": [item["model"] for item in old["models"]] + [MODEL],
        "test_samples": old["test_samples"],
        "threshold_policy": old["threshold_policy"],
        "device": "CPU",
        "precision": "float32",
        "batch_size": 1,
        "threads": 1,
        "source_lava5_protocol": str((LAVA5 / "protocol/protocol.json").relative_to(ROOT)),
        "source_lava5_protocol_sha256": sha256(LAVA5 / "protocol/protocol.json"),
        "existing_models_reexecuted": [],
        "new_model_executed": MODEL,
        "unseen": "NOT_AVAILABLE",
        "physical_replay": "NOT_AVAILABLE",
        "robustness_scope": "DIAGNOSTIC_SUBSET_100",
        "note": "Four LAVA-trained lightweight artifacts and two external references; heterogeneous initialization and training provenance.",
    }
    target = output / "protocol/protocol.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Six-model protocol changed; preserve this output and choose another root")
    write_json_atomic(target, protocol)
    write_csv(output / "protocol/test_samples.csv", read_csv(LAVA5 / "protocol/test_samples.csv"))
    diagnostic = dict(
        protocol,
        evaluation_scope="DIAGNOSTIC_SUBSET",
        canonical_test_samples=old["test_samples"],
        test_samples=old_diagnostic["test_samples"],
        subset_selection=old_diagnostic["subset_selection"],
        source_lava5_diagnostic_protocol=str((LAVA5 / DIAGNOSTIC / "protocol/protocol.json").relative_to(ROOT)),
        source_lava5_diagnostic_protocol_sha256=sha256(LAVA5 / DIAGNOSTIC / "protocol/protocol.json"),
    )
    write_json_atomic(output / DIAGNOSTIC / "protocol/protocol.json", diagnostic)
    write_csv(
        output / DIAGNOSTIC / "protocol/test_samples.csv",
        read_csv(LAVA5 / DIAGNOSTIC / "protocol/test_samples.csv"),
    )
    write_json_atomic(output / "protocol/shufflenet_artifact_audit.json", audit)
    print(json.dumps({"status": "PASS", "model": MODEL, "manifest": audit["manifest_compatibility"], "old_models_rerun": 0}, indent=2))


def verify(output: Path) -> dict:
    protocol = json.loads((output / "protocol/protocol.json").read_text(encoding="utf-8"))
    if protocol["source_lava5_protocol_sha256"] != sha256(LAVA5 / "protocol/protocol.json"):
        raise ValueError("Historical LAVA-5 protocol changed")
    audit = protocol["models"][-1]
    if audit["model"] != MODEL:
        raise ValueError("Incremental protocol has an unexpected new detector")
    for path, digest in audit["artifact_hashes"].items():
        if sha256(ROOT / path.replace("\\", "/")) != digest:
            raise ValueError(f"ShuffleNet artifact changed: {path}")
    return protocol


def _runtime_and_load(output: Path) -> tuple[Runtime, dict]:
    protocol = verify(output)
    audit = protocol["models"][-1]
    runtime = Runtime(MODEL)
    runtime.load()
    rows = read_csv(output / "protocol/test_samples.csv")
    probes = [next(row for row in rows if int(row["label"]) == label) for label in (0, 1)]
    scores = [runtime.predict(str(ROOT / row["path"])) for row in probes]
    validate_p_fake(scores)
    load = {
        "model": MODEL,
        "load_status": "PASS",
        "adapter_parity": "PASS",
        "scores": scores,
        "parameter_count": runtime.parameter_count(),
        "parameter_count_source": runtime.parameter_count_source,
        "rss_before_load": runtime.rss_before,
        "rss_after_load": runtime.rss_after,
        "load_seconds": runtime.load_seconds,
        "framework_version": runtime.version,
        "input_shape": runtime.input_shape,
        "output_shape": runtime.output_shape,
    }
    if load["parameter_count"] != audit["source_metadata"]["parameter_count"]:
        raise ValueError("Loaded parameter count does not match ShuffleNet metadata")
    write_json_atomic(output / "protocol/shufflenetv2_lstm_load.json", load)
    audit_path = output / "protocol/shufflenet_artifact_audit.json"
    audited = json.loads(audit_path.read_text(encoding="utf-8"))
    audited.update(
        load_status="PASS",
        validated_parameter_count=load["parameter_count"],
        validated_input_shape=load["input_shape"],
        validated_output_shape=load["output_shape"],
        validated_score_range=[min(scores), max(scores)],
    )
    write_json_atomic(audit_path, audited)
    return runtime, audit


def _write_result(output: Path, runtime: Runtime, audit: dict, rows: list[dict], *, suite: str, condition: str, source_root: Path | None = None) -> None:
    folder = output / ("clean" if suite == "clean" else f"robustness/{suite}/{condition}") / MODEL
    summary_path, scores_path = folder / "summary.json", folder / "scores.csv"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("scores_sha256") == sha256(scores_path) and summary.get("samples") == len(rows):
            print(f"Verified existing ShuffleNet {suite}/{condition}")
            return
        raise ValueError(f"Existing ShuffleNet result is inconsistent: {folder}")
    condition_map = None
    if suite != "clean":
        assert source_root is not None
        condition_path = source_root / "protocol/conditions" / suite / f"{condition}.csv"
        condition_map = {row["sample_id"]: row for row in read_csv(condition_path)}
        if set(condition_map) != {row["sample_id"] for row in rows}:
            raise ValueError("Historical stress manifest does not match diagnostic sample IDs")
    results = []
    started = time.perf_counter()
    folder.mkdir(parents=True, exist_ok=True)
    pending = folder / "scores.pending.csv"
    with pending.open("w", encoding="utf-8", newline="") as stream:
        import csv
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "true_label", "p_fake", "threshold", "predicted_label", "correct"])
        writer.writeheader()
        for index, row in enumerate(rows):
            if condition_map is None:
                path, expected_hash = ROOT / row["path"], row["sha256"]
            else:
                entry = condition_map[row["sample_id"]]
                if entry["source_sha256"] != row["sha256"] or int(entry["label"]) != int(row["label"]):
                    raise ValueError("Stress/source identity mismatch")
                path, expected_hash = source_root / entry["path"], entry["sha256"]
            if sha256(path) != expected_hash:
                raise ValueError(f"Input checksum changed: {path}")
            score = runtime.predict(str(path))
            label, threshold = int(row["label"]), float(audit["threshold"])
            predicted = int(score >= threshold)
            result = {"sample_id": row["sample_id"], "true_label": label, "p_fake": score, "threshold": threshold, "predicted_label": predicted, "correct": int(predicted == label)}
            results.append(result)
            writer.writerow(result)
            if index % 100 == 0:
                stream.flush()
                print(f"ShuffleNet {suite}/{condition}: {index + 1}/{len(rows)}", flush=True)
    os.replace(pending, scores_path)
    summary = {
        "model": MODEL,
        "registry_name": MODEL,
        "condition": condition,
        "status": "BENCHMARKED" if suite == "clean" else "DIAGNOSTIC",
        "samples": len(rows),
        "threshold": audit["threshold"],
        "manifest_hash": verify(output)["manifest"]["manifest_hash"],
        "scores_sha256": sha256(scores_path),
        "elapsed_seconds": time.perf_counter() - started,
        **metrics([row["true_label"] for row in results], [row["p_fake"] for row in results], audit["threshold"]),
    }
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


def clean(output: Path) -> None:
    runtime, audit = _runtime_and_load(output)
    _write_result(output, runtime, audit, read_csv(output / "protocol/test_samples.csv"), suite="clean", condition="clean")


def diagnostic_clean(output: Path) -> None:
    full_rows = read_csv(output / "clean" / MODEL / "scores.csv")
    by_id = {row["sample_id"]: row for row in full_rows}
    subset = read_csv(output / DIAGNOSTIC / "protocol/test_samples.csv")
    selected = [by_id[row["sample_id"]] for row in subset]
    destination = output / DIAGNOSTIC / "clean" / MODEL
    write_csv(destination / "scores.csv", selected)
    threshold = float(selected[0]["threshold"])
    summary = {
        "model": MODEL,
        "registry_name": MODEL,
        "condition": "clean",
        "status": "DIAGNOSTIC",
        "samples": len(selected),
        "threshold": threshold,
        "manifest_hash": verify(output)["manifest"]["manifest_hash"],
        "parent_scores_sha256": sha256(output / "clean" / MODEL / "scores.csv"),
        "scores_sha256": sha256(destination / "scores.csv"),
        **metrics([int(row["true_label"]) for row in selected], [float(row["p_fake"]) for row in selected], threshold),
    }
    write_json_atomic(destination / "summary.json", summary)


def run_stress(output: Path, suite: str) -> None:
    runtime, audit = _runtime_and_load(output)
    source_root = LAVA5 / DIAGNOSTIC
    rows = read_csv(output / DIAGNOSTIC / "protocol/test_samples.csv")
    conditions = [path.stem for path in sorted((source_root / "protocol/conditions" / suite).glob("*.csv"))]
    if not conditions:
        raise ValueError(f"No historical {suite} condition exists")
    for condition in conditions:
        _write_result(output / DIAGNOSTIC, runtime, audit, rows, suite=suite, condition=condition, source_root=source_root)


def run_efficiency(output: Path) -> None:
    runtime, _ = _runtime_and_load(output)
    path = next(row for row in read_csv(output / "protocol/test_samples.csv") if int(row["label"]) == 0)["path"]
    target = output / "efficiency" / MODEL / "summary.json"
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous.get("runs", 0) >= 50 and previous.get("warmup", 0) >= 10:
            print("Verified existing ShuffleNet efficiency")
            return
        raise ValueError("Existing ShuffleNet efficiency is not official-protocol compatible")
    write_json_atomic(target, efficiency(runtime, str(ROOT / path), 3.0, warmup=10, runs=50))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "load", "clean", "diagnostic-clean", "noise", "compression", "replay", "efficiency"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        prepare(output)
    elif args.command == "load":
        _runtime_and_load(output)
    elif args.command == "clean":
        clean(output)
    elif args.command == "diagnostic-clean":
        diagnostic_clean(output)
    elif args.command == "efficiency":
        run_efficiency(output)
    else:
        run_stress(output, args.command)


if __name__ == "__main__":
    main()
