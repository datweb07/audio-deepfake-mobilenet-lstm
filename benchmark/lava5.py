"""Reproducible, inference-only LAVA-5 execution. Never imports a trainer.

Run `python -m benchmark.lava5 prepare`, then `... run --suite clean`.
Existing completed outputs are verified and reused, never silently overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

import config
from src.lava.data.manifest import validate_manifest_files
from src.lava.registry import get_spec
from src.lava.artifacts import load_threshold, write_json_atomic
from src.lava.evaluation_metrics import compute_eer
from src.lava.score_semantics import decisions_from_p_fake, validate_p_fake

ROOT = Path(config.BASE_DIR)
DEFAULT_OUTPUT = ROOT / "outputs/lava_5"
MODELS = {
    "mobilenetv3_lstm": "mobilenetv3_lstm",
    "efficientnet_b0_lstm": "efficientnet_b0_lstm",
    "mnasnet_lstm": "mnasnet_lstm",
    "rawnet2": "rawnet2_pretrained",
    "aasist": "aasist_pretrained",
}
VERSION = "lava5-inference-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_rows():
    return [r for r in read_csv(ROOT / "data/manifests/split_manifest.csv") if r["split"] == "test"]


def metrics(y, scores, threshold):
    y = np.asarray(y, dtype=int)
    scores = validate_p_fake(scores)
    prediction = decisions_from_p_fake(scores, threshold)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    eer, _ = compute_eer(y, scores)
    return dict(accuracy=float(accuracy_score(y, prediction)),
                precision=float(precision_score(y, prediction, zero_division=0)),
                recall=float(recall_score(y, prediction, zero_division=0)),
                f1=float(f1_score(y, prediction, zero_division=0)),
                macro_f1=float(f1_score(y, prediction, average="macro", zero_division=0)),
                roc_auc=float(roc_auc_score(y, scores)), eer=eer,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def prepare(output):
    manifest = validate_manifest_files()
    rows = test_rows()
    # Recheck actual bytes, not just internal CSV consistency.
    for i, row in enumerate(rows):
        path = ROOT / row["path"]
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Source audio checksum changed: {path}")
        if i % 500 == 0:
            print(f"Integrity: {i}/{len(rows)}", flush=True)
    audits = []
    for name, registry_name in MODELS.items():
        spec = get_spec(registry_name)
        metadata = json.loads(spec.metadata_artifact.read_text(encoding="utf-8"))
        external = registry_name.endswith("_pretrained")
        paths = [spec.model_artifact, spec.threshold_artifact, spec.metadata_artifact]
        if name == "mobilenetv3_lstm":
            paths += [p for p in spec.model_artifact.parent.glob("lava_mobilenetv3_lstm.weights.*")]
        artifact_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
        training_hash = metadata.get("training_manifest_hash", metadata.get("manifest_hash", "UNKNOWN"))
        audits.append(dict(model=name, registry_name=registry_name, group=spec.group,
            framework=spec.framework, inference_framework="onnxruntime" if external else "tensorflow",
            artifact=str(spec.model_artifact.relative_to(ROOT)), artifact_hashes=artifact_hashes,
            size_bytes=spec.model_artifact.stat().st_size,
            threshold=load_threshold(spec), threshold_path=str(spec.threshold_artifact.relative_to(ROOT)),
            threshold_source=metadata.get("threshold_source", "UNKNOWN"),
            metadata_path=str(spec.metadata_artifact.relative_to(ROOT)),
            checkpoint_origin="external_reference" if external else "LAVA_training",
            trained_with_lava_pipeline=not external,
            original_training_dataset="UNKNOWN" if external else "canonical LAVA dataset",
            original_checkpoint_source=metadata.get("source", "UNKNOWN"),
            training_manifest_hash=training_hash,
            manifest_compatibility="EXTERNAL_CHECKPOINT" if external else (
                "MANIFEST_MATCHED" if training_hash == manifest["manifest_hash"] else "LEGACY_ARTIFACT"),
            duration=spec.audio_duration, sample_rate=spec.sample_rate, input_type=spec.input_type,
            label_mapping={"REAL": 0, "FAKE": 1}, score_semantics=metadata.get("score_semantics"),
            source_metadata=metadata, load_status="NOT_RUN"))
    protocol = dict(version=VERSION, title="LAVA-5 INTERIM BENCHMARK", manifest=manifest,
        manifest_file_hashes={p.name: sha256(p) for p in (ROOT / "data/manifests").glob("*") if p.is_file()},
        models=audits, test_samples=len(rows), excluded={"shufflenetv2_lstm": "TRAINING_PENDING"},
        threshold_policy="Existing thresholds fixed before test; no test calibration",
        device="CPU", precision="float32", batch_size=1, threads=1,
        note="Unequal training provenance, initialization and native duration; not a controlled training comparison.",
        unseen="NOT_AVAILABLE", physical_replay="NOT_AVAILABLE")
    target = output / "protocol/protocol.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Protocol changed: use a new --output directory; historical results retained")
    else:
        write_json_atomic(target, protocol)
        write_csv(output / "protocol/test_samples.csv", rows)
    print(json.dumps({"integrity": "PASS", "samples": len(rows), "manifest_hash": manifest["manifest_hash"]}), flush=True)


def verify_protocol(output):
    protocol = json.loads((output / "protocol/protocol.json").read_text(encoding="utf-8"))
    for filename, digest in protocol["manifest_file_hashes"].items():
        if sha256(ROOT / "data/manifests" / filename) != digest:
            raise ValueError("Canonical manifest changed during benchmark")
    for audit in protocol["models"]:
        for path, digest in audit["artifact_hashes"].items():
            if sha256(ROOT / path.replace("\\", "/")) != digest:
                raise ValueError(f"Artifact changed during benchmark: {path}")
    environment_path = output / "protocol/execution_environment.json"
    if environment_path.exists():
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        for path, digest in environment.get("production_source_sha256", {}).items():
            if sha256(ROOT / path) != digest:
                raise ValueError(f"Production inference source changed during benchmark: {path}")
    return protocol


def prepare_diagnostic(parent, samples=100):
    """Select from labels/IDs only, never from model predictions or errors."""
    protocol = verify_protocol(parent)
    rows = read_csv(parent / "protocol/test_samples.csv")
    if not 2 <= samples < len(rows):
        raise ValueError("Diagnostic subset must contain >=2 and fewer than all test samples")
    rng = np.random.default_rng(42)
    real = [i for i, r in enumerate(rows) if int(r["label"]) == 0]
    fake = [i for i, r in enumerate(rows) if int(r["label"]) == 1]
    n_real = min(len(real), max(1, min(samples-1, round(samples * len(real) / len(rows)))))
    selected = sorted(list(rng.choice(real, n_real, replace=False)) + list(rng.choice(fake, samples-n_real, replace=False)))
    subset = [rows[i] for i in selected]
    output = parent / f"diagnostic_{samples}"
    protocol = dict(protocol, evaluation_scope="DIAGNOSTIC_SUBSET", canonical_test_samples=len(rows), test_samples=samples,
                    subset_selection=dict(seed=42, method="stratified proportional, without replacement, manifest order retained", real=n_real, fake=samples-n_real), parent_output=str(parent.resolve()))
    target = output / "protocol/protocol.json"
    if target.exists() and json.loads(target.read_text()) != protocol:
        raise ValueError("Diagnostic protocol changed")
    write_json_atomic(target, protocol)
    write_csv(output / "protocol/test_samples.csv", subset)
    for name in ["execution_environment", "benchmark_implementation", "rawnet2_native_audit", "aasist_native_audit", "reference_origin_hashes", "unit_test_results"] + [f"{n}_load" for n in MODELS]:
        source = parent / "protocol" / f"{name}.json"
        if source.exists():
            write_json_atomic(output / "protocol" / source.name, json.loads(source.read_text()))
    write_json_atomic(parent / "protocol/robustness_execution_plan.json", dict(scope="DIAGNOSTIC_SUBSET", samples=samples,
        reason="User selected fixed-subset robustness after measured full CPU runtime estimate", output=str(output.resolve()),
        degradation_baseline="same selected sample IDs from completed full clean scores", full_robustness_status="NOT_RUN"))
    # A rerun after full clean completion derives the exact paired clean baseline.
    for name in MODELS:
        source = parent / "clean" / name
        if not (source / "summary.json").exists():
            continue
        original = json.loads((source / "summary.json").read_text())
        if original["status"] != "BENCHMARKED" or sha256(source / "scores.csv") != original["scores_sha256"]:
            raise ValueError("Parent clean result not verified")
        original_rows = read_csv(source / "scores.csv")
        if [r["sample_id"] for r in original_rows] != [r["sample_id"] for r in rows]:
            raise ValueError("Parent clean alignment mismatch")
        selected_rows = [original_rows[i] for i in selected]
        destination = output / "clean" / name
        write_csv(destination / "scores.csv", selected_rows)
        summary = dict(model=name, registry_name=MODELS[name], condition="clean", status="DIAGNOSTIC", samples=samples,
            threshold=original["threshold"], manifest_hash=original["manifest_hash"], parent_scores_sha256=original["scores_sha256"],
            scores_sha256=sha256(destination / "scores.csv"),
            **metrics([int(r["true_label"]) for r in selected_rows], [float(r["p_fake"]) for r in selected_rows], original["threshold"]))
        write_json_atomic(destination / "summary.json", summary)
    print(json.dumps(dict(output=str(output), scope="DIAGNOSTIC", real=n_real, fake=samples-n_real)), flush=True)


def worker(output, name, suite, limit=None):
    from benchmark.lava5_runtime import Runtime, efficiency
    protocol = verify_protocol(output)
    audit = next(a for a in protocol["models"] if a["model"] == name)
    efficiency_path = output / "efficiency" / name / "summary.json"
    if suite == "efficiency" and efficiency_path.exists():
        previous = json.loads(efficiency_path.read_text())
        if (previous.get("status") != "BENCHMARKED" or previous.get("device") != "CPU"
                or previous.get("batch_size") != 1 or previous.get("runs", 0) < 50
                or previous.get("warmup", 0) < 10):
            raise ValueError("Existing efficiency result is incompatible; use a new output directory")
        for component in ("model_only", "preprocessing", "end_to_end"):
            if len(previous[component]["runs_ms"]) != previous["runs"]:
                raise ValueError("Incomplete existing timing samples")
        print(f"Preserved existing efficiency measurement: {name}", flush=True)
        return
    rows = read_csv(output / "protocol/test_samples.csv")
    if limit:
        rows = [r for label in (0, 1) for r in [q for q in rows if int(q["label"]) == label][:max(1, limit // 2)]]
    runtime = Runtime(audit["registry_name"])
    try:
        runtime.load()
    except Exception as exc:
        write_json_atomic(output / "protocol/failures" / f"{name}_{time.time_ns()}.json",
                          dict(status="BLOCKED_ARTIFACT", model=name, error=f"{type(exc).__name__}: {exc}"))
        raise
    probe_rows = [next(r for r in rows if int(r["label"]) == label) for label in (0, 1)]
    probe = [runtime.predict(str(ROOT / r["path"])) for r in probe_rows]
    public = runtime.detector.predict_scores([str(ROOT / r["path"]) for r in probe_rows])
    np.testing.assert_allclose(probe, public, atol=2e-5, rtol=2e-5)
    load_result = dict(model=name, load_status="PASS", adapter_parity="PASS", scores=probe,
        parameter_count=runtime.parameter_count(), parameter_count_source=runtime.parameter_count_source,
        rss_before_load=runtime.rss_before, rss_after_load=runtime.rss_after,
        load_seconds=runtime.load_seconds, framework_version=runtime.version,
        input_shape=runtime.input_shape, output_shape=runtime.output_shape)
    write_json_atomic(output / "protocol" / f"{name}_load.json", load_result)
    if suite == "load":
        return
    if suite == "efficiency":
        write_json_atomic(efficiency_path, efficiency(runtime, str(ROOT / probe_rows[0]["path"]), audit["duration"]))
        return
    conditions = ["clean"] if suite == "clean" else [p.stem for p in sorted((output / "protocol/conditions" / suite).glob("*.csv"))]
    if not conditions:
        raise ValueError(f"No generated conditions for {suite}")
    for condition in conditions:
        folder = output / ("clean" if suite == "clean" else f"robustness/{suite}/{condition}") / name
        if limit:
            folder = output / "diagnostic" / suite / condition / name
        folder.mkdir(parents=True, exist_ok=True)
        summary_path, scores_path = folder / "summary.json", folder / "scores.csv"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary.get("scores_sha256") == sha256(scores_path) and summary.get("samples") == len(rows):
                print(f"Verified existing {name}/{condition}", flush=True)
                continue
            raise ValueError(f"Existing result inconsistent: {folder}")
        if scores_path.exists():
            # Recover a crash between the atomic score rename and summary write.
            # The full prefix is validated below before its summary is regenerated.
            recovery = folder / "scores.pending.csv"
            if recovery.exists():
                raise ValueError(f"Ambiguous unfinalized scores require inspection: {folder}")
            scores_path.rename(recovery)
        condition_map = None
        if suite != "clean":
            condition_map = {r["sample_id"]: r for r in read_csv(output / "protocol/conditions" / suite / f"{condition}.csv")}
            if set(condition_map) != {r["sample_id"] for r in read_csv(output / "protocol/test_samples.csv")}:
                raise ValueError("Condition does not cover exactly the canonical test set")
        results = []
        start = time.perf_counter()
        pending = folder / "scores.pending.csv"
        # Only complete, validated prefixes can resume. A torn final CSV row
        # fails closed instead of quietly skipping a sample.
        if pending.exists():
            results = read_csv(pending)
            if len(results) > len(rows):
                raise ValueError("Pending result has too many samples")
            for previous, expected in zip(results, rows):
                if previous["sample_id"] != expected["sample_id"] or int(previous["true_label"]) != int(expected["label"]):
                    raise ValueError("Pending result does not match canonical prefix")
                previous.update(true_label=int(previous["true_label"]), p_fake=float(previous["p_fake"]),
                    threshold=float(previous["threshold"]), predicted_label=int(previous["predicted_label"]), correct=int(previous["correct"]))
                validate_p_fake(previous["p_fake"])
                if previous["threshold"] != audit["threshold"] or previous["predicted_label"] != int(previous["p_fake"] >= audit["threshold"]):
                    raise ValueError("Pending result decision contract changed")
        with pending.open("a" if results else "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_id", "true_label", "p_fake", "threshold", "predicted_label", "correct"])
            if not results:
                writer.writeheader()
            completed_prefix = len(results)
            for i, row in enumerate(rows):
                if condition_map is None:
                    path, expected_hash = ROOT / row["path"], row["sha256"]
                else:
                    entry = condition_map[row["sample_id"]]
                    if entry["source_sha256"] != row["sha256"] or int(entry["label"]) != int(row["label"]):
                        raise ValueError("Stress/source identity or label mismatch")
                    path, expected_hash = output / entry["path"], entry["sha256"]
                if sha256(path) != expected_hash:
                    raise ValueError(f"Audio changed: {path}")
                if i < completed_prefix:
                    continue
                score = runtime.predict(str(path))
                label, threshold = int(row["label"]), audit["threshold"]
                predicted = int(score >= threshold)
                result = dict(sample_id=row["sample_id"], true_label=label, p_fake=score,
                              threshold=threshold, predicted_label=predicted, correct=int(predicted == label))
                results.append(result)
                writer.writerow(result)
                if i % 100 == 0:
                    stream.flush()
                    print(f"{name}/{condition} {i + 1}/{len(rows)}; {time.perf_counter() - start:.1f}s", flush=True)
        os.replace(pending, scores_path)
        summary = dict(model=name, registry_name=audit["registry_name"], condition=condition,
            status="DIAGNOSTIC" if limit or protocol.get("evaluation_scope") == "DIAGNOSTIC_SUBSET" else "BENCHMARKED", samples=len(rows),
            threshold=audit["threshold"], manifest_hash=protocol["manifest"]["manifest_hash"],
            scores_sha256=sha256(scores_path), elapsed_seconds=time.perf_counter() - start,
            **metrics([r["true_label"] for r in results], [r["p_fake"] for r in results], audit["threshold"]))
        write_json_atomic(summary_path, summary)
        print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "worker", "stress", "report", "execute", "diagnostic-prepare", "diagnostic-execute"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--suite", choices=["load", "clean", "noise", "compression", "replay", "efficiency"], default="clean")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--diagnostic-samples", type=int, default=100)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must include both classes (>=2)")
    if args.command == "diagnostic-execute":
        # Deliberately require full clean first; do not silently downgrade it.
        for name in MODELS:
            if not (args.output / "clean" / name / "summary.json").is_file():
                raise ValueError("Complete the full five-model clean run before diagnostic-execute")
        prepare_diagnostic(args.output, args.diagnostic_samples)
        subset_output = args.output / f"diagnostic_{args.diagnostic_samples}"
        for suite in ("noise", "compression", "replay"):
            for command in ("stress", "run"):
                subprocess.run([sys.executable, "-m", "benchmark.lava5", command, "--suite", suite, "--output", str(subset_output)], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-m", "benchmark.lava5", "run", "--suite", "efficiency", "--output", str(args.output)], cwd=ROOT, check=True)
        for name in MODELS:
            source = args.output / "efficiency" / name / "summary.json"
            measured = json.loads(source.read_text())
            measured["inherited_from"] = str(source)
            measured["inherited_sha256"] = sha256(source)
            write_json_atomic(subset_output / "efficiency" / name / "summary.json", measured)
        for directory in (args.output, subset_output):
            subprocess.run([sys.executable, "-m", "benchmark.lava5", "report", "--output", str(directory)], cwd=ROOT, check=True)
    elif args.command == "diagnostic-prepare":
        prepare_diagnostic(args.output, args.diagnostic_samples)
    elif args.command == "execute":
        if args.limit or args.models != list(MODELS):
            parser.error("execute uses exactly all five models and the complete canonical test set")
        # Each subprocess is awaited: do not leave untracked background model jobs.
        commands = [["prepare"], ["run", "--suite", "load"], ["run", "--suite", "clean"], ["report"]]
        for suite in ("noise", "compression", "replay"):
            commands += [["stress", "--suite", suite], ["run", "--suite", suite]]
        commands += [["run", "--suite", "efficiency"], ["report"]]
        for command in commands:
            subprocess.run([sys.executable, "-m", "benchmark.lava5", *command, "--output", str(args.output)], cwd=ROOT, check=True)
    elif args.command == "prepare":
        prepare(args.output)
    elif args.command == "stress":
        from benchmark.lava5_stress import generate
        generate(args.output, args.suite)
    elif args.command == "report":
        from benchmark.lava5_report import generate_report
        generate_report(args.output)
    elif args.command == "worker":
        worker(args.output, args.models[0], args.suite, args.limit)
    else:
        verify_protocol(args.output)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TF_NUM_INTRAOP_THREADS="1",
                   TF_NUM_INTEROP_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2", MPLBACKEND="Agg")
        for name in args.models:
            log = args.output / "protocol/logs" / f"{args.suite}_{name}_{time.time_ns()}.txt"
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-m", "benchmark.lava5", "worker", "--output", str(args.output),
                       "--models", name, "--suite", args.suite]
            if args.limit:
                command += ["--limit", str(args.limit)]
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        stream.write(line)
                        stream.flush()
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
            if code:
                raise RuntimeError(f"{name} failed ({code}); see {log}. No fabricated summary produced.")


if __name__ == "__main__":
    main()
