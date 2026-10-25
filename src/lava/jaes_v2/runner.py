"""Executable C1 vertical slice: audio -> ordered variants -> real model -> paired analysis."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter
import traceback

import numpy as np

from . import PROTOCOL_VERSION
from .analysis import CONDITIONS, SCORE_FIELDS, analyze
from .configuration import C1Config, load_config
from .evidence import file_hash, object_hash, read_csv, write_csv, write_json
from .provenance import audit_partitions, select_legacy_development
from .runtime import LegacyMelPredictor, preflight, source_fingerprint
from .transforms import Audio, AudioTransformChain, Transform, diagnostics, load_audio, write_audio


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def plot_diagnostics(output: Path, rows: list[dict], report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["svg.hashsalt"] = "lava-jaes-v2"
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    positions = np.arange(len(CONDITIONS))
    for label, name, color in ((0, "REAL false alarms", "#246a9e"), (1, "FAKE misses", "#bc4a3c")):
        key = "false_alarm_rate" if label == 0 else "miss_rate"
        axes[0].plot(positions, [report["conditions"][c][key] for c in CONDITIONS], marker="o", label=name, color=color)
    axes[0].set(xticks=positions, xticklabels=CONDITIONS, ylabel="Error fraction at frozen threshold", ylim=(-0.03, 1.03))
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].legend(fontsize=8)
    for label, name, color in ((0, "REAL", "#246a9e"), (1, "FAKE", "#bc4a3c")):
        for sample in sorted({row["sample_id"] for row in rows if int(row["label"]) == label}):
            values = {row["condition_id"]: float(row["raw_score"]) for row in rows if row["sample_id"] == sample}
            axes[1].plot(positions, [values[c] for c in CONDITIONS], color=color, alpha=0.45)
        axes[1].plot([], [], color=color, label=name)
    axes[1].axhline(float(rows[0]["threshold"]), color="black", linestyle="--", label="Legacy validation threshold")
    axes[1].set(xticks=positions, xticklabels=CONDITIONS, ylabel="Raw P(FAKE) score (uncalibrated)", ylim=(-0.03, 1.03))
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].legend(fontsize=8)
    figure.suptitle("C1 DEVELOPMENT PILOT — legacy data; no generalization claim")
    target = output / "figures"
    target.mkdir(exist_ok=True)
    figure.savefig(target / "paired_diagnostic.png", dpi=170)
    figure.savefig(target / "paired_diagnostic.svg", metadata={"Date": None})
    plt.close(figure)


def aggregate(output: Path) -> dict:
    """Rebuild numeric analysis from persisted CSV; never loads a detector."""
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    if "score_csv_sha256" not in metadata or file_hash(output / "scores.csv") != metadata["score_csv_sha256"]:
        raise ValueError("Score evidence changed or was not sealed")
    rows = read_csv(output / "scores.csv")
    configuration = json.loads((output / "config.json").read_text(encoding="utf-8"))
    manifest = read_csv(output / "manifest.csv")
    expected = {(row["sample_id"], condition) for row in manifest for condition in CONDITIONS}
    if len(rows) != len(expected) or {(row["sample_id"], row["condition_id"]) for row in rows} != expected:
        raise ValueError("Scores do not cover the intended manifest")
    for name, digest in metadata["evidence_hashes"].items():
        if file_hash(output / name) != digest:
            raise ValueError(f"Changed sealed evidence: {name}")
    report = analyze(rows, replicates=configuration["bootstrap_replicates"], seed=configuration["seed"])
    write_json(output / "metrics.json", report["conditions"])
    write_json(output / "statistics.json", report)
    flat = [{**record, "ci95_low": record["ci95"][0] if record["ci95"] else "",
             "ci95_high": record["ci95"][1] if record["ci95"] else ""} for record in report["contrasts"]]
    for record in flat:
        record.pop("ci95")
    write_csv(output / "interaction_report.csv", flat)
    plot_diagnostics(output, rows, report)
    return report


def run(root: Path, configuration: C1Config) -> Path:
    environment = preflight(root)
    output = root / "outputs/jaes_v2" / configuration.experiment_id
    # mkdir without exist_ok is deliberate: failed or successful runs are immutable identities.
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    metadata = {"experiment_id": configuration.experiment_id, "contribution_id": "C1", "hypothesis_id": ["H1b", "H1c"],
                "scope": configuration.scope, "protocol_version": PROTOCOL_VERSION, "status": "RUNNING",
                "start_time": timestamp(), "end_time": None, "environment": environment,
                "config_hash": configuration.fingerprint, "code": source_fingerprint(root),
                "training_seed": "LEGACY_CHECKPOINT_METADATA", "evaluation_seed": configuration.seed,
                "calibration_hash": "NOT_FITTED; legacy threshold artifact sealed below",
                "exclusions": [], "training_performed": False}
    write_json(output / "config.json", configuration.to_dict())
    write_json(output / "run_metadata.json", metadata)
    started = perf_counter()
    try:
        manifest_path = root / configuration.split_manifest
        manifest_hash = file_hash(manifest_path)
        if configuration.expected_manifest_sha256 and manifest_hash != configuration.expected_manifest_sha256:
            raise ValueError("Configured split manifest hash mismatch")
        metadata["legacy_split_manifest_sha256"] = manifest_hash
        records = select_legacy_development(root, manifest_path, per_class=configuration.per_class, seed=configuration.seed)
        metadata["provenance_audit"] = audit_partitions(records)
        write_csv(output / "manifest.csv", [record.to_dict() for record in records])
        first, second = Transform(**configuration.transform_a), Transform(**configuration.transform_b)
        chains = {"clean": AudioTransformChain(()), "A": AudioTransformChain((first,)), "B": AudioTransformChain((second,)),
                  "A_then_B": AudioTransformChain((first, second)), "B_then_A": AudioTransformChain((second, first))}
        metadata["transform_manifest_hash"] = object_hash({name: chain.fingerprint for name, chain in chains.items()})
        if any(t.kind == "codec" for t in (first, second)) and not environment["ffmpeg"]:
            raise FileNotFoundError("Codec pilot requires FFmpeg")
        if shutil.disk_usage(root).free < 256 * 1024 ** 2:
            raise OSError("Need at least 256MiB free for small pilot")
        generated, variants = [], []
        for record in records:
            audio = load_audio(root / record.path, sample_rate=configuration.sample_rate, max_seconds=configuration.max_seconds)
            peak = float(np.max(np.abs(audio.samples)))
            if peak == 0:
                raise ValueError(f"Silent source: {record.sample_id}; new cohort/config required, no silent exclusions")
            # Equal per-source headroom for every condition including clean; no output normalization.
            gain = min(1.0, configuration.common_peak / peak)
            base = Audio(audio.samples * gain, audio.sample_rate)
            for condition, chain in chains.items():
                changed, steps = chain.apply(base, sample_id=record.sample_id, seed=configuration.seed, ffmpeg=environment["ffmpeg"])
                relative = Path("audio") / condition / f"{record.sample_id}.wav"
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                write_audio(target, changed)
                generated.append({"sample_id": record.sample_id, "parent_recording_id": record.parent_recording_id,
                                  "dataset_id": record.dataset_id, "generator_id": record.generator_id, "label": record.label,
                                  "condition_id": condition, "path": relative.as_posix(), "sha256": file_hash(target),
                                  "transform_order": json.dumps([t.key for t in chain.transforms]), "transform_hash": chain.fingerprint,
                                  "random_seed": configuration.seed, "source_sample_rate": record.sample_rate,
                                  "output_sample_rate": changed.sample_rate, "source_duration": record.duration,
                                  "output_duration": len(changed.samples) / changed.sample_rate, "common_gain": gain,
                                  "measured": json.dumps(diagnostics(changed), sort_keys=True), "steps": json.dumps(steps, sort_keys=True)})
                variants.append((record, condition, target, generated[-1]["transform_order"]))
        write_csv(output / "conditions.csv", generated)
        metadata["transform_generation_seconds"] = perf_counter() - started
        predictor = LegacyMelPredictor(configuration.model_id)
        metadata.update(artifact_hashes=predictor.artifact_hashes, model_load_seconds=predictor.load_seconds,
                        threshold=predictor.threshold, warmup_calls=1, timing_scope="one sequential pass; diagnostic, not benchmark latency")
        # Verify our direct-call bridge agrees with the existing registry API before scoring.
        warm_path = variants[0][2]
        warm_score, _ = predictor.predict(warm_path)
        registry_score = float(predictor.detector.predict_scores([str(warm_path)])[0])
        if not np.isclose(warm_score, registry_score, rtol=1e-5, atol=1e-6):
            raise ValueError("Legacy registry score parity failed")
        metadata["registry_score_parity_abs_error"] = abs(warm_score - registry_score)
        scores = []
        for record, condition, path, order in variants:
            row = {field: "" for field in SCORE_FIELDS}
            row.update(run_id=configuration.experiment_id, sample_id=record.sample_id,
                       parent_recording_id=record.parent_recording_id, dataset_id=record.dataset_id,
                       generator_id=record.generator_id, speaker_id=record.speaker_id, label=record.label,
                       model_id=configuration.model_id, condition_id=condition, transform_order=order,
                       seed=configuration.seed, threshold=predictor.threshold, abstain=False, stage_used=1,
                       calibrated_score="NOT_FITTED", status="OK")
            try:
                score, timings = predictor.predict(path)
                prediction = int(score >= predictor.threshold)
                row.update(raw_score=score, predicted_label=prediction, correct=prediction == record.label, **timings)
            except Exception as error:
                row.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            scores.append(row)
            write_csv(output / "scores.csv", scores, SCORE_FIELDS)
        metadata["counts"] = {"attempted": len(scores), "successful": sum(row["status"] == "OK" for row in scores),
                              "failed": sum(row["status"] != "OK" for row in scores), "excluded": 0}
        metadata["score_csv_sha256"] = file_hash(output / "scores.csv")
        metadata["evidence_hashes"] = {name: file_hash(output / name) for name in ("config.json", "manifest.csv", "conditions.csv")}
        write_json(output / "run_metadata.json", metadata)
        if metadata["counts"]["failed"]:
            raise RuntimeError("Some inference records failed; no complete-case scientific analysis")
        aggregate(output)
        # A run is complete only after successful analysis, visualization and evidence sealing.
        metadata.update(status="PILOT_EVALUATED" if configuration.scope != "SMOKE" else "SMOKE_TESTED",
                        end_time=timestamp(), elapsed_seconds=perf_counter() - started,
                        scientific_status="PROTOCOL_EXTENSION; NOVELTY_NOT_SUPPORTED", generator_holdout="NOT_AVAILABLE")
        (output / "logs/execution.txt").write_text(f"Finished {metadata['end_time']}\nCounts: {metadata['counts']}\nNo training or threshold fitting.\n", encoding="utf-8")
        metadata["output_hashes"] = {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(output.rglob("*"))
                                     if p.is_file() and p.name != "run_metadata.json"}
        write_json(output / "run_metadata.json", metadata)
        return output
    except Exception as error:
        metadata.update(status="FAILED", end_time=timestamp(), error=f"{type(error).__name__}: {error}")
        write_json(output / "run_metadata.json", metadata)
        (output / "logs/error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--experiment-id", help="New run identity; existing output directories are never overwritten")
    parser.add_argument("--aggregate", type=Path, help="Rebuild analysis without model inference")
    parser.add_argument("--preflight", action="store_true")
    options = parser.parse_args()
    root = options.root.resolve()
    if options.preflight:
        print(json.dumps(preflight(root), indent=2))
    elif options.aggregate:
        report = aggregate(options.aggregate.resolve())
        print(json.dumps({"scope": report["scope"], "contrast_count": len(report["contrasts"])}))
    elif options.config:
        configuration = load_config(options.config)
        if options.experiment_id:
            configuration = replace(configuration, experiment_id=options.experiment_id)
        print(run(root, configuration))
    else:
        parser.error("Choose --preflight, --config or --aggregate")


if __name__ == "__main__":
    main()
