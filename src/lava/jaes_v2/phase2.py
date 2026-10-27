"""Bounded C1 scientific-control study, new run IDs only; Phase 1 is never rerun."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from time import perf_counter
import traceback

import numpy as np

from .analysis import SCORE_FIELDS
from .control_analysis import CONTROL_CONDITIONS, summarize
from .controls import (NoisePlan, align_control, codec_delay_probe, estimate_delay, ffmpeg_identity,
                       pcm16_control, prepare_headroom, resample_roundtrip)
from .evidence import file_hash, object_hash, read_csv, write_csv, write_json
from .provenance import select_legacy_development
from .runtime import LegacyMelPredictor, preflight, source_fingerprint
from .runner import timestamp
from .transforms import Audio, Transform, diagnostics, load_audio, write_audio


@dataclass(frozen=True)
class ControlConfig:
    experiment_id: str
    per_class: int = 32
    seed: int = 2804
    model_id: str = "mobilenetv3_lstm"
    sample_rate: int = 22050
    max_seconds: float = 3.0
    common_peak: float = .25
    sensitivity_peak: float = .125
    noise_snr_db: float = 10
    bootstrap_replicates: int = 2000
    split_manifest: str = "data/manifests/split_manifest.csv"
    expected_manifest_sha256: str = ""
    scope: str = "LEGACY_DEVELOPMENT"
    public_manifest: str | None = None

    def __post_init__(self):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", self.experiment_id) or self.scope not in {"LEGACY_DEVELOPMENT", "PUBLIC_DEVELOPMENT"}:
            raise ValueError("New development identity/scope required")
        if (self.scope == "PUBLIC_DEVELOPMENT") != bool(self.public_manifest):
            raise ValueError("Public scope requires an explicit imported development manifest")
        if not isinstance(self.per_class, int) or not 1 <= self.per_class <= 100 or self.seed < 0 or self.bootstrap_replicates < 100:
            raise ValueError("Invalid bounded sample/seed/bootstrap configuration")
        if self.model_id not in {"mobilenetv3_lstm", "shufflenetv2_lstm"} or self.sample_rate != 22050 or not 0 < self.max_seconds <= 3:
            raise ValueError("Unsupported detector/audio contract")
        if not 0 < self.sensitivity_peak < self.common_peak <= .5 or not np.isfinite(self.noise_snr_db) or self.noise_snr_db < 5:
            raise ValueError("Unsafe/invalid headroom or noise configuration")


def make_variants(audio: Audio, configuration: ControlConfig, *, sample_id: str, ffmpeg: str) -> tuple[dict, dict]:
    clean, headroom = prepare_headroom(audio, configuration.common_peak)
    low, low_headroom = prepare_headroom(audio, configuration.sensitivity_peak)
    noise_seed = int(object_hash({"sample_id": sample_id, "seed": configuration.seed, "noise": "phase2"})[:16], 16)
    noise = NoisePlan.build(clean, target_snr_db=configuration.noise_snr_db, seed=noise_seed)
    codec = Transform("opus24", "codec", {"encoder": "libopus", "bitrate": "24k", "extension": ".ogg"})
    coded, codec_evidence = codec.apply(clean, seed=0, ffmpeg=ffmpeg)
    lag = estimate_delay(clean.samples, coded.samples)
    quantized, pcm_evidence = pcm16_control(clean)
    identity, identity_evidence = ffmpeg_identity(clean, ffmpeg)
    resampled, resample_evidence = resample_roundtrip(clean)
    variants = {"clean_native": audio, "clean": clean, "headroom_low": low, "pcm16": quantized,
                "identity": identity, "resample": resampled, "codec": coded,
                "codec_aligned": align_control(coded, lag["lag_samples"])}
    evidence = {"clean_native": {"headroom": "NONE"}, "clean": headroom, "headroom_low": low_headroom,
                "pcm16": pcm_evidence, "identity": identity_evidence, "resample": resample_evidence,
                "codec": codec_evidence, "codec_aligned": {"measured_delay": lag,
                                                            "policy": "Explicit alignment sensitivity control; native codec preserved"}}
    for prefix, policy in (("local", "local_target_snr"), ("fixed", "fixed_absolute")):
        noisy, noise_evidence = noise.insert(clean, policy)
        noisy_coded, noisy_codec_evidence = codec.apply(noisy, seed=0, ffmpeg=ffmpeg)
        coded_noisy, coded_noise_evidence = noise.insert(coded, policy)
        variants.update({prefix + "_noise": noisy, prefix + "_A_then_B": noisy_coded, prefix + "_B_then_A": coded_noisy})
        evidence.update({prefix + "_noise": noise_evidence,
                         prefix + "_A_then_B": {"noise": noise_evidence, "codec": noisy_codec_evidence},
                         prefix + "_B_then_A": {"codec": codec_evidence, "noise": coded_noise_evidence}})
    if set(variants) != set(CONTROL_CONDITIONS):
        raise RuntimeError("Control conditions inconsistent")
    return variants, evidence


def plot_controls(output: Path, report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "lava-jaes-phase2"
    import matplotlib.pyplot as plt
    conditions = list(CONTROL_CONDITIONS)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    positions = np.arange(len(conditions))
    for key, label, color in (("false_alarm_rate", "REAL false alarm", "#2166ac"), ("miss_rate", "FAKE miss", "#b2182b")):
        axes[0].plot(positions, [report["metrics"][condition][key] for condition in conditions], marker="o", label=label, color=color)
    axes[0].set(xticks=positions, xticklabels=conditions, ylabel="Fixed-threshold error fraction", ylim=(-.03, 1.03))
    axes[0].tick_params(axis="x", labelrotation=75, labelsize=8)
    axes[0].legend()
    effects = [row for row in report["contrasts"] if row["risk"] == "error" and row["class"] == "ALL"]
    for position, row in enumerate(effects):
        value = row["estimate"]
        interval = row["ci95"] or [value, value]
        axes[1].errorbar(value, position, xerr=[[max(0, value - interval[0])], [max(0, interval[1] - value)]], fmt="o", color="#2166ac", capsize=3)
    axes[1].axvline(0, color="gray", linestyle="--")
    axes[1].set(yticks=np.arange(len(effects)), yticklabels=[row["effect"] for row in effects],
                xlabel="Risk difference; descriptive parent-bootstrap 95% interval")
    cohort = "legacy" if report["scope"].endswith("LEGACY_DEVELOPMENT") else "public development"
    figure.suptitle(f"EXPLORATORY DEVELOPMENT RESULT — {report['metrics']['clean']['recordings']} {cohort} recordings; lineage limitations apply")
    (output / "figures").mkdir(exist_ok=True)
    figure.savefig(output / "figures/control_effects.png", dpi=160)
    figure.savefig(output / "figures/control_effects.svg", metadata={"Date": None})
    plt.close(figure)


def aggregate_controls(output: Path) -> dict:
    metadata = json.loads((output / "run_metadata.json").read_text())
    for name, digest in metadata["input_hashes"].items():
        if file_hash(output / name) != digest:
            raise ValueError(f"Changed control evidence: {name}")
    for name, digest in metadata["source_input_hashes"].items():
        if file_hash(Path(name)) != digest:
            raise ValueError("Original source audio changed")
    configuration = ControlConfig(**json.loads((output / "config.json").read_text()))
    if object_hash(asdict(configuration)) != metadata["config_hash"]:
        raise ValueError("Config fingerprint mismatch")
    conditions = read_csv(output / "conditions.csv")
    if object_hash(conditions) != metadata["transform_manifest_hash"]:
        raise ValueError("Transform manifest fingerprint mismatch")
    for row in conditions:
        if file_hash(output / row["path"]) != row["sha256"]:
            raise ValueError("Generated waveform changed")
    report = summarize(read_csv(output / "scores.csv"), read_csv(output / "manifest.csv"),
                       replicates=configuration.bootstrap_replicates, seed=configuration.seed)
    report["scope"] = "EXPLORATORY_" + configuration.scope
    write_json(output / "metrics.json", report["metrics"])
    write_json(output / "statistics.json", report)
    flat = [{**row, "ci95": json.dumps(row["ci95"])} for row in report["contrasts"]]
    write_csv(output / "contrasts.csv", flat)
    plot_controls(output, report)
    return report


def run_controls(root: Path, configuration: ControlConfig) -> Path:
    environment = preflight(root)
    estimate = configuration.per_class * 2 * len(CONTROL_CONDITIONS) * configuration.max_seconds * configuration.sample_rate * 4
    if environment["free_disk_bytes"] < estimate + 1024 ** 3 or not environment["ffmpeg"]:
        raise RuntimeError("Insufficient free disk reserve or FFmpeg unavailable")
    output = root / "outputs/jaes_v2" / configuration.experiment_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    metadata = {"scope": configuration.scope, "status": "RUNNING", "start_time": timestamp(),
                "config_hash": object_hash(asdict(configuration)), "code": source_fingerprint(root), "environment": environment,
                "estimated_audio_bytes": estimate, "scientific_status": "NOT_SUPPORTED", "protocol_version": "phase2-controls-1"}
    write_json(output / "config.json", asdict(configuration))
    write_json(output / "run_metadata.json", metadata)
    started = perf_counter()
    try:
        manifest = root / (configuration.public_manifest or configuration.split_manifest)
        if configuration.expected_manifest_sha256 and file_hash(manifest) != configuration.expected_manifest_sha256:
            raise ValueError("Source split manifest changed")
        if configuration.public_manifest:
            from .datasets import load_public_manifest
            records = load_public_manifest(manifest)
            if any(record.split != "development" or record.official_split != "dev" for record in records):
                raise ValueError("Public development runner rejects train/calibration/test/confirmation records")
            if any(sum(record.label == label for record in records) != configuration.per_class for label in (0, 1)):
                raise ValueError("Public manifest must match explicit balanced cohort size; no hidden subsampling")
        else:
            records = select_legacy_development(root, manifest, per_class=configuration.per_class, seed=configuration.seed)
        metadata["source_input_hashes"] = {str((root / record.path).resolve()): record.checksum for record in records}
        if any(file_hash(Path(path)) != digest for path, digest in metadata["source_input_hashes"].items()):
            raise ValueError("Source checksum mismatch")
        write_csv(output / "manifest.csv", [row.to_dict() for row in records])
        metadata["split_hash"] = file_hash(manifest)
        write_json(output / "codec_delay_probe.json", codec_delay_probe(environment["ffmpeg"]))
        predictor = LegacyMelPredictor(configuration.model_id)
        metadata.update(artifact_hashes=predictor.artifact_hashes, threshold=predictor.threshold,
                        warmup_calls=1, model_load_seconds=predictor.load_seconds)
        conditions, scores = [], []
        for source in records:
            audio = load_audio(root / source.path, sample_rate=configuration.sample_rate, max_seconds=configuration.max_seconds)
            variants, evidence = make_variants(audio, configuration, sample_id=source.sample_id, ffmpeg=environment["ffmpeg"])
            for condition in CONTROL_CONDITIONS:
                relative = Path("audio") / condition / f"{source.sample_id}.wav"
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                write_audio(path, variants[condition])
                conditions.append({"sample_id": source.sample_id, "parent_recording_id": source.parent_recording_id,
                                   "condition_id": condition, "path": relative.as_posix(), "sha256": file_hash(path),
                                   "settings": json.dumps(evidence[condition], sort_keys=True),
                                   "waveform_diagnostics": json.dumps(diagnostics(variants[condition]), sort_keys=True)})
                if not scores:
                    predictor.predict(path)
                row = dict.fromkeys(SCORE_FIELDS, "")
                row.update(run_id=configuration.experiment_id, sample_id=source.sample_id,
                           parent_recording_id=source.parent_recording_id, dataset_id=source.dataset_id,
                           generator_id=source.generator_id, speaker_id=source.speaker_id, label=source.label,
                           model_id=configuration.model_id, condition_id=condition, transform_order=condition,
                           seed=configuration.seed, calibrated_score="NOT_FITTED", threshold=predictor.threshold,
                           abstain=False, stage_used=1, status="OK")
                try:
                    score, timing = predictor.predict(path)
                    prediction = int(score >= predictor.threshold)
                    row.update(raw_score=score, predicted_label=prediction, correct=prediction == source.label, **timing)
                except Exception as error:
                    row.update(status="FAILED", error=f"{type(error).__name__}: {error}")
                scores.append(row)
            write_csv(output / "scores.csv", scores, SCORE_FIELDS)
            write_csv(output / "conditions.csv", conditions)
            print(f"{len(scores)}/{len(records) * len(CONTROL_CONDITIONS)} scores persisted", flush=True)
        metadata["counts"] = {"attempted": len(scores), "successful": sum(row["status"] == "OK" for row in scores),
                              "failed": sum(row["status"] != "OK" for row in scores), "excluded": 0}
        metadata["input_hashes"] = {name: file_hash(output / name) for name in ("config.json", "manifest.csv", "conditions.csv", "scores.csv", "codec_delay_probe.json")}
        metadata["transform_manifest_hash"] = object_hash(read_csv(output / "conditions.csv"))
        write_json(output / "run_metadata.json", metadata)
        aggregate_controls(output)
        metadata.update(status="DEVELOPMENT_EVALUATED", end_time=timestamp(), elapsed_seconds=perf_counter() - started)
        (output / "logs/execution.txt").write_text(json.dumps(metadata["counts"]) + "\nNo training, calibration, external confirmation or test-label selection.\n")
        metadata["output_hashes"] = {path.relative_to(output).as_posix(): file_hash(path) for path in sorted(output.rglob("*"))
                                     if path.is_file() and path.name != "run_metadata.json"}
        write_json(output / "run_metadata.json", metadata)
        return output
    except Exception as error:
        metadata.update(status="FAILED", end_time=timestamp(), error=f"{type(error).__name__}: {error}")
        write_json(output / "run_metadata.json", metadata)
        (output / "logs/error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    if args.aggregate:
        report = aggregate_controls(args.aggregate)
        print(json.dumps({"scope": report["scope"], "contrasts": len(report["contrasts"])}))
    elif args.config:
        payload = json.loads(args.config.read_text())
        if args.experiment_id:
            payload["experiment_id"] = args.experiment_id
        print(run_controls(Path(__file__).resolve().parents[3], ControlConfig(**payload)))
    else:
        parser.error("Use --config or --aggregate")


if __name__ == "__main__":
    main()
