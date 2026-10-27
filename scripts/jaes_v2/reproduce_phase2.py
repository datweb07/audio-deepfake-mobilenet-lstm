"""Verify/rebuild a sealed Phase 2 run without loading TensorFlow or running inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

from src.lava.jaes_v2.evidence import file_hash, read_csv, write_json
from src.lava.jaes_v2.phase2 import ControlConfig, aggregate_controls, make_variants
from src.lava.jaes_v2.runtime import preflight
from src.lava.jaes_v2.transforms import load_audio, write_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--waveforms", action="store_true", help="Regenerate every audio control, not model predictions")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Verification report already exists")
    root = Path(__file__).resolve().parents[2]
    run = args.run.resolve()
    metadata = json.loads((run / "run_metadata.json").read_text())
    for name, digest in metadata["output_hashes"].items():
        if file_hash(run / name) != digest:
            raise ValueError(f"Changed sealed file: {name}")
    aggregate_controls(run)
    for name, digest in metadata["output_hashes"].items():
        if file_hash(run / name) != digest:
            raise ValueError(f"Non-reproducible derived file: {name}")
    configuration = ControlConfig(**json.loads((run / "config.json").read_text()))
    count = 0
    environment = preflight(root)
    if args.waveforms:
        references = {(row["sample_id"], row["condition_id"]): row for row in read_csv(run / "conditions.csv")}
        temporary_parent = root / "outputs/jaes_v2/audit"
        with tempfile.TemporaryDirectory(prefix="phase2-rebuild-", dir=temporary_parent) as directory:
            temporary = Path(directory).resolve()
            if not temporary.is_relative_to(temporary_parent.resolve()):
                raise ValueError("Temporary reconstruction outside workspace")
            for source in read_csv(run / "manifest.csv"):
                audio = load_audio(root / source["path"], sample_rate=configuration.sample_rate, max_seconds=configuration.max_seconds)
                variants, _ = make_variants(audio, configuration, sample_id=source["sample_id"], ffmpeg=environment["ffmpeg"])
                for condition, changed in variants.items():
                    path = temporary / "rebuilt.wav"
                    write_audio(path, changed)
                    if file_hash(path) != references[(source["sample_id"], condition)]["sha256"]:
                        raise ValueError(f"Waveform parity failed: {source['sample_id']} / {condition}")
                    count += 1
    if "tensorflow" in sys.modules or "torch" in sys.modules:
        raise RuntimeError("Model framework unexpectedly imported")
    report = {"status": "PASS", "sealed_files_verified": len(metadata["output_hashes"]),
              "model_free_aggregation": "PASS", "waveforms_regenerated": count,
              "waveform_hash_parity": "PASS" if args.waveforms else "NOT_RUN",
              "baseline_integrity": environment["baseline_freeze_verification"]}
    write_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
