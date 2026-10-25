"""Verify sealed evidence and deterministically rebuild actual pilot waveforms, without a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from src.lava.jaes_v2.configuration import load_config
from src.lava.jaes_v2.evidence import file_hash, read_csv, write_json
from src.lava.jaes_v2.runner import aggregate
from src.lava.jaes_v2.runtime import preflight
from src.lava.jaes_v2.transforms import Audio, AudioTransformChain, Transform, load_audio, write_audio


def verify(root: Path, output: Path) -> dict:
    import sys
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    changed = [name for name, digest in metadata["output_hashes"].items() if file_hash(output / name) != digest]
    if changed:
        raise ValueError(f"Sealed outputs changed: {changed}")
    aggregate(output)
    changed = [name for name, digest in metadata["output_hashes"].items() if file_hash(output / name) != digest]
    if changed:
        raise ValueError(f"Aggregation not byte-reproducible: {changed}")
    if "tensorflow" in sys.modules or "torch" in sys.modules:
        raise RuntimeError("Reproduction must not import a model framework")
    configuration = load_config(output / "config.json")
    first, second = Transform(**configuration.transform_a), Transform(**configuration.transform_b)
    chains = {"clean": AudioTransformChain(()), "A": AudioTransformChain((first,)), "B": AudioTransformChain((second,)),
              "A_then_B": AudioTransformChain((first, second)), "B_then_A": AudioTransformChain((second, first))}
    sources = {row["sample_id"]: row for row in read_csv(output / "manifest.csv")}
    condition_rows = read_csv(output / "conditions.csv")
    environment = preflight(root)
    mismatches, max_pcm_difference = [], 0.0
    temporary_parent = root / "outputs/jaes_v2/audit"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reproduction-", dir=temporary_parent) as temporary:
        temporary_path = Path(temporary).resolve()
        if not temporary_path.is_relative_to(temporary_parent.resolve()):
            raise ValueError("Temporary path escaped research audit directory")
        for row in condition_rows:
            source = sources[row["sample_id"]]
            if file_hash(root / source["path"]) != source["checksum"]:
                raise ValueError("Original source changed")
            audio = load_audio(root / source["path"], sample_rate=configuration.sample_rate, max_seconds=configuration.max_seconds)
            peak = float(np.max(np.abs(audio.samples)))
            audio = Audio(audio.samples * min(1.0, configuration.common_peak / peak), audio.sample_rate)
            rebuilt, _ = chains[row["condition_id"]].apply(audio, sample_id=row["sample_id"], seed=configuration.seed,
                                                         ffmpeg=environment["ffmpeg"])
            target = temporary_path / "rebuilt.wav"
            write_audio(target, rebuilt)
            existing, rate = sf.read(output / row["path"], dtype="float32")
            if rate != rebuilt.sample_rate or existing.shape != rebuilt.samples.shape:
                raise ValueError("Rebuilt waveform contract mismatch")
            max_pcm_difference = max(max_pcm_difference, float(np.max(np.abs(existing - rebuilt.samples))))
            if file_hash(target) != row["sha256"]:
                mismatches.append({"sample_id": row["sample_id"], "condition": row["condition_id"]})
    if mismatches:
        raise ValueError(f"Rebuilt WAV file hashes differ: {mismatches[:5]}; PCM max difference={max_pcm_difference}")
    return {"scope": "IMPLEMENTATION_VERIFICATION_ONLY", "sealed_output_files": len(metadata["output_hashes"]),
            "model_free_aggregation": "PASS", "regenerated_waveforms": len(condition_rows),
            "waveform_file_hash_parity": "PASS", "max_pcm_difference": max_pcm_difference,
            "baseline_integrity": environment["baseline_freeze_verification"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    options = parser.parse_args()
    if options.report.exists():
        raise FileExistsError("Use a new verification report path")
    root = Path(__file__).resolve().parents[2]
    report = verify(root, options.run.resolve())
    write_json(options.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
