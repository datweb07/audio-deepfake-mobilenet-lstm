"""Development-only shortcut, overlap and candidate near-duplicate audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.lava.jaes_v2.data_audit import audit_sources
from src.lava.jaes_v2.datasets import load_public_manifest
from src.lava.jaes_v2.evidence import read_csv, write_csv, write_json
from src.lava.jaes_v2.overlap import near_duplicate_candidates, overlap_dimensions
from src.lava.jaes_v2.provenance import load_manifest
from src.lava.jaes_v2.transforms import load_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    rows = read_csv(args.manifest)
    if not rows:
        raise ValueError("Empty source manifest")
    records = load_public_manifest(args.manifest) if "official_split" in rows[0] else load_manifest(args.manifest)
    summary, features = audit_sources(root, records)
    clips = {record.sample_id: load_audio(root / record.path, sample_rate=16000, max_seconds=3) for record in records}
    candidates = near_duplicate_candidates(clips)
    summary["near_duplicate_check"] = {"method": "8kHz resampled cross-correlation over <=3s prefix; max lag 20ms",
                                       "threshold": .98, "candidate_pairs": len(candidates), "automatic_exclusions": 0,
                                       "limitations": "Candidate-only; long-range/cropped/nonlinear re-encodings can be missed"}
    write_json(args.output / "source_audit.json", summary)
    write_json(args.output / "overlap_dimensions.json", overlap_dimensions(records))
    write_csv(args.output / "source_features.csv", features)
    write_csv(args.output / "near_duplicate_candidates.csv", candidates,
              ["sample_a", "sample_b", "status", "similarity", "lag_at_8khz", "threshold", "automatic_exclusion"])
    print(json.dumps({"recordings": len(records), "shortcut_baselines": summary.get("shortcut_baselines"),
                      "near_duplicate_check": summary["near_duplicate_check"]}, indent=2))


if __name__ == "__main__":
    main()
