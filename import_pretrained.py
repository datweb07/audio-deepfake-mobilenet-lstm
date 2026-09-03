#!/usr/bin/env python3
"""
One-time import script: wraps pretrained .pth files into LAVA-compatible artifacts.

Usage:
    python import_pretrained.py

This creates:
    models/rawnet2_pretrained/model.pt
    models/rawnet2_pretrained/threshold.json
    models/rawnet2_pretrained/metadata.json
    models/aasist_pretrained/model.pt
    models/aasist_pretrained/threshold.json
    models/aasist_pretrained/metadata.json
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
RAWNET2_SRC = BASE_DIR / "pre_trained_DF_RawNet2" / "pre_trained_DF_RawNet2.pth"
AASIST_SRC = BASE_DIR / "aasist-main" / "models" / "weights" / "AASIST.pth"

DETECTORS = {
    "rawnet2_pretrained": {
        "src": RAWNET2_SRC,
        "display_name": "RawNet2 (DF-Pretrained, 2021)",
        "target_samples": 64600,
        "sample_rate": 16000,
        "nb_classes": 2,
        "class_order": "0=FAKE(spoof), 1=REAL(bonafide)",
        "source": "ASVspoof2021 DF track baseline by Tak et al.",
    },
    "aasist_pretrained": {
        "src": AASIST_SRC,
        "display_name": "AASIST (Official Pretrained, NAVER)",
        "target_samples": 64600,
        "sample_rate": 16000,
        "nb_classes": 2,
        "class_order": "0=FAKE(spoof), 1=REAL(bonafide)",
        "source": "NAVER Corp - AASIST paper (ICASSP 2022)",
    },
}


def main() -> None:
    # Verify source checkpoints exist
    for name, info in DETECTORS.items():
        if not info["src"].is_file():
            print(f"[ERROR] Source checkpoint not found: {info['src']}")
            sys.exit(1)

    for name, info in DETECTORS.items():
        dest_dir = BASE_DIR / "models" / name
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy .pth -> model.pt
        dest_model = dest_dir / "model.pt"
        shutil.copy2(info["src"], dest_model)
        print(f"[OK] Copied {info['src'].name} -> {dest_model}")

        # Write threshold.json (default 0.5, can calibrate later)
        threshold_path = dest_dir / "threshold.json"
        threshold_path.write_text(
            json.dumps({"threshold": 0.5, "source": "default_no_calibration"}, indent=2),
            encoding="utf-8",
        )
        print(f"[OK] Written {threshold_path}")

        # Write metadata.json
        metadata_path = dest_dir / "metadata.json"
        metadata = {
            "detector_name": name,
            "display_name": info["display_name"],
            "framework": "pytorch",
            "source": info["source"],
            "target_samples": info["target_samples"],
            "sample_rate": info["sample_rate"],
            "nb_classes": info["nb_classes"],
            "class_order": info["class_order"],
            "score_semantics": "softmax(logits)[FAKE_INDEX=0] = P(FAKE)",
            "threshold_source": "default_0.5_uncalibrated",
            "note": "Use calibrate_pretrained.py on your validation set to get a better threshold.",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[OK] Written {metadata_path}")

    print()
    print("=" * 60)
    print("Import complete! Artifacts created:")
    for name in DETECTORS:
        print(f"  models/{name}/model.pt")
        print(f"  models/{name}/threshold.json")
        print(f"  models/{name}/metadata.json")
    print()
    print("Next step:")
    print("  streamlit run app.py")
    print("Then select 'RawNet2 (DF-Pretrained)' or 'AASIST (Official)' from the dropdown.")


if __name__ == "__main__":
    main()
