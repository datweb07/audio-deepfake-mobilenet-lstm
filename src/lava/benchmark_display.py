"""Optional measured benchmark card; never changes inference or thresholds."""
import csv
import hashlib
import json
from pathlib import Path

import config


def _packaged_card(spec):
    """Read the small tracked summary when ignored local outputs are unavailable."""
    path = Path(config.BASE_DIR) / "models/lava_6_benchmark.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["models"][spec.name]
        digest = hashlib.sha256()
        with spec.model_artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != row["model_sha256"]:
            return None
        return dict(row, Model=spec.name, Provenance="see detector metadata", TestSamples=payload["clean_samples"], RobustnessScope=payload["robustness_scope"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def benchmark_card(spec, directory=None):
    if directory is None:
        six = Path(config.OUTPUTS_DIR) / "lava_6"
        directory = six if (six / "lava_6_results.csv").is_file() else Path(config.OUTPUTS_DIR) / "lava_5"
    directory = Path(directory)
    protocol_path = directory / "protocol/protocol.json"
    result_path = directory / ("lava_6_results.csv" if (directory / "lava_6_results.csv").is_file() else "lava_5_results.csv")
    if not protocol_path.is_file() or not result_path.is_file():
        return _packaged_card(spec)
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        audit = next((a for a in protocol["models"] if a["registry_name"] == spec.name), None)
        if audit is None:
            return None
        # Fail closed if the model/threshold/metadata bundle has changed.
        for path, expected in audit["artifact_hashes"].items():
            digest = hashlib.sha256()
            with (Path(config.BASE_DIR) / path.replace("\\", "/")).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                return None
        with result_path.open(encoding="utf-8", newline="") as handle:
            row = next((r for r in csv.DictReader(handle) if r["Model"] == audit["model"]), None)
        if row is None or not row.get("CleanF1"):
            return None
        return dict(row, Provenance=audit["checkpoint_origin"], TestSamples=protocol["test_samples"])
    except (OSError, ValueError, KeyError, TypeError):
        return _packaged_card(spec)
