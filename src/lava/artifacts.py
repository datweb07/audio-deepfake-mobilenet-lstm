"""Canonical per-detector model, threshold, and metadata contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import config
from src.lava.contracts import DetectorSpec
from src.lava.errors import ArtifactNotReadyError


def detector_directory(name: str) -> Path:
    return Path(config.MODELS_DIR) / name


def tensorflow_artifacts(name: str) -> tuple[Path, Path, Path]:
    directory = detector_directory(name)
    return directory / "model.keras", directory / "threshold.json", directory / "metadata.json"


def torch_artifacts(name: str) -> tuple[Path, Path, Path]:
    directory = detector_directory(name)
    return directory / "model.pt", directory / "threshold.json", directory / "metadata.json"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactNotReadyError(f"Invalid artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactNotReadyError(f"Artifact must contain a JSON object: {path}")
    return payload


def save_threshold(spec: DetectorSpec, threshold: float, *, source: str = "validation") -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if spec.name == "mobilenetv3_lstm":
        temporary = Path(str(spec.threshold_artifact) + ".tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(f"{threshold:.8f}\n", encoding="utf-8")
        os.replace(temporary, spec.threshold_artifact)
        return
    write_json_atomic(spec.threshold_artifact, {"threshold": threshold, "source": source})


def load_threshold(spec: DetectorSpec) -> float:
    path = spec.threshold_artifact
    if not path.is_file():
        raise ArtifactNotReadyError(f"Threshold not found for {spec.name}. Train the detector first: python train.py --model {spec.name}")
    if spec.name == "mobilenetv3_lstm":
        try:
            value = float(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise ArtifactNotReadyError(f"Invalid threshold artifact: {path}") from exc
    else:
        value = float(load_json(path).get("threshold"))
    if not 0.0 <= value <= 1.0:
        raise ArtifactNotReadyError(f"Threshold outside [0, 1]: {path}")
    return value


def artifact_readiness(spec: DetectorSpec) -> tuple[bool, list[str]]:
    missing = [
        str(path)
        for path in (spec.model_artifact, spec.threshold_artifact, spec.metadata_artifact)
        if not path.is_file()
    ]
    return not missing, missing

