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


def mobilenet_artifacts() -> tuple[Path, Path, Path]:
    """Resolve MobileNet's backward-compatible or canonical artifact bundle.

    A bundle is selected atomically: paths are never mixed between layouts.
    Existing local training keeps the legacy contract, while a deployment may
    provide the canonical per-detector directory or explicit environment paths.
    """
    explicit_values = (
        os.getenv("LAVA_MOBILENET_MODEL_PATH"),
        os.getenv("LAVA_MOBILENET_THRESHOLD_PATH"),
        os.getenv("LAVA_MOBILENET_METADATA_PATH"),
    )
    candidates: list[tuple[Path, Path, Path]] = []
    if all(explicit_values):
        candidates.append(tuple(Path(value) for value in explicit_values))  # type: ignore[arg-type]
    legacy = (
        Path(config.MODEL_PATH), Path(config.THRESHOLD_PATH), Path(config.MODEL_METADATA_PATH)
    )
    canonical = tensorflow_artifacts("mobilenetv3_lstm")
    candidates.extend((legacy, canonical))
    for bundle in candidates:
        if all(path.is_file() for path in bundle):
            return bundle
    # Training remains backward compatible when neither complete bundle exists.
    return legacy


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
    if spec.threshold_artifact.suffix.lower() == ".txt":
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
    if path.suffix.lower() == ".txt":
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


def artifact_diagnostics(spec: DetectorSpec) -> list[str]:
    """Return actionable deployment diagnostics without loading a framework."""
    ready, missing = artifact_readiness(spec)
    if not ready:
        return [f"missing: {path}" for path in missing]
    issues: list[str] = []
    try:
        prefix = spec.model_artifact.read_bytes()[:160]
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            issues.append(f"model is a Git LFS pointer, not model bytes: {spec.model_artifact}")
        elif spec.model_artifact.stat().st_size == 0:
            issues.append(f"model file is empty: {spec.model_artifact}")
    except OSError as exc:
        issues.append(f"model is unreadable: {exc}")
    try:
        load_threshold(spec)
    except (ArtifactNotReadyError, TypeError, ValueError) as exc:
        issues.append(str(exc))
    try:
        load_json(spec.metadata_artifact)
    except ArtifactNotReadyError as exc:
        issues.append(str(exc))
    return issues
