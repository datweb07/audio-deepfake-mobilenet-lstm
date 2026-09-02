"""PyTorch detector metadata safe to import without importing torch."""

from __future__ import annotations

import config
from src.lava.artifacts import torch_artifacts
from src.lava.contracts import DetectorSpec
from src.lava.workers.torch_proxy import TorchWorkerDetector


def _spec(name: str, display_name: str) -> DetectorSpec:
    model, threshold, metadata = torch_artifacts(name)
    return DetectorSpec(
        name=name,
        display_name=display_name,
        group="reference",
        framework="pytorch",
        input_type="waveform",
        sample_rate=16_000,
        audio_duration=3.0,
        num_segments=None,
        model_artifact=model,
        threshold_artifact=threshold,
        metadata_artifact=metadata,
        pretraining_status="NOT_APPLICABLE_TRAIN_FROM_SCRATCH",
    )


RAWNET2_SPEC = _spec("rawnet2", "RawNet2")
AASIST_SPEC = _spec("aasist", "AASIST")


def rawnet2_factory() -> TorchWorkerDetector:
    return TorchWorkerDetector(RAWNET2_SPEC)


def aasist_factory() -> TorchWorkerDetector:
    return TorchWorkerDetector(AASIST_SPEC)

