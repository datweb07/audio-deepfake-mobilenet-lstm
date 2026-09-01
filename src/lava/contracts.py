"""Framework-neutral detector specification and runtime interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    display_name: str
    group: str
    framework: str
    input_type: str
    sample_rate: int
    audio_duration: float
    num_segments: int | None
    model_artifact: Path
    threshold_artifact: Path
    metadata_artifact: Path
    pretraining_status: str


class LAVADetector(ABC):
    """The benchmark-facing API; callers never access TF/PyTorch models directly."""

    spec: DetectorSpec

    @abstractmethod
    def train(self, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_scores(self, audio_paths: Sequence[str]) -> np.ndarray:
        """Return one finite P(FAKE) value in [0, 1] per audio path."""
        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def parameter_count(self) -> int:
        raise NotImplementedError

    def model_size(self) -> int:
        path = self.spec.model_artifact
        if not path.is_file():
            return 0
        return path.stat().st_size

    def benchmark_audio(self, audio_path: str, *, warmup: int, runs: int) -> dict[str, Any]:
        raise NotImplementedError(f"Efficiency timing is not implemented for {self.spec.name}")
