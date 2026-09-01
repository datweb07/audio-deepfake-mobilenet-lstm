"""Framework-neutral proxy for detectors hosted in the isolated torch environment."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import config
from src.lava.contracts import DetectorSpec, LAVADetector
from src.lava.errors import FrameworkDependencyError
from src.lava.score_semantics import validate_p_fake


def torch_python_path() -> Path:
    configured = os.environ.get("LAVA_TORCH_PYTHON")
    if configured:
        return Path(configured)
    return Path(config.BASE_DIR) / ".venv-torch" / "Scripts" / "python.exe"


def invoke_torch_worker(request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
    interpreter = torch_python_path()
    if not interpreter.is_file():
        raise FrameworkDependencyError(
            "RawNet2/AASIST require .venv-torch. Follow: python -m venv .venv-torch; "
            r".\.venv-torch\Scripts\Activate.ps1; python -m pip install -r requirements-torch.txt"
        )
    dependency_check = subprocess.run(
        [str(interpreter), "-c", "import torch"], capture_output=True, text=True, cwd=config.BASE_DIR, check=False
    )
    if dependency_check.returncode != 0:
        raise FrameworkDependencyError(
            "The isolated .venv-torch exists but PyTorch is not installed. Free disk space, activate "
            r".\.venv-torch\Scripts\Activate.ps1, then run: python -m pip install --no-cache-dir -r requirements-torch.txt"
        )
    process = subprocess.run(
        [str(interpreter), "-m", "src.lava.workers.torch_worker"],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        cwd=config.BASE_DIR,
        timeout=timeout,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or "unknown torch worker failure"
        raise RuntimeError(f"Torch worker failed: {detail}")
    try:
        response = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Torch worker returned invalid JSON: {process.stdout[-1000:]}") from exc
    if not response.get("ok", False):
        raise RuntimeError(str(response.get("error", "torch worker operation failed")))
    return response


class TorchWorkerDetector(LAVADetector):
    def __init__(self, spec: DetectorSpec) -> None:
        self.spec = spec

    def train(self, **kwargs: Any) -> None:
        invoke_torch_worker({"operation": "train", "model": self.spec.name, "options": kwargs}, timeout=None)

    def load(self) -> None:
        invoke_torch_worker({"operation": "load_check", "model": self.spec.name}, timeout=120)

    def predict_scores(self, audio_paths: Sequence[str]) -> np.ndarray:
        response = invoke_torch_worker(
            {"operation": "predict_scores", "model": self.spec.name, "audio_paths": list(audio_paths)},
            timeout=None,
        )
        return validate_p_fake(response["scores"]).astype(np.float32)

    def save(self) -> None:
        raise RuntimeError("Torch artifacts are saved atomically by the isolated training worker")

    def parameter_count(self) -> int:
        response = invoke_torch_worker({"operation": "model_info", "model": self.spec.name}, timeout=120)
        return int(response["parameter_count"])

    def benchmark_audio(self, audio_path: str, *, warmup: int, runs: int) -> dict[str, Any]:
        response = invoke_torch_worker(
            {"operation": "benchmark", "model": self.spec.name, "audio_path": audio_path,
             "warmup": warmup, "runs": runs}, timeout=None
        )
        return dict(response["timing"])
