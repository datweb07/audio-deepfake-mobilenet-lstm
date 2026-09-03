"""ONNX Runtime adapters for verified pretrained PyTorch reference exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.lava.contracts import DetectorSpec, LAVADetector
from src.lava.preprocessing.waveform import load_waveform
from src.lava.score_semantics import validate_p_fake
from src.lava.timing import timed_runs


TARGET_SAMPLES = 64_600
NATIVE_FAKE_INDEX = 0


def _softmax_fake(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return validate_p_fake(exponentials[:, NATIVE_FAKE_INDEX] / exponentials.sum(axis=1))


class OnnxPretrainedDetector(LAVADetector):
    """Serve a parity-tested native detector without importing PyTorch."""

    def __init__(self, spec: DetectorSpec) -> None:
        self.spec = spec
        self.session = None

    def train(self, **kwargs: Any) -> None:
        raise RuntimeError(
            f"{self.spec.display_name} is an imported pretrained reference model; training is disabled"
        )

    def load(self) -> None:
        if not self.spec.model_artifact.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.spec.model_artifact}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime is required for pretrained RawNet2/AASIST deployment"
            ) from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.spec.model_artifact),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]
        if input_info.name != "waveform" or output_info.name != "logits":
            raise ValueError(
                f"Invalid ONNX contract for {self.spec.name}: "
                f"input={input_info.name}, output={output_info.name}"
            )

    def _run_waveform(self, waveform: np.ndarray) -> float:
        if self.session is None:
            self.load()
        assert self.session is not None
        batch = np.asarray(waveform, dtype=np.float32).reshape(1, TARGET_SAMPLES)
        logits = self.session.run(["logits"], {"waveform": batch})[0]
        return float(_softmax_fake(logits)[0])

    def predict_scores(self, audio_paths: Sequence[str]) -> np.ndarray:
        scores = [
            self._run_waveform(
                load_waveform(path, sample_rate=self.spec.sample_rate, target_samples=TARGET_SAMPLES)
            )
            for path in audio_paths
        ]
        return validate_p_fake(scores).astype(np.float32)

    def save(self) -> None:
        raise RuntimeError("Imported ONNX reference artifacts are immutable")

    def parameter_count(self) -> int:
        with self.spec.metadata_artifact.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return int(metadata["parameter_count"])

    def model_size(self) -> int:
        return self.spec.model_artifact.stat().st_size if self.spec.model_artifact.is_file() else 0

    def benchmark_audio(self, audio_path: str, *, warmup: int, runs: int) -> dict[str, Any]:
        waveform = load_waveform(
            audio_path, sample_rate=self.spec.sample_rate, target_samples=TARGET_SAMPLES
        )
        if self.session is None:
            self.load()
        assert self.session is not None
        batch = waveform.reshape(1, TARGET_SAMPLES).astype(np.float32, copy=False)

        def model_call() -> None:
            self.session.run(["logits"], {"waveform": batch})

        def end_to_end_call() -> None:
            values = load_waveform(
                audio_path, sample_rate=self.spec.sample_rate, target_samples=TARGET_SAMPLES
            ).reshape(1, TARGET_SAMPLES)
            self.session.run(["logits"], {"waveform": values})

        return {
            "model_only": timed_runs(model_call, warmup=warmup, runs=runs),
            "preprocessing": timed_runs(
                lambda: load_waveform(
                    audio_path, sample_rate=self.spec.sample_rate, target_samples=TARGET_SAMPLES
                ),
                warmup=min(2, warmup), runs=runs,
            ),
            "end_to_end": timed_runs(end_to_end_call, warmup=min(2, warmup), runs=runs),
            "device": "cpu",
            "runtime": "onnxruntime",
        }
