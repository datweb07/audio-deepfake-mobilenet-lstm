"""Registry adapter around the unchanged production MobileNetV3Small-LSTM."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import tensorflow as tf

import config
from src.artifacts import load_production_model, save_production_model
from src.lava.contracts import LAVADetector
from src.lava.models.tensorflow.specs import MOBILENET_SPEC as SPEC
from src.lava.score_semantics import validate_p_fake
from src.model import build_hybrid_model
from src.preprocessing import process_audio_file
from src.lava.timing import timed_runs


class MobileNetV3LSTMDetector(LAVADetector):
    spec = SPEC

    def __init__(self) -> None:
        self.model: tf.keras.Model | None = None
        self.backbone: tf.keras.Model | None = None

    def build(self, *, weights: str | None = "imagenet") -> tf.keras.Model:
        self.model, self.backbone = build_hybrid_model(weights=weights)
        return self.model

    def train(self, **kwargs: Any) -> None:
        from train import train_tensorflow_detector

        train_tensorflow_detector(self.spec.name, **kwargs)

    def load(self) -> None:
        self.model = load_production_model(compile=False)

    def predict_scores(self, audio_paths: Sequence[str]) -> np.ndarray:
        if self.model is None:
            self.load()
        assert self.model is not None
        scores: list[float] = []
        for start in range(0, len(audio_paths), config.BATCH_SIZE):
            batch_paths = audio_paths[start:start + config.BATCH_SIZE]
            features = np.stack([process_audio_file(path) for path in batch_paths], axis=0)
            scores.extend(self.model.predict(features, verbose=0).reshape(-1).tolist())
        return validate_p_fake(scores).astype(np.float32)

    def predict_feature_batch(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Build or load the detector before prediction")
        return validate_p_fake(self.model.predict(features, verbose=0).reshape(-1)).astype(np.float32)

    def save(self) -> None:
        if self.model is None:
            raise RuntimeError("No model is available to save")
        self.model = save_production_model(self.model)

    def parameter_count(self) -> int:
        if self.model is None:
            self.load()
        assert self.model is not None
        return int(self.model.count_params())

    def benchmark_audio(self, audio_path: str, *, warmup: int, runs: int) -> dict[str, Any]:
        if self.model is None:
            self.load()
        assert self.model is not None
        features = process_audio_file(audio_path)
        batch = tf.convert_to_tensor(features[np.newaxis], dtype=tf.float32)
        model_only = timed_runs(lambda: self.model(batch, training=False).numpy(), warmup=warmup, runs=runs)
        preprocessing = timed_runs(lambda: process_audio_file(audio_path), warmup=min(2, warmup), runs=runs)
        end_to_end = timed_runs(
            lambda: self.model(
                tf.convert_to_tensor(process_audio_file(audio_path)[np.newaxis], dtype=tf.float32), training=False
            ).numpy(), warmup=min(2, warmup), runs=runs,
        )
        return {"model_only": model_only, "preprocessing": preprocessing, "end_to_end": end_to_end}
