"""Reusable Keras detector adapter without exposing Keras to benchmark callers."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import tensorflow as tf

import config
from src.lava.contracts import DetectorSpec, LAVADetector
from src.lava.errors import ArtifactNotReadyError
from src.lava.score_semantics import validate_p_fake
from src.preprocessing import process_audio_file
from src.lava.timing import timed_runs


Builder = Callable[[str | None], tuple[tf.keras.Model, tf.keras.Model]]


class KerasLightweightDetector(LAVADetector):
    def __init__(self, spec: DetectorSpec, builder: Builder) -> None:
        self.spec = spec
        self._builder = builder
        self.model: tf.keras.Model | None = None
        self.backbone: tf.keras.Model | None = None

    def build(self, *, weights: str | None = "imagenet") -> tf.keras.Model:
        self.model, self.backbone = self._builder(weights)
        self.validate_model(self.model)
        return self.model

    def validate_model(self, model: tf.keras.Model) -> None:
        expected_input = (None, config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS)
        if tuple(model.input_shape) != expected_input or tuple(model.output_shape) != (None, 1):
            raise ValueError(
                f"{self.spec.name} model contract mismatch: {model.input_shape} -> {model.output_shape}"
            )
        lstm = model.get_layer("temporal_lstm")
        if not isinstance(lstm, tf.keras.layers.LSTM) or lstm.units != config.LSTM_UNITS:
            raise ValueError("Lightweight detector must use the shared LSTM(128) head")
        output = model.get_layer("probability_fake")
        if tf.keras.activations.serialize(output.activation) != "sigmoid":
            raise ValueError("Detector output must be sigmoid P(FAKE)")

    def train(self, **kwargs: Any) -> None:
        from train import train_tensorflow_detector

        train_tensorflow_detector(self.spec.name, **kwargs)

    def load(self) -> None:
        if not self.spec.model_artifact.is_file():
            raise ArtifactNotReadyError(
                f"Production model not found for {self.spec.name}. Run: python train.py --model {self.spec.name}"
            )
        self.model = tf.keras.models.load_model(self.spec.model_artifact, compile=False)
        self.validate_model(self.model)
        temporal = next(layer for layer in self.model.layers if isinstance(layer, tf.keras.layers.TimeDistributed))
        self.backbone = temporal.layer

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
        self.validate_model(self.model)
        path = self.spec.model_artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(path.stem + ".pending.keras")
        if pending.exists():
            pending.unlink()
        try:
            export_model = tf.keras.models.clone_model(self.model)
            export_model.set_weights(self.model.get_weights())
            export_model.save(pending)
            verified = tf.keras.models.load_model(pending, compile=False)
            self.validate_model(verified)
            os.replace(pending, path)
        finally:
            if pending.exists():
                pending.unlink()
        self.load()

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
        model_only = timed_runs(
            lambda: self.model(batch, training=False).numpy(), warmup=warmup, runs=runs
        )
        preprocessing = timed_runs(lambda: process_audio_file(audio_path), warmup=min(2, warmup), runs=runs)
        end_to_end = timed_runs(
            lambda: self.model(
                tf.convert_to_tensor(process_audio_file(audio_path)[np.newaxis], dtype=tf.float32), training=False
            ).numpy(), warmup=min(2, warmup), runs=runs,
        )
        return {"model_only": model_only, "preprocessing": preprocessing, "end_to_end": end_to_end}
