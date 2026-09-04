"""EfficientNet-B0 + shared LSTM classifier."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import tensorflow as tf

import config
from src.lava.models.tensorflow.base import KerasLightweightDetector
from src.lava.models.tensorflow.specs import EFFICIENTNET_SPEC as SPEC
from src.lava.models.tensorflow.temporal_classifier import build_temporal_classifier
from src.lava.score_semantics import validate_p_fake
from src.preprocessing import process_audio_file


def build_model(weights: str | None = "imagenet") -> tuple[tf.keras.Model, tf.keras.Model]:
    # Keras EfficientNet includes its own 1/255 rescaling and expects float RGB
    # values in [0, 255], exactly matching the shared Mel-image contract.
    backbone = tf.keras.applications.EfficientNetB0(
        input_shape=(*config.IMAGE_SIZE, config.CHANNELS),
        include_top=False,
        weights=weights,
        pooling="avg",
    )
    return build_temporal_classifier(detector_name=SPEC.name, backbone=backbone)


class EfficientNetB0LSTMDetector(KerasLightweightDetector):
    def __init__(self) -> None:
        super().__init__(SPEC, build_model)
        # TimeDistributed expands B recordings into B*6 images. Inference uses
        # a separate memory budget, never the training batch size (16).
        self.inference_batch_size = int(os.getenv("LAVA_EFFICIENTNET_INFERENCE_BATCH_SIZE", "1"))
        if self.inference_batch_size < 1:
            raise ValueError("EfficientNet inference batch size must be positive")
        self._inference_model = None
        self._inference_function = None

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load()
        if self._inference_model is not self.model:
            model = self.model
            self._inference_function = tf.function(
                lambda x: model(x, training=False),
                input_signature=[tf.TensorSpec([None, config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS], tf.float32)],
            )
            self._inference_model = model
        return self._inference_function(tf.convert_to_tensor(batch, dtype=tf.float32)).numpy().reshape(-1)

    def predict_feature_batch(self, features: np.ndarray) -> np.ndarray:
        scores = []
        for start in range(0, len(features), self.inference_batch_size):
            scores.extend(self._infer(features[start:start + self.inference_batch_size]).tolist())
        return validate_p_fake(scores).astype(np.float32)

    def predict_scores(self, audio_paths: Sequence[str]) -> np.ndarray:
        scores = []
        for start in range(0, len(audio_paths), self.inference_batch_size):
            paths = audio_paths[start:start + self.inference_batch_size]
            features = np.stack([process_audio_file(path) for path in paths])
            scores.extend(self._infer(features).tolist())
            if len(audio_paths) > 128 and start % 128 == 0:
                print(f"EfficientNet inference {min(start + len(paths), len(audio_paths))}/{len(audio_paths)}", flush=True)
        return validate_p_fake(scores).astype(np.float32)
