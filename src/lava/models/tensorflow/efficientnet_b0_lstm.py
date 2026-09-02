"""EfficientNet-B0 + shared LSTM classifier."""

from __future__ import annotations

import tensorflow as tf

import config
from src.lava.models.tensorflow.base import KerasLightweightDetector
from src.lava.models.tensorflow.specs import EFFICIENTNET_SPEC as SPEC
from src.lava.models.tensorflow.temporal_classifier import build_temporal_classifier


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
