"""One authoritative REAL/FAKE inference contract."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

import config
from src.metrics import load_threshold
from src.preprocessing import process_audio_file
from src.lava.score_semantics import PredictionResult, classify_probability


def predict_features(
    model: tf.keras.Model,
    features: np.ndarray,
    threshold: float | None = None,
) -> PredictionResult:
    expected = (config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS)
    if features.shape != expected:
        raise ValueError(f"Expected features {expected}, received {features.shape}")
    probability = float(model.predict(features[np.newaxis, ...], verbose=0)[0][0])
    return classify_probability(probability, load_threshold() if threshold is None else threshold)


def predict_audio(
    model: tf.keras.Model,
    audio_path: str,
    threshold: float | None = None,
) -> PredictionResult:
    return predict_features(model, process_audio_file(audio_path), threshold=threshold)
