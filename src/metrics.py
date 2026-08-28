"""Callbacks, threshold calibration, and shared artifact resolution."""

from __future__ import annotations

import os
import warnings
from typing import Iterable

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

import config


def get_callbacks(phase: int) -> list[tf.keras.callbacks.Callback]:
    if phase == 1:
        model_path = config.PHASE1_MODEL_PATH
    elif phase == 2:
        model_path = config.PHASE2_MODEL_PATH
    else:
        raise ValueError("phase must be 1 or 2")
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            mode="min",
            verbose=1,
        ),
    ]


def calibrate_threshold(y_true: Iterable[int], probabilities: Iterable[float]) -> tuple[float, float]:
    labels = np.asarray(list(y_true), dtype=np.int32)
    probs = np.asarray(list(probabilities), dtype=np.float32)
    if labels.size == 0 or labels.size != probs.size:
        raise ValueError("Calibration labels/probabilities are empty or misaligned")
    candidates = np.arange(
        config.THRESHOLD_SEARCH_MIN,
        config.THRESHOLD_SEARCH_MAX + config.THRESHOLD_SEARCH_STEP / 2,
        config.THRESHOLD_SEARCH_STEP,
    )
    scores = np.asarray(
        [f1_score(labels, probs >= threshold, zero_division=0) for threshold in candidates]
    )
    best_score = float(np.max(scores))
    tied_indices = np.flatnonzero(np.isclose(scores, best_score))
    best_index = int(
        tied_indices[
            np.argmin(np.abs(candidates[tied_indices] - config.DEFAULT_THRESHOLD))
        ]
    )
    return float(candidates[best_index]), float(scores[best_index])


def save_threshold(threshold: float) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    temporary_path = config.THRESHOLD_PATH + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        handle.write(f"{threshold:.8f}\n")
    os.replace(temporary_path, config.THRESHOLD_PATH)


def load_threshold() -> float:
    if not os.path.exists(config.THRESHOLD_PATH):
        return config.DEFAULT_THRESHOLD
    try:
        with open(config.THRESHOLD_PATH, "r", encoding="utf-8") as handle:
            threshold = float(handle.read().strip())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid threshold artifact: {config.THRESHOLD_PATH}") from exc
    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError(f"Threshold outside [0, 1]: {threshold}")
    return threshold


def resolve_model_path() -> str:
    candidates = (
        config.PHASE2_MODEL_PATH,
        config.PHASE1_MODEL_PATH,
        config.LEGACY_PHASE2_MODEL_PATH,
        config.LEGACY_PHASE1_MODEL_PATH,
    )
    for path in candidates:
        if os.path.exists(path):
            if path.endswith(".h5"):
                warnings.warn(
                    "Loading a legacy pre-audit .h5 artifact. Retrain to produce a .keras "
                    "model with the corrected MobileNetV3 input scale.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return path
    raise FileNotFoundError("No trained model found. Run: python train.py")
