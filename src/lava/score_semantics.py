"""One authoritative REAL=0, FAKE=1, P(FAKE) score contract."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import config


@dataclass(frozen=True)
class PredictionResult:
    prediction: str
    confidence: float
    probability_fake: float
    threshold: float


def validate_p_fake(values: np.ndarray | list[float] | float) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("P(FAKE) contains NaN or infinite values")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("P(FAKE) must be in [0, 1]")
    return probabilities


def binary_logits_to_p_fake(logits: np.ndarray, *, fake_index: int) -> np.ndarray:
    """Convert two-class logits to P(FAKE), independent of native class ordering."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected two-class logits shaped (B, 2), received {values.shape}")
    if fake_index not in (0, 1):
        raise ValueError("fake_index must be 0 or 1")
    shifted = values - np.max(values, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return validate_p_fake(probabilities[:, fake_index]).astype(np.float32)


def decisions_from_p_fake(values: np.ndarray | list[float], threshold: float) -> np.ndarray:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    probabilities = validate_p_fake(values)
    return np.where(probabilities >= threshold, config.FAKE_LABEL, config.REAL_LABEL).astype(np.int32)


def classify_probability(probability_fake: float, threshold: float) -> PredictionResult:
    probability = float(validate_p_fake(probability_fake))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    prediction = config.FAKE_NAME if probability >= threshold else config.REAL_NAME
    confidence = probability if prediction == config.FAKE_NAME else 1.0 - probability
    return PredictionResult(prediction, confidence, probability, threshold)
