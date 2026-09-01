"""Framework-free evaluation metrics shared across TensorFlow and PyTorch."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import roc_curve

import config


def compute_eer(labels: Iterable[int], probabilities: Iterable[float]) -> tuple[float, float]:
    y_true = np.asarray(list(labels), dtype=np.int32)
    scores = np.asarray(list(probabilities), dtype=np.float32)
    if y_true.size == 0 or y_true.size != scores.size or np.unique(y_true).size != 2:
        raise ValueError("EER requires aligned non-empty REAL and FAKE scores")
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=config.FAKE_LABEL)
    fnr = 1.0 - tpr
    difference = fpr - fnr
    crossings = np.flatnonzero(np.signbit(difference[:-1]) != np.signbit(difference[1:]))
    if crossings.size == 0:
        index = int(np.argmin(np.abs(difference)))
        return float((fpr[index] + fnr[index]) / 2.0), float(thresholds[index])
    left, right = int(crossings[0]), int(crossings[0] + 1)
    denominator = difference[left] - difference[right]
    weight = 0.0 if denominator == 0 else difference[left] / denominator
    eer = fpr[left] + weight * (fpr[right] - fpr[left])
    left_threshold, right_threshold = float(thresholds[left]), float(thresholds[right])
    if np.isfinite(left_threshold) and np.isfinite(right_threshold):
        threshold = left_threshold + weight * (right_threshold - left_threshold)
    elif np.isfinite(right_threshold):
        threshold = right_threshold
    elif np.isfinite(left_threshold):
        threshold = left_threshold
    else:
        threshold = config.DEFAULT_THRESHOLD
    return float(eer), float(threshold)
