"""Validation-only diagnostics for silent score-distribution collapse."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


class ValidationDistributionDiagnostic(tf.keras.callbacks.Callback):
    """Log validation P(FAKE) distribution without changing training control flow."""

    def __init__(
        self,
        validation_dataset: tf.data.Dataset,
        *,
        start_epoch: int,
        interval: int,
    ) -> None:
        super().__init__()
        self.validation_dataset = validation_dataset
        self.start_epoch = max(1, int(start_epoch))
        self.interval = max(1, int(interval))

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        lifecycle_epoch = int(epoch) + 1
        if lifecycle_epoch < self.start_epoch or (
            lifecycle_epoch - self.start_epoch
        ) % self.interval:
            return
        scores: list[float] = []
        for features, _ in self.validation_dataset:
            probabilities = self.model(features, training=False)
            scores.extend(tf.reshape(probabilities, (-1,)).numpy().tolist())
        values = np.asarray(scores, dtype=np.float64)
        if not values.size:
            return
        fake_fraction = float(np.mean(values >= 0.5))
        mean, std = float(np.mean(values)), float(np.std(values))
        print(
            "Validation score distribution — "
            f"mean P(FAKE)={mean:.4f}, std={std:.4f}, "
            f"predicted REAL={(1.0 - fake_fraction) * 100:.2f}%, "
            f"predicted FAKE={fake_fraction * 100:.2f}%"
        )
        logs = logs or {}
        train_auc = logs.get("auc")
        validation_auc = logs.get("val_auc")
        near_random_auc = (
            train_auc is not None
            and validation_auc is not None
            and abs(float(train_auc) - 0.5) <= 0.03
            and abs(float(validation_auc) - 0.5) <= 0.03
        )
        collapsed_distribution = std <= 0.02 or fake_fraction <= 0.01 or fake_fraction >= 0.99
        if near_random_auc and collapsed_distribution:
            print(
                "WARNING: detector shows near-random ranking signal and a collapsed validation "
                "score distribution. Check initialization/training policy before continuing."
            )
