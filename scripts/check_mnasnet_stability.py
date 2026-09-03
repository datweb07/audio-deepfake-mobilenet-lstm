"""Short, non-production MnasNet stability check on canonical REAL/FAKE samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf

import config
from src.dataset import create_tf_dataset, get_class_weights, load_manifest_split
from src.lava.models.tensorflow.mnasnet_lstm import MnasNetLSTMDetector
from src.lava.models.tensorflow.temporal_classifier import enable_scratch_end_to_end


def balanced_subset(split: str, per_class: int) -> tuple[list[str], list[int]]:
    paths, labels = load_manifest_split(split)
    selected_paths: list[str] = []
    selected_labels: list[int] = []
    for label in (config.REAL_LABEL, config.FAKE_LABEL):
        matches = [(path, value) for path, value in zip(paths, labels) if value == label]
        selected_paths.extend(path for path, _ in matches[:per_class])
        selected_labels.extend(value for _, value in matches[:per_class])
    return selected_paths, selected_labels


class ValidationRecorder(tf.keras.callbacks.Callback):
    def __init__(self, dataset: tf.data.Dataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.records: list[dict[str, float]] = []

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        labels: list[int] = []
        scores: list[float] = []
        for features, batch_labels in self.dataset:
            scores.extend(self.model(features, training=False).numpy().reshape(-1).tolist())
            labels.extend(batch_labels.numpy().astype(int).reshape(-1).tolist())
        values = np.asarray(scores, dtype=np.float64)
        record = {
            "epoch": epoch + 1,
            "val_loss": float((logs or {}).get("val_loss", np.nan)),
            "val_auc": float(roc_auc_score(labels, values)),
            "mean_p_fake": float(values.mean()),
            "std_p_fake": float(values.std()),
            "predicted_fake_fraction": float(np.mean(values >= 0.5)),
        }
        if not all(np.isfinite(value) for value in record.values()):
            raise RuntimeError(f"MnasNet produced a non-finite stability record: {record}")
        self.records.append(record)
        print("MNASNET_STABILITY " + json.dumps(record, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-class", type=int, default=8)
    parser.add_argument("--validation-per-class", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    arguments = parser.parse_args()

    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    train_data = balanced_subset("train", arguments.train_per_class)
    validation_data = balanced_subset("validation", arguments.validation_per_class)
    train_dataset = create_tf_dataset(
        *train_data, batch_size=arguments.batch_size, training=True
    )
    validation_dataset = create_tf_dataset(
        *validation_data, batch_size=arguments.batch_size, training=False
    )
    detector = MnasNetLSTMDetector()
    model = detector.build(weights=None)
    assert detector.backbone is not None
    enable_scratch_end_to_end(detector.backbone)
    profile = detector.compile_for_scratch_training(model)
    recorder = ValidationRecorder(validation_dataset)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=arguments.epochs,
        class_weight=get_class_weights(train_data[1]),
        callbacks=[recorder],
        verbose=2,
    )
    if not recorder.records or not all(
        np.isfinite(value) for values in history.history.values() for value in values
    ):
        raise RuntimeError("MnasNet short stability training did not produce finite metrics")
    final = recorder.records[-1]
    escaped_single_class = (
        0.05 < final["predicted_fake_fraction"] < 0.95
        and final["std_p_fake"] >= 0.02
    )
    ranking_signal = final["val_auc"] > 0.5
    status = "PASS" if escaped_single_class and ranking_signal else "INCONCLUSIVE"
    report = {
        "status": status,
        "escaped_single_class": escaped_single_class,
        "ranking_signal": ranking_signal,
        "optimization_profile": profile,
        "epochs": recorder.records,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if status != "PASS":
        raise RuntimeError(
            "Short MnasNet run did not yet escape single-class prediction; "
            "increase --epochs or --train-per-class before accepting the fix"
        )


if __name__ == "__main__":
    main()
