"""Two-phase training and validation-only threshold calibration."""

from __future__ import annotations

import json
import os
import random

import numpy as np
import tensorflow as tf

import config
from src.dataset import create_tf_dataset, get_class_weights, scan_files, split_dataset
from src.metrics import calibrate_threshold, get_callbacks, save_threshold
from src.model import build_hybrid_model, compile_model, configure_backbone_for_phase, parameter_counts
from src.utils import plot_history


def set_reproducible_seed() -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)


def collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    labels: list[float] = []
    probabilities: list[float] = []
    for features, batch_labels in dataset:
        batch_probabilities = model.predict_on_batch(features).reshape(-1)
        labels.extend(batch_labels.numpy().reshape(-1).tolist())
        probabilities.extend(batch_probabilities.tolist())
    return np.asarray(labels, dtype=np.int32), np.asarray(probabilities, dtype=np.float32)


def print_parameter_state(label: str, model: tf.keras.Model) -> None:
    trainable, non_trainable = parameter_counts(model)
    print(f"{label}: trainable={trainable:,}; non-trainable={non_trainable:,}")


def restore_checkpoint_weights(model: tf.keras.Model, checkpoint_path: str) -> None:
    """Restore the selected full-model checkpoint without replacing backbone references."""
    selected_model = tf.keras.models.load_model(checkpoint_path, compile=False)
    model.set_weights(selected_model.get_weights())


def main() -> None:
    set_reproducible_seed()
    print("=== MobileNetV3Small-LSTM Audio Deepfake Training ===")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"Execution device: {'GPU (' + str(len(gpus)) + ')' if gpus else 'CPU'}")

    real_files, fake_files = scan_files()
    if not real_files or not fake_files:
        raise RuntimeError("Place audio files in data/REAL and data/FAKE before training")
    train_data, val_data, test_data = split_dataset(real_files, fake_files)
    print(f"Dataset: REAL={len(real_files)}, FAKE={len(fake_files)}")
    print(f"Splits: train={len(train_data[0])}, val={len(val_data[0])}, test={len(test_data[0])}")

    train_dataset = create_tf_dataset(
        *train_data,
        batch_size=config.BATCH_SIZE,
        training=True,
    )
    val_dataset = create_tf_dataset(
        *val_data,
        batch_size=config.BATCH_SIZE,
        training=False,
    )
    class_weights = get_class_weights(train_data[1])
    print(f"Training-only class weights: {class_weights}")

    model, backbone = build_hybrid_model()
    print(f"Tensor flow: input={model.input_shape} -> embeddings=(B, {config.NUM_SEGMENTS}, {backbone.output_shape[-1]}) -> output={model.output_shape}")

    configure_backbone_for_phase(backbone, phase=1)
    compile_model(model, config.PHASE1_LR)
    print_parameter_state("Phase 1", model)
    phase1_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.PHASE1_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks(1),
    )
    plot_history(phase1_history, "phase1")
    restore_checkpoint_weights(model, config.PHASE1_MODEL_PATH)

    configure_backbone_for_phase(backbone, phase=2)
    compile_model(model, config.PHASE2_LR)
    print_parameter_state("Phase 2", model)
    phase2_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.PHASE2_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks(2),
    )
    plot_history(phase2_history, "phase2")
    restore_checkpoint_weights(model, config.PHASE2_MODEL_PATH)

    y_val, validation_probabilities = collect_predictions(model, val_dataset)
    threshold, best_f1 = calibrate_threshold(y_val, validation_probabilities)
    save_threshold(threshold)
    print(f"Validation-calibrated threshold={threshold:.3f}; F1={best_f1:.4f}")

    metadata = {
        "architecture": "MobileNetV3Small+LSTM",
        "input_shape": [config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS],
        "input_scale": [config.INPUT_VALUE_MIN, config.INPUT_VALUE_MAX],
        "label_mapping": {config.REAL_NAME: config.REAL_LABEL, config.FAKE_NAME: config.FAKE_LABEL},
        "probability_semantics": "P(FAKE)",
        "threshold_source": "validation split",
        "random_seed": config.RANDOM_SEED,
    }
    with open(config.MODEL_METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print("Training complete. Run: python evaluate.py")


if __name__ == "__main__":
    main()
