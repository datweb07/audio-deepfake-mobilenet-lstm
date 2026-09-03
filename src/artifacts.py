"""Single production-model artifact contract for the root implementation."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

import config


PRODUCTION_MODEL_NOT_FOUND = "Production model not found. Run: python train.py"


def validate_model_contract(model: tf.keras.Model) -> None:
    """Reject artifacts that do not implement the production detector contract."""
    expected_input = (None, config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS)
    if tuple(model.input_shape) != expected_input:
        raise ValueError(
            f"Production model input mismatch: expected {expected_input}, got {model.input_shape}"
        )
    if tuple(model.output_shape) != (None, 1):
        raise ValueError(
            f"Production model output mismatch: expected (None, 1), got {model.output_shape}"
        )

    temporal_wrapper = model.get_layer("time_distributed_mobilenetv3small")
    backbone = temporal_wrapper.layer
    if (
        not isinstance(backbone, tf.keras.Model)
        or backbone.name.lower() != "mobilenetv3small"
        or tuple(backbone.input_shape) != (None, *config.IMAGE_SIZE, config.CHANNELS)
        or backbone.output_shape[-1] != 576
    ):
        raise ValueError("Production model must contain the MobileNetV3Small embedding backbone")

    lstm = model.get_layer("temporal_lstm")
    if not isinstance(lstm, tf.keras.layers.LSTM) or lstm.units != config.LSTM_UNITS:
        raise ValueError(f"Production model must contain LSTM({config.LSTM_UNITS})")
    dense = model.get_layer("classifier_dense")
    if (
        dense.units != config.DENSE_UNITS
        or tf.keras.activations.serialize(dense.activation) != "relu"
    ):
        raise ValueError(
            f"Production classifier must contain Dense({config.DENSE_UNITS}, activation='relu')"
        )
    dropout = model.get_layer("classifier_dropout")
    if (
        not isinstance(dropout, tf.keras.layers.Dropout)
        or abs(dropout.rate - config.DROPOUT_RATE) > 1e-9
    ):
        raise ValueError(f"Production classifier must contain Dropout({config.DROPOUT_RATE})")
    output = model.get_layer("probability_fake")
    if output.units != 1 or tf.keras.activations.serialize(output.activation) != "sigmoid":
        raise ValueError("Production output must be one sigmoid unit representing P(FAKE)")


def _load_production_weights(weights_path: str) -> tf.keras.Model:
    """Rebuild the exact architecture and restore a weights-only deployment copy."""
    # Import lazily to keep the artifact module independent of model builders at
    # import time and to make the deployment fallback explicit and testable.
    from src.model import build_hybrid_model

    model, _ = build_hybrid_model(weights=None)
    model.load_weights(weights_path)
    validate_model_contract(model)
    return model


def _save_numpy_weights(model: tf.keras.Model, weights_path: str) -> None:
    """Store ordered numeric tensors without Keras layer serialization."""
    tensors = model.get_weights()
    payload: dict[str, np.ndarray] = {
        "format_version": np.asarray([1], dtype=np.int32),
        "weight_count": np.asarray([len(tensors)], dtype=np.int32),
    }
    payload.update({f"weight_{index:04d}": tensor for index, tensor in enumerate(tensors)})
    np.savez_compressed(weights_path, **payload)


def _load_production_numpy_weights(weights_path: str) -> tf.keras.Model:
    """Restore tensors by validated order/shape, bypassing Keras deserializers."""
    from src.model import build_hybrid_model

    model, _ = build_hybrid_model(weights=None)
    expected = model.get_weights()
    with np.load(weights_path, allow_pickle=False) as archive:
        version = int(archive["format_version"][0])
        count = int(archive["weight_count"][0])
        if version != 1:
            raise ValueError(f"Unsupported NumPy deployment weights version: {version}")
        if count != len(expected):
            raise ValueError(
                f"NumPy deployment weight count mismatch: expected {len(expected)}, got {count}"
            )
        restored = []
        for index, expected_tensor in enumerate(expected):
            key = f"weight_{index:04d}"
            if key not in archive:
                raise ValueError(f"NumPy deployment weights missing tensor: {key}")
            tensor = archive[key]
            if tensor.shape != expected_tensor.shape:
                raise ValueError(
                    f"NumPy deployment tensor {key} shape mismatch: "
                    f"expected {expected_tensor.shape}, got {tensor.shape}"
                )
            restored.append(tensor)
    model.set_weights(restored)
    validate_model_contract(model)
    return model


def load_production_model(
    *,
    compile: bool = False,
    model_path: str | None = None,
    weights_path: str | None = None,
    numpy_weights_path: str | None = None,
) -> tf.keras.Model:
    """Load MobileNet production weights without silently using legacy checkpoints.

    The full ``.keras`` artifact remains the preferred local path.  Some Linux
    Keras 2.15 deployments fail to restore nested MobileNetV3 variables from
    that archive (``Layer 'Conv' expected 1 variables, but received 0``).  A
    verified weights-only copy avoids deserializing the nested architecture:
    production code rebuilds the unchanged architecture, then restores weights.
    """
    resolved_model_path = model_path or config.MODEL_PATH
    resolved_weights_path = weights_path or config.MODEL_WEIGHTS_PATH
    resolved_numpy_weights_path = numpy_weights_path or config.MODEL_NUMPY_WEIGHTS_PATH
    model_exists = os.path.isfile(resolved_model_path)
    weights_exist = os.path.isfile(resolved_weights_path)
    numpy_weights_exist = os.path.isfile(resolved_numpy_weights_path)

    if not model_exists and not weights_exist and not numpy_weights_exist:
        raise FileNotFoundError(PRODUCTION_MODEL_NOT_FOUND)

    if model_exists:
        try:
            model = tf.keras.models.load_model(resolved_model_path, compile=compile)
            validate_model_contract(model)
            return model
        except (ValueError, TypeError, OSError) as archive_error:
            if not numpy_weights_exist and not weights_exist:
                raise RuntimeError(
                    "Production .keras model could not be loaded and its deployment "
                    f"weights fallbacks are missing: {resolved_numpy_weights_path}, "
                    f"{resolved_weights_path}"
                ) from archive_error
            print(
                "[LAVA] Full-model load failed; using serialization-independent "
                "MobileNet weights "
                f"fallback ({type(archive_error).__name__}: {archive_error})"
            )

    if numpy_weights_exist:
        return _load_production_numpy_weights(resolved_numpy_weights_path)
    return _load_production_weights(resolved_weights_path)


def save_production_model(model: tf.keras.Model) -> tf.keras.Model:
    """Atomically publish verified full-model and weights-only artifacts."""
    validate_model_contract(model)
    # Export an uncompiled clone so the deployment artifact never carries an
    # optimizer slot state from either internal training stage.
    export_model = tf.keras.models.clone_model(model)
    export_model.set_weights(model.get_weights())
    validate_model_contract(export_model)
    pending_path = f"{config.MODEL_PATH}.pending.keras"
    pending_weights_path = f"{config.MODEL_WEIGHTS_PATH}.pending.weights.h5"
    pending_numpy_weights_path = f"{config.MODEL_NUMPY_WEIGHTS_PATH}.pending.npz"
    if os.path.exists(pending_path):
        os.remove(pending_path)
    if os.path.exists(pending_weights_path):
        os.remove(pending_weights_path)
    if os.path.exists(pending_numpy_weights_path):
        os.remove(pending_numpy_weights_path)
    try:
        export_model.save(pending_path)
        verified = tf.keras.models.load_model(pending_path, compile=False)
        validate_model_contract(verified)
        export_model.save_weights(pending_weights_path)
        weights_verified = _load_production_weights(pending_weights_path)
        if weights_verified.count_params() != export_model.count_params():
            raise ValueError("Weights-only deployment artifact parameter count mismatch")
        # Export ordered tensors from the freshly rebuilt canonical model, not
        # from the deserialized/training object. Nested Keras models can expose
        # a different get_weights() order when their trainable state differs.
        _save_numpy_weights(weights_verified, pending_numpy_weights_path)
        numpy_verified = _load_production_numpy_weights(pending_numpy_weights_path)
        if numpy_verified.count_params() != export_model.count_params():
            raise ValueError("NumPy deployment artifact parameter count mismatch")
        os.replace(pending_numpy_weights_path, config.MODEL_NUMPY_WEIGHTS_PATH)
        os.replace(pending_weights_path, config.MODEL_WEIGHTS_PATH)
        os.replace(pending_path, config.MODEL_PATH)
    finally:
        if os.path.exists(pending_path):
            os.remove(pending_path)
        if os.path.exists(pending_weights_path):
            os.remove(pending_weights_path)
        if os.path.exists(pending_numpy_weights_path):
            os.remove(pending_numpy_weights_path)
    return load_production_model(compile=False)


def _move_preserving_existing(source: Path, destination_dir: Path) -> str:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if destination.exists():
        stem, suffix = destination.stem, destination.suffix
        index = 1
        while destination.exists():
            destination = destination_dir / f"{stem}_{index}{suffix}"
            index += 1
    shutil.move(str(source), str(destination))
    return str(destination)


def archive_legacy_artifacts() -> list[str]:
    """Move known stage artifacts only after production verification succeeds."""
    archived: list[str] = []
    for name in (
        "best_model_phase1.keras",
        "best_model_phase2.keras",
        "best_model_phase1.h5",
        "best_model_phase2.h5",
    ):
        source = Path(config.MODELS_DIR) / name
        if source.is_file():
            archived.append(_move_preserving_existing(source, Path(config.LEGACY_MODELS_DIR)))
    for name in ("training_history_phase1.png", "training_history_phase2.png"):
        source = Path(config.PLOTS_DIR) / name
        if source.is_file():
            archived.append(_move_preserving_existing(source, Path(config.LEGACY_TRAINING_DIR)))
    return archived
