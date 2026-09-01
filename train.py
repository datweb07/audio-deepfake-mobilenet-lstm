"""Unified one-command/one-artifact LAVA detector training lifecycle."""

from __future__ import annotations

import json
import sys


def _early_torch_dispatch() -> bool:
    """Run native models without importing TensorFlow into the parent process."""
    model_name = None
    for index, argument in enumerate(sys.argv):
        if argument == "--model" and index + 1 < len(sys.argv):
            model_name = sys.argv[index + 1]
        elif argument.startswith("--model="):
            model_name = argument.split("=", 1)[1]
    if model_name not in {"rawnet2", "aasist"}:
        return False
    from src.lava.workers.torch_proxy import invoke_torch_worker

    request = (
        {"operation": "smoke", "model": model_name, "lengths": [48_000, 64_600]}
        if "--smoke-test" in sys.argv
        else {"operation": "train", "model": model_name, "options": {}}
    )
    print(f"=== LAVA training: {model_name} (pytorch isolated worker) ===")
    print(json.dumps(invoke_torch_worker(request, timeout=None), indent=2))
    return True


if __name__ == "__main__" and _early_torch_dispatch():
    raise SystemExit(0)

import argparse
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

import config
from src.artifacts import archive_legacy_artifacts, save_production_model
from src.dataset import create_tf_dataset, get_class_weights, load_manifest_split
from src.lava.artifacts import save_threshold as save_detector_threshold, write_json_atomic
from src.lava.data.manifest import MANIFEST_METADATA
from src.lava.registry import create, get_spec, names
from src.metrics import calibrate_threshold
from src.model import (
    compile_model as compile_baseline,
    freeze_backbone_for_warmup as freeze_baseline,
    unfreeze_backbone_for_finetuning as unfreeze_baseline,
)
from src.lava.models.tensorflow.temporal_classifier import (
    compile_binary_model,
    freeze_backbone,
    unfreeze_backbone,
)
from src.lava.training.policy import assert_test_isolation, require_validation_source
from src.utils import merge_histories, plot_training_history


def set_reproducible_seed(seed: int = config.RANDOM_SEED) -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    labels: list[float] = []
    probabilities: list[float] = []
    for features, batch_labels in dataset:
        probabilities.extend(model.predict_on_batch(features).reshape(-1).tolist())
        labels.extend(batch_labels.numpy().reshape(-1).tolist())
    return np.asarray(labels, dtype=np.int32), np.asarray(probabilities, dtype=np.float32)


def _balanced_subset(data: tuple[list[str], list[int]], per_class: int) -> tuple[list[str], list[int]]:
    selected_paths: list[str] = []
    selected_labels: list[int] = []
    for label in (config.REAL_LABEL, config.FAKE_LABEL):
        selected = [(path, value) for path, value in zip(*data) if value == label][:per_class]
        selected_paths.extend(path for path, _ in selected)
        selected_labels.extend(value for _, value in selected)
    return selected_paths, selected_labels


def _callbacks(checkpoint_path: Path, *, smoke_test: bool) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True, mode="min", verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=1 if smoke_test else config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, mode="min", verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1 if smoke_test else config.LR_REDUCTION_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE, mode="min", verbose=0,
        ),
    ]


def _restore_weights(model: tf.keras.Model, checkpoint_path: Path) -> None:
    if not checkpoint_path.is_file():
        raise RuntimeError("Training produced no validation-selected checkpoint")
    selected = tf.keras.models.load_model(checkpoint_path, compile=False)
    selected_by_name = {weight.name: weight for weight in selected.weights}
    missing = [weight.name for weight in model.weights if weight.name not in selected_by_name]
    if missing:
        raise RuntimeError(f"Checkpoint is missing model variables: {missing[:5]}")
    for weight in model.weights:
        source = selected_by_name[weight.name]
        if tuple(weight.shape) != tuple(source.shape):
            raise RuntimeError(
                f"Checkpoint variable shape mismatch for {weight.name}: {weight.shape} vs {source.shape}"
            )
        weight.assign(source)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=config.BASE_DIR, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _save_metadata(
    *, detector, threshold: float, threshold_f1: float, best_epoch: int,
    best_val_loss: float, history_path: str, warmup_epochs: int, finetune_epochs: int,
) -> None:
    spec = detector.spec
    with MANIFEST_METADATA.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    primary_eligible = spec.pretraining_status == "VERIFIED_IMAGENET"
    metadata: dict[str, Any] = {
        "detector_name": spec.name,
        "display_name": spec.display_name,
        "architecture": (
            f"6 chronological Mel images -> TimeDistributed({detector.backbone.name}) -> "
            "LSTM(128) -> Dense(64, relu) -> Dropout(0.4) -> Dense(1, sigmoid)"
        ),
        "group": spec.group,
        "framework": "tensorflow",
        "framework_version": tf.__version__,
        "training_manifest_hash": manifest["manifest_hash"],
        "input_contract": {
            "shape": list(detector.model.input_shape[1:]), "sample_rate": spec.sample_rate,
            "audio_duration": spec.audio_duration, "input_scale": [0.0, 255.0],
        },
        "label_mapping": {"REAL": 0, "FAKE": 1},
        "score_semantics": "sigmoid output = P(FAKE)",
        "pretraining": "imagenet" if primary_eligible else "scratch",
        "pretraining_status": spec.pretraining_status,
        "pretraining_stratum": "imagenet_verified" if primary_eligible else "scratch_experimental",
        "primary_pretrained_comparison_eligible": primary_eligible,
        "final_threshold": threshold,
        "threshold_source": "validation FAKE-class F1",
        "threshold_validation_f1": threshold_f1,
        "parameter_count": int(detector.model.count_params()),
        "serialized_size": spec.model_artifact.stat().st_size,
        "git_commit": _git_commit(),
        "training_seed": config.RANDOM_SEED,
        "hardware_summary": {"visible_gpus": [d.name for d in tf.config.list_physical_devices("GPU")]},
        "training_strategy": "warmup_then_finetune",
        "training_schedule": {
            "warmup_epochs_run": warmup_epochs, "finetune_epochs_run": finetune_epochs,
            "warmup_lr": config.WARMUP_LR, "finetune_lr": config.FINETUNE_LR,
            "finetune_layers": config.FINETUNE_LAYERS, "batch_normalization_frozen": True,
        },
        "selection": {
            "monitor": "validation_loss", "best_epoch": best_epoch,
            "best_value": best_val_loss, "test_used": False,
        },
        "load_smoke_test": "PASS",
        "training_history": os.path.relpath(history_path, config.BASE_DIR),
    }
    write_json_atomic(spec.metadata_artifact, metadata)


def train_tensorflow_detector(model_name: str, *, smoke_test: bool = False) -> dict[str, Any]:
    spec = get_spec(model_name)
    if spec.framework != "tensorflow":
        raise ValueError(f"{model_name} is not a TensorFlow detector")
    set_reproducible_seed()
    assert_test_isolation()
    train_data = load_manifest_split("train")
    validation_data = load_manifest_split("validation")
    if smoke_test:
        train_data = _balanced_subset(train_data, 1)
        validation_data = _balanced_subset(validation_data, 1)
    batch_size = 1 if smoke_test else config.BATCH_SIZE
    train_dataset = create_tf_dataset(*train_data, batch_size=batch_size, training=True)
    validation_dataset = create_tf_dataset(*validation_data, batch_size=batch_size, training=False)
    class_weights = get_class_weights(train_data[1])
    detector = create(model_name)
    build_weights = None if smoke_test or spec.pretraining_status != "VERIFIED_IMAGENET" else "imagenet"
    model = detector.build(weights=build_weights)
    backbone = detector.backbone
    assert backbone is not None
    checkpoint_path = Path(config.CHECKPOINTS_DIR) / f"{model_name}_best.keras"
    checkpoint_path.unlink(missing_ok=True)
    callbacks = _callbacks(checkpoint_path, smoke_test=smoke_test)

    if model_name == "mobilenetv3_lstm":
        freeze_baseline(backbone)
        compile_baseline(model, config.WARMUP_LR)
    else:
        freeze_backbone(backbone)
        compile_binary_model(model, config.WARMUP_LR)
    warmup_history = model.fit(
        train_dataset, validation_data=validation_dataset, epochs=1 if smoke_test else config.WARMUP_EPOCHS,
        class_weight=class_weights, callbacks=callbacks, verbose=0 if smoke_test else 1,
    )
    _restore_weights(model, checkpoint_path)
    if model_name == "mobilenetv3_lstm":
        unfreeze_baseline(backbone)
        compile_baseline(model, config.FINETUNE_LR)
    else:
        unfreeze_backbone(backbone, config.FINETUNE_LAYERS)
        compile_binary_model(model, config.FINETUNE_LR)
    warmup_run = len(warmup_history.epoch)
    finetune_history = model.fit(
        train_dataset, validation_data=validation_dataset, initial_epoch=warmup_run,
        epochs=warmup_run + (1 if smoke_test else config.FINETUNE_EPOCHS), class_weight=class_weights,
        callbacks=callbacks, verbose=0 if smoke_test else 1,
    )
    _restore_weights(model, checkpoint_path)
    validation_labels, scores = collect_predictions(model, validation_dataset)
    require_validation_source("validation")
    threshold, threshold_f1 = calibrate_threshold(validation_labels, scores)
    history = merge_histories((warmup_history, finetune_history))
    validation_losses = np.asarray(history["val_loss"], dtype=np.float64)

    if smoke_test:
        smoke_path = Path(config.CHECKPOINTS_DIR) / f"{model_name}_smoke.keras"
        smoke_path.unlink(missing_ok=True)
        model.save(smoke_path)
        loaded = tf.keras.models.load_model(smoke_path, compile=False)
        if loaded.output_shape != (None, 1):
            raise RuntimeError("Smoke save/load output contract failed")
        smoke_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
        return {
            "status": "SMOKE_TESTED", "model": model_name, "input_shape": list(model.input_shape),
            "embedding_dimension": int(backbone.output_shape[-1]), "output_shape": list(model.output_shape),
            "parameter_count": int(model.count_params()), "threshold_smoke_only": threshold,
        }

    if model_name == "mobilenetv3_lstm":
        detector.model = save_production_model(model)
        detector.backbone = detector.model.get_layer("time_distributed_mobilenetv3small").layer
        history_path = config.TRAINING_HISTORY_PATH
    else:
        detector.model = model
        detector.save()
        history_path = os.path.join(config.PLOTS_DIR, model_name, "training_history.png")
    history_path = plot_training_history(
        warmup_history, finetune_history, output_path=history_path, model_name=spec.display_name
    )
    save_detector_threshold(spec, threshold)
    _save_metadata(
        detector=detector, threshold=threshold, threshold_f1=threshold_f1,
        best_epoch=int(np.argmin(validation_losses)) + 1,
        best_val_loss=float(np.min(validation_losses)), history_path=history_path,
        warmup_epochs=warmup_run, finetune_epochs=len(finetune_history.epoch),
    )
    checkpoint_path.unlink(missing_ok=True)
    if model_name == "mobilenetv3_lstm":
        archive_legacy_artifacts()
    return {"status": "TRAINED", "model": model_name, "artifact": str(spec.model_artifact), "threshold": threshold}


def main(model_name: str = "mobilenetv3_lstm", *, smoke_test: bool = False) -> None:
    spec = get_spec(model_name)
    print(f"=== LAVA training: {spec.display_name} ({spec.framework}) ===")
    if spec.framework == "tensorflow":
        result = train_tensorflow_detector(model_name, smoke_test=smoke_test)
    elif smoke_test:
        from src.lava.workers.torch_proxy import invoke_torch_worker

        result = invoke_torch_worker(
            {"operation": "smoke", "model": model_name, "lengths": [48_000, 64_600]}, timeout=600
        )
    else:
        detector = create(model_name)
        detector.train()
        result = {"status": "TRAINED", "model": model_name, "artifact": str(spec.model_artifact)}
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=names(), default="mobilenetv3_lstm")
    parser.add_argument("--smoke-test", action="store_true", help="Tiny build/train/save/load test; never writes production artifacts")
    arguments = parser.parse_args()
    main(arguments.model, smoke_test=arguments.smoke_test)
