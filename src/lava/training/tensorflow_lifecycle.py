"""Stage-local TensorFlow optimization with explicit post-stage global selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, MutableMapping

import tensorflow as tf

import config
from src.lava.artifacts import write_json_atomic


STAGES = ("warmup", "finetune")


@dataclass(frozen=True)
class LifecyclePaths:
    directory: Path
    warmup_checkpoint: Path
    finetune_checkpoint: Path
    state: Path

    def checkpoint_for(self, stage: str) -> Path:
        if stage == "warmup":
            return self.warmup_checkpoint
        if stage == "finetune":
            return self.finetune_checkpoint
        raise ValueError(f"Unknown lifecycle stage: {stage}")


@dataclass(frozen=True)
class GlobalBestSelection:
    stage: str
    val_loss: float
    epoch: int
    checkpoint: Path


def lifecycle_paths(detector_name: str, root: str | Path | None = None) -> LifecyclePaths:
    directory = Path(root) if root is not None else Path(config.CHECKPOINTS_DIR) / detector_name
    return LifecyclePaths(
        directory=directory,
        warmup_checkpoint=directory / "warmup_best.keras",
        finetune_checkpoint=directory / "finetune_best.keras",
        state=directory / "lifecycle_state.json",
    )


def initial_lifecycle_state(
    detector_name: str,
    *,
    manifest_hash: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "detector_name": detector_name,
        "training_manifest_hash": manifest_hash,
        "seed": seed,
        "status": "INITIALIZED",
        "selection_finalized": False,
        "warmup_best_val_loss": None,
        "warmup_best_epoch": None,
        "finetune_best_val_loss": None,
        "finetune_best_epoch": None,
        "global_best_val_loss": None,
        "global_best_epoch": None,
        "global_best_stage": None,
        "warmup_lr": config.WARMUP_LR,
        "finetune_initial_lr": config.FINETUNE_LR,
        "finetune_final_lr": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_lifecycle_state(path: Path, state: MutableMapping[str, Any]) -> None:
    write_json_atomic(path, dict(state))


def select_global_best(
    *,
    warmup_val_loss: float | None,
    warmup_epoch: int | None,
    warmup_checkpoint: Path,
    finetune_val_loss: float | None,
    finetune_epoch: int | None,
    finetune_checkpoint: Path,
) -> GlobalBestSelection:
    """Choose the lower stage-best validation loss; never controls stage stopping."""
    candidates: list[GlobalBestSelection] = []
    if warmup_val_loss is not None and warmup_epoch is not None and warmup_checkpoint.is_file():
        candidates.append(
            GlobalBestSelection("warmup", float(warmup_val_loss), int(warmup_epoch), warmup_checkpoint)
        )
    if finetune_val_loss is not None and finetune_epoch is not None and finetune_checkpoint.is_file():
        candidates.append(
            GlobalBestSelection("finetune", float(finetune_val_loss), int(finetune_epoch), finetune_checkpoint)
        )
    if not candidates:
        raise RuntimeError("No valid stage checkpoint is available for global selection")
    # Stable tie policy favors warm-up, avoiding an unnecessary changed-backbone artifact.
    return min(candidates, key=lambda item: (item.val_loss, item.stage != "warmup"))


def _refresh_global_fields(
    state: MutableMapping[str, Any], paths: LifecyclePaths
) -> GlobalBestSelection | None:
    try:
        selected = select_global_best(
            warmup_val_loss=state.get("warmup_best_val_loss"),
            warmup_epoch=state.get("warmup_best_epoch"),
            warmup_checkpoint=paths.warmup_checkpoint,
            finetune_val_loss=state.get("finetune_best_val_loss"),
            finetune_epoch=state.get("finetune_best_epoch"),
            finetune_checkpoint=paths.finetune_checkpoint,
        )
    except RuntimeError:
        return None
    state["global_best_val_loss"] = selected.val_loss
    state["global_best_epoch"] = selected.epoch
    state["global_best_stage"] = selected.stage
    return selected


class StageStateRecorder(tf.keras.callbacks.Callback):
    """Persist each stage's local best independently of Keras callback internals."""

    def __init__(
        self,
        *,
        stage: str,
        state: MutableMapping[str, Any],
        paths: LifecyclePaths,
    ) -> None:
        super().__init__()
        if stage not in STAGES:
            raise ValueError(f"Unknown lifecycle stage: {stage}")
        self.stage = stage
        self.state = state
        self.paths = paths

    def on_train_begin(self, logs=None) -> None:
        self.state["status"] = f"{self.stage.upper()}_RUNNING"
        self.state["active_stage"] = self.stage
        write_lifecycle_state(self.paths.state, self.state)

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        logs = logs or {}
        value = logs.get("val_loss")
        if value is None:
            return
        loss = float(value)
        best_key = f"{self.stage}_best_val_loss"
        epoch_key = f"{self.stage}_best_epoch"
        current = self.state.get(best_key)
        if current is None or loss < float(current):
            self.state[best_key] = loss
            # Keras epochs are zero-based and remain lifecycle-global with initial_epoch.
            self.state[epoch_key] = int(epoch) + 1
        self.state[f"{self.stage}_epochs_completed"] = int(epoch) + 1
        write_lifecycle_state(self.paths.state, self.state)

    def on_train_end(self, logs=None) -> None:
        self.state["status"] = f"{self.stage.upper()}_COMPLETE"
        self.state["active_stage"] = None
        write_lifecycle_state(self.paths.state, self.state)


def stage_callbacks(
    *,
    stage: str,
    paths: LifecyclePaths,
    state: MutableMapping[str, Any],
    early_stopping_patience: int,
    lr_reduction_patience: int,
    verbose: int,
) -> list[tf.keras.callbacks.Callback]:
    """Return a fresh checkpoint/stopping/scheduler set for exactly one stage."""
    checkpoint = paths.checkpoint_for(stage)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=verbose,
        ),
        StageStateRecorder(stage=stage, state=state, paths=paths),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode="min",
            verbose=verbose,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=lr_reduction_patience,
            min_lr=config.MIN_LEARNING_RATE,
            mode="min",
            verbose=verbose,
        ),
    ]


def finalize_global_selection(
    state: MutableMapping[str, Any], paths: LifecyclePaths
) -> GlobalBestSelection:
    selected = _refresh_global_fields(state, paths)
    if selected is None:
        raise RuntimeError("Training completed without a recoverable stage checkpoint")
    state["selection_finalized"] = True
    state["status"] = "LIFECYCLE_COMPLETE"
    state["active_stage"] = None
    write_lifecycle_state(paths.state, state)
    return selected


def mark_interrupted(
    state: MutableMapping[str, Any], paths: LifecyclePaths, *, stage: str | None
) -> None:
    # Recovery guidance is deliberately not a finalized global lifecycle selection.
    try:
        recovery = select_global_best(
            warmup_val_loss=state.get("warmup_best_val_loss"),
            warmup_epoch=state.get("warmup_best_epoch"),
            warmup_checkpoint=paths.warmup_checkpoint,
            finetune_val_loss=state.get("finetune_best_val_loss"),
            finetune_epoch=state.get("finetune_best_epoch"),
            finetune_checkpoint=paths.finetune_checkpoint,
        )
    except RuntimeError:
        recovery = None
    if recovery is not None:
        state["recovery_best_stage"] = recovery.stage
        state["recovery_best_val_loss"] = recovery.val_loss
        state["recovery_best_epoch"] = recovery.epoch
    state["status"] = "INTERRUPTED"
    state["interrupted_stage"] = stage
    state["selection_finalized"] = False
    state["active_stage"] = None
    state["production_model_saved"] = False
    write_lifecycle_state(paths.state, state)


def load_selected_model(selection: GlobalBestSelection) -> tf.keras.Model:
    return tf.keras.models.load_model(selection.checkpoint, compile=False)


def optimizer_learning_rate(model: tf.keras.Model) -> float | None:
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        return None
    return float(tf.keras.backend.get_value(optimizer.learning_rate))


def batch_normalization_status(backbone: tf.keras.Model) -> dict[str, int]:
    layers = [layer for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
    trainable = sum(1 for layer in layers if layer.trainable)
    return {"total": len(layers), "trainable": trainable, "frozen": len(layers) - trainable}
