"""Plotting helpers for one complete detector-training lifecycle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

import config


def merge_histories(histories: Sequence[tf.keras.callbacks.History]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for metric, values in history.history.items():
            merged.setdefault(metric, []).extend(float(value) for value in values)
    return merged


def plot_training_history(
    warmup_history: tf.keras.callbacks.History,
    finetune_history: tf.keras.callbacks.History,
    *,
    output_path: str | None = None,
    model_name: str = "MobileNetV3Small-LSTM",
    lifecycle_state: Mapping[str, Any] | None = None,
) -> str:
    """Save one plot across warm-up and fine-tuning epochs."""
    history = merge_histories((warmup_history, finetune_history))
    warmup_epochs = len(warmup_history.epoch)
    total_epochs = len(history.get("loss", []))
    epoch_axis = list(range(1, total_epochs + 1))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    panels = (
        (axes[0], "accuracy", "val_accuracy", "Accuracy"),
        (axes[1], "loss", "val_loss", "Binary cross-entropy"),
    )
    for axis, train_key, validation_key, title in panels:
        if train_key in history:
            axis.plot(epoch_axis, history[train_key], label="Train")
        if validation_key in history:
            axis.plot(epoch_axis, history[validation_key], label="Validation")
        if finetune_history.epoch:
            axis.axvline(
                warmup_epochs + 0.5,
                color="tab:red",
                linestyle="--",
                linewidth=1.2,
                label="Fine-tuning starts",
            )
        if title == "Binary cross-entropy" and lifecycle_state:
            for stage, color, marker in (
                ("warmup", "tab:blue", "o"),
                ("finetune", "tab:orange", "s"),
            ):
                best_epoch = lifecycle_state.get(f"{stage}_best_epoch")
                best_loss = lifecycle_state.get(f"{stage}_best_val_loss")
                if best_epoch is not None and best_loss is not None:
                    axis.scatter(
                        [best_epoch], [best_loss], color=color, marker=marker, s=45,
                        zorder=4, label=f"{stage.capitalize()} best",
                    )
            global_epoch = lifecycle_state.get("global_best_epoch")
            global_loss = lifecycle_state.get("global_best_val_loss")
            if global_epoch is not None and global_loss is not None:
                axis.scatter(
                    [global_epoch], [global_loss], color="black", marker="*", s=100,
                    zorder=5, label="Global best",
                )
        axis.set_xlabel("Lifecycle epoch")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend()

    figure.suptitle(f"{model_name} complete training lifecycle")
    figure.tight_layout()
    destination = output_path or config.TRAINING_HISTORY_PATH
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination
