"""Fast structural smoke test for stage-local training and global selection."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf

import config
from src.artifacts import validate_model_contract
from src.lava.training.tensorflow_lifecycle import (
    batch_normalization_status,
    finalize_global_selection,
    initial_lifecycle_state,
    lifecycle_paths,
    load_selected_model,
    stage_callbacks,
    write_lifecycle_state,
)
from src.model import (
    build_hybrid_model,
    compile_model,
    freeze_backbone_for_warmup,
    parameter_counts,
    unfreeze_backbone_for_finetuning,
)
from src.utils import plot_training_history


def main() -> None:
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    with tempfile.TemporaryDirectory(prefix="lava_training_smoke_") as temporary_dir:
        paths = lifecycle_paths("mobilenetv3_lstm", root=os.path.join(temporary_dir, "lifecycle"))
        paths.directory.mkdir(parents=True)
        state = initial_lifecycle_state(
            "mobilenetv3_lstm", manifest_hash="SMOKE", seed=config.RANDOM_SEED
        )
        write_lifecycle_state(paths.state, state)
        model, backbone = build_hybrid_model(weights=None)
        validate_model_contract(model)
        inputs = tf.zeros((1, config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS))
        labels = tf.constant([0.0])
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(1)

        freeze_backbone_for_warmup(backbone)
        compile_model(model, config.WARMUP_LR)
        warmup_counts = parameter_counts(model)
        warmup_history = model.fit(
            dataset,
            validation_data=dataset,
            epochs=1,
            callbacks=stage_callbacks(
                stage="warmup", paths=paths, state=state,
                early_stopping_patience=1, lr_reduction_patience=1, verbose=0,
            ),
            verbose=0,
        )

        model = tf.keras.models.load_model(paths.warmup_checkpoint, compile=False)
        backbone = next(
            layer.layer for layer in model.layers if isinstance(layer, tf.keras.layers.TimeDistributed)
        )
        unfreeze_backbone_for_finetuning(backbone)
        compile_model(model, config.FINETUNE_LR)
        finetune_counts = parameter_counts(model)
        bn_status = batch_normalization_status(backbone)
        finetune_history = model.fit(
            dataset,
            validation_data=dataset,
            initial_epoch=1,
            epochs=2,
            callbacks=stage_callbacks(
                stage="finetune", paths=paths, state=state,
                early_stopping_patience=1, lr_reduction_patience=1, verbose=0,
            ),
            verbose=0,
        )
        selection = finalize_global_selection(state, paths)
        selected = load_selected_model(selection)
        validate_model_contract(selected)
        final_path = os.path.join(temporary_dir, "final.keras")
        selected.save(final_path)
        validate_model_contract(tf.keras.models.load_model(final_path, compile=False))
        history_path = plot_training_history(
            warmup_history,
            finetune_history,
            output_path=os.path.join(temporary_dir, "training_history.png"),
            lifecycle_state=state,
        )

        if finetune_counts[0] <= warmup_counts[0]:
            raise AssertionError("Fine-tuning did not increase trainable parameters")
        if bn_status["trainable"] != 0:
            raise AssertionError("BatchNormalization must remain frozen")
        print(json.dumps({
            "input_shape": selected.input_shape,
            "output_shape": selected.output_shape,
            "warmup_checkpoint": paths.warmup_checkpoint.is_file(),
            "finetune_checkpoint": paths.finetune_checkpoint.is_file(),
            "global_best_stage": selection.stage,
            "batch_normalization": bn_status,
            "final_save_load": True,
            "combined_history": os.path.isfile(history_path),
        }, indent=2))


if __name__ == "__main__":
    main()
