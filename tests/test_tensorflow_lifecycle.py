from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.lava.training.tensorflow_lifecycle import (
    finalize_global_selection,
    initial_lifecycle_state,
    lifecycle_paths,
    load_selected_model,
    mark_interrupted,
    select_global_best,
    stage_callbacks,
)
from src.metrics import calibrate_threshold


def _tiny_model(kernel: float = 1.0, bias: float = 0.0) -> tf.keras.Model:
    inputs = tf.keras.layers.Input((1,))
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer=tf.keras.initializers.Constant(kernel),
        bias_initializer=tf.keras.initializers.Constant(bias),
    )(inputs)
    return tf.keras.Model(inputs, outputs)


class TensorFlowLifecycleTest(unittest.TestCase):
    def test_a_warmup_wins_global_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = lifecycle_paths("model", root=directory)
            paths.warmup_checkpoint.touch(); paths.finetune_checkpoint.touch()
            selected = select_global_best(
                warmup_val_loss=0.20, warmup_epoch=10, warmup_checkpoint=paths.warmup_checkpoint,
                finetune_val_loss=0.40, finetune_epoch=20, finetune_checkpoint=paths.finetune_checkpoint,
            )
            self.assertEqual(selected.stage, "warmup")
            self.assertEqual(selected.epoch, 10)

    def test_b_finetune_wins_global_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = lifecycle_paths("model", root=directory)
            paths.warmup_checkpoint.touch(); paths.finetune_checkpoint.touch()
            selected = select_global_best(
                warmup_val_loss=0.20, warmup_epoch=10, warmup_checkpoint=paths.warmup_checkpoint,
                finetune_val_loss=0.15, finetune_epoch=80, finetune_checkpoint=paths.finetune_checkpoint,
            )
            self.assertEqual(selected.stage, "finetune")
            self.assertEqual(selected.epoch, 80)

    def test_c_adaptation_valley_is_not_compared_to_warmup_for_early_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = lifecycle_paths("model", root=directory)
            state = initial_lifecycle_state("model", manifest_hash="test", seed=42)
            state["warmup_best_val_loss"] = 0.27
            state["warmup_best_epoch"] = 10
            paths.warmup_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            paths.warmup_checkpoint.touch()
            callbacks = stage_callbacks(
                stage="finetune", paths=paths, state=state,
                early_stopping_patience=2, lr_reduction_patience=2, verbose=0,
            )
            early_stopping = next(
                callback for callback in callbacks if isinstance(callback, tf.keras.callbacks.EarlyStopping)
            )
            model = _tiny_model()
            model.stop_training = False
            early_stopping.set_model(model)
            early_stopping.on_train_begin()
            losses = [0.49, 0.48, 0.53, 0.42, 0.32, 0.25, 0.20, 0.15]
            for epoch, loss in enumerate(losses):
                early_stopping.on_epoch_end(epoch, {"val_loss": loss})
                self.assertFalse(model.stop_training)
            paths.finetune_checkpoint.touch()
            selected = select_global_best(
                warmup_val_loss=0.27, warmup_epoch=10, warmup_checkpoint=paths.warmup_checkpoint,
                finetune_val_loss=min(losses), finetune_epoch=18,
                finetune_checkpoint=paths.finetune_checkpoint,
            )
            self.assertEqual(selected.stage, "finetune")
            self.assertAlmostEqual(selected.val_loss, 0.15)

    def test_stage_callbacks_are_independent_and_stage_local(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = lifecycle_paths("model", root=directory)
            state = initial_lifecycle_state("model", manifest_hash="test", seed=42)
            warmup = stage_callbacks(
                stage="warmup", paths=paths, state=state,
                early_stopping_patience=3, lr_reduction_patience=2, verbose=0,
            )
            finetune = stage_callbacks(
                stage="finetune", paths=paths, state=state,
                early_stopping_patience=7, lr_reduction_patience=4, verbose=0,
            )
            warmup_checkpoint = next(c for c in warmup if isinstance(c, tf.keras.callbacks.ModelCheckpoint))
            finetune_checkpoint = next(c for c in finetune if isinstance(c, tf.keras.callbacks.ModelCheckpoint))
            warmup_stop = next(c for c in warmup if isinstance(c, tf.keras.callbacks.EarlyStopping))
            finetune_stop = next(c for c in finetune if isinstance(c, tf.keras.callbacks.EarlyStopping))
            warmup_lr = next(c for c in warmup if isinstance(c, tf.keras.callbacks.ReduceLROnPlateau))
            finetune_lr = next(c for c in finetune if isinstance(c, tf.keras.callbacks.ReduceLROnPlateau))
            self.assertIsNot(warmup_stop, finetune_stop)
            self.assertIsNot(warmup_lr, finetune_lr)
            self.assertEqual(warmup_stop.patience, 3)
            self.assertEqual(finetune_stop.patience, 7)
            self.assertTrue(str(warmup_checkpoint.filepath).endswith("warmup_best.keras"))
            self.assertTrue(str(finetune_checkpoint.filepath).endswith("finetune_best.keras"))

    def test_d_keyboard_interrupt_preserves_both_valid_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = lifecycle_paths("model", root=directory)
            paths.directory.mkdir(parents=True, exist_ok=True)
            _tiny_model(1.0, 0.0).save(paths.warmup_checkpoint)
            _tiny_model(2.0, -1.0).save(paths.finetune_checkpoint)
            state = initial_lifecycle_state("model", manifest_hash="test", seed=42)
            state.update({
                "warmup_best_val_loss": 0.27, "warmup_best_epoch": 10,
                "finetune_best_val_loss": 0.25, "finetune_best_epoch": 17,
            })
            try:
                raise KeyboardInterrupt
            except KeyboardInterrupt:
                mark_interrupted(state, paths, stage="finetune")
            tf.keras.models.load_model(paths.warmup_checkpoint, compile=False)
            tf.keras.models.load_model(paths.finetune_checkpoint, compile=False)
            persisted = json.loads(paths.state.read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "INTERRUPTED")
            self.assertIsNone(persisted["global_best_stage"])
            self.assertEqual(persisted["recovery_best_stage"], "finetune")
            self.assertFalse(persisted["selection_finalized"])
            self.assertFalse(persisted["production_model_saved"])

    def test_e_calibration_uses_selected_global_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = lifecycle_paths("model", root=directory)
            paths.directory.mkdir(parents=True, exist_ok=True)
            _tiny_model(-10.0, 5.0).save(paths.warmup_checkpoint)
            _tiny_model(10.0, -5.0).save(paths.finetune_checkpoint)
            state = initial_lifecycle_state("model", manifest_hash="test", seed=42)
            state.update({
                "warmup_best_val_loss": 0.20, "warmup_best_epoch": 10,
                "finetune_best_val_loss": 0.15, "finetune_best_epoch": 18,
            })
            selection = finalize_global_selection(state, paths)
            selected = load_selected_model(selection)
            scores = selected(np.asarray([[0.0], [1.0]], dtype=np.float32), training=False).numpy().reshape(-1)
            threshold, _ = calibrate_threshold([0, 1], scores)
            self.assertEqual(selection.stage, "finetune")
            self.assertLess(scores[0], threshold)
            self.assertGreaterEqual(scores[1], threshold)


if __name__ == "__main__":
    unittest.main()
