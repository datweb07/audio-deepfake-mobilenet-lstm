from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import tensorflow as tf

import train
from src.lava.registry import get_spec


class _FakeDetector:
    def __init__(self) -> None:
        self.spec = get_spec("mobilenetv3_lstm")
        inputs = tf.keras.layers.Input((1,), name="tiny_input")
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="tiny_output")(inputs)
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy")
        self.backbone = tf.keras.Sequential(name="tiny_backbone")

    def build(self, *, weights=None):
        return self.model


class TrainingInterruptIntegrationTest(unittest.TestCase):
    def test_keyboard_interrupt_during_finetune_preserves_stage_checkpoints(self) -> None:
        detector = _FakeDetector()
        fit_calls = 0

        def fake_fit(model_self, *args, callbacks, initial_epoch=0, **kwargs):
            nonlocal fit_calls
            fit_calls += 1
            epoch = int(initial_epoch)
            for callback in callbacks:
                callback.set_model(model_self)
                callback.set_params({"epochs": kwargs.get("epochs", epoch + 1), "verbose": 0})
                callback.on_train_begin({})
            loss = 0.27 if fit_calls == 1 else 0.40
            for callback in callbacks:
                callback.on_epoch_end(epoch, {"val_loss": loss, "loss": loss})
            if fit_calls == 2:
                raise KeyboardInterrupt
            for callback in callbacks:
                callback.on_train_end({})
            return SimpleNamespace(epoch=[epoch], history={"loss": [loss], "val_loss": [loss]})

        detector.model.fit = types.MethodType(fake_fit, detector.model)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            train.config, "CHECKPOINTS_DIR", directory
        ), patch.object(train, "load_manifest_split", return_value=(["a", "b"], [0, 1])), patch.object(
            train, "create_tf_dataset", return_value=object()
        ), patch.object(train, "get_class_weights", return_value={0: 1.0, 1: 1.0}), patch.object(
            train, "create", return_value=detector
        ), patch.object(train, "freeze_baseline"), patch.object(train, "compile_baseline"), patch.object(
            train, "unfreeze_baseline"
        ), patch.object(
            train, "batch_normalization_status", return_value={"total": 0, "trainable": 0, "frozen": 0}
        ):
            with self.assertRaises(KeyboardInterrupt):
                train.train_tensorflow_detector("mobilenetv3_lstm", smoke_test=False)
            lifecycle = Path(directory) / "mobilenetv3_lstm"
            warmup = lifecycle / "warmup_best.keras"
            finetune = lifecycle / "finetune_best.keras"
            state_path = lifecycle / "lifecycle_state.json"
            tf.keras.models.load_model(warmup, compile=False)
            tf.keras.models.load_model(finetune, compile=False)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "INTERRUPTED")
            self.assertEqual(state["interrupted_stage"], "finetune")
            self.assertFalse(state["production_model_saved"])
            self.assertFalse(state["selection_finalized"])


if __name__ == "__main__":
    unittest.main()
