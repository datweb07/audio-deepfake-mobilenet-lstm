from __future__ import annotations

from dataclasses import replace
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import tensorflow as tf

import config
import train
from src.lava.contracts import Initialization, TrainingPolicy
from src.lava.models.tensorflow.efficientnet_b0_lstm import build_model as build_efficientnet
from src.lava.models.tensorflow.mnasnet_lstm import build_model as build_mnasnet
from src.lava.models.tensorflow.mobilenetv3_lstm import MobileNetV3LSTMDetector
from src.lava.models.tensorflow.shufflenetv2_lstm import build_model as build_shufflenet
from src.lava.models.tensorflow.specs import MOBILENET_SPEC
from src.lava.models.tensorflow.specs import SHUFFLENET_SPEC
from src.lava.models.tensorflow.temporal_classifier import (
    enable_scratch_end_to_end,
    freeze_backbone,
    parameter_status,
    unfreeze_backbone,
)
from src.lava.registry import get_spec
from src.lava.training.policy import resolve_training_policy
from src.lava.training.tensorflow_lifecycle import batch_normalization_status


class TrainingPolicyResolverTest(unittest.TestCase):
    def test_registry_policy_matches_initialization(self) -> None:
        expected = {
            "mobilenetv3_lstm": TrainingPolicy.PRETRAINED_TRANSFER,
            "efficientnet_b0_lstm": TrainingPolicy.PRETRAINED_TRANSFER,
            "shufflenetv2_lstm": TrainingPolicy.SCRATCH_END_TO_END,
            "mnasnet_lstm": TrainingPolicy.SCRATCH_END_TO_END,
            "rawnet2": TrainingPolicy.NATIVE_REFERENCE,
            "aasist": TrainingPolicy.NATIVE_REFERENCE,
        }
        for name, policy in expected.items():
            with self.subTest(name=name):
                self.assertEqual(resolve_training_policy(get_spec(name)), policy)

    def test_random_backbone_cannot_use_transfer_policy(self) -> None:
        invalid = replace(
            MOBILENET_SPEC,
            initialization=Initialization.SCRATCH,
            pretraining_status="PRETRAINING_NOT_VERIFIED",
        )
        with self.assertRaisesRegex(ValueError, "Refusing to freeze"):
            resolve_training_policy(invalid)


class TensorFlowTrainabilityPolicyTest(unittest.TestCase):
    def test_pretrained_warmup_and_finetune_policy(self) -> None:
        builders = {
            "mobilenetv3_lstm": lambda: MobileNetV3LSTMDetector().build(weights=None),
            "efficientnet_b0_lstm": lambda: build_efficientnet(None)[0],
        }
        for name, builder in builders.items():
            with self.subTest(name=name):
                model = builder()
                backbone = next(
                    layer.layer for layer in model.layers
                    if isinstance(layer, tf.keras.layers.TimeDistributed)
                )
                freeze_backbone(backbone)
                self.assertEqual(parameter_status(backbone)["trainable"], 0)
                unfreeze_backbone(backbone, config.FINETUNE_LAYERS)
                status = parameter_status(backbone)
                self.assertGreater(status["trainable"], 0)
                self.assertLess(status["trainable"], status["total"])
                self.assertEqual(batch_normalization_status(backbone)["trainable"], 0)
                tf.keras.backend.clear_session()

    def test_scratch_epoch1_exact_parameter_and_bn_counts(self) -> None:
        cases = {
            "shufflenetv2_lstm": (build_shufflenet, 1_269_784, 1_253_604, 56),
            "mnasnet_lstm": (build_mnasnet, 2_639_526, 2_606_038, 49),
        }
        for name, (builder, total, trainable, batch_norm) in cases.items():
            with self.subTest(name=name):
                _, backbone = builder(None)
                enable_scratch_end_to_end(backbone)
                self.assertEqual(
                    parameter_status(backbone),
                    {"total": total, "trainable": trainable, "frozen": total - trainable},
                )
                self.assertEqual(
                    batch_normalization_status(backbone),
                    {"total": batch_norm, "trainable": batch_norm, "frozen": 0},
                )
                tf.keras.backend.clear_session()


class ScratchGradientFlowTest(unittest.TestCase):
    def _assert_gradient_flow(self, builder, representative_names: tuple[str, ...]) -> None:
        model, backbone = builder(None)
        enable_scratch_end_to_end(backbone)
        # Trainability must be established before compile/optimizer creation.
        model.compile(optimizer=tf.keras.optimizers.Adam(config.SCRATCH_LR), loss="binary_crossentropy")
        features = tf.random.uniform(
            (1, config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS),
            minval=0.0, maxval=255.0, seed=123,
        )
        labels = tf.constant([[1.0]])
        with tf.GradientTape() as tape:
            probability = model(features, training=True)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, probability))
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        by_name = dict(zip((variable.name for variable in variables), gradients))
        for needle in representative_names:
            matches = [(name, gradient) for name, gradient in by_name.items() if needle in name]
            self.assertTrue(matches, f"No representative variable matched {needle}")
            name, gradient = matches[0]
            self.assertIsNotNone(gradient, f"Gradient missing for {name}")
            self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(gradient))), f"Non-finite gradient for {name}")
            self.assertGreater(float(tf.linalg.global_norm([gradient])), 0.0, f"Zero gradient for {name}")
        tf.keras.backend.clear_session()

    def test_shufflenet_gradient_reaches_stem_middle_tail_lstm_and_head(self) -> None:
        self._assert_gradient_flow(
            build_shufflenet,
            ("stem_conv/kernel", "stage3_unit5_b2_pw1_conv/kernel", "head_conv/kernel",
             "temporal_lstm/lstm_cell/kernel", "probability_fake/kernel"),
        )

    def test_mnasnet_gradient_reaches_stem_middle_tail_lstm_and_head(self) -> None:
        self._assert_gradient_flow(
            build_mnasnet,
            ("stem_conv/kernel", "stage4_block1_expand_conv/kernel", "head_conv/kernel",
             "temporal_lstm/lstm_cell/kernel", "probability_fake/kernel"),
        )


class ScratchMetadataTest(unittest.TestCase):
    def test_required_policy_and_epoch1_fields_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory(dir=config.BASE_DIR) as directory:
            root = Path(directory)
            spec = replace(
                SHUFFLENET_SPEC,
                model_artifact=root / "model.keras",
                threshold_artifact=root / "threshold.json",
                metadata_artifact=root / "metadata.json",
            )
            spec.model_artifact.write_bytes(b"verified-model-placeholder")
            inputs = tf.keras.layers.Input((1,), name="metadata_input")
            model = tf.keras.Model(inputs, tf.keras.layers.Dense(1)(inputs))
            detector = SimpleNamespace(
                spec=spec, model=model,
                backbone=SimpleNamespace(name="shufflenetv2_1_0x_backbone"),
            )
            history = root / "history.png"
            history.write_bytes(b"plot")
            state = {
                "global_best_epoch": 7, "global_best_val_loss": 0.42,
            }
            train._save_scratch_metadata(
                detector=detector, threshold=0.51, threshold_f1=0.6,
                history_path=str(history), epochs_run=10, lifecycle_state=state,
                checkpoint_directory=root, backbone_epoch1={
                    "total": 1_269_784, "trainable": 1_253_604, "frozen": 16_180,
                },
                bn_epoch1={"total": 56, "trainable": 56, "frozen": 0},
            )
            payload = json.loads(spec.metadata_artifact.read_text(encoding="utf-8"))
            self.assertEqual(payload["initialization"], "scratch")
            self.assertEqual(payload["training_policy"], "full_end_to_end_from_epoch_1")
            self.assertEqual(payload["backbone_trainable_params_at_epoch1"], 1_253_604)
            self.assertEqual(payload["BN_trainable"], 56)
            self.assertEqual(payload["manifest_hash"], payload["training_manifest_hash"])


if __name__ == "__main__":
    unittest.main()
