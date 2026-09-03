from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

import config
from src.lava.models.tensorflow.mnasnet_lstm import (
    BN_ARGS,
    MNASNET_CLIPNORM,
    MNASNET_LABEL_SMOOTHING,
    MNASNET_LEARNING_RATE,
    MNASNET_WEIGHT_DECAY,
    MnasNetLSTMDetector,
    build_backbone,
    build_model,
)
from src.lava.models.tensorflow.shufflenetv2_lstm import ShuffleNetV2LSTMDetector


class MnasNetShapesTest(unittest.TestCase):
    def test_reference_activation_initialization_and_audio_bn_policy(self) -> None:
        backbone = build_backbone()
        self.assertEqual(BN_ARGS["momentum"], 0.9)
        self.assertIsInstance(backbone.get_layer("stem_relu"), tf.keras.layers.ReLU)
        self.assertIsNone(backbone.get_layer("stem_relu").max_value)
        stem = backbone.get_layer("stem_conv")
        initializer = stem.kernel_initializer.get_config()
        self.assertEqual(initializer["mode"], "fan_out")
        self.assertEqual(initializer["distribution"], "untruncated_normal")
        self.assertAlmostEqual(stem.kernel_regularizer.l2, MNASNET_WEIGHT_DECAY / 2.0)

    def test_mnasnet_only_stability_compile_profile(self) -> None:
        detector = MnasNetLSTMDetector()
        model = detector.build(weights=None)
        profile = detector.compile_for_scratch_training(model)
        self.assertAlmostEqual(float(model.optimizer.learning_rate.numpy()), MNASNET_LEARNING_RATE)
        self.assertAlmostEqual(float(model.optimizer.clipnorm), MNASNET_CLIPNORM)
        self.assertAlmostEqual(
            float(model.loss.get_config()["label_smoothing"]), MNASNET_LABEL_SMOOTHING
        )
        self.assertEqual(profile["weight_decay"], MNASNET_WEIGHT_DECAY)
        self.assertFalse(
            hasattr(ShuffleNetV2LSTMDetector(), "compile_for_scratch_training"),
            "The MnasNet stability recipe must not alter ShuffleNet training",
        )

    def test_stages_se_and_skip(self) -> None:
        backbone = build_backbone()
        self.assertIsNotNone(backbone.get_layer("stage3_block1_se_scale"))
        self.assertIsNotNone(backbone.get_layer("stage5_block2_skip"))
        probes = tf.keras.Model(
            backbone.input,
            [
                backbone.get_layer("stage2_block1_project_bn").output,
                backbone.get_layer("stage3_block1_project_bn").output,
                backbone.get_layer("stage4_block1_project_bn").output,
                backbone.get_layer("stage6_block1_project_bn").output,
                backbone.output,
            ],
        )
        shapes = [tuple(tensor.shape) for tensor in probes(tf.zeros((1, 224, 224, 3)), training=False)]
        self.assertEqual(shapes, [(1, 56, 56, 24), (1, 28, 28, 40), (1, 14, 14, 80), (1, 7, 7, 160), (1, 1280)])

    def test_temporal_forward_and_serialization(self) -> None:
        model, backbone = build_model(None)
        self.assertEqual(backbone.output_shape, (None, 1280))
        sample = np.zeros((1, config.NUM_SEGMENTS, *config.IMAGE_SIZE, 3), dtype=np.float32)
        output = model(sample, training=False).numpy()
        self.assertEqual(output.shape, (1, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"
            model.save(path)
            loaded = tf.keras.models.load_model(path, compile=False)
            np.testing.assert_allclose(output, loaded(sample, training=False).numpy(), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
