from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

import config
from src.lava.models.tensorflow.mnasnet_lstm import build_backbone, build_model


class MnasNetShapesTest(unittest.TestCase):
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
