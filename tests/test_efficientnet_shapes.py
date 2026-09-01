from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

import config
from src.lava.models.tensorflow.efficientnet_b0_lstm import build_model


class EfficientNetShapesTest(unittest.TestCase):
    def test_shape_forward_and_serialization(self) -> None:
        model, backbone = build_model(None)
        self.assertEqual(backbone.output_shape, (None, 1280))
        self.assertEqual(model.get_layer("time_distributed_efficientnet_b0").output_shape, (None, 6, 1280))
        sample = np.zeros((1, config.NUM_SEGMENTS, *config.IMAGE_SIZE, 3), dtype=np.float32)
        output = model(sample, training=False).numpy()
        self.assertEqual(output.shape, (1, 1))
        self.assertTrue(0.0 <= output[0, 0] <= 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"
            model.save(path)
            loaded = tf.keras.models.load_model(path, compile=False)
            np.testing.assert_allclose(output, loaded(sample, training=False).numpy(), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
