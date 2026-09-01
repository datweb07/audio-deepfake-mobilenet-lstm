from __future__ import annotations

import unittest

import numpy as np
import tensorflow as tf

import config
from src.lava.registry import create
from src.model import build_hybrid_model


class MobileNetRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tf.keras.utils.set_random_seed(123)

    def test_registry_build_matches_original_builder(self) -> None:
        original, _ = build_hybrid_model(weights=None)
        detector = create("mobilenetv3_lstm")
        wrapped = detector.build(weights=None)
        self.assertEqual(original.input_shape, wrapped.input_shape)
        self.assertEqual(original.output_shape, wrapped.output_shape)
        self.assertEqual(original.count_params(), wrapped.count_params())
        self.assertEqual([layer.name for layer in original.layers], [layer.name for layer in wrapped.layers])
        wrapped.set_weights(original.get_weights())
        sample = np.zeros((1, config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS), dtype=np.float32)
        np.testing.assert_allclose(original(sample, training=False), wrapped(sample, training=False), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
