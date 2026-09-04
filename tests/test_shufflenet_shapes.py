from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

import config
from src.lava.models.tensorflow.shufflenetv2_lstm import ChannelSplit, ChannelShuffle, build_backbone, build_model


class ShuffleNetShapesTest(unittest.TestCase):
    def tearDown(self):
        tf.keras.backend.clear_session()

    def test_channel_split_symbolic_and_serialization(self) -> None:
        inputs = tf.keras.Input((2, 2, 116))
        outputs = ChannelSplit(name="split")(inputs)
        model = tf.keras.Model(inputs, outputs)
        value = tf.reshape(tf.range(2 * 2 * 116, dtype=tf.float32), (1, 2, 2, 116))
        expected = tf.split(value, 2, axis=-1)
        for a, b in zip(model(value), expected):
            np.testing.assert_array_equal(a.numpy(), b.numpy())
        self.assertEqual(model.count_params(), 0)
        clone = tf.keras.models.clone_model(model)
        for a, b in zip(clone(value), expected):
            np.testing.assert_array_equal(a.numpy(), b.numpy())
        with self.assertRaises(ValueError):
            ChannelSplit().compute_output_shape((None, 2, 2, 115))

    def test_channel_shuffle_permutation(self) -> None:
        value = tf.reshape(tf.range(4, dtype=tf.float32), (1, 1, 1, 4))
        shuffled = ChannelShuffle(2)(value).numpy().reshape(-1)
        np.testing.assert_array_equal(shuffled, [0, 2, 1, 3])

    def test_stage_shapes(self) -> None:
        backbone = build_backbone()
        probes = tf.keras.Model(
            backbone.input,
            [
                backbone.get_layer("stage2_unit1_shuffle").output,
                backbone.get_layer("stage3_unit1_shuffle").output,
                backbone.get_layer("stage4_unit1_shuffle").output,
                backbone.output,
            ],
        )
        shapes = [tuple(tensor.shape) for tensor in probes(tf.zeros((1, 224, 224, 3)), training=False)]
        self.assertEqual(shapes, [(1, 28, 28, 116), (1, 14, 14, 232), (1, 7, 7, 464), (1, 1024)])

    def test_temporal_forward_and_serialization(self) -> None:
        model, backbone = build_model(None)
        self.assertEqual(backbone.output_shape, (None, 1024))
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
