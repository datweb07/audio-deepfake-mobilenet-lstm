import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.lava.models.tensorflow.shufflenetv2_lstm import build_model
from src.lava.models.tensorflow.temporal_classifier import enable_scratch_end_to_end, parameter_status


class ShuffleTrainingStepTest(unittest.TestCase):
    def test_update_gradient_bn_and_roundtrip(self):
        self.addCleanup(tf.keras.backend.clear_session)
        tf.keras.utils.set_random_seed(42)
        model, backbone = build_model(None)
        enable_scratch_end_to_end(backbone)
        self.assertEqual(parameter_status(backbone)["trainable"], 1253604)
        self.assertEqual(parameter_status(backbone)["total"], 1269784)
        bns = [l for l in backbone.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
        self.assertEqual(len(bns), 56)
        self.assertTrue(all(l.trainable for l in bns))
        optimizer = tf.keras.optimizers.Adam(3e-4)
        x = tf.random.uniform((1, 6, 224, 224, 3), 0, 255)
        stem = backbone.get_layer("stem_conv").kernel
        before = stem.numpy().copy()
        for label in (0., 1.):
            with tf.GradientTape() as tape:
                score = model(x, training=True)
                loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy([[label]], score))
            gradients = tape.gradient(loss, model.trainable_variables)
            named = dict(zip([w.name for w in model.trainable_variables], gradients))
            for part in ("stem_conv/kernel", "stage3_unit5_b2_pw1_conv/kernel", "head_conv/kernel",
                         "temporal_lstm/lstm_cell/kernel", "probability_fake/kernel"):
                gradient = next(g for name, g in named.items() if part in name)
                self.assertIsNotNone(gradient)
                self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(gradient))))
                self.assertGreater(float(tf.linalg.global_norm([gradient])), 0)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        self.assertFalse(np.array_equal(before, stem.numpy()))
        expected = model(x, training=False).numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.keras"
            model.save(path)
            restored = tf.keras.models.load_model(path, compile=False)
            np.testing.assert_allclose(expected, restored(x, training=False).numpy(), atol=1e-6, rtol=1e-5)

    def test_saturated_wrong_sigmoid_bce_does_not_imply_dead_gradient(self):
        z = tf.Variable([[20.]])
        with tf.GradientTape() as tape:
            p = tf.keras.activations.sigmoid(z)
            loss = tf.keras.backend.binary_crossentropy(tf.zeros_like(p), p)
        self.assertGreater(float(tape.gradient(loss, z)[0, 0]), .99)


if __name__ == "__main__":
    unittest.main()
