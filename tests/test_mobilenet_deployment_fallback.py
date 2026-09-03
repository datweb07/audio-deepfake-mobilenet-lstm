from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import config
from src.artifacts import load_production_model, validate_model_contract


class MobileNetDeploymentFallbackTest(unittest.TestCase):
    def test_weights_fallback_survives_full_model_deserialization_failure(self) -> None:
        expected = load_production_model(compile=False)
        sample = np.zeros(
            (1, config.NUM_SEGMENTS, *config.IMAGE_SIZE, config.CHANNELS),
            dtype=np.float32,
        )
        expected_score = expected(sample, training=False).numpy()

        cloud_error = ValueError(
            "Layer 'Conv' expected 1 variables, but received 0 variables during loading. "
            "Expected: ['Conv/kernel:0']"
        )
        with patch("src.artifacts.tf.keras.models.load_model", side_effect=cloud_error):
            restored = load_production_model(compile=False)

        validate_model_contract(restored)
        actual_score = restored(sample, training=False).numpy()
        np.testing.assert_allclose(actual_score, expected_score, rtol=1e-6, atol=1e-7)

    def test_weights_fallback_can_load_without_full_model_archive(self) -> None:
        restored = load_production_model(
            compile=False,
            model_path="does-not-exist.keras",
            weights_path=config.MODEL_WEIGHTS_PATH,
        )
        validate_model_contract(restored)
        self.assertEqual(restored.count_params(), 1_308_401)


if __name__ == "__main__":
    unittest.main()
