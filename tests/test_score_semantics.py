from __future__ import annotations

import unittest

import numpy as np

from src.lava.score_semantics import binary_logits_to_p_fake, decisions_from_p_fake, validate_p_fake


class ScoreSemanticsTest(unittest.TestCase):
    def test_native_spoof_zero_maps_to_fake_probability(self) -> None:
        logits = np.asarray([[5.0, -5.0], [-5.0, 5.0]], dtype=np.float32)
        probabilities = binary_logits_to_p_fake(logits, fake_index=0)
        self.assertGreater(probabilities[0], 0.99)
        self.assertLess(probabilities[1], 0.01)
        np.testing.assert_array_equal(decisions_from_p_fake(probabilities, 0.5), [1, 0])

    def test_invalid_probability_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_p_fake([0.5, 1.01])


if __name__ == "__main__":
    unittest.main()

