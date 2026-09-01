from __future__ import annotations

import unittest

import numpy as np

from src.lava.evaluation_metrics import compute_eer


class MetricsTest(unittest.TestCase):
    def test_eer_threshold_is_finite_for_tiny_perfect_case(self) -> None:
        eer, threshold = compute_eer([0, 1], [0.01, 0.99])
        self.assertEqual(eer, 0.0)
        self.assertTrue(np.isfinite(threshold))


if __name__ == "__main__":
    unittest.main()
