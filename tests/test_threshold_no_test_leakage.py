from __future__ import annotations

import unittest

from src.lava.training.policy import assert_test_isolation, require_validation_source


class ThresholdNoTestLeakageTest(unittest.TestCase):
    def test_validation_is_the_only_calibration_source(self) -> None:
        require_validation_source("validation")
        with self.assertRaises(ValueError):
            require_validation_source("test")

    def test_test_cannot_be_a_training_role(self) -> None:
        assert_test_isolation(("train", "validation"))
        with self.assertRaises(ValueError):
            assert_test_isolation(("train", "validation", "test"))


if __name__ == "__main__":
    unittest.main()
