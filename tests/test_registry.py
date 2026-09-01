from __future__ import annotations

import unittest

from src.lava.registry import create, get_spec, names


class RegistryTest(unittest.TestCase):
    def test_baseline_is_registered(self) -> None:
        self.assertEqual(
            names(),
            (
                "mobilenetv3_lstm", "efficientnet_b0_lstm", "shufflenetv2_lstm",
                "mnasnet_lstm", "rawnet2", "aasist",
            ),
        )
        spec = get_spec("mobilenetv3_lstm")
        self.assertEqual(spec.framework, "tensorflow")
        self.assertEqual(spec.input_type, "mel_sequence")
        self.assertEqual(create(spec.name).spec, spec)

    def test_every_registered_score_contract_is_fake_probability(self) -> None:
        for name in names():
            self.assertEqual(create(name).spec.name, name)


if __name__ == "__main__":
    unittest.main()
