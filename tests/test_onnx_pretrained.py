from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.lava.models.onnx_pretrained import OnnxPretrainedDetector
from src.lava.registry import create


class OnnxPretrainedDetectorTest(unittest.TestCase):
    def test_pretrained_exports_load_and_return_p_fake(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "silence.wav"
            sf.write(audio_path, np.zeros(64_600, dtype=np.float32), 16_000)
            expected_parameters = {
                "rawnet2_pretrained": 17_621_410,
                "aasist_pretrained": 297_866,
            }
            for name, parameter_count in expected_parameters.items():
                with self.subTest(name=name):
                    detector = create(name)
                    self.assertIsInstance(detector, OnnxPretrainedDetector)
                    detector.load()
                    score = float(detector.predict_scores([str(audio_path)])[0])
                    self.assertGreaterEqual(score, 0.0)
                    self.assertLessEqual(score, 1.0)
                    self.assertEqual(detector.parameter_count(), parameter_count)


if __name__ == "__main__":
    unittest.main()
