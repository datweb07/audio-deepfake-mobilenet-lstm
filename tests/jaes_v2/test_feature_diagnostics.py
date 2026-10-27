import unittest
import numpy as np

from scripts.jaes_v2.phase4_feature_diagnostics import diagnostic_features
from src.preprocessing import process_audio_data


class FeatureDiagnosticTests(unittest.TestCase):
    def test_reference_parity_and_scaled_floor_gain_invariance(self):
        rng=np.random.default_rng(4)
        samples=rng.normal(0,.02,66150).astype('float32')
        samples[:12000]=0
        original=process_audio_data(samples)
        np.testing.assert_array_equal(diagnostic_features(samples,mel_power_floor=1e-10),original)
        half=process_audio_data(samples*.5)
        self.assertGreater(float(np.max(np.abs(half-original))),1)
        corrected=diagnostic_features(samples*.5,mel_power_floor=2.5e-11)
        self.assertLess(float(np.max(np.abs(corrected-original))),.001)

    def test_invalid_diagnostic_floor_rejected(self):
        with self.assertRaises(ValueError):
            diagnostic_features(np.ones(66150),mel_power_floor=0)


if __name__=='__main__':
    unittest.main()
