import shutil
import unittest

import numpy as np

from src.lava.jaes_v2.controls import (NoisePlan, align_control, codec_delay_probe, estimate_delay,
                                      ffmpeg_identity, pcm16_control, prepare_headroom, resample_roundtrip)
from src.lava.jaes_v2.transforms import Audio


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.audio = Audio(np.random.default_rng(4).normal(0, .04, 22050), 22050)

    def test_fixed_absolute_power_local_snr_and_noise_identity(self):
        plan = NoisePlan.build(self.audio, target_snr_db=10, seed=7)
        attenuated = Audio(self.audio.samples * .5, 22050)
        original, first = plan.insert(self.audio, "fixed_absolute")
        fixed, second = plan.insert(attenuated, "fixed_absolute")
        local, third = plan.insert(attenuated, "local_target_snr")
        np.testing.assert_allclose(original.samples - self.audio.samples, fixed.samples - attenuated.samples, atol=3e-8)
        self.assertEqual(first["absolute_noise_power"], second["absolute_noise_power"])
        self.assertAlmostEqual(third["absolute_noise_power"] * 4, first["absolute_noise_power"], places=12)
        self.assertAlmostEqual(second["measured_snr_db"], 10 - 20 * np.log10(2), places=5)
        self.assertAlmostEqual(third["measured_snr_db"], 10, places=5)

    def test_known_delay_sign_and_alignment_oracle(self):
        original = np.zeros(1000, np.float32)
        original[200:300] = np.random.default_rng(1).normal(size=100)
        delayed = np.pad(original, (37, 0))[:1000]
        measured = estimate_delay(original, delayed, max_lag=80)
        self.assertEqual(measured["lag_samples"], 37)
        np.testing.assert_array_equal(align_control(Audio(delayed, 16000), 37).samples, original)
        self.assertEqual(estimate_delay(delayed, original, max_lag=80)["lag_samples"], -37)
        with self.assertRaises(ValueError):
            estimate_delay(np.zeros(3), np.zeros(3))

    def test_pcm_quantization_grid_and_error_bound(self):
        source = Audio(np.array([-.5, -.25, 0, .25, .5, .1234567], np.float32), 16000)
        changed, evidence = pcm16_control(source)
        np.testing.assert_array_equal(changed.samples[:5], source.samples[:5])
        np.testing.assert_allclose(changed.samples * 32768, np.round(changed.samples * 32768), atol=0)
        self.assertLessEqual(evidence["maximum_abs_error"], 1 / 32768)

    def test_resampling_frequency_and_duration(self):
        time = np.arange(22050) / 22050
        source = Audio(.1 * np.sin(2 * np.pi * 440 * time), 22050)
        changed, _ = resample_roundtrip(source, 24000)
        self.assertEqual(changed.samples.shape, source.samples.shape)
        self.assertGreater(np.corrcoef(source.samples[100:-100], changed.samples[100:-100])[0, 1], .9999)

    def test_headroom_is_explicit_and_preserves_relative_waveform(self):
        source = Audio(np.array([-.5, 0, .25], np.float32), 22050)
        changed, evidence = prepare_headroom(source, .25)
        unchanged, no_gain = prepare_headroom(source, None)
        np.testing.assert_array_equal(changed.samples, source.samples * .5)
        np.testing.assert_array_equal(unchanged.samples, source.samples)
        self.assertEqual(evidence["gain"], .5)
        self.assertEqual(no_gain["gain"], 1)

    @unittest.skipUnless(shutil.which("ffmpeg"), "Actual codec requires FFmpeg")
    def test_actual_identity_and_codec_delay_metadata(self):
        identity, _ = ffmpeg_identity(self.audio, shutil.which("ffmpeg"))
        quantized, _ = pcm16_control(self.audio)
        np.testing.assert_array_equal(identity.samples, quantized.samples)
        report = codec_delay_probe(shutil.which("ffmpeg"))
        self.assertLess(abs(report["net_residual_delay"]["lag_samples"]), 10)
        self.assertIn("NOT_SEPARATELY_IDENTIFIABLE", report["decoder_component_delay"])


if __name__ == "__main__":
    unittest.main()
