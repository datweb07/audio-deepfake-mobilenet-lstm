from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import soundfile as sf

from src.lava.jaes_v2.transforms import Audio, AudioTransformChain, Transform, measured_snr, write_audio


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.rate = 16000
        time = np.arange(self.rate, dtype=np.float64) / self.rate
        self.audio = Audio(0.12 * np.sin(2 * np.pi * 440 * time), self.rate)
        self.noise = Transform("noise", "awgn", {"snr_db": 10.0})
        self.band = Transform("band", "bandlimit", {"low_hz": 100, "high_hz": 3400, "order": 4})

    def test_noise_is_deterministic_measured_and_input_unmodified(self):
        original = self.audio.samples.copy()
        chain = AudioTransformChain((self.noise,))
        first, evidence = chain.apply(self.audio, sample_id="s1", seed=17)
        second, _ = chain.apply(self.audio, sample_id="s1", seed=17)
        other, _ = chain.apply(self.audio, sample_id="s2", seed=17)
        np.testing.assert_array_equal(first.samples, second.samples)
        np.testing.assert_array_equal(self.audio.samples, original)
        self.assertFalse(np.array_equal(first.samples, other.samples))
        self.assertAlmostEqual(measured_snr(original, first.samples), 10.0, places=5)
        self.assertAlmostEqual(evidence[0]["measured_snr_db"], 10.0, places=5)

    def test_order_changes_signal_and_chain_identity_but_keeps_noise_seed(self):
        forward = AudioTransformChain((self.noise, self.band))
        reverse = AudioTransformChain((self.band, self.noise))
        a, ea = forward.apply(self.audio, sample_id="same", seed=123)
        b, eb = reverse.apply(self.audio, sample_id="same", seed=123)
        self.assertNotEqual(forward.fingerprint, reverse.fingerprint)
        self.assertEqual(ea[0]["seed"], eb[1]["seed"])
        self.assertGreater(float(np.mean((a.samples - b.samples) ** 2)), 1e-6)

    def test_silence_nonfinite_bad_schema_and_nyquist_are_rejected(self):
        with self.assertRaises(ValueError):
            self.noise.apply(Audio(np.zeros(100), 16000), seed=1)
        with self.assertRaises(ValueError):
            Audio(np.array([np.nan]), 16000)
        with self.assertRaises(ValueError):
            Transform("x", "awgn", {"snr_db": 10, "typo": 1})
        invalid = Transform("x", "bandlimit", {"low_hz": 100, "high_hz": 9000, "order": 4})
        with self.assertRaises(ValueError):
            invalid.apply(self.audio, seed=1)

    def test_resample_changes_duration_by_at_most_one_output_sample(self):
        changed, _ = Transform("r", "resample", {"target_rate": 22050}).apply(self.audio, seed=0)
        self.assertEqual(changed.sample_rate, 22050)
        self.assertAlmostEqual(len(changed.samples) / 22050, 1.0, places=5)

    def test_float_wave_serialization_has_no_timestamped_peak_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a.wav", Path(directory) / "b.wav"
            write_audio(first, self.audio)
            write_audio(second, self.audio)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertNotIn(b"PEAK", first.read_bytes())
            decoded, rate = sf.read(first, dtype="float32")
            np.testing.assert_array_equal(decoded, self.audio.samples)
            self.assertEqual(rate, self.rate)

    def test_measured_rir_has_no_hidden_peak_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rir.wav"
            sf.write(path, np.array([1, 0, .5], np.float32), self.rate, subtype="FLOAT")
            changed, evidence = Transform("rir", "rir", {"path": str(path)}).apply(self.audio, seed=0)
            expected = np.convolve(self.audio.samples, [1, 0, .5])[:len(self.audio.samples)]
            np.testing.assert_allclose(changed.samples, expected, atol=1e-7)
            self.assertIn("asset_sha256", evidence)

    def test_environmental_noise_snr_and_asset_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "noise.wav"
            sf.write(path, np.random.default_rng(2).normal(0, .1, 2000), self.rate, subtype="FLOAT")
            transformed, evidence = Transform("env", "environmental_noise", {"path": str(path), "snr_db": 7}).apply(self.audio, seed=3)
            self.assertAlmostEqual(measured_snr(self.audio.samples, transformed.samples), 7, places=5)
            self.assertEqual(len(evidence["asset_sha256"]), 64)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg unavailable; actual codec test cannot run")
    def test_actual_codec_repeatability_duration_and_clipping_gate(self):
        codec = Transform("opus", "codec", {"encoder": "libopus", "bitrate": "24k", "extension": ".ogg"})
        first, evidence = codec.apply(self.audio, seed=1)
        second, _ = codec.apply(self.audio, seed=1)
        np.testing.assert_array_equal(first.samples, second.samples)
        self.assertEqual(first.sample_rate, self.rate)
        self.assertEqual(first.samples.shape, self.audio.samples.shape)
        self.assertIn("sample_delta", evidence)
        self.assertGreater(float(np.mean((first.samples - self.audio.samples) ** 2)), 0)
        with self.assertRaises(ValueError):
            codec.apply(Audio(self.audio.samples * 100, self.rate), seed=1)


if __name__ == "__main__":
    unittest.main()
