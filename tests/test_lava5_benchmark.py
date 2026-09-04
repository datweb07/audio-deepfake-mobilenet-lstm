import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf

from benchmark.lava5 import MODELS, metrics, sha256
from benchmark.lava5_stress import add_noise, simulated_replay
from src.lava.models.onnx_pretrained import _softmax_fake
from src.lava.score_semantics import decisions_from_p_fake
from benchmark.pareto import pareto_frontier


class Lava5Tests(unittest.TestCase):
    def test_report_markdown_preserves_tables(self):
        from benchmark.lava5_report import markdown_blocks
        self.assertEqual(markdown_blocks(["# Results", "", "| Model |", "|---|", "| measured |", "", "Caveat"]),
                         "# Results\n\n| Model |\n|---|\n| measured |\n\nCaveat\n")

    def test_efficiency_resume_preserves_original_measurement(self):
        import json
        from unittest.mock import patch
        from benchmark.lava5 import worker
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "efficiency/mnasnet_lstm/summary.json"
            path.parent.mkdir(parents=True)
            result = dict(status="BENCHMARKED", device="CPU", batch_size=1, runs=50, warmup=10)
            result.update({k: {"runs_ms": [1.] * 50} for k in ("model_only", "preprocessing", "end_to_end")})
            path.write_text(json.dumps(result))
            original = path.read_bytes()
            with patch("benchmark.lava5.verify_protocol", return_value={"models": [{"model": "mnasnet_lstm"}]}), patch("benchmark.lava5_runtime.Runtime") as runtime:
                worker(root, "mnasnet_lstm", "efficiency")
                runtime.assert_not_called()
            self.assertEqual(path.read_bytes(), original)

    def test_exact_five_and_external_mapping(self):
        self.assertEqual(len(MODELS), 5)
        self.assertNotIn("shufflenetv2_lstm", MODELS)
        self.assertEqual(MODELS["rawnet2"], "rawnet2_pretrained")
        self.assertEqual(MODELS["aasist"], "aasist_pretrained")

    def test_threshold_not_argmax_and_auc_raw(self):
        result = metrics([0, 1], [.5952, .95], .9)
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["roc_auc"], 1)
        self.assertEqual(result["eer"], 0)
        np.testing.assert_array_equal(decisions_from_p_fake([.5952, .9, .90001], .9), [0, 1, 1])

    def test_noise_snr_and_determinism(self):
        x = np.random.default_rng(3).normal(0, .1, 64000).astype(np.float32)
        for snr in (20, 10, 5, 0):
            y = add_noise(x, snr, 42)
            np.testing.assert_array_equal(y, add_noise(x, snr, 42))
            measured = 10 * np.log10(np.mean(x ** 2) / np.mean((y - x) ** 2))
            self.assertAlmostEqual(float(measured), snr, places=4)
        with self.assertRaises(ValueError):
            add_noise(np.zeros(100), 10, 42)

    def test_float_cache_lossless(self):
        x = np.random.default_rng(5).normal(0, 1, 1600).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signal.wav"
            sf.write(path, x, 16000, subtype="FLOAT")
            restored, _ = sf.read(path, dtype="float32")
            np.testing.assert_array_equal(x, restored)
            self.assertEqual(sha256(path), sha256(path))

    def test_replay_deterministic_finite_shape(self):
        x = np.random.default_rng(1).normal(0, .1, 64600).astype(np.float32)
        y = simulated_replay(x, 16000)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(np.isfinite(y).all())
        np.testing.assert_array_equal(y, simulated_replay(x, 16000))

    def test_onnx_native_score_order(self):
        scores = _softmax_fake(np.array([[9., -9.], [-9., 9.]], dtype=np.float32))
        np.testing.assert_array_equal(decisions_from_p_fake(scores, .5), [1, 0])

    def test_pareto_rejects_missing_or_nan(self):
        for value in [None, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                pareto_frontier([{"eer": .1}, {"eer": value}], {"eer": "min"})

    def test_pairwise_sample_misalignment_rejected(self):
        import json
        from benchmark.lava5 import write_csv
        from benchmark.lava5_report import load_result
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rows = [dict(sample_id="b", true_label=0, p_fake=.1, threshold=.5, predicted_label=0, correct=1),
                    dict(sample_id="a", true_label=1, p_fake=.9, threshold=.5, predicted_label=1, correct=1)]
            write_csv(path / "scores.csv", rows)
            summary = dict(status="BENCHMARKED", samples=2, threshold=.5, scores_sha256=sha256(path / "scores.csv"), **metrics([0, 1], [.1, .9], .5))
            (path / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                load_result(path, [{"sample_id": "a", "label": 0}, {"sample_id": "b", "label": 1}])

    def test_display_missing_results_is_optional(self):
        from src.lava.benchmark_display import benchmark_card
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(benchmark_card(None, directory))

    def test_display_hides_stale_artifact(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        from benchmark.lava5 import write_csv
        from src.lava.benchmark_display import benchmark_card
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol").mkdir()
            artifact = root / "model.bin"
            artifact.write_bytes(b"test weights")
            protocol = dict(models=[dict(registry_name="aasist_pretrained", model="aasist", artifact_hashes={"model.bin": sha256(artifact)}, checkpoint_origin="external_reference")], test_samples=2)
            (root / "protocol/protocol.json").write_text(json.dumps(protocol))
            write_csv(root / "lava_5_results.csv", [dict(Model="aasist", CleanF1=.8, AUC=.9, EER=.1)])
            with patch("src.lava.benchmark_display.config.BASE_DIR", directory):
                spec = SimpleNamespace(name="aasist_pretrained")
                self.assertIsNotNone(benchmark_card(spec, root))
                artifact.write_bytes(b"changed weights")
                self.assertIsNone(benchmark_card(spec, root))

    def test_available_codec_roundtrips(self):
        import shutil
        from benchmark.lava5_stress import CODECS, codec_roundtrip
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            self.skipTest("ffmpeg unavailable")
        with tempfile.TemporaryDirectory() as directory:
            for rate in (16000, 44100):
                x = (.1 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)).astype(np.float32)
                for name, setting in CODECS.items():
                    path = Path(directory) / f"{name}_{rate}.wav"
                    codec_roundtrip(x, rate, path, setting, ffmpeg)
                    y, restored_rate = sf.read(path, dtype="float32")
                    self.assertEqual(restored_rate, rate)
                    self.assertTrue(np.isfinite(y).all())
                    self.assertGreater(float(np.std(y)), .01)
                    self.assertLess(abs(len(y) / rate - 1), .15)

    def test_diagnostic_subset_is_fixed_and_proportional(self):
        from unittest.mock import patch
        from benchmark.lava5 import prepare_diagnostic, write_csv, read_csv
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [dict(sample_id=f"sample{i}", label=int(i >= 6)) for i in range(10)]
            write_csv(root / "protocol/test_samples.csv", rows)
            with patch("benchmark.lava5.verify_protocol", return_value={"models": []}):
                prepare_diagnostic(root, 5)
                first = read_csv(root / "diagnostic_5/protocol/test_samples.csv")
                prepare_diagnostic(root, 5)
                second = read_csv(root / "diagnostic_5/protocol/test_samples.csv")
            self.assertEqual(first, second)
            self.assertEqual(len(first), 5)
            self.assertEqual(sum(int(r["label"]) == 0 for r in first), 3)


if __name__ == "__main__":
    unittest.main()
