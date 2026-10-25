from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from src.lava.jaes_v2.configuration import C1Config
from src.lava.jaes_v2.evidence import file_hash, read_csv, write_csv
from src.lava.jaes_v2.runner import aggregate, run


class FixturePredictor:
    """Unit fixture only; the published local pilot uses an actual legacy artifact."""
    threshold = .5
    artifact_hashes = {"fixture": "not-a-model"}
    load_seconds = 0
    def __init__(self, model_id):
        self.detector = self
    def predict(self, path):
        return .7, {"preprocessing_latency_ms": 0, "model_latency_ms": 0, "end_to_end_latency_ms": 0}
    def predict_scores(self, paths):
        return np.full(len(paths), .7)


class RunnerTests(unittest.TestCase):
    def test_tiny_pipeline_rebuild_sealing_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "data/manifests/split_manifest.csv"
            manifest.parent.mkdir(parents=True)
            rows = []
            for label in (0, 1):
                path = root / f"audio{label}.wav"
                sf.write(path, .1 * np.sin(np.arange(22050) * .13), 22050, subtype="FLOAT")
                rows.append(dict(sample_id=f"s{label}", path=path.name, sha256=file_hash(path),
                                 label=label, split="validation", included="True"))
            # Distinct source checksum, not the same waveform with conflicting labels.
            path = root / "audio1.wav"
            sf.write(path, .1 * np.cos(np.arange(22050) * .17), 22050, subtype="FLOAT")
            rows[1]["sha256"] = file_hash(path)
            write_csv(manifest, rows)
            configuration = C1Config(experiment_id="integration", model_id="mobilenetv3_lstm", per_class=1,
                                     seed=2, bootstrap_replicates=100, sample_rate=22050, max_seconds=1,
                                     common_peak=.25, transform_a={"key": "n", "kind": "awgn", "parameters": {"snr_db": 10}},
                                     transform_b={"key": "b", "kind": "bandlimit", "parameters": {"low_hz": 100, "high_hz": 3400, "order": 4}})
            with patch("src.lava.jaes_v2.runner.preflight", return_value={"ffmpeg": None}), patch("src.lava.jaes_v2.runner.LegacyMelPredictor", FixturePredictor):
                output = run(root, configuration)
                self.assertEqual(len(read_csv(output / "scores.csv")), 10)
                first = (output / "statistics.json").read_bytes()
                original_svg = (output / "figures/paired_diagnostic.svg").read_bytes()
                aggregate(output)
                self.assertEqual(first, (output / "statistics.json").read_bytes())
                self.assertEqual(original_svg, (output / "figures/paired_diagnostic.svg").read_bytes())
                self.assertTrue((output / "figures/paired_diagnostic.png").is_file())
                with self.assertRaises(FileExistsError):
                    run(root, configuration)
                with (output / "scores.csv").open("a") as stream:
                    stream.write("tampered")
                with self.assertRaises(ValueError):
                    aggregate(output)


if __name__ == "__main__":
    unittest.main()
