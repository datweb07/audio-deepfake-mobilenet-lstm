from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from src.lava.jaes_v2.control_analysis import CONTROL_CONDITIONS, summarize
from src.lava.jaes_v2.datasets import ASVspoof2019LA, import_public
from src.lava.jaes_v2.evidence import file_hash, read_csv
from src.lava.jaes_v2.phase2 import ControlConfig, aggregate_controls, make_variants, run_controls


class ScoreFixture:
    threshold = .5
    artifact_hashes = {"fixture": "UNIT_TEST_ONLY"}
    load_seconds = 0
    def __init__(self, model_id):
        pass
    def predict(self, path):
        return .7, {"preprocessing_latency_ms": 0, "model_latency_ms": 0, "end_to_end_latency_ms": 0}


class Phase2Tests(unittest.TestCase):
    def test_config_rejects_confirmation_and_implicit_public_source(self):
        for values in ({"scope": "CONFIRMATORY"}, {"scope": "PUBLIC_DEVELOPMENT"}, {"per_class": 2.5}):
            with self.assertRaises(ValueError):
                ControlConfig(experiment_id="test", **values)

    @unittest.skipUnless(shutil.which("ffmpeg"), "Actual codec integration needs FFmpeg")
    def test_public_fixture_import_to_control_scores_and_model_free_aggregation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lines = []
            for index in range(4):
                identifier = f"LA_D_100000{index}"
                lines.append(f"LA_000{index} {identifier} - {'-' if index < 2 else 'A01'} {'bonafide' if index < 2 else 'spoof'}")
                path = root / "ASVspoof2019_LA_dev/flac" / (identifier + ".flac")
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, np.random.default_rng(index).normal(0, .03, 1600), 16000, subtype="PCM_16")
            protocol = root / "TEST_FIXTURE_PROTOCOL.txt"
            protocol.write_text("\n".join(lines))
            imported = root / "import"
            import_public(ASVspoof2019LA(), dataset_root=root, protocol=protocol, official_split="dev", output=imported)
            configuration = ControlConfig(experiment_id="fixture_only", per_class=2, max_seconds=.1, bootstrap_replicates=100,
                                           scope="PUBLIC_DEVELOPMENT", public_manifest="import/provenance_manifest.csv")
            environment = {"free_disk_bytes": 10 ** 12, "ffmpeg": shutil.which("ffmpeg")}
            with patch("src.lava.jaes_v2.phase2.preflight", return_value=environment), patch("src.lava.jaes_v2.phase2.LegacyMelPredictor", ScoreFixture):
                output = run_controls(root, configuration)
            scores = read_csv(output / "scores.csv")
            self.assertEqual(len(scores), 4 * len(CONTROL_CONDITIONS))
            before = (output / "statistics.json").read_bytes()
            report = aggregate_controls(output)
            self.assertEqual(before, (output / "statistics.json").read_bytes())
            self.assertEqual(report["scope"], "EXPLORATORY_PUBLIC_DEVELOPMENT")
            with self.assertRaises(ValueError):
                summarize(scores[:-1], read_csv(output / "manifest.csv"), replicates=100, seed=1)
            variant = output / read_csv(output / "conditions.csv")[0]["path"]
            variant.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                aggregate_controls(output)


if __name__ == "__main__":
    unittest.main()
