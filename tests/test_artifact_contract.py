from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.lava.artifacts import artifact_readiness, load_threshold, save_threshold, write_json_atomic
from src.lava.contracts import DetectorSpec, Initialization, TrainingPolicy


class ArtifactContractTest(unittest.TestCase):
    def test_three_artifact_readiness_and_json_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = DetectorSpec(
                name="test_detector", display_name="Test", group="lightweight", framework="tensorflow",
                input_type="mel_sequence", sample_rate=22050, audio_duration=3.0, num_segments=6,
                model_artifact=root / "model.keras", threshold_artifact=root / "threshold.json",
                metadata_artifact=root / "metadata.json", pretraining_status="TEST",
                initialization=Initialization.SCRATCH,
                training_policy=TrainingPolicy.SCRATCH_END_TO_END,
            )
            ready, missing = artifact_readiness(spec)
            self.assertFalse(ready)
            self.assertEqual(len(missing), 3)
            spec.model_artifact.write_bytes(b"model")
            save_threshold(spec, 0.42)
            write_json_atomic(spec.metadata_artifact, {"load_smoke_test": "PASS"})
            self.assertAlmostEqual(load_threshold(spec), 0.42)
            self.assertEqual(artifact_readiness(spec), (True, []))


if __name__ == "__main__":
    unittest.main()
