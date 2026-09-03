from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.lava.artifacts import (
    artifact_diagnostics,
    artifact_readiness,
    load_threshold,
    mobilenet_artifacts,
    save_threshold,
    write_json_atomic,
)
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
            self.assertEqual(artifact_diagnostics(spec), [])

    def test_mobilenet_resolver_keeps_complete_legacy_bundle(self) -> None:
        model, threshold, metadata = mobilenet_artifacts()
        self.assertEqual(model.name, "lava_mobilenetv3_lstm.keras")
        self.assertEqual(threshold.name, "best_threshold.txt")
        self.assertEqual(metadata.name, "model_metadata.json")

    def test_git_lfs_pointer_is_rejected_with_actionable_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = DetectorSpec(
                name="test_detector", display_name="Test", group="lightweight",
                framework="tensorflow", input_type="mel_sequence", sample_rate=22050,
                audio_duration=3.0, num_segments=6, model_artifact=root / "model.keras",
                threshold_artifact=root / "threshold.json",
                metadata_artifact=root / "metadata.json", pretraining_status="TEST",
                initialization=Initialization.SCRATCH,
                training_policy=TrainingPolicy.SCRATCH_END_TO_END,
            )
            spec.model_artifact.write_text(
                "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 999\n",
                encoding="utf-8",
            )
            save_threshold(spec, 0.5)
            write_json_atomic(spec.metadata_artifact, {"load_smoke_test": "PASS"})
            self.assertIn("Git LFS pointer", " ".join(artifact_diagnostics(spec)))

    def test_mobilenet_explicit_canonical_bundle_uses_json_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.keras"
            threshold = root / "threshold.json"
            metadata = root / "metadata.json"
            model.write_bytes(b"model")
            write_json_atomic(threshold, {"threshold": 0.73})
            write_json_atomic(metadata, {"load_smoke_test": "PASS"})
            environment = {
                "LAVA_MOBILENET_MODEL_PATH": str(model),
                "LAVA_MOBILENET_THRESHOLD_PATH": str(threshold),
                "LAVA_MOBILENET_METADATA_PATH": str(metadata),
            }
            with patch.dict("os.environ", environment):
                self.assertEqual(mobilenet_artifacts(), (model, threshold, metadata))
            spec = DetectorSpec(
                name="mobilenetv3_lstm", display_name="Mobile", group="lightweight",
                framework="tensorflow", input_type="mel_sequence", sample_rate=22050,
                audio_duration=3.0, num_segments=6, model_artifact=model,
                threshold_artifact=threshold, metadata_artifact=metadata,
                pretraining_status="TEST", initialization=Initialization.IMAGENET_PRETRAINED,
                training_policy=TrainingPolicy.PRETRAINED_TRANSFER,
            )
            self.assertAlmostEqual(load_threshold(spec), 0.73)


if __name__ == "__main__":
    unittest.main()
