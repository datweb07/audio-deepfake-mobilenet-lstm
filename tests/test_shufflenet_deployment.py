"""Deployment regression checks scoped only to ShuffleNetV2-LSTM."""
from __future__ import annotations

import json
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts.shufflenet_deployment import ROOT, WORK, sha


class ShuffleNetDeploymentTest(unittest.TestCase):
    def test_source_bundle_provenance(self):
        metadata = json.loads((ROOT / "ok/metadata.json").read_text(encoding="utf-8"))
        threshold = json.loads((ROOT / "ok/threshold.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["detector_name"], "shufflenetv2_lstm")
        self.assertEqual(metadata["selection"]["best_stage"], "scratch")
        self.assertFalse(metadata["selection"]["test_used"])
        self.assertEqual(metadata["best_epoch"], 16)
        self.assertEqual(metadata["epochs_run"], 28)
        self.assertEqual(threshold, {"source": "validation", "threshold": 0.12})
        self.assertEqual(metadata["training_manifest_hash"], "8b55591d58d3658b8cafe0e77b6ebdedbaa67be2e339730a7276fe9b10958df9")

    @unittest.skipUnless(
        (ROOT / "models/shufflenetv2_lstm/model.keras").exists(),
        "ShuffleNet deployment not installed",
    )
    def test_registry_load_threshold_and_parity(self):
        import tensorflow as tf
        self.addCleanup(tf.keras.backend.clear_session)
        from src.lava.artifacts import artifact_diagnostics, load_threshold
        from src.lava.registry import create, get_spec

        spec = get_spec("shufflenetv2_lstm")
        detector = create("shufflenetv2_lstm")
        detector.load()
        metadata = json.loads(spec.metadata_artifact.read_text(encoding="utf-8"))
        self.assertEqual(artifact_diagnostics(spec), [])
        self.assertEqual(detector.parameter_count(), 1_868_441)
        self.assertEqual(detector.parameter_count(), metadata["parameter_count"])
        self.assertEqual(load_threshold(spec), 0.12)
        self.assertEqual(sha(spec.model_artifact), metadata["conversion"]["converted_sha256"])
        inputs = np.load(WORK / "parity_inputs.npy")
        expected = np.load(WORK / "source_predictions.npz")["scores"].ravel()
        actual = detector.predict_feature_batch(inputs)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    @unittest.skipUnless(
        (ROOT / "models/shufflenetv2_lstm/model.keras").exists(),
        "ShuffleNet deployment not installed",
    )
    def test_streamlit_selector_discovers_shufflenet(self):
        from streamlit.testing.v1 import AppTest

        page = AppTest.from_file(str(ROOT / "app.py")).run(timeout=120)
        self.assertEqual(list(page.exception), [])
        self.assertIn("ShuffleNetV2-1.0x-LSTM", page.selectbox[0].options)
        page.selectbox[0].set_value("ShuffleNetV2-1.0x-LSTM").run(timeout=120)
        self.assertEqual(list(page.exception), [])
        audio_path = ROOT / "data/REAL/11241.wav"
        upload = io.BytesIO(audio_path.read_bytes())
        upload.name = audio_path.name
        upload.type = "audio/wav"
        with patch("streamlit.file_uploader", return_value=upload):
            page.run(timeout=120)
        self.assertEqual(list(page.exception), [])
        self.assertEqual(list(page.error), [])
        self.assertTrue(any(metric.label == "Raw FAKE score" for metric in page.metric))


if __name__ == "__main__":
    unittest.main()
