"""Regression checks scoped to the imported EfficientNet warm-up only."""
import json
import io
import gc
import zipfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts.efficientnet_warmup import clean_config, WORK, ROOT, sha


class EfficientNetConversionTest(unittest.TestCase):
    def test_inference_does_not_reuse_training_batch_size(self):
        from src.lava.models.tensorflow.efficientnet_b0_lstm import EfficientNetB0LSTMDetector
        with patch.dict("os.environ", {"LAVA_EFFICIENTNET_INFERENCE_BATCH_SIZE": "1"}):
            detector = EfficientNetB0LSTMDetector()
        sizes = []
        def infer(batch):
            sizes.append(len(batch))
            return np.full(len(batch), 0.25)
        with patch.object(detector, "_infer", side_effect=infer):
            output = detector.predict_feature_batch(np.zeros((5, 6, 1, 1, 3), dtype=np.float32))
        self.assertEqual(sizes, [1] * 5)
        np.testing.assert_array_equal(output, np.full(5, 0.25))

    def test_serialization_translation(self):
        value = {"dtype": {"class_name": "DTypePolicy", "config": {"name": "float32"}}, "activation": "silu", "scale": [2.08, 2.11, 2.10]}
        result = clean_config(value)
        self.assertEqual(result["dtype"], "float32")
        self.assertEqual(result["activation"], "swish")
        self.assertEqual(result["scale"], value["scale"])
        self.assertIsInstance(value["dtype"], dict)

    @unittest.skipUnless((ROOT / "models/efficientnet_b0_lstm/model.keras").exists(), "EfficientNet deployment not installed")
    def test_deployment_registry_parity_threshold_and_ui(self):
        import tensorflow as tf
        self.addCleanup(tf.keras.backend.clear_session)
        from src.lava.registry import create, get_spec
        from src.lava.artifacts import artifact_diagnostics, load_threshold
        from streamlit.testing.v1 import AppTest
        detector = create("efficientnet_b0_lstm")
        detector.load()
        spec = get_spec("efficientnet_b0_lstm")
        self.assertEqual(artifact_diagnostics(spec), [])
        metadata = json.loads(spec.metadata_artifact.read_text(encoding="utf-8"))
        with zipfile.ZipFile(spec.model_artifact) as archive:
            self.assertEqual(json.loads(archive.read("metadata.json"))["keras_version"], "2.15.0")
            self.assertNotIn(b"keras.src.models.functional", archive.read("config.json"))
        self.assertEqual(metadata["detector_name"], "efficientnet_b0_lstm")
        self.assertEqual(metadata["selection"]["stage"], "warmup")
        self.assertFalse(metadata["selection"]["test_used"])
        self.assertEqual(load_threshold(spec), metadata["final_threshold"])
        self.assertEqual(json.loads(spec.threshold_artifact.read_text())["source"], "validation")
        self.assertEqual(sha(spec.model_artifact), metadata["conversion"]["converted_sha256"])
        self.assertEqual(detector.parameter_count(), metadata["parameter_count"])
        if (WORK / "source_predictions.npz").exists():
            inputs = np.load(WORK / "parity_inputs.npy")
            expected = np.load(WORK / "source_predictions.npz")["scores"].ravel()
            actual = detector.predict_feature_batch(inputs)
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        del detector
        tf.keras.backend.clear_session()
        gc.collect()
        page = AppTest.from_file(str(ROOT / "app.py")).run(timeout=120)
        self.assertEqual(list(page.exception), [])
        self.assertIn("EfficientNet-B0-LSTM", page.selectbox[0].options)
        page.selectbox[0].set_value("EfficientNet-B0-LSTM").run(timeout=120)
        self.assertEqual(list(page.exception), [])
        input_report = WORK / "parity_inputs.json"
        if input_report.exists():
            audio_path = Path(json.loads(input_report.read_text())["paths"][0])
            if audio_path.exists():
                upload = io.BytesIO(audio_path.read_bytes())
                upload.name = audio_path.name
                upload.type = "audio/wav"
                with patch("streamlit.file_uploader", return_value=upload):
                    page.run(timeout=120)
                self.assertEqual(list(page.exception), [])
                self.assertEqual(list(page.error), [])
                self.assertTrue(any(m.label == "Raw FAKE score" for m in page.metric))


if __name__ == "__main__":
    unittest.main()
