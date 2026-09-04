"""Decision presentation regressions; thresholds and classifier stay unchanged."""
import io
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from src.lava.decision_display import decision_explanation, threshold_description
from src.lava.score_semantics import classify_probability


class DecisionDisplayTest(unittest.TestCase):
    def test_requested_case_and_boundary(self):
        result = classify_probability(0.5952, 0.9)
        self.assertEqual(result.prediction, "REAL")
        self.assertIn("0.5952 < threshold 0.9000", decision_explanation(result))
        self.assertEqual(classify_probability(0.9, 0.9).prediction, "FAKE")
        self.assertIn(">=", decision_explanation(classify_probability(0.9, 0.9)))

    def test_provenance_is_not_inferred_from_threshold(self):
        spec = SimpleNamespace(threshold_artifact=Path("threshold.json"), metadata_artifact=Path("metadata.json"))
        for source, expected in [("default_no_calibration", "Default"),
                                 ("validation", "Validation-selected"), ("", "unavailable")]:
            with patch("src.lava.decision_display.load_json", return_value={"source": source}):
                self.assertIn(expected, threshold_description(spec))
        spec.threshold_artifact = Path("best_threshold.txt")
        with patch("src.lava.decision_display.load_json", return_value={"threshold_source": "validation FAKE-class F1"}):
            self.assertIn("Validation-selected", threshold_description(spec))

    def test_single_score_chart(self):
        from app import render_probability
        with patch("app.st.pyplot") as pyplot:
            render_probability(0.5952, 0.9)
        axis = pyplot.call_args.args[0].axes[0]
        self.assertEqual(len(axis.collections), 1)
        np.testing.assert_allclose(axis.collections[0].get_offsets(), [[0.5952, 0.5]])
        self.assertEqual(len(axis.get_yticks()), 0)
        self.assertEqual(axis.lines[0].get_xdata()[0], 0.9)

    def test_cli_has_no_confidence_and_retains_decision(self):
        import predict
        spec = SimpleNamespace(display_name="Test", framework="test")
        detector = SimpleNamespace(load=lambda: None, predict_scores=lambda paths: [0.5952])
        output = io.StringIO()
        with patch("predict.os.path.isfile", return_value=True), \
             patch("predict.get_spec", return_value=spec), \
             patch("predict.create", return_value=detector), \
             patch("predict.load_threshold", return_value=0.9), \
             patch("predict.threshold_description", return_value="Validation-selected"), \
             patch("sys.stdout", output):
            predict.main("sample.wav", "mnasnet_lstm")
        self.assertIn("Prediction: REAL", output.getvalue())
        self.assertIn("Raw P(FAKE): 0.5952", output.getvalue())
        self.assertNotIn("Confidence:", output.getvalue())

    def test_ui_requested_case_and_default_threshold(self):
        from streamlit.testing.v1 import AppTest
        from src.lava.registry import get_spec
        import config
        spec = get_spec("mnasnet_lstm")
        detector = SimpleNamespace(spec=spec, load=lambda: None,
                                   predict_scores=lambda paths: np.array([0.5952]))
        upload = io.BytesIO(b"mock audio decoded by patched loader")
        upload.name, upload.type = "test.wav", "audio/wav"
        with patch("src.lava.registry.specs", return_value=(spec,)), \
             patch("src.lava.registry.create", return_value=detector), \
             patch("src.lava.artifacts.artifact_diagnostics", return_value=[]), \
             patch("src.lava.artifacts.load_threshold", return_value=0.9), \
             patch("src.preprocessing.load_audio", return_value=np.zeros(config.TOTAL_SAMPLES)), \
             patch("streamlit.file_uploader", return_value=upload):
            page = AppTest.from_file("app.py").run(timeout=30)
            self.assertEqual(list(page.exception), [])
            self.assertNotIn("Confidence", [m.label for m in page.metric])
            self.assertEqual(next(m.value for m in page.metric if m.label == "Raw FAKE score"), "0.5952")
            self.assertTrue(any("classified as REAL" in c.value for c in page.caption))
            self.assertTrue(any("Signal contract" in m.value for m in page.markdown))
            # Same interface, but default pretrained threshold must not claim validation.
            with patch("src.lava.decision_display.load_json", return_value={"source": "default_no_calibration"}):
                page.run(timeout=30)
            self.assertTrue(any("Default threshold" in c.value for c in page.caption))


if __name__ == "__main__":
    unittest.main()
