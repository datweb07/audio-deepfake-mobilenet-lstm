from __future__ import annotations

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


class StreamlitUITest(unittest.TestCase):
    def test_available_models_do_not_show_missing_artifact_panel(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        page = AppTest.from_file(str(app_path)).run(timeout=120)
        self.assertEqual(list(page.exception), [])
        self.assertEqual(list(page.error), [])
        self.assertEqual(list(page.warning), [])
        # A measured, hash-validated benchmark is informational, not a missing-artifact warning.
        expander_labels = [element.label for element in page.expander]
        self.assertTrue(set(expander_labels) <= {"LAVA-5 measured benchmark"})
        self.assertNotIn("Detector diagnostics", expander_labels)
        self.assertNotIn("Unavailable detectors", expander_labels)
        self.assertEqual(len(page.selectbox), 1)
        self.assertEqual(len(page.toggle), 2)
        self.assertEqual(len(page.radio), 1)
        self.assertEqual(page.radio[0].value, "Upload file")
        self.assertEqual(page.selectbox[0].value, "MobileNetV3Small-LSTM")
        self.assertIn("RawNet2 (DF-Pretrained, 2021)", page.selectbox[0].options)
        self.assertIn("AASIST (Official Pretrained, NAVER)", page.selectbox[0].options)
        rendered_text = " ".join(element.value for element in page.markdown)
        self.assertIn("Phan Khắc Anh Tuấn", rendered_text)
        self.assertIn("Trương Thành Đạt", rendered_text)


if __name__ == "__main__":
    unittest.main()
