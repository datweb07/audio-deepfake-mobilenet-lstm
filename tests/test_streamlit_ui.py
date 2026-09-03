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
        self.assertEqual(list(page.expander), [])
        self.assertEqual(len(page.selectbox), 1)
        self.assertEqual(page.selectbox[0].value, "mobilenetv3_lstm")
        self.assertIn("RawNet2 (DF-Pretrained, 2021)", page.selectbox[0].options)
        self.assertIn("AASIST (Official Pretrained, NAVER)", page.selectbox[0].options)


if __name__ == "__main__":
    unittest.main()
