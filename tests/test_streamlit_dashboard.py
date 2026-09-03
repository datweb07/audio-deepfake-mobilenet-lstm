from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import config
from app import render_full_mel, render_probability, render_segment_profile, render_waveform
from src.preprocessing import segment_audio


class StreamlitDashboardTest(unittest.TestCase):
    def test_dashboard_charts_render_for_normalized_audio(self) -> None:
        timeline = np.arange(config.TOTAL_SAMPLES, dtype=np.float32) / config.SAMPLE_RATE
        audio = (0.1 * np.sin(2 * np.pi * 440 * timeline)).astype(np.float32)
        segments = segment_audio(audio)
        with patch("app.st.pyplot") as pyplot:
            render_probability(0.72, 0.5)
            render_waveform(audio)
            render_full_mel(segments)
            render_segment_profile(segments)
        self.assertEqual(pyplot.call_count, 4)

    def test_mobile_styles_have_breakpoint_without_gradients(self) -> None:
        stylesheet = (
            Path(__file__).resolve().parents[1] / "assets" / "app.css"
        ).read_text(encoding="utf-8")
        self.assertIn("@media (max-width:768px)", stylesheet)
        self.assertIn('data-testid="stSidebar"', stylesheet)
        self.assertNotIn("gradient(", stylesheet.lower())
        self.assertIn("'Barlow'", stylesheet)
        self.assertIn("'IBM Plex Sans'", stylesheet)
        self.assertIn('data-testid="collapsedControl"', stylesheet)


if __name__ == "__main__":
    unittest.main()
