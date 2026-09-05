from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "papers"


class PublicationArtifactsTest(unittest.TestCase):
    def test_three_versions_share_six_detector_facts(self) -> None:
        paths = [
            PAPERS / "LAVA_FULL_PAPER_EN.md",
            PAPERS / "LAVA_FULL_PAPER_VI.md",
            PAPERS / "LAVA_FULL_PAPER.tex",
        ]
        detectors = ["MobileNetV3", "ShuffleNetV2", "MnasNet", "EfficientNet", "RawNet2", "AASIST"]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for detector in detectors:
                self.assertIn(detector, text, f"{detector} absent from {path.name}")
            for value in ("0.9824", "0.9929", "0.0146", "0.0208"):
                self.assertTrue(value in text or value.replace(".", ",") in text)

    def test_markdown_figure_table_counts_and_paths(self) -> None:
        for name in ("LAVA_FULL_PAPER_EN.md", "LAVA_FULL_PAPER_VI.md"):
            text = (PAPERS / name).read_text(encoding="utf-8")
            figures = re.findall(r"!\[.*?\]\((.*?)\)", text)
            captions = re.findall(r"\*\*(?:Table|Bảng)\s+\d+\.", text)
            self.assertEqual(len(figures), 18)
            self.assertEqual(len(captions), 12)
            for relative in figures:
                self.assertTrue((PAPERS / relative).is_file(), relative)

    def test_latex_structure_and_citations(self) -> None:
        tex = (PAPERS / "LAVA_FULL_PAPER.tex").read_text(encoding="utf-8")
        self.assertEqual(tex.count("\\begin{figure"), tex.count("\\end{figure"))
        self.assertEqual(tex.count("\\begin{table"), tex.count("\\end{table"))
        self.assertEqual(tex.count("\\begin{equation}"), tex.count("\\end{equation}"))
        self.assertEqual(tex.count("\\begin{figure"), 18)
        self.assertEqual(tex.count("\\begin{table"), 12)
        keys = set(re.findall(r"@\w+\{([^,]+),", (PAPERS / "references.bib").read_text(encoding="utf-8")))
        cited = set()
        for group in re.findall(r"\\cite\{([^}]+)\}", tex):
            cited.update(group.split(","))
        self.assertEqual(cited, keys)

    def test_publication_validation_gate(self) -> None:
        result = json.loads((PAPERS / "PAPER_VALIDATION.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["detectors"], 6)
        self.assertEqual(result["robustness_scope"], "DIAGNOSTIC_SUBSET_100")
        self.assertFalse(result["training_run"])
        self.assertFalse(result["inference_run"])


if __name__ == "__main__":
    unittest.main()
