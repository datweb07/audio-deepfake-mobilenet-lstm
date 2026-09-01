from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.aggregate import aggregate


class BenchmarkAggregationTest(unittest.TestCase):
    def test_rejects_stale_clean_and_smoke_efficiency_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            outputs = root / "outputs"
            (data / "manifests").mkdir(parents=True)
            (data / "manifests" / "manifest_metadata.json").write_text(
                json.dumps({"manifest_hash": "current"}), encoding="utf-8"
            )
            for name, manifest_hash in (("stale", "old"), ("valid", "current")):
                clean = outputs / "benchmark" / "clean" / name
                efficiency = outputs / "benchmark" / "efficiency" / name
                clean.mkdir(parents=True)
                efficiency.mkdir(parents=True)
                (clean / "summary.json").write_text(
                    json.dumps({
                        "status": "BENCHMARKED", "manifest_hash": manifest_hash,
                        "eer": 0.1, "f1": 0.9, "macro_f1": 0.9, "roc_auc": 0.95,
                        "pretraining_stratum": "test",
                    }), encoding="utf-8"
                )
                (efficiency / "summary.json").write_text(
                    json.dumps({"status": "SMOKE_TESTED", "parameter_count": 123}), encoding="utf-8"
                )
            with patch("benchmark.aggregate.config.DATA_DIR", str(data)), patch(
                "benchmark.aggregate.config.OUTPUTS_DIR", str(outputs)
            ):
                destination = aggregate(("stale", "valid"))
            text = destination.read_text(encoding="utf-8")
            self.assertNotIn("stale", text)
            self.assertIn("valid", text)
            self.assertIn("NOT_RUN", text)


if __name__ == "__main__":
    unittest.main()
