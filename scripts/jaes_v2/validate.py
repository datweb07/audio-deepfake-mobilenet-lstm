"""Persist research and selected frozen-legacy regression results without changing baseline."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import unittest

from src.lava.jaes_v2.evidence import write_json
from src.lava.jaes_v2.runtime import preflight, source_fingerprint

LEGACY_PATTERNS = ["test_registry.py", "test_score_semantics.py", "test_threshold_no_test_leakage.py",
                   "test_manifest_integrity.py", "test_metrics.py", "test_artifact_contract.py",
                   "test_benchmark_aggregation.py", "test_lava5_benchmark.py", "test_lava6_incremental.py",
                   "test_decision_display.py", "test_streamlit_ui.py"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "code": source_fingerprint(root),
              "baseline": preflight(root), "suites": {}}
    for name, patterns, folder in (("jaes_v2", ["test_*.py"], root / "tests/jaes_v2"),
                                   ("legacy_selected", LEGACY_PATTERNS, root / "tests")):
        with (args.output / (name + ".txt")).open("w", encoding="utf-8") as stream, redirect_stdout(stream), redirect_stderr(stream):
            suite = unittest.TestSuite(unittest.TestLoader().discover(str(folder), pattern=pattern) for pattern in patterns)
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        report["suites"][name] = {"run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                                 "skipped": len(result.skipped), "successful": result.wasSuccessful(),
                                 "failure_details": [{"test": test.id(), "traceback": details} for test, details in result.failures + result.errors]}
    report["baseline_after_tests"] = preflight(root)["baseline_freeze_verification"]
    write_json(args.output / "validation.json", report)
    print(json.dumps({name: {key: value for key, value in suite.items() if key != "failure_details"}
                      for name, suite in report["suites"].items()}, indent=2))
    sys.exit(0 if all(suite["successful"] for suite in report["suites"].values()) else 1)


if __name__ == "__main__":
    main()
