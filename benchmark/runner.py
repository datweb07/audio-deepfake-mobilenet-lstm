"""Unified LAVA benchmark entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from benchmark.aggregate import aggregate
from benchmark.clean import run_clean
from benchmark.efficiency import run_efficiency
from src.lava.artifacts import write_json_atomic
from src.lava.registry import names


def main(models: list[str], suite: str, *, limit: int | None, warmup: int, runs: int) -> dict[str, object]:
    selected = list(names()) if models == ["all"] else models
    report: dict[str, object] = {"suite": suite, "models": {}}
    for model_name in selected:
        try:
            if suite == "clean":
                result = run_clean(model_name, limit=limit)
            elif suite == "efficiency":
                result = run_efficiency(model_name, warmup=warmup, runs=runs)
            else:
                raise ValueError(f"Unsupported suite: {suite}")
            report["models"][model_name] = result
        except Exception as exc:
            report["models"][model_name] = {"status": "BLOCKED", "reason": str(exc)}
    aggregate(selected)
    destination = Path(config.OUTPUTS_DIR) / "benchmark" / f"last_{suite}_run.json"
    write_json_atomic(destination, report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["all"], help="all or one/more registry names")
    parser.add_argument("--suite", choices=("clean", "efficiency"), default="clean")
    parser.add_argument("--limit", type=int, help="Diagnostic clean subset; never marked BENCHMARKED")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    arguments = parser.parse_args()
    invalid = [name for name in arguments.models if name != "all" and name not in names()]
    if invalid:
        parser.error(f"Unknown detectors: {invalid}")
    main(arguments.models, arguments.suite, limit=arguments.limit, warmup=arguments.warmup, runs=arguments.runs)
