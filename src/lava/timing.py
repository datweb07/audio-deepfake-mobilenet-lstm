"""Common timing statistics independent of model framework."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any


def timed_runs(operation: Callable[[], Any], *, warmup: int, runs: int) -> dict[str, float | int]:
    if warmup < 0 or runs < 1:
        raise ValueError("warmup must be >= 0 and runs must be >= 1")
    for _ in range(warmup):
        operation()
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered) + 0.999999) - 1))
    return {
        "warmup_runs": warmup,
        "timed_runs": runs,
        "mean_seconds": float(statistics.fmean(samples)),
        "median_seconds": float(statistics.median(samples)),
        "std_seconds": float(statistics.pstdev(samples)),
        "p95_seconds": float(ordered[p95_index]),
    }
