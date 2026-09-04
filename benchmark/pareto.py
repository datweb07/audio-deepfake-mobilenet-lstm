"""Non-dominated-set analysis with no arbitrary weighted score."""

from __future__ import annotations

from typing import Iterable, Mapping
import math


def pareto_frontier(
    rows: Iterable[Mapping[str, object]], objectives: Mapping[str, str]
) -> list[Mapping[str, object]]:
    candidates = list(rows)
    if len(candidates) < 2:
        raise ValueError("Pareto analysis unavailable: insufficient completed benchmark results.")
    if not objectives or any(direction not in {"min", "max"} for direction in objectives.values()):
        raise ValueError("Objectives must explicitly use min or max directions")
    for row in candidates:
        for key in objectives:
            if key not in row or row[key] in (None, "", "NOT_RUN"):
                raise ValueError("Pareto analysis unavailable: incomplete selected objectives.")
            if not math.isfinite(float(row[key])):
                raise ValueError("Pareto analysis unavailable: non-finite selected objectives.")

    frontier = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            no_worse = True
            strictly_better = False
            for key, direction in objectives.items():
                candidate_value, other_value = float(candidate[key]), float(other[key])
                if direction == "min":
                    no_worse &= other_value <= candidate_value
                    strictly_better |= other_value < candidate_value
                else:
                    no_worse &= other_value >= candidate_value
                    strictly_better |= other_value > candidate_value
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier
