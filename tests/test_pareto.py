from __future__ import annotations

import unittest

from benchmark.pareto import pareto_frontier


class ParetoTest(unittest.TestCase):
    def test_non_dominated_set(self) -> None:
        rows = [
            {"model": "a", "eer": 0.10, "rtf": 0.2},
            {"model": "b", "eer": 0.08, "rtf": 0.4},
            {"model": "c", "eer": 0.12, "rtf": 0.5},
        ]
        names = {row["model"] for row in pareto_frontier(rows, {"eer": "min", "rtf": "min"})}
        self.assertEqual(names, {"a", "b"})

    def test_requires_multiple_complete_results(self) -> None:
        with self.assertRaisesRegex(ValueError, "insufficient"):
            pareto_frontier([{"eer": 0.1}], {"eer": "min"})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            pareto_frontier([{"eer": 0.1}, {"eer": "NOT_RUN"}], {"eer": "min"})


if __name__ == "__main__":
    unittest.main()
