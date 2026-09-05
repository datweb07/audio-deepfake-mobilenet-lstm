import csv
import json
from pathlib import Path
import unittest

import config
from benchmark.lava5 import sha256
from src.lava.benchmark_display import benchmark_card
from src.lava.registry import get_spec


ROOT = Path(config.BASE_DIR)
OUT = ROOT / "outputs/lava_6"
OLD = ROOT / "outputs/lava_5"
MODELS = {"mobilenetv3_lstm", "efficientnet_b0_lstm", "mnasnet_lstm", "rawnet2", "aasist", "shufflenetv2_lstm"}


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Lava6IncrementalTest(unittest.TestCase):
    def test_incremental_protocol_preserves_lava5_and_forbids_old_reruns(self):
        protocol = json.loads((OUT / "protocol/protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["existing_models_reexecuted"], [])
        self.assertEqual(protocol["new_model_executed"], "shufflenetv2_lstm")
        self.assertEqual(protocol["source_lava5_protocol_sha256"], sha256(OLD / "protocol/protocol.json"))
        self.assertEqual(set(protocol["model_order"]), MODELS)

    def test_six_model_aggregate_invariants(self):
        master = rows(OUT / "lava_6_results.csv")
        self.assertEqual(len(master), 6)
        self.assertEqual({r["Model"] for r in master}, MODELS)
        self.assertTrue(all(r["Status"] == "FULL_CLEAN_DIAGNOSTIC_ROBUSTNESS_EFFICIENCY" for r in master))
        self.assertEqual(len(rows(OUT / "tables/table_1_detector_specification_6_models.csv")), 6)
        self.assertEqual(len(rows(OUT / "tables/table_2_clean_6_models.csv")), 6)
        self.assertEqual(len(rows(OUT / "tables/table_4_efficiency_6_models.csv")), 6)
        self.assertEqual(len(rows(OUT / "pareto/pareto_results_6_models.csv")), 6)

    def test_shuffle_clean_score_contract_and_count(self):
        score_rows = rows(OUT / "clean/shufflenetv2_lstm/scores.csv")
        summary = json.loads((OUT / "clean/shufflenetv2_lstm/summary.json").read_text())
        self.assertEqual(len(score_rows), 2737)
        self.assertEqual(summary["samples"], 2737)
        self.assertEqual(summary["threshold"], .12)
        self.assertEqual(summary["scores_sha256"], sha256(OUT / "clean/shufflenetv2_lstm/scores.csv"))
        self.assertTrue(all(0 <= float(r["p_fake"]) <= 1 for r in score_rows))
        self.assertTrue(all(int(r["predicted_label"]) == int(float(r["p_fake"]) >= .12) for r in score_rows))

    def test_completed_diagnostic_conditions_have_six_models(self):
        for suite, expected in {"noise": 4, "compression": 4, "replay": 1}.items():
            self.assertFalse(list((OLD / "diagnostic_100/robustness" / suite).glob("*/shufflenetv2_lstm/summary.json")))
            self.assertEqual(len(list((OUT / "diagnostic_100/robustness" / suite).glob("*/shufflenetv2_lstm/summary.json"))), expected)
        self.assertEqual(len(rows(OUT / "tables/table_3_robustness_diagnostic_6_models.csv")), 6)

    def test_agreement_and_acceptance_scope_are_honest(self):
        agreement = rows(OUT / "error_analysis/agreement_matrix_6_models.csv")
        self.assertEqual(len(agreement), 6)
        self.assertTrue(all(set(row) == {"Model", *MODELS} for row in agreement))
        acceptance = json.loads((OUT / "report/acceptance.json").read_text())
        self.assertEqual(acceptance["agreement_shape"], [6, 6])
        self.assertEqual(acceptance["full_test_robustness"], "NOT_RUN")
        self.assertEqual(acceptance["official_pareto"], "NOT_RUN")
        self.assertFalse(acceptance["full_lava_benchmark_complete"])
        self.assertTrue(acceptance["six_detector_current_scope_complete"])

    def test_streamlit_benchmark_card_prefers_six_model_results(self):
        card = benchmark_card(get_spec("shufflenetv2_lstm"))
        self.assertIsNotNone(card)
        self.assertAlmostEqual(float(card["CleanF1"]), 0.9824109824109823)
        self.assertEqual(int(card["TestSamples"]), 2737)

    def test_streamlit_packaged_summary_works_without_ignored_outputs(self):
        card = benchmark_card(get_spec("shufflenetv2_lstm"), ROOT / "does-not-exist")
        self.assertIsNotNone(card)
        self.assertEqual(card["RobustnessScope"], "DIAGNOSTIC_SUBSET_100")
        self.assertAlmostEqual(float(card["EER"]), 0.014629948364888179)
