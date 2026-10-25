from __future__ import annotations

from dataclasses import replace
import unittest

from src.lava.jaes_v2.analysis import CONDITIONS, analyze, paired_index, parent_bootstrap
from src.lava.jaes_v2.configuration import C1Config
from src.lava.jaes_v2.provenance import SourceRecord, audit_partitions


def source(**overrides):
    values = dict(sample_id="s1", path="s1.wav", parent_recording_id="parent1", dataset_id="d", dataset_version="v1",
                  split="train", label=0, checksum="1" * 64, sample_rate=16000, duration=1)
    return SourceRecord(**{**values, **overrides})


def scores():
    records = []
    for sample, label in (("s1", 0), ("s2", 1)):
        for condition in CONDITIONS:
            score = (0.9 if label == 0 else 0.1) if condition == "A_then_B" else (0.1 if label == 0 else 0.9)
            records.append(dict(sample_id=sample, parent_recording_id=sample, label=label, threshold=.5,
                                raw_score=score, predicted_label=int(score >= .5), condition_id=condition,
                                model_id="unit-fixture", seed=1, run_id="unit-fixture", status="OK"))
    return records


class IntegrityTests(unittest.TestCase):
    def test_variant_and_byte_leakage_rejected(self):
        first = source()
        for other in (replace(first, sample_id="s2", split="test", checksum="2" * 64),
                      replace(first, sample_id="s2", split="test", parent_recording_id="other")):
            with self.assertRaises(ValueError):
                audit_partitions([first, other])

    def test_unknown_does_not_certify_holdouts(self):
        with self.assertRaises(ValueError):
            audit_partitions([source()], require_speaker_disjoint=True)
        with self.assertRaises(ValueError):
            audit_partitions([source(label=1)], require_generator_disjoint=True)
        self.assertFalse(audit_partitions([source()])["global_parent_lineage_verified"])

    def test_source_target_speaker_overlap_and_generator_overlap(self):
        first = source(label=1, source_speaker_id="speaker1", generator_id="g1")
        second = source(sample_id="s2", checksum="2" * 64, parent_recording_id="p2", split="test", label=1,
                        target_speaker_id="speaker1", generator_id="g1")
        with self.assertRaises(ValueError):
            audit_partitions([first, second], require_speaker_disjoint=True)
        with self.assertRaises(ValueError):
            audit_partitions([first, second], require_generator_disjoint=True)

    def test_cross_label_conflict_is_not_auto_relabelled(self):
        with self.assertRaises(ValueError):
            audit_partitions([source(), source(sample_id="s2", label=1)])


class AnalysisTests(unittest.TestCase):
    def test_hand_computed_interaction_order_and_class_errors(self):
        result = analyze(scores(), replicates=200, seed=7)
        error = [row for row in result["contrasts"] if row["risk"] == "error" and row["class"] == "ALL"]
        self.assertEqual([row["estimate"] for row in error], [1, 0, 1])
        self.assertEqual(result["conditions"]["A_then_B"]["false_alarm_rate"], 1)
        self.assertEqual(result["conditions"]["A_then_B"]["miss_rate"], 1)
        self.assertEqual(result["prediction_flips"]["A_then_B"]["clean_correct_to_error"], 2)
        squared = next(row for row in result["contrasts"] if row["risk"] == "squared_score_error" and row["class"] == "ALL")
        self.assertAlmostEqual(squared["estimate"], .8)

    def test_parent_estimand_not_variant_weighted(self):
        result = parent_bootstrap({"p1": [1] * 50, "p2": [0]}, replicates=200, seed=1)
        self.assertEqual(result["estimate"], .5)
        self.assertEqual(result["parents"], 2)
        self.assertEqual(result, parent_bootstrap({"p1": [1] * 50, "p2": [0]}, replicates=200, seed=1))
        self.assertIsNone(parent_bootstrap({"p1": [1, 0]}, replicates=200, seed=1)["ci95"])

    def test_missing_duplicate_failed_and_identity_mismatch_rejected(self):
        variants = [scores()[:-1], scores() + [scores()[0]]]
        bad = scores()
        bad[0]["status"] = "FAILED"
        variants.append(bad)
        bad = scores()
        bad[0]["parent_recording_id"] = "wrong"
        variants.append(bad)
        bad = scores()
        bad[0]["raw_score"] = float("nan")
        variants.append(bad)
        for rows in variants:
            with self.assertRaises(ValueError):
                paired_index(rows)


class ConfigurationTests(unittest.TestCase):
    def test_confirmatory_path_traversal_and_unknown_fields_rejected(self):
        values = dict(experiment_id="valid", model_id="mobilenetv3_lstm", per_class=1, seed=1,
                      bootstrap_replicates=200, sample_rate=22050, max_seconds=3, common_peak=.25,
                      transform_a={"key": "a", "kind": "awgn", "parameters": {"snr_db": 10}},
                      transform_b={"key": "b", "kind": "resample", "parameters": {"target_rate": 16000}})
        C1Config(**values)
        for override in ({"scope": "CONFIRMATORY"}, {"experiment_id": "../escape"}, {"common_peak": 1}):
            with self.assertRaises(ValueError):
                C1Config(**{**values, **override})
        with self.assertRaises(TypeError):
            C1Config(**values, typo=True)


if __name__ == "__main__":
    unittest.main()
