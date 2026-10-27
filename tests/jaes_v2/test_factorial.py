import unittest

from src.lava.jaes_v2.factorial import generator_channel_interaction


def fixture():
    rows = []
    for factor, cluster, repeats, changed_error in (("G0", "p0", 2, 0), ("G0", "p1", 1, 0),
                                                   ("G1", "p2", 7, 1), ("G1", "p3", 1, 0)):
        for sample in range(repeats):
            for condition in ("clean", "codec"):
                prediction = 0 if condition == "codec" and changed_error else 1
                rows.append(dict(sample_id=f"{cluster}-{sample}", label=1, condition_id=condition, predicted_label=prediction,
                                 raw_score=.9 if prediction else .1, threshold=.5, model_id="fixture", seed=0, status="OK",
                                 speaker_id=cluster, dataset_id="fixture_corpus", attack_id="A01" if factor == "G0" else "A02",
                                 generator_id="UNKNOWN", factor_group=factor))
    return rows


class FactorialTests(unittest.TestCase):
    def analyze(self, rows, **options):
        return generator_channel_interaction(rows, channel="codec", factor_definition="FIXTURE official attack groups, not model familiarity",
                                             class_label=1, replicates=200, **options)

    def test_hand_computed_equal_cluster_contrast_with_unequal_sizes(self):
        result = self.analyze(fixture())
        self.assertEqual(result["contrasts"]["generator_channel_interaction"]["estimate"], .5)
        self.assertEqual(result["contrasts"]["generator_shift_clean"]["estimate"], 0)
        self.assertEqual(result["cells"]["fixture_corpus/G1"]["recordings"], 8)
        self.assertEqual(result["cells"]["fixture_corpus/G1"]["clusters"], 2)
        self.assertEqual(result, self.analyze(fixture()))

    def test_missing_cell_single_class_and_unknown_generator_rejected(self):
        with self.assertRaises(ValueError):
            self.analyze(fixture()[:-1])
        with self.assertRaises(ValueError):
            generator_channel_interaction(fixture(), channel="codec", factor_definition="fixture", replicates=200)
        with self.assertRaises(ValueError):
            self.analyze(fixture(), factor_kind="generator")

    def test_corpus_confounding_is_not_a_generator_effect(self):
        rows = fixture()
        for row in rows:
            if row["factor_group"] == "G1":
                row["dataset_id"] = "other_corpus"
        with self.assertRaisesRegex(ValueError, "confounded"):
            self.analyze(rows)

    def test_shared_clusters_preserve_dependence_without_pairing_utterances(self):
        rows = fixture()
        for row in rows:
            row["speaker_id"] = {"p0": "s1", "p1": "s2", "p2": "s1", "p3": "s2"}[row["speaker_id"]]
        result = self.analyze(rows)
        self.assertEqual(result["contrasts"]["generator_channel_interaction"]["estimate"], .5)
        self.assertEqual(result["cluster_field"], "speaker_id")

    def test_cross_corpus_shared_clusters_and_invalid_labels_are_rejected(self):
        rows = fixture()
        duplicated = [{**row, "sample_id": "other-" + row["sample_id"], "dataset_id": "other_corpus"} for row in rows]
        with self.assertRaisesRegex(ValueError, "Cluster reused"):
            self.analyze(rows + duplicated)
        for row in rows:
            row["label"] = 2
        with self.assertRaises(ValueError):
            generator_channel_interaction(rows, channel="codec", factor_definition="fixture", replicates=200)


if __name__ == "__main__":
    unittest.main()
