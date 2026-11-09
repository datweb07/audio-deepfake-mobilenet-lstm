"""Lightweight, failure-focused tests for the Phase 10 contract."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from scripts.jaes_v2.phase10_numeric import numerical_diff
from src.lava.jaes_v2.evidence import file_hash, write_json
from src.lava.jaes_v2.phase8_training import seal_tree
from src.lava.jaes_v2.phase10_resources import GIB, pressure
from src.lava.jaes_v2.phase10_training import committed_controller
from src.lava.jaes_v2.phase10_validation import compare, plan, select, MARGINS, TIERS


class Phase10ResourceTests(unittest.TestCase):
    def test_resource_stop_categories_and_boundary(self):
        healthy = dict(c_free=4*GIB, d_free=16*GIB, commit_available=3*GIB,
                       ram_available=2*GIB, process_rss=GIB)
        self.assertIsNone(pressure(healthy))
        for field, value, expected in (
            ('d_free', 15*GIB, 'DISK_SPACE_PRESSURE'),
            ('c_free', 3*GIB-1, 'DISK_SPACE_PRESSURE'),
            ('commit_available', 3*GIB-1, 'SYSTEM_COMMIT_PRESSURE'),
            ('ram_available', 2*GIB-1, 'PROCESS_RSS_PRESSURE'),
            ('process_rss', 4*GIB+1, 'PROCESS_RSS_PRESSURE'),
        ):
            with self.subTest(field=field):
                self.assertEqual(pressure(dict(healthy, **{field: value})), expected)

    def test_absolute_tolerance_at_zero_and_nonfinite_rejected(self):
        tolerance = dict(atol=1e-6, rtol=1e-4)
        self.assertTrue(numerical_diff([0, 100], [9e-7, 100.009], tolerance)['pass_thresholds'])
        self.assertFalse(numerical_diff([0], [2e-6], tolerance)['pass_thresholds'])
        for value in ([float('nan')], [float('inf')], [0, 0]):
            with self.assertRaises(ValueError):
                numerical_diff([0], value, tolerance)


class Phase10CheckpointTests(unittest.TestCase):
    def test_pointer_commit_and_strict_contract(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            complete = out / 'checkpoints/u002'
            contract = dict(config='c', model='m', dataset='d', backend='vjp', augmentation_rng='rng',
                            batch_size=4, preprocessing='p0', source='rev')
            state = dict(contract=contract, step=2)
            write_json(complete / 'controller.json', state)
            (complete / 'state.data').write_bytes(b'optimizer payload')
            seal_tree(complete)
            write_json(out / 'latest.json', dict(checkpoint='u002', sha256=file_hash(complete / 'controller.json')))
            write_json(out / 'checkpoints/u003.pending/controller.json', dict(state, step=3))
            self.assertEqual(committed_controller(out, contract), (complete, state))
            for key in contract:
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'contract mismatch'):
                    committed_controller(out, dict(contract, **{key: 'different'}))
            (complete / 'state.data').write_bytes(b'corrupted optimizer')
            with self.assertRaisesRegex(ValueError, 'Artifact changed'):
                committed_controller(out, contract)
            write_json(out / 'latest.json', dict(checkpoint='../u002', sha256='wrong'))
            with self.assertRaises(ValueError):
                committed_controller(out, contract)


class Phase10ValidationTests(unittest.TestCase):
    def fixture(self):
        labels = [('a', 0, '-'), ('f5', 1, 'A05'), ('f6', 1, 'A06'), ('b', 0, '-')]
        records = [SimpleNamespace(sample_id=i, label=y, attack_id=a) for i,y,a in labels]
        return records, {'b'}

    def test_tier_schedule(self):
        records, b = self.fixture()
        self.assertEqual(len(list(plan(records, b, 'A'))), 3)
        self.assertEqual(len(list(plan(records, b, 'B'))), 13)
        self.assertEqual(len(list(plan(records, b, 'C'))), len(records) * len(TIERS['C']))

    def rows(self, tier='B'):
        records, b = self.fixture()
        return [dict(sample_id=r.sample_id, label=r.label, attack_id=r.attack_id,
                     condition_id=c, stratum=s, raw_score=.8 if r.label else .2,
                     predicted_label=r.label) for r, c, s in plan(records, b, tier)]

    def test_guard_fails_closed_on_missing_scores_or_subgroup_regressions(self):
        rows = self.rows()
        self.assertEqual(select(rows, rows, MARGINS, {'A_clean_FA': .10})['status'], 'ELIGIBLE')
        with self.assertRaisesRegex(ValueError, 'cohort'):
            select(rows[:-1], rows, MARGINS, {})
        candidate = [dict(r, predicted_label=1, raw_score=.8) if r['sample_id'] == 'b' else r for r in rows]
        result = select(candidate, rows, MARGINS, {})
        self.assertEqual(result['status'], 'NO_FEASIBLE_CHECKPOINT')
        self.assertEqual(result['failed'], ['B_clean_FA'])

    def test_parity_detects_flips_even_within_score_tolerance(self):
        rows = self.rows('C')
        a = [dict(rows[0], raw_score=.49999), *rows[1:]]
        b = [dict(a[0], raw_score=.50001, predicted_label=1), *rows[1:]]
        result = compare(a, b, max_score=3e-5, max_rank_metric=.005, near_margin=.001)
        self.assertEqual(result['near_threshold_rows'], 1)
        self.assertEqual(result['decision_flips'], 1)
        self.assertFalse(result['pass_thresholds'])
        self.assertTrue(compare(rows, rows, max_score=3e-5, max_rank_metric=.005, near_margin=.001)['pass_thresholds'])
