import unittest

from src.lava.jaes_v2.phase9_protocol import guarded_selection


class GuardedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.margins = dict.fromkeys(('A_clean_FA', 'A_clean_miss', 'B_clean_FA',
                                     'A05_stress_miss', 'A06_stress_miss'), '0.02')
        self.reference = dict.fromkeys((*self.margins, 'A_stress_FA', 'A_stress_miss'), (10, 100))

    def test_exact_boundary_and_balanced_stress(self):
        candidate = dict(self.reference, B_clean_FA=(12, 100), A_stress_miss=(4, 100))
        self.assertEqual(guarded_selection(candidate, self.reference, self.margins),
                         {'status': 'ELIGIBLE', 'failed': [], 'objective': 0.07})

    def test_stress_improvement_cannot_override_guard_failures(self):
        candidate = dict(self.reference, B_clean_FA=(13, 100), A05_stress_miss=(13, 100),
                         A_stress_FA=(0, 100))
        self.assertEqual(guarded_selection(candidate, self.reference, self.margins),
                         {'status': 'NO_FEASIBLE_CHECKPOINT',
                          'failed': ['B_clean_FA', 'A05_stress_miss'], 'objective': None})

    def test_missing_or_invalid_support_fails_closed(self):
        missing = self.reference.copy()
        missing.pop('A06_stress_miss')
        for candidate in (missing, dict(self.reference, A06_stress_miss=(10, 99)),
                          dict(self.reference, A06_stress_miss=(101, 100)),
                          dict(self.reference, A06_stress_miss=(float('nan'), 100))):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    guarded_selection(candidate, self.reference, self.margins)
