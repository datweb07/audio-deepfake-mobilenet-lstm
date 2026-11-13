"""Static, score and admission tests. No optimizer runs."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.jaes_v2 import phase10_size1280 as cli, size1280_profile as profile
from scripts.jaes_v2.size1280_analysis import contrast, report
from src.lava.jaes_v2 import phase8_training as prior, size1280_training as runner
from src.lava.jaes_v2.phase10_resources import GIB, MIN_D_FREE, WRITE_RESERVE, ResourceStop


class Size1280Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(profile.REGISTRY.read_text(encoding='utf-8'))

    def test_registry_rejects_changed_bytes_before_loading_config(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / profile.REGISTRY
            path.parent.mkdir(parents=True)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'registry changed'):
                profile.registry(root)

    def test_budget_and_per_checkpoint_reserve(self):
        snap = dict(d_free=17 * GIB)
        scientific = profile.storage(Path('.'), self.spec, current=snap)
        bounded = profile.storage(Path('.'), self.spec, qualification=True, current=snap)
        self.assertEqual(scientific['planned_remaining_checkpoints'], 338)
        self.assertEqual(bounded['planned_remaining_checkpoints'], 20)
        self.assertGreater(scientific['additional_disk_bytes'], 0)
        self.assertEqual(bounded['additional_disk_bytes'], 0)
        with patch.object(profile, 'require_resources', return_value={'d_free': MIN_D_FREE + WRITE_RESERVE}):
            with self.assertRaisesRegex(ResourceStop, 'BEFORE_WRITE'):
                profile.guard_disk(Path('.'), 1)

    def test_same_source_order_and_exact_epoch_transition(self):
        config = SimpleNamespace(training_seed=9207, sampling='natural', batch_size=16,
                                 warmup_epochs=1, finetune_epochs=20,
                                 learning_rate=.001, finetune_learning_rate=.0001)
        rows = list(range(1280))
        for epoch in (0, 1, 20):
            order = prior.epoch_order(rows, config, epoch)
            self.assertEqual(sorted(order), rows)
            a = runner.update_plan(rows, SimpleNamespace(**vars(config), augmentation_policy='clean'), epoch, 0, order)
            b = runner.update_plan(rows, SimpleNamespace(**vars(config), augmentation_policy='channel'), epoch, 0, order)
            self.assertEqual([r for r, _ in a], [r for r, _ in b])
            self.assertEqual(len(b), 16)
        np.testing.assert_allclose([runner.learning_rate(config, e) for e in (0, 1, 20)], [.001, .0001, .0000525])

    def test_separate_paths_and_qualification_gate(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIn('size1280_v1', profile.directory(root, self.spec, 'erm', 'scientific').parts)
            with self.assertRaises(ValueError):
                profile.directory(root, self.spec, 'erm', 'uninterrupted')
            with patch.object(cli, 'QUALIFICATION', root):
                with self.assertRaisesRegex(ValueError, 'zero-update smoke'):
                    cli.require_qualification()
                with patch.object(cli, 'require_smoke'):
                    with self.assertRaisesRegex(ValueError, 'manual bounded qualification'):
                        cli.require_qualification()

    def test_paired_size_contrast_and_negative_effect(self):
        values = (1., 3., 4., 5.)
        self.assertEqual(contrast(values), dict(delta640=2., delta1280=1., delta_size=-1.))
        boot = [np.array([value, value + 20]) for value in values]
        result = report(values, boot)
        self.assertEqual(result['delta_size']['percentile95'], [-1., -1.])
