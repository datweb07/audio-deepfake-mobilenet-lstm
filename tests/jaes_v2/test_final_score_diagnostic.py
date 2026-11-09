import shutil
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.jaes_v2.final_order_components import views
from scripts.jaes_v2.final_score_diagnostic import (
    A, B, align, bootstrap_auc, cluster_delta, expected_rows, metrics, transitions,
)
from src.lava.jaes_v2.evidence import file_hash
from src.lava.jaes_v2.transforms import Audio


class FinalDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.record = SimpleNamespace(sample_id='one', parent_recording_id='parent',
                                      speaker_id='speaker', attack_id='-', label=0,
                                      split='development')
        self.identities = expected_rows([self.record], {'one'})
        self.row = dict(self.identities['one'], sample_id='one', condition_id='clean',
                        raw_score='0.5', predicted_label='1', threshold='0.5', status='OK')

    def test_alignment_and_fake_orientation(self):
        other = dict(self.row, condition_id='noise', raw_score='0.1', predicted_label='0')
        first = align([self.row, other], self.identities, ('clean','noise'))
        second = align([other, self.row], self.identities, ('clean','noise'))
        self.assertEqual(first, second)
        self.assertEqual(first['one','clean',B]['error'], 1)
        for bad in (dict(self.row, predicted_label='0'), dict(self.row, raw_score='nan'),
                    dict(self.row, speaker_id='other'), dict(self.row, parent_recording_id='other')):
            with self.assertRaises(ValueError):
                align([bad], self.identities, ('clean',))

    def test_duplicate_or_missing_grid_rejected(self):
        with self.assertRaises(ValueError):
            align([self.row,self.row], self.identities, ('clean',))
        with self.assertRaises(ValueError):
            align([self.row], self.identities, ('clean','noise'))
        with self.assertRaises(ValueError):
            expected_rows([self.record,self.record], {'one'})
        with self.assertRaises(ValueError):
            expected_rows([self.record,SimpleNamespace(**(vars(self.record)|{'sample_id':'two'}))], {'one'})

    def test_one_class_metrics(self):
        report = metrics(list(align([self.row],self.identities,('clean',)).values()), curves=True)
        self.assertIsNone(report['auc'])
        self.assertIsNone(report['eer'])
        self.assertNotIn('roc',report)
        self.assertEqual(report['false_alarm']['errors'],1)

    def test_paired_counts_and_grouped_bootstrap(self):
        self.assertEqual(transitions([0,0,1,1],[0,1,0,1]),
                         dict(both_correct=1,both_wrong=1,only_first_wrong=1,only_second_wrong=1))
        left=[dict(speaker_id='one',error=0),dict(speaker_id='one',error=0),dict(speaker_id='two',error=1)]
        right=[dict(row,error=1-row['error']) for row in left]
        weights=np.array([[2,0],[1,1],[0,2],[0,0]])
        summary=cluster_delta(left,right,['one','two'],weights)
        self.assertAlmostEqual(summary['delta'],1/3)
        self.assertEqual((summary['valid'],summary['invalid']),(3,1))
        labels=[dict(speaker_id='one',p=.8,y=1),dict(speaker_id='two',p=.2,y=0)]
        np.testing.assert_equal(bootstrap_auc(labels,['one','two'],weights),[np.nan,1,np.nan,np.nan])

    def test_forbidden_partition_rejected_before_codec(self):
        for split in ('train','calibration','evaluation'):
            record=SimpleNamespace(**(vars(self.record)|{'split':split}))
            with self.assertRaises(ValueError):
                expected_rows([record], set())
            with self.assertRaises(ValueError):
                views(None,record,None)

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required')
    def test_deterministic_components_and_read_only_reference(self):
        protected=Path('papers/PaperID_804_final.docx')
        before=file_hash(protected)
        audio=Audio(.12*np.sin(np.arange(4000)*.05),22050)
        first,_=views(audio,self.record,shutil.which('ffmpeg'))
        second,_=views(audio,self.record,shutil.which('ffmpeg'))
        for name in ('clean','noise10','opus24','A_then_B','B_then_A'):
            np.testing.assert_array_equal(first[name].samples,second[name].samples)
        self.assertEqual(file_hash(protected),before)
