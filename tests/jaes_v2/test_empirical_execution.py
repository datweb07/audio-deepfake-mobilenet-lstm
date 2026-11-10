import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.jaes_v2.empirical_readiness import ATTACKS, commit_source_scores, sample_heldout, write_once
from scripts.jaes_v2.empirical_heldout import run, waveform_bank
from scripts.jaes_v2.phase10_medium import registered
from src.lava.jaes_v2 import phase8_training as prior
from src.lava.jaes_v2.c2_data import paired_views
from src.lava.jaes_v2.transforms import load_audio


class EmpiricalExecutionTests(unittest.TestCase):
    def test_deterministic_speaker_balanced_selection(self):
        records=[dict(sample_id=f'{attack}-{speaker}-{i}', speaker_id=speaker,
                      attack=attack,split='evaluation')
                 for attack in ATTACKS for speaker in ('a','b') for i in range(3)]
        quotas=dict.fromkeys(ATTACKS,2)
        first,support=sample_heldout(records,91109,quotas)
        self.assertEqual((first,support),sample_heldout(reversed(records),91109,quotas))
        for attack in ATTACKS:
            self.assertEqual({r['speaker_id'] for r in first if r['attack']==attack},{'a','b'})
        with self.assertRaises(ValueError):
            sample_heldout(records+[records[0]],91109,quotas)
        with self.assertRaises(ValueError):
            sample_heldout(records,91109,quotas,blocked_speakers={'a'})
        with self.assertRaises(ValueError):
            sample_heldout([dict(records[0],split='calibration')],91109,quotas)

    def test_locked_shards_resume_without_duplicate_or_changed_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            rows=[dict(model='model',condition='clean',sample_id='a',raw_score=.5,predicted_label=1)]
            self.assertTrue(commit_source_scores(root,'lock','a',rows,['model'],['clean']))
            self.assertFalse(commit_source_scores(root,'lock','a',rows,['model'],['clean']))
            for args in [('new','a',rows),('lock','a',rows+rows),('lock','a',[dict(rows[0],predicted_label=0)])]:
                with self.assertRaises(ValueError):
                    commit_source_scores(root,*args,['model'],['clean'])
            self.assertEqual(len(list(root.glob('*.json'))),1)

    def test_pending_lock_never_opens_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            lock=root/'lock.json'
            write_once(lock,{'authorized':False})
            with patch('scripts.jaes_v2.empirical_heldout.ZipFile') as archive:
                with self.assertRaisesRegex(ValueError,'not authorized'):
                    run(root,lock)
                archive.assert_not_called()

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required')
    def test_heldout_waveform_formula_matches_real_development_contract(self):
        root=Path.cwd()
        _,cfg=registered('p10_medium_erm_s9107_v2')
        _,dev,_,_=prior.dataset(root,cfg)
        record=dev[0]
        path=prior.source_path(root,record.path)
        ffmpeg=shutil.which('ffmpeg')
        registered_views,_=paired_views(load_audio(path,sample_rate=22050,max_seconds=3),
                                        record.sample_id,'development',ffmpeg)
        actual=waveform_bank(path.read_bytes(),record.sample_id.split(':')[-1],ffmpeg)
        for name, waveform in actual.items():
            np.testing.assert_array_equal(waveform.samples,registered_views[name].samples)
