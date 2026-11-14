import io
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from zipfile import BadZipFile, ZipFile, ZIP_DEFLATED

import numpy as np
import soundfile as sf

from scripts.jaes_v2 import r08_data as d, r08_external as r


class R08Tests(TestCase):
    def test_metadata_labels_reproducibility_overlap(self):
        raw = b'file,speaker,label\n0.wav,Speaker One,bona-fide\n1.wav,Speaker One,spoof\n2.wav,Other,spoof\n'
        rows = d.parse_metadata(raw)
        self.assertEqual([x['label'] for x in rows], [0, 1, 1])
        self.assertEqual(d.shortlist(rows, 1, 1), d.shortlist(rows[::-1], 1, 1))
        with self.assertRaisesRegex(ValueError, 'metadata quota'):
            d.shortlist(rows, 1, 1, blocked_speakers={'speakerone'})
        with self.assertRaisesRegex(ValueError, 'metadata quota'):
            d.shortlist(rows, 1, 1, blocked_ids={'0.wav'})
        with self.assertRaises(ValueError):
            d.parse_metadata(raw + b'0.WAV,Other,spoof\n')
        with self.assertRaises(ValueError):
            d.parse_metadata(raw.replace(b'spoof', b'unverified'))

    def test_path_and_crc(self):
        for name in ('../a.wav', 'C:/a.wav', 'sub/a.wav', 'x\\a.wav'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                d.member_name(name)
        buf = io.BytesIO()
        row = {'sample_id':'a.wav','member':d.member_name('a.wav')}
        with ZipFile(buf, 'w', compression=ZIP_DEFLATED) as z:
            z.writestr(row['member'], b'my wav bytes')
        with ZipFile(buf) as z:
            self.assertEqual(d.read_member(z,row), b'my wav bytes')
            z.getinfo(row['member']).CRC ^= 1
            with self.assertRaises(BadZipFile):
                d.read_member(z,row)

    def test_range_identity_budget_and_retry(self):
        class Reply(io.BytesIO):
            status = 206
            headers = {'Content-Range':'bytes 2-4/10','ETag':'"test"','Content-Length':'3'}
            def geturl(self):
                return 'https://example.org/object'
        debits = []
        stream = d.RemoteZip(6, lambda:None, url='https://example.org/archive', length=10, etag='"test"', debit=debits.append)
        stream.seek(2)
        bad = Reply(b'abc')
        bad.headers = dict(bad.headers, ETag='"wrong"')
        with patch.object(d, 'urlopen', return_value=bad), self.assertRaisesRegex(ValueError, 'identity'):
            stream.read(3)
        self.assertEqual(stream.tell(),2)
        with patch.object(d, 'urlopen', return_value=Reply(b'abc')) as opened:
            self.assertEqual(stream.read(3),b'abc')
            self.assertEqual(opened.call_args.args[0].get_header('Range'),'bytes=2-4')
        self.assertEqual(debits,[3,3])
        with self.assertRaisesRegex(ValueError, 'budget'):
            stream.read(1)

    def test_wav_and_duplicate_fingerprints(self):
        stream = io.BytesIO()
        sf.write(stream, np.sin(np.arange(16000,dtype=np.float32)*.1)*.1, 16000, format='WAV')
        info = d.qualified_audio(stream.getvalue())
        self.assertEqual(info['shape'],r.SHAPE)
        left = dict(info,sample_id='a.wav',source_group='a')
        right = dict(info,sample_id='b.wav',source_group='b',sha256='different')
        with self.assertRaisesRegex(ValueError,'decoded_sha256'):
            r.check_unique([left,right])
        with self.assertRaisesRegex(ValueError,'overlap'):
            r.check_unique([left],{info['sha256']})
        with self.assertRaisesRegex(ValueError,'source group'):
            r.check_unique([left,dict(right,decoded_sha256='different',source_group='a')])

    def test_authorization_and_hash(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'approval.json'
            with self.assertRaisesRegex(ValueError,'authorization'):
                r.approval(path,'R08',{'sha256':'wanted'})
            r.write_once(path,dict(scope='R08',sha256='wrong',author='Fixture',
                                   approved_at='2026-11-13T00:00:00+07:00',approval_reference='Fixture'))
            with self.assertRaisesRegex(ValueError,'authorization'):
                r.approval(path,'R08',{'sha256':'wanted'})
            self.assertEqual(r.approval(path,'R08',{'sha256':'wrong'})['author'],'Fixture')

    def test_score_threshold_identity_and_single_class(self):
        self.assertEqual(r.score_value(.5),(.5,1))
        self.assertEqual(r.score_value(.49999),(.49999,0))
        for value in (float('nan'),float('inf'),-.1,1.1):
            with self.assertRaises(ValueError):
                r.score_value(value)
        fixture = dict(sample_id='s',speaker_id='sp',label=0)
        left = r.check_prediction(dict(fixture,raw_score=.5,predicted_label=1),fixture)
        right = r.check_prediction(dict(fixture,raw_score=.4,predicted_label=0),fixture)
        with self.assertRaisesRegex(ValueError,'threshold'):
            r.check_prediction(dict(fixture,raw_score=.5,predicted_label=0),fixture)
        parts = {f'{m}_{s}':[left if m=='erm' else right] for s in r.SEEDS for m in r.METHODS}
        summary = r.summarize(parts,bootstrap=True)
        self.assertEqual(summary['models']['erm_9107']['confusion'],dict(tp=0,tn=0,fn=0,fp=1))
        self.assertIsNone(summary['models']['erm_9107']['auc'])
        self.assertIsNone(summary['models']['erm_9107']['fnr'])
        self.assertEqual(summary['paired']['9107']['delta']['fpr'],-1)
        self.assertEqual(summary['paired']['9107']['class_transitions']['0']['speaker_interval']['valid'],10000)

    def test_interrupted_scoring_skips_verified_complete_model(self):
        with TemporaryDirectory() as tmp, patch.object(r,'OUT',Path(tmp)), patch.object(r,'file_hash',return_value='protocol'):
            records = [dict(sample_id='clip.wav',speaker_id='speaker',source_group=None,label=0,sha256='bytes')]
            entry = dict(id='erm_9107',seed=9107,method='erm',checkpoint='step',model_sha256='weights')
            lock = dict(evaluation_id='fixture',rows=records,models=[entry])
            path = r.shard(Path(tmp)/'run',entry,records[0])
            identity = r.identity(lock,entry,records[0])
            r.write_once(path,dict(identity,raw_score=.5,predicted_label=1))
            with patch.object(r.engine,'phase10_tensorflow'), patch.object(r,'restore') as restore:
                r.score(lock)
                restore.assert_not_called()

    def test_extra_prediction_is_rejected_before_export(self):
        with TemporaryDirectory() as tmp, patch.object(r,'OUT',Path(tmp)), patch.object(r,'file_hash',return_value='protocol'):
            row = dict(sample_id='clip.wav',speaker_id='speaker',source_group=None,label=0,sha256='bytes')
            entry = dict(id='erm_9107',seed=9107,method='erm',checkpoint='step',model_sha256='weights')
            lock = dict(evaluation_id='fixture',rows=[row],models=[entry],predictions=1)
            path = r.shard(Path(tmp)/'run',entry,row)
            identity = r.identity(lock,entry,row)
            r.write_once(path,dict(identity,raw_score=.5,predicted_label=1))
            r.write_once(path.with_name('duplicate.json'),dict(identity,raw_score=.5,predicted_label=1))
            with self.assertRaisesRegex(ValueError,'extra prediction'):
                r.finalize(lock)
            self.assertFalse((Path(tmp)/'run'/'scores.csv').exists())

    def test_storage_stop(self):
        with patch.object(r,'require_resources',return_value={'d_free':r.MIN_D_FREE+r.WRITE_RESERVE}):
            with self.assertRaisesRegex(ValueError,'Insufficient D space'):
                r.guard(1)
