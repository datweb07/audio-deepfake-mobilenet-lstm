"""Phase 5 contracts: genuine gradients, archive safety and leakage controls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

import numpy as np

from src.lava.jaes_v2.asv_archive import safe_member,inspect_archive,extract_train_dev
from src.lava.jaes_v2.preprocessing import Preprocessor
from src.lava.jaes_v2.c2_student import build_student,initialize_tensorflow,objective,fake_probabilities
from src.lava.jaes_v2.c2_training import TrainingConfig,validate_views
from src.lava.jaes_v2.shortcut import available_source_clusters


class PreprocessingTests(unittest.TestCase):
    def test_production_parity_and_silence(self):
        from src.preprocessing import process_audio_data
        x=np.random.default_rng(3).normal(0,.05,66150).astype('float32')
        np.testing.assert_array_equal(Preprocessor('P0').extract(x),process_audio_data(x))
        for p in ('P0','P1','P2'):
            output=Preprocessor(p).extract(np.zeros(66150,dtype='float32'))
            self.assertEqual(output.shape,(6,224,224,3)); self.assertTrue(np.isfinite(output).all())
        self.assertEqual(float(Preprocessor('P1').extract(np.zeros(66150)).max()),0)

    def test_deployment_feasible_scale_invariance(self):
        x=np.random.default_rng(4).normal(0,.1,66150).astype('float32')
        for p in ('P1','P2'):
            np.testing.assert_allclose(Preprocessor(p).extract(x),Preprocessor(p).extract(x*.37),atol=.001)
        self.assertNotEqual(Preprocessor('P0').fingerprint,Preprocessor('P1').fingerprint)

    def test_invalid_inputs(self):
        for x in (np.array([]),np.array([np.nan]),np.zeros((2,4))):
            with self.assertRaises(ValueError): Preprocessor('P1').extract(x)


class ArchiveTests(unittest.TestCase):
    def test_paths(self):
        for value in ('../escape','/absolute','LA/../escape','C:/escape','LA\\escape'):
            with self.assertRaises(ValueError): safe_member(value)

    def test_selective_extraction_never_reads_eval(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); archive=root/'LA.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('LA/ASVspoof2019_LA_train/flac/train.flac',b'train')
                z.writestr('LA/ASVspoof2019_LA_dev/LICENSE.txt',b'license')
                z.writestr('LA/ASVspoof2019_LA_eval/flac/sealed.flac',b'sealed')
                z.writestr('LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt',b'sealed-labels')
            report=inspect_archive(archive)
            self.assertEqual(report['total_expanded_bytes'],5+7+6+13)
            real_open=zipfile.ZipFile.open
            def checked_open(instance,member,*args,**kwargs):
                self.assertNotIn('eval',member.filename if hasattr(member,'filename') else member)
                return real_open(instance,member,*args,**kwargs)
            # Tiny safety fixture must not depend on the host temp volume's
            # production 512MiB operational allowance (audio is only bytes here).
            with patch('src.lava.jaes_v2.asv_archive.RESERVE_BYTES',0),patch.object(zipfile.ZipFile,'open',checked_open),\
                 patch('src.lava.jaes_v2.asv_archive.shutil.disk_usage',return_value=SimpleNamespace(free=16*1024**3)):
                result=extract_train_dev(archive,root/'data',root/'audit')
            self.assertEqual(result['extracted_files'],2)
            self.assertFalse((root/'data/LA/ASVspoof2019_LA_eval').exists())

    def test_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            archive=Path(name)/'LA.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('LA/README.LA.txt',b'a'); z.writestr('LA/README.LA.txt',b'b')
            with self.assertRaises(ValueError): inspect_archive(archive)


class TrainingContractTests(unittest.TestCase):
    def test_selection_deterministic_balanced_and_speaker_aware(self):
        from types import SimpleNamespace
        from src.lava.jaes_v2.c2_data import select_balanced
        records=[SimpleNamespace(sample_id=f'{label}-{speaker}-{i}',split='train',speaker_id=str(speaker),label=label,attack_id='A01' if label else '-')
                 for label in (0,1) for speaker in range(4) for i in range(5)]
        a=select_balanced(records,per_class=8,seed=5804,role='train')
        b=select_balanced(list(reversed(records)),per_class=8,seed=5804,role='train')
        self.assertEqual([r.sample_id for r in a],[r.sample_id for r in b])
        self.assertEqual(sum(r.label for r in a),8); self.assertEqual(len({r.sample_id for r in a}),16)
        self.assertEqual({r.speaker_id for r in a},{'0','1','2','3'})

    def test_interrupted_training_preserves_failure_evidence(self):
        from types import SimpleNamespace
        from src.lava.jaes_v2.c2_training import train
        tf=initialize_tensorflow(); inputs=tf.keras.Input((6,576)); outputs=tf.keras.layers.Dense(1,activation='sigmoid')(tf.keras.layers.Flatten()(inputs))
        head=tf.keras.Model(inputs,outputs)
        from src.lava.jaes_v2.evidence import write_json,write_csv
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); cache=root/'cache'; cache.mkdir(); audit=root/'outputs/jaes_v2/audit'; audit.mkdir(parents=True)
            write_json(cache/'cache_metadata.json',dict(status='SEALED_C2_FEATURE_CACHE',output_hashes={},pretraining_weight_sha256='fixture',selection_hash='fixture',environment={}))
            write_json(audit/'phase5_training_preflight_v2.json',{'status':'PASS'})
            rows=[dict(sample_id=f'{split}-{i}',split=split,policy='P0',condition='clean',label=str(i%2),array_key=split,array_index=i,speaker_id=split,attack_id='UNKNOWN')
                  for split in ('train','development') for i in range(4)]
            write_csv(cache/'view_manifest.csv',rows)
            np.savez(cache/'embeddings.npz',train=np.zeros((4,6,576),dtype='float32'),development=np.zeros((4,6,576),dtype='float32'))
            with patch('src.lava.jaes_v2.c2_training.file_hash',return_value='fixture'),patch('src.lava.jaes_v2.c2_training.source_fingerprint',return_value={}),\
                 patch('src.lava.jaes_v2.c2_training.build_student',return_value=(head,SimpleNamespace(trainable=False),head,None)),\
                 patch('src.lava.jaes_v2.c2_training.objective',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt): train(root,TrainingConfig('interrupt_fixture',cache='cache',phase='TIER1_OVERFIT_SANITY'))
            state=json.loads((root/'outputs/jaes_v2/c2/interrupt_fixture/run_metadata.json').read_text())
            self.assertEqual(state['status'],'INTERRUPTED')
            self.assertFalse((root/'outputs/jaes_v2/c2/interrupt_fixture/validation_scores.csv').exists())

    def test_no_sealed_partitions_or_inconsistent_pairs(self):
        row=dict(sample_id='x',policy='P0',condition='clean',split='train',label='1')
        validate_views([row])
        for split in ('calibration','evaluation','test'):
            with self.assertRaises(ValueError): validate_views([{**row,'split':split}])
        with self.assertRaises(ValueError): validate_views([row,{**row,'condition':'gain','split':'development'}])
        with self.assertRaises(ValueError): validate_views([row,{**row,'condition':'gain','label':'0'}])
        with self.assertRaises(ValueError): validate_views([{**row,'waveform_sha256':'a'},{**row,'policy':'P1','waveform_sha256':'b'}])
        with self.assertRaises(ValueError): validate_views([{**row,'transform_seed':'1'},{**row,'policy':'P1','transform_seed':'2'}])

    def test_frozen_cache_rejects_scratch_and_unbounded_training(self):
        for values in ({'initialization':'scratch'},{'maximum_seconds':3600},{'threshold':.82},{'phase':'CONFIRMATION'}):
            with self.assertRaises(ValueError): TrainingConfig('fixture',**values)

    def test_official_speakers_join_components(self):
        groups=available_source_clusters([{'speaker_id':'A'},{'speaker_id':'A'},{'speaker_id':'B'},{'speaker_id':'UNKNOWN'}])
        self.assertEqual(groups[0],groups[1]); self.assertNotEqual(groups[1],groups[2]); self.assertNotEqual(groups[2],groups[3])

    def test_teacher_score_orientation(self):
        logits=np.array([[2.,0.],[0.,2.]])
        np.testing.assert_allclose(fake_probabilities(logits,fake_index=0),[.8807970779,.1192029220])
        np.testing.assert_allclose(fake_probabilities(logits,fake_index=1),[.1192029220,.8807970779])
        with self.assertRaises(ValueError): fake_probabilities(logits,fake_index=2)

    def test_bce_consistency_gradients_and_save_reload(self):
        tf=initialize_tensorflow()
        tf.keras.utils.set_random_seed(5804)
        full,backbone,head,_=build_student(seed=5804,initialization='scratch')
        self.assertTrue(backbone.trainable)
        x=tf.random.normal((2,6,576)); y=tf.constant([0.,1.])
        with tf.GradientTape() as tape:
            a=head(x,training=False); b=head(x*.5,training=False)
            value=objective(y,a,b,baseline='B4')
        gradients=tape.gradient(value,head.trainable_weights)
        self.assertTrue(all(g is not None and np.isfinite(g.numpy()).all() for g in gradients))
        self.assertGreater(float(tf.linalg.global_norm(gradients)),0)
        known=objective([0.,1.],tf.constant([[.2],[.8]]))
        self.assertAlmostEqual(float(known),-np.log(.8),places=5)
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'fixture.keras'; head.save(path)
            loaded=tf.keras.models.load_model(path,compile=False)
            np.testing.assert_allclose(loaded(x,training=False).numpy(),a.numpy(),atol=1e-6)
        with self.assertRaises(ValueError): objective(y,a,baseline='B4')


if __name__=='__main__': unittest.main()
