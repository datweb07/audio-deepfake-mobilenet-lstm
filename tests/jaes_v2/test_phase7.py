import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from src.lava.jaes_v2.phase7_data import all_attack_select,ATTACKS,safe_path,atomic_npz,load_cache
from src.lava.jaes_v2.phase7_training import Config,seeds,online_rawboost,score_metrics
from src.lava.jaes_v2.phase7_statistics import validate_scores,paired_seed_effects,speaker_risks

def fixture(role='development'):
    rows=[]
    for speaker in ('s0','s1'):
        for a in ('REAL',)+ATTACKS:
            for n in range(30): rows.append(SimpleNamespace(sample_id=f'{speaker}_{a}_{n:02}',speaker_id=speaker,label=int(a!='REAL'),attack_id=a,split=role))
    for n in range(30): rows.append(SimpleNamespace(sample_id=f'realonly{n}',speaker_id='s2',label=0,attack_id='REAL',split=role))
    return rows

class SelectionTests(unittest.TestCase):
    def test_deterministic_and_input_order_invariant(self):
        a,b,_=all_attack_select(fixture(),role='development'); x,y,_=all_attack_select(list(reversed(fixture())),role='development')
        self.assertEqual([r.sample_id for r in a],[r.sample_id for r in x]); self.assertEqual([r.sample_id for r in b],[r.sample_id for r in y])
    def test_every_speaker_six_attacks_and_class_counts(self):
        a,b,_=all_attack_select(fixture(),role='development')
        for s in ('s0','s1'):
            g=[r for r in a if r.speaker_id==s]; self.assertEqual(sum(r.label==0 for r in g),16); self.assertEqual(sum(r.label==1 for r in g),16)
            self.assertTrue(all(sum(r.attack_id==attack for r in g) in (2,3) for attack in ATTACKS))
        self.assertEqual(len(b),16); self.assertTrue(all(r.label==0 for r in b))
    def test_no_reserved_role(self):
        for role in ('calibration','evaluation','test'):
            with self.assertRaises(ValueError): all_attack_select(fixture(role),role=role)
    def test_wrong_record_partition_rejected(self):
        with self.assertRaises(ValueError): all_attack_select(fixture(),role='train')
    def test_missing_attack_fails(self):
        with self.assertRaises(ValueError): all_attack_select([r for r in fixture() if r.attack_id!='A06'],role='development')
    def test_distinct_selection_seed(self):
        a,_,_=all_attack_select(fixture(),role='development',seed=1); b,_,_=all_attack_select(fixture(),role='development',seed=2)
        self.assertNotEqual([r.sample_id for r in a],[r.sample_id for r in b])

class ContractTests(unittest.TestCase):
    def test_cropped_transformed_duplicate_rejected(self):
        from src.lava.jaes_v2.phase7_cache_audit import cross_partition_waveforms
        with self.assertRaisesRegex(ValueError,'waveform overlap'):
            cross_partition_waveforms([dict(waveform_sha256='same',split='train'),dict(waveform_sha256='same',split='development')])
        self.assertEqual(cross_partition_waveforms([dict(waveform_sha256='same',split='train'),dict(waveform_sha256='same',split='train')]),1)
    def test_unsafe_path(self):
        root=Path.cwd()
        for path in ('../private','C:/elsewhere','outputs/../../bad'):
            with self.assertRaises(ValueError): safe_path(root,path)
    def test_unknown_config_key_and_confirmation_rejected(self):
        with self.assertRaises(TypeError): Config(unsupported=True)
        with self.assertRaises(ValueError): Config('p7_bad','x','a'*64,'a',1,'x',6107,'H0',scope='CONFIRMATION')
    def test_atomic_publication(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.npz'; atomic_npz(path,x=np.arange(4)); self.assertTrue(path.exists()); self.assertFalse(path.with_suffix('.tmp.npz').exists())
            with np.load(path) as z: np.testing.assert_array_equal(z['x'],np.arange(4))
    def test_interruption_no_published_archive(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.npz'
            with patch('numpy.savez_compressed',side_effect=RuntimeError('interrupted')):
                with self.assertRaises(RuntimeError): atomic_npz(path,x=np.arange(4))
            self.assertFalse(path.exists())
    def test_stale_cache_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p=root/'outputs/jaes_v2/phase7/caches/a'; p.mkdir(parents=True)
            (p/'cache_metadata.json').write_text(json.dumps(dict(status='SEALED_PHASE7_CACHE',benchmark_only=False,contract_hash='wrong')))
            with patch('src.lava.jaes_v2.phase7_data.cache_contract',return_value=({'selection_hash':'current'},[])):
                with self.assertRaisesRegex(ValueError,'Stale'): load_cache(root,{'cache_id':'a'})
    def test_cache_interrupted_resume_and_atomic_directory_publish(self):
        from src.lava.jaes_v2.phase7_data import build_cache
        from src.lava.jaes_v2.evidence import file_hash
        from src.lava.jaes_v2.transforms import Audio
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'source.bin').write_bytes(b'fixture');digest=file_hash(root/'source.bin')
            records=[SimpleNamespace(sample_id=k,split=role,label=i,speaker_id=k,attack_id='A01',path='source.bin',checksum=digest) for i,(k,role) in enumerate([('a','train'),('b','development')])]
            def views(root,r,ffmpeg):return ({name:Audio(np.ones(32,dtype='float32')*.1,22050) for name in ('clean','gain_half','gain_quarter','noise10','noise5','opus24','opus64','A_then_B','B_then_A','rawboost0','rawboost1')},{'sample_seed':123})
            fake_tf=SimpleNamespace(config=SimpleNamespace(list_physical_devices=lambda kind:[]))
            extractor=lambda x,training=False:SimpleNamespace(numpy=lambda:np.zeros((len(x),6,576),dtype='float32'))
            cfg=dict(cache_id='unit',disk_reserve_gib=15,maximum_seconds=30)
            from contextlib import ExitStack
            with ExitStack() as stack:
                stack.enter_context(patch('src.lava.jaes_v2.phase7_data.cache_contract',return_value=({'source_hash':'fixture'},records)))
                stack.enter_context(patch('src.lava.jaes_v2.c2_student.initialize_tensorflow',return_value=fake_tf))
                stack.enter_context(patch('src.lava.jaes_v2.c2_student.build_student',return_value=(None,None,None,extractor)))
                stack.enter_context(patch('src.lava.jaes_v2.preprocessing.Preprocessor',return_value=SimpleNamespace(extract=lambda x:np.zeros((2,2),dtype='float32'))))
                stack.enter_context(patch('shutil.disk_usage',return_value=SimpleNamespace(free=32*1024**3)))
                stack.enter_context(patch('shutil.which',return_value='fixture_ffmpeg'))
                with patch('src.lava.jaes_v2.phase7_data.waveform_views',side_effect=[views(root,records[0],''),RuntimeError('interrupt after first shard')]):
                    with self.assertRaisesRegex(RuntimeError,'interrupt'):build_cache(root,cfg)
                destination=root/'outputs/jaes_v2/phase7/caches/unit'
                self.assertFalse(destination.exists());self.assertTrue(destination.with_name('unit.incomplete').exists())
                with patch('src.lava.jaes_v2.phase7_data.waveform_views',side_effect=views) as generated:
                    result=build_cache(root,cfg);self.assertEqual(result['resumed_shards'],1);self.assertEqual(generated.call_count,1)
                self.assertTrue(destination.exists());self.assertFalse(destination.with_name('unit.incomplete').exists());self.assertEqual(result['views'],13)
                # Complete cache is immutable; a second publication cannot replace it.
                with self.assertRaises(FileExistsError):build_cache(root,cfg)

class SeedAugmentationTests(unittest.TestCase):
    def test_canonical_hash_independent_of_trainability_order(self):
        from src.lava.jaes_v2.phase7_training import named_weights_hash,weights_hash
        def variable(name,value):return SimpleNamespace(name=name,shape=SimpleNamespace(as_list=lambda:[1]),numpy=lambda:np.array([value],dtype='float32'))
        weights=[variable('first/kernel',1),variable('last/kernel',2)]
        self.assertEqual(named_weights_hash(weights),named_weights_hash(list(reversed(weights))))
        self.assertNotEqual(weights_hash(weights),weights_hash(list(reversed(weights))))
    def test_rng_streams(self):
        a=seeds(6107); self.assertNotEqual(a['batch_order'],a['view_choice']); self.assertNotEqual(seeds(6107),seeds(6207))
        order=np.random.default_rng(a['batch_order']); view=np.random.default_rng(a['view_choice']); expected=order.permutation(20); view.random(100)
        np.testing.assert_array_equal(expected,np.random.default_rng(a['batch_order']).permutation(20))
    def test_online_determinism_freshness_rng_isolation(self):
        x=.1*np.sin(np.arange(4096)*.1); np.random.seed(55); before=np.random.get_state()
        a,sa=online_rawboost(x,training_seed=1,epoch=0,sample_id='a'); b,sb=online_rawboost(x,training_seed=1,epoch=0,sample_id='a'); c,sc=online_rawboost(x,training_seed=1,epoch=1,sample_id='a')
        np.testing.assert_array_equal(a,b); self.assertEqual(sa,sb); self.assertNotEqual(sa,sc); self.assertFalse(np.array_equal(a,c)); np.testing.assert_array_equal(before[1],np.random.get_state()[1]); self.assertLessEqual(np.max(np.abs(a)),1.00001); self.assertEqual(a.shape,x.shape)
    def test_rate_mapping_explicit(self):
        with self.assertRaises(ValueError): online_rawboost(np.ones(30),training_seed=1,epoch=0,sample_id='x',sample_rate=16000)

class StatisticsTests(unittest.TestCase):
    def test_one_class_metrics_null(self):
        result=score_metrics([dict(label=1,raw_score=.1),dict(label=1,raw_score=.8)])
        self.assertIsNone(result['roc_auc']); self.assertIsNone(result['eer']); self.assertIsNone(result['REAL_false_alarm']); self.assertEqual(result['FAKE_miss'],.5)
    def test_missing_prediction_not_dropped(self):
        with self.assertRaises(ValueError): validate_scores([], [('x','clean')])
    def test_duplicate_prediction_rejected(self):
        row=dict(sample_id='x',condition_id='clean')
        with self.assertRaises(ValueError): validate_scores([row,row],[('x','clean')])
    def test_paired_seed_and_speaker_units(self):
        a={1:{'s1':0.,'s2':1.},2:{'s1':.5,'s2':1.},3:{'s1':1.,'s2':1.}}; b={k:{'s1':1.,'s2':1.} for k in a}
        m=paired_seed_effects(a,b); self.assertEqual(m['seed_count'],3); self.assertEqual(m['speaker_count'],2); self.assertAlmostEqual(m['mean'],-.25)
        with self.assertRaises(ValueError): paired_seed_effects(a,{1:b[1]})
    def test_classwise_no_fabricated_support(self):
        rows=[dict(label=0,predicted_label=1,speaker_id='s1',condition_id='noise5',stratum='B_REAL_ONLY')]
        self.assertEqual(speaker_risks(rows,label=1,stratum='B_REAL_ONLY'),{}); self.assertEqual(speaker_risks(rows,label=0,stratum='B_REAL_ONLY'),{'s1':1.})

class ReadinessTests(unittest.TestCase):
    def test_confirmation_fail_closed(self):
        from src.lava.jaes_v2.phase7_readiness import ConfirmationContract,readiness,require_locked
        self.assertEqual(readiness(ConfirmationContract())['status'],'NOT_LOCKED')
        with self.assertRaises(ValueError):require_locked(ConfirmationContract(status='LOCKED'))
    def test_reserved_calibration_requires_real_lock(self):
        from src.lava.jaes_v2.phase7_readiness import CalibrationContract
        with self.assertRaises(ValueError):CalibrationContract().validate_for_fit()
    def test_supplied_temperature_preserves_ranking(self):
        from src.lava.jaes_v2.phase7_readiness import temperature_transform
        x=np.array([.1,.4,.9]);np.testing.assert_allclose(temperature_transform(x,1),x);self.assertTrue(np.all(np.diff(temperature_transform(x,2))>0))
        with self.assertRaises(ValueError):temperature_transform(x,0)
    def test_report_rejects_unsupported_promotion_and_duplicate_ids(self):
        from scripts.jaes_v2.build_master_research_report_phase7 import verify_statuses
        from copy import deepcopy
        valid=dict(confirmation='NOT_LOCKED',C2='MULTI_SEED_STANDARD_BASELINES; NOVELTY_NOT_ESTABLISHED',C3='DEFERRED',experiments=[dict(experiment_id='a')],phase7=dict(statistics=dict(seed_aggregates=[dict(method='H'+str(i),seeds=3) for i in range(5)])))
        verify_statuses(valid)
        for mutate in (lambda x:x.update(confirmation='CONFIRMED'),lambda x:x['experiments'].append(dict(experiment_id='a')),lambda x:x['phase7']['statistics']['seed_aggregates'][0].update(seeds=1)):
            invalid=deepcopy(valid);mutate(invalid)
            with self.assertRaises(ValueError):verify_statuses(invalid)

class TensorFlowLifecycleTests(unittest.TestCase):
    def test_seeded_initialization_and_real_cnn_accumulation(self):
        from src.lava.jaes_v2.c2_student import initialize_tensorflow
        tf=initialize_tensorflow()
        def model(seed):
            tf.keras.utils.set_random_seed(seed)
            m=tf.keras.Sequential([tf.keras.layers.Input((8,1)),tf.keras.layers.Conv1D(2,3),tf.keras.layers.BatchNormalization(),tf.keras.layers.Flatten(),tf.keras.layers.Dense(1)])
            m.layers[1].trainable=False
            return m
        a=model(6107); b=model(6107); c=model(6207)
        for x,y in zip(a.weights,b.weights): np.testing.assert_array_equal(x.numpy(),y.numpy())
        self.assertFalse(np.array_equal(a.weights[0].numpy(),c.weights[0].numpy()))
        frozen=[v.numpy().copy() for v in a.layers[1].weights]; old=a.weights[0].numpy().copy(); optimizer=tf.keras.optimizers.Adam(.001)
        x=tf.constant(np.arange(16,dtype='float32').reshape(2,8,1)/16); y=tf.constant([[0.],[1.]])
        with tf.GradientTape() as tape: batch_loss=tf.reduce_mean((a(x,training=True)-y)**2)
        full_batch_gradients=tape.gradient(batch_loss,a.trainable_weights)
        gradients=[tf.zeros_like(v) for v in a.trainable_weights]
        for j in range(2):
            with tf.GradientTape() as tape: loss=tf.reduce_mean((a(x[j:j+1],training=True)-y[j:j+1])**2)*.5
            gradients=[u+v for u,v in zip(gradients,tape.gradient(loss,a.trainable_weights))]
        for accumulated,reference in zip(gradients,full_batch_gradients):np.testing.assert_allclose(accumulated.numpy(),reference.numpy(),rtol=1e-5,atol=1e-6)
        self.assertGreater(float(tf.linalg.norm(gradients[0])),0); optimizer.apply_gradients(zip(gradients,a.trainable_weights)); self.assertFalse(np.array_equal(old,a.weights[0].numpy()))
        for old,v in zip(frozen,a.layers[1].weights): np.testing.assert_array_equal(old,v.numpy())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.keras'; a.save(p); restored=tf.keras.models.load_model(p,compile=False); np.testing.assert_allclose(a(x,training=False).numpy(),restored(x,training=False).numpy(),atol=1e-7)

if __name__=='__main__': unittest.main()
