"""Independent numeric oracles and fail-closed lifecycle contracts."""
import unittest,json,tempfile
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from src.lava.jaes_v2.phase8_objectives import group_key,dro_update,bce,weighted_sum,clean_constraint
from src.lava.jaes_v2.phase8_training import ExperimentConfig,epoch_order,selection_value,seal_tree,verify_tree,source_path

ROOT=Path(__file__).resolve().parents[2]

class Phase8Contracts(unittest.TestCase):
    def config(self):
        return ExperimentConfig(**json.loads((ROOT/'configs/jaes_v2/phase8/p8_tiny_erm_head_s6107_v1.json').read_text()))

    def test_unknown_fields_fail(self):
        d=self.config().__dict__.copy();d['NovelLoss']=True
        with self.assertRaises(TypeError):ExperimentConfig(**d)

    def test_reserved_manifest_rejected(self):
        with self.assertRaises(ValueError):replace(self.config(),source_manifest='calibration_manifest.csv')

    def test_no_m3_or_detached_finetuning_claim(self):
        with self.assertRaises(ValueError):replace(self.config(),training_objective='M3')
        with self.assertRaises(ValueError):replace(self.config(),trainable_backbone_policy='FULL')

    def test_accumulation_contract(self):
        with self.assertRaises(ValueError):replace(self.config(),gradient_accumulation=3)

    def test_group_identity(self):
        self.assertEqual(group_key(0,'UNKNOWN','clean'),'0:REAL:clean')
        with self.assertRaises(ValueError):group_key(1,'UNKNOWN','clean')

    def test_dro_numeric_oracle_floor(self):
        q=np.array([.5,.5]);v=dro_update(q,[0.,2.],.1,.05)
        expected=.95*np.exp([0.,.2])/np.exp([0.,.2]).sum()+.025
        np.testing.assert_allclose(v,expected,rtol=1e-14)
        self.assertAlmostEqual(v.sum(),1.);self.assertGreaterEqual(v.min(),.025)
        with self.assertRaises(ValueError):dro_update(q,[np.nan,1],.1,.05)

    def test_dro_group_complete_sampler(self):
        c=replace(self.config(),training_objective='GROUP_DRO',sampling='group_balanced')
        rows=[SimpleNamespace(label=0,attack_id='UNKNOWN') for _ in range(6)]+[SimpleNamespace(label=1,attack_id=f'A{i:02}') for i in range(1,7)]
        order=epoch_order(rows,c,0)
        for k in range(0,len(order),c.batch_size):
            self.assertEqual(len({group_key(rows[i].label,rows[i].attack_id,'clean') for i in order[k:k+c.batch_size]}),7)
        self.assertEqual(order,epoch_order(rows,c,0))

    def test_natural_unique_sampling(self):
        rows=[object() for _ in range(13)];c=self.config()
        order=epoch_order(rows,c,0);self.assertEqual(sorted(order),list(range(13)))
        self.assertNotEqual(order,epoch_order(rows,c,1))

    def test_guardrail_rejects_and_missing_metrics_fail(self):
        c=replace(self.config(),checkpoint_policy='CLEAN_RISK_CONSTRAINED_ROBUSTNESS')
        clean=dict(condition='clean',stratum='A_PAIRED',REAL_false_alarm=.2,FAKE_miss=.1,nll=.2)
        self.assertIsNone(selection_value([clean],c)[0])
        clean['REAL_false_alarm']=0.
        with self.assertRaises(ValueError):selection_value([clean],c)

    def test_seal_corruption_and_unsafe_source(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'a').write_text('a');seal_tree(p);verify_tree(p)
            (p/'a').write_text('b')
            with self.assertRaises(ValueError):verify_tree(p)
        with self.assertRaises(ValueError):source_path(ROOT,'../outside.flac')

class Phase8Numerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse initialized TF from legacy research tests without changing runtime.
        import tensorflow as tf
        cls.tf=tf

    def test_erm_and_weighted_bce_oracle(self):
        y=np.array([0,1],dtype='float32');p=np.array([.2,.6],dtype='float32')
        expected=np.array([-np.log(.8),-np.log(.6)])
        np.testing.assert_allclose(bce(y,p).numpy(),expected,rtol=1e-6)
        self.assertAlmostEqual(float(weighted_sum(y,p,[.5,.75])),float(expected@[.5,.75]),places=6)

    def test_constraint_gradient_direction(self):
        tf=self.tf;p=tf.Variable([.8,.2],dtype=tf.float32)
        with tf.GradientTape() as tape:value=clean_constraint([0,1],p,[.1,.1],2.)
        g=tape.gradient(value,p).numpy();self.assertGreater(g[0],0);self.assertLess(g[1],0)
        self.assertEqual(float(clean_constraint([0,1],[.01,.99],[.1,.1],2.)),0.)

    def test_accumulation_matches_weighted_full_gradient(self):
        tf=self.tf;w=tf.Variable(.2);x=tf.constant([1.,2.,3.]);y=tf.constant([0.,1.,1.])
        with tf.GradientTape() as tape:loss=weighted_sum(y,tf.sigmoid(w*x),[1/3]*3)
        full=tape.gradient(loss,w).numpy();parts=[]
        for i in range(3):
            with tf.GradientTape() as tape:v=weighted_sum(y[i:i+1],tf.sigmoid(w*x[i:i+1]),[1/3])
            parts.append(tape.gradient(v,w).numpy())
        self.assertAlmostEqual(float(full),float(sum(parts)),places=7)

    def test_dro_loss_no_erm_double_count(self):
        y=[0,0,1];p=[.2,.4,.7];q=[.3,.7]
        coeff=[q[0]/2,q[0]/2,q[1]]
        expected=.3*(-np.log(.8)-np.log(.6))/2+.7*(-np.log(.7))
        self.assertAlmostEqual(float(weighted_sum(y,p,coeff)),expected,places=6)

    def test_classwise_denominators_threshold_and_null(self):
        from src.lava.jaes_v2.phase7_training import score_metrics
        m=score_metrics([dict(label=0,raw_score=.5),dict(label=0,raw_score=.1),dict(label=1,raw_score=.4)])
        self.assertEqual(m['REAL_false_alarm'],.5);self.assertEqual(m['FAKE_miss'],1.)
        one=score_metrics([dict(label=1,raw_score=.8)])
        self.assertIsNone(one['REAL_false_alarm']);self.assertIsNone(one['roc_auc'])

    def test_checkpoint_optimizer_roundtrip_and_no_overwrite(self):
        from src.lava.jaes_v2.phase8_training import save_checkpoint,restore_checkpoint,variable_hash
        tf=self.tf
        model=tf.keras.Sequential([tf.keras.layers.Input((2,)),tf.keras.layers.Dense(1)])
        opt=tf.keras.optimizers.Adam(.01);opt.build(model.trainable_weights)
        with tf.GradientTape() as tape:value=tf.reduce_sum(model(tf.ones((1,2)))**2)
        opt.apply_gradients(zip(tape.gradient(value,model.trainable_weights),model.trainable_weights))
        expected=(variable_hash(model.weights),variable_hash(opt.variables()))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint'
            state=dict(contract={'data':'sealed'},step=1)
            save_checkpoint(tf,model,opt,path,state)
            with self.assertRaises(FileExistsError):save_checkpoint(tf,model,opt,path,state)
            model.weights[0].assign(tf.zeros_like(model.weights[0]));opt.iterations.assign(99)
            with self.assertRaises(ValueError):restore_checkpoint(tf,model,opt,path,{'data':'wrong'})
            restored=restore_checkpoint(tf,model,opt,path,{'data':'sealed'})
            self.assertEqual(restored,state)
            self.assertEqual((variable_hash(model.weights),variable_hash(opt.variables())),expected)

    def test_atomic_checkpoint_failure_does_not_publish(self):
        from unittest.mock import patch
        from src.lava.jaes_v2.phase8_training import save_checkpoint
        tf=self.tf;model=tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(1)])
        opt=tf.keras.optimizers.Adam(.01);opt.build(model.trainable_weights)
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'commit'
            with patch.object(tf.train.Checkpoint,'write',side_effect=RuntimeError('power loss')):
                with self.assertRaises(RuntimeError):save_checkpoint(tf,model,opt,target,{})
            self.assertFalse(target.exists())

    def test_tiny_overfit_with_real_optimizer(self):
        tf=self.tf;tf.keras.utils.set_random_seed(8123)
        model=tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(1,activation='sigmoid')])
        opt=tf.keras.optimizers.SGD(.5);x=tf.constant([[-1.],[1.]]);y=tf.constant([0.,1.])
        start=float(tf.reduce_mean(bce(y,model(x))))
        for _ in range(40):
            with tf.GradientTape() as tape:loss=tf.reduce_mean(bce(y,model(x)))
            opt.apply_gradients(zip(tape.gradient(loss,model.trainable_weights),model.trainable_weights))
        self.assertLess(float(tf.reduce_mean(bce(y,model(x)))),start*.2)

if __name__=='__main__':unittest.main()
