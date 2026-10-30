"""Phase6 scientific controls, actual reference parity, calibration and leakage contracts."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
import numpy as np
from src.lava.jaes_v2.phase6_data import speaker_select
from src.lava.jaes_v2.rawboost import rawboost,reference,PARAMETERS
from src.lava.jaes_v2.phase6_training import Config,paired_objective,reliability_metrics
from src.lava.jaes_v2.c2_student import initialize_tensorflow


class SelectionTests(unittest.TestCase):
    def test_concurrent_heavy_jobs_rejected(self):
        from scripts.jaes_v2.phase6_experiment import resource_lock
        with tempfile.TemporaryDirectory() as name:
            with resource_lock(Path(name)):
                with self.assertRaises((RuntimeError,BlockingIOError)):
                    with resource_lock(Path(name)): pass
    def fixture(self):
        return [SimpleNamespace(sample_id=f'{s}-{y}-{i}',split='train',speaker_id=str(s),label=y,attack_id='A0'+str(i%3+1) if y else '-')
                for s in range(4) for y in (0,1) for i in range(8)]
    def test_balanced_cell_determinism(self):
        a,_=speaker_select(self.fixture(),per_cell=4,seed=5806,role='train'); b,_=speaker_select(list(reversed(self.fixture())),per_cell=4,seed=5806,role='train')
        self.assertEqual([r.sample_id for r in a],[r.sample_id for r in b]); self.assertEqual(len(a),32)
        for s in range(4): self.assertEqual(sum(r.label==1 and r.speaker_id==str(s) for r in a),4)
    def test_unavailable_cells_excluded_not_fabricated(self):
        rows=[r for r in self.fixture() if not (r.speaker_id=='0' and r.label==1)]
        selected,coverage=speaker_select(rows,per_cell=4,seed=5806,role='train')
        self.assertEqual(len(selected),24); self.assertEqual(len(coverage),4)
        with self.assertRaises(ValueError): speaker_select(rows,per_cell=9,seed=5806,role='train')
    def test_reserved_roles_blocked(self):
        with self.assertRaises(ValueError): speaker_select(self.fixture(),per_cell=2,seed=1,role='calibration')
        with self.assertRaises(ValueError): speaker_select(self.fixture(),per_cell=2,seed=1,role='development')


class RawBoostTests(unittest.TestCase):
    def test_exact_algorithm5_reference_and_rng_isolation(self):
        x=np.random.default_rng(7).normal(0,.05,3200); state=np.random.get_state()
        np.random.seed(42); expected=reference.ISD_additive_noise(reference.LnL_convolutive_noise(x,**PARAMETERS,fs=22050),10,2)
        np.random.set_state(state); actual=rawboost(x,seed=42)
        np.testing.assert_array_equal(actual,expected.astype('float32'))
        after=np.random.get_state(); self.assertEqual(state[0],after[0]); np.testing.assert_array_equal(state[1],after[1]); self.assertEqual(state[2:],after[2:])
    def test_algorithms_length_finite_seed_and_silence(self):
        x=np.random.default_rng(8).normal(0,.3,3200)
        for algorithm in range(6):
            a=rawboost(x,seed=2,algorithm=algorithm); b=rawboost(x,seed=2,algorithm=algorithm)
            np.testing.assert_array_equal(a,b); self.assertEqual(a.shape,x.shape); self.assertTrue(np.isfinite(a).all())
            np.testing.assert_array_equal(rawboost(np.zeros(3200),algorithm=algorithm),np.zeros(3200))
        self.assertFalse(np.array_equal(rawboost(x,seed=1),rawboost(x,seed=2)))
        with self.assertRaises(ValueError): rawboost(x,sample_rate=8000)
    def test_impulsive_can_change_both_polarities_and_bounds(self):
        x=np.sin(np.arange(10000)*.1)*.1; y=rawboost(x,seed=3,algorithm=2); delta=y-x
        self.assertTrue(np.any(delta>0)); self.assertTrue(np.any(delta<0)); self.assertLessEqual(float(abs(y).max()),1.000001)


class ObjectiveTests(unittest.TestCase):
    def test_master_report_rejects_stale_metrics(self):
        from scripts.jaes_v2.build_master_research_report import verify_metric_rows
        rows=[dict(label='0',raw_score='.2',predicted_label='0',condition_id='clean'),dict(label='1',raw_score='.8',predicted_label='1',condition_id='clean')]
        verify_metric_rows(rows,[dict(condition='clean',error_rate=0.,n=2)])
        with self.assertRaises(ValueError): verify_metric_rows(rows,[dict(condition='clean',error_rate=.5,n=2)])
        with self.assertRaises(ValueError): verify_metric_rows(rows,[dict(condition='missing',error_rate=0.)])
    def test_equal_view_control_isolates_mse(self):
        tf=initialize_tensorflow(); p=tf.Variable([[.01],[.99]]); q=tf.constant([[.99],[.01]]); y=tf.constant([0.,1.])
        control=paired_objective(y,p,q); regularized=paired_objective(y,p,q,consistency=True,weight=.1)
        self.assertAlmostEqual(float(regularized-control),.1*.98**2,places=6)
        self.assertAlmostEqual(float(paired_objective(y,p,p,consistency=True)),float(paired_objective(y,p)),places=6)
        with tf.GradientTape() as tape: value=paired_objective(y,p,q,consistency=True)
        self.assertGreater(float(tf.linalg.norm(tape.gradient(value,p))),0)
    def test_single_class_extreme_logits_finite(self):
        tf=initialize_tensorflow()
        for y in ([0.,0.],[1.,1.]):
            with tf.GradientTape() as tape:
                logits=tf.Variable([[-100.],[100.]]); p=tf.sigmoid(logits); value=paired_objective(y,p,1-p,consistency=True)
            self.assertTrue(np.isfinite(value.numpy()))
    def test_confirmation_and_finetune_config_cannot_bypass_gate(self):
        for values in ({'scope':'CONFIRMATORY_EVALUATION'},{'training_policy':'PARTIAL_BACKBONE_FINETUNE'},{'epochs':100},{'maximum_seconds':3600}):
            with self.assertRaises(ValueError): Config('p6_fixture',**values)
    def test_reliability_semantics(self):
        result=reliability_metrics([0,1],[.25,.75],bins=4)
        self.assertAlmostEqual(result['brier'],.0625); self.assertAlmostEqual(result['nll'],-np.log(.75)); self.assertAlmostEqual(result['ece'],.25)
        self.assertIsNone(reliability_metrics([0,0],[0.,1.])['classwise']['1']['mean_score'])
        with self.assertRaises(ValueError): reliability_metrics([0],[2.])


if __name__=='__main__': unittest.main()
