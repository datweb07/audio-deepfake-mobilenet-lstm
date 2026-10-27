from dataclasses import asdict
import unittest
import numpy as np

from src.lava.jaes_v2.transforms import Audio
from src.lava.jaes_v2.nuisance_controls import POLICIES,intervene,signal_features
from src.lava.jaes_v2.precision import plan_precision
from src.lava.jaes_v2.phase4 import NuisanceConfig
from src.lava.jaes_v2.phase4_analysis import summarize,NEW_CHANNELS
from src.lava.jaes_v2.public_analysis import PUBLIC_CONDITIONS


class NuisanceTests(unittest.TestCase):
    def test_edge_repack_retains_content_exactly_and_length(self):
        x=np.r_[np.zeros(20),np.linspace(.01,.1,60),np.zeros(20)].astype('float32')
        result,evidence=intervene(Audio(x,100),'edge_repack_v1')
        np.testing.assert_array_equal(result.samples[:60],x[20:80])
        np.testing.assert_array_equal(result.samples[60:],0)
        self.assertEqual(len(result.samples),len(x))
        self.assertEqual(evidence['start_sample'],20)
        np.testing.assert_array_equal(x[:20],0)

    def test_amplitude_rms_cap_no_post_channel_normalization(self):
        audio=Audio(np.array([.1,-.2]*100,dtype='float32'),22050)
        result,evidence=intervene(audio,'global_half_v1')
        np.testing.assert_array_equal(result.samples,audio.samples*.5)
        self.assertFalse(evidence['post_channel_normalization'])
        result,evidence=intervene(audio,'rms_anchor_v1')
        self.assertAlmostEqual(np.sqrt(np.mean(result.samples**2)),.03,places=6)
        self.assertLessEqual(np.max(np.abs(result.samples)),.25)

    def test_invalid_silence_policy_and_measurement_names(self):
        audio=Audio(np.zeros(100),100)
        for policy in ('edge_repack_v1','rms_anchor_v1','invented'):
            with self.assertRaises(ValueError):
                intervene(audio,policy)
        features=signal_features(audio)
        self.assertEqual(features['silence_fraction_below_1e-4'],1)
        self.assertIsNone(features['rolloff95_hz'])

    def test_first_three_seconds_and_production_segmentation_contract(self):
        import config
        from src.preprocessing import normalize_audio_length,segment_audio
        audio=np.arange(80000,dtype='float32')
        np.testing.assert_array_equal(normalize_audio_length(audio),audio[:66150])
        self.assertEqual(segment_audio(audio).shape,(6,11025))
        self.assertEqual(config.SAMPLE_RATE,22050)

    def test_precision_uses_cluster_means_not_utterance_count(self):
        report=plan_precision({'s0':[0]*100,'s1':[1]})
        self.assertEqual(report['equal_cluster_mean'],.5)
        self.assertEqual(report['clusters_observed'],2)
        self.assertEqual(report['recordings_observed'],101)
        self.assertEqual(report['status'],'INSUFFICIENT_INDEPENDENT_GROUPS')
        zero=plan_precision({'s0':[0],'s1':[0]})
        self.assertIsNone(zero['required_clusters_at_pilot_variance'])

    def test_confirmation_and_unsealed_config_rejected(self):
        kwargs=dict(experiment_id='fixture',selection_path='x',selection_sha256='a'*64,phase3_cache='x',
                    phase3_cache_metadata_sha256='b'*64,source_matching_report='x',source_matching_sha256='c'*64)
        with self.assertRaises(ValueError):
            NuisanceConfig(**kwargs,scope='CONFIRMATORY')
        with self.assertRaises(ValueError):
            NuisanceConfig(**{**kwargs,'selection_sha256':''})

    def fixture(self):
        manifest=[];rows=[]
        for label in (0,1):
            for i in range(4):
                source=dict(sample_id=f'{label}-{i}',label=label,generator_id='A' if label else 'UNKNOWN',
                    parent_recording_id=f'p{label}{i}',source_speaker_id=f's{i%2}' if label else 'UNKNOWN',dataset_id='fixture',split='development')
                manifest.append(source)
                for policy in ('phase3_baseline',)+POLICIES:
                    for condition in PUBLIC_CONDITIONS if policy=='phase3_baseline' else NEW_CHANNELS:
                        error=policy=='edge_repack_v1' and condition=='fixed_A_then_B' and label==1
                        prediction=1-label if error else label
                        rows.append({**source,'policy':policy,'condition_id':condition,'threshold':.5,'raw_score':.9 if prediction else .1,
                                     'predicted_label':prediction,'model_id':'fixture','status':'OK'})
        return manifest,rows

    def test_paired_sensitivity_and_single_class_risks(self):
        manifest,rows=self.fixture()
        result=summarize(rows,manifest,replicates=100)
        primary=next(row for row in result['contrasts'] if row['primary'])
        self.assertEqual(primary['estimate'],1)
        self.assertEqual(primary['parents'],2)
        self.assertEqual(result['matched_generator_interaction'],'SOURCE_MATCHED_EXPERIMENT_BLOCKED')

    def test_missing_failed_or_changed_labels_not_silently_dropped(self):
        manifest,rows=self.fixture()
        for changed in (rows[:-1],[{**rows[0],'label':1}]+rows[1:],[{**rows[0],'status':'FAILED'}]+rows[1:]):
            with self.assertRaises(ValueError):
                summarize(changed,manifest,replicates=100)


if __name__=='__main__':
    unittest.main()
