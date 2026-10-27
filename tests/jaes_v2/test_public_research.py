from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from src.lava.jaes_v2.datasets import PublicSourceRecord
from src.lava.jaes_v2.mlaad import select_development, source_speaker, confounding_audit
from src.lava.jaes_v2.phase3 import PublicConfig, verify_seal
from src.lava.jaes_v2.public_analysis import PUBLIC_CONDITIONS, metrics, summarize_public, validate_public_scores
from src.lava.jaes_v2.evidence import file_hash
from src.lava.jaes_v2.provenance import audit_partitions
from src.lava.jaes_v2.factorial import generator_channel_interaction


def candidates():
    rows = []
    for group in ('B','A'):
        for i in range(16):
            rows.append(dict(sample_id=f'{group}{i}',path=f'{group}/{i}.wav',label=1,language='en',generator_id=group,
                             source_speaker_id=f'speaker{i%4}',annotation_status='AUTHOR_MODEL_NAME'))
    for i in range(40):
        rows.append(dict(sample_id=f'real{i}',path=f'real/{i}.wav',label=0,language='en',generator_id='UNKNOWN',
                         source_speaker_id='UNKNOWN',annotation_status='AUDIOFOLDER_CLASS_ONLY'))
    return rows


def scores_fixture():
    manifest, scores = [], []
    for label in (0,1):
        for i in range(8):
            source = dict(sample_id=f'{label}-{i}',label=label,dataset_id='fixture',parent_recording_id=f'parent-{label}-{i}',
                          generator_id='A' if i<4 else 'B',language='en',source_corpus='fixture-original',
                          source_speaker_id=f'speaker{i%4}' if label else 'UNKNOWN',parent_scope='LOCAL_BYTES_ONLY')
            manifest.append(source)
            for condition in PUBLIC_CONDITIONS:
                predicted = label if condition != 'fixed_A_then_B' else 0
                scores.append({**source, 'model_id':'fixture','run_id':'fixture','threshold':.5,'seed':1,'status':'OK',
                               'condition_id':condition,'raw_score':.9 if predicted else .1,'predicted_label':predicted})
    return manifest,scores


class PublicResearchTests(unittest.TestCase):
    def test_selection_deterministic_balanced_without_performance_or_test_input(self):
        rows = candidates()
        first = select_development(rows,revision='a'*40,per_group=8,group_count=2,language='en',seed=3804)
        self.assertEqual(first,select_development(list(reversed(rows)),revision='a'*40,per_group=8,group_count=2,language='en',seed=3804))
        self.assertEqual(first['groups'],['A','B'])
        self.assertEqual(sum(row['label']==0 for row in first['records']),16)
        self.assertEqual(first['scope'],'PUBLIC_DEVELOPMENT')

    def test_selection_insufficient_group_language_or_class_rejected(self):
        for rows in (candidates()[:16],[row for row in candidates() if row['label']],
                     [{**row,'language':'de' if row['label'] else 'en'} for row in candidates()]):
            with self.assertRaises(ValueError):
                select_development(rows,revision='a'*40,per_group=8,group_count=2,language='en',seed=1)

    def test_source_speaker_does_not_guess_synthesized_voice(self):
        self.assertEqual(source_speaker('en_US/by_book/female/alice/book/wavs/a.wav'),'M-AILABS:en_US/by_book/female/alice')
        self.assertEqual(source_speaker('book_a.wav'),'UNKNOWN')

    def test_classwise_denominators_do_not_invent_auc_eer_f1(self):
        _,rows = scores_fixture()
        result = metrics([row for row in rows if row['label']==1 and row['condition_id']=='clean'])
        self.assertEqual(result['miss_rate'],0)
        for key in ('auc','eer','f1','false_alarm_rate'):
            self.assertIsNone(result[key])

    def test_paired_analysis_unknown_cluster_and_reproducible_interactions(self):
        manifest,rows = scores_fixture()
        result = summarize_public(rows,manifest,replicates=100,seed=1)
        self.assertEqual(result,summarize_public(rows,manifest,replicates=100,seed=1))
        effects = [row for row in result['order_contrasts'] if row['group']=='A' and row.get('risk')=='error']
        self.assertEqual(effects[0]['estimate'],1)
        self.assertTrue(any(row['status']=='NOT_IDENTIFIABLE_UNKNOWN_CLUSTER' for row in result['order_contrasts']))
        self.assertEqual(result['independent_population_units'],'UNKNOWN')

    def test_missing_pair_duplicate_or_changed_label_is_rejected(self):
        manifest,rows = scores_fixture()
        for changed in (rows[:-1],rows+[rows[0]],[{**rows[0],'label':1}]+rows[1:]):
            with self.assertRaises(ValueError):
                validate_public_scores(changed,manifest)

    def test_language_and_source_corpus_factor_confounding(self):
        from test_factorial import fixture
        for field in ('language','source_corpus'):
            rows = [{**row,field:row['factor_group']} for row in fixture()]
            with self.assertRaisesRegex(ValueError,'CONFOUNDED_FACTOR'):
                generator_channel_interaction(rows,channel='codec',factor_definition='fixture',class_label=1,replicates=100)

    def test_confirmation_and_unsealed_config_blocked(self):
        kwargs = dict(experiment_id='fixture',manifest='m.csv',manifest_sha256='a'*64,selection_sha256='b'*64)
        with self.assertRaises(ValueError):
            PublicConfig(**kwargs,scope='CONFIRMATORY')
        with self.assertRaises(ValueError):
            PublicConfig(**{**kwargs,'manifest_sha256':''})

    def test_seal_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'audio.wav').write_bytes(b'fixture')
            meta = {'output_hashes':{'audio.wav':file_hash(root/'audio.wav')}}
            self.assertEqual(verify_seal(root,meta),1)
            (root/'audio.wav').write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                verify_seal(root,meta)


if __name__ == '__main__':
    unittest.main()
