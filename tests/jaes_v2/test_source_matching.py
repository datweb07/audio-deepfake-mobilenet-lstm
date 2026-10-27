from pathlib import Path
import unittest

from src.lava.jaes_v2.source_matching import relationship,MATCHING_VERSION
from src.lava.jaes_v2.matched_analysis import select_matched,matched_interaction


def score_fixture():
    rows=[]
    for i,cluster in ((0,'s0'),(1,'s0'),(2,'s1')):
        for system in ('G0','G1'):
            for condition in ('AB','BA'):
                error=i==0 and system=='G0' and condition=='AB'
                rows.append(dict(sample_id=f'{i}-{system}',scope='PUBLIC_DEVELOPMENT',source_recording_id=f'original/{i}.wav',
                    source_speaker_id=cluster,generator_id=system,matching_level=3,condition_id=condition,label=1,
                    status='OK',raw_score=.1 if error else .9,threshold=.5,predicted_label=int(not error),
                    language='en',source_corpus='source',model_id='fixture',seed=0))
    return rows


class SourceMatchingTests(unittest.TestCase):
    def test_full_author_id_and_text_does_not_become_level4(self):
        a=dict(source_recording_id='speaker/book/a.wav',transcript='Same  text',source_speaker_id='s0')
        b={**a,'transcript':'Same text'}
        result=relationship(a,b)
        self.assertEqual(result['matching_level'],3)
        self.assertEqual(result['verification_status'],'AUTHOR_ID_MATCH')

    def test_basename_collision_is_not_a_match(self):
        result=relationship(dict(source_recording_id='speakerA/book/a.wav'),dict(source_recording_id='speakerB/other/a.wav'))
        self.assertEqual(result['verification_status'],'AMBIGUOUS_SOURCE_MATCH')
        self.assertEqual(result['matching_level'],0)

    def test_unknown_and_speaker_only_and_text_only_are_distinct(self):
        self.assertEqual(relationship({}, {})['matching_level'],0)
        self.assertEqual(relationship(dict(source_speaker_id='s'),dict(source_speaker_id='s'))['matching_level'],1)
        self.assertEqual(relationship(dict(transcript='Hello'),dict(transcript='Hello'))['matching_level'],2)

    def test_conflicting_author_text_rejected(self):
        result=relationship(dict(source_recording_id='a',transcript='Yes'),dict(source_recording_id='a',transcript='No'))
        self.assertEqual(result['verification_status'],'CONFLICTING_AUTHOR_TEXT')

    def test_deterministic_balanced_matched_selection_and_minimum_support(self):
        rows=[]
        for i in range(12):
            for system in ('A','B'):
                rows.append(dict(source_recording_id=f'p{i}',source_speaker_id=f's{i%6}',generator_id=system,
                                 language='en',matching_level=3,verification_status='AUTHOR_ID_MATCH'))
        result=select_matched(rows,systems=('A','B'))
        self.assertEqual(result,select_matched(list(reversed(rows)),systems=('A','B')))
        self.assertEqual(len(result['records']),16)
        self.assertEqual(select_matched(rows[:6],systems=('A','B'))['status'],'INSUFFICIENT_SOURCE_MATCHES')
        for row in rows:
            row['source_speaker_id']='s0'
        self.assertEqual(select_matched(rows,systems=('A','B'))['status'],'INSUFFICIENT_INDEPENDENT_GROUPS')

    def test_hand_computed_matched_interaction_with_unequal_cluster_sizes(self):
        result=matched_interaction(score_fixture(),first_system='G0',second_system='G1',replicates=100)
        self.assertEqual(result['estimate'],.25)
        self.assertEqual(result['parents'],2)
        self.assertEqual(result['source_recordings'],3)
        self.assertFalse(result['causal_interpretation'])
        self.assertEqual(result,matched_interaction(score_fixture(),first_system='G0',second_system='G1',replicates=100))

    def test_missing_cells_unknown_source_and_mismatched_record_rejected(self):
        self.assertEqual(matched_interaction(score_fixture()[:-1],first_system='G0',second_system='G1')['status'],'MISSING_FACTORIAL_CELL')
        rows=score_fixture(); rows[0]['matching_level']=1
        with self.assertRaises(ValueError):
            matched_interaction(rows,first_system='G0',second_system='G1')
        rows=score_fixture(); rows[0]['scope']='external_confirmation'
        with self.assertRaises(ValueError):
            matched_interaction(rows,first_system='G0',second_system='G1')

    def test_language_or_corpus_confounding_not_estimated_as_zero(self):
        for field in ('language','source_corpus'):
            rows=score_fixture()
            for row in rows:
                row[field]=row['generator_id']
            result=matched_interaction(rows,first_system='G0',second_system='G1')
            self.assertIsNone(result['estimate'])
            self.assertTrue(result['status'].startswith('CONFOUNDED_FACTOR'))


if __name__=='__main__':
    unittest.main()
