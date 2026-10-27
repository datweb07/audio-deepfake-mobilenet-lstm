import unittest
from src.lava.jaes_v2.confirmation_gate import assess_confirmation


class ConfirmationGateTests(unittest.TestCase):
    def test_incomplete_confirmation_cannot_be_locked(self):
        report=assess_confirmation({'confirmation_labels_opened':False},development_ids={'dev'})
        self.assertEqual(report['status'],'CONFIRMATION_NOT_LOCKED')
        self.assertFalse(report['confirmation_execution_enabled'])
        self.assertFalse(report['labels_read'])

    def test_shared_development_ids_and_missing_source_evidence_reject(self):
        draft=dict(validation_kind='A_SOURCE_MATCHED_HYPOTHESIS',sample_ids=['dev'],matching_level=1,
                   verified_independent_clusters=4,confirmation_labels_opened=False)
        result=assess_confirmation(draft,development_ids={'dev'})
        self.assertIn('development_confirmation_disjointness',result['missing_requirements'])
        self.assertIn('authoritative_source_content_support',result['missing_requirements'])
        self.assertIn('sufficient_verified_independent_clusters',result['missing_requirements'])

    def test_transfer_does_not_imply_matched_hypothesis_confirmation(self):
        result=assess_confirmation(dict(validation_kind='B_CROSS_CORPUS_TRANSFER',confirmation_labels_opened=False),development_ids=set())
        self.assertEqual(result['validation_kind'],'B_CROSS_CORPUS_TRANSFER')
        self.assertNotIn('authoritative_source_content_support',result['missing_requirements'])
        self.assertEqual(result['status'],'CONFIRMATION_NOT_LOCKED')


if __name__=='__main__':
    unittest.main()
