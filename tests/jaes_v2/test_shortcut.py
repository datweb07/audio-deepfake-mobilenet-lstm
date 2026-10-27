import unittest

from src.lava.jaes_v2.shortcut import available_source_clusters, grouped_shortcuts


class ShortcutTests(unittest.TestCase):
    def test_shared_parent_and_source_speaker_form_connected_components(self):
        rows=[dict(parent_recording_id='p0',source_speaker_id='s0'),
              dict(parent_recording_id='p1',source_speaker_id='s0'),
              dict(parent_recording_id='p1',source_speaker_id='UNKNOWN'),
              dict(parent_recording_id='p2',source_speaker_id='UNKNOWN')]
        groups=available_source_clusters(rows)
        self.assertEqual(len(set(groups[:3])),1)
        self.assertNotEqual(groups[2],groups[3])

    def test_constant_feature_is_not_a_zero_accuracy_discovery(self):
        manifest=[dict(sample_id=str(i),label=i%2,split='development',parent_recording_id=str(i),source_speaker_id='UNKNOWN') for i in range(32)]
        features=[{'sample_id':str(i),'duration':i%2+1,'sample_rate':16000,'silence_fraction_below_1e-4':.1,'rms':.1,'peak':.2} for i in range(32)]
        report=grouped_shortcuts(features,manifest)
        constant=next(row for row in report['results'] if row['baseline']=='sample_rate_only')
        self.assertIsNone(constant['accuracy'])
        self.assertEqual(constant['status'],'NONINFORMATIVE_CONSTANT_FEATURE')
        self.assertEqual(next(row for row in report['results'] if row['baseline']=='duration_only')['accuracy'],1)
        self.assertEqual(report,grouped_shortcuts(features,manifest))

    def test_test_or_calibration_labels_cannot_fit_shortcuts(self):
        for split in ('test','calibration','external_confirmation'):
            with self.assertRaises(ValueError):
                grouped_shortcuts([], [dict(split=split)])


if __name__=='__main__':
    unittest.main()
