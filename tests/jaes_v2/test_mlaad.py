"""Author-format fixtures exercise real import gates; never benchmark evidence."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from src.lava.jaes_v2.datasets import load_public_manifest
from src.lava.jaes_v2.evidence import file_hash, object_hash, write_json
from src.lava.jaes_v2.mlaad import import_mlaad, metadata_inventory, confounding_audit
from src.lava.jaes_v2.provenance import audit_partitions


class MLAADTests(unittest.TestCase):
    def fixture(self,root):
        discovery, audio = root/'discovery',root/'audio'
        discovery.mkdir(); audio.mkdir()
        meta = discovery/'fake/en/System/meta.csv'
        meta.parent.mkdir(parents=True)
        meta.write_text('path|original_file|language|model_name|architecture\n'
                        './fake/en/System/a.wav|en_US/by_book/female/alice/book/wavs/a.wav|en|AuthorSystem|Family\n')
        inventory = []
        files = []
        for index,path in enumerate(('fake/en/System/a.wav','original/en/b.wav')):
            target = audio/path
            target.parent.mkdir(parents=True,exist_ok=True)
            sf.write(target,np.random.default_rng(index).normal(0,.01,1600),16000)
            digest = file_hash(target)
            inventory.append(dict(path=path,size=target.stat().st_size,sha256=digest,git_blob='UNKNOWN'))
            files.append(dict(path=path,checksum=digest))
        write_json(discovery/'inventory.json',inventory)
        release = dict(repository='mueller91/MLAAD-tiny',revision='a'*40,inventory_sha256=file_hash(discovery/'inventory.json'),
                       files=[dict(path='fake/en/System/meta.csv',checksum=file_hash(meta))])
        write_json(discovery/'discovery.json',release)
        records,summary = metadata_inventory(discovery)
        selection = dict(scope='PUBLIC_DEVELOPMENT',revision='a'*40,records=records)
        acquisition = dict(status='ACQUIRED_VERIFIED',repository=release['repository'],revision='a'*40,
                           selection_hash=object_hash(selection),files=files,license={'fixture':True})
        write_json(audio/'acquisition.json',acquisition)
        return discovery,audio,selection

    def test_author_generator_mapping_not_folder_and_unknown_voice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            discovery,audio,selection = self.fixture(root)
            with patch('src.lava.jaes_v2.mlaad.license_gate',return_value={}):
                result = import_mlaad(discovery,audio,selection,root/'import')
            records = load_public_manifest(root/'import/provenance_manifest.csv')
            fake = next(row for row in records if row.label)
            real = next(row for row in records if not row.label)
            self.assertEqual(fake.generator_id,'AuthorSystem')
            self.assertEqual(fake.generator_family,'Family')
            self.assertEqual(fake.speaker_id,'UNKNOWN')
            self.assertEqual(fake.target_speaker_id,'UNKNOWN')
            self.assertEqual(real.source_speaker_id,'UNKNOWN')
            self.assertEqual(fake.official_split,'UNKNOWN')
            self.assertFalse(result['partition']['global_parent_lineage_verified'])
            with self.assertRaises(ValueError):
                audit_partitions(records,require_speaker_disjoint=True)
            self.assertFalse(confounding_audit(records)['language_label_complete_separation'])

    def test_missing_audio_and_modified_author_annotation_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            discovery,audio,selection = self.fixture(root)
            selection['records'][0]['label']=1-selection['records'][0]['label']
            # Even a maliciously updated acquisition selection hash cannot relabel author data.
            acq = json.loads((audio/'acquisition.json').read_text())
            acq['selection_hash']=object_hash(selection)
            write_json(audio/'acquisition.json',acq)
            with patch('src.lava.jaes_v2.mlaad.license_gate',return_value={}):
                with self.assertRaisesRegex(ValueError,'annotation'):
                    import_mlaad(discovery,audio,selection,root/'bad')
            self.assertFalse((root/'bad/provenance_manifest.csv').exists())
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            discovery,audio,selection=self.fixture(root)
            (audio/selection['records'][0]['path']).unlink()
            with patch('src.lava.jaes_v2.mlaad.license_gate',return_value={}):
                with self.assertRaises(FileNotFoundError):
                    import_mlaad(discovery,audio,selection,root/'missing')

    def test_metadata_revision_and_audio_checksum_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            discovery,audio,selection=self.fixture(root)
            (audio/selection['records'][0]['path']).write_bytes(b'corrupted')
            with patch('src.lava.jaes_v2.mlaad.license_gate',return_value={}):
                with self.assertRaisesRegex(ValueError,'checksum'):
                    import_mlaad(discovery,audio,selection,root/'corrupt')


if __name__=='__main__':
    unittest.main()
