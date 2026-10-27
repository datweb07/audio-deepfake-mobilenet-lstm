import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.lava.jaes_v2.acquisition import (RemoteFile, acquire_file, acquire_selection,
                                         license_gate, pinned_revision, safe_relative, REPOSITORY)
from src.lava.jaes_v2.evidence import file_hash, write_json

REVISION = 'a'*40


class AcquisitionTests(unittest.TestCase):
    def test_revision_pinning_and_path_escape(self):
        for value in ('main','abc','A'*40,'../main'):
            with self.assertRaises(ValueError):
                pinned_revision(value)
        for value in ('../audio.wav','/audio.wav','C:/x.wav','x\\y.wav'):
            with self.assertRaises(ValueError):
                safe_relative(value)
        self.assertEqual(pinned_revision(REVISION),REVISION)

    def test_download_hash_size_revision_and_attribution(self):
        payload = b'real fixture file content'
        item = RemoteFile('original/en/a.wav',len(payload),hashlib.sha256(payload).hexdigest())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = acquire_file(item,repository=REPOSITORY,revision=REVISION,destination=root,fetch=lambda url,size:payload)
            self.assertEqual(result['status'],'VERIFIED')
            self.assertIn(REVISION,result['url'])
            self.assertEqual(result['repository'],REPOSITORY)
            self.assertEqual(result['checksum'],file_hash(root/item.path))
            with self.assertRaises(FileExistsError):
                acquire_file(item,repository=REPOSITORY,revision=REVISION,destination=root,fetch=lambda u,s:payload)

    def test_partial_and_checksum_mismatch_never_publish_audio(self):
        item = RemoteFile('a.wav',5,hashlib.sha256(b'abcde').hexdigest())
        for payload in (b'ab',b'wrong'):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.assertRaises(ValueError):
                    acquire_file(item,repository=REPOSITORY,revision=REVISION,destination=root,fetch=lambda u,s:payload)
                self.assertFalse((root/'a.wav').exists())
                self.assertTrue((root/'a.wav.part').exists())

    def test_audio_without_remote_checksum_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError,'SHA-256'):
                acquire_file(RemoteFile('a.wav',10),repository=REPOSITORY,revision=REVISION,destination=Path(temporary))

    def test_license_and_noncommercial_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                license_gate(root,purpose='noncommercial_research')
            (root/'original').mkdir()
            for path,text in [('LICENSE','CC BY-NC 4.0'),('original/LICENSE','M-AILABS'),
                              ('README.md','fixture'),('NOTICE_NO_COMMERCIAL_USE.txt','No commercial use')]:
                (root/path).write_text(text)
            self.assertEqual(license_gate(root,purpose='noncommercial_research')['commercial_permission'],'NOT_GRANTED')
            with self.assertRaises(ValueError):
                license_gate(root,purpose='commercial')

    def test_selection_budget_blocks_network_and_no_full_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root/'inventory.json',[dict(path='a.wav',size=100,sha256='b'*64,git_blob='UNKNOWN')])
            write_json(root/'discovery.json',dict(revision=REVISION,inventory_sha256=file_hash(root/'inventory.json')))
            selection = dict(revision=REVISION,scope='PUBLIC_DEVELOPMENT',records=[dict(path='a.wav')])
            with patch('src.lava.jaes_v2.acquisition.license_gate',return_value={}), \
                    patch('src.lava.jaes_v2.acquisition.acquire_file') as download:
                with self.assertRaisesRegex(ValueError,'budget'):
                    acquire_selection(root,selection,root/'audio',maximum_bytes=50,purpose='noncommercial_research')
                download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
