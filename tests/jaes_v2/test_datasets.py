from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from src.lava.jaes_v2.datasets import ASVspoof2019LA, import_public, load_public_manifest
from src.lava.jaes_v2.overlap import near_duplicate_candidates, overlap_dimensions
from src.lava.jaes_v2.provenance import audit_partitions
from src.lava.jaes_v2.transforms import Audio


class PublicDatasetTests(unittest.TestCase):
    def fixture(self, root):
        protocol = root / "official_format_fixture.txt"
        protocol.write_text("LA_0001 LA_D_1000001 - - bonafide\nLA_0002 LA_D_1000002 - A01 spoof\n")
        for index in (1, 2):
            path = root / "ASVspoof2019_LA_dev/flac" / f"LA_D_100000{index}.flac"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, np.random.default_rng(index).normal(0, .05, 16000), 16000, subtype="PCM_16")
        return protocol

    def test_official_mapping_provenance_and_attack_not_generator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = self.fixture(root)
            result = import_public(ASVspoof2019LA(), dataset_root=root, protocol=protocol, official_split="dev", output=root / "import")
            records = load_public_manifest(root / "import/provenance_manifest.csv")
            self.assertEqual(result["status"], "IMPORTED")
            self.assertEqual([row.label for row in records], [0, 1])
            self.assertEqual(records[1].attack_id, "A01")
            self.assertEqual(records[1].generator_id, "UNKNOWN")
            self.assertEqual(records[1].speaker_id, "LA_0002")
            self.assertEqual(records[1].source_speaker_id, "UNKNOWN")
            self.assertEqual(records[1].official_split, "dev")
            self.assertEqual(records[1].research_split, "development")
            self.assertEqual(records[1].source_recording_id, "UNKNOWN")
            with self.assertRaises(ValueError):
                audit_partitions([records[0], replace(records[1], split="test", speaker_id=records[0].speaker_id)], require_speaker_disjoint=True)

    def test_missing_audio_fails_without_partial_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = self.fixture(root)
            (root / "ASVspoof2019_LA_dev/flac/LA_D_1000002.flac").unlink()
            with self.assertRaises(FileNotFoundError):
                import_public(ASVspoof2019LA(), dataset_root=root, protocol=protocol, official_split="dev", output=root / "missing")
            self.assertEqual(load_public_manifest(root / "missing/provenance_manifest.csv"), [])
            self.assertTrue((root / "missing/source_inventory.csv").exists())

    def test_bad_label_partition_or_protocol_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = self.fixture(root)
            original = protocol.read_text()
            for text in (original.replace("bonafide", "REAL"), original.replace("LA_D_", "LA_E_"), original + original):
                protocol.write_text(text)
                with self.assertRaises(ValueError):
                    ASVspoof2019LA().parse(protocol, "dev")

    def test_candidate_near_duplicate_is_flagged_not_excluded(self):
        original = np.random.default_rng(9).normal(0, .1, 16000).astype(np.float32)
        shifted = np.pad(original * .7, (8, 0))[:16000]
        unrelated = np.random.default_rng(19).normal(0, .1, 16000).astype(np.float32)
        result = near_duplicate_candidates({"a": Audio(original, 16000), "b": Audio(shifted, 16000), "c": Audio(unrelated, 16000)})
        self.assertEqual([(row["sample_a"], row["sample_b"]) for row in result], [("a", "b")])
        self.assertFalse(result[0]["automatic_exclusion"])


if __name__ == "__main__":
    unittest.main()
