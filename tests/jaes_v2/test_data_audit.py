from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from src.lava.jaes_v2.data_audit import audit_sources
from src.lava.jaes_v2.evidence import file_hash
from src.lava.jaes_v2.provenance import SourceRecord


class SourceAuditTests(unittest.TestCase):
    def test_same_pcm_different_container_detects_split_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Identical quantized PCM; separate float->PCM encoders need not round alike.
            signal = np.arange(500, dtype=np.int16)
            records = []
            for index, (suffix, subtype, split) in enumerate(((".wav", "PCM_16", "train"), (".flac", "PCM_16", "test"))):
                path = root / ("source" + suffix)
                sf.write(path, signal, 16000, subtype=subtype)
                records.append(SourceRecord(sample_id=str(index), path=path.name, checksum=file_hash(path), label=0,
                                            split=split, parent_recording_id=str(index), dataset_id="fixture",
                                            dataset_version="1", sample_rate=16000, duration=len(signal) / 16000))
            self.assertNotEqual(records[0].checksum, records[1].checksum)
            with self.assertRaisesRegex(ValueError, "Decoded-waveform"):
                audit_sources(root, records)

    def test_shortcut_model_never_fit_on_test_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "source.wav"
            sf.write(path, np.arange(500) / 1000, 16000, subtype="FLOAT")
            record = SourceRecord(sample_id="s", path=path.name, checksum=file_hash(path), label=0, split="test",
                                  parent_recording_id="p", dataset_id="fixture", dataset_version="1", sample_rate=16000,
                                  duration=500 / 16000)
            summary, _ = audit_sources(root, [record])
            self.assertEqual(summary["shortcut_baseline"], "NOT_RUN")


if __name__ == "__main__":
    unittest.main()
