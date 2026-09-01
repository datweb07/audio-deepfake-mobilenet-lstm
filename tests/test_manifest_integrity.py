from __future__ import annotations

import csv
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from src.lava.data.manifest import build_manifests, validate_manifest_files


def write_wav(path: Path, value: int) -> None:
    samples = np.full(800, value, dtype=np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(samples.tobytes())


class ManifestIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.real = self.root / "data" / "REAL"
        self.fake = self.root / "data" / "FAKE"
        self.output = self.root / "data" / "manifests"
        self.real.mkdir(parents=True)
        self.fake.mkdir(parents=True)
        for index, value in enumerate((10, 20, 30, 40, 50, 60)):
            write_wav(self.real / f"real_{index}.wav", value)
        for index, value in enumerate((-10, -20, -30, -40, -50, -60)):
            write_wav(self.fake / f"fake_{index}.wav", value)
        (self.real / "real_duplicate.wav").write_bytes((self.real / "real_0.wav").read_bytes())
        (self.fake / "conflict.wav").write_bytes((self.real / "real_1.wav").read_bytes())

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _read(self, name: str) -> list[dict[str, str]]:
        with (self.output / name).open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_integrity_and_determinism(self) -> None:
        first = build_manifests(real_dir=self.real, fake_dir=self.fake, output_dir=self.output, base_dir=self.root)
        split_rows = self._read("split_manifest.csv")
        dataset_rows = self._read("dataset_manifest.csv")
        conflicts = [row for row in dataset_rows if row["integrity_status"] == "LABEL_CONFLICT"]
        self.assertEqual(len(conflicts), 2)
        self.assertTrue(all(row["sample_id"] not in {item["sample_id"] for item in split_rows} for row in conflicts))
        checksum_splits: dict[str, set[str]] = {}
        group_splits: dict[str, set[str]] = {}
        for row in split_rows:
            checksum_splits.setdefault(row["sha256"], set()).add(row["split"])
            group_splits.setdefault(row["duplicate_group_id"], set()).add(row["split"])
        self.assertTrue(all(len(value) == 1 for value in checksum_splits.values()))
        self.assertTrue(all(len(value) == 1 for value in group_splits.values()))
        self.assertEqual({int(row["label"]) for row in split_rows}, {0, 1})
        second = build_manifests(real_dir=self.real, fake_dir=self.fake, output_dir=self.output, base_dir=self.root)
        self.assertEqual(first["manifest_hash"], second["manifest_hash"])
        self.assertEqual(first["split_counts"], second["split_counts"])
        verified = validate_manifest_files(self.output)
        self.assertEqual(first["manifest_hash"], verified["manifest_hash"])


if __name__ == "__main__":
    unittest.main()
