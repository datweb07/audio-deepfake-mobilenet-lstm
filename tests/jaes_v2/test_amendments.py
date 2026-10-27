import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.lava.jaes_v2.evidence import file_hash, write_json
from src.lava.jaes_v2.runtime import preflight


class BaselineAmendmentTests(unittest.TestCase):
    def test_exact_authorized_amendment_only_and_original_snapshot_preserved(self):
        repository = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "papers").mkdir()
            shutil.copyfile(repository / "papers/PaperID_804_final.docx", root / "papers/PaperID_804_final.docx")
            source = root / "src/lava/benchmark_display.py"
            source.parent.mkdir(parents=True)
            source.write_text("before")
            before = file_hash(source)
            snapshot = root / "outputs/jaes_v2/audit/baseline_freeze.json"
            write_json(snapshot, {"protected_files": {"src/lava/benchmark_display.py": {"sha256": before}}})
            original_snapshot_bytes = snapshot.read_bytes()
            source.write_text("after")
            amendment = root / "outputs/jaes_v2/audit/phase2_baseline_amendment.json"
            values = {"parent_snapshot_sha256": file_hash(snapshot), "approved_changes": {
                "src/lava/benchmark_display.py": {"before_sha256": before, "after_sha256": file_hash(source), "reason": "Phase 2 optional spec bug"}}}
            write_json(amendment, values)
            with patch("src.lava.jaes_v2.runtime.shutil.which", return_value=None), patch("src.lava.jaes_v2.runtime.subprocess.check_output", return_value="fixture"):
                self.assertIn("1 explicit Phase 2", preflight(root)["baseline_freeze_verification"])
                self.assertEqual(snapshot.read_bytes(), original_snapshot_bytes)
                source.write_text("unapproved later edit")
                with self.assertRaisesRegex(ValueError, "Protected baseline changed"):
                    preflight(root)
                values["approved_changes"]["models/model.pt"] = values["approved_changes"]["src/lava/benchmark_display.py"]
                write_json(amendment, values)
                with self.assertRaisesRegex(ValueError, "Unapproved legacy amendment"):
                    preflight(root)


if __name__ == "__main__":
    unittest.main()
