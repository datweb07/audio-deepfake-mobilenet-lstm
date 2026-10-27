"""Read-only legacy detector bridge and reproducibility/resource preflight."""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np

from .evidence import file_hash, object_hash

ACCEPTED_SHA256 = "91fa8753025895bbf88470f4ebe7e6c283e897c8370fa8e91b1719abaac61073"


def source_fingerprint(root: Path) -> dict:
    paths = set()
    for folder in ("src", "benchmark", "scripts", "tests", "configs/jaes_v2"):
        paths.update(path for path in (root / folder).rglob("*") if path.suffix in {".py", ".json"})
    paths.update(root / name for name in ("config.py", "train.py", "predict.py", "evaluate.py", "app.py", "requirements.txt", "requirements-torch.txt"))
    hashes = {path.relative_to(root).as_posix(): file_hash(path) for path in sorted(paths) if path.is_file()}
    return {"files": hashes, "hash": object_hash(hashes)}


def preflight(root: Path, *, verify_freeze: bool = True) -> dict:
    manuscript = root / "papers/PaperID_804_final.docx"
    if file_hash(manuscript) != ACCEPTED_SHA256:
        raise ValueError("Accepted manuscript hash changed")
    packages = {}
    for name in ("numpy", "scipy", "soundfile", "matplotlib", "scikit-learn", "tensorflow", "onnxruntime"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    executable = shutil.which("ffmpeg")
    version = subprocess.check_output([executable, "-version"], text=True, timeout=15).splitlines()[0] if executable else None
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = "NOT_DETECTED"
    freeze_status = "NOT_CHECKED"
    if verify_freeze:
        import json
        path = root / "outputs/jaes_v2/audit/baseline_freeze.json"
        frozen = json.loads(path.read_text(encoding="utf-8"))
        amendment_path = root / "outputs/jaes_v2/audit/phase2_baseline_amendment.json"
        exceptions = {}
        if amendment_path.is_file():
            amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
            if amendment["parent_snapshot_sha256"] != file_hash(path):
                raise ValueError("Baseline amendment references a different original snapshot")
            exceptions = amendment["approved_changes"]
            allowed = {"src/lava/benchmark_display.py", "tests/test_decision_display.py", "tests/test_streamlit_ui.py"}
            if set(exceptions) - allowed:
                raise ValueError("Unapproved legacy amendment paths")
            for name, change in exceptions.items():
                if change["before_sha256"] != frozen["protected_files"][name]["sha256"] or not change.get("reason"):
                    raise ValueError("Invalid baseline amendment provenance")
        changed = [name for name, info in frozen["protected_files"].items()
                   if not (root / name).is_file() or file_hash(root / name) != exceptions.get(name, {}).get("after_sha256", info["sha256"])]
        if changed:
            raise ValueError(f"Protected baseline changed: {changed[:10]}")
        freeze_status = f"PASS ({len(frozen['protected_files'])} files)"
        if exceptions:
            freeze_status += f"; {len(exceptions)} explicit Phase 2 source/test amendments; original snapshot retained"
    return {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "gpu_inventory_not_used_by_c1": gpu, "packages": packages,
            "free_disk_bytes": shutil.disk_usage(root).free, "ffmpeg": executable, "ffmpeg_version": version,
            "ffmpeg_sha256": file_hash(Path(executable)) if executable else None,
            "accepted_paper_sha256": ACCEPTED_SHA256, "baseline_freeze_verification": freeze_status,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
            "execution_policy": "CPU-only; no model training; no probability calibration or policy fitting"}


class LegacyMelPredictor:
    """Reuse artifact loading and preprocessing. Time actual synchronous forward calls."""

    def __init__(self, model_id: str) -> None:
        # These are set before the lazy TensorFlow import in a fresh CLI process.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        from src.lava.registry import create
        from src.lava.artifacts import load_threshold
        from src.preprocessing import process_audio_file
        self.tensorflow = tf
        self.preprocess = process_audio_file
        self.detector = create(model_id)
        started = perf_counter()
        self.detector.load()
        self.load_seconds = perf_counter() - started
        self.threshold = load_threshold(self.detector.spec)
        self.artifact_hashes = {}
        for path in (self.detector.spec.model_artifact, self.detector.spec.threshold_artifact, self.detector.spec.metadata_artifact):
            self.artifact_hashes[str(path)] = file_hash(path)
        # MobileNet's deployment loader may restore one of these verified fallbacks.
        if model_id == "mobilenetv3_lstm":
            for path in self.detector.spec.model_artifact.parent.glob("lava_mobilenetv3_lstm.weights.*"):
                self.artifact_hashes[str(path)] = file_hash(path)

    def predict(self, path: Path) -> tuple[float, dict]:
        started = perf_counter()
        features = self.preprocess(str(path))
        tensor = self.tensorflow.convert_to_tensor(features[np.newaxis], dtype=self.tensorflow.float32)
        feature_done = perf_counter()
        score = float(self.detector.model(tensor, training=False).numpy().reshape(-1)[0])
        done = perf_counter()
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Detector violated P(FAKE) contract")
        return score, {"preprocessing_latency_ms": (feature_done - started) * 1000,
                       "model_latency_ms": (done - feature_done) * 1000,
                       "end_to_end_latency_ms": (done - started) * 1000}
