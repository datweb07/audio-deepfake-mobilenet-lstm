"""Model-independent warm latency/throughput/RTF foundation."""

from __future__ import annotations

import os
import platform
import time
import ctypes
from ctypes import wintypes
from pathlib import Path

import config
from src.lava.data.loader import load_split
from src.lava.artifacts import artifact_readiness, write_json_atomic
from src.lava.registry import create, get_spec


def _rss_mib() -> float | None:
    try:
        import psutil
        return float(psutil.Process().memory_info().rss / 1024**2)
    except ImportError:
        if os.name != "nt":
            return None
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        success = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        return float(counters.WorkingSetSize / 1024**2) if success else None


def run_efficiency(model_name: str, *, warmup: int = 10, runs: int = 50) -> dict[str, object]:
    spec = get_spec(model_name)
    ready, missing = artifact_readiness(spec)
    if not ready:
        raise FileNotFoundError(f"Detector artifacts are not ready for {model_name}: {missing}")
    representative = load_split("test")[0][0]
    detector = create(model_name)
    rss_before = _rss_mib()
    cold_start = time.perf_counter()
    detector.load()
    cold_start_seconds = time.perf_counter() - cold_start
    rss_after = _rss_mib()
    timing = detector.benchmark_audio(representative, warmup=warmup, runs=runs)
    model_mean = float(timing["model_only"]["mean_seconds"])
    end_mean = float(timing["end_to_end"]["mean_seconds"])
    result = {
        "status": "BENCHMARKED" if runs >= 10 else "SMOKE_TESTED",
        "detector": model_name,
        "framework": spec.framework,
        "batch_size": 1,
        "warmup": warmup,
        "runs": runs,
        "parameter_count": detector.parameter_count(),
        "serialized_size_bytes": detector.model_size(),
        "cold_start_seconds_including_load": cold_start_seconds,
        "process_startup_excluded_from_warm_timing": True,
        "model_only": timing["model_only"],
        "preprocessing": timing["preprocessing"],
        "end_to_end": timing["end_to_end"],
        "model_only_throughput_files_per_second": 1.0 / model_mean,
        "end_to_end_throughput_files_per_second": 1.0 / end_mean,
        "model_only_rtf": model_mean / spec.audio_duration,
        "end_to_end_rtf": end_mean / spec.audio_duration,
        "process_rss_mib": {"before_load": rss_before, "after_load": rss_after},
        "memory_note": "Process RSS snapshots are not isolated model peak memory; PyTorch child RSS is not captured.",
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count()},
        "representative_file": representative,
    }
    output = Path(config.OUTPUTS_DIR) / "benchmark" / "efficiency" / model_name / "summary.json"
    write_json_atomic(output, result)
    return result
