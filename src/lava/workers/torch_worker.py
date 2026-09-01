"""JSON stdin/stdout worker for process-isolated native PyTorch detectors."""

from __future__ import annotations

import json
import sys
import traceback
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.lava.models.pytorch.specs import AASIST_SPEC, RAWNET2_SPEC
from src.lava.preprocessing.waveform import load_waveform
from src.lava.timing import timed_runs


def _build(name: str):
    if name == "rawnet2":
        from src.lava.models.pytorch.rawnet2 import NATIVE_FAKE_INDEX, build_model
    elif name == "aasist":
        from src.lava.models.pytorch.aasist import NATIVE_FAKE_INDEX, build_model
    else:
        raise ValueError(f"Unknown torch detector: {name}")
    return build_model(), NATIVE_FAKE_INDEX


def _spec(name: str):
    return RAWNET2_SPEC if name == "rawnet2" else AASIST_SPEC


def _load(name: str, device: torch.device):
    spec = _spec(name)
    if not spec.model_artifact.is_file():
        raise FileNotFoundError(f"Production model not found for {name}. Run: python train.py --model {name}")
    checkpoint = torch.load(spec.model_artifact, map_location=device)
    model, fake_index = _build(name)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, fake_index, int(checkpoint.get("target_samples", 48_000))


def handle(request: dict[str, Any]) -> dict[str, Any]:
    operation = str(request.get("operation"))
    name = str(request.get("model"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if operation == "train":
        from src.lava.training.torch_training import train_detector

        return {"ok": True, "metadata": train_detector(name, request.get("options") or {})}
    if operation == "model_info":
        model, _ = _build(name)
        return {
            "ok": True,
            "framework": "pytorch",
            "framework_version": torch.__version__,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "native_class_order": ["FAKE", "REAL"],
        }
    if operation == "smoke":
        model, fake_index = _build(name)
        model.eval()
        lengths = request.get("lengths") or [48_000]
        outputs = {}
        with torch.no_grad():
            for length in lengths:
                logits = model(torch.zeros(1, int(length)))
                scores = torch.softmax(logits, dim=1)[:, fake_index]
                outputs[str(length)] = {"logits_shape": list(logits.shape), "p_fake": float(scores[0])}
        return {"ok": True, "outputs": outputs}
    model, fake_index, target_samples = _load(name, device)
    if operation == "load_check":
        return {"ok": True, "target_samples": target_samples}
    if operation == "predict_scores":
        scores = []
        with torch.no_grad():
            for audio_path in request.get("audio_paths", []):
                waveform = load_waveform(audio_path, sample_rate=16_000, target_samples=target_samples)
                logits = model(torch.from_numpy(waveform).unsqueeze(0).to(device))
                scores.append(float(torch.softmax(logits, dim=1)[0, fake_index].cpu()))
        return {"ok": True, "scores": scores}
    if operation == "benchmark":
        audio_path = str(request["audio_path"])
        warmup, runs = int(request.get("warmup", 10)), int(request.get("runs", 50))
        waveform = load_waveform(audio_path, sample_rate=16_000, target_samples=target_samples)
        tensor = torch.from_numpy(waveform).unsqueeze(0).to(device)

        def synchronize() -> None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def model_call() -> None:
            synchronize()
            with torch.no_grad():
                torch.softmax(model(tensor), dim=1)[:, fake_index]
            synchronize()

        def end_to_end_call() -> None:
            values = load_waveform(audio_path, sample_rate=16_000, target_samples=target_samples)
            batch = torch.from_numpy(values).unsqueeze(0).to(device)
            synchronize()
            with torch.no_grad():
                torch.softmax(model(batch), dim=1)[:, fake_index]
            synchronize()

        timing = {
            "model_only": timed_runs(model_call, warmup=warmup, runs=runs),
            "preprocessing": timed_runs(
                lambda: load_waveform(audio_path, sample_rate=16_000, target_samples=target_samples),
                warmup=min(2, warmup), runs=runs,
            ),
            "end_to_end": timed_runs(end_to_end_call, warmup=min(2, warmup), runs=runs),
            "device": str(device),
            "gpu_synchronized": device.type == "cuda",
        }
        return {"ok": True, "timing": timing}
    raise ValueError(f"Unknown operation: {operation}")


def main() -> None:
    try:
        request = json.loads(sys.stdin.read())
        print(json.dumps(handle(request)))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "traceback": traceback.format_exc()}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
