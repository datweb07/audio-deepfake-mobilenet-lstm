"""Validation-only model selection and threshold calibration for native PyTorch detectors."""

from __future__ import annotations

import csv
import json
import os
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import config
from src.lava.data.manifest import MANIFEST_METADATA, SPLIT_MANIFEST
from src.lava.models.pytorch.specs import AASIST_SPEC, RAWNET2_SPEC
from src.lava.preprocessing.waveform import load_waveform
from src.lava.training.policy import assert_test_isolation, require_validation_source


class WaveformManifestDataset(Dataset):
    def __init__(self, split: str, target_samples: int):
        self.rows: list[tuple[str, int]] = []
        with SPLIT_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["split"] != split:
                    continue
                path = Path(row["path"])
                if not path.is_absolute():
                    path = Path(config.BASE_DIR) / path
                self.rows.append((str(path), int(row["label"])))
        if not self.rows:
            raise RuntimeError(f"No canonical samples for split '{split}'")
        self.target_samples = target_samples

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, lava_label = self.rows[index]
        waveform = load_waveform(path, sample_rate=16_000, target_samples=self.target_samples)
        # Native reference heads are [spoof=0, bonafide=1].
        native_label = 0 if lava_label == config.FAKE_LABEL else 1
        return torch.from_numpy(waveform), torch.tensor(native_label, dtype=torch.long)


def _build_model(name: str) -> nn.Module:
    if name == "rawnet2":
        from src.lava.models.pytorch.rawnet2 import build_model
    elif name == "aasist":
        from src.lava.models.pytorch.aasist import build_model
    else:
        raise ValueError(f"Unsupported torch detector: {name}")
    return build_model()


def _spec(name: str):
    return RAWNET2_SPEC if name == "rawnet2" else AASIST_SPEC


def _forward(model: nn.Module, inputs: torch.Tensor, name: str, training: bool) -> torch.Tensor:
    if name == "aasist":
        return model(inputs, frequency_mask=training)
    return model(inputs)


def _validation(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, name: str
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    labels: list[int] = []
    scores: list[float] = []
    with torch.no_grad():
        for inputs, native_labels in loader:
            inputs, native_labels = inputs.to(device), native_labels.to(device)
            logits = _forward(model, inputs, name, False)
            total_loss += float(criterion(logits, native_labels).item()) * inputs.size(0)
            probabilities = torch.softmax(logits, dim=1)[:, 0]
            scores.extend(probabilities.cpu().numpy().tolist())
            labels.extend((1 - native_labels).cpu().numpy().tolist())
    return total_loss / len(loader.dataset), np.asarray(labels, dtype=np.int32), np.asarray(scores, dtype=np.float32)


def _calibrate(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.arange(0.10, 0.901, 0.01):
        predictions = scores >= threshold
        tp = int(np.sum((predictions == 1) & (labels == 1)))
        fp = int(np.sum((predictions == 1) & (labels == 0)))
        fn = int(np.sum((predictions == 0) & (labels == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1 or (np.isclose(f1, best_f1) and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_threshold, best_f1 = float(threshold), float(f1)
    return best_threshold, best_f1


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=config.BASE_DIR, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def train_detector(name: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one complete native lifecycle; test data is never loaded here."""
    options = options or {}
    assert_test_isolation()
    if not SPLIT_MANIFEST.is_file() or not MANIFEST_METADATA.is_file():
        raise FileNotFoundError("Canonical manifest missing. Run: python -m src.lava.data.manifest build")
    seed = int(options.get("seed", config.RANDOM_SEED))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_samples = int(options.get("target_samples", 48_000))
    epochs = int(options.get("epochs", 100))
    batch_size = int(options.get("batch_size", 8))
    patience = int(options.get("patience", 10))
    learning_rate = float(options.get("learning_rate", 1e-4))
    weight_decay = float(options.get("weight_decay", 1e-4))

    train_data = WaveformManifestDataset("train", target_samples)
    validation_data = WaveformManifestDataset("validation", target_samples)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    model = _build_model(name).to(device)
    counts = Counter(native_label for _, lava_label in train_data.rows for native_label in [0 if lava_label == 1 else 1])
    total = len(train_data)
    class_weights = torch.tensor([total / (2 * counts[0]), total / (2 * counts[1])], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=5e-6)
        if name == "aasist" else None
    )
    checkpoint = Path(config.CHECKPOINTS_DIR) / f"{name}_best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, native_labels in train_loader:
            inputs, native_labels = inputs.to(device), native_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(_forward(model, inputs, name, True), native_labels)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        validation_loss, _, _ = _validation(model, validation_loader, criterion, device, name)
        if validation_loss < best_loss:
            best_loss, best_epoch, stale_epochs = validation_loss, epoch, 0
            torch.save({"state_dict": model.state_dict(), "target_samples": target_samples}, checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if not checkpoint.is_file():
        raise RuntimeError("Training produced no validation-selected checkpoint")
    selected = torch.load(checkpoint, map_location=device)
    model.load_state_dict(selected["state_dict"])
    _, validation_labels, validation_scores = _validation(model, validation_loader, criterion, device, name)
    require_validation_source("validation")
    threshold, threshold_f1 = _calibrate(validation_labels, validation_scores)

    spec = _spec(name)
    spec.model_artifact.parent.mkdir(parents=True, exist_ok=True)
    pending = spec.model_artifact.with_suffix(".pending.pt")
    torch.save(
        {
            "detector_name": name,
            "architecture_version": "lava-native-v1",
            "state_dict": model.cpu().state_dict(),
            "target_samples": target_samples,
            "native_class_order": ["FAKE", "REAL"],
        },
        pending,
    )
    verified = torch.load(pending, map_location="cpu")
    verifier = _build_model(name)
    verifier.load_state_dict(verified["state_dict"])
    os.replace(pending, spec.model_artifact)
    spec.threshold_artifact.write_text(
        json.dumps({"threshold": threshold, "source": "validation FAKE-class F1"}, indent=2), encoding="utf-8"
    )
    with MANIFEST_METADATA.open("r", encoding="utf-8") as handle:
        manifest_metadata = json.load(handle)
    metadata = {
        "detector_name": name,
        "architecture": "RawNet2 native waveform" if name == "rawnet2" else "AASIST native spectro-temporal graph",
        "group": "reference",
        "framework": "pytorch",
        "framework_version": torch.__version__,
        "training_manifest_hash": manifest_metadata["manifest_hash"],
        "input_contract": {"sample_rate": 16000, "target_samples": target_samples},
        "duration_stratum": "lava_common_48000" if target_samples == 48_000 else f"native_or_custom_{target_samples}",
        "duration_fidelity_status": "SHAPE_VERIFIED_PERFORMANCE_COMPARISON_PENDING",
        "primary_duration_comparison_eligible": False,
        "label_mapping": {"REAL": 0, "FAKE": 1},
        "native_class_order": ["FAKE", "REAL"],
        "score_semantics": "softmax(logits)[:, 0] = P(FAKE)",
        "pretraining": "scratch",
        "pretraining_stratum": "native_reference_scratch",
        "final_threshold": threshold,
        "threshold_validation_f1": threshold_f1,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "serialized_size": spec.model_artifact.stat().st_size,
        "git_commit": _git_commit(),
        "training_seed": seed,
        "hardware_summary": {"device": str(device), "cuda_available": torch.cuda.is_available()},
        "selection": {"monitor": "validation_loss", "best_epoch": best_epoch, "best_value": best_loss, "test_used": False},
        "load_smoke_test": "PASS",
        "creation_time": datetime.now(timezone.utc).isoformat(),
    }
    spec.metadata_artifact.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    checkpoint.unlink(missing_ok=True)
    return metadata
