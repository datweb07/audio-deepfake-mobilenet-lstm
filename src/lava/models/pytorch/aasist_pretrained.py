"""Bridge for the original AASIST pretrained weights (NAVER Corp, ICASSP 2022).

Imports the original model class from aasist-main/models/AASIST.py.
Checkpoint format: flat state_dict with keys encoder.*, GAT_layer_S*, etc.
Output: (_, logits) where logits [spoof, bonafide] -> P(FAKE) = softmax(logits)[0]
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch import Tensor

# parents[4] = D:\audio-deepfake-mobilenet-lstm (repo root)
_AASIST_REPO = Path(__file__).parents[4] / "aasist-main"
_AASIST_MODELS = _AASIST_REPO / "models"
if str(_AASIST_MODELS) not in sys.path:
    sys.path.insert(0, str(_AASIST_MODELS))
if str(_AASIST_REPO) not in sys.path:
    sys.path.insert(0, str(_AASIST_REPO))

from AASIST import Model as _OriginalAASIST  # noqa: E402

NATIVE_FAKE_INDEX: int = 0
_TARGET_SAMPLES: int = 64_600
_MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


class AASISTPretrained(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._inner = _OriginalAASIST(_MODEL_CONFIG)

    def forward(self, waveform: Tensor) -> Tensor:
        """waveform: (B, samples) float32. Returns logits (B, 2)."""
        _, output = self._inner(waveform)
        return output


def build_model() -> AASISTPretrained:
    return AASISTPretrained()


def load_pretrained(checkpoint_path, device: torch.device) -> AASISTPretrained:
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model = build_model()
    model._inner.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def target_samples() -> int:
    return _TARGET_SAMPLES
