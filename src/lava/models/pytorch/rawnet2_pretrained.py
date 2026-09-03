"""Bridge for the original RawNet2-DF pretrained weights (ASVspoof2021 DF track).

Imports the original model class from 2021-main/LA/Baseline-RawNet2/model.py.
Checkpoint format: flat state_dict (no wrapper key).
Output: logits [spoof, bonafide] -> P(FAKE) = softmax(logits)[0]
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch import Tensor

# parents[4] = D:\audio-deepfake-mobilenet-lstm (repo root)
_RAWNET2_REPO = Path(__file__).parents[4] / "2021-main" / "LA" / "Baseline-RawNet2"
if str(_RAWNET2_REPO) not in sys.path:
    sys.path.insert(0, str(_RAWNET2_REPO))

from model import RawNet as _OriginalRawNet2  # noqa: E402 (class is named RawNet in original repo)

NATIVE_FAKE_INDEX: int = 0
_TARGET_SAMPLES: int = 64_600
_MODEL_CONFIG = {
    "nb_samp": 64_600, "first_conv": 1024, "in_channels": 1,
    "filts": [20, [20, 20], [20, 128], [128, 128]],
    "blocks": [2, 4], "nb_fc_node": 1024, "gru_node": 1024,
    "nb_gru_layer": 3, "nb_classes": 2,
}


class RawNet2Pretrained(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self._inner = _OriginalRawNet2(_MODEL_CONFIG, device)
        self._device = device

    def forward(self, waveform: Tensor) -> Tensor:
        # Original RawNet.forward() expects (batch, samples) — 2D input.
        # It internally does x.view(nb_samp, 1, len_seq) to add the channel dim.
        # Do NOT unsqueeze(1) here; that would corrupt the shape contract.
        if waveform.ndim != 2:
            raise ValueError(f"RawNet2 (pretrained) expects (B, samples) 2D input, got {tuple(waveform.shape)}")
        return self._inner(waveform)


def build_model(device: torch.device | None = None) -> RawNet2Pretrained:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return RawNet2Pretrained(device)


def load_pretrained(checkpoint_path, device: torch.device) -> RawNet2Pretrained:
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model = build_model(device)
    model._inner.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def target_samples() -> int:
    return _TARGET_SAMPLES
