"""Modern device-safe adaptation of the MIT RawNet2 anti-spoofing architecture."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


NATIVE_FAKE_INDEX = 0
NATIVE_REAL_INDEX = 1


def _fixed_mel_sinc_filters(out_channels: int, kernel_size: int, sample_rate: int) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    frequencies = np.linspace(0.0, sample_rate / 2.0, 257)
    mel = 2595.0 * np.log10(1.0 + frequencies / 700.0)
    mel_edges = np.linspace(mel.min(), mel.max(), out_channels + 2)
    hz_edges = 700.0 * (10.0 ** (mel_edges / 2595.0) - 1.0)
    support = np.arange(-(kernel_size - 1) / 2, (kernel_size - 1) / 2 + 1)
    window = np.hamming(kernel_size)
    filters = np.zeros((out_channels, kernel_size), dtype=np.float32)
    for index in range(out_channels):
        lower, upper = hz_edges[index], hz_edges[index + 1]
        high = (2 * upper / sample_rate) * np.sinc(2 * upper * support / sample_rate)
        low = (2 * lower / sample_rate) * np.sinc(2 * lower * support / sample_rate)
        filters[index] = window * (high - low)
    return torch.from_numpy(filters[:, None, :])


class FixedSincConv1d(nn.Module):
    def __init__(self, out_channels: int = 128, kernel_size: int = 128, sample_rate: int = 16_000):
        super().__init__()
        filters = _fixed_mel_sinc_filters(out_channels, kernel_size, sample_rate)
        self.register_buffer("filters", filters, persistent=True)
        self.kernel_size = int(filters.shape[-1])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.conv1d(inputs, self.filters)


class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, first: bool = False):
        super().__init__()
        self.first = first
        self.bn1 = nn.Identity() if first else nn.BatchNorm1d(in_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.3)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.projection = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.pool = nn.MaxPool1d(3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = self.projection(inputs)
        features = inputs if self.first else self.activation(self.bn1(inputs))
        features = self.conv1(features)
        features = self.conv2(self.activation(self.bn2(features)))
        return self.pool(features + identity)


class RawNet2(nn.Module):
    """RawNet2 topology with native [spoof, bonafide] two-logit output ordering."""

    def __init__(self):
        super().__init__()
        self.sinc = FixedSincConv1d(128, 128, 16_000)
        self.first_bn = nn.BatchNorm1d(128)
        self.selu = nn.SELU(inplace=False)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock1d(128, 128, first=True),
                ResidualBlock1d(128, 128),
                ResidualBlock1d(128, 512),
                ResidualBlock1d(512, 512),
                ResidualBlock1d(512, 512),
                ResidualBlock1d(512, 512),
            ]
        )
        channels = (128, 128, 512, 512, 512, 512)
        self.attention = nn.ModuleList([nn.Linear(channel, channel) for channel in channels])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.before_gru = nn.BatchNorm1d(512)
        self.gru = nn.GRU(512, 1024, num_layers=3, batch_first=True)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError(f"RawNet2 expects (B, samples), received {tuple(waveform.shape)}")
        features = self.sinc(waveform.unsqueeze(1))
        features = F.max_pool1d(torch.abs(features), 3)
        features = self.selu(self.first_bn(features))
        for block, attention in zip(self.blocks, self.attention):
            block_output = block(features)
            weights = torch.sigmoid(attention(self.pool(block_output).flatten(1))).unsqueeze(-1)
            features = block_output * weights + weights
        features = self.selu(self.before_gru(features)).transpose(1, 2)
        self.gru.flatten_parameters()
        features, _ = self.gru(features)
        return self.fc2(self.fc1(features[:, -1, :]))


def build_model() -> RawNet2:
    return RawNet2()

