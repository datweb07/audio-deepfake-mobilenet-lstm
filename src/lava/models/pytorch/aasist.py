"""MIT-derived faithful AASIST spectro-temporal graph architecture for LAVA."""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lava.models.pytorch.rawnet2 import _fixed_mel_sinc_filters


NATIVE_FAKE_INDEX = 0
NATIVE_REAL_INDEX = 1


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(torch.empty(out_dim, 1))
        nn.init.xavier_normal_(self.att_weight)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(0.2)
        self.activation = nn.SELU(inplace=False)
        self.temperature = temperature

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.input_drop(inputs)
        pairs = values.unsqueeze(2) * values.unsqueeze(1)
        attention = torch.matmul(torch.tanh(self.att_proj(pairs)), self.att_weight)
        attention = F.softmax(attention / self.temperature, dim=-2).squeeze(-1)
        output = self.proj_with_att(torch.matmul(attention, values)) + self.proj_without_att(values)
        shape = output.shape
        output = self.bn(output.reshape(-1, shape[-1])).reshape(shape)
        return self.activation(output)


class HeterogeneousGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_proj_master = nn.Linear(in_dim, out_dim)
        self.att_weight11 = self._parameter(out_dim, 1)
        self.att_weight22 = self._parameter(out_dim, 1)
        self.att_weight12 = self._parameter(out_dim, 1)
        self.att_weight_master = self._parameter(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_att_master = nn.Linear(in_dim, out_dim)
        self.proj_without_att_master = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(0.2)
        self.activation = nn.SELU(inplace=False)
        self.temperature = temperature

    @staticmethod
    def _parameter(*shape: int) -> nn.Parameter:
        parameter = nn.Parameter(torch.empty(*shape))
        nn.init.xavier_normal_(parameter)
        return parameter

    def _master_update(self, nodes: torch.Tensor, master: torch.Tensor) -> torch.Tensor:
        attention = torch.tanh(self.att_proj_master(nodes * master))
        attention = torch.matmul(attention, self.att_weight_master)
        attention = F.softmax(attention / self.temperature, dim=-2)
        attended = torch.matmul(attention.squeeze(-1).unsqueeze(1), nodes)
        return self.proj_with_att_master(attended) + self.proj_without_att_master(master)

    def forward(
        self, type1: torch.Tensor, type2: torch.Tensor, master: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        type1 = self.proj_type1(type1)
        type2 = self.proj_type2(type2)
        count1, count2 = type1.size(1), type2.size(1)
        nodes = torch.cat([type1, type2], dim=1)
        if master is None:
            master = torch.mean(nodes, dim=1, keepdim=True)
        if master.size(0) == 1 and nodes.size(0) != 1:
            master = master.expand(nodes.size(0), -1, -1)
        values = self.input_drop(nodes)
        pairs = values.unsqueeze(2) * values.unsqueeze(1)
        projected = torch.tanh(self.att_proj(pairs))
        board = torch.zeros_like(projected[..., :1])
        board[:, :count1, :count1] = torch.matmul(projected[:, :count1, :count1], self.att_weight11)
        board[:, count1:, count1:] = torch.matmul(projected[:, count1:, count1:], self.att_weight22)
        board[:, :count1, count1:] = torch.matmul(projected[:, :count1, count1:], self.att_weight12)
        board[:, count1:, :count1] = torch.matmul(projected[:, count1:, :count1], self.att_weight12)
        attention = F.softmax(board / self.temperature, dim=-2).squeeze(-1)
        updated_master = self._master_update(values, master)
        output = self.proj_with_att(torch.matmul(attention, values)) + self.proj_without_att(values)
        shape = output.shape
        output = self.activation(self.bn(output.reshape(-1, shape[-1])).reshape(shape))
        return output[:, :count1], output[:, count1:count1 + count2], updated_master


class GraphPool(nn.Module):
    def __init__(self, ratio: float, in_dim: int, dropout: float):
        super().__init__()
        self.ratio = ratio
        self.projection = nn.Linear(in_dim, 1)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        scores = torch.sigmoid(self.projection(self.dropout(nodes)))
        keep = max(int(nodes.size(1) * self.ratio), 1)
        _, indices = torch.topk(scores, keep, dim=1)
        indices = indices.expand(-1, -1, nodes.size(2))
        return torch.gather(nodes * scores, 1, indices)


class FixedSincFrontEnd(nn.Module):
    def __init__(self, out_channels: int = 70, kernel_size: int = 128, sample_rate: int = 16_000):
        super().__init__()
        filters = _fixed_mel_sinc_filters(out_channels, kernel_size, sample_rate)
        self.register_buffer("filters", filters, persistent=True)

    def forward(self, waveform: torch.Tensor, frequency_mask: bool = False) -> torch.Tensor:
        filters = self.filters
        if frequency_mask and self.training:
            maximum = min(20, filters.size(0) - 1)
            width = random.randint(0, maximum)
            if width:
                start = random.randint(0, filters.size(0) - width)
                filters = filters.clone()
                filters[start:start + width] = 0
        return F.conv1d(waveform, filters)


class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, first: bool = False):
        super().__init__()
        self.first = first
        self.bn1 = nn.Identity() if first else nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (2, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (2, 3), padding=(0, 1))
        self.projection = (
            nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1))
            if in_channels != out_channels else nn.Identity()
        )
        self.pool = nn.MaxPool2d((1, 3))
        self.activation = nn.SELU(inplace=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = self.projection(inputs)
        # The released AASIST implementation defines bn1 but feeds inputs
        # directly to conv1. Preserve this behavior for checkpoint parity.
        output = self.conv1(inputs)
        output = self.conv2(self.activation(self.bn2(output)))
        return self.pool(output + identity)


class AASIST(nn.Module):
    """Released AASIST topology with native [spoof, bonafide] logits."""

    def __init__(self):
        super().__init__()
        filters = [70, (1, 32), (32, 32), (32, 64), (64, 64)]
        gat_dims = (64, 32)
        temperatures = (2.0, 2.0, 100.0, 100.0)
        self.frontend = FixedSincFrontEnd(filters[0], 128, 16_000)
        self.first_bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU(inplace=False)
        self.encoder = nn.Sequential(
            ResidualBlock2d(*filters[1], first=True),
            ResidualBlock2d(*filters[2]),
            ResidualBlock2d(*filters[3]),
            ResidualBlock2d(*filters[4]),
            ResidualBlock2d(*filters[4]),
            ResidualBlock2d(*filters[4]),
        )
        self.positional_spectral = nn.Parameter(torch.randn(1, 23, filters[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.gat_spectral = GraphAttentionLayer(filters[-1][-1], gat_dims[0], temperatures[0])
        self.gat_temporal = GraphAttentionLayer(filters[-1][-1], gat_dims[0], temperatures[1])
        self.pool_spectral = GraphPool(0.5, gat_dims[0], 0.3)
        self.pool_temporal = GraphPool(0.7, gat_dims[0], 0.3)
        self.heterogeneous11 = HeterogeneousGraphAttentionLayer(gat_dims[0], gat_dims[1], temperatures[2])
        self.heterogeneous12 = HeterogeneousGraphAttentionLayer(gat_dims[1], gat_dims[1], temperatures[2])
        self.heterogeneous21 = HeterogeneousGraphAttentionLayer(gat_dims[0], gat_dims[1], temperatures[2])
        self.heterogeneous22 = HeterogeneousGraphAttentionLayer(gat_dims[1], gat_dims[1], temperatures[2])
        self.pool_hs1 = GraphPool(0.5, gat_dims[1], 0.3)
        self.pool_ht1 = GraphPool(0.5, gat_dims[1], 0.3)
        self.pool_hs2 = GraphPool(0.5, gat_dims[1], 0.3)
        self.pool_ht2 = GraphPool(0.5, gat_dims[1], 0.3)
        self.drop_way = nn.Dropout(0.2)
        self.drop = nn.Dropout(0.5)
        self.classifier = nn.Linear(5 * gat_dims[1], 2)

    def _heterogeneous_path(
        self,
        temporal: torch.Tensor,
        spectral: torch.Tensor,
        first: HeterogeneousGraphAttentionLayer,
        second: HeterogeneousGraphAttentionLayer,
        pool_t: GraphPool,
        pool_s: GraphPool,
        master: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out_t, out_s, out_master = first(temporal, spectral, master=master)
        out_s, out_t = pool_s(out_s), pool_t(out_t)
        aug_t, aug_s, aug_master = second(out_t, out_s, master=out_master)
        return out_t + aug_t, out_s + aug_s, out_master + aug_master

    def forward(self, waveform: torch.Tensor, frequency_mask: bool = False) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError(f"AASIST expects (B, samples), received {tuple(waveform.shape)}")
        features = self.frontend(waveform.unsqueeze(1), frequency_mask=frequency_mask)
        features = F.max_pool2d(torch.abs(features.unsqueeze(1)), (3, 3))
        features = self.selu(self.first_bn(features))
        encoded = self.encoder(features)
        spectral = torch.max(torch.abs(encoded), dim=3).values.transpose(1, 2)
        if spectral.size(1) != self.positional_spectral.size(1):
            raise ValueError(
                f"AASIST spectral node mismatch: expected {self.positional_spectral.size(1)}, got {spectral.size(1)}"
            )
        spectral = self.pool_spectral(self.gat_spectral(spectral + self.positional_spectral))
        temporal = torch.max(torch.abs(encoded), dim=2).values.transpose(1, 2)
        temporal = self.pool_temporal(self.gat_temporal(temporal))

        temporal1, spectral1, master1 = self._heterogeneous_path(
            temporal, spectral, self.heterogeneous11, self.heterogeneous12,
            self.pool_ht1, self.pool_hs1, self.master1,
        )
        temporal2, spectral2, master2 = self._heterogeneous_path(
            temporal, spectral, self.heterogeneous21, self.heterogeneous22,
            self.pool_ht2, self.pool_hs2, self.master2,
        )
        temporal = torch.maximum(self.drop_way(temporal1), self.drop_way(temporal2))
        spectral = torch.maximum(self.drop_way(spectral1), self.drop_way(spectral2))
        master = torch.maximum(self.drop_way(master1), self.drop_way(master2))
        hidden = torch.cat(
            [
                torch.max(torch.abs(temporal), dim=1).values,
                torch.mean(temporal, dim=1),
                torch.max(torch.abs(spectral), dim=1).values,
                torch.mean(spectral, dim=1),
                master.squeeze(1),
            ],
            dim=1,
        )
        return self.classifier(self.drop(hidden))


def build_model() -> AASIST:
    return AASIST()

