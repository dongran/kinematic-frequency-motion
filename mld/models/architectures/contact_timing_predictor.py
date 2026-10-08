from __future__ import annotations

import torch
from torch import nn


def _conv_same(in_channels: int, out_channels: int, kernel_size: int) -> nn.Conv1d:
    padding = kernel_size // 2
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=padding,
    )


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)
        self.conv = _conv_same(channels, channels, kernel_size)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        x = self.dropout(x)
        return residual + x


class ContactTimingPredictor(nn.Module):
    """Predict per-leg contact logits from timing-related motion features."""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 2,
        num_blocks: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.stem = nn.Sequential(
            _conv_same(self.input_dim, hidden_dim, 11),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualTemporalBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(int(num_blocks))
            ]
        )
        self.head = _conv_same(hidden_dim, self.output_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected x [B,T,C], got {tuple(x.shape)}")
        x = x.transpose(1, 2)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x)
        return logits.transpose(1, 2)


class ContactTimingConditionEncoder(nn.Module):
    """Encode contact/timing sequences into a single condition token."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        output_dim: int = 512,
        num_blocks: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            _conv_same(input_dim, hidden_dim, 7),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualTemporalBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(int(num_blocks))
            ]
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        timing: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if timing.dim() != 3:
            raise ValueError(f"expected timing [B,T,C], got {tuple(timing.shape)}")
        x = timing.transpose(1, 2)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)
        if mask is not None:
            if mask.dim() != 2:
                raise ValueError(f"expected mask [B,T], got {tuple(mask.shape)}")
            weight = mask.unsqueeze(-1).to(dtype=x.dtype)
            pooled = (x * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        else:
            pooled = x.mean(dim=1)
        return self.proj(pooled).unsqueeze(1)
