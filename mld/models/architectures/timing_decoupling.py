from __future__ import annotations

import torch
from torch import nn


def _masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).to(device=sequence.device, dtype=sequence.dtype)
    return (sequence * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def extract_phase_features(
    sequence: torch.Tensor,
    *,
    mask: torch.Tensor,
    num_harmonics: int = 1,
) -> torch.Tensor:
    if sequence.dim() != 3:
        raise ValueError(f"expected sequence [B,T,C], got {tuple(sequence.shape)}")
    if mask.dim() != 2:
        raise ValueError(f"expected mask [B,T], got {tuple(mask.shape)}")
    if sequence.shape[:2] != mask.shape:
        raise ValueError(
            f"sequence/mask shape mismatch: sequence={tuple(sequence.shape)}, mask={tuple(mask.shape)}"
        )

    lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0).to(dtype=sequence.dtype)
    offset = _masked_mean(sequence, mask)
    centered = (sequence - offset.unsqueeze(1)) * mask.unsqueeze(-1).to(dtype=sequence.dtype)

    spectrum = torch.fft.rfft(centered, dim=1)
    if spectrum.shape[1] <= 1:
        phase_sin = torch.zeros_like(offset)
        phase_cos = torch.ones_like(offset)
        amplitude = torch.zeros_like(offset)
        frequency = torch.zeros_like(offset)
        features = [offset, amplitude, frequency, phase_sin, phase_cos]
        for _ in range(max(int(num_harmonics), 1) - 1):
            zeros = torch.zeros_like(offset)
            features.extend([zeros, zeros, zeros, zeros])
        return torch.cat(features, dim=-1)

    usable = spectrum[:, 1:, :]
    requested_harmonics = max(int(num_harmonics), 1)
    topk = min(requested_harmonics, usable.shape[1])
    magnitude = torch.abs(usable)
    values, indices = torch.topk(magnitude, k=topk, dim=1)
    indices = indices + 1
    coeff = torch.gather(spectrum, dim=1, index=indices)

    frequency = indices.to(dtype=sequence.dtype) / lengths.unsqueeze(-1)
    amplitude = 2.0 * torch.abs(coeff) / lengths.unsqueeze(-1)
    phase = torch.angle(coeff)
    phase_sin = torch.sin(phase)
    phase_cos = torch.cos(phase)

    features = [offset]
    for harmonic_idx in range(topk):
        features.extend(
            [
                amplitude[:, harmonic_idx, :],
                frequency[:, harmonic_idx, :],
                phase_sin[:, harmonic_idx, :],
                phase_cos[:, harmonic_idx, :],
            ]
        )
    if topk < requested_harmonics:
        zeros = torch.zeros_like(offset)
        for _ in range(requested_harmonics - topk):
            features.extend([zeros, zeros, zeros, zeros])
    return torch.cat(features, dim=-1)


class TimingPhaseConditionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        if phase_features.dim() != 2:
            raise ValueError(f"expected phase_features [B,C], got {tuple(phase_features.shape)}")
        return self.net(phase_features).unsqueeze(1)


class TimingBranchFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        fusion_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.mix = nn.Sequential(
            nn.Linear(hidden_dim * 2, fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(fusion_hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        event_hidden: torch.Tensor | None,
        phase_hidden: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if event_hidden is None and phase_hidden is None:
            return None
        if event_hidden is None:
            return phase_hidden
        if phase_hidden is None:
            return event_hidden
        if event_hidden.shape != phase_hidden.shape:
            raise ValueError(
                f"event/phase hidden shape mismatch: {tuple(event_hidden.shape)} vs {tuple(phase_hidden.shape)}"
            )
        event_token = event_hidden.squeeze(1)
        phase_token = phase_hidden.squeeze(1)
        fused = torch.cat([event_token, phase_token], dim=-1)
        gate = self.gate(fused)
        mixed = self.mix(fused)
        out = gate * event_token + (1.0 - gate) * phase_token + mixed
        return out.unsqueeze(1)
