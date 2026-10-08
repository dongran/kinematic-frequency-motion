from __future__ import annotations

import torch
import torch.nn as nn


class FrequencyBranch(nn.Module):
    def __init__(self, motion_dim: int = 263, imf_dim: int = 189, hidden_dim: int = 512):
        super().__init__()
        self.motion_encoder = nn.Sequential(
            nn.Conv1d(motion_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.imf_encoder = nn.Sequential(
            nn.Conv1d(imf_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.motion_proj = nn.Linear(hidden_dim, hidden_dim)
        self.imf_proj = nn.Linear(hidden_dim, hidden_dim)

    def _pool(self, feats, lengths):
        max_len = feats.shape[-1]
        mask = (torch.arange(max_len, device=feats.device)[None, :] < lengths[:, None]).float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (feats * mask[:, None, :]).sum(dim=-1) / denom
        return pooled

    def _mask_motion(self, motion, feature_indices=None):
        if feature_indices is None:
            return motion
        if len(feature_indices) == 0:
            return torch.zeros_like(motion)
        index_tensor = torch.as_tensor(feature_indices, device=motion.device, dtype=torch.long)
        masked = torch.zeros_like(motion)
        masked[..., index_tensor] = motion[..., index_tensor]
        return masked

    def encode_motion(self, motion, lengths, feature_indices=None):
        motion = self._mask_motion(motion, feature_indices=feature_indices)
        feats = self.motion_encoder(motion.transpose(1, 2))
        pooled = self._pool(feats, lengths)
        return self.motion_proj(pooled).unsqueeze(1)

    def _mask_bands(self, imfs, band_indices=None):
        if band_indices is None:
            return imfs
        if imfs.ndim != 4:
            raise ValueError(f"Expected IMF tensor shape [B, bands, dof, T], got {tuple(imfs.shape)}")
        if len(band_indices) == 0:
            return torch.zeros_like(imfs)
        mask = torch.zeros(imfs.shape[1], device=imfs.device, dtype=imfs.dtype)
        mask[band_indices] = 1.0
        return imfs * mask.view(1, -1, 1, 1)

    def encode_imfs(self, imfs, lengths, band_indices=None):
        masked = self._mask_bands(imfs, band_indices=band_indices)
        flat = masked.reshape(masked.shape[0], -1, masked.shape[-1])
        feats = self.imf_encoder(flat)
        pooled = self._pool(feats, lengths)
        return self.imf_proj(pooled).unsqueeze(1)
