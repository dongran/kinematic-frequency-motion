from __future__ import annotations

import torch
import torch.nn.functional as F


def _expand_mask(mask: torch.Tensor, target_dim: int) -> torch.Tensor:
    while mask.dim() < target_dim:
        mask = mask.unsqueeze(-1)
    return mask


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = _expand_mask(mask.to(device=pred.device, dtype=pred.dtype), pred.dim())
    loss = (pred - target).pow(2)
    return (loss * mask).sum() / mask.expand_as(loss).sum().clamp_min(1.0)


def masked_bce_with_probs(
    pred_prob: torch.Tensor,
    target_prob: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred_prob = pred_prob.clamp(1.0e-4, 1.0 - 1.0e-4)
    mask = _expand_mask(mask.to(device=pred_prob.device, dtype=pred_prob.dtype), pred_prob.dim())
    loss = F.binary_cross_entropy(pred_prob, target_prob, reduction="none")
    return (loss * mask).sum() / mask.expand_as(loss).sum().clamp_min(1.0)


def masked_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = _expand_mask(mask.to(device=pred.device, dtype=pred.dtype), pred.dim())
    loss = torch.abs(pred - target)
    return (loss * mask).sum() / mask.expand_as(loss).sum().clamp_min(1.0)
