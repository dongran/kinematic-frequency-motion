from __future__ import annotations

import torch
import torch.nn.functional as F


def _expanded_time_mask(tensor, lengths):
    max_len = tensor.shape[-1]
    view_shape = [1] * tensor.ndim
    view_shape[-1] = max_len
    time_index = torch.arange(max_len, device=tensor.device).view(*view_shape)
    length_shape = [tensor.shape[0]] + [1] * (tensor.ndim - 1)
    valid_lengths = torch.as_tensor(lengths, device=tensor.device).view(*length_shape)
    return (time_index < valid_lengths).float().expand_as(tensor)


def _expand_selective_mask(tensor, lengths, *, band_indices=None, dof_slice=None, dof_indices=None):
    mask = _expanded_time_mask(tensor, lengths)
    if tensor.ndim < 4:
        return mask
    if band_indices is not None:
        band_mask = torch.zeros(tensor.shape[1], device=tensor.device, dtype=tensor.dtype)
        if len(band_indices) > 0:
            band_mask[torch.as_tensor(band_indices, device=tensor.device, dtype=torch.long)] = 1.0
        mask = mask * band_mask.view(1, -1, 1, 1)
    if dof_slice is not None or dof_indices is not None:
        dof_mask = torch.zeros(tensor.shape[2], device=tensor.device, dtype=tensor.dtype)
        if dof_slice is not None:
            start, end = int(dof_slice[0]), int(dof_slice[1])
            dof_mask[start:end] = 1.0
        if dof_indices is not None:
            if len(dof_indices) > 0:
                dof_mask[torch.as_tensor(dof_indices, device=tensor.device, dtype=torch.long)] = 1.0
        mask = mask * dof_mask.view(1, 1, -1, 1)
    return mask


def masked_l1(pred, target, lengths):
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    mask = _expanded_time_mask(pred, lengths)
    denom = mask.sum().clamp_min(1.0)
    return (torch.abs(pred - target) * mask).sum() / denom


def masked_mse(pred, target, lengths):
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    mask = _expanded_time_mask(pred, lengths)
    denom = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def masked_l1_dof(pred, target, lengths, dof_slice):
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    if pred.ndim < 3:
        raise ValueError(f"Expected tensor with a DOF axis, got shape {pred.shape}")
    start, end = int(dof_slice[0]), int(dof_slice[1])
    if end <= start:
        raise ValueError(f"Invalid dof slice: {dof_slice}")
    return masked_l1(pred[..., start:end, :], target[..., start:end, :], lengths)


def masked_l1_selective(
    pred,
    target,
    lengths,
    *,
    band_indices=None,
    dof_slice=None,
    dof_indices=None,
):
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    mask = _expand_selective_mask(
        pred,
        lengths,
        band_indices=band_indices,
        dof_slice=dof_slice,
        dof_indices=dof_indices,
    )
    denom = mask.sum().clamp_min(1.0)
    return (torch.abs(pred - target) * mask).sum() / denom


def hht_feature_loss(detector, pred_imfs, target_imfs, lengths):
    pred_amp, pred_freq = detector.compute_hht_features(
        pred_imfs.reshape(pred_imfs.shape[0], -1, pred_imfs.shape[-1])
    )
    target_amp, target_freq = detector.compute_hht_features(
        target_imfs.reshape(target_imfs.shape[0], -1, target_imfs.shape[-1])
    )
    pred_amp = pred_amp.reshape(pred_imfs.shape)
    target_amp = target_amp.reshape(target_imfs.shape)
    pred_freq = pred_freq.reshape(pred_imfs.shape[0], pred_imfs.shape[1], pred_imfs.shape[2], -1)
    target_freq = target_freq.reshape(target_imfs.shape[0], target_imfs.shape[1], target_imfs.shape[2], -1)
    freq_lengths = torch.clamp(lengths - 1, min=1)
    amp_loss = masked_l1(pred_amp, target_amp, lengths)
    freq_loss = masked_l1(pred_freq, target_freq, freq_lengths)
    return amp_loss, freq_loss


def cosine_global_loss(pred_global, target_global):
    pred_flat = pred_global.reshape(-1, pred_global.shape[-1])
    target_flat = target_global.reshape(-1, target_global.shape[-1])
    labels = torch.ones(pred_flat.shape[0], device=pred_flat.device)
    return F.cosine_embedding_loss(pred_flat, target_flat.detach(), labels)
