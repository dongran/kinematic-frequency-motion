from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F

try:
    import ptwt
except ImportError:
    ptwt = None


def _resolve_band_masks(freqs: torch.Tensor, band_edges_hz) -> Dict[str, torch.Tensor]:
    low_max, mid_max, high_max = [float(v) for v in band_edges_hz]
    nyquist = float(freqs[-1].item()) if freqs.numel() > 0 else 0.0
    high_cut = min(high_max, nyquist)
    return {
        "all": torch.ones_like(freqs, dtype=torch.bool),
        "low": freqs <= low_max,
        "mid": (freqs > low_max) & (freqs <= mid_max),
        "high": (freqs > mid_max) & (freqs <= high_cut),
    }


def _init_loss_dict(device: torch.device, names=None) -> Dict[str, torch.Tensor]:
    zero = torch.tensor(0.0, device=device)
    keys = names if names is not None else ("all", "low", "mid", "high")
    return {name: zero.clone() for name in keys}


def _finalize_loss_dict(losses: Dict[str, torch.Tensor], counts: Dict[str, int]) -> Dict[str, torch.Tensor]:
    final = {}
    ref = next(iter(losses.values()))
    for name, value in losses.items():
        denom = max(counts[name], 1)
        final[name] = value / denom if counts[name] > 0 else torch.zeros_like(ref)
    return final


def _prepare_motion_pair(pred_motion: torch.Tensor, target_motion: torch.Tensor, length: int):
    pred_seq = pred_motion[:length].transpose(0, 1).float()
    target_seq = target_motion[:length].transpose(0, 1).float()
    return pred_seq, target_seq


def _group_indices_by_length(lengths, *, min_length: int) -> Dict[int, list[int]]:
    groups: Dict[int, list[int]] = {}
    for idx, length in enumerate(lengths):
        length_i = int(length)
        if length_i < min_length:
            continue
        groups.setdefault(length_i, []).append(idx)
    return groups


def _magnitude_transform(values: torch.Tensor, log_magnitude: bool) -> torch.Tensor:
    magnitudes = torch.abs(values)
    return torch.log1p(magnitudes) if log_magnitude else magnitudes


def _build_haar_filters(num_channels: int, device: torch.device, dtype: torch.dtype):
    scale = 1.0 / math.sqrt(2.0)
    low = torch.tensor([scale, scale], device=device, dtype=dtype).view(1, 1, 2)
    high = torch.tensor([scale, -scale], device=device, dtype=dtype).view(1, 1, 2)
    return low.repeat(num_channels, 1, 1), high.repeat(num_channels, 1, 1)


def _haar_dwt_components(sequence: torch.Tensor, levels: int) -> Dict[str, torch.Tensor]:
    current = sequence.unsqueeze(0)
    components: Dict[str, torch.Tensor] = {}
    for level in range(1, max(int(levels), 1) + 1):
        if current.shape[-1] < 2:
            break
        if current.shape[-1] % 2 == 1:
            current = current[..., :-1]
        channels = current.shape[1]
        low_filter, high_filter = _build_haar_filters(channels, current.device, current.dtype)
        approx = F.conv1d(current, low_filter, stride=2, groups=channels)
        detail = F.conv1d(current, high_filter, stride=2, groups=channels)
        components[f"detail_l{level}"] = detail.squeeze(0)
        current = approx
    if components:
        components[f"approx_l{len(components)}"] = current.squeeze(0)
    return components


def fft_spectral_losses(
    pred_motion: torch.Tensor,
    target_motion: torch.Tensor,
    lengths,
    *,
    sample_rate: float,
    band_edges_hz,
    log_magnitude: bool = True,
) -> Dict[str, torch.Tensor]:
    device = pred_motion.device
    losses = _init_loss_dict(device)
    counts = {name: 0 for name in losses}

    for idx, length in enumerate(lengths):
        length_i = int(length)
        if length_i < 2:
            continue
        pred_seq, target_seq = _prepare_motion_pair(pred_motion[idx], target_motion[idx], length_i)
        pred_fft = torch.fft.rfft(pred_seq, dim=-1, norm="ortho")
        target_fft = torch.fft.rfft(target_seq, dim=-1, norm="ortho")
        pred_mag = _magnitude_transform(pred_fft, log_magnitude)
        target_mag = _magnitude_transform(target_fft, log_magnitude)
        diff = torch.abs(pred_mag - target_mag)
        freqs = torch.fft.rfftfreq(length_i, d=1.0 / float(sample_rate), device=device)
        for name, mask in _resolve_band_masks(freqs, band_edges_hz).items():
            if mask.any():
                losses[name] = losses[name] + diff[:, mask].mean()
                counts[name] += 1

    return _finalize_loss_dict(losses, counts)


def stft_spectral_losses(
    pred_motion: torch.Tensor,
    target_motion: torch.Tensor,
    lengths,
    *,
    sample_rate: float,
    band_edges_hz,
    n_fft: int,
    hop_length: int,
    win_length: int,
    log_magnitude: bool = True,
) -> Dict[str, torch.Tensor]:
    device = pred_motion.device
    losses = _init_loss_dict(device)
    counts = {name: 0 for name in losses}

    for idx, length in enumerate(lengths):
        length_i = int(length)
        if length_i < 8:
            continue
        pred_seq, target_seq = _prepare_motion_pair(pred_motion[idx], target_motion[idx], length_i)
        n_fft_i = min(int(n_fft), length_i)
        if n_fft_i < 8:
            continue
        win_length_i = min(int(win_length), n_fft_i)
        hop_length_i = min(int(hop_length), max(1, win_length_i // 2))
        window = torch.hann_window(win_length_i, device=device)
        pred_stft = torch.stft(
            pred_seq,
            n_fft=n_fft_i,
            hop_length=hop_length_i,
            win_length=win_length_i,
            window=window,
            center=False,
            return_complex=True,
        )
        target_stft = torch.stft(
            target_seq,
            n_fft=n_fft_i,
            hop_length=hop_length_i,
            win_length=win_length_i,
            window=window,
            center=False,
            return_complex=True,
        )
        pred_mag = _magnitude_transform(pred_stft, log_magnitude)
        target_mag = _magnitude_transform(target_stft, log_magnitude)
        diff = torch.abs(pred_mag - target_mag)
        freqs = torch.fft.rfftfreq(n_fft_i, d=1.0 / float(sample_rate), device=device)
        for name, mask in _resolve_band_masks(freqs, band_edges_hz).items():
            if mask.any():
                losses[name] = losses[name] + diff[:, mask, :].mean()
                counts[name] += 1

    return _finalize_loss_dict(losses, counts)


def wavelet_spectral_losses(
    pred_motion: torch.Tensor,
    target_motion: torch.Tensor,
    lengths,
    *,
    levels: int = 3,
    log_magnitude: bool = True,
) -> Dict[str, torch.Tensor]:
    device = pred_motion.device
    levels_i = max(int(levels), 1)
    component_names = [f"approx_l{levels_i}"] + [
        f"detail_l{level}" for level in range(levels_i, 0, -1)
    ]
    losses = _init_loss_dict(device, names=["all", *component_names])
    counts = {name: 0 for name in losses}

    for idx, length in enumerate(lengths):
        length_i = int(length)
        if length_i < 2 ** levels_i:
            continue
        pred_seq, target_seq = _prepare_motion_pair(pred_motion[idx], target_motion[idx], length_i)
        pred_coeffs = _haar_dwt_components(pred_seq, levels_i)
        target_coeffs = _haar_dwt_components(target_seq, levels_i)
        if not pred_coeffs or not target_coeffs:
            continue

        sample_component_losses = []
        for name in component_names:
            pred_coeff = pred_coeffs.get(name)
            target_coeff = target_coeffs.get(name)
            if pred_coeff is None or target_coeff is None:
                continue
            pred_mag = _magnitude_transform(pred_coeff, log_magnitude)
            target_mag = _magnitude_transform(target_coeff, log_magnitude)
            value = torch.abs(pred_mag - target_mag).mean()
            losses[name] = losses[name] + value
            counts[name] += 1
            sample_component_losses.append(value)

        if sample_component_losses:
            losses["all"] = losses["all"] + torch.stack(sample_component_losses).mean()
            counts["all"] += 1

    return _finalize_loss_dict(losses, counts)


def wavelet_spectral_losses_ptwt(
    pred_motion: torch.Tensor,
    target_motion: torch.Tensor,
    lengths,
    *,
    levels: int = 3,
    wavelet: str = "haar",
    mode: str = "reflect",
    log_magnitude: bool = True,
) -> Dict[str, torch.Tensor]:
    if ptwt is None:
        raise ImportError(
            "ptwt is required for wavelet_spectral_losses_ptwt. Install it with `pip install ptwt`."
        )

    device = pred_motion.device
    levels_i = max(int(levels), 1)
    component_names = [f"approx_l{levels_i}"] + [
        f"detail_l{level}" for level in range(levels_i, 0, -1)
    ]
    losses = _init_loss_dict(device, names=["all", *component_names])
    counts = {name: 0 for name in losses}

    length_groups = _group_indices_by_length(lengths, min_length=2 ** levels_i)
    for length_i, indices in length_groups.items():
        batch_index = torch.as_tensor(indices, device=device, dtype=torch.long)
        pred_group = pred_motion.index_select(0, batch_index)[:, :length_i, :].transpose(1, 2).float()
        target_group = target_motion.index_select(0, batch_index)[:, :length_i, :].transpose(1, 2).float()

        pred_coeffs = ptwt.wavedec(
            pred_group,
            wavelet,
            mode=mode,
            level=levels_i,
            axis=-1,
        )
        target_coeffs = ptwt.wavedec(
            target_group,
            wavelet,
            mode=mode,
            level=levels_i,
            axis=-1,
        )
        if not pred_coeffs or not target_coeffs:
            continue

        sample_component_losses = []

        pred_mag = _magnitude_transform(pred_coeffs[0], log_magnitude)
        target_mag = _magnitude_transform(target_coeffs[0], log_magnitude)
        approx_loss = torch.abs(pred_mag - target_mag).flatten(1).mean(dim=1)
        losses[f"approx_l{levels_i}"] = losses[f"approx_l{levels_i}"] + approx_loss.sum()
        counts[f"approx_l{levels_i}"] += approx_loss.numel()
        sample_component_losses.append(approx_loss)

        for level in range(levels_i, 0, -1):
            coeff_idx = levels_i - level + 1
            pred_mag = _magnitude_transform(pred_coeffs[coeff_idx], log_magnitude)
            target_mag = _magnitude_transform(target_coeffs[coeff_idx], log_magnitude)
            detail_loss = torch.abs(pred_mag - target_mag).flatten(1).mean(dim=1)
            name = f"detail_l{level}"
            losses[name] = losses[name] + detail_loss.sum()
            counts[name] += detail_loss.numel()
            sample_component_losses.append(detail_loss)

        if sample_component_losses:
            all_loss = torch.stack(sample_component_losses, dim=0).mean(dim=0)
            losses["all"] = losses["all"] + all_loss.sum()
            counts["all"] += all_loss.numel()

    return _finalize_loss_dict(losses, counts)
