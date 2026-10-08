"""Visualization-only foot-contact cleanup for HumanML 22-joint motions.

The optimizer locks detected foot contacts, keeps bone lengths, and stays close
to the generated pose. It is the post-process used before rendering. Quantitative
metrics in the paper are computed on the raw generated joints, before this step.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

# HumanML3D / T2M kinematic chain.
T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]
LEFT_FOOT = (8, 11)
RIGHT_FOOT = (7, 10)
FOOT_JOINTS = LEFT_FOOT + RIGHT_FOOT


def _parents(njoints: int) -> np.ndarray:
    parents = np.full((njoints,), -1, dtype=np.int64)
    for chain in T2M_KINEMATIC_CHAIN:
        for i in range(1, len(chain)):
            parents[chain[i]] = chain[i - 1]
    return parents


def floor_height_from_joints(joints: np.ndarray) -> float:
    foot_y = joints[:, list(FOOT_JOINTS), 1]
    return float(np.quantile(foot_y, 0.05))


def compute_binary_contact(
    joints: np.ndarray,
    *,
    floor_height: float,
    velocity_threshold: float,
    height_threshold: float,
) -> np.ndarray:
    def group_contact(indices: tuple[int, int]) -> np.ndarray:
        foot = joints[:, indices]
        velocity = np.zeros((foot.shape[0], foot.shape[1]), dtype=np.float32)
        velocity[1:] = np.linalg.norm(foot[1:] - foot[:-1], axis=-1)
        relative_height = foot[..., 1] - floor_height
        return np.logical_and(
            velocity < velocity_threshold,
            relative_height < height_threshold,
        ).any(axis=1)

    return np.stack(
        [group_contact(LEFT_FOOT), group_contact(RIGHT_FOOT)],
        axis=-1,
    )


def _fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            out[i:j] = True
        i = j
    return out


def _remove_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i
        while j < n and out[j]:
            j += 1
        if (j - i) < min_len:
            out[i:j] = False
        i = j
    return out


def stabilize_contact(contact: np.ndarray, *, min_contact_frames: int, gap_fill: int) -> np.ndarray:
    out = contact.copy()
    for side in range(out.shape[1]):
        out[:, side] = _remove_short_runs(
            _fill_short_gaps(out[:, side], gap_fill),
            min_contact_frames,
        )
    return out


def _segment_slices(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    n = len(mask)
    start = 0
    while start < n:
        while start < n and not mask[start]:
            start += 1
        if start >= n:
            break
        end = start
        while end + 1 < n and mask[end + 1]:
            end += 1
        yield start, end + 1
        start = end + 1


def build_foot_targets(
    joints: np.ndarray,
    contact: np.ndarray,
    *,
    floor_height: float,
    floor_blend: float,
) -> tuple[np.ndarray, np.ndarray]:
    targets = joints.copy()
    mask = np.zeros((joints.shape[0], joints.shape[1]), dtype=bool)
    for side, indices in enumerate((LEFT_FOOT, RIGHT_FOOT)):
        for start, end in _segment_slices(contact[:, side]):
            for idx in indices:
                anchor = joints[start:end, idx].mean(axis=0).astype(np.float32)
                anchor[1] = (1.0 - floor_blend) * anchor[1] + floor_blend * floor_height
                targets[start:end, idx] = anchor
                mask[start:end, idx] = True
    return targets.astype(np.float32), mask


def fix_foot_sliding(
    joints: np.ndarray,
    *,
    velocity_threshold: float = 0.045,
    height_threshold: float = 0.08,
    min_contact_frames: int = 4,
    gap_fill: int = 2,
    floor_blend: float = 1.0,
    steps: int = 800,
    lr: float = 0.03,
    device: str = "cpu",
    w_foot: float = 80.0,
    w_bone: float = 60.0,
    w_pose: float = 8.0,
    w_root: float = 20.0,
    w_vel: float = 8.0,
    w_accel: float = 2.0,
    w_floor: float = 20.0,
) -> tuple[np.ndarray, dict]:
    """Return foot-fixed joints and a small before/after contact summary.

    Defaults match the visualization post-process used for the paper figures.
    """
    raw = np.asarray(joints, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1] < 22 or raw.shape[2] != 3:
        raise ValueError(f"Expected joints [T, 22, 3], got {raw.shape}")
    raw = raw[:, :22]

    floor_height = floor_height_from_joints(raw)
    contact = stabilize_contact(
        compute_binary_contact(
            raw,
            floor_height=floor_height,
            velocity_threshold=velocity_threshold,
            height_threshold=height_threshold,
        ),
        min_contact_frames=min_contact_frames,
        gap_fill=gap_fill,
    )
    targets, target_mask = build_foot_targets(
        raw,
        contact,
        floor_height=floor_height,
        floor_blend=floor_blend,
    )

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    raw_t = torch.from_numpy(raw).to(dev)
    targets_t = torch.from_numpy(targets).to(dev)
    target_mask_t = torch.from_numpy(target_mask).to(dev)
    floor_t = torch.tensor(float(floor_height), dtype=raw_t.dtype, device=dev)
    parents = _parents(raw.shape[1])
    edges = [(j, int(p)) for j, p in enumerate(parents) if p >= 0]
    child_idx = torch.tensor([j for j, _ in edges], dtype=torch.long, device=dev)
    parent_idx = torch.tensor([p for _, p in edges], dtype=torch.long, device=dev)

    raw_bone = torch.linalg.norm(raw_t[:, child_idx] - raw_t[:, parent_idx], dim=-1)
    raw_vel = raw_t[1:] - raw_t[:-1]
    raw_accel = raw_t[2:] - 2 * raw_t[1:-1] + raw_t[:-2]

    x = torch.nn.Parameter(raw_t.clone())
    opt = torch.optim.Adam([x], lr=lr)
    pose_weights = torch.ones((1, raw.shape[1], 1), dtype=raw_t.dtype, device=dev)
    pose_weights[:, list(FOOT_JOINTS), :] = 0.25
    root_weight = torch.zeros_like(pose_weights)
    root_weight[:, 0:1, :] = 1.0

    for _ in range(int(steps)):
        opt.zero_grad()
        if target_mask_t.any():
            foot_loss = ((x[target_mask_t] - targets_t[target_mask_t]) ** 2).mean()
        else:
            foot_loss = x.new_zeros(())
        bone = torch.linalg.norm(x[:, child_idx] - x[:, parent_idx], dim=-1)
        bone_loss = ((bone - raw_bone) ** 2).mean()
        pose_loss = (((x - raw_t) ** 2) * pose_weights).mean()
        root_loss = (((x - raw_t) ** 2) * root_weight).mean()
        vel_loss = (((x[1:] - x[:-1]) - raw_vel) ** 2).mean()
        if x.shape[0] > 2:
            accel_loss = (((x[2:] - 2 * x[1:-1] + x[:-2]) - raw_accel) ** 2).mean()
        else:
            accel_loss = x.new_zeros(())
        floor_loss = torch.relu(floor_t - x[:, list(FOOT_JOINTS), 1]).pow(2).mean()
        loss = (
            w_foot * foot_loss
            + w_bone * bone_loss
            + w_pose * pose_loss
            + w_root * root_loss
            + w_vel * vel_loss
            + w_accel * accel_loss
            + w_floor * floor_loss
        )
        loss.backward()
        opt.step()

    fixed = x.detach().cpu().numpy().astype(np.float32)
    info = {
        "floor_height": floor_height,
        "left_contact_frames": int(contact[:, 0].sum()),
        "right_contact_frames": int(contact[:, 1].sum()),
        "mean_joint_delta": float(np.linalg.norm(fixed - raw, axis=-1).mean()),
    }
    return fixed, info
