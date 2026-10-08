from __future__ import annotations

from typing import Sequence

import torch


FEATURE_DIMS = {
    "hip": 3,
    "root": 3,
    "root_vel": 3,
    "yaw": 1,
}


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def compute_contact_timing_features(joints: torch.Tensor) -> dict[str, torch.Tensor]:
    if joints.dim() != 4 or joints.shape[-1] != 3:
        raise ValueError(f"expected joints [B,T,J,3], got {tuple(joints.shape)}")

    root = joints[:, :, 0]
    left_hip = joints[:, :, 2]
    right_hip = joints[:, :, 1]

    across = left_hip - right_hip
    across_xz = across[..., [0, 2]]
    across_xz = across_xz / torch.linalg.norm(across_xz, dim=-1, keepdim=True).clamp_min(1e-6)

    forward_xz = torch.stack([-across_xz[..., 1], across_xz[..., 0]], dim=-1)
    yaw = torch.atan2(forward_xz[..., 0], forward_xz[..., 1]).to(dtype=joints.dtype)

    root_vel = torch.zeros_like(root)
    root_vel[:, 1:] = root[:, 1:] - root[:, :-1]

    delta_yaw = torch.zeros_like(yaw)
    if joints.shape[1] > 1:
        delta_yaw[:, 1:] = wrap_angle(yaw[:, 1:] - yaw[:, :-1]).to(dtype=joints.dtype)
        delta_yaw[:, 0] = delta_yaw[:, 1]

    hip = torch.zeros((joints.shape[0], joints.shape[1], 3), dtype=joints.dtype, device=joints.device)
    if joints.shape[1] > 1:
        dx = root_vel[:, 1:, 0]
        dz = root_vel[:, 1:, 2]
        c = torch.cos(-yaw[:, :-1])
        s = torch.sin(-yaw[:, :-1])
        hip[:, 1:, 0] = c * dx - s * dz
        hip[:, 1:, 1] = s * dx + c * dz
        hip[:, 1:, 2] = delta_yaw[:, 1:]
        hip[:, 0] = hip[:, 1]

    return {
        "hip": hip,
        "yaw": yaw.unsqueeze(-1),
        "root": root,
        "root_vel": root_vel,
    }


def build_contact_timing_feature_tensor(
    joints: torch.Tensor,
    feature_names: Sequence[str],
) -> torch.Tensor:
    features = compute_contact_timing_features(joints)
    chunks: list[torch.Tensor] = []
    for name in feature_names:
        key = str(name)
        if key not in features:
            raise KeyError(f"unsupported contact timing feature: {key}")
        chunks.append(features[key])
    if not chunks:
        raise ValueError("feature_names cannot be empty")
    return torch.cat(chunks, dim=-1)


def build_root_trajectory_tensor(joints: torch.Tensor) -> torch.Tensor:
    features = compute_contact_timing_features(joints)
    root = features["root"]
    yaw = features["yaw"]
    return torch.cat([root[..., [0, 2]], yaw], dim=-1)


def normalize_root_trajectory_tensor(
    trajectory: torch.Tensor,
    *,
    align_xz: bool = True,
    align_yaw: bool = True,
) -> torch.Tensor:
    if trajectory.dim() != 3 or trajectory.shape[-1] != 3:
        raise ValueError(f"expected trajectory [B,T,3], got {tuple(trajectory.shape)}")
    traj = trajectory.clone()
    if align_xz:
        traj[..., :2] = traj[..., :2] - traj[:, :1, :2]
    if align_yaw:
        traj[..., 2:3] = wrap_angle(traj[..., 2:3] - traj[:, :1, 2:3])
    return traj


def compute_binary_foot_contact_from_joints(
    joints: torch.Tensor,
    *,
    left_indices: Sequence[int] = (10,),
    right_indices: Sequence[int] = (11,),
    velocity_threshold: float = 0.02,
    height_threshold: float = 0.05,
) -> torch.Tensor:
    if joints.dim() != 4 or joints.shape[-1] != 3:
        raise ValueError(f"expected joints [B,T,J,3], got {tuple(joints.shape)}")

    def _group_contact(indices: Sequence[int]) -> torch.Tensor:
        feet = joints[:, :, tuple(int(idx) for idx in indices), :]
        velocity = torch.zeros(
            (feet.shape[0], feet.shape[1], feet.shape[2]),
            dtype=feet.dtype,
            device=feet.device,
        )
        velocity[:, 1:] = torch.linalg.norm(feet[:, 1:] - feet[:, :-1], dim=-1)
        height = feet[..., 1]
        contact = torch.logical_and(
            velocity < float(velocity_threshold),
            height < float(height_threshold),
        )
        return contact.any(dim=-1).to(dtype=joints.dtype)

    left_contact = _group_contact(left_indices)
    right_contact = _group_contact(right_indices)
    return torch.stack([left_contact, right_contact], dim=-1)
