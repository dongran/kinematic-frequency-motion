"""Normalization stats and HumanML joint recovery for the public demo.

The released motion-transfer demo does not need the full FineMotion dataset.
It only needs the mean and standard deviation that the released weights were
trained with, plus the standard HumanML rotation-invariant joint recovery.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from mld_clean.models.architectures.utils.quaternion import qinv, qrot


def recover_root_rot_pos(data: torch.Tensor):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), device=data.device, dtype=data.dtype)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,), device=data.device, dtype=data.dtype)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data: torch.Tensor, joints_num: int) -> torch.Tensor:
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)),
        positions,
    )
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    return torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)


class ReleaseMotionStats:
    """Stand-in for the training datamodule used by the diffusion model."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, njoints: int = 22):
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        self.hparams = SimpleNamespace(
            mean=mean,
            std=std,
            mean_eval=mean,
            std_eval=std,
        )
        self.nfeats = int(mean.shape[-1])
        self.njoints = int(njoints)
        self.is_mm = False

    def feats2joints(self, features: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.hparams.mean, device=features.device, dtype=features.dtype)
        std = torch.as_tensor(self.hparams.std, device=features.device, dtype=features.dtype)
        return recover_from_ric(features * std + mean, self.njoints)

    def joints2feats(self, features):
        raise NotImplementedError(
            "joints2feats is not part of the released motion-transfer demo."
        )

    def renorm4t2m(self, features: torch.Tensor) -> torch.Tensor:
        return features
