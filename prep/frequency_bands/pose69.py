"""SMPL-24 clips to the 69-D kinematic signal used for MEMD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class LoadedSMPL24:
    poses: np.ndarray  # (T, 24, 3) axis-angle
    trans: np.ndarray  # (T, 3)
    src_fps: float


def load_smpl24_npz(path: Path, *, default_fps: float = 30.0) -> LoadedSMPL24:
    with np.load(str(path), allow_pickle=False) as blob:
        if "poses" not in blob.files:
            raise KeyError(f"missing poses in {path} (keys={list(blob.files)})")
        poses = np.asarray(blob["poses"], dtype=np.float32)
        if poses.ndim != 3 or poses.shape[1:] != (24, 3):
            raise ValueError(f"{path}: expected poses (T, 24, 3), got {poses.shape}")
        if "trans" not in blob.files:
            raise KeyError(f"missing trans in {path}")
        trans = np.asarray(blob["trans"], dtype=np.float32)
        if trans.shape != (poses.shape[0], 3):
            raise ValueError(f"{path}: bad trans shape {trans.shape} for poses {poses.shape}")
        if "mocap_framerate" in blob.files:
            fps = float(np.asarray(blob["mocap_framerate"]).reshape(-1)[0])
        elif "fps" in blob.files:
            fps = float(np.asarray(blob["fps"]).reshape(-1)[0])
        else:
            fps = float(default_fps)
    if fps <= 0:
        fps = float(default_fps)
    return LoadedSMPL24(poses=poses, trans=trans, src_fps=fps)


def _aa_to_quat_wxyz(aa: np.ndarray) -> np.ndarray:
    ang = np.linalg.norm(aa, axis=-1, keepdims=True)
    half = 0.5 * ang
    axis = aa / (ang + 1e-8)
    axis = np.where(ang < 1e-8, 0.0, axis)
    xyz = axis * np.sin(half)
    w = np.cos(half)
    return np.concatenate([w, xyz], axis=-1).astype(np.float32, copy=False)


def _quat_wxyz_to_aa(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / (n + 1e-8)
    w = np.clip(q[..., 0:1], -1.0, 1.0)
    xyz = q[..., 1:4]
    ang = 2.0 * np.arccos(w)
    s = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    axis = xyz / (s + 1e-8)
    axis = np.where(s < 1e-6, 0.0, axis)
    return (axis * ang).astype(np.float32, copy=False)


def _slerp_quat_wxyz(q0: np.ndarray, q1: np.ndarray, w: np.ndarray) -> np.ndarray:
    q0 = q0 / (np.linalg.norm(q0, axis=-1, keepdims=True) + 1e-8)
    q1 = q1 / (np.linalg.norm(q1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    flip = dot < 0.0
    q1 = np.where(flip, -q1, q1)
    dot = np.clip(np.where(flip, -dot, dot), -1.0, 1.0)
    w = np.asarray(w, dtype=np.float32)
    while w.ndim < dot.ndim:
        w = w[..., None]
    close = dot > 0.9995
    theta0 = np.arccos(dot)
    sin0 = np.sin(theta0)
    s0 = np.sin((1.0 - w) * theta0) / (sin0 + 1e-8)
    s1 = np.sin(w * theta0) / (sin0 + 1e-8)
    out = np.where(close, (1.0 - w) * q0 + w * q1, s0 * q0 + s1 * q1)
    out = out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-8)
    return out.astype(np.float32, copy=False)


def _dest_index(num_src: int, src_fps: float, dst_fps: float) -> np.ndarray:
    duration = (num_src - 1) / float(src_fps)
    num_dst = max(2, int(np.floor(duration * float(dst_fps))) + 1)
    index = (np.arange(num_dst, dtype=np.float64) / float(dst_fps)) * float(src_fps)
    return np.clip(index, 0.0, float(num_src - 1))


def resample_poses_slerp(poses: np.ndarray, *, src_fps: float, dst_fps: float) -> np.ndarray:
    """Resample axis-angle poses (T, J, 3) onto the 20 Hz grid used for HumanML-263."""
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[-1] != 3:
        raise ValueError(f"expected poses (T, J, 3), got {poses.shape}")
    num_src = int(poses.shape[0])
    if num_src <= 1 or abs(float(src_fps) - float(dst_fps)) < 1e-8:
        return poses.copy()
    index = _dest_index(num_src, src_fps, dst_fps)
    i0 = np.floor(index).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, num_src - 1)
    weight = (index - i0.astype(np.float64)).astype(np.float32)
    quat = _aa_to_quat_wxyz(poses)
    out = _quat_wxyz_to_aa(_slerp_quat_wxyz(quat[i0], quat[i1], weight[:, None, None]))
    return out.reshape(index.shape[0], poses.shape[1], 3).astype(np.float32, copy=False)


def resample_translation(trans: np.ndarray, *, src_fps: float, dst_fps: float) -> np.ndarray:
    trans = np.asarray(trans, dtype=np.float32)
    num_src = int(trans.shape[0])
    if num_src <= 1 or abs(float(src_fps) - float(dst_fps)) < 1e-8:
        return trans.copy()
    index = _dest_index(num_src, src_fps, dst_fps)
    src_index = np.arange(num_src, dtype=np.float64)
    out = np.zeros((index.shape[0], 3), dtype=np.float32)
    for axis in range(3):
        out[:, axis] = np.interp(index, src_index, trans[:, axis]).astype(np.float32)
    return out


def build_pose69(poses: np.ndarray, trans: np.ndarray, *, src_fps: float, dst_fps: float) -> np.ndarray:
    """Return (T, 23, 3): root rotation, 21 body joints, root translation, at dst_fps."""
    poses_dst = resample_poses_slerp(poses, src_fps=src_fps, dst_fps=dst_fps)
    trans_dst = resample_translation(trans, src_fps=src_fps, dst_fps=dst_fps)
    num = min(int(poses_dst.shape[0]), int(trans_dst.shape[0]))
    root_rot = poses_dst[:num, 0:1, :]
    body = poses_dst[:num, 1:22, :]
    trans3 = trans_dst[:num, None, :]
    signal = np.concatenate([root_rot, body, trans3], axis=1)
    if signal.shape[1:] != (23, 3):
        raise ValueError(f"unexpected pose69 shape: {signal.shape}")
    return signal.astype(np.float32, copy=False)
