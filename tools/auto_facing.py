from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass
class FacingMeta:
    mode: str
    enabled: bool
    initial_facing_deg: float | None
    target_facing_deg: float
    auto_yaw_deg: float
    joint_frames: int

    def to_dict(self) -> dict[str, float | int | bool | str | None]:
        return asdict(self)


def _wrap_degrees(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def _rotate_vertices_z(vertices: np.ndarray, yaw_deg: float, center_xy: np.ndarray | None = None) -> np.ndarray:
    if abs(float(yaw_deg)) < 1e-7:
        return np.asarray(vertices, dtype=np.float32).copy()
    out = np.asarray(vertices, dtype=np.float32).copy()
    yaw = math.radians(float(yaw_deg))
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    if center_xy is None:
        center_xy = out[..., :2].mean(axis=(0, 1))
    xy = out[..., :2] - center_xy.reshape(1, 1, 2)
    out[..., :2] = np.einsum("...i,ij->...j", xy, rot.T) + center_xy.reshape(1, 1, 2)
    return out


def _estimate_initial_facing_deg(joints_bl: np.ndarray, *, frames: int) -> tuple[float | None, int]:
    if joints_bl.ndim != 3 or joints_bl.shape[-1] != 3:
        return None, 0
    n = min(max(1, int(frames)), joints_bl.shape[0])
    js = joints_bl[:n]
    pairs: list[tuple[int, int]] = []
    # SMPL/SMPLH common first joints: L/R hip = 1/2, L/R shoulder = 16/17.
    if js.shape[1] > 17:
        pairs.append((16, 17))
    if js.shape[1] > 2:
        pairs.append((1, 2))
    if not pairs:
        return None, 0

    right_vecs = []
    for left_idx, right_idx in pairs:
        right_vecs.append(js[:, right_idx, :2] - js[:, left_idx, :2])
    right_vec = np.concatenate(right_vecs, axis=0).mean(axis=0)
    norm = float(np.linalg.norm(right_vec))
    if norm < 1e-8:
        return None, n
    right_vec = right_vec / norm
    # Blender Z-up right-handed coords: forward = Z axis cross body-right.
    forward = np.array([-right_vec[1], right_vec[0]], dtype=np.float32)
    return float(math.degrees(math.atan2(float(forward[1]), float(forward[0])))), n


def apply_auto_facing(
    vertices_bl: np.ndarray,
    joints_bl: np.ndarray | None,
    *,
    mode: str = "off",
    target_facing_deg: float = -90.0,
    estimate_frames: int = 12,
) -> tuple[np.ndarray, FacingMeta]:
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"0", "false", "none"}:
        mode_norm = "off"
    if mode_norm not in {"off", "front", "camera", "auto"}:
        raise ValueError(f"Unknown auto facing mode: {mode}")

    initial_deg: float | None = None
    joint_frames = 0
    auto_yaw = 0.0
    out = np.asarray(vertices_bl, dtype=np.float32).copy()
    if mode_norm != "off":
        if joints_bl is None:
            raise ValueError("auto_facing requires joints_bl")
        initial_deg, joint_frames = _estimate_initial_facing_deg(joints_bl, frames=int(estimate_frames))
        if initial_deg is None:
            raise ValueError("Failed to estimate initial facing direction from joints")
        auto_yaw = _wrap_degrees(float(target_facing_deg) - float(initial_deg))
        out = _rotate_vertices_z(out, auto_yaw)

    meta = FacingMeta(
        mode=mode_norm,
        enabled=mode_norm != "off",
        initial_facing_deg=initial_deg,
        target_facing_deg=float(target_facing_deg),
        auto_yaw_deg=float(auto_yaw),
        joint_frames=int(joint_frames),
    )
    return out.astype(np.float32, copy=False), meta
