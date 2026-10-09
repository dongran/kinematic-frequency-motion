from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class GroundingMeta:
    mode: str
    enabled: bool
    auto_shift: float
    manual_shift: float
    total_shift: float
    target_z: float
    estimated_floor_z: float | None
    clamp_min: float
    clamp_max: float
    bottom_percentile: float
    contact_height_percentile: float
    contact_velocity_percentile: float
    contact_frames: int

    def to_dict(self) -> dict[str, float | int | bool | str | None]:
        return asdict(self)


def parse_shift_clamp(text: str) -> tuple[float, float]:
    vals = [float(x.strip()) for x in str(text).split(",")]
    if len(vals) != 2:
        raise ValueError(f"Expected two comma-separated clamp values, got: {text}")
    lo, hi = vals
    if lo > hi:
        raise ValueError(f"Invalid clamp range: {text}")
    return float(lo), float(hi)


def apply_auto_ground(
    vertices_bl: np.ndarray,
    *,
    mode: str = "off",
    manual_shift: float = 0.0,
    target_z: float = -0.005,
    clamp: tuple[float, float] = (-0.08, 0.04),
    bottom_percentile: float = 0.2,
    contact_height_percentile: float = 20.0,
    contact_velocity_percentile: float = 65.0,
    min_contact_frames: int = 8,
    fps: float = 20.0,
) -> tuple[np.ndarray, GroundingMeta]:
    """Apply robust vertical grounding to Blender-coordinate vertices.

    The estimate uses the low-Z vertex percentile per frame instead of the
    single global minimum, which is less sensitive to toe tips or noisy vertices.
    """
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"0", "false", "none"}:
        mode_norm = "off"
    if mode_norm not in {"off", "quantile", "feet", "auto"}:
        raise ValueError(f"Unknown auto ground mode: {mode}")

    out = np.asarray(vertices_bl, dtype=np.float32).copy()
    lo, hi = clamp
    auto_shift = 0.0
    estimated_floor_z: float | None = None
    contact_frames = 0

    if mode_norm != "off":
        if out.ndim != 3 or out.shape[-1] != 3 or out.shape[0] == 0:
            raise ValueError(f"Expected vertices shape (T,V,3), got {out.shape}")
        z = out[..., 2].astype(np.float32, copy=False)
        low_z = np.percentile(z, float(bottom_percentile), axis=1)
        if low_z.shape[0] > 1:
            vel = np.empty_like(low_z)
            vel[1:] = np.abs(np.diff(low_z)) * float(fps)
            vel[0] = vel[1]
        else:
            vel = np.zeros_like(low_z)

        height_thr = float(np.percentile(low_z, float(contact_height_percentile)))
        vel_thr = float(np.percentile(vel, float(contact_velocity_percentile)))
        contact = (low_z <= height_thr) & (vel <= vel_thr)
        if int(contact.sum()) < int(min_contact_frames):
            contact = low_z <= height_thr
        if int(contact.sum()) < int(min_contact_frames):
            n = min(max(1, int(min_contact_frames)), low_z.shape[0])
            contact = np.zeros_like(low_z, dtype=bool)
            contact[np.argsort(low_z)[:n]] = True

        contact_frames = int(contact.sum())
        estimated_floor_z = float(np.median(low_z[contact]))
        auto_shift = float(np.clip(float(target_z) - estimated_floor_z, lo, hi))

    total_shift = float(auto_shift + float(manual_shift))
    if total_shift != 0.0:
        out[..., 2] += total_shift

    meta = GroundingMeta(
        mode=mode_norm,
        enabled=mode_norm != "off",
        auto_shift=float(auto_shift),
        manual_shift=float(manual_shift),
        total_shift=float(total_shift),
        target_z=float(target_z),
        estimated_floor_z=estimated_floor_z,
        clamp_min=float(lo),
        clamp_max=float(hi),
        bottom_percentile=float(bottom_percentile),
        contact_height_percentile=float(contact_height_percentile),
        contact_velocity_percentile=float(contact_velocity_percentile),
        contact_frames=int(contact_frames),
    )
    return out.astype(np.float32, copy=False), meta
