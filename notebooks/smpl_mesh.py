"""Simple SMPL mesh preview from raw HumanML-263 features.

This is not a Blender render. It runs the neutral SMPL layer and draws the
mesh with matplotlib. Root rotation is the HumanML yaw. Local poses are the
21 continuous 6D joint rotations stored in the feature, with the two SMPL
hand joints left at zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mld_clean.data.release_stats import recover_root_rot_pos
from mld_clean.utils.rotation_conversions import quaternion_to_matrix, rotation_6d_to_matrix

ROT_START = 4 + 21 * 3
ROT_END = ROT_START + 21 * 6


def raw_feats_to_smpl_params(feats: torch.Tensor):
    """feats: raw, unnormalized [T, 263]. Returns rotation matrices."""
    root_quat, root_pos = recover_root_rot_pos(feats)
    global_orient = quaternion_to_matrix(root_quat).unsqueeze(-3)
    rot6d = feats[..., ROT_START:ROT_END].reshape(*feats.shape[:-1], 21, 6)
    local_mats = rotation_6d_to_matrix(rot6d)
    eye = torch.eye(3, device=feats.device, dtype=feats.dtype)
    body = eye.view(1, 1, 3, 3).expand(feats.shape[0], 23, 3, 3).clone()
    body[:, :21, :, :] = local_mats
    return global_orient, body, root_pos


def _patch_numpy_for_chumpy() -> None:
    import numpy as np

    aliases = {
        "bool": np.bool_,
        "int": np.int_,
        "float": np.float64,
        "complex": np.complex128,
        "object": np.object_,
        "str": np.str_,
        "unicode": np.str_,
    }
    for name, value in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def load_smpl_layer(smpl_pkl: str | Path, device: torch.device):
    _patch_numpy_for_chumpy()
    from smplx import SMPLLayer

    layer = SMPLLayer(
        model_path=str(smpl_pkl),
        gender="neutral",
        num_betas=10,
        pose2rot=True,
    )
    return layer.to(device)


def feats_to_vertices(feats: np.ndarray, smpl_pkl: str | Path, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    torch_device = torch.device(device)
    layer = load_smpl_layer(smpl_pkl, torch_device)
    raw = torch.as_tensor(feats, dtype=torch.float32, device=torch_device)
    global_orient, body_pose, transl = raw_feats_to_smpl_params(raw)
    betas = torch.zeros(raw.shape[0], layer.num_betas, device=torch_device)
    with torch.no_grad():
        output = layer(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            betas=betas,
        )
    vertices = output.vertices.detach().cpu().numpy()
    faces = np.asarray(layer.faces, dtype=np.int64)
    return vertices, faces


def save_mesh_gif(
    vertices: np.ndarray,
    faces: np.ndarray,
    out_path: str | Path,
    *,
    title: str = "SMPL",
    fps: int = 20,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(np.ceil(len(vertices) / 80)))
    frame_ids = list(range(0, len(vertices), stride))
    # Matplotlib's vertical axis is z. SMPL and HumanML use y as up.
    plotted = np.stack([vertices[..., 0], vertices[..., 2], vertices[..., 1]], axis=-1)
    shown = plotted[frame_ids]
    flat = shown.reshape(-1, 3)
    center = (flat.min(axis=0) + flat.max(axis=0)) / 2.0
    span = max(float((flat.max(axis=0) - flat.min(axis=0)).max()), 1.2)
    half = span / 2.0
    face_ids = faces[::2]

    fig = plt.figure(figsize=(4.2, 4.6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_id):
        ax.cla()
        verts = plotted[frame_id]
        tris = verts[face_ids]
        collection = Poly3DCollection(tris, alpha=0.95)
        collection.set_facecolor((0.86, 0.72, 0.62, 1.0))
        collection.set_edgecolor((0.35, 0.28, 0.24, 0.15))
        ax.add_collection3d(collection)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.view_init(elev=18, azim=-70)
        ax.set_axis_off()
        ax.set_title(f"{title}  frame {frame_id}")
        return [collection]

    anim = FuncAnimation(fig, update, frames=frame_ids, interval=1000 / fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=max(1, int(round(fps / stride)))))
    plt.close(fig)
    return out_path
