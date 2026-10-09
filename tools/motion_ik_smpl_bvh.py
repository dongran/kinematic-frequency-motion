#!/usr/bin/env python3
"""
HumanML-263 features or 22 joints to a SMPL npz through joints2smpl, plus an optional BVH.
"""

from __future__ import annotations

import pickle
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Literal

warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")

import numpy as np

# NumPy 2.0 dropped aliases such as np.float_. Restore them only when missing.
for _name, _val in [
    ("bool", getattr(np, "bool_", bool)),
    ("int", getattr(np, "int_", np.int64)),
    ("float", getattr(np, "float_", np.float64)),
    ("complex", getattr(np, "complex_", np.complex128)),
    ("object", object),
    ("str", str),
    ("unicode", str),
    ("nan", np.nan),
    ("inf", np.inf),
]:
    if not hasattr(np, _name):
        setattr(np, _name, _val)

import torch

IkVariant = Literal["clean", "patched"]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_smpl_path(user_smpl_path: str, joints2smpl_dir: Path) -> str:
    for cand in [
        user_smpl_path or "",
        str(REPO_ROOT / "third_party" / "smpl"),
        str(REPO_ROOT / "deps" / "smpl_models" / "smpl"),
        str(REPO_ROOT / "deps" / "smpl_models" / "motionclip" / "smpl"),
        str(REPO_ROOT / "deps" / "motionclip" / "smpl"),
        str(joints2smpl_dir / "smpl_models" / "smpl"),
    ]:
        if cand and (Path(cand) / "SMPL_NEUTRAL.pkl").exists():
            return cand
    raise FileNotFoundError(
        "SMPL_NEUTRAL.pkl was not found. Pass smpl_path, "
        "or place the file in third_party/smpl/."
    )


def resolve_smpl2bvh_model_path(smpl_path: str) -> str:
    smpl_dir = Path(smpl_path)
    if smpl_dir.name == "smpl" and (smpl_dir / "SMPL_NEUTRAL.pkl").exists():
        return str(smpl_dir.parent)
    if (smpl_dir / "smpl").is_dir():
        return str(smpl_dir)
    return smpl_path


def resolve_mesh_model_path(smpl_path: str) -> str:
    """Return the parent directory that contains smpl/SMPL_NEUTRAL.pkl."""
    return resolve_smpl2bvh_model_path(smpl_path)


def load_feats263_array(data: np.ndarray) -> np.ndarray:
    if data.ndim == 3:
        data = data[0]
    if data.shape[-1] != 263:
        raise ValueError(f"Expected a 263-D feature, got shape={data.shape}")
    return data.astype(np.float32)


def feats263_to_joints22_humanml(data: np.ndarray, joints_num: int = 22) -> np.ndarray:
    from mld.data.humanml.scripts.motion_process import recover_from_ric

    tensor = torch.from_numpy(data).to("cpu")
    joints = recover_from_ric(tensor, joints_num).cpu().numpy()
    return joints.astype(np.float32)


def humanml3d_to_smpl_joints(
    joints: np.ndarray,
    *,
    flip_y: bool = False,
    flip_z: bool = False,
) -> np.ndarray:
    out = np.stack([-joints[..., 0], joints[..., 2], joints[..., 1]], axis=-1)
    if flip_y:
        out[..., 1] *= -1
    if flip_z:
        out[..., 2] *= -1
    return out


def prepare_joints_for_ik(
    joints_humanml22: np.ndarray,
    variant: IkVariant,
    *,
    flip_y_joints: bool = False,
    flip_z_joints: bool = False,
) -> np.ndarray:
    if variant == "clean":
        return joints_humanml22.astype(np.float32)
    return humanml3d_to_smpl_joints(
        joints_humanml22, flip_y=flip_y_joints, flip_z=flip_z_joints
    )


def yup_to_zup(
    poses72: np.ndarray,
    trans: np.ndarray,
    *,
    transform_trans: bool = True,
    trans_sign: str = "xzy",
    flip_forward: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    from mld.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle

    if transform_trans:
        if trans_sign == "xzy":
            trans_zup = np.stack([trans[:, 0], trans[:, 2], trans[:, 1]], axis=-1)
        else:
            trans_zup = np.stack([trans[:, 0], -trans[:, 2], trans[:, 1]], axis=-1)
    else:
        trans_zup = trans.copy()

    poses_24 = poses72.reshape(poses72.shape[0], 24, 3)
    root_aa = poses_24[:, 0:1, :]
    root_mat = axis_angle_to_matrix(torch.from_numpy(root_aa)).numpy()

    rad_x = np.deg2rad(-90.0)
    rx = np.array([rad_x, 0.0, 0.0], dtype=np.float32)
    rx_mat = axis_angle_to_matrix(
        torch.from_numpy(rx[np.newaxis, np.newaxis, :])
    ).numpy()
    rx_mat = np.broadcast_to(rx_mat, root_mat.shape)
    combined = np.matmul(rx_mat, root_mat)

    if flip_forward:
        rad_z = np.deg2rad(180.0)
        rz = np.array([0.0, 0.0, rad_z], dtype=np.float32)
        rz_mat = axis_angle_to_matrix(
            torch.from_numpy(rz[np.newaxis, np.newaxis, :])
        ).numpy()
        rz_mat = np.broadcast_to(rz_mat, combined.shape)
        combined = np.matmul(rz_mat, combined)
        trans_zup = np.stack(
            [-trans_zup[:, 0], -trans_zup[:, 1], trans_zup[:, 2]], axis=-1
        )

    poses_24 = poses_24.copy()
    poses_24[:, 0, :] = matrix_to_axis_angle(
        torch.from_numpy(combined.squeeze(1))
    ).numpy()
    return poses_24, trans_zup


def postprocess_smpl_pose_trans(
    poses72: np.ndarray,
    trans: np.ndarray,
    variant: IkVariant,
    *,
    no_zup_trans: bool = False,
    trans_sign: str = "xzy",
    flip_forward: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if variant == "clean":
        rots24 = poses72.reshape(poses72.shape[0], 24, 3).astype(np.float32)
        return rots24, trans.astype(np.float32)
    rots24, trans_out = yup_to_zup(
        poses72,
        trans,
        transform_trans=not no_zup_trans,
        trans_sign=trans_sign,
        flip_forward=flip_forward,
    )
    return rots24.astype(np.float32), trans_out.astype(np.float32)


def _load_joints2smpl_init(
    joints2smpl_dir: Path,
    smpl_path: str,
    device: str,
    num_iters: int,
    *,
    use_collision: bool = True,
):
    j2s_src = joints2smpl_dir / "src"
    if str(j2s_src) not in sys.path:
        sys.path.insert(0, str(j2s_src))
    if str(joints2smpl_dir) not in sys.path:
        sys.path.insert(0, str(joints2smpl_dir))

    import config as j2s_config
    from smplify import SMPLify3D
    import smplx

    orig_config = {
        "SMPL_MODEL_DIR": getattr(j2s_config, "SMPL_MODEL_DIR", None),
        "GMM_MODEL_DIR": getattr(j2s_config, "GMM_MODEL_DIR", None),
        "SMPL_MEAN_FILE": getattr(j2s_config, "SMPL_MEAN_FILE", None),
        "Part_Seg_DIR": getattr(j2s_config, "Part_Seg_DIR", None),
    }

    smpl_base = Path(smpl_path)
    if (smpl_base / "SMPL_NEUTRAL.pkl").exists():
        j2s_config.SMPL_MODEL_DIR = str(smpl_base.parent)
    elif (smpl_base / "smpl" / "SMPL_NEUTRAL.pkl").exists():
        j2s_config.SMPL_MODEL_DIR = str(smpl_base)
    else:
        raise FileNotFoundError(f"SMPL_NEUTRAL.pkl was not found under: {smpl_path}")

    j2s_config.GMM_MODEL_DIR = str(joints2smpl_dir / "smpl_models")
    if use_collision:
        try:
            import mesh_intersection  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "SMPLify collision is on, but this Python environment has no "
                "`mesh_intersection`. Install that package, "
                "or pass --no_collision / USE_COLLISION=0."
            ) from e
        part_segm = joints2smpl_dir / "smpl_models" / "smplx_parts_segm.pkl"
        if not part_segm.is_file():
            raise FileNotFoundError(
                "Collision needs smplx_parts_segm.pkl, which was not found: "
                f"{part_segm}"
            )
        j2s_config.Part_Seg_DIR = str(part_segm)
    mean_file = None
    for cand in [
        joints2smpl_dir / "smpl_models" / "neutral_smpl_mean_params.h5",
        REPO_ROOT / "deps" / "smpl_models" / "neutral_smpl_mean_params.h5",
    ]:
        if cand.exists():
            mean_file = cand
            break
    if mean_file is not None:
        j2s_config.SMPL_MEAN_FILE = str(mean_file)

    if str(device).startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(device)
    else:
        dev = torch.device("cpu")

    smpl_model = smplx.create(
        j2s_config.SMPL_MODEL_DIR,
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        batch_size=1,
    ).to(dev)

    smplify = SMPLify3D(
        smplxmodel=smpl_model,
        batch_size=1,
        joints_category="AMASS",
        num_iters=num_iters,
        device=dev,
        use_collision=use_collision,
    )

    init_mean_pose = torch.zeros(1, 72, device=dev)
    init_mean_shape = torch.zeros(1, 10, device=dev)
    try:
        import h5py

        mean_file_path = Path(j2s_config.SMPL_MEAN_FILE)
        if mean_file_path.exists():
            with h5py.File(mean_file_path, "r") as f:
                init_mean_pose = (
                    torch.from_numpy(f["pose"][:]).unsqueeze(0).float().to(dev)
                )
                init_mean_shape = (
                    torch.from_numpy(f["shape"][:]).unsqueeze(0).float().to(dev)
                )
    except ImportError:
        pass

    return dev, smplify, init_mean_pose, init_mean_shape, orig_config


def _restore_joints2smpl_config(orig_config: dict) -> None:
    import config as j2s_config

    for key, value in orig_config.items():
        if value is not None:
            setattr(j2s_config, key, value)


def fit_joints_sequence(
    joints_3d: np.ndarray,
    joints2smpl_dir: Path,
    smpl_path: str,
    device: str,
    num_iters: int,
    *,
    log_every: int = 20,
    use_collision: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dev, smplify, init_mean_pose, init_mean_shape, orig_config = _load_joints2smpl_init(
        joints2smpl_dir=joints2smpl_dir,
        smpl_path=smpl_path,
        device=device,
        num_iters=num_iters,
        use_collision=use_collision,
    )

    try:
        num_frames = joints_3d.shape[0]
        poses = np.zeros((num_frames, 72), dtype=np.float32)
        trans = np.zeros((num_frames, 3), dtype=np.float32)
        betas = np.zeros((num_frames, 10), dtype=np.float32)

        pred_pose = init_mean_pose.clone()
        pred_betas = init_mean_shape.clone()
        pred_cam_t = torch.zeros(1, 3, device=dev)
        keypoints_3d = torch.zeros(1, 22, 3, device=dev)
        confidence = torch.ones(22, device=dev)

        for idx in range(num_frames):
            keypoints_3d[0] = torch.from_numpy(joints_3d[idx]).float().to(dev)

            if idx > 0:
                pred_pose = torch.from_numpy(poses[idx - 1]).unsqueeze(0).float().to(dev)
                pred_betas = torch.from_numpy(betas[idx - 1]).unsqueeze(0).float().to(dev)
                pred_cam_t = torch.from_numpy(trans[idx - 1]).unsqueeze(0).float().to(dev)

            _, _, new_pose, new_betas, new_cam_t, _ = smplify(
                pred_pose.detach(),
                pred_betas.detach(),
                pred_cam_t.detach(),
                keypoints_3d,
                conf_3d=confidence,
                seq_ind=idx,
            )

            poses[idx] = new_pose.detach().cpu().numpy().squeeze()
            trans[idx] = new_cam_t.detach().cpu().numpy().squeeze()
            betas[idx] = new_betas.detach().cpu().numpy().squeeze()

            if log_every > 0 and ((idx + 1) % log_every == 0 or idx == num_frames - 1):
                print(f"  joints2smpl: {idx + 1}/{num_frames} frames")

        return poses, trans, betas
    finally:
        _restore_joints2smpl_config(orig_config)


def save_smpl_npz_bvh(
    out_dir: Path,
    joints22_humanml: np.ndarray,
    rots24: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray,
    fps: int,
    smpl_path: str,
    *,
    skip_bvh: bool = False,
) -> tuple[Path, Path | None, Path]:
    """
    Write smpl_poses.npz (with a batch axis, for smpl2bvh), smpl_poses_mesh.npz
    (T, 24, 3, for the mesh renderer), and an optional motion.bvh.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "joints22.npy", joints22_humanml)

    smpl_npz = out_dir / "smpl_poses.npz"
    np.savez(
        smpl_npz,
        poses=rots24[np.newaxis].astype(np.float32),
        trans=trans[np.newaxis].astype(np.float32),
        betas=betas[-1],
        fps=float(fps),
    )

    mesh_npz = out_dir / "smpl_poses_mesh.npz"
    np.savez(
        mesh_npz,
        poses=rots24.astype(np.float32),
        trans=trans.astype(np.float32),
    )

    bvh_path: Path | None = None
    if not skip_bvh:
        sys.path.insert(0, str(REPO_ROOT / "third_party" / "smpl2bvh"))
        from smpl2bvh import smpl2bvh

        bvh_path = out_dir / "motion.bvh"
        smpl2bvh(
            model_path=resolve_smpl2bvh_model_path(smpl_path),
            poses=str(smpl_npz),
            output=str(bvh_path),
            mirror=False,
            model_type="smpl",
            gender="NEUTRAL",
            num_betas=10,
            fps=fps,
        )

    return smpl_npz, bvh_path, mesh_npz


def run_ik_pipeline_from_joints22(
    joints_humanml22: np.ndarray,
    out_dir: Path,
    *,
    smpl_path: str,
    joints2smpl_dir: Path,
    device: str,
    num_iters: int,
    fps: int,
    variant: IkVariant = "clean",
    skip_bvh: bool = False,
    flip_y_joints: bool = False,
    flip_z_joints: bool = False,
    no_zup_trans: bool = False,
    trans_sign: str = "xzy",
    flip_forward: bool = False,
    max_frames: int = 0,
    log_every: int = 20,
    use_collision: bool = True,
) -> tuple[Path, Path | None, Path]:
    j = joints_humanml22.astype(np.float32)
    if max_frames > 0:
        j = j[:max_frames]
    j_ik = prepare_joints_for_ik(
        j, variant, flip_y_joints=flip_y_joints, flip_z_joints=flip_z_joints
    )
    poses72, trans, betas = fit_joints_sequence(
        j_ik,
        joints2smpl_dir=joints2smpl_dir,
        smpl_path=smpl_path,
        device=device,
        num_iters=num_iters,
        log_every=log_every,
        use_collision=use_collision,
    )
    rots24, trans_out = postprocess_smpl_pose_trans(
        poses72,
        trans,
        variant,
        no_zup_trans=no_zup_trans,
        trans_sign=trans_sign,
        flip_forward=flip_forward,
    )
    return save_smpl_npz_bvh(
        out_dir,
        j,
        rots24,
        trans_out,
        betas,
        fps,
        smpl_path,
        skip_bvh=skip_bvh,
    )


def run_ik_pipeline_from_feats263(
    feats263: np.ndarray,
    out_dir: Path,
    **kwargs: Any,
) -> tuple[Path, Path | None, Path]:
    data = load_feats263_array(feats263)
    max_frames = int(kwargs.get("max_frames", 0))
    if max_frames > 0:
        data = data[:max_frames]
    joints = feats263_to_joints22_humanml(data)
    return run_ik_pipeline_from_joints22(joints, out_dir, **kwargs)


def ensure_joints22(joints: np.ndarray) -> np.ndarray:
    """Return joints as [T, 22, 3]. A [T, 263] array is converted first."""
    if joints.ndim == 2:
        return feats263_to_joints22_humanml(joints)
    if joints.shape[-1] == 3 and joints.shape[-2] >= 22:
        return joints[..., :22, :].astype(np.float32)
    return joints.astype(np.float32)


def align_joints_length(
    *arrays: np.ndarray, target_len: int
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for a in arrays:
        T = a.shape[0]
        if T >= target_len:
            out.append(a[:target_len].copy())
        else:
            pad = np.tile(a[-1:], (target_len - T, 1, 1))
            out.append(np.concatenate([a, pad], axis=0))
    return out


def load_style_transfer_pkl(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_pair_stems_from_id(id_str: str) -> tuple[str, str] | None:
    """
    Parse an id of the form content<stem>_style<stem>_scale_...
    """
    if "_style" not in id_str:
        return None
    pre, post = id_str.split("_style", 1)
    if not pre.startswith("content"):
        return None
    content_stem = pre[len("content") :]
    style_part = post.split("_scale_", 1)[0]
    return content_stem, style_part


def sanitize_dir_component(name: str, max_len: int = 120) -> str:
    s = re.sub(r"[^\w.\-]+", "_", name)
    s = s.strip("._")
    if len(s) > max_len:
        s = s[:max_len]
    return s or "pair"


def stems_from_pkl_entry(
    pkl_data: dict[str, Any], index: int
) -> tuple[str, str] | None:
    """Read the content and style stems from the id string."""
    ids = pkl_data.get("id")
    if ids is None or index >= len(ids):
        return None
    return parse_pair_stems_from_id(str(ids[index]))
