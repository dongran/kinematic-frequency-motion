#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Tuple

import imageio.v2 as imageio
import numpy as np
import smplx
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.auto_grounding import apply_auto_ground, parse_shift_clamp  # noqa: E402
from tools.auto_facing import apply_auto_facing  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from blender_cycles_prefs import default_render_device_token, normalize_render_device  # noqa: E402
DEFAULT_SOTA_RENDER = (
    Path("/home/randong/MCM-LDM/decoupling_contact/SOTAtest-HHT-Motion/tools")
    / "render_finemotion_mesh_from_segids_mp4_smplh_hands.py"
)
DEFAULT_BLENDER_BIN = ROOT_DIR / "deps" / "blender-3.6.17-linux-x64" / "blender"
DEFAULT_BLENDER_SCRIPT = ROOT_DIR / "tools" / "blender_render_vertices_shadow.py"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export SMPLH vertices for a FineMotion clip and render shadows with headless Blender."
    )
    ap.add_argument(
        "--dataset_root",
        type=str,
        default=str(ROOT_DIR / "datasets" / "FineMotion" / "humanml3d_20hz_for_Tranning263" / "finemotion_263_v4"),
        help="FineMotion 20Hz dataset root containing new_joint_vecs.",
    )
    ap.add_argument(
        "--feature_vec_root",
        type=str,
        default="",
        help="Optional direct directory containing <id>.npy; defaults to dataset_root/new_joint_vecs.",
    )
    ap.add_argument(
        "--finemotion_root",
        type=str,
        default=str(ROOT_DIR / "datasets" / "FineMotion"),
        help="FineMotion 30Hz source root.",
    )
    ap.add_argument("--ids", type=str, nargs="+", required=True, help="Motion ids to export/render.")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory for outputs.")
    ap.add_argument(
        "--smplh_model_root",
        type=str,
        required=True,
        help="SMPLH model root or direct SMPLH_NEUTRAL.npz path.",
    )
    ap.add_argument(
        "--hand_preset",
        type=str,
        default="handpose_zip_4_400_tight65",
        help="Fixed hand preset name from render_finemotion_mesh_from_segids_mp4_smplh_hands.py.",
    )
    ap.add_argument("--hand_preset_scale", type=float, default=1.0)
    ap.add_argument("--target_fps", type=float, default=20.0)
    ap.add_argument("--out_fps", type=float, default=0.0, help="0 = target_fps / stride")
    ap.add_argument("--src_fps_override", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--preset", type=str, default="diag", choices=["front", "top", "diag", "lookdown", "bird"])
    ap.add_argument("--elev_deg", type=float, default=8.0)
    ap.add_argument("--azim_deg", type=float, default=270.0)
    ap.add_argument("--cam_dist", type=float, default=3.2)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--camera_mode", type=str, default="track", choices=["track", "fixed"])
    ap.add_argument("--camera_projection", type=str, default="perspective", choices=["perspective", "orthographic"])
    ap.add_argument("--ortho_scale", type=float, default=0.0)
    ap.add_argument("--fixed_anchor", type=str, default="first", choices=["first", "mean", "origin"])
    ap.add_argument("--yaw_deg", type=float, default=0.0, help="Actor yaw rotation in Blender degrees.")
    ap.add_argument("--quality_profile", type=str, default="standard", choices=["standard", "paper"])
    ap.add_argument("--samples", type=int, default=24, help="Cycles samples for Blender.")
    ap.add_argument(
        "--render_device",
        type=normalize_render_device,
        default=default_render_device_token(),
        help=(
            "Blender Cycles: cpu | cuda | optix | profile id "
            "(blender_machine_profiles.json / blender_machine_local.json). "
            "Default: $BLENDER_RENDER_DEVICE or $BLENDER_MACHINE_PROFILE or cuda."
        ),
    )
    ap.add_argument("--sun_energy", type=float, default=3.4)
    ap.add_argument("--sun_elev_deg", type=float, default=42.0)
    ap.add_argument("--sun_azim_deg", type=float, default=35.0)
    ap.add_argument("--sun_angle_deg", type=float, default=7.0, help="Larger means softer shadows.")
    ap.add_argument("--area_light_energy", type=float, default=0.0)
    ap.add_argument("--area_light_size", type=float, default=4.0)
    ap.add_argument("--area_light_offset", type=str, default="0.0,-2.8,3.2")
    ap.add_argument("--floor_mode", type=str, default="plane", choices=["plane", "none", "checker"])
    ap.add_argument("--plane_color", type=str, default="0.96,0.96,0.96,1.0")
    ap.add_argument("--perspective_fov_deg", type=float, default=45.0)
    ap.add_argument("--checker_color1", type=str, default="0.32,0.32,0.32,1.0")
    ap.add_argument("--checker_color2", type=str, default="0.52,0.52,0.52,1.0")
    ap.add_argument("--checker_scale", type=float, default=5.5)
    ap.add_argument("--checker_ref_plane", type=float, default=5.2)
    ap.add_argument("--no_checker_rescale_to_plane", action="store_true")
    ap.add_argument("--floor_roughness", type=float, default=0.88)
    ap.add_argument("--floor_min_size", type=float, default=6.0)
    ap.add_argument("--floor_motion_scale", type=float, default=2.6)
    ap.add_argument("--floor_pad_mesh", type=float, default=0.0)
    ap.add_argument("--mesh_color", type=str, default="0.66,0.66,0.70,1.0")
    ap.add_argument("--mesh_roughness", type=float, default=0.55)
    ap.add_argument("--mesh_subsurface", type=float, default=0.0)
    ap.add_argument("--mesh_z_shift", type=float, default=0.0)
    ap.add_argument(
        "--auto_ground",
        default="auto",
        choices=["off", "quantile", "feet", "auto"],
        help="Optional robust foot/floor estimate. Default auto keeps paper/showcase renders grounded.",
    )
    ap.add_argument("--ground_target_z", type=float, default=-0.005)
    ap.add_argument("--ground_shift_clamp", default="-0.08,0.04")
    ap.add_argument("--ground_bottom_percentile", type=float, default=0.2)
    ap.add_argument("--ground_contact_height_percentile", type=float, default=20.0)
    ap.add_argument("--ground_contact_velocity_percentile", type=float, default=65.0)
    ap.add_argument("--ground_min_contact_frames", type=int, default=8)
    ap.add_argument(
        "--auto_facing",
        default="camera",
        choices=["off", "front", "camera", "auto"],
        help="Rotate initial body facing to --facing_target_deg before Blender yaw trim. Default camera faces the diag/lookdown camera.",
    )
    ap.add_argument("--facing_target_deg", type=float, default=-90.0, help="-90 faces the default diag/lookdown camera.")
    ap.add_argument("--facing_estimate_frames", type=int, default=12)
    ap.add_argument("--world_strength", type=float, default=0.35)
    ap.add_argument("--white_background", action="store_true")
    ap.add_argument("--transparent_background", action="store_true")
    ap.add_argument(
        "--blender_bin",
        type=str,
        default=str(DEFAULT_BLENDER_BIN),
        help="Headless Blender binary path.",
    )
    ap.add_argument(
        "--blender_script",
        type=str,
        default=str(DEFAULT_BLENDER_SCRIPT),
        help="Background Blender Python render script.",
    )
    ap.add_argument(
        "--source_script",
        type=str,
        default=str(DEFAULT_SOTA_RENDER),
        help="Path to render_finemotion_mesh_from_segids_mp4_smplh_hands.py for helper reuse.",
    )
    ap.add_argument("--keep_frames", action="store_true", help="Keep rendered PNG frames.")
    ap.add_argument("--keep_assets", action="store_true", help="Keep intermediate vertices/faces/meta files.")
    ap.add_argument("--frames_only", action="store_true", help="Render PNG frames without writing an MP4.")
    ap.add_argument("--only_frame", type=int, default=-1, help=">=0 renders only that frame index after export.")
    return ap.parse_args()


def _parse_rgba(text: str) -> Tuple[float, float, float, float]:
    vals = [float(x.strip()) for x in str(text).split(",")]
    if len(vals) == 3:
        vals.append(1.0)
    if len(vals) != 4:
        raise ValueError(f"Expected 3 or 4 comma-separated values, got: {text}")
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])


def _load_source_module(script_path: Path) -> Any:
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing helper source script: {script_path}")
    spec = importlib.util.spec_from_file_location("sota_smplh_hands", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _camera_angles_from_preset(preset: str, elev_deg: float, azim_deg: float) -> Tuple[float, float]:
    preset = str(preset).strip().lower()
    if preset == "top":
        return 75.0, 180.0
    if preset == "front":
        return 8.0, 0.0
    # Looking down: about 27 degrees, same azimuth as the default diag view.
    if preset in ("lookdown", "bird"):
        return 27.0, 270.0
    return float(elev_deg), float(azim_deg)


def _camera_offset_blender(cam_dist: float, elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = math.radians(float(elev_deg))
    azim = math.radians(float(azim_deg))
    offset_y_up = np.array(
        [
            float(cam_dist) * math.sin(azim) * math.cos(elev),
            float(cam_dist) * math.sin(elev),
            float(cam_dist) * math.cos(azim) * math.cos(elev),
        ],
        dtype=np.float32,
    )
    return np.array(
        [offset_y_up[2], offset_y_up[0], offset_y_up[1]],
        dtype=np.float32,
    )


def _build_vertices_for_motion(
    *,
    mid: str,
    dataset_root: Path,
    feature_vec_root: Path | None,
    finemotion_root: Path,
    smplh_model_root: Path,
    hand_preset: str,
    hand_preset_scale: float,
    target_fps: float,
    out_fps: float,
    src_fps_override: float,
    stride: int,
    max_frames: int,
    batch_size: int,
    source_mod: Any,
    mesh_z_shift: float = 0.0,
    auto_ground: str = "auto",
    ground_target_z: float = -0.005,
    ground_shift_clamp: tuple[float, float] = (-0.08, 0.04),
    ground_bottom_percentile: float = 0.2,
    ground_contact_height_percentile: float = 20.0,
    ground_contact_velocity_percentile: float = 65.0,
    ground_min_contact_frames: int = 8,
    auto_facing: str = "camera",
    facing_target_deg: float = -90.0,
    facing_estimate_frames: int = 12,
) -> Tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    vec_root = feature_vec_root if feature_vec_root is not None else dataset_root / "new_joint_vecs"
    feat_path = vec_root / f"{mid}.npy"
    if not feat_path.is_file():
        raise FileNotFoundError(f"Missing feature npy for id={mid}: {feat_path}")
    feat = np.load(feat_path, mmap_mode="r")
    t_feat = int(feat.shape[0])
    if t_feat <= 0:
        raise ValueError(f"Feature npy is empty for id={mid}: {feat_path}")

    source_mid = mid[:-2] if mid.endswith("-0") else mid
    base_id, start_idx, _end_idx = source_mod._base_id(source_mid)
    src_npz = source_mod._infer_src_npz(finemotion_root, base_id)
    poses30, trans30, src_fps, betas10 = source_mod._load_npz_motion(src_npz)
    if src_fps_override > 0:
        src_fps = float(src_fps_override)

    t_tgt = (float(start_idx) + np.arange(t_feat, dtype=np.float32)) / float(target_fps)
    t_src = np.arange(poses30.shape[0], dtype=np.float32) / float(src_fps)
    t_tgt = np.clip(t_tgt, float(t_src[0]), float(t_src[-1]))
    poses20 = source_mod._interp_pose_rotvec_slerp(poses30, t_src=t_src, t_tgt=t_tgt)
    trans20 = source_mod._interp_trans(trans30, t_src=t_src, t_tgt=t_tgt)

    frame_ids = list(range(0, poses20.shape[0], max(1, int(stride))))
    if int(max_frames) > 0:
        frame_ids = frame_ids[: int(max_frames)]
    poses = poses20[frame_ids]
    trans = trans20[frame_ids]
    video_fps = float(out_fps) if float(out_fps) > 0 else float(target_fps) / float(max(1, int(stride)))

    use_pca, flat_hand_mean, num_pca_comps, left_hand_pose_np, right_hand_pose_np = source_mod._resolve_hand_pose_preset(
        hand_preset=hand_preset,
        hand_preset_scale=float(hand_preset_scale),
    )
    device = torch.device("cpu")
    max_bs = max(1, min(int(batch_size), poses.shape[0]))
    faces: Optional[np.ndarray] = None
    verts_all: list[np.ndarray] = []
    joints_all: list[np.ndarray] = []

    for chunk_start in range(0, poses.shape[0], max_bs):
        chunk_end = min(poses.shape[0], chunk_start + max_bs)
        curr_bs = chunk_end - chunk_start
        poses_chunk = poses[chunk_start:chunk_end]
        trans_chunk = trans[chunk_start:chunk_end]

        body_model = smplx.create(
            model_path=str(smplh_model_root),
            model_type="smplh",
            gender="neutral",
            ext="npz",
            flat_hand_mean=bool(flat_hand_mean),
            use_pca=bool(use_pca),
            num_pca_comps=int(num_pca_comps),
            batch_size=curr_bs,
        ).to(device)
        body_model.eval()
        if faces is None:
            faces = np.asarray(body_model.faces, dtype=np.int32)

        root_orient = torch.from_numpy(poses_chunk[:, 0, :]).to(device)
        body_pose = torch.from_numpy(poses_chunk[:, 1:22, :].reshape(curr_bs, -1)).to(device)
        transl = torch.from_numpy(trans_chunk).to(device)
        left_hand_pose = torch.from_numpy(
            np.repeat(left_hand_pose_np.reshape(1, -1), repeats=curr_bs, axis=0)
        ).to(device)
        right_hand_pose = torch.from_numpy(
            np.repeat(right_hand_pose_np.reshape(1, -1), repeats=curr_bs, axis=0)
        ).to(device)
        betas_curr = None
        if betas10 is not None:
            betas_np = np.asarray(betas10, dtype=np.float32).reshape(1, -1)
            betas_curr = torch.from_numpy(
                np.repeat(betas_np, repeats=curr_bs, axis=0)
            ).to(device)

        with torch.no_grad():
            out = body_model(
                global_orient=root_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                transl=transl,
                betas=betas_curr,
            )
        verts_all.append(out.vertices.detach().cpu().numpy().astype(np.float32))
        joints_all.append(out.joints.detach().cpu().numpy().astype(np.float32))

    if faces is None:
        raise RuntimeError("Failed to produce SMPLH faces.")
    vertices_y_up = np.concatenate(verts_all, axis=0)
    joints_y_up = np.concatenate(joints_all, axis=0)
    vertices_bl = vertices_y_up[..., [2, 0, 1]].copy()
    joints_bl = joints_y_up[..., [2, 0, 1]].copy()
    vertices_bl[..., 2] -= float(np.min(vertices_bl[..., 2]))
    vertices_bl, ground_meta = apply_auto_ground(
        vertices_bl,
        mode=auto_ground,
        manual_shift=float(mesh_z_shift),
        target_z=float(ground_target_z),
        clamp=ground_shift_clamp,
        bottom_percentile=float(ground_bottom_percentile),
        contact_height_percentile=float(ground_contact_height_percentile),
        contact_velocity_percentile=float(ground_contact_velocity_percentile),
        min_contact_frames=int(ground_min_contact_frames),
        fps=float(video_fps),
    )
    vertices_bl, facing_meta = apply_auto_facing(
        vertices_bl,
        joints_bl,
        mode=auto_facing,
        target_facing_deg=float(facing_target_deg),
        estimate_frames=int(facing_estimate_frames),
    )

    meta = {
        "sample_id": mid,
        "src_npz": str(src_npz),
        "feature_path": str(feat_path),
        "hand_preset": hand_preset,
        "hand_preset_scale": float(hand_preset_scale),
        "use_pca": bool(use_pca),
        "flat_hand_mean": bool(flat_hand_mean),
        "num_pca_comps": int(num_pca_comps),
        "target_fps": float(target_fps),
        "video_fps": float(video_fps),
        "stride": int(stride),
        "frames": int(vertices_bl.shape[0]),
        "mesh_z_shift": float(mesh_z_shift),
        "grounding": ground_meta.to_dict(),
        "facing": facing_meta.to_dict(),
    }
    return vertices_bl.astype(np.float32), faces, float(video_fps), meta


def _write_video_from_frames(frames_dir: Path, out_mp4: Path, fps: float) -> None:
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"No PNG frames found in: {frames_dir}")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_mp4),
        fps=float(fps),
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))
    finally:
        writer.close()


def _run_blender(
    *,
    blender_bin: Path,
    blender_script: Path,
    vertices_npy: Path,
    faces_npy: Path,
    frames_dir: Path,
    width: int,
    height: int,
    camera_mode: str,
    camera_projection: str,
    ortho_scale: float,
    fixed_anchor: str,
    samples: int,
    quality_profile: str,
    render_device: str,
    camera_offset: np.ndarray,
    yaw_deg: float,
    sun_energy: float,
    sun_elev_deg: float,
    sun_azim_deg: float,
    sun_angle_deg: float,
    area_light_energy: float,
    area_light_size: float,
    area_light_offset: str,
    floor_mode: str,
    plane_color: Tuple[float, float, float, float],
    mesh_color: Tuple[float, float, float, float],
    mesh_roughness: float,
    mesh_subsurface: float,
    world_strength: float,
    white_background: bool,
    transparent_background: bool,
    perspective_fov_deg: float = 45.0,
    checker_color1: Tuple[float, float, float, float] = (0.32, 0.32, 0.32, 1.0),
    checker_color2: Tuple[float, float, float, float] = (0.52, 0.52, 0.52, 1.0),
    checker_scale: float = 5.5,
    checker_ref_plane: float = 5.2,
    checker_rescale_to_plane: bool = True,
    floor_roughness: float = 0.88,
    floor_min_size: float = 6.0,
    floor_motion_scale: float = 2.6,
    floor_pad_mesh: float = 0.0,
    only_frame: int = -1,
) -> None:
    if not blender_bin.is_file():
        raise FileNotFoundError(f"Blender binary not found: {blender_bin}")
    if not blender_script.is_file():
        raise FileNotFoundError(f"Blender render script not found: {blender_script}")

    cmd = [
        str(blender_bin),
        "--background",
        "--python",
        str(blender_script),
        "--",
        "--vertices_npy",
        str(vertices_npy),
        "--faces_npy",
        str(faces_npy),
        "--frames_dir",
        str(frames_dir),
        "--width",
        str(int(width)),
        "--height",
        str(int(height)),
        "--camera_mode",
        str(camera_mode),
        "--camera_projection",
        str(camera_projection),
        "--ortho_scale",
        str(float(ortho_scale)),
        "--fixed_anchor",
        str(fixed_anchor),
        "--samples",
        str(int(samples)),
        "--quality_profile",
        str(quality_profile),
        "--render_device",
        str(render_device),
        f"--camera_offset={','.join(f'{float(v):.6f}' for v in camera_offset.tolist())}",
        "--yaw_deg",
        str(float(yaw_deg)),
        "--sun_energy",
        str(float(sun_energy)),
        "--sun_elev_deg",
        str(float(sun_elev_deg)),
        "--sun_azim_deg",
        str(float(sun_azim_deg)),
        "--sun_angle_deg",
        str(float(sun_angle_deg)),
        "--area_light_energy",
        str(float(area_light_energy)),
        "--area_light_size",
        str(float(area_light_size)),
        "--area_light_offset",
        str(area_light_offset),
        "--floor_mode",
        str(floor_mode),
        "--plane_color",
        ",".join(f"{float(v):.6f}" for v in plane_color),
        "--mesh_color",
        ",".join(f"{float(v):.6f}" for v in mesh_color),
        "--mesh_roughness",
        str(float(mesh_roughness)),
        "--mesh_subsurface",
        str(float(mesh_subsurface)),
        "--world_strength",
        str(float(world_strength)),
        "--perspective_fov_deg",
        str(float(perspective_fov_deg)),
        "--checker_color1",
        ",".join(f"{float(v):.6f}" for v in checker_color1),
        "--checker_color2",
        ",".join(f"{float(v):.6f}" for v in checker_color2),
        "--checker_scale",
        str(float(checker_scale)),
        "--checker_ref_plane",
        str(float(checker_ref_plane)),
        "--floor_roughness",
        str(float(floor_roughness)),
        "--floor_min_size",
        str(float(floor_min_size)),
        "--floor_motion_scale",
        str(float(floor_motion_scale)),
        "--floor_pad_mesh",
        str(float(floor_pad_mesh)),
        "--only_frame",
        str(int(only_frame)),
    ]
    if not checker_rescale_to_plane:
        cmd.append("--no_checker_rescale_to_plane")
    if white_background:
        cmd.append("--white_background")
    if transparent_background:
        cmd.append("--transparent_background")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    feature_vec_root = Path(args.feature_vec_root).expanduser().resolve() if args.feature_vec_root else None
    finemotion_root = Path(args.finemotion_root).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    blender_bin = Path(args.blender_bin).expanduser().resolve()
    blender_script = Path(args.blender_script).expanduser().resolve()
    source_script = Path(args.source_script).expanduser().resolve()
    smplh_model_root = Path(args.smplh_model_root).expanduser().resolve()
    plane_color = _parse_rgba(args.plane_color)
    checker_color1 = _parse_rgba(args.checker_color1)
    checker_color2 = _parse_rgba(args.checker_color2)
    mesh_color = _parse_rgba(args.mesh_color)

    source_mod = _load_source_module(source_script)
    elev_deg, azim_deg = _camera_angles_from_preset(args.preset, args.elev_deg, args.azim_deg)
    camera_offset = _camera_offset_blender(args.cam_dist, elev_deg, azim_deg)

    for mid in args.ids:
        stem = Path(mid).stem
        sample_dir = out_root / stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        vertices_npy = sample_dir / f"{stem}_vertices_blender.npy"
        faces_npy = sample_dir / f"{stem}_faces.npy"
        meta_json = sample_dir / f"{stem}_meta.json"
        frames_dir = sample_dir / f"{stem}_frames"
        out_mp4 = sample_dir / f"{stem}_shadow.mp4"

        vertices, faces, fps, meta = _build_vertices_for_motion(
            mid=mid,
            dataset_root=dataset_root,
            feature_vec_root=feature_vec_root,
            finemotion_root=finemotion_root,
            smplh_model_root=smplh_model_root,
            hand_preset=args.hand_preset,
            hand_preset_scale=float(args.hand_preset_scale),
            target_fps=float(args.target_fps),
            out_fps=float(args.out_fps),
            src_fps_override=float(args.src_fps_override),
            stride=int(args.stride),
            max_frames=int(args.max_frames),
            batch_size=int(args.batch_size),
            source_mod=source_mod,
            mesh_z_shift=float(args.mesh_z_shift),
            auto_ground=str(args.auto_ground),
            ground_target_z=float(args.ground_target_z),
            ground_shift_clamp=parse_shift_clamp(args.ground_shift_clamp),
            ground_bottom_percentile=float(args.ground_bottom_percentile),
            ground_contact_height_percentile=float(args.ground_contact_height_percentile),
            ground_contact_velocity_percentile=float(args.ground_contact_velocity_percentile),
            ground_min_contact_frames=int(args.ground_min_contact_frames),
            auto_facing=str(args.auto_facing),
            facing_target_deg=float(args.facing_target_deg),
            facing_estimate_frames=int(args.facing_estimate_frames),
        )
        grounding = meta.get("grounding", {})
        print(
            "[INFO] grounding "
            f"mode={grounding.get('mode')} auto_shift={float(grounding.get('auto_shift', 0.0)):.4f} "
            f"manual_shift={float(grounding.get('manual_shift', 0.0)):.4f} "
            f"total_shift={float(grounding.get('total_shift', 0.0)):.4f}",
            flush=True,
        )
        facing = meta.get("facing", {})
        print(
            "[INFO] facing "
            f"mode={facing.get('mode')} initial={facing.get('initial_facing_deg')} "
            f"target={float(facing.get('target_facing_deg', -90.0)):.1f} "
            f"auto_yaw={float(facing.get('auto_yaw_deg', 0.0)):.1f} "
            f"trim_yaw={float(args.yaw_deg):.1f}",
            flush=True,
        )
        np.save(vertices_npy, vertices)
        np.save(faces_npy, faces)
        meta.update(
            {
                "camera_offset_blender": [float(x) for x in camera_offset.tolist()],
                "render_width": int(args.width),
                "render_height": int(args.height),
                "camera_mode": str(args.camera_mode),
                "fixed_anchor": str(args.fixed_anchor),
                "render_samples": int(args.samples),
                "blender_bin": str(blender_bin),
            }
        )
        meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        _run_blender(
            blender_bin=blender_bin,
            blender_script=blender_script,
            vertices_npy=vertices_npy,
            faces_npy=faces_npy,
            frames_dir=frames_dir,
            width=int(args.width),
            height=int(args.height),
            camera_mode=str(args.camera_mode),
            camera_projection=str(args.camera_projection),
            ortho_scale=float(args.ortho_scale),
            fixed_anchor=str(args.fixed_anchor),
            samples=int(args.samples),
            quality_profile=str(args.quality_profile),
            render_device=str(args.render_device),
            camera_offset=camera_offset,
            yaw_deg=float(args.yaw_deg),
            sun_energy=float(args.sun_energy),
            sun_elev_deg=float(args.sun_elev_deg),
            sun_azim_deg=float(args.sun_azim_deg),
            sun_angle_deg=float(args.sun_angle_deg),
            area_light_energy=float(args.area_light_energy),
            area_light_size=float(args.area_light_size),
            area_light_offset=str(args.area_light_offset),
            floor_mode=str(args.floor_mode),
            plane_color=plane_color,
            mesh_color=mesh_color,
            mesh_roughness=float(args.mesh_roughness),
            mesh_subsurface=float(args.mesh_subsurface),
            world_strength=float(args.world_strength),
            white_background=bool(args.white_background),
            transparent_background=bool(args.transparent_background),
            perspective_fov_deg=float(args.perspective_fov_deg),
            checker_color1=checker_color1,
            checker_color2=checker_color2,
            checker_scale=float(args.checker_scale),
            checker_ref_plane=float(args.checker_ref_plane),
            checker_rescale_to_plane=not bool(args.no_checker_rescale_to_plane),
            floor_roughness=float(args.floor_roughness),
            floor_min_size=float(args.floor_min_size),
            floor_motion_scale=float(args.floor_motion_scale),
            floor_pad_mesh=float(args.floor_pad_mesh),
            only_frame=int(args.only_frame),
        )
        if not args.frames_only:
            _write_video_from_frames(frames_dir, out_mp4, fps=fps)

        if not args.keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)
        if not args.keep_assets:
            vertices_npy.unlink(missing_ok=True)
            faces_npy.unlink(missing_ok=True)

        print(f"[ok] wrote Blender shadow render to: {out_mp4}")


if __name__ == "__main__":
    main()
