#!/usr/bin/env python3
"""
Render SMPL mesh sequences from smpl_poses_mesh.npz (keys poses + trans) via Blender Cycles.

This is a standalone helper for already-converted SMPL transfer outputs. It does not replace or
modify the existing FineMotion/content-style Blender pipeline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.motion_ik_smpl_bvh import resolve_mesh_model_path, resolve_smpl_path  # noqa: E402
from tools.auto_grounding import apply_auto_ground, parse_shift_clamp  # noqa: E402
from tools.auto_facing import apply_auto_facing  # noqa: E402


def _load_fin_render_module() -> Any:
    path = REPO_ROOT / "tools" / "render_finemotion_smplh_hands_blender_shadow.py"
    spec = importlib.util.spec_from_file_location("finemotion_blender_shadow", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_source_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("finemotion_source_render", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_rgba(text: str) -> tuple[float, float, float, float]:
    vals = [float(x.strip()) for x in str(text).split(",")]
    if len(vals) == 3:
        vals.append(1.0)
    if len(vals) != 4:
        raise ValueError(f"Expected 3 or 4 comma-separated RGBA values: {text}")
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])


def smpl_vertices_from_npz(
    npz_path: Path,
    *,
    smpl_path_resolved: str,
    max_frames: int,
    batch_size: int,
    body_model_type: str = "smpl",
    smplh_model_root: Path | None = None,
    source_script: Path | None = None,
    hand_preset: str = "handpose_zip_4_400_tight65",
    hand_preset_scale: float = 1.0,
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
    fps: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import smplx
    import torch

    data = np.load(npz_path, allow_pickle=True)
    poses = np.asarray(data["poses"], dtype=np.float32)
    trans = np.asarray(data["trans"], dtype=np.float32)
    if poses.ndim != 3 or poses.shape[-1] != 3:
        raise ValueError(f"poses expect (T,J,3), got {poses.shape}")
    if trans.ndim != 2 or trans.shape[-1] != 3:
        raise ValueError(f"trans expect (T,3), got {trans.shape}")
    if poses.shape[0] != trans.shape[0]:
        raise ValueError("poses/trans frame count mismatch")
    if poses.shape[1] != 24:
        raise ValueError(f"This helper expects SMPL (T,24,3); got J={poses.shape[1]}.")

    if max_frames > 0:
        poses = poses[:max_frames]
        trans = trans[:max_frames]

    body_model_type = str(body_model_type).strip().lower()
    if body_model_type not in {"smpl", "smplh"}:
        raise ValueError(f"Unsupported body model: {body_model_type}")

    use_pca = False
    flat_hand_mean = False
    num_pca_comps = 6
    left_hand_pose_np = np.zeros((45,), dtype=np.float32)
    right_hand_pose_np = np.zeros((45,), dtype=np.float32)
    if body_model_type == "smplh":
        if smplh_model_root is None:
            raise ValueError("--smplh_model_root is required when --body_model=smplh")
        if source_script is None:
            raise ValueError("--source_script is required when --body_model=smplh")
        source_mod = _load_source_module(source_script.expanduser().resolve())
        use_pca, flat_hand_mean, num_pca_comps, left_hand_pose_np, right_hand_pose_np = (
            source_mod._resolve_hand_pose_preset(
                hand_preset=hand_preset,
                hand_preset_scale=float(hand_preset_scale),
            )
        )
    else:
        model_root = resolve_mesh_model_path(smpl_path_resolved)

    faces = None
    verts_all: list[np.ndarray] = []
    joints_all: list[np.ndarray] = []
    bs = max(1, int(batch_size))
    for start in range(0, poses.shape[0], bs):
        end = min(poses.shape[0], start + bs)
        b = end - start
        if body_model_type == "smplh":
            body_model = smplx.create(
                str(smplh_model_root.expanduser().resolve()),
                model_type="smplh",
                gender="neutral",
                ext="npz",
                flat_hand_mean=bool(flat_hand_mean),
                use_pca=bool(use_pca),
                num_pca_comps=int(num_pca_comps),
                batch_size=b,
            ).eval()
        else:
            body_model = smplx.create(
                model_root,
                model_type="smpl",
                gender="neutral",
                ext="pkl",
                batch_size=b,
            ).eval()
        if faces is None:
            faces = np.asarray(body_model.faces, dtype=np.int32)

        chunk = poses[start:end].reshape(b, -1)
        global_orient = torch.from_numpy(chunk[:, :3]).float()
        transl = torch.from_numpy(trans[start:end]).float()
        betas = torch.zeros((b, 10), dtype=torch.float32)
        with torch.no_grad():
            if body_model_type == "smplh":
                body_pose = torch.from_numpy(poses[start:end, 1:22, :].reshape(b, -1)).float()
                left_hand_pose = torch.from_numpy(
                    np.repeat(left_hand_pose_np.reshape(1, -1), repeats=b, axis=0)
                ).float()
                right_hand_pose = torch.from_numpy(
                    np.repeat(right_hand_pose_np.reshape(1, -1), repeats=b, axis=0)
                ).float()
                out = body_model(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    transl=transl,
                    betas=betas,
                )
            else:
                body_pose = torch.from_numpy(chunk[:, 3:72]).float()
                out = body_model(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    transl=transl,
                    betas=betas,
                )
        verts_all.append(out.vertices.detach().cpu().numpy().astype(np.float32))
        joints_all.append(out.joints.detach().cpu().numpy().astype(np.float32))

    if faces is None:
        raise RuntimeError(f"No frames found in {npz_path}")
    vertices_y_up = np.concatenate(verts_all, axis=0)
    joints_y_up = np.concatenate(joints_all, axis=0)
    vertices_bl = vertices_y_up[..., [2, 0, 1]].copy()
    joints_bl = joints_y_up[..., [2, 0, 1]].copy()
    z_min_subtract = float(np.min(vertices_bl[..., 2]))
    vertices_bl[..., 2] -= z_min_subtract
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
        fps=float(fps),
    )
    facing_center_xy = (
        float(np.mean(vertices_bl[..., 0])),
        float(np.mean(vertices_bl[..., 1])),
    )
    vertices_bl, facing_meta = apply_auto_facing(
        vertices_bl,
        joints_bl,
        mode=auto_facing,
        target_facing_deg=float(facing_target_deg),
        estimate_frames=int(facing_estimate_frames),
    )
    meta = {
        "body_model": body_model_type,
        "hand_preset": str(hand_preset) if body_model_type == "smplh" else "",
        "hand_preset_scale": float(hand_preset_scale) if body_model_type == "smplh" else 0.0,
        "grounding": ground_meta.to_dict(),
        "facing": facing_meta.to_dict(),
        "z_min_subtract": float(z_min_subtract),
        "facing_center_xy": [float(facing_center_xy[0]), float(facing_center_xy[1])],
    }
    return vertices_bl.astype(np.float32), faces, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz_path", required=True, type=Path)
    ap.add_argument("--out_mp4", required=True, type=Path)
    ap.add_argument("--smpl_path", default="", help="Directory containing SMPL_NEUTRAL.pkl or empty for defaults.")
    ap.add_argument("--body_model", default="smpl", choices=["smpl", "smplh"], help="Use smplh to render fixed hand poses.")
    ap.add_argument("--smplh_model_root", type=Path, default=REPO_ROOT / "deps" / "smpl_models")
    ap.add_argument(
        "--source_script",
        type=Path,
        default=None,
        help="Required only for --body_model=smplh. Path to the SMPL-H hand-pose helper.",
    )
    ap.add_argument("--hand_preset", default="handpose_zip_4_400_tight65")
    ap.add_argument("--hand_preset_scale", type=float, default=1.0)
    ap.add_argument("--blender_bin", required=True, type=Path)
    ap.add_argument(
        "--blender_script",
        type=Path,
        default=REPO_ROOT / "tools" / "blender_render_vertices_shadow.py",
    )
    ap.add_argument("--max_frames", type=int, default=0, help="0 = full sequence.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--stride", type=int, default=1, help="Sample every Nth frame from vertices when rendering.")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--quality_profile", default="standard", choices=["standard", "paper"])
    ap.add_argument("--render_device", default="cuda", choices=["cpu", "cuda", "optix"])
    ap.add_argument(
        "--preset",
        default="diag",
        choices=["front", "top", "diag", "lookdown", "bird"],
        help="diag: eye-level three-quarter. lookdown/bird: about 27 degrees down. top: overhead. front: front view.",
    )
    ap.add_argument("--elev_deg", type=float, default=8.0)
    ap.add_argument("--azim_deg", type=float, default=270.0)
    ap.add_argument("--cam_dist", type=float, default=3.2)
    ap.add_argument("--camera_mode", default="track", choices=["track", "fixed"])
    ap.add_argument("--camera_projection", default="perspective", choices=["perspective", "orthographic"])
    ap.add_argument("--ortho_scale", type=float, default=0.0)
    ap.add_argument("--fixed_anchor", default="first", choices=["first", "mean", "origin"])
    ap.add_argument("--yaw_deg", type=float, default=0.0)
    ap.add_argument(
        "--props_boxes_yup_json",
        type=str,
        default="",
        help="Optional Y-up AABB props JSON; transformed with transfer mesh grounding/facing.",
    )
    ap.add_argument(
        "--props_boxes_json",
        type=str,
        default="",
        help="Optional Blender Z-up AABB props JSON already aligned to this mesh.",
    )
    ap.add_argument("--sun_energy", type=float, default=3.4)
    ap.add_argument("--sun_elev_deg", type=float, default=42.0)
    ap.add_argument("--sun_azim_deg", type=float, default=35.0)
    ap.add_argument("--sun_angle_deg", type=float, default=7.0)
    ap.add_argument("--area_light_energy", type=float, default=0.0)
    ap.add_argument("--area_light_size", type=float, default=4.0)
    ap.add_argument("--area_light_offset", default="0.0,-2.8,3.2")
    ap.add_argument(
        "--floor_mode",
        default="plane",
        choices=["plane", "none", "checker"],
        help="plane = legacy solid floor; checker = paper-style procedural grid; none = invisible floor.",
    )
    ap.add_argument(
        "--perspective_fov_deg",
        type=float,
        default=45.0,
        help="Perspective only; typical legacy ~45, paper checker line often ~50.",
    )
    ap.add_argument("--checker_color1", default="0.32,0.32,0.32,1.0")
    ap.add_argument("--checker_color2", default="0.52,0.52,0.52,1.0")
    ap.add_argument("--checker_scale", type=float, default=5.5)
    ap.add_argument("--checker_ref_plane", type=float, default=5.2)
    ap.add_argument(
        "--no_checker_rescale_to_plane",
        action="store_true",
        help="Pass through to Blender; disable checker density rescale for large floors.",
    )
    ap.add_argument("--floor_roughness", type=float, default=0.88, help="Checker floor roughness (floor_mode=checker).")
    ap.add_argument("--floor_min_size", type=float, default=6.0)
    ap.add_argument("--floor_motion_scale", type=float, default=2.6)
    ap.add_argument("--floor_pad_mesh", type=float, default=0.0)
    ap.add_argument("--only_frame", type=int, default=-1, help=">=0: render only this frame (preview).")
    ap.add_argument(
        "--preview_png",
        type=Path,
        default=None,
        help="If set, copy the first rendered PNG here (e.g. after max_frames=1).",
    )
    ap.add_argument("--plane_color", default="0.96,0.96,0.96,1.0")
    ap.add_argument("--mesh_color", default="0.90,0.38,0.10,1.0")
    ap.add_argument("--mesh_roughness", type=float, default=0.55)
    ap.add_argument("--mesh_subsurface", type=float, default=0.0)
    ap.add_argument("--world_strength", type=float, default=0.35)
    ap.add_argument(
        "--mesh_z_shift",
        type=float,
        default=0.0,
        help=(
            "Manual trim after initial min-Z grounding and optional auto grounding. "
            "Negative moves the actor downward (often reduces apparent floating)."
        ),
    )
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
    ap.add_argument("--white_background", action="store_true")
    ap.add_argument("--transparent_background", action="store_true")
    ap.add_argument(
        "--skip_mp4",
        action="store_true",
        help="Write PNG frames only. Do not encode an mp4.",
    )
    ap.add_argument(
        "--export_png_dir",
        type=Path,
        default=None,
        help="If set, copy frame_*.png out of the temporary frames directory before it is deleted.",
    )
    ap.add_argument("--keep_frames", action="store_true")
    ap.add_argument("--keep_assets", action="store_true")
    args = ap.parse_args()

    fin = _load_fin_render_module()
    npz_path = args.npz_path.expanduser().resolve()
    out_mp4 = args.out_mp4.expanduser().resolve()
    blender_bin = args.blender_bin.expanduser().resolve()
    blender_script = args.blender_script.expanduser().resolve()

    smpl_resolved = resolve_smpl_path(args.smpl_path, REPO_ROOT / "third_party" / "joints2smpl")
    vertices, faces, render_meta = smpl_vertices_from_npz(
        npz_path,
        smpl_path_resolved=smpl_resolved,
        body_model_type=str(args.body_model),
        smplh_model_root=args.smplh_model_root,
        source_script=args.source_script,
        hand_preset=str(args.hand_preset),
        hand_preset_scale=float(args.hand_preset_scale),
        max_frames=int(args.max_frames),
        batch_size=int(args.batch_size),
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
        fps=float(args.fps),
    )
    ground_meta = render_meta["grounding"]
    facing_meta = render_meta["facing"]
    stride = max(1, int(args.stride))
    if stride > 1:
        vertices = vertices[::stride]

    work = out_mp4.parent / f".assets_{out_mp4.stem}"
    work.mkdir(parents=True, exist_ok=True)
    verts_disk = work / "vertices_blender.npy"
    faces_disk = work / "faces.npy"
    np.save(verts_disk, vertices)
    np.save(faces_disk, faces)
    (work / "grounding.json").write_text(json.dumps(ground_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (work / "facing.json").write_text(json.dumps(facing_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    props_boxes_json = None
    props_bl = str(getattr(args, "props_boxes_json", "") or "").strip()
    props_yup = str(getattr(args, "props_boxes_yup_json", "") or "").strip()
    if props_bl:
        props_boxes_json = Path(props_bl).expanduser().resolve()
        if not props_boxes_json.is_file():
            raise FileNotFoundError(props_boxes_json)
        print(f"[INFO] props boxes (blender) -> {props_boxes_json}", flush=True)
    elif props_yup:
        from tools.props_boxes import load_yup_boxes, write_boxes_json, yup_boxes_to_blender

        bl_boxes = yup_boxes_to_blender(
            load_yup_boxes(props_yup),
            z_min_subtract=float(render_meta.get("z_min_subtract", 0.0)),
            ground_total_shift=float(ground_meta["total_shift"]),
            facing_yaw_deg=float(facing_meta["auto_yaw_deg"]),
            facing_center_xy=render_meta.get("facing_center_xy"),
            ground_target_z=float(args.ground_target_z),
        )
        props_boxes_json = work / "props_boxes_blender.json"
        write_boxes_json(props_boxes_json, bl_boxes, meta={"source_yup": props_yup})
        print(f"[INFO] props boxes -> {props_boxes_json} ({len(bl_boxes)} boxes)", flush=True)
    print(
        "[INFO] grounding "
        f"mode={ground_meta['mode']} auto_shift={ground_meta['auto_shift']:.4f} "
        f"manual_shift={ground_meta['manual_shift']:.4f} total_shift={ground_meta['total_shift']:.4f}",
        flush=True,
    )
    print(
        "[INFO] facing "
        f"mode={facing_meta['mode']} initial={facing_meta['initial_facing_deg']} "
        f"target={facing_meta['target_facing_deg']:.1f} auto_yaw={facing_meta['auto_yaw_deg']:.1f} "
        f"trim_yaw={float(args.yaw_deg):.1f}",
        flush=True,
    )
    print(
        "[INFO] body_model "
        f"type={render_meta['body_model']} hand_preset={render_meta['hand_preset']}",
        flush=True,
    )

    elev_deg, azim_deg = fin._camera_angles_from_preset(args.preset, args.elev_deg, args.azim_deg)
    camera_offset = fin._camera_offset_blender(float(args.cam_dist), elev_deg, azim_deg)
    plane_color = _parse_rgba(args.plane_color)
    mesh_color = _parse_rgba(args.mesh_color)
    checker_color1 = _parse_rgba(args.checker_color1)
    checker_color2 = _parse_rgba(args.checker_color2)

    frames_dir = work / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    fin._run_blender(
        blender_bin=blender_bin,
        blender_script=blender_script,
        vertices_npy=verts_disk,
        faces_npy=faces_disk,
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
        **({"props_boxes_json": props_boxes_json} if props_boxes_json is not None else {}),
    )

    if args.export_png_dir is not None:
        export_dst = args.export_png_dir.expanduser().resolve()
        export_dst.mkdir(parents=True, exist_ok=True)
        exported = 0
        for p in sorted(frames_dir.glob("frame_*.png")):
            shutil.copy2(p, export_dst / p.name)
            exported += 1
        print(f"[INFO] export_png_dir: copied {exported} PNG -> {export_dst}", flush=True)

    if args.preview_png is not None:
        preview_dst = args.preview_png.expanduser().resolve()
        preview_dst.parent.mkdir(parents=True, exist_ok=True)
        frame_pngs = sorted(frames_dir.glob("frame_*.png"))
        if not frame_pngs:
            raise RuntimeError(f"No frame PNG in {frames_dir}")
        shutil.copy2(frame_pngs[0], preview_dst)

    video_fps = float(args.fps) / float(stride)
    if bool(args.skip_mp4):
        print("[INFO] --skip_mp4: skip ffmpeg mp4 encode", flush=True)
    else:
        fin._write_video_from_frames(frames_dir, out_mp4, fps=video_fps)

    if not args.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
    if not args.keep_assets:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
