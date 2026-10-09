"""Load the released dual-style model and run one motion transfer.

The notebook and the command-line check both call this module. Inputs are
HumanML-263 feature files, shape [T, 263], the same representation as
``data/examples``. The released weights apply ``data/stats/Mean.npy`` and
``data/stats/Std.npy`` internally. Do not pre-normalize the files.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np

T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]
CHAIN_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]


def find_repo_root(start: Path | None = None) -> Path:
    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "demo_dual_style.py").is_file() and (
            candidate / "configs" / "dual_style_hht.yaml"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find the kinematic-frequency-motion repository. "
        "Run this notebook from the repo, or set REPO_URL so the setup cell can clone it."
    )


def prepare_repo(repo: Path) -> Path:
    repo = Path(repo).resolve()
    os.chdir(repo)
    root = str(repo)
    if root not in sys.path:
        sys.path.insert(0, root)
    return repo


def _torch_load(path: str):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(repo: Path, checkpoint: str | None = None, device: str | None = None):
    """Build the released model and load the four frozen checkpoints."""
    import torch
    from omegaconf import OmegaConf

    repo = prepare_repo(repo)
    from mld.config import get_module_config
    from mld.data.release_stats import ReleaseMotionStats
    from mld.models.get_model import get_model

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this environment cannot see a GPU.")

    cfg_base = OmegaConf.load("configs/base.yaml")
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load("configs/dual_style_hht.yaml"))
    cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
    cfg = OmegaConf.merge(cfg_exp, cfg_model, OmegaConf.load("configs/assets.yaml"))
    if checkpoint:
        cfg.TEST.CHECKPOINTS = checkpoint
    if torch_device.type == "cpu":
        cfg.ACCELERATOR = "cpu"

    mean_np = np.load("data/stats/Mean.npy")
    std_np = np.load("data/stats/Std.npy")
    cfg.DATASET.NFEATS = int(mean_np.shape[-1])
    cfg.DATASET.NJOINTS = 22
    dataset = ReleaseMotionStats(mean_np, std_np)
    model = get_model(cfg, dataset)

    checkpoint_obj = _torch_load(cfg.TEST.CHECKPOINTS)
    state_dict = (
        checkpoint_obj["state_dict"]
        if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj
        else checkpoint_obj
    )
    unused_prefixes = (
        "t2m_textencoder.",
        "t2m_moveencoder.",
        "t2m_motionencoder.",
    )
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(unused_prefixes)
    }
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            "Checkpoint is missing weights required by the released model: "
            + ", ".join(incompatible.missing_keys[:20])
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint has unexpected weights for the released model: "
            + ", ".join(incompatible.unexpected_keys[:20])
        )

    model.sample_mean = bool(OmegaConf.select(cfg, "TEST.MEAN", default=False))
    model.fact = float(OmegaConf.select(cfg, "TEST.FACT", default=1.0))
    model.to(torch_device)
    model.eval()
    return model


def resolve_npy(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Motion path does not exist: {candidate}")
    if candidate.is_file():
        if candidate.suffix != ".npy":
            raise ValueError(f"Expected a .npy file, got {candidate}")
        return candidate
    files = sorted(p for p in candidate.rglob("*.npy") if p.is_file())
    if len(files) != 1:
        names = ", ".join(p.name for p in files[:8]) or "(none)"
        raise ValueError(
            f"{candidate} must contain exactly one .npy file for this notebook. Found {len(files)}: {names}"
        )
    return files[0]


def load_features(path: str | Path) -> np.ndarray:
    resolved = resolve_npy(path)
    motion = np.load(resolved)
    if motion.ndim != 2 or motion.shape[-1] != 263:
        raise ValueError(
            f"{resolved} has shape {tuple(motion.shape)}. "
            "This demo expects one HumanML-263 feature array of shape [T, 263]."
        )
    if motion.shape[0] < 2:
        raise ValueError(f"{resolved} is too short: T={motion.shape[0]}.")
    return np.asarray(motion, dtype=np.float32), resolved


def features_to_joints(features: np.ndarray) -> np.ndarray:
    """Recover joint positions from raw, unnormalized HumanML features."""
    import torch
    from mld.data.release_stats import recover_from_ric

    joints = recover_from_ric(torch.as_tensor(features, dtype=torch.float32), 22)
    return joints.detach().cpu().numpy()


def _as_joints(result) -> np.ndarray:
    import torch

    joints = result
    if isinstance(joints, dict):
        joints = joints["joints"]
    if isinstance(joints, (list, tuple)):
        joints = joints[0]
    if torch.is_tensor(joints):
        joints = joints.detach().cpu().numpy()
    joints = np.asarray(joints)
    if joints.ndim == 4:
        joints = joints[0]
    if joints.ndim != 3 or joints.shape[-2:] != (22, 3):
        raise RuntimeError(f"Expected joints [T, 22, 3], got {joints.shape}")
    return joints


def transfer(
    model,
    content_features: np.ndarray,
    style_features: np.ndarray,
    *,
    coarse_features: np.ndarray | None = None,
    fine_features: np.ndarray | None = None,
    scale: float = 2.5,
    fine_scale: float = 1.5,
    seed: int | None = 0,
    return_raw_feats: bool = False,
):
    """Run one dual-style transfer. The result length follows the content motion."""
    import torch

    if coarse_features is None:
        coarse_features = style_features
    if fine_features is None:
        fine_features = style_features
    device = next(model.parameters()).device
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def _batch(array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, dtype=torch.float32, device=device).unsqueeze(0)

    model.guidance_scale = float(scale)
    model.guidance_fine_scale = float(fine_scale)
    batch = {
        "length": [int(content_features.shape[0])],
        "content_motion": _batch(content_features),
        "style_motion": _batch(fine_features),
        "coarse_style_motion": _batch(coarse_features),
        "fine_style_motion": _batch(fine_features),
        "tag_scale": float(scale),
        "tag_scale_fine": float(fine_scale),
        "return_motion_feats": bool(return_raw_feats),
    }
    with torch.no_grad():
        output = model(batch)
    if not return_raw_feats:
        return _as_joints(output)
    joints = _as_joints(output["joints"])
    feats = output["motion_feats"]
    if isinstance(feats, (list, tuple)):
        feats = feats[0]
    if torch.is_tensor(feats):
        feats = feats.detach().cpu().numpy()
    mean = np.load("data/stats/Mean.npy").astype(np.float32)
    std = np.load("data/stats/Std.npy").astype(np.float32)
    return joints, np.asarray(feats, dtype=np.float32) * std + mean


def motion_stats(name: str, joints: np.ndarray) -> dict:
    if not np.isfinite(joints).all():
        raise RuntimeError(f"{name} joints contain NaN or Inf.")
    span = joints.reshape(-1, 3).max(axis=0) - joints.reshape(-1, 3).min(axis=0)
    root = joints[:, 0, :]
    travel = float(np.linalg.norm(np.diff(root, axis=0), axis=1).sum()) if len(root) > 1 else 0.0
    stats = {
        "name": name,
        "frames": int(joints.shape[0]),
        "height_span_m": float(span[1]),
        "root_travel_m": travel,
        "root_height_mean_m": float(root[:, 1].mean()),
    }
    if stats["height_span_m"] < 0.2 or stats["height_span_m"] > 5.0:
        raise RuntimeError(
            f"{name} does not look like a HumanML skeleton: vertical span is {stats['height_span_m']:.3f} m."
        )
    return stats


def _draw_pose(ax, pose: np.ndarray, limits) -> None:
    ax.cla()
    for chain, color in zip(T2M_KINEMATIC_CHAIN, CHAIN_COLORS):
        pts = pose[chain]
        ax.plot(pts[:, 0], pts[:, 2], pts[:, 1], color=color, linewidth=2.0)
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], color=color, s=8, depthshade=False)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_zlim(limits[2])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=18, azim=-70)
    ax.set_box_aspect((1, 1, 1))


def _limits(joints: np.ndarray):
    flat = joints.reshape(-1, 3)
    center = (flat.min(axis=0) + flat.max(axis=0)) / 2.0
    span = max(float((flat.max(axis=0) - flat.min(axis=0)).max()), 1.2)
    half = span / 2.0
    return (
        (center[0] - half, center[0] + half),
        (center[2] - half, center[2] + half),
        (center[1] - half, center[1] + half),
    )


def _frame_at(joints: np.ndarray, index: int) -> np.ndarray:
    return joints[min(max(index, 0), len(joints) - 1)]


def save_comparison(
    content: np.ndarray,
    style: np.ndarray,
    result: np.ndarray,
    out_dir: str | Path,
    *,
    fps: int = 20,
    titles: tuple[str, str, str] = ("Content", "Style", "Transferred"),
) -> dict:
    """Write a stick-figure gif. Y is up, units are meters."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sequences = [content, style, result]
    limits = [_limits(seq) for seq in sequences]

    stride = max(1, int(np.ceil(len(result) / 160)))
    frame_ids = list(range(0, len(result), stride))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), subplot_kw={"projection": "3d"})

    def update(frame_id):
        for ax, seq, title, limit in zip(axes, sequences, titles, limits):
            _draw_pose(ax, _frame_at(seq, frame_id), limit)
            ax.set_title(f"{title}  frame {min(frame_id, len(seq) - 1)}")
        return axes

    anim = FuncAnimation(fig, update, frames=frame_ids, interval=1000 / fps, blit=False)
    gif_path = out_dir / "preview.gif"
    anim.save(gif_path, writer=PillowWriter(fps=max(1, int(round(fps / stride)))))
    plt.close(fig)

    np.save(out_dir / "transferred_joints.npy", result)
    return {"gif": gif_path, "joints": out_dir / "transferred_joints.npy"}


def run_pair(
    model,
    content_path: str | Path,
    style_path: str | Path,
    out_dir: str | Path,
    *,
    coarse_path: str | Path | None = None,
    fine_path: str | Path | None = None,
    scale: float = 2.5,
    fine_scale: float = 1.5,
    seed: int | None = 0,
    foot_fix: bool = True,
) -> dict:
    content, content_file = load_features(content_path)
    style, style_file = load_features(style_path)
    coarse = load_features(coarse_path)[0] if coarse_path else None
    fine = load_features(fine_path)[0] if fine_path else None
    content_joints = features_to_joints(content)
    style_joints = features_to_joints(style)
    stats = [
        motion_stats("content", content_joints),
        motion_stats("style", style_joints),
    ]
    raw_result = transfer(
        model,
        content,
        style,
        coarse_features=coarse,
        fine_features=fine,
        scale=scale,
        fine_scale=fine_scale,
        seed=seed,
    )
    result = raw_result
    if foot_fix:
        import torch
        from mld.postprocess.foot_fix import fix_foot_sliding

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        result, _info = fix_foot_sliding(raw_result, device=device)
    stats.append(motion_stats("transferred", result))
    paths = save_comparison(content_joints, style_joints, result, out_dir)
    raw_path = Path(out_dir) / "transferred_joints_raw.npy"
    np.save(raw_path, raw_result)
    paths["joints_raw"] = raw_path
    return {
        "content_file": content_file,
        "style_file": style_file,
        "stats": stats,
        "paths": paths,
        "scale": float(scale),
        "fine_scale": float(fine_scale),
        "seed": seed,
        "foot_fix": bool(foot_fix),
    }


def _print_report(report: dict) -> None:
    print(f"content: {report['content_file']}")
    print(f"style:   {report['style_file']}")
    print(f"scales:  global={report['scale']} fine={report['fine_scale']} seed={report['seed']}")
    for item in report["stats"]:
        print(
            f"{item['name']}: frames={item['frames']} "
            f"height_span={item['height_span_m']:.2f}m "
            f"root_height={item['root_height_mean_m']:.2f}m "
            f"root_travel={item['root_travel_m']:.2f}m"
        )
    for key, path in report["paths"].items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run one released dual-style transfer.")
    parser.add_argument("--content", default="data/examples/turn_footwork/content")
    parser.add_argument("--style", default="data/examples/turn_footwork/style")
    parser.add_argument("--coarse", default="")
    parser.add_argument("--fine", default="")
    parser.add_argument("--scale", type=float, default=2.5)
    parser.add_argument("--fine-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-foot-fix", action="store_true")
    parser.add_argument("--out", default="outputs/notebook_demo/turn_footwork")
    args = parser.parse_args()

    loaded = load_model(find_repo_root())
    report = run_pair(
        loaded,
        args.content,
        args.style,
        args.out,
        coarse_path=args.coarse or None,
        fine_path=args.fine or None,
        scale=args.scale,
        fine_scale=args.fine_scale,
        seed=args.seed,
        foot_fix=not args.no_foot_fix,
    )
    _print_report(report)
