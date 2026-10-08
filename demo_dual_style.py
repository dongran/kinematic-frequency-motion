"""Dual-style motion transfer demo for the public release.

The demo combines one content motion, one coarse/global style motion, and one
fine/detail style motion. Inputs are HumanML-style `.npy` feature files with
shape [T, 263], using the shipped FineMotion normalization statistics.
"""

from __future__ import annotations

import os
from pathlib import Path
import pickle

import numpy as np
import torch
from omegaconf import OmegaConf

from mld.config import parse_args
from mld.data.release_stats import ReleaseMotionStats
from mld.models.get_model import get_model
from mld.postprocess.foot_fix import fix_foot_sliding
from mld.utils.logger import create_logger


def _torch_load_compat(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _get_test_scalars(cfg):
    mean = float(OmegaConf.select(cfg, "TEST.MEAN", default=0.0))
    fact = float(OmegaConf.select(cfg, "TEST.FACT", default=1.0))
    return mean, fact


def _iter_npy_files(root: str):
    files = sorted([p for p in Path(root).rglob("*.npy") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No .npy files found under: {root}")
    return files


def _load_motion(path: Path, device: torch.device):
    motion = np.load(str(path))
    if motion.ndim != 2:
        raise ValueError(f"Expected [T, C] motion array, got shape {motion.shape} from {path}")
    return torch.tensor(np.array([motion]), dtype=torch.float32, device=device)


def _label_from_stem(stem: str) -> str:
    return stem.rsplit("-", 1)[-1]


def main():
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    create_logger(cfg, phase="demo")

    content_path = cfg.DEMO.content_motion_dir or "data/content_motion"
    style_path = cfg.DEMO.style_motion_dir or "data/fine_style_motion"
    coarse_path = cfg.DEMO.coarse_style_motion_dir or "data/coarse_style_motion"
    fine_path = cfg.DEMO.fine_style_motion_dir or style_path

    label_mode = str(OmegaConf.select(cfg, "DEMO.matrix_label_mode", default="fine")).lower()
    if label_mode not in {"fine", "coarse"}:
        raise ValueError(f"Unsupported matrix_label_mode={label_mode!r}")

    eval_name = "dual_style"
    fine_scale = float(getattr(cfg.DEMO, "fine_scale", 1.0))
    traj_scale = OmegaConf.select(cfg, "DEMO.traj_scale", default=None)
    preserve_scale = OmegaConf.select(cfg, "DEMO.preserve_scale", default=None)
    timing_scale = OmegaConf.select(cfg, "DEMO.timing_scale", default=None)
    save_root = Path(cfg.FOLDER) / str(cfg.model.model_type) / str(cfg.NAME)
    save_root.mkdir(parents=True, exist_ok=True)
    joints_dir = save_root / "joints"
    joints_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = save_root / "joints_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    apply_foot_fix = bool(getattr(cfg.DEMO, "foot_fix", True))
    save_path = save_root / (
        f"{eval_name}_scale_{str(cfg.DEMO.scale).replace('.', '-')}"
        f"_fine_{str(fine_scale).replace('.', '-')}.pkl"
    )

    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in cfg.DEVICE)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    mean_np = np.load("data/stats/Mean.npy")
    std_np = np.load("data/stats/Std.npy")
    cfg.DATASET.NFEATS = int(mean_np.shape[-1])
    cfg.DATASET.NJOINTS = 22
    dataset = ReleaseMotionStats(mean_np, std_np)
    model = get_model(cfg, dataset)
    checkpoint = _torch_load_compat(cfg.TEST.CHECKPOINTS)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    # The training checkpoint also stores the T2M evaluator.
    # This motion-to-motion demo does not build those modules.
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
    mean, fact = _get_test_scalars(cfg)
    model.sample_mean = mean
    model.fact = fact
    model.to(device)
    model.eval()

    scale = cfg.DEMO.scale
    model.guidance_scale = float(scale)
    model.guidance_fine_scale = fine_scale
    if timing_scale is not None:
        model.guidance_contact_timing_scale = float(timing_scale)

    content_files = _iter_npy_files(content_path)
    coarse_files = _iter_npy_files(coarse_path)
    fine_files = _iter_npy_files(fine_path)
    total = len(content_files) * len(coarse_files) * len(fine_files)

    save_all = {
        "joints": [],
        "id": [],
        "label_content": [],
        "label_style": [],
        "content_id": [],
        "coarse_style_id": [],
        "fine_style_id": [],
        "label_coarse_style": [],
        "label_fine_style": [],
        "meta": {
            "mode": "dual_style_hht",
            "label_style_source": label_mode,
            "scale": float(scale),
            "fine_scale": float(fine_scale),
            "traj_scale": None if traj_scale is None else float(traj_scale),
            "preserve_scale": None if preserve_scale is None else float(preserve_scale),
            "timing_scale": None if timing_scale is None else float(timing_scale),
            "foot_fix": apply_foot_fix,
        },
    }

    idx = 0
    for content_p in content_files:
        content_stem = content_p.stem
        content_motion = _load_motion(content_p, device)
        lengths = [int(content_motion.shape[1])]

        for coarse_p in coarse_files:
            coarse_stem = coarse_p.stem
            coarse_motion = _load_motion(coarse_p, device)

            for fine_p in fine_files:
                fine_stem = fine_p.stem
                fine_motion = _load_motion(fine_p, device)
                print(f"[demo] sample {idx + 1}/{total}: content={content_stem}, coarse={coarse_stem}, fine={fine_stem}")
                idx += 1

                with torch.no_grad():
                    batch = {
                        "length": lengths,
                        "content_motion": content_motion,
                        "style_motion": fine_motion,
                        "coarse_style_motion": coarse_motion,
                        "fine_style_motion": fine_motion,
                        "tag_scale": scale,
                        "tag_scale_fine": fine_scale,
                    }
                    if traj_scale is not None:
                        batch["tag_scale_traj"] = float(traj_scale)
                    if preserve_scale is not None:
                        batch["tag_scale_preserve"] = float(preserve_scale)
                    if timing_scale is not None:
                        batch["tag_scale_contact_timing"] = float(timing_scale)
                    joints = model(batch)

                sample_id = (
                    f"content_{content_stem}__coarse_{coarse_stem}__fine_{fine_stem}"
                    f"__scale_{str(scale).replace('.', '-')}"
                    f"__fine_{str(fine_scale).replace('.', '-')}"
                )
                label_coarse = _label_from_stem(coarse_stem)
                label_fine = _label_from_stem(fine_stem)
                joints_np = joints[0].detach().cpu().numpy()
                raw_npy = raw_dir / f"{sample_id}.npy"
                np.save(raw_npy, joints_np)
                if apply_foot_fix:
                    fixed_np, foot_info = fix_foot_sliding(
                        joints_np,
                        device="cuda:0" if device.type == "cuda" else "cpu",
                    )
                    print(
                        "[demo] foot fix "
                        f"delta={foot_info['mean_joint_delta']:.4f} "
                        f"contact L/R={foot_info['left_contact_frames']}/"
                        f"{foot_info['right_contact_frames']}"
                    )
                else:
                    fixed_np = joints_np
                sample_npy = joints_dir / f"{sample_id}.npy"
                np.save(sample_npy, fixed_np)
                print(f"[demo] joints {tuple(fixed_np.shape)} -> {sample_npy}")
                print(f"[demo] raw joints -> {raw_npy}")
                save_all["joints"].append(fixed_np)
                save_all["id"].append(sample_id)
                save_all["label_content"].append(_label_from_stem(content_stem))
                save_all["label_style"].append(label_fine if label_mode == "fine" else label_coarse)
                save_all["content_id"].append(content_stem)
                save_all["coarse_style_id"].append(coarse_stem)
                save_all["fine_style_id"].append(fine_stem)
                save_all["label_coarse_style"].append(label_coarse)
                save_all["label_fine_style"].append(label_fine)

    with open(save_path, "wb") as f:
        pickle.dump(save_all, f)
    print(f"[OK] saved {save_path}")


if __name__ == "__main__":
    main()

