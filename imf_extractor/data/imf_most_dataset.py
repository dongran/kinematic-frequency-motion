import os
from os.path import join as pjoin
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data.imf_dataset import imf_collate_fn


def _safe_load_npz(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"npz not found: {path}")
    return np.load(path)


def _flatten_clips_to_motion(clip_11_t_21: np.ndarray) -> np.ndarray:
    # clip: (11, T, 21) -> motion: (T, 231)
    if clip_11_t_21.ndim != 3 or clip_11_t_21.shape[0] != 11 or clip_11_t_21.shape[2] != 21:
        raise ValueError(f"Unexpected clips shape: {tuple(clip_11_t_21.shape)} (expected (11,T,21))")
    clip_t_11_21 = clip_11_t_21.transpose(1, 0, 2)  # (T,11,21)
    return clip_t_11_21.reshape(clip_t_11_21.shape[0], -1)  # (T,231)


def _infer_valid_length_from_nan(motion_t_d: np.ndarray) -> int:
    # Valid frame: all dims finite. Tail is expected to be NaN padded.
    if motion_t_d.ndim != 2:
        raise ValueError(f"Unexpected motion shape: {tuple(motion_t_d.shape)}")
    finite = np.isfinite(motion_t_d).all(axis=1)  # (T,)
    if not finite.any():
        return 0
    # take last finite index + 1 (more robust than sum if any sporadic NaNs exist)
    last = int(np.where(finite)[0][-1])
    return last + 1


def _load_manifest_seg2idx(path: str) -> Dict[str, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"manifest not found: {path}")
    seg2idx: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            seg_id = str(obj.get("seg_id"))
            idx = int(obj.get("idx"))
            seg2idx[seg_id] = idx
    return seg2idx


def _canonicalize_imfs(imfs: np.ndarray) -> np.ndarray:
    """Canonicalize IMF labels to [3,63,T]. Compatible with:
    - [3,63,T]
    - [3,T,63]
    - [3,T,21,3]
    - [3,21,3,T]
    """
    if imfs.ndim == 3 and imfs.shape[0] == 3 and imfs.shape[1] == 63:
        return imfs
    if imfs.ndim == 3 and imfs.shape[0] == 3 and imfs.shape[2] == 63:
        return imfs.transpose(0, 2, 1)
    if imfs.ndim == 4 and imfs.shape[0] == 3 and imfs.shape[2] == 21 and imfs.shape[3] == 3:
        T = imfs.shape[1]
        return imfs.reshape(3, T, 63).transpose(0, 2, 1)
    if imfs.ndim == 4 and imfs.shape[0] == 3 and imfs.shape[1] == 21 and imfs.shape[2] == 3:
        T = imfs.shape[3]
        return imfs.reshape(3, 63, T)
    raise ValueError(f"Unsupported IMF shape: {tuple(imfs.shape)}")


class ImfMostNpzDataset(Dataset):
    """MoST preprocessed clips -> BVH rotvec63 IMF labels dataset.

    Input:
    - root/train.npz: clips shape (N, 11, max_frame, 21) with NaN padding on tail
    - root/distribution.npz: Xmean/Xstd shape (11,1,21)
    - root/manifest_train.jsonl: idx<->seg_id mapping
    - root/splits_mostcontent9_v1/{train,val,test}.txt: seg_id lists

    Label:
    - imf_dir/<seg_id>.npy: typically (3,T,21,3) (rotvec63) -> canonicalized to (3,63,T)

    The dataset returns variable-length cropped windows aligned for motion and IMFs,
    with unit_length alignment and shared random start (similar to ImfDecompDataset).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        split_dir: Optional[str] = None,
        imf_dir: Optional[str] = None,
        max_motion_length: int = 196,
        min_motion_length: int = 40,
        unit_length: int = 4,
        debug: bool = False,
    ):
        super().__init__()
        self.root = os.path.abspath(root)
        self.split = split
        self.max_motion_length = int(max_motion_length)
        self.min_motion_length = int(min_motion_length)
        self.unit_length = int(unit_length)
        self.debug = bool(debug)

        # paths
        if split_dir is None:
            self.split_dir = pjoin(self.root, "splits_mostcontent9_v1")
        else:
            if os.path.isabs(split_dir):
                self.split_dir = split_dir
            else:
                cand = pjoin(self.root, split_dir)
                self.split_dir = cand if os.path.isdir(cand) else os.path.abspath(split_dir)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"splits dir not found: {self.split_dir}")

        if imf_dir is None:
            self.imf_dir = None
        else:
            if os.path.isabs(imf_dir):
                self.imf_dir = imf_dir
            else:
                cand = pjoin(self.root, imf_dir)
                self.imf_dir = cand if os.path.isdir(cand) else os.path.abspath(imf_dir)
        if self.imf_dir is None or (not os.path.isdir(self.imf_dir)):
            raise FileNotFoundError(f"IMF dir not found: {self.imf_dir}")

        self.train_npz_path = pjoin(self.root, "train.npz")
        self.dist_npz_path = pjoin(self.root, "distribution.npz")
        self.manifest_path = pjoin(self.root, "manifest_train.jsonl")

        # load stats for motion (from distribution.npz)
        dist = _safe_load_npz(self.dist_npz_path)
        if ("Xmean" not in dist) or ("Xstd" not in dist):
            raise KeyError(f"distribution.npz missing Xmean/Xstd: {self.dist_npz_path}")
        Xmean = np.asarray(dist["Xmean"])
        Xstd = np.asarray(dist["Xstd"])
        if Xmean.shape != (11, 1, 21) or Xstd.shape != (11, 1, 21):
            raise ValueError(f"Unexpected Xmean/Xstd shape: {Xmean.shape} / {Xstd.shape} (expected (11,1,21))")
        self.motion_mean = Xmean.reshape(-1).astype(np.float32)  # (231,)
        self.motion_std = np.maximum(Xstd.reshape(-1).astype(np.float32), 1e-8)  # (231,)

        # load stats for IMF labels (must exist; generated by tools/compute_imf_most_dataset_stats.py)
        mean_imf_path = pjoin(self.root, "Mean_imf.npy")
        std_imf_path = pjoin(self.root, "Std_imf.npy")
        if not (os.path.exists(mean_imf_path) and os.path.exists(std_imf_path)):
            raise FileNotFoundError(
                f"Missing Mean_imf/Std_imf under {self.root}. "
                f"Please run tools/compute_imf_most_dataset_stats.py first.\n"
                f"  - expected: {mean_imf_path}\n"
                f"  - expected: {std_imf_path}\n"
            )
        self.mean_imf = np.load(mean_imf_path).astype(np.float32)  # (3,63)
        self.std_imf = np.load(std_imf_path).astype(np.float32)
        if self.mean_imf.shape != (3, 63) or self.std_imf.shape != (3, 63):
            raise ValueError(
                f"Unexpected Mean_imf/Std_imf shape: {self.mean_imf.shape} / {self.std_imf.shape} (expected (3,63))"
            )

        # build seg_id list from split file
        split_file = pjoin(self.split_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file, "r") as f:
            seg_ids = [ln.strip() for ln in f if ln.strip()]

        # build seg_id -> idx mapping
        seg2idx = _load_manifest_seg2idx(self.manifest_path)

        # load clips array once (fork-friendly on Linux: workers share pages by COW)
        npz = _safe_load_npz(self.train_npz_path)
        if "clips" not in npz:
            raise KeyError(f"train.npz missing 'clips': {self.train_npz_path}")
        self.clips = np.asarray(npz["clips"])  # (N,11,max_frame,21)

        self.samples: List[Tuple[str, int]] = []
        for sid in seg_ids:
            idx = seg2idx.get(sid, None)
            if idx is None:
                if self.debug:
                    print(f"[ImfMostNpzDataset] skip {sid}: missing in manifest")
                continue
            imf_path = pjoin(self.imf_dir, sid + ".npy")
            if not os.path.exists(imf_path):
                if self.debug:
                    print(f"[ImfMostNpzDataset] skip {sid}: missing imf label")
                continue
            self.samples.append((sid, int(idx)))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found for split={split} under {self.root}")

        if self.debug:
            print(
                f"[ImfMostNpzDataset] root={self.root}, split={split}, samples={len(self.samples)}, "
                f"split_dir={self.split_dir}, imf_dir={self.imf_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        sid, clip_idx = self.samples[idx]

        clip = self.clips[clip_idx]  # (11,T,21)
        motion_full = _flatten_clips_to_motion(np.asarray(clip))  # (T,231), may contain NaNs
        T_full = motion_full.shape[0]

        valid_len = _infer_valid_length_from_nan(motion_full)
        if valid_len < self.min_motion_length:
            if self.debug:
                print(f"[ImfMostNpzDataset] skip {sid}: valid_len={valid_len} < min={self.min_motion_length}")
            return None

        # Determine m_length (unit_length alignment) following ImfDecompDataset logic
        m_length = int(valid_len)
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"
        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length

        if m_length > valid_len:
            m_length = int(valid_len)
        if self.max_motion_length is not None and m_length > int(self.max_motion_length):
            m_length = int(self.max_motion_length)
        if m_length < self.unit_length:
            return None

        start = np.random.randint(0, max(int(valid_len) - m_length + 1, 1))
        end = start + m_length

        motion = motion_full[start:end]  # (m_length,231), may contain NaNs if sporadic
        # replace any remaining NaNs/Infs with 0 (loss masking happens later, but keep finite)
        motion = np.where(np.isfinite(motion), motion, 0.0).astype(np.float32)

        # load & crop IMF label
        imf_path = pjoin(self.imf_dir, sid + ".npy")
        imfs_raw = np.load(imf_path)
        imfs = _canonicalize_imfs(imfs_raw)  # (3,63,T_label)

        # common off-by-one fix (some pipelines save T+1)
        if imfs.shape[2] == T_full + 1:
            imfs = imfs[:, :, :-1]

        if imfs.shape[2] < end:
            if self.debug:
                print(f"[ImfMostNpzDataset] skip {sid}: imf shorter than motion slice ({imfs.shape[2]} < {end})")
            return None
        imfs = imfs[:, :, start:end]  # (3,63,m_length)

        # normalize motion with distribution mean/std
        motion = (motion - self.motion_mean[None, :]) / self.motion_std[None, :]

        # normalize imfs with Mean_imf/Std_imf
        mean_imf_exp = self.mean_imf[:, :, None]
        std_imf_exp = np.maximum(self.std_imf[:, :, None], 1e-8)
        imfs = (imfs.astype(np.float32) - mean_imf_exp) / std_imf_exp

        return {
            "id": sid,
            "motion": motion.astype(np.float32),  # (T,231)
            "imfs": imfs.astype(np.float32),      # (3,63,T)
            "length": int(m_length),
        }


def build_most_dataloader(
    root: str,
    split: str = "train",
    split_dir: Optional[str] = None,
    imf_dir: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    max_motion_length: int = 196,
    min_motion_length: int = 40,
    unit_length: int = 4,
    debug: bool = False,
) -> DataLoader:
    dataset = ImfMostNpzDataset(
        root=root,
        split=split,
        split_dir=split_dir,
        imf_dir=imf_dir,
        max_motion_length=max_motion_length,
        min_motion_length=min_motion_length,
        unit_length=unit_length,
        debug=debug,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=imf_collate_fn,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return loader

