import os
import json
from os.path import join as pjoin
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from data.imf_shape_utils import canonicalize_imfs


class ImfDecompDataset(Dataset):
    """IMF pretraining set on HumanML3D axis-angle features.

    The directory layout matches the HumanML3D preprocessing used by this project:

    - root/
      - HumanML3D-axisAngle-all/HumanML3D/
        - train.txt / val.txt / test.txt (or root/splits/*.txt)
        - new_joint_vecs/*.npy          (raw 263-D features)
        - joints_imf/*.npy              (IMF of shape 3 x 63 x T; imf_dir may point elsewhere)
        - Mean_orig.npy, Std_orig.npy
        - Mean_imf.npy,  Std_imf.npy
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        split_dir: Optional[str] = None,
        imf_dir: Optional[str] = None,
        stats_dir: Optional[str] = None,
        max_motion_length: int = 196,
        min_motion_length: int = 40,
        unit_length: int = 4,
        debug: bool = False,
    ):
        super().__init__()
        self.root = os.path.abspath(root)
        self.split = split
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.debug = debug

        # Paths.
        self.motion_dir = self._resolve_dir(["new_joint_vecs_i", "new_joint_vecs"])
        if imf_dir is not None:
            self.imf_dir = os.path.abspath(imf_dir)
            if not os.path.isdir(self.imf_dir):
                raise FileNotFoundError(f"IMF dir not found: {self.imf_dir}")
        else:
            self.imf_dir = self._resolve_dir(["joints_imf"])

        self.stats_dir = os.path.abspath(stats_dir) if stats_dir is not None else self.root
        if not os.path.isdir(self.stats_dir):
            raise FileNotFoundError(f"Stats dir not found: {self.stats_dir}")

        # Normalization statistics.
        self.mean_orig, self.std_orig = self._load_stats("Mean_orig.npy", "Std_orig.npy")
        self.mean_imf, self.std_imf = self._load_stats("Mean_imf.npy", "Std_imf.npy")
        if self.mean_imf.ndim != 2 or int(self.mean_imf.shape[0]) != 3:
            raise ValueError(f"Expected Mean_imf shape [3,dof], got {tuple(self.mean_imf.shape)}")
        self.motion_dof = int(self.mean_orig.shape[-1])
        self.imf_dof = int(self.mean_imf.shape[-1])

        # Clip ids.
        split_file = self._resolve_split_file(split=split, split_dir=split_dir)
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as f:
            ids = [line.strip() for line in f.readlines() if line.strip()]

        self.samples: List[str] = []
        lengths: List[int] = []

        for sid in ids:
            m_path = pjoin(self.motion_dir, sid + ".npy")
            imf_path = pjoin(self.imf_dir, sid + ".npy")
            if not (os.path.exists(m_path) and os.path.exists(imf_path)):
                if self.debug:
                    print(f"[ImfDecompDataset] skip {sid}: missing motion/imf file")
                continue
            # Only the length is needed here. mmap avoids reading the whole array.
            motion = np.load(m_path, mmap_mode="r")
            if motion.shape[0] < self.min_motion_length or motion.shape[0] >= 200:
                continue
            self.samples.append(sid)
            lengths.append(motion.shape[0])

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found under {self.root} for split={split}")

        self.length_arr = np.asarray(lengths)
        if self.debug:
            print(
                f"[ImfDecompDataset] root={self.root}, split={split}, "
                f"samples={len(self.samples)}, motion_dir={self.motion_dir}, imf_dir={self.imf_dir}"
            )

    # --------------------------------------------------------------------- utils
    def _resolve_split_file(self, split: str, split_dir: Optional[str]) -> str:
        """Two split-file layouts are supported:

        1) root/{train,val,test}.txt
        2) root/splits/{train,val,test}.txt  (this is the finemotion_263_v4 layout)
        An explicit split_dir is also accepted, and an absolute path wins.
        """
        candidates: List[str] = []
        if split_dir:
            candidates.append(pjoin(split_dir, f"{split}.txt"))
        candidates.append(pjoin(self.root, f"{split}.txt"))
        candidates.append(pjoin(self.root, "splits", f"{split}.txt"))

        for p in candidates:
            if os.path.exists(p):
                return p

        # Return the most likely path so the error message can point at it.
        return candidates[0] if candidates else pjoin(self.root, f"{split}.txt")

    def _resolve_dir(self, candidates: List[str]) -> str:
        """Search candidate subdirectories under root, in order."""
        for name in candidates:
            path = pjoin(self.root, name)
            if os.path.isdir(path):
                return path
        raise FileNotFoundError(
            f"None of {candidates} found under {self.root}; "
            "Check the HumanML3D-axisAngle-all/HumanML3D path."
        )

    def _load_stats(self, mean_name: str, std_name: str):
        mean_path = pjoin(self.stats_dir, mean_name)
        std_path = pjoin(self.stats_dir, std_name)
        if not (os.path.exists(mean_path) and os.path.exists(std_path)):
            raise FileNotFoundError(
                f"Missing stats: {mean_path} or {std_path}. "
                "Compute Mean_*.npy and Std_*.npy in the HumanML3D directory first."
            )
        mean = np.load(mean_path)
        std = np.load(std_path)
        return mean, std

    # ----------------------------------------------------------------- Dataset API
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Crop one sample, following Text2MotionDatasetV2 in MCM-LDM.

        - The raw length is T. __init__ already keeps 40 <= T < 200.
        - Align T down to a multiple of unit_length, usually 4:
            * coin2 = "single"  -> m_length = floor(T/4)*4
            * coin2 = "double"  -> m_length = (floor(T/4)-1)*4   (slightly shorter, more variety)
        - Pick a random start and crop motion and IMFs to that length together.
        """
        sid = self.samples[idx]
        m_path = pjoin(self.motion_dir, sid + ".npy")
        imf_path = pjoin(self.imf_dir, sid + ".npy")

        motion = np.load(m_path)  # [T, motion_dof]
        imfs_raw = np.load(imf_path)
        imfs = canonicalize_imfs(imfs_raw, expected_dof=self.imf_dof)  # [3, dof, T']

        if motion.ndim != 2 or int(motion.shape[1]) != self.motion_dof:
            raise ValueError(
                f"Bad motion shape for {sid}: {tuple(motion.shape)} (expected [T,{self.motion_dof}])"
            )

        T = motion.shape[0]
        m_length = T

        # Same length alignment as Text2MotionDatasetV2 (unit_length=4).
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:  # "single"
            m_length = (m_length // self.unit_length) * self.unit_length

        # Keep the cropped length inside a safe range.
        # - not longer than T
        # - not longer than max_motion_length (usually 196)
        if m_length > T:
            m_length = T
        if m_length > self.max_motion_length:
            m_length = self.max_motion_length

        # Drop the sample if the crop collapses below unit_length.
        if m_length < self.unit_length:
            if self.debug:
                print(f"[ImfDecompDataset] m_length too small for {sid}, skip sample")
            return None

        start = np.random.randint(0, max(T - m_length + 1, 1))
        end = start + m_length

        motion = motion[start:end]  # [m_length, 263]

        # Crop the IMF on the same time range.
        if imfs.shape[2] == T + 1:
            imfs = imfs[:, :, :-1]
        if imfs.shape[2] < m_length:
            # Bad sample. Return None so collate_fn can drop it.
            if self.debug:
                print(f"[ImfDecompDataset] IMF shorter than motion for {sid}, skip sample")
            return None
        imfs = imfs[:, :, start:end]  # [3, dof, m_length]

        # Normalize motion with Mean_orig/Std_orig and IMFs with Mean_imf/Std_imf.
        motion = (motion - self.mean_orig) / self.std_orig  # [m_length, motion_dof]

        mean_imf_exp = self.mean_imf[:, :, None]
        std_imf_exp = np.maximum(self.std_imf[:, :, None], 1e-8)
        imfs = (imfs - mean_imf_exp) / std_imf_exp

        sample: Dict[str, Any] = {
            "id": sid,
            "motion": motion.astype(np.float32),  # [T_i, motion_dof]
            "imfs": imfs.astype(np.float32),      # [3, dof, T_i]
            "length": int(m_length),
        }
        return sample


def _collate_tensors(batch: List[torch.Tensor]) -> torch.Tensor:
    """Right-pad variable-length tensors, as in mld.data.utils.collate_tensors.

    - Input: a list of length B. Each item can be [T_i, D], [C1, C2, T_i], and so on.
    - Output: shape [B, ...]. The time axis is padded to the batch maximum T_max.
    """
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def imf_collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
    """Drop None samples and pad like mld_collate in MCM-LDM.

    - Each sample is the variable-length crop from __getitem__, length T_i.
    - Pad `motion` and `imfs` to the batch maximum T_max.
    - Also return each sample's true length for a later mask.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}

    ids = [b["id"] for b in batch]
    motion_list = [torch.from_numpy(b["motion"]) for b in batch]  # each [T_i, 263]
    imf_list = [torch.from_numpy(b["imfs"]) for b in batch]       # each [3, 63, T_i]
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)

    motions = _collate_tensors(motion_list)  # [B, T_max, 263]
    imfs = _collate_tensors(imf_list)        # [B, 3, 63, T_max]

    return {
        "ids": ids,
        "motion": motions,
        "imfs": imfs,
        "length": lengths,
    }


def build_dataloader(
    root: str,
    split: str = "train",
    split_dir: Optional[str] = None,
    imf_dir: Optional[str] = None,
    stats_dir: Optional[str] = None,
    sample_weight_path: Optional[str] = None,
    sample_weight_replacement: bool = True,
    batch_size: int = 64,
    num_workers: int = 4,
    max_motion_length: int = 196,
    min_motion_length: int = 40,
    unit_length: int = 4,
    debug: bool = False,
) -> DataLoader:
    dataset = ImfDecompDataset(
        root=root,
        split=split,
        split_dir=split_dir,
        imf_dir=imf_dir,
        stats_dir=stats_dir,
        max_motion_length=max_motion_length,
        min_motion_length=min_motion_length,
        unit_length=unit_length,
        debug=debug,
    )
    sampler = None
    if sample_weight_path is not None and split == "train":
        weight_path = os.path.abspath(sample_weight_path)
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Sample weight file not found: {weight_path}")
        with open(weight_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        weight_map = payload.get("weights", payload) if isinstance(payload, dict) else {}
        weights = [float(weight_map.get(sid, 1.0)) for sid in dataset.samples]
        weights_t = torch.as_tensor(weights, dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights=weights_t,
            num_samples=len(dataset),
            replacement=bool(sample_weight_replacement),
        )
        print(
            "[ImfDecompDataset] weighted sampler enabled:",
            f"path={weight_path}, min={weights_t.min().item():.4f},",
            f"max={weights_t.max().item():.4f}, mean={weights_t.mean().item():.4f}",
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=imf_collate_fn,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return loader


