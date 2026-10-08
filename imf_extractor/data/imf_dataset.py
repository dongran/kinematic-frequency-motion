import os
import json
from os.path import join as pjoin
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from data.imf_shape_utils import canonicalize_imfs


class ImfDecompDataset(Dataset):
    """HumanML3D 轴角预处理数据上的 IMF 预训练数据集。

    约定数据组织方式与你当前工程中的 HumanML3D 预处理结果一致：

    - root/
      - HumanML3D-axisAngle-all/HumanML3D/
        - train.txt / val.txt / test.txt（或 root/splits/*.txt）
        - new_joint_vecs/*.npy          (原始 263 维特征)
        - joints_imf/*.npy              (3 x 63 x T 的 IMF；也支持外部 imf_dir 指向独立目录)
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

        # 关键路径
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

        # 统计量
        self.mean_orig, self.std_orig = self._load_stats("Mean_orig.npy", "Std_orig.npy")
        self.mean_imf, self.std_imf = self._load_stats("Mean_imf.npy", "Std_imf.npy")
        if self.mean_imf.ndim != 2 or int(self.mean_imf.shape[0]) != 3:
            raise ValueError(f"Expected Mean_imf shape [3,dof], got {tuple(self.mean_imf.shape)}")
        self.motion_dof = int(self.mean_orig.shape[-1])
        self.imf_dof = int(self.mean_imf.shape[-1])

        # ID 列表
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
            # 这里只需要 length，用 mmap 避免全量读入
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
        """支持两种 split 文件布局：

        1) root/{train,val,test}.txt
        2) root/splits/{train,val,test}.txt  （你的 finemotion_263_v4 就是这种）
        也支持显式传入 split_dir（绝对路径优先）。
        """
        candidates: List[str] = []
        if split_dir:
            candidates.append(pjoin(split_dir, f"{split}.txt"))
        candidates.append(pjoin(self.root, f"{split}.txt"))
        candidates.append(pjoin(self.root, "splits", f"{split}.txt"))

        for p in candidates:
            if os.path.exists(p):
                return p

        # 返回一个“最可能”的路径，便于报错定位
        return candidates[0] if candidates else pjoin(self.root, f"{split}.txt")

    def _resolve_dir(self, candidates: List[str]) -> str:
        """在 root 下按顺序查找若干候选子目录。"""
        for name in candidates:
            path = pjoin(self.root, name)
            if os.path.isdir(path):
                return path
        raise FileNotFoundError(
            f"None of {candidates} found under {self.root}; "
            f"please check your HumanML3D-axisAngle-all/HumanML3D 路径。"
        )

    def _load_stats(self, mean_name: str, std_name: str):
        mean_path = pjoin(self.stats_dir, mean_name)
        std_path = pjoin(self.stats_dir, std_name)
        if not (os.path.exists(mean_path) and os.path.exists(std_path)):
            raise FileNotFoundError(
                f"Missing stats: {mean_path} or {std_path}. "
                "请确认已在 HumanML3D 目录下计算好 Mean_*.npy / Std_*.npy。"
            )
        mean = np.load(mean_path)
        std = np.load(std_path)
        return mean, std

    # ----------------------------------------------------------------- Dataset API
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """单个样本的裁剪逻辑，尽量复刻 MCM-LDM 中 Text2MotionDatasetV2 的行为。

        - 原始 motion 长度记为 T，先在 __init__ 中保证 40 <= T < 200；
        - 这里以 T 为基础，按 unit_length=4 对齐到最近的下界：
            * coin2 = "single"  -> m_length = floor(T/4)*4
            * coin2 = "double"  -> m_length = (floor(T/4)-1)*4   （略短一点，增加多样性）
        - 随机选择一个起点，在 motion / imfs 上同时裁出长度为 m_length 的片段。
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

        # 与 Text2MotionDatasetV2 一致的长度对齐策略（unit_length=4）
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:  # "single"
            m_length = (m_length // self.unit_length) * self.unit_length

        # 额外安全防护：确保最终长度在合理范围内
        # - 不大于 T
        # - 不超过 max_motion_length（通常为 196）
        if m_length > T:
            m_length = T
        if m_length > self.max_motion_length:
            m_length = self.max_motion_length

        # 若长度退化到过小，可以直接丢弃该样本
        if m_length < self.unit_length:
            if self.debug:
                print(f"[ImfDecompDataset] m_length too small for {sid}, skip sample")
            return None

        start = np.random.randint(0, max(T - m_length + 1, 1))
        end = start + m_length

        motion = motion[start:end]  # [m_length, 263]

        # IMF 在时间上对齐，同样裁剪
        if imfs.shape[2] == T + 1:
            imfs = imfs[:, :, :-1]
        if imfs.shape[2] < m_length:
            # 样本异常，返回 None 让 collate_fn 过滤
            if self.debug:
                print(f"[ImfDecompDataset] IMF shorter than motion for {sid}, skip sample")
            return None
        imfs = imfs[:, :, start:end]  # [3, dof, m_length]

        # 归一化：motion 使用 Mean_orig/Std_orig；IMF 使用 Mean_imf/Std_imf
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
    """模仿 MCM-LDM 中 mld.data.utils.collate_tensors 的行为，对变长序列做右侧 padding。

    - 输入：一个长度为 B 的列表，每个元素形状可以是 [T_i, D] 或 [C1, C2, T_i] 等；
    - 输出：形状为 [B, ...]，时间维被 pad 到本 batch 内的最大长度 T_max，其余维度按最大值对齐。
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
    """collate：过滤 None，并模仿 MCM-LDM 的 mld_collate 行为进行 padding。

    - 对每个样本使用 __getitem__ 裁剪得到的变长片段（长度记为 T_i）；
    - 在 collate 时，将 `motion` 和 `imfs` 统一 pad 到 batch 内的 T_max；
    - 同时返回每个样本的真实 length，方便将来需要 mask 时使用。
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}

    ids = [b["id"] for b in batch]
    motion_list = [torch.from_numpy(b["motion"]) for b in batch]  # 每个 [T_i, 263]
    imf_list = [torch.from_numpy(b["imfs"]) for b in batch]       # 每个 [3, 63, T_i]
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


