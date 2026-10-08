from os.path import join as pjoin

import numpy as np
import torch

from .base import BASEDataModule


class Nao23MotionDataset(torch.utils.data.Dataset):
    """SA26-shaped NAO23 motion dataset: npy [T, 23] plus dummy text fields."""

    def __init__(
        self,
        split_file,
        split="train",
        mean=None,
        std=None,
        max_motion_length=196,
        min_motion_length=40,
        max_text_len=20,
        unit_length=4,
        motion_dir=None,
        tiny=False,
        debug=False,
        progress_bar=True,
        max_ids=0,
        **kwargs,
    ):
        super().__init__()
        self.split = split
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.max_motion_length = int(max_motion_length)
        self.min_motion_length = int(min_motion_length)
        self.max_text_len = int(max_text_len)
        self.unit_length = int(unit_length)
        self.motion_dir = motion_dir
        self.nfeats = int(self.mean.shape[-1])
        self.njoints = self.nfeats
        dummy_len = self.max_text_len + 2
        self._dummy_word = np.zeros((dummy_len, 300), dtype=np.float32)
        self._dummy_pos = np.zeros((dummy_len, 15), dtype=np.float32)

        with open(split_file, "r", encoding="utf-8") as handle:
            ids = [line.strip() for line in handle if line.strip()]
        if tiny or debug:
            ids = ids[: min(32, len(ids))]
        if int(max_ids) > 0:
            ids = ids[: int(max_ids)]

        self.name_list = []
        self.length_arr = []
        iterator = ids
        if progress_bar:
            try:
                from tqdm import tqdm

                iterator = tqdm(ids, desc=f"nao23[{split}]")
            except Exception:
                iterator = ids
        for clip_id in iterator:
            path = pjoin(self.motion_dir, clip_id + ".npy")
            try:
                motion = np.load(path, mmap_mode="r")
            except Exception:
                continue
            if motion.ndim != 2 or int(motion.shape[1]) != self.nfeats:
                continue
            frames = int(motion.shape[0])
            if frames < self.min_motion_length:
                continue
            self.name_list.append(clip_id)
            self.length_arr.append(frames)
        if not self.name_list:
            raise RuntimeError(f"No NAO23 clips found for split={split} in {split_file}")
        self.length_arr = np.asarray(self.length_arr, dtype=np.int64)
        if self.nfeats is None and self.name_list:
            sample = np.load(pjoin(self.motion_dir, self.name_list[0] + ".npy"))
            self.nfeats = int(sample.shape[1])

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        clip_id = self.name_list[item]
        motion = np.load(pjoin(self.motion_dir, clip_id + ".npy")).astype(np.float32)
        frames = int(motion.shape[0])
        m_length = frames
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"
        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
        m_length = max(self.unit_length, min(m_length, frames, self.max_motion_length))
        start = int(np.random.randint(0, max(frames - m_length + 1, 1)))
        motion = motion[start : start + m_length]
        motion = (motion - self.mean) / np.maximum(self.std, 1e-8)
        return {
            "motion": motion.astype(np.float32),
            "text": clip_id,
            "length": int(m_length),
            "word_embs": self._dummy_word.copy(),
            "pos_ohot": self._dummy_pos.copy(),
            "text_len": np.array(1, dtype=np.int64),
            "tokens": ["unk/OTHER"],
        }


class NAO23DataModule(BASEDataModule):
    def __init__(
        self,
        cfg,
        batch_size,
        num_workers,
        collate_fn=None,
        phase="train",
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        self.save_hyperparameters(logger=False)
        self.name = "nao23"
        self.njoints = 23
        self.Dataset = Nao23MotionDataset
        self.cfg = cfg
        sample_overrides = {
            "split": "val",
            "tiny": True,
            "progress_bar": False,
            "max_ids": int(getattr(getattr(cfg, "DATA", None), "MAX_IDS", 0) or 0),
        }
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        self.nfeats = self._sample_set.nfeats

    def get_sample_set(self, overrides=None):
        sample_params = dict(self.hparams)
        sample_params.update(overrides or {})
        sample_params.setdefault(
            "max_ids",
            int(getattr(getattr(self.cfg, "DATA", None), "MAX_IDS", 0) or 0),
        )
        split_file = pjoin(
            eval(f"self.cfg.DATASET.{self.name.upper()}.SPLIT_ROOT"),
            self.cfg.EVAL.SPLIT + ".txt",
        )
        return self.Dataset(split_file=split_file, **sample_params)

    def feats2joints(self, features):
        mean = torch.as_tensor(self.hparams.mean, device=features.device, dtype=features.dtype)
        std = torch.as_tensor(self.hparams.std, device=features.device, dtype=features.dtype)
        return features * std + mean

    def joints2feats(self, features):
        mean = np.asarray(self.hparams.mean, dtype=np.float32)
        std = np.asarray(self.hparams.std, dtype=np.float32)
        return (np.asarray(features, dtype=np.float32) - mean) / np.maximum(std, 1e-8)

    def renorm4t2m(self, features):
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(
                self.name_list,
                min(int(self.cfg.TEST.MM_NUM_SAMPLES), len(self.name_list)),
                replace=False,
            )
            self.test_dataset.name_list = list(self.mm_list)
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
