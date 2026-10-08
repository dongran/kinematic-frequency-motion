from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .contact_timing_paths import resolve_label_path


def _read_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def build_feature_matrix(arrays: np.lib.npyio.NpzFile, feature_names: Sequence[str]) -> np.ndarray:
    features: list[np.ndarray] = []
    seq_len = None
    for name in feature_names:
        if name not in arrays:
            raise KeyError(f"missing feature '{name}' in {arrays.files}")
        value = np.asarray(arrays[name], dtype=np.float32)
        if value.ndim == 1:
            value = value[:, None]
        if value.ndim != 2:
            raise ValueError(f"feature '{name}' must be [T,C], got {value.shape}")
        if seq_len is None:
            seq_len = value.shape[0]
        elif value.shape[0] != seq_len:
            raise ValueError(
                f"feature '{name}' length mismatch: expected {seq_len}, got {value.shape[0]}"
            )
        features.append(value)
    if not features:
        raise ValueError("feature_names cannot be empty")
    return np.concatenate(features, axis=-1).astype(np.float32)


@dataclass(frozen=True)
class ContactTimingSample:
    stem: str
    path: Path


def load_split_samples(
    *,
    label_root: Path,
    split_file: Path,
    max_items: int = 0,
    strict: bool = False,
) -> list[ContactTimingSample]:
    stems = _read_lines(split_file)
    samples: list[ContactTimingSample] = []
    missing: list[str] = []
    for stem in stems:
        path = resolve_label_path(label_root, stem)
        if path.is_file():
            samples.append(ContactTimingSample(stem=stem, path=path))
        else:
            missing.append(stem)
        if max_items > 0 and len(samples) >= max_items:
            break
    if missing and strict:
        raise FileNotFoundError(
            f"{len(missing)} label files are missing under {label_root}; "
            f"first few: {missing[:5]}"
        )
    return samples


def load_contact_timing_arrays(
    npz_path: Path,
    *,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    arrays = np.load(str(npz_path))
    try:
        features = build_feature_matrix(arrays, feature_names)
        contact = np.asarray(arrays["contact"], dtype=np.float32)
    finally:
        arrays.close()
    if contact.ndim == 1:
        contact = contact[:, None]
    if contact.ndim != 2:
        raise ValueError(f"contact must be [T,C], got {contact.shape}")
    if contact.shape[0] != features.shape[0]:
        raise ValueError(
            f"contact length mismatch: features={features.shape[0]}, contact={contact.shape[0]}"
        )
    return features, contact


class ContactTimingWindowDataset(Dataset):
    def __init__(
        self,
        *,
        label_root: str | Path,
        split_file: str | Path,
        input_features: Sequence[str] = ("hip",),
        window_size: int = 120,
        sampling: str = "random",
        max_items: int = 0,
        strict: bool = False,
    ) -> None:
        self.label_root = Path(label_root).expanduser().resolve()
        self.split_file = Path(split_file).expanduser().resolve()
        self.input_features = tuple(str(name) for name in input_features)
        self.window_size = int(window_size)
        self.sampling = str(sampling).strip().lower()
        self.samples = load_split_samples(
            label_root=self.label_root,
            split_file=self.split_file,
            max_items=max_items,
            strict=strict,
        )
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        if self.sampling not in {"random", "center", "head", "tail"}:
            raise ValueError(f"unsupported sampling mode: {self.sampling}")

    def __len__(self) -> int:
        return len(self.samples)

    def _choose_start(self, seq_len: int) -> int:
        if seq_len <= self.window_size:
            return 0
        if self.sampling == "random":
            return int(np.random.randint(0, seq_len - self.window_size + 1))
        if self.sampling == "center":
            return max((seq_len - self.window_size) // 2, 0)
        if self.sampling == "tail":
            return max(seq_len - self.window_size, 0)
        return 0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        features, contact = load_contact_timing_arrays(
            sample.path,
            feature_names=self.input_features,
        )
        seq_len = int(features.shape[0])
        start = self._choose_start(seq_len)
        stop = min(start + self.window_size, seq_len)
        feature_clip = features[start:stop]
        contact_clip = contact[start:stop]

        x = np.zeros((self.window_size, feature_clip.shape[-1]), dtype=np.float32)
        y = np.zeros((self.window_size, contact_clip.shape[-1]), dtype=np.float32)
        mask = np.zeros((self.window_size,), dtype=np.float32)
        valid = feature_clip.shape[0]
        if valid > 0:
            x[:valid] = feature_clip
            y[:valid] = contact_clip
            mask[:valid] = 1.0
            if valid < self.window_size:
                x[valid:] = feature_clip[-1]
                y[valid:] = contact_clip[-1]

        return {
            "name": sample.stem,
            "x": torch.from_numpy(x),
            "contact": torch.from_numpy(y),
            "mask": torch.from_numpy(mask),
            "length": torch.tensor(valid, dtype=torch.long),
        }
