from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from prep.frequency_bands.memd.MEMD_all import avgA2, avgFsig


@dataclass(frozen=True)
class AlignmentConfig:
    fs: float = 20.0
    frequency_threshold: float = 0.5
    n_select: int = 3
    alpha: float = 1.0
    beta: float = 0.0


@dataclass(frozen=True)
class AlignmentResult:
    imfs_aligned: np.ndarray  # (n_select, T, J, C) or (n_select, T, channels) depending on input
    selected_indices: np.ndarray  # (n_select,) indices into original IMF axis (excluding residue handling explained in meta)
    target_frequencies: np.ndarray  # (n_select,)
    avgf: np.ndarray  # (K-1,) per-IMF avg frequency (excluding residue)
    avga: np.ndarray  # (K-1,) per-IMF avg amplitude (excluding residue)
    meta_json: str


def _as_str_meta(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        # np scalar -> python scalar
        if isinstance(x, np.ndarray) and x.shape == ():
            return str(x.item())
    except Exception:
        pass
    return str(x)


def _load_npz(path: Path) -> Tuple[Dict[str, np.ndarray], str]:
    data = np.load(str(path), allow_pickle=False)
    out: Dict[str, np.ndarray] = {k: data[k] for k in data.files}
    meta = _as_str_meta(out.get("meta", ""))
    return out, meta


def load_imfs_file(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Load IMFs from either:
    - .npz (keys: imfs, optional x/meta)
    - .npy (raw array, legacy format)
    """
    suf = path.suffix.lower()
    if suf == ".npz":
        d, meta = _load_npz(path)
        if "imfs" not in d:
            raise KeyError(f"Missing 'imfs' in {path}")
        x = d.get("x", None)
        return d["imfs"], x, meta
    if suf == ".npy":
        imfs = np.load(str(path))
        return imfs, None, ""
    raise ValueError(f"Unsupported file type: {path}")


def iter_imf_files(root: Path, *, exts: Sequence[str] = ("npz", "npy")) -> List[Path]:
    exts = tuple([e.lower().lstrip(".") for e in exts])
    paths: List[Path] = []
    for ext in exts:
        paths.extend([p for p in root.rglob(f"*.{ext}") if p.is_file()])
    return sorted(paths)


def _imfs_to_kct(imfs: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """Convert various IMF layouts to (K, channels, T) expected by avgFsig/avgA2."""
    if imfs.ndim == 4:
        # (K, T, J, C) -> (K, channels, T)
        K, T, J, C = imfs.shape
        kct = np.transpose(imfs.reshape(K, T, J * C), (0, 2, 1))
        return kct, {"K": K, "T": T, "J": J, "C": C, "channels": J * C}
    if imfs.ndim == 3:
        # Could be (K, channels, T) or (K, T, channels). Disambiguate by assuming time axis is the one matching the last dim.
        K = int(imfs.shape[0])
        # Heuristic: MEMD_all outputs (K, channels, T) typically; our extract stores (K, T, ...)
        # If second dim is "time-like" and last dim is "channels-like", swap.
        # But for safety, we treat (K, channels, T) as default.
        kct = imfs
        return kct, {"K": K, "channels": int(imfs.shape[1]), "T": int(imfs.shape[2])}
    raise ValueError(f"Unsupported imfs shape: {imfs.shape}")


def _kct_to_original_like(imfs_aligned_kct: np.ndarray, original_imfs: np.ndarray) -> np.ndarray:
    """Convert (n_select, channels, T) back to the original layout of `original_imfs`."""
    if original_imfs.ndim == 4:
        # original: (K, T, J, C)
        n_select, channels, T = imfs_aligned_kct.shape
        _K, _T, J, C = original_imfs.shape
        if channels != J * C:
            raise ValueError(f"Channel mismatch: got {channels}, expected {J*C} for original {original_imfs.shape}")
        ktc = np.transpose(imfs_aligned_kct, (0, 2, 1))  # (n_select, T, channels)
        return ktc.reshape(n_select, T, J, C).astype(np.float32, copy=False)
    if original_imfs.ndim == 3:
        # original: (K, channels, T)
        return imfs_aligned_kct.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported original imfs shape: {original_imfs.shape}")


def compute_imf_avgf_avga(imfs: np.ndarray, *, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (avgf, avga) each shape (K-1,). The last component (residue) is excluded."""
    kct, info = _imfs_to_kct(imfs)
    dt = 1.0 / float(fs)
    # avgFsig/avgA2 return shape (K-1, channels)
    avgf = np.nanmean(avgFsig(kct, dt), axis=1)
    avga = np.nanmean(avgA2(kct, dt), axis=1)
    if avgf.ndim != 1 or avga.ndim != 1:
        raise ValueError(f"Unexpected avgf/avga shapes: {avgf.shape}, {avga.shape} for imfs {imfs.shape} ({info})")
    return avgf.astype(np.float32, copy=False), avga.astype(np.float32, copy=False)


def select_imfs_by_targets(
    avgf: np.ndarray,
    avga: np.ndarray,
    target_frequencies: Sequence[float],
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> List[int]:
    """Match target frequencies sequentially, preventing duplicate IMF selection."""
    selected: List[int] = []
    available = set(range(int(avgf.shape[0])))
    max_avga = float(np.max(avga)) if avga.size else 1e-9
    for tf in target_frequencies:
        distances: List[float] = []
        for i in range(int(avgf.shape[0])):
            if i not in available:
                distances.append(float("inf"))
                continue
            freq_diff = abs(float(avgf[i]) - float(tf))
            amp_penalty = max_avga - float(avga[i])
            distances.append(float(alpha) * freq_diff + float(beta) * amp_penalty)
        closest = int(np.argmin(np.asarray(distances, dtype=np.float64)))
        selected.append(closest)
        available.remove(closest)
    return selected


def compute_target_frequencies(
    files: Iterable[Path],
    *,
    fs: float = 20.0,
    frequency_threshold: float = 0.5,
    n_select: int = 3,
    max_files: Optional[int] = None,
) -> np.ndarray:
    """Compute global target frequencies following your `alignIMFs.py` strategy."""
    collected: List[np.ndarray] = []
    n = 0
    for p in files:
        if max_files is not None and n >= int(max_files):
            break
        try:
            imfs, _x, _meta = load_imfs_file(p)
            avgf, avga = compute_imf_avgf_avga(imfs, fs=fs)  # (K-1,)
        except Exception:
            continue

        valid = avgf >= float(frequency_threshold)
        f = avgf[valid]
        a = avga[valid]
        if f.shape[0] < int(n_select):
            continue

        top_idx = np.argsort(a)[-int(n_select) :]
        top_freqs_sorted = np.sort(f[top_idx])  # ascending
        collected.append(top_freqs_sorted.astype(np.float32, copy=False))
        n += 1

    if not collected:
        raise ValueError(
            "No clip had enough IMFs at or above the frequency threshold. "
            "Use longer motions that contain more than one oscillatory mode."
        )

    arr = np.stack(collected, axis=0)  # (N, n_select)
    target = np.mean(arr, axis=0).astype(np.float32, copy=False)
    # match your implementation: return from high to low
    return target[::-1].copy()


def align_imfs(
    imfs: np.ndarray,
    *,
    target_frequencies: Sequence[float],
    cfg: AlignmentConfig,
    source_id: str = "",
) -> AlignmentResult:
    """Align a single clip's IMFs and return selected IMFs in the same layout as input.

    Notes:
    - avgf/avga are computed for IMFs excluding the last residue/trend (K-1 values).
    - selection is performed after filtering by `frequency_threshold`.
    """
    avgf, avga = compute_imf_avgf_avga(imfs, fs=cfg.fs)  # (K-1,)

    valid = avgf >= float(cfg.frequency_threshold)
    candidate_imf_indices = np.nonzero(valid)[0].astype(np.int64)  # indices within [0..K-2]
    if candidate_imf_indices.shape[0] < int(cfg.n_select):
        raise ValueError(f"Not enough valid IMFs after filtering: {candidate_imf_indices.shape[0]} < {cfg.n_select}")

    f = avgf[valid]
    a = avga[valid]
    chosen_local = select_imfs_by_targets(
        f,
        a,
        target_frequencies,
        alpha=cfg.alpha,
        beta=cfg.beta,
    )
    chosen_local = np.asarray(chosen_local, dtype=np.int64)
    chosen_imf_indices = candidate_imf_indices[chosen_local]  # indices into original IMF axis (excluding residue)

    # Select from original IMFs (exclude residue by limiting to K-1 first).
    imfs_no_residue = imfs[:-1]
    if imfs_no_residue.shape[0] <= int(np.max(chosen_imf_indices)):
        raise ValueError(f"Chosen indices out of range: {chosen_imf_indices} for imfs {imfs.shape}")

    selected_imfs = imfs_no_residue[chosen_imf_indices]
    # Ensure output is float32
    selected_imfs = selected_imfs.astype(np.float32, copy=False)

    meta = {
        "source_id": source_id,
        "fs": float(cfg.fs),
        "frequency_threshold": float(cfg.frequency_threshold),
        "n_select": int(cfg.n_select),
        "alpha": float(cfg.alpha),
        "beta": float(cfg.beta),
        "target_frequencies": [float(x) for x in target_frequencies],
        "selected_indices_excluding_residue": [int(x) for x in chosen_imf_indices.tolist()],
        "note": "selected_indices are 0-based indices into imfs[:-1] (residue removed).",
    }

    # Normalize output layout:
    # if input was (K,T,J,C) -> keep (n_select,T,J,C)
    # if input was (K,channels,T) -> keep (n_select,channels,T)
    return AlignmentResult(
        imfs_aligned=selected_imfs,
        selected_indices=chosen_imf_indices,
        target_frequencies=np.asarray(list(target_frequencies), dtype=np.float32),
        avgf=avgf,
        avga=avga,
        meta_json=json.dumps(meta, ensure_ascii=False),
    )

