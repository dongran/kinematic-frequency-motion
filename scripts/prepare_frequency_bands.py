#!/usr/bin/env python3
"""Prepare the three frequency bands used to train the IMF extractor.

Input: one SMPL-24 ``.npz`` per clip, with ``poses`` (T, 24, 3) axis-angle,
``trans`` (T, 3), and optional ``mocap_framerate`` (default 30).

The script resamples each clip to 20 Hz, runs MEMD on the 69-D kinematic
signal, estimates three dataset frequencies, and writes aligned bands in
high, mid, low order. Clips shorter than 70 frames at 20 Hz are skipped:
MEMD needs more frames than the 69 channels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prep.frequency_bands.alignment import (  # noqa: E402
    AlignmentConfig,
    align_imfs,
    compute_target_frequencies,
)
from prep.frequency_bands.memd.MEMD_all import memd  # noqa: E402
from prep.frequency_bands.pose69 import build_pose69, load_smpl24_npz  # noqa: E402


def _save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **arrays)


def extract_one(path: Path, out_path: Path, *, target_fps: float, num_directions: int, min_frames: int) -> str:
    if out_path.exists():
        return "exists"
    loaded = load_smpl24_npz(path)
    signal = build_pose69(
        loaded.poses,
        loaded.trans,
        src_fps=loaded.src_fps,
        dst_fps=target_fps,
    )
    num_frames = int(signal.shape[0])
    if num_frames < int(min_frames):
        return f"short:{num_frames}"
    flat = signal.reshape(num_frames, -1)
    imfs_kct = memd(flat, int(num_directions))
    if imfs_kct.ndim != 3 or imfs_kct.shape[1] != flat.shape[1] or imfs_kct.shape[2] != num_frames:
        raise ValueError(f"unexpected MEMD shape {imfs_kct.shape} for input {flat.shape}")
    count = int(imfs_kct.shape[0])
    imfs = np.transpose(imfs_kct, (0, 2, 1)).reshape(count, num_frames, 23, 3).astype(np.float32)
    meta = {
        "source": path.name,
        "src_fps": float(loaded.src_fps),
        "target_fps": float(target_fps),
        "T": num_frames,
        "num_directions": int(num_directions),
        "K": count,
        "channel_layout": ["root_rot", "body21", "trans"],
    }
    _save_npz(out_path, imfs=imfs, x=signal, meta=np.asarray(json.dumps(meta)))
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="MEMD and three-band alignment for your own SMPL clips.")
    parser.add_argument("--smpl_dir", type=str, required=True, help="Directory of SMPL-24 npz clips.")
    parser.add_argument("--out_dir", type=str, required=True, help="Where to write base_imfs/, aligned/, and target_frequencies.json.")
    parser.add_argument("--target_fps", type=float, default=20.0)
    parser.add_argument("--num_directions", type=int, default=160, help="MEMD projection directions. 160 is the released setting.")
    parser.add_argument("--frequency_threshold", type=float, default=0.5, help="Keep IMFs at or above this frequency, in Hz.")
    parser.add_argument("--n_select", type=int, default=3)
    parser.add_argument("--min_frames", type=int, default=70)
    args = parser.parse_args()

    smpl_dir = Path(args.smpl_dir)
    out_dir = Path(args.out_dir)
    base_dir = out_dir / "base_imfs"
    aligned_dir = out_dir / "aligned"
    base_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(smpl_dir.glob("*.npz"))
    if not clips:
        raise SystemExit(f"no npz files in {smpl_dir}")

    ok = skipped = failed = 0
    started = time.time()
    for index, path in enumerate(clips, 1):
        out_path = base_dir / f"{path.stem}.npz"
        try:
            status = extract_one(
                path,
                out_path,
                target_fps=float(args.target_fps),
                num_directions=int(args.num_directions),
                min_frames=int(args.min_frames),
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[fail] {path.name}: {type(exc).__name__}: {exc}")
            continue
        if status == "ok" or status == "exists":
            ok += 1
        else:
            skipped += 1
            print(f"[skip] {path.name}: {status}")
        if index == len(clips) or index % 20 == 0:
            print(f"[extract] {index}/{len(clips)} ok={ok} skip={skipped} fail={failed} elapsed={time.time() - started:.1f}s")

    base_files = sorted(base_dir.glob("*.npz"))
    if not base_files:
        raise SystemExit("no MEMD files were written")

    try:
        targets = compute_target_frequencies(
            base_files,
            fs=float(args.target_fps),
            frequency_threshold=float(args.frequency_threshold),
            n_select=int(args.n_select),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    target_path = out_dir / "target_frequencies.json"
    target_path.write_text(
        json.dumps(
            {
                "order": ["high", "mid", "low"],
                "hz": [float(x) for x in targets.tolist()],
                "frequency_threshold_hz": float(args.frequency_threshold),
                "fs_hz": float(args.target_fps),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[targets] high/mid/low Hz = {', '.join(f'{float(x):.4f}' for x in targets)}")

    cfg = AlignmentConfig(
        fs=float(args.target_fps),
        frequency_threshold=float(args.frequency_threshold),
        n_select=int(args.n_select),
        alpha=1.0,
        beta=0.0,
    )
    aligned = align_skipped = 0
    for path in base_files:
        out_path = aligned_dir / path.name
        if out_path.exists():
            aligned += 1
            continue
        imfs = np.load(str(path))["imfs"]
        try:
            result = align_imfs(imfs, target_frequencies=targets, cfg=cfg, source_id=path.stem)
        except ValueError as exc:
            align_skipped += 1
            print(f"[skip-align] {path.name}: {exc}")
            continue
        _save_npz(
            out_path,
            imfs=result.imfs_aligned,
            target_frequencies=result.target_frequencies,
            selected_indices=result.selected_indices,
            meta=np.asarray(result.meta_json),
        )
        aligned += 1
    print(f"[align] wrote {aligned} clips, skipped {align_skipped}. Labels are in {aligned_dir}")
    if aligned == 0:
        raise SystemExit("no clip could be aligned to three bands")
    print("Band 0 is high, band 1 is mid, band 2 is low. These files supervise the IMF extractor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
