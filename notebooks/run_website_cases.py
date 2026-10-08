"""Run the website dance-control cases and one throw exaggeration.

Persona style features are not in this script: the 263-d PerMo punch file
is not on the lab disk. The throw case uses one FineMotion throw as both
content and style, with the fine scale turned up.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

import transfer_runtime as tr


def _pick_throw(finemotion_vecs: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "throw.npy"
    if dest.is_file():
        return dest
    candidates = sorted(finemotion_vecs.glob("*Throw*.npy"))
    best = None
    best_len = 10**9
    for path in candidates:
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or array.shape[-1] != 263:
            continue
        length = int(array.shape[0])
        if 60 <= length <= 180 and length < best_len:
            best = path
            best_len = length
    if best is None:
        raise FileNotFoundError(f"No [T, 263] throw feature under {finemotion_vecs}")
    shutil.copyfile(best, dest)
    print(f"throw source: {best.name} T={best_len}")
    return dest


def main() -> None:
    repo = tr.find_repo_root(Path(__file__).resolve().parents[1])
    throw = _pick_throw(
        Path("/home/randong/mydata/MCM-LDM/datasets/FineMotion/humanml3d_20hz_for_Tranning263/finemotion_263_v4/new_joint_vecs"),
        repo / "data" / "examples" / "throw_exag",
    )
    cases = [
        ("dance_walk", repo / "data/examples/turn_footwork/content", repo / "data/examples/turn_footwork/style", 2.5, 10.0),
        ("dance_c5_f5", repo / "data/examples/turn_footwork/content", repo / "data/examples/turn_footwork/style", 5.0, 5.0),
        ("dance_c5_f7", repo / "data/examples/turn_footwork/content", repo / "data/examples/turn_footwork/style", 5.0, 7.0),
        ("dancekick", repo / "data/examples/dancekick/content", repo / "data/examples/dancekick/style", 2.5, 10.0),
        ("throw_exag", throw.parent, throw.parent, 2.5, 10.0),
    ]
    model = tr.load_model(repo)
    for name, content, style, scale, fine in cases:
        out = repo / "outputs" / "website_cases" / name
        print(f"\n=== {name} scale={scale} fine={fine} ===")
        report = tr.run_pair(model, content, style, out, scale=scale, fine_scale=fine, seed=0)
        tr._print_report(report)


if __name__ == "__main__":
    main()
