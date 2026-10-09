"""Stair walk x energetic style. Coarse scale 2.5, fine scale 0 / 1 / 3.

The default clips are the ones shipped in data/examples/stair.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import transfer_runtime as tr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content", type=Path, default=None, help="Content .npy or a directory with one .npy.")
    parser.add_argument("--style", type=Path, default=None, help="Style .npy or a directory with one .npy.")
    parser.add_argument("--device", default=None, help="cuda:0 or cpu. Default: CUDA if it is available.")
    args = parser.parse_args()

    repo = tr.find_repo_root(Path(__file__).resolve().parents[1])
    content = args.content or (repo / "data" / "examples" / "stair" / "content")
    style = args.style or (repo / "data" / "examples" / "stair" / "style")
    model = tr.load_model(repo, device=args.device)
    for fine in (0.0, 1.0, 3.0):
        name = f"fine{int(fine)}"
        out = repo / "outputs" / "stair_control" / name
        print(f"\n=== {name} scale=2.5 fine={fine} ===", flush=True)
        report = tr.run_pair(
            model,
            content,
            style,
            out,
            scale=2.5,
            fine_scale=fine,
            seed=0,
        )
        tr._print_report(report)


if __name__ == "__main__":
    main()
