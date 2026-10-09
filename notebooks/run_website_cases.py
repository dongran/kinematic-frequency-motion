"""Run the shipped qualitative cases.

Defaults use the clips in data/examples. Pass --content and --style to replace a pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import transfer_runtime as tr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="cuda:0 or cpu. Default: CUDA if it is available.")
    args = parser.parse_args()

    repo = tr.find_repo_root(Path(__file__).resolve().parents[1])
    turn_content = repo / "data" / "examples" / "turn_footwork" / "content"
    turn_style = repo / "data" / "examples" / "turn_footwork" / "style"
    dance_content = repo / "data" / "examples" / "dancekick" / "content"
    dance_style = repo / "data" / "examples" / "dancekick" / "style"
    throw = repo / "data" / "examples" / "throw_exag"
    cases = [
        ("dance_walk", turn_content, turn_style, 2.5, 10.0),
        ("dance_c5_f5", turn_content, turn_style, 5.0, 5.0),
        ("dance_c5_f7", turn_content, turn_style, 5.0, 7.0),
        ("dancekick", dance_content, dance_style, 2.5, 10.0),
        ("throw_exag", throw, throw, 2.5, 10.0),
    ]
    model = tr.load_model(repo, device=args.device)
    for name, content, style, scale, fine in cases:
        out = repo / "outputs" / "website_cases" / name
        print(f"\n=== {name} scale={scale} fine={fine} ===")
        report = tr.run_pair(model, content, style, out, scale=scale, fine_scale=fine, seed=0)
        tr._print_report(report)


if __name__ == "__main__":
    main()
