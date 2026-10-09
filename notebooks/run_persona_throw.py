"""Circling walk at fine 2, a self-styled walk at fine 2, and a throw at fine 6 and 10.

Defaults use the clips in data/examples. The mesh preview needs SMPL_NEUTRAL.pkl,
which is not shipped. Pass --smpl, or place the file at third_party/smpl/SMPL_NEUTRAL.pkl.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import smpl_mesh
import transfer_runtime as tr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smpl", type=Path, default=None, help="Path to SMPL_NEUTRAL.pkl.")
    parser.add_argument("--device", default=None, help="cuda:0 or cpu. Default: CUDA if it is available.")
    args = parser.parse_args()

    repo = tr.find_repo_root(Path(__file__).resolve().parents[1])
    smpl = args.smpl or (repo / "third_party" / "smpl" / "SMPL_NEUTRAL.pkl")
    if not smpl.is_file():
        raise FileNotFoundError(
            f"SMPL_NEUTRAL.pkl was not found at {smpl}. "
            "Download it from https://smpl.is.tue.mpg.de/ and pass --smpl."
        )
    persona = repo / "data" / "examples" / "persona_walk"
    throw = repo / "data" / "examples" / "throw_exag"
    cases = [
        ("circle_fine2", repo / "data/examples/turn_footwork/content", repo / "data/examples/turn_footwork/style", 2.5, 2.0),
        ("persona_self_fine2", persona, persona, 2.5, 2.0),
        ("throw_fine6", throw, throw, 2.5, 6.0),
        ("throw_fine10", throw, throw, 2.5, 10.0),
    ]
    model = tr.load_model(repo, device=args.device)
    for name, content, style, scale, fine in cases:
        out = repo / "outputs" / "website_cases" / name
        print(f"\n=== {name} scale={scale} fine={fine} ===", flush=True)
        content_feats, _ = tr.load_features(content)
        style_feats, _ = tr.load_features(style)
        joints, raw = tr.transfer(
            model,
            content_feats,
            style_feats,
            scale=scale,
            fine_scale=fine,
            seed=0,
            return_raw_feats=True,
        )
        content_j = tr.features_to_joints(content_feats)
        style_j = tr.features_to_joints(style_feats)
        stick = tr.save_comparison(content_j, style_j, joints, out)
        print("stick", stick["gif"], flush=True)
        vertices, faces = smpl_mesh.feats_to_vertices(raw, smpl, device="cpu")
        mesh = smpl_mesh.save_mesh_gif(vertices, faces, out / "smpl.gif", title=name)
        print("mesh", mesh, "verts", vertices.shape, flush=True)


if __name__ == "__main__":
    main()
