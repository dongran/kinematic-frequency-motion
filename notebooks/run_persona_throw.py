"""Persona self-transfer at fine 2, circling walk at fine 2, throw at fine 6 and 10.

The missing PerMo style file is replaced by using the content motion as its own
style. Fine 2 is the circling / rotating-hands setting. Fine 6 and 10 stay on
the throw.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import transfer_runtime as tr
import smpl_mesh

SMPL_PKL = Path("/home/randong/mydata/MCM-LDM/deps/smpl_models/smpl/SMPL_NEUTRAL.pkl")
PERSONA_SRC = Path(
    "/home/randong/mydata/MCM-LDM/datasets/FineMotion/humanml3d_20hz_for_Tranning263/"
    "finemotion_263_v4/new_joint_vecs/"
    "motionx_clean_30hz__idea400__subset_0052__Simultaneously_Turn_The_Neck_And_Walking_clip_1.npy"
)


def main() -> None:
    repo = tr.find_repo_root(Path(__file__).resolve().parents[1])
    persona_dir = repo / "data" / "examples" / "persona_walk"
    persona_dir.mkdir(parents=True, exist_ok=True)
    persona = persona_dir / "motion.npy"
    if not persona.is_file():
        shutil.copyfile(PERSONA_SRC, persona)
    throw = repo / "data" / "examples" / "throw_exag" / "throw.npy"
    cases = [
        ("circle_fine2", repo / "data/examples/turn_footwork/content", repo / "data/examples/turn_footwork/style", 2.5, 2.0),
        ("persona_self_fine2", persona_dir, persona_dir, 2.5, 2.0),
        ("throw_fine6", throw.parent, throw.parent, 2.5, 6.0),
        ("throw_fine10", throw.parent, throw.parent, 2.5, 10.0),
    ]
    model = tr.load_model(repo)
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
        tr.motion_stats("content", content_j)
        tr.motion_stats("style", style_j)
        tr.motion_stats("transferred", joints)
        stick = tr.save_comparison(content_j, style_j, joints, out)
        print("stick", stick["gif"], flush=True)
        vertices, faces = smpl_mesh.feats_to_vertices(raw, SMPL_PKL, device="cpu")
        mesh = smpl_mesh.save_mesh_gif(vertices, faces, out / "smpl.gif", title=name)
        print("mesh", mesh, "verts", vertices.shape, flush=True)


if __name__ == "__main__":
    main()
