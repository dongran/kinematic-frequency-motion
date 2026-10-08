import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/randong/mydata/MCM-LDM/release/Dual-Style-HHT")
sys.path.insert(0, "/home/randong/mydata/MCM-LDM/release/Dual-Style-HHT/notebooks")

import smpl_mesh

src = Path(
    "/home/randong/mydata/MCM-LDM/datasets/FineMotion/humanml3d_20hz_for_Tranning263/"
    "finemotion_263_v4/new_joint_vecs/"
    "motionx_clean_30hz__idea400__subset_0052__Simultaneously_Turn_The_Neck_And_Walking_clip_1.npy"
)
pkl = Path("/home/randong/mydata/MCM-LDM/deps/smpl_models/smpl/SMPL_NEUTRAL.pkl")
feats = np.load(src)
print("feats", feats.shape)
verts, faces = smpl_mesh.feats_to_vertices(feats[:40], pkl, device="cpu")
print("verts", verts.shape, "span", verts.reshape(-1, 3).max(0) - verts.reshape(-1, 3).min(0))
out = Path("/tmp/persona_content_smpl.gif")
smpl_mesh.save_mesh_gif(verts, faces, out, title="persona content")
print("saved", out, out.stat().st_size)
