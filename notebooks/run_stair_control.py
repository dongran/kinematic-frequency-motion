"""Stair-walking content x energetic multi-action style.

Website control: coarse scale 2.5, fine scale 0 / 1 / 3.
Content 010511 walks up an obstacle and back down.
Style is the LaFAN multiple-actions clip.
"""

from pathlib import Path

import transfer_runtime as tr

REPO = Path("/data1/randong/MCM-LDM/release/Dual-Style-HHT")
VEC = Path(
    "/home/randong/mydata/MCM-LDM/datasets/FineMotion/"
    "humanml3d_20hz_for_Tranning263/finemotion_263_v4/new_joint_vecs"
)
CONTENT = VEC / "010511.npy"
STYLE = VEC / "lafan1_30hz__multipleActions1_subject2Re__s000880_e001077.npy"


def main() -> None:
    model = tr.load_model(REPO, device="cuda:0")
    for fine in (0.0, 1.0, 3.0):
        name = f"fine{int(fine)}"
        out = REPO / "outputs" / "stair_control" / name
        print(f"\n=== {name} scale=2.5 fine={fine} ===", flush=True)
        report = tr.run_pair(
            model,
            CONTENT,
            STYLE,
            out,
            scale=2.5,
            fine_scale=fine,
            seed=0,
        )
        tr._print_report(report)


if __name__ == "__main__":
    main()
