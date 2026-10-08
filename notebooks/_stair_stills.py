#!/usr/bin/env python3
"""Still frames and a root-height curve for the stair control run."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import transfer_runtime as tr

REPO = Path("/data1/randong/MCM-LDM/release/Dual-Style-HHT")
VEC = Path(
    "/home/randong/mydata/MCM-LDM/datasets/FineMotion/"
    "humanml3d_20hz_for_Tranning263/finemotion_263_v4/new_joint_vecs"
)
OUT = REPO / "outputs" / "stair_control"


def main() -> None:
    content = tr.features_to_joints(tr.load_features(VEC / "010511.npy")[0])
    style = tr.features_to_joints(
        tr.load_features(VEC / "lafan1_30hz__multipleActions1_subject2Re__s000880_e001077.npy")[0]
    )
    results = {
        name: np.load(OUT / name / "transferred_joints.npy")
        for name in ("fine0", "fine1", "fine3")
    }
    def follow(pose: np.ndarray):
        center = pose.mean(axis=0)
        return (
            (center[0] - 0.85, center[0] + 0.85),
            (center[2] - 0.85, center[2] + 0.85),
            (center[1] - 0.15, center[1] + 1.55),
        )

    frames = [
        ("walking", 20),
        ("stepping up", 90),
        ("highest step", 106),
        ("stepping down", 128),
    ]
    columns = [
        ("Content", content),
        ("Fine 0", results["fine0"]),
        ("Fine 1", results["fine1"]),
        ("Fine 3", results["fine3"]),
    ]
    fig, axes = plt.subplots(4, 4, figsize=(12, 13), subplot_kw={"projection": "3d"})
    for row, (label, frame_id) in enumerate(frames):
        for col, (title, seq) in enumerate(columns):
            ax = axes[row, col]
            pose = tr._frame_at(seq, frame_id)
            tr._draw_pose(ax, pose, follow(pose))
            ax.view_init(elev=12, azim=-58)
            shown = min(frame_id, len(seq) - 1)
            ax.set_title(f"{title}  {label}\nframe {shown}", fontsize=9)
    fig.suptitle("Stair content, coarse 2.5. Camera follows the body.", fontsize=13)
    fig.tight_layout()
    still = OUT / "stair_frames.png"
    fig.savefig(still, dpi=130)
    plt.close(fig)

    for name, seq in [("content", content), *results.items()]:
        speed = np.linalg.norm(np.diff(seq, axis=0), axis=-1).mean()
        print(f"{name} mean_joint_speed={speed:.4f}")

    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    t = np.arange(len(content)) / 20.0
    ax.plot(t, content[:, 0, 1], label="content", color="#333333", linewidth=2)
    colors = {"fine0": "#4C78A8", "fine1": "#F58518", "fine3": "#E45756"}
    for name, seq in results.items():
        n = min(len(seq), len(t))
        ax.plot(t[:n], seq[:n, 0, 1], label=name, color=colors[name], linewidth=1.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("root height (m)")
    ax.set_title("Root height stays on the stair path")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    curve = OUT / "root_height.png"
    fig.savefig(curve, dpi=130)
    plt.close(fig)
    print(f"peak_frame={peak} ground_frame={ground}")
    print(still)
    print(curve)


if __name__ == "__main__":
    main()
