# Included examples

Each example is one content motion and one style motion. The demo script feeds the style motion to both the coarse and fine branches.

Features are HumanML-263 arrays, shape `[T, 263]`, from the FineMotion representation used to train the released weights. The demo applies `data/stats/Mean.npy` and `data/stats/Std.npy`.

| Example | Content | Style | Command |
|---|---|---|---|
| `turn_footwork` | turn / footwork (`011973`) | periodic dance kick (`013476`) | `bash scripts/run_demo.sh` |
| `dancekick` | dance kick (`005411`) | periodic dance kick (`013476`) | `EXAMPLE=dancekick bash scripts/run_demo.sh` |

Rendered versions of these transfers, and transfers at other guidance scales, are on the project page.
