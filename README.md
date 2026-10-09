# Learning Kinematic Frequency-Aware Disentanglement for Motion Style Transfer and Editing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dongran/kinematic-frequency-motion/blob/main/notebooks/kinematic_frequency_demo.ipynb)

This repository accompanies **SIGGRAPH Asia 2026** and provides:

- inference code for the dual-style latent diffusion model
- the Colab notebook and the example HumanML-263 clips
- the checksums for the five released weight files

![Teaser](asset/teaser.jpg)

Paper figures and results are on the project page:
[kinematic-frequency-motion](https://www.dr-lab.org/projects/kinematic-frequency-motion/).

This repository runs the transfer and writes joint positions. Mesh rendering is a separate Blender step and needs an SMPL body model, which is not included. Training follows the four scripts below. The FineMotion clips themselves are not in this repository.

### Weights

The figure is the released model. Panel (a) is one transfer. Panel (b) is the IMF extractor that splits a motion into high, mid, and low bands. Five files, about 2.4 GB in total, cover the blocks below. Click a name to download that file into the path shown in the table.

![Dual-style network](asset/model-all.jpg)

The content motion is encoded by the motion VAE into a content latent, and its root path is a separate trajectory condition. The style motion is passed through the IMF extractor. High and mid bands become the fine-style condition. A summary of the low band is encoded by MotionCLIP into the coarse-style token. The dual-style denoiser is the network trained in the last step. The VAE decoder turns its latent back into a motion. The dashed box on the right is used only while training: the same frozen IMF extractor and the frequency losses supervise the generated motion. They are not another checkpoint. The released IMF extractor and the denoiser are one trained set. Use the five files together.

`contact_timing.pt` is a small frozen predictor of foot contact on the content motion. It conditions the trajectory branch. The denoiser also learns its own contact-timing encoder, and those weights are inside `dual_style_denoiser.ckpt`. The four training scripts below follow this split: steps 1–3 produce the frozen files, and step 4 trains the denoiser.

| File | Size | What it is |
| --- | --- | --- |
| [`checkpoints/motion_vae.ckpt`](https://www.dr-lab.org/projects/kinematic-frequency-motion/releases/checkpoints/motion_vae.ckpt) | 929 MB | Motion VAE. Frozen while the denoiser trains. |
| [`checkpoints/imf_extractor.pt`](https://www.dr-lab.org/projects/kinematic-frequency-motion/releases/checkpoints/imf_extractor.pt) | 91 MB | Three-band IMF extractor for this motion set. Train one before the denoiser, and keep it with these files. |
| [`checkpoints/contact_timing.pt`](https://www.dr-lab.org/projects/kinematic-frequency-motion/releases/checkpoints/contact_timing.pt) | 1.0 MB | Content-side contact-timing predictor. Frozen. |
| [`checkpoints/motionclip_checkpoint/motionclip.pth.tar`](https://www.dr-lab.org/projects/kinematic-frequency-motion/releases/checkpoints/motionclip_checkpoint/motionclip.pth.tar) | 217 MB | Pretrained [MotionCLIP from MCM-LDM](https://github.com/XingliangJin/MCM-LDM). Frozen coarse-style token. |
| [`checkpoints/dual_style_denoiser.ckpt`](https://www.dr-lab.org/projects/kinematic-frequency-motion/releases/checkpoints/dual_style_denoiser.ckpt) | 1.03 GB | Dual-style denoiser trained with the released IMF extractor. |

`checkpoints/SHA256SUMS` lists the SHA-256 of each file. CLIP ViT-B/32 is not one of these five files. The `clip` package downloads it on the first run.

```bash
mkdir -p checkpoints/motionclip_checkpoint
cd checkpoints
curl -L -O https://www.dr-lab.org/projects/kinematic-frequency-motion/releases/checkpoints/motion_vae.ckpt
curl -L -O https://www.dr-lab.org/projects/kinematic-frequency-motion/releases/checkpoints/imf_extractor.pt
curl -L -O https://www.dr-lab.org/projects/kinematic-frequency-motion/releases/checkpoints/contact_timing.pt
curl -L -O https://www.dr-lab.org/projects/kinematic-frequency-motion/releases/checkpoints/dual_style_denoiser.ckpt
curl -L -o motionclip_checkpoint/motionclip.pth.tar \
  https://www.dr-lab.org/projects/kinematic-frequency-motion/releases/checkpoints/motionclip_checkpoint/motionclip.pth.tar
```

### Training

Train on HumanML-263 features at 20 Hz. `configs/assets_finemotion.yaml` points at the FineMotion set used for the released weights. Those clips are not in this repository. One GPU is enough for each script. The diffusion code is `mld`. The IMF extractor code is `imf_extractor`. Layer sizes, learning rates, and loss weights are in the YAML file for each step.

#### Step 1: Motion VAE

Train the motion VAE, then keep it frozen. Settings are in `configs/motion_vae_finemotion.yaml`.

```bash
bash scripts/train_motion_vae.sh
```

#### Step 2: IMF extractor

Train the IMF extractor before the denoiser. It splits a motion into high, mid, and low bands, and the denoiser uses those bands to disentangle style from content. A different motion set needs its own extractor. The released `imf_extractor.pt` belongs with the released denoiser; keep the five files together when you use this checkpoint. Settings are in `configs/imf_pose69_teacher.yaml`.

```bash
bash scripts/train_imf_extractor.sh
```

#### Step 3: Contact-timing predictor

Train the foot-contact predictor on the content motion, then keep it frozen. Each clip needs a contact-label `.npz` under `DATA.LABEL_ROOT` in `configs/contact_timing_finemotion.yaml`.

```bash
bash scripts/train_contact_timing.sh
```

#### MotionCLIP

MotionCLIP is not trained here. The coarse-style token uses the pretrained checkpoint from [MCM-LDM](https://github.com/XingliangJin/MCM-LDM) ([project page](https://xingliangjin.github.io/MCM-LDM-Web/)). The file in this repository is that checkpoint. It stays frozen.

#### Step 4: Dual-style denoiser

Train the denoiser with the motion VAE, the IMF extractor, the contact-timing predictor, and MotionCLIP frozen. Settings and loss weights are in `configs/dual_style_finemotion_scratch.yaml`.

```bash
bash scripts/train_dual_style.sh
```

Normalization uses `data/stats/Mean.npy` and `data/stats/Std.npy`. Inference uses DDIM with 50 steps. The paper protocol is global scale **2.5** and fine scale **1.5**.

### Run a transfer

```bash
pip install -r requirements.txt
bash scripts/run_demo.sh
```

`scripts/run_demo.sh` uses the turning-walk example at the paper scales. The qualitative dance setting is:

```bash
STYLE_SCALE=2.5 FINE_SCALE=10 bash scripts/run_demo.sh
```

The command writes two joint files:

- `outputs/.../joints_raw/*.npy` is the diffusion output. Paper numbers use this file.
- `outputs/.../joints/*.npy` is the same motion after the foot-contact lock. Render this file.

Foot cleanup is on by default and only moves joint positions. Pass `--no_foot_fix` to skip it.

The first cell of the notebook clones this repository and downloads the five weight files. In the example cell, `FOOT_FIX = False` shows the raw joints.

### Your own motions

Use a raw HumanML3D feature, shape `[T, 263]`, 20 Hz. Do not normalize it yourself.

SMPL motion (`poses` and `trans` in an `.npz`) is converted by the [HumanML3D](https://github.com/EricGuo5513/HumanML3D) notebooks `raw_pose_processing.ipynb` and then `motion_representation.ipynb`. Keep the `new_joint_vecs` array.

A motion-capture BVH file is not read here. Retarget it to SMPL with [tempo-changing-music2motion](https://github.com/dongran/tempo-changing-music2motion), then run the two HumanML3D notebooks.

### Generated motions

Both stick figures are rendered after the default foot-contact cleanup.

Turning walk with a dance-kick style, coarse scale 2.5 and fine scale 10.

![Turning walk with dance-kick style, after foot-contact cleanup](asset/turning_walk.gif)

Stair walk with an energetic style, coarse scale 2.5 and fine scale 3. The camera keeps one scale for the whole clip, so the step up and the step down stay visible.

![Stair walk, fine scale 3, after foot-contact cleanup](asset/stair_fine3.gif)

### Mesh rendering

The same transfers can be drawn as an SMPL mesh in Blender. This step is not part of the Colab notebook. Install [Blender](https://www.blender.org/download/) yourself. The SMPL body model is not in this repository: download `SMPL_NEUTRAL.pkl` from the [SMPL website](https://smpl.is.tue.mpg.de/) and place it at `third_party/smpl/SMPL_NEUTRAL.pkl`.

The renderer reads an `.npz` with `poses` and `trans`. `tools/motion_ik_smpl_bvh.py` can fit that file from a HumanML-263 clip or from the `joints` array written by the transfer, using [joints2smpl](https://github.com/wangsen1312/joints2smpl). Then:

```bash
BLENDER_BIN=/path/to/blender SMPL_PATH=third_party/smpl \
  bash scripts/render_mesh_blender.sh outputs/mesh/smpl_poses_mesh.npz outputs/mesh.mp4
```

One example, the turning walk with a dance-kick style at coarse 2.5 and fine 10. Left to right: content in gray, style in blue, and the transfer in orange.

![Content, style, and transfer as SMPL meshes](asset/dance_walk_mesh.gif)

### Citation

```bibtex
@inproceedings{dong2026kinematic,
  title={Learning Kinematic Frequency-Aware Disentanglement for Motion Style Transfer and Editing},
  author={Dong, Ran and Xie, Haoran and Yang, Xi},
  booktitle={SIGGRAPH Asia 2026 Conference Papers},
  year={2026},
  note={to appear}
}
```
