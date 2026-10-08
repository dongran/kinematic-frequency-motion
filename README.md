# Learning Kinematic Frequency-Aware Disentanglement for Motion Style Transfer and Editing

This repository accompanies **SIGGRAPH Asia 2026** and provides:

- inference code for the dual-style latent diffusion model
- the Colab notebook and the example HumanML-263 clips
- the checksums for the five released weight files

![Teaser](asset/teaser.jpg)

Paper figures and results are on the project page:
[kinematic-frequency-motion](https://www.dr-lab.org/projects/kinematic-frequency-motion/).

This repository runs the transfer and writes joint positions. It does not include SMPL mesh or Blender rendering. The training launchers and the FineMotion split are also not in this repository. The sections below record how the released networks were trained, and where to put the weights.

### Weights

Five files, about 2.4 GB in total. Download them from:

https://www.dr-lab.org/projects/kinematic-frequency-motion/releases/

| File | Size | What it is |
| --- | --- | --- |
| `checkpoints/motion_vae.ckpt` | 929 MB | Motion VAE, epoch 599. Latent shape 7 × 256. Frozen during diffusion training. |
| `checkpoints/imf_extractor.pt` | 91 MB | Three-band IMF extractor. Frozen. Do not replace this file with a later extractor. |
| `checkpoints/contact_timing.pt` | 1.0 MB | Content-side contact-timing predictor. Frozen. |
| `checkpoints/motionclip_checkpoint/motionclip.pth.tar` | 217 MB | MotionCLIP. Frozen. It supplies the coarse style token. |
| `checkpoints/dual_style_denoiser.ckpt` | 1.03 GB | Dual-style denoiser, epoch 1999. This is the network that was trained. |

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

### How the networks were trained

Training has four frozen models and one trained denoiser. The released config is `configs/dual_style_hht.yaml`.

1. **Motion VAE.** An encoder–decoder transformer on raw HumanML-263 features: 9 layers, 4 heads, feed-forward size 1024, latent shape 7 × 256. It was trained on FineMotion at 20 Hz and stopped at epoch 599. Diffusion training does not update it.

2. **IMF extractor.** A frozen teacher that splits a motion into three intrinsic mode functions at 20 Hz. Bands 0 and 1 are the fine style. Band 2 is the coarse band. The trajectory branch reads degrees of freedom 66–69. The diffusion model was trained against this exact checkpoint. A later extractor is not a substitute.

3. **Contact timing and MotionCLIP.** Both stay frozen. Contact timing is a content-side condition. MotionCLIP, with CLIP ViT-B/32, embeds the style motion into the coarse style token. The denoiser still has its own contact-timing encoder, and those weights are inside `dual_style_denoiser.ckpt`: hidden size 128, output size 512, 2 blocks, kernel size 5.

4. **Dual-style denoiser.** A transformer encoder, also 9 layers, 4 heads, and feed-forward size 1024, with dropout 0.1 and GELU. The fine-style condition is injected from layer 6. It was trained from scratch with AdamW at `1e-4` for 2000 epochs, and the released file is epoch 1999. The noise schedule is DDPM with 1000 training steps, `scaled_linear` betas from `0.00085` to `0.012`. Each condition is dropped with probability 0.25. The diffusion losses recorded in the config are:

   - reconstruction, generation, and cross-reconstruction: `1.0`
   - IMF reconstruction: `0.1`
   - HHT amplitude and HHT frequency: `0.02` each
   - global frequency and branch frequency: `0.01` and `0.1`
   - KL: `1e-4`
   - latent: `1e-5`

The training motions are FineMotion clips stored as HumanML-263 features, normalized with `data/stats/Mean.npy` and `data/stats/Std.npy`. Sampling is separate from those loss weights. Inference uses DDIM with 50 steps. The paper protocol is global scale **2.5** and fine scale **1.5**.

More notes are in `docs/TRAINING.md`.

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

[Open the notebook in Colab](https://colab.research.google.com/github/dongran/kinematic-frequency-motion/blob/main/notebooks/dual_style_transfer_demo.ipynb). The first cell clones this repository and downloads the five weight files. In the example cell, `FOOT_FIX = False` shows the raw joints.

### Your own motions

Use a raw HumanML3D feature, shape `[T, 263]`, 20 Hz. Do not normalize it yourself.

SMPL motion (`poses` and `trans` in an `.npz`) is converted by the [HumanML3D](https://github.com/EricGuo5513/HumanML3D) notebooks `raw_pose_processing.ipynb` and then `motion_representation.ipynb`. Keep the `new_joint_vecs` array.

A motion-capture BVH file is not read here. Retarget it to SMPL with [tempo-changing-music2motion](https://github.com/dongran/tempo-changing-music2motion), then run the two HumanML3D notebooks.

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
