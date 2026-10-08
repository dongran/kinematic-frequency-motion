#!/usr/bin/env bash
# Step 1. Motion VAE on FineMotion HumanML-263.
# AdamW, lr 1e-4, batch 128, latent 7 x 256. The config runs for 1000 epochs.
# The released checkpoint is epoch 599.
set -euo pipefail
cd "$(dirname "$0")/.."
python train.py \
  --cfg configs/motion_vae_finemotion.yaml \
  --cfg_assets configs/assets_finemotion.yaml \
  --device 0 \
  --nodebug
