#!/usr/bin/env bash
# Step 4. Dual-style denoiser, trained from scratch.
# The motion VAE, IMF extractor, contact-timing predictor, and MotionCLIP stay frozen.
# AdamW, lr 1e-4, batch 128, 2000 epochs. The released checkpoint is epoch 1999.
set -euo pipefail
cd "$(dirname "$0")/.."
python train.py \
  --cfg configs/dual_style_finemotion_scratch.yaml \
  --cfg_assets configs/assets_finemotion.yaml \
  --device 0 \
  --nodebug
