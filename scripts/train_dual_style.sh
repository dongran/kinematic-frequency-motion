#!/usr/bin/env bash
# Step 4. Train the dual-style denoiser with the IMF extractor for this motion set.
# The motion VAE, IMF extractor, contact-timing predictor, and MotionCLIP stay frozen.
# Settings are in configs/dual_style_finemotion_scratch.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
python train.py \
  --cfg configs/dual_style_finemotion_scratch.yaml \
  --cfg_assets configs/assets_finemotion.yaml \
  --device 0 \
  --nodebug
