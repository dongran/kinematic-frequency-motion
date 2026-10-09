#!/usr/bin/env bash
# Step 1. Motion VAE on FineMotion HumanML-263 at 20 Hz.
# Settings are in configs/motion_vae_finemotion.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
python train.py \
  --cfg configs/motion_vae_finemotion.yaml \
  --cfg_assets configs/assets_finemotion.yaml \
  --device 0 \
  --nodebug
