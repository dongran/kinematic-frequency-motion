#!/usr/bin/env bash
# Step 2. Train the three-band IMF extractor used with the dual-style denoiser.
# Settings are in configs/imf_pose69_teacher.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
python imf_extractor/train_imf.py \
  --config configs/imf_pose69_teacher.yaml \
  --device cuda:0 \
  --epochs 8 \
  --batch_size 128
