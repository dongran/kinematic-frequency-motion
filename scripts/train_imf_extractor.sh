#!/usr/bin/env bash
# Step 2. Body-63 recovery finetune of the pose-69 IMF extractor.
# Only the body-63 decoders and refiners train, for 8 epochs at 1e-6, batch 128.
# The released teacher is epoch 6 of this run.
# Put the pose-69 checkpoint it continues from at checkpoints/imf_pose69_init.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
python imf_extractor/train_imf.py \
  --config configs/imf_pose69_teacher.yaml \
  --device cuda:0 \
  --epochs 8 \
  --batch_size 128
