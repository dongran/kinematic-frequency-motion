#!/usr/bin/env bash
# Step 3. Content-side contact-timing predictor.
# 40 epochs, batch 64, AdamW lr 1e-4. The released file is checkpoints/best.pt
# from this run. Labels are one npz per FineMotion clip under DATA.LABEL_ROOT.
set -euo pipefail
cd "$(dirname "$0")/.."
python tools/train_contact_timing_predictor.py \
  --config configs/contact_timing_finemotion.yaml \
  --device cuda:0
