#!/usr/bin/env bash
# Step 3. Content-side contact-timing predictor.
# Settings are in configs/contact_timing_finemotion.yaml.
# Labels are one npz per clip under DATA.LABEL_ROOT.
set -euo pipefail
cd "$(dirname "$0")/.."
python tools/train_contact_timing_predictor.py \
  --config configs/contact_timing_finemotion.yaml \
  --device cuda:0
