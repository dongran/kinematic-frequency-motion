#!/usr/bin/env bash
set -euo pipefail

# Paper protocol defaults: global style scale 2.5, fine-style scale 1.5.
# Override STYLE_SCALE / FINE_SCALE to match other settings shown on the project page.
# The same style motion is used for the coarse and fine branches in these examples.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"

EXAMPLE="${EXAMPLE:-turn_footwork}"
CONTENT_DIR="${CONTENT_DIR:-data/examples/${EXAMPLE}/content}"
STYLE_DIR="${STYLE_DIR:-data/examples/${EXAMPLE}/style}"

cmd=(
  python demo_dual_style.py
  --cfg configs/dual_style_hht.yaml
  --cfg_assets configs/assets.yaml
  --checkpoints checkpoints/dual_style_denoiser.ckpt
  --content_motion_dir "${CONTENT_DIR}"
  --coarse_style_motion_dir "${STYLE_DIR}"
  --fine_style_motion_dir "${STYLE_DIR}"
  --scale "${STYLE_SCALE:-2.5}"
  --fine_scale "${FINE_SCALE:-1.5}"
)
if [[ -n "${DEMO_SEED:-}" ]]; then
  cmd+=(--demo_seed "${DEMO_SEED}")
fi

"${cmd[@]}"
