#!/usr/bin/env bash
# Render an SMPL npz (keys: poses, trans) to an mp4 with Blender.
# SMPL_NEUTRAL.pkl is not included. Set SMPL_PATH to the directory that contains it.
# Set BLENDER_BIN to the Blender executable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"

NPZ="${1:?Pass an npz with poses and trans}"
OUT="${2:-outputs/mesh.mp4}"

if [[ -z "${BLENDER_BIN:-}" ]]; then
  echo "Set BLENDER_BIN to the Blender executable." >&2
  exit 2
fi
if [[ -z "${SMPL_PATH:-}" ]]; then
  echo "Set SMPL_PATH to the directory that contains SMPL_NEUTRAL.pkl." >&2
  exit 2
fi

python tools/render_smpl_npz_blender.py \
  --npz_path "${NPZ}" \
  --out_mp4 "${OUT}" \
  --smpl_path "${SMPL_PATH}" \
  --blender_bin "${BLENDER_BIN}"
