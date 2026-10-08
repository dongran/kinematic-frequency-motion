$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not $env:STYLE_SCALE) { $env:STYLE_SCALE = "2.5" }
if (-not $env:FINE_SCALE) { $env:FINE_SCALE = "1.5" }
if (-not $env:EXAMPLE) { $env:EXAMPLE = "turn_footwork" }

python demo_dual_style.py `
  --cfg configs/dual_style_hht.yaml `
  --cfg_assets configs/assets.yaml `
  --checkpoints checkpoints/dual_style_denoiser.ckpt `
  --content_motion_dir "data/examples/$($env:EXAMPLE)/content" `
  --coarse_style_motion_dir "data/examples/$($env:EXAMPLE)/style" `
  --fine_style_motion_dir "data/examples/$($env:EXAMPLE)/style" `
  --scale $env:STYLE_SCALE `
  --fine_scale $env:FINE_SCALE
