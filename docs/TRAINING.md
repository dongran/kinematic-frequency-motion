# Training

The four training scripts and the settings that produced the released checkpoints are in the README. Run them in this order:

1. `bash scripts/train_motion_vae.sh`
2. `bash scripts/train_imf_extractor.sh`
3. `bash scripts/train_contact_timing.sh`
4. `bash scripts/train_dual_style.sh`

FineMotion HumanML-263 clips are not included. `configs/assets_finemotion.yaml` is where their path is set.
