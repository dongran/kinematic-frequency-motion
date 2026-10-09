# Training

The README part "Train on your own data" is this procedure. Prepare HumanML-263 features and the three aligned frequency bands with `scripts/prepare_frequency_bands.py`, then run the four scripts below. FineMotion will be released separately; the same steps apply to another SMPL dataset. The released weights are the demonstration set.

The four training scripts are in the README. Layer sizes and loss weights are in each step's YAML file. Run the scripts in this order:

1. `bash scripts/train_motion_vae.sh`
2. `bash scripts/train_imf_extractor.sh` — train this before the denoiser. A new motion set needs its own IMF extractor.
3. `bash scripts/train_contact_timing.sh`
4. `bash scripts/train_dual_style.sh`

The coarse-style token uses the trained MotionCLIP checkpoint from [MCM-LDM](https://github.com/XingliangJin/MCM-LDM). Retrain it for a custom token on your own data.

FineMotion HumanML-263 clips are not included. `configs/assets_finemotion.yaml` is where their path is set.
