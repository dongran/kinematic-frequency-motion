# Training notes

This repository releases the inference stack and the five weight files used together. It does not launch training by itself. The README lists each file, its size, and the download URL.

The released denoiser is a from-scratch FineMotion run, stopped at epoch 1999. It was trained with the other three checkpoints frozen:

1. **Motion VAE.** FineMotion HumanML-263 features, latent shape 7 x 256, epoch 599. File: `checkpoints/motion_vae.ckpt`.
2. **IMF extractor.** Three-band pose-69 teacher, epoch 6. File: `checkpoints/imf_extractor.pt`. The diffusion model was trained against this extractor. A later extractor, even one trained after a Hilbert-unwrap correction, is not a drop-in replacement.
3. **Contact-timing predictor.** Frozen content-side contact timing. File: `checkpoints/contact_timing.pt`. The denoiser still learns its own contact-timing encoder; those weights are inside `checkpoints/dual_style_denoiser.ckpt`.
4. **Dual-style denoiser.** Coarse style, fine IMF style, trajectory IMF, and contact timing. File: `checkpoints/dual_style_denoiser.ckpt`, epoch 1999. The coarse style token is a frozen MotionCLIP embedding (`checkpoints/motionclip_checkpoint/motionclip.pth.tar`).

`configs/dual_style_hht.yaml` is the inference configuration for that denoiser. The loss weights in that file are the ones used during diffusion training:

- IMF reconstruction weight `0.1`
- HHT amplitude and frequency weights `0.02` each
- global and branch frequency weights `0.01` and `0.1`

Guidance at sampling time is separate from those loss weights. The paper protocol is `--scale 2.5 --fine_scale 1.5`.

Training data are FineMotion motions stored as HumanML-263 features at 20 Hz, normalized with `data/stats/Mean.npy` and `data/stats/Std.npy`. Reproducing training also needs the original MCM-LDM training entry points and the FineMotion split used for this model. Those launchers are not part of this inference release.
