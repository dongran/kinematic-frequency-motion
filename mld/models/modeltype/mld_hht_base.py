from __future__ import annotations

import time

import numpy as np
import torch

from mld.models.architectures.imf_detector import (
    IMFDetectorAdapter,
    build_fixed_band_detector,
)
from mld.models.modeltype.mld import MLD
from mld.utils.temos_utils import lengths_to_mask, remove_padding


class MLD_HHT_BASE(MLD):
    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__(cfg=cfg, datamodule=datamodule, **kwargs)
        data_cfg = getattr(cfg, "DATA", None)
        decomposer_type = str(getattr(data_cfg, "DECOMPOSER_TYPE", "imf")).strip().lower()
        self.decomposer_type = decomposer_type
        self.decomposer_repr = str(
            getattr(
                data_cfg,
                "DECOMPOSER_REPR",
                getattr(data_cfg, "FREQ_REPR", "full"),
            )
        ).lower()
        self.decomposer_slice_start = getattr(data_cfg, "DECOMPOSER_SLICE_START", None)
        self.decomposer_slice_end = getattr(data_cfg, "DECOMPOSER_SLICE_END", None)
        self.decomposer_frame_rate = float(
            getattr(
                data_cfg,
                "DECOMPOSER_FRAME_RATE",
                getattr(data_cfg, "IMF_FRAME_RATE", 20.0),
            )
        )
        if decomposer_type == "imf":
            checkpoint_path = getattr(data_cfg, "IMF_CHECKPOINT", None)
            if not checkpoint_path:
                raise ValueError("DATA.IMF_CHECKPOINT must be set when DATA.DECOMPOSER_TYPE is 'imf'.")
            self.imf_detector = IMFDetectorAdapter(
                checkpoint_path=checkpoint_path,
                nfeats=cfg.DATASET.NFEATS,
                latent_dim=int(getattr(data_cfg, "IMF_LATENT_DIM", 512)),
                imf_count=int(getattr(data_cfg, "IMF_COUNT", 3)),
                frame_rate=float(getattr(data_cfg, "IMF_FRAME_RATE", 20.0)),
                freeze=bool(getattr(data_cfg, "FREEZE_IMF_EXTRACTOR", True)),
            )
        else:
            self.imf_detector = build_fixed_band_detector(
                decomposer_type=decomposer_type,
                nfeats=int(cfg.DATASET.NFEATS),
                frame_rate=self.decomposer_frame_rate,
                repr_name=self.decomposer_repr,
                slice_start=self.decomposer_slice_start,
                slice_end=self.decomposer_slice_end,
                fft_band_edges_hz=getattr(data_cfg, "FFT_BAND_EDGES_HZ", [1.0, 3.0, 8.0]),
                dwt_levels=int(getattr(data_cfg, "DWT_LEVELS", 2)),
            )
        detector = getattr(self.imf_detector, "detector", self.imf_detector)
        self.imf_dof = int(getattr(detector, "imfdof", 63))
        self.lambda_imf = float(getattr(cfg.LOSS, "LAMBDA_IMF", 0.0))
        self.lambda_hht_amp = float(getattr(cfg.LOSS, "LAMBDA_HHT_AMP", 0.0))
        self.lambda_hht_freq = float(getattr(cfg.LOSS, "LAMBDA_HHT_FREQ", 0.0))
        self.lambda_global = float(getattr(cfg.LOSS, "LAMBDA_FREQ_GLOBAL", 0.0))
        self.lambda_branch = float(getattr(cfg.LOSS, "LAMBDA_FREQ_BRANCH", 0.0))
        self.hht_target_mode = str(getattr(data_cfg, "HHT_TARGET_MODE", "offline")).lower()
        self.guidance_fine_scale = float(
            getattr(getattr(cfg, "model", None), "guidance_fine_scale", 1.0)
        )
        self.guidance_uncodp_fine = float(
            getattr(getattr(cfg, "model", None), "guidance_uncondp_fine", self.guidance_uncodp)
        )

    def _parse_dof_slice(self, value, *, total_dof=None):
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if ":" in text:
                start_text, end_text = text.split(":", 1)
                start = int(start_text)
                end = int(end_text)
            else:
                start = int(text)
                end = start + 1
        elif isinstance(value, int):
            start = int(value)
            end = start + 1
        else:
            values = list(value)
            if len(values) != 2:
                raise ValueError(f"Expected DOF slice [start, end], got {value}")
            start, end = int(values[0]), int(values[1])
        if total_dof is not None and (start < 0 or end > total_dof):
            raise ValueError(f"Invalid DOF slice {start}:{end}; valid range is [0, {total_dof}]")
        if end <= start:
            raise ValueError(f"Invalid DOF slice {start}:{end}")
        return (start, end)

    def _mask_imfs(self, imfs, *, band_indices=None, dof_slice=None, dof_indices=None):
        if band_indices is None and dof_slice is None and dof_indices is None:
            return imfs
        masked = imfs
        if band_indices is not None:
            band_mask = torch.zeros(imfs.shape[1], device=imfs.device, dtype=imfs.dtype)
            if len(band_indices) > 0:
                band_mask[torch.as_tensor(band_indices, device=imfs.device, dtype=torch.long)] = 1.0
            masked = masked * band_mask.view(1, -1, 1, 1)
        if dof_slice is not None or dof_indices is not None:
            dof_mask = torch.zeros(imfs.shape[2], device=imfs.device, dtype=imfs.dtype)
            if dof_slice is not None:
                dof_mask[dof_slice[0]:dof_slice[1]] = 1.0
            if dof_indices is not None and len(dof_indices) > 0:
                dof_mask[torch.as_tensor(dof_indices, device=imfs.device, dtype=torch.long)] = 1.0
            masked = masked * dof_mask.view(1, 1, -1, 1)
        return masked

    def _joint_ids_to_imf_dof_indices(self, joint_ids):
        indices = []
        for joint_id in joint_ids:
            base = int(joint_id) * 3
            if base + 2 >= self.imf_dof:
                raise ValueError(
                    f"Joint id {joint_id} exceeds available IMF rotational DOF ({self.imf_dof})."
                )
            indices.extend([base, base + 1, base + 2])
        return sorted(set(indices))

    def _joint_ids_to_motion_feature_indices(self, joint_ids):
        if int(self.cfg.DATASET.NFEATS) != 263:
            raise ValueError(
                "Selective content suppression currently assumes HumanML3D 263D features."
            )
        indices = []
        joint_ids = sorted({int(j) for j in joint_ids})
        ric_start = 4
        rot_start = ric_start + 21 * 3
        vel_start = rot_start + 21 * 6
        for joint_id in joint_ids:
            if joint_id > 0:
                ric_base = ric_start + (joint_id - 1) * 3
                rot_base = rot_start + (joint_id - 1) * 6
                indices.extend(range(ric_base, ric_base + 3))
                indices.extend(range(rot_base, rot_base + 6))
            vel_base = vel_start + joint_id * 3
            indices.extend(range(vel_base, vel_base + 3))
        return sorted(set(indices))

    def _scale_motion_feature_indices(self, motion, feature_indices, scale):
        if feature_indices is None or len(feature_indices) == 0 or float(scale) == 1.0:
            return motion
        scaled = motion.clone()
        index_tensor = torch.as_tensor(feature_indices, device=motion.device, dtype=torch.long)
        scaled[..., index_tensor] = scaled[..., index_tensor] * float(scale)
        return scaled

    def _normalize_external_motion(self, motion):
        mean = self.mean.to(motion.device)
        std = self.std.to(motion.device)
        return (motion - mean) / std

    def _select_decomposer_motion(self, motion: torch.Tensor) -> torch.Tensor:
        if self.decomposer_repr in {"full", "full263"}:
            return motion
        if self.decomposer_repr in {"pose63", "rot63"}:
            start = 3 if self.decomposer_slice_start is None else int(self.decomposer_slice_start)
            end = 66 if self.decomposer_slice_end is None else int(self.decomposer_slice_end)
            return motion[..., start:end]
        if self.decomposer_slice_start is None or self.decomposer_slice_end is None:
            raise ValueError(
                f"Custom decomposer repr '{self.decomposer_repr}' requires DECOMPOSER_SLICE_START and DECOMPOSER_SLICE_END."
            )
        start = int(self.decomposer_slice_start)
        end = int(self.decomposer_slice_end)
        if end <= start:
            raise ValueError(f"Invalid decomposer slice {start}:{end}")
        return motion[..., start:end]

    def _build_content_condition(self, motion_norm, lengths, duplicate_for_guidance=False):
        feats_content = motion_norm.clone()
        feats_content[..., :3] = 0.0
        with torch.no_grad():
            z_content, _ = self.vae.encode(feats_content.float(), lengths)
        if duplicate_for_guidance:
            return torch.cat([z_content, z_content], dim=1).permute(1, 0, 2)
        return z_content.permute(1, 0, 2)

    def _build_style_condition(
        self,
        motion_norm,
        lengths,
        *,
        duplicate_for_guidance=False,
        apply_dropout=False,
    ):
        motion_seq = motion_norm * self.std.to(motion_norm.device) + self.mean.to(motion_norm.device)
        motion_seq = motion_seq.clone()
        motion_seq[..., :3] = 0.0
        motion_seq = motion_seq.unsqueeze(-1).permute(0, 2, 3, 1)
        with torch.no_grad():
            motion_emb = self.motionclip.encoder(
                {
                    "x": motion_seq.float(),
                    "y": torch.zeros(motion_seq.shape[0], dtype=int, device=motion_seq.device),
                    "mask": lengths_to_mask(
                        lengths,
                        device=motion_seq.device,
                        max_len=motion_seq.shape[-1],
                    ),
                }
            )["mu"].unsqueeze(1)
        if apply_dropout:
            mask_uncond = torch.rand(motion_emb.shape[0], device=motion_emb.device) < self.guidance_uncodp
            motion_emb = motion_emb.clone()
            motion_emb[mask_uncond, ...] = 0
        if duplicate_for_guidance:
            motion_emb = torch.cat([torch.zeros_like(motion_emb), motion_emb], dim=0)
        return motion_emb

    def _build_trans_condition(self, motion_norm, duplicate_for_guidance=False):
        trans_cond = motion_norm[..., :3]
        if duplicate_for_guidance:
            trans_cond = torch.cat([trans_cond, trans_cond], dim=0)
        return trans_cond

    def _extract_online_imfs(self, motion_norm, lengths):
        with torch.no_grad():
            return self.imf_detector(motion_norm, lengths)

    def _decode_to_joints(self, latents, lengths):
        with torch.no_grad():
            feats_rst = self.vae.decode(latents, lengths)
        joints = self.feats2joints(feats_rst.detach().cpu())
        return feats_rst, remove_padding(joints, lengths)

    def _postprocess_t2m_eval(
        self,
        batch,
        motions,
        feats_rst,
        lengths,
        start_time,
        *,
        word_embs=None,
        pos_ohot=None,
        text_lengths=None,
    ):
        end = time.time()
        self.times.append(end - start_time)

        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(motions)

        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)

        m_lens = torch.as_tensor(lengths, dtype=torch.long)
        align_idx = torch.argsort(m_lens, descending=True)
        align_idx_device = align_idx.to(motions.device)
        motions = motions.index_select(0, align_idx_device)
        feats_rst = feats_rst.index_select(0, align_idx_device)
        m_lens = m_lens.index_select(0, align_idx)
        m_lens = torch.div(
            m_lens,
            self.cfg.DATASET.HUMANML3D.UNIT_LEN,
            rounding_mode="floor",
        )

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        if word_embs is None:
            word_embs = batch["word_embs"].detach().clone()
        if pos_ohot is None:
            pos_ohot = batch["pos_ohot"].detach().clone()
        if text_lengths is None:
            text_lengths = batch["text_len"].detach().clone()
        text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[align_idx]

        return {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
        }

    def _encode_core_conditions(self, feats_ref, lengths):
        feats_content = feats_ref.clone()
        feats_content[..., :3] = 0.0
        with torch.no_grad():
            z, _ = self.vae.encode(feats_ref, lengths)
            z_content, _ = self.vae.encode(feats_content, lengths)
            cond_emb = z_content.permute(1, 0, 2)
        motion_seq = feats_ref * self.std.to(feats_ref.device) + self.mean.to(feats_ref.device)
        motion_seq[..., :3] = 0.0
        motion_seq = motion_seq.unsqueeze(-1).permute(0, 2, 3, 1)
        mask = lengths_to_mask(lengths, device=motion_seq.device)
        with torch.no_grad():
            motion_emb = self.motionclip.encoder(
                {
                    "x": motion_seq.float(),
                    "y": torch.zeros(motion_seq.shape[0], dtype=int, device=motion_seq.device),
                    "mask": mask,
                }
            )["mu"].unsqueeze(1)
        mask_uncond = torch.rand(motion_emb.shape[0], device=motion_emb.device) < self.guidance_uncodp
        motion_emb = motion_emb.clone()
        motion_emb[mask_uncond, ...] = 0
        trans_cond = feats_ref[..., :3]
        return z, cond_emb, motion_emb, trans_cond

    def _diffusion_process_with_state(
        self,
        latents,
        encoder_hidden_states,
        lengths=None,
        denoiser_kwargs=None,
    ):
        latents = latents.permute(1, 0, 2)
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise, timesteps)
        denoiser_kwargs = denoiser_kwargs or {}
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            lengths=lengths,
            return_dict=False,
            **denoiser_kwargs,
        )[0]
        return {
            "noise": noise,
            "noise_prior": 0,
            "noise_pred": noise_pred,
            "noise_pred_prior": 0,
            "noisy_latents": noisy_latents,
            "timesteps": timesteps,
        }

    def _decode_prediction(self, rs_set, lengths):
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(rs_set["noise_pred"].device)
        t = rs_set["timesteps"]
        a_bar = alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_ab = torch.sqrt(1 - a_bar)
        sqrt_ab = torch.sqrt(a_bar)
        x0 = (rs_set["noisy_latents"] - sqrt_one_minus_ab * rs_set["noise_pred"]) / (sqrt_ab + 1e-8)
        return self.vae.decode(x0.permute(1, 0, 2), lengths)

    def _align_imf_pair(self, pred_imfs: torch.Tensor, target_imfs: torch.Tensor):
        if pred_imfs.shape[-1] == target_imfs.shape[-1]:
            return pred_imfs, target_imfs
        shared_t = min(pred_imfs.shape[-1], target_imfs.shape[-1])
        return pred_imfs[..., :shared_t], target_imfs[..., :shared_t]

    def _resolve_target_imfs(self, batch, lengths):
        target_imfs = None
        if self.hht_target_mode == "offline" and "imfs" in batch:
            target_imfs = batch["imfs"]
        with torch.no_grad():
            online_target_imfs, target_global, _ = self.imf_detector(batch["motion"], lengths)
        if target_imfs is None:
            target_imfs = online_target_imfs
        return target_imfs, target_global

    def _build_aux_outputs(self, batch, motion_pred, *, target_imfs=None, target_global=None):
        lengths = batch["length"]
        pred_imfs, pred_global, _ = self.imf_detector(motion_pred, lengths)
        aux = {
            "motion_pred": motion_pred,
            "pred_imfs": pred_imfs,
            "pred_global": pred_global,
        }
        if target_imfs is None or target_global is None:
            target_imfs, target_global = self._resolve_target_imfs(batch, lengths)
        pred_imfs, target_imfs = self._align_imf_pair(pred_imfs, target_imfs)
        aux["pred_imfs"] = pred_imfs
        aux["target_imfs"] = target_imfs
        aux["target_global"] = target_global
        return aux

    def _compute_aux_losses(self, rs_set, batch):
        return torch.tensor(0.0, device=rs_set["noise_pred"].device), {}

    def train_diffusion_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]
        z, cond_emb, motion_emb, trans_cond = self._encode_core_conditions(feats_ref, lengths)
        target_imfs, target_global = self._resolve_target_imfs(batch, lengths)
        encoder_hidden_states = self._compose_conditions(batch, cond_emb, motion_emb, trans_cond)
        rs_set = self._diffusion_process_with_state(z, encoder_hidden_states, lengths)
        motion_pred = self._decode_prediction(rs_set, lengths)
        rs_set.update(
            self._build_aux_outputs(
                batch,
                motion_pred,
                target_imfs=target_imfs,
                target_global=target_global,
            )
        )
        return rs_set

    def _compose_conditions(self, batch, cond_emb, motion_emb, trans_cond):
        return [cond_emb, motion_emb, trans_cond]

    def allsplit_step(self, split: str, batch, batch_idx):
        if split in ["train", "val"]:
            if self.stage == "vae":
                rs_set = self.train_vae_forward(batch)
                rs_set["lat_t"] = rs_set["lat_m"]
            elif self.stage == "diffusion":
                rs_set = self.train_diffusion_forward(batch)
            elif self.stage == "vae_diffusion":
                raise ValueError("HHT variants currently only support diffusion stage.")
            else:
                raise ValueError(f"Not support this stage {self.stage}!")

            base_loss = self.losses[split].update(rs_set)
            aux_loss, aux_logs = self._compute_aux_losses(rs_set, batch)
            loss = base_loss + aux_loss
            if not self.trainer.sanity_checking:
                prefix = "train" if split == "train" else "val"
                self.log(f"hht/aux_total/{prefix}", aux_loss.detach(), sync_dist=True, rank_zero_only=True)
                for name, value in aux_logs.items():
                    self.log(f"hht/{name}/{prefix}", value.detach(), sync_dist=True, rank_zero_only=True)
        if split in ["val", "test"]:
            rs_set = self.t2m_eval(batch)
            if self.trainer.datamodule.is_mm:
                metrics_dicts = ["MMMetrics"]
            else:
                metrics_dicts = self.metrics_dict
            for metric in metrics_dicts:
                if metric == "TemosMetric":
                    phase = split if split != "val" else "eval"
                    if eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower() not in ["humanml3d", "kit"]:
                        raise TypeError("APE and AVE metrics only support humanml3d and kit datasets now")
                    getattr(self, metric).update(rs_set["joints_rst"], rs_set["joints_ref"], batch["length"])
                elif metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        rs_set["lat_t"], rs_set["lat_rm"], rs_set["lat_m"], batch["length"]
                    )
                elif metric == "UncondMetrics":
                    getattr(self, metric).update(
                        recmotion_embeddings=rs_set["lat_rm"],
                        gtmotion_embeddings=rs_set["lat_m"],
                        lengths=batch["length"],
                    )
                elif metric == "MRMetrics":
                    getattr(self, metric).update(rs_set["joints_rst"], rs_set["joints_ref"], batch["length"])
                elif metric == "MMMetrics":
                    getattr(self, metric).update(rs_set["lat_rm"].unsqueeze(0), batch["length"])
                else:
                    raise TypeError(f"Not support this metric {metric}")
        if split in ["test"]:
            return rs_set["joints_rst"], batch["length"]
        return loss
