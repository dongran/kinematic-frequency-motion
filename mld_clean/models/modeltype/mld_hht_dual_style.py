from __future__ import annotations

import time
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mld_clean.models.architectures.contact_timing_predictor import (
    ContactTimingConditionEncoder,
    ContactTimingPredictor,
)
from mld_clean.models.architectures.contact_timing_runtime import (
    FEATURE_DIMS,
    build_contact_timing_feature_tensor,
    build_root_trajectory_tensor,
    normalize_root_trajectory_tensor,
    wrap_angle,
)
from mld_clean.models.architectures.frequency_branch import FrequencyBranch
from mld_clean.models.architectures.timing_decoupling import (
    TimingBranchFusion,
    TimingPhaseConditionEncoder,
    extract_phase_features,
)
from mld_clean.models.losses.hht_aux_losses import (
    cosine_global_loss,
    hht_feature_loss,
    masked_l1,
    masked_l1_dof,
    masked_l1_selective,
)
from mld_clean.models.losses.spectral_aux_losses import (
    fft_spectral_losses,
    wavelet_spectral_losses,
)
from mld_clean.models.losses.timing_aux_losses import (
    masked_bce_with_probs,
    masked_mse as masked_timing_mse,
)
from mld_clean.models.modeltype.mld_hht_base import MLD_HHT_BASE
from mld_clean.utils.temos_utils import lengths_to_mask, remove_padding


def _torch_load_compat(path: str | Path, *, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class MLD_HHT_DUAL_STYLE(MLD_HHT_BASE):
    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__(cfg=cfg, datamodule=datamodule, **kwargs)
        detector = getattr(self.imf_detector, "detector", self.imf_detector)
        imf_count = int(getattr(getattr(cfg, "DATA", None), "IMF_COUNT", 3))
        imf_dof = int(getattr(detector, "imfdof", 63))
        self.imf_dof = imf_dof
        data_cfg = getattr(cfg, "DATA", None)
        model_cfg = getattr(cfg, "model", None)
        self.frequency_branch = FrequencyBranch(
            motion_dim=cfg.DATASET.NFEATS,
            imf_dim=imf_count * imf_dof,
            hidden_dim=512,
        )
        self.total_imf_bands = imf_count
        self.fine_band_indices = self._parse_band_indices(
            getattr(getattr(cfg, "DATA", None), "IMF_FINE_BANDS", [0, 1]),
            total_bands=imf_count,
        )
        self.coarse_band_indices = self._parse_band_indices(
            getattr(getattr(cfg, "DATA", None), "IMF_COARSE_BANDS", []),
            total_bands=imf_count,
        )
        self.traj_band_indices = self._parse_band_indices(
            getattr(model_cfg, "traj_imf_bands", []),
            total_bands=imf_count,
        )
        self.trans_dof_slice = self._parse_dof_slice(
            getattr(data_cfg, "IMF_TRANS_DOF_SLICE", None),
            total_dof=imf_dof,
        )
        self.lambda_imf_trans = float(getattr(cfg.LOSS, "LAMBDA_IMF_TRANS", 0.0))
        self.traj_imf_scale = float(getattr(model_cfg, "traj_imf_scale", 1.0))
        self.guidance_traj_scale = float(getattr(model_cfg, "guidance_traj_scale", self.traj_imf_scale))
        self.guidance_uncodp_traj = float(
            getattr(model_cfg, "guidance_uncondp_traj", self.guidance_uncodp_fine)
        )
        self.guidance_zero_uncond_traj = bool(
            getattr(model_cfg, "guidance_zero_uncond_traj", True)
        )
        self.content_traj_band_indices = self._parse_band_indices(
            getattr(model_cfg, "content_traj_imf_bands", []),
            total_bands=imf_count,
        )
        self.content_traj_imf_scale = float(getattr(model_cfg, "content_traj_imf_scale", 1.0))
        self.guidance_content_traj_scale = float(
            getattr(model_cfg, "guidance_content_traj_scale", self.content_traj_imf_scale)
        )
        self.content_imf_remover_band_indices = self._parse_band_indices(
            getattr(model_cfg, "content_imf_remover_bands", []),
            total_bands=imf_count,
        )
        self.content_imf_remover_scale = float(
            getattr(model_cfg, "content_imf_remover_scale", 1.0)
        )
        self.guidance_content_imf_remover_scale = float(
            getattr(
                model_cfg,
                "guidance_content_imf_remover_scale",
                self.content_imf_remover_scale,
            )
        )
        self.style_source_mix_enabled = bool(
            getattr(model_cfg, "style_source_mix_enabled", False)
        )
        self.style_source_mix_prob = float(
            getattr(model_cfg, "style_source_mix_prob", 0.0)
        )
        self.style_source_mix_coarse_prob = float(
            getattr(model_cfg, "style_source_mix_coarse_prob", self.style_source_mix_prob)
        )
        self.style_source_mix_fine_prob = float(
            getattr(model_cfg, "style_source_mix_fine_prob", self.style_source_mix_prob)
        )
        self.style_source_mix_traj_prob = float(
            getattr(model_cfg, "style_source_mix_traj_prob", self.style_source_mix_fine_prob)
        )
        self.train_coarse_branch_dropout = float(
            getattr(model_cfg, "train_coarse_branch_dropout", 0.0)
        )
        self.train_fine_branch_dropout = float(
            getattr(model_cfg, "train_fine_branch_dropout", 0.0)
        )
        self.train_traj_branch_dropout = float(
            getattr(model_cfg, "train_traj_branch_dropout", 0.0)
        )
        if not self.fine_band_indices:
            raise ValueError("Dual-style models require at least one IMF fine band.")
        if (
            self.lambda_imf_trans > 0.0
            or self.traj_band_indices
            or self.content_traj_band_indices
        ) and self.trans_dof_slice is None:
            raise ValueError(
                "DATA.IMF_TRANS_DOF_SLICE must be set when using trans IMF split loss or trajectory IMF routing."
            )
        self.coarse_imf_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        self.selective_preset = str(getattr(model_cfg, "selective_preset", "")).strip().lower()
        self.selective_enabled = self.selective_preset not in {"", "none"}
        self.selective_content_preserve = float(
            getattr(model_cfg, "selective_content_preserve", 1.0)
        )
        self.guidance_content_preserve = float(
            getattr(model_cfg, "guidance_content_preserve", self.selective_content_preserve)
        )
        self.selective_branch_motion_mask = bool(
            getattr(model_cfg, "selective_branch_motion_mask", True)
        )
        self.lambda_imf_region_target = float(
            getattr(cfg.LOSS, "LAMBDA_IMF_REGION_TARGET", 0.0)
        )
        self.lambda_imf_region_keep = float(
            getattr(cfg.LOSS, "LAMBDA_IMF_REGION_KEEP", 0.0)
        )
        self.native_loss_type = str(getattr(cfg.LOSS, "NATIVE_LOSS_TYPE", "none")).strip().lower()
        self.native_fft_band_edges_hz = list(
            getattr(cfg.LOSS, "NATIVE_FFT_BAND_EDGES_HZ", [1.0, 3.0, 8.0])
        )
        self.native_fft_log_magnitude = bool(
            getattr(cfg.LOSS, "NATIVE_FFT_LOG_MAGNITUDE", True)
        )
        self.native_fft_weights = {
            "all": float(getattr(cfg.LOSS, "LAMBDA_NATIVE_FFT_ALL", 0.0)),
            "low": float(getattr(cfg.LOSS, "LAMBDA_NATIVE_FFT_LOW", 0.0)),
            "mid": float(getattr(cfg.LOSS, "LAMBDA_NATIVE_FFT_MID", 0.0)),
            "high": float(getattr(cfg.LOSS, "LAMBDA_NATIVE_FFT_HIGH", 0.0)),
        }
        self.native_wavelet_levels = max(
            int(getattr(cfg.LOSS, "NATIVE_WAVELET_LEVELS", 2)),
            1,
        )
        self.native_wavelet_log_magnitude = bool(
            getattr(cfg.LOSS, "NATIVE_WAVELET_LOG_MAGNITUDE", True)
        )
        self.native_wavelet_weights = {
            "all": float(getattr(cfg.LOSS, "LAMBDA_NATIVE_WAVELET_ALL", 0.0)),
            f"approx_l{self.native_wavelet_levels}": float(
                getattr(
                    cfg.LOSS,
                    f"LAMBDA_NATIVE_WAVELET_APPROX_L{self.native_wavelet_levels}",
                    0.0,
                )
            ),
        }
        self.native_wavelet_weights.update(
            {
                f"detail_l{level}": float(
                    getattr(cfg.LOSS, f"LAMBDA_NATIVE_WAVELET_DETAIL_L{level}", 0.0)
                )
                for level in range(1, self.native_wavelet_levels + 1)
            }
        )
        self.lambda_coarse_fine_decorrelation = float(
            getattr(cfg.LOSS, "LAMBDA_COARSE_FINE_DECORR", 0.0)
        )
        self.selective_target_band_indices = list(self.fine_band_indices)
        self.selective_target_dof_indices = None
        self.selective_keep_dof_indices = None
        self.selective_target_motion_feature_indices = None
        if self.selective_enabled:
            if imf_dof < 66:
                raise ValueError(
                    "Selective preset v1 currently requires a pose69 IMF extractor with at least 66 rotational DOF."
                )
            preset_cfg = self._resolve_selective_preset(self.selective_preset)
            self.selective_target_band_indices = preset_cfg["target_bands"]
            self.selective_target_dof_indices = self._joint_ids_to_imf_dof_indices(
                preset_cfg["target_joints"]
            )
            self.selective_keep_dof_indices = self._joint_ids_to_imf_dof_indices(
                preset_cfg["keep_joints"]
            )
            self.selective_target_motion_feature_indices = self._joint_ids_to_motion_feature_indices(
                preset_cfg["target_joints"]
            )
        self.contact_timing_enabled = False
        self.contact_timing_ckpt = ""
        self.contact_timing_source = "content"
        self.contact_timing_scale = 1.0
        self.guidance_contact_timing_scale = 1.0
        self.guidance_uncondp_contact_timing = self.guidance_uncodp
        self.guidance_zero_uncond_contact_timing = True
        self.contact_timing_feature_names: tuple[str, ...] = ()
        self.contact_timing_predictor = None
        self.contact_timing_encoder = None
        self.contact_timing_phase_encoder = None
        self.contact_timing_branch_fusion = None
        self.contact_timing_route = "trajectory"
        self.phase_timing_enabled = False
        self.contact_timing_phase_num_harmonics = 1
        self.lambda_contact_timing = float(
            getattr(getattr(cfg, "LOSS", None), "LAMBDA_CONTACT_TIMING", 0.0)
        )
        self.lambda_contact_timing_bce = float(
            getattr(getattr(cfg, "LOSS", None), "LAMBDA_CONTACT_TIMING_BCE", 0.0)
        )
        self.lambda_contact_timing_traj = float(
            getattr(getattr(cfg, "LOSS", None), "LAMBDA_CONTACT_TIMING_TRAJ", 0.0)
        )
        self.lambda_contact_timing_phase = float(
            getattr(getattr(cfg, "LOSS", None), "LAMBDA_CONTACT_TIMING_PHASE", 0.0)
        )
        self.lambda_root_traj_xz = float(
            getattr(getattr(cfg, "LOSS", None), "LAMBDA_ROOT_TRAJ_XZ", 0.0)
        )
        self.lambda_root_traj_yaw = float(
            getattr(getattr(cfg, "LOSS", None), "LAMBDA_ROOT_TRAJ_YAW", 0.0)
        )
        self.root_traj_align_xz = bool(getattr(model_cfg, "root_traj_align_xz", True))
        self.root_traj_align_yaw = bool(getattr(model_cfg, "root_traj_align_yaw", True))
        self._init_contact_timing_modules(model_cfg)

    def _parse_band_indices(self, value, *, total_bands):
        if value is None:
            return []
        if isinstance(value, str):
            if not value.strip():
                return []
            if value.strip().lower() == "all":
                values = list(range(total_bands))
            else:
                values = [v for v in value.split(",") if v.strip()]
        elif isinstance(value, int):
            values = [value]
        else:
            values = list(value)
        indices = sorted({int(v) for v in values})
        for idx in indices:
            if idx < 0 or idx >= total_bands:
                raise ValueError(f"Invalid IMF band index {idx}; valid range is [0, {total_bands - 1}]")
        return indices

    def _resolve_selective_preset(self, preset_name):
        lower = [0, 1, 2, 4, 5, 7, 8, 10, 11]
        torso = [3, 6, 9, 12]
        arms_head = [13, 14, 15, 16, 17, 18, 19, 20, 21]
        presets = {
            "upper_expressive": {
                "target_joints": torso + arms_head,
                "keep_joints": lower,
                "target_bands": [0, 1],
            },
            "preset_upper_expressive": {
                "target_joints": torso + arms_head,
                "keep_joints": lower,
                "target_bands": [0, 1],
            },
        }
        if preset_name not in presets:
            raise ValueError(f"Unsupported selective preset: {preset_name}")
        return presets[preset_name]

    def _as_scalar(self, value, default):
        if value is None:
            return float(default)
        if torch.is_tensor(value):
            return float(value.reshape(-1)[0].item())
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return float(default)
            return float(value[0])
        return float(value)

    def _init_contact_timing_modules(self, model_cfg):
        ckpt_path = str(getattr(model_cfg, "contact_timing_ckpt", "")).strip()
        if ckpt_path.lower() in {"", "none"}:
            return

        ckpt_path = str(Path(ckpt_path).expanduser().resolve())
        allow_missing_ckpt_fallback = os.environ.get(
            "MCM_LDM_ALLOW_CONTACT_TIMING_CKPT_FALLBACK",
            "0",
        ) == "1"
        predictor_state_dict = None
        if not Path(ckpt_path).is_file() and not allow_missing_ckpt_fallback:
            raise FileNotFoundError(f"contact timing checkpoint not found: {ckpt_path}")

        if Path(ckpt_path).is_file():
            ckpt = _torch_load_compat(ckpt_path, map_location="cpu")
            predictor_cfg = ckpt.get("config", {})
            predictor_state_dict = ckpt.get("model", ckpt.get("state_dict"))
        else:
            # The main diffusion checkpoint can contain the frozen predictor weights.
            # In that case we only need to instantiate the expected architecture here;
            # the subsequent strict load_state_dict call fills in the actual weights.
            predictor_cfg = {
                "DATA": {"INPUT_FEATURES": ["hip"]},
                "MODEL": {
                    "HIDDEN_DIM": 64,
                    "OUTPUT_DIM": 2,
                    "NUM_BLOCKS": 4,
                    "KERNEL_SIZE": 5,
                    "DROPOUT": 0.1,
                },
            }
        data_cfg = predictor_cfg.get("DATA", {})
        predictor_model_cfg = predictor_cfg.get("MODEL", {})
        feature_names = tuple(str(x) for x in data_cfg.get("INPUT_FEATURES", ["hip"]))
        if not feature_names:
            raise ValueError("contact timing predictor must define at least one input feature")
        unsupported = [name for name in feature_names if name not in FEATURE_DIMS]
        if unsupported:
            raise ValueError(
                f"unsupported contact timing predictor input features: {unsupported}; "
                f"supported={sorted(FEATURE_DIMS)}"
            )
        input_dim = sum(FEATURE_DIMS[name] for name in feature_names)
        output_dim = int(predictor_model_cfg.get("OUTPUT_DIM", 2))
        predictor = ContactTimingPredictor(
            input_dim=input_dim,
            hidden_dim=int(predictor_model_cfg.get("HIDDEN_DIM", 64)),
            output_dim=output_dim,
            num_blocks=int(predictor_model_cfg.get("NUM_BLOCKS", 4)),
            kernel_size=int(predictor_model_cfg.get("KERNEL_SIZE", 5)),
            dropout=float(predictor_model_cfg.get("DROPOUT", 0.0)),
        )
        if predictor_state_dict is None and not allow_missing_ckpt_fallback:
            raise KeyError("contact timing checkpoint must contain 'model' or 'state_dict'")
        if predictor_state_dict is not None:
            predictor.load_state_dict(predictor_state_dict, strict=True)
        predictor.eval()
        for parameter in predictor.parameters():
            parameter.requires_grad = False

        expected_dim = int(self.denoiser.emb_proj_contact_timing[1].in_features)
        encoder_output_dim = int(
            getattr(model_cfg, "contact_timing_encoder_output_dim", expected_dim)
        )
        if encoder_output_dim != expected_dim:
            raise ValueError(
                f"contact timing encoder output dim must match denoiser input dim "
                f"({expected_dim}), got {encoder_output_dim}"
            )
        self.contact_timing_encoder = ContactTimingConditionEncoder(
            input_dim=output_dim,
            hidden_dim=int(getattr(model_cfg, "contact_timing_encoder_hidden_dim", 128)),
            output_dim=encoder_output_dim,
            num_blocks=int(getattr(model_cfg, "contact_timing_encoder_num_blocks", 2)),
            kernel_size=int(getattr(model_cfg, "contact_timing_encoder_kernel_size", 5)),
            dropout=float(getattr(model_cfg, "contact_timing_encoder_dropout", 0.1)),
        )
        self.contact_timing_route = str(
            getattr(model_cfg, "contact_timing_route", "trajectory")
        ).strip().lower()
        if self.contact_timing_route not in {"trajectory", "timing", "both"}:
            raise ValueError(
                "contact_timing_route must be one of: trajectory, timing, both; "
                f"got {self.contact_timing_route}"
            )
        self.phase_timing_enabled = bool(getattr(model_cfg, "phase_timing_enabled", False))
        self.contact_timing_phase_num_harmonics = max(
            int(getattr(model_cfg, "contact_timing_phase_num_harmonics", 1)),
            1,
        )
        phase_input_dim = output_dim * (1 + 4 * self.contact_timing_phase_num_harmonics)
        phase_output_dim = int(
            getattr(model_cfg, "contact_timing_phase_output_dim", encoder_output_dim)
        )
        if phase_output_dim != encoder_output_dim:
            raise ValueError(
                "contact timing phase output dim must match contact timing encoder output dim "
                f"({encoder_output_dim}), got {phase_output_dim}"
            )
        if self.phase_timing_enabled:
            self.contact_timing_phase_encoder = TimingPhaseConditionEncoder(
                input_dim=phase_input_dim,
                hidden_dim=int(getattr(model_cfg, "contact_timing_phase_hidden_dim", 128)),
                output_dim=phase_output_dim,
                dropout=float(getattr(model_cfg, "contact_timing_encoder_dropout", 0.1)),
            )
        self.contact_timing_branch_fusion = TimingBranchFusion(
            hidden_dim=encoder_output_dim,
            fusion_hidden_dim=int(
                getattr(model_cfg, "contact_timing_branch_fusion_hidden_dim", 256)
            ),
        )
        self.contact_timing_predictor = predictor
        self.contact_timing_feature_names = feature_names
        self.contact_timing_ckpt = ckpt_path
        self.contact_timing_source = str(
            getattr(model_cfg, "contact_timing_source", "content")
        ).strip().lower()
        self.contact_timing_scale = float(getattr(model_cfg, "contact_timing_scale", 1.0))
        self.guidance_contact_timing_scale = float(
            getattr(model_cfg, "guidance_contact_timing_scale", self.contact_timing_scale)
        )
        self.guidance_uncondp_contact_timing = float(
            getattr(model_cfg, "guidance_uncondp_contact_timing", self.guidance_uncodp)
        )
        self.guidance_zero_uncond_contact_timing = bool(
            getattr(model_cfg, "guidance_zero_uncond_contact_timing", True)
        )
        self.contact_timing_enabled = True

    def _resolve_contact_timing_inputs(
        self,
        *,
        default_motion,
        default_lengths,
        content_motion=None,
        content_lengths=None,
        style_motion=None,
        style_lengths=None,
        conditioned_content_motion=None,
        conditioned_content_lengths=None,
    ):
        source = self.contact_timing_source
        if source in {"content", "content_motion"} and content_motion is not None:
            return content_motion, content_lengths
        if source in {"conditioned_content", "conditioned_content_motion"} and conditioned_content_motion is not None:
            return conditioned_content_motion, conditioned_content_lengths
        if source in {"style", "style_motion"} and style_motion is not None:
            return style_motion, style_lengths
        if source in {"self", "motion", "default"}:
            return default_motion, default_lengths
        if source in {"content", "content_motion"}:
            return default_motion, default_lengths
        raise ValueError(f"unsupported contact timing source: {self.contact_timing_source}")

    def _build_contact_timing_condition(
        self,
        motion_norm,
        lengths,
        *,
        duplicate_for_guidance=False,
        apply_dropout=False,
        scale=1.0,
    ):
        timing_stats = self._predict_contact_timing_sequence(
            motion_norm,
            lengths,
            allow_grad=False,
        )
        if timing_stats is None:
            return None
        timing_cond, _timing_branch = self._build_contact_timing_conditions_from_stats(
            timing_stats,
            duplicate_for_guidance=duplicate_for_guidance,
            apply_dropout=apply_dropout,
            scale=scale,
        )
        return timing_cond

    def _apply_condition_guidance(
        self,
        cond,
        *,
        duplicate_for_guidance=False,
        apply_dropout=False,
        dropout_prob=None,
        scale=1.0,
        zero_uncond_for_guidance=True,
        shared_mask_uncond=None,
    ):
        if cond is None:
            return None, shared_mask_uncond
        if apply_dropout:
            if shared_mask_uncond is None:
                drop_prob = (
                    self.guidance_uncondp_contact_timing
                    if dropout_prob is None
                    else float(dropout_prob)
                )
                shared_mask_uncond = (
                    torch.rand(cond.shape[0], device=cond.device) < float(drop_prob)
                )
            cond = cond.clone()
            cond[shared_mask_uncond, ...] = 0
        scale = float(scale)
        if scale != 1.0:
            cond = cond * scale
        if duplicate_for_guidance:
            if zero_uncond_for_guidance:
                cond = torch.cat([torch.zeros_like(cond), cond], dim=0)
            else:
                cond = torch.cat([cond, cond], dim=0)
        return cond, shared_mask_uncond

    def _predict_contact_timing_sequence(self, motion_norm, lengths, *, allow_grad=False):
        if (
            not self.contact_timing_enabled
            or self.contact_timing_predictor is None
            or self.contact_timing_encoder is None
        ):
            return None

        self.contact_timing_predictor.eval()

        def _compute():
            joints = self.feats2joints(motion_norm)
            timing_features = build_contact_timing_feature_tensor(
                joints,
                self.contact_timing_feature_names,
            )
            timing_logits = self.contact_timing_predictor(timing_features)
            timing_prob = torch.sigmoid(timing_logits)
            timing_mask = lengths_to_mask(
                lengths,
                device=motion_norm.device,
                max_len=timing_prob.shape[1],
            )
            trajectory = build_root_trajectory_tensor(joints)
            return {
                "joints": joints,
                "timing_features": timing_features,
                "timing_logits": timing_logits,
                "timing_prob": timing_prob,
                "timing_mask": timing_mask,
                "trajectory": trajectory,
            }

        if allow_grad:
            return _compute()
        with torch.no_grad():
            return _compute()

    def _build_root_trajectory_from_motion(self, motion_norm):
        joints = self.feats2joints(motion_norm)
        trajectory = build_root_trajectory_tensor(joints)
        return normalize_root_trajectory_tensor(
            trajectory,
            align_xz=self.root_traj_align_xz,
            align_yaw=self.root_traj_align_yaw,
        )

    def _build_contact_timing_conditions_from_stats(
        self,
        timing_stats,
        *,
        duplicate_for_guidance=False,
        apply_dropout=False,
        scale=1.0,
    ):
        if timing_stats is None:
            return None, None

        event_hidden = self.contact_timing_encoder(
            timing_stats["timing_prob"],
            mask=timing_stats["timing_mask"],
        )
        phase_hidden = None
        if self.phase_timing_enabled and self.contact_timing_phase_encoder is not None:
            phase_features = extract_phase_features(
                timing_stats["timing_prob"],
                mask=timing_stats["timing_mask"],
                num_harmonics=self.contact_timing_phase_num_harmonics,
            )
            phase_hidden = self.contact_timing_phase_encoder(phase_features)
        timing_branch_hidden = (
            self.contact_timing_branch_fusion(event_hidden, phase_hidden)
            if self.contact_timing_branch_fusion is not None
            else event_hidden
        )
        shared_mask_uncond = None
        event_hidden, shared_mask_uncond = self._apply_condition_guidance(
            event_hidden,
            duplicate_for_guidance=duplicate_for_guidance,
            apply_dropout=apply_dropout,
            dropout_prob=self.guidance_uncondp_contact_timing,
            scale=scale,
            zero_uncond_for_guidance=self.guidance_zero_uncond_contact_timing,
            shared_mask_uncond=shared_mask_uncond,
        )
        timing_branch_hidden, _shared_mask_uncond = self._apply_condition_guidance(
            timing_branch_hidden,
            duplicate_for_guidance=duplicate_for_guidance,
            apply_dropout=apply_dropout,
            dropout_prob=self.guidance_uncondp_contact_timing,
            scale=scale,
            zero_uncond_for_guidance=self.guidance_zero_uncond_contact_timing,
            shared_mask_uncond=shared_mask_uncond,
        )
        return event_hidden, timing_branch_hidden

    def _apply_selective_content_suppression(self, motion, *, preserve_scale=None):
        if not self.selective_enabled or self.selective_target_motion_feature_indices is None:
            return motion
        scale = self._as_scalar(preserve_scale, self.selective_content_preserve)
        return self._scale_motion_feature_indices(
            motion,
            self.selective_target_motion_feature_indices,
            scale,
        )

    def _get_pose_dof_slice(self):
        pose_end = self.imf_dof
        if self.trans_dof_slice is not None:
            pose_end = int(self.trans_dof_slice[0])
        if pose_end <= 0:
            return None
        return (0, pose_end)

    def _encode_imf_condition(
        self,
        masked_imfs,
        lengths,
        *,
        duplicate_for_guidance=False,
        apply_dropout=False,
        scale=1.0,
        dropout_prob=None,
        zero_uncond_for_guidance=True,
    ):
        length_tensor = torch.as_tensor(lengths, device=masked_imfs.device)
        eff_lengths = torch.clamp(length_tensor, max=masked_imfs.shape[-1])
        imf_emb = self.frequency_branch.encode_imfs(
            masked_imfs,
            eff_lengths,
        )
        if apply_dropout:
            drop_prob = self.guidance_uncodp_fine if dropout_prob is None else float(dropout_prob)
            mask_uncond = torch.rand(imf_emb.shape[0], device=imf_emb.device) < drop_prob
            imf_emb = imf_emb.clone()
            imf_emb[mask_uncond, ...] = 0
        scale = float(scale)
        if scale != 1.0:
            imf_emb = imf_emb * scale
        if duplicate_for_guidance:
            if zero_uncond_for_guidance:
                imf_emb = torch.cat([torch.zeros_like(imf_emb), imf_emb], dim=0)
            else:
                imf_emb = torch.cat([imf_emb, imf_emb], dim=0)
        return imf_emb

    def _build_selected_imf_condition(
        self,
        imfs,
        lengths,
        band_indices,
        *,
        dof_indices=None,
        duplicate_for_guidance=False,
        apply_dropout=False,
        scale=1.0,
    ):
        if not band_indices:
            return None
        masked_imfs = self._mask_imfs(
            imfs,
            band_indices=band_indices,
            dof_indices=dof_indices,
        )
        return self._encode_imf_condition(
            masked_imfs,
            lengths,
            duplicate_for_guidance=duplicate_for_guidance,
            apply_dropout=apply_dropout,
            scale=scale,
            dropout_prob=self.guidance_uncodp_fine,
            zero_uncond_for_guidance=True,
        )

    def _build_fine_style_condition(
        self,
        imfs,
        lengths,
        *,
        duplicate_for_guidance=False,
        apply_dropout=False,
        scale=1.0,
    ):
        dof_indices = self.selective_target_dof_indices if self.selective_enabled else None
        band_indices = (
            self.selective_target_band_indices if self.selective_enabled else self.fine_band_indices
        )
        return self._build_selected_imf_condition(
            imfs,
            lengths,
            band_indices,
            dof_indices=dof_indices,
            duplicate_for_guidance=duplicate_for_guidance,
            apply_dropout=apply_dropout,
            scale=scale,
        )

    def _build_traj_imf_condition(
        self,
        imfs,
        lengths,
        *,
        duplicate_for_guidance=False,
        apply_dropout=False,
        scale=1.0,
    ):
        if not self.traj_band_indices or self.trans_dof_slice is None:
            return None
        masked_imfs = self._mask_imfs(
            imfs,
            band_indices=self.traj_band_indices,
            dof_slice=self.trans_dof_slice,
        )
        return self._encode_imf_condition(
            masked_imfs,
            lengths,
            duplicate_for_guidance=duplicate_for_guidance,
            apply_dropout=apply_dropout,
            scale=scale,
            dropout_prob=self.guidance_uncodp_traj,
            zero_uncond_for_guidance=self.guidance_zero_uncond_traj,
        )

    def _build_content_traj_condition(
        self,
        imfs,
        lengths,
        *,
        duplicate_for_guidance=False,
        scale=1.0,
    ):
        if not self.content_traj_band_indices or self.trans_dof_slice is None:
            return None
        masked_imfs = self._mask_imfs(
            imfs,
            band_indices=self.content_traj_band_indices,
            dof_slice=self.trans_dof_slice,
        )
        return self._encode_imf_condition(
            masked_imfs,
            lengths,
            duplicate_for_guidance=duplicate_for_guidance,
            apply_dropout=False,
            scale=scale,
            zero_uncond_for_guidance=False,
        )

    def _build_content_imf_remover_condition(
        self,
        imfs,
        lengths,
        *,
        duplicate_for_guidance=False,
        scale=1.0,
    ):
        if not self.content_imf_remover_band_indices:
            return None
        pose_dof_slice = self._get_pose_dof_slice()
        if pose_dof_slice is None:
            return None
        masked_imfs = self._mask_imfs(
            imfs,
            band_indices=self.content_imf_remover_band_indices,
            dof_slice=pose_dof_slice,
        )
        return self._encode_imf_condition(
            masked_imfs,
            lengths,
            duplicate_for_guidance=duplicate_for_guidance,
            apply_dropout=False,
            scale=scale,
            zero_uncond_for_guidance=False,
        )

    def _compose_dual_conditions(
        self,
        content_cond,
        coarse_style_cond,
        fine_style_cond,
        trans_cond,
        traj_fine_cond=None,
        content_traj_cond=None,
        content_imf_remover_cond=None,
    ):
        return [
            content_cond,
            coarse_style_cond,
            fine_style_cond,
            trans_cond,
            traj_fine_cond,
            content_traj_cond,
            content_imf_remover_cond,
        ]

    def _merge_coarse_imf_summary(
        self,
        coarse_style_emb,
        imfs,
        lengths,
        *,
        duplicate_for_guidance=False,
    ):
        coarse_imf_emb = self._build_selected_imf_condition(
            imfs,
            lengths,
            self.coarse_band_indices,
            duplicate_for_guidance=duplicate_for_guidance,
        )
        if coarse_imf_emb is None:
            return coarse_style_emb
        return coarse_style_emb + self.coarse_imf_proj(coarse_imf_emb)

    def _sample_source_mix_indices(self, batch_size, device, prob):
        if (
            not self.training
            or not self.style_source_mix_enabled
            or batch_size <= 1
            or float(prob) <= 0.0
        ):
            return None
        mix_mask = torch.rand(batch_size, device=device) < float(prob)
        if not bool(mix_mask.any()):
            return None
        perm = torch.randperm(batch_size, device=device)
        same = perm == torch.arange(batch_size, device=device)
        if bool(same.any()) and batch_size > 1:
            perm[same] = (perm[same] + 1) % batch_size
        indices = torch.arange(batch_size, device=device)
        return torch.where(mix_mask, perm, indices)

    def _index_lengths(self, lengths, indices):
        if indices is None:
            return lengths
        index_list = indices.detach().cpu().tolist()
        return [int(lengths[int(i)]) for i in index_list]

    def _apply_training_branch_dropout(self, cond, prob):
        if cond is None or not self.training or float(prob) <= 0.0:
            return cond
        drop_mask = torch.rand(cond.shape[0], device=cond.device) < float(prob)
        if not bool(drop_mask.any()):
            return cond
        cond = cond.clone()
        cond[drop_mask, ...] = 0
        return cond

    def _coarse_fine_decorrelation_loss(self, coarse_cond, fine_cond):
        if coarse_cond is None or fine_cond is None:
            return None
        coarse = coarse_cond.reshape(coarse_cond.shape[0], -1)
        fine = fine_cond.reshape(fine_cond.shape[0], -1)
        shared_dim = min(coarse.shape[-1], fine.shape[-1])
        if shared_dim <= 0:
            return None
        coarse = coarse[..., :shared_dim]
        fine = fine[..., :shared_dim]
        coarse_norm = torch.linalg.vector_norm(coarse, dim=-1)
        fine_norm = torch.linalg.vector_norm(fine, dim=-1)
        valid = (coarse_norm > 1e-6) & (fine_norm > 1e-6)
        if not bool(valid.any()):
            return coarse.sum() * 0.0
        cosine = F.cosine_similarity(coarse[valid], fine[valid], dim=-1, eps=1e-6)
        return (cosine ** 2).mean()

    def train_diffusion_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]
        target_imfs, target_global = self._resolve_target_imfs(batch, lengths)
        batch_size = feats_ref.shape[0]
        coarse_indices = self._sample_source_mix_indices(
            batch_size,
            feats_ref.device,
            self.style_source_mix_coarse_prob,
        )
        fine_indices = self._sample_source_mix_indices(
            batch_size,
            feats_ref.device,
            self.style_source_mix_fine_prob,
        )
        traj_indices = self._sample_source_mix_indices(
            batch_size,
            feats_ref.device,
            self.style_source_mix_traj_prob,
        )
        coarse_motion = feats_ref if coarse_indices is None else feats_ref[coarse_indices]
        fine_imfs = target_imfs if fine_indices is None else target_imfs[fine_indices]
        traj_imfs = target_imfs if traj_indices is None else target_imfs[traj_indices]
        coarse_lengths = self._index_lengths(lengths, coarse_indices)
        fine_lengths = self._index_lengths(lengths, fine_indices)
        traj_lengths = self._index_lengths(lengths, traj_indices)
        coarse_imfs = target_imfs if coarse_indices is None else target_imfs[coarse_indices]
        conditioned_motion = self._apply_selective_content_suppression(feats_ref)
        with torch.no_grad():
            z, _ = self.vae.encode(feats_ref, lengths)
        cond_emb = self._build_content_condition(conditioned_motion, lengths)
        coarse_style_emb = self._build_style_condition(
            coarse_motion,
            coarse_lengths,
            apply_dropout=True,
        )
        coarse_style_emb = self._merge_coarse_imf_summary(
            coarse_style_emb,
            coarse_imfs,
            coarse_lengths,
        )
        coarse_style_emb_for_decor = None
        if self.lambda_coarse_fine_decorrelation > 0.0:
            coarse_style_emb_for_decor = self._build_style_condition(
                coarse_motion,
                coarse_lengths,
                apply_dropout=False,
            )
            coarse_style_emb_for_decor = self._merge_coarse_imf_summary(
                coarse_style_emb_for_decor,
                coarse_imfs,
                coarse_lengths,
            )
        coarse_style_emb = self._apply_training_branch_dropout(
            coarse_style_emb,
            self.train_coarse_branch_dropout,
        )
        fine_style_cond = self._build_fine_style_condition(
            fine_imfs,
            fine_lengths,
            apply_dropout=True,
        )
        fine_style_cond_for_decor = None
        if self.lambda_coarse_fine_decorrelation > 0.0:
            fine_style_cond_for_decor = self._build_fine_style_condition(
                fine_imfs,
                fine_lengths,
                apply_dropout=False,
            )
        fine_style_cond = self._apply_training_branch_dropout(
            fine_style_cond,
            self.train_fine_branch_dropout,
        )
        trans_cond = self._build_trans_condition(conditioned_motion)
        traj_fine_cond = self._build_traj_imf_condition(
            traj_imfs,
            traj_lengths,
            apply_dropout=True,
            scale=self.traj_imf_scale,
        )
        traj_fine_cond = self._apply_training_branch_dropout(
            traj_fine_cond,
            self.train_traj_branch_dropout,
        )
        content_traj_cond = self._build_content_traj_condition(
            target_imfs,
            lengths,
            scale=self.content_traj_imf_scale,
        )
        content_imf_remover_cond = self._build_content_imf_remover_condition(
            target_imfs,
            lengths,
            scale=self.content_imf_remover_scale,
        )
        contact_timing_motion, contact_timing_lengths = self._resolve_contact_timing_inputs(
            default_motion=feats_ref,
            default_lengths=lengths,
            content_motion=feats_ref,
            content_lengths=lengths,
            conditioned_content_motion=conditioned_motion,
            conditioned_content_lengths=lengths,
            style_motion=feats_ref,
            style_lengths=lengths,
        )
        contact_timing_stats = self._predict_contact_timing_sequence(
            contact_timing_motion,
            contact_timing_lengths,
            allow_grad=False,
        )
        contact_timing_cond, timing_branch_hidden = self._build_contact_timing_conditions_from_stats(
            contact_timing_stats,
            apply_dropout=True,
            scale=self.contact_timing_scale,
        )
        encoder_hidden_states = self._compose_dual_conditions(
            cond_emb,
            coarse_style_emb,
            fine_style_cond,
            trans_cond,
            traj_fine_cond=traj_fine_cond,
            content_traj_cond=content_traj_cond,
            content_imf_remover_cond=content_imf_remover_cond,
        )
        denoiser_kwargs = {}
        if contact_timing_cond is not None and self.contact_timing_route in {"trajectory", "both"}:
            denoiser_kwargs["contact_timing_hidden"] = contact_timing_cond
        if timing_branch_hidden is not None and self.contact_timing_route in {"timing", "both"}:
            denoiser_kwargs["timing_hidden"] = timing_branch_hidden
        rs_set = self._diffusion_process_with_state(
            z,
            encoder_hidden_states,
            lengths,
            denoiser_kwargs=denoiser_kwargs,
        )
        rs_set["coarse_style_emb_for_decor"] = coarse_style_emb_for_decor
        rs_set["fine_style_emb_for_decor"] = fine_style_cond_for_decor
        motion_pred = self._decode_prediction(rs_set, lengths)
        rs_set.update(
            self._build_aux_outputs(
                batch,
                motion_pred,
                target_imfs=target_imfs,
                target_global=target_global,
            )
        )
        if self.lambda_root_traj_xz > 0.0 or self.lambda_root_traj_yaw > 0.0:
            rs_set["root_traj_pred"] = self._build_root_trajectory_from_motion(motion_pred)
            with torch.no_grad():
                rs_set["root_traj_ref"] = self._build_root_trajectory_from_motion(feats_ref)
            rs_set["root_traj_mask"] = lengths_to_mask(
                lengths,
                device=motion_pred.device,
                max_len=rs_set["root_traj_pred"].shape[1],
            )
        if contact_timing_stats is not None:
            rs_set["contact_timing_ref_motion"] = contact_timing_motion
            rs_set["contact_timing_ref_lengths"] = contact_timing_lengths
            rs_set["contact_timing_ref_prob"] = contact_timing_stats["timing_prob"]
            rs_set["contact_timing_ref_mask"] = contact_timing_stats["timing_mask"]
            rs_set["contact_timing_ref_traj"] = contact_timing_stats["trajectory"]
        motion_lengths = torch.as_tensor(lengths, device=motion_pred.device)
        rs_set["fine_target_emb"] = self._build_fine_style_condition(
            rs_set["target_imfs"],
            lengths,
        )
        motion_feature_indices = (
            self.selective_target_motion_feature_indices
            if self.selective_enabled and self.selective_branch_motion_mask
            else None
        )
        rs_set["fine_motion_emb"] = self.frequency_branch.encode_motion(
            motion_pred,
            torch.clamp(motion_lengths, max=motion_pred.shape[1]),
            feature_indices=motion_feature_indices,
        )
        return rs_set

    def _compute_aux_losses(self, rs_set, batch):
        total = torch.tensor(0.0, device=rs_set["noise_pred"].device)
        logs = {}

        if "target_imfs" in rs_set and (self.lambda_imf > 0.0 or self.lambda_imf_trans > 0.0):
            lengths = torch.as_tensor(batch["length"], device=rs_set["pred_imfs"].device)
            eff_lengths = torch.clamp(lengths, max=rs_set["pred_imfs"].shape[-1])
            loss_imf = masked_l1(rs_set["pred_imfs"], rs_set["target_imfs"], eff_lengths)
            if self.lambda_imf > 0.0:
                total = total + self.lambda_imf * loss_imf
            logs["imf"] = loss_imf
            if self.lambda_imf_trans > 0.0 and self.trans_dof_slice is not None:
                loss_imf_trans = masked_l1_dof(
                    rs_set["pred_imfs"],
                    rs_set["target_imfs"],
                    eff_lengths,
                    self.trans_dof_slice,
                )
                total = total + self.lambda_imf_trans * loss_imf_trans
                logs["imf_trans"] = loss_imf_trans
            if self.lambda_hht_amp > 0.0 or self.lambda_hht_freq > 0.0:
                amp_loss, freq_loss = hht_feature_loss(
                    self.imf_detector,
                    rs_set["pred_imfs"],
                    rs_set["target_imfs"],
                    eff_lengths,
                )
                if self.lambda_hht_amp > 0.0:
                    total = total + self.lambda_hht_amp * amp_loss
                    logs["hht_amp"] = amp_loss
                if self.lambda_hht_freq > 0.0:
                    total = total + self.lambda_hht_freq * freq_loss
                    logs["hht_freq"] = freq_loss

        if self.lambda_global > 0.0 and "target_global" in rs_set:
            global_loss = cosine_global_loss(rs_set["pred_global"], rs_set["target_global"])
            total = total + self.lambda_global * global_loss
            logs["global"] = global_loss

        if self.selective_enabled and "target_imfs" in rs_set:
            lengths = torch.as_tensor(batch["length"], device=rs_set["pred_imfs"].device)
            eff_lengths = torch.clamp(lengths, max=rs_set["pred_imfs"].shape[-1])
            if self.lambda_imf_region_target > 0.0 and self.selective_target_dof_indices:
                selective_target_loss = masked_l1_selective(
                    rs_set["pred_imfs"],
                    rs_set["target_imfs"],
                    eff_lengths,
                    band_indices=self.selective_target_band_indices,
                    dof_indices=self.selective_target_dof_indices,
                )
                total = total + self.lambda_imf_region_target * selective_target_loss
                logs["imf_region_target"] = selective_target_loss
            if self.lambda_imf_region_keep > 0.0 and self.selective_keep_dof_indices:
                selective_keep_loss = masked_l1_selective(
                    rs_set["pred_imfs"],
                    rs_set["target_imfs"],
                    eff_lengths,
                    dof_indices=self.selective_keep_dof_indices,
                )
                total = total + self.lambda_imf_region_keep * selective_keep_loss
                logs["imf_region_keep"] = selective_keep_loss

        if self.lambda_branch > 0.0 and "fine_target_emb" in rs_set:
            branch_loss = F.mse_loss(rs_set["fine_motion_emb"], rs_set["fine_target_emb"].detach())
            total = total + self.lambda_branch * branch_loss
            logs["branch"] = branch_loss

        if (
            self.lambda_coarse_fine_decorrelation > 0.0
            and "coarse_style_emb_for_decor" in rs_set
            and "fine_style_emb_for_decor" in rs_set
        ):
            decor_loss = self._coarse_fine_decorrelation_loss(
                rs_set["coarse_style_emb_for_decor"],
                rs_set["fine_style_emb_for_decor"],
            )
            if decor_loss is not None:
                total = total + self.lambda_coarse_fine_decorrelation * decor_loss
                logs["coarse_fine_decor"] = decor_loss

        native_lengths = torch.as_tensor(batch["length"], device=rs_set["noise_pred"].device)
        native_motion_pred = self._select_decomposer_motion(rs_set["motion_pred"])
        native_motion_ref = self._select_decomposer_motion(batch["motion"])
        if self.native_loss_type == "fft":
            band_losses = fft_spectral_losses(
                native_motion_pred,
                native_motion_ref,
                native_lengths,
                sample_rate=self.decomposer_frame_rate,
                band_edges_hz=self.native_fft_band_edges_hz,
                log_magnitude=self.native_fft_log_magnitude,
            )
            for name, value in band_losses.items():
                logs[f"native_fft_{name}"] = value
                weight = self.native_fft_weights.get(name, 0.0)
                if weight > 0.0:
                    total = total + weight * value
        elif self.native_loss_type in {"wavelet", "dwt"}:
            coeff_losses = wavelet_spectral_losses(
                native_motion_pred,
                native_motion_ref,
                native_lengths,
                levels=self.native_wavelet_levels,
                log_magnitude=self.native_wavelet_log_magnitude,
            )
            for name, value in coeff_losses.items():
                logs[f"native_wavelet_{name}"] = value
                weight = self.native_wavelet_weights.get(name, 0.0)
                if weight > 0.0:
                    total = total + weight * value

        if (
            (self.lambda_root_traj_xz > 0.0 or self.lambda_root_traj_yaw > 0.0)
            and "root_traj_pred" in rs_set
            and "root_traj_ref" in rs_set
            and "root_traj_mask" in rs_set
        ):
            root_traj_mask = rs_set["root_traj_mask"]
            if self.lambda_root_traj_xz > 0.0:
                root_traj_xz = masked_timing_mse(
                    rs_set["root_traj_pred"][..., :2],
                    rs_set["root_traj_ref"][..., :2].detach(),
                    root_traj_mask,
                )
                total = total + self.lambda_root_traj_xz * root_traj_xz
                logs["root_traj_xz"] = root_traj_xz
            if self.lambda_root_traj_yaw > 0.0:
                yaw_delta = wrap_angle(
                    rs_set["root_traj_pred"][..., 2:3]
                    - rs_set["root_traj_ref"][..., 2:3].detach()
                )
                root_traj_yaw = masked_timing_mse(
                    yaw_delta,
                    torch.zeros_like(yaw_delta),
                    root_traj_mask,
                )
                total = total + self.lambda_root_traj_yaw * root_traj_yaw
                logs["root_traj_yaw"] = root_traj_yaw

        if (
            (
                self.lambda_contact_timing > 0.0
                or self.lambda_contact_timing_bce > 0.0
                or self.lambda_contact_timing_traj > 0.0
                or self.lambda_contact_timing_phase > 0.0
            )
            and "contact_timing_ref_prob" in rs_set
            and "motion_pred" in rs_set
        ):
            pred_timing_stats = self._predict_contact_timing_sequence(
                rs_set["motion_pred"],
                batch["length"],
                allow_grad=True,
            )
            if pred_timing_stats is not None:
                timing_mask = torch.logical_and(
                    pred_timing_stats["timing_mask"].bool(),
                    rs_set["contact_timing_ref_mask"].bool(),
                )
                ref_prob = rs_set["contact_timing_ref_prob"].detach()
                pred_prob = pred_timing_stats["timing_prob"]
                if self.lambda_contact_timing > 0.0:
                    timing_loss = masked_timing_mse(pred_prob, ref_prob, timing_mask)
                    total = total + self.lambda_contact_timing * timing_loss
                    logs["contact_timing"] = timing_loss
                if self.lambda_contact_timing_bce > 0.0:
                    timing_bce = masked_bce_with_probs(pred_prob, ref_prob, timing_mask)
                    total = total + self.lambda_contact_timing_bce * timing_bce
                    logs["contact_timing_bce"] = timing_bce
                if self.lambda_contact_timing_traj > 0.0:
                    traj_loss = masked_timing_mse(
                        pred_timing_stats["trajectory"],
                        rs_set["contact_timing_ref_traj"].detach(),
                        timing_mask,
                    )
                    total = total + self.lambda_contact_timing_traj * traj_loss
                    logs["contact_timing_traj"] = traj_loss
                if self.lambda_contact_timing_phase > 0.0:
                    pred_phase = extract_phase_features(
                        pred_prob,
                        mask=timing_mask,
                        num_harmonics=self.contact_timing_phase_num_harmonics,
                    )
                    ref_phase = extract_phase_features(
                        ref_prob,
                        mask=timing_mask,
                        num_harmonics=self.contact_timing_phase_num_harmonics,
                    ).detach()
                    phase_loss = F.mse_loss(pred_phase, ref_phase)
                    total = total + self.lambda_contact_timing_phase * phase_loss
                    logs["contact_timing_phase"] = phase_loss

        return total, logs

    def forward(self, batch):
        content_motion = self._normalize_external_motion(batch["content_motion"].float())
        style_motion = self._normalize_external_motion(batch["style_motion"].float())
        coarse_style_motion = batch.get("coarse_style_motion")
        fine_style_motion = batch.get("fine_style_motion")
        if coarse_style_motion is not None:
            coarse_style_motion = self._normalize_external_motion(coarse_style_motion.float())
        else:
            coarse_style_motion = style_motion
        if fine_style_motion is not None:
            fine_style_motion = self._normalize_external_motion(fine_style_motion.float())
        else:
            fine_style_motion = style_motion
        content_lengths = (
            batch["length"]
            if "length" in batch and batch["length"] is not None
            else [int(content_motion.shape[1])] * content_motion.shape[0]
        )
        style_lengths = [int(style_motion.shape[1])] * style_motion.shape[0]
        coarse_style_lengths = [int(coarse_style_motion.shape[1])] * coarse_style_motion.shape[0]
        fine_style_lengths = [int(fine_style_motion.shape[1])] * fine_style_motion.shape[0]
        coarse_scale = batch["tag_scale"]
        fine_scale = self._as_scalar(batch.get("tag_scale_fine"), self.guidance_fine_scale)
        traj_scale = self._as_scalar(batch.get("tag_scale_traj"), self.guidance_traj_scale)
        content_traj_scale = self._as_scalar(
            batch.get("tag_scale_content_traj"),
            self.guidance_content_traj_scale,
        )
        content_imf_remover_scale = self._as_scalar(
            batch.get("tag_scale_content_imf_remover"),
            self.guidance_content_imf_remover_scale,
        )
        contact_timing_scale = self._as_scalar(
            batch.get("tag_scale_contact_timing"),
            self.guidance_contact_timing_scale,
        )
        preserve_scale = self._as_scalar(
            batch.get("tag_scale_preserve"),
            self.guidance_content_preserve,
        )
        conditioned_content_motion = self._apply_selective_content_suppression(
            content_motion,
            preserve_scale=preserve_scale,
        )

        content_cond = self._build_content_condition(
            conditioned_content_motion,
            content_lengths,
            duplicate_for_guidance=True,
        )
        coarse_style_cond = self._build_style_condition(
            coarse_style_motion,
            coarse_style_lengths,
            duplicate_for_guidance=True,
        )
        content_imfs, _, _ = self._extract_online_imfs(content_motion, content_lengths)
        coarse_style_imfs, _, _ = self._extract_online_imfs(
            coarse_style_motion,
            coarse_style_lengths,
        )
        fine_style_imfs, _, _ = self._extract_online_imfs(
            fine_style_motion,
            fine_style_lengths,
        )
        coarse_style_cond = self._merge_coarse_imf_summary(
            coarse_style_cond,
            coarse_style_imfs,
            coarse_style_lengths,
            duplicate_for_guidance=True,
        )
        fine_style_cond = self._build_fine_style_condition(
            fine_style_imfs,
            fine_style_lengths,
            duplicate_for_guidance=True,
            scale=fine_scale,
        )
        trans_cond = self._build_trans_condition(
            conditioned_content_motion,
            duplicate_for_guidance=True,
        )
        traj_fine_cond = self._build_traj_imf_condition(
            fine_style_imfs,
            fine_style_lengths,
            duplicate_for_guidance=True,
            scale=traj_scale,
        )
        content_traj_cond = self._build_content_traj_condition(
            content_imfs,
            content_lengths,
            duplicate_for_guidance=True,
            scale=content_traj_scale,
        )
        content_imf_remover_cond = self._build_content_imf_remover_condition(
            content_imfs,
            content_lengths,
            duplicate_for_guidance=True,
            scale=content_imf_remover_scale,
        )
        contact_timing_motion, contact_timing_lengths = self._resolve_contact_timing_inputs(
            default_motion=content_motion,
            default_lengths=content_lengths,
            content_motion=content_motion,
            content_lengths=content_lengths,
            style_motion=style_motion,
            style_lengths=style_lengths,
            conditioned_content_motion=conditioned_content_motion,
            conditioned_content_lengths=content_lengths,
        )
        contact_timing_stats = self._predict_contact_timing_sequence(
            contact_timing_motion,
            contact_timing_lengths,
            allow_grad=False,
        )
        contact_timing_cond, timing_branch_hidden = self._build_contact_timing_conditions_from_stats(
            contact_timing_stats,
            duplicate_for_guidance=True,
            scale=contact_timing_scale,
        )
        denoiser_kwargs = {}
        if contact_timing_cond is not None and self.contact_timing_route in {"trajectory", "both"}:
            denoiser_kwargs["contact_timing_hidden"] = contact_timing_cond
        if timing_branch_hidden is not None and self.contact_timing_route in {"timing", "both"}:
            denoiser_kwargs["timing_hidden"] = timing_branch_hidden
        z = self._diffusion_reverse(
            self._compose_dual_conditions(
                content_cond,
                coarse_style_cond,
                fine_style_cond,
                trans_cond,
                traj_fine_cond=traj_fine_cond,
                content_traj_cond=content_traj_cond,
                content_imf_remover_cond=content_imf_remover_cond,
            ),
            content_lengths,
            coarse_scale,
            denoiser_kwargs=denoiser_kwargs,
        )
        feats_rst, joints = self._decode_to_joints(z, content_lengths)
        if bool(batch.get("return_motion_feats", False)):
            return {
                "joints": joints,
                "motion_feats": remove_padding(feats_rst.detach().cpu(), content_lengths),
            }
        return joints

    def t2m_eval(self, batch):
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()

        start = time.time()

        if self.trainer.datamodule.is_mm:
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            text_lengths = text_lengths.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        conditioned_motions = self._apply_selective_content_suppression(
            motions,
            preserve_scale=self.guidance_content_preserve,
        )
        content_cond = self._build_content_condition(
            conditioned_motions,
            lengths,
            duplicate_for_guidance=True,
        )
        coarse_style_cond = self._build_style_condition(
            motions,
            lengths,
            duplicate_for_guidance=True,
        )
        style_imfs, _, _ = self._extract_online_imfs(motions, lengths)
        coarse_style_cond = self._merge_coarse_imf_summary(
            coarse_style_cond,
            style_imfs,
            lengths,
            duplicate_for_guidance=True,
        )
        fine_style_cond = self._build_fine_style_condition(
            style_imfs,
            lengths,
            duplicate_for_guidance=True,
            scale=self.guidance_fine_scale,
        )
        trans_cond = self._build_trans_condition(
            conditioned_motions,
            duplicate_for_guidance=True,
        )
        traj_fine_cond = self._build_traj_imf_condition(
            style_imfs,
            lengths,
            duplicate_for_guidance=True,
            scale=self.guidance_traj_scale,
        )
        content_traj_cond = self._build_content_traj_condition(
            style_imfs,
            lengths,
            duplicate_for_guidance=True,
            scale=self.guidance_content_traj_scale,
        )
        content_imf_remover_cond = self._build_content_imf_remover_condition(
            style_imfs,
            lengths,
            duplicate_for_guidance=True,
            scale=self.guidance_content_imf_remover_scale,
        )
        contact_timing_motion, contact_timing_lengths = self._resolve_contact_timing_inputs(
            default_motion=motions,
            default_lengths=lengths,
            content_motion=motions,
            content_lengths=lengths,
            style_motion=motions,
            style_lengths=lengths,
            conditioned_content_motion=conditioned_motions,
            conditioned_content_lengths=lengths,
        )
        contact_timing_stats = self._predict_contact_timing_sequence(
            contact_timing_motion,
            contact_timing_lengths,
            allow_grad=False,
        )
        contact_timing_cond, timing_branch_hidden = self._build_contact_timing_conditions_from_stats(
            contact_timing_stats,
            duplicate_for_guidance=True,
            scale=self.guidance_contact_timing_scale,
        )
        denoiser_kwargs = {}
        if contact_timing_cond is not None and self.contact_timing_route in {"trajectory", "both"}:
            denoiser_kwargs["contact_timing_hidden"] = contact_timing_cond
        if timing_branch_hidden is not None and self.contact_timing_route in {"timing", "both"}:
            denoiser_kwargs["timing_hidden"] = timing_branch_hidden
        z = self._diffusion_reverse(
            self._compose_dual_conditions(
                content_cond,
                coarse_style_cond,
                fine_style_cond,
                trans_cond,
                traj_fine_cond=traj_fine_cond,
                content_traj_cond=content_traj_cond,
                content_imf_remover_cond=content_imf_remover_cond,
            ),
            lengths,
            scale=self.guidance_scale,
            denoiser_kwargs=denoiser_kwargs,
        )

        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)

        return self._postprocess_t2m_eval(
            batch,
            motions,
            feats_rst,
            lengths,
            start,
            word_embs=word_embs,
            pos_ohot=pos_ohot,
            text_lengths=text_lengths,
        )
