from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

# The IMF model lives in the top-level models package of this extractor.
from models.imf_extractor import ht


class ImfLoss:
    """IMF pretraining loss: L1, EMD, and HHT.

    This is a lighter form of the IMF terms in `MLDLosses`. It keeps only what
    IMFExtractor pretraining needs:

    - L1 alignment of the predicted IMF against the offline MEMD IMF
    - EMD mean: channel means match on an RMS scale
    - EMD extrema: extrema density per frame matches
    - HHT amplitude and frequency: instantaneous amplitude and normalized frequency match
    """

    def __init__(
        self,
        frame_rate: float = 20.0,
        lambda_decomp: float = 1.0,
        lambda_emd: float = 0.5,
        lambda_ht: float = 0.5,
        alpha_ht_amp: float = 0.5,
        alpha_ht_freq: float = 0.5,
        beta_emd_mean: float = 0.5,
        beta_emd_ext: float = 0.5,
        temporal_aux_cfg: Optional[Dict[str, Any]] = None,
        definition_unsup_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.dt = 1.0 / float(frame_rate)
        self.f_nyquist = float(frame_rate) / 2.0

        self.lambda_decomp = float(lambda_decomp)
        self.lambda_emd = float(lambda_emd)
        self.lambda_ht = float(lambda_ht)
        self.alpha_ht_amp = float(alpha_ht_amp)
        self.alpha_ht_freq = float(alpha_ht_freq)
        self.beta_emd_mean = float(beta_emd_mean)
        self.beta_emd_ext = float(beta_emd_ext)

        aux_cfg = temporal_aux_cfg if isinstance(temporal_aux_cfg, dict) else {}
        aux_groups = aux_cfg.get("groups", []) or []
        self.temporal_aux_group_names = {str(name).strip() for name in aux_groups if str(name).strip()}
        self.temporal_normalize_by_target_rms = bool(aux_cfg.get("normalize_by_target_rms", True))
        self.base_lambda_temporal_delta = float(aux_cfg.get("lambda_delta", 0.0))
        self.base_lambda_temporal_accel = float(aux_cfg.get("lambda_accel", 0.0))
        self.base_lambda_ht_phase = float(aux_cfg.get("lambda_ht_phase", 0.0))
        self.lambda_temporal_delta = self.base_lambda_temporal_delta
        self.lambda_temporal_accel = self.base_lambda_temporal_accel
        self.lambda_ht_phase = self.base_lambda_ht_phase
        self.temporal_schedule_cfg = aux_cfg.get("schedule", {}) or {}
        self.phase_amp_weighted = bool(aux_cfg.get("phase_amp_weighted", True))
        self.phase_weight_clip = float(aux_cfg.get("phase_weight_clip", 3.0))
        aux_has_weight = (
            self.base_lambda_temporal_delta > 0.0
            or self.base_lambda_temporal_accel > 0.0
            or self.base_lambda_ht_phase > 0.0
        )
        enabled_flag = aux_cfg.get("enabled", None)
        self.temporal_aux_enabled = aux_has_weight if enabled_flag is None else (bool(enabled_flag) and aux_has_weight)
        self._temporal_runtime_scales = {"delta": 1.0, "accel": 1.0, "phase": 1.0}

        def_cfg = definition_unsup_cfg if isinstance(definition_unsup_cfg, dict) else {}
        def_groups = def_cfg.get("groups", []) or []
        self.definition_unsup_group_names = {
            str(name).strip()
            for name in def_groups
            if str(name).strip() and str(name).strip().lower() not in ("all", "*")
        }
        self.definition_unsup_normalize_by_rms = bool(def_cfg.get("normalize_by_rms", True))
        self.definition_unsup_rms_source = str(def_cfg.get("rms_source", "pred")).strip().lower()
        if self.definition_unsup_rms_source not in ("pred", "target", "none"):
            self.definition_unsup_rms_source = "pred"

        env_cfg = def_cfg.get("envelope", {}) or {}
        env_pool_kernel = max(int(env_cfg.get("pool_kernel", 9)), 1)
        if env_pool_kernel % 2 == 0:
            env_pool_kernel += 1
        self.definition_env_pool_kernel = env_pool_kernel

        zero_cfg = def_cfg.get("zero_cross", {}) or {}
        self.definition_softsign_scale = max(float(zero_cfg.get("softsign_scale", 8.0)), 1e-3)
        self.definition_zero_ext_margin = max(float(zero_cfg.get("margin", 1.0)), 0.0)

        self.base_lambda_definition_env_mean = float(def_cfg.get("lambda_env_mean", 0.0))
        self.base_lambda_definition_zero_ext = float(def_cfg.get("lambda_zero_ext", 0.0))
        self.base_lambda_definition_dc_mean = float(def_cfg.get("lambda_dc_mean", 0.0))
        self.lambda_definition_env_mean = self.base_lambda_definition_env_mean
        self.lambda_definition_zero_ext = self.base_lambda_definition_zero_ext
        self.lambda_definition_dc_mean = self.base_lambda_definition_dc_mean
        self.definition_schedule_cfg = def_cfg.get("schedule", {}) or {}

        def_has_weight = (
            self.base_lambda_definition_env_mean > 0.0
            or self.base_lambda_definition_zero_ext > 0.0
            or self.base_lambda_definition_dc_mean > 0.0
        )
        def_enabled_flag = def_cfg.get("enabled", None)
        self.definition_unsup_enabled = (
            def_has_weight if def_enabled_flag is None else (bool(def_enabled_flag) and def_has_weight)
        )
        self._definition_runtime_scales = {"env_mean": 1.0, "zero_ext": 1.0, "dc_mean": 1.0}

    # ----------------------------------------------------------------- public API
    def __call__(
        self,
        pred_imf: torch.Tensor,
        target_imf: torch.Tensor,
        debug_ctx: Optional[Dict] = None,
        time_mask: Optional[torch.Tensor] = None,
        group_specs: Optional[Sequence[Dict[str, Any]]] = None,
        group_loss_mode: str = "group_only",
        group_aux_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mode = str(group_loss_mode or "group_only").strip().lower()
        if group_specs and mode == "full_plus_aux":
            return self._call_full_plus_aux(
                pred_imf=pred_imf,
                target_imf=target_imf,
                debug_ctx=debug_ctx,
                time_mask=time_mask,
                group_specs=group_specs,
                group_aux_weight=float(group_aux_weight),
            )
        if group_specs and mode in ("group_only", "group"):
            return self._call_grouped(
                pred_imf=pred_imf,
                target_imf=target_imf,
                debug_ctx=debug_ctx,
                time_mask=time_mask,
                group_specs=group_specs,
            )
        if group_specs and mode not in ("none", "off", "single", "full"):
            raise ValueError(f"Unsupported group_loss_mode: {group_loss_mode}")
        return self._call_single(
            pred_imf=pred_imf,
            target_imf=target_imf,
            debug_ctx=debug_ctx,
            time_mask=time_mask,
            group_specs=group_specs,
        )

    def _compute_group_breakdown(
        self,
        pred_imf: torch.Tensor,
        target_imf: torch.Tensor,
        debug_ctx: Optional[Dict] = None,
        time_mask: Optional[torch.Tensor] = None,
        group_specs: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        prefixed_losses: Dict[str, torch.Tensor] = {}
        aggregate_losses: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=pred_imf.device)

        for group in group_specs or []:
            start = int(group["start"])
            end = int(group["end"])
            if end <= start:
                continue
            weight = float(group.get("weight", 1.0))
            name = str(group.get("name", f"{start}:{end}"))
            group_ctx = dict(debug_ctx or {})
            group_ctx["group_name"] = name
            group_total, group_losses = self._call_single(
                pred_imf=pred_imf[:, start:end, :],
                target_imf=target_imf[:, start:end, :],
                debug_ctx=group_ctx,
                time_mask=time_mask,
                current_group_name=name,
            )
            prefixed_losses[f"{name}/total"] = group_total
            total = total + weight * group_total
            for key, value in group_losses.items():
                if key == "total":
                    continue
                prefixed_losses[f"{name}/{key}"] = value
                if key not in aggregate_losses:
                    aggregate_losses[key] = weight * value
                else:
                    aggregate_losses[key] = aggregate_losses[key] + weight * value

        return total, prefixed_losses, aggregate_losses

    def _call_grouped(
        self,
        pred_imf: torch.Tensor,
        target_imf: torch.Tensor,
        debug_ctx: Optional[Dict] = None,
        time_mask: Optional[torch.Tensor] = None,
        group_specs: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total, prefixed_losses, aggregate_losses = self._compute_group_breakdown(
            pred_imf=pred_imf,
            target_imf=target_imf,
            debug_ctx=debug_ctx,
            time_mask=time_mask,
            group_specs=group_specs,
        )
        losses: Dict[str, torch.Tensor] = {}
        losses.update(prefixed_losses)
        losses.update(aggregate_losses)
        losses["total"] = total
        return total, losses

    def set_temporal_runtime_scales(
        self,
        *,
        delta_scale: Optional[float] = None,
        accel_scale: Optional[float] = None,
        phase_scale: Optional[float] = None,
    ) -> None:
        if delta_scale is not None:
            self._temporal_runtime_scales["delta"] = max(float(delta_scale), 0.0)
        if accel_scale is not None:
            self._temporal_runtime_scales["accel"] = max(float(accel_scale), 0.0)
        if phase_scale is not None:
            self._temporal_runtime_scales["phase"] = max(float(phase_scale), 0.0)

        self.lambda_temporal_delta = self.base_lambda_temporal_delta * self._temporal_runtime_scales["delta"]
        self.lambda_temporal_accel = self.base_lambda_temporal_accel * self._temporal_runtime_scales["accel"]
        self.lambda_ht_phase = self.base_lambda_ht_phase * self._temporal_runtime_scales["phase"]

    def get_temporal_weight_state(self) -> Dict[str, float]:
        return {
            "base_lambda_temporal_delta": float(self.base_lambda_temporal_delta),
            "base_lambda_temporal_accel": float(self.base_lambda_temporal_accel),
            "base_lambda_ht_phase": float(self.base_lambda_ht_phase),
            "delta_scale": float(self._temporal_runtime_scales["delta"]),
            "accel_scale": float(self._temporal_runtime_scales["accel"]),
            "phase_scale": float(self._temporal_runtime_scales["phase"]),
            "lambda_temporal_delta": float(self.lambda_temporal_delta),
            "lambda_temporal_accel": float(self.lambda_temporal_accel),
            "lambda_ht_phase": float(self.lambda_ht_phase),
        }

    def set_definition_runtime_scales(
        self,
        *,
        env_mean_scale: Optional[float] = None,
        zero_ext_scale: Optional[float] = None,
        dc_mean_scale: Optional[float] = None,
    ) -> None:
        if env_mean_scale is not None:
            self._definition_runtime_scales["env_mean"] = max(float(env_mean_scale), 0.0)
        if zero_ext_scale is not None:
            self._definition_runtime_scales["zero_ext"] = max(float(zero_ext_scale), 0.0)
        if dc_mean_scale is not None:
            self._definition_runtime_scales["dc_mean"] = max(float(dc_mean_scale), 0.0)

        self.lambda_definition_env_mean = (
            self.base_lambda_definition_env_mean * self._definition_runtime_scales["env_mean"]
        )
        self.lambda_definition_zero_ext = (
            self.base_lambda_definition_zero_ext * self._definition_runtime_scales["zero_ext"]
        )
        self.lambda_definition_dc_mean = (
            self.base_lambda_definition_dc_mean * self._definition_runtime_scales["dc_mean"]
        )

    def get_definition_weight_state(self) -> Dict[str, float]:
        return {
            "base_lambda_definition_env_mean": float(self.base_lambda_definition_env_mean),
            "base_lambda_definition_zero_ext": float(self.base_lambda_definition_zero_ext),
            "base_lambda_definition_dc_mean": float(self.base_lambda_definition_dc_mean),
            "definition_env_mean_scale": float(self._definition_runtime_scales["env_mean"]),
            "definition_zero_ext_scale": float(self._definition_runtime_scales["zero_ext"]),
            "definition_dc_mean_scale": float(self._definition_runtime_scales["dc_mean"]),
            "lambda_definition_env_mean": float(self.lambda_definition_env_mean),
            "lambda_definition_zero_ext": float(self.lambda_definition_zero_ext),
            "lambda_definition_dc_mean": float(self.lambda_definition_dc_mean),
        }

    def _call_full_plus_aux(
        self,
        pred_imf: torch.Tensor,
        target_imf: torch.Tensor,
        debug_ctx: Optional[Dict] = None,
        time_mask: Optional[torch.Tensor] = None,
        group_specs: Optional[Sequence[Dict[str, Any]]] = None,
        group_aux_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        main_total, main_losses = self._call_single(
            pred_imf=pred_imf,
            target_imf=target_imf,
            debug_ctx=debug_ctx,
            time_mask=time_mask,
            group_specs=group_specs,
        )
        group_total, prefixed_losses, _ = self._compute_group_breakdown(
            pred_imf=pred_imf,
            target_imf=target_imf,
            debug_ctx=debug_ctx,
            time_mask=time_mask,
            group_specs=group_specs,
        )
        aux_weight = float(group_aux_weight)
        aux_total = aux_weight * group_total
        total = main_total + aux_total

        losses = dict(main_losses)
        losses.update(prefixed_losses)
        losses["main_total"] = main_total
        losses["group_total"] = group_total
        losses["group_aux"] = aux_total
        losses["total"] = total
        return total, losses

    def _call_single(
        self,
        pred_imf: torch.Tensor,
        target_imf: torch.Tensor,
        debug_ctx: Optional[Dict] = None,
        time_mask: Optional[torch.Tensor] = None,
        group_specs: Optional[Sequence[Dict[str, Any]]] = None,
        current_group_name: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Total loss and its terms.

        Args:
            pred_imf:   [B, C, T] IMF predicted by IMFExtractor (C = 3 * 63).
            target_imf: [B, C, T] offline MEMD IMF target.
            time_mask:  [B, T] valid-frame mask. True means the frame is real.
                        After padding a batch, this keeps padding out of the loss.
        """
        assert pred_imf.shape == target_imf.shape, "pred_imf / target_imf shape mismatch"

        if time_mask is not None:
            m = time_mask
            if m.dim() == 3 and m.shape[1] == 1:
                m = m[:, 0, :]
            if m.dtype != torch.bool:
                m = m > 0
            time_mask = m.to(device=pred_imf.device)
            assert time_mask.shape[0] == pred_imf.shape[0] and time_mask.shape[-1] == pred_imf.shape[-1]

        losses: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=pred_imf.device)

        # 1) L1 decomposition alignment.
        if time_mask is None:
            l_decomp = F.l1_loss(pred_imf, target_imf)
        else:
            # L1 on valid frames only, so padding does not count.
            eps = 1e-8
            pred_f = pred_imf.float()
            target_f = target_imf.float()
            mask_bc = time_mask[:, None, :].float()  # [B,1,T]
            denom = mask_bc.sum() * float(pred_f.shape[1]) + eps
            l_decomp = ((pred_f - target_f).abs() * mask_bc).sum() / denom
        losses["imf_decomp"] = l_decomp
        total = total + self.lambda_decomp * l_decomp

        # 2) EMD: mean and extrema density.
        l_emd_mean = self.compute_emd_mean_loss(pred_imf, target_imf, time_mask=time_mask)
        l_emd_ext = self.compute_emd_extremum_loss(pred_imf, target_imf, time_mask=time_mask)
        losses["imf_emd_mean"] = l_emd_mean
        losses["imf_emd_extremum"] = l_emd_ext

        # Normalize the weights.
        s_emd = max(self.beta_emd_mean + self.beta_emd_ext, 1e-6)
        w_mean = self.beta_emd_mean / s_emd
        w_ext = self.beta_emd_ext / s_emd
        l_emd_total = self.lambda_emd * (w_mean * l_emd_mean + w_ext * l_emd_ext)
        losses["imf_emd"] = l_emd_total
        total = total + l_emd_total

        # 3) HHT: amplitude and frequency.
        if self.lambda_ht > 0.0 and (self.alpha_ht_amp + self.alpha_ht_freq) > 0.0:
            amp_loss, freq_loss = self.compute_hht_losses(
                pred_imf,
                target_imf,
                debug_ctx=debug_ctx,
                time_mask=time_mask,
            )
            losses["imf_ht_amp"] = amp_loss
            losses["imf_ht_freq"] = freq_loss

            s_ht = max(self.alpha_ht_amp + self.alpha_ht_freq, 1e-6)
            w_amp = self.alpha_ht_amp / s_ht
            w_freq = self.alpha_ht_freq / s_ht
            l_ht_total = self.lambda_ht * (w_amp * amp_loss + w_freq * freq_loss)
            losses["imf_ht"] = l_ht_total
            total = total + l_ht_total
        else:
            losses["imf_ht_amp"] = torch.tensor(0.0, device=pred_imf.device)
            losses["imf_ht_freq"] = torch.tensor(0.0, device=pred_imf.device)
            losses["imf_ht"] = torch.tensor(0.0, device=pred_imf.device)

        aux_pred, aux_target = self._select_temporal_aux_tensors(
            pred_imf=pred_imf,
            target_imf=target_imf,
            group_specs=group_specs,
            current_group_name=current_group_name,
        )
        if aux_pred is not None and aux_target is not None:
            if self.lambda_temporal_delta > 0.0:
                l_temporal_delta = self.compute_temporal_derivative_loss(
                    aux_pred,
                    aux_target,
                    order=1,
                    time_mask=time_mask,
                )
            else:
                l_temporal_delta = torch.tensor(0.0, device=pred_imf.device)

            if self.lambda_temporal_accel > 0.0:
                l_temporal_accel = self.compute_temporal_derivative_loss(
                    aux_pred,
                    aux_target,
                    order=2,
                    time_mask=time_mask,
                )
            else:
                l_temporal_accel = torch.tensor(0.0, device=pred_imf.device)

            if self.lambda_ht_phase > 0.0:
                l_ht_phase = self.compute_hht_phase_loss(
                    aux_pred,
                    aux_target,
                    debug_ctx=debug_ctx,
                    time_mask=time_mask,
                )
            else:
                l_ht_phase = torch.tensor(0.0, device=pred_imf.device)

            l_temporal_total = (
                self.lambda_temporal_delta * l_temporal_delta
                + self.lambda_temporal_accel * l_temporal_accel
                + self.lambda_ht_phase * l_ht_phase
            )
        else:
            l_temporal_delta = torch.tensor(0.0, device=pred_imf.device)
            l_temporal_accel = torch.tensor(0.0, device=pred_imf.device)
            l_ht_phase = torch.tensor(0.0, device=pred_imf.device)
            l_temporal_total = torch.tensor(0.0, device=pred_imf.device)

        losses["imf_temporal_delta"] = l_temporal_delta
        losses["imf_temporal_accel"] = l_temporal_accel
        losses["imf_ht_phase"] = l_ht_phase
        losses["imf_temporal_aux"] = l_temporal_total
        total = total + l_temporal_total

        def_pred = self._select_definition_unsup_tensor(
            pred_imf=pred_imf,
            group_specs=group_specs,
            current_group_name=current_group_name,
        )
        if def_pred is not None:
            if self.lambda_definition_env_mean > 0.0:
                l_def_env_mean = self.compute_definition_envelope_mean_loss(
                    def_pred,
                    time_mask=time_mask,
                )
            else:
                l_def_env_mean = torch.tensor(0.0, device=pred_imf.device)

            if self.lambda_definition_zero_ext > 0.0:
                l_def_zero_ext = self.compute_definition_zero_ext_gap_loss(
                    def_pred,
                    time_mask=time_mask,
                )
            else:
                l_def_zero_ext = torch.tensor(0.0, device=pred_imf.device)

            if self.lambda_definition_dc_mean > 0.0:
                l_def_dc_mean = self.compute_definition_dc_mean_loss(
                    def_pred,
                    time_mask=time_mask,
                )
            else:
                l_def_dc_mean = torch.tensor(0.0, device=pred_imf.device)

            l_definition_total = (
                self.lambda_definition_env_mean * l_def_env_mean
                + self.lambda_definition_zero_ext * l_def_zero_ext
                + self.lambda_definition_dc_mean * l_def_dc_mean
            )
        else:
            l_def_env_mean = torch.tensor(0.0, device=pred_imf.device)
            l_def_zero_ext = torch.tensor(0.0, device=pred_imf.device)
            l_def_dc_mean = torch.tensor(0.0, device=pred_imf.device)
            l_definition_total = torch.tensor(0.0, device=pred_imf.device)

        losses["imf_def_env_mean"] = l_def_env_mean
        losses["imf_def_zero_ext"] = l_def_zero_ext
        losses["imf_def_dc_mean"] = l_def_dc_mean
        losses["imf_definition_unsup"] = l_definition_total
        total = total + l_definition_total

        losses["total"] = total
        return total, losses

    def _select_temporal_aux_tensors(
        self,
        pred_imf: torch.Tensor,
        target_imf: torch.Tensor,
        group_specs: Optional[Sequence[Dict[str, Any]]] = None,
        current_group_name: Optional[str] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.temporal_aux_enabled:
            return None, None

        if current_group_name is not None:
            if self.temporal_aux_group_names and current_group_name not in self.temporal_aux_group_names:
                return None, None
            return pred_imf, target_imf

        if not self.temporal_aux_group_names:
            return pred_imf, target_imf
        if not group_specs:
            return None, None

        pred_chunks = []
        target_chunks = []
        for group in group_specs:
            name = str(group.get("name", "")).strip()
            if name not in self.temporal_aux_group_names:
                continue
            start = int(group["start"])
            end = int(group["end"])
            if end <= start:
                continue
            pred_chunks.append(pred_imf[:, start:end, :])
            target_chunks.append(target_imf[:, start:end, :])

        if not pred_chunks:
            return None, None
        return torch.cat(pred_chunks, dim=1), torch.cat(target_chunks, dim=1)

    def _select_definition_unsup_tensor(
        self,
        pred_imf: torch.Tensor,
        group_specs: Optional[Sequence[Dict[str, Any]]] = None,
        current_group_name: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        if not self.definition_unsup_enabled:
            return None

        if current_group_name is not None:
            if self.definition_unsup_group_names and current_group_name not in self.definition_unsup_group_names:
                return None
            return pred_imf

        if not self.definition_unsup_group_names:
            return pred_imf
        if not group_specs:
            return None

        pred_chunks = []
        for group in group_specs:
            name = str(group.get("name", "")).strip()
            if name not in self.definition_unsup_group_names:
                continue
            start = int(group["start"])
            end = int(group["end"])
            if end <= start:
                continue
            pred_chunks.append(pred_imf[:, start:end, :])

        if not pred_chunks:
            return None
        return torch.cat(pred_chunks, dim=1)

    @staticmethod
    def _normalize_time_mask(
        time_mask: Optional[torch.Tensor],
        *,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if time_mask is None:
            return None
        m = time_mask
        if m.dim() == 3 and m.shape[1] == 1:
            m = m[:, 0, :]
        if m.dtype != torch.bool:
            m = m > 0
        return m.to(device=device)

    @classmethod
    def _prepare_signal_and_mask(
        cls,
        tensor: torch.Tensor,
        time_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        tensor_f = tensor.float()
        mask = cls._normalize_time_mask(time_mask, device=tensor_f.device)
        if mask is None:
            batch_size = int(tensor_f.shape[0])
            seq_len = int(tensor_f.shape[-1])
            lengths = torch.full((batch_size,), float(seq_len), device=tensor_f.device, dtype=tensor_f.dtype)
            return tensor_f, None, None, lengths

        batch_size, channels, seq_len = tensor_f.shape
        lengths = mask.sum(dim=-1).clamp(min=1)
        last_idx = (lengths - 1).view(batch_size, 1, 1).expand(batch_size, channels, 1)
        last_val = tensor_f.gather(dim=-1, index=last_idx)
        mask_bc = mask[:, None, :]
        filled = torch.where(mask_bc, tensor_f, last_val.expand(batch_size, channels, seq_len))
        return filled, mask_bc.float(), lengths.float().unsqueeze(1), lengths.float()

    def _definition_rms_scale(
        self,
        pred_imf: torch.Tensor,
        time_mask: Optional[torch.Tensor] = None,
        target_imf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        eps = 1e-8
        if (not self.definition_unsup_normalize_by_rms) or self.definition_unsup_rms_source == "none":
            return torch.ones(pred_imf.shape[:2], device=pred_imf.device, dtype=pred_imf.dtype)

        if self.definition_unsup_rms_source == "target" and target_imf is not None:
            ref = target_imf.float()
        else:
            ref = pred_imf.float()
        mask = self._normalize_time_mask(time_mask, device=ref.device)
        if mask is None:
            return torch.sqrt(torch.mean(ref**2, dim=-1) + eps)

        mask_bc = mask[:, None, :].float()
        denom_t = mask.sum(dim=-1).clamp(min=1).float().unsqueeze(1)
        return torch.sqrt((ref**2 * mask_bc).sum(dim=-1) / denom_t + eps)

    # -------------------------------------------------------------- EMD helpers
    def compute_emd_mean_loss(
        self,
        pred_imf: torch.Tensor,
        target_imf: torch.Tensor,
        time_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """IMF mean loss: match channel means on an RMS scale."""
        # fp16 autocast underflows here (eps=1e-8 becomes 0) and blows up low-energy
        # channels into Inf/NaN. Force float32.
        eps = 1e-8
        pred_f = pred_imf.float()
        target_f = target_imf.float()
        # Use the target channel RMS over time as the scale. The mask ignores padding.
        if time_mask is None:
            target_rms = torch.sqrt(torch.mean(target_f**2, dim=-1) + eps)  # [B, C]
            denom_t = None
            mask_bc = None
        else:
            m = time_mask
            if m.dtype != torch.bool:
                m = m > 0
            m = m.to(device=target_f.device)
            mask_bc = m[:, None, :].float()  # [B,1,T]
            denom_t = m.sum(dim=-1).clamp(min=1).float().unsqueeze(1)  # [B,1]
            target_rms = torch.sqrt((target_f**2 * mask_bc).sum(dim=-1) / denom_t + eps)  # [B,C]
        target_rms_exp = target_rms.unsqueeze(-1)  # [B, C, 1]

        pred_norm = pred_f / (target_rms_exp + eps)
        target_norm = target_f / (target_rms_exp + eps)
        if time_mask is None:
            pred_mean = torch.mean(pred_norm, dim=-1)
            target_mean = torch.mean(target_norm, dim=-1)
        else:
            pred_mean = (pred_norm * mask_bc).sum(dim=-1) / denom_t
            target_mean = (target_norm * mask_bc).sum(dim=-1) / denom_t
        return F.mse_loss(pred_mean, target_mean)

    def compute_emd_extremum_loss(
        self,
        pred_imf: torch.Tensor,
        target_imf: torch.Tensor,
        time_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """IMF extrema loss: match extrema density, counted per frame."""
        batch_size, total_channels, seq_len = pred_imf.shape

        if time_mask is None:
            pred_ext = self.count_extrema(pred_imf, dim=-1)
            target_ext = self.count_extrema(target_imf, dim=-1)
            T = float(seq_len)
            pred_density = pred_ext / max(T, 1.0)
            target_density = target_ext / max(T, 1.0)
            return F.l1_loss(pred_density, target_density)

        # With a mask, count extrema only on valid frames.
        m = time_mask
        if m.dtype != torch.bool:
            m = m > 0
        m = m.to(device=pred_imf.device)
        lengths = m.sum(dim=-1).clamp(min=3).float()  # [B], long enough for a sign change

        # diff: [B,C,T-1] ; sign_changes: [B,C,T-2]
        diff_p = torch.diff(pred_imf, dim=-1)
        diff_t = torch.diff(target_imf, dim=-1)
        sc_p = diff_p[..., 1:] * diff_p[..., :-1] < 0
        sc_t = diff_t[..., 1:] * diff_t[..., :-1] < 0

        # Mask sign changes. For a valid length L, valid indices are [0, L-3], so there are L-2 of them.
        sc_len = max(seq_len - 2, 1)
        idx = torch.arange(sc_len, device=pred_imf.device)[None, :]  # [1, T-2]
        sc_mask = idx < (lengths - 2).clamp(min=0).long().unsqueeze(1)  # [B, T-2]
        sc_mask = sc_mask[:, None, :]  # [B,1,T-2]

        pred_ext = (sc_p & sc_mask).sum(dim=-1).float()  # [B,C]
        target_ext = (sc_t & sc_mask).sum(dim=-1).float()  # [B,C]

        pred_density = pred_ext / lengths.unsqueeze(1)
        target_density = target_ext / lengths.unsqueeze(1)
        return F.l1_loss(pred_density, target_density)

    @staticmethod
    def count_extrema(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Count extrema along one dimension."""
        if dim < 0:
            dim = tensor.dim() + dim
        diff = torch.diff(tensor, dim=dim)
        sign_changes = diff[..., 1:] * diff[..., :-1] < 0
        extrema_count = sign_changes.sum(dim=dim).float()
        return extrema_count

    @staticmethod
    def count_zero_crossings(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Count zero crossings along one dimension."""
        if dim < 0:
            dim = tensor.dim() + dim
        prod = tensor[..., 1:] * tensor[..., :-1]
        zero_cross = prod < 0
        zero_cross_count = zero_cross.sum(dim=dim).float()
        return zero_cross_count

    def compute_definition_envelope_mean_loss(
        self,
        pred_imf: torch.Tensor,
        time_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Approximate the upper and lower envelopes with max-pooling and pull their local mean toward 0."""
        eps = 1e-8
        pred_f, mask_bc, _denom_t, _lengths = self._prepare_signal_and_mask(pred_imf, time_mask=time_mask)
        kernel = max(int(self.definition_env_pool_kernel), 1)
        pad = kernel // 2
        upper = F.max_pool1d(pred_f, kernel_size=kernel, stride=1, padding=pad)
        lower = -F.max_pool1d(-pred_f, kernel_size=kernel, stride=1, padding=pad)
        local_mean = 0.5 * (upper + lower)
        scale = self._definition_rms_scale(pred_f, time_mask=time_mask).unsqueeze(-1)
        local_mean = local_mean / (scale + eps)
        if mask_bc is None:
            return local_mean.abs().mean()
        denom = mask_bc.sum() * float(local_mean.shape[1]) + eps
        return (local_mean.abs() * mask_bc).sum() / denom

    def compute_definition_zero_ext_gap_loss(
        self,
        pred_imf: torch.Tensor,
        time_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Soft form of the IMF condition |N_ext - N_zero| <= 1."""
        eps = 1e-8
        pred_f, _mask_bc, _denom_t, lengths = self._prepare_signal_and_mask(pred_imf, time_mask=time_mask)
        softness = max(float(self.definition_softsign_scale), 1e-3)

        sign_soft = torch.tanh(softness * pred_f)
        zero_prob = 0.5 * (1.0 - sign_soft[..., 1:] * sign_soft[..., :-1])
        zero_mask = self._shrink_time_mask(
            time_mask=time_mask,
            order=1,
            target_length=zero_prob.shape[-1],
            device=pred_f.device,
        )
        if zero_mask is not None:
            zero_prob = zero_prob * zero_mask[:, None, :].float()
        zero_count = zero_prob.sum(dim=-1)

        diff = torch.diff(pred_f, dim=-1)
        diff_sign_soft = torch.tanh(softness * diff)
        ext_prob = 0.5 * (1.0 - diff_sign_soft[..., 1:] * diff_sign_soft[..., :-1])
        ext_mask = self._shrink_time_mask(
            time_mask=time_mask,
            order=2,
            target_length=ext_prob.shape[-1],
            device=pred_f.device,
        )
        if ext_mask is not None:
            ext_prob = ext_prob * ext_mask[:, None, :].float()
        ext_count = ext_prob.sum(dim=-1)

        gap = F.relu((ext_count - zero_count).abs() - float(self.definition_zero_ext_margin))
        return (gap / lengths.unsqueeze(1).clamp(min=1.0)).mean() + 0.0 * eps

    def compute_definition_dc_mean_loss(
        self,
        pred_imf: torch.Tensor,
        time_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Global zero-mean guard. Separates a DC bias from a local envelope-mean error."""
        eps = 1e-8
        pred_f = pred_imf.float()
        scale = self._definition_rms_scale(pred_f, time_mask=time_mask).unsqueeze(-1)
        pred_n = pred_f / (scale + eps)
        mask = self._normalize_time_mask(time_mask, device=pred_f.device)
        if mask is None:
            pred_mean = pred_n.mean(dim=-1)
        else:
            mask_bc = mask[:, None, :].float()
            denom_t = mask.sum(dim=-1).clamp(min=1).float().unsqueeze(1)
            pred_mean = (pred_n * mask_bc).sum(dim=-1) / denom_t
        return pred_mean.abs().mean()

    # -------------------------------------------------------------- HHT helpers
    def compute_hht_losses(
        self,
        pred_imf: torch.Tensor,
        target_imf: torch.Tensor,
        debug_ctx: Optional[Dict] = None,
        time_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """HHT loss on normalized amplitude and frequency."""
        dt = self.dt

        base_ctx = debug_ctx if isinstance(debug_ctx, dict) else {}
        pred_ctx = dict(base_ctx)
        pred_ctx["side"] = "pred"
        target_ctx = dict(base_ctx)
        target_ctx["side"] = "target"

        # Padding is zeros, and that leaks into the FFT and the Hilbert transform.
        # Before ht(), repeat the last valid frame across the padded tail, then mask the loss.
        pred_f = pred_imf.float()
        target_f = target_imf.float()
        time_mask_bool = None
        mask_bc = None
        denom_t = None
        if time_mask is not None:
            m = time_mask
            if m.dtype != torch.bool:
                m = m > 0
            time_mask_bool = m.to(device=pred_f.device)
            if time_mask_bool.dim() == 3 and time_mask_bool.shape[1] == 1:
                time_mask_bool = time_mask_bool[:, 0, :]
            B, C, T = pred_f.shape
            lengths = time_mask_bool.sum(dim=-1).clamp(min=1)  # [B]
            last_idx = (lengths - 1).view(B, 1, 1).expand(B, C, 1)
            pred_last = pred_f.gather(dim=-1, index=last_idx)  # [B,C,1]
            target_last = target_f.gather(dim=-1, index=last_idx)
            mask_bc = time_mask_bool[:, None, :]  # [B,1,T] (bool)
            pred_f = torch.where(mask_bc, pred_f, pred_last.expand(B, C, T))
            target_f = torch.where(mask_bc, target_f, target_last.expand(B, C, T))

            # Kept for the masked statistics below.
            mask_bc = mask_bc.float()
            denom_t = lengths.float().unsqueeze(1)  # [B,1]

        # Hilbert transform: instantaneous amplitude and frequency.
        pred_amp, pred_freq = ht(pred_f, dt, debug_ctx=pred_ctx)
        target_amp, target_freq = ht(target_f, dt, debug_ctx=target_ctx)

        eps = 1e-8
        # Amplitude, normalized by the target RMS over time.
        if time_mask is None:
            target_rms = torch.sqrt(torch.mean(target_amp**2, dim=-1) + eps)  # [B, C]
        else:
            target_rms = torch.sqrt((target_amp**2 * mask_bc).sum(dim=-1) / denom_t + eps)  # [B,C]
        target_rms_exp = target_rms.unsqueeze(-1)  # [B, C, 1]
        pred_amp_n = pred_amp / (target_rms_exp + eps)
        target_amp_n = target_amp / (target_rms_exp + eps)
        if time_mask is None:
            amp_loss_norm = F.mse_loss(pred_amp_n, target_amp_n)
        else:
            denom = mask_bc.sum() * float(pred_amp_n.shape[1]) + eps
            amp_loss_norm = (((pred_amp_n - target_amp_n) ** 2) * mask_bc).sum() / denom

        # Frequency, normalized by Nyquist and clipped to [0, f_nyq].
        f_nyq = max(self.f_nyquist, 1e-6)
        pred_f_n = torch.clamp(pred_freq, min=0.0, max=f_nyq) / f_nyq
        target_f_n = torch.clamp(target_freq, min=0.0, max=f_nyq) / f_nyq
        if time_mask is None:
            freq_loss_norm = F.mse_loss(pred_f_n, target_f_n)
        else:
            denom = mask_bc.sum() * float(pred_f_n.shape[1]) + eps
            freq_loss_norm = (((pred_f_n - target_f_n) ** 2) * mask_bc).sum() / denom

        return amp_loss_norm, freq_loss_norm

    @staticmethod
    def _shrink_time_mask(
        time_mask: Optional[torch.Tensor],
        order: int,
        target_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if time_mask is None:
            return None

        m = time_mask
        if m.dim() == 3 and m.shape[1] == 1:
            m = m[:, 0, :]
        if m.dtype != torch.bool:
            m = m > 0
        m = m.to(device=device)

        lengths = m.sum(dim=-1).clamp(min=order + 1)
        valid_lengths = (lengths - order).clamp(min=1)
        idx = torch.arange(target_length, device=device)[None, :]
        return idx < valid_lengths.unsqueeze(1)

    def compute_temporal_derivative_loss(
        self,
        pred_imf: torch.Tensor,
        target_imf: torch.Tensor,
        order: int = 1,
        time_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")

        pred_f = pred_imf.float()
        target_f = target_imf.float()
        for _ in range(int(order)):
            pred_f = torch.diff(pred_f, dim=-1)
            target_f = torch.diff(target_f, dim=-1)

        eps = 1e-8
        diff_mask = self._shrink_time_mask(
            time_mask=time_mask,
            order=int(order),
            target_length=pred_f.shape[-1],
            device=pred_f.device,
        )

        if self.temporal_normalize_by_target_rms:
            if diff_mask is None:
                target_rms = torch.sqrt(torch.mean(target_f**2, dim=-1) + eps)
            else:
                mask_bc = diff_mask[:, None, :].float()
                denom_t = diff_mask.sum(dim=-1).clamp(min=1).float().unsqueeze(1)
                target_rms = torch.sqrt((target_f**2 * mask_bc).sum(dim=-1) / denom_t + eps)
            scale = target_rms.unsqueeze(-1) + eps
            pred_f = pred_f / scale
            target_f = target_f / scale

        if diff_mask is None:
            return F.l1_loss(pred_f, target_f)

        mask_bc = diff_mask[:, None, :].float()
        denom = mask_bc.sum() * float(pred_f.shape[1]) + eps
        return ((pred_f - target_f).abs() * mask_bc).sum() / denom

    def compute_hht_phase_loss(
        self,
        pred_imf: torch.Tensor,
        target_imf: torch.Tensor,
        debug_ctx: Optional[Dict] = None,
        time_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dt = self.dt
        base_ctx = debug_ctx if isinstance(debug_ctx, dict) else {}
        pred_ctx = dict(base_ctx)
        pred_ctx["side"] = "pred"
        target_ctx = dict(base_ctx)
        target_ctx["side"] = "target"

        pred_f = pred_imf.float()
        target_f = target_imf.float()
        mask_bc = None
        denom_t = None
        if time_mask is not None:
            m = time_mask
            if m.dtype != torch.bool:
                m = m > 0
            m = m.to(device=pred_f.device)
            if m.dim() == 3 and m.shape[1] == 1:
                m = m[:, 0, :]
            lengths = m.sum(dim=-1).clamp(min=1)
            last_idx = (lengths - 1).view(pred_f.shape[0], 1, 1).expand(pred_f.shape[0], pred_f.shape[1], 1)
            pred_last = pred_f.gather(dim=-1, index=last_idx)
            target_last = target_f.gather(dim=-1, index=last_idx)
            mask_bc = m[:, None, :]
            pred_f = torch.where(mask_bc, pred_f, pred_last.expand_as(pred_f))
            target_f = torch.where(mask_bc, target_f, target_last.expand_as(target_f))
            mask_bc = mask_bc.float()
            denom_t = lengths.float().unsqueeze(1)

        pred_amp, _, pred_phase_real, pred_phase_imag = ht(
            pred_f,
            dt,
            debug_ctx=pred_ctx,
            return_analytic=True,
        )
        target_amp, _, target_phase_real, target_phase_imag = ht(
            target_f,
            dt,
            debug_ctx=target_ctx,
            return_analytic=True,
        )

        phase_dot = pred_phase_real * target_phase_real + pred_phase_imag * target_phase_imag
        phase_dist = 1.0 - torch.clamp(phase_dot, min=-1.0, max=1.0)

        eps = 1e-8
        if self.phase_amp_weighted:
            if time_mask is None:
                target_rms = torch.sqrt(torch.mean(target_amp**2, dim=-1) + eps)
            else:
                target_rms = torch.sqrt((target_amp**2 * mask_bc).sum(dim=-1) / denom_t + eps)
            phase_weight = target_amp / (target_rms.unsqueeze(-1) + eps)
            phase_weight = torch.clamp(phase_weight, min=0.0, max=max(self.phase_weight_clip, 0.0))
        else:
            phase_weight = torch.ones_like(phase_dist)

        if time_mask is None:
            denom = phase_weight.sum() + eps
            return (phase_dist * phase_weight).sum() / denom

        weighted_mask = phase_weight * mask_bc
        denom = weighted_mask.sum() + eps
        return (phase_dist * weighted_mask).sum() / denom


