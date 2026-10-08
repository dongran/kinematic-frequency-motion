from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mld_clean.models.architectures.mld_denoiser import MldDenoiser, modulate
from timm.models.vision_transformer import Attention, Mlp


class DualStyleDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation_coarse = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )
        self.adaLN_modulation_fine = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )
        self.adaLN_modulation_timing = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )
        self.adaLN_modulation_trans = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation_timing[1].weight)
        nn.init.zeros_(self.adaLN_modulation_timing[1].bias)

    def forward(self, x, coarse_style, trajectory, fine_style=None, timing_style=None):
        shift_c, scale_c, gate_c = self.adaLN_modulation_coarse(coarse_style).chunk(3, dim=1)
        if fine_style is not None:
            shift_f, scale_f, gate_f = self.adaLN_modulation_fine(fine_style).chunk(3, dim=1)
        else:
            shift_f = torch.zeros_like(shift_c)
            scale_f = torch.zeros_like(scale_c)
            gate_f = torch.zeros_like(gate_c)
        if timing_style is not None:
            shift_t, scale_t, gate_t = self.adaLN_modulation_timing(timing_style).chunk(3, dim=1)
        else:
            shift_t = torch.zeros_like(shift_c)
            scale_t = torch.zeros_like(scale_c)
            gate_t = torch.zeros_like(gate_c)
        shift = shift_c + shift_f + shift_t
        scale = scale_c + scale_f + scale_t
        gate = gate_c + gate_f + gate_t
        x = x + gate.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift, scale))

        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation_trans(trajectory).chunk(3, dim=1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class MldDenoiserDualStyle(MldDenoiser):
    def __init__(
        self,
        *args,
        motion_encoded_dim: int = 512,
        fine_inject_start_layer: int = 6,
        num_heads: int = 4,
        **kwargs,
    ):
        super().__init__(*args, motion_encoded_dim=motion_encoded_dim, num_heads=num_heads, **kwargs)
        self.emb_proj_fine = nn.Sequential(
            nn.ReLU(),
            nn.Linear(motion_encoded_dim, self.latent_dim),
        )
        self.emb_proj_traj_fine = nn.Sequential(
            nn.ReLU(),
            nn.Linear(motion_encoded_dim, self.latent_dim),
        )
        content_traj_linear = nn.Linear(motion_encoded_dim, self.latent_dim)
        contact_timing_linear = nn.Linear(motion_encoded_dim, self.latent_dim)
        self.emb_proj_traj_content = nn.Sequential(
            nn.SiLU(),
            content_traj_linear,
        )
        self.emb_proj_contact_timing = nn.Sequential(
            nn.SiLU(),
            contact_timing_linear,
        )
        timing_branch_linear = nn.Linear(motion_encoded_dim, self.latent_dim)
        self.emb_proj_timing_branch = nn.Sequential(
            nn.SiLU(),
            timing_branch_linear,
        )
        self.emb_proj_content_imf_remover = nn.Linear(motion_encoded_dim, self.latent_dim)
        nn.init.zeros_(content_traj_linear.weight)
        nn.init.zeros_(content_traj_linear.bias)
        nn.init.zeros_(contact_timing_linear.weight)
        nn.init.zeros_(contact_timing_linear.bias)
        nn.init.zeros_(timing_branch_linear.weight)
        nn.init.zeros_(timing_branch_linear.bias)
        nn.init.zeros_(self.emb_proj_content_imf_remover.weight)
        nn.init.zeros_(self.emb_proj_content_imf_remover.bias)
        self.fine_inject_start_layer = int(fine_inject_start_layer)
        num_layers = len(self.blocks)
        self.blocks = nn.ModuleList(
            [
                DualStyleDiTBlock(
                    hidden_size=self.latent_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        lengths=None,
        **kwargs,
    ):
        sample = sample.permute(1, 0, 2)
        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        time_emb = self.time_embedding(time_emb).unsqueeze(0)

        if len(encoder_hidden_states) not in (4, 5, 7):
            raise ValueError(
                "Dual-style denoiser expects [content, coarse_style, fine_style, trajectory] "
                "or [content, coarse_style, fine_style, trajectory, traj_fine] conditions, "
                "or the extended 7-slot condition list with content-traj/content-remover branches."
            )
        if len(encoder_hidden_states) == 7:
            (
                content_hidden,
                coarse_style_hidden,
                fine_style_hidden,
                trans_cond,
                traj_fine_hidden,
                content_traj_hidden,
                content_imf_remover_hidden,
            ) = encoder_hidden_states
        else:
            content_hidden, coarse_style_hidden, fine_style_hidden, trans_cond = encoder_hidden_states[:4]
            traj_fine_hidden = encoder_hidden_states[4] if len(encoder_hidden_states) == 5 else None
            content_traj_hidden = None
            content_imf_remover_hidden = None
        coarse_style = coarse_style_hidden.permute(1, 0, 2)
        content_emb = content_hidden.permute(1, 0, 2)
        fine_style = fine_style_hidden.permute(1, 0, 2)
        if traj_fine_hidden is not None:
            traj_fine = traj_fine_hidden.permute(1, 0, 2)
        else:
            traj_fine = None
        if content_traj_hidden is not None:
            content_traj = content_traj_hidden.permute(1, 0, 2)
        else:
            content_traj = None
        if content_imf_remover_hidden is not None:
            content_imf_remover = content_imf_remover_hidden.permute(1, 0, 2)
        else:
            content_imf_remover = None
        contact_timing_hidden = kwargs.get("contact_timing_hidden")
        if contact_timing_hidden is not None:
            if contact_timing_hidden.dim() == 2:
                contact_timing_hidden = contact_timing_hidden.unsqueeze(1)
            contact_timing = contact_timing_hidden.permute(1, 0, 2)
        else:
            contact_timing = None
        timing_hidden = kwargs.get("timing_hidden")
        if timing_hidden is not None:
            if timing_hidden.dim() == 2:
                timing_hidden = timing_hidden.unsqueeze(1)
            timing_style = timing_hidden.permute(1, 0, 2)
        else:
            timing_style = None

        content_emb_latent = self.IN(content_emb.permute(1, 2, 0)).permute(2, 0, 1)
        if content_imf_remover is not None:
            remover_logits = self.emb_proj_content_imf_remover(content_imf_remover).squeeze(0)
            suppress = F.softplus(remover_logits) - math.log(2.0)
            content_emb_latent = content_emb_latent * torch.exp(-suppress.unsqueeze(0))
        content_emb_latent = content_emb_latent + time_emb
        content_emb_latent = self.pe_content(content_emb_latent)
        content_emb_latent = self.seqTransEncoder(content_emb_latent).permute(1, 0, 2)
        content_emb_latent = self.linear(
            content_emb_latent.reshape(content_emb_latent.shape[0], -1)
        ).reshape(content_emb_latent.shape[0], 6, 256)
        content_emb_latent = content_emb_latent.permute(1, 0, 2)
        xseq = torch.cat((content_emb_latent, sample), axis=0)

        coarse_style_latent = self.emb_proj_st(coarse_style)
        # Only remove the sequence-length dim; keep batch dim even when batch_size == 1.
        coarse_style_latent = (time_emb + coarse_style_latent).squeeze(0)

        fine_style_latent = self.emb_proj_fine(fine_style)
        # Only remove the sequence-length dim; keep batch dim even when batch_size == 1.
        fine_style_latent = (time_emb + fine_style_latent).squeeze(0)

        trans_emb = self.trans_Encoder(trans_cond, lengths)
        if content_traj is not None:
            trans_emb = trans_emb + self.emb_proj_traj_content(content_traj)
        if traj_fine is not None:
            trans_emb = trans_emb + self.emb_proj_traj_fine(traj_fine)
        if contact_timing is not None:
            trans_emb = trans_emb + self.emb_proj_contact_timing(contact_timing)
        # Only remove the sequence-length dim; keep batch dim even when batch_size == 1.
        trans_emb = (trans_emb + time_emb).squeeze(0)
        timing_latent = None
        if timing_style is not None:
            timing_latent = (self.emb_proj_timing_branch(timing_style) + time_emb).squeeze(0)

        xseq = self.query_pos(xseq).permute(1, 0, 2)
        for block_idx, block in enumerate(self.blocks):
            fine_cond = fine_style_latent if block_idx >= self.fine_inject_start_layer else None
            xseq = block(
                xseq,
                coarse_style_latent,
                trans_emb,
                fine_style=fine_cond,
                timing_style=timing_latent,
            )
        sample = xseq[:, content_emb_latent.shape[0] :, :]
        return (sample,)
