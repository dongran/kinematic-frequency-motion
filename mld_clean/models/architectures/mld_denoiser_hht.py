from __future__ import annotations

import torch
import torch.nn as nn

from mld_clean.models.architectures.mld_denoiser import MldDenoiser


class MldDenoiserHHT(MldDenoiser):
    def __init__(self, *args, motion_encoded_dim: int = 512, **kwargs):
        super().__init__(*args, motion_encoded_dim=motion_encoded_dim, **kwargs)
        self.emb_proj_freq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(motion_encoded_dim, self.latent_dim),
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

        if len(encoder_hidden_states) == 4:
            content_hidden, style_hidden, freq_hidden, trans_cond = encoder_hidden_states
        else:
            content_hidden, style_hidden, trans_cond = encoder_hidden_states
            freq_hidden = None

        style_emb = style_hidden.permute(1, 0, 2)
        content_emb = content_hidden.permute(1, 0, 2)

        content_emb_latent = self.IN(content_emb.permute(1, 2, 0)).permute(2, 0, 1)
        content_emb_latent = content_emb_latent + time_emb
        content_emb_latent = self.pe_content(content_emb_latent)
        content_emb_latent = self.seqTransEncoder(content_emb_latent).permute(1, 0, 2)
        content_emb_latent = self.linear(
            content_emb_latent.reshape(content_emb_latent.shape[0], -1)
        ).reshape(content_emb_latent.shape[0], 6, 256)
        content_emb_latent = content_emb_latent.permute(1, 0, 2)
        xseq = torch.cat((content_emb_latent, sample), axis=0)

        style_emb_latent = self.emb_proj_st(style_emb)
        style_emb_latent = time_emb + style_emb_latent
        if freq_hidden is not None:
            freq_emb = freq_hidden.permute(1, 0, 2)
            freq_emb_latent = self.emb_proj_freq(freq_emb)
            style_emb_latent = style_emb_latent + freq_emb_latent
        # Only remove the sequence-length dim; keep batch dim even when batch_size == 1.
        style_emb_latent = style_emb_latent.squeeze(0)

        trans_emb = self.trans_Encoder(trans_cond, lengths)
        trans_emb = trans_emb + time_emb
        # Only remove the sequence-length dim; keep batch dim even when batch_size == 1.
        trans_emb = trans_emb.squeeze(0)

        xseq = self.query_pos(xseq).permute(1, 0, 2)
        for block in self.blocks:
            xseq = block(xseq, style_emb_latent, trans_emb)
        sample = xseq[:, content_emb_latent.shape[0] :, :]
        return (sample,)
