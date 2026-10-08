from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .imf_extractor import ConvBlock, IMFExtractor, IMF_edit
except ImportError:  # standalone extractor entry points
    from models.imf_extractor import ConvBlock, IMFExtractor, IMF_edit


class Body63ResidualRefineIMFExtractor(IMFExtractor):
    """Grouped-head IMF extractor with an extra residual refiner on body63 outputs."""

    def __init__(
        self,
        *args,
        body_refine_group_name: str = "body63",
        body_refine_hidden: int = 96,
        body_refine_kernel_size: int = 5,
        body_refine_norm: str = "bn",
        body_refine_dropout: float = 0.1,
        body_refine_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not self.shared_backbone or self.group_decoders is None:
            raise ValueError("Body63ResidualRefineIMFExtractor requires shared_backbone/grouped-head setup.")

        self.body_refine_group_name = str(body_refine_group_name)
        body_group = next(
            (group for group in self.output_groups if str(group.get("name", "")) == self.body_refine_group_name),
            None,
        )
        if body_group is None:
            raise ValueError(f"Group not found for residual refine: {self.body_refine_group_name}")

        self.body_refine_scale = float(body_refine_scale)
        self.body_refine_dof = int(body_group["dof"])
        hidden = max(int(body_refine_hidden), 1)
        dropout = None if body_refine_dropout is None else float(body_refine_dropout)
        kernel_size = int(body_refine_kernel_size)
        norm = str(body_refine_norm)
        self.body_refiners = nn.ModuleList()
        for _ in range(self.imf_count):
            refiner = nn.Sequential(
                ConvBlock(
                    kernel_size=kernel_size,
                    in_channels=self.body_refine_dof,
                    out_channels=hidden,
                    norm=norm,
                    dropout=dropout,
                    acti="lrelu",
                ),
                ConvBlock(
                    kernel_size=kernel_size,
                    in_channels=hidden,
                    out_channels=self.body_refine_dof,
                    norm=norm,
                    dropout=dropout,
                    acti="none",
                ),
            )
            last_conv = refiner[-1].model[1]
            if isinstance(last_conv, nn.Conv1d):
                nn.init.zeros_(last_conv.weight)
                if last_conv.bias is not None:
                    nn.init.zeros_(last_conv.bias)
            self.body_refiners.append(refiner)

    def forward(self, x, lengths=None, label=None, gamma=1.0):
        batch_size, _seq_len, _ = x.shape

        x_trans = x.transpose(1, 2)
        xpe = self.pos_encoder(x_trans)
        if self.pe_proj is not None:
            xpe = self.pe_proj(xpe)

        imf_outputs = []
        global_features = []

        shared_latent = self.shared_encoder(xpe)
        if self.transformer_enabled:
            shared_latent_trans = shared_latent.permute(2, 0, 1)
            shared_latent_trans = self.shared_transformer(shared_latent_trans)
            shared_latent = shared_latent_trans.permute(1, 2, 0)

        for i in range(self.imf_count):
            latent = self.band_adapters[i](shared_latent)
            global_feat = self.attention_poolings[i](latent)
            global_features.append(global_feat)

            group_outputs = []
            for group in self.output_groups:
                name = str(group["name"])
                group_out = self.group_decoders[i][name](latent)
                if name == self.body_refine_group_name:
                    residual = self.body_refiners[i](group_out)
                    group_out = group_out + self.body_refine_scale * residual
                group_outputs.append(group_out)
            imf_outputs.append(torch.cat(group_outputs, dim=1))

        imfs_flat = torch.cat(imf_outputs, dim=1)
        if label is not None:
            imfs_flat = IMF_edit(imfs_flat, label, gamma)
        imfs_reshaped = imfs_flat.reshape(batch_size, self.imf_count, self.imfdof, -1)
        global_features = torch.stack(global_features, dim=1)
        return imfs_reshaped, global_features, imfs_flat
