from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

try:
    from .imf_extractor import IMFExtractor
    from .imf_extractor_body63_refine import Body63ResidualRefineIMFExtractor
    from .imf_group_utils import normalize_output_groups
except ImportError:  # pragma: no cover - keep standalone training entrypoints working
    from models.imf_extractor import IMFExtractor
    from models.imf_extractor_body63_refine import Body63ResidualRefineIMFExtractor
    from models.imf_group_utils import normalize_output_groups


def _as_int_list(x: Optional[Sequence[Any]]) -> Optional[list[int]]:
    if x is None:
        return None
    return [int(v) for v in x]


def build_imf_extractor(model_cfg: Dict[str, Any]) -> IMFExtractor:
    """Build IMFExtractor from a config dict.

    Backward compatible: if new fields (variant/cnn/transformer) are absent,
    this returns the same baseline CNN as before.
    """

    variant = str(model_cfg.get("variant", "cnn")).lower()
    nfeats = int(model_cfg["nfeats"])
    latent_dim = int(model_cfg.get("latent_dim", 512))
    imf_count = int(model_cfg.get("imf_count", 3))
    imf_dof = int(model_cfg.get("imf_dof", 63))
    frame_rate = float(model_cfg.get("frame_rate", 20.0))
    pe_dim = int(model_cfg.get("pe_dim", 4))
    pe_mode = str(model_cfg.get("pe_mode", "time_concat"))
    pe_log_space = bool(model_cfg.get("pe_log_space", False))
    attention_dim = int(model_cfg.get("attention_dim", 128))

    cnn_cfg = model_cfg.get("cnn", {}) or {}
    transformer_cfg = model_cfg.get("transformer", {}) or {}

    # CNN knobs (optional)
    encoder_channels = _as_int_list(cnn_cfg.get("encoder_channels", None))
    decoder_channels = _as_int_list(cnn_cfg.get("decoder_channels", None))
    cnn_kernel_size = int(cnn_cfg.get("kernel_size", 5))
    cnn_norm = str(cnn_cfg.get("norm", "bn"))
    cnn_dropout = cnn_cfg.get("dropout", 0.2)
    cnn_dropout = None if cnn_dropout is None else float(cnn_dropout)
    shared_backbone = bool(
        model_cfg.get(
            "shared_backbone",
            variant in ("cnn_grouped", "cnn_grouped_heads", "grouped", "grouped_heads"),
        )
    )
    band_adapter_resblocks = int(cnn_cfg.get("band_adapter_resblocks", 2 if shared_backbone else 0))
    output_groups = normalize_output_groups(model_cfg.get("output_groups"), imf_dof)
    body_refine_cfg = model_cfg.get("body_refine", {}) or {}

    # Transformer knobs (optional)
    # variant hint: if user sets variant=cnn_transformer but forgets transformer.enabled,
    # we treat it as enabled by default.
    transformer_enabled = bool(transformer_cfg.get("enabled", False))
    if (variant in ("cnn_transformer", "transformer", "cnn+transformer")) and ("enabled" not in transformer_cfg):
        transformer_enabled = True
    transformer_num_layers = int(transformer_cfg.get("num_layers", 1))
    transformer_heads = int(transformer_cfg.get("num_heads", model_cfg.get("transformer_heads", 4)))
    transformer_ff_dim = int(transformer_cfg.get("ff_dim", model_cfg.get("transformer_ff_dim", 512)))
    transformer_dropout = float(transformer_cfg.get("dropout", 0.1))

    if variant in (
        "cnn_grouped_heads_body63_refine",
        "grouped_heads_body63_refine",
        "body63_residual_refine",
        "body63_refine",
    ):
        return Body63ResidualRefineIMFExtractor(
            nfeats=nfeats,
            latent_dim=latent_dim,
            imf_count=imf_count,
            imf_dof=imf_dof,
            pe_dim=pe_dim,
            pe_mode=pe_mode,
            pe_log_space=pe_log_space,
            attention_dim=attention_dim,
            frame_rate=frame_rate,
            cnn_encoder_channels=encoder_channels,
            cnn_decoder_channels=decoder_channels,
            cnn_kernel_size=cnn_kernel_size,
            cnn_norm=cnn_norm,
            cnn_dropout=cnn_dropout,
            shared_backbone=True,
            output_groups=output_groups,
            band_adapter_resblocks=band_adapter_resblocks,
            transformer_enabled=transformer_enabled,
            transformer_num_layers=transformer_num_layers,
            transformer_heads=transformer_heads,
            transformer_ff_dim=transformer_ff_dim,
            transformer_dropout=transformer_dropout,
            body_refine_group_name=str(body_refine_cfg.get("group_name", "body63")),
            body_refine_hidden=int(body_refine_cfg.get("hidden_channels", 96)),
            body_refine_kernel_size=int(body_refine_cfg.get("kernel_size", cnn_kernel_size)),
            body_refine_norm=str(body_refine_cfg.get("norm", cnn_norm)),
            body_refine_dropout=body_refine_cfg.get("dropout", cnn_dropout),
            body_refine_scale=float(body_refine_cfg.get("scale", 1.0)),
        )

    # Variant presets (only applied when channels are not explicitly provided)
    if encoder_channels is None and decoder_channels is None:
        if variant == "cnn_deeper":
            encoder_channels = [nfeats, 128, 256, latent_dim, latent_dim]
            decoder_channels = [latent_dim, latent_dim, 128, 96, imf_dof]
        elif variant == "cnn_wider":
            encoder_channels = [nfeats, 192, 384, latent_dim]
            # decoder keeps baseline unless user overrides
            decoder_channels = None

    return IMFExtractor(
        nfeats=nfeats,
        latent_dim=latent_dim,
        imf_count=imf_count,
        imf_dof=imf_dof,
        pe_dim=pe_dim,
        pe_mode=pe_mode,
        pe_log_space=pe_log_space,
        attention_dim=attention_dim,
        frame_rate=frame_rate,
        cnn_encoder_channels=encoder_channels,
        cnn_decoder_channels=decoder_channels,
        cnn_kernel_size=cnn_kernel_size,
        cnn_norm=cnn_norm,
        cnn_dropout=cnn_dropout,
        shared_backbone=shared_backbone,
        output_groups=output_groups,
        band_adapter_resblocks=band_adapter_resblocks,
        transformer_enabled=transformer_enabled,
        transformer_num_layers=transformer_num_layers,
        transformer_heads=transformer_heads,
        transformer_ff_dim=transformer_ff_dim,
        transformer_dropout=transformer_dropout,
    )

