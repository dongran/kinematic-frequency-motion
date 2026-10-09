import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .imf_group_utils import normalize_output_groups
except ImportError:  # standalone extractor entry points
    from models.imf_group_utils import normalize_output_groups


class PositionalEncoder(nn.Module):
    """Sine-cosine positional encoder for input points."""

    def __init__(
        self,
        d_input: int,
        n_freqs: int,
        log_space: bool = False,
        *,
        concat_dim: int = -1,
    ):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        # NOTE:
        # - Baseline: for input [B, C, T], concatenate on dim=-1 (time), so T -> T*(1+2*K).
        # - Transformer variant: concatenate on dim=1 (channels), so C -> C*(1+2*K), and T stays the same.
        self.concat_dim = int(concat_dim)
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.0 ** torch.linspace(0.0, self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** (self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x) -> torch.Tensor:
        """Apply positional encoding to input."""
        return torch.concat([fn(x) for fn in self.embed_fns], dim=self.concat_dim)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dim_feedforward: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=float(dropout),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=int(num_layers),
        )

    def forward(self, x):
        return self.transformer_encoder(x)


class FeatureWiseAttentionPooling(nn.Module):
    def __init__(self, feature_dim, attention_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(feature_dim, attention_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(attention_dim, feature_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: [batch_size, channels, seq_len]
        attn_weights = self.attention(x)  # [batch_size, channels, seq_len]
        weighted_x = x * attn_weights  # [batch_size, channels, seq_len]
        pooled_x = torch.mean(weighted_x, dim=2)  # [batch_size, channels]
        return pooled_x


class ConvBlock(nn.Module):
    def __init__(
        self,
        kernel_size,
        in_channels,
        out_channels,
        stride=1,
        norm="bn",
        dropout=None,
        acti="lrelu",
    ):
        super().__init__()
        layers = []

        # Padding
        pad_l = (kernel_size - 1) // 2
        pad_r = kernel_size - 1 - pad_l
        layers.append(nn.ReflectionPad1d((pad_l, pad_r)))

        # Convolution
        layers.append(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
            )
        )

        # Normalization
        if norm == "bn":
            layers.append(nn.BatchNorm1d(out_channels))
        elif norm == "in":
            layers.append(nn.InstanceNorm1d(out_channels, affine=True))

        # Dropout
        if dropout is not None:
            layers.append(nn.Dropout(p=dropout))

        # Activation
        if acti == "lrelu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif acti == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif acti == "none":
            pass

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        kernel_size,
        channels,
        stride=1,
        pad_type="reflect",
        norm="bn",
        acti="lrelu",
    ):
        super().__init__()
        self.conv1 = ConvBlock(
            kernel_size,
            channels,
            channels,
            stride=stride,
            norm=norm,
            acti=acti,
        )
        self.conv2 = ConvBlock(
            kernel_size,
            channels,
            channels,
            stride=stride,
            norm=norm,
            acti="none",
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out


def build_conv_stack(kernel_size, channels, norm="bn", dropout=None):
    layers = []
    for j in range(len(channels) - 1):
        layers.append(
            ConvBlock(
                kernel_size=kernel_size,
                in_channels=channels[j],
                out_channels=channels[j + 1],
                stride=1,
                norm=norm,
                dropout=dropout,
                acti="lrelu",
            )
        )
    return nn.Sequential(*layers) if layers else nn.Identity()


def build_res_stack(kernel_size, channels, num_blocks, norm="bn"):
    count = max(int(num_blocks), 0)
    if count == 0:
        return nn.Identity()
    return nn.Sequential(
        *[
            ResBlock(
                kernel_size=kernel_size,
                channels=channels,
                stride=1,
                norm=norm,
                acti="lrelu",
            )
            for _ in range(count)
        ]
    )


def IMF_edit(imfs, label, gamma=1.0):
    """Edit IMF features.

    Args:
        imfs: [batch_size, imf_count * channels, seq_len]
        label: which IMF index to edit
        gamma: edit scale
    """
    batch_size, total_channels, seq_len = imfs.shape
    imf_count = 3  # Three IMFs.
    channels_per_imf = total_channels // imf_count

    edited_imfs = imfs.clone()

    # Edit one sample at a time.
    for i in range(batch_size):
        if hasattr(label, "__iter__"):
            curr_label = label[i] if i < len(label) else -1
        else:
            curr_label = label

        # The label selects which IMF to scale.
        if curr_label == 0:
            edited_imfs[i, :channels_per_imf, :] *= gamma
        elif curr_label == 1:
            edited_imfs[i, channels_per_imf : 2 * channels_per_imf, :] *= gamma
        elif curr_label == 2:
            edited_imfs[i, 2 * channels_per_imf :, :] *= gamma
        else:  # Scale every IMF.
            edited_imfs[i] *= gamma

    return edited_imfs


def diff(x, dim=-1, same_size=False):
    """Finite difference."""
    if same_size:
        return F.pad(x[..., 1:] - x[..., :-1], (1, 0))
    else:
        return x[..., 1:] - x[..., :-1]


def unwrap(phi, dim=-1):
    """Unwrap phase.

    Standard atan2 unwrap:
    - jump threshold = pi
    - modulo period = 2 pi

    An old bug treated `asin(1)` as pi. It is pi/2, so the threshold and the
    period were both half of what they should be.
    """
    pi = torch.pi
    twopi = 2.0 * pi
    dphi = diff(phi, same_size=False)
    dphi_m = ((dphi + pi) % twopi) - pi
    dphi_m[(dphi_m == -pi) & (dphi > 0)] = pi
    phi_adj = dphi_m - dphi
    phi_adj[dphi.abs() < pi] = 0

    # Accumulate the phase correction along the given dimension.
    if dim < 0:
        dim = phi.dim() + dim

    result = torch.cat(
        [
            phi[..., :1],
            phi[..., 1:] + torch.cumsum(phi_adj, dim=dim),
        ],
        dim=dim,
    )

    return result


def ht(sig, dt, debug_ctx=None, min_radius: float = 1e-6, return_analytic: bool = False):
    """Hilbert transform: instantaneous amplitude and frequency, with a safe atan2.

    Args:
        sig: [batch_size, channels, seq_len]
        dt: time step
        debug_ctx: optional debug context for logs
        min_radius: floor used near zero so atan2 stays stable
        return_analytic: when True, also return the unit real and imaginary
            parts of the analytic signal for a phase-alignment loss.
    """
    assert len(sig.shape) == 3, "Input tensor must be 3-dimensional"
    batch_size, channels, length = sig.shape
    device = sig.device

    # AMP fp16/bf16: cuFFT rejects half precision unless the length is a power of two
    # (length 196 fails). Run the FFT and the Hilbert transform in float32 so
    # mixed precision does not stop in ht().
    sig_f = sig.float()

    # Frequency axis.
    fsig = torch.fft.fftfreq(length, dt, device=device)
    jsgn = (torch.sign(fsig) * 1.0j).to(torch.complex64)

    # Hilbert transform.
    sig_fft = torch.fft.fft(sig_f, dim=-1)  # complex64
    sig_h_c = torch.fft.ifft(-jsgn * sig_fft, dim=-1)  # complex64
    sig_h = torch.real(sig_h_c)  # hilbert transform (real)

    # Instantaneous amplitude.
    # amp = sqrt(x^2 + y^2) has an undefined gradient at (0, 0) and backprop becomes NaN.
    # A tiny minimum radius mr gives amp = sqrt(x^2 + y^2 + mr^2) and keeps the gradient finite.
    mr = max(float(min_radius), 0.0)
    mr2 = mr * mr
    radius_sq = sig_f**2 + sig_h**2
    amp_inst = torch.sqrt(radius_sq + mr2)

    # Phase with a safe atan2. Replace samples whose radius is near zero.
    y = sig_h
    x = sig_f
    try:
        # Test near-zero with radius_sq. amp_inst already includes mr, so it is never below mr.
        zero_mask = radius_sq < mr2
        num_zero = int(zero_mask.sum().item())
        ctx = debug_ctx if isinstance(debug_ctx, dict) else {}
        ht_loss_weight = float(ctx.get("ht_loss_weight", 1.0))
        # Optional logging: only emit NaNGuard warnings when explicitly enabled
        log_warning = bool(ctx.get("ht_log_warning", False))
        if log_warning and num_zero > 0 and ht_loss_weight > 0.0:
            from logging import getLogger

            logger = getLogger()
            logger.warning(
                "[NaNGuard][ht] near-zero radius before atan2: count=%d, shape=%s, ctx=%s",
                num_zero,
                tuple(y.shape),
                {k: ctx.get(k) for k in ["loss", "split", "epoch", "batch_idx", "ht_loss_weight"]},
            )
        # For near-zero radius, clamp x and y to avoid gradient explosion in atan2.
        x = torch.where(zero_mask, torch.full_like(x, mr), x)
        y = torch.where(zero_mask, torch.zeros_like(y), y)
    except Exception:
        pass
    phase_inst = torch.atan2(y, x)

    # Unwrap the phase.
    phase_inst_unwrapped = unwrap(phase_inst)

    # Instantaneous frequency.
    freq_inst = torch.gradient(phase_inst_unwrapped, dim=-1)[0] / dt / (2.0 * torch.pi)

    if return_analytic:
        phase_real = x / amp_inst
        phase_imag = y / amp_inst
        return amp_inst, freq_inst, phase_real, phase_imag

    return amp_inst, freq_inst


class IMFExtractor(nn.Module):
    """Three-band IMF extractor.

    The layout matches `mld_imf_extractor.IMFExtractor` in the original MCM-LDM
    code, so the same checkpoint can be loaded on both sides.
    """

    def __init__(
        self,
        nfeats,
        latent_dim=512,
        imf_count=3,
        imf_dof=63,
        pe_dim=4,
        pe_mode: str = "time_concat",
        pe_log_space: bool = False,
        attention_dim=128,
        transformer_heads=4,
        transformer_ff_dim=512,
        # CNN knobs (optional; keep baseline defaults if None)
        cnn_encoder_channels=None,
        cnn_decoder_channels=None,
        cnn_kernel_size: int = 5,
        cnn_norm: str = "bn",
        cnn_dropout: float = 0.2,
        shared_backbone: bool = False,
        output_groups=None,
        band_adapter_resblocks: int = 0,
        # Transformer knobs (optional; default disabled)
        transformer_enabled: bool = False,
        transformer_num_layers: int = 1,
        transformer_dropout: float = 0.1,
        frame_rate: float = 20.0,
    ):
        super().__init__()
        self.nfeats = nfeats  # Input size, for example 263 for HumanML3D.
        self.imf_count = imf_count  # Number of IMFs. Default is 3.
        self.imfdof = int(imf_dof)  # Degrees of freedom in each IMF, for example 63 or 69.
        self.shared_backbone = bool(shared_backbone)
        self.output_groups = normalize_output_groups(output_groups, self.imfdof)
        self.group_names = [str(g["name"]) for g in self.output_groups]
        self.band_adapter_resblocks = max(int(band_adapter_resblocks), 0)

        # Frame rate and time step, used by the HHT.
        self.frame_rate = float(frame_rate) if frame_rate is not None else 20.0
        self.dt = 1.0 / self.frame_rate

        # latent_dim defaults to 512, matching the original code. A config may override it.
        self.latent_dim = int(latent_dim)

        # Transformer switch. Off by default, so the baseline behavior stays the same.
        self.transformer_enabled = bool(transformer_enabled)
        self.transformer_num_layers = int(transformer_num_layers)
        self.transformer_dropout = float(transformer_dropout)

        # Fourier positional encoding, in the NeRF style.
        # - baseline: pe_mode=time_concat concatenates on time and does not change the channel count.
        # - For the transformer variant, pe_mode=channel_concat concatenates on channels and a 1x1 projection returns nfeats.
        self.pe_dim = int(pe_dim)
        self.pe_mode = str(pe_mode or "time_concat").lower()
        self.pe_log_space = bool(pe_log_space)
        self.pe_proj = None
        if self.pe_dim > 0:
            if self.pe_mode in ("channel", "channel_concat", "channels"):
                self.pos_encoder = PositionalEncoder(
                    self.nfeats,
                    self.pe_dim,
                    log_space=self.pe_log_space,
                    concat_dim=1,
                )
                in_ch = int(self.nfeats) * (1 + 2 * int(self.pe_dim))
                # 1x1 conv: project [B, C*(1+2K), T] back to [B, C, T].
                self.pe_proj = nn.Conv1d(in_ch, int(self.nfeats), kernel_size=1, bias=True)
            else:
                # Default: concatenate along time. This is the historical behavior.
                self.pos_encoder = PositionalEncoder(
                    self.nfeats,
                    self.pe_dim,
                    log_space=self.pe_log_space,
                    concat_dim=-1,
                )
        else:
            # pe_dim=0: no Fourier expansion, identity only.
            self.pos_encoder = PositionalEncoder(
                self.nfeats,
                0,
                log_space=self.pe_log_space,
                concat_dim=-1,
            )

        self.shared_encoder = None
        self.shared_transformer = None
        self.band_adapters = None
        self.group_decoders = None
        self.imf_encoders = nn.ModuleList()
        self.imf_transformers = nn.ModuleList()
        self.imf_decoders = nn.ModuleList()
        self.attention_poolings = nn.ModuleList()

        # CNN channel layout. The default is the baseline.
        enc_channels = (
            list(map(int, cnn_encoder_channels))
            if cnn_encoder_channels is not None
            else [int(self.nfeats), 128, 256, int(self.latent_dim)]
        )
        # If encoder_channels is set, its last width becomes latent_dim so the two cannot disagree.
        if enc_channels:
            self.latent_dim = int(enc_channels[-1])

        dec_channels = (
            list(map(int, cnn_decoder_channels))
            if cnn_decoder_channels is not None
            else [int(self.latent_dim), 128, 96, int(self.imfdof)]
        )

        # Normalize the first and last channel widths. A small config mistake should not crash.
        if len(enc_channels) >= 1:
            enc_channels[0] = int(self.nfeats)
        if len(dec_channels) >= 1:
            dec_channels[0] = int(self.latent_dim)
        if len(dec_channels) >= 1:
            dec_channels[-1] = int(self.imfdof)

        ksz = int(cnn_kernel_size)
        norm = str(cnn_norm)
        dropout = None if cnn_dropout is None else float(cnn_dropout)

        if self.shared_backbone:
            # Shared temporal backbone. The split is only at the IMF-band and body/root/translation heads.
            self.shared_encoder = build_conv_stack(
                kernel_size=ksz,
                channels=enc_channels,
                norm=norm,
                dropout=dropout,
            )
            if self.transformer_enabled:
                self.shared_transformer = TransformerEncoder(
                    input_dim=self.latent_dim,
                    num_heads=transformer_heads,
                    dim_feedforward=transformer_ff_dim,
                    num_layers=self.transformer_num_layers,
                    dropout=self.transformer_dropout,
                )
            else:
                self.shared_transformer = nn.Identity()
            self.band_adapters = nn.ModuleList()
            self.group_decoders = nn.ModuleList()
            for _ in range(imf_count):
                self.band_adapters.append(
                    build_res_stack(
                        kernel_size=ksz,
                        channels=self.latent_dim,
                        num_blocks=self.band_adapter_resblocks,
                        norm=norm,
                    )
                )
                self.attention_poolings.append(
                    FeatureWiseAttentionPooling(
                        feature_dim=self.latent_dim,
                        attention_dim=attention_dim,
                    )
                )
                group_decoder_dict = nn.ModuleDict()
                for group in self.output_groups:
                    group_dec_channels = list(dec_channels)
                    group_dec_channels[-1] = int(group["dof"])
                    group_decoder_dict[str(group["name"])] = build_conv_stack(
                        kernel_size=ksz,
                        channels=group_dec_channels,
                        norm=norm,
                        dropout=dropout,
                    )
                self.group_decoders.append(group_decoder_dict)
        else:
            for _ in range(imf_count):
                # Encoder path. The last channel count is 512.
                self.imf_encoders.append(
                    build_conv_stack(
                        kernel_size=ksz,
                        channels=enc_channels,
                        norm=norm,
                        dropout=dropout,
                    )
                )

                # Optional transformer encoder. Off by default.
                self.imf_transformers.append(
                    TransformerEncoder(
                        input_dim=self.latent_dim,
                        num_heads=transformer_heads,
                        dim_feedforward=transformer_ff_dim,
                        num_layers=self.transformer_num_layers,
                        dropout=self.transformer_dropout,
                    )
                    if self.transformer_enabled
                    else nn.Identity()
                )

                # Attention pooling. Input and output are both 512.
                self.attention_poolings.append(
                    FeatureWiseAttentionPooling(
                        feature_dim=self.latent_dim,
                        attention_dim=attention_dim,
                    )
                )

                # Decoder path: latent_dim -> 128 -> 96 -> imfdof.
                self.imf_decoders.append(
                    build_conv_stack(
                        kernel_size=ksz,
                        channels=dec_channels,
                        norm=norm,
                        dropout=dropout,
                    )
                )

    def forward(self, x, lengths=None, label=None, gamma=1.0):
        """Forward pass.

        Args:
            x: [batch_size, seq_len, nfeats], a normalized motion.
            lengths: optional sequence lengths. Unused in this implementation.
            label: optional IMF edit index.
            gamma: optional IMF edit scale.

        Returns:
            imfs_reshaped: [B, imf_count, imfdof, T]
            global_features: [B, imf_count, latent_dim]
            imfs_flat: [B, imf_count * imfdof, T]
        """
        batch_size, seq_len, _ = x.shape

        # Apply positional encoding. First reshape to [B, C, T].
        x_trans = x.transpose(1, 2)  # [batch_size, nfeats, seq_len]
        xpe = self.pos_encoder(x_trans)
        if self.pe_proj is not None:
            xpe = self.pe_proj(xpe)

        imf_outputs = []
        global_features = []

        if self.shared_backbone:
            shared_latent = self.shared_encoder(xpe)
            if self.transformer_enabled:
                shared_latent_trans = shared_latent.permute(2, 0, 1)  # [T, B, latent_dim]
                shared_latent_trans = self.shared_transformer(shared_latent_trans)
                shared_latent = shared_latent_trans.permute(1, 2, 0)

            for i in range(self.imf_count):
                latent = self.band_adapters[i](shared_latent)
                global_feat = self.attention_poolings[i](latent)
                global_features.append(global_feat)

                group_outputs = []
                for group in self.output_groups:
                    group_outputs.append(self.group_decoders[i][str(group["name"])](latent))
                imf_outputs.append(torch.cat(group_outputs, dim=1))
        else:
            for i in range(self.imf_count):
                # Encode into the latent space.
                latent = self.imf_encoders[i](xpe)  # [B, latent_dim, T']

                # Optional transformer. Off by default, so the baseline stays the same.
                if self.transformer_enabled:
                    latent_trans = latent.permute(2, 0, 1)  # [T', B, latent_dim]
                    latent_trans = self.imf_transformers[i](latent_trans)
                    latent = latent_trans.permute(1, 2, 0)  # [B, latent_dim, T']

                # One global vector per IMF.
                global_feat = self.attention_poolings[i](latent)  # [B, latent_dim]
                global_features.append(global_feat)

                # Decode back to IMF space. The time length returns to T.
                imf = self.imf_decoders[i](latent)  # [B, imfdof, T]
                imf_outputs.append(imf)

        # Stack IMF outputs to [B, imf_count * imfdof, T].
        imfs_flat = torch.cat(imf_outputs, dim=1)

        # Optional edit.
        if label is not None:
            imfs_flat = IMF_edit(imfs_flat, label, gamma)

        # Reshape to [B, imf_count, imfdof, T].
        imfs_reshaped = imfs_flat.reshape(batch_size, self.imf_count, self.imfdof, -1)

        # Stack global features to [B, imf_count, latent_dim].
        global_features = torch.stack(global_features, dim=1)

        return imfs_reshaped, global_features, imfs_flat

    def extract_imf_features(self, x, lengths=None):
        """IMF features only, for the decomposition loss.

        Args:
            x: [B, T, nfeats]
        Returns:
            imfs_flat: [B, imf_count * imfdof, T]
        """
        _, _, imfs_flat = self.forward(x, lengths)
        return imfs_flat

    def compute_hht_features(self, imfs):
        """HHT features: instantaneous amplitude and frequency."""
        amp, freq = ht(imfs, self.dt)
        return amp, freq


