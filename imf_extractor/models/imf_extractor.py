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
        # - baseline 历史行为：对形状 [B,C,T] 的输入，在 dim=-1（时间维）拼接，T -> T*(1+2*K)
        # - Transformer 友好变体：在 dim=1（通道维）拼接，C -> C*(1+2*K)，并保持 T 不变
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
    """编辑IMF特征

    Args:
        imfs: [batch_size, imf_count * channels, seq_len] - IMF特征
        label: 指定要编辑的IMF索引列表
        gamma: 编辑系数
    """
    batch_size, total_channels, seq_len = imfs.shape
    imf_count = 3  # 假设有3个IMF
    channels_per_imf = total_channels // imf_count

    edited_imfs = imfs.clone()

    # 针对每个样本进行编辑
    for i in range(batch_size):
        if hasattr(label, "__iter__"):
            curr_label = label[i] if i < len(label) else -1
        else:
            curr_label = label

        # 根据标签决定编辑哪个IMF
        if curr_label == 0:
            edited_imfs[i, :channels_per_imf, :] *= gamma
        elif curr_label == 1:
            edited_imfs[i, channels_per_imf : 2 * channels_per_imf, :] *= gamma
        elif curr_label == 2:
            edited_imfs[i, 2 * channels_per_imf :, :] *= gamma
        else:  # 编辑所有IMF
            edited_imfs[i] *= gamma

    return edited_imfs


def diff(x, dim=-1, same_size=False):
    """计算差分"""
    if same_size:
        return F.pad(x[..., 1:] - x[..., :-1], (1, 0))
    else:
        return x[..., 1:] - x[..., :-1]


def unwrap(phi, dim=-1):
    """相位展开。

    使用标准 atan2 相位展开约定：
    - 跳变阈值 = π
    - 取模周期 = 2π

    历史 bug：曾把 `asin(1)` 误当作 π（实际是 π/2），导致阈值/周期各缩小一倍。
    """
    pi = torch.pi
    twopi = 2.0 * pi
    dphi = diff(phi, same_size=False)
    dphi_m = ((dphi + pi) % twopi) - pi
    dphi_m[(dphi_m == -pi) & (dphi > 0)] = pi
    phi_adj = dphi_m - dphi
    phi_adj[dphi.abs() < pi] = 0

    # 沿指定维度累加相位调整
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
    """Hilbert变换，计算瞬时幅值和频率（带安全 atan2 防护）

    Args:
        sig: [batch_size, channels, seq_len] - 信号
        dt: 时间步长
        debug_ctx: 可选调试上下文（用于日志）
        min_radius: 在近零半径处使用的最小半径，用于避免 atan2 的数值不稳定
        return_analytic: 若为 True，同时返回解析信号的单位实部/虚部，
            便于在 loss 中做相位对齐约束。
    """
    assert len(sig.shape) == 3, "Input tensor must be 3-dimensional"
    batch_size, channels, length = sig.shape
    device = sig.device

    # 重要：在 AMP(fp16/bf16) 下，cuFFT 对非 2^k 长度的 half 精度有硬限制，
    # 会直接报错（例如 length=196）。这里强制用 float32 做 FFT/Hilbert，
    # 既保证数值稳定，也保证 mixed-precision 训练不被 ht() 卡住。
    sig_f = sig.float()

    # 计算频率
    fsig = torch.fft.fftfreq(length, dt, device=device)
    jsgn = (torch.sign(fsig) * 1.0j).to(torch.complex64)

    # Hilbert变换
    sig_fft = torch.fft.fft(sig_f, dim=-1)  # complex64
    sig_h_c = torch.fft.ifft(-jsgn * sig_fft, dim=-1)  # complex64
    sig_h = torch.real(sig_h_c)  # hilbert transform (real)

    # 计算瞬时幅值
    # 注意：amp = sqrt(x^2 + y^2) 在 (x,y)=(0,0) 处梯度不定义，会在反向出现 0/0 -> NaN。
    # 这里加入一个极小的最小半径 mr，使得 amp = sqrt(x^2 + y^2 + mr^2)，避免 NaN 梯度。
    mr = max(float(min_radius), 0.0)
    mr2 = mr * mr
    radius_sq = sig_f**2 + sig_h**2
    amp_inst = torch.sqrt(radius_sq + mr2)

    # 计算相位（安全 atan2）：对近零半径的位置进行安全替换
    y = sig_h
    x = sig_f
    try:
        # 用 radius_sq 做 near-zero 判断（不要用 amp_inst，因为它已加入 mr，不会再小于 mr）
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

    # 相位展开
    phase_inst_unwrapped = unwrap(phase_inst)

    # 计算瞬时频率
    freq_inst = torch.gradient(phase_inst_unwrapped, dim=-1)[0] / dt / (2.0 * torch.pi)

    if return_analytic:
        phase_real = x / amp_inst
        phase_imag = y / amp_inst
        return amp_inst, freq_inst, phase_real, phase_imag

    return amp_inst, freq_inst


class IMFExtractor(nn.Module):
    """3-band IMF 提取网络。

    当前实现与原 MCM-LDM 工程中的 `mld_imf_extractor.IMFExtractor` 结构保持一致，
    方便在两边直接加载/共享权重。
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
        self.nfeats = nfeats  # 输入维度（例如 HumanML3D 的 263 维）
        self.imf_count = imf_count  # IMF 数量（默认为 3）
        self.imfdof = int(imf_dof)  # 每个 IMF 的自由度（例如 63 / 69）
        self.shared_backbone = bool(shared_backbone)
        self.output_groups = normalize_output_groups(output_groups, self.imfdof)
        self.group_names = [str(g["name"]) for g in self.output_groups]
        self.band_adapter_resblocks = max(int(band_adapter_resblocks), 0)

        # 保存帧率与时间步长（供 HHT 使用）
        self.frame_rate = float(frame_rate) if frame_rate is not None else 20.0
        self.dt = 1.0 / self.frame_rate

        # latent_dim：默认 512（与原工程保持一致），但允许通过 config 进行变体实验
        self.latent_dim = int(latent_dim)

        # Transformer 开关（默认关闭，保持 baseline 行为不变）
        self.transformer_enabled = bool(transformer_enabled)
        self.transformer_num_layers = int(transformer_num_layers)
        self.transformer_dropout = float(transformer_dropout)

        # Fourier/PE（NeRF-style）编码
        # - baseline: pe_mode=time_concat（在时间维拼接，不改变通道数）
        # - transformer 变体推荐：pe_mode=channel_concat（在通道维拼接 + 1x1 投影回 nfeats）
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
                # 1x1 conv：把 [B, C*(1+2K), T] 投影回 [B, C, T]
                self.pe_proj = nn.Conv1d(in_ch, int(self.nfeats), kernel_size=1, bias=True)
            else:
                # 默认：沿时间维拼接（历史行为）
                self.pos_encoder = PositionalEncoder(
                    self.nfeats,
                    self.pe_dim,
                    log_space=self.pe_log_space,
                    concat_dim=-1,
                )
        else:
            # pe_dim=0：不做 Fourier 扩展（仅 identity）
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

        # CNN 通道配置（缺省为 baseline）
        enc_channels = (
            list(map(int, cnn_encoder_channels))
            if cnn_encoder_channels is not None
            else [int(self.nfeats), 128, 256, int(self.latent_dim)]
        )
        # 若用户提供 encoder_channels，则以其最后一项作为 latent_dim（避免冲突）
        if enc_channels:
            self.latent_dim = int(enc_channels[-1])

        dec_channels = (
            list(map(int, cnn_decoder_channels))
            if cnn_decoder_channels is not None
            else [int(self.latent_dim), 128, 96, int(self.imfdof)]
        )

        # 规范化首尾维度（尽量容错，避免因小配置错误直接 crash）
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
            # 新实验：共享时序主干，只在 IMF band / body-root-trans 头上分流。
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
                # 编码器路径 - 最终通道数为 512
                self.imf_encoders.append(
                    build_conv_stack(
                        kernel_size=ksz,
                        channels=enc_channels,
                        norm=norm,
                        dropout=dropout,
                    )
                )

                # 可选的 Transformer 编码器（当前默认关闭）
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

                # 注意力池化 - 输入/输出均为 512
                self.attention_poolings.append(
                    FeatureWiseAttentionPooling(
                        feature_dim=self.latent_dim,
                        attention_dim=attention_dim,
                    )
                )

                # 解码器路径：latent_dim -> 128 -> 96 -> imfdof
                self.imf_decoders.append(
                    build_conv_stack(
                        kernel_size=ksz,
                        channels=dec_channels,
                        norm=norm,
                        dropout=dropout,
                    )
                )

    def forward(self, x, lengths=None, label=None, gamma=1.0):
        """前向计算。

        Args:
            x: [batch_size, seq_len, nfeats] - 输入动作序列（已标准化）
            lengths: 可选的序列长度（当前实现未使用，占位）
            label: 可选的 IMF 编辑标签
            gamma: IMF 编辑缩放系数

        Returns:
            imfs_reshaped: [B, imf_count, imfdof, T]
            global_features: [B, imf_count, latent_dim]
            imfs_flat: [B, imf_count * imfdof, T]
        """
        batch_size, seq_len, _ = x.shape

        # 应用位置编码：先转为 [B, C, T]
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
                # 编码到 latent 空间
                latent = self.imf_encoders[i](xpe)  # [B, latent_dim, T']

                # 可选 Transformer：默认关闭（保持 baseline 行为不变）
                if self.transformer_enabled:
                    latent_trans = latent.permute(2, 0, 1)  # [T', B, latent_dim]
                    latent_trans = self.imf_transformers[i](latent_trans)
                    latent = latent_trans.permute(1, 2, 0)  # [B, latent_dim, T']

                # 全局特征（每个 IMF 一个向量）
                global_feat = self.attention_poolings[i](latent)  # [B, latent_dim]
                global_features.append(global_feat)

                # 解码到 IMF 空间，输出时间长度回到 T
                imf = self.imf_decoders[i](latent)  # [B, imfdof, T]
                imf_outputs.append(imf)

        # 堆叠 IMF 输出 [B, imf_count * imfdof, T]
        imfs_flat = torch.cat(imf_outputs, dim=1)

        # 可选编辑
        if label is not None:
            imfs_flat = IMF_edit(imfs_flat, label, gamma)

        # 重塑为 [B, imf_count, imfdof, T]
        imfs_reshaped = imfs_flat.reshape(batch_size, self.imf_count, self.imfdof, -1)

        # 堆叠全局特征 [B, imf_count, latent_dim]
        global_features = torch.stack(global_features, dim=1)

        return imfs_reshaped, global_features, imfs_flat

    def extract_imf_features(self, x, lengths=None):
        """仅提取 IMF 特征，用于 IMF 分解监督。

        Args:
            x: [B, T, nfeats]
        Returns:
            imfs_flat: [B, imf_count * imfdof, T]
        """
        _, _, imfs_flat = self.forward(x, lengths)
        return imfs_flat

    def compute_hht_features(self, imfs):
        """计算 HHT 特征（瞬时幅值和频率）。"""
        amp, freq = ht(imfs, self.dt)
        return amp, freq


