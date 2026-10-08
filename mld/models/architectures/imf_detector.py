from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _candidate_extractor_roots() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    candidates = []
    for name in ("imf_extractor",):
        candidate = repo_root / name
        if (candidate / "models" / "imf_factory.py").exists():
            candidates.append(candidate)
    return candidates


def _purge_module_prefix(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.pop(name, None)


def _resolve_imf_symbols():
    errors: list[str] = []

    try:
        factory_mod = importlib.import_module("imf_extractor.models.imf_factory")
        extractor_mod = importlib.import_module("imf_extractor.models.imf_extractor")
        return factory_mod.build_imf_extractor, extractor_mod.IMFExtractor, extractor_mod.ht
    except Exception as exc:  # pragma: no cover - exercised only in mixed repo layouts
        errors.append(f"package import failed: {exc}")
        _purge_module_prefix("imf_extractor")

    for extractor_root in _candidate_extractor_roots():
        root_text = str(extractor_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        importlib.invalidate_caches()
        _purge_module_prefix("models")
        try:
            factory_mod = importlib.import_module("models.imf_factory")
            extractor_mod = importlib.import_module("models.imf_extractor")
            return factory_mod.build_imf_extractor, extractor_mod.IMFExtractor, extractor_mod.ht
        except Exception as exc:  # pragma: no cover - exercised only when extractor checkout is broken
            errors.append(f"top-level import via {extractor_root.name} failed: {exc}")

    joined_errors = " | ".join(errors) if errors else "no extractor candidates found"
    raise ImportError(
        "Could not import IMF extractor implementation from the workspace. "
        f"Tried packaged import and repo-root fallback. Details: {joined_errors}"
    )


build_imf_extractor, IMFExtractor, compute_ht = _resolve_imf_symbols()


def _torch_load_compat(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _infer_repr_dim(nfeats: int, repr_name: str, slice_start=None, slice_end=None) -> int:
    repr_key = str(repr_name or "full").lower()
    if repr_key in {"full", "full263"}:
        return int(nfeats)
    if repr_key in {"pose63", "rot63"}:
        start = 3 if slice_start is None else int(slice_start)
        end = 66 if slice_end is None else int(slice_end)
        return end - start
    if slice_start is None or slice_end is None:
        raise ValueError(
            f"Custom decomposer repr '{repr_name}' requires both slice start and slice end."
        )
    return int(slice_end) - int(slice_start)


def _select_motion_repr(motion: torch.Tensor, repr_name: str, slice_start=None, slice_end=None):
    repr_key = str(repr_name or "full").lower()
    if repr_key in {"full", "full263"}:
        return motion
    if repr_key in {"pose63", "rot63"}:
        start = 3 if slice_start is None else int(slice_start)
        end = 66 if slice_end is None else int(slice_end)
        return motion[..., start:end]
    if slice_start is None or slice_end is None:
        raise ValueError(
            f"Custom decomposer repr '{repr_name}' requires both slice start and slice end."
        )
    start = int(slice_start)
    end = int(slice_end)
    if end <= start:
        raise ValueError(f"Invalid decomposer slice {start}:{end}")
    return motion[..., start:end]


def _band_rms_pool(bands: torch.Tensor, lengths) -> torch.Tensor:
    max_len = bands.shape[-1]
    time_index = torch.arange(max_len, device=bands.device).view(1, 1, 1, -1)
    valid_lengths = torch.as_tensor(lengths, device=bands.device).view(-1, 1, 1, 1)
    mask = (time_index < valid_lengths).float()
    denom = mask.sum(dim=-1).clamp_min(1.0)
    energy = (bands.float() ** 2) * mask
    return torch.sqrt(energy.sum(dim=-1) / denom + 1e-8)


class FixedBandDetectorBase(nn.Module):
    def __init__(
        self,
        *,
        nfeats: int = 263,
        frame_rate: float = 20.0,
        repr_name: str = "full",
        slice_start=None,
        slice_end=None,
    ):
        super().__init__()
        self.imf_count = 3
        self.frame_rate = float(frame_rate)
        self.dt = 1.0 / self.frame_rate
        self.repr_name = str(repr_name or "full")
        self.slice_start = slice_start
        self.slice_end = slice_end
        self.imfdof = _infer_repr_dim(nfeats, self.repr_name, slice_start, slice_end)

    def _select_motion(self, motion: torch.Tensor) -> torch.Tensor:
        return _select_motion_repr(
            motion,
            self.repr_name,
            slice_start=self.slice_start,
            slice_end=self.slice_end,
        )

    def _decompose(self, signal: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, motion, lengths):
        selected = self._select_motion(motion).float().transpose(1, 2)
        bands = self._decompose(selected)
        global_features = _band_rms_pool(bands, lengths)
        flat = bands.reshape(bands.shape[0], -1, bands.shape[-1])
        return bands, global_features, flat

    def extract_imf_features(self, motion, lengths):
        _, _, flat = self.forward(motion, lengths)
        return flat

    def compute_hht_features(self, imfs):
        return compute_ht(imfs, self.dt)


class FFTBandDetector(FixedBandDetectorBase):
    def __init__(self, *, band_edges_hz=None, **kwargs):
        super().__init__(**kwargs)
        edges = list(band_edges_hz or [1.0, 3.0, 8.0])
        if len(edges) != 3:
            raise ValueError(f"FFT band edges must have three values, got {edges}")
        self.band_edges_hz = [float(v) for v in edges]

    def _decompose(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.shape[-1] == 0:
            raise ValueError("Cannot decompose an empty motion sequence.")
        signal_f = signal.float()
        signal_fft = torch.fft.rfft(signal_f, dim=-1, norm="ortho")
        freqs = torch.fft.rfftfreq(
            signal_f.shape[-1],
            d=1.0 / self.frame_rate,
            device=signal_f.device,
        )
        low_max, mid_max, high_max = self.band_edges_hz
        nyquist = float(freqs[-1].item()) if freqs.numel() > 0 else 0.0
        high_cut = min(high_max, nyquist)
        masks = {
            "high": (freqs > mid_max) & (freqs <= high_cut),
            "mid": (freqs > low_max) & (freqs <= mid_max),
            "low": freqs <= low_max,
        }

        outputs = []
        for band_name in ("high", "mid", "low"):
            band_fft = torch.zeros_like(signal_fft)
            mask = masks[band_name]
            if mask.any():
                band_fft[..., mask] = signal_fft[..., mask]
            outputs.append(
                torch.fft.irfft(
                    band_fft,
                    n=signal_f.shape[-1],
                    dim=-1,
                    norm="ortho",
                )
            )
        return torch.stack(outputs, dim=1)


def _haar_forward_level(signal: torch.Tensor):
    if signal.shape[-1] % 2 != 0:
        raise ValueError(f"Haar forward expects even length, got {signal.shape[-1]}")
    scale = 1.0 / torch.sqrt(torch.tensor(2.0, device=signal.device, dtype=signal.dtype))
    even = signal[..., 0::2]
    odd = signal[..., 1::2]
    approx = (even + odd) * scale
    detail = (even - odd) * scale
    return approx, detail


def _haar_inverse_level(approx: torch.Tensor, detail: torch.Tensor):
    if approx.shape != detail.shape:
        raise ValueError(f"Haar inverse shape mismatch: {approx.shape} vs {detail.shape}")
    scale = 1.0 / torch.sqrt(torch.tensor(2.0, device=approx.device, dtype=approx.dtype))
    output = torch.empty(
        *approx.shape[:-1],
        approx.shape[-1] * 2,
        device=approx.device,
        dtype=approx.dtype,
    )
    output[..., 0::2] = (approx + detail) * scale
    output[..., 1::2] = (approx - detail) * scale
    return output


class DWTBandDetector(FixedBandDetectorBase):
    def __init__(self, *, levels: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.levels = int(levels)
        if self.levels != 2:
            raise ValueError(
                "DWTBandDetector currently supports exactly 2 levels so it maps cleanly to 3 bands."
            )

    def _decompose(self, signal: torch.Tensor) -> torch.Tensor:
        signal_f = signal.float()
        orig_len = signal_f.shape[-1]
        if orig_len == 0:
            raise ValueError("Cannot decompose an empty motion sequence.")
        pad_multiple = 2 ** self.levels
        pad_len = (-orig_len) % pad_multiple
        if pad_len > 0:
            pad_mode = "replicate" if orig_len > 1 else "constant"
            signal_f = F.pad(signal_f, (0, pad_len), mode=pad_mode)

        approx_l1, detail_l1 = _haar_forward_level(signal_f)
        approx_l2, detail_l2 = _haar_forward_level(approx_l1)

        zeros_l2 = torch.zeros_like(approx_l2)
        zeros_l1 = torch.zeros_like(approx_l1)

        high = _haar_inverse_level(zeros_l1, detail_l1)
        mid_l1 = _haar_inverse_level(zeros_l2, detail_l2)
        mid = _haar_inverse_level(mid_l1, torch.zeros_like(detail_l1))
        low_l1 = _haar_inverse_level(approx_l2, zeros_l2)
        low = _haar_inverse_level(low_l1, torch.zeros_like(detail_l1))

        return torch.stack(
            [
                high[..., :orig_len],
                mid[..., :orig_len],
                low[..., :orig_len],
            ],
            dim=1,
        )


def build_fixed_band_detector(
    *,
    decomposer_type: str,
    nfeats: int = 263,
    frame_rate: float = 20.0,
    repr_name: str = "full",
    slice_start=None,
    slice_end=None,
    fft_band_edges_hz=None,
    dwt_levels: int = 2,
):
    kwargs = {
        "nfeats": nfeats,
        "frame_rate": frame_rate,
        "repr_name": repr_name,
        "slice_start": slice_start,
        "slice_end": slice_end,
    }
    decomp_key = str(decomposer_type or "imf").lower()
    if decomp_key == "fft":
        return FFTBandDetector(band_edges_hz=fft_band_edges_hz, **kwargs)
    if decomp_key in {"dwt", "wavelet"}:
        return DWTBandDetector(levels=dwt_levels, **kwargs)
    raise ValueError(f"Unsupported fixed decomposer type: {decomposer_type}")


def build_imf_detector(
    checkpoint_path: str,
    *,
    nfeats: int = 263,
    latent_dim: int = 512,
    imf_count: int = 3,
    frame_rate: float = 20.0,
    freeze: bool = True,
) -> IMFExtractor:
    ckpt = _torch_load_compat(checkpoint_path, map_location="cpu")
    model_cfg = None
    if isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
        model_cfg = dict(ckpt["config"].get("model", {}))
    if model_cfg:
        model_cfg.setdefault("nfeats", nfeats)
        model_cfg.setdefault("latent_dim", latent_dim)
        model_cfg.setdefault("imf_count", imf_count)
        model_cfg.setdefault("frame_rate", frame_rate)
        model = build_imf_extractor(model_cfg)
    else:
        model = IMFExtractor(
            nfeats=nfeats,
            latent_dim=latent_dim,
            imf_count=imf_count,
            frame_rate=frame_rate,
        )
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load IMF detector cleanly from {checkpoint_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    return model


class IMFDetectorAdapter(nn.Module):
    def __init__(self, checkpoint_path: str, **kwargs):
        super().__init__()
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"IMF detector checkpoint not found: {checkpoint_path}")
        self.detector = build_imf_detector(checkpoint_path, **kwargs)

    def forward(self, motion, lengths):
        return self.detector(motion, lengths)

    def extract_imf_features(self, motion, lengths):
        return self.detector.extract_imf_features(motion, lengths)

    def compute_hht_features(self, imfs):
        return self.detector.compute_hht_features(imfs)
