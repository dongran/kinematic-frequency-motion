import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

_PKG = os.path.dirname(os.path.abspath(__file__))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

# NOTE: 在当前独立仓库结构下，data / losses / models 目录位于项目根目录，
# 因此直接以顶层包名导入即可。
from data.dataloader_factory import build_dataloader_by_type
from losses.imf_losses import ImfLoss
from models.imf_factory import build_imf_extractor
from models.imf_group_utils import group_specs_from_model_cfg, resolve_group_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone IMFExtractor training")
    parser.add_argument(
        "--config",
        type=str,
        # 在独立仓库中，配置文件默认位于 configs/ 目录
        default="configs/imf_pose69_teacher.yaml",
        help="Path to YAML config file (relative to project root)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs in config (useful for quick smoke test)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Stop after N optimizer steps (for quick benchmark/debug).",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch = batch_size * grad_accum_steps.",
    )
    parser.add_argument(
        "--no_val",
        action="store_true",
        help="Disable validation even if val split exists.",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Run validation every N epochs (default: 1).",
    )
    parser.add_argument(
        "--nan_debug",
        action="store_true",
        help="Abort immediately when any NaN/Inf appears (prints batch ids and stats).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable torch.cuda.amp (mixed precision) for speed/memory.",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="AMP autocast dtype: fp16 (default) or bf16 (recommended on RTX 40xx).",
    )
    parser.add_argument(
        "--amp_init_scale",
        type=float,
        default=65536.0,
        help="GradScaler init_scale for fp16 AMP (ignored for bf16).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override dataset.batch_size in config",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help="Override dataset.val_batch_size in config",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override dataset.num_workers in config",
    )
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_device(arg_device: Optional[str]) -> torch.device:
    if arg_device is not None:
        return torch.device(arg_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loss_group_config(model_cfg, loss_cfg):
    groups = group_specs_from_model_cfg(model_cfg)
    if len(groups) <= 1:
        return None
    group_loss_cfg = loss_cfg.get("group_loss", {}) or {}
    if not isinstance(group_loss_cfg, dict):
        group_loss_cfg = {}
    if group_loss_cfg.get("enabled", True) is False:
        return None

    mode = str(group_loss_cfg.get("mode", loss_cfg.get("group_loss_mode", "group_only"))).strip().lower()
    aux_weight = float(group_loss_cfg.get("aux_weight", loss_cfg.get("group_aux_weight", 1.0)))
    weight_strategy = str(
        group_loss_cfg.get(
            "weight_strategy",
            loss_cfg.get("group_weight_strategy", "equal"),
        )
    ).strip().lower()
    weight_cfg = group_loss_cfg.get("weights", loss_cfg.get("group_weights"))
    weights = resolve_group_weights(groups, weight_cfg, strategy=weight_strategy)
    out = []
    for group, weight in zip(groups, weights):
        row = dict(group)
        row["weight"] = float(weight)
        out.append(row)
    return {
        "mode": mode,
        "aux_weight": aux_weight,
        "weight_strategy": weight_strategy,
        "specs": out,
    }


def maybe_load_init_checkpoint(model, train_cfg):
    ckpt_path = train_cfg.get("init_checkpoint", None)
    if not ckpt_path:
        return None

    ckpt_abs = os.path.expanduser(str(ckpt_path))
    if not os.path.isabs(ckpt_abs):
        ckpt_abs = os.path.join(os.getcwd(), ckpt_abs)

    strict = bool(train_cfg.get("init_checkpoint_strict", True))
    print(f"[IMF] Loading init checkpoint: {ckpt_abs}")
    ckpt = torch.load(ckpt_abs, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    load_result = model.load_state_dict(state_dict, strict=strict)

    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    print(
        "[IMF] Init checkpoint loaded: "
        f"strict={strict}, missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
    )
    if missing_keys:
        print(f"[IMF]   missing_keys: {missing_keys[:10]}")
    if unexpected_keys:
        print(f"[IMF]   unexpected_keys: {unexpected_keys[:10]}")
    return ckpt


def _normalize_prefix_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    out: List[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _matches_prefix(name: str, prefixes: Sequence[str]) -> bool:
    if not prefixes:
        return False
    for prefix in prefixes:
        if prefix in ("*", "all"):
            return True
        if name == prefix or name.startswith(f"{prefix}."):
            return True
    return False


def _ensure_prefix_hits(names: Sequence[str], prefixes: Sequence[str], kind: str) -> Dict[str, List[str]]:
    hits: Dict[str, List[str]] = {}
    for prefix in prefixes:
        matched = [name for name in names if _matches_prefix(name, [prefix])]
        if not matched:
            raise ValueError(f"{kind} prefix matched nothing: {prefix}")
        hits[prefix] = matched
    return hits


def apply_freeze_plan(model, train_cfg):
    trainable_prefixes = _normalize_prefix_list(train_cfg.get("trainable_modules"))
    freeze_prefixes = _normalize_prefix_list(train_cfg.get("freeze_modules"))
    freeze_modules_eval = bool(train_cfg.get("freeze_modules_eval", bool(trainable_prefixes or freeze_prefixes)))

    named_params = list(model.named_parameters())
    param_names = [name for name, _ in named_params]
    module_names = [name for name, _ in model.named_modules() if name]

    if trainable_prefixes:
        _ensure_prefix_hits(param_names, trainable_prefixes, kind="trainable_modules")
        _ensure_prefix_hits(module_names, trainable_prefixes, kind="trainable_modules(module)")
        for _name, param in named_params:
            param.requires_grad = False
        for name, param in named_params:
            if _matches_prefix(name, trainable_prefixes):
                param.requires_grad = True
    else:
        for _name, param in named_params:
            param.requires_grad = True

    if freeze_prefixes:
        _ensure_prefix_hits(param_names, freeze_prefixes, kind="freeze_modules")
        _ensure_prefix_hits(module_names, freeze_prefixes, kind="freeze_modules(module)")
        for name, param in named_params:
            if _matches_prefix(name, freeze_prefixes):
                param.requires_grad = False

    trainable_param_names = [name for name, param in named_params if param.requires_grad]
    if not trainable_param_names:
        raise ValueError("Freeze plan produced zero trainable parameters.")

    return {
        "enabled": bool(trainable_prefixes or freeze_prefixes),
        "trainable_prefixes": trainable_prefixes,
        "freeze_prefixes": freeze_prefixes,
        "freeze_modules_eval": freeze_modules_eval,
        "mode_strategy": "trainable_only" if trainable_prefixes else ("freeze_list" if freeze_prefixes else "full"),
        "trainable_param_names": trainable_param_names,
        "trainable_module_names": [name for name in module_names if _matches_prefix(name, trainable_prefixes)],
        "frozen_module_names": [name for name in module_names if _matches_prefix(name, freeze_prefixes)],
    }


def reapply_train_mode_policy(model, freeze_plan):
    if not freeze_plan or not freeze_plan.get("enabled"):
        model.train()
        return

    if not freeze_plan.get("freeze_modules_eval", True):
        model.train()
        return

    trainable_prefixes = freeze_plan.get("trainable_prefixes", []) or []
    freeze_prefixes = freeze_plan.get("freeze_prefixes", []) or []
    mode_strategy = str(freeze_plan.get("mode_strategy", "full"))

    if mode_strategy == "trainable_only" and trainable_prefixes:
        model.eval()
        for name, module in model.named_modules():
            if not name:
                continue
            if _matches_prefix(name, trainable_prefixes):
                module.train()
        return

    model.train()
    if mode_strategy == "freeze_list" and freeze_prefixes:
        for name, module in model.named_modules():
            if not name:
                continue
            if _matches_prefix(name, freeze_prefixes):
                module.eval()


def _resolve_schedule_epoch(spec: Dict[str, Any], key: str, total_epochs: int, default: int) -> int:
    if key in spec:
        return max(int(spec[key]), 1)
    ratio_key = key.replace("_epoch", "_ratio")
    if ratio_key in spec:
        return max(int(round(float(spec[ratio_key]) * float(total_epochs))), 1)
    return max(int(default), 1)


def _schedule_scale_for_epoch(epoch: int, total_epochs: int, spec: Optional[Dict[str, Any]]) -> float:
    if not isinstance(spec, dict) or not spec:
        return 1.0

    start_epoch = _resolve_schedule_epoch(spec, "start_epoch", total_epochs, 1)
    end_epoch = _resolve_schedule_epoch(spec, "end_epoch", total_epochs, start_epoch)
    start_scale = float(spec.get("start_scale", 0.0 if start_epoch > 1 else 1.0))
    end_scale = float(spec.get("end_scale", 1.0))
    if end_epoch <= start_epoch:
        return end_scale if epoch >= end_epoch else start_scale
    if epoch <= start_epoch:
        return start_scale
    if epoch >= end_epoch:
        return end_scale
    alpha = float(epoch - start_epoch) / float(max(end_epoch - start_epoch, 1))
    return start_scale + (end_scale - start_scale) * alpha


def apply_temporal_schedule(imf_loss_fn: ImfLoss, loss_cfg: Dict[str, Any], epoch: int, total_epochs: int) -> Dict[str, float]:
    temporal_aux_cfg = loss_cfg.get("temporal_aux", {}) or {}
    schedule_cfg = temporal_aux_cfg.get("schedule", {}) or {}
    delta_scale = _schedule_scale_for_epoch(epoch, total_epochs, schedule_cfg.get("delta"))
    accel_scale = _schedule_scale_for_epoch(epoch, total_epochs, schedule_cfg.get("accel"))
    phase_scale = _schedule_scale_for_epoch(epoch, total_epochs, schedule_cfg.get("phase"))
    imf_loss_fn.set_temporal_runtime_scales(
        delta_scale=delta_scale,
        accel_scale=accel_scale,
        phase_scale=phase_scale,
    )
    state = imf_loss_fn.get_temporal_weight_state()
    state["schedule_enabled"] = bool(schedule_cfg)
    return state


def apply_definition_schedule(imf_loss_fn: ImfLoss, loss_cfg: Dict[str, Any], epoch: int, total_epochs: int) -> Dict[str, float]:
    def_cfg = loss_cfg.get("definition_unsup", {}) or {}
    schedule_cfg = def_cfg.get("schedule", {}) or {}
    env_mean_scale = _schedule_scale_for_epoch(epoch, total_epochs, schedule_cfg.get("env_mean"))
    zero_ext_scale = _schedule_scale_for_epoch(epoch, total_epochs, schedule_cfg.get("zero_ext"))
    dc_mean_scale = _schedule_scale_for_epoch(epoch, total_epochs, schedule_cfg.get("dc_mean"))
    imf_loss_fn.set_definition_runtime_scales(
        env_mean_scale=env_mean_scale,
        zero_ext_scale=zero_ext_scale,
        dc_mean_scale=dc_mean_scale,
    )
    state = imf_loss_fn.get_definition_weight_state()
    state["schedule_enabled"] = bool(schedule_cfg)
    return state


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 设定随机种子
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device(args.device)
    print(f"[IMF] Using device: {device}")
    if torch.cuda.is_available():
        # 更稳定/更快的卷积算法选择（不会影响数值语义）
        torch.backends.cudnn.benchmark = True

    # ------------------------------ Data
    ds_cfg = cfg["dataset"]
    if args.batch_size is not None:
        ds_cfg["batch_size"] = int(args.batch_size)
    if args.val_batch_size is not None:
        ds_cfg["val_batch_size"] = int(args.val_batch_size)
    if args.num_workers is not None:
        ds_cfg["num_workers"] = int(args.num_workers)
    data_root = ds_cfg["root"]
    # 以当前工作目录为基准解析相对路径
    if not os.path.isabs(data_root):
        data_root = os.path.join(os.getcwd(), data_root)

    # split 文件目录（可选）：支持 root/splits/*.txt 的新布局
    split_dir = ds_cfg.get("split_dir", None)
    split_dir_abs = None
    if split_dir:
        if os.path.isabs(split_dir):
            split_dir_abs = split_dir
        else:
            cand = os.path.join(data_root, split_dir)
            split_dir_abs = cand if os.path.isdir(cand) else os.path.join(os.getcwd(), split_dir)

    # IMF 标签目录（可选）：支持把 imfs3 放在独立目录（例如 finemotion_263_v4_memd_pose63/aligned3/imfs3）
    imf_dir = ds_cfg.get("imf_dir", None)
    imf_dir_abs = None
    if imf_dir:
        imf_dir_abs = imf_dir if os.path.isabs(imf_dir) else os.path.join(os.getcwd(), imf_dir)

    stats_dir = ds_cfg.get("stats_dir", None)
    stats_dir_abs = None
    if stats_dir:
        if os.path.isabs(stats_dir):
            stats_dir_abs = stats_dir
        else:
            stats_dir_abs = os.path.join(data_root, stats_dir)

    sample_weight_path = ds_cfg.get("sample_weight_path", None)
    sample_weight_path_abs = None
    if sample_weight_path:
        if os.path.isabs(sample_weight_path):
            sample_weight_path_abs = sample_weight_path
        else:
            sample_weight_path_abs = os.path.join(data_root, sample_weight_path)

    # 训练集
    train_split = ds_cfg.get("split", "train")
    dataset_type = ds_cfg.get("type", None)
    train_loader = build_dataloader_by_type(
        dataset_type=dataset_type,
        root=data_root,
        split=train_split,
        split_dir=split_dir_abs,
        imf_dir=imf_dir_abs,
        stats_dir=stats_dir_abs,
        sample_weight_path=sample_weight_path_abs,
        sample_weight_replacement=bool(ds_cfg.get("sample_weight_replacement", True)),
        batch_size=ds_cfg.get("batch_size", 64),
        num_workers=ds_cfg.get("num_workers", 4),
        max_motion_length=ds_cfg.get("max_motion_length", 196),
        min_motion_length=ds_cfg.get("min_motion_length", 40),
        unit_length=ds_cfg.get("unit_length", 4),
    )

    # 验证集（如果存在 val.txt，则自动启用）
    val_loader = None
    val_split = ds_cfg.get("val_split", "val")
    try:
        val_loader = build_dataloader_by_type(
            dataset_type=dataset_type,
            root=data_root,
            split=val_split,
            split_dir=split_dir_abs,
            imf_dir=imf_dir_abs,
            stats_dir=stats_dir_abs,
            sample_weight_path=None,
            sample_weight_replacement=bool(ds_cfg.get("sample_weight_replacement", True)),
            batch_size=ds_cfg.get("val_batch_size", ds_cfg.get("batch_size", 64)),
            num_workers=ds_cfg.get("num_workers", 4),
            max_motion_length=ds_cfg.get("max_motion_length", 196),
            min_motion_length=ds_cfg.get("min_motion_length", 40),
            unit_length=ds_cfg.get("unit_length", 4),
        )
    except FileNotFoundError as e:
        print(
            f"[IMF] Validation split not available (skip val evaluation): {e}"
        )
    except RuntimeError as e:
        print(f"[IMF] Validation split has no valid samples (skip val evaluation): {e}")

    if args.no_val and val_loader is not None:
        print("[IMF] --no_val is set: disable validation.")
        val_loader = None

    # 简要数据集信息（类似 MCM-LDM 的 dataset summary）
    dataset = train_loader.dataset
    num_samples = len(dataset)
    num_batches = len(train_loader)
    print("\n=== IMF Dataset summary ===")
    print(f"Root path: {data_root}")
    print(f"Train split: {train_split}")
    print(
        f"Samples: {num_samples}, Batches/epoch: {num_batches}, "
        f"batch_size={ds_cfg.get('batch_size', 64)}"
    )
    if val_loader is not None:
        val_ds = val_loader.dataset
        print(
            f"Val split: {val_split}, Samples: {len(val_ds)}, "
            f"Batches/epoch: {len(val_loader)}, "
            f"batch_size={ds_cfg.get('val_batch_size', ds_cfg.get('batch_size', 64))}"
        )

    # 抽一个 batch 看一下张量形状（来自训练集）
    sample_batch = None
    for b in train_loader:
        if b:
            sample_batch = b
            break
    if sample_batch:
        print("Example batch shapes:")
        print(f"  motion: {tuple(sample_batch['motion'].shape)}")
        print(f"  imfs  : {tuple(sample_batch['imfs'].shape)}")
        print(f"  length: {tuple(sample_batch['length'].shape)}")

    # ------------------------------ Model
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    imf_extractor = build_imf_extractor(model_cfg).to(device)
    maybe_load_init_checkpoint(imf_extractor, train_cfg)
    freeze_plan = apply_freeze_plan(imf_extractor, train_cfg)
    expected_imf_dof = int(model_cfg.get("imf_dof", getattr(imf_extractor, "imfdof", 63)))

    # 模型参数量统计
    total_params = sum(p.numel() for p in imf_extractor.parameters())
    trainable_params = sum(p.numel() for p in imf_extractor.parameters() if p.requires_grad)
    print("\n=== IMFExtractor model summary ===")
    print(f"Total parameters   : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ------------------------------ Loss
    loss_cfg = cfg["loss"]
    imf_loss_fn = ImfLoss(
        frame_rate=model_cfg.get("frame_rate", 20.0),
        lambda_decomp=loss_cfg.get("lambda_decomp", 1.0),
        lambda_emd=loss_cfg.get("lambda_emd", 0.5),
        lambda_ht=loss_cfg.get("lambda_ht", 0.5),
        alpha_ht_amp=loss_cfg.get("alpha_ht_amp", 0.5),
        alpha_ht_freq=loss_cfg.get("alpha_ht_freq", 0.5),
        beta_emd_mean=loss_cfg.get("beta_emd_mean", 0.5),
        beta_emd_ext=loss_cfg.get("beta_emd_ext", 0.5),
        temporal_aux_cfg=loss_cfg.get("temporal_aux", {}),
        definition_unsup_cfg=loss_cfg.get("definition_unsup", {}),
    )
    loss_group_cfg = build_loss_group_config(model_cfg, loss_cfg)
    loss_group_specs = loss_group_cfg["specs"] if loss_group_cfg else None
    if loss_group_cfg:
        print("\n=== Grouped loss setup ===")
        print(f"  mode: {loss_group_cfg['mode']}")
        print(f"  aux_weight: {loss_group_cfg['aux_weight']:.4f}")
        print(f"  weight_strategy: {loss_group_cfg['weight_strategy']}")
        for group in loss_group_specs:
            print(
                "  - "
                f"{group['name']}: channels[{group['start']}:{group['end']}] "
                f"(dof={group['dof']}), weight={group['weight']:.4f}"
            )

    if imf_loss_fn.temporal_aux_enabled:
        temporal_aux_cfg = loss_cfg.get("temporal_aux", {}) or {}
        group_desc = ", ".join(temporal_aux_cfg.get("groups", []) or ["all"])
        print("\n=== Temporal alignment aux setup ===")
        print(f"  groups: {group_desc}")
        print(f"  lambda_delta(base): {imf_loss_fn.base_lambda_temporal_delta:.4f}")
        print(f"  lambda_accel(base): {imf_loss_fn.base_lambda_temporal_accel:.4f}")
        print(f"  lambda_ht_phase(base): {imf_loss_fn.base_lambda_ht_phase:.4f}")
        print(f"  normalize_by_target_rms: {imf_loss_fn.temporal_normalize_by_target_rms}")
        print(f"  phase_amp_weighted: {imf_loss_fn.phase_amp_weighted}")
        print(f"  phase_weight_clip: {imf_loss_fn.phase_weight_clip:.4f}")
        if imf_loss_fn.temporal_schedule_cfg:
            print("  schedule:")
            for key in ("delta", "accel", "phase"):
                spec = imf_loss_fn.temporal_schedule_cfg.get(key, None)
                if isinstance(spec, dict) and spec:
                    print(f"    - {key}: {spec}")

    if imf_loss_fn.definition_unsup_enabled:
        def_cfg = loss_cfg.get("definition_unsup", {}) or {}
        def_group_desc = ", ".join(def_cfg.get("groups", []) or ["all"])
        print("\n=== IMF definition unsup setup ===")
        print(f"  groups: {def_group_desc}")
        print(f"  lambda_env_mean(base): {imf_loss_fn.base_lambda_definition_env_mean:.4f}")
        print(f"  lambda_zero_ext(base): {imf_loss_fn.base_lambda_definition_zero_ext:.4f}")
        print(f"  lambda_dc_mean(base): {imf_loss_fn.base_lambda_definition_dc_mean:.4f}")
        print(f"  normalize_by_rms: {imf_loss_fn.definition_unsup_normalize_by_rms}")
        print(f"  rms_source: {imf_loss_fn.definition_unsup_rms_source}")
        print(f"  envelope.pool_kernel: {imf_loss_fn.definition_env_pool_kernel}")
        print(f"  zero_cross.softsign_scale: {imf_loss_fn.definition_softsign_scale:.4f}")
        print(f"  zero_cross.margin: {imf_loss_fn.definition_zero_ext_margin:.4f}")
        if imf_loss_fn.definition_schedule_cfg:
            print("  schedule:")
            for key in ("env_mean", "zero_ext", "dc_mean"):
                spec = imf_loss_fn.definition_schedule_cfg.get(key, None)
                if isinstance(spec, dict) and spec:
                    print(f"    - {key}: {spec}")

    if freeze_plan.get("enabled"):
        print("\n=== Module freeze setup ===")
        print(f"  mode_strategy: {freeze_plan['mode_strategy']}")
        print(f"  freeze_modules_eval: {freeze_plan['freeze_modules_eval']}")
        if freeze_plan.get("trainable_prefixes"):
            print(f"  trainable_modules: {', '.join(freeze_plan['trainable_prefixes'])}")
        if freeze_plan.get("freeze_prefixes"):
            print(f"  freeze_modules: {', '.join(freeze_plan['freeze_prefixes'])}")
        print(f"  trainable parameter tensors: {len(freeze_plan['trainable_param_names'])}")

    # ------------------------------ Optimizer
    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    if args.max_steps is not None:
        train_cfg["max_steps"] = int(args.max_steps)
    trainable_param_list = [p for p in imf_extractor.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_param_list,
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    # 学习率调度器（ReduceLROnPlateau）
    scheduler_cfg = train_cfg.get("scheduler", {})
    use_scheduler = scheduler_cfg.get("type", None) == "reduce_on_plateau"
    scheduler = None
    if use_scheduler:
        plateau_kwargs = {
            "mode": "min",
            "factor": scheduler_cfg.get("factor", 0.5),
            "patience": scheduler_cfg.get("patience", 80),
            "threshold": scheduler_cfg.get("threshold", 1e-4),
            "threshold_mode": scheduler_cfg.get("threshold_mode", "rel"),
            "min_lr": scheduler_cfg.get("min_lr", 1e-6),
        }
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                verbose=True,
                **plateau_kwargs,
            )
        except TypeError:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **plateau_kwargs,
            )

    # Early stopping 配置
    early_cfg = train_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience", 200))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    early_min_epochs = int(early_cfg.get("min_epochs", 0))
    best_val_loss = None
    epochs_no_improve = 0

    # ------------------------------ Logging & checkpoints
    out_dir = train_cfg.get("out_dir", "imf_extractor/outputs")
    os.makedirs(out_dir, exist_ok=True)
    run_name = cfg["experiment"].get("name", "imf_run")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(out_dir, f"{run_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # 保存最终解析后的 config（已包含 CLI overrides），用于实验可复现与对比
    try:
        resolved_cfg_path = os.path.join(log_dir, "config_resolved.yaml")
        with open(resolved_cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"[IMF] Saved resolved config to: {resolved_cfg_path}")
    except Exception as e:
        print(f"[IMF] WARNING: failed to save config_resolved.yaml: {e}")

    # TensorBoard 记录一些 run 元信息（便于对比不同变体）
    try:
        model_variant = str(model_cfg.get("variant", "cnn"))
        writer.add_text("config/model_variant", model_variant, 0)
        writer.add_scalar("meta/total_params", float(total_params), 0)
        writer.add_scalar("meta/trainable_params", float(trainable_params), 0)
        if loss_group_cfg:
            group_desc = "\n".join(
                [
                    f"- {group['name']}: {group['start']}:{group['end']} "
                    f"(dof={group['dof']}, weight={group['weight']:.4f})"
                    for group in loss_group_specs
                ]
            )
            writer.add_text(
                "config/output_groups",
                "\n".join(
                    [
                        f"mode: {loss_group_cfg['mode']}",
                        f"aux_weight: {loss_group_cfg['aux_weight']:.6f}",
                        f"weight_strategy: {loss_group_cfg['weight_strategy']}",
                        group_desc,
                    ]
                ),
                0,
            )
        if freeze_plan.get("enabled"):
            writer.add_text(
                "config/freeze_plan",
                "\n".join(
                    [
                        f"mode_strategy: {freeze_plan['mode_strategy']}",
                        f"freeze_modules_eval: {freeze_plan['freeze_modules_eval']}",
                        f"trainable_modules: {', '.join(freeze_plan.get('trainable_prefixes', []) or ['<none>'])}",
                        f"freeze_modules: {', '.join(freeze_plan.get('freeze_prefixes', []) or ['<none>'])}",
                        f"trainable_param_tensors: {len(freeze_plan['trainable_param_names'])}",
                    ]
                ),
                0,
            )
    except Exception:
        pass

    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ------------------------------ Training loop
    epochs = train_cfg.get("epochs", 50)
    ckpt_interval = train_cfg.get("ckpt_interval", 5)
    max_steps = train_cfg.get("max_steps", None)
    if max_steps is not None:
        max_steps = int(max_steps)

    use_amp = bool(args.amp) and torch.cuda.is_available()
    amp_dtype = str(getattr(args, "amp_dtype", "fp16")).lower()
    if amp_dtype not in ("fp16", "bf16"):
        amp_dtype = "fp16"
    autocast_dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    use_scaler = bool(use_amp) and (amp_dtype == "fp16")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler, init_scale=float(args.amp_init_scale))
    if use_amp:
        print(f"[IMF] AMP enabled: dtype={amp_dtype}, grad_scaler={'on' if use_scaler else 'off'}")

    grad_accum_steps = max(int(args.grad_accum_steps), 1)
    opt_step = 0
    stop_early = False

    for epoch in range(1, epochs + 1):
        temporal_state = apply_temporal_schedule(imf_loss_fn, loss_cfg, epoch, epochs)
        definition_state = apply_definition_schedule(imf_loss_fn, loss_cfg, epoch, epochs)
        reapply_train_mode_policy(imf_extractor, freeze_plan)
        running_loss = 0.0
        epoch_loss_sums = None
        epoch_batch_count = 0
        micro_step_in_epoch = 0
        optimizer.zero_grad(set_to_none=True)
        # 使用 tqdm 显示当前 epoch 的 batch 进度
        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        ):
            if not batch:
                continue

            batch_ids = batch.get("ids", None)
            motion = batch["motion"].to(device)  # [B, T_pad, 263]
            target_imfs = batch["imfs"].to(device)  # [B, 3, 63, T_pad]
            lengths = batch["length"].to(device)  # [B]
            if int(target_imfs.shape[2]) != expected_imf_dof:
                raise ValueError(
                    f"Target IMF dof mismatch: batch has {target_imfs.shape[2]}, model expects {expected_imf_dof}"
                )

            # 有效帧 mask（用于解决 batch 内 padding 的影响）
            T_pad = motion.shape[1]
            time_mask = (
                torch.arange(T_pad, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            )  # [B, T_pad] bool

            # padding 末帧延拓：将 motion 的 padding 部分填充为最后一帧，减少边界伪影对模型输出的影响
            # （损失计算仍会用 time_mask 排除 padding 帧）
            last_idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(
                motion.shape[0], 1, motion.shape[2]
            )
            last_motion = motion.gather(dim=1, index=last_idx)  # [B,1,263]
            motion = torch.where(time_mask.unsqueeze(-1), motion, last_motion.expand_as(motion))

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
                # 预测 IMF
                pred_imfs_flat = imf_extractor.extract_imf_features(motion)  # [B, 3*63, T_pred]
                target_flat = target_imfs.reshape(
                    target_imfs.shape[0], -1, target_imfs.shape[-1]
                )  # [B, 3*63, T_gt]

                # 与主工程 MCM-LDM 中一致：若预测长度与 GT 长度不完全相同，
                # 在时间维上取二者的最小值进行对齐，避免 stride/下采样导致的 shape mismatch。
                T_used = min(pred_imfs_flat.shape[-1], target_flat.shape[-1])
                pred_imfs_used = pred_imfs_flat[..., :T_used]
                target_imfs_used = target_flat[..., :T_used]
                time_mask_used = time_mask[:, :T_used]

                # 为 HHT / NaNGuard 提供更完整的调试上下文，方便日志查看
                debug_ctx = {
                    "loss": "imf_pretrain",
                    "split": ds_cfg.get("split", "train"),
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    # 这里使用 lambda_ht 作为 ht_loss_weight 的近似权重，便于在日志中识别
                    "ht_loss_weight": loss_cfg.get("lambda_ht", 0.5),
                }

                # 对 loss 做 float32 计算（尤其是 EMD/HHT 相关项），避免 fp16 下溢/溢出。
                with torch.cuda.amp.autocast(enabled=False):
                    total_loss, loss_dict = imf_loss_fn(
                        pred_imf=pred_imfs_used.float(),
                        target_imf=target_imfs_used.float(),
                        debug_ctx=debug_ctx,
                        time_mask=time_mask_used,
                        group_specs=loss_group_specs,
                        group_loss_mode=(loss_group_cfg or {}).get("mode", "group_only"),
                        group_aux_weight=float((loss_group_cfg or {}).get("aux_weight", 1.0)),
                    )

            if args.nan_debug:
                def _isfinite(x: torch.Tensor) -> bool:
                    return bool(torch.isfinite(x).all().item())

                bad = []
                if not _isfinite(motion):
                    bad.append("motion")
                if not _isfinite(target_imfs_used):
                    bad.append("target_imf")
                if not _isfinite(pred_imfs_used):
                    bad.append("pred_imf")
                if not _isfinite(total_loss):
                    bad.append("total_loss")
                for k, v in loss_dict.items():
                    if torch.is_tensor(v) and (not _isfinite(v)):
                        bad.append(f"loss[{k}]")

                if bad:
                    print("\n[NaNDebug] Non-finite detected:", bad)
                    print("  epoch:", epoch, "batch_idx:", batch_idx, "T_used:", T_used)
                    try:
                        print("  batch_ids (first 8):", batch_ids[:8] if batch_ids else None)
                    except Exception:
                        print("  batch_ids:", batch_ids)
                    print("  motion absmax:", float(motion.abs().max().item()))
                    print("  target absmax:", float(target_imfs_used.abs().max().item()))
                    print("  pred   absmax:", float(pred_imfs_used.abs().max().item()))
                    print("  loss_dict:")
                    for kk, vv in loss_dict.items():
                        if torch.is_tensor(vv):
                            vv_f = float(vv.detach().item()) if vv.numel() == 1 else float("nan")
                            print(f"    - {kk}: {vv_f}")
                    return

            # 梯度累积：保持“有效 batch”变大但显存不爆
            loss_scaled = total_loss / float(grad_accum_steps)
            scaler.scale(loss_scaled).backward()
            micro_step_in_epoch += 1

            do_step = (micro_step_in_epoch % grad_accum_steps) == 0
            if do_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

            running_loss += float(total_loss.item())
            epoch_batch_count += 1

            # 统计本 epoch 内各项 loss 的和，方便在 epoch 结束时输出一次平均值
            if epoch_loss_sums is None:
                epoch_loss_sums = {k: v.detach().clone() for k, v in loss_dict.items()}
            else:
                for k, v in loss_dict.items():
                    epoch_loss_sums[k] += v.detach()

            if max_steps is not None and opt_step >= max_steps:
                stop_early = True
                break

        avg_loss = running_loss / max(epoch_batch_count, 1)
        # 计算并打印本 epoch 平均 loss（各子项）
        if epoch_loss_sums is not None and epoch_batch_count > 0:
            epoch_loss_means = {
                k: v / float(epoch_batch_count) for k, v in epoch_loss_sums.items()
            }
            loss_str = " | ".join(
                [f"{k}={v.item():.4e}" for k, v in epoch_loss_means.items()]
            )
            print(
                f"[Epoch {epoch}/{epochs}] "
                f"avg_total={avg_loss:.6f} | {loss_str}"
            )
            # TensorBoard：训练集，每个 epoch 结束时记录一次
            writer.add_scalar("loss/epoch_avg", avg_loss, epoch)
            for k, v in epoch_loss_means.items():
                writer.add_scalar(f"loss/{k}", v.item(), epoch)
        else:
            print(f"[Epoch {epoch}/{epochs}] avg_loss={avg_loss:.6f}")
            writer.add_scalar("loss/epoch_avg", avg_loss, epoch)
        if imf_loss_fn.temporal_aux_enabled:
            writer.add_scalar("loss_runtime/lambda_temporal_delta", temporal_state["lambda_temporal_delta"], epoch)
            writer.add_scalar("loss_runtime/lambda_temporal_accel", temporal_state["lambda_temporal_accel"], epoch)
            writer.add_scalar("loss_runtime/lambda_ht_phase", temporal_state["lambda_ht_phase"], epoch)
            writer.add_scalar("loss_runtime/delta_scale", temporal_state["delta_scale"], epoch)
            writer.add_scalar("loss_runtime/accel_scale", temporal_state["accel_scale"], epoch)
            writer.add_scalar("loss_runtime/phase_scale", temporal_state["phase_scale"], epoch)
        if imf_loss_fn.definition_unsup_enabled:
            writer.add_scalar(
                "loss_runtime/lambda_definition_env_mean",
                definition_state["lambda_definition_env_mean"],
                epoch,
            )
            writer.add_scalar(
                "loss_runtime/lambda_definition_zero_ext",
                definition_state["lambda_definition_zero_ext"],
                epoch,
            )
            writer.add_scalar(
                "loss_runtime/lambda_definition_dc_mean",
                definition_state["lambda_definition_dc_mean"],
                epoch,
            )
            writer.add_scalar(
                "loss_runtime/definition_env_mean_scale",
                definition_state["definition_env_mean_scale"],
                epoch,
            )
            writer.add_scalar(
                "loss_runtime/definition_zero_ext_scale",
                definition_state["definition_zero_ext_scale"],
                epoch,
            )
            writer.add_scalar(
                "loss_runtime/definition_dc_mean_scale",
                definition_state["definition_dc_mean_scale"],
                epoch,
            )

        # ------------------------------ Validation loop
        val_avg_loss = None
        val_interval = max(int(args.val_interval), 1)
        if val_loader is not None and (epoch % val_interval == 0):
            imf_extractor.eval()
            val_running_loss = 0.0
            val_loss_sums = None
            val_batch_count = 0

            for batch_idx, batch in enumerate(
                tqdm(val_loader, desc=f"[Val] Epoch {epoch}/{epochs}", leave=False)
            ):
                if not batch:
                    continue

                motion = batch["motion"].to(device)  # [B, T_pad, 263]
                target_imfs = batch["imfs"].to(device)  # [B, 3, 63, T_pad]
                lengths = batch["length"].to(device)  # [B]
                if int(target_imfs.shape[2]) != expected_imf_dof:
                    raise ValueError(
                        f"Target IMF dof mismatch: batch has {target_imfs.shape[2]}, model expects {expected_imf_dof}"
                    )

                T_pad = motion.shape[1]
                time_mask = (
                    torch.arange(T_pad, device=device).unsqueeze(0) < lengths.unsqueeze(1)
                )  # [B, T_pad] bool

                # 同训练：末帧延拓，减少 padding 影响
                last_idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(
                    motion.shape[0], 1, motion.shape[2]
                )
                last_motion = motion.gather(dim=1, index=last_idx)  # [B,1,263]
                motion = torch.where(time_mask.unsqueeze(-1), motion, last_motion.expand_as(motion))

                # 预测 IMF
                pred_imfs_flat = imf_extractor.extract_imf_features(
                    motion
                )  # [B, 3*63, T_pred]
                target_flat = target_imfs.reshape(
                    target_imfs.shape[0], -1, target_imfs.shape[-1]
                )  # [B, 3*63, T_gt]

                # 时间维对齐
                T_used = min(pred_imfs_flat.shape[-1], target_flat.shape[-1])
                pred_imfs_used = pred_imfs_flat[..., :T_used]
                target_imfs_used = target_flat[..., :T_used]
                time_mask_used = time_mask[:, :T_used]

                debug_ctx = {
                    "loss": "imf_pretrain",
                    "split": val_split,
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "ht_loss_weight": loss_cfg.get("lambda_ht", 0.5),
                }

                with torch.no_grad():
                    val_total_loss, val_loss_dict = imf_loss_fn(
                        pred_imf=pred_imfs_used.float(),
                        target_imf=target_imfs_used.float(),
                        debug_ctx=debug_ctx,
                        time_mask=time_mask_used,
                        group_specs=loss_group_specs,
                        group_loss_mode=(loss_group_cfg or {}).get("mode", "group_only"),
                        group_aux_weight=float((loss_group_cfg or {}).get("aux_weight", 1.0)),
                    )

                val_running_loss += float(val_total_loss.item())
                val_batch_count += 1

                if val_loss_sums is None:
                    val_loss_sums = {
                        k: v.detach().clone() for k, v in val_loss_dict.items()
                    }
                else:
                    for k, v in val_loss_dict.items():
                        val_loss_sums[k] += v.detach()

            if val_batch_count > 0 and val_loss_sums is not None:
                val_avg_loss = val_running_loss / float(val_batch_count)
                val_loss_means = {
                    k: v / float(val_batch_count) for k, v in val_loss_sums.items()
                }
                val_loss_str = " | ".join(
                    [f"{k}={v.item():.4e}" for k, v in val_loss_means.items()]
                )
                print(
                    f"[Val   {epoch}/{epochs}] "
                    f"avg_total={val_avg_loss:.6f} | {val_loss_str}"
                )
                # TensorBoard：验证集，单独放到 val/ 前缀下
                writer.add_scalar("val/loss/epoch_avg", val_avg_loss, epoch)
                for k, v in val_loss_means.items():
                    writer.add_scalar(f"val/loss/{k}", v.item(), epoch)

        # ------------------------------ Scheduler step & early stopping
        # 仅在存在验证集时才基于 val loss 调整学习率 / 判断是否早停
        if scheduler is not None and val_avg_loss is not None:
            scheduler.step(val_avg_loss)

        # 记录当前学习率到 TensorBoard，方便观察调度行为（即便没有 scheduler 也能看到恒定 lr）
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("lr", current_lr, epoch)

        # Early stopping 判断
        if early_enabled and val_avg_loss is not None:
            if (best_val_loss is None) or (
                val_avg_loss < best_val_loss - early_min_delta
            ):
                best_val_loss = val_avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch >= early_min_epochs and epochs_no_improve >= early_patience:
                print(
                    f"[EarlyStopping] Stop at epoch {epoch}: "
                    f"best_val_loss={best_val_loss:.6f}, "
                    f"epochs_no_improve={epochs_no_improve}"
                )
                # 触发 early stopping 时，额外保存一次 checkpoint
                ckpt_path = os.path.join(ckpt_dir, f"imf_epoch_{epoch}_early_stop.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": imf_extractor.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": cfg,
                    },
                    ckpt_path,
                )
                print(f"  Saved early-stop checkpoint to {ckpt_path}")
                break

        # 保存 checkpoint
        if epoch % ckpt_interval == 0 or epoch == epochs:
            ckpt_path = os.path.join(ckpt_dir, f"imf_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": imf_extractor.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint to {ckpt_path}")

        if stop_early:
            print(
                f"[IMF] Reached max_steps={max_steps} (optimizer steps), stop early. "
                f"grad_accum_steps={grad_accum_steps}"
            )
            break

    writer.close()
    print(f"Training finished. Logs and checkpoints are under: {log_dir}")
    if torch.cuda.is_available():
        max_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        max_resv = torch.cuda.max_memory_reserved() / (1024**2)
        print(f"[CUDA] max_memory_allocated={max_alloc:.1f} MiB, max_memory_reserved={max_resv:.1f} MiB")


if __name__ == "__main__":
    main()


