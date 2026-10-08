#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mld.data.contact_timing_dataset import ContactTimingWindowDataset
from mld.models.architectures.contact_timing_predictor import (
    ContactTimingPredictor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a contact/timing predictor.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/contact_timing_finemotion.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--device", type=str, default="", help="Override device, e.g. cuda:0")
    parser.add_argument("--label_root", type=str, default="", help="Override DATA.LABEL_ROOT")
    parser.add_argument("--split_root", type=str, default="", help="Override DATA.SPLIT_ROOT")
    parser.add_argument("--epochs", type=int, default=0, help="Override TRAIN.EPOCHS")
    parser.add_argument("--batch_size", type=int, default=0, help="Override TRAIN.BATCH_SIZE")
    parser.add_argument("--num_workers", type=int, default=-1, help="Override TRAIN.NUM_WORKERS")
    parser.add_argument("--lr", type=float, default=0.0, help="Override TRAIN.LR")
    parser.add_argument("--window_size", type=int, default=0, help="Override DATA.WINDOW_SIZE")
    parser.add_argument("--max_train_items", type=int, default=0, help="Override DATA.MAX_TRAIN_ITEMS")
    parser.add_argument("--max_val_items", type=int, default=0, help="Override DATA.MAX_VAL_ITEMS")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint .pt")
    parser.add_argument("--no_tensorboard", action="store_true", help="Disable tensorboard logging")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def torch_load_compat(path: str | Path, *, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _event_frames(contact: np.ndarray) -> np.ndarray:
    change = np.zeros((contact.shape[0],), dtype=bool)
    change[1:] = np.any(contact[1:] != contact[:-1], axis=-1)
    return np.flatnonzero(change)


def contact_precision_recall(
    predicted_contact: np.ndarray,
    target_contact: np.ndarray,
    *,
    tolerance_frames: int = 2,
) -> tuple[float, float]:
    pred_events = _event_frames(predicted_contact)
    gt_events = _event_frames(target_contact)
    if len(pred_events) == 0 and len(gt_events) == 0:
        return 1.0, 1.0
    if len(pred_events) == 0 or len(gt_events) == 0:
        return 0.0, 0.0

    def _count_matches(src: np.ndarray, dst: np.ndarray) -> int:
        matches = 0
        for frame in src:
            if np.any(np.abs(dst - frame) <= tolerance_frames):
                matches += 1
        return matches

    precision = _count_matches(pred_events, gt_events) / max(len(pred_events), 1)
    recall = _count_matches(gt_events, pred_events) / max(len(gt_events), 1)
    return float(precision), float(recall)


def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=pos_weight,
    )
    while mask.dim() < loss.dim():
        mask = mask.unsqueeze(-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def masked_temporal_l1(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    order: int,
) -> torch.Tensor:
    if order not in (1, 2):
        raise ValueError(f"unsupported order: {order}")
    if values.shape[1] <= order:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    if order == 1:
        diff = values[:, 1:] - values[:, :-1]
        valid = mask[:, 1:] * mask[:, :-1]
    else:
        diff = values[:, 2:] - 2.0 * values[:, 1:-1] + values[:, :-2]
        valid = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
    while valid.dim() < diff.dim():
        valid = valid.unsqueeze(-1)
    return (diff.abs() * valid).sum() / valid.sum().clamp_min(1.0)


def frame_binary_accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).to(dtype=target.dtype)
    valid = mask.unsqueeze(-1)
    correct = ((pred == target).to(dtype=target.dtype) * valid).sum()
    total = valid.sum() * target.shape[-1]
    return float((correct / total.clamp_min(1.0)).item())


def build_dataloader(dataset: ContactTimingWindowDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
    )


def evaluate(
    model: ContactTimingPredictor,
    loader: DataLoader,
    device: torch.device,
    *,
    pos_weight: torch.Tensor | None,
    lambda_tv: float,
    lambda_acc: float,
    event_tolerance_frames: int,
) -> dict[str, float]:
    model.eval()
    losses_total: list[float] = []
    losses_bce: list[float] = []
    losses_tv: list[float] = []
    losses_acc: list[float] = []
    accs: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            target = batch["contact"].to(device)
            mask = batch["mask"].to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            loss_bce = masked_bce_with_logits(logits, target, mask, pos_weight=pos_weight)
            loss_tv = masked_temporal_l1(probs, mask, order=1)
            loss_acc = masked_temporal_l1(probs, mask, order=2)
            loss_total = loss_bce + lambda_tv * loss_tv + lambda_acc * loss_acc

            losses_total.append(float(loss_total.item()))
            losses_bce.append(float(loss_bce.item()))
            losses_tv.append(float(loss_tv.item()))
            losses_acc.append(float(loss_acc.item()))
            accs.append(frame_binary_accuracy(logits, target, mask))

            pred_np = (probs.detach().cpu().numpy() > 0.5).astype(np.float32)
            target_np = target.detach().cpu().numpy().astype(np.float32)
            mask_np = (mask.detach().cpu().numpy() > 0.5)
            for sample_pred, sample_target, sample_mask in zip(pred_np, target_np, mask_np):
                valid = np.flatnonzero(sample_mask)
                if len(valid) == 0:
                    continue
                pred_valid = sample_pred[: len(valid)]
                target_valid = sample_target[: len(valid)]
                precision, recall = contact_precision_recall(
                    pred_valid,
                    target_valid,
                    tolerance_frames=event_tolerance_frames,
                )
                precisions.append(precision)
                recalls.append(recall)

    precision_mean = float(np.mean(precisions)) if precisions else 0.0
    recall_mean = float(np.mean(recalls)) if recalls else 0.0
    f1 = 0.0
    if precision_mean + recall_mean > 0.0:
        f1 = 2.0 * precision_mean * recall_mean / (precision_mean + recall_mean)
    return {
        "loss_total": float(np.mean(losses_total)) if losses_total else 0.0,
        "loss_bce": float(np.mean(losses_bce)) if losses_bce else 0.0,
        "loss_tv": float(np.mean(losses_tv)) if losses_tv else 0.0,
        "loss_acc": float(np.mean(losses_acc)) if losses_acc else 0.0,
        "frame_acc": float(np.mean(accs)) if accs else 0.0,
        "event_precision": precision_mean,
        "event_recall": recall_mean,
        "event_f1": float(f1),
    }


def _prepare_run_dir(cfg: dict[str, Any], resume_path: str) -> Path:
    if resume_path:
        ckpt_path = Path(resume_path).expanduser().resolve()
        return ckpt_path.parent.parent
    output_root = Path(cfg["OUTPUT"]["ROOT"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{cfg['NAME']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.label_root:
        cfg["DATA"]["LABEL_ROOT"] = args.label_root
    if args.split_root:
        cfg["DATA"]["SPLIT_ROOT"] = args.split_root
    if args.epochs > 0:
        cfg["TRAIN"]["EPOCHS"] = int(args.epochs)
    if args.batch_size > 0:
        cfg["TRAIN"]["BATCH_SIZE"] = int(args.batch_size)
    if args.num_workers >= 0:
        cfg["TRAIN"]["NUM_WORKERS"] = int(args.num_workers)
    if args.lr > 0:
        cfg["TRAIN"]["LR"] = float(args.lr)
    if args.window_size > 0:
        cfg["DATA"]["WINDOW_SIZE"] = int(args.window_size)
    if args.max_train_items > 0:
        cfg["DATA"]["MAX_TRAIN_ITEMS"] = int(args.max_train_items)
    if args.max_val_items > 0:
        cfg["DATA"]["MAX_VAL_ITEMS"] = int(args.max_val_items)
    if args.no_tensorboard:
        cfg["TRAIN"]["USE_TENSORBOARD"] = False


def train(cfg: dict[str, Any], *, device_override: str = "", resume_path: str = "") -> Path:
    seed = int(cfg.get("SEED", 1234))
    set_seed(seed)

    run_dir = _prepare_run_dir(cfg, resume_path)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.json"
    config_dump_path = run_dir / "config_resolved.yaml"
    with open(config_dump_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    data_cfg = cfg["DATA"]
    train_cfg = cfg["TRAIN"]
    model_cfg = cfg["MODEL"]
    eval_cfg = cfg.get("EVAL", {})

    label_root = Path(data_cfg["LABEL_ROOT"]).expanduser().resolve()
    split_root = Path(data_cfg["SPLIT_ROOT"]).expanduser().resolve()
    if not label_root.is_dir():
        raise FileNotFoundError(f"label_root 不存在: {label_root}")
    if not split_root.is_dir():
        raise FileNotFoundError(f"split_root 不存在: {split_root}")

    train_split = split_root / str(data_cfg.get("TRAIN_SPLIT", "train.txt"))
    val_split = split_root / str(data_cfg.get("VAL_SPLIT", "val.txt"))
    test_split = split_root / str(data_cfg.get("TEST_SPLIT", "test.txt"))

    input_features = tuple(str(x) for x in data_cfg.get("INPUT_FEATURES", ["hip"]))
    train_set = ContactTimingWindowDataset(
        label_root=label_root,
        split_file=train_split,
        input_features=input_features,
        window_size=int(data_cfg.get("WINDOW_SIZE", 120)),
        sampling=str(data_cfg.get("TRAIN_SAMPLING", "random")),
        max_items=int(data_cfg.get("MAX_TRAIN_ITEMS", 0)),
        strict=bool(data_cfg.get("STRICT_SPLITS", False)),
    )
    val_set = ContactTimingWindowDataset(
        label_root=label_root,
        split_file=val_split,
        input_features=input_features,
        window_size=int(data_cfg.get("WINDOW_SIZE", 120)),
        sampling=str(data_cfg.get("EVAL_SAMPLING", "center")),
        max_items=int(data_cfg.get("MAX_VAL_ITEMS", 0)),
        strict=bool(data_cfg.get("STRICT_SPLITS", False)),
    )
    test_set = None
    if test_split.is_file():
        test_set = ContactTimingWindowDataset(
            label_root=label_root,
            split_file=test_split,
            input_features=input_features,
            window_size=int(data_cfg.get("WINDOW_SIZE", 120)),
            sampling=str(data_cfg.get("EVAL_SAMPLING", "center")),
            max_items=int(data_cfg.get("MAX_TEST_ITEMS", 0)),
            strict=bool(data_cfg.get("STRICT_SPLITS", False)),
        )

    batch_size = int(train_cfg.get("BATCH_SIZE", 64))
    eval_batch_size = int(train_cfg.get("EVAL_BATCH_SIZE", batch_size))
    num_workers = int(train_cfg.get("NUM_WORKERS", 4))
    train_loader = build_dataloader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = build_dataloader(val_set, batch_size=eval_batch_size, num_workers=num_workers, shuffle=False)
    test_loader = (
        build_dataloader(test_set, batch_size=eval_batch_size, num_workers=num_workers, shuffle=False)
        if test_set is not None
        else None
    )
    print(f"[TimingPred] label_root={label_root}")
    print(f"[TimingPred] train_split={train_split} samples={len(train_set)}")
    print(f"[TimingPred] val_split={val_split} samples={len(val_set)}")
    if test_set is not None:
        print(f"[TimingPred] test_split={test_split} samples={len(test_set)}")
    print(f"[TimingPred] input_features={list(input_features)} window_size={data_cfg.get('WINDOW_SIZE', 120)}")
    print(f"[TimingPred] run_dir={run_dir}")

    device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
    input_dim = int(sum(int(sample.shape[-1]) for sample in [np.zeros((1, 3), dtype=np.float32)]))  # placeholder
    # Infer real input dim from config names rather than hard-coding.
    dummy_feature_dims = {
        "hip": 3,
        "root": 3,
        "root_vel": 3,
        "yaw": 1,
    }
    input_dim = int(sum(dummy_feature_dims[name] for name in input_features))

    model = ContactTimingPredictor(
        input_dim=input_dim,
        hidden_dim=int(model_cfg.get("HIDDEN_DIM", 64)),
        output_dim=int(model_cfg.get("OUTPUT_DIM", 2)),
        num_blocks=int(model_cfg.get("NUM_BLOCKS", 4)),
        kernel_size=int(model_cfg.get("KERNEL_SIZE", 5)),
        dropout=float(model_cfg.get("DROPOUT", 0.0)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("LR", 1e-4)),
        weight_decay=float(train_cfg.get("WEIGHT_DECAY", 1e-2)),
    )

    scheduler = None
    if str(train_cfg.get("LR_SCHEDULER", "none")).strip().lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(train_cfg.get("EPOCHS", 40)), 1),
        )

    pos_weight_list = train_cfg.get("POS_WEIGHT", None)
    pos_weight = None
    if pos_weight_list is not None:
        pos_weight = torch.as_tensor(pos_weight_list, dtype=torch.float32, device=device)

    lambda_tv = float(train_cfg.get("LAMBDA_TEMPORAL_TV", 0.0))
    lambda_acc = float(train_cfg.get("LAMBDA_TEMPORAL_ACC", 0.0))
    event_tolerance = int(eval_cfg.get("EVENT_TOLERANCE_FRAMES", 2))

    writer = None
    if bool(train_cfg.get("USE_TENSORBOARD", True)):
        if SummaryWriter is None:
            raise RuntimeError("TensorBoard is not installed. Install `tensorboard` or disable it.")
        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    best_path = checkpoints_dir / "best.pt"
    latest_path = checkpoints_dir / "latest.pt"
    periodic_dir = checkpoints_dir / "periodic"
    periodic_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val = float("inf")
    history: list[dict[str, float]] = []
    if resume_path:
        resume_state = torch_load_compat(Path(resume_path).expanduser().resolve(), map_location="cpu")
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        if scheduler is not None and "scheduler" in resume_state and resume_state["scheduler"] is not None:
            scheduler.load_state_dict(resume_state["scheduler"])
        start_epoch = int(resume_state.get("epoch", 0))
        best_val = float(resume_state.get("best_val_loss", best_val))
        if history_path.is_file():
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)

    try:
        for epoch in range(start_epoch, int(train_cfg.get("EPOCHS", 40))):
            model.train()
            train_total: list[float] = []
            train_bce: list[float] = []
            train_tv: list[float] = []
            train_acc: list[float] = []
            train_frame_acc: list[float] = []

            for batch in train_loader:
                x = batch["x"].to(device)
                target = batch["contact"].to(device)
                mask = batch["mask"].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                probs = torch.sigmoid(logits)
                loss_bce = masked_bce_with_logits(logits, target, mask, pos_weight=pos_weight)
                loss_tv = masked_temporal_l1(probs, mask, order=1)
                loss_acc = masked_temporal_l1(probs, mask, order=2)
                loss_total = loss_bce + lambda_tv * loss_tv + lambda_acc * loss_acc
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(train_cfg.get("GRAD_CLIP_NORM", 1.0)),
                )
                optimizer.step()

                train_total.append(float(loss_total.item()))
                train_bce.append(float(loss_bce.item()))
                train_tv.append(float(loss_tv.item()))
                train_acc.append(float(loss_acc.item()))
                train_frame_acc.append(frame_binary_accuracy(logits, target, mask))

            if scheduler is not None:
                scheduler.step()

            val_metrics = evaluate(
                model,
                val_loader,
                device,
                pos_weight=pos_weight,
                lambda_tv=lambda_tv,
                lambda_acc=lambda_acc,
                event_tolerance_frames=event_tolerance,
            )

            epoch_info = {
                "epoch": float(epoch + 1),
                "train_loss_total": float(np.mean(train_total)) if train_total else 0.0,
                "train_loss_bce": float(np.mean(train_bce)) if train_bce else 0.0,
                "train_loss_tv": float(np.mean(train_tv)) if train_tv else 0.0,
                "train_loss_acc": float(np.mean(train_acc)) if train_acc else 0.0,
                "train_frame_acc": float(np.mean(train_frame_acc)) if train_frame_acc else 0.0,
                "val_loss_total": val_metrics["loss_total"],
                "val_loss_bce": val_metrics["loss_bce"],
                "val_loss_tv": val_metrics["loss_tv"],
                "val_loss_acc": val_metrics["loss_acc"],
                "val_frame_acc": val_metrics["frame_acc"],
                "val_event_precision": val_metrics["event_precision"],
                "val_event_recall": val_metrics["event_recall"],
                "val_event_f1": val_metrics["event_f1"],
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(epoch_info)
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

            state = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best_val_loss": best_val,
                "metrics": epoch_info,
                "config": cfg,
            }
            torch.save(state, latest_path)

            save_every = int(train_cfg.get("SAVE_EVERY_EPOCHS", 5))
            if save_every > 0 and (epoch + 1) % save_every == 0:
                torch.save(state, periodic_dir / f"epoch_{epoch + 1:04d}.pt")

            if epoch_info["val_loss_total"] < best_val:
                best_val = epoch_info["val_loss_total"]
                state["best_val_loss"] = best_val
                torch.save(state, best_path)

            if writer is not None:
                for key, value in epoch_info.items():
                    if key != "epoch":
                        writer.add_scalar(key, value, epoch + 1)

            print(
                "[TimingPred] "
                f"epoch={epoch + 1:03d} "
                f"train_total={epoch_info['train_loss_total']:.4f} "
                f"val_total={epoch_info['val_loss_total']:.4f} "
                f"val_acc={epoch_info['val_frame_acc']:.4f} "
                f"val_f1={epoch_info['val_event_f1']:.4f}"
            )
    finally:
        if writer is not None:
            writer.close()

    if test_loader is not None and best_path.is_file():
        best_state = torch_load_compat(best_path, map_location="cpu")
        model.load_state_dict(best_state["model"])
        test_metrics = evaluate(
            model.to(device),
            test_loader,
            device,
            pos_weight=pos_weight,
            lambda_tv=lambda_tv,
            lambda_acc=lambda_acc,
            event_tolerance_frames=event_tolerance,
        )
        with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)
        print(
            "[TimingPred][TEST] "
            f"loss={test_metrics['loss_total']:.4f} "
            f"acc={test_metrics['frame_acc']:.4f} "
            f"f1={test_metrics['event_f1']:.4f}"
        )

    print(f"[OK] run_dir: {run_dir}")
    print(f"[OK] best checkpoint: {best_path}")
    return best_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)
    train(cfg, device_override=args.device, resume_path=args.resume)


if __name__ == "__main__":
    main()
