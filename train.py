import os
from collections import OrderedDict
from types import MethodType

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

if not hasattr(np, "Inf"):
    np.Inf = np.inf  # type: ignore[attr-defined]

from mld.callback import ProgressLogger
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger


def _normalize_precision(precision):
    if precision is None:
        return 32
    if isinstance(precision, (int, float)):
        return int(precision)
    text = str(precision).strip().lower()
    aliases = {
        "32-true": 32,
        "16-mixed": 16,
        "16-true": 16,
        "64-true": 64,
        "bf16-mixed": "bf16",
        "bf16-true": "bf16",
    }
    if text in aliases:
        return aliases[text]
    if text in {"16", "32", "64"}:
        return int(text)
    if text == "bf16":
        return text
    raise ValueError(
        f"Unsupported precision value '{precision}'. Use one of: 32, 16, 64, bf16, "
        "or aliases like 16-mixed / bf16-mixed."
    )


def _torch_load_compat(path: str, *, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _install_resume_checkpoint_loader(trainer):
    def _compat_load_checkpoint(self, path, map_location=None):
        return _torch_load_compat(path, map_location=map_location)

    trainer.strategy.checkpoint_io.load_checkpoint = MethodType(
        _compat_load_checkpoint,
        trainer.strategy.checkpoint_io,
    )


def _install_resume_state_fallback(trainer, logger):
    original_restore = trainer._checkpoint_connector.restore_optimizers_and_schedulers

    def _safe_restore(self):
        try:
            return original_restore()
        except ValueError as exc:
            msg = str(exc)
            if "parameter group" not in msg:
                raise
            warn_msg = (
                "Resume checkpoint optimizer state does not match the current model parameter groups. "
                "Falling back to weights+loop-state resume and reinitializing optimizer/lr scheduler state."
            )
            if logger is not None:
                logger.warning(warn_msg)
            else:
                print(f"[WARN] {warn_msg}")
            if isinstance(getattr(self, "_loaded_checkpoint", None), dict):
                self._loaded_checkpoint["optimizer_states"] = []
                self._loaded_checkpoint["lr_schedulers"] = []
            return

    trainer._checkpoint_connector.restore_optimizers_and_schedulers = MethodType(
        _safe_restore,
        trainer._checkpoint_connector,
    )


def _load_resume_config(cfg):
    resume = cfg.TRAIN.RESUME
    backcfg = cfg.TRAIN.copy()
    if not os.path.exists(resume):
        raise ValueError("Resume path is not right.")

    if os.path.isfile(resume):
        resume_ckpt = resume
        resume_dir = os.path.dirname(os.path.dirname(resume)) if os.path.basename(
            os.path.dirname(resume)
        ) == "checkpoints" else os.path.dirname(resume)
    else:
        resume_ckpt = None
        resume_dir = resume

    file_list = sorted(os.listdir(resume_dir), reverse=True)
    for item in file_list:
        if item.endswith(".yaml"):
            cfg = OmegaConf.load(os.path.join(resume_dir, item))
            cfg.TRAIN = backcfg
            break

    if resume_ckpt is None:
        checkpoints = sorted(
            os.listdir(os.path.join(resume_dir, "checkpoints")),
            key=lambda x: int(x[6:-5]) if "epoch=" in x else -1,
            reverse=True,
        )
        for checkpoint in checkpoints:
            if "epoch=" in checkpoint:
                resume_ckpt = os.path.join(resume_dir, "checkpoints", checkpoint)
                break

    if resume_ckpt is None:
        raise ValueError("No checkpoint found to resume from.")

    cfg.TRAIN.PRETRAINED = resume_ckpt
    cfg.TRAIN.RESUME = resume

    if os.path.exists(os.path.join(resume_dir, "wandb")):
        wandb_list = sorted(os.listdir(os.path.join(resume_dir, "wandb")), reverse=True)
        for item in wandb_list:
            if "run-" in item:
                cfg.LOGGER.WANDB.RESUME_ID = item.split("-")[-1]
                break

    return cfg


def main():
    cfg = parse_args()
    if cfg.TRAIN.RESUME:
        cfg = _load_resume_config(cfg)
    logger = create_logger(cfg, phase="train")

    pl.seed_everything(cfg.SEED_VALUE)

    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    loggers = []
    if cfg.LOGGER.WANDB.PROJECT:
        wandb_logger = pl_loggers.WandbLogger(
            project=cfg.LOGGER.WANDB.PROJECT,
            offline=cfg.LOGGER.WANDB.OFFLINE,
            id=cfg.LOGGER.WANDB.RESUME_ID,
            save_dir=cfg.FOLDER_EXP,
            version="",
            name=cfg.NAME,
            anonymous=False,
            log_model=False,
        )
        loggers.append(wandb_logger)
    if cfg.LOGGER.TENSORBOARD:
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=cfg.FOLDER_EXP,
            sub_dir="tensorboard",
            version="",
            name="",
        )
        loggers.append(tb_logger)
    logger.info(OmegaConf.to_yaml(cfg))

    datasets = get_datasets(cfg, logger=logger)
    logger.info("datasets module {} initialized".format("".join(cfg.TRAIN.DATASETS)))

    model = get_model(cfg, datasets[0])
    logger.info("model {} loaded".format(cfg.model.model_type))

    metric_monitor = {
        "Train_jf": "recons/text2jfeats/train",
        "Val_jf": "recons/text2jfeats/val",
        "Train_rf": "recons/text2rfeats/train",
        "Val_rf": "recons/text2rfeats/val",
        "APE root": "Metrics/APE_root",
        "APE mean pose": "Metrics/APE_mean_pose",
        "AVE root": "Metrics/AVE_root",
        "AVE mean pose": "Metrics/AVE_mean_pose",
        "R_TOP_1": "Metrics/R_precision_top_1",
        "R_TOP_2": "Metrics/R_precision_top_2",
        "R_TOP_3": "Metrics/R_precision_top_3",
        "gt_R_TOP_1": "Metrics/gt_R_precision_top_1",
        "gt_R_TOP_2": "Metrics/gt_R_precision_top_2",
        "gt_R_TOP_3": "Metrics/gt_R_precision_top_3",
        "FID": "Metrics/FID",
        "gt_FID": "Metrics/gt_FID",
        "Diversity": "Metrics/Diversity",
        "gt_Diversity": "Metrics/gt_Diversity",
        "MM dist": "Metrics/Matching_score",
        "Accuracy": "Metrics/accuracy",
        "gt_Accuracy": "Metrics/gt_accuracy",
    }

    callbacks = [
        pl.callbacks.RichProgressBar(),
        ProgressLogger(metric_monitor=metric_monitor),
        ModelCheckpoint(
            dirpath=os.path.join(cfg.FOLDER_EXP, "checkpoints"),
            filename="{epoch}",
            monitor="step",
            mode="max",
            every_n_epochs=cfg.LOGGER.SACE_CHECKPOINT_EPOCH,
            save_top_k=-1,
            save_last=False,
            save_on_train_epoch_end=True,
        ),
    ]
    logger.info("Callbacks initialized")

    ddp_strategy = "ddp" if len(cfg.DEVICE) > 1 else None
    train_precision = _normalize_precision(getattr(cfg.TRAIN, "PRECISION", 32))
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        strategy=ddp_strategy,
        precision=train_precision,
        default_root_dir=cfg.FOLDER_EXP,
        log_every_n_steps=cfg.LOGGER.VAL_EVERY_STEPS,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
    )
    if cfg.TRAIN.RESUME:
        _install_resume_checkpoint_loader(trainer)
        _install_resume_state_fallback(trainer, logger)
    logger.info("Trainer initialized")

    if cfg.TRAIN.PRETRAINED_VAE:
        logger.info("Loading pretrain vae from {}".format(cfg.TRAIN.PRETRAINED_VAE))
        state_dict = _torch_load_compat(cfg.TRAIN.PRETRAINED_VAE, map_location="cpu")["state_dict"]
        vae_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.split(".")[0] == "vae":
                vae_dict[k.replace("vae.", "")] = v
        model.vae.load_state_dict(vae_dict, strict=True)

    if cfg.TRAIN.PRETRAINED:
        logger.info("Loading pretrain mode from {}".format(cfg.TRAIN.PRETRAINED))
        logger.info("Attention! VAE will be recovered")
        state_dict = _torch_load_compat(cfg.TRAIN.PRETRAINED, map_location="cpu")["state_dict"]
        # Keep IMF teacher from DATA.IMF_CHECKPOINT when finetuning with a new teacher.
        skip_imf = os.environ.get("PRETRAINED_SKIP_IMF", "0") == "1"
        new_state_dict = OrderedDict()
        skipped_imf = 0
        for k, v in state_dict.items():
            if k in ["denoiser.sequence_pos_encoding.pe"]:
                continue
            if skip_imf and (k == "imf_detector" or k.startswith("imf_detector.")):
                skipped_imf += 1
                continue
            new_state_dict[k] = v
        if skip_imf:
            logger.info(
                "PRETRAINED_SKIP_IMF=1: skipped %d imf_detector keys from PRETRAINED",
                skipped_imf,
            )
        model.load_state_dict(new_state_dict, strict=False)

    if cfg.TRAIN.RESUME:
        trainer.fit(model, datamodule=datasets[0], ckpt_path=cfg.TRAIN.PRETRAINED)
    else:
        trainer.fit(model, datamodule=datasets[0])

    checkpoint_folder = trainer.checkpoint_callback.dirpath
    logger.info(f"The checkpoints are stored in {checkpoint_folder}")
    logger.info(f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
    logger.info("Training ends!")


if __name__ == "__main__":
    main()
