import importlib


def get_model(cfg, datamodule, phase="train"):
    return get_module(cfg, datamodule)


def get_module(cfg, datamodule):
    modeltype = cfg.model.model_type
    model_module = importlib.import_module(
        f".modeltype.{cfg.model.model_type}", package="mld_clean.models")
    Model = model_module.__getattribute__(f"{modeltype.upper()}")
    return Model(cfg=cfg, datamodule=datamodule)
