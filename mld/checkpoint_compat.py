"""Load checkpoints that were saved when this package was still named mld_clean."""

from __future__ import annotations

import pickle


def rename_legacy_module(module: str) -> str:
    """Map old package names onto the names used in this repository."""
    replacements = (
        ("mld_clean.", "mld."),
        ("mcm_imf_extractor.", "imf_extractor."),
    )
    for old, new in replacements:
        if module.startswith(old):
            return new + module[len(old):]
    if module == "mld_clean":
        return "mld"
    if module == "mcm_imf_extractor":
        return "imf_extractor"
    return module


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(rename_legacy_module(module), name)


class CompatPickle:
    """Drop-in pickle module for torch.load."""

    Unpickler = _CompatUnpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)


_INSTALLED = False


def install_checkpoint_compat() -> None:
    """Make torch.load understand checkpoints saved under the old package names."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        import torch
    except ImportError:
        return

    original = torch.load

    def _load(*args, **kwargs):
        if kwargs.get("weights_only") is not True:
            kwargs.setdefault("pickle_module", CompatPickle)
        return original(*args, **kwargs)

    _load._mld_compat = True
    torch.load = _load
    _INSTALLED = True
