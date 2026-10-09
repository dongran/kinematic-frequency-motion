#!/usr/bin/env python3
"""
Shared Blender Cycles GPU presets (machine-specific without editing render scripts).

Committed profiles live in ``blender_machine_profiles.json``. Optional per-host overrides:
copy ``blender_machine_local.example.json`` to ``blender_machine_local.json`` (gitignored).

Environment (optional default when CLI omits explicit choice):
  BLENDER_RENDER_DEVICE   e.g. cuda, optix, a400
  BLENDER_MACHINE_PROFILE same as above if BLENDER_RENDER_DEVICE unset
"""

from __future__ import annotations

import json
import os
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent

_BUILTIN = frozenset({"cpu", "cuda", "optix"})
_PROFILE_CACHE: dict[str, dict] | None = None


def profiles_path() -> Path:
    return TOOLS_DIR / "blender_machine_profiles.json"


def local_profiles_path() -> Path:
    return TOOLS_DIR / "blender_machine_local.json"


def load_machine_profiles() -> dict[str, dict]:
    global _PROFILE_CACHE
    if _PROFILE_CACHE is not None:
        return _PROFILE_CACHE
    merged: dict[str, dict] = {}
    if profiles_path().is_file():
        raw = json.loads(profiles_path().read_text(encoding="utf-8"))
        merged.update(dict(raw.get("profiles", {})))
    if local_profiles_path().is_file():
        raw = json.loads(local_profiles_path().read_text(encoding="utf-8"))
        merged.update(dict(raw.get("profiles", {})))
    _PROFILE_CACHE = merged
    return merged


def clear_machine_profiles_cache() -> None:
    global _PROFILE_CACHE
    _PROFILE_CACHE = None


def list_profile_ids() -> list[str]:
    return sorted(load_machine_profiles().keys())


def default_render_device_token() -> str:
    for key in ("BLENDER_RENDER_DEVICE", "BLENDER_MACHINE_PROFILE"):
        v = os.environ.get(key, "").strip()
        if v:
            return v.strip().lower()
    return "cuda"


def default_render_device_token_paper_strip() -> str:
    """Paper-strip script historically defaults to cpu unless env overrides."""
    if os.environ.get("BLENDER_RENDER_DEVICE", "").strip() or os.environ.get("BLENDER_MACHINE_PROFILE", "").strip():
        return default_render_device_token()
    return "cpu"


def render_device_choices() -> list[str]:
    return sorted(_BUILTIN | set(load_machine_profiles().keys()))


def normalize_render_device(token: str) -> str:
    t = str(token).strip().lower()
    if t in _BUILTIN:
        return t
    if t in load_machine_profiles():
        return t
    choices = ", ".join(render_device_choices())
    raise ValueError(f"Unknown --render_device {token!r}; expected one of: {choices}")


def describe_profiles_help() -> str:
    ids = list_profile_ids()
    if not ids:
        return "No machine profiles in blender_machine_profiles.json."
    return "Machine profiles: " + ", ".join(ids)
