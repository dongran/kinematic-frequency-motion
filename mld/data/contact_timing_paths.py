from __future__ import annotations

from pathlib import Path
from urllib.parse import quote


def normalize_motion_stem(stem: str) -> str:
    return str(stem).replace("\\", "/").strip().strip("/")


def motion_stem_to_label_stem(stem: str) -> str:
    normalized = normalize_motion_stem(stem)
    return quote(normalized, safe="._-")


def motion_stem_to_label_filename(stem: str) -> str:
    return f"{motion_stem_to_label_stem(stem)}.npz"


def resolve_label_path(label_root: Path, stem: str) -> Path:
    return label_root / motion_stem_to_label_filename(stem)


def motion_path_to_relative_stem(motion_path: Path, *, vecs_root: Path) -> str:
    return motion_path.relative_to(vecs_root).with_suffix("").as_posix()
