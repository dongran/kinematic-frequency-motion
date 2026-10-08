from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def normalize_output_groups(
    output_groups: Optional[Sequence[Any]],
    imf_dof: int,
    *,
    default_name: str = "all",
) -> List[Dict[str, Any]]:
    """Normalize output group specs into contiguous channel ranges."""

    total_dof = int(imf_dof)
    if total_dof <= 0:
        raise ValueError(f"imf_dof must be positive, got: {imf_dof}")

    if not output_groups:
        return [{"name": str(default_name), "dof": total_dof, "start": 0, "end": total_dof}]

    groups: List[Dict[str, Any]] = []
    cursor = 0
    for idx, item in enumerate(output_groups):
        if isinstance(item, dict):
            name = str(item.get("name") or f"group{idx}")
            dof = int(item.get("dof", 0))
            extra = dict(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            name = str(item[0])
            dof = int(item[1])
            extra = {}
        else:
            raise ValueError(f"Bad output_groups item at index {idx}: {item!r}")

        if dof <= 0:
            raise ValueError(f"Group {name!r} has non-positive dof: {dof}")

        start = cursor
        end = start + dof
        group = {"name": name, "dof": dof, "start": start, "end": end}
        for key, value in extra.items():
            if key not in group:
                group[key] = value
        groups.append(group)
        cursor = end

    if cursor != total_dof:
        raise ValueError(
            f"output_groups total dof mismatch: sum={cursor}, expected imf_dof={total_dof}"
        )
    return groups


def group_specs_from_model_cfg(model_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    return normalize_output_groups(
        model_cfg.get("output_groups"),
        int(model_cfg.get("imf_dof", 63)),
    )


def resolve_group_weights(
    groups: Sequence[Dict[str, Any]],
    weight_cfg: Optional[Any] = None,
    *,
    strategy: str = "equal",
) -> List[float]:
    """Resolve normalized weights for each group.

    Supports:
    - None + strategy=equal: equal weights
    - None + strategy=by_dof: proportional to group dof
    - dict: keyed by group name
    - list/tuple: same order as groups
    """

    if not groups:
        return []

    strategy = str(strategy or "equal").strip().lower()
    raw_weights: List[float] = []
    if weight_cfg is None:
        if strategy in ("by_dof", "dof", "channel", "channels", "proportional"):
            raw_weights = [float(max(int(g.get("dof", 0)), 0)) for g in groups]
        else:
            raw_weights = [float(g.get("weight", 1.0)) for g in groups]
    elif isinstance(weight_cfg, dict):
        for group in groups:
            raw_weights.append(float(weight_cfg.get(group["name"], group.get("weight", 1.0))))
    elif isinstance(weight_cfg, (list, tuple)):
        if len(weight_cfg) != len(groups):
            raise ValueError(
                f"group_weights length mismatch: got {len(weight_cfg)}, expected {len(groups)}"
            )
        raw_weights = [float(w) for w in weight_cfg]
    else:
        raise ValueError(f"Unsupported group_weights type: {type(weight_cfg)!r}")

    raw_weights = [max(float(w), 0.0) for w in raw_weights]
    total = sum(raw_weights)
    if total <= 0.0:
        raw_weights = [1.0 for _ in groups]
        total = float(len(raw_weights))
    return [float(w) / float(total) for w in raw_weights]
