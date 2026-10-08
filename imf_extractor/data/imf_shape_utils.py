from __future__ import annotations

from typing import Optional

import numpy as np


def canonicalize_imfs(imfs: np.ndarray, expected_dof: Optional[int] = None) -> np.ndarray:
    """将不同保存格式的 IMF 统一成 `[3, dof, T]`。

    支持：
    - `[3, dof, T]`
    - `[3, T, dof]`
    - `[3, T, groups, 3]`
    - `[3, groups, 3, T]`
    """

    if imfs.ndim == 3 and imfs.shape[0] == 3:
        if expected_dof is not None:
            if imfs.shape[1] == int(expected_dof):
                return imfs
            if imfs.shape[2] == int(expected_dof):
                return imfs.transpose(0, 2, 1)
            raise ValueError(
                f"Unsupported IMF shape for expected_dof={expected_dof}: {tuple(imfs.shape)}"
            )

        if imfs.shape[1] % 3 == 0 and (imfs.shape[1] < imfs.shape[2] or imfs.shape[2] % 3 != 0):
            return imfs
        if imfs.shape[2] % 3 == 0:
            return imfs.transpose(0, 2, 1)
        raise ValueError(f"Cannot infer IMF dof from shape: {tuple(imfs.shape)}")

    if imfs.ndim == 4 and imfs.shape[0] == 3 and imfs.shape[-1] == 3:
        t = int(imfs.shape[1])
        dof = int(imfs.shape[2]) * 3
        if expected_dof is not None and dof != int(expected_dof):
            raise ValueError(
                f"Unexpected IMF dof={dof} for expected_dof={expected_dof}, shape={tuple(imfs.shape)}"
            )
        return imfs.reshape(3, t, dof).transpose(0, 2, 1)

    if imfs.ndim == 4 and imfs.shape[0] == 3 and imfs.shape[2] == 3:
        t = int(imfs.shape[3])
        dof = int(imfs.shape[1]) * 3
        if expected_dof is not None and dof != int(expected_dof):
            raise ValueError(
                f"Unexpected IMF dof={dof} for expected_dof={expected_dof}, shape={tuple(imfs.shape)}"
            )
        return imfs.reshape(3, dof, t)

    raise ValueError(f"Unsupported IMF shape: {tuple(imfs.shape)}")
