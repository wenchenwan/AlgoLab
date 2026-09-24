"""SE(3) 齐次变换。"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def make_transform(R: ArrayLike | None = None, p: ArrayLike | None = None) -> NDArray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def inv_transform(T: ArrayLike) -> NDArray:
    """利用 R^T 求逆，比 np.linalg.inv 更快更准。"""
    T = np.asarray(T, dtype=float)
    R, p = T[:3, :3], T[:3, 3]
    return make_transform(R.T, -R.T @ p)


def transform_points(T: ArrayLike, points: ArrayLike) -> NDArray:
    """对 (N, 3) 或 (3,) 点集做变换。"""
    T = np.asarray(T, dtype=float)
    pts = np.asarray(points, dtype=float)
    return pts @ T[:3, :3].T + T[:3, 3]
