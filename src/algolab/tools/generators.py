"""随机测试数据生成。传入 seed 或 np.random.Generator，保证失败用例可复现。"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from algolab.spatial.so3 import quat_to_matrix

RngLike = int | np.random.Generator | None


def _rng(seed: RngLike) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def random_joint_config(lower: ArrayLike, upper: ArrayLike, seed: RngLike = None) -> NDArray:
    """在关节限位 [lower, upper] 内均匀采样一组关节角。"""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    return _rng(seed).uniform(lower, upper)


def random_rotation(seed: RngLike = None) -> NDArray:
    """在 SO(3) 上均匀采样旋转矩阵（归一化高斯四元数）。"""
    q = _rng(seed).normal(size=4)
    return quat_to_matrix(q / np.linalg.norm(q))


def random_transform(pos_range: float = 1.0, seed: RngLike = None) -> NDArray:
    rng = _rng(seed)
    T = np.eye(4)
    T[:3, :3] = random_rotation(rng)
    T[:3, 3] = rng.uniform(-pos_range, pos_range, size=3)
    return T
