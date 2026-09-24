"""数值校验工具：对拍、数值微分、旋转矩阵合法性检查。"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def compare(
    impl: Callable[..., Any],
    reference: Callable[..., Any],
    gen: Callable[[int], tuple[Any, ...]],
    rounds: int = 200,
    atol: float = 1e-9,
    rtol: float = 1e-7,
) -> None:
    """用 gen(seed) 生成参数，逐轮按容差比较 impl 与 reference 的输出。

    典型用法：新写的解析解 vs 数值解、优化版本 vs 朴素版本、自己的实现 vs 第三方库。
    不一致时抛出 AssertionError，附带 seed 与输入，方便复现。
    """
    for seed in range(rounds):
        args = gen(seed)
        got, want = impl(*args), reference(*args)
        if not np.allclose(got, want, atol=atol, rtol=rtol):
            err = np.max(np.abs(np.asarray(got) - np.asarray(want)))
            raise AssertionError(
                f"seed={seed} 不一致，最大误差 {err:.3e}\n"
                f"  输入: {args!r}\n  得到: {got!r}\n  期望: {want!r}"
            )


def numerical_jacobian(
    f: Callable[[NDArray], ArrayLike], x: ArrayLike, eps: float = 1e-6
) -> NDArray:
    """中心差分求 df/dx，用于校验解析雅可比/梯度。"""
    x = np.asarray(x, dtype=float)
    f0 = np.atleast_1d(np.asarray(f(x), dtype=float))
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        J[:, i] = (np.ravel(f(x + dx)) - np.ravel(f(x - dx))) / (2 * eps)
    return J


def is_rotation_matrix(R: ArrayLike, atol: float = 1e-9) -> bool:
    """检查 R^T R = I 且 det(R) = +1。"""
    R = np.asarray(R, dtype=float)
    return (
        R.shape == (3, 3)
        and np.allclose(R.T @ R, np.eye(3), atol=atol)
        and np.isclose(np.linalg.det(R), 1.0, atol=atol)
    )
