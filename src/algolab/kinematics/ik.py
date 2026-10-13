"""基于阻尼最小二乘的数值逆运动学。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from algolab.kinematics.dh import SerialArm
from algolab.kinematics.jacobian import geometric_jacobian
from algolab.spatial.so3 import log_so3


@dataclass(frozen=True)
class IkResult:
    q: NDArray[np.float64]
    success: bool
    iterations: int
    position_error: float
    orientation_error: float


def inverse_kinematics(
    arm: SerialArm,
    target: ArrayLike,
    q0: ArrayLike,
    *,
    max_iterations: int = 200,
    position_tolerance: float = 1e-8,
    orientation_tolerance: float = 1e-8,
    damping: float = 1e-4,
    max_step: float = 0.2,
) -> IkResult:
    """求解末端目标位姿，保持输入种子附近的连续解支。

    误差和几何雅可比均表达在世界坐标系，更新量使用阻尼最小二乘计算。
    """
    target = np.asarray(target, dtype=float)
    q = np.asarray(q0, dtype=float).copy()
    if target.shape != (4, 4):
        raise ValueError("target 必须为 4x4 齐次变换")
    if q.shape != (arm.dof,):
        raise ValueError(f"q0 维度应为 ({arm.dof},)")
    if max_iterations <= 0 or damping <= 0 or max_step <= 0:
        raise ValueError("迭代数、阻尼和最大步长必须为正")

    position_norm = float("inf")
    orientation_norm = float("inf")
    for iteration in range(1, max_iterations + 1):
        current = arm.fk(q)
        position_error = target[:3, 3] - current[:3, 3]
        orientation_error = log_so3(target[:3, :3] @ current[:3, :3].T)
        position_norm = float(np.linalg.norm(position_error))
        orientation_norm = float(np.linalg.norm(orientation_error))
        if position_norm <= position_tolerance and orientation_norm <= orientation_tolerance:
            return IkResult(q, True, iteration - 1, position_norm, orientation_norm)

        error = np.concatenate((position_error, orientation_error))
        jacobian = geometric_jacobian(arm, q)
        regularized = jacobian @ jacobian.T + damping * damping * np.eye(6)
        step = jacobian.T @ np.linalg.solve(regularized, error)
        step_norm = float(np.linalg.norm(step))
        if step_norm > max_step:
            step *= max_step / step_norm
        q += step

    return IkResult(q, False, max_iterations, position_norm, orientation_norm)
