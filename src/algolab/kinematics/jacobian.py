"""几何雅可比。"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from algolab.kinematics.dh import SerialArm


def geometric_jacobian(arm: SerialArm, q: ArrayLike) -> NDArray:
    """世界系下的 6xn 几何雅可比，行顺序为 [v; w]（线速度在前）。

    转动关节 i: Jv_i = z_{i-1} × (p_e - p_{i-1}),  Jw_i = z_{i-1}
    """
    frames = arm.fk_all(q)
    p_e = (frames[-1] @ arm.tool)[:3, 3]
    J = np.zeros((6, arm.dof))
    for i in range(arm.dof):
        z = frames[i][:3, 2]
        p = frames[i][:3, 3]
        J[:3, i] = np.cross(z, p_e - p)
        J[3:, i] = z
    return J


def manipulability(J: ArrayLike) -> float:
    """Yoshikawa 可操作度 sqrt(det(J J^T))，趋近 0 表示接近奇异。"""
    J = np.asarray(J, dtype=float)
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
