"""SO(3) 旋转相关运算。

约定:
- 四元数顺序为 (w, x, y, z)，单位四元数，w >= 0。
- RPY 为绕固定轴 X-Y-Z 依次旋转，即 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def skew(v: ArrayLike) -> NDArray:
    """向量 -> 反对称矩阵，满足 skew(a) @ b == cross(a, b)。"""
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def unskew(S: ArrayLike) -> NDArray:
    """反对称矩阵 -> 向量（skew 的逆）。"""
    S = np.asarray(S, dtype=float)
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def rot_x(a: float) -> NDArray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> NDArray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> NDArray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> NDArray:
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_rpy(R: ArrayLike) -> NDArray:
    """旋转矩阵 -> (roll, pitch, yaw)。pitch = ±pi/2 时为奇异位形，令 roll = 0。"""
    R = np.asarray(R, dtype=float)
    pitch = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
    if np.isclose(np.cos(pitch), 0.0, atol=1e-9):
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def quat_to_matrix(q: ArrayLike) -> NDArray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(R: ArrayLike) -> NDArray:
    """旋转矩阵 -> 四元数 (w, x, y, z)，Shepperd 方法，数值稳定。"""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def exp_so3(omega: ArrayLike) -> NDArray:
    """轴角向量 -> 旋转矩阵（Rodrigues 公式）。"""
    omega = np.asarray(omega, dtype=float)
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3) + skew(omega)
    K = skew(omega / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def log_so3(R: ArrayLike) -> NDArray:
    """旋转矩阵 -> 轴角向量，theta ∈ [0, pi]。"""
    R = np.asarray(R, dtype=float)
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-9:
        return unskew(R - R.T) / 2.0
    if np.pi - theta < 1e-6:
        # 接近 pi 时 sin(theta)≈0，改用 R = 2aa^T - I 求轴
        k = int(np.argmax(np.diag(R)))
        axis = (R[:, k] + np.eye(3)[:, k]) / np.sqrt(2.0 * (1.0 + R[k, k]))
        return theta * axis / np.linalg.norm(axis)
    return theta / (2.0 * np.sin(theta)) * unskew(R - R.T)
