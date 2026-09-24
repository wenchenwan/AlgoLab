"""标准 DH 参数建模与正运动学（全部为转动关节）。

单关节变换: T_i = Rz(theta) · Tz(d) · Tx(a) · Rx(alpha)，其中 theta = q_i + offset。
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class DHLink:
    a: float
    alpha: float
    d: float
    offset: float = 0.0


def dh_transform(a: float, alpha: float, d: float, theta: float) -> NDArray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


class SerialArm:
    """串联机械臂。base/tool 为基座和工具坐标系的附加变换。"""

    def __init__(
        self,
        links: Sequence[DHLink],
        base: ArrayLike | None = None,
        tool: ArrayLike | None = None,
        name: str = "arm",
    ) -> None:
        self.links = list(links)
        self.base = np.eye(4) if base is None else np.asarray(base, dtype=float)
        self.tool = np.eye(4) if tool is None else np.asarray(tool, dtype=float)
        self.name = name

    @property
    def dof(self) -> int:
        return len(self.links)

    def fk_all(self, q: ArrayLike) -> list[NDArray]:
        """返回 [T_0, T_1, ..., T_n]，T_i 为第 i 个连杆坐标系在世界系下的位姿（T_0 = base）。

        末端工具位姿为 fk_all(q)[-1] @ tool，即 fk(q)。
        """
        q = np.asarray(q, dtype=float)
        if q.shape != (self.dof,):
            raise ValueError(f"q 维度应为 ({self.dof},)，实际为 {q.shape}")
        frames = [self.base]
        T = self.base
        for link, qi in zip(self.links, q, strict=True):
            T = T @ dh_transform(link.a, link.alpha, link.d, qi + link.offset)
            frames.append(T)
        return frames

    def fk(self, q: ArrayLike) -> NDArray:
        """末端（含工具）位姿，4x4 齐次矩阵。"""
        return self.fk_all(q)[-1] @ self.tool
