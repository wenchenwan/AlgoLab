"""轨迹质量检查：对离散采样的关节轨迹做差分，检查速度/加速度/jerk 是否超限。

适用于检查规划器输出，或实机/仿真记录的轨迹数据。
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class TrajectoryReport:
    max_vel: NDArray  # 每个关节的最大 |速度|
    max_acc: NDArray
    max_jerk: NDArray
    violations: list[str]

    @property
    def ok(self) -> bool:
        return not self.violations

    def __str__(self) -> str:
        def fmt(a: NDArray) -> str:
            return np.array2string(a, precision=4, suppress_small=True)

        lines = [
            f"max |vel|  = {fmt(self.max_vel)}",
            f"max |acc|  = {fmt(self.max_acc)}",
            f"max |jerk| = {fmt(self.max_jerk)}",
        ]
        lines += ["超限:"] + [f"  - {v}" for v in self.violations] if self.violations else ["OK"]
        return "\n".join(lines)


def check_trajectory(
    t: ArrayLike,
    q: ArrayLike,
    v_max: ArrayLike | None = None,
    a_max: ArrayLike | None = None,
    j_max: ArrayLike | None = None,
    tol: float = 1e-6,
) -> TrajectoryReport:
    """t: (N,) 时间戳；q: (N,) 或 (N, dof) 关节位置。限值可为标量或 (dof,)。

    速度/加速度/jerk 由 np.gradient 差分得到，采样越密越准确。
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[:, None]
    if np.any(np.diff(t) <= 0):
        raise ValueError("时间戳必须严格递增")

    vel = np.gradient(q, t, axis=0)
    acc = np.gradient(vel, t, axis=0)
    jerk = np.gradient(acc, t, axis=0)
    maxima = {
        "速度": np.max(np.abs(vel), axis=0),
        "加速度": np.max(np.abs(acc), axis=0),
        "jerk": np.max(np.abs(jerk), axis=0),
    }

    violations = []
    for name, limit in (("速度", v_max), ("加速度", a_max), ("jerk", j_max)):
        if limit is None:
            continue
        limit = np.broadcast_to(np.asarray(limit, dtype=float), maxima[name].shape)
        for j in np.flatnonzero(maxima[name] > limit * (1 + tol)):
            violations.append(f"关节 {j} {name} {maxima[name][j]:.4g} > 限值 {limit[j]:.4g}")

    return TrajectoryReport(maxima["速度"], maxima["加速度"], maxima["jerk"], violations)
