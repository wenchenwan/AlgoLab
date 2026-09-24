"""单轴梯形速度曲线（点到点，起止速度为 0）。

距离不足以加速到 v_max 时退化为三角形曲线。
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class TrapezoidalProfile:
    q0: float
    q1: float
    v_max: float
    a_max: float

    def __post_init__(self) -> None:
        if self.v_max <= 0 or self.a_max <= 0:
            raise ValueError("v_max 和 a_max 必须为正")

    @property
    def distance(self) -> float:
        return abs(self.q1 - self.q0)

    @property
    def v_peak(self) -> float:
        """实际达到的最大速度（三角形曲线时小于 v_max）。"""
        return min(self.v_max, float(np.sqrt(self.distance * self.a_max)))

    @property
    def t_acc(self) -> float:
        return self.v_peak / self.a_max

    @property
    def t_const(self) -> float:
        if self.v_peak == 0:
            return 0.0
        return (self.distance - self.v_peak * self.t_acc) / self.v_peak

    @property
    def duration(self) -> float:
        return 2 * self.t_acc + self.t_const

    def sample(self, t: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
        """返回时刻 t 的 (位置, 速度, 加速度)，t 可为标量或数组，超出 [0, T] 会被截断。"""
        t = np.clip(np.asarray(t, dtype=float), 0.0, self.duration)
        sign = 1.0 if self.q1 >= self.q0 else -1.0
        a, vp, ta, tc = self.a_max, self.v_peak, self.t_acc, self.t_const
        td = t - ta - tc  # 减速段内的时间

        acc_phase = t < ta
        dec_phase = t > ta + tc
        s = np.where(
            acc_phase,
            0.5 * a * t**2,
            np.where(dec_phase, vp * (ta / 2 + tc) + vp * td - 0.5 * a * td**2, vp * (t - ta / 2)),
        )
        v = np.where(acc_phase, a * t, np.where(dec_phase, vp - a * td, vp))
        acc = np.where(acc_phase, a, np.where(dec_phase, -a, 0.0))
        return self.q0 + sign * s, sign * v, sign * acc
