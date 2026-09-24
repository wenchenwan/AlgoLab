"""离散 PID（位置式）。

- 微分作用于测量值，避免设定值阶跃引起微分冲击。
- 输出限幅时停止积分（条件积分抗饱和）。
"""

from dataclasses import dataclass, field


@dataclass
class PID:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    dt: float = 0.001
    out_min: float = float("-inf")
    out_max: float = float("inf")

    _integral: float = field(default=0.0, init=False, repr=False)
    _prev_meas: float | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_meas = None

    def update(self, setpoint: float, measurement: float) -> float:
        error = setpoint - measurement
        d_meas = 0.0 if self._prev_meas is None else (measurement - self._prev_meas) / self.dt
        self._prev_meas = measurement

        integral = self._integral + error * self.dt
        u = self.kp * error + self.ki * integral - self.kd * d_meas
        u_sat = min(max(u, self.out_min), self.out_max)
        if u_sat == u or (u > u_sat) != (error > 0):
            # 未饱和，或积分方向有助于退出饱和时才累积
            self._integral = integral
        return u_sat
