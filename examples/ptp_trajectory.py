"""示例：6 轴同步梯形速度点到点运动，检查限值并画图。

uv run python examples/ptp_trajectory.py
"""

import numpy as np

from algolab.models import ur5
from algolab.tools import check_trajectory
from algolab.tools.plot import plot_joint_trajectory
from algolab.trajectory.trapezoidal import TrapezoidalProfile

V_MAX = np.full(6, np.pi / 2)
A_MAX = np.full(6, np.pi)

if __name__ == "__main__":
    arm = ur5()
    q_start = np.zeros(6)
    q_goal = np.array([1.2, -1.0, 1.5, -0.5, 1.0, 2.0])

    # 各轴独立规划，按最慢轴的时长等比缩放速度/加速度，使各轴同时到达
    profiles = [
        TrapezoidalProfile(a, b, v, acc)
        for a, b, v, acc in zip(q_start, q_goal, V_MAX, A_MAX, strict=True)
    ]
    T = max(p.duration for p in profiles)
    synced = []
    for p in profiles:
        k = p.duration / T if p.duration > 0 else 1.0
        synced.append(TrapezoidalProfile(p.q0, p.q1, p.v_peak * k, p.a_max * k**2))

    t = np.linspace(0, T, 1001)
    q = np.column_stack([p.sample(t)[0] for p in synced])

    print(f"总时长 {T:.3f} s")
    print(check_trajectory(t, q, v_max=V_MAX, a_max=A_MAX))
    print("终点 TCP 位置:", np.round(arm.fk(q[-1])[:3, 3], 4))
    plot_joint_trajectory(t, q, title="UR5 PTP (trapezoidal, synchronized)")
