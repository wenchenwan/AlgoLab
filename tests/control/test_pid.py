import pytest

from algolab.control.pid import PID


def simulate_first_order(pid: PID, setpoint: float, tau: float = 0.1, steps: int = 3000) -> float:
    """一阶惯性对象 tau·y' + y = u，前向欧拉仿真，返回最终输出。"""
    y = 0.0
    for _ in range(steps):
        u = pid.update(setpoint, y)
        y += (u - y) / tau * pid.dt
    return y


def test_pi_eliminates_steady_state_error():
    assert simulate_first_order(PID(kp=2.0, ki=20.0, dt=0.001), 1.0) == pytest.approx(1.0, abs=1e-3)


def test_p_only_has_steady_state_error():
    y = simulate_first_order(PID(kp=2.0, dt=0.001), 1.0)
    assert y == pytest.approx(2.0 / 3.0, abs=1e-3)  # Kp/(1+Kp)


def test_output_saturation_and_anti_windup():
    pid = PID(kp=1.0, ki=100.0, dt=0.001, out_min=-0.5, out_max=0.5)
    for _ in range(1000):
        assert abs(pid.update(10.0, 0.0)) <= 0.5
    # 抗饱和生效时积分不会累积，误差反向后输出应立即打到反向限幅
    # （若积分饱和，积分项 ≈ 1000，输出会一直顶在 +0.5）
    assert pid.update(-10.0, 0.0) == -0.5
