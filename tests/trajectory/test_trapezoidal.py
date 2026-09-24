import numpy as np
import pytest

from algolab.tools import check_trajectory
from algolab.trajectory.trapezoidal import TrapezoidalProfile


@pytest.mark.parametrize(
    ("q0", "q1", "v_max", "a_max"),
    [
        (0.0, 2.0, 1.0, 2.0),  # 梯形
        (0.0, 0.1, 1.0, 2.0),  # 距离太短 -> 三角形
        (1.0, -1.5, 0.8, 3.0),  # 反向运动
        (0.5, 0.5, 1.0, 1.0),  # 零位移
    ],
)
def test_profile_endpoints_and_limits(q0, q1, v_max, a_max):
    prof = TrapezoidalProfile(q0, q1, v_max, a_max)
    t = np.linspace(0, prof.duration, 2001)
    q, v, a = prof.sample(t)

    assert q[0] == pytest.approx(q0)
    assert q[-1] == pytest.approx(q1)
    assert v[0] == pytest.approx(0) and v[-1] == pytest.approx(0)
    assert np.all(np.abs(v) <= v_max + 1e-12)
    assert np.all(np.abs(a) <= a_max + 1e-12)


def test_profile_is_consistent_with_numerical_derivative():
    prof = TrapezoidalProfile(0.0, 2.0, 1.0, 2.0)
    t = np.linspace(0, prof.duration, 20001)
    q, v, _ = prof.sample(t)
    np.testing.assert_allclose(np.gradient(q, t), v, atol=1e-3)
    report = check_trajectory(t, q, v_max=1.0)
    assert report.ok, str(report)


def test_triangular_peak_velocity():
    prof = TrapezoidalProfile(0.0, 0.1, 1.0, 2.0)
    assert prof.t_const == pytest.approx(0.0)
    assert prof.v_peak == pytest.approx(np.sqrt(0.1 * 2.0))
