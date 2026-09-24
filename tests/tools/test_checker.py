import numpy as np
import pytest

from algolab.tools import check_trajectory, compare, numerical_jacobian


def test_compare_passes_within_tolerance():
    compare(lambda x: np.sin(x) ** 2, lambda x: 1 - np.cos(x) ** 2, lambda seed: (seed * 0.1,))


def test_compare_reports_seed_on_mismatch():
    with pytest.raises(AssertionError, match="seed=3"):
        compare(lambda x: x, lambda x: x if x != 3 else -1, lambda seed: (seed,))


def test_numerical_jacobian():
    f = lambda x: np.array([x[0] * x[1], np.sin(x[0])])  # noqa: E731
    x = np.array([0.3, 2.0])
    expected = np.array([[2.0, 0.3], [np.cos(0.3), 0.0]])
    np.testing.assert_allclose(numerical_jacobian(f, x), expected, atol=1e-8)


def test_check_trajectory_detects_violation():
    t = np.linspace(0, 1, 1001)
    q = np.column_stack([np.sin(2 * np.pi * t), 0.1 * t])  # 关节 0 峰值速度 2π
    report = check_trajectory(t, q, v_max=[5.0, 1.0])
    assert not report.ok
    assert len(report.violations) == 1 and "关节 0" in report.violations[0]
