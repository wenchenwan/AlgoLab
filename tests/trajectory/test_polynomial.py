import numpy as np
import pytest

from algolab.trajectory.polynomial import QuinticPolynomial


def test_quintic_boundary_conditions():
    poly = QuinticPolynomial.from_boundary(0.2, 1.5, duration=2.0, v0=0.1, v1=-0.3, a0=0.5, a1=0.0)
    q, v, a = poly.sample(np.array([0.0, 2.0]))
    np.testing.assert_allclose(q, [0.2, 1.5], atol=1e-12)
    np.testing.assert_allclose(v, [0.1, -0.3], atol=1e-12)
    np.testing.assert_allclose(a, [0.5, 0.0], atol=1e-12)


def test_quintic_rest_to_rest_peak_velocity():
    # 静止到静止时峰值速度在中点，等于 15/8 · D/T
    poly = QuinticPolynomial.from_boundary(0.0, 1.0, duration=1.0)
    _, v, _ = poly.sample(0.5)
    assert float(v) == pytest.approx(15 / 8)
