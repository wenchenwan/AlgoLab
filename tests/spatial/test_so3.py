import numpy as np
import pytest

from algolab.spatial.so3 import (
    exp_so3,
    log_so3,
    matrix_to_quat,
    matrix_to_rpy,
    quat_to_matrix,
    rot_z,
    rpy_to_matrix,
    skew,
)
from algolab.tools import is_rotation_matrix, random_rotation


def test_skew_matches_cross(rng):
    a, b = rng.normal(size=3), rng.normal(size=3)
    np.testing.assert_allclose(skew(a) @ b, np.cross(a, b))


def test_rpy_roundtrip(rng):
    for _ in range(100):
        rpy = rng.uniform([-np.pi, -np.pi / 2 + 0.01, -np.pi], [np.pi, np.pi / 2 - 0.01, np.pi])
        np.testing.assert_allclose(matrix_to_rpy(rpy_to_matrix(*rpy)), rpy, atol=1e-9)


def test_rpy_gimbal_lock_still_reconstructs_matrix():
    R = rpy_to_matrix(0.3, np.pi / 2, -0.4)
    np.testing.assert_allclose(rpy_to_matrix(*matrix_to_rpy(R)), R, atol=1e-9)


def test_quat_roundtrip(rng):
    for _ in range(200):
        R = random_rotation(rng)
        assert is_rotation_matrix(R)
        np.testing.assert_allclose(quat_to_matrix(matrix_to_quat(R)), R, atol=1e-9)


def test_exp_log_roundtrip(rng):
    for _ in range(200):
        w = rng.normal(size=3)
        w *= rng.uniform(0, np.pi - 1e-3) / np.linalg.norm(w)
        np.testing.assert_allclose(log_so3(exp_so3(w)), w, atol=1e-9)


@pytest.mark.parametrize("theta", [0.0, 1e-10, np.pi])
def test_log_so3_special_angles(theta):
    R = rot_z(theta)
    np.testing.assert_allclose(exp_so3(log_so3(R)), R, atol=1e-9)
