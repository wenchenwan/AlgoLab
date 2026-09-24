import numpy as np
import pytest

from algolab.tools import is_rotation_matrix, random_joint_config


def test_ur5_zero_pose(arm):
    # UR5 零位时 TCP 位置（官方 DH 数据推算）
    p = arm.fk(np.zeros(6))[:3, 3]
    np.testing.assert_allclose(p, [-0.81725, -0.19145, -0.005491], atol=1e-6)


def test_fk_returns_valid_pose(arm, rng):
    for _ in range(50):
        T = arm.fk(random_joint_config(-np.pi * np.ones(6), np.pi * np.ones(6), rng))
        assert is_rotation_matrix(T[:3, :3])
        np.testing.assert_allclose(T[3], [0, 0, 0, 1])


def test_fk_rejects_wrong_dof(arm):
    with pytest.raises(ValueError):
        arm.fk(np.zeros(5))
