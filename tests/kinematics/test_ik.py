import numpy as np

from algolab.kinematics.ik import inverse_kinematics


def test_inverse_kinematics_recovers_a_nearby_pose(arm):
    q_start = np.deg2rad([-25.0, -75.0, 95.0, -105.0, -90.0, 20.0])
    q_target = q_start + np.deg2rad([3.0, -4.0, 2.0, 3.0, -2.0, 2.0])

    result = inverse_kinematics(arm, arm.fk(q_target), q_start)

    assert result.success
    np.testing.assert_allclose(arm.fk(result.q), arm.fk(q_target), atol=1e-7)
