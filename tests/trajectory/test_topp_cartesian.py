import numpy as np

from algolab.spatial.so3 import exp_so3
from algolab.trajectory.topp import (
    CartesianLineRequest,
    CartesianPathRequest,
    CartesianToppLimits,
    CartesianToppPlanner,
    DiagonalJointDynamics,
)


def _limits() -> CartesianToppLimits:
    return CartesianToppLimits(
        joint_velocity=np.deg2rad(np.full(6, 120.0)),
        joint_acceleration=np.deg2rad(np.full(6, 300.0)),
        joint_jerk=np.deg2rad(np.full(6, 2000.0)),
        cartesian_velocity=0.25,
        cartesian_acceleration=1.0,
        cartesian_jerk=10.0,
        angular_velocity=1.5,
        angular_acceleration=3.0,
        angular_jerk=20.0,
        joint_torque=np.array([60.0, 60.0, 40.0, 20.0, 10.0, 8.0]),
    )


def test_cartesian_line_uses_two_stage_lp_and_respects_limits(arm):
    q_start = np.deg2rad([-25.0, -75.0, 95.0, -105.0, -90.0, 20.0])
    target = arm.fk(q_start).copy()
    target[:3, 3] += np.array([0.04, -0.03, 0.04])
    target[:3, :3] = target[:3, :3] @ exp_so3([0.0, 0.0, np.deg2rad(10.0)])
    dynamics = DiagonalJointDynamics(
        inertia=np.array([5.0, 4.0, 2.0, 1.0, 0.5, 0.2]),
        gravity_coefficient=np.array([15.0, 12.0, 6.0, 3.0, 1.0, 0.5]),
    )
    limits = _limits()

    result = CartesianToppPlanner(arm, dynamics=dynamics).plan_line(
        CartesianLineRequest(q_start, target, limits, grid_points=41)
    )

    assert result.duration > 0.0
    assert np.all(np.diff(result.time) > 0.0)
    assert result.path_speed[0] == 0.0
    assert result.path_speed[-1] == 0.0
    assert result.maximum_cartesian_error < 1e-8
    assert result.maximum_orientation_error < 1e-8
    np.testing.assert_allclose(arm.fk(result.joint_position[-1]), target, atol=1e-8)
    assert np.max(np.linalg.norm(result.tcp_velocity, axis=1)) <= limits.cartesian_velocity + 1e-9
    assert np.all(np.max(np.abs(result.joint_velocity), axis=0) <= limits.joint_velocity + 1e-9)
    assert np.all(
        np.max(np.abs(result.joint_acceleration), axis=0) <= limits.joint_acceleration + 1e-9
    )
    assert np.all(np.max(np.abs(result.tcp_acceleration), axis=0) <= 1.0 + 1e-9)
    assert np.all(np.max(np.abs(result.linearized_joint_jerk), axis=0) <= limits.joint_jerk + 1e-7)
    assert np.max(np.abs(result.linearized_path_jerk)) <= limits.cartesian_jerk + 1e-7
    assert result.torque is not None
    assert np.all(np.max(np.abs(result.torque), axis=0) <= limits.joint_torque + 1e-9)
    assert result.stats.stage2_constraints > result.stats.stage1_constraints
    assert result.stats.stage1_iterations > 0
    assert result.stats.stage2_iterations > 0


def test_complex_cartesian_path_can_activate_each_constraint_family(arm):
    q_start = np.deg2rad([-25.0, -75.0, 95.0, -105.0, -90.0, 20.0])
    start = arm.fk(q_start)
    u = np.linspace(0.0, 1.0, 121)
    positions = start[:3, 3] + np.column_stack(
        (
            0.16 * u,
            0.04 * np.sin(2.0 * np.pi * u),
            0.03 * np.sin(4.0 * np.pi * u) + 0.03 * u,
        )
    )
    angle = np.deg2rad(25.0) * np.sin(np.pi * u) + np.deg2rad(15.0) * u
    rotations = np.asarray([start[:3, :3] @ exp_so3([0.0, 0.0, value]) for value in angle])
    limits = CartesianToppLimits(
        joint_velocity=np.deg2rad([120.0, 120.0, 120.0, 120.0, 120.0, 55.0]),
        joint_acceleration=np.deg2rad([500.0, 500.0, 500.0, 240.0, 500.0, 500.0]),
        joint_jerk=np.deg2rad([6000.0, 6000.0, 3000.0, 6000.0, 6000.0, 6000.0]),
        cartesian_velocity=0.25,
        cartesian_acceleration=1.5,
        cartesian_jerk=20.0,
        angular_velocity=2.0,
        angular_acceleration=8.0,
        angular_jerk=60.0,
    )

    result = CartesianToppPlanner(arm).plan_path(
        CartesianPathRequest(q_start, positions, rotations, limits)
    )

    utilization = np.asarray(
        [
            np.max(np.linalg.norm(result.tcp_velocity, axis=1)) / limits.cartesian_velocity,
            np.max(np.abs(result.joint_velocity) / limits.joint_velocity),
            np.max(np.abs(result.joint_acceleration) / limits.joint_acceleration),
            np.max(np.abs(result.linearized_path_jerk)) / limits.cartesian_jerk,
            np.max(np.abs(result.linearized_joint_jerk) / limits.joint_jerk),
        ]
    )
    assert np.all(utilization <= 1.0 + 1e-8)
    assert np.all(utilization >= 0.999)
    assert result.maximum_cartesian_error < 1e-8
    assert result.maximum_orientation_error < 1e-8
