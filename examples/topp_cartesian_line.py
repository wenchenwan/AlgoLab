"""UR5 笛卡尔直线运动的两阶段 HiGHS TOPP 规划与曲线绘制。

PowerShell:
    uv run python examples/topp_cartesian_line.py
    uv run python examples/topp_cartesian_line.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from algolab.models import ur5
from algolab.spatial.so3 import exp_so3, log_so3
from algolab.trajectory.topp import (
    CartesianLineRequest,
    CartesianToppLimits,
    CartesianToppPlanner,
    CartesianToppResult,
    DiagonalJointDynamics,
)


def plan_demo(grid_points: int = 81) -> tuple[CartesianToppResult, CartesianToppLimits]:
    arm = ur5()
    q_start = np.deg2rad([-25.0, -75.0, 95.0, -105.0, -90.0, 20.0])
    target_pose = arm.fk(q_start).copy()
    target_pose[:3, 3] += np.array([0.04, -0.03, 0.04])
    target_pose[:3, :3] = target_pose[:3, :3] @ exp_so3([0.0, 0.0, np.deg2rad(10.0)])
    limits = CartesianToppLimits(
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

    # 演示参数仅用于验证力矩约束通路；接入实体机器人时替换为辨识后的 RNEA 模型。
    dynamics = DiagonalJointDynamics(
        inertia=np.array([5.0, 4.0, 2.0, 1.0, 0.5, 0.2]),
        gravity_coefficient=np.array([15.0, 12.0, 6.0, 3.0, 1.0, 0.5]),
    )
    result = CartesianToppPlanner(arm, dynamics=dynamics).plan_line(
        CartesianLineRequest(q_start, target_pose, limits, grid_points)
    )
    return result, limits


def validate_result(result: CartesianToppResult, limits: CartesianToppLimits) -> None:
    tolerance = 1e-7
    if result.maximum_cartesian_error > 1e-7:
        raise AssertionError("Cartesian interpolation error exceeded")
    if result.maximum_orientation_error > 1e-7:
        raise AssertionError("Cartesian orientation interpolation error exceeded")
    if np.any(np.abs(result.joint_velocity) > limits.joint_velocity + tolerance):
        raise AssertionError("joint velocity limit exceeded")
    if np.any(np.abs(result.joint_acceleration) > limits.joint_acceleration + tolerance):
        raise AssertionError("joint acceleration limit exceeded")
    if np.any(np.abs(result.linearized_joint_jerk) > limits.joint_jerk + tolerance):
        raise AssertionError("joint jerk limit exceeded")
    if np.max(np.linalg.norm(result.tcp_velocity, axis=1)) > limits.cartesian_velocity + tolerance:
        raise AssertionError("Cartesian velocity limit exceeded")
    if np.any(np.abs(result.tcp_acceleration) > limits.cartesian_acceleration + tolerance):
        raise AssertionError("Cartesian acceleration component limit exceeded")
    if np.max(np.abs(result.linearized_path_jerk)) > limits.cartesian_jerk + tolerance:
        raise AssertionError("path jerk limit exceeded")
    if result.torque is not None and limits.joint_torque is not None:
        if np.any(np.abs(result.torque) > limits.joint_torque + tolerance):
            raise AssertionError("joint torque limit exceeded")


def plot_result(result: CartesianToppResult, limits: CartesianToppLimits, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, 6))
    fig = plt.figure(figsize=(15, 18))
    grid = fig.add_gridspec(5, 2)

    path_axis = fig.add_subplot(grid[0, 0], projection="3d")
    path_axis.plot(*result.desired_tcp_position.T, "k--", linewidth=2, label="desired")
    path_axis.plot(*result.tcp_position.T, color="#2563eb", linewidth=1.5, label="IK/FK")
    path_axis.scatter(*result.tcp_position[0], color="#16a34a", s=40, label="start")
    path_axis.scatter(*result.tcp_position[-1], color="#dc2626", s=40, label="end")
    path_axis.set_title("Cartesian straight-line path")
    path_axis.set_xlabel("x [m]")
    path_axis.set_ylabel("y [m]")
    path_axis.set_zlabel("z [m]")
    path_axis.legend(fontsize="small")

    xyz_axis = fig.add_subplot(grid[0, 1])
    for index, name in enumerate("xyz"):
        xyz_axis.plot(result.time, result.tcp_position[:, index], label=name)
    xyz_axis.set_title("TCP position")
    xyz_axis.set_ylabel("position [m]")
    xyz_axis.legend(ncol=3)

    speed_axis = fig.add_subplot(grid[1, 0])
    speed_axis.plot(result.time, result.path_speed, color="#2563eb", label="path/TCP speed")
    speed_axis.axhline(limits.cartesian_velocity, color="k", linestyle="--", label="limit")
    speed_axis.set_title("Path speed")
    speed_axis.set_ylabel("speed [m/s]")
    speed_axis.legend()

    path_dynamics_axis = fig.add_subplot(grid[1, 1])
    path_dynamics_axis.plot(
        result.acceleration_time,
        result.path_acceleration,
        color="#059669",
        label="path acceleration",
    )
    path_dynamics_axis.axhline(limits.cartesian_acceleration, color="#475569", linestyle="--")
    path_dynamics_axis.axhline(-limits.cartesian_acceleration, color="#475569", linestyle="--")
    jerk_axis = path_dynamics_axis.twinx()
    jerk_axis.plot(
        result.time[1:-1],
        result.linearized_path_jerk,
        color="#7c3aed",
        label="linearized jerk",
    )
    jerk_axis.axhline(limits.cartesian_jerk, color="#7c3aed", linestyle=":")
    jerk_axis.axhline(-limits.cartesian_jerk, color="#7c3aed", linestyle=":")
    path_dynamics_axis.set_title("Path acceleration and stage-2 jerk")
    path_dynamics_axis.set_ylabel("acceleration [m/s2]")
    jerk_axis.set_ylabel("jerk [m/s3]")
    path_dynamics_axis.legend(loc="upper left")
    jerk_axis.legend(loc="upper right")

    joint_position_axis = fig.add_subplot(grid[2, 0])
    joint_velocity_axis = fig.add_subplot(grid[2, 1])
    joint_acceleration_axis = fig.add_subplot(grid[3, 0])
    joint_jerk_axis = fig.add_subplot(grid[3, 1])
    for joint, color in enumerate(colors):
        name = f"q{joint + 1}"
        joint_position_axis.plot(
            result.time, np.rad2deg(result.joint_position[:, joint]), color=color, label=name
        )
        joint_velocity_axis.plot(
            result.time, np.rad2deg(result.joint_velocity[:, joint]), color=color, label=name
        )
        joint_acceleration_axis.plot(
            result.acceleration_time,
            np.rad2deg(result.joint_acceleration[:, joint]),
            color=color,
            label=name,
        )
        joint_jerk_axis.plot(
            result.time[1:-1],
            np.rad2deg(result.linearized_joint_jerk[:, joint]),
            color=color,
            label=name,
        )
    joint_position_axis.set_title("Joint position")
    joint_velocity_axis.set_title("Joint velocity")
    joint_acceleration_axis.set_title("Joint acceleration")
    joint_jerk_axis.set_title("Stage-2 linearized joint jerk")
    joint_position_axis.set_ylabel("angle [deg]")
    joint_velocity_axis.set_ylabel("velocity [deg/s]")
    joint_acceleration_axis.set_ylabel("acceleration [deg/s2]")
    joint_jerk_axis.set_ylabel("jerk [deg/s3]")
    joint_position_axis.legend(ncol=3, fontsize="small")

    torque_axis = fig.add_subplot(grid[4, 0])
    if result.torque is not None:
        for joint, color in enumerate(colors):
            torque_axis.plot(
                result.acceleration_time,
                result.torque[:, joint],
                color=color,
                label=f"tau{joint + 1}",
            )
    torque_axis.set_title("Joint torque model")
    torque_axis.set_ylabel("torque [N m]")
    torque_axis.set_xlabel("time [s]")
    torque_axis.legend(ncol=3, fontsize="small")

    error_axis = fig.add_subplot(grid[4, 1])
    error_axis.semilogy(
        result.time,
        np.maximum(
            np.linalg.norm(result.tcp_position - result.desired_tcp_position, axis=1), 1e-16
        ),
        color="#dc2626",
    )
    orientation_error = np.asarray(
        [
            np.linalg.norm(log_so3(desired @ actual.T))
            for actual, desired in zip(
                result.tcp_rotation, result.desired_tcp_rotation, strict=True
            )
        ]
    )
    error_axis.semilogy(
        result.time,
        np.maximum(orientation_error, 1e-16),
        color="#7c3aed",
        label="orientation [rad]",
    )
    error_axis.set_title("Cartesian pose interpolation error")
    error_axis.set_ylabel("error")
    error_axis.set_xlabel("time [s]")
    error_axis.legend()

    for axis in fig.axes:
        if hasattr(axis, "grid"):
            axis.grid(True, alpha=0.25)
    fig.suptitle("UR5 Cartesian TOPP — HiGHS two-stage linear programming", fontsize=15)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan and plot a Cartesian TOPP motion")
    parser.add_argument("--grid-points", type=int, default=81)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/topp_cartesian_line.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    result, limits = plan_demo(args.grid_points)
    validate_result(result, limits)
    plot_result(result, limits, args.output)
    print(f"duration: {result.duration:.6f} s")
    print(f"max Cartesian error: {result.maximum_cartesian_error:.3e} m")
    print(f"max orientation error: {result.maximum_orientation_error:.3e} rad")
    print(
        f"stage 1: {result.stats.stage1_constraints} rows, "
        f"{result.stats.stage1_iterations} iterations"
    )
    print(
        f"stage 2: {result.stats.stage2_constraints} rows, "
        f"{result.stats.stage2_iterations} iterations"
    )
    print(f"plot: {args.output.resolve()}")

    if args.show:
        import matplotlib.pyplot as plt

        image = plt.imread(args.output)
        plt.figure(figsize=(12, 10))
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
