"""复杂三维笛卡尔路径及 TOPP 各类活动约束的可视化。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from algolab.models import ur5
from algolab.spatial.so3 import exp_so3
from algolab.trajectory.topp import (
    CartesianPathRequest,
    CartesianToppLimits,
    CartesianToppPlanner,
    CartesianToppResult,
)


def create_plan() -> tuple[CartesianToppResult, CartesianToppLimits]:
    arm = ur5()
    q_start = np.deg2rad([-25.0, -75.0, 95.0, -105.0, -90.0, 20.0])
    start = arm.fk(q_start)
    u = np.linspace(0.0, 1.0, 121)

    # 三维双频 S 曲线。x 单调前进，y/z 采用不同频率，避免退化成平面圆弧。
    positions = start[:3, 3] + np.column_stack(
        (
            0.16 * u,
            0.04 * np.sin(2.0 * np.pi * u),
            0.03 * np.sin(4.0 * np.pi * u) + 0.03 * u,
        )
    )
    # 同时绕工具 z 轴平滑摆动，激活角运动和第六关节速度约束。
    orientation_angle = np.deg2rad(25.0) * np.sin(np.pi * u) + np.deg2rad(15.0) * u
    rotations = np.asarray(
        [start[:3, :3] @ exp_so3([0.0, 0.0, angle]) for angle in orientation_angle]
    )

    # 限制经过选择，使不同路径区段分别由 TCP 速度、q6 速度、q4 加速度和 q3 jerk 主导。
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
    return result, limits


def utilization_curves(
    result: CartesianToppResult, limits: CartesianToppLimits
) -> dict[str, np.ndarray]:
    return {
        "TCP speed": np.linalg.norm(result.tcp_velocity, axis=1) / limits.cartesian_velocity,
        "joint speed": np.max(np.abs(result.joint_velocity) / limits.joint_velocity, axis=1),
        "path acceleration": np.abs(result.path_acceleration) / limits.cartesian_acceleration,
        "joint acceleration": np.max(
            np.abs(result.joint_acceleration) / limits.joint_acceleration, axis=1
        ),
        "path jerk": np.abs(result.linearized_path_jerk) / limits.cartesian_jerk,
        "joint jerk": np.max(np.abs(result.linearized_joint_jerk) / limits.joint_jerk, axis=1),
    }


def plot_plan(result: CartesianToppResult, limits: CartesianToppLimits, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = utilization_curves(result, limits)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, 6))
    fig = plt.figure(figsize=(15, 15))
    grid = fig.add_gridspec(4, 2)

    path_axis = fig.add_subplot(grid[0, 0], projection="3d")
    path_axis.plot(*result.desired_tcp_position.T, "k--", linewidth=2.5, label="desired")
    points = path_axis.scatter(
        *result.tcp_position.T,
        c=result.path_speed,
        cmap="viridis",
        s=12,
        label="IK/FK samples",
    )
    fig.colorbar(points, ax=path_axis, shrink=0.65, label="path speed [m/s]")
    path_axis.set_title("3D Cartesian S-curve with orientation motion")
    path_axis.set_xlabel("x [m]")
    path_axis.set_ylabel("y [m]")
    path_axis.set_zlabel("z [m]")
    path_axis.legend(fontsize="small")

    tcp_axis = fig.add_subplot(grid[0, 1])
    tcp_speed = np.linalg.norm(result.tcp_velocity, axis=1)
    tcp_axis.plot(result.time, tcp_speed, color="#2563eb", linewidth=2, label="TCP speed")
    tcp_axis.axhline(limits.cartesian_velocity, color="k", linestyle="--", label="TCP speed limit")
    tcp_axis.set_title("TCP speed reaches its limit")
    tcp_axis.set_ylabel("speed [m/s]")
    tcp_axis.legend()

    joint_velocity_axis = fig.add_subplot(grid[1, 0])
    for joint, color in enumerate(colors):
        joint_velocity_axis.plot(
            result.time,
            np.abs(result.joint_velocity[:, joint]) / limits.joint_velocity[joint],
            color=color,
            label=f"q{joint + 1}",
        )
    joint_velocity_axis.axhline(1.0, color="k", linestyle="--", label="limit")
    joint_velocity_axis.set_title("Normalized joint-speed constraints")
    joint_velocity_axis.set_ylabel("utilization")
    joint_velocity_axis.legend(ncol=4, fontsize="small")

    acceleration_axis = fig.add_subplot(grid[1, 1])
    acceleration_axis.plot(
        result.acceleration_time,
        curves["path acceleration"],
        color="#059669",
        linewidth=2,
        label="path acceleration",
    )
    acceleration_axis.plot(
        result.acceleration_time,
        curves["joint acceleration"],
        color="#dc2626",
        linewidth=2,
        label="max joint acceleration",
    )
    acceleration_axis.axhline(1.0, color="k", linestyle="--", label="limit")
    acceleration_axis.set_title("Normalized acceleration constraints")
    acceleration_axis.set_ylabel("utilization")
    acceleration_axis.legend()

    jerk_axis = fig.add_subplot(grid[2, 0])
    jerk_axis.plot(
        result.time[1:-1],
        curves["path jerk"],
        color="#7c3aed",
        linewidth=2,
        label="path jerk",
    )
    jerk_axis.plot(
        result.time[1:-1],
        curves["joint jerk"],
        color="#ea580c",
        linewidth=2,
        label="max joint jerk",
    )
    jerk_axis.axhline(1.0, color="k", linestyle="--", label="limit")
    jerk_axis.set_title("Normalized stage-2 jerk constraints")
    jerk_axis.set_ylabel("utilization")
    jerk_axis.legend()

    profile_axis = fig.add_subplot(grid[2, 1])
    profile_axis.plot(result.time, result.path_position, label="path position")
    profile_axis.plot(result.time, result.path_speed, label="path speed")
    profile_axis.set_title("TOPP path profile")
    profile_axis.set_ylabel("SI units")
    profile_axis.legend()

    joint_axis = fig.add_subplot(grid[3, 0])
    for joint, color in enumerate(colors):
        joint_axis.plot(
            result.time,
            np.rad2deg(result.joint_position[:, joint]),
            color=color,
            label=f"q{joint + 1}",
        )
    joint_axis.set_title("Continuous IK joint trajectory")
    joint_axis.set_ylabel("joint angle [deg]")
    joint_axis.set_xlabel("time [s]")
    joint_axis.legend(ncol=3, fontsize="small")

    maxima = {name: float(np.max(values)) for name, values in curves.items()}
    summary_axis = fig.add_subplot(grid[3, 1])
    labels = list(maxima)
    bars = summary_axis.barh(labels, list(maxima.values()), color=plt.cm.Set2.colors[: len(labels)])
    summary_axis.axvline(1.0, color="k", linestyle="--", label="active limit")
    summary_axis.set_xlim(0.0, 1.08)
    summary_axis.set_title("Maximum constraint utilization")
    summary_axis.set_xlabel("max |value| / limit")
    summary_axis.bar_label(bars, fmt="%.3f", padding=3)
    summary_axis.legend()

    for axis in fig.axes:
        if hasattr(axis, "grid"):
            axis.grid(True, alpha=0.25)
    fig.suptitle("Complex Cartesian TOPP — independently active constraint families", fontsize=15)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot a complex Cartesian TOPP trajectory")
    parser.add_argument(
        "--output", type=Path, default=Path("docs/assets/topp_cartesian_complex.png")
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    result, limits = create_plan()
    curves = utilization_curves(result, limits)
    plot_plan(result, limits, args.output)
    print(f"duration: {result.duration:.6f} s")
    for name, values in curves.items():
        print(f"{name:<20}: {np.max(values):.6f}")
    print(f"max TCP position error: {result.maximum_cartesian_error:.3e} m")
    print(f"max TCP orientation error: {result.maximum_orientation_error:.3e} rad")
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
