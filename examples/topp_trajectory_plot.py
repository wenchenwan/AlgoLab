"""规划几组 TOPP 标量路径并绘制位置、速度、加速度和线性化 jerk。

PowerShell:
    uv run python examples/topp_trajectory_plot.py
    uv run python examples/topp_trajectory_plot.py --show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from algolab.trajectory.topp import PathCommand, PathPlanningRequest, ToppRuntime
from algolab.trajectory.topp.highs_solver import (
    linearized_jerk_values,
    sample_squared_speed_profile,
)


@dataclass(frozen=True)
class SampledPlan:
    name: str
    time: NDArray[np.float64]
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64]
    speed_limit: NDArray[np.float64]
    acceleration_limit: NDArray[np.float64]
    jerk_time: NDArray[np.float64]
    jerk: NDArray[np.float64]
    jerk_limit: NDArray[np.float64]
    segment_ends: tuple[float, ...]
    pipeline_cycles: int

    @property
    def duration(self) -> float:
        return float(self.time[-1])

    @property
    def distance(self) -> float:
        return float(self.position[-1])


TRAJECTORIES: dict[str, tuple[PathCommand, ...]] = {
    # 距离不足以达到速度上限，退化为三角速度曲线。
    "short jerk limited": (PathCommand(1, 0.25, 2.0, 2.0),),
    # 中间存在明显的恒速平台。
    "long trapezoidal": (PathCommand(1, 4.0, 1.0, 1.0),),
    # 较小的加速度上限主导总时长。
    "acceleration limited": (PathCommand(1, 1.5, 2.0, 0.3),),
    # 不同长度和约束的连续路径段，用于验证窗口提交和分段衔接。
    "four segment path": (
        PathCommand(1, 1.2, 1.0, 2.0),
        PathCommand(2, 0.4, 0.8, 1.5),
        PathCommand(3, 2.0, 1.2, 2.5),
        PathCommand(4, 0.8, 0.9, 1.8),
    ),
}


def plan_and_sample(
    name: str, commands: tuple[PathCommand, ...], samples_per_segment: int = 251
) -> SampledPlan:
    """通过恢复的 ToppRuntime 规划，并将每个提交段采样为连续标量路径。"""
    runtime = ToppRuntime()
    prepared = runtime.prepare()
    if not prepared.ok:
        raise RuntimeError(f"{name}: runtime prepare failed: {prepared.error}")

    result = runtime.run(PathPlanningRequest(commands=commands))
    if not result.ok or result.value is None:
        raise RuntimeError(f"{name}: planning failed: {result.error}")
    if not result.value.planning_success:
        raise RuntimeError(f"{name}: one or more committed segments failed")

    command_by_id = {command.command_id: command for command in commands}
    times: list[NDArray[np.float64]] = []
    positions: list[NDArray[np.float64]] = []
    velocities: list[NDArray[np.float64]] = []
    accelerations: list[NDArray[np.float64]] = []
    speed_limits: list[NDArray[np.float64]] = []
    acceleration_limits: list[NDArray[np.float64]] = []
    jerk_times: list[NDArray[np.float64]] = []
    jerks: list[NDArray[np.float64]] = []
    jerk_limits: list[NDArray[np.float64]] = []
    segment_ends: list[float] = []
    time_offset = 0.0
    position_offset = 0.0

    for index, segment in enumerate(result.value.segments):
        command = command_by_id[segment.curve_id]
        local_time, position, velocity, acceleration = sample_squared_speed_profile(
            segment.path_positions,
            segment.squared_speeds,
        )
        if not np.isclose(local_time[-1], segment.duration, rtol=1e-10, atol=1e-10):
            raise AssertionError(
                f"curve {segment.curve_id}: runtime duration {segment.duration} "
                f"!= sampled duration {local_time[-1]}"
            )

        sample_stride = max(1, len(local_time) // samples_per_segment)
        local_time = local_time[::sample_stride]
        position = position[::sample_stride] + position_offset
        velocity = velocity[::sample_stride]
        acceleration = acceleration[::sample_stride]
        if index:
            local_time = local_time[1:]
            position = position[1:]
            velocity = velocity[1:]
            acceleration = acceleration[1:]

        times.append(local_time + time_offset)
        positions.append(position)
        velocities.append(velocity)
        accelerations.append(acceleration)
        speed_limits.append(np.full(local_time.shape, command.speed_limit))
        acceleration_limits.append(np.full(local_time.shape, command.acceleration_limit))
        node_positions = np.asarray(segment.path_positions)
        node_speeds = np.sqrt(np.maximum(np.asarray(segment.squared_speeds), 0.0))
        node_durations = 2.0 * np.diff(node_positions) / (node_speeds[:-1] + node_speeds[1:])
        node_times = np.concatenate(([0.0], np.cumsum(node_durations)))
        segment_jerk = linearized_jerk_values(
            segment.path_positions,
            segment.squared_speeds,
            segment.reference_squared_speeds,
        )
        jerk_times.append(node_times[1:-1] + time_offset)
        jerks.append(segment_jerk)
        jerk_limits.append(np.full(segment_jerk.shape, command.jerk_limit))
        time_offset += segment.duration
        position_offset += command.length
        segment_ends.append(time_offset)

    sampled = SampledPlan(
        name=name,
        time=np.concatenate(times),
        position=np.concatenate(positions),
        velocity=np.concatenate(velocities),
        acceleration=np.concatenate(accelerations),
        speed_limit=np.concatenate(speed_limits),
        acceleration_limit=np.concatenate(acceleration_limits),
        jerk_time=np.concatenate(jerk_times),
        jerk=np.concatenate(jerks),
        jerk_limit=np.concatenate(jerk_limits),
        segment_ends=tuple(segment_ends),
        pipeline_cycles=result.value.pipeline_cycles,
    )
    validate_sampled_plan(sampled, sum(command.length for command in commands))
    return sampled


def validate_sampled_plan(plan: SampledPlan, expected_distance: float) -> None:
    """检查终点、单调性及逐段速度/加速度约束。"""
    tolerance = 1e-10
    if not np.isclose(plan.position[-1], expected_distance, atol=tolerance):
        raise AssertionError(f"{plan.name}: endpoint mismatch")
    if np.any(np.diff(plan.position) < -tolerance):
        raise AssertionError(f"{plan.name}: path position is not monotonic")
    if np.any(np.abs(plan.velocity) > plan.speed_limit + tolerance):
        raise AssertionError(f"{plan.name}: speed limit exceeded")
    if np.any(np.abs(plan.acceleration) > plan.acceleration_limit + tolerance):
        raise AssertionError(f"{plan.name}: acceleration limit exceeded")
    if np.any(np.abs(plan.jerk) > plan.jerk_limit + 1e-7):
        raise AssertionError(f"{plan.name}: linearized jerk limit exceeded")


def plot_plans(plans: list[SampledPlan], output: Path, show: bool) -> None:
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, len(plans), figsize=(18, 11), squeeze=False)
    colors = ("#2563eb", "#059669", "#dc2626")

    for column, plan in enumerate(plans):
        series = (plan.position, plan.velocity, plan.acceleration)
        labels = ("path position s", "path speed ds/dt", "path accel d2s/dt2")
        for row, (values, label, color) in enumerate(zip(series, labels, colors, strict=True)):
            axis = axes[row, column]
            axis.plot(plan.time, values, color=color, linewidth=2.0)
            for boundary in plan.segment_ends[:-1]:
                axis.axvline(boundary, color="#64748b", linestyle=":", alpha=0.7)
            axis.grid(True, alpha=0.25)
            if column == 0:
                axis.set_ylabel(label)

        jerk_axis = axes[3, column]
        jerk_axis.plot(plan.jerk_time, plan.jerk, color="#7c3aed", linewidth=1.8)
        for boundary in plan.segment_ends[:-1]:
            jerk_axis.axvline(boundary, color="#64748b", linestyle=":", alpha=0.7)
        jerk_axis.set_xlabel("time [s]")
        jerk_axis.grid(True, alpha=0.25)
        if column == 0:
            jerk_axis.set_ylabel("linearized jerk")

        axes[0, column].set_title(plan.name)
        axes[1, column].plot(
            plan.time,
            plan.speed_limit,
            color="#0f172a",
            linestyle="--",
            linewidth=1.0,
            label="speed limit",
        )
        axes[2, column].plot(
            plan.time,
            plan.acceleration_limit,
            color="#0f172a",
            linestyle="--",
            linewidth=1.0,
            label="+/- accel limit",
        )
        axes[2, column].plot(
            plan.time,
            -plan.acceleration_limit,
            color="#0f172a",
            linestyle="--",
            linewidth=1.0,
        )
        jerk_axis.plot(
            plan.jerk_time,
            plan.jerk_limit,
            color="#0f172a",
            linestyle="--",
            linewidth=1.0,
            label="+/- jerk limit",
        )
        jerk_axis.plot(
            plan.jerk_time,
            -plan.jerk_limit,
            color="#0f172a",
            linestyle="--",
            linewidth=1.0,
        )
        axes[1, column].legend(loc="best", fontsize="small")
        axes[2, column].legend(loc="best", fontsize="small")
        jerk_axis.legend(loc="best", fontsize="small")

    fig.suptitle("TOPP recovered runtime: scalar path profiles", fontsize=15)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    print(f"plot: {output.resolve()}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan and plot several TOPP trajectories")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/topp_trajectory_profiles.png"),
        help="PNG output path",
    )
    parser.add_argument("--show", action="store_true", help="open an interactive plot window")
    args = parser.parse_args()

    plans = [plan_and_sample(name, commands) for name, commands in TRAJECTORIES.items()]
    print(f"{'trajectory':<24} {'distance':>10} {'duration':>10} {'peak speed':>12} {'cycles':>8}")
    for plan in plans:
        print(
            f"{plan.name:<24} {plan.distance:>10.3f} {plan.duration:>10.3f} "
            f"{np.max(plan.velocity):>12.3f} {plan.pipeline_cycles:>8}"
        )
    plot_plans(plans, args.output, args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
