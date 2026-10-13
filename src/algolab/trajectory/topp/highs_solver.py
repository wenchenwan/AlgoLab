"""使用 HiGHS 实现的两阶段 TOPP 线性规划求解器。

决策变量为离散网格上的路径速度平方 ``x_i = (ds/dt)_i²``。阶段一求解
速度和加速度约束下的最大进给率；阶段二利用阶段一速度作为参考，把 jerk
约束线性化后再次求解。该结构对应目标库中的 Stage1 ->
Stage2MatrixTransformer -> Stage2 调用链。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt

import highspy
import numpy as np
from numpy.typing import NDArray

from .model import (
    FeedrateOptimizerConfig,
    OptimizationWindow,
    OptimizedProfile,
    Result,
    ToppErrorCode,
    WindowSolveResult,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]


@dataclass(frozen=True)
class HighsSolveStats:
    stage1_objective: float
    stage2_objective: float
    stage1_iterations: int
    stage2_iterations: int
    variables: int
    stage1_constraints: int
    stage2_constraints: int


@dataclass(frozen=True)
class _WindowGrid:
    positions: FloatArray
    speed_upper_squared: FloatArray
    interval_acceleration_limit: FloatArray
    interval_jerk_limit: FloatArray
    curve_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class LpSolution:
    values: FloatArray
    objective: float
    iterations: int


class HighsTwoStageFeedrateSolver:
    """HiGHS 两阶段线性规划求解器。

    Stage 1
        最大化速度平方曲线的路径积分，并满足速度与加速度约束。

    Stage 2
        用 Stage 1 的速度线性化 ``j = v * da/ds``，加入 jerk 约束，
        再次最大化同一个进给率目标。
    """

    def __init__(self, config: FeedrateOptimizerConfig) -> None:
        self.config = config
        self.abort_requested: Callable[[], bool] | None = None
        self.last_stats: HighsSolveStats | None = None

    @property
    def backend_name(self) -> str:
        return f"HiGHS {highspy.Highs().version()} two-stage LP"

    def set_abort_flag(self, abort_requested: Callable[[], bool] | None) -> None:
        self.abort_requested = abort_requested

    def solve_window(self, window: OptimizationWindow) -> Result[WindowSolveResult]:
        self.last_stats = None
        if self._cancelled():
            return Result.failure(ToppErrorCode.CANCELLED)
        if any(curve.fail_solver for curve in window.curves):
            return Result.failure(ToppErrorCode.DEMO_SOLVER_FAILED)

        grid = self._build_grid(window)
        if len(grid.positions) == 1:
            profiles = tuple(
                OptimizedProfile(
                    curve_id=curve.curve_id,
                    duration=0.0,
                    peak_speed=0.0,
                    acceleration=0.0,
                    path_positions=(0.0,),
                    squared_speeds=(0.0,),
                )
                for curve in window.curves
            )
            return Result.success(WindowSolveResult(profiles))

        lower = np.zeros(len(grid.positions), dtype=np.float64)
        upper = grid.speed_upper_squared.copy()
        start_value = min(window.start_squared_speed, upper[0])
        lower[0] = start_value
        upper[0] = start_value
        if window.terminal:
            lower[-1] = 0.0
            upper[-1] = 0.0

        base_rows = self._acceleration_rows(grid)
        path_weights = self._path_integration_weights(grid.positions)
        stage1 = self.solve_lp(path_weights, lower, upper, base_rows)
        if stage1 is None or self._cancelled():
            error = (
                ToppErrorCode.CANCELLED if self._cancelled() else ToppErrorCode.DEMO_SOLVER_FAILED
            )
            return Result.failure(error)

        stage2_rows = [*base_rows, *self._linearized_jerk_rows(grid, stage1.values)]
        stage2 = self.solve_lp(path_weights, lower, upper, stage2_rows)
        if stage2 is None or self._cancelled():
            error = (
                ToppErrorCode.CANCELLED if self._cancelled() else ToppErrorCode.DEMO_SOLVER_FAILED
            )
            return Result.failure(error)

        profiles = self._distribute_profiles(window, grid, stage1.values, stage2.values)
        if profiles is None:
            return Result.failure(ToppErrorCode.DEMO_SOLVER_FAILED)

        self.last_stats = HighsSolveStats(
            stage1_objective=stage1.objective,
            stage2_objective=stage2.objective,
            stage1_iterations=stage1.iterations,
            stage2_iterations=stage2.iterations,
            variables=len(grid.positions),
            stage1_constraints=len(base_rows),
            stage2_constraints=len(stage2_rows),
        )
        return Result.success(
            WindowSolveResult(
                profiles=profiles,
                stage1_objective=stage1.objective,
                stage2_objective=stage2.objective,
                stage1_iterations=stage1.iterations,
                stage2_iterations=stage2.iterations,
            )
        )

    def _cancelled(self) -> bool:
        return self.abort_requested is not None and self.abort_requested()

    def _build_grid(self, window: OptimizationWindow) -> _WindowGrid:
        positions = [0.0]
        speed_upper_squared = [np.inf]
        acceleration_limits: list[float] = []
        jerk_limits: list[float] = []
        curve_ranges: list[tuple[int, int]] = []

        for curve in window.curves:
            start = len(positions) - 1
            speed_upper_squared[start] = min(
                speed_upper_squared[start], curve.speed_limit * curve.speed_limit
            )
            if curve.length > 0:
                intervals = self.config.grid_intervals_per_curve
                spacing = curve.length / intervals
                for _ in range(intervals):
                    positions.append(positions[-1] + spacing)
                    speed_upper_squared.append(curve.speed_limit * curve.speed_limit)
                    acceleration_limits.append(curve.acceleration_limit)
                    jerk_limits.append(curve.jerk_limit)
            curve_ranges.append((start, len(positions) - 1))

        return _WindowGrid(
            positions=np.asarray(positions, dtype=np.float64),
            speed_upper_squared=np.asarray(speed_upper_squared, dtype=np.float64),
            interval_acceleration_limit=np.asarray(acceleration_limits, dtype=np.float64),
            interval_jerk_limit=np.asarray(jerk_limits, dtype=np.float64),
            curve_ranges=tuple(curve_ranges),
        )

    @staticmethod
    def _acceleration_rows(
        grid: _WindowGrid,
    ) -> list[tuple[IntArray, FloatArray, float, float]]:
        rows: list[tuple[IntArray, FloatArray, float, float]] = []
        spacing = np.diff(grid.positions)
        for index, (delta, limit) in enumerate(
            zip(spacing, grid.interval_acceleration_limit, strict=True)
        ):
            bound = 2.0 * limit * delta
            rows.append(
                (
                    np.asarray([index, index + 1], dtype=np.int32),
                    np.asarray([-1.0, 1.0], dtype=np.float64),
                    -bound,
                    bound,
                )
            )
        return rows

    def _linearized_jerk_rows(
        self, grid: _WindowGrid, reference: FloatArray
    ) -> list[tuple[IntArray, FloatArray, float, float]]:
        rows: list[tuple[IntArray, FloatArray, float, float]] = []
        spacing = np.diff(grid.positions)
        for node in range(1, len(grid.positions) - 1):
            ds_left = spacing[node - 1]
            ds_right = spacing[node]
            reference_speed = sqrt(
                max(
                    float(reference[node]),
                    self.config.stage2_speed_floor * self.config.stage2_speed_floor,
                )
            )
            factor = 2.0 * reference_speed / (ds_left + ds_right)
            coefficients = factor * np.asarray(
                [
                    1.0 / (2.0 * ds_left),
                    -1.0 / (2.0 * ds_left) - 1.0 / (2.0 * ds_right),
                    1.0 / (2.0 * ds_right),
                ],
                dtype=np.float64,
            )
            jerk_limit = min(grid.interval_jerk_limit[node - 1], grid.interval_jerk_limit[node])
            rows.append(
                (
                    np.asarray([node - 1, node, node + 1], dtype=np.int32),
                    coefficients,
                    -jerk_limit,
                    jerk_limit,
                )
            )
        return rows

    @staticmethod
    def _path_integration_weights(positions: FloatArray) -> FloatArray:
        spacing = np.diff(positions)
        weights = np.zeros(len(positions), dtype=np.float64)
        weights[:-1] += 0.5 * spacing
        weights[1:] += 0.5 * spacing
        return weights

    @staticmethod
    def solve_lp(
        cost: FloatArray,
        lower: FloatArray,
        upper: FloatArray,
        rows: list[tuple[IntArray, FloatArray, float, float]],
    ) -> LpSolution | None:
        """向 HiGHS 提交一个行式 LP，目标方向固定为最大化。"""
        lp = highspy.HighsLp()
        lp.num_col_ = len(cost)
        lp.num_row_ = len(rows)
        lp.sense_ = highspy.ObjSense.kMaximize
        lp.col_cost_ = cost
        lp.col_lower_ = lower
        lp.col_upper_ = upper
        lp.row_lower_ = np.asarray([row[2] for row in rows], dtype=np.float64)
        lp.row_upper_ = np.asarray([row[3] for row in rows], dtype=np.float64)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise

        starts = [0]
        indices: list[int] = []
        values: list[float] = []
        for row_indices, row_values, _, _ in rows:
            indices.extend(int(index) for index in row_indices)
            values.extend(float(value) for value in row_values)
            starts.append(len(indices))
        lp.a_matrix_.start_ = np.asarray(starts, dtype=np.int32)
        lp.a_matrix_.index_ = np.asarray(indices, dtype=np.int32)
        lp.a_matrix_.value_ = np.asarray(values, dtype=np.float64)

        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        highs.setOptionValue("threads", 1)
        highs.setOptionValue("parallel", "off")
        highs.setOptionValue("solver", "simplex")
        if highs.passModel(lp) == highspy.HighsStatus.kError:
            return None
        if highs.run() == highspy.HighsStatus.kError:
            return None
        if highs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return None

        solution = np.asarray(highs.getSolution().col_value, dtype=np.float64)
        solution[np.abs(solution) < 1e-12] = 0.0
        info = highs.getInfo()
        return LpSolution(
            values=solution,
            objective=float(highs.getObjectiveValue()),
            iterations=int(info.simplex_iteration_count),
        )

    @staticmethod
    def _distribute_profiles(
        window: OptimizationWindow,
        grid: _WindowGrid,
        reference_squared_speeds: FloatArray,
        squared_speeds: FloatArray,
    ) -> tuple[OptimizedProfile, ...] | None:
        profiles: list[OptimizedProfile] = []
        for curve, (start, end) in zip(window.curves, grid.curve_ranges, strict=True):
            local_positions = grid.positions[start : end + 1] - grid.positions[start]
            local_reference = np.maximum(reference_squared_speeds[start : end + 1], 0.0)
            local_squared_speeds = np.maximum(squared_speeds[start : end + 1], 0.0)
            if start == end:
                duration = 0.0
                maximum_acceleration = 0.0
            else:
                spacing = np.diff(local_positions)
                speeds = np.sqrt(local_squared_speeds)
                speed_sums = speeds[:-1] + speeds[1:]
                if np.any(speed_sums <= 1e-12):
                    return None
                duration = float(np.sum(2.0 * spacing / speed_sums))
                acceleration = np.diff(local_squared_speeds) / (2.0 * spacing)
                maximum_acceleration = float(np.max(np.abs(acceleration)))

            profiles.append(
                OptimizedProfile(
                    curve_id=curve.curve_id,
                    duration=duration,
                    peak_speed=float(np.sqrt(np.max(local_squared_speeds))),
                    acceleration=maximum_acceleration,
                    path_positions=tuple(float(value) for value in local_positions),
                    squared_speeds=tuple(float(value) for value in local_squared_speeds),
                    reference_squared_speeds=tuple(float(value) for value in local_reference),
                )
            )
        return tuple(profiles)


def sample_squared_speed_profile(
    path_positions: tuple[float, ...],
    squared_speeds: tuple[float, ...],
    samples_per_interval: int = 8,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """对 LP 的分段常加速度结果采样，返回 ``(t, s, v, a)``。"""
    if len(path_positions) != len(squared_speeds):
        raise ValueError("path_positions 与 squared_speeds 长度必须相同")
    if not path_positions:
        return (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
    if samples_per_interval < 2:
        raise ValueError("samples_per_interval 必须不小于 2")
    if len(path_positions) == 1:
        return (
            np.asarray([0.0]),
            np.asarray([path_positions[0]]),
            np.asarray([sqrt(max(squared_speeds[0], 0.0))]),
            np.asarray([0.0]),
        )

    all_time: list[FloatArray] = []
    all_position: list[FloatArray] = []
    all_velocity: list[FloatArray] = []
    all_acceleration: list[FloatArray] = []
    time_offset = 0.0

    for interval in range(len(path_positions) - 1):
        start_position = path_positions[interval]
        distance = path_positions[interval + 1] - start_position
        start_speed = sqrt(max(squared_speeds[interval], 0.0))
        end_speed = sqrt(max(squared_speeds[interval + 1], 0.0))
        if distance <= 0:
            raise ValueError("path_positions 必须严格递增")
        if start_speed + end_speed <= 1e-12:
            raise ValueError("非零路径区间的两端速度不能同时为零")

        duration = 2.0 * distance / (start_speed + end_speed)
        acceleration = (end_speed - start_speed) / duration
        local_time = np.linspace(0.0, duration, samples_per_interval)
        if interval:
            local_time = local_time[1:]
        all_time.append(local_time + time_offset)
        all_position.append(
            start_position + start_speed * local_time + 0.5 * acceleration * local_time**2
        )
        all_velocity.append(start_speed + acceleration * local_time)
        all_acceleration.append(np.full(local_time.shape, acceleration))
        time_offset += duration

    return (
        np.concatenate(all_time),
        np.concatenate(all_position),
        np.concatenate(all_velocity),
        np.concatenate(all_acceleration),
    )


def linearized_jerk_values(
    path_positions: tuple[float, ...],
    squared_speeds: tuple[float, ...],
    reference_squared_speeds: tuple[float, ...],
    speed_floor: float = 1e-3,
) -> FloatArray:
    """计算阶段二 LP 在内部网格节点使用的线性化路径 jerk。"""
    if not (len(path_positions) == len(squared_speeds) == len(reference_squared_speeds)):
        raise ValueError("位置、阶段一和阶段二 profile 的长度必须相同")
    if len(path_positions) < 3:
        return np.zeros(0, dtype=np.float64)

    positions = np.asarray(path_positions, dtype=np.float64)
    values = np.asarray(squared_speeds, dtype=np.float64)
    reference = np.asarray(reference_squared_speeds, dtype=np.float64)
    spacing = np.diff(positions)
    left_acceleration = (values[1:-1] - values[:-2]) / (2.0 * spacing[:-1])
    right_acceleration = (values[2:] - values[1:-1]) / (2.0 * spacing[1:])
    reference_speed = np.sqrt(np.maximum(reference[1:-1], speed_floor * speed_floor))
    return (
        reference_speed
        * 2.0
        / (spacing[:-1] + spacing[1:])
        * (right_acceleration - left_acceleration)
    )
