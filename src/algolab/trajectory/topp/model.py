"""TOPP 流水线的可执行结构模型。

该模块复现从 AArch64 ``libtopp_core.so`` 恢复出的调度语义，并由 highspy 执行
HiGHS 两阶段线性规划。当前公开输入是标量路径约束，尚未包含目标库完整的机器人
关节、笛卡尔动力学和 B 样条约束，因此不用于生产机器人控制。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from math import sqrt
from typing import Protocol


class ToppErrorCode(IntEnum):
    """已从目标函数确认的错误码，以及一个仅供 demo 使用的求解错误码。"""

    PIPELINE_NO_PROGRESS = 19
    MAX_CYCLES_EXCEEDED = 20
    INPUT_ARENA_ALLOCATION_FAILED = 33
    RUNTIME_ALREADY_PREPARED = 34
    RUNTIME_NOT_READY = 35
    RUNTIME_RESET_NOT_ALLOWED = 36
    CANCELLED = 411
    DEMO_SOLVER_FAILED = 9001


@dataclass(frozen=True)
class Result[T]:
    """原库 ``Result<T>`` 的轻量等价物。"""

    value: T | None = None
    error: int | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def success(cls, value: T) -> Result[T]:
        return cls(value=value)

    @classmethod
    def failure(cls, error: int | ToppErrorCode) -> Result[T]:
        return cls(error=int(error))


class RuntimeState(IntEnum):
    """``ToppRuntime::run`` 中确认的状态数值。"""

    UNPREPARED = 0
    READY = 1
    RUNNING = 2
    SUCCEEDED = 3
    FAILED = 4


class FeedrateOptimizerStatus(IntEnum):
    """``FeedrateOptimizer::step`` 对成员 ``status_`` 的写入值。"""

    IDLE = 0
    WAITING_FOR_INPUT = 1
    OUTPUT_READY = 2
    SOLVING = 3
    TERMINAL = 4
    FAILED = 5
    OUTPUT_READY_AND_TERMINAL = 6


class WindowSelectionKind(Enum):
    WINDOW = auto()
    NEED_INPUT = auto()
    TERMINAL = auto()
    INVALID = auto()


@dataclass(frozen=True)
class PathCommand:
    """Demo 使用的标量路径命令。

    ``length`` 可理解为几何路径参数总长度。``speed_limit`` 和
    ``acceleration_limit`` 对应路径速度/加速度上界。
    """

    command_id: int
    length: float
    speed_limit: float
    acceleration_limit: float
    fail_preprocessing: bool = False
    fail_solver: bool = False
    jerk_limit: float = 10.0

    def __post_init__(self) -> None:
        if self.length < 0:
            raise ValueError("length 不能为负")
        if self.speed_limit <= 0 or self.acceleration_limit <= 0:
            raise ValueError("速度和加速度限制必须为正")
        if self.jerk_limit <= 0:
            raise ValueError("jerk 限制必须为正")


@dataclass(frozen=True)
class PreparedCurve:
    curve_id: int
    length: float
    speed_limit: float
    acceleration_limit: float
    fail_solver: bool = False
    jerk_limit: float = 10.0


@dataclass(frozen=True)
class PreparedCurveFailure:
    curve_id: int
    error: int


@dataclass(frozen=True)
class OptimizedProfile:
    curve_id: int
    duration: float
    peak_speed: float
    acceleration: float
    path_positions: tuple[float, ...] = ()
    squared_speeds: tuple[float, ...] = ()
    reference_squared_speeds: tuple[float, ...] = ()


@dataclass(frozen=True)
class CommittedSegment:
    curve_id: int
    success: bool
    duration: float = 0.0
    peak_speed: float = 0.0
    error: int | None = None
    path_positions: tuple[float, ...] = ()
    squared_speeds: tuple[float, ...] = ()
    reference_squared_speeds: tuple[float, ...] = ()


@dataclass(frozen=True)
class TrajectorySample:
    curve_id: int
    start_time: float
    end_time: float
    peak_speed: float


@dataclass(frozen=True)
class OptimizationWindow:
    curves: tuple[PreparedCurve, ...]
    commit_count: int
    start_squared_speed: float = 0.0
    terminal: bool = False


@dataclass(frozen=True)
class WindowSelection:
    kind: WindowSelectionKind
    window: OptimizationWindow | None = None
    error: int | None = None


@dataclass(frozen=True)
class WindowSolveResult:
    profiles: tuple[OptimizedProfile, ...]
    stage1_objective: float = 0.0
    stage2_objective: float = 0.0
    stage1_iterations: int = 0
    stage2_iterations: int = 0


@dataclass(frozen=True)
class OptimizerStepResult:
    progressed: bool
    segments: tuple[CommittedSegment, ...] = ()
    terminal: bool = False


@dataclass(frozen=True)
class PipelineCycleResult:
    made_progress: bool


@dataclass(frozen=True)
class PathPlanningRequest:
    commands: tuple[PathCommand, ...]
    max_cycles: int = 1_000
    terminal_after_commands: bool = True
    abort_requested: Callable[[], bool] | None = None
    force_stall: bool = False

    def __post_init__(self) -> None:
        if self.max_cycles < 0:
            raise ValueError("max_cycles 不能为负")


@dataclass(frozen=True)
class PathPlanningResult:
    segments: tuple[CommittedSegment, ...]
    samples: tuple[TrajectorySample, ...]
    pipeline_cycles: int
    diagnostics: tuple[str, ...]

    @property
    def total_duration(self) -> float:
        return self.samples[-1].end_time if self.samples else 0.0

    @property
    def planning_success(self) -> bool:
        """流水线成功结束，且所有已提交轨迹段都求解成功。"""
        return all(segment.success for segment in self.segments)


@dataclass(frozen=True)
class FeedrateOptimizerConfig:
    window_size: int = 3
    overlap: int = 1
    grid_intervals_per_curve: int = 40
    stage2_speed_floor: float = 1e-3

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size 必须为正")
        if not 0 <= self.overlap < self.window_size:
            raise ValueError("overlap 必须满足 0 <= overlap < window_size")
        if self.grid_intervals_per_curve < 2:
            raise ValueError("grid_intervals_per_curve 必须不小于 2")
        if self.stage2_speed_floor <= 0:
            raise ValueError("stage2_speed_floor 必须为正")


class PreparedCurveBuffer:
    """保存尚未提交的曲线、失败项及重叠窗口剖面。"""

    def __init__(self) -> None:
        self.curves: deque[PreparedCurve] = deque()
        self.failures: deque[PreparedCurveFailure] = deque()
        self.optimized_profiles: dict[int, OptimizedProfile] = {}
        self.terminal = False

    def reset(self) -> None:
        self.curves.clear()
        self.failures.clear()
        self.optimized_profiles.clear()
        self.terminal = False

    def push_curve(self, curve: PreparedCurve) -> None:
        self.curves.append(curve)

    def push_failure(self, failure: PreparedCurveFailure) -> None:
        self.failures.append(failure)

    def mark_terminal(self) -> None:
        self.terminal = True

    def store_profiles(self, profiles: Sequence[OptimizedProfile]) -> None:
        self.optimized_profiles.update((profile.curve_id, profile) for profile in profiles)

    def commit(self, count: int) -> None:
        for _ in range(min(count, len(self.curves))):
            curve = self.curves.popleft()
            self.optimized_profiles.pop(curve.curve_id, None)

    @property
    def available_count(self) -> int:
        return len(self.curves)


class WindowSelector:
    def __init__(self, config: FeedrateOptimizerConfig) -> None:
        self.config = config

    def select_next_window(self, buffer: PreparedCurveBuffer) -> WindowSelection:
        available = buffer.available_count
        if available == 0:
            kind = (
                WindowSelectionKind.TERMINAL if buffer.terminal else WindowSelectionKind.NEED_INPUT
            )
            return WindowSelection(kind)

        if available < self.config.window_size and not buffer.terminal:
            return WindowSelection(WindowSelectionKind.NEED_INPUT)

        count = min(available, self.config.window_size)
        curves = tuple(list(buffer.curves)[:count])
        commit_count = (
            count if buffer.terminal and count == available else count - self.config.overlap
        )
        if commit_count <= 0:
            return WindowSelection(
                WindowSelectionKind.INVALID,
                error=int(ToppErrorCode.PIPELINE_NO_PROGRESS),
            )
        return WindowSelection(
            WindowSelectionKind.WINDOW,
            OptimizationWindow(
                curves=curves,
                commit_count=commit_count,
                start_squared_speed=self._start_squared_speed(buffer, curves[0]),
                terminal=buffer.terminal and count == available,
            ),
        )

    @staticmethod
    def _start_squared_speed(buffer: PreparedCurveBuffer, curve: PreparedCurve) -> float:
        overlap = buffer.optimized_profiles.get(curve.curve_id)
        if overlap is None or not overlap.squared_speeds:
            return 0.0
        return overlap.squared_speeds[0]


class FeedrateSolver(Protocol):
    abort_requested: Callable[[], bool] | None

    def set_abort_flag(self, abort_requested: Callable[[], bool] | None) -> None: ...

    def solve_window(self, window: OptimizationWindow) -> Result[WindowSolveResult]: ...


class DemoTwoStageFeedrateSolver:
    """保留给单元测试的解析求解替身；ToppRuntime 已改用 HiGHS LP。

    每条曲线按零起止速度的梯形/三角速度律求解。接口边界对应原库
    ``TwoStageFeedrateSolver::solve_window``。
    """

    def __init__(self) -> None:
        self.abort_requested: Callable[[], bool] | None = None

    def set_abort_flag(self, abort_requested: Callable[[], bool] | None) -> None:
        self.abort_requested = abort_requested

    def solve_window(self, window: OptimizationWindow) -> Result[WindowSolveResult]:
        if self.abort_requested is not None and self.abort_requested():
            return Result.failure(ToppErrorCode.CANCELLED)
        if any(curve.fail_solver for curve in window.curves):
            return Result.failure(ToppErrorCode.DEMO_SOLVER_FAILED)

        profiles = tuple(self._solve_curve(curve) for curve in window.curves)
        return Result.success(WindowSolveResult(profiles))

    @staticmethod
    def _solve_curve(curve: PreparedCurve) -> OptimizedProfile:
        if curve.length == 0:
            return OptimizedProfile(curve.curve_id, 0.0, 0.0, curve.acceleration_limit)

        triangular_peak = sqrt(curve.length * curve.acceleration_limit)
        peak = min(curve.speed_limit, triangular_peak)
        acceleration_time = peak / curve.acceleration_limit
        acceleration_distance = peak * acceleration_time
        cruise_time = max(0.0, curve.length - acceleration_distance) / peak
        duration = 2.0 * acceleration_time + cruise_time
        return OptimizedProfile(curve.curve_id, duration, peak, curve.acceleration_limit)


class CommitSegmentBuilder:
    @staticmethod
    def build_failed_segment(curve_id: int, error: int) -> CommittedSegment:
        return CommittedSegment(curve_id=curve_id, success=False, error=error)

    @staticmethod
    def build_committed_segments(
        solved: WindowSolveResult, commit_count: int
    ) -> tuple[CommittedSegment, ...]:
        return tuple(
            CommittedSegment(
                curve_id=profile.curve_id,
                success=True,
                duration=profile.duration,
                peak_speed=profile.peak_speed,
                path_positions=profile.path_positions,
                squared_speeds=profile.squared_speeds,
                reference_squared_speeds=profile.reference_squared_speeds,
            )
            for profile in solved.profiles[:commit_count]
        )


class FeedrateOptimizer:
    """``FeedrateOptimizer::step`` 的可执行语义恢复。"""

    def __init__(
        self,
        config: FeedrateOptimizerConfig,
        solver: FeedrateSolver,
        buffer: PreparedCurveBuffer,
    ) -> None:
        self.config = config
        self.solver = solver
        self.buffer = buffer
        self.selector = WindowSelector(config)
        self.builder = CommitSegmentBuilder()
        self.status = FeedrateOptimizerStatus.IDLE
        self.flow_observations: list[str] = []

    def reset_for_next_run(self) -> None:
        self.buffer.reset()
        self.status = FeedrateOptimizerStatus.IDLE
        self.flow_observations.clear()

    def step(self) -> Result[OptimizerStepResult]:
        # 0x14e70c: 失败输入拥有最高优先级。
        if self.buffer.failures:
            failure = self.buffer.failures.popleft()
            segment = self.builder.build_failed_segment(failure.curve_id, failure.error)
            self.status = FeedrateOptimizerStatus.OUTPUT_READY
            self.flow_observations.append(f"curve {failure.curve_id}: preprocessing failed")
            return Result.success(OptimizerStepResult(progressed=True, segments=(segment,)))

        selection = self.selector.select_next_window(self.buffer)
        if selection.kind is WindowSelectionKind.NEED_INPUT:
            self.status = FeedrateOptimizerStatus.WAITING_FOR_INPUT
            return Result.success(OptimizerStepResult(progressed=False))
        if selection.kind is WindowSelectionKind.TERMINAL:
            self.status = FeedrateOptimizerStatus.TERMINAL
            return Result.success(OptimizerStepResult(progressed=True, terminal=True))
        if selection.kind is WindowSelectionKind.INVALID or selection.window is None:
            self.status = FeedrateOptimizerStatus.FAILED
            return Result.failure(selection.error or ToppErrorCode.PIPELINE_NO_PROGRESS)

        window = selection.window
        self.status = FeedrateOptimizerStatus.SOLVING
        solved = self.solver.solve_window(window)

        # 原实现的求解失败分支为窗口内每条曲线构造失败段。
        if not solved.ok:
            error = solved.error or int(ToppErrorCode.DEMO_SOLVER_FAILED)
            segments = tuple(
                self.builder.build_failed_segment(curve.curve_id, error) for curve in window.curves
            )
            self.buffer.commit(len(window.curves))
            self.status = FeedrateOptimizerStatus.FAILED
            self.flow_observations.append(f"window solve failed: {error}")
            return Result.success(OptimizerStepResult(progressed=True, segments=segments))

        assert solved.value is not None
        self.buffer.store_profiles(solved.value.profiles)
        segments = self.builder.build_committed_segments(solved.value, window.commit_count)
        self.buffer.commit(window.commit_count)
        terminal = self.buffer.terminal and self.buffer.available_count == 0
        self.status = (
            FeedrateOptimizerStatus.OUTPUT_READY_AND_TERMINAL
            if terminal
            else FeedrateOptimizerStatus.OUTPUT_READY
        )
        self.flow_observations.append(
            f"solved {len(window.curves)} curves, committed {window.commit_count}; "
            f"LP iterations={solved.value.stage1_iterations}+{solved.value.stage2_iterations}"
        )
        return Result.success(
            OptimizerStepResult(progressed=True, segments=segments, terminal=terminal)
        )


class PipelineOrchestrator:
    """三阶段流水线：预处理 -> 进给优化 -> 轨迹重建。"""

    def __init__(self, optimizer: FeedrateOptimizer, input_capacity: int = 2) -> None:
        if input_capacity <= 0:
            raise ValueError("input_capacity 必须为正")
        self.optimizer = optimizer
        self.input_capacity = input_capacity
        self.input_queue: deque[PathCommand] = deque()
        self.segment_queue: deque[CommittedSegment] = deque()
        self.segments: list[CommittedSegment] = []
        self.samples: list[TrajectorySample] = []
        self.input_eos = False
        self.optimizer_eos = False
        self.force_stall = False

    def reset_for_next_run(self) -> None:
        self.optimizer.reset_for_next_run()
        self.input_queue.clear()
        self.segment_queue.clear()
        self.segments.clear()
        self.samples.clear()
        self.input_eos = False
        self.optimizer_eos = False
        self.force_stall = False

    @property
    def queue0_is_full(self) -> bool:
        return len(self.input_queue) >= self.input_capacity

    def add_input_item(self, command: PathCommand) -> bool:
        if self.queue0_is_full or self.input_eos:
            return False
        self.input_queue.append(command)
        return True

    def signal_eos(self) -> bool:
        if self.queue0_is_full:
            return False
        self.input_eos = True
        return True

    def run_one_cycle(self) -> Result[PipelineCycleResult]:
        if self.force_stall:
            return Result.success(PipelineCycleResult(False))

        progressed = False

        if self.input_queue:
            command = self.input_queue.popleft()
            if command.fail_preprocessing:
                self.optimizer.buffer.push_failure(
                    PreparedCurveFailure(command.command_id, int(ToppErrorCode.DEMO_SOLVER_FAILED))
                )
            else:
                self.optimizer.buffer.push_curve(
                    PreparedCurve(
                        curve_id=command.command_id,
                        length=command.length,
                        speed_limit=command.speed_limit,
                        acceleration_limit=command.acceleration_limit,
                        fail_solver=command.fail_solver,
                        jerk_limit=command.jerk_limit,
                    )
                )
            progressed = True
        elif self.input_eos and not self.optimizer.buffer.terminal:
            self.optimizer.buffer.mark_terminal()
            progressed = True

        optimizer_step = self.optimizer.step()
        if not optimizer_step.ok:
            return Result.failure(optimizer_step.error or ToppErrorCode.PIPELINE_NO_PROGRESS)
        assert optimizer_step.value is not None
        if optimizer_step.value.progressed:
            progressed = True
        self.segment_queue.extend(optimizer_step.value.segments)
        if optimizer_step.value.terminal:
            self.optimizer_eos = True

        if self.segment_queue:
            segment = self.segment_queue.popleft()
            self.segments.append(segment)
            start = self.samples[-1].end_time if self.samples else 0.0
            self.samples.append(
                TrajectorySample(
                    curve_id=segment.curve_id,
                    start_time=start,
                    end_time=start + segment.duration,
                    peak_speed=segment.peak_speed,
                )
            )
            progressed = True

        return Result.success(PipelineCycleResult(progressed))

    def is_complete(self) -> bool:
        return (
            self.input_eos
            and self.optimizer_eos
            and not self.input_queue
            and not self.segment_queue
            and self.optimizer.buffer.available_count == 0
            and not self.optimizer.buffer.failures
        )


class ToppPipelineRunner:
    """已恢复的 ``ToppPipelineRunner::run`` 调度结构。"""

    def __init__(self, orchestrator: PipelineOrchestrator) -> None:
        self.orchestrator = orchestrator
        self.abort_requested: Callable[[], bool] | None = None

    def _cancelled(self) -> bool:
        return self.abort_requested is not None and self.abort_requested()

    def _run_one_required_cycle(self, maximum: int, current: int) -> Result[int]:
        if self._cancelled():
            return Result.failure(ToppErrorCode.CANCELLED)
        if current >= maximum:
            return Result.failure(ToppErrorCode.MAX_CYCLES_EXCEEDED)
        step = self.orchestrator.run_one_cycle()
        if not step.ok:
            return Result.failure(step.error or ToppErrorCode.PIPELINE_NO_PROGRESS)
        assert step.value is not None
        if not step.value.made_progress:
            return Result.failure(ToppErrorCode.PIPELINE_NO_PROGRESS)
        return Result.success(1)

    def _feed_input_commands(self, request: PathPlanningRequest) -> Result[int]:
        used = 0
        for command in request.commands:
            while self.orchestrator.queue0_is_full:
                cycle = self._run_one_required_cycle(request.max_cycles, used)
                if not cycle.ok:
                    return cycle
                used += cycle.value or 0
            if not self.orchestrator.add_input_item(command):
                return Result.failure(ToppErrorCode.PIPELINE_NO_PROGRESS)
        return Result.success(used)

    def _signal_eos_when_ready(self, maximum: int, current: int) -> Result[int]:
        used = 0
        while self.orchestrator.queue0_is_full:
            cycle = self._run_one_required_cycle(maximum, current + used)
            if not cycle.ok:
                return cycle
            used += cycle.value or 0
        if not self.orchestrator.signal_eos():
            return Result.failure(ToppErrorCode.PIPELINE_NO_PROGRESS)
        return Result.success(used)

    def run(self, request: PathPlanningRequest) -> Result[PathPlanningResult]:
        self.orchestrator.reset_for_next_run()
        self.orchestrator.force_stall = request.force_stall
        self.abort_requested = request.abort_requested

        try:
            fed = self._feed_input_commands(request)
            if not fed.ok:
                return Result.failure(fed.error or ToppErrorCode.PIPELINE_NO_PROGRESS)
            cycles = fed.value or 0

            if self._cancelled():
                return Result.failure(ToppErrorCode.CANCELLED)

            if request.terminal_after_commands:
                eos = self._signal_eos_when_ready(request.max_cycles, cycles)
                if not eos.ok:
                    return Result.failure(eos.error or ToppErrorCode.PIPELINE_NO_PROGRESS)
                cycles += eos.value or 0

            while cycles < request.max_cycles and not self.orchestrator.is_complete():
                if self._cancelled():
                    return Result.failure(ToppErrorCode.CANCELLED)
                step = self.orchestrator.run_one_cycle()
                if not step.ok:
                    return Result.failure(step.error or ToppErrorCode.PIPELINE_NO_PROGRESS)
                assert step.value is not None
                cycles += 1
                if not step.value.made_progress and not self.orchestrator.is_complete():
                    return Result.failure(ToppErrorCode.PIPELINE_NO_PROGRESS)

            if not self.orchestrator.is_complete():
                return Result.failure(ToppErrorCode.MAX_CYCLES_EXCEEDED)

            failed = sum(not segment.success for segment in self.orchestrator.segments)
            diagnostics = (
                f"segments={len(self.orchestrator.segments)}",
                f"failed={failed}",
                f"optimizer_status={self.orchestrator.optimizer.status.name}",
                *self.orchestrator.optimizer.flow_observations,
            )
            return Result.success(
                PathPlanningResult(
                    segments=tuple(self.orchestrator.segments),
                    samples=tuple(self.orchestrator.samples),
                    pipeline_cycles=cycles,
                    diagnostics=diagnostics,
                )
            )
        finally:
            # 0x190eb0：所有正常/错误返回路径都清除 runner 的取消指针。
            self.abort_requested = None


class ToppRuntime:
    """``ToppRuntime::run`` 的 Windows 可执行状态机模型。"""

    def __init__(self, config: FeedrateOptimizerConfig | None = None) -> None:
        from .highs_solver import HighsTwoStageFeedrateSolver

        resolved_config = config or FeedrateOptimizerConfig()
        self.state = RuntimeState.UNPREPARED
        self.solver = HighsTwoStageFeedrateSolver(resolved_config)
        self.buffer = PreparedCurveBuffer()
        self.optimizer = FeedrateOptimizer(resolved_config, self.solver, self.buffer)
        self.orchestrator = PipelineOrchestrator(self.optimizer)
        self.runner = ToppPipelineRunner(self.orchestrator)

    def prepare(self) -> Result[None]:
        if self.state is not RuntimeState.UNPREPARED:
            return Result.failure(ToppErrorCode.RUNTIME_ALREADY_PREPARED)
        self.state = RuntimeState.READY
        return Result.success(None)

    def reset(self) -> Result[None]:
        if self.state not in (RuntimeState.SUCCEEDED, RuntimeState.FAILED):
            return Result.failure(ToppErrorCode.RUNTIME_RESET_NOT_ALLOWED)
        self.orchestrator.reset_for_next_run()
        self.state = RuntimeState.READY
        return Result.success(None)

    def run(self, request: PathPlanningRequest) -> Result[PathPlanningResult]:
        if self.state is not RuntimeState.READY:
            return Result.failure(ToppErrorCode.RUNTIME_NOT_READY)

        self.state = RuntimeState.RUNNING
        self.solver.set_abort_flag(request.abort_requested)
        try:
            result = self.runner.run(request)
        finally:
            # 0x192df0：无论 runner 成功与否都解除 HiGHS 的取消指针。
            self.solver.set_abort_flag(None)

        self.state = RuntimeState.SUCCEEDED if result.ok else RuntimeState.FAILED
        return result
