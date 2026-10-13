import pytest

from algolab.trajectory.topp import (
    FeedrateOptimizerConfig,
    PathCommand,
    PathPlanningRequest,
    RuntimeState,
    ToppErrorCode,
    ToppRuntime,
)
from algolab.trajectory.topp.highs_solver import (
    HighsTwoStageFeedrateSolver,
    linearized_jerk_values,
)
from algolab.trajectory.topp.model import (
    DemoTwoStageFeedrateSolver,
    FeedrateOptimizer,
    FeedrateOptimizerStatus,
    PreparedCurve,
    PreparedCurveBuffer,
)


def commands(*, fail_solver: bool = False):
    return (
        PathCommand(1, 1.0, 1.0, 2.0),
        PathCommand(2, 0.5, 0.8, 1.5, fail_solver=fail_solver),
        PathCommand(3, 1.5, 1.2, 2.5),
        PathCommand(4, 0.7, 0.9, 1.8),
    )


def test_runtime_requires_prepare():
    runtime = ToppRuntime()
    result = runtime.run(PathPlanningRequest(commands()))
    assert not result.ok
    assert result.error == ToppErrorCode.RUNTIME_NOT_READY
    assert runtime.state is RuntimeState.UNPREPARED


def test_successful_overlapping_windows_commit_each_curve_once():
    runtime = ToppRuntime(FeedrateOptimizerConfig(window_size=3, overlap=1))
    runtime.prepare()

    result = runtime.run(PathPlanningRequest(commands()))

    assert result.ok
    assert runtime.state is RuntimeState.SUCCEEDED
    assert result.value is not None
    assert [segment.curve_id for segment in result.value.segments] == [1, 2, 3, 4]
    assert all(segment.success for segment in result.value.segments)
    assert result.value.planning_success
    assert result.value.total_duration > 0
    assert [sample.start_time for sample in result.value.samples[1:]] == [
        sample.end_time for sample in result.value.samples[:-1]
    ]
    assert runtime.solver.abort_requested is None
    assert runtime.runner.abort_requested is None


def test_solver_failure_becomes_failed_segments_not_runtime_failure():
    runtime = ToppRuntime()
    runtime.prepare()

    result = runtime.run(PathPlanningRequest(commands(fail_solver=True)))

    assert result.ok
    assert result.value is not None
    assert not result.value.planning_success
    failed = [segment for segment in result.value.segments if not segment.success]
    assert [segment.curve_id for segment in failed] == [1, 2, 3]
    assert all(segment.error == ToppErrorCode.DEMO_SOLVER_FAILED for segment in failed)


def test_cancel_is_propagated_and_abort_hooks_are_cleared():
    runtime = ToppRuntime()
    runtime.prepare()

    result = runtime.run(PathPlanningRequest(commands(), abort_requested=lambda: True))

    assert not result.ok
    assert result.error == ToppErrorCode.CANCELLED
    assert runtime.state is RuntimeState.FAILED
    assert runtime.solver.abort_requested is None
    assert runtime.runner.abort_requested is None


def test_cycle_limit_is_enforced():
    runtime = ToppRuntime()
    runtime.prepare()

    result = runtime.run(PathPlanningRequest(commands(), max_cycles=1))

    assert not result.ok
    assert result.error == ToppErrorCode.MAX_CYCLES_EXCEEDED


def test_no_progress_is_detected():
    runtime = ToppRuntime()
    runtime.prepare()

    result = runtime.run(PathPlanningRequest(commands(), force_stall=True))

    assert not result.ok
    assert result.error == ToppErrorCode.PIPELINE_NO_PROGRESS


def test_runtime_state_guards_prepare_and_reset():
    runtime = ToppRuntime()
    assert runtime.reset().error == ToppErrorCode.RUNTIME_RESET_NOT_ALLOWED

    assert runtime.prepare().ok
    assert runtime.prepare().error == ToppErrorCode.RUNTIME_ALREADY_PREPARED
    assert runtime.reset().error == ToppErrorCode.RUNTIME_RESET_NOT_ALLOWED

    assert runtime.run(PathPlanningRequest(commands())).ok
    assert runtime.reset().ok
    assert runtime.state is RuntimeState.READY


def test_feedrate_step_marks_a_successful_final_batch():
    buffer = PreparedCurveBuffer()
    buffer.push_curve(PreparedCurve(1, 1.0, 1.0, 2.0))
    buffer.mark_terminal()
    optimizer = FeedrateOptimizer(
        FeedrateOptimizerConfig(window_size=1, overlap=0),
        DemoTwoStageFeedrateSolver(),
        buffer,
    )

    result = optimizer.step()

    assert result.ok
    assert result.value is not None
    assert result.value.terminal
    assert result.value.segments[0].success
    assert optimizer.status is FeedrateOptimizerStatus.OUTPUT_READY_AND_TERMINAL


def test_feedrate_step_converts_solver_error_to_failed_segment():
    buffer = PreparedCurveBuffer()
    buffer.push_curve(PreparedCurve(7, 1.0, 1.0, 2.0, fail_solver=True))
    buffer.mark_terminal()
    optimizer = FeedrateOptimizer(
        FeedrateOptimizerConfig(window_size=1, overlap=0),
        DemoTwoStageFeedrateSolver(),
        buffer,
    )

    result = optimizer.step()

    assert result.ok
    assert result.value is not None
    assert result.value.segments[0].error == ToppErrorCode.DEMO_SOLVER_FAILED
    assert optimizer.status is FeedrateOptimizerStatus.FAILED


@pytest.mark.parametrize(
    ("length", "speed_limit", "acceleration_limit", "expected_peak", "expected_duration"),
    [
        (0.25, 2.0, 2.0, 0.5**0.5, 0.5**0.5),
        (4.0, 1.0, 1.0, 1.0, 5.0),
        (1.5, 2.0, 0.3, 0.45**0.5, 2 * (5.0**0.5)),
    ],
)
def test_representative_trajectory_profiles(
    length, speed_limit, acceleration_limit, expected_peak, expected_duration
):
    runtime = ToppRuntime()
    runtime.prepare()

    result = runtime.run(
        PathPlanningRequest(
            (PathCommand(1, length, speed_limit, acceleration_limit, jerk_limit=1e9),)
        )
    )

    assert result.ok
    assert result.value is not None
    segment = result.value.segments[0]
    assert segment.peak_speed == pytest.approx(expected_peak)
    assert segment.duration == pytest.approx(expected_duration)


def test_runtime_uses_highs_for_both_lp_stages():
    runtime = ToppRuntime()
    runtime.prepare()

    result = runtime.run(PathPlanningRequest((PathCommand(1, 0.25, 2.0, 2.0),)))

    assert result.ok
    assert isinstance(runtime.solver, HighsTwoStageFeedrateSolver)
    stats = runtime.solver.last_stats
    assert stats is not None
    assert stats.variables == 41
    assert stats.stage1_constraints == 40
    assert stats.stage2_constraints == 79
    assert stats.stage1_iterations > 0
    assert stats.stage2_iterations > 0


def test_stage2_jerk_constraint_changes_the_stage1_bang_bang_profile():
    constrained = ToppRuntime()
    unconstrained = ToppRuntime()
    constrained.prepare()
    unconstrained.prepare()

    constrained_result = constrained.run(
        PathPlanningRequest((PathCommand(1, 0.25, 2.0, 2.0, jerk_limit=10.0),))
    )
    unconstrained_result = unconstrained.run(
        PathPlanningRequest((PathCommand(1, 0.25, 2.0, 2.0, jerk_limit=1e9),))
    )

    assert constrained_result.value is not None
    assert unconstrained_result.value is not None
    constrained_segment = constrained_result.value.segments[0]
    unconstrained_segment = unconstrained_result.value.segments[0]
    assert constrained_segment.peak_speed < unconstrained_segment.peak_speed
    assert constrained_segment.duration > unconstrained_segment.duration
    constrained_jerk = linearized_jerk_values(
        constrained_segment.path_positions,
        constrained_segment.squared_speeds,
        constrained_segment.reference_squared_speeds,
    )
    assert max(abs(constrained_jerk)) <= 10.0 + 1e-7
