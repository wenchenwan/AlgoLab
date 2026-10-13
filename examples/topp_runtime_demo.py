"""Windows 可运行的 TOPP 流水线 demo。

PowerShell：
    uv run python examples/topp_runtime_demo.py
    uv run python examples/topp_runtime_demo.py --scenario solver-failure
"""

from __future__ import annotations

import argparse
import json

from algolab.trajectory.topp import PathCommand, PathPlanningRequest, ToppRuntime


def make_request(scenario: str) -> PathPlanningRequest:
    commands = (
        PathCommand(1, length=1.2, speed_limit=1.0, acceleration_limit=2.0),
        PathCommand(
            2,
            length=0.4,
            speed_limit=0.8,
            acceleration_limit=1.5,
            fail_solver=scenario == "solver-failure",
        ),
        PathCommand(3, length=2.0, speed_limit=1.2, acceleration_limit=2.5),
        PathCommand(4, length=0.8, speed_limit=0.9, acceleration_limit=1.8),
    )
    return PathPlanningRequest(
        commands=commands,
        max_cycles=1 if scenario == "timeout" else 100,
        abort_requested=(lambda: True) if scenario == "cancel" else None,
        force_stall=scenario == "stalled",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="TOPP 逆向调度模型 demo")
    parser.add_argument(
        "--scenario",
        choices=("success", "solver-failure", "timeout", "cancel", "stalled"),
        default="success",
    )
    args = parser.parse_args()

    runtime = ToppRuntime()
    runtime.prepare()
    result = runtime.run(make_request(args.scenario))
    planning_success = result.ok and (result.value is None or result.value.planning_success)

    payload: dict[str, object] = {
        "scenario": args.scenario,
        "solver": runtime.solver.backend_name,
        "success": planning_success,
        "api_success": result.ok,
        "runtime_state": runtime.state.name,
        "error": result.error,
    }
    if result.value is not None:
        payload.update(
            {
                "pipeline_cycles": result.value.pipeline_cycles,
                "total_duration": result.value.total_duration,
                "segments": [
                    {
                        "curve_id": segment.curve_id,
                        "success": segment.success,
                        "duration": segment.duration,
                        "peak_speed": segment.peak_speed,
                        "error": segment.error,
                    }
                    for segment in result.value.segments
                ],
                "diagnostics": result.value.diagnostics,
            }
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if planning_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
