"""运动学计算耗时：uv run python benchmarks/bench_kinematics.py

控制周期通常 1ms 或更短，关注单次调用耗时是否留有余量。
"""

import numpy as np

from algolab.kinematics.jacobian import geometric_jacobian
from algolab.models import ur5
from algolab.tools import benchmark

if __name__ == "__main__":
    arm = ur5()
    qs = np.random.default_rng(0).uniform(-np.pi, np.pi, size=(1000, 6))

    def run_fk():
        for q in qs:
            arm.fk(q)

    def run_jac():
        for q in qs:
            geometric_jacobian(arm, q)

    for name, fn in (("fk", run_fk), ("geometric_jacobian", run_jac)):
        per_call_us = benchmark(fn) / len(qs) * 1e6
        print(f"{name:<20} {per_call_us:8.2f} us/次")
