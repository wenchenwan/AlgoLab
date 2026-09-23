"""计时与简单基准测试。"""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def timer(label: str = "elapsed") -> Iterator[None]:
    """with timer("xxx"): ... 打印代码块耗时。"""
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label}: {(time.perf_counter() - start) * 1000:.3f} ms")


def benchmark(func: Callable[..., Any], *args: Any, repeat: int = 5) -> float:
    """运行 func(*args) repeat 次，返回最短耗时（秒）。"""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        best = min(best, time.perf_counter() - start)
    return best
