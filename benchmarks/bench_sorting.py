"""排序算法基准：uv run python benchmarks/bench_sorting.py"""

from algolab.algorithms.sorting.quick_sort import quick_sort
from algolab.tools import benchmark, random_ints

if __name__ == "__main__":
    for n in (1_000, 10_000, 100_000):
        data = random_ints(n, seed=0)
        t_qs = benchmark(quick_sort, data)
        t_builtin = benchmark(sorted, data)
        print(f"n={n:>7}  quick_sort={t_qs * 1000:8.2f} ms  sorted={t_builtin * 1000:8.2f} ms")
