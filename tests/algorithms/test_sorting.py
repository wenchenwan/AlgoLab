import pytest

from algolab.algorithms.sorting.quick_sort import quick_sort
from algolab.tools import compare, random_ints


@pytest.mark.parametrize(
    "data",
    [[], [1], [2, 1], [3, 3, 3], [5, -1, 0, 5, 2], list(range(10, 0, -1))],
)
def test_quick_sort_edge_cases(data):
    assert quick_sort(data) == sorted(data)


def test_quick_sort_does_not_mutate_input():
    data = [3, 1, 2]
    quick_sort(data)
    assert data == [3, 1, 2]


def test_quick_sort_random():
    compare(quick_sort, sorted, lambda seed: (random_ints(seed % 100, seed=seed),))
