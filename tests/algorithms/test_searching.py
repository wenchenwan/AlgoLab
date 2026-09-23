from algolab.algorithms.searching.binary_search import binary_search
from algolab.tools import sorted_ints


def test_binary_search_found_and_missing():
    items = [1, 3, 5, 7, 9]
    for i, x in enumerate(items):
        assert binary_search(items, x) == i
    assert binary_search(items, 4) == -1
    assert binary_search([], 1) == -1


def test_binary_search_random(seeds):
    for seed in seeds:
        items = sorted(set(sorted_ints(50, seed=seed)))
        for i, x in enumerate(items):
            assert binary_search(items, x) == i
