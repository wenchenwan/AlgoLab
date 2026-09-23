import pytest

from algolab.tools import compare


def test_compare_passes_for_equal_impls():
    compare(lambda x: x * 2, lambda x: x + x, lambda seed: (seed,))


def test_compare_reports_seed_on_mismatch():
    with pytest.raises(AssertionError, match="seed=3"):
        compare(lambda x: x, lambda x: x if x != 3 else -1, lambda seed: (seed,))
