import numpy as np
import pytest

from algolab.models import ur5


@pytest.fixture
def rng() -> np.random.Generator:
    """固定种子的随机数生成器，保证测试可复现。"""
    return np.random.default_rng(0)


@pytest.fixture
def arm():
    return ur5()
