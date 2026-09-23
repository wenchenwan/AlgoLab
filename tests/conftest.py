import pytest


@pytest.fixture
def seeds() -> range:
    """随机测试统一使用的种子范围。"""
    return range(50)
