"""二分查找。"""

from collections.abc import Sequence
from typing import Any


def binary_search(items: Sequence[Any], target: Any) -> int:
    """在升序序列中查找 target，返回下标；不存在返回 -1。

    时间复杂度: O(log n)
    """
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
