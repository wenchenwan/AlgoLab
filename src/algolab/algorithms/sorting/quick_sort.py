"""快速排序。"""

from collections.abc import Sequence
from typing import Any


def quick_sort(items: Sequence[Any]) -> list[Any]:
    """返回排好序的新列表（三路划分，不修改输入）。

    时间复杂度: 平均 O(n log n)，最坏 O(n^2)
    空间复杂度: O(n)
    """
    if len(items) <= 1:
        return list(items)
    pivot = items[len(items) // 2]
    less = [x for x in items if x < pivot]
    equal = [x for x in items if x == pivot]
    greater = [x for x in items if x > pivot]
    return quick_sort(less) + equal + quick_sort(greater)
