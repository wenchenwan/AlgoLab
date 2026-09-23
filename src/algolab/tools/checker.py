"""对拍工具：用参考实现校验待测实现。"""

from collections.abc import Callable
from typing import Any


def compare(
    impl: Callable[..., Any],
    reference: Callable[..., Any],
    gen: Callable[[int], tuple[Any, ...]],
    rounds: int = 200,
) -> None:
    """用 gen(seed) 生成参数，逐轮比较 impl 与 reference 的输出。

    发现不一致时抛出 AssertionError，并附带 seed 和输入，方便复现。
    """
    for seed in range(rounds):
        args = gen(seed)
        got, want = impl(*args), reference(*args)
        if got != want:
            raise AssertionError(
                f"seed={seed} 不一致\n  输入: {args!r}\n  得到: {got!r}\n  期望: {want!r}"
            )
