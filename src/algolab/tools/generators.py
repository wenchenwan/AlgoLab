"""随机测试数据生成器。所有函数都支持 seed，便于复现失败用例。"""

import random
import string


def random_ints(n: int, lo: int = -1000, hi: int = 1000, seed: int | None = None) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(lo, hi) for _ in range(n)]


def sorted_ints(n: int, lo: int = -1000, hi: int = 1000, seed: int | None = None) -> list[int]:
    return sorted(random_ints(n, lo, hi, seed))


def random_string(n: int, alphabet: str = string.ascii_lowercase, seed: int | None = None) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(alphabet) for _ in range(n))
