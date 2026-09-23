"""自用测试工具：数据生成、对拍、计时/基准。"""

from algolab.tools.checker import compare
from algolab.tools.generators import random_ints, random_string, sorted_ints
from algolab.tools.timer import benchmark, timer

__all__ = ["benchmark", "compare", "random_ints", "random_string", "sorted_ints", "timer"]
