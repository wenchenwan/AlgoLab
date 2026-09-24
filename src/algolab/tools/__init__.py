"""自用测试工具：随机数据生成、数值对拍、轨迹检查、计时、绘图。"""

from algolab.tools.checker import compare, is_rotation_matrix, numerical_jacobian
from algolab.tools.generators import random_joint_config, random_rotation, random_transform
from algolab.tools.timer import benchmark, timer
from algolab.tools.trajectory_check import TrajectoryReport, check_trajectory

__all__ = [
    "TrajectoryReport",
    "benchmark",
    "check_trajectory",
    "compare",
    "is_rotation_matrix",
    "numerical_jacobian",
    "random_joint_config",
    "random_rotation",
    "random_transform",
    "timer",
]
