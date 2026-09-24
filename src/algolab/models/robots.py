"""机械臂 DH 模型。新增型号时照 ur5() 的形式加一个工厂函数。"""

import numpy as np

from algolab.kinematics.dh import DHLink, SerialArm


def ur5() -> SerialArm:
    """UR5，标准 DH 参数（Universal Robots 官方数据，单位 m / rad）。"""
    links = [
        DHLink(a=0.0, alpha=np.pi / 2, d=0.089159),
        DHLink(a=-0.425, alpha=0.0, d=0.0),
        DHLink(a=-0.39225, alpha=0.0, d=0.0),
        DHLink(a=0.0, alpha=np.pi / 2, d=0.10915),
        DHLink(a=0.0, alpha=-np.pi / 2, d=0.09465),
        DHLink(a=0.0, alpha=0.0, d=0.0823),
    ]
    return SerialArm(links, name="UR5")
