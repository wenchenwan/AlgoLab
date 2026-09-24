import numpy as np

from algolab.kinematics.jacobian import geometric_jacobian, manipulability
from algolab.spatial.so3 import unskew
from algolab.tools import numerical_jacobian, random_joint_config


def test_geometric_jacobian_matches_numerical(arm, rng):
    lim = np.pi * np.ones(6)
    for _ in range(20):
        q = random_joint_config(-lim, lim, rng)
        J = geometric_jacobian(arm, q)

        # 线速度部分：直接对末端位置做数值微分
        Jv = numerical_jacobian(lambda x: arm.fk(x)[:3, 3], q)
        np.testing.assert_allclose(J[:3], Jv, atol=1e-6)

        # 角速度部分：dR/dq_i · R^T = skew(w_i)
        R = arm.fk(q)[:3, :3]
        dR = numerical_jacobian(lambda x: arm.fk(x)[:3, :3].ravel(), q)
        Jw = np.column_stack([unskew(dR[:, i].reshape(3, 3) @ R.T) for i in range(6)])
        np.testing.assert_allclose(J[3:], Jw, atol=1e-6)


def test_manipulability_zero_at_singularity(arm):
    # UR5 零位时第 2、3、4 关节轴平行且臂完全伸直 -> 奇异
    assert manipulability(geometric_jacobian(arm, np.zeros(6))) < 1e-6
    q = np.array([0.0, -1.0, 1.5, -0.5, 1.2, 0.0])
    assert manipulability(geometric_jacobian(arm, q)) > 1e-3
