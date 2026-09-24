"""轨迹可视化。matplotlib 为可选依赖，只在调用时导入。"""

import numpy as np
from numpy.typing import ArrayLike


def plot_joint_trajectory(
    t: ArrayLike,
    q: ArrayLike,
    qd: ArrayLike | None = None,
    qdd: ArrayLike | None = None,
    title: str = "",
    show: bool = True,
):
    """画关节位置/速度/加速度曲线；qd、qdd 未给时由差分得到。返回 matplotlib Figure。"""
    import matplotlib.pyplot as plt

    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    q = q[:, None] if q.ndim == 1 else q
    qd = np.gradient(q, t, axis=0) if qd is None else np.asarray(qd).reshape(q.shape)
    qdd = np.gradient(qd, t, axis=0) if qdd is None else np.asarray(qdd).reshape(q.shape)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
    for ax, data, label in zip(
        axes, (q, qd, qdd), ("q [rad]", "qd [rad/s]", "qdd [rad/s^2]"), strict=True
    ):
        for j in range(data.shape[1]):
            ax.plot(t, data[:, j], label=f"J{j + 1}")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", ncol=min(q.shape[1], 6), fontsize="small")
    axes[-1].set_xlabel("t [s]")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig
