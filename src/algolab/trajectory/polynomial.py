"""五次多项式插值：给定两端位置/速度/加速度，保证加速度连续（jerk 有界）。"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class QuinticPolynomial:
    """q(t) = sum(c_k t^k), k = 0..5，t ∈ [0, T]。"""

    coeffs: NDArray
    duration: float

    @classmethod
    def from_boundary(
        cls,
        q0: float,
        q1: float,
        duration: float,
        v0: float = 0.0,
        v1: float = 0.0,
        a0: float = 0.0,
        a1: float = 0.0,
    ) -> "QuinticPolynomial":
        T = duration
        A = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [1, T, T**2, T**3, T**4, T**5],
                [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
                [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
            ],
            dtype=float,
        )
        b = np.array([q0, v0, a0, q1, v1, a1], dtype=float)
        return cls(np.linalg.solve(A, b), T)

    def sample(self, t: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
        """返回 (位置, 速度, 加速度)。"""
        t = np.clip(np.asarray(t, dtype=float), 0.0, self.duration)
        c = self.coeffs
        q = np.polynomial.polynomial.polyval(t, c)
        v = np.polynomial.polynomial.polyval(t, np.polynomial.polynomial.polyder(c))
        a = np.polynomial.polynomial.polyval(t, np.polynomial.polynomial.polyder(c, 2))
        return q, v, a
