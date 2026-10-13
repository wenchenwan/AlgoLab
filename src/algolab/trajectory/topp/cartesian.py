"""笛卡尔路径、机器人约束预计算与两阶段 HiGHS TOPP。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from algolab.kinematics.dh import SerialArm
from algolab.kinematics.ik import inverse_kinematics
from algolab.spatial.so3 import exp_so3, log_so3

from .highs_solver import (
    HighsSolveStats,
    HighsTwoStageFeedrateSolver,
    linearized_jerk_values,
)
from .model import FeedrateOptimizerConfig

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
ConstraintRow = tuple[IntArray, FloatArray, float, float]


class PathDynamicsModel(Protocol):
    """将机器人动力学投影到标量路径坐标的接口。"""

    def path_terms(
        self, q: FloatArray, q_s: FloatArray, q_ss: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """返回 ``tau = m(s)*s_ddot + c(s)*s_dot² + g(s)`` 的三项。"""


@dataclass(frozen=True)
class DiagonalJointDynamics:
    """用于演示力矩约束的可配置对角动力学模型。

    生产使用时应以真实 RNEA/机器人辨识模型替换，但 LP 接口无需改变。
    """

    inertia: FloatArray
    gravity_coefficient: FloatArray

    def path_terms(
        self, q: FloatArray, q_s: FloatArray, q_ss: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        inertia = np.asarray(self.inertia, dtype=float)
        gravity = np.asarray(self.gravity_coefficient, dtype=float)
        if inertia.shape != (q.shape[1],) or gravity.shape != (q.shape[1],):
            raise ValueError("动力学参数维度必须等于机器人自由度")
        return inertia * q_s, inertia * q_ss, gravity * np.sin(q)


@dataclass(frozen=True)
class CartesianToppLimits:
    joint_velocity: FloatArray
    joint_acceleration: FloatArray
    joint_jerk: FloatArray
    cartesian_velocity: float
    cartesian_acceleration: float
    cartesian_jerk: float
    angular_velocity: float = 1.5
    angular_acceleration: float = 3.0
    angular_jerk: float = 20.0
    joint_torque: FloatArray | None = None

    def validate(self, dof: int) -> None:
        for name, value in (
            ("joint_velocity", self.joint_velocity),
            ("joint_acceleration", self.joint_acceleration),
            ("joint_jerk", self.joint_jerk),
        ):
            array = np.asarray(value, dtype=float)
            if array.shape != (dof,) or np.any(array <= 0):
                raise ValueError(f"{name} 必须是长度为 {dof} 的正数数组")
        if self.joint_torque is not None:
            torque = np.asarray(self.joint_torque, dtype=float)
            if torque.shape != (dof,) or np.any(torque <= 0):
                raise ValueError(f"joint_torque 必须是长度为 {dof} 的正数数组")
        scalar_limits = (
            self.cartesian_velocity,
            self.cartesian_acceleration,
            self.cartesian_jerk,
            self.angular_velocity,
            self.angular_acceleration,
            self.angular_jerk,
        )
        if any(limit <= 0 for limit in scalar_limits):
            raise ValueError("笛卡尔速度、加速度和 jerk 限制必须为正")


@dataclass(frozen=True)
class CartesianLineRequest:
    q_start: FloatArray
    target_pose: FloatArray
    limits: CartesianToppLimits
    grid_points: int = 81

    def __post_init__(self) -> None:
        if self.grid_points < 5:
            raise ValueError("grid_points 必须不小于 5")


@dataclass(frozen=True)
class CartesianPathRequest:
    """已离散的连续笛卡尔位姿路径；首个位姿必须与 ``q_start`` 对应。"""

    q_start: FloatArray
    positions: FloatArray
    rotations: FloatArray
    limits: CartesianToppLimits


@dataclass(frozen=True)
class CartesianToppResult:
    time: FloatArray
    acceleration_time: FloatArray
    path_position: FloatArray
    path_speed: FloatArray
    path_acceleration: FloatArray
    reference_squared_speed: FloatArray
    squared_speed: FloatArray
    joint_position: FloatArray
    joint_velocity: FloatArray
    joint_acceleration: FloatArray
    tcp_position: FloatArray
    desired_tcp_position: FloatArray
    tcp_rotation: FloatArray
    desired_tcp_rotation: FloatArray
    tcp_velocity: FloatArray
    tcp_acceleration: FloatArray
    linearized_path_jerk: FloatArray
    linearized_joint_jerk: FloatArray
    torque: FloatArray | None
    stats: HighsSolveStats

    @property
    def duration(self) -> float:
        return float(self.time[-1])

    @property
    def maximum_cartesian_error(self) -> float:
        return float(np.max(np.linalg.norm(self.tcp_position - self.desired_tcp_position, axis=1)))

    @property
    def maximum_orientation_error(self) -> float:
        errors = [
            np.linalg.norm(log_so3(desired @ actual.T))
            for actual, desired in zip(self.tcp_rotation, self.desired_tcp_rotation, strict=True)
        ]
        return float(np.max(errors))


@dataclass(frozen=True)
class _PreparedCartesianPath:
    s: FloatArray
    desired_position: FloatArray
    desired_rotation: FloatArray
    orientation_coordinate: FloatArray
    q: FloatArray
    q_s: FloatArray
    q_ss: FloatArray
    p_s: FloatArray
    p_ss: FloatArray
    rotation_s: FloatArray
    rotation_ss: FloatArray


class CartesianToppPlanner:
    """笛卡尔几何路径、约束预计算和两阶段 TOPP。"""

    def __init__(
        self,
        arm: SerialArm,
        *,
        dynamics: PathDynamicsModel | None = None,
        numerical_config: FeedrateOptimizerConfig | None = None,
    ) -> None:
        self.arm = arm
        self.dynamics = dynamics
        self.config = numerical_config or FeedrateOptimizerConfig()
        self.lp_solver = HighsTwoStageFeedrateSolver(self.config)

    def plan_line(self, request: CartesianLineRequest) -> CartesianToppResult:
        request.limits.validate(self.arm.dof)
        return self._solve_path(self._prepare_line_path(request), request.limits)

    def plan_path(self, request: CartesianPathRequest) -> CartesianToppResult:
        """规划任意已采样的连续笛卡尔位姿路径。"""
        request.limits.validate(self.arm.dof)
        path = self._prepare_pose_samples(request.q_start, request.positions, request.rotations)
        return self._solve_path(path, request.limits)

    def _solve_path(
        self, path: _PreparedCartesianPath, limits: CartesianToppLimits
    ) -> CartesianToppResult:
        grid_points = len(path.s)
        lower = np.zeros(grid_points)
        upper = self._velocity_upper_bounds(path, limits)
        lower[0] = upper[0] = 0.0
        lower[-1] = upper[-1] = 0.0

        stage1_rows: list[ConstraintRow] = []
        path_acceleration = self._affine_acceleration_rows(
            path.s,
            np.ones((grid_points, 1)),
            np.zeros((grid_points, 1)),
            np.asarray([limits.cartesian_acceleration]),
        )
        joint_acceleration = self._affine_acceleration_rows(
            path.s, path.q_s, path.q_ss, limits.joint_acceleration
        )
        cartesian_acceleration = self._affine_acceleration_rows(
            path.s,
            path.p_s,
            path.p_ss,
            np.full(3, limits.cartesian_acceleration),
        )
        angular_acceleration = self._affine_acceleration_rows(
            path.s,
            path.rotation_s,
            path.rotation_ss,
            np.full(3, limits.angular_acceleration),
        )
        stage1_rows.extend(path_acceleration[0])
        stage1_rows.extend(joint_acceleration[0])
        stage1_rows.extend(cartesian_acceleration[0])
        stage1_rows.extend(angular_acceleration[0])

        dynamics_terms = None
        if self.dynamics is not None and limits.joint_torque is not None:
            dynamics_terms = self.dynamics.path_terms(path.q, path.q_s, path.q_ss)
            stage1_rows.extend(self._torque_rows(path.s, dynamics_terms, limits.joint_torque))

        weights = self._path_weights(path.s)
        stage1 = self.lp_solver.solve_lp(weights, lower, upper, stage1_rows)
        if stage1 is None:
            raise RuntimeError("HiGHS stage 1 failed or the Cartesian path is infeasible")

        stage2_rows = list(stage1_rows)
        stage2_rows.extend(
            self._affine_jerk_rows(
                path.s,
                joint_acceleration[1],
                stage1.values,
                limits.joint_jerk,
            )
        )
        stage2_rows.extend(
            self._affine_jerk_rows(
                path.s,
                cartesian_acceleration[1],
                stage1.values,
                np.full(3, limits.cartesian_jerk),
            )
        )
        stage2_rows.extend(
            self._affine_jerk_rows(
                path.s,
                angular_acceleration[1],
                stage1.values,
                np.full(3, limits.angular_jerk),
            )
        )
        stage2_rows.extend(self._path_jerk_rows(path.s, stage1.values, limits.cartesian_jerk))
        stage2 = self.lp_solver.solve_lp(weights, lower, upper, stage2_rows)
        if stage2 is None:
            raise RuntimeError("HiGHS stage 2 failed or jerk constraints are infeasible")

        stats = HighsSolveStats(
            stage1_objective=stage1.objective,
            stage2_objective=stage2.objective,
            stage1_iterations=stage1.iterations,
            stage2_iterations=stage2.iterations,
            variables=grid_points,
            stage1_constraints=len(stage1_rows),
            stage2_constraints=len(stage2_rows),
        )
        return self._build_result(path, limits, stage1.values, stage2.values, stats, dynamics_terms)

    def _prepare_line_path(self, request: CartesianLineRequest) -> _PreparedCartesianPath:
        q_start = np.asarray(request.q_start, dtype=float)
        target = np.asarray(request.target_pose, dtype=float)
        if q_start.shape != (self.arm.dof,) or target.shape != (4, 4):
            raise ValueError("q_start 或 target_pose 维度错误")

        start = self.arm.fk(q_start)
        displacement = target[:3, 3] - start[:3, 3]
        length = float(np.linalg.norm(displacement))
        if length <= 1e-6:
            raise ValueError("笛卡尔直线长度必须大于 1e-6 m")

        s = np.linspace(0.0, length, request.grid_points)
        ratio = s / length
        rotation_vector = log_so3(start[:3, :3].T @ target[:3, :3])
        desired_position = start[:3, 3] + ratio[:, None] * displacement
        orientation_coordinate = ratio[:, None] * rotation_vector
        desired_rotation = np.asarray(
            [start[:3, :3] @ exp_so3(coordinate) for coordinate in orientation_coordinate]
        )

        return self._prepare_pose_samples(q_start, desired_position, desired_rotation)

    def _prepare_pose_samples(
        self, q_start: ArrayLike, positions: ArrayLike, rotations: ArrayLike
    ) -> _PreparedCartesianPath:
        q_start = np.asarray(q_start, dtype=float)
        desired_position = np.asarray(positions, dtype=float)
        desired_rotation = np.asarray(rotations, dtype=float)
        if q_start.shape != (self.arm.dof,):
            raise ValueError(f"q_start 维度应为 ({self.arm.dof},)")
        if desired_position.ndim != 2 or desired_position.shape[1] != 3:
            raise ValueError("positions 维度应为 (N, 3)")
        sample_count = len(desired_position)
        if sample_count < 5 or desired_rotation.shape != (sample_count, 3, 3):
            raise ValueError("rotations 维度应为 (N, 3, 3)，且 N 不小于 5")

        start = self.arm.fk(q_start)
        if not np.allclose(desired_position[0], start[:3, 3], atol=1e-8) or not np.allclose(
            desired_rotation[0], start[:3, :3], atol=1e-8
        ):
            raise ValueError("路径首个位姿必须与 q_start 的正运动学结果一致")
        spacing = np.linalg.norm(np.diff(desired_position, axis=0), axis=1)
        if np.any(spacing <= 1e-8):
            raise ValueError("相邻笛卡尔路径点必须具有非零平移距离")
        s = np.concatenate(([0.0], np.cumsum(spacing)))
        orientation_coordinate = np.asarray(
            [log_so3(start[:3, :3].T @ rotation) for rotation in desired_rotation]
        )

        q_values = np.empty((sample_count, self.arm.dof))
        q_values[0] = q_start
        seed = q_start
        for index in range(1, sample_count):
            pose = np.eye(4)
            pose[:3, :3] = desired_rotation[index]
            pose[:3, 3] = desired_position[index]
            ik = inverse_kinematics(
                self.arm,
                pose,
                seed,
                max_iterations=100,
                position_tolerance=2e-9,
                orientation_tolerance=2e-9,
            )
            if not ik.success:
                raise RuntimeError(
                    f"Cartesian IK failed at grid {index}: "
                    f"position={ik.position_error:.3e}, orientation={ik.orientation_error:.3e}"
                )
            q_values[index] = ik.q
            seed = ik.q

        q_s = np.gradient(q_values, s, axis=0, edge_order=2)
        q_ss = np.gradient(q_s, s, axis=0, edge_order=2)
        p_s = np.gradient(desired_position, s, axis=0, edge_order=2)
        p_ss = np.gradient(p_s, s, axis=0, edge_order=2)
        rotation_s = self._rotation_derivative(desired_rotation, s)
        rotation_ss = np.gradient(rotation_s, s, axis=0, edge_order=2)
        return _PreparedCartesianPath(
            s=s,
            desired_position=desired_position,
            desired_rotation=desired_rotation,
            orientation_coordinate=orientation_coordinate,
            q=q_values,
            q_s=q_s,
            q_ss=q_ss,
            p_s=p_s,
            p_ss=p_ss,
            rotation_s=rotation_s,
            rotation_ss=rotation_ss,
        )

    @staticmethod
    def _rotation_derivative(rotations: FloatArray, s: FloatArray) -> FloatArray:
        """以世界坐标表达离散姿态路径的角速度路径导数 ``omega / s_dot``。"""
        interval = np.asarray(
            [
                log_so3(right @ left.T) / ds
                for left, right, ds in zip(rotations[:-1], rotations[1:], np.diff(s), strict=True)
            ]
        )
        derivative = np.empty((len(s), 3))
        derivative[0] = interval[0]
        derivative[-1] = interval[-1]
        derivative[1:-1] = 0.5 * (interval[:-1] + interval[1:])
        return derivative

    @staticmethod
    def _velocity_upper_bounds(
        path: _PreparedCartesianPath, limits: CartesianToppLimits
    ) -> FloatArray:
        upper = np.full(len(path.s), limits.cartesian_velocity**2)
        joint_velocity = np.asarray(limits.joint_velocity, dtype=float)
        for joint in range(path.q.shape[1]):
            derivative = np.abs(path.q_s[:, joint])
            mask = derivative > 1e-10
            upper[mask] = np.minimum(upper[mask], (joint_velocity[joint] / derivative[mask]) ** 2)
        translation_scale = np.linalg.norm(path.p_s, axis=1)
        mask = translation_scale > 1e-10
        upper[mask] = np.minimum(
            upper[mask], (limits.cartesian_velocity / translation_scale[mask]) ** 2
        )
        rotation_scale = np.linalg.norm(path.rotation_s, axis=1)
        mask = rotation_scale > 1e-10
        upper[mask] = np.minimum(upper[mask], (limits.angular_velocity / rotation_scale[mask]) ** 2)
        return upper

    @staticmethod
    def _affine_acceleration_rows(
        s: FloatArray,
        first_derivative: FloatArray,
        second_derivative: FloatArray,
        limits: ArrayLike,
    ) -> tuple[list[ConstraintRow], FloatArray]:
        dimensions = first_derivative.shape[1]
        bounds = np.asarray(limits, dtype=float)
        if bounds.shape != (dimensions,):
            raise ValueError("加速度限制维度与路径导数不一致")
        coefficients = np.empty((len(s) - 1, dimensions, 2))
        rows: list[ConstraintRow] = []
        for interval, ds in enumerate(np.diff(s)):
            first = 0.5 * (first_derivative[interval] + first_derivative[interval + 1])
            second = 0.5 * (second_derivative[interval] + second_derivative[interval + 1])
            coefficients[interval, :, 0] = -first / (2.0 * ds) + 0.5 * second
            coefficients[interval, :, 1] = first / (2.0 * ds) + 0.5 * second
            for dimension in range(dimensions):
                rows.append(
                    (
                        np.asarray([interval, interval + 1], dtype=np.int32),
                        coefficients[interval, dimension].copy(),
                        -float(bounds[dimension]),
                        float(bounds[dimension]),
                    )
                )
        return rows, coefficients

    def _affine_jerk_rows(
        self,
        s: FloatArray,
        acceleration_coefficients: FloatArray,
        reference: FloatArray,
        limits: ArrayLike,
    ) -> list[ConstraintRow]:
        bounds = np.asarray(limits, dtype=float)
        dimensions = acceleration_coefficients.shape[1]
        rows: list[ConstraintRow] = []
        spacing = np.diff(s)
        for node in range(1, len(s) - 1):
            speed = np.sqrt(max(reference[node], self.config.stage2_speed_floor**2))
            factor = 2.0 * speed / (spacing[node - 1] + spacing[node])
            for dimension in range(dimensions):
                left = acceleration_coefficients[node - 1, dimension]
                right = acceleration_coefficients[node, dimension]
                values = factor * np.asarray([-left[0], right[0] - left[1], right[1]], dtype=float)
                rows.append(
                    (
                        np.asarray([node - 1, node, node + 1], dtype=np.int32),
                        values,
                        -float(bounds[dimension]),
                        float(bounds[dimension]),
                    )
                )
        return rows

    def _path_jerk_rows(
        self, s: FloatArray, reference: FloatArray, jerk_limit: float
    ) -> list[ConstraintRow]:
        first = np.ones((len(s), 1))
        second = np.zeros((len(s), 1))
        _, acceleration_coefficients = self._affine_acceleration_rows(
            s, first, second, np.asarray([np.inf])
        )
        return self._affine_jerk_rows(
            s, acceleration_coefficients, reference, np.asarray([jerk_limit])
        )

    @staticmethod
    def _torque_rows(
        s: FloatArray,
        dynamics_terms: tuple[FloatArray, FloatArray, FloatArray],
        torque_limits: FloatArray,
    ) -> list[ConstraintRow]:
        inertia, velocity_squared, gravity = dynamics_terms
        rows: list[ConstraintRow] = []
        for interval, ds in enumerate(np.diff(s)):
            m = 0.5 * (inertia[interval] + inertia[interval + 1])
            c = 0.5 * (velocity_squared[interval] + velocity_squared[interval + 1])
            g = 0.5 * (gravity[interval] + gravity[interval + 1])
            for joint in range(inertia.shape[1]):
                values = np.asarray(
                    [
                        -m[joint] / (2.0 * ds) + 0.5 * c[joint],
                        m[joint] / (2.0 * ds) + 0.5 * c[joint],
                    ],
                    dtype=float,
                )
                rows.append(
                    (
                        np.asarray([interval, interval + 1], dtype=np.int32),
                        values,
                        -float(torque_limits[joint]) - float(g[joint]),
                        float(torque_limits[joint]) - float(g[joint]),
                    )
                )
        return rows

    @staticmethod
    def _path_weights(s: FloatArray) -> FloatArray:
        spacing = np.diff(s)
        weights = np.zeros(len(s))
        weights[:-1] += 0.5 * spacing
        weights[1:] += 0.5 * spacing
        return weights

    def _build_result(
        self,
        path: _PreparedCartesianPath,
        limits: CartesianToppLimits,
        reference: FloatArray,
        squared_speed: FloatArray,
        stats: HighsSolveStats,
        dynamics_terms: tuple[FloatArray, FloatArray, FloatArray] | None,
    ) -> CartesianToppResult:
        squared_speed = np.maximum(squared_speed, 0.0)
        speed = np.sqrt(squared_speed)
        spacing = np.diff(path.s)
        interval_time = 2.0 * spacing / (speed[:-1] + speed[1:])
        time = np.concatenate(([0.0], np.cumsum(interval_time)))
        acceleration_time = 0.5 * (time[:-1] + time[1:])
        path_acceleration = np.diff(squared_speed) / (2.0 * spacing)
        interval_squared_speed = 0.5 * (squared_speed[:-1] + squared_speed[1:])
        joint_velocity = path.q_s * speed[:, None]
        tcp_velocity = path.p_s * speed[:, None]
        tcp_poses = np.asarray([self.arm.fk(q) for q in path.q])
        tcp_position = tcp_poses[:, :3, 3]
        tcp_rotation = tcp_poses[:, :3, :3]
        path_jerk = linearized_jerk_values(
            tuple(path.s), tuple(squared_speed), tuple(reference), self.config.stage2_speed_floor
        )

        _, joint_coefficients = self._affine_acceleration_rows(
            path.s, path.q_s, path.q_ss, np.asarray(limits.joint_acceleration)
        )
        joint_acceleration = np.einsum(
            "ijk,ik->ij",
            joint_coefficients,
            np.column_stack((squared_speed[:-1], squared_speed[1:])),
        )
        _, tcp_coefficients = self._affine_acceleration_rows(
            path.s,
            path.p_s,
            path.p_ss,
            np.full(3, limits.cartesian_acceleration),
        )
        tcp_acceleration = np.einsum(
            "ijk,ik->ij",
            tcp_coefficients,
            np.column_stack((squared_speed[:-1], squared_speed[1:])),
        )
        joint_jerk = self._evaluate_affine_jerk(
            path.s, joint_coefficients, reference, squared_speed
        )

        torque = None
        if dynamics_terms is not None:
            inertia, velocity_squared, gravity = dynamics_terms
            torque = (
                0.5 * (inertia[:-1] + inertia[1:]) * path_acceleration[:, None]
                + 0.5
                * (velocity_squared[:-1] + velocity_squared[1:])
                * interval_squared_speed[:, None]
                + 0.5 * (gravity[:-1] + gravity[1:])
            )
        return CartesianToppResult(
            time=time,
            acceleration_time=acceleration_time,
            path_position=path.s,
            path_speed=speed,
            path_acceleration=path_acceleration,
            reference_squared_speed=reference,
            squared_speed=squared_speed,
            joint_position=path.q,
            joint_velocity=joint_velocity,
            joint_acceleration=joint_acceleration,
            tcp_position=tcp_position,
            desired_tcp_position=path.desired_position,
            tcp_rotation=tcp_rotation,
            desired_tcp_rotation=path.desired_rotation,
            tcp_velocity=tcp_velocity,
            tcp_acceleration=tcp_acceleration,
            linearized_path_jerk=path_jerk,
            linearized_joint_jerk=joint_jerk,
            torque=torque,
            stats=stats,
        )

    def _evaluate_affine_jerk(
        self,
        s: FloatArray,
        acceleration_coefficients: FloatArray,
        reference: FloatArray,
        values: FloatArray,
    ) -> FloatArray:
        spacing = np.diff(s)
        result = np.empty((len(s) - 2, acceleration_coefficients.shape[1]))
        for node in range(1, len(s) - 1):
            speed = np.sqrt(max(reference[node], self.config.stage2_speed_floor**2))
            factor = 2.0 * speed / (spacing[node - 1] + spacing[node])
            left = acceleration_coefficients[node - 1]
            right = acceleration_coefficients[node]
            left_value = left[:, 0] * values[node - 1] + left[:, 1] * values[node]
            right_value = right[:, 0] * values[node] + right[:, 1] * values[node + 1]
            result[node - 1] = factor * (right_value - left_value)
        return result
