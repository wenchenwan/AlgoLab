# AlgoLab

机械臂运动控制算法的个人实验室：算法实现、单元测试，以及配套的测试/分析工具（Python）。

## 目录结构

```
AlgoLab/
├── src/algolab/
│   ├── spatial/         # 空间变换基础：旋转矩阵、RPY、四元数、SO(3) exp/log、齐次变换
│   ├── kinematics/      # 运动学：DH 建模、正运动学、几何雅可比、可操作度；逆解
│   ├── dynamics/        # 动力学：RNEA、拉格朗日、摩擦模型、参数辨识
│   ├── trajectory/      # 轨迹规划：梯形/S 型速度曲线、五次多项式、样条、笛卡尔插补
│   ├── planning/        # 路径规划：RRT/PRM、碰撞检测、路径平滑
│   ├── control/         # 控制器：PID、计算力矩、阻抗/导纳
│   ├── filters/         # 滤波与估计：低通/陷波、卡尔曼、速度估计
│   ├── models/          # 机械臂模型参数（UR5 ...）
│   └── tools/           # 自用测试工具
│       ├── generators.py        # 随机关节角 / 旋转 / 位姿（可指定 seed 复现）
│       ├── checker.py           # 数值对拍、数值雅可比、旋转矩阵合法性
│       ├── trajectory_check.py  # 轨迹速度/加速度/jerk 超限检查
│       ├── plot.py              # 关节轨迹曲线
│       └── timer.py             # 计时 / 基准
├── tests/               # pytest 测试，目录与 src/algolab 一一对应
├── benchmarks/          # 耗时基准（关注是否满足控制周期）
├── examples/            # 可直接运行的示例脚本
├── notebooks/           # Jupyter 实验、调参、可视化
├── data/                # 实测/仿真数据（默认不入库，小样例放 data/samples/）
└── docs/                # 算法笔记、推导
```

已实现的是一套骨架示例，未实现的模块只有 `__init__.py` 占位。

## 环境

使用 [uv](https://docs.astral.sh/uv/) 管理，Python ≥ 3.12。

```bash
uv sync                                        # 创建 .venv 并安装依赖
uv run pytest                                  # 运行全部测试
uv run pytest tests/kinematics                 # 只跑某个模块
uv run pytest --cov=algolab                    # 覆盖率
uv run ruff check . ; uv run ruff format .     # 代码检查 / 格式化
uv run python examples/ptp_trajectory.py       # 示例：6 轴同步梯形 PTP
uv run python benchmarks/bench_kinematics.py   # FK / 雅可比耗时
```

## 约定

- 单位：长度 m，角度 rad，时间 s。
- 四元数顺序 `(w, x, y, z)`；RPY 为 `R = Rz(yaw) · Ry(pitch) · Rx(roll)`。
- 雅可比行顺序 `[v; ω]`，表达在基座系。
- DH 使用标准 DH（非 MDH）。
- 新算法：在对应模块下建文件，docstring 写清公式/约定；测试至少包含
  1. 已知解析结果（如特殊位形、边界条件）；
  2. 随机数据的性质检验或对拍（如解析雅可比 vs `numerical_jacobian`，逆解 → 正解回代）。

## 测试工具示例

```python
import numpy as np
from algolab.models import ur5
from algolab.kinematics.jacobian import geometric_jacobian
from algolab.tools import check_trajectory, numerical_jacobian, random_joint_config

arm = ur5()
q = random_joint_config(-np.pi * np.ones(6), np.pi * np.ones(6), seed=42)

# 解析雅可比 vs 数值微分
Jv_num = numerical_jacobian(lambda x: arm.fk(x)[:3, 3], q)
assert np.allclose(geometric_jacobian(arm, q)[:3], Jv_num, atol=1e-6)

# 检查一条轨迹（规划输出或实测记录）是否超限
report = check_trajectory(t, q_traj, v_max=np.pi / 2, a_max=np.pi)
print(report)
```

## TOPP 逆向调度 demo

项目包含从 AArch64 `libtopp_core.so` 恢复的 TOPP 流水线控制流模型，可在 Windows 上直接运行：

```powershell
uv run python examples/topp_runtime_demo.py
uv run python examples/topp_runtime_demo.py --scenario solver-failure
uv run python examples/topp_trajectory_plot.py
uv run python examples/topp_cartesian_line.py
uv run python examples/topp_cartesian_complex.py
uv run pytest tests/trajectory/test_topp_runtime.py
uv run pytest tests/trajectory/test_topp_cartesian.py
```

实现复现 `FeedrateOptimizer::step`、`ToppPipelineRunner::run` 和 `ToppRuntime::run` 的窗口、状态机、错误传播、取消及周期控制语义。数值部分通过 `highspy` 调用 HiGHS：阶段一优化速度平方曲线，阶段二基于阶段一参考速度加入线性化 jerk 约束并重新优化。详细证据见 [`docs/TOPP_HIGHS_TWO_STAGE_LP.md`](docs/TOPP_HIGHS_TWO_STAGE_LP.md)、[`docs/TOPP_FEEDRATE_RUNTIME_REVERSE.md`](docs/TOPP_FEEDRATE_RUNTIME_REVERSE.md) 与 [`docs/TOPP_PIPELINE_RUNNER_RUN_REVERSE.md`](docs/TOPP_PIPELINE_RUNNER_RUN_REVERSE.md)。

绘图示例测试短距离 jerk 受限、长距离梯形速度、加速度受限和四段混合路径，默认输出 `docs/assets/topp_trajectory_profiles.png`；传入 `--show` 可同时打开交互窗口。

笛卡尔 demo 对 UR5 TCP 直线进行连续 IK 和路径约束预计算，然后将关节/TCP 的速度、
加速度、jerk 及可插拔动力学力矩限制统一写入两阶段 LP。它会输出三维路径、TCP、关节、
jerk 和力矩曲线到 `docs/assets/topp_cartesian_line.png`。实现与模型接口说明见
[`docs/TOPP_CARTESIAN_PORT.md`](docs/TOPP_CARTESIAN_PORT.md)。

复杂轨迹示例使用三维双频 S 曲线并叠加工具姿态摆动，特意让 TCP 速度、关节速度、
路径/关节加速度和阶段二 jerk 在不同区段成为活动约束。默认输出
`docs/assets/topp_cartesian_complex.png`，图中的归一化利用率达到 `1.0` 表示约束被触及。
