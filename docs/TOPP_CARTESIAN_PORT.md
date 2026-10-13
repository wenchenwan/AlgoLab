# 笛卡尔 TOPP 移植说明

本实现将逆向得到的 `ToppPipelineRunner::run`、`FeedrateOptimizer::step`、
`ToppRuntime::run` 控制流，与可在 Windows 上运行的机器人笛卡尔约束预计算连接起来。
数值求解仍使用 HiGHS 两阶段线性规划。

## 处理链

1. 在 TCP 起点和目标位姿之间生成笛卡尔直线与 SO(3) 指数插值。
2. 使用前一离散点作为种子进行阻尼最小二乘 IK，保持关节解支连续。
3. 对离散关节路径计算 `q'(s)`、`q''(s)`，并以 `x = s_dot^2` 为 LP 变量。
4. 阶段一加入速度、加速度和可选力矩约束，最大化加权速度平方。
5. 以阶段一速度为参考，将 jerk 约束线性化后加入阶段二，再次调用 HiGHS。
6. 由最终 `x` 重建节点时间、关节/TCP 速度、区间加速度、线性化 jerk 和力矩。

核心关系为：

```text
q_dot  = q'(s) s_dot
q_ddot = q'(s) s_ddot + q''(s) s_dot^2
x      = s_dot^2
s_ddot = (x[i+1] - x[i]) / (2 delta_s)
tau    = m(s) s_ddot + c(s) s_dot^2 + g(s)
```

因此区间上的加速度和力矩均是相邻 `x[i]`、`x[i+1]` 的仿射函数，可以直接写入
LP。阶段二使用阶段一参考速度，把相邻区间加速度差除以近似时间，得到对三个相邻
节点变量的线性 jerk 约束。

## 代码对应关系

- `src/algolab/kinematics/ik.py`：连续解支数值 IK。
- `src/algolab/trajectory/topp/cartesian.py`：笛卡尔路径预计算、所有约束族、两阶段求解及结果重建。
- `src/algolab/trajectory/topp/highs_solver.py`：HiGHS LP 建模与通用两阶段标量求解器。
- `src/algolab/trajectory/topp/model.py`：逆向恢复的窗口调度、状态机和 runtime。
- `examples/topp_cartesian_line.py`：UR5 端到端 Windows demo 和曲线输出。
- `examples/topp_cartesian_complex.py`：任意离散位姿路径及活动约束利用率示例。
- `tests/trajectory/test_topp_cartesian.py`：路径精度及速度/加速度/jerk/力矩约束测试。

## 运行

```powershell
uv sync
uv run python examples/topp_cartesian_line.py
uv run python examples/topp_cartesian_line.py --show
uv run python examples/topp_cartesian_complex.py --show
uv run pytest tests/trajectory/test_topp_cartesian.py
```

默认图片输出到 `docs/assets/topp_cartesian_line.png`。

`CartesianPathRequest` 可直接接收 `(N, 3)` TCP 位置和 `(N, 3, 3)` 姿态序列。
复杂示例生成三维双频 S 曲线及工具 z 轴摆动，并对每类限制绘制
`max(abs(value) / limit)`。示例配置下，TCP 速度、关节速度、路径/关节加速度、
路径/关节 jerk 六项最大利用率都为 `1.0`。

## 动力学接口边界

`PathDynamicsModel.path_terms()` 接收 `q(s)`、`q'(s)`、`q''(s)`，返回路径投影后的
`m(s)`、`c(s)`、`g(s)`。演示中的 `DiagonalJointDynamics` 只用于完整验证力矩约束的 LP
通路，不代表 UR5 的真实惯量和重力参数。部署到实体机器人时应替换为根据机器人惯性参数、
负载和摩擦得到的 RNEA/辨识模型；规划器和 LP 结构不需要修改。

原闭源库中的 B 样条构造、具体机器人动力学参数和 `PreparedCurve` 私有矩阵没有包含在
现有二进制符号及项目配置里，因此此处移植的是已经验证的控制流、数据流和全部约束族，
而不是声称与原库对任意机器人逐位数值相同。
