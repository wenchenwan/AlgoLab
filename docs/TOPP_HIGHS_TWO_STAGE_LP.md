# HiGHS 两阶段 TOPP 线性规划恢复

## 板端调用链证据

目标符号 `topp::TwoStageFeedrateSolver::solve_window(...)` 位于 `0x14a728`，大小 `0x3f9c`。反汇编确认主路径依次调用：

```text
LpBlockAssembler::assemble_stage1
  -> StartAnchorBuilder::apply_start_anchor
  -> StartStopAnchorBuilder::apply
  -> OverlapAnchorBuilder::apply_stage1_overlap_anchor
  -> AnchorMatrixMerger::materialize_stage1
  -> HighsLpProblemBuilder::build_stage1
  -> HighsLpSolver::configure_problem
  -> HighsLpSolver::solve

逐 ConstraintMatrixBlock：
Stage2MatrixTransformer::transform(block, stage1_profile, ...)
  -> LpBlockAssembler::assemble_stage2
  -> StartStopAnchorBuilder::apply
  -> StartAnchorBuilder::apply_start_anchor
  -> OverlapAnchorBuilder::apply_stage2_overlap_anchor
  -> AnchorMatrixMerger::materialize_stage2
  -> HighsLpProblemBuilder::build_stage2
  -> HighsLpSolver::configure_problem
  -> HighsLpSolver::solve
  -> strip_errors
  -> ResultDistributor::distribute
```

库中还保留 `ToppraStage1Solver::solve` 作为阶段一的另一条路径。约束构建符号显示原实现包含 feedrate、关节/笛卡尔速度、加速度、jerk、动力学、B 样条基函数、起停段和重叠窗口约束。

## Windows 实现的 LP 变量

在路径网格 `s_0 ... s_n` 上定义：

```text
x_i = v_i² = (ds/dt)_i²
```

采用速度平方后，速度和路径加速度约束都是线性的：

```text
0 <= x_i <= v_max,i²

-2 a_max,i Δs_i <= x_(i+1) - x_i <= 2 a_max,i Δs_i

a_i = (x_(i+1) - x_i) / (2 Δs_i)
```

首窗口固定 `x_0=0`，终止窗口固定 `x_n=0`。重叠窗口的 `x_0` 使用上一窗口为重叠曲线保存的速度平方，从而让已经提交的前缀和重新优化的尾部连续。

## 阶段一：最大进给率

第一阶段在速度、加速度和边界锚点约束下求解：

```text
maximize  Σ w_i x_i
```

`w_i` 是路径网格的梯形积分权重。这是时间最优目标的线性代理，并产生用于第二阶段线性化的最大可达速度平方 profile。

## 阶段二：线性化 jerk

路径 jerk 为：

```text
j = da/dt = (da/ds) (ds/dt) = sqrt(x) da/ds
```

使用阶段一结果 `x*` 固定乘子 `sqrt(x*)`。在内部节点 `i`，左右区间加速度为：

```text
a_left  = (x_i     - x_(i-1)) / (2 Δs_left)
a_right = (x_(i+1) - x_i)     / (2 Δs_right)
```

因此下面的 jerk 约束对第二阶段变量 `x` 仍然是线性的：

```text
-j_max <= sqrt(x_i*) * 2/(Δs_left+Δs_right)
          * (a_right-a_left)
       <= j_max
```

第二阶段保留第一阶段的速度、加速度和锚点约束，加入上述 jerk 行，再次最大化 `Σw_i x_i`。两阶段都通过 `highspy.Highs` 求解，Windows 下固定单线程以获得可复现结果。

## 解的时间重建

每个区间的速度由 `v_i=sqrt(x_i)` 得到，区间内加速度恒定。持续时间为：

```text
Δt_i = 2 Δs_i / (v_i + v_(i+1))
```

`ResultDistributor` 的对应实现把全窗口解按曲线网格范围拆分为 `OptimizedProfile`，随后 `CommitSegmentBuilder` 只提交非重叠前缀。绘图脚本直接采样 LP 输出的速度平方节点，不再重新构造解析梯形曲线。

## 实现位置与边界

- HiGHS 两阶段求解器：`src/algolab/trajectory/topp/highs_solver.py`
- 窗口/重叠锚点与流水线：`src/algolab/trajectory/topp/model.py`
- LP 曲线绘图：`examples/topp_trajectory_plot.py`

当前实现恢复了两阶段结构、HiGHS 后端、速度平方变量、加速度约束、阶段二 jerk 线性化、起止锚点和重叠锚点。目标库完整的 B 样条控制变量以及机器人关节、笛卡尔和动力学矩阵需要真实机器人模型和 `PreparedCurve` 数据；目前的 `PathCommand` 标量接口没有这些输入，不能声称与生产库数值逐点一致。
