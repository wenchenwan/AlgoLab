# `FeedrateOptimizer::step` 与 `ToppRuntime::run` 逆向恢复

## 目标与实现范围

- 目标库：开发板 `/home/topp-test/lib/libtopp_core.so`
- 架构：AArch64 ELF64，保留 C++ 符号，无 DWARF 类型信息
- SHA-256：`b7ede831359dd359acc67b4060f1a7adda7f22eeba70460d272bb5abaf282d61`
- Build ID：`63b24657c574a6448f9f1b118018121e9e7f3a8b`
- `topp::FeedrateOptimizer::step()`：`0x14e6c8`，大小 `0xe54`
- `topp::ToppRuntime::run(PathPlanningRequest const&)`：`0x192d10`，大小 `0x1ce8`

本文的控制流、直接调用、状态值和对象偏移来自板端 `nm`、`readelf`、`objdump` 与 GDB。字段名是依据行为推断的；伪代码是语义恢复，不是原始源码。REA 6.4.0 已尝试，但其无后端模式只支持 Linux x64，无法直接分析本目标的 AArch64 ELF。

## `FeedrateOptimizer::step`

### 已确认的成员与状态

反汇编直接访问的主要成员如下：

```cpp
PreparedCurveBuffer prepared_buffer_;       // this + 0x138
// ...
WindowSelector* window_selector_;            // this + 0x580（按调用语义命名）
TwoStageFeedrateSolver* solver_;             // this + 0x588（按调用语义命名）
CommitSegmentBuilder* segment_builder_;      // this + 0x590（按调用语义命名）
int status_;                                 // this + 0x598
unsigned step_count_;                        // this + 0x59c
// ...
HeapVector<OptimizerFlowObservation, 64> observations_; // this + 0x5e0
```

`status_` 的写入值：

| 数值 | 恢复名称 | 触发时机 |
|---:|---|---|
| 0 | `Idle` | reset 后 |
| 1 | `WaitingForInput` | 当前曲线不足以形成窗口 |
| 2 | `OutputReady` | 成功提交一批段，后面还有数据 |
| 3 | `Solving` | 调用两阶段求解器之前 |
| 4 | `Terminal` | 已收到终止且没有待处理窗口 |
| 5 | `Failed` | 窗口无效或求解失败 |
| 6 | `OutputReadyAndTerminal` | 本批有输出，并且它是最后一批 |

枚举名字是按行为命名，数值和写入位置是确定的。

### 恢复出的处理顺序

```cpp
Result<OptimizerStepResult> FeedrateOptimizer::step()
{
    // 1. 预处理失败具有最高优先级，一次弹出一个。
    if (prepared_buffer_.has_failed()) {
        auto failed = prepared_buffer_.pop_failed();
        if (!failed)
            return failure(failed.error());

        auto segment = segment_builder_->build_failed_segment(
            failed->curve_id, failed->source_index, failed->error);
        return success(OptimizerStepResult::with_segment(std::move(segment)));
    }

    // 2. 选择一个可解窗口。这里还传入左右零速边界与配置项。
    auto selected = window_selector_->select_next_window(
        prepared_buffer_, config_value_, left_boundary_, right_boundary_);
    if (!selected)
        return failure(selected.error());

    if (selected->kind == NeedInput) {
        status_ = 1;
        return success(OptimizerStepResult::need_input());
    }
    if (selected->kind == Terminal) {
        status_ = 4;
        return success(OptimizerStepResult::no_more_work());
    }
    if (!selected->window.has_value()) {
        status_ = 5;
        return success(OptimizerStepResult::failed_or_no_work());
    }

    const OptimizationWindow& window = *selected->window;

    // 3. 从 buffer 取得 [begin,end) 指针集合，构造约束和重叠参考。
    auto curves = prepared_buffer_.selectable_curves(window.begin, window.end);
    if (!curves)
        return failure(curves.error());

    auto context = build_default_feedrate_solve_context(window, *curves, config_);
    if (!context)
        return failure(context.error());

    auto overlap = prepared_buffer_.build_overlap_reference(window);
    if (!overlap)
        return failure(overlap.error());
    context->overlap_reference = std::move(*overlap);

    // 4. 调用真正的 LP/两阶段进给率求解器。
    status_ = 3;
    auto solved = solver_->solve_window(window, *curves, *context);

    if (!solved) {
        // 求解失败不是本函数外层 Result 的失败：窗口中每条曲线被转换成
        // CommittedSegment{failure}，并作为一个有效 OptimizerStepResult 返回。
        status_ = 5;
        std::vector<CommittedSegment> failed_segments;
        for (const PreparedCurve* curve : *curves) {
            failed_segments.push_back(segment_builder_->build_failed_segment(
                curve->curve_id, window.source_index, solved.error()));
        }
        observations_.push_back(make_failure_observation(...));
        prepared_buffer_.mark_optimized(window);
        return success(OptimizerStepResult::failed_segments(
            std::move(failed_segments)));
    }

    // 5. 成功时只提交窗口的非重叠前缀，尾部保留给下一个窗口。
    const size_t commit_count = window_selector_->compute_commit_count(window);
    auto segments = segment_builder_->build_committed_segments(*solved, commit_count);
    if (!segments)
        return failure(segments.error());

    observations_.push_back(make_success_observation(...));
    if (!prepared_buffer_.store_optimized_profiles(*solved))
        return failure(...);

    prepared_buffer_.mark_optimized(window);
    prepared_buffer_.mark_committed(commit_count);
    ++step_count_;

    const bool final_batch =
        prepared_buffer_.terminal_received() &&
        prepared_buffer_.available_count() == 0;
    status_ = final_batch ? 6 : 2;
    return success(OptimizerStepResult::segments(
        std::move(*segments), final_batch));
}
```

### 关键结论

- 失败输入先于正常窗口处理，因此预处理失败不会被后面的正常曲线阻塞。
- 重叠窗口由 `compute_commit_count` 决定提交前缀，窗口尾部可进入下一轮，以保持边界速度连续。
- 构建上下文、重叠参考、提交段和保存 profile 的基础设施错误会通过外层 `Result` 传播。
- `solve_window` 的数值求解失败则被降级为失败段，外层 `Result` 仍然成功；调用者必须检查每个 `CommittedSegment`。
- 成功求解最后一批时状态为 6，不是普通的状态 2。

## `ToppRuntime::run`

运行时状态位于 `this + 0xA570`。已确认的状态值为：

| 数值 | 恢复名称 |
|---:|---|
| 0 | `Unprepared` |
| 1 | `Ready` |
| 2 | `Running` |
| 3 | `Succeeded` |
| 4 | `Failed` |

`run` 本身是一个很薄的生命周期包装器；函数体很大主要是因为编译器内联展开了巨大结果对象的移动/拷贝和清理。

```cpp
Result<PathPlanningResult>
ToppRuntime::run(const PathPlanningRequest& request)
{
    if (state_ != Ready)
        return failure(static_cast<ToppErrorCode>(35));

    state_ = Running;

    // request + 0x3960（十进制 14688）
    highs_solver_.set_abort_flag(request.abort_flag);

    auto result = pipeline_runner_.run(request);

    // 位于 pipeline 调用之后，成功和失败路径都会先解除求解器取消指针。
    highs_solver_.set_abort_flag(nullptr);

    if (!result) {
        state_ = Failed;
        return failure(result.error());
    }

    state_ = Succeeded;
    return success(std::move(result.value()));
}
```

相关状态守卫也从相邻函数得到确认：

- `prepare()` 只接受状态 0，否则返回错误 `34`；成功后进入状态 1。
- `run()` 只接受状态 1，否则返回错误 `35`。
- `reset()` 只接受状态 3 或 4，否则返回错误 `36`；清理本轮所有队列、buffer、solver 和 arena 后回到状态 1。

## Windows 可执行 demo

实现位于：

- `src/algolab/trajectory/topp/model.py`
- `examples/topp_runtime_demo.py`
- `tests/trajectory/test_topp_runtime.py`

在项目根目录的 PowerShell 中运行：

```powershell
uv sync
uv run python examples/topp_runtime_demo.py
uv run python examples/topp_runtime_demo.py --scenario solver-failure
uv run python examples/topp_runtime_demo.py --scenario timeout
uv run python examples/topp_runtime_demo.py --scenario cancel
uv run python examples/topp_runtime_demo.py --scenario stalled
uv run python examples/topp_trajectory_plot.py
uv run python examples/topp_trajectory_plot.py --show
uv run pytest tests/trajectory/test_topp_runtime.py
```

输出中的两个成功字段含义不同：

- `api_success`：流水线外层 `Result` 是否成功。
- `success`：外层成功并且所有提交段均成功。

因此 `solver-failure` 会展示 `api_success: true`、`success: false`，准确表达原库将数值求解失败包装为失败段的行为。

`topp_trajectory_plot.py` 规划并检查四组代表性路径：短距离 jerk 受限曲线、长距离梯形速度曲线、加速度受限曲线和四段混合路径。默认将位置、速度、加速度和阶段二线性化 jerk 图保存到 `docs/assets/topp_trajectory_profiles.png`；传入 `--show` 会同时打开 Matplotlib 窗口。

## Demo 与原库的边界

Windows demo 已通过 `highspy` 调用真实 HiGHS，执行阶段一速度/加速度 LP 和阶段二线性化 jerk LP，同时保留状态机、重叠窗口、失败段、EOS、取消、最大周期和无进展检测。当前输入模型仍只有标量路径长度、速度、加速度和 jerk 上限；目标库中的机器人关节/笛卡尔动力学约束、B 样条基函数和完整 start-stop 几何尚未恢复，因此不应直接用于生产机器人轨迹下发。两阶段矩阵与公式见 `TOPP_HIGHS_TWO_STAGE_LP.md`。
