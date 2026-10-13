# `ToppPipelineRunner::run` 逆向恢复

## 目标与证据

- 目标文件：`/home/topp-test/lib/libtopp_core.so`
- 架构：AArch64 ELF64
- SHA-256：`b7ede831359dd359acc67b4060f1a7adda7f22eeba70460d272bb5abaf282d61`
- Build ID：`63b24657c574a6448f9f1b118018121e9e7f3a8b`
- 目标符号：`topp::ToppPipelineRunner::run(topp::PathPlanningRequest const&)`
- 地址：`0x190de8`
- 大小：`0x5b4` 字节

目标库保留了 C++ 符号，但没有 DWARF 类型信息。下面的控制流、被调用函数、常量和字段偏移来自符号表与 AArch64 反汇编；字段名称是根据调用语义恢复的。代码是语义等价实现，不是原始源码的逐字还原。

## 恢复出的对象布局

构造函数位于 `0x18b570`，可以恢复出 `ToppPipelineRunner` 的主要成员：

```cpp
class ToppPipelineRunner {
    PlanningContext* context_;                 // +0x00
    PathCommandProcessor* command_processor_;  // +0x08
    PipelineOrchestrator* orchestrator_;        // +0x10
    // PipelineDiagnosticsCollector 很可能是空对象，未占独立存储
    PreparedCurveStore* prepared_store_;       // +0x20
    Arena* arena_;                              // +0x28
    const std::atomic<bool>* abort_flag_;       // +0x30
};
```

`PathPlanningRequest` 中被本函数直接使用的字段：

```cpp
// 偏移是确定的，名称由行为推断。
initial_joints             // +0x3908，Vector<double, 6>
terminal_after_commands   // +0x3940，bool
max_cycles                // +0x3948，std::size_t
abort_flag                // +0x3960，const std::atomic<bool>*
```

## 语义等价实现

为突出调度逻辑，以下代码省略编译器生成的栈保护，以及临时 `OptionalStorage`、`HeapVector` 的析构展开。原二进制通过 RAII 完成这些清理。

```cpp
Result<PathPlanningResult>
ToppPipelineRunner::run(const PathPlanningRequest& request)
{
    // 反汇编会根据 Arena 当前 scope 类型，把 arena_ 临时绑定到对应的
    // thread_local 槽，并在所有返回/异常路径上恢复旧值和 arena 游标。
    ArenaScope arena_scope(*arena_);

    abort_flag_ = request.abort_flag;

    prepared_store_->clear();
    command_processor_->reset_for_next_run();

    orchestrator_->initialize_referencejoints_disc(request.initial_joints);
    orchestrator_->initialize_referencejoints_inter(request.initial_joints);

    // 对首条 MoveL/MoveC 的起始关节做近奇异检查。若命中，该辅助函数
    // 会向流水线发送带 PipelinePlanningStop 的 EOS。
    auto singular_start = stop_if_first_cartesian_start_is_singular(request);
    if (!singular_start) {
        abort_flag_ = nullptr;
        return Result<PathPlanningResult>::failure(singular_start.error());
    }

    if (singular_start.value()) {
        auto drained = orchestrator_->run_all(request.max_cycles);
        if (!drained) {
            abort_flag_ = nullptr;
            return Result<PathPlanningResult>::failure(drained.error());
        }

        auto result = make_result(request, drained.value());
        abort_flag_ = nullptr;
        return result;
    }

    auto fed = feed_input_commands(request, request.max_cycles);
    if (!fed) {
        abort_flag_ = nullptr;
        return Result<PathPlanningResult>::failure(fed.error());
    }

    std::size_t cycle_count = fed.value();

    if (abort_flag_ != nullptr && abort_flag_->load()) {
        abort_flag_ = nullptr;
        return Result<PathPlanningResult>::failure(
            static_cast<ToppErrorCode>(411));  // 取消
    }

    if (request.terminal_after_commands) {
        // queue0 满时先执行必要周期，直到能发送 EOS；返回内部消耗的周期数。
        auto eos = signal_eos_when_ready(request.max_cycles, cycle_count);
        if (!eos) {
            abort_flag_ = nullptr;
            return Result<PathPlanningResult>::failure(eos.error());
        }
        cycle_count += eos.value();
    }

    std::size_t executed = 0;
    const std::size_t remaining =
        cycle_count < request.max_cycles
            ? request.max_cycles - cycle_count
            : 0;

    while (executed < remaining && !orchestrator_->is_complete()) {
        if (abort_flag_ != nullptr && abort_flag_->load()) {
            abort_flag_ = nullptr;
            return Result<PathPlanningResult>::failure(
                static_cast<ToppErrorCode>(411));  // 取消
        }

        auto step = orchestrator_->run_one_cycle();
        if (!step) {
            abort_flag_ = nullptr;
            return Result<PathPlanningResult>::failure(step.error());
        }

        ++executed;

        // run_one_cycle 没有推动任何阶段，并且流水线又没有完成，继续循环
        // 不可能产生新结果，因此立即失败，避免死循环。
        if (!step.value().made_progress && !orchestrator_->is_complete()) {
            abort_flag_ = nullptr;
            return Result<PathPlanningResult>::failure(
                static_cast<ToppErrorCode>(19));  // pipeline no progress
        }
    }

    if (!orchestrator_->is_complete()) {
        abort_flag_ = nullptr;
        return Result<PathPlanningResult>::failure(
            static_cast<ToppErrorCode>(20));      // max cycles exceeded
    }

    auto result = make_result(request, cycle_count + executed);
    abort_flag_ = nullptr;
    return result;
}
```

## 执行流程

```text
绑定本线程 Arena
  -> 清空 PreparedCurveStore / 重置命令处理器
  -> 设置离散与插补参考关节
  -> 检查首个笛卡尔运动是否从奇异位姿开始
       -> 是：发送 PlanningStop，run_all 排空，组装结果
       -> 否：继续
  -> 将请求命令转换为 PipelineItem 并写入 queue0
       -> queue0 满时运行必要周期释放空间
  -> 检查取消
  -> terminal_after_commands=true 时发送 EOS
       -> queue0 满时运行必要周期，等待 EOS 可写入
  -> 循环 run_one_cycle
       -> 失败：传播下层错误
       -> 无进展且未完成：错误 19
       -> 超过 max_cycles：错误 20
       -> 收到取消：错误 411
  -> make_result(request, total_cycles)
  -> 清除 abort_flag_，恢复 Arena
```

## 辅助函数的恢复结论

### `feed_input_commands(request, max_cycles)`

1. 从 `Arena` 申请一个最大 512 项的临时 `PipelineItem` 容器；单项大小为 `0x610`。
2. 调用 `PathCommandProcessor::build_raw_items` 把 MoveJ/MoveL/MoveC 请求转换成原始流水线项。
3. 逐项调用 `PipelineOrchestrator::add_input_item`。
4. 若 queue0 已满，反复调用 `run_one_required_cycle` 释放容量。
5. 返回投喂过程中消耗的流水线周期数。
6. Arena 申请失败返回错误码 `33`。

### `signal_eos_when_ready(max_cycles, cycles_already_used)`

1. 如果预处理器已经结束，直接成功返回 0。
2. queue0 满时，通过 `run_one_required_cycle` 推进流水线。
3. queue0 有空间后调用 `PipelineOrchestrator::signal_eos()`。
4. 返回为发送 EOS 而额外执行的周期数。

### `run_one_required_cycle(max_cycles, current_cycles)`

1. 先检查取消标志，命中时返回 `411`。
2. 若 `current_cycles >= max_cycles`，返回 `20`。
3. 调用 `orchestrator_->run_one_cycle()`。
4. 下层失败时原样传播错误。
5. 若周期成功但没有任何阶段取得进展，返回 `19`。
6. 否则返回本次消耗的周期数 1。

### `stop_if_first_cartesian_start_is_singular(request)`

1. 空命令或首条不是 MoveL/MoveC 时返回 `false`。
2. 调用机器人模型的近奇异检查回调检查 `initial_joints`。
3. 命中时根据奇异位掩码构造 `PipelinePlanningStop`，错误细分值包括 `-11`、`-12`、`-13` 和 `-4009`。
4. 将该停止原因随 EOS 写入流水线，成功后返回 `true`。

## `make_result` 的职责

`make_result` 很大（`0x5048` 字节），但它不参与主调度决策。其直接调用证明它负责：

- 从 `PipelineOrchestrator` 取得关节位置轨迹；
- 检查流水线是否完成；
- 读取原始曲线数和已提交段数；
- 调用 `PipelineDiagnosticsCollector::collect`；
- 合并诊断、失败记录、警告和已提交段；
- 把总周期数写入最终结果。

## 验证

使用板端 `topp_offline_reproducer_v2` 执行 `cases/example.yaml`：

```text
result success=true prepared=2 committed=2 samples=727
pipeline cycles=6 complete=true raw_commands=2 raw_curves=2
committed_segments=2 failures=0 queues=[0,0,0]
```

这与恢复的正常路径一致：两条命令被投喂，EOS 后流水线在 6 个周期内完成，三个队列最终均为空，再由 `make_result` 生成 727 个插补样本。

## 可信度边界

- 高可信：控制流、调用顺序、字段偏移、周期计数方式、错误码数值、取消/EOS/超时行为。
- 中等可信：未导出枚举名称，因此错误码 `19/20/411` 的文字名称是按行为命名。
- 未恢复：原始局部变量名、注释、模板别名和 `Result` 的确切声明形式。
- REA 6.4.0 已尝试运行，但其无后端 ELF 分析器拒绝 AArch64 目标；本次深度证据来自目标板的 `nm`、`objdump`、`readelf` 及离线复现器。
