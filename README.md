# AlgoLab

个人算法实现、测试，以及配套的测试工具集（Python）。

## 目录结构

```
AlgoLab/
├── src/algolab/
│   ├── algorithms/          # 算法实现，按类别分子包
│   │   ├── sorting/         #   排序
│   │   ├── searching/       #   查找
│   │   ├── graph/           #   图论
│   │   ├── dp/              #   动态规划
│   │   └── string/          #   字符串
│   ├── data_structures/     # 数据结构（并查集、堆、线段树、Trie ...）
│   └── tools/               # 自用测试工具
│       ├── generators.py    #   随机数据生成（支持 seed 复现）
│       ├── checker.py       #   对拍：待测实现 vs 参考实现
│       └── timer.py         #   计时 / 基准
├── tests/                   # pytest 测试，目录与 src/algolab 一一对应
├── benchmarks/              # 性能基准脚本
├── notebooks/               # Jupyter 实验与可视化
└── docs/                    # 算法笔记
```

## 环境

使用 [uv](https://docs.astral.sh/uv/) 管理，Python ≥ 3.12。

```bash
uv sync                                   # 创建 .venv 并安装依赖
uv run pytest                             # 运行全部测试
uv run pytest tests/algorithms -k sort    # 只跑部分测试
uv run pytest --cov=algolab               # 覆盖率
uv run ruff check . && uv run ruff format .   # 代码检查 / 格式化
uv run python benchmarks/bench_sorting.py     # 基准测试
```

## 新增算法的约定

1. 在 `src/algolab/algorithms/<类别>/` 下新建 `<算法名>.py`，docstring 注明时间/空间复杂度。
2. 在 `tests/algorithms/test_<类别>.py` 中补测试：边界用例 + 用 `tools.compare` 对拍随机数据。
3. 需要看性能时，在 `benchmarks/` 下加 `bench_<类别>.py`。

## 测试工具示例

```python
from algolab.tools import compare, random_ints, timer

# 对拍：出错时会打印 seed 和输入，方便复现
compare(my_sort, sorted, lambda seed: (random_ints(100, seed=seed),))

with timer("my_sort 1e5"):
    my_sort(random_ints(100_000))
```
