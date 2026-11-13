# Hughes 计算引擎复用指南

## 1. 结论

另一个项目可以直接复用本仓库的计算引擎来运行不含管控机制的 Hughes 型宏观人群仿真，不需要先修改数值内核。推荐使用配置驱动入口 `codes/simulate_from_config.py` 或 Python 函数 `crowd_bellman.config_workflow.run_from_config()`。

需要准确理解“不使用 M”：当前统一求解器内部始终以 `m11`、`m12`、`m22` 表示局部度量，但当配置中不声明任何 `[[stages.controls]]` 时，编译器会自动令

```text
m11 = 1, m12 = 0, m22 = 1，即 M = I。
```

此时 M 不产生各向异性引导，模型退化为各向同性 Hughes 基线。方向控制和入口速率控制也都是可选机制，省略对应配置后不会生效。因此，从模型功能上可以直接复用；从长期的软件依赖管理上，当前代码还不是一个成熟的独立 Python 包，如果另一个项目需要稳定的进程内 API、无文件输出运行或独立版本管理，建议按第 8 节抽离。

## 2. 本指南所指的功能边界

另一个项目需要保留的计算链为：

```text
总密度 rho
  -> Greenshields 速度 f(rho)
  -> 各向同性 Bellman 最短时间势函数 phi
  -> 最优行进方向 u
  -> 守恒律通量更新
  -> 出口移除质量
```

以下机制不属于本次复用范围：

| 不需要的机制 | 当前实现位置 | 关闭方式 |
|---|---|---|
| M 的各向异性几何引导 | `routes.toml` 的 `[[stages.controls]]` | 不写任何 `[[stages.controls]]`，使 `M=I` |
| 通道通行方向控制 | `stages.allowed_directions` 或控制项的 `allowed_directions` | 两处都不写，编译器默认允许 8 个离散方向 |
| 内部通道入口通行速率控制 | `routes.toml` 的 `[[capacity_controls]]` | 不写任何 `[[capacity_controls]]` |
| 外部持续入流 | `population.toml` 的 `[[inflow_groups]]` | 不写 `[[inflow_groups]]`，只配置初始人群 |
| 通道流量统计 | `scene.toml` 的 `[[channels]]` | 不需要通道指标时可以全部省略 |
| 多阶段切换和概率分流 | `next_stage`、`decision_region`、`targets`、`kappa` | 使用单个 stage，并省略这些字段 |

“内部通道入口通行速率控制”和“外部持续入流”不是同一功能。前者限制已有行人在内部截面上的通行通量，后者向计算域内持续增加人群。若另一个项目两者都不需要，应同时省略 `[[capacity_controls]]` 和 `[[inflow_groups]]`。

密度上限 `rho_max` 及其数值通量限制属于基础求解器的拥挤与数值稳定机制，不是入口速率控制，不建议删除。

## 3. 最快的直接复用方式

### 3.1 环境要求

当前仓库没有 `pyproject.toml`、`setup.py` 或依赖锁文件。运行完整配置工作流至少需要：

```powershell
python -m pip install numpy matplotlib numba
```

默认 `bellman_backend = "sweeping"` 依赖 Numba。若暂时不安装 Numba，可以在 `run.toml` 中设置 `bellman_backend = "optimized"`，但大网格运行会明显更慢。

另一个项目可先选择以下任一接入方式：

1. 将本仓库作为 Git submodule 或固定提交的 vendor 依赖，并从命令行调用 `simulate_from_config.py`。该方式对现有代码侵入最小。
2. 把 `D:\CrowdModels\codes` 加入 `PYTHONPATH`，直接调用 `run_from_config()`。该方式适合原型验证，但不建议依赖一个会随研究代码变化的绝对路径。
3. 按第 8 节抽离并打包为独立库。该方式适合长期维护和多个项目共同依赖。

不要只复制 `core.py` 后立即接入生产代码。完整工作流还依赖 `scenes.py`、配置 loader/compiler、`runner.py`、`metrics.py` 和 `plotting.py`，只复制单文件很容易遗漏数据结构、CFL、密度上限和输出语义。

### 3.2 命令行调用

在本仓库根目录执行：

```powershell
python codes\simulate_from_config.py --config <另一个项目中的配置目录>\run.toml
```

入口会依次加载四个 TOML 文件、编译栅格场景和人群、执行仿真，并写出结果。

### 3.3 Python 调用

当 `codes` 已位于 `PYTHONPATH` 中时：

```python
from pathlib import Path

from crowd_bellman.config_workflow import run_from_config

summary = run_from_config(
    Path(r"D:\AnotherProject\simulation\run.toml"),
    output_root=Path(r"D:\AnotherProject\outputs\hughes"),
    simulation_overrides={
        "steps": 800,
        "time_horizon": 120.0,
        "save_every": 100,
    },
)

print(summary["case_id"])
print(summary["group_count"])
```

当前 `run_from_config()` 会创建结果目录并写配置快照、CSV、JSON 和图片。它不是纯内存函数，调用方应为输出目录预留写权限。

## 4. 纯 Hughes 最小配置

建议在另一个项目中建立独立目录，例如：

```text
simulation/hughes_case/
├── run.toml
├── scene.toml
├── population.toml
└── routes.toml
```

### 4.1 run.toml

```toml
[simulation]
nx = 60
ny = 40
dx = 0.5
steps = 300
time_horizon = 40.0
vmax = 1.5
rho_max = 5.0
bellman_every = 1
bellman_f_eps = 0.08
cfl = 0.9
dt_cap = 0.18
save_every = 50
bellman_backend = "sweeping"
direction_recovery_backend = "vectorized"

# 以下值明确关闭额外墙体代价；普通障碍仍由 walkable mask 生效。
wall_avoidance_weight = 0.0
wall_clearance_cells = 0

[objective]
name = "pure_hughes"

[scene]
file = "scene.toml"

[population]
file = "population.toml"

[routes]
file = "routes.toml"

[outputs]
output_root = "./results"
```

`dx` 是单元格边长，`vmax` 是自由流速度，`rho_max` 是最大密度。三个量必须采用彼此一致的长度、时间和人数单位。数组坐标使用 `[y, x]`，TOML 矩形使用 `x0:x1, y0:y1` 的左闭右开语义。

### 4.2 scene.toml

```toml
block_boundaries = true
obstacles = ["wall"]

[[regions]]
name = "spawn"
x0 = 2
x1 = 12
y0 = 10
y1 = 30

[[regions]]
name = "exit"
x0 = 58
x1 = 59
y0 = 16
y1 = 24

[[regions]]
name = "wall"
x0 = 28
x1 = 31
y0 = 0
y1 = 15

[[exits]]
name = "exit"
region = "exit"
```

这里故意没有 `[[channels]]`。通道仅用于通量统计及入口限流引用，不是 Hughes 求解的必需几何元素。

### 4.3 population.toml

```toml
[[initial_groups]]
group_id = "pedestrians"
stage_id = "evacuation"
region = "spawn"
density = 2.0
```

这里故意没有 `[[inflow_groups]]`，因此仿真过程中不会按速率注入新行人。

### 4.4 routes.toml

```toml
[case]
case_id = "pure_hughes_demo"
title = "Pure Hughes Demo"

[[stages]]
stage_id = "evacuation"
group_key = [1, 1]
goal_region = "exit"
sink_region = "exit"
```

这个文件中不要增加以下内容：

```toml
# 不要配置 stages.allowed_directions
# 不要配置 [[stages.controls]]
# 不要配置 [[capacity_controls]]
```

编译后的有效状态应为：

```text
M = I
allowed_mask = 全部 8 个方向，但不可通行单元为 0
inflows = ()
gates = {}
gate_capacity_schedules = ()
transitions = ()
```

障碍物边角处可能因 `block_diagonal_corner_cutting = true` 而屏蔽穿墙式对角移动。这是几何一致性处理，不是通道方向管控。

## 5. 输入、运行过程与输出

### 5.1 必需输入

纯 Hughes 单阶段场景至少需要：

- 计算网格尺寸 `nx`、`ny` 和单元格边长 `dx`；
- 可通行域和障碍物；
- 一个非空初始密度区域；
- 一个非空目标/出口区域；
- `vmax`、`rho_max`、CFL 和停止条件；
- 一个 stage，使初始群体的 `stage_id` 能映射到目标出口。

所有密度数组、mask 和目标区域必须具有 `(ny, nx)` 形状。出口应位于可通行域内，初始密度不应落在障碍物上。

### 5.2 每步计算

`runner.simulate_case()` 每步执行以下基础过程：

1. 汇总所有群体密度；纯 Hughes 示例中只有一个群体。
2. 用 `greenshields_speed()` 计算 `f(rho)`。
3. 用 `solve_bellman()` 计算到出口的值函数。
4. 用 `recover_optimal_direction()` 恢复 8 邻域最优方向。
5. 根据 CFL 条件确定时间步长。
6. 计算迎风面通量并执行密度上限约束。
7. 用守恒更新推进密度，并移除到达 `sink_mask` 的质量。
8. 记录指标并按 `save_every` 写快照。

### 5.3 默认输出

输出目录为 `<output_root>/<case_id>/`，主要文件包括：

- `summary.json`：汇总指标和实际生效的仿真配置；
- `timeseries.csv`：逐步时间序列；
- `timeseries.png`：时间序列图；
- `snapshot_XXXX.png`：密度、势函数和方向场快照；
- `config_snapshot/`：本次运行的四个 TOML 文件副本及 manifest；
- 根目录 `config_run_summary.json`：配置工作流汇总。

完全省略 `[[channels]]` 可以正常仿真。当前绘图代码可能输出一次 `No artists with labels found to put in legend` 警告，这是空通道图例造成的非致命警告，不影响数值结果。

## 6. 接入时应做的校验

首次迁移场景后至少检查：

1. `summary.json` 中 `group_count = 1`、`transition_count = 0`。
2. 编译后的每个 group 满足 `m11=1`、`m12=0`、`m22=1`。
3. `case.inflows`、`case.gate_capacity_schedules` 为空，`case.gates` 为空字典。
4. 不可通行区域密度始终为 0，出口势函数为 0，出口外存在有限势函数。
5. 在无持续入流条件下，初始质量约等于当前域内质量、累计出口质量以及数值诊断项之和。
6. 逐步减小 `dx` 或 `dt_cap` 时，关键结果趋势应稳定。
7. 用小场景分别运行 `optimized` 与 `sweeping` 后端，确认结果在项目可接受误差内一致。

本仓库已有 `codes/tests/test_core_backends.py`、`codes/tests/test_mass_conservation.py` 和 `codes/tests/test_scene_walls.py`，抽离后应保留对应回归测试。

## 7. 可以直接复用到什么程度

### 7.1 适合直接复用

满足以下条件时，可以先直接依赖当前源码：

- 另一个项目接受四个 TOML 文件描述场景；
- 仿真以离线批处理方式运行；
- 接受引擎自行写 JSON、CSV 和图片；
- 接受当前的 Greenshields 速度-密度关系和 8 邻域离散；
- 能把本仓库固定在明确的 Git commit，避免上游研究代码变化导致结果漂移。

### 7.2 不建议直接依赖

出现以下任一需求时，应抽离独立库：

- 需要从另一个项目的服务进程中稳定 import；
- 需要纯内存计算，不希望强制生成图片和文件；
- 需要逐步控制、暂停、恢复或把状态交给其他模块；
- 需要替换速度-密度关系、数值边界或输出指标；
- 多个项目要共享同一版本并进行语义化版本管理；
- 需要明确的公共 API 和向后兼容承诺。

当前限制的根因是：仓库没有包构建元数据，`crowd_bellman/__init__.py` 没有导出公共接口，`simulate_case()` 同时承担数值循环、指标、绘图和文件写入。因此它在功能上可复用，但还不是稳定、轻量的 SDK。

## 8. 推荐的抽离方案

### 8.1 目标目录

建议新建独立仓库或本仓库子包：

```text
hughes-engine/
├── pyproject.toml
├── src/hughes_engine/
│   ├── __init__.py
│   ├── api.py          # 稳定公共类型和 simulate_hughes
│   ├── numerics.py     # Bellman、方向恢复、CFL、守恒通量
│   ├── model.py        # 输入、状态、结果 dataclass
│   ├── adapters/
│   │   └── toml.py     # 当前四文件配置的兼容适配器
│   └── outputs.py      # 可选的 CSV/JSON/绘图写出
└── tests/
```

公共层只暴露纯 Hughes 概念，不暴露 `M`、`allowed_mask`、gate 或 capacity schedule。兼容适配器可以继续读取当前 TOML，但应在边界处拒绝任何非纯 Hughes 配置。

### 8.2 建议的公共接口

```python
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class HughesConfig:
    dx: float
    vmax: float
    rho_max: float
    steps: int
    time_horizon: float
    cfl: float = 0.9
    dt_cap: float = 0.18
    bellman_every: int = 1
    bellman_f_eps: float = 0.08
    bellman_backend: str = "sweeping"


@dataclass(frozen=True)
class HughesProblem:
    walkable: np.ndarray
    initial_density: np.ndarray
    exit_mask: np.ndarray


@dataclass(frozen=True)
class HughesResult:
    final_density: np.ndarray
    simulated_time: float
    step_count: int
    initial_mass: float
    remaining_mass: float
    evacuated_mass: float


StepObserver = Callable[[int, float, np.ndarray], None]


def simulate_hughes(
    problem: HughesProblem,
    config: HughesConfig,
    *,
    observer: StepObserver | None = None,
) -> HughesResult:
    ...
```

接口边界应完成一次集中校验：三个数组形状一致，mask 为布尔数组，密度有限且非负，出口是可通行域的子集，`dx/vmax/rho_max` 为正数。内部实现可继续构造 `M=I` 和全方向 `allowed_mask` 来复用现有内核，但这些兼容细节不应成为新项目的公共契约。

### 8.3 文件迁移映射

| 当前文件 | 抽离后的处理 |
|---|---|
| `crowd_bellman/core.py` | 保留基础 Bellman、方向恢复、CFL、通量和密度更新；把多阶段转移留在扩展包 |
| `crowd_bellman/scenes.py` | 新建只含 `HughesConfig/Problem/Result` 的模型；不向公共层暴露 M、方向和限流类型 |
| `crowd_bellman/runner.py` | 拆为无 I/O 的仿真循环与可选输出 writer |
| `crowd_bellman/metrics.py` | 只保留基础质量、时间和密度指标；通道与 gate 指标移入扩展模块 |
| `crowd_bellman/plotting.py` | 作为可选依赖，不由核心循环强制调用 |
| `loaders/`、`compilers/` | 作为 TOML adapter 保留，或由另一个项目直接构造 `HughesProblem` |
| `g1` 至 `g7`、HCMBO 和 bund runner | 不进入通用 Hughes 包 |

### 8.4 推荐抽离顺序

1. 先增加一个只描述输入和结果的公共契约，以及边界校验测试。
2. 从 `simulate_case()` 提取无文件 I/O 的单群体循环，保持现有算法和默认值不变。
3. 在新循环内部固定构造 `M=I` 和全方向 mask，先保证与当前纯 Hughes 配置结果逐步一致。
4. 将指标和绘图改为 observer/writer 插件，核心函数只返回结果。
5. 将 TOML 工作流改为调用新的公共 API，作为兼容层而不是核心依赖。
6. 增加 `pyproject.toml`、版本号和最小依赖；将 Numba 与绘图声明为可选或明确依赖。
7. 用固定小场景比较旧入口与新包的 `phi`、方向场、每步密度、时间步长和质量收支。
8. 两个项目都改为依赖同一已发布版本或同一 Git tag，避免复制后分叉。

### 8.5 抽离验收门槛

抽离完成至少应满足：

- 旧配置入口和新 API 在纯 Hughes 案例上逐步数值一致；
- 核心包无需 Matplotlib 也能运行；
- 不传 observer/writer 时不产生任何文件；
- 公共 API 中不出现 M、通道方向或入口限流字段；
- 质量守恒、出口移除、障碍不可穿越和后端一致性测试通过；
- 安装方式为标准的 `pip install` 或固定版本依赖，不再要求手工修改 `PYTHONPATH`。

## 9. 关键源码索引

- `codes/crowd_bellman/config_workflow.py`：四文件配置的加载、编译、运行和快照编排。
- `codes/crowd_bellman/compilers/config_compiler.py`：默认 `M=I`、全方向 mask，以及可选 inflow/gate 的编译逻辑。
- `codes/crowd_bellman/core.py`：Bellman、最优方向、CFL、通量限制和密度更新数值内核。
- `codes/crowd_bellman/runner.py`：当前完整仿真循环。
- `codes/crowd_bellman/scenes.py`：运行时数据结构。
- `codes/scenes/README.md`：现有 TOML 字段参考。

## 10. 当前验证记录

本指南撰写时已基于当前源码执行 2 步短仿真验证，使用单群体、无 stage controls、无 allowed directions、无 channels、无 inflow groups、无 capacity controls。编译结果为 `M=I`，且 `channels=0`、`inflows=0`、`gates=0`、`capacity schedules=0`，仿真成功完成。验证说明当前代码路径支持直接运行纯 Hughes 模式，但不替代另一个项目在其真实场景、网格尺度和单位体系下的收敛性验证。
