# Scenes Config Guide

`codes/scenes/` 下的配置采用 4 个 TOML 文件分工：

- `run.toml`: 运行入口，负责把其他 3 个文件串起来，并定义数值参数与目标函数参数。
- `scene.toml`: 场景几何，负责区域、障碍、出口、通道统计口径。
- `population.toml`: 初始人群，负责哪些群体从哪里开始、初始密度是多少。
- `routes.toml`: 路线与控制逻辑，负责 case 标识、阶段切换、分流和控制策略。

主入口命令：

```bash
python codes/simulate_from_config.py --config codes/scenes/examples/single_stage/run.toml
```

可直接参考的样例：

- `codes/scenes/examples/bund_simplified/`: 南京东路-外滩简化场景，`5/6` 号通道进场，平台向南游览，再分流为 `9/10` 号通道离场或 `5/6` 原路返回
- `codes/scenes/examples/g2_multistage_directional/`: G2 多阶段浏览场景，基线偏好经中通道进场、南向游览、再经中通道回流，可在批量实验里扫描三条通道的方向设置
- `codes/scenes/examples/single_stage/`: 最小单阶段例子
- `codes/scenes/examples/multi_stage/`: 多阶段与固定概率分流例子
- `codes/scenes/examples/three_channel_hardcoded/`: 三通道主场景例子
- `codes/scenes/examples/tour_hardcoded/`: 游览型多阶段场景例子

## 1. run.toml

`run.toml` 是唯一直接传给命令行入口的文件。

最小结构：

```toml
[simulation]
nx = 96
ny = 72
dx = 0.5
steps = 600
time_horizon = 40.0
vmax = 1.5
rho_max = 5.0
rho_init = 2.2
bellman_every = 1
bellman_f_eps = 0.08
cfl = 0.9
dt_cap = 0.18
save_every = 40

[objective]
name = "default_objective"
lambda_j1 = 1.0
lambda_j2 = 1.0
lambda_j5 = 1.0
rho_safe = 3.5
use_normalized_terms = false
j1_scale = 1.0
j2_scale = 1.0
j5_scale = 1.0

[scene]
file = "scene.toml"

[population]
file = "population.toml"

[routes]
file = "routes.toml"

[outputs]
output_root = "../../../results/my_case"
```

字段说明：

- `[simulation]`
  - 控制网格、时间推进、速度函数和保存频率。
  - 这些字段会直接构造成 `SimulationConfig`，因此字段名必须和代码一致。
- `[objective]`
  - 定义 `J = lambda_j1 * J1 + lambda_j2 * J2 + lambda_j5 * J5`。
  - `rho_safe` 是高密暴露阈值。
  - `use_normalized_terms = true` 时，会先除以 `j1_scale/j2_scale/j5_scale` 再加权。
- `[scene] / [population] / [routes]`
  - `file` 填相对路径或绝对路径都可以。
  - 相对路径是相对于当前 `run.toml` 所在目录解析。
- `[outputs]`
  - `output_root` 是结果根目录。
  - 每个 case 会在里面再创建一个以 `case_id` 命名的子目录。

## 2. scene.toml

`scene.toml` 只负责定义几何和统计区域，不负责人群和路线。

最小结构：

```toml
block_boundaries = true
obstacles = ["wall_1", "wall_2"]

[[regions]]
name = "spawn_left"
x0 = 1
x1 = 15
y0 = 1
y1 = 71

[[regions]]
name = "exit_east"
x0 = 94
x1 = 95
y0 = 1
y1 = 71

[[exits]]
name = "exit_east"
region = "exit_east"

[[channels]]
name = "main_channel"
region = "spawn_left"
probe_x = 10
```

字段说明：

- `block_boundaries`
  - `true` 时自动把四周边界设为不可通行。
- `obstacles`
  - 列表中的每个名字都必须在 `[[regions]]` 里定义过。
  - 这些区域会被从可通行区域里扣掉。
- `[[regions]]`
  - 当前仅支持轴对齐矩形区域。
  - `x0:x1, y0:y1` 采用左闭右开切片语义。
  - 所有后续引用都靠 `name`。
- `[[exits]]`
  - 用于构造出口 mask。
  - 支持 `region = "name"` 或 `regions = ["a", "b"]`。
- `[[channels]]`
  - 用于统计通道平均密度和累计流量。
  - 支持 `region = "name"` 或 `regions = ["a", "b"]`。
  - `probe_x` 是统计穿越流量时采用的 x 位置；不写时会取该区域 x 中心。

建议写法：

- 把“几何区域”和“统计区域”都命名清楚，后面 `routes.toml` 会复用这些名称。
- 先定义通道、平台、出口、障碍，再写 `obstacles`、`exits`、`channels`。
- 如果某个区域只是用来做控制区或决策区，也要先在 `[[regions]]` 里声明。

## 3. population.toml

`population.toml` 负责初始群体分布。

最小结构：

```toml
[[initial_groups]]
group_id = "main_crowd"
stage_id = "main_exit"
region = "spawn_left"
density = 2.2

[[inflow_groups]]
group_id = "late_arrivals"
stage_id = "main_exit"
region = "spawn_left"
rate = 0.8
time_start = 5.0
time_end = 40.0
rho_cap = 2.5
```

字段说明：

- `group_id`
  - 当前主要用于配置可读性，运行时不会单独作为主键。
- `stage_id`
  - 必须和 `routes.toml` 中某个 `[[stages]].stage_id` 一致。
  - 该群体会被放到对应阶段的 `group_key` 上。
- `region`
  - 必须引用 `scene.toml` 中已定义的区域名。
- `density`
  - 初始密度值，会在对应区域内均匀赋值。

`[[inflow_groups]]` 用于持续入流源。

- `group_id`
  - 仅用于可读性和输出标识。
- `stage_id`
  - 必须指向已有阶段；入流质量会直接注入该阶段群体。
- `region`
  - 入流作用区域，必须是 `scene.toml` 中已定义区域。
- `rate`
  - 总入流质量速率，单位是“每单位时间注入到整个区域的总质量”。
- `time_start` / `time_end`
  - 入流生效时间窗；`time_end` 可省略，表示一直持续到仿真结束。
- `rho_cap`
  - 源区总密度上限；达到该上限后该步不再继续注入。

注意：

- 同一个 `stage_id` 可以有多个 `[[initial_groups]]` 条目，它们会叠加。
- `[[initial_groups]]` 和 `[[inflow_groups]]` 可以同时存在。

## 4. routes.toml

`routes.toml` 是最关键的文件，负责定义 case、阶段、控制策略和阶段切换。

最小单阶段结构：

```toml
[case]
case_id = "case1_baseline"
title = "Case 1: baseline"

[[stages]]
stage_id = "main_exit"
group_key = [1, 1]
goal_region = "exit_east"
sink_region = "exit_east"
```

多阶段结构示例：

```toml
[case]
case_id = "phase1_multi_stage"
title = "Phase 1 Multi Stage Config"

[[stages]]
stage_id = "entry"
group_key = [1, 1]
goal_region = "platform_upper"
decision_region = "platform_upper"
next_stage = "tour_down"
kappa = 2.0

[[stages.controls]]
mode = "target_region"
target_region = "platform_upper"
alpha = 9.0
beta = 0.3

[[stages]]
stage_id = "tour_down"
group_key = [2, 1]
goal_region = "platform_lower"
decision_region = "platform_lower"
kappa = 1.8

[[stages.targets]]
stage_id = "route8"
probability = 0.2

[[stages.targets]]
stage_id = "route9"
probability = 0.3

[[stages.targets]]
stage_id = "route10"
probability = 0.5
```

### 4.1 [case]

- `case_id`: 结果目录名和 summary 中的 case 主键。
- `title`: 图表和 summary 中显示的标题。

### 4.2 [[stages]]

每个 `[[stages]]` 定义一个阶段群体。

必填字段：

- `stage_id`: 阶段标识，必须唯一。
- `group_key = [a, b]`: 两个整数，运行时群体键，必须唯一。
- `goal_region` 或 `goal_regions`: 该阶段 Bellman 求解的目标区域。

常用可选字段：

- `sink_region` 或 `sink_regions`
  - 真正从系统中移除质量的区域。
  - 如果不写且该阶段没有转移，则默认等于 `goal_region`。
  - 如果该阶段有 `next_stage` 或 `targets`，默认不会自动出流。
- `decision_region` 或 `decision_regions`
  - 发生阶段切换或概率分流的区域。
  - 不写时默认用 `goal_region`。
- `allowed_directions`
  - 限制可选方向，例如 `["E"]`、`["E", "NE", "SE"]`。
- `next_stage`
  - 确定性切换到下一个阶段。
- `targets`
  - 概率分流目标，和 `next_stage` 二选一，不能同时写。
- `kappa`
  - 阶段切换率。

### 4.3 [[stages.controls]]

每个阶段可以挂多个控制项，按声明顺序覆盖到区域上。

当前支持的 `mode`：

- `identity`
  - 不做任何修改。
- `isotropic`
  - 把区域内张量改成各向同性。
  - 需要 `value`。
- `fixed_direction`
  - 在区域内设置固定优先方向。
  - 需要 `direction`, `alpha`, `beta`。
- `target_region`
  - 朝某个区域中心施加各向异性引导。
  - 需要 `target_region`, `alpha`, `beta`。
- `target_point`
  - 朝某个点施加各向异性引导。
  - 需要 `target_point = [x, y]`, `alpha`, `beta`。

控制项通用字段：

- `region`
  - 控制生效区域名。
  - 不写时默认对整个可通行域生效。
- `allowed_directions`
  - 可以在控制区局部覆盖方向约束。

方向名必须来自 8 邻域方向集：

- `E`, `W`, `N`, `S`, `NE`, `NW`, `SE`, `SW`

常见示例：

```toml
[[stages.controls]]
mode = "fixed_direction"
region = "middle_channel"
direction = "E"
alpha = 10.0
beta = 0.2
allowed_directions = ["E"]
```

```toml
[[stages.controls]]
mode = "target_region"
region = "feeder_band"
target_region = "platform_upper"
alpha = 8.0
beta = 0.35
```

### 4.4 [[stages.targets]]

用于固定概率分流。

```toml
[[stages.targets]]
stage_id = "route8"
probability = 0.2
```

要求：

- `stage_id` 必须指向已定义阶段。
- 一个阶段下的所有 `probability` 建议加总为 1.0。
- 当前代码不会自动帮你补齐缺失概率。

## 5. 推荐编写顺序

推荐按下面顺序写新 case：

1. 先写 `scene.toml`
   - 把所有会被引用的区域名先定下来。
2. 再写 `routes.toml`
   - 先定义阶段，再补控制区和切换逻辑。
3. 再写 `population.toml`
   - 把初始群体绑定到已有阶段。
4. 最后写 `run.toml`
   - 串起 3 个文件，并设定 simulation/objective/output。

## 6. 常见错误

- 区域名拼错
  - `scene.toml` 中没定义，`routes.toml` 或 `population.toml` 却引用了。
- `stage_id` 不匹配
  - `population.toml` 写的 `stage_id` 在 `routes.toml` 里不存在。
- `group_key` 重复
  - 每个阶段必须唯一。
- 同时写 `next_stage` 和 `targets`
  - 当前实现不允许。
- `goal_region` 为空或被障碍完全覆盖
  - 编译阶段会直接报错。
- `allowed_directions` 或 `direction` 名称非法
  - 只能用当前支持的 8 个方向名。

## 7. 验证建议

每次写完新配置，建议至少做两步：

1. 先跑一个小 smoke：

```bash
python codes/simulate_from_config.py --config codes/scenes/examples/your_case/run.toml
```

2. 检查输出目录：

- `summary.json` 是否生成
- `config_snapshot/` 是否包含 4 份配置快照
- `timeseries.csv` 是否有合理数值
- `snapshot_*.png` 是否能看出人群往预期方向移动

如果是批量比较，还可以再跑：

```bash
python codes/evaluate_objectives.py --input codes/results/your_group/comparison_summary.json --weights codes/scenes/examples/objective_sets/section_5_1.toml
```
