# HCMBO 相对 TPE-Mixed BO 的改进咨询报告

**日期**：2026年5月25日  
**用途**：用于向优化方法、交通人群仿真和实验设计专家请教  
**核心问题**：当前 G6 横向实验中，HCMBO 尚未在全方法口径下优于 `tpe_mixed_bo`。需要判断 HCMBO 应如何进一步改进，才能在公平预算、统一高保真评价和可解释管控约束下实现相对 TPE-Mixed BO 的稳定优势。

---

## 1. 报告摘要

本报告围绕当前 CrowdModels 项目中入口容量控制版本的 HCMBO 方法展开。模型已经从旧的方向与几何引导强度优化，切换为当前的方向配置与内部入口通行速率联合优化：

```text
z = (s, q)
```

其中 `s` 是通道方向配置，`q` 是内部通道入口的分段通行速率上限。几何引导强度 `eta` 不再作为优化变量，而是固定为模型参数 `eta0`。

当前已经完成两组关键实验：

1. **G5 消融与小预算长时程实验**：验证 `z=(s,q)` 是否有优化潜力，并暴露低保真筛选、候选来源归因和入口排队约束问题。
2. **G6 横向对比实验**：在相同场景、相同高保真评价和相同优化预算下，对比 `baseline_prior_best`、`random_search`、`pure_sa`、`enum_de`、`tpe_mixed_bo` 和 `hcmbo_proposed`。

当前 G6 的主要结果为：

| 方法 | 平均 best HF objective | 中位数 | 最好值 | 最差值 | 可行率 |
|---|---:|---:|---:|---:|---:|
| baseline_prior_best | 5.477363 | 5.477363 | 5.477363 | 5.477363 | 0% |
| random_search | 3.011173 | 2.979732 | 2.882092 | 3.228684 | 40% |
| pure_sa | 2.924760 | 2.920316 | 2.746847 | 3.104941 | 60% |
| enum_de | 3.059686 | 3.100384 | 2.732891 | 3.250017 | 40% |
| hcmbo_proposed | 2.884847 | 2.861739 | 2.700227 | 3.042659 | 80% |
| tpe_mixed_bo | 2.740328 | 2.628788 | 2.518036 | 3.134514 | 80% |

从这个表看，HCMBO 在排除 TPE 的方法集合中表现最好，并且可行率达到 80%。但是在全方法口径下，TPE-Mixed BO 的平均目标值、最好目标值和中位数均优于 HCMBO。因此，当前不能写成“HCMBO 已经优于 TPE-Mixed BO”。更准确的判断是：

```text
HCMBO 当前优于 baseline、random search、pure SA 和 enum-DE 的趋势较清楚；
但相对于 TPE-Mixed BO 仍有差距，尤其是在 J2 高密度暴露项和整体目标值上。
```

本报告建议将下一阶段目标定义为：

```text
在相同预算、相同高保真复核、相同随机种子和可行解优先排序口径下，
改进 HCMBO，使其在平均 best feasible HF objective、配对胜率、可行率和入口排队指标上稳定优于 TPE-Mixed BO。
```

---

## 2. 前因后果

### 2.1 模型为什么从旧变量切换到 `z=(s,q)`

早期优化框架主要围绕通道方向 `s` 和几何引导强度 `eta` 展开。后续分析认为，`eta` 更适合作为固定的几何响应参数，而不适合作为现场管理方可以实时调节的控制量。现场真正可操作的管理动作包括：

- 调整某条通道允许正向、反向、双向或关闭；
- 调整某条通道入口的放行速率；
- 通过限流把拥堵从通道内部转移到入口上游的可控等待区。

因此，当前正式模型把控制变量写为：

```text
z = (s, q)
```

其中：

- `s_c in {E, W, FREE, CLOSED}` 或等价数学记号 `{+1, -1, 0, empty}`；
- `q_c^+(t)` 和 `q_c^-(t)` 是第 `c` 条通道两个方向入口的最大允许通行速率；
- `q` 是内部界面通量约束，不是外部边界入流；
- 如果尝试进入通道的流量超过 `q`，超出部分留在入口上游等待，不从系统中删除。

这个建模改变使优化问题更贴近实际人群管控，也使优化变量变成典型的混合变量结构：

- 离散变量：通道方向配置 `s`；
- 连续变量：在给定 `s` 下的容量曲线参数 `x`，经过映射 `q=T_s(x)` 生成实际容量。

### 2.2 G5 结果为什么推动 G6 横向实验

G5 是当前 HCMBO 改进的前置实验。它不是最终横向比较，而是用于回答以下问题：

1. `z=(s,q)` 是否比单独优化方向或单独使用先验容量更有潜力；
2. 低保真筛选是否能可靠筛出长时程高保真下的好方向；
3. HCMBO 内层 BO、DFO 精修、等待惩罚等组件是否有效；
4. 当前 `feasible` 标记是否足以代表方案可执行。

G5 小预算长时程实验使用：

```text
optimization.steps = 1600
optimization.time_horizon = 160.0
high_fidelity.steps = 1600
high_fidelity.time_horizon = 160.0
direction_candidate_limit = 12
shortlist_size = 4
initial_samples = 8
bo_iterations = 12
dfo_evaluations = 5
random_search_evaluations = 100
```

G5 的主要结果为：

| 实验 | best objective | feasible | 最优方向配置 |
|---|---:|---:|---|
| no_jb | 2.601517 | True | top:W, middle:E, lower_middle:E, bottom:W |
| no_lf_selection | 2.614016 | True | top:W, middle:E, lower_middle:E, bottom:W |
| main_hcmbo_full | 2.944064 | False | top:W, middle:E, lower_middle:E, bottom:W |
| no_dfo | 2.944064 | False | top:W, middle:E, lower_middle:E, bottom:W |
| random_search | 2.944064 | False | top:W, middle:E, lower_middle:E, bottom:W |
| only_q_prior | 3.313980 | True | top:FREE, middle:E, lower_middle:W, bottom:FREE |
| only_s_high | 3.375114 | True | top:E, middle:W, lower_middle:W, bottom:E |

G5 得到三条重要经验。

第一，`z=(s,q)` 的联合优化方向是有意义的。`no_lf_selection` 明显优于 `only_q_prior` 和 `only_s_high`，说明只给方向或只给容量先验都不够。

第二，默认低保真筛选存在严重风险。后续表现最好的方向：

```text
top:W, middle:E, lower_middle:E, bottom:W
```

在 `main_hcmbo_full` 的短时低保真筛选中只排第 10，因此被筛掉。`no_lf_selection` 保留全部 12 个方向后，才找到更好的高保真候选。

第三，`main_hcmbo_full` 的 best 不能直接视为 HCMBO best。G5 中 `main_hcmbo_full` 的最优候选来自内部 random search，而不是 HCMBO 自身生成的候选。这说明后续必须拆分候选来源，不能把混合候选池的最优值当成 HCMBO 的纯性能。

基于这些结论，G6 设计成横向公平比较，不再只看消融矩阵，而是把所有方法放到同一评价口径下。

---

## 3. 当前方法定义

### 3.1 HCMBO 的全称与定位

当前 HCMBO 写作：

```text
Hierarchical Constrained Mixed-variable Black-box Optimization
```

中文可写作：

```text
分层约束混合变量黑箱优化
```

它的核心思想是：

1. 外层处理离散方向配置 `s`；
2. 内层在固定方向下优化连续容量参数 `x`；
3. 通过映射 `q=T_s(x)` 保证方向和容量硬一致；
4. 在统一高保真设置下复核候选；
5. 最终排序只依据高保真结果。

### 3.2 当前 G6 中 HCMBO 的实现口径

G6 的 `hcmbo_proposed` 与 G5 初版相比已经做了一个关键改变：不再使用低保真 hard shortlist，而是把 `shortlist_size` 设置为 `direction_candidate_limit`，即保留全部 12 个方向候选。

当前 G6 HCMBO 的实际预算结构为：

```text
direction_candidate_count = 12
每个方向 initial_samples = 8
每个方向 bo_iterations = 12
每个方向 dfo_evaluations = 5
每个方向结构化评估 = 8 + 12 + 5 = 25
结构化 HCMBO 评估 = 12 * 25 = 300
内部 random_search 评估 = 100
总优化评估 = 400
高保真复核候选 = 10
```

因此，当前 `hcmbo_proposed` 不是 400 次纯结构化 BO，而是：

```text
300 次方向分层结构化搜索 + 100 次内部随机搜索
```

这一点对专家咨询很重要。若要严谨证明 HCMBO 的结构化搜索优于 TPE，需要进一步输出：

- HCMBO structured-only best；
- HCMBO internal-random-only best；
- combined-pool best；
- 高保真 top-10 中每个候选的来源。

当前 G6 汇总表中，高保真复核后的 best `source` 都被写成 `high_fidelity_recheck`，没有保留原始生成来源，因此还不能直接判断每个 HCMBO seed 的 best 是来自结构化 BO 还是内部随机搜索。

### 3.3 当前 TPE-Mixed BO 的实现口径

TPE 的全称是：

```text
Tree-structured Parzen Estimator
```

当前代码中的 `tpe_mixed_bo` 标记为：

```text
local_tree_parzen_style
```

它不是直接调用 Optuna 的官方 TPE，而是一个本地实现的 TPE-style 混合变量搜索：

1. 从同一批方向候选中随机 warmup；
2. 将历史候选按目标值排序；
3. 取前 25% 作为 elite；
4. 以 80% 概率围绕 elite 的容量向量做高斯扰动；
5. 以 20% 概率重新随机采样方向和容量；
6. 总优化评估次数为 400；
7. 高保真复核 top 10。

这意味着当前 TPE 基线虽然叫 `tpe_mixed_bo`，但严格来说更接近“局部 elite sampling 的 tree-Parzen-style 方法”。若最终论文要面对高水平审稿，建议至少做一次官方 Optuna TPE 或 SMAC 类基线作为补充，否则专家可能会质疑外部强基线的标准性。

---

## 4. G6 实验设计

### 4.1 统一场景与预算

G6 使用当前四通道外滩观景平台场景，控制变量为 `z=(s,q)`。实验结果目录为：

```text
codes/results/g6_horizontal_comparison
```

配置文件为：

```text
codes/scenes/examples/g6_horizontal_comparison/g6.toml
```

核心设置：

```text
profile = "full"
seeds = [11, 23, 37, 51, 73]
methods = [
  "baseline_prior_best",
  "random_search",
  "pure_sa",
  "tpe_mixed_bo",
  "enum_de",
  "hcmbo_proposed",
]

optimization.steps = 1600
optimization.time_horizon = 160.0
high_fidelity.steps = 1600
high_fidelity.time_horizon = 160.0
random_search_evaluations = 400
high_fidelity_top_k = 10
```

所有横向方法使用相同优化保真和高保真复核设置。

### 4.2 目标函数与可行性

当前目标函数为：

```text
J = lambda_j1 * J1_eval
  + lambda_j2 * J2_eval
  + lambda_j5 * J5_eval
  + lambda_jb * JB_normalized
  + lambda_jr * JR_normalized
  + infeasible_penalty
```

G6 权重为：

```text
lambda_j1 = 1.0
lambda_j2 = 1.0
lambda_j5 = 1.0
lambda_jb = 1.0
lambda_jr = 0.1
```

可行性当前主要由 `cap_removed_relative` 判定：

```text
cap_removed_relative = final_cap_removed_cumulative / total_mass_reference
feasible = cap_removed_relative <= 0.02
```

这里需要特别说明：当前 `feasible=True` 只表示密度 cap 删除量没有超过阈值，并不表示入口拒绝流、等待峰值或排队时间一定较小。因此，当前结果中存在 `feasible=True` 但 `gate_rejected` 仍然很高的方案。

### 4.3 方法比较对象

G6 主方法含义如下：

| 方法 | 含义 | 作用 |
|---|---|---|
| baseline_prior_best | 先验基线 | 判断优化是否明显优于专家/规则方案 |
| random_search | 随机搜索 | 无模型黑箱搜索下界 |
| pure_sa | 纯模拟退火 | 传统混合变量扰动搜索 |
| enum_de | 方向枚举加差分进化 | 每个方向上做连续全局优化 |
| tpe_mixed_bo | TPE-style 混合变量 BO | 当前最强外部黑箱基线 |
| hcmbo_proposed | HCMBO | 本文结构化分层约束优化方法 |

---

## 5. G6 结果分析

### 5.1 全方法结果

G6 的全方法排序如下：

| 排名 | 方法 | 平均 best HF objective | 可行率 |
|---:|---|---:|---:|
| 1 | tpe_mixed_bo | 2.740328 | 80% |
| 2 | hcmbo_proposed | 2.884847 | 80% |
| 3 | pure_sa | 2.924760 | 60% |
| 4 | random_search | 3.011173 | 40% |
| 5 | enum_de | 3.059686 | 40% |
| 6 | baseline_prior_best | 5.477363 | 0% |

HCMBO 的优势：

- 相比 baseline，平均目标从 5.477363 降到 2.884847；
- 相比 random search，平均目标降低约 0.126326；
- 相比 pure SA，平均目标略低；
- 相比 enum-DE，平均目标更低，可行率更高；
- 在排除 TPE 的图表集合中是当前最佳方法。

HCMBO 的不足：

- 相比 TPE-Mixed BO，平均目标高 0.144519；
- 中位数高 0.232951；
- 最好值高 0.182190；
- 配对 seed 中只赢 1 次，输 4 次；
- J2、J5、JB、JR 和 `gate_rejected` 的平均值均未优于 TPE。

### 5.2 HCMBO 与 TPE 的逐 seed 对比

| seed | HCMBO objective | TPE objective | HCMBO - TPE | HCMBO feasible | TPE feasible | HCMBO gate_rejected | TPE gate_rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 2.861739 | 2.596353 | +0.265386 | True | True | 2680.56 | 2843.56 |
| 23 | 3.028568 | 2.518036 | +0.510532 | True | True | 3197.61 | 2815.21 |
| 37 | 2.700227 | 2.628788 | +0.071438 | True | False | 2970.60 | 2805.96 |
| 51 | 3.042659 | 2.823949 | +0.218710 | False | True | 3206.29 | 3479.34 |
| 73 | 2.791041 | 3.134514 | -0.343472 | True | True | 3345.74 | 2774.24 |

按当前 `best_hf_objective_default` 口径，HCMBO 仅在 seed 73 优于 TPE。seed 37 比较特殊：TPE 的最低 objective 候选不可行，但 TPE 在该 seed 仍有可行候选，且 best feasible objective 为 2.644526，仍低于 HCMBO 的 2.700227。因此即使改成 feasible-only 口径，当前 TPE 仍然领先。

### 5.3 Feasible-only 口径补算

当前 `G6_method_summary.csv` 的 best 是“目标值最小候选”，不是“可行候选中的目标值最小候选”。从 `G6_hf_candidates.csv` 后处理可得到 HCMBO 与 TPE 的 best feasible 结果：

| 方法 | seed 11 | seed 23 | seed 37 | seed 51 | seed 73 | mean best feasible |
|---|---:|---:|---:|---:|---:|---:|
| hcmbo_proposed | 2.861739 | 3.028568 | 2.700227 | 3.121375 | 2.791041 | 2.900590 |
| tpe_mixed_bo | 2.596353 | 2.518036 | 2.644526 | 2.823949 | 3.134514 | 2.743476 |

可行解优先后，TPE 的平均 best feasible objective 仍低于 HCMBO：

```text
HCMBO feasible-only mean = 2.900590
TPE feasible-only mean   = 2.743476
gap                      = 0.157114
```

因此，下一步不能只通过“过滤不可行候选”来实现 HCMBO 超越 TPE，必须真正改进 HCMBO 搜索能力或目标/约束建模。

### 5.4 分项目标对比

按 5 个 seed 平均，关键分项如下：

| 方法 | J1 | J2 | J5 | JB | JR | gate_rejected |
|---|---:|---:|---:|---:|---:|---:|
| baseline_prior_best | 0.3577 | 2.7525 | 0.7278 | 0.0647 | 0.0000 | 4055.57 |
| random_search | 0.3877 | 2.1375 | 0.2841 | 0.0588 | 0.1061 | 3042.27 |
| pure_sa | 0.4010 | 2.0526 | 0.3258 | 0.0624 | 0.1502 | 2997.82 |
| enum_de | 0.3952 | 2.1688 | 0.3021 | 0.0590 | 0.0874 | 2981.20 |
| hcmbo_proposed | 0.4006 | 2.0631 | 0.2804 | 0.0633 | 0.1424 | 3080.16 |
| tpe_mixed_bo | 0.3886 | 2.0151 | 0.2658 | 0.0573 | 0.1320 | 2943.66 |

这张表说明，TPE 的优势不是单个偶然 seed 造成的。TPE 在平均 `J2_eval`、`J5_eval`、`JB_normalized`、`JR_normalized` 和 `gate_rejected` 上都优于 HCMBO。HCMBO 当前没有在分项目标上形成明显补偿优势。

需要注意，当前 `J2_eval` 的量级大约在 2.0 左右，是总目标中最主要的差异来源。HCMBO 要超越 TPE，首先要降低高密度暴露项 `J2_eval`，或者在新的管理偏好下明确强化均衡、平滑、等待等结构性指标。

### 5.5 统计检验结果

当前只有 5 个随机种子，统计检验能力有限。HCMBO vs TPE 的结果为：

```text
paired_seed_count = 5
mean_delta_hcmbo_minus_tpe = 0.144519
median_delta_hcmbo_minus_tpe = 0.218710
HCMBO wins = 1
TPE wins = 4
sign_test_p = 0.375
Vargha-Delaney A12 = 0.2
Cliff's delta = -0.6
```

这说明当前数据趋势上 TPE 更强，但由于 seed 数少，不能把差异写成统计显著。反过来，也不能把 HCMBO 写成已经超过 TPE。对专家咨询时，建议表述为：

```text
当前 G6 结果显示 TPE-Mixed BO 是最强基线；
HCMBO 在非 TPE 方法中表现最好，但仍需要方法改进才能在全方法比较中胜出。
```

---

## 6. 为什么当前 HCMBO 没有超过 TPE

### 6.1 HCMBO 的预算被均匀分配到所有方向，TPE 更容易集中到好区域

当前 HCMBO 对 12 个方向均分结构化预算，每个方向只有 25 次评估：

```text
8 initial + 12 BO + 5 DFO = 25
```

如果某个方向本身很差，这 25 次预算会被浪费。如果某个方向很好，25 次又可能不够充分优化容量曲线。TPE-style 方法虽然不显式分层，但它会围绕 elite 候选持续扰动，更容易把预算集中到已经表现好的方向和容量区域。

G5 已经证明，方向选择非常关键，且短时低保真筛选会误判方向。当前 G6 为避免误筛，把所有方向都保留了，但代价是每个方向内层预算过少。这是 HCMBO 当前性能低于 TPE 的最可能原因之一。

### 6.2 当前 HCMBO 的 BO 代理过弱

当前 `propose_lcb_candidate()` 不是严格的 GP-BO、RF-BO 或 TPE。它大致做法是：

1. 保存已经评估的 `x` 和目标值；
2. 随机生成候选池；
3. 加入围绕当前 best 的局部扰动；
4. 用距离加权均值估计候选目标；
5. 用距离表示不确定性；
6. 计算简化 LCB：

```text
acquisition = weighted_mean - kappa * y_std * uncertainty
```

这个代理在低维平滑函数上可能可用，但当前问题具有：

- 方向与容量强耦合；
- 容量变量维度随方向变化；
- 仿真目标存在非平滑阈值、排队和拥堵突变；
- `J2` 高密度暴露对容量变化非常敏感；
- 高保真 top-k 选择存在可行性与随机性。

因此，当前 HCMBO 的内层 BO 可能不足以在每个方向 25 次预算内找到接近 TPE 的容量曲线。

### 6.3 HCMBO 的结构化优势没有转化为目标函数优势

理论上，HCMBO 的优势应该来自：

- 利用方向结构；
- 在方向固定后降低连续优化维度；
- 用方向和容量硬一致性减少无效候选；
- 更容易解释每个通道的管控策略。

但当前 G6 中，所有方法都使用同一个 `control_from_x()` 映射，TPE 也能自然满足方向和容量一致性。也就是说，HCMBO 的“可行映射优势”已经被所有方法共享，剩下的差别主要是搜索策略。

如果 HCMBO 不进一步利用更强的方向先验、可行性代理、排队约束或容量结构，那么它相对于 TPE 的独有优势就不明显。

### 6.4 当前可行性指标没有约束入口拒绝流

当前 feasible 主要看：

```text
cap_removed_relative <= 0.02
```

但当前所有较优方法的 `gate_rejected` 都仍然接近 3000。HCMBO 平均 `gate_rejected=3080.16`，TPE 平均 `2943.66`。这说明当前目标函数和可行性判定没有充分惩罚入口拒绝流。

如果专家认为入口拒绝流代表真实排队压力，那么当前 objective 的最优方案可能并不是管理上最好的方案。HCMBO 若要体现“结构化管控”的优势，应把入口等待、拒绝流、绑定时间等指标纳入更强约束，而不是只追逐当前标量目标。

### 6.5 TPE 当前在同一最优方向上找到了更好的容量曲线

HCMBO 最好 seed 37 与 TPE 最好 seed 23 都使用了同一方向：

```text
top:W, middle:E, lower_middle:E, bottom:W
```

但 TPE 的目标值更低：

| 指标 | HCMBO best seed 37 | TPE best seed 23 |
|---|---:|---:|
| objective | 2.700227 | 2.518036 |
| J1_eval | 0.409340 | 0.414665 |
| J2_eval | 1.941559 | 1.694826 |
| J5_eval | 0.274207 | 0.338092 |
| JB_normalized | 0.059709 | 0.059643 |
| JR_normalized | 0.154121 | 0.108107 |
| gate_rejected | 2970.60 | 2815.21 |

这个对比说明，差距不只是方向选择，而是容量曲线优化。HCMBO 的容量曲线使 `J5` 更好，但 TPE 大幅降低了 `J2`，且 `JR` 更平滑，最终总目标更低。当前权重下，`J2` 的收益超过了 `J5` 的损失。

---

## 7. 建议的 HCMBO 改进方向

### 7.1 改进一：从均匀分配方向预算改为自适应方向 racing

当前 HCMBO 最大问题是所有方向均分预算。建议改成三阶段方向 racing：

#### 阶段 A：全方向低预算初始化

对 12 个方向都给少量初始化预算，例如：

```text
每方向 4 个容量样本
总预算 12 * 4 = 48
```

每个方向记录：

- 最好 objective；
- 最好 feasible objective；
- `J2_eval`；
- `gate_rejected`；
- `waiting_mass_peak`；
- `binding_time_ratio_max`；
- 容量曲线平滑度。

#### 阶段 B：保留 top 方向并增量分配预算

用 soft score 排序方向，而不是 hard LF screen。可考虑：

```text
score_s = best_objective_s
        + alpha * infeasible_rate_s
        + beta * normalized_gate_rejected_s
        + gamma * uncertainty_s
```

保留 top 4 到 top 6 个方向继续优化。每一轮给这些方向分配更多预算，并定期重新排序。

#### 阶段 C：精修 top 方向

最后只对 top 2 到 top 3 个方向做局部精修：

- trust-region BO；
- local DE；
- coordinate search；
- pattern search；
- 多起点容量扰动。

这样能避免两个极端：

- 不像 G5 那样用短时低保真 hard screen 误删好方向；
- 不像当前 G6 那样对所有方向平均撒预算。

建议实验名：

```text
hcmbo_adaptive_racing
```

### 7.2 改进二：引入约束感知 acquisition

当前 acquisition 只看目标值，不显式建模可行性和排队风险。建议把 HCMBO 内层从单目标 LCB 改为约束感知 acquisition。

候选指标：

```text
objective_value
feasible
cap_removed_relative
gate_rejected
waiting_mass_peak
binding_time_ratio_max
```

可使用如下采样准则：

```text
constrained_score(x)
  = predicted_objective(x)
  - kappa * predicted_uncertainty(x)
  + mu1 * predicted_infeasibility_risk(x)
  + mu2 * predicted_queue_risk(x)
```

或者使用可行性优先：

```text
先最大化 P(feasible and queue_safe)
再在可行区域中最小化 expected objective
```

对当前问题而言，`gate_rejected` 和 `waiting_mass_peak` 不应只作为报告指标，而应进入搜索导向。否则优化器可能继续找到“目标值低但入口排队大”的策略。

建议实验名：

```text
hcmbo_constrained_ei
hcmbo_queue_aware_lcb
```

### 7.3 改进三：增强内层代理模型

当前距离加权 LCB 过于简化。可考虑三种升级路线。

#### 路线 1：Random Forest / Extra Trees surrogate

适合样本少、变量维度中等、目标不平滑的问题。输入为容量参数 `x`，输出为 objective 和 constraint。优点是实现成本较低，能处理非线性和局部突变。

#### 路线 2：Trust-region BO

对每个方向维护局部 trust region。若新候选改善，则扩大区域；若连续失败，则缩小区域。该路线适合当前容量曲线优化，因为同一方向下连续变量空间仍然较高维，但局部好区域可能相对稳定。

#### 路线 3：HCMBO 内部使用 TPE-like per-direction optimizer

这条路线最直接：承认 TPE 在连续/混合局部搜索上强，把 TPE 的优点放进 HCMBO 的内层，但外层仍保留方向分层、约束处理和可解释管控结构。

此时方法可以定义为：

```text
Hierarchical Constrained TPE-assisted MBO
```

关键是论文叙事要清楚：贡献不在“发明一个完全不同的采样器”，而在于“把人群通道管控问题分解成方向层、容量层、约束层和高保真复核层”。专家需要判断这种方法创新是否足够。

### 7.4 改进四：改进容量参数化

当前容量为 4 个时间段的分段常数。对单向通道每个方向是 4 维，对 FREE 通道是 8 维。当前映射中 FREE 通道使用：

```text
qbar_total = max(qbar_plus, qbar_minus)
```

再在 plus/minus 间分配。这可能低估双向通道总容量，或使 FREE 的容量表示与物理通道容量不完全一致。建议专家评估以下替代：

1. `FREE qbar_total = qbar_plus + qbar_minus`；
2. plus 和 minus 独立受各自 `qbar` 限制；
3. 对 FREE 增加方向偏置参数和总容量参数；
4. 对所有通道使用平滑基函数，而不是完全自由的 4 段常数；
5. 对容量加入最小持续时间和变化率约束。

建议实验名：

```text
hcmbo_qmap_sum_free
hcmbo_qmap_independent_free
hcmbo_smooth_capacity_basis
```

### 7.5 改进五：修改最终 high-fidelity top-k 选择

当前高保真复核选取优化阶段 objective 最小的 top 10。若优化保真与高保真之间存在排序误差，top 10 可能缺少多样性。建议 top-k 选择引入：

- objective top candidates；
- feasible top candidates；
- low `J2` candidates；
- low `gate_rejected` candidates；
- direction-diverse candidates；
- capacity-profile-diverse candidates。

例如：

```text
HF top-k = 10
其中：
3 个按 objective
2 个按 feasible objective
2 个按 J2
1 个按 gate_rejected
2 个按方向/容量多样性补齐
```

这样可以减少“中保真排序误差导致高保真漏掉好候选”的风险。

### 7.6 改进六：重新定义可行性和主指标

如果专家认可 `gate_rejected` 代表入口排队压力，则建议把最终比较主指标从：

```text
best_hf_objective_default
```

改为：

```text
best_feasible_hf_objective_under_queue_constraints
```

可行性至少包括：

```text
cap_removed_relative <= threshold_cap
gate_rejected <= threshold_rejected
waiting_mass_peak <= threshold_waiting
binding_time_ratio_max <= threshold_binding
```

这样 HCMBO 的结构化控制优势才有机会显现。否则 TPE 只要找到目标值低的容量曲线，就可能在当前标量目标上领先，即使该方案在现场排队解释上不理想。

---

## 8. 建议的下一轮实验设计

### 8.1 实验 G7-A：后处理诊断，不重跑仿真

目的：在不重跑实验的情况下，先搞清楚当前 HCMBO 输给 TPE 的具体原因。

输入：

```text
codes/results/g6_horizontal_comparison/G6_evaluation_log.csv
codes/results/g6_horizontal_comparison/G6_hf_candidates.csv
各 method/seed/method_summary.json
```

输出：

1. HCMBO high-fidelity top-10 的原始来源：
   - hcmbo_init；
   - hcmbo_bo；
   - hcmbo_dfo；
   - internal random_search。
2. HCMBO vs TPE 的 best feasible 对比；
3. 各 seed 中同方向候选的容量曲线差异；
4. `J2`、`J5`、`gate_rejected` 的 Pareto 散点；
5. 高保真 top-k 候选是否集中于同一方向。

若后处理发现 HCMBO 的 best 多数来自内部 random，则需要先修正方法归因。若 best 多数来自结构化 BO，则继续优化 acquisition 和预算分配。

### 8.2 实验 G7-B：同预算 HCMBO 变体比较

目的：在不改变外部对比基线的情况下，比较 HCMBO 自身改进路线。

建议方法：

| 方法 | 含义 |
|---|---|
| hcmbo_current | 当前 G6 实现 |
| hcmbo_structured_only | 去掉内部 random search，只保留结构化搜索 |
| hcmbo_adaptive_racing | 方向层自适应预算分配 |
| hcmbo_queue_aware_lcb | 目标加排队约束的 acquisition |
| hcmbo_trust_region | 每个方向下 trust-region BO |
| hcmbo_diverse_hf_topk | 多样性高保真候选选择 |

预算：

```text
B = 400
HF top_k = 10
seeds = [11, 23, 37, 51, 73]
```

成功标准：

```text
至少有一个 HCMBO 变体的 mean best feasible objective < 当前 tpe_mixed_bo
并且 feasible rate >= 80%
并且 gate_rejected 不高于当前 HCMBO
```

### 8.3 实验 G7-C：HCMBO 改进版 vs TPE 正式横向比较

目的：验证改进后的 HCMBO 是否真正优于 TPE。

建议方法：

```text
baseline_prior_best
random_search
pure_sa
enum_de
tpe_mixed_bo_current
optuna_tpe_mixed_bo
hcmbo_current
hcmbo_improved
```

预算两档：

```text
Budget-S: B=400, HF top_k=10, seeds=5
Budget-M: B=800, HF top_k=20, seeds=10
```

主指标：

```text
mean best feasible HF objective
median best feasible HF objective
paired win count vs TPE
feasible rate
gate_rejected
waiting_mass_peak
binding_time_ratio_max
convergence AUC
```

统计分析：

```text
paired sign test
Wilcoxon signed-rank test
Vargha-Delaney A12
Cliff's delta
Holm-Bonferroni correction
```

建议声明 HCMBO 优于 TPE 的最低证据标准：

1. HCMBO improved 在 10 个 seed 中至少赢 TPE 7 个；
2. mean best feasible objective 低于 TPE；
3. feasible rate 不低于 TPE；
4. `gate_rejected` 和 `waiting_mass_peak` 不高于 TPE；
5. 效应量至少达到中等水平；
6. 最好能通过 Wilcoxon 或 sign test 给出趋势性或显著性支持。

### 8.4 实验 G7-D：目标函数与约束敏感性

目的：判断 HCMBO 是否在更合理的现场管理目标下更有优势。

建议比较：

```text
default objective
queue-aware objective
strict feasible-only ranking
high lambda_jb
explicit gate_rejected penalty
binding_time constraint
```

建议输出：

| 指标 | 解释 |
|---|---|
| objective | 当前综合目标 |
| best feasible objective | 可行候选最优目标 |
| J1 | 总旅行时间 |
| J2 | 高密度暴露 |
| J5 | 通道负载不均衡 |
| JB | 等待暴露 |
| JR | 控制平滑性 |
| gate_rejected | 被入口容量拒绝的累计通量 |
| waiting_mass_peak | 入口等待峰值 |
| binding_time_ratio_max | 容量长期绑定比例 |

如果 HCMBO 在 queue-aware 目标下相对 TPE 优势更明显，则论文可以把贡献从“纯 objective 最优”调整为“结构化约束管控下更可执行、更稳定”。

---

## 9. 可向专家请教的问题

### 9.1 关于主结论口径

1. 当前是否应以 `best_hf_objective_default` 作为主指标，还是应以 `best_feasible_hf_objective` 作为主指标？
2. 如果某个方法的最低 objective 候选不可行，但存在稍差的可行候选，应如何在主表中呈现？
3. 对人群管控问题，`gate_rejected` 是否应作为硬约束，而不是仅作为诊断指标？
4. `feasible` 只基于 `cap_removed_relative <= 0.02` 是否过弱？

### 9.2 关于 HCMBO 方法创新

1. HCMBO 的贡献应强调“分层约束建模框架”，还是强调“具体 BO 采样器”？
2. 如果内层改用 TPE、RF 或 trust-region BO，是否会削弱 HCMBO 的方法独立性？
3. HCMBO 是否应该保留内部 random search？如果保留，如何避免和外部 random baseline 混淆？
4. 方向层是否应使用 multi-armed bandit / successive halving / racing 机制？

### 9.3 关于 TPE 基线

1. 当前 `local_tree_parzen_style` 是否足以称为 TPE-Mixed BO？
2. 是否必须加入 Optuna TPE 或 SMAC 才能作为高质量期刊中的强基线？
3. 若 HCMBO 只优于当前本地 TPE-style，而不优于 Optuna TPE，论文应如何表述？

### 9.4 关于目标函数

1. 当前 `J2_eval` 在总目标中影响最大，是否符合管理目标？
2. 通道均衡 `J5` 是否需要更高权重，才能体现分流管控价值？
3. `lambda_jb=1` 是否足以约束入口等待？
4. `lambda_jr=0.1` 是否足以约束容量曲线可执行性？
5. 是否应采用 Pareto 前沿而不是单一权重标量化？

### 9.5 关于实验规模

1. 5 个 seed 是否足以作为论文主结果？
2. Budget-S 的 `B=400` 是否对 HCMBO 不利，因为它要覆盖 12 个方向？
3. 是否应增加 Budget-M，使用 `B=800` 或 `B=1200` 检查 HCMBO 是否在更大预算下体现结构化优势？
4. 是否应增加局部入口决策场景，专门验证入口速率控制而不是完整全局场景？

---

## 10. 建议给专家的简短问题陈述

可以将咨询问题压缩为以下版本：

```text
当前模型已从旧的方向-几何强度优化改为方向配置与内部入口通行速率控制联合优化，即 z=(s,q)。
G5 消融实验显示，联合优化有潜力，但短时低保真筛选会误删长时程下的优良方向。
因此 G6 改为横向公平比较，在 B=400、HF top-10、5 个 seed 下比较 baseline、random、pure SA、enum-DE、TPE-Mixed BO 和 HCMBO。

结果显示，HCMBO 在排除 TPE 的方法集合中平均目标最低、可行率最高；
但全方法口径下 TPE-Mixed BO 平均目标 2.7403，HCMBO 为 2.8848，HCMBO 只在 1/5 个 seed 上赢 TPE。
feasible-only 后处理下，TPE 的 mean best feasible objective 仍为 2.7435，HCMBO 为 2.9006。

希望专家判断：
1. 当前主指标与可行性定义是否合理；
2. HCMBO 是否应从均匀方向预算改为自适应 direction racing；
3. 是否应引入约束感知 acquisition 和 queue-aware objective；
4. TPE 基线是否应换成官方 Optuna/SMAC；
5. 怎样设计下一轮实验，才能严谨验证 HCMBO 是否真正优于 TPE。
```

---

## 11. 当前阶段可写入论文的保守结论

在未完成 HCMBO 改进前，建议只写以下保守结论：

1. `z=(s,q)` 的入口容量控制模型明显优于先验 baseline。
2. HCMBO 在非 TPE 的结构化和启发式方法集合中取得最低平均目标和最高可行率。
3. TPE-Mixed BO 是当前全方法口径下最强基线，HCMBO 尚未稳定超过 TPE。
4. HCMBO 的下一步改进应集中在方向预算自适应分配、约束感知 acquisition、入口排队约束和高保真候选多样性选择。
5. 当前所有较优方法仍存在较高 `gate_rejected`，因此仅报告综合 objective 不足以说明方案现场可执行性。

不建议写：

```text
HCMBO 全面优于 TPE-Mixed BO。
```

建议写：

```text
HCMBO 在结构化方法集合中表现最好，并揭示出方向-容量联合控制的有效性；
TPE-Mixed BO 在当前全方法比较中仍是更强的通用混合变量 BO 基线。
后续将通过自适应方向预算、约束感知 acquisition 和 queue-aware objective 进一步提升 HCMBO。
```

---

## 12. 证据文件

本报告主要基于以下本地文件：

```text
methodology/model.md
methodology/optimization.md
codes/scenes/examples/g6_horizontal_comparison/g6.toml
codes/g6_horizontal_comparison.py
codes/crowd_bellman/g5_hcmbo.py
records/20260520-g5-small-1600-analysis.md
reports/20260525-周报.md
codes/results/g6_horizontal_comparison/G6_full_report.md
codes/results/g6_horizontal_comparison/G6_method_summary.csv
codes/results/g6_horizontal_comparison/G6_seed_summary.csv
codes/results/g6_horizontal_comparison/G6_hf_candidates.csv
codes/results/g6_horizontal_comparison/G6_statistical_tests.csv
```

---

## 13. 建议下一步落地工作

优先级从高到低：

1. 写一个 `g6_posthoc_source_audit.py`，从现有结果中恢复或近似判断 HCMBO top-k 候选来源，输出 HCMBO structured-only、internal-random-only 和 combined-pool 的差异。
2. 修改 G6 汇总逻辑，增加 `best_feasible_hf_objective`、`best_feasible_case_id`、`best_feasible_gate_rejected` 和 feasible-only 方法汇总。
3. 实现 `hcmbo_adaptive_racing`，把方向预算从均匀分配改为自适应分配。
4. 实现 `hcmbo_queue_aware_lcb`，把 `gate_rejected`、`waiting_mass_peak` 和 `binding_time_ratio_max` 纳入 acquisition 或可行性约束。
5. 实现 official Optuna TPE baseline，确认当前 TPE-style 结果是否代表强基线。
6. 以 `B=400, seeds=5` 做小规模筛选，再以 `B=800, seeds=10` 做论文级复验。

最终目标不是让 HCMBO 在某张排除 TPE 的图中显得最好，而是在完整、公平、可复核的横向实验中，给出 HCMBO 相对 TPE 的明确优势、适用条件和失败边界。
