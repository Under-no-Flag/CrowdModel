# G6 横向对比实验方案：HCMBO 与外部优化方法比较

## 1. 实验目标

G5 当前主要是纵向消融实验，用来回答“低保真筛选、DFO、等待惩罚、只优化方向、只优化容量”等模块是否有效。G6 应改为横向对比实验，用来回答：

1. 在相同场景、相同目标函数、相同评价预算下，本文提出的 HCMBO 是否优于常用黑箱优化方法。
2. HCMBO 的优势来自结构化变量分解与条件容量映射，而不是来自更大的预算、更多高保真复验或不公平的调参。
3. HCMBO 得到的最优方案是否同时具有较低目标值、可行性和可解释的管控模式。

G6 的主结论应建立在统一高保真评价上，而不是 LF/MF 排名上。

## 2. 与 G5 的关系

G5：纵向消融。

- `main_hcmbo_full`
- `no_lf_selection`
- `no_dfo`
- `no_jb`
- `only_s_high`
- `only_q_prior`
- `random_search`

其中 `no_lf_selection`、`no_dfo`、`no_jb` 是方法内部消融，不适合作为横向外部算法。`only_s_high` 和 `only_q_prior` 可以作为弱结构基线，但也更像消融。G6 应把它们放到辅助表或附录，不作为主横向基线。

G6：横向算法比较。

- Baseline / expert policy
- Random Search
- Pure SA / Mixed Simulated Annealing
- TPE 或 SMAC 类混合变量 BO
- Differential Evolution / CMA-ES / MADS 类连续黑箱优化基线
- Proposed HCMBO

## 3. 推荐主对比算法

为满足高质量期刊的审稿标准，不能只选择明显弱的算法。建议主文放 4 个外部对比 + 1 个本文方法：

| 方法 | 类型 | 选择理由 | 期刊说服力 |
|---|---|---|---|
| Baseline / no-control / prior policy | 工程基线 | 说明优化是否优于无优化和专家规则 | 必须有 |
| Random Search / LHS Random Search | 无模型黑箱搜索 | 显示优化算法是否优于均匀探索 | 必须有 |
| Mixed Simulated Annealing, Pure-SA | 混合变量启发式 | 与旧 G4/SA 思路相关，能说明 HCMBO 优于传统扰动搜索 | 推荐 |
| TPE 或 SMAC | 混合变量贝叶斯优化 | 适合离散-连续条件变量，是强基线 | 强烈推荐 |
| Enum-DE 或 CMA-ES/DE per direction | 连续黑箱全局优化 | 对每个方向优化容量，代表成熟连续优化方法 | 推荐 |
| HCMBO | 本文方法 | 结构化方向-容量混合优化 | 主方法 |

建议主文最终保留：

1. `baseline_prior_best`
2. `random_search`
3. `pure_sa`
4. `tpe_mixed_bo`
5. `enum_de`
6. `hcmbo_proposed`

如果实现成本需要控制，优先级为：

1. `random_search`
2. `pure_sa`
3. `tpe_mixed_bo`
4. `hcmbo_proposed`

`enum_de` 可放到第二批实验或附录。

## 4. 不建议的算法选择方式

不建议为了突出 HCMBO 而只选择弱基线。审稿人通常会质疑：

- Random Search 太弱，不能代表优化方法。
- Grid Search 在高维混合变量下不现实。
- 只比较 baseline、random 和内部消融，缺少外部强基线。
- 每个方法预算不同，HCMBO 用了更多仿真次数。
- HCMBO 内部包含 random search，但又单独把 random search 作为低预算基线。

正确做法是选择 2 到 3 个有代表性的强基线，并保证预算公平。这样即使 HCMBO 只是显著优于其中大部分，也比“挑弱基线全赢”更可信。

## 5. G6 主实验场景

第一阶段仍使用当前 G5 全局到达场景：

- baseline config: `codes/scenes/examples/g2_multistage_directional/run_baseline.toml`
- 控制变量：`z = (s, q)`
- 方向状态：`E, W, FREE, CLOSED`
- 通道：`top, middle, lower_middle, bottom`
- 容量时间段：`time_segments = 4`
- 候选方向数量：`direction_candidate_limit = 12`
- 默认目标权重：`lambda = (1, 1, 1, 1, 0.1)`
- 评价保真：`steps = 1600`, `time_horizon = 160.0`, `bellman_every = 5`

第二阶段可新增局部入口决策场景：

- 人群初始化在通道入口前的决策区域或等待区。
- 目的地仍按阶段设置为平台或返回出口。
- 该场景用于验证入口管控算法本身，避免短时 LF 还没覆盖控制生效阶段。

主文建议先报告全局场景，局部场景作为鲁棒性补充。

## 6. 公平预算设计

G6 必须统一预算。建议采用两档：

### Budget-S：可运行主实验

- 每个方法每个 seed 最多 `B = 400` 次优化保真仿真。
- 高保真复验 `K = 10` 个候选。
- seed 数：`5`。
- 适合当前计算资源。

### Budget-M：期刊增强实验

- 每个方法每个 seed 最多 `B = 800` 次优化保真仿真。
- 高保真复验 `K = 20` 个候选。
- seed 数：`10`。
- 用于最终论文或补充材料。

当前 G5 小预算结果中 HCMBO 约为：

- LF screen: `24`
- optimization: `404`
- HF recheck: `5`

G6 中应把优化预算明确写成 `B=400` 或 `B=404`，并让所有横向算法使用同等预算。LF screen 如果保留，只作为廉价诊断，不应让 HCMBO 获得额外高保真评价优势。

推荐主预算：

- `B = 400`
- `K = 10`
- `seeds = [11, 23, 37, 51, 73]`

## 7. 各算法实现口径

### 7.1 Baseline / Prior

用途：工程参照，不参与“优化预算公平”比较。

候选：

- no-cap all-FREE reference
- prior direction: `DEFAULT_PRIOR_DIRECTIONS`
- prior direction with high / medium / low capacity

报告时取其中默认目标下最好的一个作为 `baseline_prior_best`。

### 7.2 Random Search

统一候选空间：

- 方向从同一 `direction_candidates` 中采样。
- 容量变量 `x` 在 `[0,1]^d` 中采样。
- 使用 `control_from_x()` 映射到合法 `q`。

预算：

- `B` 次优化保真仿真。
- 选 top `K` 做高保真复验。

说明：

- Random Search 是必要下界。
- 建议同时实现 `lhs_random_search` 或 Sobol Search，避免审稿人认为纯随机过弱。

### 7.3 Pure-SA / Mixed Simulated Annealing

变量邻域：

- 离散方向：随机选择一个通道，在 `E/W/FREE/CLOSED` 中合法切换，并保证 `min_open_channels`。
- 连续容量：对当前 `x` 加高斯扰动或分段扰动，然后裁剪到 `[0,1]`。

接受准则：

- 若目标下降则接受。
- 否则按 `exp(-(f_new-f_old)/T)` 接受。
- 温度从 `T0` 按指数或线性退火到 `Tmin`。

预算：

- 初始随机点若干。
- 总评价次数严格等于 `B`。

选择理由：

- 与既有 SA-HBO/G4 方法有历史联系。
- 能说明 HCMBO 不只是比随机好，也优于传统混合扰动搜索。

### 7.4 TPE / SMAC 类混合变量 BO

推荐优先实现 TPE，因为 Optuna 实现成本低，天然支持条件变量：

- `s_top, s_middle, s_lower_middle, s_bottom` 为 categorical。
- 对每个 active gate 和 time segment 采样容量比例。
- 若方向为 `E` 只开放 plus，`W` 只开放 minus，`FREE` 开放双侧，`CLOSED` 置零。

预算：

- `B` 次 trial。
- 选 top `K` 做高保真复验。

选择理由：

- TPE/SMAC 是混合变量黑箱优化常用强基线。
- 审稿人更容易接受它作为外部算法代表。

如果不引入第三方库，可先实现简化 `RF-BO`：

- 用随机森林回归器拟合目标。
- one-hot 编码方向。
- 连续变量为容量参数。
- 用候选池 LCB/EI 选择下一点。

但从期刊说服力看，使用 Optuna TPE 或 SMAC 更好。

### 7.5 Enum-DE / Direction-wise Differential Evolution

设计：

- 使用同一批方向候选。
- 对每个方向分配预算 `B / N_direction`。
- 每个方向内部用 Differential Evolution 优化连续容量变量。
- 汇总所有方向的 top 候选，再做高保真复验。

选择理由：

- 代表成熟连续全局优化方法。
- 与 HCMBO 的区别清楚：DE 不利用分层候选、代理排序和结构化内层 BO。

注意：

- 若方向数多，Enum-DE 每个方向预算会很小，可能不公平。可以设置两种版本：
  - `enum_de_equal_direction`: 每个方向均分预算。
  - `de_mixed_encoded`: 方向 one-hot/整数编码后统一优化。

主文建议只保留更稳定的版本。

### 7.6 Proposed HCMBO

G6 中的 HCMBO 应使用当前 G5 结论后的推荐配置：

- 不使用 LF hard shortlist。
- `shortlist_size = direction_candidate_limit`。
- 保留 LF 记录用于诊断，但不让 LF 排名剪枝。
- 默认目标权重不变。
- DFO 可保留，但需要在文中说明当前预算下 DFO 不是主要收益来源。

为了避免“HCMBO 包含 random search，而 random search 又作为基线”的质疑，G6 需要明确：

- 若 HCMBO 内部保留 random exploration，则这部分计入 HCMBO 的总预算 `B`。
- 横向 random search 也使用同样总预算 `B`。
- 最终按相同 `K` 个高保真候选复验。

## 8. 统一评价指标

主指标：

- `best_hf_objective_default`: 默认权重下的最佳高保真目标值。
- `feasible_best`: 最优候选是否可行。
- `best_feasible_hf_objective`: 只在可行候选中取最优。

分项指标：

- `J1_eval`: 效率 / 旅行时间。
- `J2_eval`: 安全 / 高密度风险。
- `J5_eval`: 通道流量均衡。
- `JB_normalized`: 入口等待。
- `JR_normalized`: 控制平滑性。
- `gate_rejected`: 被容量限制拒绝的累计流量。
- `binding_time_ratio_max`: 容量约束长期绑定程度。

过程指标：

- best-so-far 曲线。
- 达到某目标阈值所需评价次数。
- 每个方法运行时间。
- 每个方法有效可行候选比例。

统计指标：

- 均值、标准差。
- 中位数、IQR。
- paired Wilcoxon signed-rank test。
- Holm-Bonferroni 多重检验校正。
- Cliff's delta 或 Vargha-Delaney A12 效应量。

## 9. 图表设计

主文建议 4 张图：

1. 横向方法 best HF objective 箱线图。
2. best-so-far convergence curve，横轴为评价次数，纵轴为当前最优目标。
3. 最优候选分项指标雷达图或柱状图。
4. 最优管控方案的方向与容量 profile 图。

附录图：

- 每个 seed 的排名表。
- feasible rate 对比。
- `J1-J2` 或 `J2-J5` Pareto 投影。
- 局部入口场景和全局场景的结果一致性。

## 10. 预期结论写法

不要写成“我们选择了弱方法，所以 HCMBO 全面最优”。建议写成：

1. HCMBO 在默认权重和相同预算下取得最低或并列最低的高保真目标值。
2. 与 Random Search 相比，HCMBO 更快找到可行低目标候选。
3. 与 Pure-SA 相比，HCMBO 的结构化方向-容量分解降低了混合变量搜索难度。
4. 与 TPE/SMAC 相比，HCMBO 利用了通道方向的结构约束和容量映射，因此在有限预算下更稳定。
5. 若 `no_jb` 或其他权重变体出现更低数值，必须说明它们优化的是不同目标，不作为默认目标下的横向主结论。

当前 G5 结果可作为 G6 动机：

- 禁用 LF hard shortlist 后，HCMBO 在当前小预算下达到 `2.614016`。
- Random Search 为 `2.944064`。
- only-q 和 only-s 均在 `3.3` 以上。
- 说明结构化联合优化比单独优化方向或容量更有效。

但正式论文结论必须基于 G6 多 seed 横向结果。

## 11. 输出目录与文件建议

建议新增：

- `codes/g6_horizontal_comparison.py`
- `codes/scenes/examples/g6_horizontal_comparison/g6.toml`
- `codes/results/g6_horizontal_comparison/`

输出文件：

- `G6_manifest.json`
- `G6_method_summary.csv`
- `G6_seed_summary.csv`
- `G6_hf_candidates.csv`
- `G6_convergence_curves.csv`
- `G6_statistical_tests.csv`
- `G6_full_report.md`

## 12. 实施顺序

第一步：实现 G6 runner 框架。

- 复用 `G5EvaluationCache`。
- 统一候选空间和 `qbar` 估计。
- 所有方法写入独立子目录。
- 所有方法最终只用 HF 排名。

第二步：实现最小横向比较。

- baseline_prior_best
- random_search
- pure_sa
- hcmbo_proposed

第三步：加入强基线。

- tpe_mixed_bo
- enum_de

第四步：多 seed 正式实验。

- Budget-S: `B=400, K=10, seeds=5`
- 输出统计检验和论文图。

第五步：增强实验。

- Budget-M: `B=800, K=20, seeds=10`
- 新增局部入口决策场景。

## 13. 最终推荐的 G6 主实验组合

主文方法：

1. `baseline_prior_best`
2. `random_search`
3. `pure_sa`
4. `tpe_mixed_bo`
5. `enum_de`
6. `hcmbo_proposed`

附录方法：

1. `only_s_high`
2. `only_q_prior`
3. `no_lf_selection`
4. `no_dfo`
5. `no_jb`

这样设计的优点是：

- 主文横向对比有外部强基线。
- 附录保留内部消融。
- HCMBO 的优势可以归因到结构化联合优化，而不是预算或基线选择偏差。
- 结果更符合高质量期刊对公平性、统计性和可复现性的要求。
