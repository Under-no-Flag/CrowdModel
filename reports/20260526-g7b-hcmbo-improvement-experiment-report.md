# G7-B HCMBO 改进与变体消融实验报告

## 1. 实验背景

G6 横向对比实验中，当前 HCMBO 实现虽然明显优于 baseline、random search、pure SA 等方法，但在 `seed=23` 上落后于 `tpe_mixed_bo`。进一步查看后处理诊断后可以看到，HCMBO 的高保真最优候选主要来自结构化方向内搜索，而不是内部 random search。这说明当前实现中“结构化搜索 + 随机补预算”的预算分配可能并不理想，随机部分没有稳定贡献更优候选，反而可能稀释了结构化搜索预算。

因此，本轮 G7-B 实验的目标不是重新做横向对比，而是在 HCMBO 内部做变体消融，回答以下问题：

1. 去掉 internal random search 后，结构化搜索是否更强。
2. 引入排队/拒绝流量感知的搜索评分是否能改善通道入口行为。
3. 自适应方向竞赛、随机森林代理、trust-region 和多样化高保真复验是否能进一步提升结果。

## 2. 改进思路

本轮实验保留 G5/G6 的基本问题定义，即优化混合变量 `z=(s,q)`：

- `s` 表示四条通道的方向配置。
- `q(t)` 表示各入口在分段时间上的容量控制。
- 目标函数仍综合考虑效率、通行时间、平滑性、入口等待/拒绝等项。
- 最终结论只使用高保真复验结果，不使用低/中保真排序直接作为最终最优。

在此基础上，G7-B 主要做了四类改进：

### 2.1 去掉内部随机搜索

当前 HCMBO 会先对每个方向做结构化优化，再用剩余预算做 random search。G7-B 增加 `hcmbo_structured_only`，将全部 400 次优化预算投入方向内结构化搜索，检验 random search 是否真正有价值。

### 2.2 引入 queue-aware 搜索评分

G7-B 增加 `hcmbo_queue_aware_lcb`，不只按原始 objective 排序候选，而是用 queue-aware score：

```text
score = objective
      + 0.0001 * gate_rejected
      + 0.005  * waiting_mass_peak
      + 0.5    * binding_time_ratio_max
      + cap_removed penalty
      + infeasible penalty
```

这样做的目的是避免算法只追求综合目标数值，却产生明显入口排队、入口拒绝或过强管控。

### 2.3 自适应方向竞赛

G7-B 增加 `hcmbo_adaptive_racing`。它不再平均给所有方向完整预算，而是分阶段筛方向：

```text
12 个方向各 4 次
保留 6 个方向，各加 12 次
保留 4 个方向，各加 25 次
保留 3 个方向，各加 40 次
保留 2 个方向，各加 30 次
总预算 400 次
```

该变体用于检验：如果早期淘汰差方向，把预算集中到较优方向，是否能提升搜索效率。

### 2.4 增加代理模型、局部搜索和 HF 候选选择消融

G7-B 还加入了以下变体：

- `hcmbo_rf_constrained_bo`：用 ExtraTrees/随机森林风格代理模型拟合 queue-aware score，并按预测均值与不确定性选择候选。
- `hcmbo_trust_region`：围绕当前最优容量向量做局部 trust-region 搜索。
- `hcmbo_diverse_hf_topk`：候选生成仍与 current 相同，只改变高保真 top-k 选择策略，加入 J2、gate_rejected、J5 和方向多样性。
- `hcmbo_adaptive_racing_queue_aware`：组合自适应方向竞赛与 queue-aware score。

## 3. 实验设计

实验入口为 `codes/g7_hcmbo_variant_ablation.py`，配置为 `codes/scenes/examples/g7_hcmbo_variant_ablation/g7.toml`。

本轮完整实验采用单 seed 验证：

- seed：`23`
- 方向候选数：`12`
- 优化预算：`400` 次中保真候选评估
- 高保真复验：每个变体 `top_k=10`
- 优化保真度：`steps=1600, time_horizon=160.0`
- 高保真复验：`steps=1600, time_horizon=160.0`
- 并行数：`workers=4`
- 输出目录：`codes/results/g7_b_variant_ablation`

对比变体如下：

| 变体 | 含义 |
|---|---|
| `hcmbo_current` | 当前 HCMBO，实现为结构化搜索加 internal random search |
| `hcmbo_structured_only` | 去掉 internal random，将全部预算用于结构化方向内搜索 |
| `hcmbo_adaptive_racing` | 分阶段筛选方向，把预算集中到较优方向 |
| `hcmbo_queue_aware_lcb` | 用 queue-aware score 做 LCB 搜索和 HF 候选选择 |
| `hcmbo_rf_constrained_bo` | 用 ExtraTrees/随机森林风格代理模型拟合 queue-aware score |
| `hcmbo_adaptive_racing_queue_aware` | 自适应方向竞赛与 queue-aware score 的组合 |
| `hcmbo_diverse_hf_topk` | 只改变 HF top-k 候选选择策略 |
| `hcmbo_trust_region` | 对每个方向做局部 trust-region 容量搜索 |

## 4. 实验结果

G7-B 完整实验已完成，8 个变体全部成功生成高保真复验结果。最终结果如下，按高保真目标值从小到大排序：

| 排名 | 方法 | HF 最优目标 | 可行 | J2_eval | gate_rejected | 相对 current |
|---:|---|---:|---|---:|---:|---:|
| 1 | `hcmbo_structured_only` | **2.449453** | True | **1.675949** | 2545.508 | **-19.1%** |
| 2 | `hcmbo_queue_aware_lcb` | 2.616214 | True | 1.852077 | **2412.404** | -13.6% |
| 2 | `hcmbo_rf_constrained_bo` | 2.616214 | True | 1.852077 | **2412.404** | -13.6% |
| 4 | `hcmbo_adaptive_racing` | 2.656076 | True | 1.947425 | 3204.194 | -12.3% |
| 5 | `hcmbo_adaptive_racing_queue_aware` | 2.668322 | True | 1.786214 | 2664.373 | -11.9% |
| 6 | `hcmbo_current` | 3.028568 | True | 2.231950 | 3197.608 | 0.0% |
| 6 | `hcmbo_diverse_hf_topk` | 3.028568 | True | 2.231950 | 3197.608 | 0.0% |
| 8 | `hcmbo_trust_region` | 3.031677 | True | 1.922380 | 3967.226 | +0.1% |

与 G6 横向对比中 `seed=23` 的主要方法相比：

| 方法 | HF 目标 | 可行 | J2_eval | gate_rejected |
|---|---:|---|---:|---:|
| G7-B `hcmbo_structured_only` | **2.449453** | True | **1.675949** | 2545.508 |
| G6 `tpe_mixed_bo` | 2.518036 | True | 1.694826 | 2815.208 |
| G6 `enum_de` | 2.732891 | True | 1.931630 | 2959.627 |
| G6 `pure_sa` | 2.920316 | True | 2.148054 | 3350.641 |
| G6 `random_search` | 2.944064 | False | 1.951716 | 3087.072 |
| G6 `hcmbo_proposed/current` | 3.028568 | True | 2.231950 | 3197.608 |
| G6 `baseline_prior_best` | 5.477363 | False | 2.752489 | 4055.573 |

在 `seed=23` 上，`hcmbo_structured_only` 不仅优于当前 HCMBO，也超过了 G6 中最强的 `tpe_mixed_bo`。相对于 TPE，其高保真目标值降低约 2.7%，同时 J2_eval 和 gate_rejected 也更低。

## 5. 结果分析

### 5.1 structured-only 是本轮最有效改进

`hcmbo_structured_only` 的目标值为 2.449453，相比 `hcmbo_current` 降低约 19.1%。这说明在当前预算和场景下，internal random search 没有提供有效补充，反而占用了结构化搜索可用预算。更合理的方向是强化结构化搜索本身，而不是在同一预算内加入随机搜索。

### 5.2 queue-aware 变体改善了入口行为

`hcmbo_queue_aware_lcb` 的目标值为 2.616214，虽然不是最优，但 gate_rejected 为 2412.404，是所有变体中最低的。相比 `hcmbo_current` 的 3197.608，入口拒绝量下降约 24.6%。这表明 queue-aware score 确实能把搜索引向入口行为更好的控制策略。

### 5.3 RF 代理没有带来额外收益

`hcmbo_rf_constrained_bo` 与 `hcmbo_queue_aware_lcb` 得到完全相同的最优高保真候选。这说明在当前 seed 和候选空间下，ExtraTrees 代理没有明显优于 LCB 搜索。它可能需要更多 seed 或更大候选池才能体现差异。

### 5.4 adaptive racing 有收益，但不稳定改善入口拒绝

`hcmbo_adaptive_racing` 的目标值为 2.656076，优于 current，但 gate_rejected 为 3204.194，基本没有改善入口拒绝问题。这说明方向预算集中策略能提升目标值，但如果方向排序仍只看 objective，就不会自然优化入口排队/拒绝行为。

### 5.5 diverse HF top-k 不是主要瓶颈

`hcmbo_diverse_hf_topk` 与 current 得到完全相同的最优结果，说明当前问题的瓶颈不是“高保真复验候选怎么选”，而是“中保真候选池里有没有足够好的候选”。因此后续改进应优先放在候选生成阶段。

### 5.6 trust-region 不适合作为主方法

`hcmbo_trust_region` 的目标值略差于 current，且 gate_rejected 最高。这说明当前容量控制空间可能存在多峰和非光滑特征，单纯局部扰动容易陷入不理想区域，不适合作为主搜索策略。

## 6. 结论

本轮 G7-B 实验证明，当前 HCMBO 的主要问题不是方向结构建模无效，而是预算分配和候选生成策略不够合理。最关键的发现是：

1. 去掉 internal random search 后，`hcmbo_structured_only` 明显优于当前 HCMBO。
2. queue-aware score 能显著降低 gate_rejected，适合作为后续约束感知优化的重要组成部分。
3. 只改 HF top-k 选择策略无法带来收益，说明候选生成阶段才是主要改进点。
4. trust-region 局部搜索不适合当前问题，不建议作为主线。
5. 在 `seed=23` 上，改进后的 HCMBO 已经超过 G6 中最强的 TPE 基线。

因此，下一阶段建议以 `hcmbo_structured_only` 作为新的主干版本，并吸收 queue-aware score 作为约束/排队风险控制项。更稳妥的下一步实验是开展多 seed 复验，至少重新运行 `hcmbo_structured_only`、`hcmbo_queue_aware_lcb`、`hcmbo_rf_constrained_bo`、`tpe_mixed_bo` 和 `hcmbo_current`，确认该结论是否具有稳定性。

## 7. 需要注意的限制

本轮 G7-B 只使用了 `seed=23`，因此它可以支持“发现了明确改进方向”，但还不足以支持“统计显著优于所有方法”的强结论。论文中应表述为：

> 在 `seed=23` 的完整预算验证中，structured-only HCMBO 获得了当前最优的高保真目标值，并超过了 G6 中最强的 TPE 基线；后续需要多 seed 实验验证稳定性。

## 8. 输出文件

- `codes/results/g7_b_variant_ablation/G7B_method_summary.csv`
- `codes/results/g7_b_variant_ablation/G7B_seed_summary.csv`
- `codes/results/g7_b_variant_ablation/G7B_hf_candidates.csv`
- `codes/results/g7_b_variant_ablation/G7B_pairwise_deltas_vs_current.csv`
- `codes/results/g7_b_variant_ablation/G7B_variant_ablation_report.md`
