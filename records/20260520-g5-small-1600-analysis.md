# 2026-05-20 G5 small 1600 实验结果分析记录

## 1. 实验背景

本记录分析 `codes/results/g5_small_1600` 的 G5 V2 小预算 1600-step 实验结果。该实验是在 `g5_small_2h` 的基础上，将优化阶段仿真也改为长时程：

```text
optimization.steps = 1600
optimization.time_horizon = 160.0
optimization.bellman_every = 5
optimization.save_every = 100000
high_fidelity.steps = 1600
high_fidelity.time_horizon = 160.0
```

小预算搜索规模保持不变，主要参数为：

```text
direction_candidate_limit = 12
shortlist_size = 4
initial_samples = 8
bo_iterations = 12
dfo_evaluations = 5
high_fidelity_top_k = 5
random_search_evaluations = 100
workers = 3
```

因此，`main_hcmbo_full` 中 HCMBO 内层优化预算约为：

```text
shortlist_size * (initial_samples + bo_iterations + dfo_evaluations)
= 4 * (8 + 12 + 5)
= 100
```

实验完成时间为 `2026-05-20 16:50:53` 左右，所有 7 个子实验均完成，`run_stderr.log` 为空。

## 2. 主要结果

顶层 `G5_matrix_summary.csv` 给出的各实验最佳 HF 候选如下：

| 实验 | best objective | feasible | 方向配置 | evaluation rows |
|---|---:|---:|---|---:|
| `no_jb` | 2.601517 | True | top:W, middle:E, lower_middle:E, bottom:W | 234 |
| `no_lf_selection` | 2.614016 | True | top:W, middle:E, lower_middle:E, bottom:W | 434 |
| `main_hcmbo_full` | 2.944064 | False | top:W, middle:E, lower_middle:E, bottom:W | 234 |
| `no_dfo` | 2.944064 | False | top:W, middle:E, lower_middle:E, bottom:W | 214 |
| `random_search` | 2.944064 | False | top:W, middle:E, lower_middle:E, bottom:W | 106 |
| `only_q_prior` | 3.313980 | True | top:FREE, middle:E, lower_middle:W, bottom:FREE | 31 |
| `only_s_high` | 3.375114 | True | top:E, middle:W, lower_middle:W, bottom:E | 18 |

直接按 `best_objective` 看，`no_jb` 最低。但这个排序不能直接作为最终结论，因为 `no_jb` 使用了不同目标函数：它将 `lambda_jb` 置为 0，和默认目标函数不一致。

使用 `G5_weight_sensitivity.csv` 中的默认权重后处理重排后：

```text
default rank 1: no_lf_selection, objective = 2.614016
default rank 2: no_jb, objective = 2.662389
default rank 11: main_hcmbo_full/random_search shared candidate, objective = 2.944064
```

因此，在默认权重口径下，本轮最佳 HF 候选来自 `no_lf_selection`，不是 `no_jb`。

## 3. 关键异常与解释

### 3.1 main_hcmbo_full 的最佳结果不是 HCMBO 找到的

`main_hcmbo_full` 的 `G5_method_comparison.csv` 显示：

| method | evaluation_count | best_objective | best_case_id | feasible |
|---|---:|---:|---|---:|
| `random_search_mf` | 100 | 2.944064 | `g5_mf_0009_rnd_03369d3dfb` | False |
| `hcmbo_mf` | 100 | 3.251352 | `g5_mf_0123_hcmbo_bo_f160de95f9` | True |
| `high_fidelity_recheck` | 5 | 2.944064 | `g5_hf_0001_hf_03369d3dfb` | False |

这说明 `main_hcmbo_full` 的最终最佳 HF 候选来自它内部附带的 `random_search_mf`，不是 HCMBO 本身。`main_hcmbo_full` 当前更像“baseline + random search + HCMBO 的混合候选池”，而不是纯 HCMBO 结果。

因此，不能写成：

```text
main_hcmbo_full 表示 HCMBO 的性能。
```

更准确的写法是：

```text
main_hcmbo_full 表示完整混合候选池的结果，其中最终 best 可能来自 random、baseline 或 HCMBO。
```

后续报告必须拆分：

- `combined_pool_best`
- `hcmbo_only_best`
- `random_only_best`
- `baseline_best`

### 3.2 no_lf_selection 更好，说明低保真筛选失效

`main_hcmbo_full` 的 shortlist 只保留 4 个方向组合：

```text
1. top:E,    middle:W, lower_middle:W,    bottom:E
2. top:FREE, middle:E, lower_middle:W,    bottom:FREE
3. top:E,    middle:E, lower_middle:W,    bottom:W
4. top:W,    middle:E, lower_middle:W,    bottom:E
```

而后续表现最好的方向：

```text
top:W, middle:E, lower_middle:E, bottom:W
```

在 `main_hcmbo_full` 的低保真筛选中排第 10，因此被筛掉。`no_lf_selection` 不做这种硬筛选，保留全部 12 个方向候选，因此能在该方向上继续优化容量 `q`，最终得到更优候选：

```text
no_lf_selection best objective = 2.614016
feasible = True
```

这一现象表明当前 LF screening 的预测能力很弱。具体表现为，screen 阶段多数方向的目标值几乎都在 `0.91456` 附近，且 `J2_eval=0`、`J5=0`，排序主要由极小差异决定。这种 `time_horizon=4.0` 的短时程 screen 很难预测 160 秒长时程下的拥堵、排队与通道流量均衡。

因此，这不是简单的仿真 bug，更像是多保真筛选设计问题。

### 3.3 no_jb 的低 objective 不能直接与默认目标比较

`no_jb` 的 own objective 为：

```text
2.601517
```

但该值是在 `lambda_jb=0` 下计算的。用默认权重重新计算后，它的 candidate-library rerank objective 为：

```text
2.662389
```

它仍然较好，但已经不是全局第一。因此 `G5_full_report.md` 当前直接跨实验比较 `best_objective` 有口径风险。报告需要同时输出：

- `own_objective`
- `default_weight_objective`
- `feasible`
- `source_method`

### 3.4 feasible 定义不足以刻画入口排队问题

本轮结果中存在一个重要矛盾：不少候选虽然 `feasible=True`，但仍有很大的 `gate_rejected`。

例如：

```text
no_lf_selection best:
gate_rejected = 2599.64
feasible = True

no_jb best:
gate_rejected = 3239.30
feasible = True

only_q_prior best:
gate_rejected = 3926.37
feasible = True
```

当前 `feasible` 主要由 `cap_removed_relative` 判断，而不是由 `gate_rejected`、`waiting_mass_peak` 或 `binding_time_ratio` 判断。因此 `feasible=True` 只能说明密度 cap 删除量未超阈值，不代表入口无拥堵、不代表通道前没有排队。

这与前面观察到的“人堆积在通道前，不进入通道”是一致的：模型确实在容量门控处拒绝了大量尝试进入流量。

## 4. 是代码错误还是方法设计问题

### 4.1 暂未发现仿真求解器层面的明显错误

从本轮数据看，以下链路是自洽的：

- 方向配置会改变通道入流方向；
- 容量配置会限制内部 gate 的实际通过流；
- `gate_attempted`、`gate_actual`、`gate_rejected` 能反映入口排队和容量拒绝；
- HF 复验能复现 MF 候选的主要排序趋势；
- 所有子实验完成且日志无异常。

因此，目前没有证据说明 Python 仿真求解器本身算错。

### 4.2 存在实验汇总口径问题

更明确的问题在实验编排和报告层：

1. `main_hcmbo_full` 混合了 baseline、random search、HCMBO，再统一做 HF top-k，导致 best case 不一定来自 HCMBO。
2. `no_jb` 使用不同目标函数，不能直接用 own objective 与默认目标下的实验比较。
3. 顶层报告没有默认过滤 `feasible=False` 候选。
4. `no_lf_selection` 的 HCMBO 评估次数为 300，而 `main_hcmbo_full` 的 HCMBO 评估次数为 100，不能直接作为同预算公平比较。

这些属于实现层的实验口径问题，不是仿真物理错误。

### 4.3 存在方法设计问题

方法层的主要问题包括：

1. **低保真筛选不足。** `screen` 太短，无法预测长时程拥堵和安全风险。
2. **目标函数尺度失衡。** `J2_eval` 通常在 1.7 到 2.8，明显主导总目标；`J_B`、`J_R` 多为 0.05 到 0.4，影响较弱。
3. **排队与拒绝流没有被强约束。** 大量 `gate_rejected` 仍可出现在最优候选中。
4. **BO 代理较弱。** 当前 `propose_lcb_candidate()` 是基于距离加权均值和 LCB 的启发式搜索，不是严格的 GP/RF/TPE BO。对于方向变量强影响、容量维度较高的问题，它很依赖 shortlist 是否可靠。
5. **FREE 通道容量映射有进一步检查空间。** 当前 `FREE` 通道使用 `max(qbar_plus, qbar_minus)` 作为总容量基准再按比例分配，可能低估或扭曲双向通道可用容量，需要专门消融。

## 5. 排查建议

### 5.1 不重跑的 posthoc 排查

优先基于现有 `g5_small_1600` 结果做以下分析：

1. **修正版 summary：**
   - 统一用默认权重重算所有候选；
   - 输出 `own_objective` 和 `default_weight_objective`；
   - 增加 `feasible_only_rank`。

2. **来源拆分：**
   - 对每个 experiment 输出 `best_by_source`；
   - 区分 `random_search_mf`、`hcmbo_mf`、`baseline`、`high_fidelity_recheck`。

3. **LF-HF 相关性诊断：**
   - 对 12 个方向候选计算 LF rank；
   - 对每个方向取 MF/HF 最佳；
   - 统计 Spearman 相关系数、Top-k hit rate。

4. **拥堵诊断：**
   - 按 `gate_rejected`、`waiting_mass_peak`、`binding_time_ratio_max` 排序；
   - 检查低 objective 候选是否依赖大量入口拒绝流。

5. **可行性过滤：**
   - 比较 all candidates ranking 与 feasible-only ranking；
   - 明确 `feasible=False` 是否允许作为最终候选。

### 5.2 需要改代码的排查

建议增加几个报告字段和脚本：

1. 在 `G5_matrix_summary.csv` 中加入：
   - `best_source_method`
   - `own_objective`
   - `default_weight_objective`
   - `feasible_rank`
   - `hcmbo_only_best_objective`
   - `random_only_best_objective`

2. 新增 posthoc 诊断脚本：
   - `codes/g5_result_diagnostics.py`
   - 输入 result root；
   - 输出 LF-HF rank audit、source split、feasible-only summary。

3. 修改 `G5_full_report.md` 生成逻辑：
   - 不跨目标函数直接比较 `no_jb`；
   - 明确 `candidate-library reranking`；
   - 最终结论默认使用 HF + feasible-only + default weights。

## 6. 下一步实验设计

### 实验 A：低保真筛选可靠性

目的：确认 LF screening 是否真的能预测 HF 表现。

建议设置：

```text
screen.time_horizon = 4, 20, 60
screen.steps        = 60, 240, 600
direction_candidate_limit = 12
shortlist_size = 4
optimization/high_fidelity = 1600/160
```

输出指标：

- LF rank 与 HF rank 的 Spearman 相关；
- HF best 是否进入 LF top-4；
- 不同 screen horizon 下 shortlist 的稳定性。

如果 `time_horizon=4` 的 Top-k hit rate 很差，而 `20/60` 明显改善，则说明当前 main 失败主要来自 LF 过短。

### 实验 B：同预算算法比较

目的：公平比较 HCMBO 与 random search，而不是比较不同预算。

建议设置：

```text
budget = 100, 200, 400
methods:
  random_search
  hcmbo_shortlist
  hcmbo_all_directions
seeds = 5
HF top_k = 5 or 10
```

关键要求：

- HCMBO 只统计 HCMBO 自己生成的候选；
- random search 只统计 random 自己生成的候选；
- 不把 `main_hcmbo_full` 的混合候选池 best 当成 HCMBO best。

### 实验 C：约束与排队敏感性

目的：解决“目标函数低但入口仍严重排队”的问题。

建议比较：

```text
default objective
feasible-only ranking
strong cap_removed penalty
strong gate_rejected penalty
strong waiting_mass penalty
lambda_jb = 1, 5, 10
```

输出指标：

- objective；
- `gate_rejected`；
- `waiting_mass_peak`；
- `binding_time_ratio_max`；
- `cap_removed_relative`；
- 通道前密度快照。

### 实验 D：FREE 容量映射消融

目的：检查 `FREE` 通道容量映射是否扭曲了双向通道容量。

候选实现：

```text
FREE qbar_total = max(qbar_plus, qbar_minus)
FREE qbar_total = qbar_plus + qbar_minus
FREE plus/minus 独立控制
```

观察 `only_q_prior` 和 all-FREE 方向下的 `J1/J2/J5/JB/gate_rejected` 变化。

## 7. 当前结论

本轮实验不能支持“当前 HCMBO 主实验稳定优于 random search”的结论。更准确的结论是：

1. `main_hcmbo_full` 的最佳候选来自内部 random search，而不是 HCMBO。
2. 默认 LF screening 会筛掉后续表现最好的方向，导致 HCMBO 在 main 设置下表现较差。
3. `no_lf_selection` 在默认权重重排下取得当前最优 HF 结果，说明固定方向容量优化本身有潜力。
4. `no_jb` 的 own objective 最低，但不能直接跨目标函数比较；用默认权重重排后低于 `no_lf_selection`。
5. 当前可行性定义没有充分约束入口拒绝流和排队，因此最优候选仍可能存在大量 `gate_rejected`。
6. 下一步应先修正实验报告口径，再做 LF 可靠性、同预算、多 seed、queue-aware objective 实验。

论文或阶段报告中建议使用以下表述：

```text
G5 V2 小预算长时程实验表明，容量控制与方向配置联合优化可以找到优于纯方向或固定先验容量的候选；
但当前 HCMBO 主流程受低保真筛选影响较大，默认 screen 会筛掉后续高保真表现较好的方向。
因此，当前结果更支持“多保真筛选和排队约束需要改进”，还不能支持“完整 HCMBO 已稳定优于随机搜索”的结论。
```
