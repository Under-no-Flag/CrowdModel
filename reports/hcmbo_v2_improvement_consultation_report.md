# HCMBO 优化方法改进咨询报告

**报告主题**：如何改进当前 HCMBO，使其在公平预算、统一高保真评价和可解释管控约束下稳定优于 TPE-Mixed BO  
**项目背景**：人群管控模型与优化算法  
**报告日期**：2026-05-25  
**建议版本定位**：HCMBO-v2：自适应方向预算 + 约束感知内层搜索 + 排队安全主指标 + 多样性高保真复核

---

## 1. 总体判断

当前不应把问题理解为 **“HCMBO 表述不够强”**，而应理解为：

> **HCMBO 的结构化优势还没有真正进入搜索决策。**

当前 HCMBO 已经证明 `z=(s,q)` 的方向—容量联合优化是有价值的，但还没有证明其搜索机制稳定优于 TPE-Mixed BO。

G6 结果显示：

| 方法 | 平均 best HF objective | 可行率 | 结论 |
|---|---:|---:|---|
| TPE-Mixed BO | 2.740328 | 80% | 当前全方法最优 |
| HCMBO proposed | 2.884847 | 80% | 非 TPE 方法中较优，但未超过 TPE |
| pure SA | 2.924760 | 60% | 略弱于 HCMBO |
| random search | 3.011173 | 40% | 明显弱于 HCMBO |
| enum-DE | 3.059686 | 40% | 明显弱于 HCMBO |
| baseline prior best | 5.477363 | 0% | 显著落后 |

因此，当前可以说：

> HCMBO 在结构化方法集合中表现较好，并验证了方向—容量联合控制的潜力；但在全方法比较中，TPE-Mixed BO 仍是更强的通用混合变量 BO 基线。

不建议在论文或汇报中直接表述为：

```text
HCMBO 全面优于 TPE-Mixed BO。
```

更稳妥的表述是：

```text
当前 HCMBO 已经优于 baseline、random search、pure SA 和 enum-DE 的趋势较清楚；
但相对于 TPE-Mixed BO 仍有差距。下一步需要通过自适应方向预算、
约束感知 acquisition、入口排队约束和高保真候选多样性选择来进一步增强。
```

---

## 2. 当前 HCMBO 输给 TPE 的主要原因

### 2.1 预算分配方式不合理

当前 HCMBO 对 12 个方向均分结构化预算。每个方向约为：

```text
8 initial + 12 BO + 5 DFO = 25 次评估
```

总结构化预算为：

```text
12 × 25 = 300
```

另外还有：

```text
内部 random search = 100
总优化预算 = 400
```

问题在于：HCMBO 的外层分层是正确的，但均匀撒预算是低效的。好方向没有得到足够深的容量曲线优化，差方向却消耗了固定预算。

TPE-Mixed BO 虽然不显式分层，但会围绕 elite 候选持续扰动，更容易把预算集中到好方向和好容量区域。

**诊断结论**：HCMBO 不是输在“有分层”，而是输在“分层之后没有自适应预算转移”。

---

### 2.2 内层 BO 代理太弱

当前 `propose_lcb_candidate()` 本质上是距离加权均值 + 简化 LCB，不是严格的 GP-BO、RF-BO、TPE 或 trust-region BO。

而当前问题具有以下困难特征：

- 方向变量 `s` 与容量变量 `q` 强耦合；
- 不同方向下连续变量的有效维度不同；
- `J2` 高密度暴露指标具有阈值型突变；
- 入口排队、容量绑定、密度 cap 删除等指标会产生非平滑响应；
- 中保真排序与高保真排序可能存在偏差。

因此，简化 LCB 很难在每个方向仅 25 次评估内找到接近 TPE 的容量曲线。

**诊断结论**：HCMBO 的结构化思想没有问题，但内层连续搜索器需要显著升级。

---

### 2.3 HCMBO 的可行映射优势被所有方法共享

理论上，HCMBO 的优势来自：

```text
方向层 s
容量层 q
硬一致性映射 q = T_s(x)
```

但在 G6 中，所有方法都使用同一个 `control_from_x()` 映射。也就是说，TPE-Mixed BO 同样可以生成满足方向—容量一致性的方案。

这会稀释 HCMBO 的独有优势。此时，HCMBO 与 TPE 的竞争主要变成搜索策略之间的竞争，而不是结构建模能力之间的竞争。

**诊断结论**：下一版 HCMBO 必须把结构信息真正用于 acquisition、预算分配、约束排序和高保真候选选择，而不能只用于变量拆分。

---

### 2.4 可行性定义过窄

当前可行性主要由：

```text
cap_removed_relative <= 0.02
```

判定。这表示密度 cap 删除量不超过阈值，但不表示入口拒绝流、等待峰值或排队持续时间一定可接受。

当前较优方案中，`gate_rejected` 仍接近 3000。若 `gate_rejected` 代表入口上游排队压力，那么当前的 `feasible=True` 只是“数值可行”，不是“管理可行”。

**诊断结论**：HCMBO 作为管控优化方法，应把入口排队和等待风险纳入约束或 acquisition，而不是只在结果表格中报告。

---

### 2.5 同一方向下 TPE 找到了更好的容量曲线

最关键的诊断现象是：HCMBO 最好 seed 与 TPE 最好 seed 使用了同一方向配置：

```text
top:W, middle:E, lower_middle:E, bottom:W
```

但 TPE 的 objective 更低，且 J2、JR、gate_rejected 等指标更优。

这说明差距不只是方向选择，而是容量曲线优化。

**诊断结论**：下一步重点应放在 `q(t)` 参数化、内层优化器和 J2/排队风险 acquisition 上。

---

## 3. HCMBO-v2 的核心改进方案

建议将下一版方法定义为：

```text
HCMBO-v2 =
    adaptive direction racing
  + constrained / queue-aware inner acquisition
  + stronger per-direction surrogate
  + feasible/J2/queue/diversity mixed HF top-k
  + source-preserving audit
```

---

## 4. 改法一：自适应方向预算 racing

### 4.1 问题

当前 HCMBO 对 12 个方向均匀分配预算，导致好方向深度不足、差方向浪费预算。

### 4.2 建议方案

将方向层从“均匀分配”改成“自适应 direction racing”。

一个可直接落地的 B=400 分配方案如下：

```text
阶段 A：全方向探测
12 个方向 × 4 个容量样本 = 48

阶段 B1：保留 top 6 方向
6 个方向 × 12 个增量样本 = 72
累计 120

阶段 B2：保留 top 4 方向
4 个方向 × 25 个增量样本 = 100
累计 220

阶段 C：保留 top 3 方向
3 个方向 × 40 个精修样本 = 120
累计 340

阶段 D：top 2 局部精修
2 个方向 × 30 个样本 = 60
累计 400
```

### 4.3 方向评分函数

方向筛选不要只看当前 best objective。建议使用综合方向评分：

```text
score_s =
    best_feasible_objective_s
  + alpha * infeasible_rate_s
  + beta  * normalized_gate_rejected_s
  + gamma * normalized_waiting_peak_s
  + delta * binding_time_ratio_max_s
  - kappa * uncertainty_s
```

其中：

- `best_feasible_objective_s`：该方向下当前最好可行目标；
- `infeasible_rate_s`：不可行候选比例；
- `normalized_gate_rejected_s`：入口拒绝流归一化指标；
- `normalized_waiting_peak_s`：等待区峰值压力；
- `binding_time_ratio_max_s`：容量长时间绑定比例；
- `uncertainty_s`：方向层不确定性，用于保留探索。

低分方向继续获得预算。

### 4.4 推荐实验名

```text
hcmbo_adaptive_racing
```

这是第一优先级改进。

---

## 5. 改法二：约束感知、排队感知 acquisition

### 5.1 当前问题

当前 acquisition 主要看目标值或简化 LCB，没有显式建模：

- 可行性概率；
- 入口排队；
- 等待峰值；
- 容量长时间绑定风险；
- 高密度暴露 J2 风险。

这会导致优化器找到“目标值较低但入口排队大”的方案。

### 5.2 建议约束定义

建议把约束写成：

```text
g_cap(z)  = cap_removed_relative - tau_cap
g_rej(z)  = gate_rejected_normalized - tau_rej
g_wait(z) = waiting_mass_peak_normalized - tau_wait
g_bind(z) = binding_time_ratio_max - tau_bind
```

建议默认阈值：

```text
tau_cap  = 0.02
tau_rej  = 根据管理可接受入口排队设定
tau_wait = 根据等待区容量设定
tau_bind = 0.7 或 0.8
```

### 5.3 约束 LCB acquisition

可使用如下候选评分：

```text
acq(x | s) =
    mu_J(x)
  - kappa * sigma_J(x)
  + lambda_cap  * E[max(0, g_cap(x))]
  + lambda_rej  * E[max(0, g_rej(x))]
  + lambda_wait * E[max(0, g_wait(x))]
  + lambda_bind * E[max(0, g_bind(x))]
```

或者采用更严格的可行性优先排序：

```text
先最大化 P(feasible and queue_safe)
再在可行区域内最小化 expected objective
```

### 5.4 推荐实验名

```text
hcmbo_queue_aware_lcb
hcmbo_constrained_ei
```

优先实现 `hcmbo_queue_aware_lcb`，因为它和当前 LCB 框架兼容，工程改动较小。

---

## 6. 改法三：升级内层代理模型

当前内层代理过弱。建议按实现成本从低到高分三层推进。

---

### 6.1 第一层：Extra Trees / Random Forest surrogate

对每个方向单独训练 surrogate，输入容量参数 `x`，输出多任务指标：

```text
J
J2
J5
JB
JR
gate_rejected
waiting_mass_peak
binding_time_ratio_max
feasible
```

Extra Trees / Random Forest 的优点：

- 适合小样本；
- 对非平滑目标较稳健；
- 不要求目标函数可微或平滑；
- 可以通过 ensemble 方差近似不确定性；
- 可以同时预测目标和约束。

推荐实验名：

```text
hcmbo_rf_constrained_bo
```

---

### 6.2 第二层：per-direction trust-region BO

每个方向维护一个局部 trust region：

```text
若连续改善：扩大 trust region
若连续失败：缩小 trust region
若长期无改善：重启到新的容量中心
```

这比全局代理更适合当前问题，因为同一方向下容量空间仍然可能高维，但好解附近通常存在局部结构。

推荐实验名：

```text
hcmbo_trust_region
```

---

### 6.3 第三层：per-direction TPE-assisted HCMBO

最务实的版本是把 TPE 作为内层固定方向优化器：

```text
外层：方向 racing + 约束管理 + 高保真多样性选择
内层：固定方向下 TPE / RF-BO / trust-region BO
```

这时 HCMBO 的贡献不是“发明一个全新采样器”，而是：

```text
把人群通道管控问题分解成方向层、容量层、约束层和高保真复核层，
并在每层使用与问题结构匹配的优化策略。
```

推荐实验名：

```text
hcmbo_tpe_assisted
```

最可能快速超过当前 TPE 的组合是：

```text
hcmbo_adaptive_racing
+ per-direction TPE or ExtraTrees
+ queue-aware acquisition
```

---

## 7. 改法四：重点压低 J2，而不是平均优化所有项

当前结果显示，TPE 在 J2、J5、JB、JR 和 `gate_rejected` 平均上均优于 HCMBO。其中 J2 是总目标差异的主要来源之一。

因此，下一版 HCMBO 不应只优化综合 `J`，而应显式降低 J2。

### 7.1 安全优先排序

建议使用：

```text
若候选 feasible:
    primary   = J2_quantile_75
    secondary = objective
    tertiary  = gate_rejected
else:
    primary   = violation_sum
```

### 7.2 双阶段目标

也可以使用双阶段优化：

```text
阶段 1：找到 J2 低于当前 HCMBO best 或低于 TPE 均值的候选区域
阶段 2：在该区域内优化 J、J5、JR 和 gate_rejected
```

如果继续让 HCMBO 均匀追逐综合 objective，它可能会为了 J5 或容量平滑牺牲 J2；但当前权重下，TPE 正是靠降低 J2 获胜。

---

## 8. 改法五：重新设计容量参数化

### 8.1 当前问题

当前容量为 4 个时间段分段常数。对 FREE 通道而言，plus/minus 两个方向都需要容量参数，导致维度上升。

此外，当前若使用：

```text
qbar_total = max(qbar_plus, qbar_minus)
```

可能低估双向通道总容量，或造成物理解释不一致。

### 8.2 推荐参数化

建议改成：

```text
Q_c(t)      = 通道总入口服务能力
r_c(t)      = plus 方向分配比例
q_c^+(t)    = Q_c(t) * r_c(t)
q_c^-(t)    = Q_c(t) * (1 - r_c(t))
```

用平滑基函数参数化：

```text
Q_c(t) = q_min + (q_max - q_min) * sigmoid(B(t) theta_Q)
r_c(t) = sigmoid(B(t) theta_r)
```

其中 `B(t)` 可用 3 个平滑基函数，而不是 4 段完全自由常数。

### 8.3 优点

这种参数化有五个优点：

1. 保证容量非负；
2. 保证 plus/minus 共享同一物理通道容量；
3. 降低有效维度；
4. 避免容量曲线剧烈跳变；
5. 更容易解释为“总限流强度 + 双向分配比例”。

对单向通道，只保留 `Q_c(t)`；对关闭通道，固定 `Q_c(t)=0`；对 FREE 通道，同时优化 `Q_c(t)` 和 `r_c(t)`。

推荐实验名：

```text
hcmbo_smooth_capacity_basis
hcmbo_total_capacity_split_ratio
```

---

## 9. 改法六：高保真 top-k 候选多样性选择

### 9.1 当前问题

如果高保真复核 top-10 只按优化阶段 objective 选择，则在中保真排序误差下可能漏掉：

- 低 J2 候选；
- 低排队候选；
- 方向多样性好的候选；
- 容量曲线多样性好的候选。

### 9.2 推荐 top-k 组成

若 `HF top_k = 10`，建议固定为：

```text
3 个：优化阶段 objective 最低
2 个：best feasible objective 最低
2 个：J2 最低
1 个：gate_rejected 最低
1 个：方向配置多样性补齐
1 个：容量曲线多样性补齐
```

若 `top_k = 20`，可按比例扩展。

推荐实验名：

```text
hcmbo_diverse_hf_topk
```

这一步工程改动小，但可能显著降低“好候选没进高保真复核”的风险。

---

## 10. 建议重新定义主指标和可行性

当前主指标如果仍是：

```text
best_hf_objective_default
```

HCMBO 很难体现“可解释约束管控”的优势。

建议把主指标改为：

```text
best_feasible_hf_objective_under_queue_constraints
```

新的可行性定义至少包括：

```text
cap_removed_relative <= threshold_cap
gate_rejected <= threshold_rejected
waiting_mass_peak <= threshold_waiting
binding_time_ratio_max <= threshold_binding
```

推荐两层结果报告。

### 10.1 算法优化主表

```text
best feasible HF objective under queue constraints
feasible rate
paired win count
gate_rejected
waiting_mass_peak
J2
```

### 10.2 补充表

```text
default objective
J1 / J2 / J5 / JB / JR
convergence AUC
top-k diversity
```

这样既能体现管控可执行性，也不会被质疑为“只改指标让 HCMBO 赢”。

---

## 11. 必须先做的诊断：HCMBO best 来源归因

当前 `hcmbo_proposed` 实际是：

```text
300 次方向分层结构化搜索 + 100 次内部随机搜索
```

而高保真复核后的 best `source` 被写成：

```text
high_fidelity_recheck
```

没有保留原始生成来源。

因此，不能直接判断 HCMBO 的 best 来自：

- structured BO；
- internal random search；
- DFO refinement；
- high-fidelity recheck pool。

建议首先写一个后处理脚本：

```text
g6_posthoc_source_audit.py
```

输出：

```text
HCMBO structured-only best
HCMBO internal-random-only best
HCMBO combined-pool best
HF top-10 original source
各 source 的 J2 / gate_rejected / feasible rate
```

如果发现 HCMBO best 多数来自 internal random，那么当前方法归因要重写；如果多数来自 structured BO，则说明结构化 BO 方向值得继续强化。

---

## 12. 下一轮实验设计

### 12.1 G7-A：不重跑的后处理诊断

目的：确认 HCMBO 输在哪里。

输出：

```text
1. HCMBO top-10 原始来源
2. structured-only vs internal-random-only vs combined-pool
3. HCMBO vs TPE best feasible 对比
4. 同方向下容量曲线差异
5. J2-J5-gate_rejected Pareto 散点
6. top-k 是否方向过度集中
```

这是最高优先级，成本最低。

---

### 12.2 G7-B：同预算 HCMBO 变体消融

预算设定：

```text
B = 400
HF top_k = 10
seeds = [11, 23, 37, 51, 73]
```

方法矩阵：

```text
hcmbo_current
hcmbo_structured_only
hcmbo_adaptive_racing
hcmbo_queue_aware_lcb
hcmbo_trust_region
hcmbo_rf_constrained_bo
hcmbo_diverse_hf_topk
hcmbo_adaptive_racing + queue_aware_lcb
```

成功标准：

```text
mean best feasible HF objective < 当前 tpe_mixed_bo
feasible rate >= 80%
gate_rejected 不高于当前 HCMBO
J2 至少不高于 TPE
```

---

### 12.3 G7-C：正式横向比较

方法：

```text
baseline_prior_best
random_search
pure_sa
enum_de
tpe_mixed_bo_current
official_optuna_tpe
SMAC 或 sklearn forest BO
hcmbo_current
hcmbo_improved
```

预算两档：

```text
Budget-S: B=400, HF top_k=10, seeds=5
Budget-M: B=800, HF top_k=20, seeds=10
```

统计检验：

```text
paired sign test
Wilcoxon signed-rank test
Vargha-Delaney A12
Cliff's delta
Holm-Bonferroni correction
```

声明 HCMBO improved 优于 TPE 的最低标准建议为：

```text
10 个 seed 至少赢 7 个
mean best feasible objective 更低
feasible rate 不低于 TPE
gate_rejected 和 waiting_mass_peak 不高于 TPE
效应量至少达到中等
```

---

## 13. 推荐实现优先级

### 第 1 优先级：结果归因与指标修正

```text
1. g6_posthoc_source_audit.py
2. 增加 best_feasible_hf_objective
3. 增加 best_feasible_gate_rejected
4. 增加 source-preserving HF recheck
5. 输出 J2/J5/gate_rejected Pareto 图
```

理由：先确认 HCMBO 的结构化部分是否真的有效。

---

### 第 2 优先级：HCMBO adaptive racing

实现：

```text
hcmbo_adaptive_racing
```

理由：这是最可能提升性能的结构性改法，也是 HCMBO 区别于 TPE 的关键。

---

### 第 3 优先级：queue-aware constrained acquisition

实现：

```text
hcmbo_queue_aware_lcb
```

理由：当前可行性没有管住入口排队，HCMBO 应该在“管控可执行性”上建立优势。

---

### 第 4 优先级：内层 surrogate 升级

先实现：

```text
Extra Trees surrogate + constrained LCB
```

再实现：

```text
per-direction trust-region BO
```

最后尝试：

```text
per-direction TPE-assisted HCMBO
```

理由：不要一开始就把 HCMBO 变成“套壳 TPE”；先证明结构化 racing 和约束 acquisition 的贡献。

---

### 第 5 优先级：容量参数化与 top-k 多样性

实现：

```text
smooth_capacity_basis
q_total + split_ratio
diverse_hf_topk
```

理由：这会提高工程解释性和高保真鲁棒性。

---

## 14. 论文表述建议

当前阶段不建议写：

```text
HCMBO 全面优于 TPE-Mixed BO。
```

建议写：

```text
HCMBO 在结构化方法集合中表现最好，并验证了方向—容量联合控制的有效性；
但在当前全方法比较中，TPE-Mixed BO 仍是更强的通用混合变量 BO 基线。
进一步提升 HCMBO 的关键在于自适应方向预算、约束感知 acquisition、
入口排队约束和高保真候选多样性选择。
```

如果 G7-B 或 G7-C 后 HCMBO-v2 成功超过 TPE，可以改写为：

```text
通过引入自适应方向 racing、排队感知约束 acquisition 和多样性高保真复核，
HCMBO-v2 在相同预算和统一高保真评价下，相比 TPE-Mixed BO 获得更低的
best feasible objective，并在入口排队风险和高密度暴露指标上表现更稳健。
```

---

## 15. 最终建议

最值得立即推进的版本是：

```text
HCMBO-v2 =
    adaptive direction racing
  + per-direction ExtraTrees / trust-region BO
  + queue-aware constrained acquisition
  + feasible/J2/queue/diversity mixed HF top-k
  + source-preserving result audit
```

这版方法比当前 HCMBO 更有可能在公平预算下超过 TPE，同时也更符合“人群管控优化”的论文贡献：

> 不是单纯找一个数值目标最小的黑箱方案，而是在方向、容量、排队、安全和高保真复核之间形成可解释、可执行、可审计的优化框架。

---

## 16. 可执行任务清单

建议下一步按如下顺序落地：

```text
[ ] 增加 high-fidelity recheck 的 original_source 字段
[ ] 写 g6_posthoc_source_audit.py
[ ] 输出 structured-only / random-only / combined-pool best
[ ] 增加 best_feasible_hf_objective_under_queue_constraints
[ ] 增加 gate_rejected / waiting_peak / binding_ratio 约束
[ ] 实现 hcmbo_adaptive_racing
[ ] 实现 hcmbo_queue_aware_lcb
[ ] 实现 ExtraTrees surrogate
[ ] 实现 diverse_hf_topk
[ ] 运行 G7-B HCMBO 变体消融
[ ] 运行 G7-C 正式横向比较
[ ] 用配对统计检验证明 HCMBO-v2 是否稳定优于 TPE
```

