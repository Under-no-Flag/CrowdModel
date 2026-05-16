# model v2 可实施性评估与下一步计划

## 1. 结论

`methodology/model v2.md` 的方向是可实施的，而且比继续优化 `eta` 更接近真实管理动作：它把连续控制变量从“通道几何引导强度”改为“内部通道入口允许通行速率” `q_c^+(t), q_c^-(t)`，本质上是在现有 Bellman 路径选择和显式守恒格式之间加入内部瓶颈通量约束。

但它不适合直接替换当前论文和 G4 代码。原因是：当前论文、实验和 `codes/crowd_bellman/g4_sahbo.py` 已经围绕 `z=(s, eta)`、固定 `p_hat`、SA-HBO 精简实现形成闭环；`model v2` 会把下层仿真器、目标函数、优化变量、G4 实验和论文 4/5 章一起牵动。因此建议按“先验证、再并行实现、最后决定是否替换论文主线”的路线推进。

推荐定位：

- 短期：把 `model v2` 作为下一版容量管控模型 `z=(s,q)` 的候选方案。
- 中期：新增独立代码与实验根目录，避免覆盖当前 `z=(s,eta)` 的结果。
- 论文层面：只有在 `eta` 敏感性确认为弱、`q` 限流机制验证通过、G2-v2 容量响应实验能形成可解释 trade-off、且 G4-v2 实验显著优于现有方案后，才正式把论文主线从 `eta` 改到 `q`。

### 2026-05-16 新版补充后的更新判断

新版 `methodology/model v2.md` 已经补齐了原计划中若干理论缺口：符号表、`q_c^\pm/A_c^\pm/\widehat A_c^\pm/R_c` 的区分、入口穿越方向 `e_c^\pm`、等待区指标 `J_B`、可选平滑性指标 `J_R`，以及“内部瓶颈约束不是外边界入流条件”的说明。因此，本计划不需要推翻，只需要把后续执行口径从“先补理论定义”调整为“按新版理论实现和验证”。

## 2. 可实施性评估

### 2.1 理论可行性

理论上成立。`model v2` 保留现有多阶段多路线 Bellman-HJB 路径选择层和守恒方程，只在内部通道入口截面 `Sigma_c^+ / Sigma_c^-` 上限制实际通量：

```text
actual_flow = min(attempted_flow, allowed_rate)
```

这和有限体积格式中的内部瓶颈界面约束一致，且不会破坏质量守恒：被限制的质量不跨过入口面，而是留在上游单元中形成排队。

需要补强的理论点：

- 新版已经把控制速率 $q_c^\pm(t)$、尝试流量 $A_c^\pm(t)$、实际通过量 $\widehat A_c^\pm(t)$ 和通道累计通过量 $R_c$ 区分开，论文可以沿用 $q$ 作为容量控制变量，不必强制改名为 $r$。
- 仍需在代码实现时明确离散单位：当前代码面通量 $fx/fy$ 是“密度 * 速度”，若 $q_c^\pm$ 的单位是“人/时间”，离散聚合时应使用 `sum(face_flux) * dx` 得到入口尝试通过率；如果内部函数把面长已经并入通量，则需要在函数名和注释里写清楚。
- 新版已修正旧版章节跳号、符号混用和残留文字等问题，后续文档工作重点转为和代码配置字段、summary 输出字段一一对齐。
- $\eta$ 不应直接写成“对目标函数不起作用”作为论文结论，需要先做敏感性验证。更稳妥的表述是：“本版本固定几何引导强度 $\eta_0$，将优化重点转向更具管理含义的入口通行能力控制。”

### 2.2 代码可行性

代码上可实施，但不是只改 G4 优化器。关键改动在下层仿真器。

当前已有支点：

- `codes/crowd_bellman/core.py` 已有 `compute_face_fluxes()` 和 `update_density()`，通量计算已经显式存在。
- `codes/crowd_bellman/runner.py` 已在每步为各群体计算 `vx/vy`，再调用 `update_density()` 推进密度。
- `codes/crowd_bellman/metrics.py` 已有通道累计流量统计，可扩展为入口尝试流量、允许流量、被限流量和等待区质量。
- `codes/crowd_bellman/g4_sahbo.py` 已有配置生成式 G4 评估器，可复用其“生成 routes/run 配置 -> 仿真 -> 收集 summary”的结构。

主要缺口：

- 当前 `update_density()` 是逐群体独立更新，但 `q_c^\pm` 约束要求先跨所有群体汇总尝试流量，再把同一个缩放系数应用回每个群体，所以必须把密度更新改成“两阶段通量流程”：
  1. 为所有群体计算自由面通量 `fx/free`, `fy/free`。
  2. 对内部入口面集合聚合尝试流量，计算 `lambda_c^\pm`。
  3. 将限制后的通量用于所有群体的守恒更新。
- 当前 `scene.toml` 只定义 `channels` 区域和 `probe_x`，还没有内部入口截面集合 `F_c^+ / F_c^-`、入口法向、等待区 `W_c^\pm`。
- 当前目标函数只有 `J1/J2/J5`，需要增加 `J_B` 或至少增加排队诊断量，否则优化器可能通过过度限流降低某些流量指标但造成入口外严重积压。
- 新版还新增了可选平滑性指标 `J_R`，代码层面应先把它作为目标函数可选项实现，默认权重为 0，避免影响旧实验。
- 当前 `enforce_total_density_cap()` 会在总密度超过 `rho_max` 时缩放密度。引入入口限流后，上游排队更容易触发 cap，必须记录 cap 削减量或重新处理拥堵上限，否则会出现“限流导致质量被数值裁掉”的解释风险。

### 2.3 实验可行性

实验上可行，但需要先做小实验验证机制，再进入 G4-v2 优化。

最低验证链路：

- V0：`eta` 敏感性验证。固定方向配置 `s`，扫描 `eta in {1,4,8,12}`，比较 `J1/J2/J5` 和通道流量占比。如果变化小于预设阈值，才有依据固定 `eta_0`。
- V1：单通道内部瓶颈验证。只对一条东西向通道设置 `q^+`，验证 `q=inf` 退化到 baseline、`q=0` 阻断入口、有限 `q` 会在上游等待区形成高密度。
- V2：多群体比例限流验证。多个阶段/路线群体同时尝试进入同一入口时，确认限流按比例缩放且总通过率不超过 `q`。
- V3：`s` 与 `q` 耦合验证。`s=E` 时 `q^-` 自动为 0，`s=W` 时 `q^+` 自动为 0，`s=FREE` 时满足 `q^+ + q^- <= qbar`，`s=CLOSED` 时两侧均为 0。
- V4：`G2-v2` 容量控制响应与可优化性验证。固定优化器，只做人工设计的 `q` 策略矩阵，证明 `q` 会产生可区分、可解释的系统响应，并证明 `J1/J2/J5/J_B` 之间存在非一致最优。
- V5：G4-v2 搜索验证。用 `z=(s,q)` 进行小预算搜索，并与当前 `z=(s,eta)`、方向-only、固定容量 baseline 对照。
- V6：平滑性惩罚验证。固定同一组 `s,q` 候选，比较 `lambda_R=0` 与 `lambda_R>0` 时的容量时间序列跳变、目标值和排队指标，确认 `J_R` 只约束管控动作可实施性，不掩盖物理指标解释。

## 3. 下一步计划

## 3.1 文档与理论

- [x] 新版 `methodology/model v2.md` 已补充符号表、连续形式、离散形式和论文表述建议；后续若不再大改，可直接作为论文改写的底稿。
- [x] 新版已用 $q_c^\pm(t)$ 表示控制速率上限，用 $A_c^\pm(t)$ 表示尝试流量，用 $\widehat A_c^\pm(t)$ 表示实际通过量，用 $R_c$ 表示通道累计通过量；后续计划沿用这套命名。
- [x] 统一控制变量：
  - 旧版：`z=(s,eta)`；
  - v2：`z=(s,q)`，其中 `eta=eta_0` 固定。
- [x] 补充离散单位说明：
  - `face_flux = rho * velocity`；
  - `attempted_rate = sum(max(face_flux,0)) * dx`，除非代码中的面通量对象已经乘过面长；
  - `lambda = min(attempted_rate, q) / (attempted_rate + eps)`。
- [x] 新版已增加 `s` 与 `q` 的可行域定义：
  - `E`: `q^- = 0`；
  - `W`: `q^+ = 0`；
  - `FREE`: `q^+ + q^- <= qbar`；
  - `CLOSED`: `q^+ = q^- = 0`。
- [x] 新版已将排队指标写清楚：
  - `B_c^\pm(t)` 是等待区质量；
  - `J_B` 是排队暴露；
  - `J_B` 建议进入优化目标，避免过度限流。
- [x] 将可选平滑性指标 `J_R` 纳入目标函数说明和代码配置说明，默认 `lambda_R=0`。

验收标准：理论文档能独立解释 $\Sigma_c^\pm$、离散入口面、限流单位、质量守恒、$s-q$ 耦合和 $J_B$。

更新后验收标准：3.1 文档与理论已完成；后续重点转入 3.2 验证，尤其是 `eta` 敏感性、单通道限流 smoke、质量守恒和 `J_R` 平滑性验证。

## 3.2 验证

- [x] 做 `eta` 敏感性验证，输出 `eta_sensitivity.csv` 和一张对比图。
- [x] 做单通道限流 smoke case，至少包含 `q=inf`、`q=high`、`q=medium`、`q=0` 四组。
- [x] 做质量守恒验证：
  - 初始质量 + 入流 - 离场 - 当前质量 - cap削减量 应接近 0；
  - 被拒绝通量不应消失，应表现为上游等待区质量上升。
- [x] 做方向耦合验证，确认 `s` 变化会自动约束对应方向的入口容量。
- [x] 做多群体验证，确认共享入口容量按同一 `lambda` 作用于各群体。
- [x] 做 `J_R` 验证，确认平滑性惩罚不会改变质量守恒与入口限流机制，只影响相邻时间段容量跳变。

验收标准：每个验证都有可复现命令、结果目录、summary 和失败判定阈值。

执行结果（2026-05-16）：

- 验证脚本：`codes/validate_model_v2_3_2.py`。
- 复现命令：`D:\Anaconda\envs\interpreter\python.exe codes\validate_model_v2_3_2.py`。
- 结果目录：`codes/results/model_v2_3_2_validation`。
- 总结文件：`codes/results/model_v2_3_2_validation/validation_summary.json`。
- `eta` 敏感性：已生成 `eta_sensitivity.csv` 和 `eta_sensitivity.png`；固定方向 `FREE,E,W,FREE` 下，`eta in {1,4,8,12}` 的目标函数相对变化为 `0.044014`，低于当前 `0.05` 弱敏感阈值。
- 单通道限流 smoke：`q=inf/high/medium/0` 四组通过，`q=0` 阻断入口，`q=medium` 产生绑定和拒绝通量，`q=inf` 与 `q=high` 不绑定。
- 质量守恒：最大残差 `1.4210854715202004e-13`，低于 `1e-8` 阈值；被拒绝通量表现为上游等待区质量上升。
- 方向耦合：`E/W/FREE/CLOSED` 投影规则通过，`FREE` 满足 `q^+ + q^- <= qbar`。
- 多群体共享容量：总尝试率 `15.0`、实际通过率 `6.0`，共享 `lambda=0.4`，两个群体按同一比例缩放。
- `J_R` 验证：粗糙容量序列 `J_R=1.0`，平滑序列 `J_R=0.0`；改变 `lambda_jr` 只改变目标函数附加项，不破坏质量守恒和限流约束。
- 口径说明：`eta` 敏感性使用当前 G4/runner 小预算仿真；入口限流、方向耦合、多群体和 `J_R` 是独立有限体积 smoke 验证，生产级接入仍属于 3.5 代码阶段。

## 3.3 G2-v2 容量控制响应与可优化性验证（优化前）

### 必要性评估

需要补充这一组实验。3.2 已经证明 `q` 限流机制在数值上可运行、守恒且满足方向耦合，但它还没有回答论文实验层面的关键问题：`q` 作为新控制变量是否真的能在完整行为模型下诱发可区分、可解释、可优化的系统响应。原 G2 对 `s` 做过这一层论证，`q` 也必须做同样的优化前验证，否则直接进入 G4-v2 会缺少“为什么这个连续控制变量值得优化”的实验依据。

该实验应放在 G4-v2 优化之前，执行位置是 3.5 阶段 A-C 完成之后、3.5 阶段 D 优化器实现之前。它不搜索最优解，而是用人工设计的容量策略矩阵建立响应面和候选基线。

### 论文定位

论文中可作为实验章节的 `G2-v2：入口容量控制响应与可优化性验证`，与旧 G2 的逻辑一致：

1. 旧 G2 证明离散通道配置 `s` 会引起可解释响应；
2. 新 G2-v2 证明连续入口容量 `q` 会引起可解释响应；
3. 两者共同支撑后续 G4-v2 中 `z=(s,q)` 的优化变量选择。

### 目标

G2-v2 的目标不是回答“哪组 `q` 最好”，而是证明在固定 `eta_0` 和完整行为模型下：

1. 不同入口容量策略 `q_c^\pm(t)` 会引起可区分、可解释的系统响应；
2. 效率 `J1`、安全 `J2`、均衡 `J5`、排队暴露 `J_B` 之间存在非一致最优，因此不能靠单一经验规则选容量；
3. 容量响应具有足够稳定性，可形成 G4-v2 优化器的候选解空间、baseline 和参数范围。

### 研究问题

1. 当固定方向配置 `s`、只改变入口容量强度时，系统是否出现可解释的通过量下降、等待区质量上升、热点迁移和风险变化？
2. 在相同总容量预算下，不同通道之间的容量分配是否会改变通道负载均衡、局部拥堵位置和总滞留时间？
3. `q` 的作用是否存在阈值区间：过高时接近 no-cap，过低时造成排队和 `J_B` 上升，中间区间才产生有效调控？
4. 分段容量调度是否能改变排队峰值与风险时段，但同时引入管控动作跳变，从而需要 `J_R` 作为可选可实施性惩罚？
5. 这些响应是否在轻微需求扰动或重复运行下保持排序稳定，足以支撑优化器评估？

### 实验矩阵

建议使用当前 G4-v2 目标场景和四通道集合 `top/middle/lower_middle/bottom`；如果论文图表需要延续三通道 G2 表述，可在可视化中选择代表性三通道子集，但代码和结果字段优先与四通道 G4-v2 对齐。

| 组别 | 名称 | 固定条件 | 自变量 | 目的 |
|------|------|------|------|------|
| Q0 | no-cap 退化基线 | 固定 `s=s_0`、`eta=eta_0` | `q=inf` | 给出无入口容量约束下的自然尝试流量、通道负载和热点基线 |
| Q1 | 容量强度扫描 | 固定 `s=s_0`、所有开放方向同一容量比例 | `q in {high, medium, low}` | 验证 `q` 强度是否形成单调或阈值型响应 |
| Q2 | 单入口瓶颈 | 固定总容量，其余入口 high | 逐一降低某一关键入口容量 | 识别哪个入口对 `J1/J2/J5/J_B` 更敏感 |
| Q3 | 同总预算空间分配 | 固定总容量预算 `sum q` | uniform、middle-priority、edge-priority、demand-proportional | 验证容量分配而非容量总量是否改变系统响应 |
| Q4 | 方向耦合容量 | 固定代表性 `s` 组合 | `E/W/FREE/CLOSED` 下的可行 `q^+,q^-` 投影 | 验证 `s-q` 耦合在完整仿真中的策略含义 |
| Q5 | 分段调度 | 固定 `s=s_0` 和总容量预算 | `L=1` constant、`L=2` front-loaded、`L=2` return-priority、`L=4` smooth | 验证时变容量对排队峰值、通过率和 `J_R` 的影响 |
| Q6 | 稳定性复核 | 选取 Q1-Q5 中 3-5 个代表方案 | 轻微需求扰动或不同随机种子/入流强度 | 判断排序和响应差异是否足够稳定 |

容量水平不建议直接写死为绝对值。先用 Q0 的 no-cap 结果估计各入口自然尝试流量参考值 `A_ref`，再定义：

- `high = 0.9 * A_ref` 或接近不绑定的容量；
- `medium = 0.6 * A_ref`，应产生部分绑定；
- `low = 0.3 * A_ref`，应产生明显排队但不能导致不可解释的数值裁剪；
- `zero` 只作为机制或封控边界，不作为主要管理方案。

### 指标

主指标：

- `J1`：总系统滞留或总旅行时间近似；
- `J2`：高密度暴露；
- `J5`：通道负载方差；
- `J_B`：等待区排队暴露。

容量诊断指标：

- `gate_attempted_rate/cumulative`；
- `gate_allowed_rate/cumulative`；
- `gate_actual_rate/cumulative`；
- `gate_rejected_rate/cumulative`；
- `binding_time_ratio`；
- `waiting_mass_peak`；
- `cap_removed_mass`。

解释性指标：

- 通道流量占比；
- 高密度热点位置与峰值时刻；
- 等待区质量时间序列；
- `J_R`，仅在 Q5 中作为可实施性解释项，不作为主物理收益指标。

### 主要输出

- `g2_capacity_response_summary.csv`：每个 `q` 策略的 `J1/J2/J5/J_B/J_R` 和 gate 诊断量；
- `g2_capacity_response_pareto.png`：`J1/J2/J5/J_B` 的二维或三维权衡图；
- `g2_capacity_levels.png`：容量强度扫描下的响应曲线；
- `g2_capacity_allocation_loads.png`：同总预算不同空间分配下的通道负载图；
- `g2_waiting_mass_timeseries.png`：代表方案的等待区质量时间序列；
- `g2_capacity_hotspot_migration.png`：关键方案的高密度热点迁移图；
- `g2_capacity_tradeoff_table.md`：论文可直接引用的候选策略表。

### 判定方式

G2-v2 不要求找到最优容量，而是判断是否满足优化前提：

- 若 `q=inf/high/medium/low` 形成可解释的通过量、等待区质量和风险响应差异，则说明 `q` 不是无效变量；
- 若同总容量预算下，不同空间分配导致 `J5`、热点位置或 `J_B` 变化，则说明 `q` 的通道分配具有优化价值；
- 若效率、安全、均衡、排队暴露之间出现非一致最优，则说明后续 G4-v2 优化必要；
- 若所有方案几乎同质，或只有“越大越好”且没有目标冲突，则不应急于做高维 `q` 优化，应先收缩变量维度或调整容量参数化；
- 若低容量方案触发明显 `cap_removed_mass`，应把该容量范围标记为不可解释或加入排队/容量下界约束。

验收标准：形成可复现命令、独立结果目录、summary、图表和论文可写结论，并明确给出 G4-v2 的容量范围、候选 baseline 和是否允许进入优化的结论。

执行结果（2026-05-16）：

- 实验脚本：`codes/g2_capacity_response_runner.py`。
- 复现命令：`D:\Anaconda\envs\interpreter\python.exe codes\g2_capacity_response_runner.py --output-root codes/results/g2_capacity_response --steps 600 --time-horizon 35 --bellman-every 5 --save-every 100000 --density-contour-levels off`。
- 结果目录：`codes/results/g2_capacity_response`。
- 总结文件：`codes/results/g2_capacity_response/g2_capacity_response_report.json`。
- 主表：`codes/results/g2_capacity_response/g2_capacity_response_summary.csv`。
- 论文表格草案：`codes/results/g2_capacity_response/g2_capacity_tradeoff_table.md`。
- 图表：`g2_capacity_levels.png`、`g2_capacity_response_pareto.png`、`g2_capacity_allocation_loads.png`、`g2_waiting_mass_timeseries.png`、`g2_capacity_hotspot_migration.png`。
- 实验矩阵已覆盖 13 个方案：no-cap 参考、high/medium/low 容量强度扫描、单入口瓶颈、同总预算容量分配、分段调度。
- 安全指标已由硬阈值暴露更新为 `J2_soft = int[((rho-rho_safe)_+ / rho_safe)^gamma] dxdt`，当前 G2-v2 采用 `gamma=1.0`、`j2_scale=0.001`。
- 关键响应：`J2_soft` / `J2_eval` 相对变化 `0.726902`，`J5` 相对变化 `0.284912`，`J_B` 相对变化 `0.489792`，拒绝通量相对变化 `1.391538`，说明 `q` 会引起可区分的系统响应。
- 原始归一化 `J2_soft` 仍是平均相对超密度，数值范围约 `0.000174-0.000466`；进入标量目标时使用 `J2_eval=J2_soft/0.001`，范围约 `0.174-0.466`，已与 `J1/J5` 处于同一优化驱动量级。
- no-cap 参考的 gate 自然尝试率已记录为 G4-v2 容量范围参考，其中 `middle:plus` 为主要受压入口，参考尝试率约 `4.25037`。
- `max_cap_removed_relative=0.005567`，低于当前 `0.02` 阈值；cap 削减存在但相对总质量较小，后续 G4-v2 仍需保留该诊断字段。
- `allow_g4_v2_optimization=true`：在小预算实验下，`q` 的响应差异、目标 trade-off 和容量范围足以支撑进入 G4-v2 小矩阵优化。

## 3.4 G4-v2 优化实验设计

- [ ] 在 `experiments/实验设计.md` 中新增 `G4-v2 内部入口容量控制` 小节，保持旧 G4 结果不被覆盖。
- [ ] 设计四类对照：
  - no-cap：`q=inf`，只保留方向控制；
  - uniform-cap：所有入口同一容量；
  - scheduled-cap：按时间分段容量；
  - optimized-cap：`z=(s,q)` 自动搜索。
- [ ] 设计消融实验：
  - 是否加入 `J_B`；
  - `L=1/2/4` 个时间段的容量参数化；
  - 固定 `s` 优化 `q` vs 联合优化 `s,q`；
  - 只限进场入口 vs 进出双向入口都限。
- [ ] 保持结果目录隔离：
  - 当前旧结果：`codes/results/g4_minimal_matrix`；
  - 新结果建议：`codes/results/g4_capacity_v2`。
- [ ] 主要指标：
  - 原有 `J1/J2_soft/J5`；
  - 新增 `J_B`；
  - 可选 `J_R`；
  - `gate_attempted_rate`；
  - `gate_allowed_rate`；
  - `gate_rejected_rate`；
  - `binding_time_ratio`；
  - `waiting_mass_peak`。

验收标准：实验矩阵能回答“入口容量控制是否产生可解释收益，以及是否值得替换 `eta` 作为连续控制变量”。

## 3.5 代码

建议按低风险增量实现，不直接重写现有 G4。

### 阶段 A：数据结构与配置

- [ ] 在 `scene_spec.py` 增加内部入口面定义，例如 `ChannelGateSpec`：
  - `channel`
  - `side = plus/minus`
  - `axis = x/y`
  - `face_index`
  - `span`
  - `waiting_region`
- [ ] 在 `scene.toml` 增加每条通道的 `plus_gate` 和 `minus_gate`。
- [x] 在路线或 case 配置中增加容量控制表，支持分段常数：
  - `channel`
  - `side`
  - `time_start/time_end`
  - `rate`
- [x] 编译为运行时对象 `ChannelGateModel` / `GateCapacitySchedule`。

说明：本轮没有要求手写 `scene.toml` 中的 gate 坐标，而是先从现有 `channel` 区域自动推导 `plus/minus` 入口面和等待区，以便 G2-v2 先形成可运行结果。若论文需要精确展示 $\Sigma_c^\pm$ 的几何定义，后续仍可补显式 `ChannelGateSpec`。

### 阶段 B：通量限制器

- [x] 拆分 `update_density()`：
  - `compute_face_fluxes()` 保留；
  - 新增 `apply_internal_gate_limits(fx_by_group, fy_by_group, gates, schedules, time, dx)`；
  - 新增“用给定通量更新密度”的函数。
- [x] 在 `runner.py` 中改为先计算所有群体自由通量，再统一限流，最后更新密度。
- [x] 记录限流诊断量并传给 `metrics.record_step()`。
- [x] 增加质量守恒日志，特别是 cap 削减量。

### 阶段 C：指标与目标函数

- [x] 扩展 `CaseStats`：
  - gate attempted/allowed/rejected cumulative；
  - waiting mass cumulative/peak；
  - binding time ratio。
- [x] 扩展 `ObjectiveConfig`，新增 `lambda_jb` 和 `lambda_jr`，默认 0，保证旧实验不受影响。
- [x] 扩展 `summary.json` 与 `timeseries.csv` 输出。

### 阶段 D：G4-v2 优化器

- [ ] 新建 `codes/crowd_bellman/g4_capacity_control.py` 或在旧 G4 中新增独立 `ControlVectorV2`，不要覆盖 `ControlVector(directions, eta)`。
- [ ] `ControlVectorV2` 包含：
  - `directions`
  - `q` 或 `rates`，形状为 `C x 2 x L`
  - `eta0` 只作为固定参数写入配置。
- [ ] 先实现小预算随机搜索和坐标搜索，再考虑是否沿用 SA-HBO 的分块框架：
  - 离散块：`s` 邻域搜索；
  - 连续块：`q` 的投影随机近似或坐标扰动；
  - 投影：保证 `s-q` 方向一致性和 `FREE` 的 `q^+ + q^- <= qbar`。

阶段 D 需等 3.3 的 G2-v2 容量响应实验完成后再进入，避免在 `q` 的响应面和有效范围尚不清楚时直接做高维搜索。

验收标准：旧 `g4_sahbo.py` 和旧结果可继续运行，新 G4-v2 有独立 CLI、配置和结果目录。

## 3.6 论文

论文不建议立即整体替换。建议分两步。

第一步：作为扩展方案写入方法讨论或附录。

- [ ] 在第 4 章方法中增加“内部入口容量控制扩展”小节。
- [ ] 保留当前 `U(x)` 表示硬方向规则、`M(x)` 表示软几何引导的叙述。
- [ ] 将 `eta` 从“优化变量”降级为“固定几何引导强度”时，要先引用敏感性验证结果。
- [ ] 在实验章节中增加 `G2-v2：入口容量控制响应与可优化性验证`，先报告人工容量策略矩阵的响应面、目标 trade-off 和容量范围，再进入 G4-v2 优化结果。

第二步：如果 G4-v2 实验通过，再替换主线。

- [ ] 改写 4.7 优化变量：`z=(s,eta)` -> `z=(s,q)`。
- [ ] 改写优化目标：`J = lambda_1 J1 + lambda_2 J2 + lambda_5 J5 + lambda_B J_B + lambda_R J_R`，其中 `lambda_R` 可取 0。
- [ ] 改写 G4 实验：
  - 旧 G4 作为 baseline 或消融；
  - 新 G4-v2 作为主结果；
  - 报告入口限流带来的排队、通过率和风险热点变化。
- [ ] 更新图表：
  - 入口限流机制示意图；
  - G2-v2 容量强度响应曲线；
  - G2-v2 同总预算容量分配对比图；
  - `q` 时间分段示意；
  - 限流前后等待区质量曲线；
  - G4-v2 方法对比；
  - 最优 `s,q` 可视化。

验收标准：论文中的变量、代码配置、实验结果和图表使用同一套 `z=(s,q)` 口径，不再出现理论、代码、结果三套变量不一致。

## 4. 推荐执行顺序

1. **先做验证，不先改论文主线**：完成 `eta` 敏感性和单通道限流 smoke。
2. **再实现通量限制器与指标**：把内部入口限流放进下层仿真器，并保证旧实验兼容。
3. **先跑 G2-v2 容量响应实验**：证明 `q` 会产生可解释响应、目标之间存在 trade-off，并确定可优化容量范围。
4. **再跑 G4-v2 小矩阵**：用独立结果目录比较 `z=(s,eta)` 与 `z=(s,q)`，并基于 G2-v2 的范围限制搜索空间。
5. **最后改论文主线**：只有当 G2-v2 和 G4-v2 的机制、质量守恒、响应差异和目标改善都成立，才把论文主线切换到 `model v2`。

## 5. 当前风险清单

- `eta` 已完成小预算固定方向敏感性验证，目标函数相对变化低于 5% 阈值；但这还不是完整 G4-v2 替换依据，后续正式实验仍需报告更大预算和更多方向配置下的边界。
- 内部入口面的几何定义如果只用 `probe_x` 近似，会不足以支撑论文中的 `Sigma_c^\pm` 表述。
- 限流后排队可能触发 `rho_max` 裁剪，导致质量损失，需要显式记录或调整处理方式。
- `q` 已通过机制 smoke，但尚未证明在完整行为模型下会形成足够稳定、可解释的响应面；必须先完成 G2-v2，再进入 G4-v2 优化。
- `q` 参数维度较高，四通道、双方向、四时段就是 32 个连续变量，直接做高维黑箱优化预算会不够。
- 如果不加入 `J_B` 或排队约束，优化器可能选择过强限流，得到管理上不可接受的方案。
- 论文中已有 SA-HBO 和 G4 结果，切换变量会牵动第 4 章方法、第 5 章实验、图表、摘要和结论。
