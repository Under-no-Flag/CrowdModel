# 稿件系统性审阅意见（整合版）

**审阅对象**：*Macroscopic Crowd Control with Direction Constraints and Entrance-Rate Control for Urban Pedestrian Zones*

**总体审阅结论**：建议按 **“大修（Major Revision）”** 处理。

**核心定位建议**：将稿件定位为 *“面向开放步行区多通道管控的可解释宏观建模与仿真优化框架”*，而非 *“已可直接指导真实外滩现场部署的成熟系统”*。

> 本文档整合了两份独立审阅意见，并补充了关于“无量纲数字可读性”和“数值/符号排版”的专项意见，去重后形成统一的、可直接用于回复作者的完整审稿报告。文末附 P0/P1/P2 修改清单、可直接套用的摘要降调版本与 Discussion 限制说明。

---

## 1. 总体评价

### 1.1 选题与框架价值

稿件选题具有明确应用价值。论文在 Hughes 连续介质人群模型基础上，引入局部可行方向集合 $U(x;s)$ 表示单向、双向和关闭通道规则，引入固定各向异性张量 $M(x;\eta_0)$ 表示通道几何引导，再通过内部入口流率上界 $q_c^\pm(t)$ 表示“通道开放但限流”的管控动作，并使用 HCMBO 联合搜索方向配置与入口放行强度。该框架解释性较好，契合开放景区、滨水观景平台、节假日人流组织等应用场景。

### 1.2 主要优点

1. 将单向通行规则、几何引导、多阶段路线选择、入口限流和负载均衡纳入统一的宏观模型框架；
2. 明确区分固定行为偏好参数 $\hat p$、固定几何引导参数 $\eta_0$ 与可控变量 $z=(s,q)$，形成清晰的“行为层—管控层”结构；
3. 通过机制验证、方向响应、入口流率响应、优化器横向比较、HCMBO 消融和外滩迁移实验，形成了相对完整的实验链条；
4. 论文在讨论部分主动承认简化实验与真实部署之间的差距，也诚实指出了外滩迁移实验中的效率/安全 trade-off，这降低了过度泛化风险，是值得肯定的科学态度。

### 1.3 主要结构性风险

1. 数学定义尚不够严谨，特别是 $M(x;\eta_0)$ 的物理含义、$U(x;s)$ 在非通道/关闭/障碍区域的定义、Bellman–守恒律耦合的适定性、入口限流下的质量守恒，以及 density-cap removal 的处理；
2. 优化目标的标准化方式与指标编号不统一，导致不同实验中的 objective 数值缺乏可比性；**且文中出现大量无量纲数字，对读者而言缺乏直接物理意义**（详见 §2.5、§2.8）；
3. HCMBO 算法描述偏流程化，缺少可复现的伪代码、代理模型、候选选择准则和多保真细节，关键超参数（含权重 $\lambda$）缺失；
4. 主优化实验仅 5 个随机种子，且 HCMBO 与 TPE-Mixed BO 的均值差异相对标准差并不大，结论不宜写得过强；baseline 比较存在 G6/G7-D 混用的公平性疑问；
5. 外滩迁移实验主要体现负载均衡改善，同时伴随效率项、安全暴露和残留质量的潜在恶化，必须避免笼统表述为“效率与安全同时改善”；
6. 安全暴露 $J_2$ 仅作为加权软目标，对“人群安全”这一核心价值主张构成风险（优化器可用安全换取其他指标）。

**综合判断**：论文主线值得保留，但需要系统性补强 **模型闭合性、目标函数可解释性、算法可复现性、统计结论稳健性，以及数值结果的可读性**。

---

## 2. 最高优先级修改问题（P0）

### 2.1 重新澄清 $M(x;\eta_0)$ 的物理含义（可能的实质性数学错误）✅️

稿件称 $M(x;\eta_0)$ 为 anisotropic metric tensor，并在 Bellman 更新式(10)中使用：

$$
\phi(x)=\min_{u\in U(x;s)}\left[\phi(x+\Delta x\, u)+\frac{\Delta x}{f(\rho(x,t))}\frac{1}{\sqrt{u^\top M(x;\eta_0)u}}\right].
$$

**问题**：在该式中，$u^\top M u$ 越大，步进代价越小。这与传统各向异性 eikonal/Finsler 框架中“代价正比于 $\sqrt{u^\top M u}$（沿高代价方向走代价大）”的约定方向相反。因此 $M$ 在此处更像是 **mobility tensor / inverse metric（co-metric）**，而非传统“代价度量张量”。

同时式(9)定义 $M_c(x;\eta_0)=\beta_c\!\left(\eta_0\,\tau_c\tau_c^\top+n_c n_c^\top\right),\ \eta_0\ge 1$，并称“larger $\eta_0$ reduces effective cost of tangential motion”。**请务必核对式(9)与式(10)在数学上是否自洽**——这是一个可能导致全文机制解释方向相反的实质性问题。

**建议改写**：

> We interpret $M(x;\eta_0)$ as an anisotropic *mobility* tensor, or equivalently the inverse of a local travel-cost metric. A larger value of $u^\top M(x;\eta_0)u$ indicates a lower effective step cost in direction $u$.

若作者坚持沿用 “metric tensor” 术语，必须明确说明它是 “the inverse metric used in the discrete Bellman cost”。

### 2.2 完整定义 $U(x;s)$，并保证数值一致性

当前式(8)仅在通道区域 $\Omega_c$ 内定义：

$$
U_c(x;s_c)=
\begin{cases}
\{\tau_c(x)\}, & s_c=+1,\\
\{-\tau_c(x)\}, & s_c=-1,\\
\{\tau_c(x),-\tau_c(x)\}, & s_c=0,\\
\varnothing, & s_c=\varnothing.
\end{cases}
$$

存在三个缺口：

1. **非通道区域的 $U(x;s)$** 未明确：是全方向集合、八邻域方向集合，还是受墙体/障碍约束后的邻接方向集合？
2. **关闭通道 $s_c=\varnothing$** 的数值处理未明确：若 $U=\varnothing$，则 $\min_{u\in\varnothing}$ 如何处理？是令该网格不可达、令 $\phi=+\infty$，还是将通道区域从 $\Omega_w$ 中删除？
3. **$\tau_c(x)$ 与网格方向不一致**时的离散方式未明确：实际求解通常基于有限邻域方向，而非任意连续方向；$u$ 是单位方向还是带模长？$x+\Delta x\,u$ 取 4 邻域还是 8 邻域？

**建议将式(10)改写为更数值一致的离散形式**：

$$
\phi_i=\min_{k\in \mathcal A_i(s)}
\left[
\phi_{i+k}
+
\frac{\ell_k}{f(\rho_i)+\varepsilon_f}\,
\frac{1}{\sqrt{e_k^\top M_i e_k}}
\right],
$$

其中 $\mathcal A_i(s)$ 是由障碍物、边界和通道规则共同决定的可行邻接方向集合，$\ell_k$ 是轴向或对角步长，$e_k$ 是有限邻接方向。

此外，建议至少引用 fast sweeping / fast marching / ordered upwind 的收敛理论，说明本文离散格式继承了哪些适定性与收敛性质。

### 2.3 讨论 Bellman–守恒律耦合系统的适定性与数值稳定性

式(2)、(5)的守恒律与式(10)的 Bellman 双向耦合（密度影响势函数，势函数决定速度方向进而影响密度演化）。这种耦合系统的适定性、是否产生振荡或非物理解，文中完全未讨论；而 Hughes 模型本身解的正则性即存在争议，本文的多阶段 + 方向约束扩展使问题更复杂。

**建议**在 Section III-E 补充：CFL 条件的具体形式、Bellman 更新频率对稳定性的影响、是否观察到振荡或激波、以及多阶段转移项是否引入额外刚性。

### 2.4 明确入口限流下的质量守恒（含 density-cap removal 的正式定义）

稿件提出 attempted entrance flux $A_c^\pm(t)$、admitted flux $\hat A_c^\pm(t)=\min\{A_c^\pm(t),q_c^\pm(t)\}$，并用比例 $\theta_c^\pm(t)$ 缩放入口通量。设计合理，但未放行质量的保存方式仍不清楚：

1. 未放行质量是保留在上游有限体积单元中，还是进入单独队列变量 $B_c^\pm(t)$？
2. 若上游密度已接近 $\rho_{\max}$，未放行质量是否会触发 density cap removal？
3. 若存在 density cap removal，它是否破坏质量守恒？是否进入目标惩罚或可行性约束？
4. 第 IV-A 将 density-cap removal mass 不超过参考质量 2% 作为可行性标准，但方法部分**没有对这个量做正式定义**。

**建议新增方法说明**：

> The queue variable $B_c^\pm(t)$ is not merely a diagnostic quantity. When $A_c^\pm(t)>\hat A_c^\pm(t)$, the unreleased flux is retained in the upstream control volume or accumulated in an explicit queue reservoir, with subsequent release governed by [...]. Density clipping, if used, is recorded as $M_{\mathrm{cap}}$ and either penalized in the objective or imposed as a feasibility constraint.

可借助 Table VII 已有的量（Final cumulative inflow / sink flow / system mass）做一次显式质量闭合检查（总流入 = 系统内 + 流出 + 等待 + 削除），并报告残差，以正面回应守恒性质疑。

### 2.5 统一目标函数、诊断指标与编号 ✅️

方法部分定义五项指标：效率 $J_1$、安全 $J_2$、负载均衡 $J_3$、等待 $J_4$、平滑性 $J_5$，并写成 $J(z)=\sum_{k=1}^5\lambda_k\tilde J_k(z)$。但实验部分有时只报告 $\tilde J_1,\tilde J_2,\tilde J_3$，有时写成 $\tilde J=\tilde J_1+\tilde J_2+\tilde J_3$，有时又出现 $J_2^{\mathrm{eval}}$。这带来明显的可读性与可复现性问题。

**建议统一**：

1. 机制与响应实验使用三项诊断目标：$J_{\mathrm{diag}}=\tilde J_1+\tilde J_2+\tilde J_3$；
2. 优化实验使用五项完整目标：$J_{\mathrm{opt}}=\sum_{k=1}^5\lambda_k\tilde J_k$；
3. 迁移实验若采用不同权重或不同项，单独定义 $J_{\mathrm{transfer}}$。

正文应明确说明 Table II、III、IV、VII 中的 Objective 分别属于哪个目标。

**必须补充标准化方式**，例如：

$$
\tilde J_k=\frac{J_k-J_k^{\mathrm{ref}}}{s_k}
\qquad\text{或}\qquad
\tilde J_k=\frac{J_k}{J_k^{\mathrm{base}}}.
$$

并**必须报告权重 $\lambda=(\lambda_1,\dots,\lambda_5)$ 的具体取值**——它直接决定所有 objective 数值，目前正文缺失。

> 注：2.6338、0.5102、1.6851、0.7410 等目标值出现在不同实验中，若标准化基准或目标项不同，这些数值**不能直接横向比较**，相关结论需谨慎。

### 2.6 补足 HCMBO 的可复现细节（含完整算法与超参数）部分✅️，差下面8点明确

当前 HCMBO 主要通过 Table I 的流程表和文字描述给出，不足以支撑算法论文的复现。Section III-G 的 Step 3 “structured black-box search”——方法的核心——被一笔带过，读者无法复现。

**建议补充正式 Algorithm 1（带输入、输出、循环、终止条件），并明确**：

1. $x\in[0,1]^d$ 的维度 $d$ 如何确定；
2. $T_s(x)$ 的具体映射方式（线性缩放、分段常数、样条或其他）；
3. surrogate model 的类型（GP、random forest、TPE、RBF、ranking surrogate 或自定义）；
4. acquisition / 候选选择准则（EI、LCB、概率可行性、rank-based 或启发式规则）；
5. multi-fidelity 的低/中/高保真定义（时间步数、网格、Bellman 更新频率、复核数量）；
6. 方向候选集 $S_{\mathrm{cand}}$ 的连通性约束；
7. 各 baseline 的预算是否完全相同（含低保真评估、高保真复核与最终排名成本）；
8. HCMBO 是否使用了比 TPE/DE 更多的结构先验；若是，应**诚实表述为结构化方法的优势**，而非简单声称通用算法性能更差。

**建议伪代码骨架**：

```text
Algorithm 1: HCMBO for joint direction–entrance-rate control
Input: direction state set, capacity bounds, budget B,
       high-fidelity recheck budget H, fixed p_hat, fixed eta_0
1. Generate feasible direction set S_cand (apply connectivity/operational constraints)
2. For each s in S_cand, construct capacity parameterization q = T_s(x)
3. Initialize samples by structured design
4. While budget remains:
   4.1 Fit/update surrogate using evaluated samples
   4.2 Select direction-capacity candidates under feasibility constraints
   4.3 Evaluate candidates using low/medium-fidelity simulator
   4.4 Promote promising candidates to high-fidelity pool
5. Re-evaluate finalists under unified high-fidelity setting
6. Return the best high-fidelity candidate z*
```

### 2.7 降低优化结果的统计结论强度，并澄清 baseline 公平性

**统计层面**：Table IV 中 HCMBO 平均目标 2.6338，TPE-Mixed BO 为 2.7403，差值约 0.1065；但 HCMBO 标准差 0.3106、TPE 标准差 0.2475，且仅 5 个种子。“平均更好”可以说，但不宜写成强统计结论。

**建议**：

1. 增加 paired-seed 对照表；
2. 增加 paired t-test 或 Wilcoxon signed-rank test，报告 $p$ 值；
3. 报告 95% 置信区间与 effect size；
4. 若暂不做检验，将结论改为：
   > HCMBO obtains the best average objective in the tested five-seed protocol and improves over TPE-Mixed BO in four out of five paired seeds.
5. 避免 “statistically superior”“consistently outperforms” 等过强表达；建议种子数扩至 ≥10–15。

**公平性层面**：Section IV-E 提到 “HCMBO data taken from G7-D mainline, other baselines from G6 horizontal comparison”。这立刻引发质疑：

1. HCMBO 与 baselines 是否在完全相同的评估预算（$B=400$）、相同种子、相同高保真协议下比较？G6 与 G7-D 两套实验混用可能引入偏差，须明确说明一致性。
2. 各方法超参数是否经过同等调优（TPE 对自身超参数亦敏感）；
3. Prior Baseline 的 feasible rate 为 0%、目标高达 5.4774，与其余方法差距悬殊，需解释它代表什么，避免被视为“稻草人对手”。

### 2.8 处理“大量无量纲数字缺乏可读性”问题（**新增重点**）✅️

全文出现大量无量纲/标准化数字，例如目标值 2.6338、0.5102、1.6851、0.7410、1.0140，份额类 48.78%、100.00%、82.30%、91.34%，以及 415.75、1213.76、427.89、3239.295、2545.508 等 raw counts。**这些数字对读者而言缺乏直接物理意义**，难以判断其工程含义与重要性，削弱了结果的说服力与可解释性。具体问题与建议：

1. **标准化目标值（如 2.6338、0.7410）**：在未给出标准化公式与基准前，读者无法判断 2.6338 是“好”还是“差”，也无法判断 0.1065 的均值差是否重要。**建议**：每个无量纲指标首次出现时给出标准化定义与参考基准；并优先报告**相对改进（%）**和**相对于某一可解释基线的比值**，而非裸的绝对值。摘要尤其应强调相对改进与可解释性，而非绝对数 2.6338。
2. **raw counts（如 415.75 reverse-direction mass time、1213.76 counterflow cell time、427.89 unreleased attempted flow、3239.295 unreleased flow）**：这些量纲为“mass·time”“cell·time”或“mass”的累积量，单位与物理意义未交代。**建议**：在表注或正文中给出每个量的定义式与单位（或明确其为无量纲仿真单位 a.u.），并在可能时归一化为“占总流入/总质量的比例”或“相对参考工况的倍数”。例如将“unreleased attempted flow = 427.89”改写为“占总尝试流入的 X%”。
3. **百分比份额**：通道流量份额（48.78%→100.00% 等）解释性较好，建议保留并作为正面范例，但应统一说明分母（是瞬时流量、累计通过量还是吸收量）。
4. **有效数字**：3239.295、2412.404、2545.508 等给到 3–4 位小数，超过仿真可重复精度，建议统一为 2–3 位有效数字，并在表注说明精度来源。
5. **单位与符号统一**：建议在方法部分新增一张“符号与单位对照表（Nomenclature）”，列出 $\rho,v,f,\phi,M,U,q,A,\hat A,\theta,B,R,J_k,\tilde J_k$ 等的定义、单位（或注明无量纲）与取值范围；这样所有后续无量纲数字都能对照查阅。
6. **可解释性锚点**：建议为关键无量纲指标提供一个“物理锚点”，例如把 high-density exposure 折算为“面积·时间（m²·s）中密度超过 $\rho_{\mathrm{safe}}$ 的占比”，把 travel time 折算为“平均每位行人停留时长”的近似，让读者把无量纲数字映射回现实直觉。

> 小结：无量纲化本身是合理的（不同量纲指标需标准化后加权），问题在于**缺少定义、缺少基准、缺少物理锚点、有效数字过多**。补足这四点后，现有数字即可由“难以解读”转为“可解释、可比较”。

### 2.9 外滩迁移实验结论必须降调

外滩迁移实验显示，受控组目标从 1.0140 降至 0.7410，主要来自负载均衡项 $J_3$（0.6043→0.2908）；但效率项 $J_1$ 从 0.3813 升至 0.4145，安全暴露 $J_2^{\mathrm{eval}}$ 从 0.0284 升至 0.0357，累计流出更小，系统残留质量更大。这说明该策略并非“效率、安全、均衡全部改善”，而是：**在当前权重下，通过牺牲部分效率与安全暴露，换取显著的负载均衡改善。**

“26.9% 相对改进”的表述容易误导，因为它掩盖了效率与安全的退化。**建议改写**：

> The Bund-inspired transfer experiment suggests that the proposed control variables can redistribute channel loads under a simplified geometry. However, under the current weights, the improvement mainly comes from reduced load imbalance, while travel-time and high-density-exposure terms do not improve simultaneously.

### 2.10 将安全暴露纳入硬约束考量

多处结果显示 HCMBO 一个种子安全暴露偏高（IV-E、结论），外滩实验中安全也变差。对于人群安全应用，把 $J_2$ 仅作为加权软目标存在风险——优化器可用安全换取其他指标。**建议**讨论或实现将密度上限/安全暴露作为可行性硬约束：

$$
J_2(z)\le \epsilon
\qquad\text{或}\qquad
\max_{x,t}\rho(x,t)\le \rho_{\mathrm{crit}}.
$$

这也与 Table IV 的 feasible rate 定义（density-cap removal mass ≤ 2%）形成更一致的安全框架。

---

## 3. 逐章节修改建议

### 3.1 摘要

摘要信息完整，但过长、过满、过强。建议压缩为四层：问题、方法、实验、结论。需修改：

1. “novel”“fine-grained large-scale crowd management”“ultimately”“effective decision-support tool” 等表述偏强；
2. HCMBO 的优势应限定在 tested protocol 内；
3. 应明确实验是 simplified scenic-platform scenario，而非已完成真实外滩部署验证；
4. “consistently outperforms baseline algorithms” 改为 “achieves the best mean objective under the tested protocol”；
5. 核心数字 2.6338 应改为以**相对改进**为主表述，并说明其为固定权重下的标准化标量目标（呼应 §2.8）。

### 3.2 引言

现实背景较充分，但部分表述偏政策化（如 “tests urban governance capacity and the modernization level of social safety governance systems”），IEEE 风格建议改为更技术化表述。建议重构逻辑：

1. 开放步行区与大型活动场景存在多通道、多阶段、高密度、强管控约束；
2. 现有方法不足：静态容量、单 OD、缺少单向规则与入口限流联动、缺少结构化优化；
3. 本文问题：在固定行为偏好与几何引导参数下，联合优化通道方向 $s$ 与入口放行 $q$；
4. 本文贡献：模型、入口限流、指标体系、HCMBO、实验验证。

引言末尾提前说明，避免读者误以为实验已直接还原真实现场：

> The Bund case motivates the scenario class, while the experiments use controlled abstract geometries to verify mechanisms and algorithmic behavior.

### 3.3 相关工作

覆盖面较广但主线略散。建议按本文缺口重组为四类：(1) crowd control 与 facility/flow regulation；(2) macroscopic pedestrian models 与 Hughes-type route choice；(3) directional constraints 与 anisotropic/Finsler/HJB modeling；(4) simulation-based mixed-variable optimization。

- **第三类需重点补强**：当前提到 constrained HJB formulations 但文献支撑不足，建议补充 anisotropic eikonal、Finsler metric、ordered upwind / fast marching、HJB with state/control constraints 等研究，否则 $U+M$ 的模型创新显得突兀。
- **第四类需新增背景**：目前仅在实验中用 TPE 作对比，正文缺少对 mixed-variable / Bayesian optimization 的背景综述，难以支撑 HCMBO 的方法学定位。
- 应在引言或相关工作结尾用一段明确区分“哪些借用已有方法、哪些为本文首创”，强化原创点界定。

### 3.4 方法部分

方法部分是核心，需更严谨。建议新增或强化：

1. **假设清单**：宏观连续介质、固定偏好 $\hat p$、固定 $\eta_0$、同质速度–密度关系、局部最优路径选择、无显式个体避碰；
2. **边界条件**：入口、出口、墙体、障碍物、封闭通道与内部入口界面；
3. **数值格式**：upwind flux 表达、CFL 条件、Bellman 更新频率、密度上限处理；
4. **质量账本**：阶段转移、入口限流、出口吸收、density cap removal 的逐项收支；
5. **标准化目标**：所有 $\tilde J_k$ 的公式、权重与基准（呼应 §2.5、§2.8）；
6. **符号与单位对照表（Nomenclature）**（呼应 §2.8）；
7. **HCMBO 伪代码**：输入、输出、循环、候选生成、代理模型、保真切换与高保真复核。

### 3.5 实验部分

素材较丰富，建议显式设置 research questions：

- **RQ1**：$U(x;s)$ 是否能消除禁行方向与逆向流？
- **RQ2**：$M(x;\eta_0)$ 是否能改变吸引域与入口预对齐？
- **RQ3**：方向配置 $s$ 是否带来效率–安全–均衡的非单调权衡？
- **RQ4**：入口流率 $q$ 是否是必要控制变量？
- **RQ5**：HCMBO 是否在同预算下优于通用混合变量优化器？

对 Table IV 建议增加：(1) 每个 seed 的 paired result；(2) convergence 曲线的置信区间或 seed-wise 曲线；(3) 运行时间；(4) 同等高保真评估次数；(5) 高保真复核前后排名变化；(6) 各项指标分解，而非仅 scalar objective。

对入口流率实验建议强调：

> 入口限流不是越小越安全。过强限流可能导致入口上游排队与密度累积，从而提高安全暴露。因此入口限流应与方向配置联合优化，而非作为单调安全控制项。

### 3.6 讨论与结论

讨论较诚实，应保留并将部分内容前移至实验开头说明：

1. 四通道场景是机制验证与算法行为分析，不是实地复现；
2. 外滩迁移实验是抽象 transfer demonstration，不是现场部署评估；
3. 真实部署需要流入、密度、速度、通道流量、排队长度与路线选择数据校准。

结论中将 “HCMBO consistently outperforms baseline algorithms.” 改为：

> HCMBO achieves the best average objective under the tested five-seed, 400-evaluation protocol.

并补充限制：(1) 仍为四通道抽象场景；(2) 随机种子数量有限；(3) 权重设置影响方法排序；(4) 一个 HCMBO seed 仍有较高安全暴露；(5) 尚未完成真实数据校准与在线需求更新。

---

## 4. 必做 / 建议补强实验

### 4.1 统计显著性实验（P1，必做）

至少对以下比较进行 paired test：HCMBO vs TPE-Mixed BO、vs Random Search、vs Pure SA、vs Enum-DE。报告：mean difference、median difference、paired t-test 或 Wilcoxon signed-rank test、95% 置信区间、effect size。

### 4.2 权重敏感性实验（P1）

外滩实验显示结果主要受负载均衡项驱动。建议扫描：安全优先（提高 $\lambda_2$）、效率优先（$\lambda_1$）、均衡优先（$\lambda_3$）、等待优先（$\lambda_4$）、平滑优先（$\lambda_5$），展示最优策略的方向配置是否变化及各项 trade-off。

### 4.3 需求强度敏感性实验（P1）

设置低/中/高 inflow 或初始密度，证明方法不仅在单一需求水平有效：低需求（拥堵不明显，方向规则影响弱）、中需求（局部拥挤，方向与限流有效）、高需求（入口限流与队列主导）。

### 4.4 安全硬约束实验（P2）

将 $J_2$ 由加权软目标改为硬约束形式 $J_2(z)\le\epsilon$ 或 $\max_{x,t}\rho\le\rho_{\mathrm{crit}}$，更贴近公共安全管理场景（呼应 §2.10）。

### 4.5 质量闭合性验证（P1）

利用 Table VII 已有量做一次显式质量平衡检查并报告残差，正面回应守恒性质疑（呼应 §2.4）。

---

## 5. 图表与版式修改建议

1. **Fig. 1 / Fig. 10 / Fig. 11**：关键场景与结果图，提升分辨率与标注清晰度，确保印刷质量。
2. **Fig. 1 现场照片版权**：需注明来源/授权；若作者拍摄，标 “Photo credit: authors.”。
3. **Fig. 2 框架图**：信息量过大，建议拆为两张——(a) 模型结构图（输入、控制变量、Bellman–守恒律模拟器、输出）；(b) 优化流程图（方向候选、容量映射、代理搜索、多保真评估、高保真复核）。同时修正符号渲染异常（如 $\rho=\sum_\sigma\sum_r \rho_{\sigma,r}$ 中 $\sigma$ 误显示为 $o$）。
4. **Fig. 4**：展示 $U+M$ 机制空间迁移的重要图，建议改双栏宽图、放大色标与子图标题、标出 top/middle/lower-middle/bottom 通道名、加箭头说明 dominant flow channel 迁移。
5. **Fig. 5 / Table II**：Case 编号跳跃（C1、C3、C4、C6、C8、C9），正文称有 13 个方向设置而表中仅列 6 个非支配解；应在表题/正文明确这是从 13 个中筛出的非支配集，避免困惑。
6. **Fig. 9**：入口流率热图需统一色标与单位说明，并在正文解释：为何相同方向结构 $[T,M,LM,B]=[W,E,E,W]$ 下，不同方法的分段流率仍造成性能差异（结合通道流率–等待–安全暴露的关系）。
7. **Table IV**：作为主结果应单独突出，避免与 Fig. 5、Table III、Fig. 6、Fig. 7 同页拥挤；建议增列 “paired wins vs HCMBO/TPE” 与 “mean high-density exposure term”。
8. **所有表格的数字精度**：统一为 2–3 位有效数字，并在表注说明精度与单位（呼应 §2.8）。

---

## 6. 语言与表述修改建议

### 6.1 降低过强措辞

| 原表达 | 建议表达 |
| --- | --- |
| novel HCMBO algorithm | we develop an HCMBO algorithm |
| effectively captures | is able to reproduce / represents |
| consistently outperforms | achieves the best average performance under the tested protocol |
| effective decision-support tool | potential simulation-based decision-support tool |
| improves safety and efficiency | changes the efficiency–safety–balance trade-off |

### 6.2 统一术语

1. 用 “internal entrance-rate upper-bound control” 统一替代 “entrance capacity control / entrance-rate control / release intensity” 的混用；
2. 用 “realized channel-throughput imbalance” 替代 “channel load variance / channel-flow variance”；
3. 用 “high-density exposure” 替代 “density exposure / safety exposure”；
4. 用 “anisotropic mobility tensor” 替代易误解的 “metric tensor”，或明确为 inverse metric（呼应 §2.1）；
5. 统一 channel / passage / corridor 三词，或在术语表中明确区分。

### 6.3 其他语言问题

1. 避免中文论文式长句，引言与摘要中多个从句叠加的长句应拆分；
2. 删除/改写 “the same logic as the Chinese experimental outline”（IV 节开头）这类疑似内部草稿直译的表述；
3. 修正全文上下标与数学符号转换错误，例如式(8)的闭合符号 $\varnothing$、式(10)根号排版、$q_c^\pm(t)$、$\hat A_c^\pm(t)$、$\theta_c^\pm(t)$ 等的上下标错乱（在 LaTeX 源文件中逐一核对编译后版本）。
4. 作者信息：首页 “Author Name” 与脚注 “Department, Institution, City, Country” 为占位符，投稿前须按期刊要求填写（双盲则保持匿名）。

---

## 7. 建议重写的贡献表述

建议将贡献点收敛为更集中、更技术化的三点，避免把“指标体系”包装成过大的独立创新：

> The contributions of this paper are threefold.
> First, we formulate a multi-stage, multi-route macroscopic crowd model in which subpopulations have target-specific Bellman potentials while sharing a total-density-dependent speed function.
> Second, we integrate hard direction constraints and soft geometric guidance through a local admissible control set $U(x;s)$ and an anisotropic mobility tensor $M(x;\eta_0)$, and further introduce internal entrance-rate upper bounds to distinguish attempted and admitted channel fluxes.
> Third, we develop a hierarchical constrained mixed-variable black-box optimization procedure for the joint direction–rate control problem and evaluate it under unified high-fidelity rechecking against several baseline optimizers.

---

## 8. 投稿前修改清单

### P0：必须修改

- [ ] 明确 $M$ 是 metric、inverse metric 还是 mobility tensor，并核对式(9)与式(10)的数学自洽性；
- [ ] 完整定义非通道区域、关闭通道、障碍边界下的 $U(x;s)$，并给出数值一致的离散 Bellman 形式；
- [ ] 讨论 Bellman–守恒律耦合的适定性与数值稳定性（CFL、振荡）；
- [ ] 补充有限体积通量、Bellman 离散、density cap removal 的数值细节与质量守恒说明；
- [ ] 给出 $\tilde J_1,\ldots,\tilde J_5$ 的完整标准化公式与权重 $\lambda$ 取值；
- [ ] 区分 diagnostic / optimization / transfer objective，统一指标编号；
- [ ] **为所有无量纲数字补充定义、基准、单位与物理锚点，统一有效数字，新增 Nomenclature 表**；
- [ ] 补充 HCMBO 伪代码（Algorithm 1）与关键超参数；
- [ ] 澄清 G6/G7-D 混用问题，保证 baseline 比较的预算/种子/协议一致；
- [ ] 降低摘要与结论中的泛化性表述；
- [ ] 外滩迁移实验不得表述为效率与安全同时改善。

### P1：强烈建议修改

- [ ] 增加 paired-seed 结果与统计检验（p 值、CI、effect size），扩大种子数；
- [ ] 增加权重敏感性分析；
- [ ] 增加需求强度敏感性分析；
- [ ] 补充质量闭合性验证（残差报告）；
- [ ] 报告每种优化方法的运行时间与同等高保真评估预算；
- [ ] 报告各项指标分解，而非仅 scalar objective；
- [ ] 在相关工作中补强 anisotropic/HJB/Finsler 与 mixed-variable BO 两类文献。

### P2：可选增强

- [ ] 增加真实/半真实数据校准流程图；
- [ ] 将 scalarized optimization 扩展为 Pareto weight sweep；
- [ ] 引入硬安全约束；
- [ ] 增加 SMAC、CMA-ES with categorical encoding、mixed-integer BO 等更强 baseline；
- [ ] 增加入口等待长度、残留质量、通道流量份额等工程指标（并归一化呈现）。

---

## 9. 建议的最终定位

本稿最适合定位为：

> 一个面向开放步行区多通道管控的**可解释宏观建模与仿真优化框架**。

不建议包装为已能直接指导外滩现场实操的成熟系统。只要补足 **数学定义、目标标准化与可读性、算法可复现性、统计支撑**，稿件的说服力会显著提高。论文思路清晰、问题定义合理、实验较诚实（主动承认局限与 trade-off），主要风险集中于：单场景普适性、统计支撑薄弱、核心算法描述不足、度量张量公式自洽性，以及大量无量纲数字的可读性。妥善处理上述 P0 问题后，本工作具备发表价值。

---

## 10. 可直接套用的摘要降调版本

> Open pedestrian zones and major event venues often experience interacting pedestrian streams, localized congestion, and operational constraints such as one-way channels and entrance metering. This paper develops a macroscopic simulation-based optimization framework for direction-constrained crowd control in multi-channel pedestrian zones. Building on a Hughes-type continuum model, the proposed simulator couples density conservation with target-specific Bellman potentials for multi-stage and multi-route subpopulations. Hard operational rules are represented by local admissible direction sets, while soft channel alignment is modeled through an anisotropic mobility tensor. Internal entrance-rate upper bounds are further introduced to distinguish attempted and admitted channel fluxes. A scalarized objective combines travel time, high-density exposure, realized channel-throughput imbalance, entrance waiting, and control smoothness. To solve the resulting mixed discrete-continuous black-box problem, a hierarchical constrained mixed-variable optimization procedure is developed. Experiments on a simplified scenic-platform scenario show that the proposed model reproduces direction-rule effects, channel-guidance responses, and non-monotonic trade-offs among efficiency, safety, and load balance. Under the tested five-seed protocol, HCMBO achieves the best mean objective and improves over TPE-Mixed BO in four out of five paired seeds. These results suggest that the framework can serve as a simulation-based tool for analyzing crowd-routing and entrance-metering strategies, while further data calibration is required before site-specific deployment.

---

## 11. 可直接套用的 Discussion 限制说明

> The present experiments are conducted on simplified geometries designed to isolate the effects of direction constraints, anisotropic channel guidance, and entrance-rate control. They should not be interpreted as a calibrated reproduction of a specific event in the Shanghai Bund area. Site-specific deployment would require empirical calibration of inflow demand, speed-density relations, route-choice preferences, channel capacities, queue dynamics, and density observations. Moreover, the reported optimizer ranking is based on a limited number of random seeds and a fixed scalarization of multiple objectives. Different safety priorities or waiting penalties may lead to different preferred strategies. Future work will integrate real-time sensing data, explicit queue-reservoir dynamics, hard safety constraints, and online demand updating.
