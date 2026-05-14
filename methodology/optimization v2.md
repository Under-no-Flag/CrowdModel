# SA-HBO 优化方法与对比实验设计（通道方向配置版）

## 0. 研究定位

本文的策略优化问题是一个由人群疏散仿真驱动的混合变量黑箱优化问题。给定行为转移参数估计值 $\hat p$，管控策略记为

$$
z=(s,\eta)\in\mathcal Z,
$$

其中：

- $s$：离散通道方向配置；
- $\eta$：连续控制参数，例如分流比例、诱导强度、服从引导比例、限流强度或其他连续管控强度；
- $\mathcal Z$：满足工程可行性、安全规则、通道连通性和控制约束的可行策略集合。

需要特别说明：原人群动力学模型中已经使用 $s=1,\dots,S$ 表示行为阶段。为了避免论文中记号冲突，正式写作时建议将通道方向配置记为 $\sigma$ 或 $s^{\mathrm{cfg}}$。本文件为保持与你当前优化章节记号一致，仍使用 $s$ 表示通道方向配置；若涉及行为阶段，建议改用 $b$ 或 $h$ 表示行为阶段索引。

对任意给定策略 $z$，需要运行完整人群动力学仿真，并计算

$$
\tilde J_1(z\mid \hat p),\qquad
\tilde J_2(z\mid \hat p),\qquad
\tilde J_5(z\mid \hat p).
$$

在给定权重 $(\lambda_1,\lambda_2,\lambda_5)$ 下，标量化目标函数为

$$
J(z\mid \hat p)
=
\lambda_1\tilde J_1(z\mid \hat p)
+
\lambda_2\tilde J_2(z\mid \hat p)
+
\lambda_5\tilde J_5(z\mid \hat p),
$$

优化问题写为

$$
\min_{z\in\mathcal Z}J(z\mid \hat p).
$$

由于 $J(z\mid\hat p)$ 由“势场求解—密度演化—指标统计”这一完整仿真流程隐式给出，通常不可导、非凸、非光滑，且单次评价成本较高，因此不适合直接使用梯度型优化方法。本文采用 SA-HBO 作为主优化算法。

---

# 1. 离散变量 $s$ 的正确定义

## 1.1 通道方向配置

本文中的离散变量 $s$ 表示通道方向配置，而不是泛泛的策略类型标签。设共有 $C$ 条可管控通道，则

$$
s=(s_1,\dots,s_C),
\qquad
s_c\in\{-1,0,+1,\varnothing\}.
$$

其中：

| 取值 | 含义 |
|---:|---|
| $s_c=+1$ | 第 $c$ 条通道按参考切向正向单向通行 |
| $s_c=-1$ | 第 $c$ 条通道按参考切向反向单向通行 |
| $s_c=0$ | 第 $c$ 条通道保持双向通行 |
| $s_c=\varnothing$ | 第 $c$ 条通道完全关闭 |

因此，若不考虑额外工程约束，完整离散配置空间规模为

$$
|\mathcal S|=4^C.
$$

当 $C=4$ 时，

$$
|\mathcal S|=4^4=256.
$$

这说明 $s$ 虽然是一个离散变量，但它具有明确的通道级组合结构，不应被简单视为 $1,2,\dots,256$ 这类互不相关的类别标签。

---

## 1.2 通道方向配置对 Bellman 势场求解的影响

设第 $c$ 条通道区域为 $\Omega_c$，其参考切向方向为 $\tau_c(x)$。在无通道方向管控时，基础允许方向集合记为 $U^0(x)$。给定通道方向配置 $s$ 后，通道区域内的允许方向集合可写为

$$
U^{(s)}(x)=
\begin{cases}
\{u\in U^0(x):u^\top\tau_c(x)\ge 0\}, & x\in\Omega_c,\ s_c=+1,\\[3pt]
\{u\in U^0(x):u^\top\tau_c(x)\le 0\}, & x\in\Omega_c,\ s_c=-1,\\[3pt]
U^0(x), & x\in\Omega_c,\ s_c=0,\\[3pt]
\varnothing, & x\in\Omega_c,\ s_c=\varnothing,\\[3pt]
U^0(x), & x\notin\bigcup_{c=1}^{C}\Omega_c.
\end{cases}
$$

在离散网格实现中，也可以将 $u^\top\tau_c(x)\ge0$ 和 $u^\top\tau_c(x)\le0$ 替换为基于离散邻接方向的正向/反向通行规则，或加入角度容忍阈值。

当 $s_c=\varnothing$ 时，该通道关闭。关闭状态在下层 Bellman 求解中实现为

$$
U^{(s)}(x)=\varnothing,
\qquad x\in\Omega_c.
$$

数值上可等价处理为：

1. 该区域不可达；或
2. 所有从该区域出发的步进代价设为 $+\infty$；或
3. 在图搜索/动态规划中删除该区域对应节点或边。

因此，对子群体的 Bellman 方程应理解为在由 $s$ 修改后的允许方向集合上求解：

$$
\phi_{b,r}^{(s)}(x)
=
\min_{u\in U_{b,r}^{(s)}(x)}
\left(
\phi_{b,r}^{(s)}(x+\Delta x\,u)
+
\frac{\Delta x}{f(\rho)}
\frac{1}{\sqrt{u^\top M_{b,r}(x)u}}
\right),
$$

其中 $b$ 表示行为阶段，$r$ 表示路线类型。若某处 $U_{b,r}^{(s)}(x)=\varnothing$，则该处无可行步进方向，势函数可设为 $+\infty$ 或按不可达节点处理。

---

# 2. 主优化算法：SA-HBO

## 2.1 算法基本思想

本文采用

$$
\text{SA-HBO}
=
\text{Simulated Annealing exploration}
+
\text{Hybrid Bayesian Optimization exploitation}.
$$

其中：

- **HBO** 负责建立混合变量代理模型，在有限仿真预算下提高样本效率；
- **SA** 负责在不同通道方向配置 $s$ 之间进行全局跳转，避免搜索过程过早集中于局部配置区域；
- **mixed-variable surrogate model** 负责同时处理离散通道方向变量 $s_c\in\{-1,0,+1,\varnothing\}$ 和连续变量 $\eta$。

需要强调的是，本文中的 SA 不直接替代 BO，也不应在真实目标函数 $J(z)$ 上进行大量试探。由于真实目标评价非常昂贵，SA 更适合用于候选点生成、混合邻域跳转或采集函数搜索。真正的高精度人群疏散仿真只对最终选中的候选策略执行。

---

## 2.2 为什么不能对每个完整 $s$ 都采样若干 $\eta$

若共有 $C=4$ 条通道，则完整通道方向配置数为

$$
4^4=256.
$$

若采用“对每一种完整配置 $s$ 至少采样 $2\sim3$ 个连续参数 $\eta$”的初始化方式，则仅初始阶段就需要

$$
256\times(2\sim3)=512\sim768
$$

次高精度仿真。该成本只会出现在**全配置覆盖式初始化**中，并不是 SA-HBO 必然需要的成本。对于昂贵人群仿真，本文不采用这种方式。

正确做法是：把 $s$ 视为通道级组合向量，而不是无结构类别标签。初始样本数由总仿真预算决定，而不是由 $4^C$ 决定。

设总仿真预算为

$$
B.
$$

初始样本数取

$$
n_0=\alpha B,
\qquad
\alpha\in[0.1,0.25].
$$

例如，当

$$
B=100,
$$

可以取

$$
n_0=15\sim25,
$$

而不是 $512\sim768$。

---

# 3. SA-HBO 算法结构

## 3.1 第一步：预算约束的初始实验设计

初始数据集记为

$$
\mathcal D_0
=
\{(z_i,J(z_i))\}_{i=1}^{n_0},
\qquad
z_i=(s_i,\eta_i).
$$

其中

$$
s_i=(s_{i1},\dots,s_{iC}),
\qquad
s_{ic}\in\{-1,0,+1,\varnothing\}.
$$

由于 $s$ 是完整通道方向配置，初始化的目标不是覆盖所有 $4^C$ 个组合，而是保证每条通道、每种方向状态和连续控制参数空间具有基本代表性。

推荐初始样本由以下部分组成。

| 初始样本来源 | 建议数量 | 作用 |
|---|---:|---|
| 工程基准配置 | 3–5 | 包括 no-control、现行配置、全双向配置、安全优先配置等 |
| 正交设计或分数因子设计 | 10–20 | 使每条通道的 $-1,0,+1,\varnothing$ 状态得到基本覆盖 |
| 随机组合配置 | 5–10 | 增加探索性，避免初始样本过于规则 |
| 连续参数采样 | 与上述 $s$ 配对 | 使用 LHS、Sobol 或均匀采样生成 $\eta$ |

若 no-control 对应“所有通道保持双向”，则可写为

$$
s^{\mathrm{base}}=(0,0,\dots,0).
$$

若实际工程现状中部分通道本来就是单向或关闭，则 no-control 应取现行通道方向配置，而不是强行设为全双向。

若 $\eta$ 是分流比例向量，例如

$$
\eta=(\eta_1,\dots,\eta_m),
\qquad
\sum_{j=1}^{m}\eta_j=1,
\qquad
\eta_j\ge0,
$$

则初始采样应在 simplex 上进行。可以先生成随机向量，再归一化；也可以使用 Dirichlet 分布或 simplex LHS。

---

## 3.2 第二步：建立混合变量代理模型

由于 $s$ 是完整通道方向配置，主方法不建议对每个完整 $s$ 单独建立代理模型。否则当 $|\mathcal S|=4^C$ 较大时，每个子模型几乎没有足够数据。

主方法采用联合混合变量代理模型：

$$
\widehat J(z)=\widehat J(s,\eta).
$$

### 3.2.1 结构感知离散核函数

为了让代理模型利用相似通道配置之间的信息共享，可定义通道方向配置之间的距离。给定

$$
s=(s_1,\dots,s_C),
\qquad
s'=(s_1',\dots,s_C'),
$$

一种简单选择是加权 Hamming 距离：

$$
d_{\mathcal S}(s,s')
=
\sum_{c=1}^{C}w_c\mathbf 1(s_c\ne s_c'),
$$

其中 $w_c$ 表示第 $c$ 条通道的重要性权重。主通道、关键瓶颈通道或靠近出口的通道可赋予更高权重。

也可以进一步区分方向状态之间的差异。例如，将关闭状态与通行状态的差异设得更大：

$$
d_c(s_c,s_c')=
\begin{cases}
0, & s_c=s_c',\\
\delta_{\mathrm{dir}}, & s_c,s_c'\in\{-1,0,+1\},\ s_c\ne s_c',\\
\delta_{\mathrm{close}}, & \text{one of }s_c,s_c'\text{ equals }\varnothing,
\end{cases}
$$

其中通常取

$$
\delta_{\mathrm{close}}>\delta_{\mathrm{dir}}>0.
$$

此时

$$
d_{\mathcal S}(s,s')
=
\sum_{c=1}^{C}w_c d_c(s_c,s_c').
$$

离散核函数可定义为

$$
k_{\mathcal S}(s,s')
=
\exp\left(-\frac{d_{\mathcal S}(s,s')}{\ell_s}\right).
$$

### 3.2.2 连续变量核函数

连续变量 $\eta$ 可使用 Matérn 核或 RBF 核：

$$
k_\eta(\eta,\eta').
$$

若 $\eta$ 是 simplex 上的分流比例向量，可以直接在欧氏空间中建模，也可以使用 log-ratio transformation 后建模。

### 3.2.3 混合核函数

最终混合核函数可以写为

$$
k(z,z')=k_{\mathcal S}(s,s')k_{\eta}(\eta,\eta'),
\qquad
z=(s,\eta),\quad z'=(s',\eta').
$$

这样，代理模型可以利用不同通道方向配置之间的结构相似性，而不是把 $4^C$ 个配置当作完全无关的类别标签。

若 GP 在高维离散空间中不稳定，也可以使用随机森林、TPE 或 SMAC 类代理模型作为替代。论文主方法可采用结构感知 GP，强基线可采用 TPE 或 SMAC。

---

## 3.3 第三步：采集函数选择

本文为单目标标量化最小化问题，可以使用

$$
\mathrm{EI}(z),
\qquad
\mathrm{UCB}(z),
\qquad
\mathrm{LCB}(z).
$$

由于目标是最小化，本文采用 Expected Improvement。设当前已观测最优值为

$$
J_{\min}=\min_{(z_i,J_i)\in\mathcal D_n}J_i.
$$

在代理模型给出预测分布

$$
J(z)\sim\mathcal N(\mu_n(z),\sigma_n^2(z))
$$

时，Expected Improvement 定义为

$$
\alpha_{\mathrm{EI}}(z)
=
\mathbb E\left[\max(J_{\min}-J(z),0)\right].
$$

若 $\sigma_n(z)>0$，其闭式形式为

$$
\alpha_{\mathrm{EI}}(z)
=
(J_{\min}-\mu_n(z))\Phi(\gamma(z))
+
\sigma_n(z)\phi(\gamma(z)),
$$

其中

$$
\gamma(z)=\frac{J_{\min}-\mu_n(z)}{\sigma_n(z)},
$$

$\Phi(\cdot)$ 和 $\phi(\cdot)$ 分别为标准正态分布函数和密度函数。

EI 同时考虑 exploitation 和 exploration：预测均值越小，说明该点潜在目标值越好；预测方差越大，说明该点不确定性越高，具有探索价值。

---

## 3.4 第四步：模拟退火辅助候选生成

SA 部分不直接替代 BO，而作为混合变量候选生成或采集函数搜索机制。

在第 $n$ 次迭代中，基于当前数据集 $\mathcal D_n$ 建立代理模型，并得到采集函数 $\alpha_n(z)$。然后生成候选点集合

$$
\mathcal C_n=\{z_{\mathrm{BO}},z_{\mathrm{SA}},z_{\mathrm{rand}}\}.
$$

### BO 候选点

$$
z_{\mathrm{BO}}
=
\arg\max_{z\in\mathcal Z}\alpha_n(z).
$$

该候选点由常规采集函数优化得到。

### SA 候选点

$$
z_{\mathrm{SA}}
\approx
\arg\max_{z\in\mathcal Z}\alpha_n(z),
$$

但其搜索过程使用模拟退火在混合变量空间中进行邻域跳转。

从当前候选 $z=(s,\eta)$ 出发，生成邻域点

$$
z'=(s',\eta').
$$

其中：

- $s'$ 由通道方向配置邻域生成；
- $\eta'=\eta+\epsilon$，扰动幅度随温度 $T_k$ 缩小；
- 若 $\eta$ 是分流比例向量，则扰动后投影回 simplex：

$$
\eta'=\Pi_{\Delta}(\eta+\epsilon).
$$

SA 接受概率基于采集函数，而不基于真实仿真目标：

$$
P_{\mathrm{accept}}
=
\min\left\{1,
\exp\left(\frac{\alpha_n(z')-\alpha_n(z)}{T_k}\right)
\right\}.
$$

当 $\alpha_n(z')\ge \alpha_n(z)$ 时，候选点直接接受；当 $\alpha_n(z')<\alpha_n(z)$ 时，仍以一定概率接受，从而保留跨区域探索能力。

温度更新可采用

$$
T_{k+1}=\gamma T_k,
\qquad
0<\gamma<1.
$$

### 随机候选点

$$
z_{\mathrm{rand}}
\sim \mathrm{Uniform}(\mathcal Z),
$$

或由混合 LHS/Sobol 生成，用于维持全局探索能力。

最终可选择

$$
z_{n+1}
=
\arg\max_{z\in\mathcal C_n}\alpha_n(z).
$$

若采用批量并行仿真，也可以从 $\mathcal C_n$ 中选择多个采集函数较高且彼此差异较大的候选点进行评价。

---

## 3.5 通道方向配置邻域设计

由于

$$
s=(s_1,\dots,s_C),
\qquad
s_c\in\{-1,0,+1,\varnothing\},
$$

邻域不应定义为随机替换一个 $1\sim4^C$ 的完整配置标签，而应定义为通道级方向扰动。

### 单通道方向替换

随机选择一条通道 $c$，将其状态替换为另一种可行状态：

$$
s_c\rightarrow s_c',
\qquad
s_c'\in\{-1,0,+1,\varnothing\}\setminus\{s_c\}.
$$

例如：

$$
0\rightarrow +1,
\qquad
+1\rightarrow -1,
\qquad
0\rightarrow\varnothing.
$$

### 单通道方向反转

当通道当前为单向时，可以进行方向反转：

$$
+1\leftrightarrow -1.
$$

该操作适合描述临时改变通道方向的管控措施。

### 双向与单向切换

可将双向通道改为正向或反向单向，也可将单向通道恢复为双向：

$$
0\leftrightarrow +1,
\qquad
0\leftrightarrow -1.
$$

### 通道关闭与恢复

可将通道关闭，或将关闭通道恢复为某种通行状态：

$$
0,+1,-1\rightarrow\varnothing,
\qquad
\varnothing\rightarrow0,+1,-1.
$$

为了避免生成明显不可行的策略，若某次关闭导致出入口不连通或可通行能力低于最低需求，则该邻域点应被 repair 或直接判为不可行。

### 多通道小幅扰动

在较高温度阶段，可以以较小概率同时改变两条通道状态：

$$
(s_i,s_j)\rightarrow(s_i',s_j').
$$

该操作可以增强全局探索能力。随着温度降低，应减少多通道扰动比例，使搜索逐渐转向局部精修。

### 连续控制参数扰动

连续参数采用

$$
\eta' = \Pi_{\mathcal H_s}(\eta+\epsilon),
$$

其中 $\Pi_{\mathcal H_s}$ 表示投影到当前通道方向配置 $s$ 对应的连续可行域。

若 $\eta$ 是分流比例向量：

$$
\eta=(\eta_1,\dots,\eta_m),
\qquad
\sum_{j=1}^{m}\eta_j=1,
\qquad
\eta_j\ge0,
$$

则采用 simplex 投影：

$$
\eta' = \Pi_{\Delta_m}(\eta+\epsilon).
$$

这种邻域设计具有明确工程解释性，也可以避免产生大量无意义的完整配置跳转。

---

## 3.6 第五步：真实仿真评价与数据更新

对选出的候选策略 $z_{n+1}$ 运行完整人群疏散仿真，得到原始指标：

$$
J_1(z_{n+1}),
\qquad
J_2(z_{n+1}),
\qquad
J_5(z_{n+1}).
$$

然后计算标准化指标：

$$
\tilde J_1(z_{n+1}),
\qquad
\tilde J_2(z_{n+1}),
\qquad
\tilde J_5(z_{n+1}).
$$

最后计算标量目标：

$$
J(z_{n+1})
=
\lambda_1\tilde J_1(z_{n+1})
+
\lambda_2\tilde J_2(z_{n+1})
+
\lambda_5\tilde J_5(z_{n+1}).
$$

更新数据集：

$$
\mathcal D_{n+1}
=
\mathcal D_n\cup\{(z_{n+1},J(z_{n+1}))\}.
$$

若需要比较多组权重，则应保存完整指标库：

$$
\mathcal D
=
\{z_i,\tilde J_1(z_i),\tilde J_2(z_i),\tilde J_5(z_i)\}_{i=1}^{N}.
$$

由于不同权重只改变加权方式，不改变单个策略的仿真结果，因此同一批仿真数据可以被多组权重重复使用。

---

## 3.7 停止条件

算法可在以下条件之一满足时停止：

1. 达到最大仿真预算 $B$；
2. 连续若干轮 best-so-far 目标值无明显改善；
3. 代理模型推荐点高度重复，说明搜索趋于稳定；
4. 达到预设工程性能阈值。

最终输出

$$
z^*=
\arg\min_{z_i\in\mathcal D_N}J(z_i).
$$

同时报告该策略对应的原始指标和标准化指标。

---

# 4. SA-HBO 伪代码

```text
Input:
    Feasible strategy space Z
    Channel number C
    Channel direction states {-1, 0, +1, empty}
    Simulation budget B
    Initial sample size n0
    Weights (lambda_1, lambda_2, lambda_5)
    Initial temperature T0
    Cooling factor gamma

Output:
    Best strategy z*
    Evaluation database D

1. Construct an initial design D0:
       Represent each discrete configuration as
           s = (s_1, ..., s_C), s_c in {-1, 0, +1, empty}.
       Use engineering baseline configurations, orthogonal/fractional
       factorial design, and random configurations.
       Sample eta by LHS, Sobol, uniform sampling, or simplex sampling.

2. For each z_i = (s_i, eta_i) in D0:
       Apply s_i to the lower-level Bellman solver by modifying
       the channel-wise feasible direction sets U^(s_i)(x).
       Run the crowd simulation.
       Compute J1, J2, J5.
       Compute normalized indicators J1_tilde, J2_tilde, J5_tilde.
       Compute scalar objective J(z_i).

3. Set D = D0.

4. For n = n0, ..., B-1:
       Fit a mixed-variable surrogate model using D.
       Construct the acquisition function alpha_n(z), e.g. EI.

       Generate candidate z_BO by maximizing alpha_n(z)
       using a standard acquisition optimizer.

       Generate candidate z_SA by simulated annealing search
       over the mixed-variable space:
           - propose channel-level direction moves for s;
           - perturb continuous eta;
           - project eta back to the feasible set if necessary;
           - reject or repair infeasible channel configurations;
           - accept or reject candidates according to alpha_n(z),
             not according to the expensive true objective J(z).

       Generate optional random candidate z_rand.

       Select next evaluation point:
           z_{n+1} = argmax alpha_n(z) over the candidate set.

       Run the full crowd simulation at z_{n+1}.
       Compute J1, J2, J5 and their normalized forms.
       Compute scalar objective J(z_{n+1}).
       Update D = D union {(z_{n+1}, J(z_{n+1}))}.

5. Return:
       z* = argmin_{z_i in D} J(z_i).
```

---

# 5. 关键计算成本说明

对于 $C$ 条通道，每条通道有 4 种方向状态：

$$
s_c\in\{-1,0,+1,\varnothing\}.
$$

完整配置数为

$$
4^C.
$$

当 $C=4$ 时，完整配置数为

$$
4^4=256.
$$

“对每个完整 $s$ 采样 $2\sim3$ 个 $\eta$”将导致

$$
512\sim768
$$

次初始仿真。该成本只会出现在全配置覆盖式初始化中，并不是 SA-HBO 必然需要的成本。

本文采用预算约束设计后，初始仿真次数为

$$
n_0=\alpha B.
$$

例如

$$
B=100,
\qquad
n_0=20,
\qquad
n_{\mathrm{seq}}=80.
$$

此时总高精度仿真次数为 100，而不是 512–768。SA 在算法中主要搜索采集函数，因此其内部尝试的大量邻域点不需要运行真实仿真。

---

# 6. 对比算法设计

为了满足高质量学术期刊对算法实验的要求，应设置工程基线、无信息搜索基线、经典无导数优化算法、模型辅助优化算法和本文方法之间的系统比较。

## 6.1 工程基线

### Baseline 1：No-control / current practice

No-control 表示不施加额外管控或采用现有默认通道方向配置。若默认状态为全部双向，则

$$
s^{\mathrm{base}}=(0,0,\dots,0).
$$

若现实场景中已有固定单向通道、关闭通道或临时通行规则，则应以实际现行配置作为 $s^{\mathrm{base}}$。

该基线用于回答：优化方法相对现有管理方式是否带来实际改善。

---

## 6.2 无信息搜索基线

### Baseline 2：Random Search / LHS

Random Search 或 Latin Hypercube Sampling 在相同仿真预算 $B$ 下随机生成策略并评价。

离散部分可从可行方向配置集合 $\mathcal S_{\mathrm{fea}}$ 中随机采样：

$$
s\sim\mathrm{Uniform}(\mathcal S_{\mathrm{fea}}),
$$

连续部分 $\eta$ 使用 LHS、Sobol、均匀采样或 simplex 采样。

该基线用于证明 SA-HBO 不是单纯依靠“多试几个策略”取得改进，而是有效利用了历史仿真结果和代理模型。

---

## 6.3 经典无导数优化算法

### Baseline 3：Simulated Annealing, SA

由于主算法包含 SA 机制，因此必须比较

$$
\text{SA-HBO}\quad\text{vs.}\quad\text{SA only}.
$$

SA-only 直接在真实目标 $J(z)$ 上进行邻域搜索。每评价一个候选策略，都需要运行真实仿真。

SA-only 使用与 SA-HBO 相同的通道方向邻域，例如：

$$
s_c\rightarrow s_c',
\qquad
s_c'\in\{-1,0,+1,\varnothing\},
$$

并同步扰动 $\eta$。

该对比用于回答：HBO 代理模型是否显著提高样本效率。

公平性要求：

$$
B_{\mathrm{SA}}=B_{\mathrm{SA-HBO}}.
$$

### Baseline 4：Genetic Algorithm, GA

当 $\eta$ 维度较低，且 $s$ 可以清晰编码时，GA 是合适的经典无导数优化基线。

编码方式可以为

$$
z=(s_1,\dots,s_C,\eta),
\qquad
s_c\in\{-1,0,+1,\varnothing\}.
$$

其中：

- $s_c$ 使用整数编码或逐通道 one-hot 编码；
- $\eta$ 使用实数编码；
- 交叉和变异后若违反连通性、最小通行能力或 simplex 约束，则使用 repair 或 penalty 处理。

GA 的作用是代表传统群智能或进化类全局搜索方法。

### Baseline 5：CMA-ES

CMA-ES 主要适合连续变量非凸优化，不天然适合直接处理 $s_c\in\{-1,0,+1,\varnothing\}$ 这类离散方向配置。因此，当 $s$ 是完整通道方向配置时，不建议将 CMA-ES 作为完整混合变量优化的主基线。

更合理的使用方式是：固定若干工程上重要的通道方向配置 $s$，仅用 CMA-ES 优化连续变量 $\eta$：

$$
\eta_s^*=
\arg\min_{\eta}J(s,\eta).
$$

然后在这些固定配置之间比较：

$$
z^*=
\arg\min_s J(s,\eta_s^*).
$$

可选固定配置包括：

1. no-control 配置；
2. 当前工程配置；
3. 全双向配置；
4. 安全优先配置；
5. 由初步筛选得到的若干高潜力配置。

若对 $4^C$ 个完整配置逐一运行 CMA-ES，则成本会过高，不建议作为主实验。

---

## 6.4 模型辅助优化基线

仅与 SA、GA、Random Search 比较还不够，因为 SA-HBO 本质上是代理模型辅助优化方法。因此建议加入以下模型辅助基线。

### Baseline 6：Standard BO with one-hot encoding

将离散通道方向配置编码为 one-hot，再与连续变量 $\eta$ 拼接，使用标准 Bayesian Optimization。

有两种编码方式：

1. 完整配置 one-hot：将 $s\in\mathcal S$ 编码为 $4^C$ 维 one-hot；
2. 逐通道 one-hot：将每个 $s_c\in\{-1,0,+1,\varnothing\}$ 编码为 4 维 one-hot，总维度为 $4C$。

对于 $C=4$，完整配置 one-hot 维度为 256，逐通道 one-hot 维度为 16。为了避免过高维度，建议使用逐通道 one-hot 作为 standard BO 的较强基线。

该基线用于证明：本文的结构感知混合变量建模和通道级邻域设计是否优于直接套用普通 BO。

### Baseline 7：TPE 或 SMAC

TPE 和 SMAC 适合处理包含离散变量、连续变量和条件变量的黑箱优化问题。二者可选其一作为强模型辅助基线。

建议：

- 若实现复杂度需要控制，选择 TPE；
- 若策略空间具有较多类别变量和条件结构，选择 SMAC；
- 若存在明显黑箱约束或失败仿真，也可以考虑 NOMAD/MADS 作为附加基线。

---

## 6.5 推荐的主实验算法组合

主实验建议采用以下算法组合：

$$
\boxed{
\text{No-control}
+
\text{Random/LHS}
+
\text{SA}
+
\text{GA}
+
\text{standard BO}
+
\text{TPE/SMAC}
+
\text{SA-HBO}
}
$$


但需要明确说明 CMA-ES 的适用范围。对于完整通道方向配置问题，CMA-ES 应作为固定 $s$ 下连续参数优化的辅助基线，而不是完整混合变量优化的主基线。

---

# 7. 实验设计

## 7.1 场景设置

建议设置多个仿真实例，而不是只在单一场景中验证算法。可以采用以下结构：

$$
\text{几何场景}
\times
\text{需求强度}
\times
\text{行为参数情形}.
$$

例如：

| 维度 | 建议设置 |
|---|---|
| 几何场景 | 简单瓶颈场景、中等复杂多通道场景、真实或准真实场馆场景 |
| 需求强度 | low、medium、high |
| 行为情形 | nominal、biased、uncertain |

若计算资源允许，可以构造

$$
3\times3\times3=27
$$

个问题实例。若仿真成本较高，可以减少为

$$
2\times3\times3=18
$$

个问题实例。至少应避免只使用一个实例进行算法比较。

---

## 7.2 权重组合设置

由于目标函数为

$$
J
=
\lambda_1\tilde J_1
+
\lambda_2\tilde J_2
+
\lambda_5\tilde J_5,
$$

建议设置多组权重以反映不同管理偏好。

| 权重方案 | $(\lambda_1,\lambda_2,\lambda_5)$ | 管理含义 |
|---|---:|---|
| Equal | $(1/3,1/3,1/3)$ | 效率、安全、均衡同等重要 |
| Efficiency-first | $(0.7,0.2,0.1)$ | 优先减少总体滞留 |
| Safety-first | $(0.2,0.7,0.1)$ | 优先降低高密度暴露 |
| Balance-first | $(0.2,0.2,0.6)$ | 优先均衡通道负载 |
| Safety-balance | $(0.25,0.5,0.25)$ | 偏重安全，同时控制通道过载 |

对于不同权重组合，若仿真数据库已保存

$$
\tilde J_1(z_i),
\qquad
\tilde J_2(z_i),
\qquad
\tilde J_5(z_i),
$$

则无需重复仿真同一策略，只需要重新计算加权目标即可。

---

## 7.3 仿真预算与重复次数

所有优化算法必须使用相同高精度仿真预算。

例如：

$$
B\in\{50,100,150\}.
$$

若仿真成本较高，可以使用：

$$
B\in\{30,60,100\}.
$$

每个算法在每个实例上应进行多随机种子重复，建议：

$$
R=20.
$$

若计算资源允许，可取

$$
R=30.
$$

每次运行应记录 best-so-far 曲线：

$$
b\mapsto\min_{1\le i\le b}J(z_i),
\qquad
b=1,\dots,B.
$$

---

## 7.4 评价指标

### 优化性能指标

报告以下指标：

1. 最终最优目标值：

   $$
   J_{\min}^{(B)}=
   \min_{1\le i\le B}J(z_i).
   $$

2. best-so-far 收敛曲线：

   $$
   b\mapsto J_{\min}^{(b)}.
   $$

3. normalized regret：

   $$
   r_b=
   \frac{J_{\min}^{(b)}-J^\dagger}{|J^\dagger|+\epsilon},
   $$

   其中 $J^\dagger$ 是所有算法、所有运行中观察到的最好结果。

4. 达到目标阈值所需仿真次数。

### 工程性能指标

除标量目标外，还应报告原始物理指标：

$$
J_1,
\qquad
J_2,
\qquad
J_5.
$$

同时建议报告：

- 最后一人离开时间；
- $T_{90}$ 或 $T_{95}$，即 90% 或 95% 人群完成疏散时间；
- 最大密度 $\max_{x,t}\rho(x,t)$；
- 高密度区域持续时间；
- 各通道累计流量 $F_c$；
- 通道流量占比 $p_c=F_c/F$；
- 策略实施成本，例如关闭通道数、单向化通道数、连续诱导强度或限流强度。

---

# 8. 消融实验设计

为证明 SA-HBO 不是简单的算法拼接，需要做消融实验。

建议比较以下版本：

| 版本 | 目的 |
|---|---|
| Full SA-HBO | 完整方法 |
| HBO without SA | 验证 SA 全局跳转是否有贡献 |
| SA only | 验证代理模型是否提高样本效率 |
| Standard BO with one-hot $s$ | 验证结构感知混合变量建模是否有用 |
| SA-HBO with generic random moves | 验证通道方向邻域设计是否有用 |
| SA-HBO without structured initialization | 验证结构化初始样本是否有用 |

最关键的消融组合为：

$$
\boxed{
\text{Full SA-HBO}
\quad\text{vs.}\quad
\text{HBO without SA}
\quad\text{vs.}\quad
\text{SA only}
\quad\text{vs.}\quad
\text{standard BO}
}
$$

如果完整方法显著优于这些版本，说明算法优势来自混合变量代理模型、退火式跳转机制和结构化初始化的共同作用，而不是简单地把 SA 与 BO 拼接。

---

# 9. 鲁棒性测试

如果行为转移概率 $\hat p$、入流强度、初始密度或人群服从比例存在不确定性，应进行鲁棒性测试。

设扰动场景为

$$
\xi^{(1)},\xi^{(2)},\dots,\xi^{(N_{\mathrm{test}})}.
$$

对优化得到的策略 $z^*$，计算测试性能：

$$
\bar J_{\mathrm{test}}(z^*)
=
\frac{1}{N_{\mathrm{test}}}
\sum_{\ell=1}^{N_{\mathrm{test}}}
J(z^*\mid \xi^{(\ell)}).
$$

建议报告：

$$
\mathbb E[J],
\qquad
\operatorname{Std}[J],
\qquad
\operatorname{CVaR}_{0.9}[J],
\qquad
\Pr\left(\max_{x,t}\rho(x,t)>\rho_{\mathrm{crit}}\right).
$$

扰动类型可以包括：

1. 入流强度增加 10% 或 20%；
2. 初始密度空间分布偏移；
3. 行为转移概率 $p$ 扰动；
4. 部分人群不服从引导；
5. 某条通道容量下降；
6. 某出口临时受阻。

---

# 10. 统计检验

对于多算法、多实例实验，不应只报告平均值。建议采用：

1. Friedman test 比较多个算法整体差异；
2. Wilcoxon signed-rank test 比较 SA-HBO 与主要基线的成对差异；
3. 95% bootstrap confidence interval 报告结果不确定性；
4. 平均排名或 critical difference diagram 展示算法整体排序。

统计检验的样本单位应是“问题实例”或“实例—权重组合”，不宜简单把所有随机种子视为完全独立样本，否则可能夸大显著性。

---

# 11. 推荐图表

## 表 1：场景参数表

包括：

- 几何场景；
- 初始人数；
- 入流强度；
- 出口数量；
- 通道数量 $C$；
- 可管控通道区域 $\Omega_c$；
- 参考切向方向 $\tau_c(x)$；
- $\rho_{\mathrm{safe}}$；
- $\rho_{\max}$；
- 仿真时长 $T$。

## 表 2：算法参数表

包括：

- 算法名称；
- $s$ 的编码方式；
- $\eta$ 的采样或扰动方式；
- 初始样本数；
- 总预算；
- 关键超参数；
- 是否支持混合变量；
- 是否使用代理模型。

## 图 1：SA-HBO 流程图

建议展示：

$$
\text{initial design}
\rightarrow
\text{simulation}
\rightarrow
\text{mixed surrogate}
\rightarrow
\text{acquisition function}
\rightarrow
\text{SA candidate generation}
\rightarrow
\text{next simulation}.
$$

## 图 2：anytime convergence curves

横轴为高精度仿真次数，纵轴为 best-so-far $J$。

建议展示 median 和 interquartile range。

## 图 3：最终结果箱线图

展示不同算法最终获得的：

$$
J,
\qquad
\tilde J_1,
\qquad
\tilde J_2,
\qquad
\tilde J_5.
$$

## 图 4：典型策略密度热力图

比较：

- No-control；
- GA best；
- standard BO best；
- SA-HBO best。

## 图 5：通道累计流量分布

展示各通道

$$
F_c
$$

或

$$
p_c=F_c/F.
$$

用于说明 $\tilde J_5$ 的变化。

## 图 6：权重敏感性图

展示不同权重组合下最优策略和指标变化。

---

# 12. 论文中可使用的表述

可在方法章节中写：

> 本文优化问题是由人群动力学仿真驱动的混合变量黑箱优化问题。每次目标函数评价均需完整求解势场方程与密度演化方程，且决策变量同时包含通道方向配置 $s=(s_1,\dots,s_C)$ 与连续控制强度 $\eta$。其中 $s_c\in\{-1,0,+1,\varnothing\}$ 分别表示第 $c$ 条通道按参考切向正向单向、反向单向、保持双向和完全关闭。关闭通道在 Bellman 求解中通过令对应通道区域允许方向集合为空实现。由于目标函数非显式、不可导且仿真评价昂贵，传统梯度型优化和凸优化方法并不适用。本文提出 SA-HBO 框架，将模拟退火的全局跳转能力与 Bayesian optimization 的样本效率相结合，用于在有限仿真预算下搜索较优通道方向与连续控制策略。

还可进一步写：

> 当离散变量 $s$ 表示完整通道方向组合时，对每一种完整配置逐一初始化会导致不可接受的仿真成本。本文将 $s$ 保持为通道级方向向量，并采用预算约束的结构化初始设计，使初始样本覆盖关键通道状态而非覆盖全部组合配置。后续通过结构感知代理模型在相似配置之间共享信息，并利用退火式采集函数搜索机制自适应选择高价值仿真样本。

---

# 13. 最终建议

本文主算法和实验设计建议概括为：

$$
\boxed{
\text{SA-HBO}
=
\text{budgeted structured initialization}
+
\text{mixed-variable surrogate modeling}
+
\text{annealed candidate generation}
+
\text{simulation-based objective evaluation}
}
$$

由于 $s$ 是通道方向配置

$$
s=(s_1,\dots,s_C),
\qquad
s_c\in\{-1,0,+1,\varnothing\},
$$

不应对每个完整 $s$ 都采样若干 $\eta$，而应采用通道级编码、结构化初始设计和联合混合变量代理模型。

推荐主实验算法组合为：

$$
\boxed{
\text{No-control}
+
\text{Random/LHS}
+
\text{SA}
+
\text{GA}
+
\text{standard BO}
+
\text{TPE/SMAC}
+
\text{SA-HBO}
}
$$

CMA-ES 可以作为固定通道方向配置下连续参数优化的辅助基线，但不建议作为完整混合变量优化的主基线。
