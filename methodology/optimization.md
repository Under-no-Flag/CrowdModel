# 优化

## 优化变量

本文区分“可控变量”和“行为参数”。

### 可控变量（当前论文实际优化）
1. **通道方向配置**
   $$
   s=(s_1,\dots,s_C),\qquad s_c\in\{-1,0,+1\},
   $$
   其中 $s_c=+1$、$-1$、$0$ 分别表示第 $c$ 条通道按参考切向正向单向、反向单向或保持双向。

2. **几何引导强度**
   为避免同时优化 $\alpha,\beta$ 带来的尺度耦合，本文只优化各向异性比值：
   $$
   M_c(x;\eta_c)=\beta_c\Big(\eta_c\,\tau_c\tau_c^\top + n_c n_c^\top\Big),\qquad \eta_c\ge 1,
   $$
   即
   $$
   \eta_c=\alpha_c/\beta_c.
   $$

### 外生行为参数（当前论文固定，不直接优化）
1. **分流偏好参数**
   $$
   p=\hat p,
   $$
   由历史数据识别得到，在优化过程中固定。它描述游客在阶段切换时的路线偏好，属于行为参数而非直接管控变量。

2. **决策区域位置参数**
   本文当前版本不将 $\xi$ 纳入优化，仅保留为后续扩展接口。

因此，当前论文中的实际优化变量写为
$$
z=(s,\eta).
$$

---

## 1. 问题定义

基于现有 Bellman–守恒律耦合人群模型，本文在给定几何区域 $\Omega$、固定行为阶段结构以及固定偏好参数 $\hat p$ 的条件下，对可控管控变量 $(s,\eta)$ 进行优化。现有模型中，每个“阶段–路线”子群体 $(\sigma,r)$ 具有独立的势函数 $\phi_{\sigma,r}$、最优方向 $u^*_{\sigma,r}$ 与速度场 $\mathbf v_{\sigma,r}$，而拥堵项由总密度
$$
\rho(x,t)=\sum_{\sigma=1}^{S}\sum_{r=1}^{R_\sigma}\rho_{\sigma,r}(x,t)
$$
统一决定；势函数通过离散 Bellman 方程计算，密度通过显式守恒格式推进。

在此基础上，本文考虑如下**固定权重下的标量化优化问题**：
$$
\min_{z\in\mathcal Z} J(z\mid \hat p)
=
\lambda_1 J_1(z\mid \hat p)+\lambda_2 J_2(z\mid \hat p)+\lambda_5 J_5(z\mid \hat p),
$$
其中：
- $J_1$：总旅行时间；
- $J_2$：高密度暴露时间；
- $J_5$：通道累计流量方差。

需要强调的是，本文算法针对的是**多目标管控问题在一组给定权重下的标量化求解**，而不是直接输出完整 Pareto 前沿的多目标算法。若需要近似 Pareto 集，可在不同权重组合 $(\lambda_1,\lambda_2,\lambda_5)$ 下重复运行本文算法。

---

## 2. 下层评估器

对任意给定控制变量 $z=(s,\eta)$，定义下层仿真算子
$$
(\rho,\phi,\mathbf v)=\mathcal S(z;\hat p),
$$
其计算过程如下。

首先，根据 $z$ 构造方向允许集合 $U_{\sigma,r}(x;z)$ 与度量张量 $M_{\sigma,r}(x;z)$，并固定使用由历史数据识别得到的偏好参数 $\hat p$。然后，对每个子群体 $(\sigma,r)$ 求解离散 Bellman 方程
$$
\phi_{\sigma,r}(x)=
\min_{u\in U_{\sigma,r}(x;z)}
\left(
\phi_{\sigma,r}(x+\Delta x\,u)
+
\frac{\Delta x}{f(\rho)}\frac{1}{\sqrt{u^\top M_{\sigma,r}(x;z)u}}
\right),
$$
再由
$$
u_{\sigma,r}^*(x)=
\operatorname*{arg\,min}_{u\in U_{\sigma,r}(x;z)}
\left(
\phi_{\sigma,r}(x+\Delta x\,u)
+
\frac{\Delta x}{f(\rho)}\frac{1}{\sqrt{u^\top M_{\sigma,r}(x;z)u}}
\right)
$$
恢复最优方向，并得到速度场
$$
\mathbf v_{\sigma,r}(x)=f(\rho)\,u^*_{\sigma,r}(x).
$$

最后，利用显式守恒格式与固定的阶段转移/分流参数 $\hat p$ 推进各子群体密度：
$$
\frac{\partial \rho_{\sigma,r}}{\partial t}
+
\nabla\cdot(\rho_{\sigma,r}\mathbf v_{\sigma,r})
=
Q^{\text{in}}_{\sigma,r}(\hat p)-Q^{\text{out}}_{\sigma,r}(\hat p).
$$

因此，优化算法不改动下层模型本身，而是将现有仿真器视为结构化评估器。

---

## 3. SA-HBO 的定位：结构感知框架及其精简实例

直接将该问题视为黑箱优化会忽略三个关键结构：

1. 控制变量具有“离散方向配置 + 连续引导强度”的混合特征；
2. Bellman 势场对 $U(x)$ 和 $M(x)$ 的变化具有明显的局部响应特征；
3. 下层模型采用“共享拥堵、分离势场”的多子群体结构，单次全仿真代价高。

基于此，本文提出一种**结构感知的混合分块优化框架**
**Structure-Aware Hybrid Block Optimization (SA-HBO)**。需要特别说明：

- **SA-HBO 是一个面向 Bellman–守恒律耦合模型的结构感知优化框架；**
- **当前论文实现的是该框架在 $z=(s,\eta)$、$p=\hat p$ 固定条件下的精简实例。**

当前实现包含四个核心组件：

1. **离散块邻域搜索**：对通道方向配置 $s$ 进行局部搜索，而非全枚举；
2. **代理预筛选**：对邻域候选先做局部代理评价，仅将少量候选送入全仿真；
3. **Bellman 热启动与局部重求解**：复用上一轮势场，降低候选评估成本；
4. **连续块投影随机近似更新**：对 $\eta$ 做低成本扰动搜索，并加入接受/回退机制保证流程闭合。

因此，本文不将 SA-HBO 表述为通用元启发式算法，而将其表述为：
**利用现有 PDE 结构、数值接口和局部响应特征而构造的结构感知优化框架。**

---

## 4. 离散块更新：通道方向配置的邻域搜索

固定连续变量 $\eta^k$，第 $k$ 次迭代的离散子问题为
$$
s^{k+1/2}
\approx
\arg\min_{s\in\mathcal S}
J(s,\eta^k\mid \hat p).
$$

考虑到通道方向变量 $s_c$ 仅会改变对应通道区域内的允许方向集合 $U(x)$，从而局部影响 Bellman 步进结构，而不会改变整个区域的数学形式，本文采用邻域搜索而非全局枚举。定义当前方向配置 $s^k$ 的邻域为
$$
\mathcal N(s^k)
=
\left\{
s:\ \|s-s^k\|_0\le \nu
\right\},
$$
其中 $\nu$ 为允许同时翻转的通道数，通常取 $1$ 或 $2$。

### 4.1 代理预筛选

为减少全仿真次数，对每个候选 $s'\in\mathcal N(s^k)$ 构造局部代理评价：
$$
\widehat J(s',\eta^k)
=
\mu_1 \widehat J_1^{\text{local}}
+
\mu_2 \widehat J_2^{\text{local}}
+
\mu_5 \widehat J_5^{\text{flow}},
$$
其中：
- $\widehat J_1^{\text{local}}$ 表示候选通道邻域内的密度时间累计变化；
- $\widehat J_2^{\text{local}}$ 表示邻域内超阈值暴露变化；
- $\widehat J_5^{\text{flow}}$ 表示与相关通道流量再分配有关的局部代理量。

需要强调：**代理目标仅用于预筛选，不作为最终接受准则。**
对代理值最优的前 $K$ 个候选组成集合 $\mathcal C_K(s^k)\subset\mathcal N(s^k)$，再对其执行全量 PDE 仿真，并以真实目标值
$$
J_{\text{full}}(s,\eta^k\mid \hat p)
$$
进行最终比较：
$$
s^{k+1/2}
=
\arg\min_{s\in\mathcal C_K(s^k)}J_{\text{full}}(s,\eta^k\mid \hat p).
$$

### 4.2 代理有效性报告

为说明代理筛选不会系统性误导搜索，实验中应报告代理排序与全仿真排序的一致性，例如：
- Top-$K$ 命中率；
- Kendall $\tau$ 或 Spearman 相关系数；
- 真实最优候选是否被代理保留在 $\mathcal C_K$ 中。

这一步是算法完整性的重要组成部分。

---

## 5. 连续块更新：几何引导强度的投影随机近似

固定离散变量 $s^{k+1/2}$，对连续变量 $\eta$ 求解
$$
\eta^{k+1}
\approx
\arg\min_{\eta\in\mathcal H} J(s^{k+1/2},\eta\mid \hat p),
$$
其中
$$
\mathcal H=\{\eta:\ \eta_c\ge 1,\ c=1,\dots,C\}.
$$

由于 $J$ 通过 Bellman 求解器与显式守恒格式间接依赖 $\eta$，难以获得解析梯度，因此本文采用投影随机近似更新。第 $k$ 次迭代中生成 Rademacher 扰动向量
$$
\Delta_\eta^k\in\{-1,+1\}^{C},
$$
并构造两侧扰动：
$$
\eta_+^k=\eta^k+c_k\Delta_\eta^k,\qquad
\eta_-^k=\eta^k-c_k\Delta_\eta^k.
$$
将其投影回可行域：
$$
\tilde\eta_\pm^k=\Pi_{\mathcal H}(\eta_\pm^k).
$$

对应的随机近似梯度为
$$
\widehat g_\eta^k
=
\frac{
J(s^{k+1/2},\tilde\eta_+^k\mid \hat p)
-
J(s^{k+1/2},\tilde\eta_-^k\mid \hat p)
}{2c_k}
\odot (\Delta_\eta^k)^{-1}.
$$

给出候选更新：
$$
\eta_{\text{cand}}^{k+1}
=
\Pi_{\mathcal H}
\left(
\eta^k-\alpha_k D_\eta^{-1}\widehat g_\eta^k
\right),
$$
其中 $D_\eta$ 为对角预条件矩阵。

### 5.1 接受/回退机制

为保证算法流程闭合，连续块更新不默认接受，而采用如下准则。

先计算候选目标值
$$
J_{\text{cand}}=J(s^{k+1/2},\eta_{\text{cand}}^{k+1}\mid \hat p),
$$
并与当前目标值
$$
J_{\text{cur}}=J(s^{k+1/2},\eta^k\mid \hat p)
$$
比较。

- 若
  $$
  J_{\text{cand}}\le J_{\text{cur}}-\varepsilon_{\text{acc}},
  $$
  则接受更新，令
  $$
  \eta^{k+1}=\eta_{\text{cand}}^{k+1}.
  $$

- 若上述条件不满足，则执行回退。可采用两种等价实现：
  1. **直接回退**：令
     $$
     \eta^{k+1}=\eta^k;
     $$
  2. **回溯缩步**：将步长缩小为 $\alpha_k\leftarrow \gamma\alpha_k$（$0<\gamma<1$），重新生成候选，直到满足接受条件或达到最大回溯次数。

在实现中，同时维护历史最好解
$$
z_{\text{best}}=(s_{\text{best}},\eta_{\text{best}})
$$
及其目标值 $J_{\text{best}}$，保证即使中间发生非单调更新，最终输出仍为历史最优解。

---

## 6. Bellman 热启动与局部重求解

在现有模型中，势函数通过离散 Bellman 方程求解，而控制变量变化通常只会影响局部区域内的 $U_{\sigma,r}(x)$ 或 $M_{\sigma,r}(x)$。因此，若每次候选解评估都从头全域求解势场，则计算代价过高。为此，本文引入 Bellman 热启动与局部重求解策略。

设第 $k$ 次迭代已得到势函数 $\phi_{\sigma,r}^k$。从 $z^k$ 更新到候选控制 $\tilde z$ 后，识别受影响区域
$$
\Omega_{\mathrm{chg}}^{k}
=
\left\{
x\in\Omega:\
U_{\sigma,r}(x;\tilde z)\neq U_{\sigma,r}(x;z^k)
\ \text{or}\
M_{\sigma,r}(x;\tilde z)\neq M_{\sigma,r}(x;z^k)
\right\}.
$$

以旧势场为初值：
$$
\phi_{\sigma,r}^{k,0}(x)=\phi_{\sigma,r}^{k}(x),
$$
仅在扩张邻域 $\Omega_{\mathrm{chg}}^{k,\delta}$ 内迭代：
$$
\phi_{\sigma,r}^{k,m+1}(x)
=
\min_{u\in U_{\sigma,r}(x;\tilde z)}
\left(
\phi_{\sigma,r}^{k,m}(x+\Delta x u)
+
\frac{\Delta x}{f(\rho)}\frac{1}{\sqrt{u^\top M_{\sigma,r}(x;\tilde z)u}}
\right),
\quad x\in\Omega_{\mathrm{chg}}^{k,\delta}.
$$
当
$$
\max_{x\in\Omega_{\mathrm{chg}}^{k,\delta}}
\left|
\phi_{\sigma,r}^{k,m+1}(x)-\phi_{\sigma,r}^{k,m}(x)
\right|
\le \varepsilon_\phi
$$
时停止局部更新。

需要说明的是，局部重求解主要用于**候选评估加速**。在实际实现中，可每隔若干轮迭代或在接受较大更新后执行一次全域校正，以抑制由长期局部近似带来的误差累积。

---

## 7. 算法 1：当前论文实现的精简版 SA-HBO

### 输入
- 初始控制 $z^0=(s^0,\eta^0)$；
- 固定偏好参数 $\hat p$；
- 固定权重 $(\lambda_1,\lambda_2,\lambda_5)$；
- 最大迭代次数 $K_{\max}$；
- 步长序列 $\{\alpha_k\},\{c_k\}$；
- 邻域半径 $\nu$；
- 代理保留数 $K$。

### 输出
- 历史最好控制变量 $z_{\mathrm{best}}=(s_{\mathrm{best}},\eta_{\mathrm{best}})$。

### 步骤
1. 初始化势函数与密度场，计算
   $$
   J(z^0\mid \hat p).
   $$
   令 $z_{\mathrm{best}}=z^0$。

2. 对 $k=0,1,\dots,K_{\max}-1$ 重复：

   **离散块更新**
   1. 固定 $\eta^k$，生成离散邻域 $\mathcal N(s^k)$；
   2. 对每个候选 $s'\in\mathcal N(s^k)$ 计算代理目标 $\widehat J(s',\eta^k)$；
   3. 选取前 $K$ 个候选组成 $\mathcal C_K(s^k)$；
   4. 对 $s\in\mathcal C_K(s^k)$ 执行 Bellman 热启动局部重求解与全量密度推进，计算真实目标值 $J(s,\eta^k\mid \hat p)$；
   5. 更新
      $$
      s^{k+1/2}=\arg\min_{s\in\mathcal C_K(s^k)}J(s,\eta^k\mid \hat p).
      $$

   **连续块更新**
   6. 生成扰动 $\Delta_\eta^k$，计算两侧扰动目标值；
   7. 构造随机近似梯度 $\widehat g_\eta^k$；
   8. 生成候选更新 $\eta_{\mathrm{cand}}^{k+1}$；
   9. 依据接受/回退机制确定 $\eta^{k+1}$。

   **状态更新**
   10. 令
       $$
       z^{k+1}=(s^{k+1/2},\eta^{k+1});
       $$
   11. 计算 $J(z^{k+1}\mid \hat p)$；
   12. 若
       $$
       J(z^{k+1}\mid \hat p)<J(z_{\mathrm{best}}\mid \hat p),
       $$
       则更新历史最好解：
       $$
       z_{\mathrm{best}}\leftarrow z^{k+1}.
       $$

   **停止检查**
   13. 若满足停止条件，则退出。

3. 返回 $z^\star=z_{\mathrm{best}}$。

---

## 8. 停止准则

本文采用如下停止准则：

1. 标量化目标函数变化足够小：
   $$
   |J(z^{k+1}\mid \hat p)-J(z^k\mid \hat p)|\le \varepsilon_J;
   $$

2. 连续 $L$ 次迭代中离散变量不再变化，即
   $$
   s^{k+1}=s^k=\cdots=s^{k-L+1};
   $$

3. 或达到最大迭代次数、最大仿真次数、最大 CPU 时间限制。

在工程实现中，常将 1 和 2 联合使用，以避免过早终止。

---

## 9. 复杂度分析

设：
- 网格节点数为 $N$；
- 时间步数为 $T_n$；
- 子群体总数为
  $$
  G=\sum_{\sigma=1}^{S}R_\sigma;
  $$
- 单次全域 Bellman 求解复杂度记为 $C_B(N)$；
- 单次密度推进复杂度为 $O(GNT_n)$。

则一次完整全仿真评估的复杂度为
$$
C_{\mathrm{full}}
=
O\!\left(
G\,C_B(N)+GNT_n
\right).
$$

若直接采用通用群智能算法，对每个候选解都做全仿真，则总成本通常为
$$
O(N_{\mathrm{eval}}\,C_{\mathrm{full}}).
$$

本文算法中，离散块每次只对邻域候选做廉价代理评估，再对少量候选做真实评估。若受影响区域大小为 $N_{\mathrm{loc}}\ll N$，则局部 Bellman 更新复杂度可近似记为
$$
C_B^{\mathrm{loc}}(N_{\mathrm{loc}})
\ll
C_B(N).
$$
因此，单次离散块更新的复杂度约为
$$
O\!\left(
|\mathcal N(s^k)|\,C_{\mathrm{proxy}}
+
K\,[G\,C_B^{\mathrm{loc}}(N_{\mathrm{loc}})+GNT_n]
\right),
$$
其中 $C_{\mathrm{proxy}}$ 为代理评价成本。

连续块更新中，当前论文只对 $\eta$ 做一次双侧扰动，因此其主成本约为
$$
O(2\,C_{\mathrm{full}}),
$$
而不随 $\eta$ 维数线性增长。故当 $N_{\mathrm{loc}}\ll N$ 且通道数较多时，精简版 SA-HBO 相较于完全黑箱优化仍具有明显计算优势。

---

## 10. 方法创新点表述

本文的方法创新不在于调用某一通用元启发式方法，而在于：针对现有 Bellman–守恒律耦合人群模型的计算结构，提出了一个**结构感知的混合优化框架**，并在当前论文中实现了其针对 $(s,\eta)$ 的精简实例。具体体现在以下五个方面。

### 10.1 可控变量与行为参数的清晰分离
本文明确将通道方向配置 $s$ 和几何引导强度 $\eta$ 视为可控管控变量，而将游客路线偏好参数 $p=\hat p$ 视为由历史数据识别得到的外生行为参数，不将其直接纳入优化。这使得优化问题与实际管理权限保持一致。

### 10.2 基于 PDE 结构的离散–连续分块优化
本文不是将问题简单视为统一黑箱搜索，而是利用 $s$ 的离散性和 $\eta$ 的连续性，分别设计邻域搜索与投影随机近似更新，从而与控制变量的物理含义相匹配。

### 10.3 代理预筛选与真实评估分离
本文提出“代理仅用于预筛，真实目标决定接受”的两阶段候选评估机制，并要求在实验中报告代理排序一致率，从而提高计算效率同时保持算法决策的可靠性。

### 10.4 Bellman 热启动与局部重求解
利用单向规则和几何引导仅通过 $U(x)$ 与 $M(x)$ 局部改变 Bellman 步进代价的特点，提出势场热启动与局部重求解机制，降低候选解评估成本。

### 10.5 面向标量化多目标管控优化的求解器
本文算法针对固定权重下的标量化目标
$$
\lambda_1J_1+\lambda_2J_2+\lambda_5J_5
$$
求解，在效率、安全与均衡三类指标之间提供可调权衡。若需要近似 Pareto 前沿，可通过多组权重重复运行该框架实现。

---

## 11. 当前论文与后续扩展的边界

为避免算法声明范围超过当前论文实现范围，本文明确区分：

### 当前论文已实现
- 优化变量：$z=(s,\eta)$；
- 行为偏好：$p=\hat p$，由历史数据识别并固定；
- 目标形式：固定权重下的标量化目标；
- 算法实现：精简版 SA-HBO。

### 后续可扩展方向
- 将决策区域位置 $\xi$ 纳入连续块；
- 在更高维参数下使用更一般的分块随机近似；
- 在多组权重下重复运行，构造 Pareto 近似前沿；
- 在行为层模型升级后，引入时间变或状态依赖的偏好参数识别。

这种表述方式可以保证方法叙述、实现边界和论文贡献三者一致。
