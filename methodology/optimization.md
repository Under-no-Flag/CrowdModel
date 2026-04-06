# 优化

## 优化变量
1. 通道方向配置
2. 几何引导强度
实际优化时，不建议直接同时优化 $\alpha,\beta$，更稳的是改成：
$$M_c(x;\eta_c)=\beta_c\Big(\eta_c\,\tau_c\tau_c^\top + nn^\top\Big),\qquad \eta_c\ge 1.$$
也就是只优化各向异性比值
$$\eta_c = \alpha_c/\beta_c.$$
3. 分流概率
4. 决策区域位置


## 1. 问题定义

基于现有 Bellman–守恒律耦合人群模型，本文考虑在给定几何区域 $\Omega$ 和固定行为阶段结构下，对管控变量进行联合优化。现有模型中，每个“阶段–路线”子群体 $(s,r)$ 具有独立的势函数 $\phi_{s,r}$、最优方向 $u^*_{s,r}$ 与速度场 $\mathbf v_{s,r}$，而拥堵项由总密度
$$\rho(x,t)=\sum_{s=1}^{S}\sum_{r=1}^{R_s}\rho_{s,r}(x,t)$$
统一决定；势函数通过离散 Bellman 方程计算，密度通过显式守恒格式推进。
设优化变量为
$$z=(s,\eta,p,\xi),$$
其中：
$$s=(s_1,\dots,s_C),\qquad s_c\in\{-1,0,+1\},$$
表示第 $c$ 条通道的方向配置；
$$\eta=(\eta_1,\dots,\eta_C),\qquad \eta_c\ge 1,$$
表示通道各向异性引导强度；
$$p=\{p_{(s,r)\to q}\},
\qquad p_{(s,r)\to q}\ge 0,\ \sum_q p_{(s,r)\to q}=1,$$
表示固定概率分流参数；
$$\xi=(\xi_1,\dots,\xi_K),$$
表示决策区域位置参数。
对应的多目标优化问题写为
$$\min_{z\in\mathcal Z} J(z)
=
\lambda_1 J_1(z)+\lambda_2 J_2(z)+\lambda_5 J_5(z),$$
其中 $J_1$ 为总旅行时间，$J_2$ 为高密度暴露时间，$J_5$ 为出口负载方差。之所以采用混合变量表示，是因为你的现有模型中单向规则通过允许方向集合 $U_{s,r}(x)$ 控制，几何引导通过度量张量 $M_{s,r}(x)$ 控制，分流机制通过阶段转移项 $Q_{(s,r)\to(s+1,q)}$ 控制，因此天然形成离散–连续并存的优化结构。


## 2. 下层评估器
对任意给定控制变量 $z$，定义下层仿真算子
$$(\rho,\phi,\mathbf v)=\mathcal S(z),$$
其计算过程如下：
首先，根据 $z$ 构造方向允许集合 $U_{s,r}(x;z)$、度量张量 $M_{s,r}(x;z)$、决策区 $D_s(\xi)$ 与分流参数 $p$。然后，对每个子群体 $(s,r)$ 求解离散 Bellman 方程
$$\phi_{s,r}(x)=
\min_{u\in U_{s,r}(x;z)}
\left(
\phi_{s,r}(x+\Delta x\,u)
+
\frac{\Delta x}{f(\rho)}\frac{1}{\sqrt{u^\top M_{s,r}(x;z)u}}
\right),$$
再由
$$u_{s,r}^*(x)=
\operatorname*{arg\,min}_{u\in U_{s,r}(x;z)}
\left(
\phi_{s,r}(x+\Delta x\,u)
+
\frac{\Delta x}{f(\rho)}\frac{1}{\sqrt{u^\top M_{s,r}(x;z)u}}
\right)$$
恢复最优方向，并得到速度场
$$\mathbf v_{s,r}(x)=f(\rho)\,u^*_{s,r}(x).$$
最后，利用显式守恒格式与固定概率分流项推进各子群体密度
$$\frac{\partial \rho_{s,r}}{\partial t}
+
\nabla\cdot(\rho_{s,r}\mathbf v_{s,r})
=
Q^{\text{in}}_{s,r}-Q^{\text{out}}_{s,r}.$$
这些方程和数值接口与现有模型完全一致，因此优化算法无需改动下层模型本身。

## 3. 算法思想
直接将问题视为黑箱优化会忽略现有模型的三个重要结构：
一是控制变量的混合离散–连续特征；二是 Bellman 势场对局部管控变量变化的局部响应特征；三是多阶段–多路线子群体之间“共享拥堵、分离势场”的块结构。基于此，本文提出一种结构感知的混合分块优化算法（Structure-Aware Hybrid Block Optimization, SA-HBO），核心思想包括：

第一，将控制变量拆分为离散块与连续块：
$$z_d=s,\qquad z_c=(\eta,p,\xi).$$
第二，对离散块采用灵敏度引导的邻域搜索，避免全枚举。

第三，对连续块采用分块随机近似梯度更新，利用模型的块结构降低梯度估计噪声。

第四，在每次候选解评估时，采用Bellman 势场热启动与局部重求解，避免从头进行全域求解。

这四个组件共同构成算法创新的主体，而非简单调用通用遗传算法或粒子群算法。

## 4. 离散块更新

固定连续变量 $z_c^k$，第 $k$ 次迭代的离散子问题为
$$s^{k+1}
\approx
\arg\min_{s\in\mathcal S}
J(s,z_c^k).$$
考虑到通道方向变量 $s_c$ 仅会改变对应通道区域内的允许方向集合 $U(x)$，从而局部影响 Bellman 步进结构，而不会改变整个区域的数学形式，本文采用邻域搜索而非全局枚举。定义当前方向配置 $s^k$ 的邻域为
$$\mathcal N(s^k)
=
\left\{
s:\ \|s-s^k\|_0\le \nu
\right\},$$
其中 $\nu$ 为允许同时翻转的通道数，通常取 $1$ 或 $2$。
为减少全仿真次数，对每个候选 $s'\in\mathcal N(s^k)$ 构造局部代理评价：
$$\widehat J(s',z_c^k)
=
\mu_1 \widehat J_1^{\text{local}}
+
\mu_2 \widehat J_2^{\text{local}}
+
\mu_5 \widehat J_5^{\text{flow}},$$
其中 $\widehat J_1^{\text{local}}$ 表示候选通道邻域内的密度时间累计变化，$\widehat J_2^{\text{local}}$ 表示邻域内超阈值暴露变化，$\widehat J_5^{\text{flow}}$ 表示相关出口流量不均衡变化。选择代理值最优的前 $K$ 个候选，再做全量 PDE 仿真。于是离散更新可写成
$$s^{k+1}
=
\arg\min_{s\in\mathcal C_K(s^k)}
J_{\text{full}}(s,z_c^k),$$
其中 $\mathcal C_K(s^k)\subset\mathcal N(s^k)$ 为代理筛选后的候选集合。
这种做法的关键不在于“搜索”本身，而在于利用了现有模型中单向规则通过 $U(x)$ 的局部接口化表达

## 连续块更新

固定离散变量 $s^{k+1}$，对连续变量
$$w=(\eta,p,\xi)$$
求解
$$w^{k+1}
\approx
\arg\min_{w\in\mathcal W} J(s^{k+1},w).$$
由于 $J$ 通过 Bellman 求解器与显式守恒格式间接依赖 $w$，很难获得解析梯度，因此本文采用分块随机近似梯度。将 $w$ 进一步划分为三块：
$$w_1=\eta,\qquad w_2=p,\qquad w_3=\xi.$$
对第 $b$ 块变量，在第 $k$ 次迭代中生成 Rademacher 扰动向量
$$\Delta_b^k\in\{-1,+1\}^{d_b},$$
并构造两次对称扰动：
$$w_{b,+}^k=w_b^k+c_k\Delta_b^k,\qquad
w_{b,-}^k=w_b^k-c_k\Delta_b^k.$$
相应的梯度估计为
$$\widehat g_b^k
=
\frac{
J(s^{k+1},w_{b,+}^k,w_{-b}^k)
-
J(s^{k+1},w_{b,-}^k,w_{-b}^k)
}{2c_k}
\odot (\Delta_b^k)^{-1}.$$
然后采用投影更新：
$$w_b^{k+1}
=
\Pi_{\mathcal W_b}
\left(
w_b^k-\alpha_k D_b^{-1}\widehat g_b^k
\right),$$
其中 $D_b$ 为对角预条件矩阵，$\Pi_{\mathcal W_b}$ 表示投影到可行域。若分流概率直接优化，则使用 softmax 重新归一化：
$$p_{(s,r)\to q}^{k+1}
=
\frac{\exp(y_{(s,r)\to q}^{k+1})}
{\sum_j \exp(y_{(s,r)\to j}^{k+1})}.$$
与普通 SPSA 相比，这种按 $\eta$、$p$、$\xi$ 分块的近似梯度构造能够显著降低变量尺度差异导致的估计波动，因为现有模型中这三类变量分别作用于 $M$、$Q$ 和决策区域几何，物理含义与数值影响路径是不同的。


## 6. Bellman 热启动与局部重求解

在现有模型中，势函数通过离散 Bellman 方程求解，而控制变量变化通常只会影响局部区域内的 $U_{s,r}(x)$ 或 $M_{s,r}(x)$。因此，若每次候选解评估都从头全域求解势场，则计算代价过高。为此，本文引入 Bellman 热启动与局部重求解策略。
设第 $k$ 次迭代已得到势函数 $\phi_{s,r}^k$。从 $z^k$ 更新到 $z^{k+1}$ 后，识别受影响区域
$$\Omega_{\mathrm{chg}}^{k}
=
\left\{
x\in\Omega:\
U_{s,r}(x;z^{k+1})\neq U_{s,r}(x;z^k)
\ \text{or}\
M_{s,r}(x;z^{k+1})\neq M_{s,r}(x;z^k)
\right\}.$$
以旧势场为初值：
$$\phi_{s,r}^{k+1,0}(x)=\phi_{s,r}^k(x),$$
仅在扩张邻域 $\Omega_{\mathrm{chg}}^{k,\delta}$ 内迭代：
$$\phi_{s,r}^{k+1,m+1}(x)
=
\min_{u\in U_{s,r}(x;z^{k+1})}
\left(
\phi_{s,r}^{k+1,m}(x+\Delta x u)
+
\frac{\Delta x}{f(\rho)}\frac{1}{\sqrt{u^\top M_{s,r}(x;z^{k+1})u}}
\right),
\quad x\in\Omega_{\mathrm{chg}}^{k,\delta}.$$
当
$$\max_{x\in\Omega_{\mathrm{chg}}^{k,\delta}}
\left|
\phi_{s,r}^{k+1,m+1}(x)-\phi_{s,r}^{k+1,m}(x)
\right|
\le \varepsilon_\phi$$
时停止局部更新。
由于你现有模型中的 Bellman 形式已经是“邻域状态 + 局部步进代价”的离散动态规划格式，因此这种热启动更新与原求解器是严格兼容的。



## 7. 完整算法
基于上述设计，本文提出的 SA-HBO 算法流程如下。
### 算法 1：Structure-Aware Hybrid Block Optimization (SA-HBO)
**输入**： 初始控制 $z^0=(s^0,\eta^0,p^0,\xi^0)$，最大迭代次数 $K_{\max}$，步长序列 $\{\alpha_k\},\{c_k\}$。

**输出**： 优化后的控制变量 $z^\star$。


1. 初始化势函数与密度场，计算 $J(z^0)$；


2. 对 $k=0,1,\dots,K_{\max}-1$ 重复：

    2.1 固定 $(\eta^k,p^k,\xi^k)$，生成离散邻域 $\mathcal N(s^k)$；

    2.2 对每个候选 $s'\in\mathcal N(s^k)$ 计算代理目标 $\widehat J(s',z_c^k)$；

    2.3 选取前 $K$ 个候选组成 $\mathcal C_K(s^k)$；

    2.4 对 $s\in\mathcal C_K(s^k)$ 执行 Bellman 热启动局部重求解与全量密度推进，计算 $J(s,z_c^k)$；

    2.5 更新$$s^{k+1}=\arg\min_{s\in\mathcal C_K(s^k)}J(s,z_c^k);$$

    2.6 对连续块 $w_b\in\{\eta,p,\xi\}$ 逐块执行：

    - 生成扰动 $\Delta_b^k$；
    - 计算两侧扰动目标值；
    - 构造 $\widehat g_b^k$；
    - 更新 $$w_b^{k+1}
=\Pi_{\mathcal W_b}\left(w_b^k-\alpha_k D_b^{-1}\widehat g_b^k\right);$$


    2.7 计算 $J(z^{k+1})$；

    2.8 若满足停止条件，则退出。


3. 返回 $z^\star=z^{k+1}$。

## 8. 收敛与停止准则
本文采用如下停止准则：
$$|J(z^{k+1})-J(z^k)|\le \varepsilon_J$$
且连续 $L$ 次迭代中离散变量不再变化，即
$$s^{k+1}=s^k=\cdots=s^{k-L+1}.$$
在实际计算中，也可附加最大仿真次数或最大 CPU 时间限制。
由于离散块采用有限邻域下降，连续块采用随机近似梯度投影更新，因此算法整体可视为一种混合型块坐标下降框架。严格全局最优性一般难以保证，但从工程优化角度，该算法可获得稳定、可重复的高质量局部最优解，并且比完全黑箱方法更有效地利用了现有 PDE 结构。


## 9. 复杂度分析
设：
- 网格节点数为 $N$；
- 时间步数为 $T_n$；
- 子群体总数为
$$G=\sum_{s=1}^{S}R_s;$$


- 单次全域 Bellman 求解复杂度记为 $C_B(N)$；


- 单次密度推进复杂度为 $O(GNT_n)$。


则一次完整全仿真评估的复杂度为
$$C_{\mathrm{full}}
=
O\!\left(
G\,C_B(N)+GNT_n
\right).$$
若直接采用通用群智能算法，对每个候选解都做全仿真，则总成本通常为
$$O(N_{\mathrm{eval}}\,C_{\mathrm{full}}).$$
本文算法中，离散块每次只评估 $K$ 个候选，且每个候选采用局部 Bellman 更新。若受影响区域大小为 $N_{\mathrm{loc}}\ll N$，则局部 Bellman 更新复杂度可近似记为
$$C_B^{\mathrm{loc}}(N_{\mathrm{loc}})
\ll
C_B(N).$$
因此，单次离散块更新的复杂度约为
$$O\!\left(
|\mathcal N(s^k)|\,C_{\mathrm{proxy}}
+
K\,[G\,C_B^{\mathrm{loc}}(N_{\mathrm{loc}})+GNT_n]
\right),$$
其中 $C_{\mathrm{proxy}}$ 为代理评价成本。

连续块更新中，若采用分块 SPSA，则每一块只需两次目标评估，因此 3 个连续块总成本约为
$$O(6\,C_{\mathrm{full}}),$$
而不随连续变量维度线性增长。相比逐坐标有限差分
$$O(d\,C_{\mathrm{full}})$$
的代价，分块随机近似明显更适合高维参数场景。

因此，在 $N_{\mathrm{loc}}\ll N$ 且连续变量维数较高时，SA-HBO 相较于全黑箱优化具有显著的计算优势。


## 10. 方法创新点表述
本文的算法创新不在于引入某一通用元启发式方法，而在于针对现有 Bellman–守恒律耦合模型的计算结构，提出了结构感知的混合优化框架，具体体现在以下四个方面。

首先，针对通道方向、几何引导强度、分流概率与决策区域位置并存的控制问题，构建了一个混合离散–连续分块优化框架，避免将问题简单地视为统一黑箱搜索问题。

其次，利用单向管控变量仅通过允许方向集合 $U(x)$ 和局部度量张量 $M(x)$ 影响 Bellman 步进代价的特征，提出了Bellman 势场热启动与局部重求解机制，显著降低了候选解评估成本。你的现有模型正是通过 $U(x)$ 与 $M(x)$ 作为接口承载通道方向控制与几何引导，因此这种局部更新是模型结构直接支持的。

再次，针对多阶段–多路线密度 $\rho_{s,r}$ 的块状结构，提出了离散邻域搜索与连续分块随机近似梯度相结合的协调更新机制，从而兼顾了解质量与优化效率。

最后，在不修改下层人群模型数学形式的前提下，本文将算法创新集中于评估复用、局部更新、分块优化与多层筛选，因此具有较好的可移植性：只要下层模型仍保留 Bellman 势场和守恒律推进结构，该算法框架即可复用。