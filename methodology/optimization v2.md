# 优化 V2：基于通道方向配置与内部入口速率约束的混合整数 PDE 约束优化

> 本文件用于替代旧版 `optimization.md` 中以 $z=(s,\eta)$ 为优化变量的表述。V2 版本中，几何引导强度 $\eta$ 不再作为优化变量，而固定为 $\eta_0$；优化变量改为通道方向配置 $s$ 与内部通道入口通行速率 $q$。因此，优化问题从“方向配置 + 几何引导强度搜索”转为“方向配置 + 入口容量控制搜索”。

---

## 0. 本章目标与核心思想

V2 优化模块的目标是在给定场景几何、行人阶段结构、路线偏好参数 $\hat p$ 和固定几何引导张量 $M(x;\eta_0)$ 的条件下，自动寻找一组可实施的管控策略

$$
z=(s,q),
$$

使系统在效率、安全、通道负载均衡、入口等待和管控平滑性之间取得可解释的折中。

其中：

- $s=(s_1,\dots,s_C)$ 是通道方向配置，决定每条通道允许正向、反向、双向还是关闭；
- $q=\{q_{c,\ell}^+,q_{c,\ell}^-\}$ 是内部入口通行速率上限，决定每条通道每个时间段在两个方向上最多允许多少人进入；
- $\hat p$ 是固定的外生行为偏好参数，不作为优化变量；
- $\eta_0$ 是固定的几何引导强度，只影响通道内部方向偏好，不承担精确流量控制功能。

从优化角度看，该问题是一个**混合整数 PDE 约束优化问题的黑盒实现版本**：

1. $s$ 是离散变量，$q$ 是连续变量；
2. 目标函数必须通过整段 Bellman--守恒律仿真得到；
3. Bellman 方程中的 $\min$ 与 $\operatorname*{arg\,min}$、入口通量裁剪中的 $\min\{A,q\}$、安全指标中的阈值函数都会使目标函数非光滑；
4. 下层仿真器通常无法提供可靠解析梯度，因此优化算法应以无导数、可行性保持和多保真筛选为主；
5. 当前版本固定 $\hat p$，因此默认是确定性仿真优化；若未来加入随机需求、随机偏好或随机边界入流，则可扩展为随机仿真优化。

推荐的主流程是：

$$
\boxed{\text{外层方向配置枚举/筛选} + \text{内层约束 BO 优化 } q + \text{DFO/NOMAD 局部抛光} + \text{高保真复验}}.
$$

这个流程的优势是把离散结构、连续容量控制和高代价 PDE 仿真分开处理，避免把全部变量简单丢给一个统一黑箱优化器。

---

## 1. 优化变量与可行控制集合

### 1.1 通道方向配置变量

设共有 $C$ 条受控通道。第 $c$ 条通道的方向变量为

$$
s_c\in\{+1,-1,0,\varnothing\}.
$$

其含义为：

| 取值 | 管控含义 | 允许方向集合 |
|---|---|---|
| $s_c=+1$ | 只允许沿参考切向 $+\tau_c$ 方向进入并通过 | $U_c=\{+\tau_c\}$ |
| $s_c=-1$ | 只允许沿参考切向 $-\tau_c$ 方向进入并通过 | $U_c=\{-\tau_c\}$ |
| $s_c=0$ | 双向通行 | $U_c=\{+\tau_c,-\tau_c\}$ 或离散可行双向步进集合 |
| $s_c=\varnothing$ | 通道关闭 | $U_c=\varnothing$ |

因此，离散方向配置空间为

$$
\mathcal S
=
\{+1,-1,0,\varnothing\}^C.
$$

若 $C$ 较小，可以对 $\mathcal S$ 做完整枚举；若 $C$ 较大，则需要对 $\mathcal S$ 做结构化筛选，例如只枚举管理上可能出现的单向组合、保留专家基线附近的邻域、排除明显不可行的关闭组合等。

### 1.2 入口通行速率变量

第 $c$ 条通道有正向入口 $\Sigma_c^+$ 和反向入口 $\Sigma_c^-$。对应的入口速率上限为

$$
q_c^+(t)\in[0,\bar q_c^+],
\qquad
q_c^-(t)\in[0,\bar q_c^-].
$$

其中 $q_c^+(t)$ 表示沿 $+\tau_c$ 方向进入通道的最大允许速率，$q_c^-(t)$ 表示沿 $-\tau_c$ 方向进入通道的最大允许速率。

为降低优化维度，令 $q_c^\pm(t)$ 在 $L$ 个时间段上分段常数。设

$$
0=t_0<t_1<\cdots<t_L=T,
$$

则

$$
q_c^\pm(t)=q_{c,\ell}^\pm,
\qquad t\in[t_\ell,t_{\ell+1}),
\qquad \ell=0,\dots,L-1.
$$

于是连续控制变量变为有限维向量

$$
q=\{q_{c,\ell}^+,q_{c,\ell}^-\}_{c=1,\dots,C;\ \ell=0,\dots,L-1}.
$$

在没有方向约束剪枝时，$q$ 的名义维度为 $2CL$；在单向或关闭通道下，实际自由维度会减少。

### 1.3 方向--速率一致性约束

方向配置与入口速率必须一致。对任意 $c$ 和 $\ell$，有

$$
s_c=+1\Rightarrow q_{c,\ell}^-=0,
$$

$$
s_c=-1\Rightarrow q_{c,\ell}^+=0,
$$

$$
s_c=\varnothing\Rightarrow q_{c,\ell}^+=q_{c,\ell}^-=0.
$$

当 $s_c=0$ 时，两个方向均可通行，但建议加入通道总容量约束：

$$
q_{c,\ell}^+ + q_{c,\ell}^- \le \bar q_c,
\qquad \ell=0,\dots,L-1.
$$

该约束表示双向共用同一条通道的物理容量。若正向和反向入口确实独立，也可以改用两个单独上界 $\bar q_c^+$ 和 $\bar q_c^-$，但仍建议在双向通道中保留一个总容量约束以避免不现实的容量叠加。

### 1.4 可选的管控平滑性硬约束

若现场执行不允许入口速率频繁剧烈变化，可对相邻时间段加入变化幅度约束：

$$
|q_{c,\ell+1}^+ - q_{c,\ell}^+|\le \Delta q_{c,\max}^+,
$$

$$
|q_{c,\ell+1}^- - q_{c,\ell}^-|\le \Delta q_{c,\max}^-,
$$

其中 $\Delta q_{c,\max}^\pm$ 是每个时间段允许调整的最大速率。该约束也可以不作为硬约束，而通过平滑性目标项 $\tilde J_R$ 软惩罚。

### 1.5 可选的最低服务能力约束

入口限流可能降低区域内部拥堵，但若 $q$ 被优化器压得过低，会在入口等待区造成过长积压。除使用 $\tilde J_B$ 惩罚入口等待外，还可以加入最低服务能力约束，例如

$$
\sum_{c=1}^C\left(q_{c,\ell}^+ + q_{c,\ell}^-\right) \ge q_{\min,\ell},
\qquad \ell=0,\dots,L-1.
$$

或者要求某些关键通道不能全部关闭：

$$
\#\{c:s_c\ne\varnothing\}\ge C_{\min}^{\mathrm{open}}.
$$

这类约束应由管理规则或疏散需求给出，不建议由优化器通过惩罚项自行学习。

### 1.6 完整可行集合

综合上述约束，V2 的可行控制集合定义为

$$
\mathcal Z
=
\left\{
(s,q):
\begin{array}{l}
 s_c\in\{+1,-1,0,\varnothing\},\\[2mm]
 0\le q_{c,\ell}^\pm\le \bar q_c^\pm,\\[2mm]
 s_c=+1\Rightarrow q_{c,\ell}^-=0,\\
 s_c=-1\Rightarrow q_{c,\ell}^+=0,\\
 s_c=\varnothing\Rightarrow q_{c,\ell}^+=q_{c,\ell}^-=0,\\[1mm]
 s_c=0\Rightarrow q_{c,\ell}^+ + q_{c,\ell}^-\le \bar q_c,\\[1mm]
 \text{满足可选平滑性、最低服务能力和工程实施约束}
\end{array}
\right\}.
$$

优化器只能在 $\mathcal Z$ 中生成候选解。方向--速率一致性、通道总容量和关闭状态等规则是硬约束，应在候选生成或变量变换阶段直接保证，而不是作为目标函数中的惩罚项。

---

## 2. 下层 Bellman--守恒律评估器

### 2.1 仿真算子

对任意可行控制变量 $z=(s,q)\in\mathcal Z$，定义下层仿真算子

$$
\mathcal S(z;\hat p,\eta_0)
=
\left\{\rho_g^n,\phi_g^n,u_g^{*,n},v_g^n,\widehat F_{g,f}^n\right\}_{g\in\mathcal G,\ n=0,\dots,N}.
$$

该算子由以下部分组成：

1. 根据 $s$ 构造每个通道区域内的允许方向集合 $U_g(x;s)$；
2. 根据固定 $\eta_0$ 构造几何引导张量 $M_g(x;\eta_0)$；
3. 在每个时间层基于总密度 $\rho=\sum_g\rho_g$ 求解各子群体 Bellman 方程；
4. 恢复最优方向 $u_g^*$ 和速度场 $v_g=F(\rho)u_g^*$；
5. 计算未受限自由通量 $F_{g,f}^{\mathrm{free},n}$；
6. 在内部入口截面 $\mathcal F_c^\pm$ 上根据 $q_{c,\ell}^\pm$ 裁剪通量；
7. 使用有限体积守恒格式推进各子群体密度；
8. 累计效率、安全、通道实际通过量、入口等待和管控平滑性指标。

优化算法不直接修改 PDE 模型，只通过 $s$ 和 $q$ 改变下层仿真器中的方向集合与内部入口通量上限。

### 2.2 Bellman 路径选择层

对每个阶段--路线子群体 $g\in\mathcal G$，给定总密度 $\rho$、方向配置 $s$ 和固定度量张量 $M_g(x;\eta_0)$ 后，势函数满足离散 Bellman 方程

$$
\phi_g(x)
=
\min_{u\in U_g(x;s)}
\left[
\phi_g(x+\Delta x u)
+
\frac{\Delta x}{F(\rho(x,t))}
\frac{1}{\sqrt{u^\top M_g(x;\eta_0)u}}
\right].
$$

最优方向为

$$
u_g^*(x,t)
=
\operatorname*{arg\,min}_{u\in U_g(x;s)}
\left[
\phi_g(x+\Delta x u)
+
\frac{\Delta x}{F(\rho(x,t))}
\frac{1}{\sqrt{u^\top M_g(x;\eta_0)u}}
\right],
$$

速度场为

$$
v_g(x,t)=F(\rho(x,t))u_g^*(x,t).
$$

其中速度--密度关系可取 Greenshields 型：

$$
F(\rho)=v_{\max}\left(1-\frac{\rho}{\rho_{\max}}\right)_+.
$$

### 2.3 内部入口速率约束

对第 $c$ 条通道的正向入口离散网格面集合 $\mathcal F_c^+$，第 $n$ 个时间层的总尝试进入通量为

$$
A_c^{+,n}
=
\sum_{g\in\mathcal G}
\sum_{f\in\mathcal F_c^+}
\left(F_{g,f}^{\mathrm{free},n}\right)_+.
$$

若代码中 $F_{g,f}^{\mathrm{free},n}$ 存储的是通量密度而不是已经对边长积分后的面通量，则应写为

$$
A_c^{+,n}
=
\sum_{g\in\mathcal G}
\sum_{f\in\mathcal F_c^+}
|f|\left(F_{g,f}^{\mathrm{free},n}\right)_+.
$$

实际允许通过量为

$$
\widehat A_c^{+,n}
=
\min\{A_c^{+,n},q_c^{+,n}\}.
$$

缩放系数为

$$
\theta_c^{+,n}
=
\begin{cases}
\widehat A_c^{+,n}/A_c^{+,n}, & A_c^{+,n}>0,\\[1mm]
1, & A_c^{+,n}=0.
\end{cases}
$$

于是受限通量为

$$
\widehat F_{g,f}^n
=
\theta_c^{+,n}\left(F_{g,f}^{\mathrm{free},n}\right)_+
+
\left(F_{g,f}^{\mathrm{free},n}\right)_-,
\qquad f\in\mathcal F_c^+.
$$

反向入口 $\mathcal F_c^-$ 完全同理。该处理方式只限制进入通道方向的正通量，不删除被拦截的人群质量；未通过的人群保留在上游单元，从而自然形成等待和积压。

### 2.4 有限体积密度更新

对每个群体 $g$ 和单元 $K_i$，密度更新为

$$
\rho_{g,i}^{n+1}
=
\rho_{g,i}^{n}
-
\frac{\Delta t}{|K_i|}
\sum_{f\subset\partial K_i}
\widehat F_{g,f}^{n}
+
\Delta t\,S_{g,i}^{n}(\hat p).
$$

其中 $S_{g,i}^{n}(\hat p)$ 表示多阶段转移与路线分流源汇项。由于 $\hat p$ 固定，优化器不会修改行为层偏好，只通过 $s$ 和 $q$ 改变路径可行性与入口容量。

---

## 3. 优化目标函数

### 3.1 原始物理指标

给定 $z=(s,q)$ 后，通过下层仿真得到以下原始指标。

#### 3.1.1 效率指标：总旅行时间

$$
J_1(s,q\mid\hat p)
=
\int_0^T\int_{\Omega_w}\rho(x,t)\,dx\,dt.
$$

该指标衡量人群在系统中的总体滞留量。$J_1$ 越小，说明整体通行效率越高。

#### 3.1.2 安全指标：高密度暴露时间

$$
J_2(s,q\mid\hat p)
=
\int_0^T\int_{\Omega_w}
\mathbf 1_{\rho(x,t)>\rho_{\mathrm{safe}}}\,dx\,dt.
$$

该指标衡量高密度状态在时空域中的累计覆盖程度。$J_2$ 越小，说明高风险拥挤暴露越少。

#### 3.1.3 通道负载不均衡指标

第 $c$ 条通道在仿真时段内的实际累计通过量记为

$$
R_c
=
\int_0^T
\left[\widehat A_c^+(t)+\widehat A_c^-(t)\right]dt.
$$

通道负载不均衡指标为

$$
J_5(s,q\mid\hat p)
=
\operatorname{Var}(R_1,\dots,R_C).
$$

这里使用实际通过量 $R_c$，而不是控制上限 $q_c^\pm(t)$。因为管理上真正关心的是人群实际使用通道是否过度集中，而不是名义上给了多少入口容量。

#### 3.1.4 入口等待指标

若 $B_c^+(t)$ 和 $B_c^-(t)$ 分别表示第 $c$ 条通道两个入口等待区中的积压质量，则入口等待指标为

$$
J_B(s,q\mid\hat p)
=
\int_0^T
\sum_{c=1}^C\left[B_c^+(t)+B_c^-(t)\right]dt.
$$

该指标用于防止优化器通过过度降低 $q$ 来“把拥堵挪到入口外侧”。在 V2 中，$J_B$ 应作为入口限流模型的重要目标项。

#### 3.1.5 管控平滑性指标

若 $q$ 分为 $L$ 个时间段，则可定义

$$
J_R(q)
=
\sum_{c=1}^C\sum_{\ell=0}^{L-2}
\left[
(q_{c,\ell+1}^+-q_{c,\ell}^+)^2
+
(q_{c,\ell+1}^--q_{c,\ell}^-)^2
\right].
$$

该项不是人群状态指标，而是管控动作可实施性指标。若只设置一个时间段 $L=1$，则 $J_R=0$。

### 3.2 无量纲标准化指标

不同指标的量纲和数量级不同，优化目标应使用标准化后的指标。

设

$$
M_{\mathrm{tot}}=M_{\mathrm{init}}+M_{\mathrm{in}},
$$

其中 $M_{\mathrm{init}}$ 是初始人群质量，$M_{\mathrm{in}}$ 是仿真期内外部进入系统的总质量。设 $|\Omega_w|$ 是可通行区域面积。

效率标准化指标为

$$
\tilde J_1
=
\frac{J_1}{M_{\mathrm{tot}}T}.
$$

安全标准化指标为

$$
\tilde J_2
=
\frac{J_2}{|\Omega_w|T}.
$$

若总通过量 $R=\sum_{c=1}^C R_c>0$，令

$$
p_c=\frac{R_c}{R},
$$

则负载均衡标准化指标为

$$
\tilde J_5
=
\frac{C^2}{C-1}\operatorname{Var}(p_1,\dots,p_C).
$$

当 $R=0$ 时，说明候选解没有产生有效通道通过量，应将其标记为不可行或给予极大目标值，而不应简单令 $\tilde J_5=0$ 后误判为负载均衡。

入口等待标准化指标为

$$
\tilde J_B
=
\frac{J_B}{M_{\mathrm{tot}}T}.
$$

管控平滑性可标准化为

$$
\tilde J_R
=
\frac{1}{2C\max(L-1,1)}
\sum_{c=1}^C\sum_{\ell=0}^{L-2}
\left[
\left(
\frac{q_{c,\ell+1}^+-q_{c,\ell}^+}{\bar q_c^+}
\right)^2
+
\left(
\frac{q_{c,\ell+1}^--q_{c,\ell}^-}{\bar q_c^-}
\right)^2
\right],
$$

其中上界为 0 的方向不参与该项，或在代码中使用可行方向集合自动跳过。

### 3.3 标量化优化目标

在给定权重

$$
\lambda_1,\lambda_2,\lambda_5,\lambda_B,\lambda_R\ge0
$$

的条件下，V2 优化目标定义为

$$
\min_{(s,q)\in\mathcal Z}
J(s,q\mid\hat p)
=
\lambda_1\tilde J_1
+
\lambda_2\tilde J_2
+
\lambda_5\tilde J_5
+
\lambda_B\tilde J_B
+
\lambda_R\tilde J_R.
$$

若暂不考虑管控动作平滑性，可取 $\lambda_R=0$。若暂不启用入口等待区诊断，可取 $\lambda_B=0$；但在入口速率 $q$ 被纳入优化后，建议默认开启 $\lambda_B>0$，否则优化器可能倾向于过度限流。



---

## 4. 问题性质与算法选择

### 4.1 问题性质

V2 优化问题具有以下特点：

| 性质 | 说明 | 对算法的影响 |
|---|---|---|
| 混合变量 | $s$ 离散，$q$ 连续 | 应分块处理，不宜直接连续松弛后四舍五入 |
| PDE 约束 | 每次目标评价都需完整仿真 | 单次评价昂贵，需要多保真筛选和代理模型 |
| 非凸 | 通道方向改变会导致路径拓扑变化 | 局部算法需多初值，不能只依赖单次局部搜索 |
| 非光滑 | Bellman 最优方向切换、入口通量裁剪、安全阈值函数 | 不适合依赖梯度，宜用 BO、MADS/NOMAD、信赖域 DFO |
| 可行域结构强 | 很多约束由管理规则明确给出 | 应在候选生成阶段保证可行，不应让优化器通过惩罚项学习 |
| 多目标标量化 | 权重体现管理偏好 | 最优解随权重变化，需要权重敏感性分析 |

### 4.2 算法框架

采用分层算法：

1. **离散层预处理**：生成或枚举合法方向配置 $s$，剔除明显不合法或不可实施的方案；
2. **低保真筛选**：用粗网格、短时段或少量代表性 $q$ 模式快速筛掉差的 $s$；
3. **固定 $s$ 的连续优化**：对每个入围 $s$，用约束贝叶斯优化搜索 $q$；
4. **局部抛光**：对 BO 得到的少数最优 $q$，使用 MADS/NOMAD 或信赖域无导数搜索做局部改进；
5. **高保真复验**：在完整网格、完整时间步和完整场景下复评最优候选；
6. **权重与鲁棒性分析**：检验候选在不同权重、随机种子、网格精度或需求强度下是否稳定。

该框架可称为 **V2-HCMBO**：Hierarchical Constrained Mixed-variable Black-box Optimization。

---

## 5. 变量参数化与可行候选生成

### 5.1 为什么要归一化 $q$

优化器内部建议使用归一化变量

$$
x\in[0,1]^{d_s},
$$

再通过映射

$$
q=\mathcal T_s(x)
$$

生成物理速率。这样可以避免不同通道容量数量级不同造成代理模型训练困难，也方便在不同方向配置 $s$ 下自动减少自由变量维度。

### 5.2 固定 $s$ 下的自由维度

给定 $s$ 后，定义第 $c$ 条通道在一个时间段中的自由方向集合：

$$
\mathcal D_c(s_c)=
\begin{cases}
\{+\}, & s_c=+1,\\
\{-\}, & s_c=-1,\\
\{+,-\}, & s_c=0,\\
\varnothing, & s_c=\varnothing.
\end{cases}
$$

于是固定 $s$ 的连续自由维度为

$$
d_s=L\sum_{c=1}^C |\mathcal D_c(s_c)|,
$$

如果双向总容量约束采用二元 simplex 参数化，则每个双向通道每个时间段仍可保留两个内部变量，但必须经过 simplex 投影或容量归一化。

### 5.3 映射 $q=\mathcal T_s(x)$

对单向通道，可以直接用线性映射：

$$
s_c=+1:
\quad
q_{c,\ell}^+=\bar q_c^+x_{c,\ell}^+,
\quad
q_{c,\ell}^-=0.
$$

$$
s_c=-1:
\quad
q_{c,\ell}^-=\bar q_c^-x_{c,\ell}^-,
\quad
q_{c,\ell}^+=0.
$$

对关闭通道，有

$$
s_c=\varnothing:
\quad
q_{c,\ell}^+=q_{c,\ell}^-=0.
$$

对双向通道，若采用总容量 $\bar q_c$，可用如下投影式映射。先生成两个非负原始变量

$$
r_{c,\ell}^+=x_{c,\ell}^+,
\qquad
r_{c,\ell}^-=x_{c,\ell}^-.
$$

若 $r_{c,\ell}^++r_{c,\ell}^-\le1$，则

$$
q_{c,\ell}^+=\bar q_c r_{c,\ell}^+,
\qquad
q_{c,\ell}^-=\bar q_c r_{c,\ell}^-.
$$

若 $r_{c,\ell}^++r_{c,\ell}^->1$，则投影到容量 simplex：

$$
q_{c,\ell}^+
=\bar q_c\frac{r_{c,\ell}^+}{r_{c,\ell}^++r_{c,\ell}^-},
\qquad
q_{c,\ell}^-
=\bar q_c\frac{r_{c,\ell}^-}{r_{c,\ell}^++r_{c,\ell}^-}.
$$

这样生成的 $q$ 天然满足双向总容量约束。

### 5.4 平滑性投影

若使用平滑性硬约束，可在 $q=\mathcal T_s(x)$ 后做一次时间方向投影。简单实现为对每个通道方向递推截断：

$$
q_{c,\ell+1}^{\pm}
\leftarrow
\min\left\{q_{c,\ell}^{\pm}+\Delta q_{c,\max}^{\pm},
\max\left(q_{c,\ell}^{\pm}-\Delta q_{c,\max}^{\pm},q_{c,\ell+1}^{\pm}\right)
\right\}.
$$

如果平滑性只作为目标项 $\tilde J_R$，则不需要该投影。

---

## 6. 外层方向配置生成与筛选

### 6.1 完整枚举

当 $C$ 较小，例如 $C\le 6$，可以直接枚举

$$
|\mathcal S|=4^C
$$

个方向配置，并对每个配置做硬约束过滤。完整枚举的优点是不会漏掉离散最优方向配置；缺点是在 $C$ 较大时成本迅速上升。例如 $C=10$ 时，$4^{10}=1,048,576$，直接全量评估不可行。

### 6.2 结构化候选集

当 $C$ 较大时，建议构造候选集合 $\mathcal S_{\mathrm{cand}}\subset\mathcal S$。候选来源包括：

1. 管理经验方案，例如常用的“某些通道只上、某些通道只下”；
2. 全双向开放方案；
3. 单向环流方案；
4. 只关闭一个通道或少量通道的方案；
5. 专家基线附近的 Hamming 邻域；
6. 随机采样但满足最低开放通道数量和最低服务能力的方案；
7. 历史实验中表现好的方向配置。

### 6.3 离散硬约束过滤

对每个 $s$，先做以下过滤：

- 若所有通道关闭，则剔除；
- 若开放通道数量小于 $C_{\min}^{\mathrm{open}}$，则剔除；
- 若关键 OD 对没有可行通路，则剔除；
- 若根据容量上界估计的总服务能力低于最低需求，则剔除；
- 若管理上禁止某类方向组合，例如相邻两条关键通道同时反向，则剔除。

这些规则是硬约束，应在外层候选生成时执行，不应交给内层 BO 或 DFO 通过目标惩罚处理。

### 6.4 低保真筛选

对保留下来的每个 $s$，使用少量代表性 $q$ 模式进行低保真仿真。典型 $q$ 模式包括：

| 模式 | 定义 | 作用 |
|---|---|---|
| 高容量 | 所有允许方向 $q$ 取容量上界 | 评估方向配置本身的通行能力 |
| 中容量 | 所有允许方向 $q=0.5\bar q$ | 评估温和限流下表现 |
| 低容量 | 所有允许方向 $q=0.25\bar q$ | 评估强限流下入口等待风险 |
| 均衡容量 | 双向或多通道容量按均衡原则分配 | 评估负载均衡潜力 |
| 专家容量 | 使用人工给定容量曲线 | 对齐现场管理经验 |

低保真可以通过以下方式降低成本：

- 使用更粗空间网格；
- 使用更少时间步；
- 缩短仿真时间，只评估拥堵形成的关键阶段；
- 减少保存输出频率；
- 使用较少群体或简化行为层。

记低保真目标为 $J^{\mathrm{LF}}(s,q)$。对每个 $s$，取若干代表性 $q$ 中的最小值：

$$
\bar J^{\mathrm{LF}}(s)
=
\min_{q\in\mathcal Q_{\mathrm{base}}(s)}J^{\mathrm{LF}}(s,q).
$$

然后保留排名前 $K_s$ 个方向配置，得到入围集合

$$
\mathcal S_{\mathrm{short}}
=
\operatorname{TopK}_{s\in\mathcal S_{\mathrm{cand}}}\left[-\bar J^{\mathrm{LF}}(s)\right].
$$

---

## 7. 固定方向配置下的约束贝叶斯优化

### 7.1 内层连续子问题

对每个入围方向配置 $s\in\mathcal S_{\mathrm{short}}$，求解固定 $s$ 下的连续优化子问题：

$$
\min_{q\in\mathcal Q(s)}J(s,q\mid\hat p),
$$

其中 $\mathcal Q(s)$ 是在该方向配置下满足所有速率上界、方向一致性、双向容量和平滑性规则的连续可行集合。

通过归一化变量 $x\in[0,1]^{d_s}$ 和可行映射 $q=\mathcal T_s(x)$，该问题变为

$$
\min_{x\in[0,1]^{d_s}}
\mathcal J_s(x)
:=J(s,\mathcal T_s(x)\mid\hat p).
$$

### 7.2 初始设计

BO 的初始样本不应完全随机，应包含人工基线和空间填充样本。建议初始集合为

$$
\mathcal X_0(s)=
\mathcal X_{\mathrm{base}}(s)
\cup
\mathcal X_{\mathrm{lhs}}(s)
\cup
\mathcal X_{\mathrm{near}}(s).
$$

其中：

- $\mathcal X_{\mathrm{base}}$ 包含高容量、中容量、低容量、均衡容量、专家容量等基线；
- $\mathcal X_{\mathrm{lhs}}$ 是 Latin Hypercube 或 Sobol 低差异采样点；
- $\mathcal X_{\mathrm{near}}$ 是围绕低保真最好容量模式的小扰动点。

每个初始点都必须先经过 $\mathcal T_s$ 映射和可行性检查，再调用高保真或中保真仿真。

### 7.3 代理模型

BO 对固定 $s$ 的黑盒函数 $\mathcal J_s(x)$ 建立代理模型。可选模型包括：

1. 高斯过程代理，适合 $d_s$ 较低且评价次数较少的情况；
2. 随机森林或 TPE，适合维度较高、变量较多或目标噪声较强的情况；
3. 多任务或多保真 GP，适合同时使用低保真和高保真评价的情况。

如果仿真输出是确定性的，观测模型可以写为

$$
y_i=\mathcal J_s(x_i).
$$

如果仿真中包含随机种子、随机入流或随机偏好，则可写为

$$
y_i=\mathcal J_s(x_i)+\epsilon_i,
\qquad
\epsilon_i\sim\mathcal N(0,\sigma_\epsilon^2),
$$

并对同一候选解进行多种子重复评估。

### 7.4 采集函数

常用采集函数包括期望改进 EI、概率改进 PI 和置信下界 LCB。以 LCB 为例，若代理模型给出均值 $\mu_s(x)$ 和标准差 $\sigma_s(x)$，则下一候选点可由

$$
x_{\mathrm{next}}
=
\arg\min_{x\in[0,1]^{d_s}}
\left[
\mu_s(x)-\kappa\sigma_s(x)
\right]
$$

给出，其中 $\kappa>0$ 控制探索强度。

如果存在仿真层面的软可行性约束 $g_j(x)\le0$，例如终端剩余质量或最大密度限制，可单独训练约束代理，并使用可行性加权采集函数：

$$
a_{\mathrm{con}}(x)
=a(x)\prod_j \mathbb P(g_j(x)\le0).
$$

但方向--速率一致性、上下界、双向总容量等已知硬约束不应通过 $g_j$ 学习，而应由 $\mathcal T_s$ 直接保证。

### 7.5 BO 迭代步骤

固定 $s$ 的 BO 过程如下：

1. 生成初始样本 $\mathcal X_0(s)$；
2. 对每个 $x_i\in\mathcal X_0(s)$，计算 $q_i=\mathcal T_s(x_i)$ 并运行仿真，得到 $y_i=J(s,q_i)$；
3. 使用样本集 $\mathcal D_s=\{(x_i,y_i)\}$ 训练代理模型；
4. 优化采集函数得到 $x_{\mathrm{next}}$；
5. 计算 $q_{\mathrm{next}}=\mathcal T_s(x_{\mathrm{next}})$ 并运行仿真；
6. 将新样本加入 $\mathcal D_s$；
7. 若达到评估预算或连续若干轮改进不足，则停止；否则返回第 3 步。

输出固定 $s$ 下的前 $K_q$ 个候选：

$$
\mathcal Q_{\mathrm{BO}}^{\mathrm{top}}(s)
=
\operatorname{TopK}_{q\in\mathcal D_s}\left[-J(s,q)\right].
$$

---

## 8. DFO/NOMAD 局部抛光

### 8.1 局部抛光的必要性

BO 擅长在有限预算下全局探索，但其最后给出的最优点未必在局部已经充分优化。V2 目标函数又具有通量裁剪、安全阈值和 Bellman 方向切换导致的非光滑结构，因此推荐对 BO 的最好 1--3 个候选进行无导数局部抛光。

可选方法包括：

- MADS/NOMAD；
- 坐标方向 direct search；
- 信赖域无导数优化；
- pattern search；
- 对低维 $q$ 的局部网格细化。

### 8.2 固定 $s$ 的局部搜索形式

对 BO 给出的候选 $x^{(0)}$，求解

$$
\min_{x\in[0,1]^{d_s}}\mathcal J_s(x).
$$

每次生成试探点 $x'$ 后，先投影到 $[0,1]^{d_s}$，再由 $q'=\mathcal T_s(x')$ 保证物理可行。

### 8.3 简化 MADS/NOMAD 步骤

设当前点为 $x^k$，网格尺度为 $\Delta_k$。一次迭代包括：

1. 生成一组方向 $D_k=\{d_1,\dots,d_m\}$，例如坐标正负方向和若干随机方向；
2. 构造试探点
   $$
   x_j'=\Pi_{[0,1]^{d_s}}(x^k+\Delta_k d_j);
   $$
3. 对每个 $x_j'$ 计算 $q_j'=\mathcal T_s(x_j')$，运行仿真得到目标值；
4. 若存在 $J(s,q_j')<J(s,q^k)-\epsilon_{\mathrm{imp}}$，接受最优试探点并适当增大或保持 $\Delta_k$；
5. 否则拒绝本轮试探并缩小 $\Delta_k$；
6. 当 $\Delta_k<\Delta_{\min}$ 或评估次数达到预算时停止。

该步骤保留 BO 的全局探索结果，同时对局部容量曲线做细化。

---

## 9. 高保真复验与权重敏感性分析

### 9.1 高保真复验

BO 和 DFO 阶段可以使用中等保真度以节省计算，但最终排序必须使用高保真仿真。对全局候选集

$$
\mathcal C_{\mathrm{final}}
=
\{(s,q):s\in\mathcal S_{\mathrm{short}},\ q\in\mathcal Q_{\mathrm{BO/DFO}}^{\mathrm{top}}(s)\}
$$

运行完整网格、完整时间步和完整行为层仿真，重新计算

$$
\tilde J_1,
\tilde J_2,
\tilde J_5,
\tilde J_B,
\tilde J_R,
J.
$$

最终推荐方案应以高保真目标值为准，而不是以低保真或代理模型预测为准。

### 9.2 权重敏感性分析

由于目标函数是标量化加权和，最优解会受到权重影响。建议至少测试以下权重组：

| 权重组 | 管理含义 |
|---|---|
| 效率优先 | 较大 $\lambda_1$ |
| 安全优先 | 较大 $\lambda_2$ |
| 均衡优先 | 较大 $\lambda_5$ |
| 入口等待优先 | 较大 $\lambda_B$ |
| 平滑执行优先 | 较大 $\lambda_R$ |
| 综合折中 | 各项权重按管理偏好归一化 |

对每组权重，不一定重新运行完整优化；可以先用已评估候选库重新计算 $J$ 并排序。如果候选库不足以覆盖新权重偏好，再针对该权重组补充 BO/DFO 搜索。

### 9.3 鲁棒性检查

最终候选应在以下扰动下保持基本稳定：

- 空间网格加密或粗化；
- 时间步长变化；
- 不同随机种子；
- 不同初始人群规模；
- 不同路线偏好参数 $\hat p$ 的合理扰动；
- 不同安全阈值 $\rho_{\mathrm{safe}}$；
- 通道容量上界 $\bar q_c^\pm$ 的估计误差。

如果某个候选只在单一权重、单一网格或单一需求强度下表现最好，而稍有扰动即明显变差，则不应作为强推荐方案。

---

## 10. 完整算法

### 算法 1：V2 分层约束混合黑盒优化

**输入：**

- 场景几何与可通行区域 $\Omega_w$；
- 通道区域 $\Omega_c$、入口截面 $\Sigma_c^\pm$、参考方向 $\tau_c$；
- 固定行为参数 $\hat p$；
- 固定几何引导强度 $\eta_0$；
- 容量上界 $\bar q_c^\pm$ 与时间分段 $\{t_\ell\}$；
- 权重 $\lambda_1,\lambda_2,\lambda_5,\lambda_B,\lambda_R$；
- 外层方向候选预算 $K_s$；
- 每个方向配置下的 BO 预算 $N_{\mathrm{BO}}$；
- 局部抛光预算 $N_{\mathrm{DFO}}$；
- 高保真复验候选数 $K_{\mathrm{HF}}$。

**输出：**

- 最优候选 $(s^*,q^*)$；
- 高保真指标 $\tilde J_1,\tilde J_2,\tilde J_5,\tilde J_B,\tilde J_R,J$；
- 候选排行榜与权重敏感性报告。

**步骤：**

```text
1. 生成方向配置候选集 S_cand。
2. 对每个 s in S_cand：
      2.1 检查离散硬约束，例如开放通道数、关键连通性、最低服务能力。
      2.2 若不满足硬约束，则剔除。
3. 对每个保留的 s：
      3.1 构造代表性容量模式 Q_base(s)。
      3.2 使用低保真仿真评估 min_{q in Q_base(s)} J_LF(s,q)。
4. 选取低保真表现最好的 K_s 个方向配置，形成 S_short。
5. 对每个 s in S_short：
      5.1 构造归一化变量 x 和可行映射 q=T_s(x)。
      5.2 生成 BO 初始设计 X_0(s)，包括人工基线和空间填充样本。
      5.3 对 X_0(s) 逐点评估中/高保真目标，得到数据集 D_s。
      5.4 while BO 未达到停止条件：
              a. 拟合代理模型。
              b. 优化采集函数，得到 x_next。
              c. 计算 q_next=T_s(x_next)。
              d. 运行仿真，得到 J(s,q_next)。
              e. 更新 D_s。
      5.5 保存固定 s 下 BO 排名前 K_q 的候选。
      5.6 对最好的若干候选执行 DFO/NOMAD 局部抛光。
6. 汇总所有 s 下的 BO/DFO 候选，形成 C_final。
7. 对 C_final 中排名前 K_HF 的候选运行高保真复验。
8. 按高保真目标值排序，得到 (s*,q*)。
9. 对候选库执行权重敏感性和鲁棒性检查。
10. 输出最优策略、指标表、候选排行榜和诊断图表。
```

### 算法 2：固定 $s$ 的可行速率映射 $q=\mathcal T_s(x)$

```text
Input: direction setting s, normalized vector x, capacity bounds qbar
Output: feasible piecewise-constant capacity q

for each channel c:
    for each time segment ell:
        if s_c == PLUS:
            q_plus[c,ell]  = qbar_plus[c] * x_plus[c,ell]
            q_minus[c,ell] = 0

        else if s_c == MINUS:
            q_plus[c,ell]  = 0
            q_minus[c,ell] = qbar_minus[c] * x_minus[c,ell]

        else if s_c == CLOSED:
            q_plus[c,ell]  = 0
            q_minus[c,ell] = 0

        else if s_c == FREE:
            r_plus  = max(0, x_plus[c,ell])
            r_minus = max(0, x_minus[c,ell])
            if r_plus + r_minus <= 1:
                q_plus[c,ell]  = qbar_total[c] * r_plus
                q_minus[c,ell] = qbar_total[c] * r_minus
            else:
                q_plus[c,ell]  = qbar_total[c] * r_plus  / (r_plus + r_minus)
                q_minus[c,ell] = qbar_total[c] * r_minus / (r_plus + r_minus)

apply optional smoothness projection if smoothness is a hard constraint
return q
```

### 算法 3：候选解评估器

```text
Input: feasible control z=(s,q), fixed behavior parameter p_hat, fixed eta0
Output: objective value J and diagnostic metrics

initialize group densities rho_g^0
initialize accumulators J1, J2, R_c, J_B, snapshots/logs

for n = 0,...,N-1:
    rho_total = sum_g rho_g^n
    speed = F(rho_total)

    for each group g:
        build allowed direction set U_g(x;s)
        use fixed metric tensor M_g(x;eta0)
        solve Bellman equation for phi_g^n
        recover optimal direction u_g^{*,n}
        compute velocity v_g^n = speed * u_g^{*,n}

    compute free finite-volume face fluxes F_free

    for each controlled channel c and direction +/-:
        collect entrance faces F_c^+ or F_c^-
        compute attempted entrance rate A_c^{+/-,n}
        locate time segment ell with t^n in [t_ell,t_{ell+1})
        compute allowed rate q_c^{+/-,ell}
        compute actual rate Ahat_c^{+/-,n} = min(A_c^{+/-,n}, q_c^{+/-,ell})
        scale positive entrance fluxes by theta = Ahat / A
        keep opposite-sign fluxes unchanged

    update each rho_g by finite volume conservation law
    apply stage transition/source terms with fixed p_hat

    accumulate J1, J2, R_c, J_B

compute normalized metrics Jtilde_1, Jtilde_2, Jtilde_5, Jtilde_B, Jtilde_R
compute scalarized objective J
return metrics and logs
```

---

## 11. 停止准则

### 11.1 BO 停止准则

固定 $s$ 的 BO 可使用双重停止准则：

1. 达到最大评估次数 $N_{\mathrm{BO}}$；
2. 连续 $r_{\mathrm{stall}}$ 轮没有显著改进，即

$$
J_{\mathrm{best}}^{k-r_{\mathrm{stall}}}-J_{\mathrm{best}}^k<\epsilon_{\mathrm{BO}}.
$$

其中 $\epsilon_{\mathrm{BO}}$ 可以取目标函数量级的 $10^{-3}$ 到 $10^{-2}$。

### 11.2 DFO/NOMAD 停止准则

局部抛光可在以下条件之一满足时停止：

- 网格尺度或信赖域半径小于阈值：
  $$
  \Delta_k<\Delta_{\min};
  $$
- 达到最大局部评估次数 $N_{\mathrm{DFO}}$；
- 连续若干轮目标改进小于 $\epsilon_{\mathrm{DFO}}$；
- 新候选全部被可行性规则拒绝。

### 11.3 高保真复验停止准则

高保真复验不是看优化器是否收敛，而是看候选排名是否稳定。若排名前几的候选在不同网格、不同种子或不同权重下频繁交换，说明仍需增加复验或给出多个备选方案，而不应只报告单一最优解。

---

## 12. 复杂度与评估预算

设：

- $N_{\mathrm{dir}}$ 为经过硬约束过滤后的方向配置数量；
- $N_{\mathrm{base}}$ 为每个方向配置用于低保真筛选的基线容量模式数量；
- $K_s$ 为入围方向配置数量；
- $N_{\mathrm{init}}$ 为 BO 初始设计点数；
- $N_{\mathrm{BO}}$ 为 BO 迭代评估次数；
- $K_q$ 为每个方向配置进入局部抛光的候选数；
- $N_{\mathrm{DFO}}$ 为每个候选的局部抛光评估预算；
- $K_{\mathrm{HF}}$ 为最终高保真复验候选数。

则仿真评估次数大致为

$$
N_{\mathrm{eval}}
\approx
N_{\mathrm{dir}}N_{\mathrm{base}}^{\mathrm{LF}}
+
K_s(N_{\mathrm{init}}+N_{\mathrm{BO}})
+
K_sK_qN_{\mathrm{DFO}}
+
K_{\mathrm{HF}}.
$$

其中第一项通常为低保真评估，成本远低于完整仿真。总计算成本由下层 Bellman--守恒律仿真决定。如果一次完整仿真成本为 $C_{\mathrm{sim}}^{\mathrm{HF}}$，一次低保真仿真成本为 $C_{\mathrm{sim}}^{\mathrm{LF}}$，则总成本约为

$$
C_{\mathrm{total}}
\approx
N_{\mathrm{dir}}N_{\mathrm{base}}C_{\mathrm{sim}}^{\mathrm{LF}}
+
\left[K_s(N_{\mathrm{init}}+N_{\mathrm{BO}})+K_sK_qN_{\mathrm{DFO}}+K_{\mathrm{HF}}\right]C_{\mathrm{sim}}^{\mathrm{HF/MF}}.
$$

因此，外层低保真筛选的核心意义是减少进入高成本内层优化的方向配置数量。

---

## 13. 推荐默认参数

以下参数适合作为第一版G5 实验的起点，后续可根据计算资源调整。

| 参数 | 推荐值 | 说明 |
|---|---:|---|
| 时间分段数 $L$ | 2--4 | 先从低维容量控制开始 |
| 入围方向数 $K_s$ | 5--20 | 取决于 $C$ 和计算预算 |
| 每个 $s$ 的初始 BO 点数 $N_{\mathrm{init}}$ | $2d_s+5$ 或 10--30 | 维度越高越需要更多初始点 |
| 每个 $s$ 的 BO 迭代数 $N_{\mathrm{BO}}$ | 20--80 | 初版可用 20--30 |
| 局部抛光候选数 $K_q$ | 1--3 | 对每个方向配置只抛光少量最好候选 |
| DFO 预算 $N_{\mathrm{DFO}}$ | 10--50 | 根据仿真成本调整 |
| 高保真复验候选数 $K_{\mathrm{HF}}$ | 5--20 | 覆盖不同方向配置和不同权衡 |
| BO 停滞轮数 $r_{\mathrm{stall}}$ | 5--10 | 连续无改进则停止 |
| LCB 探索系数 $\kappa$ | 1--3 | 值越大越偏探索 |
| 平滑性权重 $\lambda_R$ | 0 或小正数 | 若现场要求平滑切换则开启 |
| 入口等待权重 $\lambda_B$ | 建议开启 | 防止过度限流 |

若 $C$ 较小且 $L=1$，可以先做较完整的 $s$ 枚举和 $q$ 网格/随机搜索，作为 BO 结果的验证基线。若 $C$ 和 $L$ 较大，应优先使用结构化候选集和 BO/DFO。

---

## 14. 实验基线与消融对比

为了说明优化确实有效，实验中至少应包含以下基线：

| 基线 | 说明 |
|---|---|
| 全双向开放 + 最大容量 | 不做方向限制、不做限流的自然基线 |
| 管理经验方案 | 例如固定若干通道只上或只下 |
| 固定高 $q$ | 所有允许方向取高容量 |
| 固定中 $q$ | 所有允许方向取中容量 |
| 固定低 $q$ | 强限流基线，用于展示入口等待惩罚必要性 |
| 只优化 $s$ | $q$ 固定，只搜索方向配置 |
| 只优化 $q$ | $s$ 固定，只优化入口速率 |
| 随机搜索 | 混合变量随机采样可行候选 |
| 网格搜索 | 低维情况下对 $q$ 做粗网格枚举 |
| BO 无 DFO | 检查局部抛光是否带来改进 |
| 无低保真筛选 | 检查筛选机制是否节省计算且不明显损害最优性 |
| 不含 $J_B$ | 展示入口等待惩罚对避免过度限流的重要性 |

消融实验应报告每种方法的最佳目标值、评估次数、最优方向配置、容量曲线、各指标分项和通道实际流量分布。

---

## 15. 输出文件与诊断内容

一次完整优化建议输出以下文件：

| 文件 | 内容 |
|---|---|
| `G5_evaluation_log.csv` | 每次仿真的 $s,q,J$ 与分项指标 |
| `G5_top_candidates.csv` | 高保真复验后的候选排行榜 |
| `G5_method_comparison.csv` | 基线、随机搜索、BO、BO+DFO 等方法对比 |
| `G5_config_summary.json` | 优化参数、权重、容量上界、时间分段等配置 |
| `G5_best_control.json` | 最优 $s^*$ 与 $q^*$ 的机器可读版本 |
| `G5_capacity_profiles.png` | 最优方案的入口速率时间曲线 |
| `G5_flux_share.png` | 各通道实际累计通过量占比 |
| `G5_objective_trace.png` | 优化过程中的 best-so-far 曲线 |
| `G5_pareto_j1_j2.png` | 候选在效率--安全平面上的分布 |
| `G5_density_snapshots/` | 关键时刻密度场快照 |
| `G5_report.md` | 面向论文或项目汇报的文字总结 |

诊断内容至少应包括：

1. 最优方案的方向配置；
2. 每条通道、每个时间段的 $q_{c,\ell}^\pm$；
3. 每条通道实际通过量 $R_c$ 与占比 $p_c$；
4. 入口等待区积压曲线 $B_c^\pm(t)$；
5. 高密度暴露区域随时间变化；
6. 目标函数分项贡献；
7. 与基线方案相比的改进幅度；
8. 权重变化后的排名稳定性。

---

## 16. 实现注意事项

### 16.1 不要混淆三类流量

实现中必须区分：

1. $q_c^\pm(t)$：优化器给出的入口允许速率上限；
2. $A_c^\pm(t)$：当前密度和速度自然产生的尝试进入流量；
3. $\widehat A_c^\pm(t)$：经过入口速率约束后的实际允许通过量。

优化变量是 $q$，通道负载均衡指标应基于实际通过量 $\widehat A$ 积分得到的 $R_c$，而不是基于 $q$ 本身。

### 16.2 硬约束必须写死

方向--速率一致性、关闭通道容量为零、单向通道反向容量为零、双向总容量上界等都是已知硬约束。推荐通过变量映射 $\mathcal T_s$ 保证，而不是在目标函数中加入大惩罚。

### 16.3 不建议把 $s$ 连续松弛后再取整

$s_c$ 是规则型离散变量，其取值改变的是允许方向集合 $U(x;s)$，会导致 Bellman 最优路径结构发生离散变化。把 $s_c$ 临时编码为连续数再四舍五入，通常会使代理模型看到大量不真实的中间状态，也可能导致采集函数优化退化。更稳妥的做法是外层枚举、筛选或离散邻域搜索。

### 16.4 低保真只能用于筛选，不能作为最终结论

粗网格或简化时间步可能改变拥堵峰值、入口等待和通道流量分配。因此，低保真结果只能用于减少候选数量；最终排序必须由高保真仿真给出。

### 16.5 入口等待惩罚很重要

若只优化区域内部安全指标 $\tilde J_2$，优化器可能通过降低 $q$ 把人群阻挡在入口外侧，从而降低内部密度。这种方案在管理上不可接受。因此只要 $q$ 是优化变量，就应记录并通常纳入 $\tilde J_B$。

### 16.6 最优方案应以候选集形式报告

由于权重和需求场景会影响最优解，最终报告不应只给一个方案。建议至少报告综合最优、安全优先最优、效率优先最优和入口等待较低的稳健候选，方便管理方按现场偏好选择。

---




## 18. 当前论文/项目中的推荐表述

可以在论文或项目报告中这样概括本章方法：

> 本文将基于内部入口速率约束的通道管控设计形式化为混合整数 PDE 约束优化问题。离散变量 $s$ 描述通道方向规则，连续变量 $q$ 描述各内部入口在不同时间段的允许通行速率。给定 $s$ 与 $q$ 后，下层 Bellman--守恒律模型计算人群密度演化、最优行走方向和受限入口通量，并输出效率、安全、通道负载均衡、入口等待和管控平滑性指标。由于目标函数由高代价、非光滑的动态仿真隐式给出，本文采用分层无导数优化框架：首先对方向配置进行硬约束过滤和低保真筛选，然后在固定方向配置下使用约束贝叶斯优化搜索入口速率，最后对少数优良候选进行无导数局部抛光和高保真复验。该方法将方向规则、入口容量控制和几何引导机制分离，既保持了模型物理解释，又提高了优化搜索的可实施性。

---

## 19. 最小可运行实验建议

若需要快速完成第一版G5 实验，可采用以下最小设置：

1. 选择 $C=4$ 条主要通道；
2. 设置 $L=2$ 个时间段，例如前半程和后半程；
3. 枚举或采样 20--50 个合法方向配置 $s$；
4. 对每个 $s$ 用高、中、低三种容量模式做低保真筛选；
5. 保留前 $K_s=5$ 个方向配置；
6. 对每个入围 $s$ 做 $N_{\mathrm{init}}=10$ 个初始点和 $N_{\mathrm{BO}}=20$ 次 BO 迭代；
7. 对每个 $s$ 的最好 1 个候选做 $N_{\mathrm{DFO}}=10$ 次局部抛光；
8. 汇总全部候选，选前 10 个做高保真复验；
9. 与全双向开放、管理经验方案、只优化 $s$、只优化 $q$ 和随机搜索对比；
10. 输出最优方向配置、容量曲线、密度快照、通道流量占比和目标分项表。

该最小实验能够验证 V2 入口速率控制是否确实改善拥堵分布，并能展示 $q$ 相对于旧版 $\eta$ 优化的直接管控意义。

---

## 20. 小结

V2 优化问题的关键不是单纯更换优化器，而是将管控机制重新分解为三层：

1. **方向规则层**：由 $s$ 决定通道允许方向；
2. **入口容量层**：由 $q$ 决定内部入口允许通行速率；
3. **几何引导层**：由固定 $M(x;\eta_0)$ 决定通道内部的路径偏好。

在这一分解下，优化问题可清晰写为

$$
\min_{(s,q)\in\mathcal Z}
\lambda_1\tilde J_1
+
\lambda_2\tilde J_2
+
\lambda_5\tilde J_5
+
\lambda_B\tilde J_B
+
\lambda_R\tilde J_R,
$$

其中 $\mathcal Z$ 显式包含所有方向--速率一致性和容量约束。推荐算法采用“外层 $s$ 筛选 + 内层 $q$ 的约束 BO + DFO/NOMAD 抛光 + 高保真复验”的结构。这样既能适应 Bellman--守恒律仿真的非光滑黑盒特征，又能保证候选方案始终满足现场管控规则。
