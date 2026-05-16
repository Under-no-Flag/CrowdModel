# V2 模型：基于内部通道入口通行速率约束的 Bellman--守恒律人群管控模型

> 本版模型将几何引导强度 $\eta$ 从优化变量中删除。度量张量 $M(x)$ 仍保留为**通道内部几何引导机制**，但不承担精确流量控制功能。新的主要管控变量为通道方向配置 $s$ 和内部通道入口通行速率 $q$。

---

## 0. 符号、索引与约定

### 0.1 时间、空间与网格符号

| 符号 | 含义 |
|---|---|
| $T$ | 仿真或优化时间上限。 |
| $t\in[0,T]$ | 连续时间变量。 |
| $t^n=n\Delta t$ | 第 $n$ 个离散时间层。 |
| $\Delta t$ | 时间步长。 |
| $\Omega\subset\mathbb R^2$ | 计算域。 |
| $\Omega_w\subset\Omega$ | 可通行区域，障碍物、墙体等不可行区域不属于 $\Omega_w$。 |
| $\partial\Omega$ | 计算域外边界。本文的通道入口截面通常不在 $\partial\Omega$ 上，而是内部界面。 |
| $x=(x_1,x_2)$ | 空间位置变量。 |
| $K_i$ | 有限体积网格单元。 |
| $\|K_i\|$ | 网格单元 $K_i$ 的面积。 |
| $f\subset\partial K_i$ | 网格单元边界上的一个面或边。为避免与速度函数混淆，本文在有限体积部分用 $f$ 表示网格面。 |
| $\mathcal F_c^\pm$ | 第 $c$ 条通道正向或反向入口截面对应的离散内部网格面集合。 |
| $\Delta x$ | 网格空间步长；在 Bellman 离散方程中表示一次步进的基本长度。 |
| $dS$ | 沿截面或边界的线积分测度。 |

### 0.2 通道与方向符号

| 符号 | 含义 |
|---|---|
| $C$ | 通道总数。 |
| $c=1,\dots,C$ | 通道编号。 |
| $\Omega_c\subset\Omega_w$ | 第 $c$ 条通道区域。 |
| $\chi_c(x)$ | 第 $c$ 条通道区域指示函数，若 $x\in\Omega_c$ 则 $\chi_c(x)=1$，否则为 $0$。 |
| $\tau_c(x)$ | 第 $c$ 条通道的参考切向单位向量场。$+\tau_c$ 称为通道正向，$-\tau_c$ 称为通道反向。 |
| $\nu_c(x)$ | 与 $\tau_c(x)$ 垂直的通道横向单位向量，用于构造度量张量。为避免混淆，本文用 $\nu_c$ 表示横向法向，而不用 $n_c$。 |
| $\Sigma_c^+$ | 第 $c$ 条通道的正向入口截面，即允许沿 $+\tau_c$ 进入通道的内部截面。 |
| $\Sigma_c^-$ | 第 $c$ 条通道的反向入口截面，即允许沿 $-\tau_c$ 进入通道的内部截面。 |
| $e_c^+(x)$ | 在 $\Sigma_c^+$ 上指向通道内部、与 $+\tau_c$ 一致的穿越方向。 |
| $e_c^-(x)$ | 在 $\Sigma_c^-$ 上指向通道内部、与 $-\tau_c$ 一致的穿越方向。 |
| $W_c^+$ | 第 $c$ 条通道正向入口外侧的等待或排队诊断区域。 |
| $W_c^-$ | 第 $c$ 条通道反向入口外侧的等待或排队诊断区域。 |

说明：原文中同时使用 $n_c$ 表示通道横向法向，又使用 $n_c^\pm$ 表示入口截面穿越方向，容易造成歧义。本文将**通道横向法向**记为 $\nu_c$，将**入口截面穿越方向**记为 $e_c^\pm$。

### 0.3 群体、阶段与路线符号

| 符号 | 含义 |
|---|---|
| $\sigma$ | 行为阶段编号，例如进场、游览、离场或返回。 |
| $r$ | 某一阶段内的路线或目标类型编号。 |
| $g=(\sigma,r)$ | 一个阶段--路线子群体。 |
| $\mathcal G$ | 所有子群体的集合。 |
| $\rho_g(x,t)$ | 子群体 $g$ 在位置 $x$、时间 $t$ 的密度。 |
| $\rho(x,t)$ | 总密度，定义为 $\rho(x,t)=\sum_{g\in\mathcal G}\rho_g(x,t)$。 |
| $\hat p$ | 固定的外生行为偏好参数，由历史数据或经验设定得到，在本优化问题中不作为控制变量。 |
| $G_g$ | 子群体 $g$ 当前阶段的完成目标区域。 |
| $\chi_{G_g}(x)$ | 目标区域 $G_g$ 的指示函数。 |
| $\kappa_g$ | 子群体 $g$ 在目标区域内发生阶段切换的速率。 |
| $p_{g\to h}(x,t)$ | 子群体 $g$ 转入下一子群体 $h$ 的路线选择概率。可取固定概率，也可取 Logit 型概率。 |
| $Q_{g\to h}(x,t)$ | 从子群体 $g$ 转入子群体 $h$ 的源项。 |
| $S_g(\rho;\hat p)$ | 子群体 $g$ 的总源汇项，包括流入该群体的阶段转移项和流出该群体的阶段转移项。 |

### 0.4 速度、势函数与通量符号

| 符号 | 含义 |
|---|---|
| $v_{\max}$ | 自由流最大速度。 |
| $\rho_{\max}$ | 最大密度。 |
| $\rho_{\mathrm{safe}}$ | 安全密度阈值。 |
| $F(\rho)$ 或 $f_\rho(\rho)$ | 速度--密度关系。为避免和网格面 $f$ 混淆，若需要可记为 $V(\rho)$。本文仍写 $F(\rho)=v_{\max}(1-\rho/\rho_{\max})_+$。 |
| $\phi_g(x,t)$ | 子群体 $g$ 的势函数，表示到当前目标的剩余最小代价或时间。 |
| $U_g(x;s)$ | 子群体 $g$ 在位置 $x$ 的允许方向集合，由通道方向配置 $s$ 决定。 |
| $u_g^*(x,t)$ | 子群体 $g$ 的最优行走方向。 |
| $v_g(x,t)$ | 子群体 $g$ 的速度场，$v_g=F(\rho)u_g^*$。 |
| $M_g(x)$ | 子群体 $g$ 使用的空间度量张量。本文保留为几何引导机制，但不再把其强度作为优化变量。 |
| $A_c^\pm(t)$ | 第 $c$ 条通道正向或反向入口的总尝试流量。 |
| $\widehat A_c^\pm(t)$ | 第 $c$ 条通道正向或反向入口的实际允许通过量。 |
| $q_c^\pm(t)$ | 第 $c$ 条通道正向或反向入口的控制速率上限。 |
| $\bar q_c^\pm$ | 第 $c$ 条通道正向或反向入口的最大可实施通行能力。 |
| $F_{g,f}^{\mathrm{free},n}$ | 第 $n$ 个时间层、群体 $g$ 在离散网格面 $f$ 上的未受限自由数值通量。 |
| $\widehat F_{g,f}^{n}$ | 施加入口速率约束后的实际数值通量。 |
| $\varepsilon_{\mathrm{cap}}$ | 防止除零的极小正常数，只用于通量缩放系数。 |

注意：本文同时存在三类“流量/通量”量，需要区分：

1. $q_c^\pm(t)$ 是**控制变量**，表示允许通行速率上限；
2. $A_c^\pm(t)$ 是**尝试通过量**，由人群密度和速度自然产生；
3. $\widehat A_c^\pm(t)$ 是**实际通过量**，满足 $\widehat A_c^\pm(t)\le q_c^\pm(t)$。

---

## 1. 控制变量

本版将几何引导强度 $\eta$ 从优化变量中移出，并作为固定参数 $\eta_0$ 保留在度量张量中；其弱敏感性需要由后续实验验证。新的控制变量定义为

$$
z=(s,q),
$$

其中

$$
s=(s_1,\dots,s_C)
$$

表示通道方向配置；

$$
q=\{q_c^+(t),q_c^-(t)\}_{c=1}^C
$$

表示每条通道两个方向入口上的允许通行速率。

与旧版优化变量

$$
z_{\mathrm{old}}=(s,\eta)
$$

相比，V2 模型的优化变量为

$$
z_{\mathrm{v2}}=(s,q).
$$

其中 $s$ 仍表示通道方向配置，$q$ 表示内部入口通行能力；$\eta$ 不再参与搜索，而是固定为 $\eta_0$，只承担通道内部几何引导作用。这样可以把“方向规则”“入口容量控制”和“几何引导”三类机制分开：$s$ 决定哪些方向允许通行，$q$ 决定入口每单位时间允许通过多少人，$M(x;\eta_0)$ 只影响已允许方向内部的几何偏好。

### 1.1 方向配置变量

对第 $c$ 条通道，方向变量定义为

$$
s_c\in\{+1,-1,0,\varnothing\}.
$$

其含义如下：

| 取值 | 含义 | 对允许方向集合的影响 |
|---|---|---|
| $s_c=+1$ | 第 $c$ 条通道只允许沿 $+\tau_c$ 方向通行 | $U_c(x;s_c)=\{+\tau_c(x)\}$ |
| $s_c=-1$ | 第 $c$ 条通道只允许沿 $-\tau_c$ 方向通行 | $U_c(x;s_c)=\{-\tau_c(x)\}$ |
| $s_c=0$ | 第 $c$ 条通道双向自由通行 | $U_c(x;s_c)=\{+\tau_c(x),-\tau_c(x)\}$ 或离散方向集合中的可行双向步进 |
| $s_c=\varnothing$ | 第 $c$ 条通道关闭 | $U_c(x;s_c)=\varnothing$ |

在非通道区域，$U_g(x;s)$ 保持为普通可行方向集合；在通道区域 $\Omega_c$ 内，$U_g(x;s)$ 根据 $s_c$ 修改。

### 1.2 入口速率变量

第 $c$ 条通道的正向和反向入口速率分别为

$$
q_c^+(t)\in[0,\bar q_c^+],
\qquad
q_c^-(t)\in[0,\bar q_c^-].
$$

其中：

- $q_c^+(t)$：允许从第 $c$ 条通道的负端入口进入，并沿 $+\tau_c$ 方向通过的最大速率；
- $q_c^-(t)$：允许从第 $c$ 条通道的正端入口进入，并沿 $-\tau_c$ 方向通过的最大速率。

这里的 $+$ 和 $-$ 不是坐标轴方向，而是相对于通道参考切向 $\tau_c$ 而言。

### 1.3 方向与速率的一致性约束

方向配置 $s_c$ 与入口速率 $q_c^\pm(t)$ 必须满足一致性约束：

$$
s_c=+1\Rightarrow q_c^-(t)=0,
$$

$$
s_c=-1\Rightarrow q_c^+(t)=0,
$$

$$
s_c=\varnothing\Rightarrow q_c^+(t)=q_c^-(t)=0.
$$

若 $s_c=0$，即双向通行，则允许两个方向都有流量，但建议加入总容量约束：

$$
q_c^+(t)+q_c^-(t)\le \bar q_c.
$$

这样，$s_c$ 负责“允许什么方向通行”，$q_c^\pm(t)$ 负责“每个方向每单位时间最多允许多少人通过”。

### 1.4 分段常数参数化

为避免连续控制函数维度过高，令 $q_c^\pm(t)$ 在给定时间分段上保持常数。设

$$
0=t_0<t_1<\cdots<t_L=T,
$$

则

$$
q_c^\pm(t)=q_{c,\ell}^\pm,
\qquad
 t\in[t_\ell,t_{\ell+1}).
$$

因此，连续控制变量变成有限维向量

$$
q=\{q_{c,\ell}^+,q_{c,\ell}^-\}_{c=1,\dots,C;\,\ell=0,\dots,L-1}.
$$

例如四条通道、四个时间段时，最多有

$$
4\times 2\times 4=32
$$

个速率变量；若某些通道被设置为单向或关闭，则实际自由变量数量会进一步减少。

---

## 2. 几何定义：内部通道入口

设整个可行域为

$$
\Omega_w\subset\Omega.
$$

第 $c$ 条通道区域为

$$
\Omega_c\subset\Omega_w.
$$

每条通道给定参考切向单位向量场

$$
\tau_c(x),
\qquad |\tau_c(x)|=1.
$$

如果通道近似为东西向，则可以理解为

$$
+\tau_c:\text{东向},
\qquad
-\tau_c:\text{西向}.
$$

但数学上不要求通道必须东西向，只要求每条通道给定一个参考正向。

与通道切向垂直的横向单位向量记为

$$
\nu_c(x),
\qquad
\tau_c(x)\cdot \nu_c(x)=0,
\qquad
|\nu_c(x)|=1.
$$

对第 $c$ 条通道，定义两个内部入口截面：

$$
\Sigma_c^+,
\qquad
\Sigma_c^-.
$$

其中：

- $\Sigma_c^+$：沿 $+\tau_c$ 进入通道的入口截面；
- $\Sigma_c^-$：沿 $-\tau_c$ 进入通道的入口截面。

注意：

$$
\Sigma_c^\pm\not\subset\partial\Omega.
$$

它们是场景内部的曲线或网格面集合，位于“通道外部区域”和“通道内部区域”之间。因此，通道入口速率控制不是外边界入流条件，而是内部界面通量约束。

在两个入口截面上分别定义穿越方向

$$
e_c^+(x),\qquad x\in\Sigma_c^+,
$$

$$
e_c^-(x),\qquad x\in\Sigma_c^-.
$$

其中 $e_c^+$ 指向通道内部并与 $+\tau_c$ 通行方向一致，$e_c^-$ 指向通道内部并与 $-\tau_c$ 通行方向一致。

因此，通道入口通行速率控制作用在以下截面通量上：

$$
\int_{\Sigma_c^\pm}\rho_g v_g\cdot e_c^\pm\,dS,
$$

而不是作用在外边界 $\partial\Omega$ 上。

---

## 3. 连续模型

### 3.1 多阶段、多路线群体

记一个阶段--路线子群体为

$$
g=(\sigma,r),
$$

其中 $\sigma$ 是阶段编号，$r$ 是该阶段内的路线或目标类型编号。所有子群体组成集合 $\mathcal G$。

子群体密度为

$$
\rho_g(x,t).
$$

总密度为

$$
\rho(x,t)=\sum_{g\in\mathcal G}\rho_g(x,t).
$$

速度--密度关系采用 Greenshields 型：

$$
F(\rho)=v_{\max}\left(1-\frac{\rho}{\rho_{\max}}\right)_+,
$$

其中

$$
(a)_+=\max(a,0).
$$

这里用 $F(\rho)$ 表示速度--密度函数，是为了避免与有限体积网格面符号 $f$ 混淆。若不涉及离散网格面，也可继续写作 $f(\rho)$。

### 3.2 Bellman--HJB 路径选择层

对每个群体 $g$，定义势函数

$$
\phi_g(x,t),
$$

表示子群体 $g$ 从当前位置到当前目标的最小剩余代价或时间。

在给定总密度 $\rho$、方向配置 $s$ 和固定度量张量 $M_g(x)$ 的条件下，离散 Bellman 方程写为

$$
\phi_g(x)
=
\min_{u\in U_g(x;s)}
\left[
\phi_g(x+\Delta x u)
+
\frac{\Delta x}{F(\rho(x,t))}
\frac{1}{\sqrt{u^\top M_g(x)u}}
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
\frac{1}{\sqrt{u^\top M_g(x)u}}
\right].
$$

速度场为

$$
v_g(x,t)=F(\rho(x,t))u_g^*(x,t).
$$

其中 $U_g(x;s)$ 是由通道方向配置 $s$ 决定的允许方向集合。

### 3.3 几何引导张量

$M_g(x)$ 保留为通道内部几何引导张量，但不作为优化变量。对第 $c$ 条通道，可取

$$
M_c(x)
=
\beta_c
\left(
\eta_0\,\tau_c(x)\tau_c(x)^\top
+
\nu_c(x)\nu_c(x)^\top
\right),
\qquad x\in\Omega_c.
$$

其中：

- $\beta_c>0$ 是尺度因子；
- $\eta_0\ge 1$ 是固定的各向异性强度；
- $\tau_c\tau_c^\top$ 表示沿通道方向的投影；
- $\nu_c\nu_c^\top$ 表示横向投影。

在通道外部区域，可以取

$$
M_g(x)=I,
\qquad x\notin\bigcup_{c=1}^C\Omega_c,
$$

或取其他固定背景度量。本文不再优化 $\eta_0$，其作用仅是使行人倾向于沿通道方向行走，而不是精确控制入口流量。

### 3.4 多阶段源汇项

对每个群体 $g$，密度守恒方程为

$$
\frac{\partial \rho_g}{\partial t}
+
\nabla\cdot(\rho_g v_g)
=
S_g(\rho;\hat p).
$$

其中 $S_g(\rho;\hat p)$ 表示多阶段转移和路线分流造成的源汇项。一般可写为

$$
S_g(\rho;\hat p)
=
\sum_{h\in\mathcal G} Q_{h\to g}(x,t;\hat p)
-
\sum_{h\in\mathcal G} Q_{g\to h}(x,t;\hat p).
$$

若群体 $g$ 到达阶段完成目标区域 $G_g$ 后转入群体 $h$，则可取

$$
Q_{g\to h}(x,t;\hat p)
=
p_{g\to h}(x,t;\hat p)\,\kappa_g\,\chi_{G_g}(x)\,\rho_g(x,t).
$$

其中：

- $p_{g\to h}(x,t;\hat p)$ 为路线选择概率；
- $\kappa_g$ 为阶段切换速率；
- $\chi_{G_g}(x)$ 为阶段完成目标区域的指示函数。

固定概率分流、Logit 分流和拥堵修正分流均可由 $p_{g\to h}$ 的不同形式实现。本文的入口速率控制 $q$ 不改变阶段转移方程本身，而是在内部通道入口截面上修改实际数值通量。

---

## 4. 内部入口速率控制的连续形式

### 4.1 尝试通量

对第 $c$ 条通道的正向入口截面 $\Sigma_c^+$，群体 $g$ 的正向入口尝试通量密度为

$$
j_{g,c}^+(x,t)
=
\rho_g(x,t)v_g(x,t)\cdot e_c^+(x),
\qquad x\in\Sigma_c^+.
$$

第 $c$ 条通道正向入口的总尝试流量为

$$
A_c^+(t)
=
\sum_{g\in\mathcal G}
\int_{\Sigma_c^+}
\left(j_{g,c}^+(x,t)\right)_+\,dS.
$$

同理，在反向入口截面 $\Sigma_c^-$ 上定义

$$
j_{g,c}^-(x,t)
=
\rho_g(x,t)v_g(x,t)\cdot e_c^-(x),
\qquad x\in\Sigma_c^-,
$$

$$
A_c^-(t)
=
\sum_{g\in\mathcal G}
\int_{\Sigma_c^-}
\left(j_{g,c}^-(x,t)\right)_+\,dS.
$$

这里 $A_c^\pm(t)$ 是人群在当前密度、速度和路径选择下“想要进入通道”的总流量。

### 4.2 受控实际通过量

引入入口速率上限

$$
q_c^+(t)\in[0,\bar q_c^+],
\qquad
q_c^-(t)\in[0,\bar q_c^-].
$$

则实际允许通过量定义为

$$
\widehat A_c^+(t)=\min\{A_c^+(t),q_c^+(t)\},
$$

$$
\widehat A_c^-(t)=\min\{A_c^-(t),q_c^-(t)\}.
$$

因此，内部入口截面满足通量约束

$$
\sum_{g\in\mathcal G}
\int_{\Sigma_c^+}
\widehat j_{g,c}^+(x,t)\,dS
\le q_c^+(t),
$$

$$
\sum_{g\in\mathcal G}
\int_{\Sigma_c^-}
\widehat j_{g,c}^-(x,t)\,dS
\le q_c^-(t).
$$

### 4.3 比例缩放形式

为了在多个群体、多个网格面之间分配受限通量，可使用比例缩放。定义

$$
\theta_c^+(t)
=
\begin{cases}
\dfrac{\widehat A_c^+(t)}{A_c^+(t)}, & A_c^+(t)>0,\\[6pt]
1, & A_c^+(t)=0,
\end{cases}
$$

$$
\theta_c^-(t)
=
\begin{cases}
\dfrac{\widehat A_c^-(t)}{A_c^-(t)}, & A_c^-(t)>0,\\[6pt]
1, & A_c^-(t)=0.
\end{cases}
$$

则正向入口截面上的实际通量密度可写为

$$
\widehat j_{g,c}^+(x,t)
=
\theta_c^+(t)\left(j_{g,c}^+(x,t)\right)_+
+
\left(j_{g,c}^+(x,t)\right)_-,
$$

其中

$$
(a)_-=\min(a,0).
$$

反向入口同理：

$$
\widehat j_{g,c}^-(x,t)
=
\theta_c^-(t)\left(j_{g,c}^-(x,t)\right)_+
+
\left(j_{g,c}^-(x,t)\right)_-.
$$

该形式表示：

1. 进入通道方向的正通量受到 $q_c^\pm(t)$ 限制；
2. 反向离开该入口截面的通量不受该方向入口上限限制；
3. 被限制掉的人群质量不会消失，而是留在上游区域，从而自然形成入口前等待、排队和高密度。

---

## 5. 离散有限体积实现

对每个群体 $g$，每个网格单元 $K_i$ 的显式有限体积更新写为

$$
\rho_{g,i}^{n+1}
=
\rho_{g,i}^{n}
-
\frac{\Delta t}{|K_i|}
\sum_{f\subset\partial K_i}
\widehat F_{g,f}^{n}
+
\Delta t\,S_{g,i}^{n}.
$$

其中：

- $\rho_{g,i}^{n}$ 是第 $n$ 个时间层中群体 $g$ 在单元 $K_i$ 上的平均密度；
- $S_{g,i}^n$ 是源汇项在单元 $K_i$ 上的离散值；
- $\widehat F_{g,f}^{n}$ 是施加内部入口速率约束后的实际数值通量；
- 对于非受控网格面，$\widehat F_{g,f}^{n}=F_{g,f}^{\mathrm{free},n}$。

### 5.1 正向入口截面上的通量约束

设 $\mathcal F_c^+$ 是 $\Sigma_c^+$ 对应的离散内部网格面集合。对每个 $f\in\mathcal F_c^+$，约定 $F_{g,f}^{\mathrm{free},n}$ 的正方向为“从入口外侧进入通道内部并沿 $+\tau_c$ 方向通过”。

总尝试进入通量为

$$
A_c^{+,n}
=
\sum_{g\in\mathcal G}
\sum_{f\in\mathcal F_c^+}
\left(F_{g,f}^{\mathrm{free},n}\right)_+.
$$

实际允许通量为

$$
\widehat A_c^{+,n}
=
\min\left\{A_c^{+,n},q_c^{+,n}\right\}.
$$

缩放系数为

$$
\lambda_c^{+,n}
=
\frac{\widehat A_c^{+,n}}{A_c^{+,n}+\varepsilon_{\mathrm{cap}}}.
$$

其中 $\varepsilon_{\mathrm{cap}}>0$ 是防止除零的极小正常数。当 $A_c^{+,n}=0$ 时，也可以直接令 $\lambda_c^{+,n}=1$。

受限通量为

$$
\widehat F_{g,f}^{n}
=
\lambda_c^{+,n}
\left(F_{g,f}^{\mathrm{free},n}\right)_+
+
\left(F_{g,f}^{\mathrm{free},n}\right)_-,
\qquad f\in\mathcal F_c^+.
$$

### 5.2 反向入口截面上的通量约束

设 $\mathcal F_c^-$ 是 $\Sigma_c^-$ 对应的离散内部网格面集合。对每个 $f\in\mathcal F_c^-$，约定 $F_{g,f}^{\mathrm{free},n}$ 的正方向为“从入口外侧进入通道内部并沿 $-\tau_c$ 方向通过”。

总尝试进入通量为

$$
A_c^{-,n}
=
\sum_{g\in\mathcal G}
\sum_{f\in\mathcal F_c^-}
\left(F_{g,f}^{\mathrm{free},n}\right)_+.
$$

实际允许通量为

$$
\widehat A_c^{-,n}
=
\min\left\{A_c^{-,n},q_c^{-,n}\right\}.
$$

缩放系数为

$$
\lambda_c^{-,n}
=
\frac{\widehat A_c^{-,n}}{A_c^{-,n}+\varepsilon_{\mathrm{cap}}}.
$$

受限通量为

$$
\widehat F_{g,f}^{n}
=
\lambda_c^{-,n}
\left(F_{g,f}^{\mathrm{free},n}\right)_+
+
\left(F_{g,f}^{\mathrm{free},n}\right)_-,
\qquad f\in\mathcal F_c^-.
$$

### 5.3 守恒性说明

对同一个内部网格面 $f$，相邻两个网格单元必须使用同一个受限通量 $\widehat F_{g,f}^{n}$，只是在两个单元的更新式中符号相反。这样，被入口速率约束截留的质量不会消失，而是留在入口上游单元内。因此，该方案是内部瓶颈限流，而不是外部边界注入或删除质量。

### 5.4 离散单位与代码实现口径

连续定义中的 $q_c^\pm(t)$、$A_c^\pm(t)$ 和 $\widehat A_c^\pm(t)$ 的单位均为“人/时间”。在有限体积离散中，需要区分两种实现口径。

若 $F_{g,f}^{\mathrm{free},n}$ 表示已经对网格面长度积分后的面通量，则上文中的

$$
A_c^{+,n}
=
\sum_{g\in\mathcal G}
\sum_{f\in\mathcal F_c^+}
\left(F_{g,f}^{\mathrm{free},n}\right)_+
$$

可以直接与 $q_c^{+,n}$ 比较。

若代码中的面通量数组存储的是通量密度，即

$$
F_{g,f}^{\mathrm{free},n}\approx \rho_g v_g\cdot n_f,
$$

其单位为“人/(长度·时间)”，则入口尝试通过率应写为

$$
A_c^{+,n}
=
\sum_{g\in\mathcal G}
\sum_{f\in\mathcal F_c^+}
|f|
\left(F_{g,f}^{\mathrm{free},n}\right)_+.
$$

在均匀方格网格中，内部边长度 $|f|=\Delta x$，因此可实现为

$$
A_c^{+,n}
=
\Delta x
\sum_{g\in\mathcal G}
\sum_{f\in\mathcal F_c^+}
\left(F_{g,f}^{\mathrm{free},n}\right)_+.
$$

当前代码中的 `compute_face_fluxes()` 返回的是 `rho * velocity` 型面通量密度，并在密度更新中通过除以 `dx` 形成散度。因此，若沿用该口径，计算入口尝试通过率和通道累计通过量时应乘以面长 `dx`；但限流系数 $\lambda_c^{\pm,n}$ 仍可直接乘回原始面通量密度数组。反向入口 $\mathcal F_c^-$ 同理。

---

## 6. 排队和入口积压诊断量

由于通道入口是内部截面，被速率约束挡住的人仍然留在计算域内。因此通常不需要额外引入外部队列状态变量。

但是，为了评价限流是否把拥堵转移到入口前区域，可以定义等待区质量作为诊断量。对第 $c$ 条通道正向入口，定义

$$
B_c^+(t)
=
\int_{W_c^+}\rho(x,t)\,dx,
$$

其中 $W_c^+$ 是 $\Sigma_c^+$ 外侧的等待区或排队诊断区域。

同理，反向入口等待区质量为

$$
B_c^-(t)
=
\int_{W_c^-}\rho(x,t)\,dx.
$$

入口积压指标定义为

$$
J_B
=
\int_0^T
\sum_{c=1}^C
\left[B_c^+(t)+B_c^-(t)\right]dt.
$$

该指标反映限流带来的入口前滞留代价。若只优化区域内部高密度暴露，而不惩罚入口等待区积压，优化器可能倾向于过度降低 $q_c^\pm(t)$，从而得到不可实施的管控方案。因此建议在优化目标中加入 $J_B$。

---

## 7. 管控评价指标与优化目标

### 7.1 效率指标

系统总旅行时间定义为

$$
J_1(s,q\mid\hat p)
=
\int_0^T\int_{\Omega_w}\rho(x,t)\,dx\,dt.
$$

### 7.2 安全指标

高密度暴露时间定义为

$$
J_2(s,q\mid\hat p)
=
\int_0^T\int_{\Omega_w}
\mathbf 1_{\rho(x,t)>\rho_{\mathrm{safe}}}\,dx\,dt.
$$

其中 $\rho_{\mathrm{safe}}$ 是安全密度阈值。

### 7.3 通道负载均衡指标

为了避免与控制变量 $q_c^\pm(t)$ 混淆，本文将第 $c$ 条通道的实际累计通过量记为

$$
R_c
=
\int_0^T
\left[\widehat A_c^+(t)+\widehat A_c^-(t)\right]dt.
$$

其中 $R_c$ 是实际通过通道入口的累计流量，不是控制上限。通道负载不均衡指标定义为

$$
J_5(s,q\mid\hat p)
=
\operatorname{Var}(R_1,\dots,R_C).
$$

也可以使用标准化形式：

$$
\tilde J_5
=
\frac{C^2}{C-1}\operatorname{Var}(p_1,\dots,p_C),
\qquad
p_c=\frac{R_c}{\sum_{d=1}^C R_d}.
$$

### 7.4 入口等待标准化指标

设

$$
M_{\mathrm{tot}}=M_{\mathrm{init}}+M_{\mathrm{in}},
$$

其中 $M_{\mathrm{init}}$ 是初始总质量，$M_{\mathrm{in}}$ 是仿真期内外部进入系统的总质量。若当前场景没有外部边界入流，则可令 $M_{\mathrm{in}}=0$。

入口等待指标的标准化形式为

$$
\tilde J_B
=
\frac{1}{M_{\mathrm{tot}}T}
\int_0^T
\sum_{c=1}^C
\left[B_c^+(t)+B_c^-(t)\right]dt.
$$

### 7.5 可选的管控平滑性指标

若希望避免入口速率在相邻时间段剧烈变化，可加入平滑性指标

$$
J_R
=
\sum_{c=1}^C\sum_{\ell=0}^{L-2}
\left[
(q_{c,\ell+1}^+-q_{c,\ell}^+)^2
+
(q_{c,\ell+1}^--q_{c,\ell}^-)^2
\right].
$$

该项不是物理状态指标，而是管控动作可实施性惩罚项。

为了与其他目标项一起进入加权和，建议使用无量纲标准化形式。设

$$
\bar q_c^+>0,\qquad \bar q_c^->0
$$

为两个方向的容量上界，则可取

$$
\tilde J_R
=
\frac{1}{2C\max(L-1,1)}
\sum_{c=1}^C\sum_{\ell=0}^{L-2}
\left[
\left(
\frac{q_{c,\ell+1}^+-q_{c,\ell}^+}
{\bar q_c^+}
\right)^2
+
\left(
\frac{q_{c,\ell+1}^--q_{c,\ell}^-}
{\bar q_c^-}
\right)^2
\right].
$$

当某个方向因单向或关闭约束使 $\bar q_c^\pm=0$ 时，该方向不计入平滑性项，或在代码实现中使用一个极小正数仅用于避免除零。若只设置一个时间段 $L=1$，则定义 $\tilde J_R=0$。

### 7.6 优化目标

新的标量化优化问题可以写为

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

若暂不考虑平滑性，可取 $\lambda_R=0$，即

$$
J(s,q\mid\hat p)
=
\lambda_1\tilde J_1
+
\lambda_2\tilde J_2
+
\lambda_5\tilde J_5
+
\lambda_B\tilde J_B.
$$

其中 $\mathcal Z$ 是可行控制集合，包含方向取值约束、速率上下界约束、方向--速率一致性约束和分段常数参数化约束。

代码配置层面建议将 $\lambda_B$ 和 $\lambda_R$ 设计为可选权重，例如 `lambda_jb` 与 `lambda_jr`。两者默认值均取 0，以保证旧实验在不配置入口等待惩罚和平滑性惩罚时保持原有 `J1/J2/J5` 口径；当进入 G4-v2 容量控制实验时，再显式开启 `lambda_jb`，并视管控动作是否需要平滑切换决定是否开启 `lambda_jr`。

---

## 8. 完整模型汇总

### 8.1 控制变量

$$
z=(s,q),
$$

$$
s_c\in\{+1,-1,0,\varnothing\},
\qquad c=1,\dots,C,
$$

$$
q_c^\pm(t)\in[0,\bar q_c^\pm],
\qquad c=1,\dots,C.
$$

方向一致性约束为

$$
s_c=+1\Rightarrow q_c^-(t)=0,
$$

$$
s_c=-1\Rightarrow q_c^+(t)=0,
$$

$$
s_c=\varnothing\Rightarrow q_c^+(t)=q_c^-(t)=0.
$$

若 $s_c=0$，则

$$
q_c^+(t)+q_c^-(t)\le \bar q_c.
$$

### 8.2 总密度

$$
\rho(x,t)
=
\sum_{g\in\mathcal G}\rho_g(x,t).
$$

### 8.3 Bellman 路径选择

$$
\phi_g(x)
=
\min_{u\in U_g(x;s)}
\left[
\phi_g(x+\Delta xu)
+
\frac{\Delta x}{F(\rho)}
\frac{1}{\sqrt{u^\top M_g(x)u}}
\right].
$$

$$
u_g^*(x,t)
=
\operatorname*{arg\,min}_{u\in U_g(x;s)}
\left[
\phi_g(x+\Delta xu)
+
\frac{\Delta x}{F(\rho)}
\frac{1}{\sqrt{u^\top M_g(x)u}}
\right].
$$

$$
v_g(x,t)=F(\rho(x,t))u_g^*(x,t).
$$

### 8.4 守恒方程

$$
\frac{\partial \rho_g}{\partial t}
+
\nabla\cdot(\rho_gv_g)
=
S_g(\rho;\hat p).
$$

### 8.5 内部通道入口约束

$$
A_c^+(t)
=
\sum_{g\in\mathcal G}
\int_{\Sigma_c^+}
(\rho_gv_g\cdot e_c^+)_+\,dS,
$$

$$
A_c^-(t)
=
\sum_{g\in\mathcal G}
\int_{\Sigma_c^-}
(\rho_gv_g\cdot e_c^-)_+\,dS.
$$

$$
\widehat A_c^+(t)
=
\min\{A_c^+(t),q_c^+(t)\},
$$

$$
\widehat A_c^-(t)
=
\min\{A_c^-(t),q_c^-(t)\}.
$$

等价地，实际数值通量在内部入口面上由自由通量替换为受限通量。

### 8.6 有限体积更新

$$
\rho_{g,i}^{n+1}
=
\rho_{g,i}^{n}
-
\frac{\Delta t}{|K_i|}
\sum_{f\subset\partial K_i}
\widehat F_{g,f}^{n}
+
\Delta tS_{g,i}^{n}.
$$

若 $f\in\mathcal F_c^+$，则

$$
\widehat F_{g,f}^{n}
=
\lambda_c^{+,n}
(F_{g,f}^{\mathrm{free},n})_+
+
(F_{g,f}^{\mathrm{free},n})_-,
$$

其中

$$
\lambda_c^{+,n}
=
\frac{
\min\{A_c^{+,n},q_c^{+,n}\}
}{A_c^{+,n}+\varepsilon_{\mathrm{cap}}}.
$$

反向入口 $f\in\mathcal F_c^-$ 同理。

---

## 9. 论文表述建议

可以将本模型概括为：

> 本文在 Bellman--守恒律耦合的 Hughes 型宏观人群模型中，引入内部通道入口通行速率作为可实施管控变量。通道方向配置 $s$ 通过允许方向集合 $U(x;s)$ 表达单向、双向与关闭规则；固定度量张量 $M(x)$ 保留为通道内部几何引导机制；入口通行速率 $q_c^\pm(t)$ 通过内部截面通量约束限制进入通道的实际流量。该设计将方向规则、几何引导和入口限流统一到同一连续介质仿真框架中，并避免将内部通道入口误写为计算域外边界入流条件。

---

## 10. 本版相对于原 V2 的主要补充

1. 补充了完整符号表，明确区分通道切向 $\tau_c$、横向法向 $\nu_c$ 和入口截面穿越方向 $e_c^\pm$。
2. 明确区分控制速率 $q_c^\pm(t)$、尝试流量 $A_c^\pm(t)$ 和实际允许通过量 $\widehat A_c^\pm(t)$。
3. 补充了阶段--路线群体集合 $\mathcal G$、源汇项 $S_g$、转移项 $Q_{g\to h}$、目标区域 $G_g$、切换率 $\kappa_g$ 等符号定义。
4. 将通道负载累计量改记为 $R_c$，避免与控制变量 $q_c^\pm(t)$ 混淆。
5. 明确了内部入口限流的连续形式和有限体积离散形式之间的对应关系。
6. 补充了等待区质量 $B_c^\pm(t)$、入口积压指标 $J_B$ 和可选的管控平滑性指标 $J_R$。
7. 明确说明该方案是内部瓶颈通量约束，不是外边界 Neumann 入流条件。
