
# 目标
## 目标描述
在上海外滩区域，游客一般可以从多个阶梯通道登上外滩观景平台。为了避免游客只从一个通道登上平台或游览完后只从一个通道离开造成拥堵，管理方通过设置不同通道的通行方向来引导游客分散登上或离开平台。
外滩观景平台的通道从北向南依次编号为1,2,...,10分布。
游客的一般路线，一般是从5,6号通道登上观景平台，然后向南边游览，最后可能从 8,9,10等通道离开，在下街沿回到5,6号通道前的区域，再返回南京东路步行街。

管理方一般会设置4号通道只上不下，7号通道只下不上的单向通行规则，来引导游客分散登上和离开平台。

希望设计一个连续介质（宏观）人群模型，能够模拟不同通道的单向通行规则对人群流动的影响，并且能够模拟不同通道的几何引导。通过这个模型，可以评估不同管控措施对游客流动效率和安全性的影响，为管理方提供科学的决策支持。

## 对宏观模型的要求

- 能够模拟不同通道的单向通行规则对人群流动的影响
- 能够模拟不同通道的几何引导
- 能够设置游客路线，达到一个地点后前往另一个地点

## 连续介质（宏观）人群模型
### 连续性方程（密度更新）

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0$$



其中，速度 $\mathbf{v} = f(\rho) \frac{-\nabla \phi}{|\nabla \phi|}$

$f(\rho)$是速度-密度关系的函数。在Greenshields 模型中：


$f(\rho) = v_{max}(1 - \frac{\rho}{\rho_{max}})$

### 程函方程（势场更新）
$$|\nabla \phi| = \frac{1}{f(\rho)}$$



## 引入管控措施变量接口（通道通行方向控制、隔离栏位置（后续实现，现暂不考虑））

如何改进Hughes 人群宏观方程，使得可以设置某些通道（通道由几何边界定义）的可通行方向

### 引入度量张量 $\mathbf{M}$ 修改程函方程，实现通道几何引导（各向异性但不单向）：

#### 令$\mathbf{M}(x)$ 为对称正定矩阵（可随空间变化，通道内外不同。把程函方程改成各向异性 eikonal：


$$\sqrt{\nabla \phi^{\top} \mathbf{M}(x) \nabla \phi}=\frac{1}{f(\rho)}$$

对应的“最陡下降方向”也要改为$\mathbf{M}$意义下的梯度方向。所以速度场定义为：

$$\mathbf{v} = f(\rho) \mathbf{u}= f(\rho) \frac{-\mathbf{M}(x)\nabla \phi}{\sqrt{\nabla \phi^{\top} \mathbf{M}(x) \nabla \phi}}$$

连续性方程不变

#### 如何用$\mathbf{M}(x)$表达“通道”几何（让人更愿意沿通道走）

假设通道区域$\mathbf{\Omega}_c$由几何边界定义，并给出通道切向单位向量场 $\tau(x)$（沿通道方向），法向 $n(x)$（垂直通道方向）。构造：

$$\mathbf{M}(x)=\alpha(x) \tau \tau^{\top}+\beta(x) n n^{\top}, \quad \alpha \gg \beta(\text { 通道内 })$$
这会显著抬高横向“代价”，让$\phi$的最短时间路径更倾向于沿$\tau$ 前进。

### 单向通行：把程函方程写成“控制约束”的 Hamilton–Jacobi

把 Hughes 的“行走方向选择”从固定$\frac{-\nabla \phi}{|\nabla \phi|}$改成：

在每个位置$x$，行人只能从允许的方向集合$U(x)$ 里选方向。

令速度为
$\mathbf{v}=f(\rho)\mathbf{u}$，其中
$$\mathbf{u}(x) \in U(x), \quad |\mathbf{u}|=1$$

对应的“到出口最小时间势函数”$\phi$满足

$$\max _{\mathbf{u} \in U(x)}\{-f(\rho) \mathbf{u} \cdot \nabla \phi\}=1$$
- 若 $U(x)=\{ \mathbf{u}: |\mathbf{u}|=1\}$，就退化成$|\nabla \phi| = \frac{1}{f(\rho)}$，即原始 Hughes 程函方程。
- 若在单向通道内规定只允许沿$+\tau$走：
$$U(x)=\{ \tau(x)\}, \quad x \in \Omega_c$$
则上式直接变成一个“单向”线性一阶方程:



$$-f(\rho) \tau(x) \cdot \nabla \phi=1 \quad \Longleftrightarrow \quad \tau(x) \cdot \nabla \phi=-\frac{1}{f(\rho)} $$

然后速度方向取“达到最大值的最优控制”

$$\mathbf{u}^*(x) = \text{arg} \max_{\mathbf{u} \in U(x)} \{- \mathbf{u} \cdot \nabla \phi\},\quad \mathbf{v}=f(\rho)\mathbf{u}^*(x) $$

接口层面只需要提供：
- 通道区域指示函数 $\chi_c(x) \in \{0,1\}$
- 通道切向$\tau(x)$
- 通行方向标志$s_c\in \{+1,-1\}$



### 整合两个修改

- 势函数$\phi$ 不再解 eikonal，而解离散的 Bellman/HJB：

$$\phi(x)=\min _{u \in U(x)}\left(\phi(x+\Delta x u)+\frac{\Delta x}{f(\rho)} \cdot \frac{1}{\sqrt{u^{\top} M(x) u}}\right)$$

- 最优方向$\mathbf{u}^*$直接用 “使$\phi(\text{neighbor})+\text{stepCost}$ 最小的方向” 得到，严格对应argmax。


## 游客路线设置
多阶段 + 多候选下一目标 + 概率分流

- 处于阶段 $s$ 的游客，
- 会在若干候选路线/目标中，
- 按概率模型选择下一步去哪里
### 阶段 + 路线类型双指标
把原来的阶段密度
$$\rho_s(x,t)$$
扩展成
$$\rho_{s,r}(x,t),$$
其中：
- $s$：当前阶段；

- $r$：该阶段内所选择的路线/目标类型。


例如在“下平台离场阶段” $s=3$ 中，可以有：
- $r=1$：去 8 号通道；
- $r=2$：去 9 号通道；
- $r=3$：去 10 号通道。
总密度为
$$\rho(x,t)=\sum_s\sum_r \rho_{s,r}(x,t).$$
速度函数仍然用总密度：
$$f(\rho)=v_{\max}\left(1-\frac{\rho}{\rho_{\max}}\right).$$
这样拥堵仍然是共享的，但不同意图人群各走各的方向。
### 每个“阶段-路线”都有自己的势函数
对每个 $(s,r)$，定义势函数
$$\phi_{s,r}(x),$$
表示“处于阶段 $s$、且计划走路线 $r$ 的游客，从 $x$ 到其当前目标的最小剩余时间”。
对应 Bellman 方程：
$$\phi_{s,r}(x)=
\min_{u\in U_{s,r}(x)}
\left(
\phi_{s,r}(x+\Delta x\,u)
+
\frac{\Delta x}{f(\rho)}\frac1{\sqrt{u^\top M_{s,r}(x)u}}
\right).$$
最优方向：
$$u^*_{s,r}(x)=
\operatorname*{arg\,min}_{u\in U_{s,r}(x)}
\left(
\phi_{s,r}(x+\Delta x\,u)
+
\frac{\Delta x}{f(\rho)}\frac1{\sqrt{u^\top M_{s,r}(x)u}}
\right).$$
速度场：
$$\mathbf v_{s,r}(x)=f(\rho)\,u^*_{s,r}(x).$$

通常第一版里可以让不同 $r$ 共享同一个 $U_s,M_s$，只改目标区即可：
- $\phi_{3,1}$：目标是 8 号通道下口；
- $\phi_{3,2}$：目标是 9 号通道下口；
- $\phi_{3,3}$：目标是 10 号通道下口

### 概率选择发生在什么时候
阶段切换时选择下一路线
例如游客到达平台南侧后，进入“离场阶段”前，按概率选择去 8、9、10 号通道之一。
这样路线选择发生在一个明确事件点：
“到达当前阶段目标区 $G_s$ 时”。
若从阶段 $s$ 切换到下一阶段 $s+1$ 时，有多个候选路线 $r\in\{1,\dots,R_{s+1}\}$，则转移项写成
$$Q_{s\to (s+1,r)}(x,t)
=
p_{s\to r}(x,t)\,\kappa_s\,\chi_{G_s}(x)\,\rho_s(x,t),$$
其中：
- $\kappa_s$：切换率；
- $\chi_{G_s}(x)$：当前阶段目标区；
- $p_{s\to r}(x,t)$：选择路线 $r$ 的概率；
- $\sum_r p_{s\to r}(x,t)=1$。

于是：
$$\frac{\partial \rho_s}{\partial t}
+\nabla\cdot(\rho_s \mathbf v_s)
=
- \sum_r Q_{s\to (s+1,r)},$$
$$\frac{\partial \rho_{s+1,r}}{\partial t}
+\nabla\cdot(\rho_{s+1,r}\mathbf v_{s+1,r})
=
Q_{s\to (s+1,r)} - Q_{(s+1,r)\to \cdots}.$$

### 概率模型怎么构造

#### 方式 1：固定概率
例如在离场阶段：
$$p_{3\to 8}=0.2,\qquad
p_{3\to 9}=0.3,\qquad
p_{3\to 10}=0.5.$$
这表示进入离场阶段的人中，固定有 20%、30%、50% 分别去这些通道。
优点是简单、稳、容易实现。
缺点是不随拥堵和位置变化。
适合做第一版验证。


#### 方式 2：按势值/旅行时间的 Logit 概率（后续升级，暂不考虑）
这最适合你当前模型，因为你本来就有势函数 $\phi$，它正好代表“剩余代价”。
设某阶段有 $R$ 条候选路线，每条路线对应一个值函数 $\phi_{s+1,r}$。
在决策区域 $D_s$ 内，对每条候选路线定义当前代价：
$$C_r(x)=\phi_{s+1,r}(x).$$
然后用离散选择模型（softmax / multinomial logit）：
$$p_{s\to r}(x)
=
\frac{\exp(-\theta C_r(x))}
{\sum_{q=1}^{R}\exp(-\theta C_q(x))}.$$
其中 $\theta>0$ 是敏感度参数：
- $\theta$ 大：更偏向最短时间路线；
- $\theta$ 小：选择更随机。

这个非常自然，因为：

- $\phi$ 已经是“最小剩余时间”；
- 越小的 $\phi$ 越有吸引力；
拥堵会通过 $f(\rho)$ 自动影响 $\phi$，于是概率也会随拥堵变化。

#### 方式 3：加入路线偏好常数的 Logit（后续升级，暂不考虑）
如果现实中游客并不完全按最短时间选路，还会偏好“熟悉路线”“更宽通道”“更靠近原来入口”，可以加一个吸引力常数 $b_r$：
$$p_{s\to r}(x)
=
\frac{\exp\bigl(-\theta C_r(x)+b_r\bigr)}
{\sum_q \exp\bigl(-\theta C_q(x)+b_q\bigr)}.$$
其中 $b_r$ 可以表示：
- 8 号通道更显眼；
- 9 号通道更宽；
- 10 号通道更远但景观更好。
这会让模型更贴近外滩管理经验。

#### 方式 4：容量/拥堵修正概率（后续升级，暂不考虑）
如果想让管理策略“主动分流”，还可以把候选通道当前拥堵、容量、管控强度纳入概率：
$$C_r(x,t)=\phi_{s+1,r}(x,t)+\lambda\,\Psi_r(t),$$
其中 $\Psi_r(t)$ 可以是：
- 该通道附近平均密度；
- 排队长度；
- 实时拥堵指数。

于是
$$p_{s\to r}(x,t)
=
\frac{\exp(-\theta C_r(x,t))}
{\sum_q \exp(-\theta C_q(x,t))}.$$
这就能模拟“人会避开拥挤出口”的行为。


####
# 目标
## 目标描述
在上海外滩区域，游客一般可以从多个阶梯通道登上外滩观景平台。为了避免游客只从一个通道登上平台或游览完后只从一个通道离开造成拥堵，管理方通过设置不同通道的通行方向来引导游客分散登上或离开平台。
外滩观景平台的通道从北向南依次编号为1,2,...,10分布。
游客的一般路线，一般是从5,6号通道登上观景平台，然后向南边游览，最后可能从 8,9,10等通道离开，在下街沿回到5,6号通道前的区域，再返回南京东路步行街。

管理方一般会设置4号通道只上不下，7号通道只下不上的单向通行规则，来引导游客分散登上和离开平台。

希望设计一个连续介质（宏观）人群模型，能够模拟不同通道的单向通行规则对人群流动的影响，并且能够模拟不同通道的几何引导。通过这个模型，可以评估不同管控措施对游客流动效率和安全性的影响，为管理方提供科学的决策支持。

## 对宏观模型的要求

- 能够模拟不同通道的单向通行规则对人群流动的影响
- 能够模拟不同通道的几何引导
- 能够设置游客路线，达到一个地点后前往另一个地点

## 连续介质（宏观）人群模型
### 连续性方程（密度更新）

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0$$



其中，速度 $\mathbf{v} = f(\rho) \frac{-\nabla \phi}{|\nabla \phi|}$

$f(\rho)$是速度-密度关系的函数。在Greenshields 模型中：

$ f(\rho) = v_{max}(1 - \frac{\rho}{\rho_{max}}) $。

### 程函方程（势场更新）
$$|\nabla \phi| = \frac{1}{f(\rho)}$$



## 引入管控措施变量接口（通道通行方向控制、隔离栏位置（后续实现，现暂不考虑））

如何改进Hughes 人群宏观方程，使得可以设置某些通道（通道由几何边界定义）的可通行方向

### 引入度量张量 $\mathbf{M}$ 修改程函方程，实现通道几何引导（各向异性但不单向）：

#### 令$\mathbf{M}(x)$ 为对称正定矩阵（可随空间变化，通道内外不同。把程函方程改成各向异性 eikonal：


$$\sqrt{\nabla \phi^{\top} \mathbf{M}(x) \nabla \phi}=\frac{1}{f(\rho)}$$

对应的“最陡下降方向”也要改为$\mathbf{M}$意义下的梯度方向。所以速度场定义为：

$$\mathbf{v} = f(\rho) \mathbf{u}= f(\rho) \frac{-\mathbf{M}(x)\nabla \phi}{\sqrt{\nabla \phi^{\top} \mathbf{M}(x) \nabla \phi}}$$

连续性方程不变

#### 如何用$\mathbf{M}(x)$表达“通道”几何（让人更愿意沿通道走）

假设通道区域$\mathbf{\Omega}_c$由几何边界定义，并给出通道切向单位向量场 $\tau(x)$（沿通道方向），法向 $n(x)$（垂直通道方向）。构造：

$$\mathbf{M}(x)=\alpha(x) \tau \tau^{\top}+\beta(x) n n^{\top}, \quad \alpha \gg \beta(\text { 通道内 })$$
这会显著抬高横向“代价”，让$\phi$的最短时间路径更倾向于沿$\tau$ 前进。

### 单向通行：把程函方程写成“控制约束”的 Hamilton–Jacobi

把 Hughes 的“行走方向选择”从固定$\frac{-\nabla \phi}{|\nabla \phi|}$改成：

在每个位置$x$，行人只能从允许的方向集合$U(x)$ 里选方向。

令速度为
$\mathbf{v}=f(\rho)\mathbf{u}$，其中
$$\mathbf{u}(x) \in U(x), \quad |\mathbf{u}|=1$$

对应的“到出口最小时间势函数”$\phi$满足

$$\max _{\mathbf{u} \in U(x)}\{-f(\rho) \mathbf{u} \cdot \nabla \phi\}=1$$
- 若 $U(x)=\{ \mathbf{u}: |\mathbf{u}|=1\}$，就退化成$|\nabla \phi| = \frac{1}{f(\rho)}$，即原始 Hughes 程函方程。
- 若在单向通道内规定只允许沿$+\tau$走：
$$U(x)=\{ \tau(x)\}, \quad x \in \Omega_c$$
则上式直接变成一个“单向”线性一阶方程:



$$-f(\rho) \tau(x) \cdot \nabla \phi=1 \quad \Longleftrightarrow \quad \tau(x) \cdot \nabla \phi=-\frac{1}{f(\rho)} $$

然后速度方向取“达到最大值的最优控制”

$$\mathbf{u}^*(x) = \text{arg} \max_{\mathbf{u} \in U(x)} \{- \mathbf{u} \cdot \nabla \phi\},\quad \mathbf{v}=f(\rho)\mathbf{u}^*(x) $$

接口层面只需要提供：
- 通道区域指示函数 $\chi_c(x) \in \{0,1\}$
- 通道切向$\tau(x)$
- 通行方向标志$s_c\in \{+1,-1\}$



### 整合两个修改

- 势函数$\phi$ 不再解 eikonal，而解离散的 Bellman/HJB：

$$\phi(x)=\min _{u \in U(x)}\left(\phi(x+\Delta x u)+\frac{\Delta x}{f(\rho)} \cdot \frac{1}{\sqrt{u^{\top} M(x) u}}\right)$$

- 最优方向$\mathbf{u}^*$直接用 “使$\phi(\text{neighbor})+\text{stepCost}$ 最小的方向” 得到，严格对应argmax。


## 游客路线设置
多阶段 + 多候选下一目标 + 概率分流

- 处于阶段 $s$ 的游客，
- 会在若干候选路线/目标中，
- 按概率模型选择下一步去哪里
### 阶段 + 路线类型双指标
把原来的阶段密度
$$\rho_s(x,t)$$
扩展成
$$\rho_{s,r}(x,t),$$
其中：
- $s$：当前阶段；

- $r$：该阶段内所选择的路线/目标类型。


例如在“下平台离场阶段” $s=3$ 中，可以有：
- $r=1$：去 8 号通道；
- $r=2$：去 9 号通道；
- $r=3$：去 10 号通道。
总密度为
$$\rho(x,t)=\sum_s\sum_r \rho_{s,r}(x,t).$$
速度函数仍然用总密度：
$$f(\rho)=v_{\max}\left(1-\frac{\rho}{\rho_{\max}}\right).$$
这样拥堵仍然是共享的，但不同意图人群各走各的方向。
### 每个“阶段-路线”都有自己的势函数
对每个 $(s,r)$，定义势函数
$$\phi_{s,r}(x),$$
表示“处于阶段 $s$、且计划走路线 $r$ 的游客，从 $x$ 到其当前目标的最小剩余时间”。
对应 Bellman 方程：
$$\phi_{s,r}(x)=
\min_{u\in U_{s,r}(x)}
\left(
\phi_{s,r}(x+\Delta x\,u)
+
\frac{\Delta x}{f(\rho)}\frac1{\sqrt{u^\top M_{s,r}(x)u}}
\right).$$
最优方向：
$$u^*_{s,r}(x)=
\operatorname*{arg\,min}_{u\in U_{s,r}(x)}
\left(
\phi_{s,r}(x+\Delta x\,u)
+
\frac{\Delta x}{f(\rho)}\frac1{\sqrt{u^\top M_{s,r}(x)u}}
\right).$$
速度场：
$$\mathbf v_{s,r}(x)=f(\rho)\,u^*_{s,r}(x).$$

通常第一版里可以让不同 $r$ 共享同一个 $U_s,M_s$，只改目标区即可：
- $\phi_{3,1}$：目标是 8 号通道下口；
- $\phi_{3,2}$：目标是 9 号通道下口；
- $\phi_{3,3}$：目标是 10 号通道下口

### 概率选择发生在什么时候
阶段切换时选择下一路线
例如游客到达平台南侧后，进入“离场阶段”前，按概率选择去 8、9、10 号通道之一。
这样路线选择发生在一个明确事件点：
“到达当前阶段目标区 $G_s$ 时”。
若从阶段 $s$ 切换到下一阶段 $s+1$ 时，有多个候选路线 $r\in\{1,\dots,R_{s+1}\}$，则转移项写成
$$Q_{s\to (s+1,r)}(x,t)
=
p_{s\to r}(x,t)\,\kappa_s\,\chi_{G_s}(x)\,\rho_s(x,t),$$
其中：
- $\kappa_s$：切换率；
- $\chi_{G_s}(x)$：当前阶段目标区；
- $p_{s\to r}(x,t)$：选择路线 $r$ 的概率；
- $\sum_r p_{s\to r}(x,t)=1$。

于是：
$$\frac{\partial \rho_s}{\partial t}
+\nabla\cdot(\rho_s \mathbf v_s)
=
- \sum_r Q_{s\to (s+1,r)},$$
$$\frac{\partial \rho_{s+1,r}}{\partial t}
+\nabla\cdot(\rho_{s+1,r}\mathbf v_{s+1,r})
=
Q_{s\to (s+1,r)} - Q_{(s+1,r)\to \cdots}.$$

### 概率模型怎么构造

#### 方式 1：固定概率
例如在离场阶段：
$$p_{3\to 8}=0.2,\qquad
p_{3\to 9}=0.3,\qquad
p_{3\to 10}=0.5.$$
这表示进入离场阶段的人中，固定有 20%、30%、50% 分别去这些通道。
优点是简单、稳、容易实现。
缺点是不随拥堵和位置变化。
适合做第一版验证。


#### 方式 2：按势值/旅行时间的 Logit 概率（后续升级，暂不考虑）
这最适合你当前模型，因为你本来就有势函数 $\phi$，它正好代表“剩余代价”。
设某阶段有 $R$ 条候选路线，每条路线对应一个值函数 $\phi_{s+1,r}$。
在决策区域 $D_s$ 内，对每条候选路线定义当前代价：
$$C_r(x)=\phi_{s+1,r}(x).$$
然后用离散选择模型（softmax / multinomial logit）：
$$p_{s\to r}(x)
=
\frac{\exp(-\theta C_r(x))}
{\sum_{q=1}^{R}\exp(-\theta C_q(x))}.$$
其中 $\theta>0$ 是敏感度参数：
- $\theta$ 大：更偏向最短时间路线；
- $\theta$ 小：选择更随机。

这个非常自然，因为：

- $\phi$ 已经是“最小剩余时间”；
- 越小的 $\phi$ 越有吸引力；
拥堵会通过 $f(\rho)$ 自动影响 $\phi$，于是概率也会随拥堵变化。

#### 方式 3：加入路线偏好常数的 Logit（后续升级，暂不考虑）
如果现实中游客并不完全按最短时间选路，还会偏好“熟悉路线”“更宽通道”“更靠近原来入口”，可以加一个吸引力常数 $b_r$：
$$p_{s\to r}(x)
=
\frac{\exp\bigl(-\theta C_r(x)+b_r\bigr)}
{\sum_q \exp\bigl(-\theta C_q(x)+b_q\bigr)}.$$
其中 $b_r$ 可以表示：
- 8 号通道更显眼；
- 9 号通道更宽；
- 10 号通道更远但景观更好。
这会让模型更贴近外滩管理经验。

#### 方式 4：容量/拥堵修正概率（后续升级，暂不考虑）
如果想让管理策略“主动分流”，还可以把候选通道当前拥堵、容量、管控强度纳入概率：
$$C_r(x,t)=\phi_{s+1,r}(x,t)+\lambda\,\Psi_r(t),$$
其中 $\Psi_r(t)$ 可以是：
- 该通道附近平均密度；
- 排队长度；
- 实时拥堵指数。

于是
$$p_{s\to r}(x,t)
=
\frac{\exp(-\theta C_r(x,t))}
{\sum_q \exp(-\theta C_q(x,t))}.$$
这就能模拟“人会避开拥挤出口”的行为。

### 固定概率分流机制
设阶段 $s$ 的完成目标区域为 $G_s\subset\Omega$，其指示函数为
$$\chi_{G_s}(x)=
\begin{cases}
1, & x\in G_s,\\
0, & x\notin G_s.
\end{cases}$$
当某阶段人群到达 $G_s$ 后，将按固定概率分配到下一阶段的不同路线群体。

#### 阶段内单一路线群体转入下一阶段多路线群体
若当前阶段 $s$ 只有一个群体，记为 $\rho_s$，而下一阶段 $s+1$ 有多个候选路线 $r=1,\dots,R_{s+1}$，则定义固定概率
$$p_{s\to r}\ge 0,\qquad \sum_{r=1}^{R_{s+1}}p_{s\to r}=1.$$
转移项定义为
$$Q_{s\to (s+1,r)}(x,t)
=
p_{s\to r}\,\kappa_s\,\chi_{G_s}(x)\,\rho_s(x,t),$$
其中 $\kappa_s>0$ 为阶段切换率。
于是有
$$\frac{\partial \rho_s}{\partial t}
+
\nabla\cdot(\rho_s\mathbf v_s)
=
-\sum_{r=1}^{R_{s+1}}Q_{s\to (s+1,r)},$$
$$\frac{\partial \rho_{s+1,r}}{\partial t}
+
\nabla\cdot(\rho_{s+1,r}\mathbf v_{s+1,r})
=
Q_{s\to (s+1,r)}-Q^{\text{out}}_{s+1,r}.$$

#### 若当前已是多路线群体
若当前阶段也有多个群体，则每个群体都可用同样方式按固定概率继续分流。
一般写成
$$Q_{(s,r)\to (s+1,q)}(x,t)
=
p_{(s,r)\to q}\,\kappa_{s,r}\,\chi_{G_{s,r}}(x)\,\rho_{s,r}(x,t),$$
其中
$$p_{(s,r)\to q}\ge 0,\qquad \sum_q p_{(s,r)\to q}=1.$$


## 当前模型总结

### 1. 基础未知量
设：
- $\rho(x,t)$：人群密度；
- $\rho_{s,r}(x,t)$：处于阶段 $s$、且选择路线类型 $r$ 的人群密度；
- $\phi_{s,r}(x,t)$：对应子群体的最小剩余时间势函数；
- $\mathbf v_{s,r}(x,t)$：对应子群体的速度场。
- $\phi(x,t)$：到出口的最小剩余时间势函数；
- $\mathbf v(x,t)$：人群速度场。

其中：
- $s=1,\dots,S$ 表示行为阶段；
- $r=1,\dots,R_s$ 表示该阶段下的路线类型。
其中 $x\in\Omega\subset\mathbb R^2$，$t\ge 0$

总密度定义为
$$\rho(x,t)=\sum_{s=1}^{S}\sum_{r=1}^{R_s}\rho_{s,r}(x,t).$$

### 2. 速度大小函数
总密度决定拥堵速度函数：
$$f(\rho)=v_{\max}\left(1-\frac{\rho}{\rho_{\max}}\right)$$
这里 $f(\rho)$ 依赖总密度 $\rho$，表示所有群体共同造成拥堵。

### 3. 每个子群体的势场方程
对每个 $(s,r)$，定义允许方向集合 $U_{s,r}(x)$、度量张量 $M_{s,r}(x)$，并求解离散 Bellman 方程：
$$\phi_{s,r}(x)
=
\min_{u\in U_{s,r}(x)}
\left(
\phi_{s,r}(x+\Delta x\,u)
+
\frac{\Delta x}{f(\rho)}\frac{1}{\sqrt{u^\top M_{s,r}(x)u}}
\right).$$
这表示：
- 不同子群体可以有不同目标区；
- 也可以共享相同的 $U$ 和 $M$，仅目标不同。

### 4. 每个子群体的最优方向
$$u_{s,r}^*(x)=\operatorname*{arg\,min}_{u\in U_{s,r}(x)}\left(\phi_{s,r}(x+\Delta x\,u)+\frac{\Delta x}{f(\rho)}\frac1{\sqrt{u^\top M_{s,r}(x)u}}\right)$$
### 5. 每个子群体的速度场
$$\mathbf v_{s,r}(x)=f(\rho)\,u_{s,r}^*(x)$$
### 6. 每个子群体的密度演化
$$\frac{\partial \rho_{s,r}}{\partial t}
+
\nabla\cdot(\rho_{s,r}\mathbf v_{s,r})
=
Q^{\text{in}}_{s,r}-Q^{\text{out}}_{s,r}$$
### 7. 固定概率分流转移项
$$Q_{(s,r)\to (s+1,q)}(x,t)
=
p_{(s,r)\to q}\,\kappa_{s,r}\,\chi_{G_{s,r}}(x)\,\rho_{s,r}(x,t),
\qquad
\sum_q p_{(s,r)\to q}=1.$$







## 当前模型总结

### 1. 基础未知量
设：
- $\rho(x,t)$：人群密度；
- $\rho_{s,r}(x,t)$：处于阶段 $s$、且选择路线类型 $r$ 的人群密度；
- $\phi_{s,r}(x,t)$：对应子群体的最小剩余时间势函数；
- $\mathbf v_{s,r}(x,t)$：对应子群体的速度场。
- $\phi(x,t)$：到出口的最小剩余时间势函数；
- $\mathbf v(x,t)$：人群速度场。

其中：
- $s=1,\dots,S$ 表示行为阶段；
- $r=1,\dots,R_s$ 表示该阶段下的路线类型。
其中 $x\in\Omega\subset\mathbb R^2$，$t\ge 0$

总密度定义为
$$\rho(x,t)=\sum_{s=1}^{S}\sum_{r=1}^{R_s}\rho_{s,r}(x,t).$$

### 2. 速度大小函数
总密度决定拥堵速度函数：
$$f(\rho)=v_{\max}\left(1-\frac{\rho}{\rho_{\max}}\right)$$
这里 $f(\rho)$ 依赖总密度 $\rho$，表示所有群体共同造成拥堵。

### 3. 每个子群体的势场方程
对每个 $(s,r)$，定义允许方向集合 $U_{s,r}(x)$、度量张量 $M_{s,r}(x)$，并求解离散 Bellman 方程：
$$\phi_{s,r}(x)
=
\min_{u\in U_{s,r}(x)}
\left(
\phi_{s,r}(x+\Delta x\,u)
+
\frac{\Delta x}{f(\rho)}\frac{1}{\sqrt{u^\top M_{s,r}(x)u}}
\right).$$
这表示：
- 不同子群体可以有不同目标区；
- 也可以共享相同的 $U$ 和 $M$，仅目标不同。

### 4. 每个子群体的最优方向
$$u_{s,r}^*(x)=\operatorname*{arg\,min}_{u\in U_{s,r}(x)}\left(\phi_{s,r}(x+\Delta x\,u)+\frac{\Delta x}{f(\rho)}\frac1{\sqrt{u^\top M_{s,r}(x)u}}\right)$$
### 5. 每个子群体的速度场
$$\mathbf v_{s,r}(x)=f(\rho)\,u_{s,r}^*(x)$$
### 6. 每个子群体的密度演化
$$\frac{\partial \rho_{s,r}}{\partial t}
+
\nabla\cdot(\rho_{s,r}\mathbf v_{s,r})
=
Q^{\text{in}}_{s,r}-Q^{\text{out}}_{s,r}$$
### 7. 固定概率分流转移项
$$Q_{(s,r)\to (s+1,q)}(x,t)
=
p_{(s,r)\to q}\,\kappa_{s,r}\,\chi_{G_{s,r}}(x)\,\rho_{s,r}(x,t),
\qquad
\sum_q p_{(s,r)\to q}=1.$$





