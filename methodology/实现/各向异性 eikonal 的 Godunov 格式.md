
# 各向异性 eikonal 的 Godunov 有限差分格式
## 相关文档
[规划](../../plans/20260319-0329.md)

[基础理论-Godunov有限查分.md](../../references/基础理论/Godunov有限查分.md)

[理论模型](../model.md)

*一个从简单到一般的推导框架。真正完全一般的推导会涉及数值 Hamiltonian 的凸分析构造*

## 研究各向异性 eikonal 的 Godunov 有限差分格式的目的
### 因为连续方程本身不够，真正计算必须靠离散格式
如果没有合适数值格式，就不能稳定、正确地算出 $\phi$。
研究 Godunov 格式，就是在回答：
- 用什么离散方式算这个方程？
- 为什么这个离散方式收敛到正确解？
- 如何保证不出现逆向传播、伪振荡、错误分支？

### 因为 eikonal/HJ 方程的正确解是黏性解
一阶非线性方程经常有多个弱解。
物理上、最优控制上、最短时间上真正对应的是黏性解。

单调 Godunov 格式是逼近黏性解的标准方式之一。
不研究它，你的模型即使写得再漂亮，也可能数值上解错。

### 因为各向异性会破坏简单的“圆形传播”
在各向同性里，波前是圆形/球形传播，算法相对简单。
但各向异性下，波前变成椭圆甚至更复杂的凸前沿，传播速度依赖方向。
这会带来：
- 网格方向误差更明显；
- 交叉项离散更难；
- 简单格式可能失去单调性；
- 通道方向和网格不对齐时误差会很大。

所以必须专门研究各向异性的 Godunov 离散。

# 推导过程
## 先写成 Hamiltonian 形式
定义
$$H(p,x)=\sqrt{p^\top M(x)p}-\frac1{f(\rho(x))}=0,$$
或者更方便地平方写为
$$\widetilde H(p,x)=p^\top M(x)p-\frac1{f(\rho(x))^2}=0.$$
其中 $p=\nabla\phi$。
如果 $M$ 对称正定，则 $\widetilde H$ 对 $p$ 是凸的。
这正适合 Godunov 型单调离散。

## 先看 $M$ 对角的情形
假设在某点局部坐标系里
$$M=
\begin{pmatrix}
\alpha & 0\\
0 & \beta
\end{pmatrix},
\qquad \alpha,\beta>0.$$
则方程为
$$\alpha \phi_x^2+\beta \phi_y^2=\frac1{f(\rho)^2}.$$
这是最容易构造 Godunov 格式的情况。
直接把各向同性格式加权即可：
$$\alpha \Bigl(\max\{D^-_x\phi_{ij},-D^+_x\phi_{ij},0\}\Bigr)^2
+
\beta \Bigl(\max\{D^-_y\phi_{ij},-D^+_y\phi_{ij},0\}\Bigr)^2
=
\frac1{f_{ij}^2}.$$
这就是局部主方向坐标下的各向异性 eikonal Godunov 格式。
它的含义非常清楚：
- 切向和法向导数仍然用迎风差分；
- 只是不同方向的“代价权重”不同；
- $\alpha,\beta$ 决定传播速度椭圆而不是圆。


如果 $M=\alpha\tau\tau^\top+\beta nn^\top$，那么只要在 $(\tau,n)$ 坐标里离散，上式就成立。

## 再推广到一般 SPD 矩阵
若
$$M=
\begin{pmatrix}
m_{11} & m_{12}\\
m_{12} & m_{22}
\end{pmatrix},$$
则连续 Hamiltonian 是
$$m_{11}p_x^2+2m_{12}p_xp_y+m_{22}p_y^2.$$
难点在交叉项 $2m_{12}p_xp_y$：
你不能简单地分别对 $x,y$ 方向套 $\max\{\cdots\}$ 再直接代进去，因为 $p_x,p_y$ 的符号耦合了。
Godunov 的一般做法是：

### 第一步：列出每个方向所有可能的上风导数组合
在网格点 $(i,j)$，定义四个单边差分：
$$a^- = D^-_x\phi_{ij},\qquad a^+=D^+_x\phi_{ij},$$
$$b^- = D^-_y\phi_{ij},\qquad b^+=D^+_y\phi_{ij}.$$
由于 $p_x,p_y$ 的真实迎风符号未知，候选梯度可以来自四种组合：
$$(p_x,p_y)\in
\{(a^-,b^-),\,(a^-, -b^+),\,(-a^+,b^-),\,(-a^+,-b^+)\}.$$
这里负号的原因和各向同性 Godunov 一样：
如果信息从右侧来，就用 $-D^+_x$，从上侧来就用 $-D^+_y$。
### 第二步：在这些上风候选中构造数值 Hamiltonian
定义 Godunov 数值 Hamiltonian
$$H^G(a^-,a^+,b^-,b^+)
=
\max_{\substack{p_x\in [a^-, -a^+]\\ p_y\in [b^-, -b^+]}}
\Bigl(p^\top M p\Bigr)^{1/2}$$
或对应平方形式
$$\widetilde H^G
=
\max_{\substack{p_x\in [a^-, -a^+]\\ p_y\in [b^-, -b^+]}}
\left(p^\top M p\right).$$
这是 Godunov 构造最本质的定义：
**在由左右单边差分张成的区间盒子里，对连续 Hamiltonian 做局部极值选择。**

因为 Hamiltonian 凸，这个最大值会落在盒子的边界/顶点上，于是就变成检查有限个候选即可。
于是离散方程写成
$$H^G\bigl(D^-_x\phi,D^+_x\phi,D^-_y\phi,D^+_y\phi\bigr)
=
\frac1{f(\rho)}.$$
这就是一般各向异性 eikonal 的 Godunov 格式定义。

## 与对角化的关系
由于 $M$ 是对称正定矩阵，它可正交对角化：
$$M=Q
\begin{pmatrix}
\lambda_1 & 0\\
0 & \lambda_2
\end{pmatrix}
Q^\top.$$
如果你在局部主方向基底 $(\tau,n)$ 下离散，即令
$$p'=Q^\top p,$$
则连续方程变成
$$\lambda_1 (p'_1)^2+\lambda_2 (p'_2)^2=\frac1{f^2}.$$
这说明：*各向异性 Godunov 格式最自然的推导方式，其实是在 $M$ 的主方向坐标系中做对角型 Godunov 离散。*

也就是说，若你的通道张量本来就是
$$M=\alpha\tau\tau^\top+\beta nn^\top,$$
那么最自然的 Godunov 格式不是在全局 $x,y$ 坐标里硬处理交叉项，而是：
- 先投影到 $\tau,n$ 坐标；
- 对 $\partial_\tau \phi,\partial_n \phi$ 分别做上风差分；
- 再写成
$$\alpha (\partial_\tau^G \phi)^2+\beta (\partial_n^G \phi)^2=\frac1{f^2}.$$


这在建模上最清楚。

## 一个你可以直接使用的“通道坐标下各向异性 Godunov 格式”
若局部通道切向/法向为 $\tau,n$，且
$$M=\alpha \tau\tau^\top+\beta nn^\top,$$
那么在通道局部坐标 $(s,r)$（分别对应 $\tau,n$）中，连续方程是
$$\alpha \phi_s^2+\beta \phi_r^2=\frac1{f(\rho)^2}.$$
Godunov 离散可写成
$$\alpha \Bigl(\max\{D^-_s\phi,-D^+_s\phi,0\}\Bigr)^2
+
\beta \Bigl(\max\{D^-_r\phi,-D^+_r\phi,0\}\Bigr)^2
=
\frac1{f(\rho)^2}.$$
其中
$$D^-_s\phi=\frac{\phi_{ij}-\phi_{i-1,j}}{\Delta s},\qquad
D^+_s\phi=\frac{\phi_{i+1,j}-\phi_{ij}}{\Delta s},$$
$$D^-_r\phi=\frac{\phi_{ij}-\phi_{i,j-1}}{\Delta r},\qquad
D^+_r\phi=\frac{\phi_{i,j+1}-\phi_{ij}}{\Delta r}.$$
这就是最适合当前模型语境的一版。

## 这个格式是怎么从离散最短路/Bellman 推出来的
这点对当前模型特别重要，因为已经在用 Bellman/HJB 思想。
设局部离散 Bellman 更新为
$$\phi_{ij}
=
\min_{u\in U}
\left\{
\phi(x_{ij}+\Delta x\,u)
+
\frac{\Delta x}{f(\rho)}\frac1{\sqrt{u^\top M u}}
\right\}.$$
对小步长做一阶展开：
$$\phi(x+\Delta x\,u)\approx \phi(x)+\Delta x\,u\cdot \nabla\phi.$$
代回得
$$0=
\min_{u\in U}
\left\{
u\cdot \nabla\phi+\frac1{f(\rho)\sqrt{u^\top M u}}
\right\}.$$
再经过对偶化/优化，可得到连续 HJB 或各向异性 eikonal。
而 Godunov 格式，本质上就是这套最优控制/Bellman 更新在网格上的单调离散实现。
所以它不是纯粹技巧，而是：
>离散 Bellman 原理 + 上风因果性 = Godunov 型更新。
这也是 fast marching 之类算法为什么和最短路径/Bellman 有深层联系的原因。

## 当前仓库中的对应代码实现
需要说明的是：当前仓库**没有单独实现一个显式命名为 Godunov 的局部二次更新核**，
而是采用了统一的离散 Bellman 实现：
- 当 $U(x)$ 取全方向集合时，它承担各向异性 eikonal 的数值求解角色；
- 当 $U(x)$ 被截断时，它自然退化到受约束 HJB。

因此，下述代码应理解为“当前工程里的等价实现”，而不是一套独立的专用 Godunov kernel。

### 1. 张量 $M(x)$ 的代码构造
对应 `codes/crowd_bellman/core.py::tensor_from_tau`

```python
def tensor_from_tau(
    tau_x: np.ndarray,
    tau_y: np.ndarray,
    alpha: float | np.ndarray,
    beta: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_x = -tau_y
    n_y = tau_x
    m11 = alpha * tau_x * tau_x + beta * n_x * n_x
    m12 = alpha * tau_x * tau_y + beta * n_x * n_y
    m22 = alpha * tau_y * tau_y + beta * n_y * n_y
    return m11, m12, m22
```

这对应公式
$$
M=\alpha\tau\tau^\top+\beta nn^\top.
$$

### 2. 单步代价
对于方向 $u=(u_x,u_y)$，代码先预计算
$$
\frac{\Delta x}{\sqrt{u^\top M u}}
$$
这一部分：

```python
def precompute_step_factors(walkable, dx, m11, m12, m22):
    step_factor = np.full((walkable.shape[0], walkable.shape[1], len(DIRECTIONS.names)), np.inf, dtype=float)
    for k in range(len(DIRECTIONS.names)):
        ut_mu = (
            DIRECTIONS.ux[k] * DIRECTIONS.ux[k] * m11
            + 2.0 * DIRECTIONS.ux[k] * DIRECTIONS.uy[k] * m12
            + DIRECTIONS.uy[k] * DIRECTIONS.uy[k] * m22
        )
        denom = np.sqrt(np.maximum(ut_mu, 1.0e-12))
        step_factor[:, :, k] = dx * DIRECTIONS.step[k] / denom
```

再在 Bellman 更新里除以 `speed_safe=f(\rho)`，形成
$$
\frac{\Delta x}{f(\rho)\sqrt{u^\top M u}}.
$$

### 3. 对应 eikonal 的统一数值更新
```python
candidate = value + step_factor[py, px, k] / speed_safe[py, px]
if candidate + 1.0e-12 < phi[py, px]:
    phi[py, px] = candidate
    heappush(queue, (candidate, py, px))
```

如果此时 `allowed_mask` 在每个可行点上都等于“全八方向”，那么这段代码在工程上就对应于：
$$
\sqrt{\nabla \phi^\top M(x)\nabla \phi}=\frac1{f(\rho)}
$$
的统一离散求解。

### 4. 与理论 Godunov 公式的关系
本文前面推导的是局部上风 Godunov 形式：
$$
\alpha \Bigl(\max\{D^-_s\phi,-D^+_s\phi,0\}\Bigr)^2
+
\beta \Bigl(\max\{D^-_r\phi,-D^+_r\phi,0\}\Bigr)^2
=
\frac1{f(\rho)^2}.
$$

当前仓库没有把这一式子直接写成
`godunov_local_update(...)` 这样的函数。
相反，项目选择了更统一的实现路线：
- 通过方向离散把局部 PDE 转成 Bellman 最短路问题；
- 再用 `solve_bellman(...)` 一次性同时覆盖 eikonal 与 HJB 两类情形。

因此，如果后续需要做“纯 Godunov 格式 vs 统一 Bellman 格式”的数值对比，
还需要额外实现一个专门的 Godunov 局部更新函数。
