

## 连续介质人群模型
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


## 当前模型总结

### 1. 基础未知量
设：
- $\rho(x,t)$：人群密度；
- $\phi(x,t)$：到出口的最小剩余时间势函数；
- $\mathbf v(x,t)$：人群速度场。

其中 $x\in\Omega\subset\mathbb R^2$，$t\ge 0$
### 2. 闭环模型
(1) 密度方程
$$\frac{\partial \rho}{\partial t}+\nabla\cdot(\rho\mathbf v)=0$$
(2) 速度大小函数
$$f(\rho)=v_{\max}\left(1-\frac{\rho}{\rho_{\max}}\right)$$

(3) 势场方程

无方向约束时：各向异性 eikonal
$$\sqrt{\nabla\phi^\top M(x)\nabla\phi}=\frac1{f(\rho)}$$

有方向约束时：HJB
$$\max_{u\in U(x)}\{-f(\rho)\,u\cdot\nabla\phi\}=1$$

数值统一实现时：离散 Bellman
$$\phi(x)=\min_{u\in U(x)}\left(\phi(x+\Delta x\,u)+\frac{\Delta x}{f(\rho)}\frac1{\sqrt{u^\top M(x)u}}\right)$$
(4) 速度方向
$$u^*(x)=\operatorname*{arg\,min}_{u\in U(x)}\left(\phi(x+\Delta x\,u)+\frac{\Delta x}{f(\rho)}\frac1{\sqrt{u^\top M(x)u}}\right)$$
(5) 速度场
$$\mathbf v(x)=f(\rho)\,u^*(x)$$

### 3. 接口变量总结
当前最核心的输入可以整理为：
- 通道区域指示函数 $\chi_c(x)$；
- 通道切向单位向量场 $\tau(x)$；
- 通道法向单位向量场 $n(x)$；
- 各向异性参数 $\alpha(x),\beta(x)$；
- 通行方向标志 $s_c\in\{+1,-1\}$；
- 允许方向集合 $U(x)$。

### 4. 核心公式对应代码实现
当前仓库中的主要实现位于：
- `codes/crowd_bellman/core.py`
- `codes/crowd_bellman/scenes.py`
- `codes/crowd_bellman/runner.py`

#### (1) 速度-密度关系 $f(\rho)=v_{\max}(1-\rho/\rho_{\max})$
对应 `codes/crowd_bellman/core.py::greenshields_speed`

```python
def greenshields_speed(rho: np.ndarray, vmax: float, rho_max: float) -> np.ndarray:
    speed = vmax * (1.0 - rho / rho_max)
    return np.clip(speed, 0.0, vmax)
```

#### (2) 张量构造 $M(x)=\alpha\tau\tau^\top+\beta nn^\top$
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

在场景层，通道引导就是通过 `tau(x)` 和不同的 `alpha, beta` 生成 `M(x)`：

```python
guided_lane = scene.channel_masks[guided_channel] & case_walkable
lane_tau_x = np.ones_like(m11)
lane_tau_y = np.zeros_like(m11)
lane_m11, lane_m12, lane_m22 = tensor_from_tau(
    tau_x=lane_tau_x,
    tau_y=lane_tau_y,
    alpha=10.0,
    beta=0.20,
)
```

#### (3) 离散 Bellman
$$
\phi(x)=\min_{u\in U(x)}\left(\phi(x+\Delta x\,u)+\frac{\Delta x}{f(\rho)}\frac1{\sqrt{u^\top M(x)u}}\right)
$$
对应 `codes/crowd_bellman/core.py::precompute_step_factors` 与 `solve_bellman`

```python
def precompute_step_factors(walkable, dx, m11, m12, m22):
    step_factor = np.full((walkable.shape[0], walkable.shape[1], len(DIRECTIONS.names)), np.inf)
    for k in range(len(DIRECTIONS.names)):
        ut_mu = (
            DIRECTIONS.ux[k] * DIRECTIONS.ux[k] * m11
            + 2.0 * DIRECTIONS.ux[k] * DIRECTIONS.uy[k] * m12
            + DIRECTIONS.uy[k] * DIRECTIONS.uy[k] * m22
        )
        denom = np.sqrt(np.maximum(ut_mu, 1.0e-12))
        step_factor[:, :, k] = dx * DIRECTIONS.step[k] / denom
    step_factor[~walkable] = np.inf
    return step_factor
```

```python
candidate = value + step_factor[py, px, k] / speed_safe[py, px]
if candidate + 1.0e-12 < phi[py, px]:
    phi[py, px] = candidate
    heappush(queue, (candidate, py, px))
```

这里 `step_factor / speed_safe` 就是
$$
\frac{\Delta x}{f(\rho)}\frac1{\sqrt{u^\top M u}}
$$
的离散实现。

#### (4) 最优方向
$$
u^*(x)=\arg\min_{u\in U(x)}\left(\phi(x+\Delta x\,u)+\frac{\Delta x}{f(\rho)}\frac1{\sqrt{u^\top M(x)u}}\right)
$$
对应 `codes/crowd_bellman/core.py::recover_optimal_direction`

```python
candidate = phi[nyy, nxx] + step_factor[y, x, k] / speed_safe[y, x]
if candidate < best_value:
    best_value = candidate
    best_ux = DIRECTIONS.ux[k]
    best_uy = DIRECTIONS.uy[k]
```

#### (5) 速度场 $\mathbf v(x)=f(\rho)\,u^*(x)$
对应 `codes/crowd_bellman/runner.py::simulate_case`

```python
speed = greenshields_speed(rho, cfg.vmax, cfg.rho_max)
vx = speed * ux
vy = speed * uy
vx[~case.walkable] = 0.0
vy[~case.walkable] = 0.0
```

#### (6) 连续性方程显式更新
$$
\frac{\partial \rho}{\partial t}+\nabla\cdot(\rho\mathbf v)=0
$$
对应 `codes/crowd_bellman/core.py::compute_face_fluxes` 与 `update_density`

```python
vx_face = 0.5 * (vx[:, :-1] + vx[:, 1:])
vy_face = 0.5 * (vy[:-1, :] + vy[1:, :])

rho_x = np.where(vx_face >= 0.0, rho[:, :-1], rho[:, 1:])
rho_y = np.where(vy_face >= 0.0, rho[:-1, :], rho[1:, :])
fx = vx_face * rho_x
fy = vy_face * rho_y
```

```python
div_x[:, 1:-1] = (fx[:, 1:] - fx[:, :-1]) / dx
div_y[1:-1, :] = (fy[1:, :] - fy[:-1, :]) / dx
updated = rho - dt * (div_x + div_y)
```

#### (7) 单向约束 $U(x)=\{\tau(x)\}$
当前代码里，严格单向通道通过 `allowed_mask` 实现，而不是直接用 PDE 形式硬编码：

```python
east_mask = np.uint16(DIRECTIONS.bits[0])
allowed_mask = default_allowed_mask(case_walkable)
allowed_mask[guided_lane] = east_mask
```

这正对应了理论里的
$$
U(x)=\{\tau(x)\}
$$
只不过在代码中离散成了一个方向 bitmask。

