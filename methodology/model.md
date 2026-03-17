

## 连续介质人群模型
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

$$\mathbf{M}(x)=\alpha(x) \tau \tau^{\top}+\beta(x) n n^{\top}, \quad \beta \gg \alpha(\text { 通道内 })$$
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