# 1. 什么是 Godunov 有限差分格式
## 概述
Godunov 有限差分格式，本质上是针对一阶 Hamilton–Jacobi 方程/守恒律设计的单调迎风离散格式。
它的核心思想不是“随便把导数差分一下”，而是：

离散时必须尊重信息传播方向，只允许使用“上风侧（upwind）”的数值信息，从而选出正确的黏性解（viscosity solution）。

这对 eikonal 方程尤其重要，因为 eikonal 是一阶非线性方程，解可能有尖点、折线、不可微界面。如果差分方式不对，就会得到不稳定、非物理或错误分支的解。

## 1.1 从一阶 Hamilton–Jacobi 方程说起
设一般一阶 HJ 方程为
$$H(\nabla \phi,x)=0.$$
如果直接用中心差分，例如
$$\partial_x \phi \approx \frac{\phi_{i+1}-\phi_{i-1}}{2\Delta x},$$
往往会有两个问题：
- 不满足单调性；
- 不能自动识别信息传播方向。


而 HJ 方程的正确弱解概念是黏性解。
Barles–Souganidis 理论告诉我们：要收敛到黏性解，数值格式通常需要满足
- 一致性；
- 稳定性；
- 单调性。

Godunov 格式正是满足这类要求的经典构造。

## 1.2 Godunov 格式的核心：对每个方向做“迎风选择”
先看最简单一维 eikonal：
$$|\phi_x|=a(x), \qquad a(x)\ge 0.$$
离散时，在网格点 $x_i$ 定义左右单边差分
$$D^-_x \phi_i=\frac{\phi_i-\phi_{i-1}}{\Delta x},\qquad
D^+_x \phi_i=\frac{\phi_{i+1}-\phi_i}{\Delta x}.$$

Godunov 的思路是：
- 如果解的信息从左边传来，就该用 $D^-_x$；
- 如果从右边传来，就该用 $D^+_x$；
- 但由于真实方向事先未知，就把“最合适的迎风导数”编码到一个数值 Hamiltonian 里。

对凸 Hamiltonian，Godunov 数值 Hamiltonian 可以理解为：
在所有可能的一侧差分中，选出与连续 Hamiltonian 一致、且单调的那一个。

## 1.3 在二维各向同性 eikonal 中的经典形式
二维 eikonal：
$$|\nabla \phi|=F(x)$$
或
$$\phi_x^2+\phi_y^2=F(x)^2.$$
Godunov 格式常写成
$$\Bigl(\max\{D^-_x\phi_{ij},-D^+_x\phi_{ij},0\}\Bigr)^2
+
\Bigl(\max\{D^-_y\phi_{ij},-D^+_y\phi_{ij},0\}\Bigr)^2
=F_{ij}^2.$$
这个式子很重要。它的含义是：
- $D^-_x\phi$ 对应从左侧来的信息；
- $-D^+_x\phi$ 对应从右侧来的信息；
- 取 $\max(\cdot,0)$ 是为了只保留真正的上风贡献；
- 然后把各方向贡献按 Hamiltonian 的结构组合起来。

这就是 eikonal 的 Godunov 单调离散。

# 2. 为什么 eikonal 需要 Godunov 格式
因为 eikonal 是“传播型最短时间方程”，解的正确物理解来自上风传播。
例如
$$|\nabla \phi|=1,\qquad \phi=0 \text{ on exit},$$
其解表示到出口的最短距离/最短时间。
在网格点 $(i,j)$ 上，$\phi_{ij}$ 必须由更小的邻点值往外传播。
如果你用中心差分，它会把未到达的信息和已到达的信息混合起来，破坏“因果性”。
Godunov 格式本质上是在离散层面保留：
- 最短路传播的因果结构；
- 局部单调更新；
- 黏性解选择机制。

这也是 fast marching / fast sweeping 等算法的基础。

## 3. 各向异性 eikonal 是什么
当前模型里的各向异性 eikonal 是
$$\sqrt{\nabla\phi^\top M(x)\nabla\phi}=\frac{1}{f(\rho)}.$$
平方后写成
$$\nabla\phi^\top M(x)\nabla\phi=\frac{1}{f(\rho)^2}.$$
若在二维里写
$$M(x)=
\begin{pmatrix}
m_{11}(x) & m_{12}(x)\\
m_{12}(x) & m_{22}(x)
\end{pmatrix},$$
则方程变成
$$m_{11}\phi_x^2+2m_{12}\phi_x\phi_y+m_{22}\phi_y^2
=
\frac{1}{f(\rho)^2}.$$
这和各向同性情形最大的不同是：
现在 Hamiltonian 不只是 $\phi_x^2+\phi_y^2$，而是一个二次型，并且可能带交叉项 $2m_{12}\phi_x\phi_y$。
这就使离散变难了。