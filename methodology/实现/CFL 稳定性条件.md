

# CFL稳定性是什么
在数值模拟和计算流体力学（CFD）中，CFL 条件（Courant-Friedrichs-Lewy Condition）是一个至关重要的稳定性判据。
简单来说，它规定了在使用显式格式求解偏微分方程（如波动方程、流体方程）时，时间步长 ($\Delta t$) 不能取得太大，否则计算结果会迅速发散（出现无穷大或非物理的震荡）。
### 1. 核心物理直觉：信息传播速度
CFL 条件的本质是：数值计算捕捉信息的速度，必须大于或等于物理信息传播的真实速度。想象你在模拟一个波的移动：
- 物理速度 ($u$)： 波在现实中每秒移动的距离。
- 数值速度 ($\Delta x / \Delta t$)： 你的程序每一个时间步长在网格间“跳跃”的速度。

如果 $\Delta t$ 太大，波在现实中已经跑过了两个网格，但你的程序才计算完一个网格，那么中间流失的信息就会导致计算崩溃。

### 2. 数学表达式
对于一维线性对流方程，CFL 条件通常表示为一个无量纲数，称为 Courant 数 ($C$)：$$C = \frac{u \Delta t}{\Delta x} \le C_{max}$$
其中：
- $u$：流速或波速。
- $\Delta t$：时间步长。
- $\Delta x$：空间网格间距。
- $C_{max}$：通常对于简单的显式格式，
- $C_{max} = 1$。

结论： 为了保证稳定，你必须满足 $\Delta t \le \frac{C_{max} \cdot \Delta x}{u}$。即：网格越细（$\Delta x$ 越小），时间步长就必须压得越低。

# 为什么要研究 CFL 稳定性条件？
## 1. CFL 条件回答的是：时间步长 $\Delta t$ 到底能取多大
闭环模型里，密度更新靠连续性方程：
$$\frac{\partial \rho}{\partial t}+\nabla\cdot(\rho \mathbf v)=0.$$
离散后通常是显式更新：
$$\rho^{n+1}=\rho^n-\Delta t \,\nabla_h\cdot(\rho^n\mathbf v^n).$$
这时 $\Delta t$ 不能随便取。
如果取太大，一个时间步里“信息传播”跨过了过多网格，数值格式就会失稳，出现：
- 密度震荡；
- 局部负值；
- 人群在障碍附近穿透；
- 通道里出现非物理堆积或抽空。

CFL 条件本质上就是在限制：
>一个时间步内，信息传播距离不能超过数值格式所能分辨的网格尺度。

## 2. 模型里速度不是普通各向同性速度，而是被 $M$ 放大/扭曲过的
在经典 Hughes 模型里，速度大小大致由 $f(\rho)$ 控制，所以很多时候 CFL 只和
$f(\rho)$
有关。
但现在的速度场是
$$\mathbf v=f(\rho)\frac{-M\nabla\phi}{\sqrt{\nabla\phi^\top M\nabla\phi}}.$$
这意味着：
- 同样的 $f(\rho)$，不同方向上的传播速度不一样；
- 某些方向会被 $M$ 强化；
- 某些方向会被 $M$ 抑制；
- 通道方向和网格方向不一致时，坐标分量速度也会发生耦合变化。


所以如果还用原来“只看 $f(\rho)$”的 CFL 条件，就会低估真实传播速度，导致时间步过大，数值上不稳定。
也就是说：**引入 $M$ 之后，CFL 条件本身就必须改。**

## 3. 研究 CFL 是为了保证密度方程和势场方程耦合后的整体稳定
现在不是只解一个 eikonal 或一个 HJB，而是一个闭环系统：
- 已知 $\rho^n$，求 $\phi^n$；
- 由 $\phi^n$ 求最优方向 $u^{*,n}$ 和速度 $\mathbf v^n$；
- 再用 $\mathbf v^n$ 更新 $\rho^{n+1}$。

所以一旦 CFL 条件没处理好，问题不只是“密度更新误差变大”，而是整个闭环会出问题：
- $\rho$ 更新不稳定；
- 导致 $f(\rho)$ 出现异常；
- 再反馈到 eikonal / HJB；
- 使下一步势场和速度方向也一起失真。

因此 CFL 条件在这里的作用是：
保证“势场—速度—密度”这条闭环链条在数值上是可控的。

# 第一版：针对 $M$ 张量各向异性特性的改进 CFL 条件
考虑密度方程
$$\frac{\partial \rho}{\partial t}+\nabla\cdot(\rho \mathbf v)=0,$$
其中速度场由各向异性势场给出：
$$\mathbf v
=
f(\rho)\frac{-M(x)\nabla\phi}{\sqrt{\nabla\phi^\top M(x)\nabla\phi}}.$$
同时势函数满足各向异性 eikonal 方程
$$\sqrt{\nabla\phi^\top M(x)\nabla\phi}=\frac{1}{f(\rho)}.$$
目标是推导显式守恒格式的时间步长限制 $\Delta t$，使其显式体现张量 $M(x)$ 的方向放大效应，而不是只用各向同性速度上界 $f(\rho)$。

## 1. 经典 CFL 条件回顾
对二维显式一阶迎风/有限体积格式，若在一个时间步内将速度 $\mathbf v=(v_x,v_y)$ 视为已知，则标准 CFL 条件为
$$\Delta t
\left(
\frac{|v_x|}{\Delta x}+\frac{|v_y|}{\Delta y}
\right)\le 1$$
在全局上写成
$$\boxed{
\Delta t
\le
\frac{1}{
\displaystyle
\max_{(i,j)}
\left(
\frac{|v_{x,ij}|}{\Delta x}+\frac{|v_{y,ij}|}{\Delta y}
\right)
}
}$$
这保证显式迎风更新的单调性、正性和稳定性。
因此关键变成：如何从 $M$ 的结构出发，估计 $|v_x|$、$|v_y|$。

## 2. 各向异性速度分量上界
记
$$p=\nabla\phi.$$
则
$$\mathbf v
=
f(\rho)\frac{-Mp}{\sqrt{p^\top M p}}.$$
对任意单位方向 $\xi\in\mathbb R^2$，速度在该方向上的分量为
$$\xi\cdot \mathbf v
=
-f(\rho)\frac{\xi^\top M p}{\sqrt{p^\top M p}}.$$
对 $M$-诱导内积应用 Cauchy–Schwarz 不等式：
$$|\xi^\top M p|
\le
\sqrt{(\xi^\top M \xi)(p^\top M p)}.$$
代入得
$$|\xi\cdot \mathbf v|
\le
f(\rho)\sqrt{\xi^\top M \xi}.$$
因此沿任意方向 $\xi$ 的局部传播速度上界是
$$\boxed{
a_\xi(x,\rho)
=
f(\rho)\sqrt{\xi^\top M(x)\xi}.
}$$
这就是张量 $M$ 的方向特征速度。

3. 笛卡尔网格上的分量速度上界
取坐标方向
$$e_x=(1,0)^\top,\qquad e_y=(0,1)^\top,$$
则有
$$|v_x|
\le
f(\rho)\sqrt{e_x^\top M e_x},
\qquad
|v_y|
\le
f(\rho)\sqrt{e_y^\top M e_y}.$$
若记
$$M=
\begin{pmatrix}
m_{11} & m_{12}\\
m_{12} & m_{22}
\end{pmatrix},$$
则
$$e_x^\top M e_x=m_{11},\qquad e_y^\top M e_y=m_{22}.$$
所以
$$\boxed{
|v_x|\le f(\rho)\sqrt{m_{11}},\qquad
|v_y|\le f(\rho)\sqrt{m_{22}}.
}$$
代入标准 CFL 条件，得到改进后的各向异性 CFL：
$$\boxed{
\Delta t
\le
\frac{1}{
\displaystyle
\max_{(i,j)}
f(\rho_{ij})
\left(
\frac{\sqrt{m_{11,ij}}}{\Delta x}
+
\frac{\sqrt{m_{22,ij}}}{\Delta y}
\right)
}
}$$
这就是最实用的二维 Cartesian 网格版本。

## 4. 与 eikonal 的关系
由各向异性 eikonal 方程
$$p^\top M p=\frac{1}{f(\rho)^2},$$
也可写
$$\mathbf v=-f(\rho)^2 M p.$$
但直接用这个式子估计 $|\mathbf v|$ 往往得到较粗的谱范数上界。
上面基于方向投影得到的结果更精确，因为 CFL 真正需要的是法向/坐标方向波速，而不是单纯的欧氏模长。

## 5. 更保守但更简单的谱上界
由 Rayleigh 商可得
$$|\mathbf v|
\le
f(\rho)\sqrt{\lambda_{\max}(M)}.$$
因此也有保守 CFL
$$\Delta t
\le
\frac{1}{
\displaystyle
\max_{(i,j)}
f(\rho_{ij})\sqrt{\lambda_{\max}(M_{ij})}
\left(
\frac{1}{\Delta x}+\frac{1}{\Delta y}
\right)
}.$$
但因为
$$\sqrt{m_{11}}\le \sqrt{\lambda_{\max}(M)},\qquad
\sqrt{m_{22}}\le \sqrt{\lambda_{\max}(M)},$$
所以改进条件满足
$$f(\rho)\left(\frac{\sqrt{m_{11}}}{\Delta x}+\frac{\sqrt{m_{22}}}{\Delta y}\right)
\le
f(\rho)\sqrt{\lambda_{\max}(M)}\left(\frac1{\Delta x}+\frac1{\Delta y}\right).$$
因此改进 CFL 不比谱上界更严格，通常更宽松，但仍然稳定。

## 6. 在通道张量 $M=\alpha \tau\tau^\top+\beta nn^\top$ 下的具体形式
若
$$M(x)=\alpha(x)\tau\tau^\top+\beta(x)nn^\top,$$
其中 $\tau=(\tau_x,\tau_y)^\top$，$n=(n_x,n_y)^\top$，则
$$m_{11}=\alpha \tau_x^2+\beta n_x^2,
\qquad
m_{22}=\alpha \tau_y^2+\beta n_y^2.$$
因此改进 CFL 写成
$$\boxed{
\Delta t
\le
\frac{1}{
\displaystyle
\max_{(i,j)}
f(\rho_{ij})
\left[
\frac{\sqrt{\alpha_{ij}\tau_{x,ij}^2+\beta_{ij}n_{x,ij}^2}}{\Delta x}
+
\frac{\sqrt{\alpha_{ij}\tau_{y,ij}^2+\beta_{ij}n_{y,ij}^2}}{\Delta y}
\right]
}
}$$
这就把：
- 通道主方向 $\tau$；
- 法向 $n$；
- 各向异性强度 $\alpha,\beta$；

全部显式地编码进了时间步长限制。

## 7. 若网格与通道主方向对齐
若采用局部坐标 $(s,r)$，其中 $s$ 沿 $\tau$，$r$ 沿 $n$，则
$$M=
\begin{pmatrix}
\alpha & 0\\
0 & \beta
\end{pmatrix}.$$
这时传播速度上界直接为
$$|v_s|\le f(\rho)\sqrt{\alpha},\qquad
|v_r|\le f(\rho)\sqrt{\beta}.$$
于是 CFL 条件变成
$$\boxed{
\Delta t
\le
\frac{1}{
\displaystyle
\max
f(\rho)
\left(
\frac{\sqrt{\alpha}}{\Delta s}
+
\frac{\sqrt{\beta}}{\Delta r}
\right)
}
}$$
这是最清楚的“张量主方向版 CFL”。

## 8. 该条件为何是“改进”的
原先若忽略张量，只按各向同性速度上界 $f(\rho)$ 取 CFL，则会写成
$$\Delta t \le \frac{1}{\max f(\rho)\left(\frac1{\Delta x}+\frac1{\Delta y}\right)}.$$
这在 $M\neq I$ 时并不可靠，因为真实特征速度已经被 $M$ 放大或压缩。
而若直接用谱半径粗估，
$$\Delta t
\le
\frac{1}{
\max f(\rho)\sqrt{\lambda_{\max}(M)}
\left(\frac1{\Delta x}+\frac1{\Delta y}\right)
},$$
虽然稳定，但往往过于保守。
改进后的条件
$$\Delta t
\le
\frac{1}{
\max f(\rho)
\left(
\frac{\sqrt{m_{11}}}{\Delta x}
+
\frac{\sqrt{m_{22}}}{\Delta y}
\right)
}$$
恰好介于两者之间：
- 比“忽略 $M$”更准确；
- 比“只看 $\lambda_{\max}$”更不保守。

## 9. 一致性验证
### (1) 各向同性情形
若
$$M=I,$$
则
$$m_{11}=m_{22}=1.$$
改进 CFL 退化为
$$\Delta t
\le
\frac{1}{
\max f(\rho)\left(\frac1{\Delta x}+\frac1{\Delta y}\right)
},$$
这正是标准二维迎风格式 CFL 条件。

### (2) 主方向与网格对齐
若 $\tau=e_x$, $n=e_y$，则
$$M=
\begin{pmatrix}
\alpha & 0\\
0 & \beta
\end{pmatrix}.$$
于是
$$\Delta t
\le
\frac{1}{
\max f(\rho)\left(\frac{\sqrt{\alpha}}{\Delta x}+\frac{\sqrt{\beta}}{\Delta y}\right)
}.$$
说明沿通道方向若 $\alpha$ 较大，数值传播速度更快，相应时间步长必须更小，这与物理和数值直觉一致。

### (3) 旋转通道情形
设方形网格 $\Delta x=\Delta y=h$，且通道方向与 $x$ 轴成 $45^\circ$。
则
$$\tau=\frac1{\sqrt2}(1,1),\qquad
n=\frac1{\sqrt2}(-1,1).$$
于是
$$m_{11}=m_{22}=\frac{\alpha+\beta}{2}.$$
改进 CFL 变成
$$\Delta t
\le
\frac{h}{
\max f(\rho)\,\sqrt{2(\alpha+\beta)}
}.$$
而谱上界给出
$$\Delta t
\le
\frac{h}{2\max f(\rho)\sqrt{\max(\alpha,\beta)}}.$$
例如 $\alpha=9,\beta=1$ 时，
$$\Delta t_{\text{improved}}=\frac{h}{\max f(\rho)\sqrt{20}},
\qquad
\Delta t_{\text{spectral}}=\frac{h}{6\max f(\rho)}.$$
数值上
$$\frac{1}{\sqrt{20}}\approx 0.224,\qquad \frac16\approx 0.167.$$
说明改进 CFL 允许更大的时间步长，同时仍然保留张量各向异性信息，确实更高效。

## 10. 对显式密度更新格式的最终建议
若密度用显式守恒格式更新：
$$\rho_{ij}^{n+1}
=
\rho_{ij}^n
-\frac{\Delta t}{\Delta x}(F_{i+\frac12,j}-F_{i-\frac12,j})
-\frac{\Delta t}{\Delta y}(G_{i,j+\frac12}-G_{i,j-\frac12}),$$
且面通量采用迎风值，则建议在每个时间步按当前 $\rho^n,\phi^n$ 计算局部张量速度上界，并选取
$$\boxed{
\Delta t^n
=
\mathrm{CFL}\cdot
\frac{1}{
\displaystyle
\max_{(i,j)}
f(\rho^n_{ij})
\left(
\frac{\sqrt{m^n_{11,ij}}}{\Delta x}
+
\frac{\sqrt{m^n_{22,ij}}}{\Delta y}
\right)
}
}$$
其中 $0<\mathrm{CFL}<1$，例如取 $0.5\sim 0.9$。

# 升级（2026-03-26）：
为何在多群体、各向异性张量、固定概率转移下需要重新表述 CFL 条件。
## 2. 多群体显式输运更新
在“多阶段—多路线群体”模型中，每个群体 $(s,r)$ 的密度更新写为
$$\rho_{s,r}^{n+1}
=
\rho_{s,r}^{n}
-
\Delta t\,
\nabla_h\cdot(\rho_{s,r}^{n}\mathbf v_{s,r}^{n})
+
\Delta t\,
\bigl(Q_{s,r}^{\mathrm{in},n}-Q_{s,r}^{\mathrm{out},n}\bigr).$$
其中：
- 第一项为旧时刻密度；
- 第二项为显式输运项；
- 第三项为固定概率分流带来的群体转移项。

CFL 条件主要约束显式输运项的稳定性，而转移项将额外引入一个非负性时间步限制。

## 3. 各向异性速度场与方向速度上界
对每个群体 $(s,r)$，速度场写为
$$\mathbf v_{s,r}
=
f(\rho^{\mathrm{tot}})
\frac{-M_{s,r}\nabla \phi_{s,r}}
{\sqrt{\nabla\phi_{s,r}^\top M_{s,r}\nabla\phi_{s,r}}}.$$
设 $p=\nabla\phi_{s,r}$，则对任意单位方向 $\xi\in\mathbb R^2$，有
$$|\xi\cdot \mathbf v_{s,r}|
\le
f(\rho^{\mathrm{tot}})\sqrt{\xi^\top M_{s,r}\xi}.$$
因此，群体 $(s,r)$ 沿方向 $\xi$ 的局部传播速度上界为
$$a_{\xi}^{(s,r)}
=
f(\rho^{\mathrm{tot}})
\sqrt{\xi^\top M_{s,r}\xi}.$$

## 4. Cartesian 网格上的改进 CFL 条件
在 Cartesian 网格上，取
$$e_x=(1,0)^\top,\qquad e_y=(0,1)^\top,$$
则对每个群体 $(s,r)$ 有
$$|v_{x}^{(s,r)}|
\le
f(\rho^{\mathrm{tot}})\sqrt{m_{11}^{(s,r)}},
\qquad
|v_{y}^{(s,r)}|
\le
f(\rho^{\mathrm{tot}})\sqrt{m_{22}^{(s,r)}}.$$
因此其局部 CFL 限制可写成
$$\Delta t
\le
\frac{1}{
f(\rho^{\mathrm{tot}})
\left(
\frac{\sqrt{m_{11}^{(s,r)}}}{\Delta x}
+
\frac{\sqrt{m_{22}^{(s,r)}}}{\Delta y}
\right)
}.$$

## 5. 多群体系统中的全局 CFL 条件
由于所有群体均采用显式更新，为保证整个系统稳定，时间步长应对所有群体同时满足 CFL 条件。因此取全局上界：
$$\boxed{
\Delta t_{\mathrm{adv}}^n
=
\mathrm{CFL}\cdot
\frac{1}{
\displaystyle
\max_{(s,r)}\max_{(i,j)}
f(\rho_{ij}^{\mathrm{tot},n})
\left(
\frac{\sqrt{m_{11,ij}^{(s,r),n}}}{\Delta x}
+
\frac{\sqrt{m_{22,ij}^{(s,r),n}}}{\Delta y}
\right)
}
}$$
其中 $0<\mathrm{CFL}<1$。
若不同群体共享同一个张量场 $M$，则外层 $\max_{(s,r)}$ 可省略。

## 6. 固定概率转移项的显式时间步限制
设群体 $(s,r)$ 在决策区 $G_{s,r}$ 内，以切换率 $\kappa_{s,r}$ 按固定概率分流到下一阶段各群体，则连续转移项为
$$Q_{(s,r)\to(s+1,q)}
=
p_{(s,r)\to q}\,
\kappa_{s,r}\,
\chi_{G_{s,r}}\,\rho_{s,r}.$$
因此该群体在单位时间内的总流出率为
$$Q_{s,r}^{\mathrm{out}}
=
\sum_q Q_{(s,r)\to(s+1,q)}
=
\kappa_{s,r}\,\chi_{G_{s,r}}\,\rho_{s,r},$$
因为
$$\sum_q p_{(s,r)\to q}=1.$$
为保持显式更新后的密度非负，必须要求单步流出不超过当前密度，即
$$\Delta t\,\kappa_{s,r}\le 1.$$
因此固定概率转移项对应的时间步限制为
$$\boxed{
\Delta t_{\mathrm{transfer}}
\le
\frac{1}{\displaystyle\max_{(s,r)}\kappa_{s,r}}
}$$
等价地，若记离散转移比例
$$\eta_{s,r}=\kappa_{s,r}\Delta t,$$
则需满足
$$0\le \eta_{s,r}\le 1.$$

## 7. 总时间步长选择原则
综合输运 CFL 条件、固定概率转移项非负性限制以及人为设置的时间步上限 $dt_{\mathrm{cap}}$，在每个时间步应取
$$\boxed{
\Delta t^n
=
\min\left\{
\Delta t_{\mathrm{adv}}^n,\;
\Delta t_{\mathrm{transfer}},\;
dt_{\mathrm{cap}}
\right\}.
}$$
这保证：
- 输运项满足显式 CFL 稳定性；
- 固定概率分流不会在单步内转移超过当前群体质量；
- 时间步长不会超过设定的工程上限。

## 8. 特例验证
(1) 各向同性无分流情形
若
$$M_{s,r}=I,\qquad \kappa_{s,r}=0,$$
则有
$$m_{11}^{(s,r)}=m_{22}^{(s,r)}=1,$$
时间步条件退化为经典二维显式迎风 CFL：
$$\Delta t
\le
\frac{1}{
\max f(\rho^{\mathrm{tot}})
\left(
\frac1{\Delta x}+\frac1{\Delta y}
\right)
}.$$
(2) 固定概率分流但无输运
若暂不考虑空间输运，仅考虑群体转移，则稳定性条件退化为
$$\Delta t\le \frac{1}{\max_{(s,r)}\kappa_{s,r}},$$
即要求每一步的转移比例不超过 1。
(3) 多群体共享拥堵
若不同群体共享总密度 $\rho^{\mathrm{tot}}$，则拥堵会共同影响所有群体速度大小，而全局时间步长应由最不利群体决定。
