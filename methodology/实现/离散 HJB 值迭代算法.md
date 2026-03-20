

### 内容
当前模型中的
$$\phi(x)=\min_{u\in U(x)}\left(\phi(x+\Delta x\,u)+\frac{\Delta x}{f(\rho)}\frac1{\sqrt{u^\top M(x)u}}\right)$$
系统地推导成一个离散 HJB 值迭代算法

## 1. 从连续 HJB 到离散 Bellman
当前的连续控制模型是：
$$\max_{u\in U(x)}\{-f(\rho(x))\,u\cdot \nabla \phi(x)\}=1.$$
等价地可写成
$$\min_{u\in U(x)}
\left\{
u\cdot \nabla \phi(x)+\frac{1}{f(\rho(x))}
\right\}=0$$
这是在没有显式方向代价张量时的基本形式。
若同时考虑各向异性代价 $M(x)$，则更自然的局部步进代价写成
$$\text{stepCost}(x,u)=\frac{\Delta x}{f(\rho(x))}\frac{1}{\sqrt{u^\top M(x)u}}.$$
于是值函数满足局部 Bellman 最优性关系：
$$\phi(x)=\min_{u\in U(x)}
\left\{
\phi(x+\Delta x\,u)+\text{stepCost}(x,u)
\right\}.$$
这就是你写出的离散 HJB / 离散 Bellman 方程。
它的含义很直接：
- 从当前位置 $x$ 先走一步到 $x+\Delta x u$；
- 这一步付出时间代价 $\text{stepCost}(x,u)$；
- 然后再加上从新位置到出口的最优剩余代价；
- 对所有允许方向取最小。
## 2. 网格离散化
设计算区域 $\Omega$ 被剖分成二维网格点
$$x_{ij}=(i\Delta x,\; j\Delta y).$$
为简洁起见，下面先假设
$$\Delta x=\Delta y=h.$$
记
- $\phi_{ij}\approx \phi(x_{ij})$,
- $\rho_{ij}\approx \rho(x_{ij})$,
- $M_{ij}=M(x_{ij})$,
- $U_{ij}=U(x_{ij})$.


则离散 Bellman 方程写成
$$\phi_{ij}
=
\min_{u\in U_{ij}}
\left\{
\phi\bigl(x_{ij}+h\,u\bigr)
+
\frac{h}{f(\rho_{ij})}\frac{1}{\sqrt{u^\top M_{ij}u}}
\right\}.$$

## 3. 控制集离散化
为了真正实现算法，控制集 $U(x)$ 还要离散成有限方向集合。
设
$$U_{ij}^{(d)}=\{u^{(1)},u^{(2)},\dots,u^{(K)}\},
\qquad |u^{(k)}|=1,$$
例如可取：
- 四邻域方向：
$$(1,0),(-1,0),(0,1),(0,-1)$$
- 八邻域方向：
$$(\pm1,0),(0,\pm1),\frac{1}{\sqrt2}(\pm1,\pm1)$$
- 或更细的角度离散：
$$u^{(k)}=(\cos\theta_k,\sin\theta_k),\quad \theta_k=\frac{2\pi k}{K}.$$


若有单向通道约束，则只保留满足约束的方向，例如：
$$U_{ij}^{(d)}=
\left\{
u^{(k)}:\; u^{(k)}\cdot (s_c\tau_{ij})>0
\right\}$$
或严格单向时
$$U_{ij}^{(d)}=\{s_c\tau_{ij}\}.$$
于是离散 Bellman 变成有限维最小化：
$$\phi_{ij}
=
\min_{k=1,\dots,K_{ij}}
\left\{
\phi\bigl(x_{ij}+h\,u^{(k)}\bigr)
+
\frac{h}{f(\rho_{ij})}\frac{1}{\sqrt{(u^{(k)})^\top M_{ij}u^{(k)}}}
\right\}.$$

## 4. 邻点不落在网格上时的插值
若 $u^{(k)}$ 不是轴向或对角方向，则点
$$x_{ij}+h\,u^{(k)}$$
通常不落在网格节点上。
这时要对 $\phi$ 做插值：
$$\phi(x_{ij}+h\,u^{(k)})\approx \mathcal I_h[\phi](x_{ij}+h\,u^{(k)}),$$
其中 $\mathcal I_h$ 可取双线性插值。
因此离散更新式更准确地写为
$$\phi_{ij}
=
\min_{u\in U_{ij}^{(d)}}
\left\{
\mathcal I_h[\phi](x_{ij}+h\,u)
+
\frac{h}{f(\rho_{ij})}\frac{1}{\sqrt{u^\top M_{ij}u}}
\right\}.$$
这就是数值实现中常说的半拉格朗日离散 HJB。

## 5. 边界条件
值迭代算法必须先指定边界条件。
### 5.1 出口边界
若 $\Gamma_{\text{exit}}$ 是出口边界，则通常取
$$\phi(x)=0,\qquad x\in \Gamma_{\text{exit}}.$$
离散上即
$$\phi_{ij}=0,\qquad x_{ij}\in \Gamma_{\text{exit}}.$$
### 5.2 障碍物边界
若 $\Gamma_{\text{obs}}$ 是不可通行障碍边界，则通常取
$$\phi(x)=+\infty$$
或赋一个很大值。离散上可写成
$$\phi_{ij}=+\infty,\qquad x_{ij}\in \Gamma_{\text{obs}}.$$
### 5.3 其他区域
其余点初始化为一个大值：
$$\phi_{ij}^{(0)}=
\begin{cases}
0, & x_{ij}\in \Gamma_{\text{exit}},\\
+\infty, & \text{其他可行点}.
\end{cases}$$

## 6. 值迭代公式的推导
现在推导值迭代。
定义 Bellman 算子 $T$：
$$(T\phi)_{ij}
=
\min_{u\in U_{ij}^{(d)}}
\left\{
\mathcal I_h[\phi](x_{ij}+h\,u)
+
\frac{h}{f(\rho_{ij})}\frac{1}{\sqrt{u^\top M_{ij}u}}
\right\}.$$
离散 HJB 的求解就是找不动点
$$\phi=T\phi.$$
因此最自然的值迭代就是：
$$\phi^{(n+1)}=T\phi^{(n)}.$$
即在每个内点上更新
$$\phi_{ij}^{(n+1)}
=
\min_{u\in U_{ij}^{(d)}}
\left\{
\mathcal I_h[\phi^{(n)}](x_{ij}+h\,u)
+
\frac{h}{f(\rho_{ij})}\frac{1}{\sqrt{u^\top M_{ij}u}}
\right\}.$$
这就是离散 HJB 值迭代算法的核心公式。

## 7. 单调值迭代形式
由于出口代价是 0，其他点初始为大值，因此更常用的写法是“单调下降更新”：
$$\phi_{ij}^{(n+1)}
=
\min\left\{
\phi_{ij}^{(n)},
\;
\min_{u\in U_{ij}^{(d)}}
\left[
\mathcal I_h[\phi^{(n)}](x_{ij}+h\,u)
+
\frac{h}{f(\rho_{ij})}\frac{1}{\sqrt{u^\top M_{ij}u}}
\right]
\right\}.$$
这样可保证
$$\phi_{ij}^{(n+1)}\le \phi_{ij}^{(n)},$$
即值函数逐步下降，符合“最短时间不断改进”的直觉。

## 8. 收敛判据
迭代直到满足
$$\|\phi^{(n+1)}-\phi^{(n)}\|_\infty<\varepsilon$$
为止，其中 $\varepsilon>0$ 是给定容差。
即
$$\max_{i,j}|\phi_{ij}^{(n+1)}-\phi_{ij}^{(n)}|<\varepsilon.$$

## 9. 最优方向的离散恢复
当 $\phi$ 收敛后，在每个网格点上取使 Bellman 更新最小的控制作为最优方向：
$$u_{ij}^*
=
\operatorname*{arg\,min}_{u\in U_{ij}^{(d)}}
\left\{
\mathcal I_h[\phi](x_{ij}+h\,u)
+
\frac{h}{f(\rho_{ij})}\frac{1}{\sqrt{u^\top M_{ij}u}}
\right\}.$$
然后速度场为
$$\mathbf v_{ij}=f(\rho_{ij})\,u_{ij}^*.$$
这和你前面总结的公式完全一致，只是现在明确成了可计算的离散版本。

## 10. 整个离散 HJB 值迭代算法
可以整理成下面的算法步骤。

### 算法：离散 HJB 值迭代
#### 输入
- 当前密度场 $\rho_{ij}$；
- 速度函数 $f(\rho)$；
- 各向异性张量 $M_{ij}$；
- 离散允许方向集 $U_{ij}^{(d)}$；
- 出口集合 $\Gamma_{\text{exit}}$；
- 障碍物集合 $\Gamma_{\text{obs}}$；
- 网格步长 $h$；
- 收敛阈值 $\varepsilon$。


#### 初始化
$$\phi_{ij}^{(0)}=
\begin{cases}
0, & x_{ij}\in \Gamma_{\text{exit}},\\
+\infty, & x_{ij}\in \Gamma_{\text{obs}} \text{ 或其他未到达点}.
\end{cases}$$
#### 迭代
对所有非边界可行网格点 $x_{ij}$，计算
$$\phi_{ij}^{(n+1)}
=
\min_{u\in U_{ij}^{(d)}}
\left\{
\mathcal I_h[\phi^{(n)}](x_{ij}+h\,u)
+
\frac{h}{f(\rho_{ij})}\frac{1}{\sqrt{u^\top M_{ij}u}}
\right\}.$$
若采用单调版本，则写成
$$\phi_{ij}^{(n+1)}
=
\min\left\{\phi_{ij}^{(n)},\;
\min_{u\in U_{ij}^{(d)}}
\left(
\mathcal I_h[\phi^{(n)}](x_{ij}+h\,u)
+
\frac{h}{f(\rho_{ij})}\frac{1}{\sqrt{u^\top M_{ij}u}}
\right)
\right\}.$$
#### 停止条件
若
$$\max_{i,j}|\phi_{ij}^{(n+1)}-\phi_{ij}^{(n)}|<\varepsilon,$$
则停止。
输出


收敛势场 $\phi_{ij}$；


最优方向
$$u_{ij}^*=\arg\min_{u\in U_{ij}^{(d)}}\left\{\mathcal I_h[\phi](x_{ij}+h\,u)+\frac{h}{f(\rho_{ij})}\frac{1}{\sqrt{u^\top M_{ij}u}}\right\};$$


速度场
$$\mathbf v_{ij}=f(\rho_{ij})u_{ij}^*.$$