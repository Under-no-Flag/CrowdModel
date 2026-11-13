# 连续介质模型仿真场景构建与 PDE 耦合说明

## 1. 文档目的

本文说明当前仓库中的连续介质人群模型如何构建仿真场景，以及场景几何如何进入 Bellman--HJB 路径选择方程和密度守恒方程。重点回答以下问题：

1. 计算域、墙体、障碍物、可行区域、出口和人群区域分别如何定义；
2. AnyLogic 外滩场景如何转换为计算网格；
3. 几何对象如何编译为 NumPy mask；
4. `walkable`、`wall_cost`、目标 mask 和出口 mask 如何进入 PDE 数值求解；
5. 新场景接入时应检查哪些几何和数值问题。

本文以当前源码为准。场景配置字段可同时参考 `codes/scenes/README.md`；若需要在另一个项目中复用求解器，可继续参考 `codes/doc/Hughes计算引擎复用指南.md`。

## 2. 场景不是一张背景图，而是一组参与计算的离散集合

当前模型中的连续计算域记为

$$
\Omega\subset\mathbb R^2.
$$

离散实现使用大小为 `(ny, nx)` 的二维 Cartesian 网格，每个数组元素代表一个网格单元。代码统一按 `array[y, x]` 访问，空间步长由 `run.toml` 中的 `dx` 给出，当前实现假设两个方向采用相同网格步长：

$$
\Delta x=\Delta y=dx.
$$

场景最终不是以 PNG 或 GIS 图层直接进入 PDE，而是编译为以下离散对象：

| 离散对象 | 类型和形状 | 数学含义 | 主要用途 |
|---|---|---|---|
| `walkable` | `bool[ny,nx]` | 可行区域 $\Omega_w$ 的指示函数 | 限制 Bellman 搜索、速度和质量通量 |
| `wall_mask` | `bool[ny,nx]` | 显式 `[[walls]]` 栅格化结果 | 墙体诊断和绘图 |
| `wall_distance_cells` | `float[ny,nx]` | 到最终不可通行集合的离散距离 | 生成靠墙路径代价 |
| `wall_cost` | `float[ny,nx]` | $C_{\mathrm{wall}}(x)$ | 乘入 Bellman 单步代价 |
| `region_masks[name]` | `bool[ny,nx]` | 命名区域的指示函数 | 人群、目标、出口、通道和控制区域的公共几何基础 |
| `goal_mask` | `bool[ny,nx]` | 当前阶段目标集合 $G_g$ | Bellman 边界条件 $\phi_g=0$ |
| `sink_mask` | `bool[ny,nx]` | 吸收/离开计算系统的区域 | 从守恒更新后的密度中移除质量 |
| `initial_rho` | `float[ny,nx]` | 初始密度 $\rho(x,0)$ | PDE 初值 |

因此，地图或场景图片只负责提供几何参考。真正决定行人能否通过某个位置的是 `walkable`，决定去往何处的是每个阶段的 `goal_mask`，决定何时离开系统的是 `sink_mask`。

## 3. 场景数据流

当前配置驱动链路为：

```text
原始几何
  ├─ 手工定义的矩形、任意多边形和折线墙
  └─ AnyLogic .alp 中的 Image、RectangleNode、Path
                  │
                  ▼
scene.toml + population.toml + routes.toml + run.toml
                  │
                  ▼
load_scene_spec() / load_population_spec() / load_route_spec()
                  │
                  ▼
compile_scene()
  ├─ region_masks
  ├─ wall_mask
  ├─ walkable
  ├─ wall_distance_cells
  ├─ wall_cost
  └─ exit/channel masks
                  │
                  ▼
compile_case()
  ├─ initial_rho / initial_group_density
  ├─ per-group goal_mask / sink_mask
  ├─ allowed_mask
  └─ M fields and stage transitions
                  │
                  ▼
simulate_case()
  ├─ Bellman/HJB 路径选择
  ├─ 速度场恢复
  └─ 有限体积守恒更新
```

`scene.toml` 负责几何，`population.toml` 负责密度初值和可选持续入流，`routes.toml` 负责把命名区域解释为阶段目标、吸收区和路线，`run.toml` 负责网格尺度和数值参数。几何定义与 PDE 语义是分开的：一个 `[[regions]]` 只有在被 `obstacles`、`[[exits]]`、population 或 routes 引用后才获得相应含义。

## 4. 计算网格与坐标约定

### 4.1 网格大小

`run.toml` 中的三个字段确定计算网格：

```toml
[simulation]
nx = 512
ny = 224
dx = 0.125
```

数组形状为 `(224, 512)`。配置意义下的计算域尺度为

$$
L_x=nx\cdot dx,\qquad L_y=ny\cdot dx.
$$

上述示例对应 `64 × 28` 个配置长度单位。`dx` 的物理单位必须由场景标定确定，转换脚本不会自动从背景图推断米制尺度。

### 4.2 数组坐标

- TOML 几何点写为 `[x, y]`；
- NumPy 数组访问顺序为 `[y, x]`；
- 当前 AnyLogic 转换不翻转 Y 轴，AnyLogic 绘图空间和 NumPy 预览都采用向下为正的屏幕坐标；
- `block_boundaries = true` 时，数组最外一圈单元直接标为不可通行。

### 4.3 矩形区域

矩形采用左闭右开切片：

```toml
[[regions]]
name = "spawn"
x0 = 20
x1 = 40
y0 = 60
y1 = 90
```

对应：

```python
mask[60:90, 20:40] = True
```

超出网格的坐标会被裁剪到 `[0,nx] × [0,ny]`。若裁剪后宽度或高度不为正，则生成空 mask。

### 4.4 多边形区域

旋转通道、倾斜目标区等使用多边形：

```toml
[[regions]]
name = "channel_1"
shape = "polygon"
points = [[115.375, 75.583], [127.678, 81.056], [126.664, 83.332], [114.361, 77.860]]
axis = [0.914, 0.406]
```

编译器对候选单元中心 `(x+0.5,y+0.5)` 执行点在多边形内判定，因此一个轴对齐多边形与采用相同边界的左闭右开矩形会得到一致 mask。恰好落在多边形边上的单元中心也视为区域内部。

`axis` 表示区域主轴，主要供通道方向、斜向截面和几何引导使用；它不决定多边形是否可通行。若未显式给出，编译器采用多边形最长边方向并归一化。

## 5. 墙体、障碍物与可行区域

### 5.1 两种实体几何

当前实现支持两种不可通行几何。

第一种是区域障碍物。先定义矩形或多边形，再把名称放入 `obstacles`：

```toml
obstacles = ["building", "closed_area"]

[[regions]]
name = "building"
x0 = 100
x1 = 140
y0 = 30
y1 = 80

[[regions]]
name = "closed_area"
shape = "polygon"
points = [[160, 20], [190, 35], [175, 70], [150, 55]]
```

第二种是折线墙。每条墙由至少两个点和一个正宽度组成：

```toml
[[walls]]
name = "riverside_wall"
shape = "polyline"
points = [[20, 40], [80, 50], [140, 65]]
width = 1.0
```

`obstacles` 中的命名区域只有被列出后才会变成实体障碍；普通 `[[regions]]` 默认只是语义区域，不会自动阻挡行人。所有 `[[walls]]` 则自动作为实体墙参与可行域编译，不需要再写入 `obstacles`。

### 5.2 折线墙的栅格化

对墙体相邻控制点形成的每条线段，编译器检查局部包围盒内网格位置到线段的欧氏距离。当

$$
d((x,y),\text{segment})\le \frac{\text{width}}{2}
$$

时，该网格单元被标记为墙体。`width` 使用网格坐标单位；若 `dx` 代表物理长度，则墙体近似物理宽度为 `width × dx`。墙体过细可能因栅格化出现断点，过宽则会侵占通道，因此需要结合 `grid_overlay.png` 和编译后 `wall_mask` 检查。

### 5.3 可行区域的生成

编译过程可写为：

$$
\Omega_{\mathrm{solid}}
=
\Omega_{\mathrm{obs}}\cup\Omega_{\mathrm{wall}},
$$

$$
\Omega_w
=
\Omega\setminus\Omega_{\mathrm{solid}}.
$$

离散实现按以下顺序执行：

1. 初始化 `walkable = True`；
2. 若 `block_boundaries=true`，把上下左右最外一圈置为 `False`；
3. 栅格化所有命名区域；
4. 合并 `obstacles` 引用的区域得到 `obstacle_union`；
5. 栅格化并合并全部 `[[walls]]` 得到 `wall_union`；
6. 计算 `solid_union = obstacle_union | wall_union`；
7. 执行 `walkable[solid_union] = False`；
8. 可选地对实体区域做安全净距膨胀；
9. 基于最终 `~walkable` 计算墙距和靠墙代价。

### 5.4 墙体安全净距

`wall_clearance_cells` 用于把墙体和障碍物向外膨胀：

```toml
[simulation]
wall_clearance_cells = 2
```

每次膨胀把当前实体单元的 8 邻域一并设为不可通行。它直接改变 $\Omega_w$，属于硬几何处理。若网格物理步长为 `dx`，名义净距约为 `wall_clearance_cells × dx`，但对角方向使用方形/8 邻域膨胀，其几何边界不是严格欧氏圆。

### 5.5 对角穿墙阻断

Bellman 求解使用 E、W、N、S、NE、NW、SE、SW 八个方向。如果一个对角目标单元可行，但两者之间至少一个正交相邻单元不可行，直接走对角线可能切过墙角。默认 `block_diagonal_corner_cutting = true` 会从该源单元的 `allowed_mask` 中删除对应对角方向。

这个规则不会新增墙体单元，但会修改离散可达图，是保证路径不穿过障碍尖角的重要补充。

### 5.6 墙体避让代价

硬墙之外，模型还可对靠近任意不可通行区域的可行单元增加软代价。编译器先计算

$$
d_{\mathrm{wall}}(x)
=
\operatorname{dist}_{\mathrm{grid}}(x,\Omega\setminus\Omega_w),
$$

再生成

$$
C_{\mathrm{wall}}(x)
=
\begin{cases}
1+\gamma_{\mathrm{wall}}
\exp\left(-d_{\mathrm{wall}}(x)/\sigma_{\mathrm{wall}}\right),
& d_{\mathrm{wall}}(x)\le r_{\mathrm{wall}},\\
1,&\text{其他位置}.
\end{cases}
$$

对应配置为：

```toml
[simulation]
wall_avoidance_weight = 2.0
wall_avoidance_sigma_cells = 1.5
wall_avoidance_radius_cells = 4.0
```

这里的距离源是最终 `~walkable`，因此不仅包括显式 `[[walls]]`，还包括区域障碍和被封闭的外边界。该机制不改变可行区域，也不直接改变密度或删除质量，只提高靠墙路径的 Bellman 代价。默认 `wall_avoidance_weight=0` 时，`wall_cost` 全为 1。

## 6. 出口、目标、吸收区和初始人群

### 6.1 场景出口

`scene.toml` 中的出口引用已有命名区域：

```toml
[[exits]]
name = "main_exit"
region = "exit_region"
```

多个出口 mask 会合并为场景 `exit_mask`。选择器最终与 `walkable` 相交，因此落在墙体或障碍上的出口单元会被自动剔除。

### 6.2 阶段目标与吸收区

`routes.toml` 决定每个群体实际求解到哪里：

```toml
[[stages]]
stage_id = "evacuation"
group_key = [1, 1]
goal_region = "main_exit"
sink_region = "main_exit"
```

- `goal_region` 生成 `goal_mask`，在 Bellman 方程中取 $\phi=0$；
- `sink_region` 生成 `sink_mask`，密度到达后从系统中移除；
- 多阶段模型中，中间目标可以只触发阶段转移而不移除质量；
- 一个命名区域不会因为名字含有 `goal` 或 `exit` 自动获得 PDE 语义，必须被 `[[exits]]` 或 stage 显式引用。

### 6.3 初始密度

`population.toml` 把密度赋给命名区域：

```toml
[[initial_groups]]
group_id = "visitors"
stage_id = "evacuation"
region = "spawn"
density = 2.0
```

编译时使用 `region_mask & walkable`，因此障碍单元不会获得初始密度。网格单元中的人数质量为

$$
m_{ij}=\rho_{ij}\,dx^2.
$$

初始总人数质量为所有可行单元上 $\rho_{ij}dx^2$ 之和。

## 7. 场景几何如何进入 Bellman--HJB 方程

### 7.1 连续与离散形式

对群体 $g$，势函数 $\phi_g(x,t)$ 表示从当前位置到当前目标的最小剩余代价。当前离散 Bellman 更新为

$$
\phi_g(x)
=
\min_{u\in U_g(x)}
\left[
\phi_g(x+\Delta x u)
+
\frac{\Delta x\,C_{\mathrm{wall}}(x)}{F(\rho(x,t))}
\frac{1}{\sqrt{u^\top M_g(x)u}}
\right].
$$

场景几何通过四条路径进入这个方程：

1. `walkable` 决定状态 $x$ 和邻点 $x+\Delta xu$ 是否存在于离散可达图；
2. `goal_mask` 提供 Dirichlet 边界条件 $\phi_g=0$；
3. `corner_block_mask` 和可选方向设置共同决定允许方向集合 $U_g(x)$；
4. `wall_cost` 乘入单步代价，改变靠墙路径的相对吸引力。

不可通行单元的预计算步进代价被设为 `inf`。Bellman 求解时还会显式跳过非 `walkable` 的源单元或邻点，因此势函数不会跨墙传播。障碍区域保持 $\phi=+\infty$。

### 7.2 速度恢复

求得势函数后，代码在每个可行单元选择使 Bellman 代价最小的八邻域方向 $u_g^*$，再使用 Greenshields 关系

$$
F(\rho)=v_{\max}\left(1-\frac{\rho}{\rho_{\max}}\right)_+
$$

恢复速度场：

$$
v_g(x,t)=F(\rho(x,t))u_g^*(x,t).
$$

随后执行：

```python
vx[~walkable] = 0.0
vy[~walkable] = 0.0
```

因此墙体和障碍内部速度严格为 0。

## 8. 场景几何如何进入密度守恒 PDE

### 8.1 守恒方程

每个群体密度满足

$$
\frac{\partial\rho_g}{\partial t}
+
\nabla\cdot(\rho_g v_g)
=
S_g.
$$

网格上由 `core.py` 的 `compute_face_fluxes()` 采用显式有限体积/迎风通量更新。以 X 方向相邻单元为例，先对单元中心速度取平均得到面速度，再根据面速度符号选择上风侧密度：

$$
F_{i+1/2,j}
=
v_{x,i+1/2,j}
\begin{cases}
\rho_{i,j},&v_{x,i+1/2,j}\ge0,\\
\rho_{i+1,j},&v_{x,i+1/2,j}<0.
\end{cases}
$$

### 8.2 墙体上的无通量条件

代码对每个内部网格面执行：

```python
fx = where(walkable_left & walkable_right, fx, 0)
fy = where(walkable_up & walkable_down, fy, 0)
```

只要一个面的任一侧不可通行，该面通量就被置为 0。这等价于在墙体和障碍边界施加离散无穿透条件：

$$
(\rho_g v_g)\cdot n=0
\qquad\text{on }\partial\Omega_{\mathrm{solid}}.
$$

墙体不会吸收或删除质量；被墙挡住的流量保留在上游单元。当前质量守恒测试明确检查了内部墙前的通量为 0 且闭域总质量不变。

### 8.3 外边界

建议普通封闭场景使用 `block_boundaries=true`。此时最外一圈单元不可通行，外边界通过与墙体相同的面通量屏蔽形成无通量边界。

`block_boundaries=false` 不会自动建立一般意义上的开放边界条件。当前散度更新主要作用在内部索引，不能把该选项理解为已经实现了给定入流、出流或辐射边界。外部持续入流由 `[[inflow_groups]]` 在区域内注入，出口则由 `sink_mask` 吸收；若需要严格 PDE 外边界条件，应单独扩展边界通量实现。

### 8.4 出口吸收

完成通量更新后，代码计算 `sink_mask` 上的质量并将这些单元密度置零：

$$
m_{\mathrm{sink}}^n
=
\sum_{K_i\subset\Omega_{\mathrm{sink}}}
\rho_i^{n+1}dx^2,
$$

$$
\rho_i^{n+1}=0,
\qquad K_i\subset\Omega_{\mathrm{sink}}.
$$

因此出口在路径选择层是零势目标，在守恒层是离散吸收区。中间阶段目标若未配置为 sink，则只发生群体转移，不删除总质量。

## 9. 当前外滩 AnyLogic 场景的构建过程

### 9.1 原始数据

当前外滩批处理脚本默认读取：

```text
anylogic-scene/BundScene/converted/
```

该目录由 `anylogic-scene/BundScene/BundScene.alp` 通过 `convert_anylogic_scene.py` 生成。转换器读取 `.alp` XML 中的：

- `Image`：确定背景图在 AnyLogic 空间中的参考框；
- 有特定命名规则的 `RectangleNode`：生成通道、初始区域、目标、出口和路线交接区域；
- `Path`：生成折线墙。

背景图像素本身不会被自动分割成墙体或可行域。若 AnyLogic 中没有对应的 RectangleNode 或 Path，一个建筑轮廓即使画在图片上也不会进入 PDE。

### 9.2 坐标转换

AnyLogic 点 $(x,y)$ 按图像参考框线性映射到网格：

$$
g_x=\frac{x-x_{\mathrm{frame}}}{W_{\mathrm{frame}}}nx,
\qquad
g_y=\frac{y-y_{\mathrm{frame}}}{H_{\mathrm{frame}}}ny.
$$

RectangleNode 会先按其旋转角计算四个角点，再把四个角点转换为 `shape="polygon"`。Path 默认采用 `anchors` 模式，即按 AnyLogic Path 的主锚点组成折线；每个点也通过同一映射进入网格。墙宽换算为：

$$
w_{\mathrm{grid}}
=
\max\left(1,\frac{w_{\mathrm{AnyLogic}}}{W_{\mathrm{frame}}}nx\right).
$$

### 9.3 当前默认转换结果

截至 2026-08-02，`converted` 目录的配置和编译结果为：

| 项目 | 当前值 |
|---|---:|
| 网格 | `512 × 224` |
| `dx` | `0.125` |
| 配置计算尺度 | `64 × 28` 长度单位 |
| 命名区域 | 17 |
| 折线墙 | 5 |
| 通道 | 4 |
| 场景出口 | 1 |
| 编译后可行单元 | 112125 |
| 编译后不可行或封闭边界单元 | 2563 |
| 其中显式折线墙单元 | 1095 |
| 出口单元 | 96 |
| 初始非零密度单元 | 88 |
| stage/group 数量 | 11 |
| 阶段转移数量 | 10 |
| `wall_clearance_cells` | 0 |
| 墙体软代价激活单元 | 0，因为默认权重为 0 |

`conversion_manifest.json` 中另有 `wall_obstacle_region_count = 392`。这些矩形来自折线墙的预览栅格诊断，并写入 `wall_obstacles.csv`；当前 `scene.toml` 的 `obstacles=[]`，真实计算使用的是 5 个 `[[walls]]`，不能把 392 个预览矩形误认为 PDE 同时加载了第二套墙体。

`converted_visual_2048x896` 是更高分辨率的转换/视觉产物，但当前多个外滩实验 runner 的默认 `base-dir` 仍指向 `converted`。切换到高分辨率场景时，必须同时检查 `nx`、`ny`、`dx`、墙宽、初始质量、CFL 时间步和计算成本，不能只替换 `scene.toml`。

## 10. 新场景的推荐构建步骤

### 第一步：确定物理尺度和网格

先确定实际范围、长度单位、`nx`、`ny` 和 `dx`。不要在几何已经画完后再随意更改 `nx/ny`，否则所有以网格坐标定义的区域都会错位。

### 第二步：定义实体边界

优先画出外边界、建筑、隔离栏和不可穿越墙体。面积型实体使用 polygon/rect + `obstacles`，线型边界使用 `[[walls]]`。确保墙体闭合或与封闭边界相接，避免一格宽的意外缺口。

### 第三步：定义语义区域

添加 spawn、goal、exit、decision、channel 等区域。它们可以重叠，但必须检查与 `walkable` 相交后的有效面积非零。

### 第四步：定义初值和路线

在 `population.toml` 中把初始密度绑定到 spawn，在 `routes.toml` 中把阶段绑定到 goal/sink。先用单群体单阶段验证场景，再增加多阶段和分流。

### 第五步：编译但不长时间仿真

先调用 loader/compiler，检查：

```python
import numpy as np
from crowd_bellman.compilers.config_compiler import compile_scene

bundle = compile_scene(scene_spec, simulation_config)
scene = bundle.scene

print(scene.walkable.shape)
print("walkable", np.count_nonzero(scene.walkable))
print("walls", np.count_nonzero(scene.wall_mask))
print("exit", np.count_nonzero(scene.exit_mask))
```

同时查看 `grid_overlay.png` 或自行渲染 `walkable`、`wall_mask`、出口和初始密度叠加图。

### 第六步：执行短程守恒验证

运行少量步数，检查：

- 势函数在目标区域为 0；
- 障碍内部势函数保持无穷或不可达；
- 速度在不可行单元为 0；
- 墙体两侧没有穿透通量；
- 无入口、无出口的闭域总质量保持不变；
- 有出口时，域内质量减少量与累计 sink 质量一致；
- 不存在由墙体缺口造成的异常最短路径。

## 11. 常见误区和限制

### 11.1 把背景图当作计算几何

背景图只用于参考和绘图。必须把墙体、障碍、出口和区域显式转换为 TOML 几何对象。

### 11.2 定义了区域但没有加入 obstacles

普通区域不会自动阻挡行人。面积障碍必须在 `obstacles` 中引用；折线墙则直接使用 `[[walls]]`。

### 11.3 把墙体软代价当作硬边界

`wall_cost>1` 只影响路径选择，不能阻止穿越。真正的无穿透边界必须由 `walkable=False` 和面通量屏蔽建立。

### 11.4 把 wall_mask 当作全部不可行区域

`wall_mask` 只记录显式折线墙。完整不可行集合应查看 `~walkable`，其中还包括外边界、obstacles 和 clearance 膨胀区域。

### 11.5 出口与目标概念混用

Bellman 使用每个群体的 `goal_mask`，密度移除使用 `sink_mask`。场景 `exit_mask` 是出口集合，但多阶段模型中的中间目标不应都设成 sink。

### 11.6 过低分辨率导致几何走样

窄通道可能消失，斜墙可能出现锯齿或单元缺口。应做网格加密对比，并根据 `dx` 重新评估墙宽和 clearance。

### 11.7 当前不是一般非结构网格 PDE

虽然输入支持任意多边形和折线，求解器内部仍是规则 Cartesian 网格与 8 邻域 Bellman 图。曲线边界会被栅格化，不能获得有限元/非结构网格级别的边界贴合精度。

## 12. 源码索引

- `anylogic-scene/BundScene/convert_anylogic_scene.py`：AnyLogic 坐标、区域和 Path 墙体转换。
- `codes/crowd_bellman/spec/scene_spec.py`：region、wall、exit、channel 配置数据结构。
- `codes/crowd_bellman/loaders/config_loader.py`：TOML 解析和字段校验。
- `codes/crowd_bellman/compilers/config_compiler.py`：几何栅格化、可行域、墙距、墙体代价和角点阻断。
- `codes/crowd_bellman/core.py`：Bellman、迎风面通量和守恒更新。
- `codes/crowd_bellman/runner.py`：场景对象进入每步闭环求解的位置。
- `codes/tests/test_scene_walls.py`：矩形、多边形、折线墙、墙体代价和角点阻断测试。
- `codes/tests/test_mass_conservation.py`：墙体无通量与质量守恒测试。
