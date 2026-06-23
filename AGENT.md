AGENT.md for codex
## 项目概述

这是博士学位论文研究的部分研究内容
包含代码、方法文档、实验记录与计划、参考资料、论文写作等多个方面的文件。

## 技术栈
- python
- latex
- markdown

## 规范
### 每次完成代码实现、方法文档撰写、实验记录更新、论文写作等任务后，需在 `records/` 目录下形成对应的日报，内容包括：
- 目标任务描述
- 已完成的具体任务和产物（如代码文件、文档、实验结果等）

## 外滩密度热力图高清化

外滩场景仿真后若需要生成更清晰、平滑且不跨墙扩散的密度热力图，应优先使用 `codes/render_refined_density_heatmap.py` 读取实验结果目录中的 `fields/field_step_XXXX.npz` 数据，而不是直接使用低分辨率 snapshot。仿真时需开启字段保存，例如在外滩 HCMBO 迁移实验中使用 `--save-field-data --field-save-every 80`，字段目录中应包含 `fields_manifest.json` 和 `static_masks.npz`，其中 `static_masks.npz` 提供 `walkable` mask 供 wall-aware smoothing 使用。

推荐高清渲染命令如下，将 `<case_dir>` 替换为具体实验 case 目录：

```powershell
python codes\render_refined_density_heatmap.py <case_dir> --all --output-dir <case_dir>\refined_density_wall_aware_hq --scale 12 --smooth-sigma 2.4 --color-scale frame-percentile --color-percentiles 0,98 --cmap low-density --gamma 0.42 --fusion-mode wall-preserve --density-alpha 1.0 --overlay-threshold 0.005 --alpha-gamma 0.32 --dpi 320
```

示例：

```powershell
python codes\render_refined_density_heatmap.py codes\results\bund_hcmbo_transfer_wall_avoid_1600_full\bund_hcmbo_transfer_wall_avoid --all --output-dir codes\results\bund_hcmbo_transfer_wall_avoid_1600_full\bund_hcmbo_transfer_wall_avoid\refined_density_wall_aware_hq --scale 12 --smooth-sigma 2.4 --color-scale frame-percentile --color-percentiles 0,98 --cmap low-density --gamma 0.42 --fusion-mode wall-preserve --density-alpha 1.0 --overlay-threshold 0.005 --alpha-gamma 0.32 --dpi 320
```

参数约定：`--scale` 控制密度场插值倍率，`--smooth-sigma` 控制平滑强度，`--color-scale frame-percentile` 使每帧颜色相对当前密度分布更明显，`--cmap low-density` 使用低密度强化的非均匀色阶，`--gamma 0.42` 强化低密度颜色响应，`--fusion-mode wall-preserve` 会保留墙/障碍背景，同时让所有非墙有限密度单元都按密度色表填充，包括密度为 0 的区域，`--density-alpha 1.0` 使密度层完全覆盖非墙背景。脚本会自动从 `summary.json` 或配置快照定位 `grid_overlay.png` 作为场景结构背景；若背景定位失败，可显式传入 `--background <grid_overlay.png>`。生成外滩论文或汇报图时，优先使用 `refined_density_wall_aware_hq/density_step_1599.png` 等高清图。

## 外滩有/无管控密度热力图拼图

对 `bund_control_comparison_*` 对比实验目录，若需要生成 2 行 4 列的有/无管控高清密度对比图，使用 `codes/render_bund_comparison_panel.py`。该脚本读取 `controlled/controlled/fields` 与 `uncontrolled/uncontrolled/fields` 中保存的 `field_step_XXXX.npz`，按 `scene.toml` 中的 `[[walls]]` polyline 范围自动裁剪外滩主体区域，避免整张网格带来过多留白；两行分别为 `Uncontrolled` 与 `Controlled`，四列为输入的 4 个 step，并使用统一色条。

推荐命令如下，将 `<comparison_dir>` 替换为某个外滩对比实验目录，后面 4 个数字替换为需要展示的 step：

```powershell
python codes\render_bund_comparison_panel.py <comparison_dir> 40 400 800 1590 --output <comparison_dir>\figures\bund_comparison_density_panel_steps_0040_0400_0800_1590.png --scale 8 --smooth-sigma 5 --vmax 6 --dpi 240
```

示例：

```powershell
python codes\render_bund_comparison_panel.py codes\results\bund_control_comparison_hcmbo 40 400 800 1590 --output codes\results\bund_control_comparison_hcmbo\figures\bund_comparison_density_panel_steps_0040_0400_0800_1590.png --scale 8 --smooth-sigma 5 --vmax 6 --dpi 240
```

注意：输入的 step 必须已在 `fields_manifest.json` 中保存，例如每 10 step 保存时可使用 `40 400 800 1590`，最终帧可能是 `1599` 而不是 `1600`。若希望自动使用最近保存帧，可追加 `--nearest`。常用参数中，`--crop-padding` 控制墙线范围外保留的网格边距，默认 10；`--vmax` 控制右侧色条最大密度；`--gamma 0.42` 与低密度强化色阶配合，使低密度区域也可见。

## 写作要点
- 撰写论文内容段落时，句子之间逻辑严密，切题。
- 不要分点要成一个段落、尽量不使用双引号和破折号。
- 撰写论文方法部分时，一定要先阅读methodology目录下的相关文档，理解方法细节后再撰写。
- 上下文一致性：
    - 1. 标题、名词一致性


# 论文评价指标编号规范
代码和历史实验记录中可能保留早期变量名，例如 $J_5$、$J_B$、$J_R$。这些名称来自方案设计阶段的中间指标删减，不应直接出现在最终论文展示层。

论文正文、图、表、caption 和公式中统一使用连续编号：
- $J_1$：efficiency / total travel time；
- $J_2$：safety / high-density exposure；
- $J_3$：load balance / realized channel-throughput variance，代码或历史文件中的旧 $J_5$ 在论文中写作 $J_3$；
- $J_4$：entrance waiting / blocking，代码或历史文件中的旧 $J_B$ 在论文中写作 $J_4$；
- $J_5$：control smoothness，代码或历史文件中的旧 $J_R$ 在论文中写作 $J_5$。

对应权重在论文中写作 $\lambda_1,\ldots,\lambda_5$，标量目标优先写作 $J(z)=\sum_{k=1}^{5}\lambda_k\tilde J_k(z)$ 或等价展开式。除非明确讨论代码实现或历史字段，不要在论文展示层使用 $J_B$、$J_R$ 或把 load-balance 指标写成 $J_5$。
