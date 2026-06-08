# 2026-06-08 入口通行速率响应章节修订

## 目标任务描述

修订 `writing/期刊论文章节/05-实验.md` 中 5.4 节，使其与方法章的“内部入口通行速率”定义一致，并修复该节对应图件中文字重叠问题。按要求尽量不重跑实验，仅基于既有结果文件重画图。

## 已完成的具体任务和产物

- 将 5.4 标题、段落、表头和图注统一为“内部入口通行速率控制”“速率上限”“未放行企图通量”等表述，避免将 $q$ 误解为外部边界容量或总容量。
- 同步修订 5.5、5.6 和 5.7 中与入口通行速率相关的术语，保持实验章内部一致。
- 在 `codes/g2_capacity_response_runner.py` 中新增 `--redraw-only` 模式，可从既有 `g2_capacity_response_summary.csv` 和 `timeseries.csv` 重画图表，不触发仿真。
- 重画并同步以下图件到 `writing/images/experiment_result_archive/g2/`：
  - `g2_capacity_levels.png`
  - `g2_capacity_allocation_loads.png`
  - `g2_capacity_response_pareto.png`
  - `g2_waiting_mass_timeseries.png`
- 对图件排版进行修复：缩短标签、移动图例、调整注释偏移和坐标边距，消除原先图中文字重叠问题。
- 补充 5.4 中“入口绑定时间比例”的定义，并将表头改为“绑定时间比例”，说明其为速率上限实际起约束作用的时间步占比。
- 按论文排版反馈将图 6 的图例移动到图下方，并用 `--redraw-only` 从既有结果重画，未重跑实验。
- 按论文精简要求删除 5.4 中原图 7、原图 8 及其对应的空间分配、时间调度分析内容，并移除论文图片归档目录中的对应副本；等待质量时间序列重编号为图 7，并仅保留 no-limit、medium、low 三条与强度扫描直接对应的曲线。
- 在 5.5.1 中补充“高保真可行性”和“可行率”的统计口径：密度上限削减质量占参考质量比例不超过 $2\%$ 视为可行，可行率为最终高保真最优候选满足该条件的随机种子占比。

## 验证

- 执行 `python -m py_compile codes\g2_capacity_response_runner.py` 通过。
- 执行 `python codes\g2_capacity_response_runner.py --output-root codes\results\g2_capacity_response --redraw-only` 通过，仅重画图表，未重跑实验。
