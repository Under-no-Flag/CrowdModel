# 2026-06-08 G6 最优目标图样式修订

## 目标任务描述

根据论文图件反馈，修订横向优化对比中 `g6_paper_best_objective.png` 的种子分布表达方式，交换 Pure SA 与 HCMBO 的颜色，优化目标--可行率权衡图的标注可读性，并补强 5.5.4 的逐图分析。

## 已完成的具体任务和产物

- 在 `codes/crowd_bellman/g6_visualization.py` 中将 Pure SA 与 HCMBO 的方法配色互换：Pure SA 改为蓝色，HCMBO 改为绿色。
- 将 `g6_paper_best_objective.png` 中每个 seed 的黑色散点改为最小--最大范围线与中位数短线，减少点状标记带来的视觉噪声。
- 将 `g6_paper_objective_feasibility.png` 改为断轴式双面板：左侧放大优化方法区域，右侧保留先验基线位置；为 Random search、Enum-DE 和 Baseline prior 设置定制标注偏移和引线，避免文字重叠。
- 在 `g6_paper_convergence.png` 中增加末端放大窗，突出 $240$--$400$ 次评估区间内各方法 best-so-far 曲线的最终差异，解决主图末端曲线重叠不易辨识的问题。
- 将 `g6_paper_control_profiles.png` 图内标题和色标从 capacity 表述改为 entrance-rate / rate upper bound，以匹配正文中的“入口通行速率”术语。
- 将 `writing/期刊论文章节/05-实验.md` 的 5.5.4 从总括式描述改为图 11、图 12、图 13、图 14 分别说明和分析，并将图 14 图注改为“各方法最优入口通行速率控制剖面”。
- 使用 `codes/g6_horizontal_comparison.py --visualize-only` 基于既有 G6 汇总结果重画论文图件，未重跑实验。

## 验证

- 执行 `python -m py_compile codes\crowd_bellman\g6_visualization.py` 通过。
- 执行 `$env:PYTHONPATH='codes'; python codes\g6_horizontal_comparison.py --visualize-only` 通过。
- 人工检查 `codes/results/g6_horizontal_comparison/paper_figures_no_tpe/g6_paper_best_objective.png`，确认黑色散点已移除，Pure SA 与 HCMBO 配色已互换。
- 人工检查 `codes/results/g6_horizontal_comparison/paper_figures_no_tpe/g6_paper_objective_feasibility.png`，确认优化方法标签和 Baseline prior 标签不再重叠。
- 人工检查 `codes/results/g6_horizontal_comparison/paper_figures_no_tpe/g6_paper_convergence.png`，确认末端重叠区域可通过放大窗辨识。
- 人工检查 `codes/results/g6_horizontal_comparison/paper_figures_no_tpe/g6_paper_control_profiles.png`，确认图内术语已改为入口通行速率。
