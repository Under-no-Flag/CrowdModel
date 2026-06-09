# 2026-06-07 论文评价指标编号与图表同步

## 目标任务描述

统一英文论文展示层的评价指标编号，解决方案设计阶段遗留的非连续符号问题：旧的 load-balance 指标 `J_5` 在论文中改为 `J_3`，旧的 entrance waiting 指标 `J_B` 改为 `J_4`，旧的 control smoothness 指标 `J_R` 改为 `J_5`。

## 已完成的具体任务和产物

- 在 `AGENT.md` 中新增论文评价指标编号规范，明确代码和历史实验字段可保留旧名，但正文、图、表、caption 和公式应使用连续的 `J_1` 到 `J_5`。
- 更新英文论文 `writing/IEEE_lATEX/New_IEEEtran_how-to.tex`，同步目标函数、标准化目标、权重、实验图说明和 G2 结果表述中的编号。
- 更新方法框架图 `writing/images/framework/method_framework_tikz.tex`，将 Performance Evaluation 中的 load balance、waiting、smoothness 和 weighted objective 改为 `J_3`、`J_4`、`J_5` 及 `\sum_{k=1}^{5}` 形式。
- 更新实验图生成脚本 `codes/crowd_bellman/g2_strategy.py`，将 G2 tradeoff 图中 load-balance 指标的显示标签从 `\tilde J_5` 改为 `\tilde J_3`。
- 重新生成并覆盖论文引用的 `writing/images/experiment_result_archive/g2_control_tradeoff_summary.png`。
- 重新编译 `writing/IEEE_lATEX/New_IEEEtran_how-to.pdf`，确认 LaTeX 构建完成并输出 12 页 PDF。
