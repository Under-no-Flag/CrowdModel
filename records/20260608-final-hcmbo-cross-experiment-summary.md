# 2026-06-08 最终 HCMBO 跨实验结果整合

## 目标任务描述

根据 `20260601-周报.md`、`20260526-g7b-hcmbo-improvement-experiment-report.md`、`20260525-周报.md` 和 `记录2026-05-30-G5-G6-G7-G7C优化实验过程与结果.md` 中的既有实验结果，整合改进后的最终 HCMBO 与 TPE-Mixed BO 及其他横向基线的对比，并更新论文 5.5--5.6 的表述和图件。用户明确要求主文不再保留旧 HCMBO 对比。

## 已完成的具体任务和产物

- 新增 `codes/final_hcmbo_cross_experiment_summary.py`，仅读取既有 CSV/JSON 实验结果，不重跑仿真实验。
- 生成 `codes/results/final_hcmbo_cross_experiment_summary/final_method_summary.csv`，汇总最终 HCMBO、TPE-Mixed BO、random search、pure SA、enum-DE 和先验基线的五种子高保真统计。
- 生成 `final_seed_summary.csv`，将 G7-D 最终 HCMBO 的种子结果与 G6 横向基线种子结果合并为统一画图口径。
- 生成 `final_convergence_curves.csv`，将 G7-D 最终 HCMBO 的既有中保真 evaluation log 与 G6 基线收敛曲线合并，未重跑仿真实验。
- 生成 `final_hcmbo_tpe_seed_comparison.csv`，保留最终 HCMBO 相对 TPE-Mixed BO 的逐种子配对目标差，供文字分析使用。
- 生成 `g7c_focus_comparison.csv`，保留 seed 23 下 HCMBO、TPE-Mixed BO 与 queue-aware HCMBO 的聚焦比较。
- 生成四张论文主图，均为原 5.5.4 图型语义，并补入 TPE-Mixed BO，同时用最终 HCMBO 替换旧 HCMBO：
  - `final_method_best_objective.png/.pdf`
  - `final_method_objective_feasibility.png/.pdf`
  - `final_method_convergence.png/.pdf`
  - `final_method_control_profiles.png/.pdf`
- 更新 `writing/期刊论文章节/05-实验.md`：
  - 5.5.3 删除旧 HCMBO 行及其配对分析。
  - 5.5.4 将图 11--图 14 调整为最好目标分布、目标--可行率、收敛曲线和多方法入口通行速率剖面，并为每张图补充独立分析与结论。
  - 5.6.1--5.6.4 删除旧 HCMBO/internal-random 相关消融项，仅保留最终主干 HCMBO 与 queue-aware、RF-style、adaptive racing、trust-region 等机制证据。

## 验证

- 执行 `python -m py_compile codes\final_hcmbo_cross_experiment_summary.py` 通过。
- 执行 `python codes\final_hcmbo_cross_experiment_summary.py` 通过，未重跑实验，仅重建跨实验汇总表和图件。
- 使用 `rg` 检查 `writing/期刊论文章节/05-实验.md` 与 `codes/results/final_hcmbo_cross_experiment_summary`，未发现 `HCMBO with internal random`、`internal random`、`旧 HCMBO`、`hcmbo_proposed`、`hcmbo_current`、`内部随机` 和 `diverse HF` 残留。
- 人工查看 `final_method_objective_feasibility.png`，确认方法标注未重叠。
- 人工查看 `final_method_convergence.png`，确认 TPE-Mixed BO 与最终 HCMBO 均已加入原 best-so-far 收敛曲线，末端放大窗不再被图例遮挡。
- 人工查看 `final_method_control_profiles.png`，确认 random search、pure SA、enum-DE、TPE-Mixed BO 和最终 HCMBO 的多方法 entrance-rate profile 均已展示。
- 根据图 13 反馈，将收敛曲线末端放大窗 y 轴范围从 `2.65--3.35` 调整为 `2.45--3.25`，并提高 HCMBO 曲线线宽和绘制层级，避免 HCMBO 在 inset 中被截断或遮挡。
- 根据图 13 二次反馈，将 best-so-far 收敛曲线从“均值线 + 四分位阴影”改为“中位数线 + 四分位阴影”，避免 HCMBO 均值线落在绿色 IQR 阴影之外；同时将 HCMBO 绿色阴影透明度加深。
- 根据图 14 反馈，为每个 entrance-rate profile 子图都显示 `Gate segment` 横轴标题，并增加子图行距以避免轴标题与下一行子图标题重叠。
