# 20260620 外滩热力图高清化记录

## 目标任务描述

将外滩场景 1600 step 仿真结果的密度热力图进一步高清化，并把后续复用的脚本使用方案写入 `AGENT.md`。

## 已完成任务和产物

- 基于 `codes/results/bund_hcmbo_transfer_wall_avoid_1600_full/bund_hcmbo_transfer_wall_avoid/fields` 中已保存的字段数据重新生成高清密度热力图，未重新运行仿真。
- 使用 wall-aware smoothing 的高清渲染参数：`--scale 12 --smooth-sigma 2.4 --color-scale frame-percentile --color-percentiles 0,98 --gamma 0.45 --density-alpha 0.92 --overlay-threshold 0.01 --dpi 320`。
- 输出目录为 `codes/results/bund_hcmbo_transfer_wall_avoid_1600_full/bund_hcmbo_transfer_wall_avoid/refined_density_wall_aware_hq`，共生成 21 张 `density_step_XXXX.png`，包括最终帧 `density_step_1599.png`。
- 在 `AGENT.md` 新增“外滩密度热力图高清化”说明，记录字段数据前置条件、推荐命令、参数含义和背景图定位方式。
