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
