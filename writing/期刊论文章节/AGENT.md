# 论文评价指标编号规范
代码和历史实验记录中可能保留早期变量名，例如 $J_5$、$J_B$、$J_R$。这些名称来自方案设计阶段的中间指标删减，不应直接出现在最终论文展示层。

论文正文、图、表、caption 和公式中统一使用连续编号：
- $J_1$：efficiency / total travel time；
- $J_2$：safety / high-density exposure；
- $J_3$：load balance / realized channel-throughput variance，代码或历史文件中的旧 $J_5$ 在论文中写作 $J_3$；
- $J_4$：entrance waiting / blocking，代码或历史文件中的旧 $J_B$ 在论文中写作 $J_4$；
- $J_5$：control smoothness，代码或历史文件中的旧 $J_R$ 在论文中写作 $J_5$。

对应权重在论文中写作 $\lambda_1,\ldots,\lambda_5$，标量目标优先写作 $J(z)=\sum_{k=1}^{5}\lambda_k\tilde J_k(z)$ 或等价展开式。除非明确讨论代码实现或历史字段，不要在论文展示层使用 $J_B$、$J_R$ 或把 load-balance 指标写成 $J_5$。