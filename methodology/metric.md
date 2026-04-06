

# 管控模型的优化目标

## 效率目标

### 总旅行时间
$$J_1 = \int_0^T \int_\Omega \rho(x,t)\,dxdt$$
这其实等价于总系统旅行时间，通常比“最后一个人离开时间”更稳定。


## 安全目标
### 高密度暴露时间


$$J_2 = \int_0^T \int_\Omega \mathbf{1}_{\rho(x,t)>\rho_{\text{safe}}}\, dxdt$$

### 关键瓶颈区域风险（暂不考虑）


$$J_4 = \int_0^T \int_{\Omega_{\text{bottleneck}}} g(\rho(x,t))\,dxdt$$

### 峰值密度（暂不考虑）


$$J_3 = \max_{x,t}\rho(x,t)$$

## 公平目标、均衡性目标

### 出入口负载均衡
$$J_5 = \operatorname{Var}\big(F_1,\dots,F_m\big)$$

其中 $F_i$ 是各通道累计流量。
- 各路线平均旅行时间差异最小
- 某个区域不被过度牺牲

因为管理方不仅关心“快”，还关心“别把风险全压给 8 号或 9 号通道”。