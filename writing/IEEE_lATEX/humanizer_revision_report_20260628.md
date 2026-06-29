# Humanizer 修改意见报告

目标文件：`writing/IEEE_lATEX/New_IEEEtran_how-to.tex`

生成日期：2026-06-28

原文状态：本报告只提出修改意见，没有直接修改 `.tex` 原文。审阅前记录的 SHA256 为 `E2AA1D94D1DDFC49B8F4BDF26F3510B7DE69FA5E647037927436F692A7D706A2`。

## 1. 总体判断

这篇稿件整体是学术论文写法，主要问题不是聊天式 AI 痕迹，而是若干段落存在学术 AI 文本常见的模板化痕迹：

1. 重要性和效果表述偏满：`critical`, `key`, `significant`, `effective`, `efficient`, `provides a quantitative and interpretable...` 等词在摘要、引言、相关工作和实验总结中较集中。
2. 结果段落中 `show that`, `achieves`, `outperforms`, `best`, `lowest` 连续出现，容易显得像模型自动生成的论文结果摘要。
3. 引言中多处 `First, Second, Third, Fourth, Finally` 和贡献列表叠加，结构清楚，但略有公式化。
4. 相关工作有几句评价性词汇偏泛，例如 `elegantly captures the interplay`, `promising alternatives`, `proven effective`，建议改成更具体的技术描述。
5. 结论的未来工作句较长，列项过多，读起来像通用模板，可以拆开并聚焦最能回应审稿人的限制项。

不建议机械处理的内容：

1. IEEE 论文标题式章节名属于期刊/模板风格，不应仅因 Title Case 而改。
2. 公式、算法、表格、引用命令和 `Bellman--conservation-law` 这类 LaTeX 连字符不属于 AI 写作痕迹。
3. 全文未发现聊天式废话、知识截止声明、emoji、明显宣传口吻或 Unicode em/en dash。第 978-980 行的 `--` 是表格缺失值，不需要改。

## 2. 高优先级修改建议

### 2.1 摘要第 33-43 行：压缩泛化表达，弱化宣传式结尾

原文问题：

1. `systematic stampede risks`, `fine-grained large-scale crowd management`, `complex interactions`, `quantitative and interpretable decision-support tool` 偏泛。
2. `achieves`, `outperforms`, `provides` 连续出现，结尾像方法宣传。
3. 摘要有多句 `To...` 和 `The proposed...` 结构，节奏较整齐。

Draft rewrite:

```tex
Open scenic areas and large event venues can develop dense pedestrian flows that are difficult to evaluate before an event. Existing practice still relies on empirical rules and post-event review, which limits active prevention and quantitative comparison. In these settings, managers mainly choose passage directions, such as one-way, bidirectional, or closed, and adjust time-varying entrance release rates. Their coupled effect on network-level pedestrian flow is still not well represented in optimization models. This paper develops a macroscopic modeling and control optimization method for large-scale crowd management. The model extends the Hughes continuum framework. Local admissible direction sets represent operational states, while anisotropic mobility tensors encode channel-guidance effects. Internal channel entrance-rate controls are added so that a policy can specify both directions and release rates. To solve the resulting mixed discrete-continuous problem, we design a Hierarchical Constrained Mixed-variable Black-box Optimization (HCMBO) algorithm. Experiments on a simplified scenic-platform scenario show that the model reproduces multi-stage behavior, channel guidance, and direction constraints. Under a weighted objective combining efficiency, safety exposure, load balance, waiting, and smoothness, HCMBO gives the lowest mean score within the tested budget. Its mean objective is about 3.9\% lower than TPE-Mixed BO and lower than the other tested baselines. The method provides a quantitative basis for comparing routing and entrance-release strategies in open public spaces.
```

What still sounds AI-generated:

1. `provides a quantitative basis` is still a little generic.
2. `show that` remains a common paper phrase, acceptable but not ideal if overused elsewhere.
3. The paragraph still has several balanced two-part sentences.

Final suggested rewrite:

```tex
Open scenic areas and large event venues can develop dense pedestrian flows that are hard to evaluate before an event. Existing practice still relies heavily on empirical rules and post-event review, which leaves limited support for comparing control plans before deployment. In these settings, managers mainly choose passage directions, such as one-way, bidirectional, or closed, and adjust time-varying entrance release rates. Existing models rarely optimize these two controls together. This paper develops a macroscopic crowd-control model based on the Hughes continuum framework. Local admissible direction sets represent one-way, bidirectional, and closed passages, while anisotropic mobility tensors encode channel-guidance effects. Internal channel entrance-rate controls are added so that one policy specifies both passage directions and release rates. We then solve the mixed discrete-continuous problem with a Hierarchical Constrained Mixed-variable Black-box Optimization (HCMBO) algorithm. In a simplified scenic-platform scenario, the model reproduces multi-stage movement, channel guidance, and direction constraints. Under a weighted objective combining efficiency, safety exposure, load balance, waiting, and smoothness, HCMBO gives the lowest mean score within the tested budget. Its mean objective is about 3.9\% lower than TPE-Mixed BO and lower than the other tested baselines. These results give a simulation-based basis for comparing routing and entrance-release strategies in open public spaces.
```

### 2.2 引言第 53-57 行：去掉 not only/but also 和泛化背景句

原文问题：

1. `not only ... but also` 是典型高频 AI 结构。
2. `critical component`, `Against this backdrop`, `critical research problems` 强调过满。
3. 第 56 行把 `scientific modeling, numerical simulation, and optimization` 打成三件套，略显模板化。

Draft rewrite:

```tex
\IEEEPARstart{L}{arge-scale} crowd gatherings can threaten public safety when local congestion grows into a stampede risk. High-density activities such as holiday tourism, public events, and sports gatherings now place short, intense pedestrian loads on core urban areas \cite{RN2147,RN1998,RN2154,RN1193,RN673}. Crowd-control research therefore needs models that can test routing and release strategies before an event, rather than relying only on field experience after incidents.
```

What still sounds AI-generated:

1. `therefore needs` still announces the research gap in a conventional way.
2. `short, intense pedestrian loads` may need checking against the paper's preferred terminology.

Final suggested rewrite:

```tex
\IEEEPARstart{L}{arge-scale} crowd gatherings can threaten public safety when local congestion grows into a stampede risk. Holiday tourism, public events, and sports gatherings place short, intense pedestrian loads on core urban areas \cite{RN2147,RN1998,RN2154,RN1193,RN673}. This motivates crowd-control models that can compare routing and entrance-release strategies before an event, rather than relying only on field experience and post-event review.
```

### 2.3 引言第 59-65 行：减少抽象系统论措辞

原文问题：

1. `typical complex systems`, `multi-scale, nonlinear mechanisms`, `spatiotemporal patterns` 都正确，但连在一起会显得泛。
2. `additionally exhibits pronounced phase dependence` 可改得更直接。
3. 末句 `cannot adequately support refined management decisions` 属于常见泛化结论。

Draft rewrite:

```tex
Crowd motion at large events is shaped by individual speeds, destinations, route preferences, visible paths, obstacles, and density feedback \cite{RN1193,RN1100,RN2174}. At the aggregate level, these factors appear as destination-oriented flow, passage convergence, bottleneck queuing, and responses to direction rules \cite{RN2056,RN2102}. In open scenic areas, visitors also move through stages: entering the site, touring along a main corridor, choosing lateral exit passages, and returning through lower streets \cite{RN2302}. Static capacity analysis and single shortest-path models miss these stage changes and shared congestion effects.
```

What still sounds AI-generated:

1. The final contrast is tidy but acceptable for a research gap.
2. `shaped by` is still broad, but less inflated than the original.

Final suggested rewrite:

```tex
Crowd motion at large events depends on individual speeds, destinations, route preferences, visible paths, obstacles, and density feedback \cite{RN1193,RN1100,RN2174}. At the aggregate level, these factors appear as destination-oriented flow, passage convergence, bottleneck queuing, and responses to direction rules \cite{RN2056,RN2102}. In open scenic areas, visitors often move through stages: entering the site, touring along a main corridor, choosing lateral exit passages, and returning through lower streets \cite{RN2302}. Static capacity analysis and single shortest-path models do not capture these stage changes or the congestion shared across routes.
```

### 2.4 引言第 72-75 行：避免三点式模板感

原文问题：

1. `management difficulties appear in three aspects` 是典型提纲式表达。
2. 三句都以序数开头，和后面的 `First, Second, Third, Fourth, Finally` 形成重复。

Draft rewrite:

```tex
This scenario creates three operational issues. Direction settings on one passage can reorganize flow elsewhere. Behavioral transitions between the main tourist corridor and lateral passages make single origin--destination models insufficient. Local congestion and global route choice also feed back into each other, so a control change in one passage may redistribute loads in nearby areas.
```

What still sounds AI-generated:

1. `creates three operational issues` still sounds like a structured outline.

Final suggested rewrite:

```tex
Three operational issues follow from this layout. Direction settings on one passage can reorganize flow elsewhere. Behavioral transitions between the main tourist corridor and lateral passages make single origin--destination models insufficient. Local congestion and global route choice also feed back into each other, so a control change in one passage may redistribute loads in nearby areas.
```

### 2.5 引言第 93-102 行：把问题陈述从清单改成因果链

原文问题：

1. `First, Second, Third, Fourth, Finally` 的五段式问题陈述过于规整。
2. `remain heavily reliant`, `fall short`, `scientific evaluation and fine-grained optimization` 都是常见 AI 学术套话。
3. 第 97 行较长，信息多但主干被评价词遮住。

Draft rewrite:

```tex
Current practice in high-pedestrian-flow urban areas still depends on manual experience and static contingency plans \cite{RN1662}. Many schemes do not quantify how passage direction, entrance capacity, and phase-dependent route preference change density evolution. Single-point statistics and local density monitoring also make it hard to follow the coupling among direction rules, entrance metering, local optimal directions, density evolution, and exit-load distribution \cite{RN2124}. In open scenic areas, ignoring route preference and phase transition can misjudge passage load and high-risk locations. Simulation-optimization studies often treat the crowd model as a black-box evaluator, so the search does not use the structure in passage geometry, one-way rules, internal entrance capacities, or subpopulation routes \cite{RN2146,RN2122}. Existing work also rarely separates mechanism diagnostics from strategy-level management objectives, despite the importance of component-level testing in evacuation-model verification and validation \cite{ronchi2016vv_evacuation_models}.
```

What still sounds AI-generated:

1. The list is still long.
2. Several `also` transitions remain.

Final suggested rewrite:

```tex
Current practice in high-pedestrian-flow urban areas still depends on manual experience and static contingency plans \cite{RN1662}. Many schemes do not quantify how passage direction, entrance capacity, and phase-dependent route preference change density evolution. Single-point statistics and local density monitoring make it hard to follow the coupling among direction rules, entrance metering, local optimal directions, density evolution, and exit-load distribution \cite{RN2124}. In open scenic areas, ignoring route preference and phase transition can misjudge passage load and high-risk locations. Simulation-optimization studies often treat the crowd model as a black-box evaluator, so the search does not use the structure in passage geometry, one-way rules, internal entrance capacities, or subpopulation routes \cite{RN2146,RN2122}. Existing work also rarely separates mechanism diagnostics from strategy-level management objectives, although evacuation-model verification studies emphasize component-level testing and emergent-behavior validation \cite{ronchi2016vv_evacuation_models}.
```

### 2.6 相关工作第 121-136 行：减少泛评价词

原文问题：

1. `extensively studied`, `enhance safety and efficiency`, `proven effective`, `promising alternatives`, `significantly improve` 泛化评价偏多。
2. 相关工作应尽量用具体对象和方法说话，而不是评价技术趋势。

Draft rewrite:

```tex
Crowd management studies in urban centers have mainly used boundary regulation, facility layout, and decision-making models \cite{RN2000,RN2193}. Zhong et al. \cite{RN279} designed boundary control strategies for urban traffic-flow networks and proved convergence to uncongested states under disturbances through Lyapunov analysis \cite{RN1196}. Wadoo and Kachroo \cite{RN280} used advection and diffusion control to prevent blocking and shock waves in one-dimensional evacuation. Qin \cite{RN277} proposed a unit sliding-mode controller based on integral barrier Lyapunov functions for multi-directional disturbance propagation. Zhu et al. \cite{RN286} combined policy learning and neural networks to adjust inflow rates and free-flow speeds in heterogeneous corridors.

Infrastructure-based regulation has also been studied. Obstacle placement can change outflow, velocity, and local density in merging areas \cite{RN1662,RN2145}. Yang et al. \cite{RN2118,RN2120} used Model Predictive Control (MPC) to optimize barrier lengths at subway bottlenecks and later included entrance limiters, gates, and escalator directions. Carmona and Paricio-Garcia \cite{RN2119} proposed dynamic exit-choice recommendations using multinomial Logit models, but the approach depends on discrete speed instructions transmitted through electronic wristbands \cite{RN2196}, which are hard to deploy in open urban areas.

Recent work has also introduced optimization and reinforcement-learning methods. Liao et al. \cite{RN2121} formulated fence layout as subset selection. Subsequent work \cite{RN2122} used LSTM features and Proximal Policy Optimization (PPO) \cite{RN2189} for real-time entrance-flow regulation. Differential Evolution (DE) \cite{RN2190} has been used to optimize guardrail layout and flow guidance in scenarios such as Chengdu East Railway Station, with average travel time and crowd pressure as objectives \cite{RN2123}.
```

What still sounds AI-generated:

1. The sequence remains survey-like, which is normal for related work.
2. `Recent work has also introduced` is plain but still formulaic.

Final suggested rewrite:

```tex
Crowd management studies in urban centers have mainly used boundary regulation, facility layout, and decision-making models \cite{RN2000,RN2193}. Zhong et al. \cite{RN279} designed boundary control strategies for urban traffic-flow networks and proved convergence to uncongested states under disturbances through Lyapunov analysis \cite{RN1196}. Wadoo and Kachroo \cite{RN280} used advection and diffusion control to prevent blocking and shock waves in one-dimensional evacuation. Qin \cite{RN277} proposed a unit sliding-mode controller based on integral barrier Lyapunov functions for multi-directional disturbance propagation. Zhu et al. \cite{RN286} combined policy learning and neural networks to adjust inflow rates and free-flow speeds in heterogeneous corridors.

Infrastructure-based regulation has also been studied. Obstacle placement can change outflow, velocity, and local density in merging areas \cite{RN1662,RN2145}. Yang et al. \cite{RN2118,RN2120} used Model Predictive Control (MPC) to optimize barrier lengths at subway bottlenecks and later included entrance limiters, gates, and escalator directions. Carmona and Paricio-Garcia \cite{RN2119} proposed dynamic exit-choice recommendations using multinomial Logit models, but the approach depends on discrete speed instructions transmitted through electronic wristbands \cite{RN2196}, which are hard to deploy in open urban areas.

Other studies use optimization and reinforcement learning. Liao et al. \cite{RN2121} formulated fence layout as subset selection. Subsequent work \cite{RN2122} used LSTM features and Proximal Policy Optimization (PPO) \cite{RN2189} for real-time entrance-flow regulation. Differential Evolution (DE) \cite{RN2190} has been used to optimize guardrail layout and flow guidance in scenarios such as Chengdu East Railway Station, with average travel time and crowd pressure as objectives \cite{RN2123}.
```

### 2.7 相关工作第 143-151 行：替换 "elegantly captures the interplay"

原文问题：

1. `elegantly captures the interplay` 是典型 AI 学术夸饰。
2. `enriched the descriptive capability` 和 `well-suited` 也偏泛。

Draft rewrite:

```tex
Hughes \cite{hughes2003} pioneered a first-order continuum theory for pedestrian flow, in which walking speed depends on local density and movement direction follows the steepest-descent path of a potential function. The model links congestion and route choice through this potential. Ling et al. \cite{RN2078} generalized the Hughes formulation and solved it with a WENO scheme for the conservation law and a fast-sweeping method for the eikonal equation.

Several extensions modify the density, perception, and constraint terms in macroscopic models. Colombo et al. \cite{RN2097} proposed nonlocal flux models that replace pointwise density with a neighborhood-averaged quantity in the velocity function. Later work included anisotropic perception fields and wall or obstacle boundary effects, together with high-resolution schemes and well-posedness results for architectural domains \cite{RN2080}. Maury et al. \cite{RN2099} introduced a projection model for hard incompressibility constraints, formulating the actual velocity as a least-squares projection onto feasible non-overlapping configurations. The same idea was later recast as a Wasserstein gradient flow with density constraints \cite{RN2081}, using the Jordan--Kinderlehrer--Otto (JKO) scheme \cite{RN2188}.
```

What still sounds AI-generated:

1. `Several extensions modify...` is a survey opener, but it is concrete enough.
2. The paragraph is dense; that is acceptable for related work.

Final suggested rewrite:

```tex
Hughes \cite{hughes2003} pioneered a first-order continuum theory for pedestrian flow, in which walking speed depends on local density and movement direction follows the steepest-descent path of a potential function. The model links congestion and route choice through this potential. Ling et al. \cite{RN2078} generalized the Hughes formulation and solved it with a WENO scheme for the conservation law and a fast-sweeping method for the eikonal equation.

Several extensions modify the density, perception, and constraint terms in macroscopic models. Colombo et al. \cite{RN2097} proposed nonlocal flux models that replace pointwise density with a neighborhood-averaged quantity in the velocity function. Later work included anisotropic perception fields and wall or obstacle boundary effects, together with high-resolution schemes and well-posedness results for architectural domains \cite{RN2080}. Maury et al. \cite{RN2099} introduced a projection model for hard incompressibility constraints, formulating the actual velocity as a least-squares projection onto feasible non-overlapping configurations. The same idea was later recast as a Wasserstein gradient flow with density constraints \cite{RN2081}, using the Jordan--Kinderlehrer--Otto (JKO) scheme \cite{RN2188}.
```

### 2.8 方法第 189-192 行：把 "key modeling assumption" 改为更直接的建模决定

原文问题：

1. `The key modeling assumption is...` 是常见 AI 论文句式。
2. `should be optimized jointly` 表述像规范性判断，可改成论文实际建模选择。

Draft rewrite:

```tex
We model direction rules and entrance capacities as coupled controls. The direction configuration $s$ changes the local feasible direction set, whereas the entrance capacity $q$ changes the realized admitted flux at channel entrances. These two controls jointly affect channel loading, high-density exposure, and upstream waiting. We therefore formulate crowd management as a mixed-variable simulation-based optimization problem, rather than solving direction selection and capacity assignment as two independent subproblems \cite{RN2122,jiang2018coordinated_inflow,liu2021queuing_network_flow_control}.
```

What still sounds AI-generated:

1. The final `therefore` is ordinary academic signposting, not a serious issue.

Final suggested rewrite:

```tex
We model direction rules and entrance capacities as coupled controls. The direction configuration $s$ changes the local feasible direction set, whereas the entrance capacity $q$ changes the realized admitted flux at channel entrances. Together, they affect channel loading, high-density exposure, and upstream waiting. We therefore formulate crowd management as a mixed-variable simulation-based optimization problem, rather than solving direction selection and capacity assignment as two independent subproblems \cite{RN2122,jiang2018coordinated_inflow,liu2021queuing_network_flow_control}.
```

### 2.9 优化方法第 598-600 行：去掉 "clear management meaning"

原文问题：

1. `clear management meaning` 偏泛，像 AI 结尾。
2. 第 599 行重复 `variable determines... variable determines...`，节奏机械。

Draft rewrite:

```tex
Compared with generic mixed-variable optimization, HCMBO uses the dependence between direction choices and feasible capacity profiles. Once $s$ is fixed, the mapping $T_s$ removes capacity variables that conflict with the selected direction rules and bounds the remaining entrance-rate profiles. This reduces invalid candidates and allocates the simulation budget to policies that satisfy the operational constraints.
```

What still sounds AI-generated:

1. `allocates the simulation budget` is concise and acceptable.

Final suggested rewrite:

```tex
Compared with generic mixed-variable optimization, HCMBO uses the dependence between direction choices and feasible capacity profiles. Once $s$ is fixed, the mapping $T_s$ removes capacity variables that conflict with the selected direction rules and bounds the remaining entrance-rate profiles. This reduces invalid candidates and allocates the simulation budget to policies that satisfy the operational constraints.
```

### 2.10 实验第 813-819 行：结果表述降调，避免宣传式 outperforms

原文问题：

1. `achieves the lowest... lowest... lowest... highest...` 连续四个最高级，读起来像自动生成的结果宣传。
2. `outperforms` 重复出现。
3. 最后一句 `residual need for stronger... awareness` 偏泛。

Draft rewrite:

```tex
Table~\ref{tab:optimizer_summary} reports the smallest mean, median, and best objective for HCMBO, together with the highest feasible rate among the tested methods. Its mean objective is $3.9\%$ lower than TPE-Mixed BO, since $(2.7403-2.6338)/2.7403=3.9\%$. Because this is a normalized scalar objective, the percentage reduction reflects the fixed-weight management score, not a direct reduction in travel time or density exposure alone. In paired comparisons, HCMBO has a lower objective than TPE-Mixed BO in four of five seeds, lower than Random Search in all five seeds, and lower than Pure SA and Enum-DE in four of five seeds. The remaining high HCMBO objective is mainly caused by a larger safety-exposure term, which suggests that the current search still needs stronger safety constraints.
```

What still sounds AI-generated:

1. `suggests that` is standard but could be more direct.
2. `stronger safety constraints` is clearer than `awareness`, but the exact mechanism should match planned future work.

Final suggested rewrite:

```tex
Table~\ref{tab:optimizer_summary} reports the smallest mean, median, and best objective for HCMBO, together with the highest feasible rate among the tested methods. Its mean objective is $3.9\%$ lower than TPE-Mixed BO, since $(2.7403-2.6338)/2.7403=3.9\%$. Because this is a normalized scalar objective, the percentage reduction reflects the fixed-weight management score, not a direct reduction in travel time or density exposure alone. In paired comparisons, HCMBO has a lower objective than TPE-Mixed BO in four of five seeds, lower than Random Search in all five seeds, and lower than Pure SA and Enum-DE in four of five seeds. The remaining high HCMBO objective is mainly caused by a larger safety-exposure term, so later versions should add more explicit safety constraints during search.
```

### 2.11 实验第 844-846 行：减少视觉证据的泛化表达

原文问题：

1. `provide complementary visual evidence` 是常见套话。
2. `lowest final region` 不够具体。
3. `not only...; they also...` 是 AI 高频结构。

Draft rewrite:

```tex
Figs.~\ref{fig:optimizer_best_objective}--\ref{fig:control_profiles} compare the seed-wise objectives, convergence traces, and selected entrance-rate profiles. TPE-Mixed BO decreases quickly in the early evaluations, whereas HCMBO continues to reduce the best-so-far objective during the middle and late stages. The control profiles indicate that methods with the same direction code can still differ in segment-wise rate allocation, which changes the final objective.
```

What still sounds AI-generated:

1. `which changes the final objective` is a little flat, but clear.

Final suggested rewrite:

```tex
Figs.~\ref{fig:optimizer_best_objective}--\ref{fig:control_profiles} compare the seed-wise objectives, convergence traces, and selected entrance-rate profiles. TPE-Mixed BO decreases quickly in the early evaluations, whereas HCMBO continues to reduce the best-so-far objective during the middle and late stages. The control profiles indicate that methods with the same direction code can still choose different segment-wise rate allocations, leading to different final objectives.
```

### 2.12 消融第 851-856 行：减少 "First/Second/Third" 机械节奏

原文问题：

1. 与引言中的序数结构重复。
2. `clearly worse`, `showing that` 可改得更证据化。

Draft rewrite:

```tex
The ablation experiments test the contribution of each control and search component. Optimizing only directions with a fixed high rate gives a best high-fidelity objective of $3.3751$, and optimizing only rates under prior directions gives $3.3140$; both are higher than the joint optimization result. Short-horizon low-fidelity hard screening raises the best objective to $2.9441$ and selects an infeasible best candidate, because early simulations do not reliably capture long-horizon entrance waiting and load redistribution. Removing the entrance-waiting penalty slightly lowers the scalar objective to $2.6015$, but increases unreleased attempted flow to approximately $1.30\times10^4$~ped-eq. For this reason, the waiting term should remain in the management objective.
```

What still sounds AI-generated:

1. `For this reason` is ordinary but a bit formal.

Final suggested rewrite:

```tex
The ablation experiments test the contribution of each control and search component. Optimizing only directions with a fixed high rate gives a best high-fidelity objective of $3.3751$, and optimizing only rates under prior directions gives $3.3140$; both are higher than the joint optimization result. Short-horizon low-fidelity hard screening raises the best objective to $2.9441$ and selects an infeasible best candidate, because early simulations do not reliably capture long-horizon entrance waiting and load redistribution. Removing the entrance-waiting penalty slightly lowers the scalar objective to $2.6015$, but increases unreleased attempted flow to approximately $1.30\times10^4$~ped-eq. The waiting term should therefore remain in the management objective.
```

### 2.13 Bund 迁移第 1025-1033 行：保留审慎结论，删掉 Overall 套话

原文问题：

1. `also reveals an important trade-off` 和 `Overall... shows that...` 都是常见 AI 总结句。
2. 此段已经有清楚的限制性解释，应避免再包装成通用结论。

Draft rewrite:

```tex
The transfer experiment exposes a trade-off. Compared with the uncontrolled group, the controlled group has a higher efficiency term ($\tilde J_1$ increases from $0.3813$ to $0.4145$) and a slightly higher safety-exposure term ($\tilde J_2$ increases from $0.0284$ to $0.0357$). Its cumulative sink flow is also smaller ($1.33{\times}10^3$ versus $1.67{\times}10^3$~ped), and its final system mass is larger ($1.53{\times}10^3$ versus $1.20{\times}10^3$~ped). The transferred HCMBO policy therefore should not be read as a uniform safety or efficiency improvement. Under the present reduced-objective weights, it lowers $J^{(3)}$ mainly by redistributing channel load, while giving up part of exit throughput and travel efficiency. The capacity-scale factor of $3$ reduces unreleased attempted flow to $364$~ped-eq., but it does not remove the efficiency cost. For deployment-oriented use, the optimization should include explicit throughput constraints, waiting upper bounds, or multi-weight scenario analysis.
```

What still sounds AI-generated:

1. `For deployment-oriented use` is direct enough, but the final list still has three items.

Final suggested rewrite:

```tex
The transfer experiment exposes a trade-off. Compared with the uncontrolled group, the controlled group has a higher efficiency term ($\tilde J_1$ increases from $0.3813$ to $0.4145$) and a slightly higher safety-exposure term ($\tilde J_2$ increases from $0.0284$ to $0.0357$). Its cumulative sink flow is also smaller ($1.33{\times}10^3$ versus $1.67{\times}10^3$~ped), and its final system mass is larger ($1.53{\times}10^3$ versus $1.20{\times}10^3$~ped). The transferred HCMBO policy therefore should not be read as a uniform safety or efficiency improvement. Under the present reduced-objective weights, it lowers $J^{(3)}$ mainly by redistributing channel load, while giving up part of exit throughput and travel efficiency. The capacity-scale factor of $3$ reduces unreleased attempted flow to $364$~ped-eq., but it does not remove the efficiency cost. A deployment-oriented version should add explicit throughput constraints and waiting upper bounds, and should test several objective-weight settings before selecting a policy.
```

### 2.14 结论第 1040-1057 行：降低模板感，突出限制条件

原文问题：

1. `further shows`, `central to this performance`, `Future work will extend...` 是常见结论模板。
2. 未来工作一句列项过多。
3. 结论已有较好的审慎口径，应进一步具体化。

Draft rewrite:

```tex
This paper proposed a macroscopic crowd-management framework for open pedestrian zones with passage-direction rules, geometric guidance, multi-stage route behavior, and internal entrance-rate control. The framework couples a crowd-flow simulator with HCMBO so that direction settings and entrance-rate profiles can be optimized under a limited simulation budget.

The experiments indicate that the two local modeling components behave as intended and that direction control and entrance-rate control are not interchangeable. Direction settings create non-monotonic trade-offs among efficiency, safety exposure, and load balance. Entrance-rate limits change unreleased attempted flow, binding-time ratios, and local risk patterns. Across five random seeds with unified high-fidelity rechecking, HCMBO gives the lowest mean full management objective among the tested baselines, with a 3.9 percent lower fixed-weight score than TPE-Mixed BO and a 100 percent feasible rate. The ablation study attributes this result to joint direction-rate optimization, avoiding hard short-horizon screening, and structured direction-wise rate search.

The Bund-inspired transfer experiment indicates that an HCMBO-derived policy can alter flow organization in a simplified real-world spatial abstraction. The reduced objective decreases by 26.9 percent because the policy redistributes flow away from the dominant terminal channel and improves load balance across the middle-channel group. The same policy also produces higher residual mass, lower exit throughput, and unreleased attempted entrance flow. The transfer result should therefore be read as evidence of mechanism transfer and load redistribution, not as a deployable plan for the Bund area.

The study has several limitations. The experiments use simplified geometries, fixed objective weights, and finite random seeds. One HCMBO seed still has relatively high safety exposure. The model also does not yet include real crowd calibration, online demand updates, or executable field constraints. Future work should address these limits before operational deployment, especially by adding stronger safety and throughput constraints and by testing multiple demand and weight settings.
```

What still sounds AI-generated:

1. `The study has several limitations` is conventional but clear.
2. The second paragraph still summarizes many findings in a compact way, which is expected in a conclusion.

Final suggested rewrite:

```tex
This paper proposed a macroscopic crowd-management framework for open pedestrian zones with passage-direction rules, geometric guidance, multi-stage route behavior, and internal entrance-rate control. The framework couples a crowd-flow simulator with HCMBO so that direction settings and entrance-rate profiles can be optimized under a limited simulation budget.

The experiments indicate that the two local modeling components behave as intended and that direction control and entrance-rate control are not interchangeable. Direction settings create non-monotonic trade-offs among efficiency, safety exposure, and load balance. Entrance-rate limits change unreleased attempted flow, binding-time ratios, and local risk patterns. Across five random seeds with unified high-fidelity rechecking, HCMBO gives the lowest mean full management objective among the tested baselines, with a 3.9 percent lower fixed-weight score than TPE-Mixed BO and a 100 percent feasible rate. The ablation study attributes this result to joint direction-rate optimization, avoiding hard short-horizon screening, and structured direction-wise rate search.

The Bund-inspired transfer experiment indicates that an HCMBO-derived policy can alter flow organization in a simplified real-world spatial abstraction. The reduced objective decreases by 26.9 percent because the policy redistributes flow away from the dominant terminal channel and improves load balance across the middle-channel group. The same policy also produces higher residual mass, lower exit throughput, and unreleased attempted entrance flow. The transfer result should therefore be read as evidence of mechanism transfer and load redistribution, not as a deployable plan for the Bund area.

The study has several limitations. The experiments use simplified geometries, fixed objective weights, and finite random seeds. One HCMBO seed still has relatively high safety exposure. The model also does not yet include real crowd calibration, online demand updates, or executable field constraints. Future work should address these limits before operational deployment, especially by adding stronger safety and throughput constraints and by testing multiple demand and weight settings.
```

## 3. 中低优先级局部用词建议

这些位置不一定必须改，但如果后续统一润色，可以顺手处理。

| 原位置 | 当前表达 | 建议 |
| --- | --- | --- |
| 第 128 行 | `has also proven effective` | 改为 `has also been studied` 或直接说明具体方法和结果 |
| 第 134 行 | `have emerged as promising alternatives` | 改为 `have also been used` |
| 第 145 行 | `efficient numerical solution` | 若无复杂度或耗时证据，可改为 `numerical solution` |
| 第 291 行 | `effective cost` | 这里是技术含义，建议保留 |
| 第 545 行 | `unified high-fidelity setting` | 可保留，本文已反复定义该协议 |
| 第 600 行 | `clear management meaning` | 建议按 2.9 替换为 `satisfy the operational constraints` |
| 第 683 行 | `further shows` | 可改为 `also reports` 或 `indicates` |
| 第 685 行 | `support using` | 可改为 `support the use of` 或更直接写 `justify using` |
| 第 706 行 | `motivates automatic...` | 可改为 `makes manual rule selection insufficient for this setting` |
| 第 782 行 | `genuine continuous control variable` | 可保留，也可改为 `continuous control variable with measurable system effects` |
| 第 884 行 | `obtains the lowest scalar objective` | 建议写成 `has the lowest scalar objective in Table...` |
| 第 947 行 | `main improvement` | 建议写成 `largest reduction`，更贴合表格 |

## 4. 建议实施顺序

1. 先改摘要和结论，因为这两处最容易被审稿人或编辑快速判断为模板化。
2. 再改引言第 53-102 行，目标是减少套话和序数清单，不改变研究问题。
3. 然后处理相关工作中的评价性词汇，重点是第 121-136 行和第 143-151 行。
4. 最后统一实验结果段落，减少 `achieves`, `outperforms`, `shows that`, `best/lowest/highest` 的堆叠。

## 5. 风险提醒

1. 不要为了“去 AI 味”牺牲论文的可审计性。数值、公式、引用、指标定义应优先保留。
2. 摘要和结论中的建议稿改动较大，实施前应检查字数、IEEE 摘要长度限制，以及是否与最终表格数值一致。
3. 如果后续直接把建议稿写入 `.tex`，需要重新编译并检查 PDF 中摘要、表格脚注和结论分页。
