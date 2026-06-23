# Response Letter Draft and Revision Plan

稿件题目：Macroscopic Crowd Control with Direction Constraints and Entrance-Rate Control for Urban Pedestrian Zones

说明：本文档根据 `审稿意见by claude and gpt.md` 整理回复信草稿与正文修改说明。按照当前任务要求，本文件仅给出 response letter 和拟修改方案，不直接修改 `writing/IEEE_lATEX/New_IEEEtran_how-to.tex` 正文。

## 致审稿人

感谢审稿人对本文提出的系统性意见。我们认真检查了模型定义、数值离散、优化目标、实验解释和语言表述。审稿意见指出的核心问题是准确的：当前稿件应定位为面向开放步行区多通道管控的可解释宏观建模与仿真优化框架，而不应表述为已经可直接部署到真实外滩现场的成熟系统。根据该建议，修订将围绕四类问题展开：第一，澄清 Bellman--守恒律模型中的张量、可行方向集、入口限流和质量守恒定义；第二，统一评价指标、标准化目标和无量纲数值解释；第三，补足 HCMBO 的可复现算法描述和比较实验协议；第四，降低统计结论和外滩迁移实验结论的强度，明确收益与代价。

## 总体修改概览

| 审稿意见主题 | 拟修改位置 | 修改动作 |
|---|---|---|
| 稿件定位过强 | Abstract, Introduction, Discussion, Conclusion | 将定位调整为 simulation-based decision-support framework，删除或弱化直接部署、全面优越等表达 |
| $M(x;\eta_0)$ 含义不清 | Section III-C, equations for metric/Bellman update | 将 $M$ 明确定义为 mobility tensor 或 inverse metric，并同步修改公式说明 |
| $U(x;s)$ 定义不完整 | Section III-C | 补充非通道区域、关闭通道、障碍边界和离散邻域下的 admissible direction set |
| 耦合系统稳定性不足 | Section III-E | 增加 CFL 条件、Bellman 更新频率、有限体积格式和稳定性观察说明 |
| 入口限流与质量守恒 | Section III-D/III-E | 正式定义 admitted flux、unreleased attempted flow、density-cap removal 与质量闭合残差 |
| 指标与编号混乱 | Section III-F, Section IV tables/captions | 统一 $J_1$ 至 $J_5$，其中 $J_3$ 为 load balance，$J_4$ 为 waiting，$J_5$ 为 smoothness |
| 无量纲数字可读性 | Section III-F, Experimental setup, tables | 增加标准化公式、参考基准、单位说明、相对改进和有效数字规则 |
| HCMBO 可复现细节不足 | Section III-G | 增加 Algorithm 1、输入输出、候选生成、BO 超参数、高保真复验协议 |
| baseline 公平性与统计结论 | Section IV-D, Abstract, Conclusion | 澄清 G6/G7-D 数据来源、预算和种子协议；将结论改为 five-seed protocol 下的观察 |
| 外滩迁移结论过强 | Section IV-F, Discussion | 明确综合目标下降主要来自 $J_3$，同时 $J_1$、$J_2^{\mathrm{eval}}$、离场吞吐变差 |
| 安全作为硬约束 | Discussion/Future Work | 说明当前 $J_2$ 是软目标，后续将引入密度暴露或高密度持续时间硬约束 |

## 对 P0 意见的逐条回复

### Comment 1: Clarify the physical meaning of $M(x;\eta_0)$ and check mathematical consistency.

Response: We agree that the current wording may incorrectly suggest that $M(x;\eta_0)$ is a conventional cost metric. In the Bellman update used in this paper, a larger value of $u^\top M u$ reduces the effective step cost through the factor $1/\sqrt{u^\top M u}$. Therefore, $M$ should be interpreted as a mobility tensor, or equivalently an inverse metric, rather than a direct cost metric. This interpretation is consistent with the intended role of $\eta_0$: increasing tangential mobility along a channel axis and encouraging upstream alignment with the passage direction.

Proposed revision: In Section III-C, the subsection title will be changed from `Fixed Anisotropic Metric` to `Fixed Anisotropic Mobility Tensor`. The definitions around Eq. (9) and Eq. (10) will be revised so that $M_c(x;\eta_0)=\beta_c(\eta_0\tau_c\tau_c^\top+n_cn_c^\top)$ is explicitly described as a co-metric or mobility tensor. The text will state that the effective local cost is inversely proportional to $\sqrt{u^\top M u}$, so that larger tangential mobility lowers the Bellman step cost. All occurrences of `metric tensor` in this context will be checked and replaced by `mobility tensor` unless a conventional metric is explicitly meant.

### Comment 2: Fully define $U(x;s)$ and make the discrete Bellman form numerically consistent.

Response: We agree. The current definition focuses on channel regions and does not sufficiently specify non-channel walkable regions, obstacle boundaries, closed channels, and the discrete stencil used in computation. This can lead to ambiguity between the continuous notation and the numerical implementation.

Proposed revision: Section III-C will add a complete piecewise definition of $U(x;s)$. In channel cells, $U$ is determined by the channel state. In ordinary walkable cells outside controlled channels, $U$ equals the available grid-neighbor direction set after excluding obstacle-crossing and wall-crossing moves. In closed channels and obstacle cells, no through-channel direction is admissible, and obstacle cells are excluded from the conservation update. The Bellman update will be rewritten in discrete form as a minimization over valid neighboring cells or stencil directions, with an explicit rule for target cells, unreachable cells, and boundary cells. This makes the mathematical notation match the finite-grid solver.

### Comment 3: Discuss well-posedness and numerical stability of the Bellman--conservation coupling.

Response: We agree that the coupling between density-dependent speeds, Bellman potentials, and conservation-law updates should be treated more carefully. The revised manuscript will not claim a full analytical well-posedness theorem for the extended Hughes-type system. Instead, it will specify the numerical stability safeguards used in the simulations and state the scope of the claim.

Proposed revision: Section III-E will add a numerical-stability paragraph. It will specify that the conservation equation is advanced by an upwind finite-volume scheme under a CFL-type condition, e.g., $\Delta t \max_x |v(x,t)|/\Delta x \le C_{\mathrm{CFL}}$. The Bellman potentials are updated on the same density field at each time step or at a prescribed update interval, and the simulations monitor density range, mass balance residual, and absence of non-physical oscillatory artifacts. The text will explicitly state that the paper provides a simulation-based numerical framework rather than a proof of existence and uniqueness for the fully coupled continuum system.

### Comment 4: Clarify mass conservation under entrance metering and density-cap removal.

Response: We agree. Entrance metering is central to the model, and its mass accounting must be explicitly described. The current text already distinguishes attempted flux $A_c^\pm$ and admitted flux $\hat A_c^\pm$, but the full mass balance and density-cap treatment need clearer formalization.

Proposed revision: Section III-D and Section III-E will define attempted internal entrance flux, admitted flux, unreleased attempted flow, waiting or blocking mass, sink outflow, and density-cap removal in a single mass-accounting paragraph. The revised text will state that unreleased attempted flow is not deleted from the system but remains upstream and is accumulated in the waiting term. If density-cap removal is used as a numerical safeguard, it will be reported separately as a diagnostic loss rather than included silently in the controlled objective. A mass balance residual will be introduced, for example

$$
\epsilon_m(T)=\left|M(T)-M(0)-I(T)+O(T)+L_{\mathrm{cap}}(T)\right|,
$$

where $M(T)$ is system mass, $I(T)$ cumulative inflow, $O(T)$ cumulative sink outflow, and $L_{\mathrm{cap}}(T)$ density-cap removal. The experiments will report or at least discuss this residual.

### Comment 5: Unify objective functions, diagnostic metrics, and numbering.

Response: We agree. Historical code names such as old $J_5$, $J_B$, and $J_R$ should not appear in the final manuscript. The paper-facing notation will use continuous numbering $J_1,\ldots,J_5$: travel time, high-density exposure, realized channel-throughput imbalance, entrance waiting or blocking, and control smoothness.

Proposed revision: Section III-F will be the authoritative definition of all objectives. Experiment tables and figure captions will use the same notation. The scalar objective will be written as

$$
J(z)=\sum_{k=1}^{5}\lambda_k\tilde J_k(z),
$$

with $\lambda_1,\ldots,\lambda_5$ reported. When an experiment reports only a subset of terms for diagnostic purposes, the text will explicitly call it a diagnostic aggregate rather than the optimization objective. The notation $J_2^{\mathrm{eval}}$ will be defined as an evaluation-only safety exposure term if it differs from the standardized optimization term.

### Comment 6: Provide HCMBO pseudocode and reproducible hyperparameters.

Response: We agree that the current workflow table is insufficient as a reproducible algorithm description. HCMBO should be presented as a full algorithm with inputs, outputs, loops, evaluation budgets, candidate selection, surrogate search, constraints, and high-fidelity rechecking.

Proposed revision: Section III-G will include Algorithm 1. The algorithm will list the direction candidate set, capacity parameterization $q=T_s(x)$, initial sample count, BO iteration count, candidate pool size, acquisition or selection rule, feasibility handling, shortlist size, high-fidelity top-$k$, random seeds, simulation horizons, and termination condition. The experiment section will also state the exact high-fidelity protocol used for final ranking so that intermediate surrogate or low-fidelity results are not confused with final claims.

### Comment 7: Reduce statistical claims and clarify baseline fairness.

Response: We agree. With five random seeds, the results support a comparative observation under the tested protocol, but they should not be framed as strong statistical proof of superiority. We also agree that the use of G6 and G7-D data must be explained carefully so that the comparison protocol is transparent.

Proposed revision: In the Abstract, Section IV-D, and Conclusion, phrases such as `consistently outperforms` and `statistically superior` will be removed or softened. The revised claim will be: under the tested five-seed protocol and unified high-fidelity rechecking, HCMBO obtains the lowest mean objective and improves over TPE-Mixed BO in four out of five paired seeds. Section IV-D will explicitly state the budget, seeds, high-fidelity recheck count, and whether all baselines share the same evaluation conditions. If any historical comparison uses different generation pipelines, the manuscript will label it as a protocol-aligned comparison rather than a fully independent statistical benchmark.

### Comment 8: Improve readability of dimensionless numbers.

Response: We agree. Standardized objectives and accumulated simulation counts are necessary for optimization, but readers need definitions, baselines, units, and physical anchors to interpret them.

Proposed revision: The revised manuscript will add a nomenclature table covering $\rho$, $v$, $f$, $\phi$, $M$, $U$, $q$, $A$, $\hat A$, $\theta$, $B$, $R$, $J_k$, and $\tilde J_k$, with units or simulation-unit notes. Section III-F will define the standardization formula and reference baseline for each $\tilde J_k$. Experimental tables will prioritize relative improvement, share, and normalized ratios where possible. Raw accumulated quantities such as unreleased attempted flow will be described as simulation mass or mass-time units and, where feasible, also reported as a fraction of total attempted inflow. Excessive decimal precision will be reduced to a level consistent with simulation reproducibility.

### Comment 9: Tone down the Bund transfer experiment.

Response: We agree. The Bund-inspired transfer experiment should be treated as a supplementary validation on a simplified real-world spatial abstraction. It should not be interpreted as demonstrating simultaneous safety and efficiency improvement.

Proposed revision: Section IV-F will explicitly state that the improvement in scalar objective from $1.0140$ to $0.7410$ is mainly driven by the load-balance term $J_3$, which decreases from $0.6043$ to $0.2908$. It will also state the costs: $J_1$ increases from $0.3813$ to $0.4145$, $J_2^{\mathrm{eval}}$ increases from $0.0284$ to $0.0357$, final cumulative sink flow decreases from $8.5673$ to $6.8237$, and final system mass increases from $6.1822$ to $7.8455$. The conclusion will therefore be framed as improved scalar objective and channel redistribution under current weights, with a throughput and efficiency trade-off.

### Comment 10: Consider hard safety constraints.

Response: We agree that using safety exposure only as a soft weighted term is insufficient for deployment-oriented crowd management. An optimizer could reduce the scalar objective while allowing higher safety exposure if the weights permit it.

Proposed revision: The Discussion will add a limitation and future-work paragraph stating that deployment-oriented optimization should impose safety-related feasibility constraints, such as $\max_{x,t}\rho(x,t)\le \rho_{\max}$, upper bounds on high-density exposure, or constraints on the duration and area of cells exceeding $\rho_{\mathrm{safe}}$. The current paper will be positioned as a fixed-weight scalarized framework, while hard safety constraints will be identified as a necessary extension for field use.

## 对逐章节意见的回复与修改说明

### Abstract

Response: We agree that the abstract currently overstates the maturity and generality of the framework. The revised abstract will be shortened and reorganized into problem, method, experiment, and conclusion. It will emphasize that the work is a simulation-based optimization framework rather than a field-ready deployment system.

Proposed revision: Replace absolute claims such as `consistently outperforms` with protocol-limited statements such as `under the tested five-seed protocol`. The abstract will report relative and paired-seed results rather than relying only on the raw mean objective value 2.6338.

### Introduction

Response: We agree that some policy-oriented statements should be rewritten in a more technical IEEE style. The introduction should better separate real-world motivation, modeling gap, control variables, and contributions.

Proposed revision: The revised introduction will reduce broad governance language and focus on multi-stage pedestrian movement, direction constraints, internal entrance metering, and mixed discrete-continuous simulation optimization. The contributions will be compressed into three technical points: the Bellman--conservation-law model with admissible directions and mobility guidance, the entrance-rate control and objective framework, and the HCMBO optimization procedure with high-fidelity rechecking.

### Related Work

Response: We agree that the related work should be reorganized around the paper's technical gap.

Proposed revision: The related-work section will be grouped into crowd control and flow regulation, macroscopic pedestrian and Hughes-type models, anisotropic/HJB/Finsler or constrained route-choice modeling, and mixed-variable simulation-based optimization. Additional references will be added for anisotropic eikonal/HJB methods, ordered upwind or fast marching/sweeping solvers, and mixed-variable Bayesian optimization.

### Methodology

Response: We agree that this is the part requiring the most rigorous revision. The method section must distinguish assumptions, continuous formulation, discrete implementation, and optimization objective.

Proposed revision: The method section will add a compact assumption list, a complete definition of $U(x;s)$, the corrected mobility-tensor interpretation of $M(x;\eta_0)$, a discrete Bellman update consistent with the solver, a finite-volume and CFL description, explicit mass accounting under metering, and a nomenclature table. The objective definitions and HCMBO algorithm will be treated as the authoritative references for all experiments.

### Experiments

Response: We agree that the experiments should be organized as research questions and that the main optimizer comparison requires clearer protocol reporting.

Proposed revision: Section IV will be prefaced by research questions: whether $U+M$ changes flow structure as expected, whether direction and entrance capacity create non-monotonic trade-offs, whether HCMBO improves the fixed scalar objective under the tested budget, and whether the optimized policy transfers to a Bund-inspired abstraction. The optimizer comparison will report seed-wise outcomes, paired wins, high-fidelity recheck budget, feasibility rate, and objective-term decomposition. The Bund-inspired experiment will remain supplementary and will emphasize load redistribution and its cost.

### Discussion and Conclusion

Response: We agree that the discussion should more explicitly address limitations and deployment requirements.

Proposed revision: The discussion will state that the present experiments are simplified and not calibrated to site observations. It will also clarify that field deployment requires validated geometry, measured demand, calibrated behavioral preferences, executable management rules, staff and facility constraints, and hard safety constraints. The conclusion will retain the technical contribution but remove overgeneralized claims.

## 对 P1/P2 补强建议的处理

### Paired-seed statistics and larger seed sets

Response: We agree. The current five-seed result is informative but limited. The revised manuscript will at least report paired wins, mean paired differences, and confidence intervals where supported by the data. If additional simulations are not completed before submission, the limitation will be explicitly acknowledged rather than overstated.

### Weight sensitivity and Pareto analysis

Response: We agree that the Bund transfer result shows dependence on the current scalar weights. The manuscript will add this as a limitation and, if time permits, include a supplementary weight-sensitivity table or describe it as future work. A Pareto or weight-sweep analysis is the appropriate next step because safety, efficiency, waiting, and load balance do not improve monotonically together.

### Demand intensity sensitivity

Response: We agree. The method is intended for demand levels beyond normal daily flow, so inflow-rate sensitivity is important. If not added as a full experiment, the revised discussion will state that broader inflow multipliers are needed to test robustness.

### Mass closure verification

Response: We agree. Mass closure should be added as a diagnostic check. The revised experiment section will report the mass balance residual or explicitly state where residual diagnostics are stored.

### Runtime and equal-budget reporting

Response: We agree. The optimizer comparison will report evaluation budget, high-fidelity recheck count, and runtime or wall-clock cost where available. If exact runtime was not recorded for all historical runs, the manuscript will state that limitation and avoid runtime-based claims.

### Additional baselines

Response: We agree that stronger mixed-variable baselines such as SMAC, categorical CMA-ES, or mixed-integer BO would strengthen the paper. These will be treated as future work unless new experiments are completed. The current comparison will be described as a finite-budget comparison against representative baselines rather than an exhaustive optimizer benchmark.

### Figures and tables

Response: We agree that some figures and tables need improved readability. The framework figure can be split into model and optimization workflows, the $U+M$ mechanism figure should use a clearer two-column layout, and the optimizer summary table should be placed where it can be read without crowding. Tables will reduce excessive decimal precision and add units or normalization notes.

## 建议用于正文的关键改写片段

### Mobility tensor wording

Proposed text:

In the Bellman update, $M(x;\eta_0)$ is used as an anisotropic mobility tensor rather than a conventional cost metric. A larger value of $u^\top M(x;\eta_0)u$ reduces the local step cost through the factor $1/\sqrt{u^\top M(x;\eta_0)u}$, and therefore larger $\eta_0$ increases the effective mobility along the channel tangent direction.

### Objective-standardization wording

Proposed text:

The raw objective terms have different units and numerical ranges. Each term is therefore transformed into a standardized value $\tilde J_k$ using a fixed reference scale defined from the experimental baseline set. The scalar objective used for optimization is $J(z)=\sum_{k=1}^{5}\lambda_k\tilde J_k(z)$. Diagnostic aggregates reported in individual experiments are not used as replacement optimization objectives unless explicitly stated.

### Bund transfer wording

Proposed text:

The Bund-inspired transfer experiment should be interpreted as a supplementary structural validation rather than as evidence of field-ready deployment. The controlled case reduces the scalar objective mainly through the load-balance term $J_3$, but it also increases the travel-time term and safety-exposure evaluation term, decreases cumulative sink flow, and leaves more mass in the system. Thus, the transferred policy improves channel redistribution under the current weights while sacrificing part of throughput and travel efficiency.

### Statistical-claim wording

Proposed text:

Under the tested five-seed protocol and unified high-fidelity rechecking, HCMBO obtains the lowest mean objective among the tested methods and improves over TPE-Mixed BO in four out of five paired seeds. These results indicate favorable finite-budget performance in the present experimental setting, but larger seed sets and additional mixed-variable baselines are needed before making stronger statistical claims.

## 给编辑和审稿人的简短回复版本

We thank the reviewers for the detailed and constructive comments. We agree that the manuscript should be positioned as an interpretable macroscopic modeling and simulation-optimization framework for multi-channel pedestrian control, rather than as a directly deployable Bund management system. In the revision, we will clarify the mathematical meaning of the anisotropic tensor as a mobility tensor, provide a complete discrete definition of the admissible direction set, add numerical stability and mass-balance descriptions, unify the objective notation and standardization, and provide a reproducible HCMBO algorithm with hyperparameters and high-fidelity rechecking protocol. We will also reduce the strength of statistical and deployment-oriented claims. In particular, the Bund-inspired transfer experiment will be presented as a supplementary validation showing scalar-objective reduction and channel-load redistribution, while explicitly reporting the accompanying costs in travel time, safety-exposure evaluation, exit throughput, and residual system mass. These revisions are intended to make the manuscript more mathematically precise, reproducible, and appropriately scoped.

