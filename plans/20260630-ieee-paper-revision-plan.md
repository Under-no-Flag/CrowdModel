# IEEE 论文摘要与 Introduction 强化修改计划

## 1. 修改目标

本轮修改面向 `writing/IEEE_lATEX/New_IEEEtran_how-to.tex`，目标不是重写全文，而是增强论文主线表达，使读者能更快理解：

1. 为什么城市旅游热点和大型活动场景需要定量化人群管控方法。
2. 为什么本文选择宏观模型，而不是以微观行人模型作为主线。
3. 本文的模型、控制变量、优化方法和实验结果分别取得了什么成果。
4. 摘要、Introduction、贡献项和 Conclusion 是否围绕同一条逻辑展开。

建议按“摘要压缩 -> Introduction 首句与逻辑检查 -> 宏观模型动机补强 -> 成果表达统一 -> 编译验证”的顺序推进。

## 2. 当前问题判断

### 2.1 摘要偏长，方法细节略多

当前摘要已经包含背景、方法、优化器和结果，但第二句过长，连续列出方向配置、入口强度、Hughes 框架、方向集、各向异性张量、入口流率、HCMBO、实验结果等信息。问题是信息密度过高，核心贡献被淹没。

摘要应压缩到 6 到 7 句，保留如下信息：

- 背景：大规模旅游热点和活动场景存在高密度人流风险。
- 缺口：经验式管控难以量化通道方向与入口释放率的耦合作用。
- 方法：提出基于 Hughes 连续体/Bellman--conservation-law 的宏观人群控制模型。
- 控制：统一表示通道方向、几何引导和入口流率控制。
- 优化：使用 HCMBO 搜索混合离散-连续控制策略。
- 结果：HCMBO 在测试预算内取得最低平均目标值，比 TPE-Mixed BO 低约 3.9%。
- 意义：为城市旅游热点的通道组织和入口释放策略比较提供仿真依据。

### 2.2 Introduction 的段落首句总体合理，但第 2 段需要加强

当前 Introduction 段落逻辑基本是：

1. 城市治理背景与人群安全问题。
2. 大规模行人流复杂性。
3. 开放景区/外滩场景的代表性。
4. 现有安保实践。
5. 现有实践和仿真优化研究的不足。
6. 本文提出的方法。
7. 本文贡献。

这个顺序合理，但第 2 段首句 `Large-scale pedestrian flow are complex systems.` 存在语法问题，应改为 `Large-scale pedestrian flows are complex systems.` 更重要的是，这一段虽然区分了宏观和微观层面的现象，但没有明确说明为什么本文的问题更适合宏观模型。

### 2.3 “为什么不用微观模型”说明不足

当前 Related Work 中有一句：

```tex
Macroscopic crowd models treat pedestrian flows as continuous media, offering computational efficiency and analytical tractability compared to microscopic approaches.
```

这能说明宏观模型的优势，但不足以回应审稿人可能提出的“为什么不用微观模型”。需要补充本文任务场景下微观模型不适用的原因：

- 本文目标是策略级搜索，不是个体避碰轨迹复现。
- 需要在有限仿真预算内反复评估数百个方向-容量候选策略。
- 开放景区通常缺少足够精细的个体轨迹校准数据。
- 管理者关心的输出是区域密度演化、通道负载、入口等待和高密度暴露，而不是单个行人的瞬时决策。

### 2.4 成果表达需要更集中

当前 Conclusion 已经给出 HCMBO 优势、消融结果和 Bund-inspired transfer 结果，但成果表达仍可更清楚地区分为四类：

- 模型成果：多阶段、多路径宏观模型。
- 控制成果：通道方向、各向异性引导和入口流率控制的统一算子。
- 优化成果：HCMBO 对混合离散-连续控制变量的结构化搜索。
- 实验成果：机制验证、横向优化比较、消融和 Bund-inspired transfer 验证。

## 3. 具体修改任务

### Task 1: 压缩摘要

**目标：** 将摘要压缩为更清楚的“问题-方法-结果-意义”结构。

**建议动作：**

- 合并或删减过细的方法实现描述。
- 保留 HCMBO、3.9% improvement、simulation-based basis 等核心结果。
- 避免在摘要中重复解释方向集、各向异性张量、入口流率等细节，细节留给 Methodology。

**验收标准：**

- 摘要读完后能清楚回答：本文研究什么问题、提出什么方法、取得什么结果。
- 摘要中不出现超过 45 到 50 个词的超长句。
- 摘要结果与正文表格和 Conclusion 中的数值一致。

## Task 2: 检查 Introduction 每段第一句

**目标：** 确保每段首句承担明确的逻辑功能。

**逐段处理建议：**

1. 第 1 段：保留治理背景，但可把问题导向“pre-event quantitative assessment”。
2. 第 2 段：修正语法，并增加宏观模型适配性说明。
3. 第 3 段：保留开放景区/外滩场景代表性。
4. 第 4 段：保留现有实践说明。
5. 第 5 段：保留不足分析，并突出“缺乏结构化利用通道几何、方向规则和入口容量”的问题。
6. 第 6 段：提出本文框架时明确它是 policy-level macroscopic modeling and optimization framework。
7. 贡献段：每个贡献项都应对应一个具体成果，而不是只描述工作内容。

**验收标准：**

- 只读每段第一句时，能看出 Introduction 的逻辑链：背景 -> 复杂性 -> 场景 -> 实践 -> 缺口 -> 方法 -> 贡献。
- 第 2 段不再只是泛泛说“复杂”，而是引出为什么需要宏观建模。

## Task 3: 补强为什么研究宏观模型

**目标：** 在 Introduction 或 Related Work 中明确说明宏观模型与本文问题的匹配关系。

**推荐插入位置：**

- 优先放在 Introduction 第 2 段末尾。
- 如果担心 Introduction 太长，也可在 Related Work 的 `Macroscopic Crowd Modeling` 开头补充一段。

**可用英文草稿：**

```tex
For this policy-level problem, the key quantities are not individual collision-avoidance trajectories, but area-level density evolution, channel-load redistribution, entrance waiting, and high-density exposure. A macroscopic formulation is therefore suitable because it directly represents the aggregate variables used by crowd managers and can be repeatedly evaluated within an optimization loop.
```

**中文意图：**

本文关注的是区域级密度、通道负载、入口等待和高密度暴露，不是个体避碰轨迹；宏观模型既对应管理指标，也适合在优化循环中反复调用。

**验收标准：**

- 读者能明确看到“宏观模型”不是默认选择，而是由本文管理目标和优化需求决定。
- 该段不否定微观模型价值，只说明本文任务更适合宏观模型。

## Task 4: 说明为什么微观模型不适用本文主线

**目标：** 回答潜在审稿问题：为什么不用微观模型。

**推荐写法：**

```tex
Microscopic pedestrian models can describe individual interactions in detail, but they are less suitable for the present strategy-search setting because hundreds of candidate direction--capacity policies must be evaluated under a limited simulation budget. They also require detailed individual-level calibration data that are often unavailable in open scenic areas. In contrast, the proposed macroscopic model provides a tractable and interpretable representation of policy-induced density and flow redistribution.
```

**写作注意：**

- 不要写成“microscopic models are unsuitable for crowd management”，这会过度否定已有研究。
- 更稳妥的表述是“less suitable for the present strategy-search setting”。
- 强调本文的目标是策略比较和区域级管理，而不是个体行为复现。

**验收标准：**

- 文中至少有 2 到 3 句直接解释微观模型在本文任务中的局限。
- 这些句子与“limited simulation budget”“policy-level objective”“aggregate density and flow variables”关联起来。

## Task 5: 凝练本研究取得的成果

**目标：** 统一摘要、贡献项、实验结果和结论中的成果表述。

**建议成果框架：**

1. 模型成果：构建多阶段、多路径宏观人群模型，表达入口、观景、离场、返程和路径再分配。
2. 控制成果：在 Bellman--conservation-law 框架中统一方向规则、几何引导和入口流率控制。
3. 优化成果：提出 HCMBO，结构化搜索方向配置和入口流率的混合离散-连续控制空间。
4. 实验成果：机制验证表明模型组件按预期工作；横向对比中 HCMBO 平均目标值最低；Bund-inspired 场景显示策略能重分配通道负载。

**可补入 Conclusion 的表达方向：**

```tex
The main outcome of this study is not a deployable field-control plan, but a reproducible modeling and optimization framework that can compare direction--capacity policies before implementation.
```

**验收标准：**

- Conclusion 不只重复数值，而能说明本文贡献的可复用价值。
- Bund-inspired transfer 结果被表述为机制转移和负载重分配证据，而不是现实部署方案。

## 4. 推荐实施顺序

1. 先改摘要，形成压缩后的全文主线。
2. 修改 Introduction 第 2 段，补充宏观模型动机和微观模型不适用原因。
3. 检查 Introduction 每段首句，使段落逻辑连贯。
4. 调整贡献项，使四条贡献分别对应模型、控制、评估、优化。
5. 修改 Conclusion，使成果表达与摘要和贡献项一致。
6. 最后做全文一致性验证，包括术语、缩写、数值和 LaTeX 编译。

## 5. 验证清单

- [ ] 摘要压缩后仍包含核心结果：HCMBO、3.9%、simulation-based basis。
- [ ] Introduction 第 2 段明确说明为什么采用宏观模型。
- [ ] 文中明确说明为什么微观模型不适合作为本文主线。
- [ ] 贡献项与 Conclusion 的成果分类一致。
- [ ] `TPE-Mixed BO`、`HCMBO`、`CFL`、`RMT`、`CAT` 等缩写首次出现均已展开。
- [ ] 结果数值与表格一致：3.9%、100% feasible rate、26.9% transfer objective reduction 等。
- [ ] 编译命令通过：

```powershell
cd D:\CrowdModels\writing\IEEE_lATEX
latexmk -pdf -interaction=nonstopmode -halt-on-error New_IEEEtran_how-to.tex
```

- [ ] Git 检查通过：

```powershell
cd D:\CrowdModels
git diff --check -- writing/IEEE_lATEX/New_IEEEtran_how-to.tex
```

## 6. 风险与处理

| 风险 | 影响 | 处理 |
|---|---|---|
| 摘要压缩后丢失关键结果 | 审稿人难以快速判断贡献 | 保留 HCMBO、3.9% 和策略比较意义 |
| 对微观模型表述过强 | 容易被认为否定已有微观研究 | 使用 `less suitable for the present strategy-search setting` |
| 宏观模型动机与 Related Work 重复 | 文字冗余 | Introduction 讲任务动机，Related Work 讲模型谱系 |
| Bund transfer 结果表述过度 | 被质疑可部署性 | 强调机制转移和负载重分配，不称为现场部署方案 |
| 新增文字引发版面或编译问题 | PDF 页数或排版变化 | 每次修改后运行 `latexmk` 和 `git diff --check` |

## 7. 预期完成状态

完成后，论文应能更直接地回答四个问题：

1. 为什么研究这个场景：城市旅游热点和大型活动存在需要提前评估的高密度人流管控问题。
2. 为什么用宏观模型：本文关注区域级密度、通道负载、入口等待和策略比较，宏观模型更匹配这些管理指标。
3. 为什么不用微观模型作为主线：微观模型适合个体行为细节，但在本文的多候选策略搜索、有限仿真预算和缺少个体级校准数据条件下成本过高。
4. 本文取得了什么成果：提出统一的宏观建模与优化框架，验证了机制有效性，并在测试预算内取得优于对比方法的策略搜索结果。
