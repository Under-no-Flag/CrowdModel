# Humanizer 复查报告

目标文件：`writing/IEEE_lATEX/New_IEEEtran_how-to.tex`

复查日期：2026-06-28

当前稿件 SHA256：`443107F12836D5B0E96582F88F26DD01C669C5C9F003CFFDE7E56C68C37AD418`

本次只读复查，没有修改 `.tex` 原文。

## 1. 总体结论

这轮修改已经明显降低了 AI 写作痕迹。摘要、引言前两段、相关工作和结论主体都比上一版更具体，夸饰性词汇和三段式套话少了很多。当前稿件没有明显聊天式痕迹、知识截止声明、emoji、宣传式语气，也没有 Unicode em dash 或 en dash。第 961-963 行的 `--` 是表格缺失值，不属于问题。

仍建议处理的点很少，主要是结论中有一个重复句，另有少数残留模板句可以继续压缩。

## 2. 必改项

### 2.1 第 1033-1034 行：结论重复

当前两句意思几乎相同：

```tex
The transfer result should therefore be read as evidence of mechanism transfer and load redistribution, not as a deployable plan for the Bund area.
Therefore, the transfer result should be interpreted as evidence of mechanistic transferability and load-redistribution capability, not as a ready-to-deploy operational plan for the Bund area.
```

问题：

1. `therefore` 连续出现。
2. `evidence of mechanism transfer...` 和 `evidence of mechanistic transferability...` 是同义复述。
3. 这是当前全文最明显的残留编辑痕迹，也会显得像 AI 重写后未去重。

Draft rewrite:

```tex
The transfer result should therefore be read as evidence of mechanism transfer and load redistribution, not as a deployable plan for the Bund area.
```

What still sounds AI-generated:

1. `should therefore be read as evidence` 仍然稍微公式化，但在论文结论中可以接受。

Final suggested rewrite:

```tex
The transfer result supports mechanism transfer and load redistribution, but it is not a deployable plan for the Bund area.
```

建议：删除第 1034 行，用上面的 final 句替换第 1033 行，或者至少保留两句中的一句。

## 3. 建议再润色项

### 3.1 第 50 行：`highly representative` 仍偏泛

当前句：

```tex
Among various urban crowd gathering scenarios, open scenic areas, waterfront sightseeing platforms, and historic commercial districts are highly representative.
```

问题：`highly representative` 是常见泛化评价，信息量不高。

Final suggested rewrite:

```tex
This study focuses on open scenic areas, waterfront sightseeing platforms, and historic commercial districts, where pedestrian routes often remain open and partially reversible.
```

### 3.2 第 84-88 行：引出贡献的模板感仍在

当前句群：

```tex
To address the above issues, this paper proposes a macroscopic crowd modeling and control optimization framework for the ``Bund sightseeing platform--multi-stepped passage'' open scenic area scenario.
The framework connects multi-stage pedestrian movement, passage direction management, geometric guidance, and entrance release control in a unified simulation-and-optimization workflow.
It aims to provide a quantitative basis for comparing candidate crowd-control strategies under limited simulation budgets, rather than relying only on empirical rules or post-event assessment.

The main contributions of this paper are as follows.
```

问题：

1. `To address the above issues` 和 `The main contributions...` 都是论文常见模板。
2. `It aims to provide...` 可更直接。

Final suggested rewrite:

```tex
We propose a macroscopic crowd modeling and control optimization framework for the ``Bund sightseeing platform--multi-stepped passage'' open scenic area scenario.
The framework connects multi-stage pedestrian movement, passage direction management, geometric guidance, and entrance release control in one simulation-and-optimization workflow.
It compares candidate strategies under limited simulation budgets, rather than relying only on empirical rules or post-event assessment.

This paper makes four contributions.
```

### 3.3 第 97 行：贡献句里 `achieves the best mean objective` 略像结果宣传

当前句：

```tex
\item We design a hierarchical constrained mixed-variable black-box optimization framework (HCMBO) for joint direction and entrance-rate control. By exploiting the direction--capacity structure and applying unified high-fidelity rechecking, HCMBO achieves the best mean objective among representative mixed-variable baselines under the tested budget.
```

问题：贡献列表里直接写 `achieves the best` 稍微像摘要宣传句。

Final suggested rewrite:

```tex
\item We design a hierarchical constrained mixed-variable black-box optimization framework (HCMBO) for joint direction and entrance-rate control. The method exploits the direction--capacity structure and uses unified high-fidelity rechecking to compare candidate policies under the tested budget.
```

### 3.4 第 667-669 行：图注和正文略重复

当前句：

```tex
\caption{Spatial transferability of the combined $U+M$ mechanism. Moving the controlled channel induces corresponding migration of dominant flow channels, streamlines, and density hotspots.}
...
Fig.~\ref{fig:g1_um_configuration} also shows that the combined $U+M$ mechanism is spatially transferable.
Changing the guided channel from the upper to the lower channel moves the dominant flow channel and local density hotspot accordingly.
These results support the use of the Bellman--conservation-law simulator as the lower-level evaluator for subsequent strategy optimization.
```

问题：正文重复了图注的 `spatially transferable` 和 `moves... hotspot`，读起来像自动扩写。

Final suggested rewrite:

```tex
Fig.~\ref{fig:g1_um_configuration} compares the upper-, middle-, and lower-channel guidance cases.
In each case, the dominant flow channel and local density hotspot move with the guided channel.
This supports using the Bellman--conservation-law simulator as the lower-level evaluator for subsequent strategy optimization.
```

### 3.5 第 797-801 行：结果比较句过长

当前第 800 行把三组 paired comparisons 塞进一个句子，读起来机械。

Final suggested rewrite:

```tex
In paired comparisons, HCMBO has a lower objective than TPE-Mixed BO in four of five seeds, with an average objective difference of $-0.1066$.
It is also lower than Random Search in all five seeds and lower than Pure SA and Enum-DE in four of five seeds.
The remaining high HCMBO objective is mainly caused by a larger safety-exposure term, so later versions should add more explicit safety constraints during search.
```

## 4. 可以保留的内容

以下位置虽然命中了关键词，但不建议为了去 AI 味强行改：

1. 第 33 行摘要中的 `HCMBO gives the lowest mean score`：这是结果事实，且上下文有实验预算限定。
2. 第 73-75 行实践背景：虽然有 `critical locations`，但这是现场管理术语，不是明显夸饰。
3. 第 143-144 行 `Despite these advances... remains underexplored`：这是相关工作 gap 句，当前版本并不过度。
4. 第 274 行 `effective cost`：这是模型术语，不能机械删。
5. 表格里的 `Best`, `Lowest`, `Efficient`：作为列名或短标签可保留。
6. 第 1040 行 `Future work should...`：结论未来工作句已经比上一版具体，不需要为了避免模板而牺牲清晰度。

## 5. 当前复查结论

如果只从 humanizer 角度看，当前稿件已经达到可接受状态。建议至少处理第 1033-1034 行重复句；其余 3.1-3.5 属于润色项，不是硬伤。
