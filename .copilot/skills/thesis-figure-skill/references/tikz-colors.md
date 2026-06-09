# TikZ 配色参考

> **何时加载**：默认无需加载——`references/tikz-template.tex` 已内嵌学术配色。
> 仅当用户明确要求"draw.io 经典风格"或需要色值语义对照时加载本文件。

## 语义映射（两套配色一致）

| 语义 | 颜色 | 用途 |
|------|-----|------|
| 蓝 | Blue | 通用基础、输入 |
| 绿 | Green | 核心模块、创新点 |
| 橙 | Orange | 数据流、传输 |
| 紫 | Purple | 决策、验证 |
| 红 | Red | 关键操作、强调 |
| 灰 | Grey | 辅助、存储 |

扩展色语义：

| 语义 | 颜色 | 用途 |
|------|-----|------|
| 金 (Gold) | Yellow-orange | 标注、高亮 |
| 青绿 (Teal) | Blue-green | 安全、验证 |
| 青 (Cyan) | Light blue | 辅助强调 |
| 粉 (Pink) | Light red | 警告、异常 |
| 黄 (Yellow) | Soft yellow | 阶段、步骤 |
| 黄绿 (Lime) | Light green | 生物、自然 |

## 方案一：学术配色（默认）

颜色饱和度更高，在论文打印和屏幕阅读中辨识度更好。**已在 `references/tikz-template.tex` 中预定义**。

```latex
% ===== 学术配色（已在 tikz-template.tex 中） =====
\definecolor{acaBlueLine}{HTML}{6080B0}      \definecolor{acaBlueFill}{HTML}{DBEAFE}
\definecolor{acaGreenLine}{HTML}{30A060}     \definecolor{acaGreenFill}{HTML}{A0D0A0}
\definecolor{acaOrangeLine}{HTML}{D06020}    \definecolor{acaOrangeFill}{HTML}{FFE6CC}
\definecolor{acaPurpleLine}{HTML}{6020D0}    \definecolor{acaPurpleFill}{HTML}{E1D5E7}
\definecolor{acaRedLine}{HTML}{B05050}       \definecolor{acaRedFill}{HTML}{F8CECC}
\definecolor{acaGreyLine}{HTML}{666666}      \definecolor{acaGreyFill}{HTML}{F5F5F5}
\definecolor{acaGoldLine}{HTML}{D09000}      \definecolor{acaGoldFill}{HTML}{F8F8E0}
\definecolor{acaTealLine}{HTML}{009060}      \definecolor{acaTealFill}{HTML}{D1FAE5}
\definecolor{acaCyanLine}{HTML}{00B0D0}      \definecolor{acaCyanFill}{HTML}{E0F7FA}
\definecolor{acaPinkLine}{HTML}{D06090}      \definecolor{acaPinkFill}{HTML}{FCE4EC}
\definecolor{acaYellowLine}{HTML}{E0C060}    \definecolor{acaYellowFill}{HTML}{FFF8E1}
\definecolor{acaLimeLine}{HTML}{80B060}      \definecolor{acaLimeFill}{HTML}{E8F5E9}
% Zone 背景（极浅）
\definecolor{zoneBlueBg}{HTML}{E0E0F8}
\definecolor{zoneGreenBg}{HTML}{ECFDF5}
\definecolor{zonePurpleBg}{HTML}{F5F3FF}
\definecolor{zoneRedBg}{HTML}{F8E0E0}
\definecolor{zoneYellowBg}{HTML}{F8F0C0}
\definecolor{zoneOrangeBg}{HTML}{FFF5EB}
```

## 方案二：draw.io 经典配色（备选）

适合需要与 draw.io 原生风格保持一致的场景。

```latex
% ===== draw.io 经典 6 色 =====
\definecolor{drawBlueFill}{HTML}{DAE8FC}    \definecolor{drawBlueLine}{HTML}{6C8EBF}
\definecolor{drawGreenFill}{HTML}{D5E8D4}   \definecolor{drawGreenLine}{HTML}{82B366}
\definecolor{drawOrangeFill}{HTML}{FFE6CC}   \definecolor{drawOrangeLine}{HTML}{D79B00}
\definecolor{drawPurpleFill}{HTML}{E1D5E7}   \definecolor{drawPurpleLine}{HTML}{9673A6}
\definecolor{drawRedFill}{HTML}{F8CECC}      \definecolor{drawRedLine}{HTML}{B85450}
\definecolor{drawGreyFill}{HTML}{F5F5F5}     \definecolor{drawGreyLine}{HTML}{666666}
```

模板里默认用 `\colorlet{drawBlueFill}{acaBlueFill}` 等别名，旧代码无需改即可复用学术配色。

## 数据可视化专用色（来自 evolution）

| 元素 | 色值 | 用途 |
|------|------|------|
| 主波形/柱状 | `#3B82F6` (waveBlue) | 蓝色系主色 |
| 副波形 | `#8B5CF6` (wavePurple) | 紫色系辅色 |
| 热力图深 | `#6D28D9` (heatDeep) | |
| 热力图中 | `#93C5FD` (heatMid) | |
| 热力图浅 | `#DBEAFE` (heatLight) | |
