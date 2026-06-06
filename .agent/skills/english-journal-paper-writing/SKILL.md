---
name: english-journal-paper-writing
description: High-quality English journal manuscript writing and revision. Use when Codex needs to draft, rewrite, translate, polish, or audit English academic paper sections including title, abstract, introduction, related work, methodology, experiments, results, discussion, conclusion, figure captions, table captions, response-ready claim wording, or journal-style English editing while preserving technical meaning, equations, citations, data, and claim boundaries.
---

# English Journal Paper Writing

## Purpose

Use this skill to turn technical drafts, Chinese notes, experiment records, or rough manuscript sections into polished English journal prose. The output should read as a formal research article, not as a project report, lab note, or promotional summary.

## First Principles

- Preserve meaning before improving style. Do not change formulas, data, method names, citations, or conclusions unless the user asks for technical revision.
- Write with evidence discipline. Every performance claim must be supported by the provided results, figures, tables, or cited literature.
- Prefer clear, precise English over ornate vocabulary. Avoid inflated words when a simple technical verb is more accurate.
- Do not overclaim novelty or superiority. Qualify claims by setting, metric, fidelity level, seeds, budget, dataset, or assumptions.
- Remove implementation traces unless they are necessary for reproducibility. Avoid code paths, file names, run labels, debugging history, and chronological narration.
- Keep terminology consistent across the manuscript. Define abbreviations at first use and reuse the same form afterward.

## Workflow

1. Identify the target section type, audience, field, and expected journal style.
2. Extract the factual core: research problem, method, assumptions, variables, datasets or scenes, metrics, baselines, results, limitations, and contribution.
3. Decide the rhetorical role of the section:
   - Abstract: compress motivation, gap, method, result, and implication.
   - Introduction: build the problem, gap, contribution, and paper structure.
   - Method: define variables, equations, components, and algorithmic flow.
   - Experiments: describe setup, baselines, metrics, results, ablations, and limitations.
   - Discussion: explain mechanisms, trade-offs, robustness, and failure cases.
   - Conclusion: summarize contribution and future work without adding new evidence.
4. Rewrite in coherent paragraphs. Avoid bullet lists unless the target artifact explicitly calls for them.
5. Run the final quality checklist before delivering.

## Section Patterns

### Title

Use a title that states the object, method, and contribution without excessive breadth.

Good patterns:
- “A Bellman-Conservation-Law Framework for Direction and Inflow Control in Crowd Management”
- “Hierarchical Mixed-Variable Black-Box Optimization for Crowd Flow Control in Open Scenic Areas”

Avoid:
- “A Novel and Efficient Method for Smart Crowd Management”
- Vague claims such as “optimal”, “intelligent”, or “revolutionary” unless precisely justified.

### Abstract

Use 4 to 6 sentences:

1. Broad problem and importance.
2. Specific technical gap.
3. Proposed method.
4. Experimental setting and key result.
5. Main implication or limitation-aware conclusion.

Rules:
- Do not cite literature in the abstract unless the journal requires it.
- Include one or two quantitative results if available.
- Avoid generic phrases such as “with the rapid development of”.
- Use present tense for the paper contribution and past tense for specific experiments.

### Introduction

Use a funnel structure:

1. Domain importance and practical context.
2. Specific modeling or optimization challenge.
3. Limitations of existing approaches.
4. Proposed solution and why its structure matches the challenge.
5. Contributions, written as concrete and verifiable claims.

Contribution bullets should follow this form:

- “We formulate ...”
- “We develop ...”
- “We evaluate ...”

Each contribution should map to a method section or experiment section.

### Related Work

Organize by themes, not by one-paper summaries. For each theme:

1. State what the literature has achieved.
2. Identify the remaining limitation relevant to this paper.
3. Explain how the present work differs.

Do not make unsupported “no work has studied” claims. Prefer:

- “Existing studies have rarely considered ... jointly.”
- “Most formulations focus on ..., whereas the present work ...”

### Method

Method prose should explain the model as a reproducible formal system:

1. Problem definition and notation.
2. Model components and equations.
3. Control variables and constraints.
4. Objective function.
5. Optimization algorithm.

Rules:
- Keep implementation details only when they define the method.
- Use “Let”, “Define”, “Given”, and “The objective is” for formal statements.
- Avoid report wording such as “we run”, “this version”, “current implementation”, “the code uses”, or “after debugging”.
- Ensure every symbol in an equation is defined close to first use.

### Experiments

Experiments should be organized around evidence:

1. Experimental setting and metrics.
2. Baselines and fairness of comparison.
3. Main comparison.
4. Ablation.
5. Sensitivity or robustness.
6. Failure cases and limitations.

Rules:
- Final performance claims should rely on the highest-fidelity or officially selected evaluation.
- Do not use internal run names as paper evidence.
- When a method wins only on average, write “achieves the best mean performance”, not “consistently outperforms”.
- Explain both positive results and trade-offs.

### Discussion

Discussion should answer why the results occur:

- Which model component explains the observed behavior?
- Which objective terms are in tension?
- Which assumptions limit generalization?
- Which failure cases suggest future work?

Avoid merely repeating table values.

### Conclusion

Use one compact paragraph or two short paragraphs:

1. Summarize the problem and method.
2. Summarize the strongest supported finding.
3. State limitations or future work.

Do not introduce new experiments, new definitions, or new claims.

## Style Rules

### Preferred Language

- Use precise verbs: formulate, derive, constrain, evaluate, compare, validate, indicate, suggest.
- Use cautious claim verbs when evidence is limited: indicate, suggest, demonstrate under, provide evidence that.
- Use strong claim verbs only when fully supported: prove, guarantee, dominate, consistently outperform.

### Avoid

- Promotional wording: novel, powerful, advanced, revolutionary, excellent, significant without data.
- Vague intensifiers: greatly, obviously, very, extremely, remarkably unless quantified.
- AI-like filler: delve into, tapestry, leverage when “use” is enough, robustly enhance, comprehensive framework without specifying content.
- Process narration: first we ran, then we found, in this round, current result, the script, the folder, the CSV.
- Overloaded sentence starts: “It is worth noting that”, “In order to”, “With the rapid development of”.

### Claim Calibration

Use calibrated templates:

- Strong: “Under the same high-fidelity budget, HCMBO achieved the lowest mean objective among the tested methods.”
- Moderate: “The results suggest that the structured search better exploits the direction-capacity hierarchy.”
- Limited: “This observation is based on the candidate library and should not be interpreted as a full Pareto analysis.”
- Failure-aware: “The weaker result under one seed indicates that safety exposure control remains sensitive to search stochasticity.”

## Translation From Chinese Drafts

When translating Chinese academic prose into English:

- Translate meaning, not sentence order.
- Convert Chinese parallel clauses into shorter English sentences.
- Replace “本文首先……然后……” with logical section prose.
- Preserve mathematical notation and Chinese-specific scene names accurately.
- If a term has no standard English equivalent, provide a concise technical translation and reuse it consistently.

## Figure And Table Writing

Captions should state what is shown and why it matters.

Figure caption pattern:

- “Comparison of ... under ... . The curves show ... , indicating ... .”

Table discussion pattern:

1. State the primary comparison.
2. Mention the strongest metric.
3. Mention a trade-off or limitation.
4. Avoid reading every row aloud.

## Final Quality Checklist

Before delivering, check:

- Does the text read like a journal article rather than a report?
- Are all data, equations, citations, and method names preserved?
- Are claims qualified by the actual evidence?
- Are limitations stated without undermining valid contributions?
- Are baselines and fairness conditions clear in experiment prose?
- Are abbreviations defined and terminology consistent?
- Are code paths, run names, file names, and debugging details removed?
- Are paragraphs cohesive rather than a list of disconnected facts?
