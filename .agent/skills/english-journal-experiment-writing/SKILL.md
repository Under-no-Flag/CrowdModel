---
name: english-journal-experiment-writing
description: Rewrite experimental sections, result analyses, ablation studies, and comparison sections into formal English journal style. Use when converting lab notes, implementation records, code-oriented descriptions, or draft result summaries into concise academic prose while preserving data, equations, figures, tables, and claim boundaries.
---

# English Journal Experiment Writing

## Purpose

Convert process-oriented experiment notes into a publishable experimental section. The output should read as a journal article, not as a progress report or implementation log.

## Workflow

1. Identify the research question, experimental setting, control variables, metrics, baselines, ablation factors, and claim boundaries.
2. Remove implementation-trace language such as code paths, result file names, run labels, debugging history, and chronological narration unless required for reproducibility.
3. Reorganize the text around the logic of evidence: setup, mechanism validation, response analysis, comparative evaluation, ablation, and discussion.
4. Write each result paragraph as purpose, observation, interpretation, and implication.
5. Preserve all numbers, formulas, table structures, figure references, method names, and limitations.

## Style Rules

- Use objective academic prose. Prefer “This study”, “the experiment”, “the results indicate”, or passive constructions where appropriate.
- Avoid conversational phrases such as “we then ran”, “this round”, “current result”, “it should be noted”, and “from the CSV”.
- Do not overclaim. Qualify superiority claims by the evaluation budget, random seeds, fidelity level, and metrics.
- Use high-fidelity validation results for final performance claims. Low- and medium-fidelity results may support screening or diagnostic discussion only.
- Treat failure cases and unstable seeds as limitations or discussion points, not as incidental notes.

## Result-Writing Pattern

Use this compact structure for key paragraphs:

1. “To evaluate whether ..., we compare ... under ...”
2. “The results show that ...”
3. “This behavior can be attributed to ...”
4. “Therefore, ... under the stated experimental budget.”

## Final Check

Before delivery, check whether the section:

- reads like a paper rather than a report,
- avoids code and file-system references,
- explains each table or figure in prose,
- separates main comparison from ablation,
- keeps all claims supported by high-fidelity results,
- states limitations without weakening valid findings.
