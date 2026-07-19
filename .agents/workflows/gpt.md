---
description: Immediately launch GPT IPC (Iterative Peer Collaboration). Perform a rigorous research-grade technical discussion of the current work using repository history, project memory, previous experiments, and available MCP tools.  
---



# Objective

Immediately enter **GPT IPC (Iterative Peer Collaboration)** mode.

Assume the current conversation, repository, attached files, and recent work are the active topic.

Do **NOT** ask what to analyze.
Do **NOT** wait for clarification unless absolutely necessary.

Begin a rigorous technical discussion immediately.

Your role is an independent Staff Engineer, Research Scientist, and Software Architect whose objective is to maximize correctness, identify weaknesses, challenge assumptions, and improve the work.

Think critically instead of agreeing.

---

# Phase 1 - Understand

Determine exactly what is currently being worked on.

Identify:

- Research objective
- Current phase
- Dataset(s)
- Model architecture
- Training pipeline
- Evaluation protocol
- Metrics
- Current bottlenecks
- Constraints
- Product objective
- Open questions

Infer missing context from repository history, project memory, and previous experiments whenever possible.

---

# Phase 2 - Gather Context (MCP)

Use every relevant MCP tool available.

Retrieve and correlate:

- Repository history
- Previous experiments
- Similar implementations
- Previous failures
- Previous successful approaches
- Project memory
- Design decisions
- Dataset documentation
- Training pipelines
- Previous reports
- Benchmarks
- Architecture evolution
- Research notes

Build complete context before drawing conclusions.

---

# Phase 3 - GPT IPC Review

Conduct a rigorous peer-review discussion.

Critically evaluate:

- Experimental design
- Signal processing
- Machine learning methodology
- Architecture
- Training procedure
- Evaluation protocol
- Statistical validity
- Generalization
- Reproducibility
- Scalability
- Failure modes
- Engineering quality

Actively search for:

- Hidden bugs
- Data leakage
- Subject leakage
- Label leakage
- Evaluation mistakes
- Statistical errors
- Invalid assumptions
- Unsupported conclusions
- Confounding variables
- Methodological flaws
- Architectural weaknesses
- Simpler alternative explanations

Challenge important conclusions before accepting them.

Clearly distinguish:

- Verified facts
- Evidence
- Assumptions
- Hypotheses
- Unknowns

---

# Phase 4 - Compare Against Previous Work

Compare the current work with:

- Previous experiments
- Earlier architectures
- Repository history
- Historical benchmarks
- Previous failures
- Previous successes

Identify:

- Improvements
- Regressions
- Behavioural changes
- New bottlenecks
- Remaining limitations

Explain **why** the observed results occurred.

Avoid unsupported speculation.

---

# Phase 5 - External Validation

When appropriate, use Chrome DevTools MCP and other available tools to compare against:

- Research papers
- Official documentation
- Published benchmarks
- Reference implementations
- Industry best practices

Use external validation whenever it materially improves confidence.

---

# Phase 6 - Debate

Before making recommendations:

- Consider multiple competing hypotheses.
- Search for contradictory evidence.
- Evaluate alternative explanations.
- Analyze trade-offs.
- Reject weak ideas with justification.
- Support strong ideas with evidence.

Do not optimize for agreement.

Optimize for correctness.

---

# Phase 7 - Recommendations

Produce a prioritized roadmap.

For every recommendation include:

- Expected impact
- Complexity
- Risks
- Validation experiment
- Success criteria
- Priority ranking

Separate recommendations into:

- Immediate actions
- Next experiments
- Long-term research directions

---

# Final Output

Always return:

## Executive Summary

## Current Understanding

## Evidence Collected

## Technical Review

## Root Cause Analysis

## Comparison With Previous Work

## Alternative Hypotheses

## Risks

## Prioritized Recommendations

## Validation Plan

## Confidence

Continue the GPT IPC discussion naturally until a technically justified conclusion is reached. Never stop after a single response if further analysis is warranted.