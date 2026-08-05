---
phase: 4
title: "Step Selection & Token Masks"
status: in-progress
priority: P1
effort: "0.5d"
dependencies: [3]
---

# Phase 4: Step Selection & Token Masks

## Overview
Dynamic truncation (Eq. 8): per sample, keep minimal step set whose cumulative spectral strength ≥ p·(total strength), with **p = 0.95 (the paper's value — fixed, not tuned; no sweep)**. Emit token-level masks M_t for training.

## Requirements
- Functional: training JSONL augmented with `loss_mask` (list[0/1] aligned to input_ids); stats report at p=0.95
- Non-functional: pure CPU, seconds-fast, deterministic

## Architecture
Per sample: sort steps by S(s) desc → greedy accumulate until Σ_{sel} S / Σ_{all} S ≥ p → selected steps' token ranges set M_t=1, others 0. Prompt tokens always M_t=0 (never supervised). Final-answer step: log how often it survives selection (if often dropped → flag; paper keeps logic steps, answer usually high-strength).

## Related Code Files
- Create: `src/step-selection.py`
- Modify: `configs/data-config.yaml` (`energy_threshold_p: 0.95`)
- Create: `tests/test-step-selection.py` (toy strengths → expected selection)

## Implementation Steps
1. Implement selection + mask emission; write `data/train-spectral.jsonl` (+ `data/train-vanilla.jsonl` baseline via `--vanilla`).
2. Stats logging — **step-drop ratio and token-drop ratio reported separately** (mean, median, corpus totals), since many short steps may be dropped while few tokens are, or vice versa. Also: samples where <3 steps kept, final-answer-step survival rate. Written to `data/selection-stats.json` + stdout table.
3. No decision gate on p — it is the paper's constant.

## Success Criteria
- [ ] Unit test passes (greedy minimal-set logic correct, ties handled deterministically)
- [ ] Step-drop and token-drop ratios logged separately at p=0.95
- [ ] Every sample retains ≥1 step; mask aligns with input_ids length

## Risk Assessment
- p=0.95 is near-inclusive: with flat strength distributions the mask may drop very little — that is the paper's setting, but log drop ratios to confirm the spectral run actually differs from vanilla
- Degenerate samples (1-2 giant steps) → selection ≈ all-or-nothing; log count, exclude if >5% of data
