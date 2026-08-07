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
Dynamic truncation (Eq. 8): per sample, keep minimal step set whose cumulative spectral strength ≥ p·(total strength). **The paper does not publish a value for p** (checked the full text and all four appendices — the only "95%" in the paper is Fig. 2's illustrative energy threshold for a different, motivating experiment, not Eq. 8's p). `p = 0.95` is therefore our engineering starting point, not a paper-derived constant — see `src/build_masks.py --sweep` below for the decision gate that replaces the earlier (mistaken) "fixed, no sweep" framing.

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
3. **Decision gate restored**: run `python src/build_masks.py --config configs/data-config.yaml --sweep 0.7,0.8,0.9,0.95` (seconds, reuses the spectral strengths already on disk, writes no dataset file) and inspect `data/selection-stats-sweep.json` before picking the `energy_threshold_p` that goes into the real training run. Pick the smallest p whose token-drop is clearly non-zero — a near-0% drop makes the spectral run statistically indistinguishable from vanilla SFT, which would silently invalidate the A/B comparison in phase 5.

## Success Criteria
- [ ] Unit test passes (greedy minimal-set logic correct, ties handled deterministically)
- [ ] Sweep run and reviewed *before* `energy_threshold_p` is finalized in `configs/data-config.yaml`
- [ ] Step-drop and token-drop ratios logged separately for the chosen p
- [ ] Every sample retains ≥1 step; mask aligns with input_ids length

## Risk Assessment
- The paper does not publish p — an uninformed choice (e.g. one close to 1.0) can make the mask near-inclusive, dropping almost nothing and making Spectral ≈ Vanilla by construction rather than by the method failing. The sweep step above exists specifically to catch this before a GPU training run is spent on it.
- Degenerate samples (1-2 giant steps) → selection ≈ all-or-nothing; log count, exclude if >5% of data
