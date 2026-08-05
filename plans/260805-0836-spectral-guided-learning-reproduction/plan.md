---
title: "Scaled-down Reproduction: Spectral-guided Learning (Reasoning Distillation)"
status: in-progress
created: 2026-08-05
updated: 2026-08-05
source: plans/reports/brainstorm-260805-0836-spectral-guided-learning-reproduction.md
blockedBy: []
blocks: []
---

# Plan: Spectral-guided Learning — Scaled-down Reproduction

Reproduce the **method** of "Uncovering the Gradient Geometry of Long CoT" (ICML 2026) at reduced scale.
Scope: method pipeline only — NO analysis figs (2-4), NO full Table 1.

## Setup Summary

| Item | Value |
|---|---|
| Student | Qwen3-1.7B-Base, full-FT bf16 |
| Compute | 1× GPU 40-48GB via SSH server (local machine has no GPU) |
| Data | ~2k samples, `nvidia/AceReason-1.1-SFT` (fields: category/source/input/output), cutoff 16k tokens |
| Runs | Vanilla SFT vs Spectral-guided (masked loss) |
| Eval | AIME24/25, MATH500, OlympiadBench, GPQA-Diamond; vLLM; temp 0.6, top-p 0.95, n=4, max 32k |
| Train HP | lr 5e-5, cosine_with_min_lr (min 1e-5), warmup 0.1, 6 epochs, global batch 32, grad ckpt, FlashAttention |

## Current State (2026-08-05)

**All pipeline code is written and validated on CPU.** GPU stages wait on server availability.
Run stages with `./scripts/run-pipeline.sh {data|capture|masks|smoke|train|eval}`.

| # | Phase | Code | Verified locally | Remaining (needs GPU/server) |
|---|-------|------|------------------|------------------------------|
| 1 | [Environment](phase-01-environment-setup.md) | ✅ | local CPU venv, 48 tests green | server env, code sync method |
| 2 | [Data prep](phase-02-data-preparation.md) | ✅ | ran on real AceReason samples; span round-trip passes | full 2k run |
| 3 | [Gradient capture](phase-03-gradient-capture-spectral-analysis.md) | ✅ | **analytic == autograd, max diff 3.7e-9**; CLI ran on real data | 2k samples w/ real 1.7B |
| 4 | [Step selection](phase-04-step-selection-masks.md) | ✅ | mask emission at p=0.95 on real data | rerun on real strengths |
| 5 | [Masked SFT](phase-05-masked-sft-training.md) | ✅ | Trainer trained a tiny model; masked loss == vanilla CE when unmasked | 2 full runs |
| 6 | [Evaluation](phase-06-evaluation-benchmarks.md) | ✅ | scoring + rescore path tested | vLLM generation, 10 runs |
| 7 | [Results report](phase-07-results-report.md) | — | — | after eval |

### Verified facts from local runs
- AceReason schema is `input`/`output`/`category`; the math filter and long-CoT parsing work as coded.
- Real data at 16k cutoff: mean ~7k tokens and **~350 steps/sample** (mean 20 tokens/step) under the
  sentence-boundary segmentation — plenty of granularity for step selection. Re-measured 2026-08-05
  after the segmentation change; the earlier 148 steps/sample figure came from paragraph splitting.
- Capture cost scales with the chunked (T×V) matmuls; expect ~3-5 s/sample on an A100 → ~2-4 h for 2k.
- Benchmark loaders check out against the live Hub: AIME24 (30), AIME25 (30), MATH500 (500),
  OlympiadBench math-text-EN (674). AIME24 stores its gold answer as `\boxed{...}` inside `solution`
  rather than an `answer` field — handled in `benchmarks.py`.

### Action needed before the server run
- **`HF_TOKEN` must be set** wherever the pipeline runs: GPQA-Diamond is gated, and the loader fails
  without authentication even though dataset access has been granted. Everything else loads anonymously.
- Decide the code-sync method to the server (git remote vs rsync) — `scripts/sync-to-server.sh` is not
  written yet because it depends on that choice.

### Open decision gates
- **k\* on the real model** (phase 3): local numbers used a randomly initialized tiny model, so they say
  nothing about low-rank structure. If k\* is not ≪ min(T,d) with real Qwen3-1.7B, the method's premise is
  weak at this scale — check before spending GPU-days on training.
- **drop ratios at p=0.95** (phase 4): p is the paper's constant, not a gate — but with near-uniform
  strengths a 0.95 threshold drops almost nothing, making Spectral ≈ Vanilla. Check the logged
  step-drop / token-drop ratios before committing GPU time to the training runs.

## Key Dependencies
- SSH GPU server access + sync workflow (phase 1)
- HF datasets: AceReason-1.1-SFT, benchmark sets (GPQA access: granted)
- Paper equations: Eq.7 (spectral strength), Eq.8 (dynamic truncation), Eq.9 (masked loss)

## Success Criteria (plan-level)
- End-to-end pipeline runs on server; analytic gradient verified vs autograd ✅ (already green locally)
- Masked loss == vanilla loss when mask is all-ones ✅ (unit test)
- Spectral-guided ≥ Vanilla SFT on avg; primary signal MATH500 + OlympiadBench
- Selection drops a meaningful fraction of tokens (step-drop and token-drop logged separately)

## Known Risks
- p=0.95 (paper value) may drop too few tokens at this scale → log step-drop and token-drop ratios in phase 4
- AIME/GPQA statistically noisy at 1.7B/2k scale — trend-level claims only
- ~2-3 GPU-days total budget
