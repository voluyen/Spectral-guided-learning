# 2026-08-05 — Brainstorm & Plan: Spectral-guided Learning Reproduction

## What happened
Read ICML 2026 paper "Uncovering the Gradient Geometry of Long CoT" (`Uncovering_the_Gradient.pdf`), brainstormed reproduction scope with user, produced approved design + 7-phase implementation plan (`plans/260805-0836-spectral-guided-learning-reproduction/`).

## Key decisions
- **Scope cut (user):** method pipeline only — no reproduction of analysis experiments (Figs 2-4), no full Table 1. Paper's full setup (8×L20, 4 models, 10k samples) infeasible: local machine has NO GPU; user has 1× 40-48GB via SSH.
- **Scale:** Qwen3-1.7B-Base full-FT, ~2k samples from `nvidia/AceReason-1.1-SFT`, cutoff 16k (not 32k). Trend-level validation: Spectral vs Vanilla SFT, eval on 5 paper benchmarks.
- **Analytic gradient capture:** g_t = W_U^T(softmax−onehot) — forward-only, avoids autograd hooks at 16k seq. Mandatory autograd cross-check in phase 3.
- **Reproduction gap found:** paper never publishes energy threshold p (Eq. 8) — plan sweeps p∈{0.7,0.8,0.9} on selection stats (cheap, no retrain) with user decision gate. Segmentation heuristic (`\n\n`) also assumed, not specified by authors.
- **Framework:** custom PyTorch + HF Trainer (user choice over paper's Llama-Factory) — mask implemented via labels=-100 + sum/Z normalization, so masked loss ≈ standard CE plumbing.

## Impacts / risks logged
- AIME24/25 + GPQA pre-registered as noise at this scale; conclusions gated on MATH500 + OlympiadBench.
- Phase-3 decision gate: if k* not ≪ min(T,d) at 1.7B, method premise weak at this scale — surface before spending GPU-days on training.
- Budget: ~2-3 GPU-days total (capture ~4h, 2× train 10-20h, eval ~1d).

## Artifacts
- Brainstorm report: `plans/reports/brainstorm-260805-0836-spectral-guided-learning-reproduction.md`
- Plan: `plans/260805-0836-spectral-guided-learning-reproduction/plan.md` (+7 phase files), set as active plan.

## Unresolved
- SSH server alias + sync method (git vs rsync) — needed at phase 1 start.
- GPQA is HF-gated — user needs access approval or benchmark dropped.
