# Brainstorm Report: Scaled-down Reproduction of Spectral-guided Learning

**Paper:** "Uncovering the Gradient Geometry of Long CoT: A Spectral-guided Approach to Reasoning Distillation" (ICML 2026, Fan et al.) — `Uncovering_the_Gradient.pdf`
**Date:** 2026-08-05 | **Status:** Design agreed

## Problem Statement

Reproduce the paper's **method** (Spectral-guided Learning) at reduced scale. Method = 2-stage selective distillation:
1. Offline: capture per-token loss gradients w.r.t. final hidden states, per-sample SVD, compute step-level spectral strength, dynamic-truncation selection of steps.
2. Online: masked SFT — loss only on selected steps' tokens (Eq. 9), full context in forward.

**Explicitly OUT of scope (user decision):** reproduction of analysis experiments Figs 2-4 (energy concentration comparison, spectral-strength-vs-PPL, cross-instance cosine similarity), full Table 1 (4 models, 10k data), ablations Table 2.

## Agreed Constraints

| Item | Decision |
|---|---|
| Compute | 1× GPU 40-48GB (A100/A6000/L40 class) on private SSH server; local machine has NO GPU |
| Student model | Qwen3-1.7B-Base, full fine-tuning |
| Data | ~2k samples from existing R1-distilled HF dataset (`nvidia/AceReason-1.1-SFT` preferred — same problem source as paper) |
| Runs | Vanilla SFT baseline vs Spectral-guided Learning |
| Eval | All 5 benchmarks: AIME24, AIME25, MATH500, OlympiadBench, GPQA-Diamond; temp 0.6, top-p 0.95, 4 samples/problem, vLLM |
| Framework | Custom PyTorch + HF Trainer (no Llama-Factory) |

## Key Technical Decisions

1. **Analytic gradient capture** (approved): `g_t = W_U^T (softmax(logits_t) − onehot(y_t))`, forward-only, d = hidden dim (2048). Verify against autograd on one small batch. Alternative (autograd hooks) rejected: slower, memory-heavy at long seq.
2. **Per-sample SVD** of `G ∈ R^{T×d}`; consensus subspace k* at 95% cumulative energy; spectral strength S(s) = mean leverage score `||(U_{1:k*})_t||²` over step tokens (Eq. 7).
3. **Dynamic truncation** (Eq. 8): rank steps by S desc, keep minimal set with cumulative strength ≥ p. **Paper does NOT publish p** → cheap sweep p ∈ {0.7, 0.8, 0.9} on selection stats (no retrain needed), pick one for training. Unresolved reproduction gap, documented.
4. **Segmentation:** split CoT into steps by `\n\n` paragraphs (paper says only "standardized step-wise segmentation").
5. **Training:** bf16 full-FT + gradient checkpointing + FlashAttention; **cutoff 16k** (not 32k; filter longer samples) to fit 48GB; global batch 32 via grad accumulation; lr 5e-5, cosine_with_min_lr (min 1e-5), warmup 0.1, 6 epochs — hyperparams per paper Table 3.
6. **Masked loss:** precomputed token mask M_t stored with data; loss = masked mean NLL; forward keeps full sequence (paper §3.3).
7. **Eval:** vLLM generation + `math-verify`-style answer checking; max 32,768 gen tokens.

## Evaluated Approaches (summary)

| Approach | Verdict |
|---|---|
| Full faithful reproduction (8×48GB, 4 models, 10k data) | Rejected — no such compute; cost $100s-1000s |
| Regenerate CoT via DeepSeek-R1-0528 API | Rejected — significant API cost; open R1-distilled data ≈ same distribution |
| Llama-Factory (as paper) | Rejected by user — masked token-level loss needs hacks; custom Trainer cleaner |
| Analysis-only reproduction | Rejected by user — method only |
| **Scaled-down method reproduction (chosen)** | Qwen3-1.7B, 2k samples, 1 GPU, trend-level validation |

## Planned Structure

```
src/
├── data-prep.py          # download AceReason-1.1-SFT, sample ~2k, segment steps, filter ≤16k tokens
├── gradient-capture.py   # analytic g_t per token, per-sample SVD, spectral strength per step
├── step-selection.py     # p-sweep stats, dynamic truncation → token masks saved to disk
├── train-sft.py          # HF Trainer subclass w/ masked loss; --mask flag switches vanilla/spectral
└── evaluate.py           # vLLM inference + answer verification, 5 benchmarks
configs/                  # per-run yaml
```

## Success Criteria

- Pipeline runs end-to-end on server; masked-loss verified (loss identical to vanilla when mask=all-ones).
- Analytic gradient matches autograd within tolerance on test batch.
- Spectral-guided ≥ Vanilla SFT average across benchmarks; primary signal on MATH500 + OlympiadBench (AIME/GPQA noisy at 1.7B scale — pre-registered expectation).
- Selection retains meaningfully fewer tokens than 100% (paper: fewer tokens, better accuracy).

## Risks

- AIME24/25 (30 problems) statistically noisy at this scale → don't over-interpret.
- Absolute numbers will NOT match Table 1 (smaller model, 5× less data) — trend reproduction only.
- Unknown p and segmentation heuristic → main fidelity risks; mitigate via sweep + sensitivity note.
- GPU budget: ~2-4h capture, ~10-20h/train run ×2, ~4-6h eval/model ×2 → ~2-3 GPU-days.

## Next Steps

1. Create implementation plan (`/ck:plan`) with phases: env setup on server → data prep → gradient capture/selection → masked SFT trainer → training runs → eval → report.
2. Confirm SSH access details / how code syncs to GPU server (git remote? rsync?).

## Unresolved Questions

- Exact value of energy threshold p (not in paper) — to be chosen via sweep.
- Exact step segmentation used by authors — using `\n\n` heuristic.
- Whether cumulative-energy cutoff for k* is 95% (Fig 2 dashed line suggests it) — adopting 95%.
- How code reaches the GPU server (user to confirm sync workflow).
