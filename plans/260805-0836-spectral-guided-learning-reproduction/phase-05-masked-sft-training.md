---
phase: 5
title: "Masked SFT Training (2 runs)"
status: in-progress
priority: P1
effort: "1-2d GPU"
dependencies: [4]
---

# Phase 5: Masked SFT Training (Vanilla + Spectral)

## Overview
Custom HF Trainer with masked NLL (Eq. 9). Two runs on identical data/HP: Vanilla (mask = all response tokens) and Spectral (mask from phase 4). Only the mask differs — clean A/B.

## Requirements
- Functional: `train-sft.py --config configs/train-{vanilla|spectral}.yaml`; checkpoints per epoch; loss curves logged
- Non-functional: fits 1×40-48GB; resumable; seed-fixed (42)

## Architecture
- Loss (Eq. 9): `L = −(1/Z) Σ_t M_t log P(y_t|y_<t)` with `Z = Σ_t M_t` **per batch** (accumulate across grad-accum steps consistently: sum masked NLL and token counts, divide once — avoids bias from varying mask density per microbatch).
- Trainer subclass overrides `compute_loss`; collator pads `input_ids`, `labels` (=-100 where M_t=0 → actually implement via labels: set label=-100 for masked-out positions, then standard CE with reduction='sum' / Z). Vanilla = labels -100 only on prompt tokens. **Simplification: mask lives entirely in labels → no custom loss needed except sum/Z normalization.**
- Memory: bf16 weights 3.4GB + grads 3.4 + Adam fp32 states ~13.6 + fp32 master ~6.8 ≈ 27GB + activations (grad ckpt, seq 16k, batch 1) → fits 40GB; if OOM → `adamw_bnb_8bit` optimizer (saves ~10GB) or DeepSpeed ZeRO-2 offload.
- HP (paper Table 3): lr 5e-5, cosine_with_min_lr min_lr 1e-5, warmup_ratio 0.1, 6 epochs, per_device_batch 1 × grad_accum 32, bf16, gradient_checkpointing, flash_attention_2 (or sdpa fallback), max_len 16384.

## Related Code Files
- Create: `src/train-sft.py`, `src/masked-data-collator.py`, `configs/train-vanilla.yaml`, `configs/train-spectral.yaml`
- Create: `tests/test-masked-loss.py`

## Implementation Steps
1. Implement dataset loader (JSONL from phase 4), collator (labels from loss_mask), Trainer with sum-CE/Z loss + `cosine_with_min_lr` scheduler (transformers supports via `lr_scheduler_kwargs={"min_lr": 1e-5}`).
2. **Sanity test:** all-ones mask ⇒ loss identical to standard SFT causal-LM loss on same batch (assert in test).
3. Smoke run: 20 samples, 10 steps on server — check VRAM, throughput, loss decreasing.
4. Full run 1: Vanilla SFT (6 epochs, ~2k samples). Save `checkpoints/vanilla/`.
5. Full run 2: Spectral (same seed/HP, masked data). Save `checkpoints/spectral/`.
6. Log: train loss curves, tokens-supervised count per run (expect Spectral ≪ Vanilla), wall time.

## Success Criteria
- [ ] Masked-loss sanity test passes
- [ ] Both runs complete 6 epochs without OOM; losses converge
- [ ] Final checkpoints saved + config/seed recorded alongside
- [ ] Supervised-token count logged (evidence of "fewer tokens" claim)

## Risk Assessment
- OOM at 16k seq → mitigations ordered: 8-bit optimizer → ZeRO-2 CPU offload → cutoff 12k (re-run phase 2 filter)
- 6 epochs × 2k may overfit small model — keep per-epoch checkpoints, eval best-of-last-two if final degrades (apply SAME rule to both runs to keep A/B fair)
- ~10-20h per run estimate; schedule sequentially (single GPU)
