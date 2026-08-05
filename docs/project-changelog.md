# Project Changelog

## 2026-08-05 — Method decisions locked; code aligned to plan

**Changed** — two method decisions, then a pass reconciling every stage with its phase spec:
- **Step definition**: CoT is split at logical boundaries via `r'([.?!\}\]])([\s\n]+)([A-Z])'`
  (sentence-ending punctuation or closing `}`/`]`, then whitespace, then a capital). **No merge step** —
  short steps survive; only zero-token pieces are dropped. Replaces the blank-line split.
  Re-measured on real data: ~350 steps/sample, mean 20 tokens/step (was 148 steps/sample).
- **Energy threshold p = 0.95**, the paper's published value — the sweep and the decision gate are gone.
  `build_masks.py` reports **step-drop and token-drop ratios separately** (mean, median, corpus totals)
  to `data/selection-stats.json`, since a run can drop many short steps but few tokens.
- `data_prep.py` now shuffles the source stream with seed 42 before filtering (the seed was previously
  set but unused, so the "2k sample" was the head of the corpus), emits the planned record schema
  (`prompt`, `response`, `response_token_span`, `steps[{token_start, token_end}]`), and logs
  token-length / steps-per-sample / tokens-per-step distributions.
- `gradient_capture.py` writes the per-sample `data/spectral/{id}.npz` (k\*, singular values, strengths)
  the plan calls for — `output_dir` was created but never written — verifies on 3 samples, and logs
  runtime plus the step-strength distribution (a flat spectrum predicts near-zero drop at p=0.95).
- `masked_loss.py` / `MaskedSFTTrainer`: one `sum / Z` per optimizer step with Z counted across all
  gradient-accumulation microbatches (Trainer's `num_items_in_batch`), per the phase-5 spec. The old
  `per_sample` averaging over-weighted sparsely supervised sequences — exactly what spectral masks create.
  `loss_normalization` dropped from the train configs; `run-summary.json` (supervised tokens, runtime,
  final loss) saved next to each checkpoint.
- `evaluate.py` persists vLLM `finish_reason` and reports a real `truncation_rate` instead of the
  "no `\boxed`" proxy.

**Verified** — 54 unit tests pass; CPU smoke test runs end to end; `data_prep.py` → `build_masks.py` →
`gradient_capture.py --verify` re-run on live AceReason data with the new schema (analytic vs autograd
1.2e-7).

## 2026-08-05 — Pipeline implementation (local, pre-GPU)

**Added** — full Spectral-guided Learning pipeline, written and validated without a GPU:
- `src/` — segmentation, analytic gradient capture, spectral analysis, step selection, masked SFT
  trainer, benchmark loaders, answer scoring, results comparison
- `configs/` — data, capture, training (vanilla + spectral), evaluation
- `tests/` — 48 unit tests
- `scripts/run-pipeline.sh` — staged server runbook; `scripts/smoke_test_pipeline.py` — CPU end-to-end check

**Verified locally**
- Analytic gradient identity matches autograd to 3.7e-9 (tiny Qwen3, float32) — the assumption the whole
  method rests on
- Masked objective reduces exactly to standard SFT cross-entropy under an all-ones mask
- `data_prep.py` runs against real `nvidia/AceReason-1.1-SFT` data: step spans reconstruct the response
  verbatim; 20/22 samples fit the 16k cutoff; mean 8.1k tokens and 148 steps per sample
- `train_sft.py` completes real training steps (cosine_with_min_lr schedule, per-epoch checkpoints)
- `build_masks.py`, `evaluate.py --rescore`, `compare_results.py` produce correct output on fixtures

**Fixed**
- `load_aime24` read a nonexistent `answer` field; math-ai/aime24 stores gold as `\boxed{...}` inside
  `solution`. Verified all four public benchmark loaders against the live Hub (GPQA needs `HF_TOKEN`)
- Invalid regex quantifier (`\b?`) in `normalize_math` that broke all math scoring
- `TrainingArguments` hard-coded bf16/gradient-checkpointing, which cannot run on CPU — now selected from
  device availability so the same script works locally and on the server
- `torch_dtype` → `dtype` for transformers ≥4.56 (requirements floor raised accordingly)

**Pending** — every GPU stage: full 2k capture, both training runs, vLLM evaluation. Waiting on server.
