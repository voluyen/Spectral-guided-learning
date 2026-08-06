# Project Changelog

## 2026-08-06 — Phase 2 segmentation made linear-time

**Fixed** — `segment_response_token_spans()` re-tokenized the cumulative prefix once per step,
costing O(steps × tokens). On a real-shaped sample (8k tokens, 400 steps) that was 1074 ms versus
5.8 ms for a single tokenizer pass — 187x, and the dominant cost of phase 2 at ~350 steps/sample.
Step boundaries are now mapped from the character offsets of one `return_offsets_mapping` pass,
bisected into token indices. Output is **bit-identical** to the old implementation (ids and spans
verified equal across long, short, unsplittable and empty responses).

**Changed** — `build_record()` takes `max_tokens` and returns `None` for over-length samples,
rejecting them right after tokenizing instead of after full span mapping; the longest samples were
the most expensive to discard.

**Added** — two regression tests with a `MergingTokenizer` stand-in that reproduces BPE's
context dependence, asserting the emitted ids are the canonical tokenization rather than a
piecewise concat. Encoding steps separately would inflate token counts ~4% and feed the model
whitespace-only tokens it was never pretrained on — a silent corruption, since the existing
decode round-trip check cannot detect it.

**Added** — progress bar for the phase-2 scan. The collection loop moved out of `main()` into
`collect_records()`, which reports accepted samples against `n_samples` with live rejection counts
(`scanned`, `skipped_long`, `skipped_few_steps`) in the postfix — a stalled bar with a climbing
`scanned` means the filters are eating the corpus. A notice prints before the bar appears, since
`datasets` yields nothing until its 10k-row shuffle buffer is downloaded. `tqdm` promoted to a
direct requirement.

**Fixed** — problem and response are NFC-normalized before tokenizing. AceReason carries
decomposed forms (U+2261 + U+0338 for "≢") that the tokenizer folds to one codepoint at encode
time, so the stored `response` was unrecoverable from its own token span on 3 of 2000 records.
Token ids were already canonical (`re-encode(response) == ids` held), so training was unaffected —
but the record's documented round-trip property was not. `verify_span_alignment()` now runs on
every record instead of the first 20; the full pass costs ~3s against a ~60s run, and the 20-record
cap is what let these three through.

**Verified** — 56 unit tests pass; CPU smoke test runs end to end; collection loop exercised
against a synthetic stream covering both rejection branches.

**Phase 2 executed** — 2,000 samples in 58s (138 MB, `data/train-2k-segmented.jsonl`), reproducible
under seed 42 across two runs. `scanned=2080 skipped_long=79 (3.8%) skipped_few_steps=0`.
tokens/sample mean 7,341 (median 6,832, max 16,362); steps/sample mean 279.3 (min 22, max 987);
tokens/step mean 26.0 (median 19, max 2,047). Independent audit of all 2,000 records: spans
contiguous, covering the response span exactly, non-empty, and decoding back to the stored text —
0 failures on every check. Note the tokens/step tail (median 19 vs max 2,047): a single kept-or-
dropped step can move ~1k tokens, which is why `build_masks.py` reports step-drop and token-drop
separately.

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
