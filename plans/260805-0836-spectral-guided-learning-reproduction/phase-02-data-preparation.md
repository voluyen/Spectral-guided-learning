---
phase: 2
title: "Data Preparation"
status: in-progress
priority: P1
effort: "3-5h"
dependencies: [1]
---

# Phase 2: Data Preparation

## Overview
Sample ~2k long-CoT trajectories from `nvidia/AceReason-1.1-SFT`, tokenize with Qwen3 chat template, filter to ≤16k tokens, segment responses into reasoning steps with token-offset boundaries.

## Requirements
- Functional: output JSONL where each record = {id, prompt, response, input_ids, response_token_span, steps: [{token_start, token_end}], n_tokens}. Step text is not duplicated in the record — decoding `input_ids[token_start:token_end]` reproduces it exactly and that round-trip is asserted at write time.
- Non-functional: deterministic (fixed seed 42); step boundaries exact under tokenizer (no drift)

## Architecture
`data-prep.py` (runs on server, CPU-only ok):
1. Load AceReason-1.1-SFT via `datasets` (streaming to avoid full download if large). Filter math samples (dataset has math+code; paper samples math problems).
2. Tokenize prompt with Qwen3-1.7B-Base tokenizer. **No chat template for Base model** — use plain `{problem}\n` + response format, OR Qwen3 template if AceReason provides one; decide from dataset format, document choice.
3. Filter: total tokens ≤16,384 AND response yields ≥2 steps. Sample 2,000 post-filter — the stream is shuffled with seed 42 (`datasets.shuffle`, 10k buffer) before filtering, so the selection is a deterministic random draw rather than the head of the corpus.
4. Segmentation (**reasoning step definition**): split CoT trajectories at logical boundaries using regex `r'([.?!\}\]])([\s\n]+)([A-Z])'` — cut before the capital letter, i.e. after sentence-ending punctuation (or a closing `}`/`]` ending inline LaTeX) plus whitespace. Deterministic, preserves semantic completeness. **No merge step** — every boundary found becomes a step, short steps are kept as-is (only zero-token pieces are dropped). Map each step to a token range by incremental prefix tokenization.

## Related Code Files
- Create: `src/data-prep.py`, `configs/data-config.yaml` (dataset name, n_samples, cutoff, seed)

## Implementation Steps
1. Inspect AceReason-1.1-SFT schema (fields, whether responses are R1-generated with `<think>` tags) — adjust parsing.
2. Implement load → filter → sample → segment → save `data/train-2k-segmented.jsonl`.
3. Log stats: token-length histogram, steps/sample distribution, tokens/step distribution.
4. Unit check: for 20 random samples, `tokenizer.decode(input_ids[token_start:token_end]) == step_text` (modulo whitespace).

## Success Criteria
- [ ] 2,000 samples, all ≤16k tokens
- [ ] Step boundary round-trip check passes on sampled records
- [ ] Stats logged (sentence-level split → expect O(100+) steps/sample for long CoT)

## Risk Assessment
- AceReason responses may embed `<think>...</think>`: keep full trajectory as supervision target (paper distills complete CoT), but segment inside think-block too
- If AceReason math subset ill-suited (e.g. short CoTs), fallback: `open-r1/OpenR1-Math-220k` — decision point logged, ask user if switch needed
