# System Architecture

Reproduction of **Spectral-guided Learning** (Fan et al., ICML 2026 — `Uncovering_the_Gradient.pdf`)
at reduced scale: Qwen3-1.7B-Base, ~2k long-CoT samples, single 40-48GB GPU.

## Pipeline

```
AceReason-1.1-SFT ──> data_prep ──> gradient_capture ──> build_masks ──> train_sft ──> evaluate
                      (segment)     (analytic ∇ + SVD)   (Eq. 8)         (Eq. 9)       (vLLM)
                          │                │                  │              │
                  segmented.jsonl   strengths.parquet   train-*.jsonl   checkpoints/
                                    + spectral/{id}.npz  + selection-stats.json
```

Record schema out of `data_prep.py`: `{id, prompt, response, input_ids, response_token_span,
steps: [{token_start, token_end}], n_tokens}`. Step text is not stored — decoding
`input_ids[token_start:token_end]` reproduces it exactly, and that round-trip is asserted during
data prep.

Two training runs share every hyperparameter and differ **only in the loss mask**, which is what makes
the comparison a clean A/B: `train-vanilla.jsonl` supervises all response tokens, `train-spectral.jsonl`
supervises only the selected high-spectral-strength steps.

## Modules (`src/`)

| File | Role |
|---|---|
| `segmentation.py` | Split CoT at sentence-level logical boundaries (`([.?!\}\]])([\s\n]+)([A-Z])`, no merging); map each step to exact token spans |
| `gradient_utils.py` | Analytic loss gradient w.r.t. hidden states (Eq. 1), chunked over positions |
| `spectral_utils.py` | SVD, cumulative energy E(k) (Eq. 4), k\*, leverage scores, step strength (Eq. 7) |
| `step_selection.py` | Dynamic truncation (Eq. 8), token-mask construction, step/token drop stats |
| `masked_loss.py` | Selective masked objective (Eq. 9) |
| `data_collator.py` | Pads batches, turns `loss_mask` into labels with `-100` |
| `benchmarks.py` | Unified loaders for the five evaluation sets |
| `answer_scoring.py` | `\boxed{}` extraction, symbolic math equivalence, MCQ letter parsing |
| `data_prep.py`, `gradient_capture.py`, `build_masks.py`, `train_sft.py`, `evaluate.py`, `compare_results.py` | CLI stages |

Importable modules use snake_case (Python cannot import kebab-case names); shell scripts use kebab-case.

## Design decisions

**Analytic gradients instead of autograd.** For target token *y* predicted by hidden state *h*,
`g = W_Uᵀ(softmax(W_U h) − onehot(y))`. One forward pass yields every per-token gradient, with no
backward graph and no per-token retain_grad, which matters at 16k sequence length. The identity is
asserted against autograd in tests and again by `gradient_capture.py --verify` before each corpus run
(observed deviation: 3.7e-9 in float32).

**Chunking over positions.** Materializing the full (T × V) logits for a 16k sequence over a 151k
vocabulary would cost ~5 GB. The capture path calls the base transformer directly (skipping `lm_head`)
and applies the unembedding chunk-wise, keeping peak memory at (chunk × V).

**Masking lives in the labels.** Masked positions are set to `-100`, so masked-out steps still take part
in the forward pass — full context is preserved, exactly as the paper specifies — but contribute no
gradient. With an all-ones mask the objective reduces exactly to standard SFT cross-entropy (unit-tested).

**Loss normalization.** One `sum / Z` per optimizer step (Eq. 9), where Z is the supervised-token count
across *all* gradient-accumulation microbatches of that step — `MaskedSFTTrainer` takes Trainer's
`num_items_in_batch` as the divisor. Averaging microbatch losses instead would over-weight sparsely
supervised sequences, which is precisely what spectral masks produce. Applied identically to both runs,
so the mask stays the only difference between them.

## Deviations from the paper

| Aspect | Paper | Here | Reason |
|---|---|---|---|
| Model | 4 models, 4B-8B | Qwen3-1.7B-Base | single GPU |
| Data | 10k samples | ~2k | compute budget |
| Cutoff | 32k | 16k | memory at full fine-tuning |
| Framework | Llama-Factory | custom HF Trainer | token-level masking needs custom loss |
| Energy threshold p | 0.95 | 0.95 (paper value) | — |
| Step segmentation | "standardized" | sentence-boundary regex split, no merging | not specified in the paper |

Absolute accuracies will not match Table 1; the target is the **trend** (Spectral ≥ Vanilla with fewer
supervised tokens).
