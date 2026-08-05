---
phase: 3
title: "Gradient Capture & Spectral Analysis"
status: in-progress
priority: P1
effort: "1d"
dependencies: [2]
---

# Phase 3: Gradient Capture & Spectral Analysis

## Overview
Stage 1 of the method: compute per-token loss gradients w.r.t. final hidden states analytically, per-sample SVD, step-level spectral strength (paper Eq. 1-7).

## Requirements
- Functional: per sample → {k*, singular_values, step_strengths[]} saved; per-token leverage scores optional debug output
- Non-functional: 2k samples processed in hours on 1 GPU; float32 SVD for numerical stability

## Architecture
Analytic gradient (avoids backward): for supervised position t with target y_t,
```
ℓ_t = CE(logits_t, y_t);  logits_t = W_U h_t
g_t = ∇_{h_t} ℓ_t = W_U^T (softmax(logits_t) − onehot(y_t)) ∈ R^d   (d=2048)
```
Compute per chunk of positions: `probs (chunk×V)`; `probs[range, y] -= 1`; `G_chunk = probs @ W_U` (V×d matmul). Avoid materializing full T×V for 16k seq — chunk 1024 positions.

Per-sample pipeline (Eq. 2-7):
1. Stack G ∈ R^{T_resp × d} over response tokens only (supervised positions).
2. SVD (float32, `torch.linalg.svd(full_matrices=False)`).
3. Cumulative energy E(k) = Σ_{i≤k} σ_i² / Σ σ_j²; k* = min k with E(k) ≥ 0.95.
4. Spectral strength per step s: S(s) = mean_{t∈s} ||U[t, :k*]||² (Eq. 7).

## Related Code Files
- Create: `src/gradient-capture.py`, `src/spectral-utils.py` (SVD, energy, strength — unit-testable, no GPU needed for tests)
- Create: `configs/capture-config.yaml` (energy_cutoff 0.95, chunk_size, dtype)
- Create: `tests/test-spectral-utils.py`, `tests/test-analytic-gradient.py`

## Implementation Steps
1. `spectral-utils.py`: pure functions (svd→k*→strengths) with synthetic-matrix unit tests (known low-rank matrix → expected k*).
2. `gradient-capture.py`: load model bf16, no_grad forward with `output_hidden_states=False` (only need logits… note: analytic form needs only logits + W_U + targets — hidden states not needed explicitly). Batch samples individually (var-length), chunked gradient computation, SVD on GPU, save `data/spectral/{id}.npz` + aggregate `data/spectral-strengths.parquet`.
3. **Verification (mandatory):** for 3 short samples, compute g_t via autograd (`h_t.retain_grad()` on last hidden state, backward on Σℓ_t) and assert `allclose(analytic, autograd, rtol=1e-3)` in float32. Note: tied vs untied embeddings — Qwen3-1.7B ties `lm_head` to `embed_tokens`; use `model.get_output_embeddings().weight` as W_U.
4. Run over 2k samples; log runtime, k* distribution, strength distributions.

## Success Criteria
- [ ] Analytic == autograd on verification samples
- [ ] Unit tests pass for spectral-utils
- [ ] 2k samples processed; k* ≪ min(T,d) for typical samples (low-rank structure sanity — paper's core claim)
- [ ] Aggregate parquet loadable by phase 4

## Risk Assessment
- SVD of 16k×2048 float32 ≈ fast on GPU but memory ~130MB/matrix — fine; fallback `torch.svd_lowrank` if slow
- If k* NOT small (no low-rank structure at 1.7B scale) → method premise weak at this scale; surface finding to user before training (decision gate)
- bf16 logits → float32 softmax to avoid precision loss in g_t
