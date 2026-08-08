"""Cheap baseline step scorers for the causal probe (Phase 3 comparison against spectral).

These operate on model output *logits* (a forward pass), NOT gradients, so they are the
forward-pass-only competitors the plan benchmarks spectral strength / diversity against:
per-step entropy (skeleton selection, cf. DEAR/Wang et al.), perplexity (PPL filtering), and
local average log-probability (LALP, "The Signal is in the Steps", arXiv 2510.03988).

All scorers take step_spans as (start, end) row indices into the token axis of `logits`.
Empty steps score 0.0. Kept dependency-light (pure torch) and CPU-unit-testable.
"""

import torch
import torch.nn.functional as F


def _mean_over_spans(values: torch.Tensor, step_spans: list[tuple[int, int]]) -> list[float]:
    out = []
    for start, end in step_spans:
        out.append(0.0 if end <= start else float(values[start:end].mean().item()))
    return out


def step_mean_entropy(logits: torch.Tensor, step_spans: list[tuple[int, int]]) -> list[float]:
    """Mean predictive entropy H = -sum p log p over the step's tokens (nats).

    High entropy marks 'decision' positions (branches/connectives); it is blind to confident-yet-
    wrong substance (the DEAR critique). Baseline for entropy-selective supervision.
    """
    log_p = F.log_softmax(logits.float(), dim=-1)
    entropy = -(log_p.exp() * log_p).sum(dim=-1)  # (N,)
    return _mean_over_spans(entropy, step_spans)


def step_perplexity(
    logits: torch.Tensor, target_ids: torch.Tensor, step_spans: list[tuple[int, int]]
) -> list[float]:
    """Per-step perplexity exp(mean NLL of the target tokens). Higher = harder for the model."""
    log_p = F.log_softmax(logits.float(), dim=-1)
    nll = -log_p[torch.arange(logits.shape[0]), target_ids]  # (N,)
    out = []
    for start, end in step_spans:
        out.append(0.0 if end <= start else float(torch.exp(nll[start:end].mean()).item()))
    return out


def step_local_logprob(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    step_spans: list[tuple[int, int]],
    window: int | None = None,
) -> list[float]:
    """Mean target log-probability over the step's tokens (LALP-style; higher = more 'natural').

    True LALP conditions each step only on the `window` preceding steps, which requires the caller
    to supply logits recomputed under truncated context (Phase 3 does this at generation time).
    This function averages whatever conditional logits it is given; `window` is accepted for API
    symmetry and recorded by the caller. Distinct from perplexity only under such local logits.
    """
    log_p = F.log_softmax(logits.float(), dim=-1)
    token_logprob = log_p[torch.arange(logits.shape[0]), target_ids]  # (N,)
    return _mean_over_spans(token_logprob, step_spans)
