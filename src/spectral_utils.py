"""Spectral analysis primitives for Loss Subspace Attribution (paper Eq. 2-7).

Pure-torch, device-agnostic, unit-testable on CPU. All computations in float32.
"""

from dataclasses import dataclass

import torch


def cumulative_energy(singular_values: torch.Tensor) -> torch.Tensor:
    """E(k) = sum_{i<=k} sigma_i^2 / sum_j sigma_j^2 (paper Eq. 4). Returns tensor of shape (r,)."""
    sq = singular_values.float() ** 2
    return torch.cumsum(sq, dim=0) / sq.sum().clamp_min(1e-12)


def effective_rank(singular_values: torch.Tensor, energy_cutoff: float = 0.95) -> int:
    """k* = smallest k with E(k) >= energy_cutoff."""
    energy = cumulative_energy(singular_values)
    return int(torch.searchsorted(energy, energy_cutoff).item()) + 1


def token_leverage_scores(u_matrix: torch.Tensor, k_star: int) -> torch.Tensor:
    """Per-token projection energy onto consensus subspace: ||(U_{1:k*})_t||^2 for each row t."""
    return (u_matrix[:, :k_star].float() ** 2).sum(dim=1)


def step_spectral_strengths(
    leverage: torch.Tensor, step_spans: list[tuple[int, int]]
) -> list[float]:
    """S(s) = mean leverage score over the step's token rows (paper Eq. 7).

    step_spans are (start, end) row indices into the gradient matrix G
    (i.e. indices relative to the response tokens, NOT absolute sequence positions).
    """
    strengths = []
    for start, end in step_spans:
        if end <= start:
            strengths.append(0.0)
        else:
            strengths.append(float(leverage[start:end].mean().item()))
    return strengths


@dataclass
class SpectralResult:
    k_star: int
    singular_values: torch.Tensor  # (r,) float32 on CPU
    step_strengths: list[float]


def analyze_gradient_matrix(
    gradient_matrix: torch.Tensor,
    step_spans: list[tuple[int, int]],
    energy_cutoff: float = 0.95,
) -> SpectralResult:
    """SVD of G (T x d) -> k* at energy_cutoff -> step-level spectral strengths.

    SVD runs in float32 on G's device; results returned on CPU.
    """
    g32 = gradient_matrix.float()
    # full_matrices=False: U is (T, r), r = min(T, d) — leverage needs only top-k* columns
    u_matrix, singular_values, _ = torch.linalg.svd(g32, full_matrices=False)
    k_star = effective_rank(singular_values, energy_cutoff)
    leverage = token_leverage_scores(u_matrix, k_star)
    return SpectralResult(
        k_star=k_star,
        singular_values=singular_values.cpu(),
        step_strengths=step_spectral_strengths(leverage.cpu(), step_spans),
    )
