"""Spectral analysis primitives for Loss Subspace Attribution (paper Eq. 2-7).

Pure-torch, device-agnostic, unit-testable on CPU. All computations in float32.
"""

import time
from dataclasses import dataclass, field

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


def token_full_leverage(u_matrix: torch.Tensor) -> torch.Tensor:
    """Full-rank leverage ||U[t,:]||^2 over ALL r columns (not just top-k*).

    Sum over rows equals r (columns of U are orthonormal). High full leverage marks a
    statistically influential / outlier row; contrast with token_leverage_scores which
    truncates to the top-k* consensus subspace.
    """
    return (u_matrix.float() ** 2).sum(dim=1)


def token_residual_energy_fraction(
    u_matrix: torch.Tensor, singular_values: torch.Tensor, k_star: int
) -> torch.Tensor:
    """Div(t): fraction of token t's Sigma-weighted gradient energy OUTSIDE the top-k* consensus.

    Each token's reduced coordinate is c_t = (U diag(sigma))_t in R^r. We return
    ||c_t[k*:]||^2 / ||c_t||^2 in [0, 1]. This equals 1 - (Sigma-weighted top-k* strength),
    i.e. the residual complement of spectral strength. Non-degenerate at T >> d (unlike a
    leave-step-out subspace residual, which collapses to ~0 when the rest of the chain already
    spans the whole low-rank rowspace).
    """
    weighted = u_matrix.float() * singular_values.float()  # c_t rows, (T, r)
    energy = weighted**2
    total = energy.sum(dim=1).clamp_min(1e-12)
    residual = energy[:, k_star:].sum(dim=1)
    return (residual / total).clamp(0.0, 1.0)


def step_diversity_scores(
    u_matrix: torch.Tensor,
    singular_values: torch.Tensor,
    k_star: int,
    step_spans: list[tuple[int, int]],
) -> list[float]:
    """Div(s) = mean residual-energy-fraction over the step's token rows. Empty step -> 0.0.

    step_spans are (start, end) row indices into G (relative to the response tokens), matching
    step_spectral_strengths.
    """
    div = token_residual_energy_fraction(u_matrix, singular_values, k_star).cpu()
    scores = []
    for start, end in step_spans:
        scores.append(0.0 if end <= start else float(div[start:end].mean().item()))
    return scores


def token_novelty_ipr(u_matrix: torch.Tensor, singular_values: torch.Tensor) -> torch.Tensor:
    """IPR novelty: effective number of singular directions a token's gradient spreads across.

    c_t = (U diag(sigma))_t; p_{t,j} = c_{t,j}^2 / ||c_t||^2; Nov(t) = exp(-sum_j p log p) in [1, r].
    Structurally distinct from leverage/residual (which measure how much mass sits in / outside the
    top-k* subspace): IPR measures the SPREAD of the row's energy across modes. A row aligned to a
    single dominant direction scores ~1 (redundant) even if fully inside top-k* (i.e. high strength);
    a row combining many directions scores high (novel combination). Non-degenerate at T>>d — a
    closed-form per-row statistic on the fixed SVD basis, no leave-one-out. (Roy-Vetterli effective
    rank / Vendi-score family; see research report 260808-1449.)
    """
    weighted = u_matrix.float() * singular_values.float()  # c_t rows, (T, r)
    energy = weighted**2
    p = energy / energy.sum(dim=1, keepdim=True).clamp_min(1e-12)
    p = p.clamp_min(1e-12)
    entropy = -(p * p.log()).sum(dim=1)
    return torch.exp(entropy)  # (T,), in [1, r]


def step_novelty_ipr(
    u_matrix: torch.Tensor,
    singular_values: torch.Tensor,
    step_spans: list[tuple[int, int]],
) -> list[float]:
    """Nov(s) = mean IPR novelty over the step's token rows. Empty step -> 0.0."""
    novelty = token_novelty_ipr(u_matrix, singular_values).cpu()
    scores = []
    for start, end in step_spans:
        scores.append(0.0 if end <= start else float(novelty[start:end].mean().item()))
    return scores


def effective_rank_entropy(singular_values: torch.Tensor) -> float:
    """Effective rank via spectral (von Neumann) entropy: exp(-sum p_i log p_i), p_i = sigma_i^2 / sum.

    Measures how evenly gradient energy spreads across orthogonal directions (1 = rank-1
    concentration, larger = more spread). Used for the leave-step-out erank drop.
    """
    p = singular_values.float() ** 2
    p = p / p.sum().clamp_min(1e-12)
    p = p.clamp_min(1e-12)
    return float(torch.exp(-(p * p.log()).sum()).item())


def step_erank_drop(
    u_matrix: torch.Tensor,
    singular_values: torch.Tensor,
    step_spans: list[tuple[int, int]],
) -> list[float]:
    """Secondary diversity signal: drop in effective-rank-entropy when a step is removed.

    Uses the exact rank-|s| Gram downdate A_{\\s} = diag(sigma^2) - Ghat_s^T Ghat_s (r x r),
    whose eigenvalues are the squared singular values of G with the step's rows removed
    (Ghat = U diag(sigma)). A large drop means the step opens spectral directions the rest of
    the chain lacks. Graded (via singular values), unlike the degenerate binary subspace residual.
    """
    sv = singular_values.float()
    weighted = u_matrix.float() * sv  # Ghat = U diag(sigma), (T, r)
    gram_full = torch.diag(sv**2)  # Ghat^T Ghat == diag(sigma^2)
    full = effective_rank_entropy(sv)
    drops = []
    for start, end in step_spans:
        if end <= start:
            drops.append(0.0)
            continue
        block = weighted[start:end]
        downdated = gram_full - block.T @ block
        sv_without = torch.linalg.eigvalsh(downdated).clamp_min(0.0).sqrt()
        drops.append(full - effective_rank_entropy(sv_without))
    return drops


@dataclass
class SpectralResult:
    k_star: int
    singular_values: torch.Tensor  # (r,) float32 on CPU
    step_strengths: list[float]
    step_diversity: list[float] = field(default_factory=list)  # Div(s), residual energy fraction
    step_novelty: list[float] = field(default_factory=list)  # Nov(s), IPR effective-#-directions
    svd_seconds: float = 0.0  # wall time of the SVD call alone (the paper never reports this)
    diversity_seconds: float = 0.0  # wall time of Div + IPR computation (should be << svd_seconds)


def analyze_gradient_matrix(
    gradient_matrix: torch.Tensor,
    step_spans: list[tuple[int, int]],
    energy_cutoff: float = 0.95,
) -> SpectralResult:
    """SVD of G (T x d) -> k* at energy_cutoff -> step-level spectral strengths.

    SVD runs in float32 on G's device; results returned on CPU. `svd_seconds` isolates the
    SVD call's wall time, synchronizing the CUDA stream around it so the number reflects the
    real device compute rather than async-launch latency.
    """
    g32 = gradient_matrix.float()
    on_cuda = g32.is_cuda
    if on_cuda:
        torch.cuda.synchronize()
    svd_start = time.perf_counter()
    # full_matrices=False: U is (T, r), r = min(T, d) — leverage needs only top-k* columns
    u_matrix, singular_values, _ = torch.linalg.svd(g32, full_matrices=False)
    if on_cuda:
        torch.cuda.synchronize()
    svd_seconds = time.perf_counter() - svd_start

    k_star = effective_rank(singular_values, energy_cutoff)
    leverage = token_leverage_scores(u_matrix, k_star)
    step_strengths = step_spectral_strengths(leverage.cpu(), step_spans)

    # Div + IPR reuse the SVD already computed; time them together (cuda-synced) to confirm the
    # added cost is negligible vs the SVD call itself (feasibility gate, plan phase 2).
    if on_cuda:
        torch.cuda.synchronize()
    div_start = time.perf_counter()
    step_diversity = step_diversity_scores(u_matrix, singular_values, k_star, step_spans)
    step_novelty = step_novelty_ipr(u_matrix, singular_values, step_spans)
    if on_cuda:
        torch.cuda.synchronize()
    diversity_seconds = time.perf_counter() - div_start

    return SpectralResult(
        k_star=k_star,
        singular_values=singular_values.cpu(),
        step_strengths=step_strengths,
        step_diversity=step_diversity,
        step_novelty=step_novelty,
        svd_seconds=svd_seconds,
        diversity_seconds=diversity_seconds,
    )
