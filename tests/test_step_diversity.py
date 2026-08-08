"""Phase 1 math-correctness + non-degeneracy tests for step-diversity metrics.

Guards the key trap from the red-team review (report 260808-1435): a leave-step-out subspace
residual passes toy tests but collapses to ~0 on real data (T >> d, many steps). The Div metric
here is the Sigma-weighted residual energy fraction, which stays graded in that regime — the
non-degeneracy test below is the one that would fail for the discarded leave-out definition.
"""

import torch

from spectral_utils import (
    analyze_gradient_matrix,
    effective_rank,
    effective_rank_entropy,
    step_diversity_scores,
    step_erank_drop,
    step_novelty_ipr,
    token_full_leverage,
    token_leverage_scores,
    token_novelty_ipr,
    token_residual_energy_fraction,
)
from step_scorers import step_mean_entropy, step_local_logprob, step_perplexity


def _svd(matrix):
    u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
    return u, s


def test_div_is_bounded_in_unit_interval():
    torch.manual_seed(0)
    u, s = _svd(torch.randn(40, 12))
    div = token_residual_energy_fraction(u, s, effective_rank(s))
    assert div.min() >= 0.0 and div.max() <= 1.0


def test_div_is_exact_residual_complement_of_weighted_strength():
    # Div(t) + (Sigma-weighted top-k* energy fraction) == 1 exactly, per token.
    torch.manual_seed(1)
    u, s = _svd(torch.randn(50, 16))
    k = effective_rank(s)
    weighted = u * s
    energy = weighted**2
    total = energy.sum(dim=1)
    top_k_fraction = energy[:, :k].sum(dim=1) / total
    div = token_residual_energy_fraction(u, s, k)
    assert torch.allclose(div + top_k_fraction, torch.ones_like(div), atol=1e-5)


def test_div_detects_planted_diversity_at_T_much_greater_than_d():
    # THE guard: T >> d, many steps. Most steps ride the shared low-rank signal (low Div); a few
    # 'diverse' steps carry strong off-signal (tail) energy. Div must SEPARATE them. A leave-step-out
    # subspace residual would read ~0 for every step here (the rest already spans the rowspace),
    # so this is the test that discriminates the chosen metric from the discarded one.
    torch.manual_seed(2)
    n_tokens, dim, rank, step = 2000, 64, 5, 20
    basis = torch.randn(rank, dim)
    gradient = torch.randn(n_tokens, rank) @ basis  # shared low-rank reasoning subspace
    spans = [(i, i + step) for i in range(0, n_tokens, step)]  # 100 steps
    diverse_steps = {10, 37, 88}
    for idx in diverse_steps:
        start, end = spans[idx]
        gradient[start:end] += 3.0 * torch.randn(step, dim)  # off-subspace (tail) energy
    u, s = _svd(gradient)
    div = torch.tensor(step_diversity_scores(u, s, effective_rank(s), spans))

    diverse_mask = torch.zeros(len(spans), dtype=torch.bool)
    diverse_mask[list(diverse_steps)] = True
    assert div[diverse_mask].min() > div[~diverse_mask].max(), "Div failed to rank diverse steps up"
    assert div.max() > 0.1  # not collapsed to ~0 like the leave-out definition would be


def test_div_is_graded_zero_in_consensus_one_in_tail():
    # Construct coords directly: a step supported only on top-k* columns -> Div 0;
    # a step supported only on tail columns -> Div 1.
    r, k = 6, 3
    singular_values = torch.ones(r)
    u = torch.zeros(8, r)
    u[0:4, 0] = 1.0  # step A: energy in column 0 (< k*) -> consensus
    u[4:8, 5] = 1.0  # step B: energy in column 5 (>= k*) -> tail
    div = step_diversity_scores(u, singular_values, k, [(0, 4), (4, 8)])
    assert abs(div[0] - 0.0) < 1e-6
    assert abs(div[1] - 1.0) < 1e-6


def test_empty_step_scores_zero():
    torch.manual_seed(3)
    u, s = _svd(torch.randn(10, 4))
    assert step_diversity_scores(u, s, effective_rank(s), [(2, 2)]) == [0.0]


def test_full_leverage_sums_to_rank():
    torch.manual_seed(4)
    u, _ = _svd(torch.randn(60, 20))
    assert torch.isclose(token_full_leverage(u).sum(), torch.tensor(20.0), atol=1e-4)


def test_full_leverage_dominates_truncated_leverage():
    torch.manual_seed(5)
    u, s = _svd(torch.randn(30, 10))
    k = effective_rank(s)
    assert torch.all(token_full_leverage(u) - token_leverage_scores(u, k) >= -1e-6)


def test_effective_rank_entropy_matches_planted_spread():
    assert abs(effective_rank_entropy(torch.tensor([5.0, 0.0, 0.0])) - 1.0) < 1e-4  # rank-1
    assert abs(effective_rank_entropy(torch.ones(4)) - 4.0) < 1e-4  # uniform -> 4


def test_erank_drop_downdate_equals_recompute():
    # Exact identity: eig(diag(sigma^2) - Ghat_s^T Ghat_s) are squared singular values of G_rest.
    torch.manual_seed(6)
    gradient = torch.randn(8, 5)
    u, s = _svd(gradient)
    span = (2, 4)  # remove rows 2..3, leaving 6 >= r rows
    drop_downdate = step_erank_drop(u, s, [span])[0]
    rest = torch.cat([gradient[: span[0]], gradient[span[1] :]], dim=0)
    expected = effective_rank_entropy(s) - effective_rank_entropy(torch.linalg.svdvals(rest))
    assert abs(drop_downdate - expected) < 1e-4


def test_analyze_gradient_matrix_emits_step_diversity():
    torch.manual_seed(7)
    gradient = torch.randn(12, 6)
    result = analyze_gradient_matrix(gradient, [(0, 4), (4, 12)])
    assert len(result.step_diversity) == 2
    assert all(0.0 <= d <= 1.0 for d in result.step_diversity)


def test_ipr_novelty_is_bounded_between_one_and_rank():
    torch.manual_seed(9)
    u, s = _svd(torch.randn(40, 12))
    nov = token_novelty_ipr(u, s)
    assert nov.min() >= 1.0 - 1e-4
    assert nov.max() <= float(s.numel()) + 1e-4


def test_ipr_is_one_for_single_mode_and_rank_for_uniform_spread():
    r = 6
    singular_values = torch.ones(r)
    u = torch.zeros(2, r)
    u[0, 0] = 1.0  # all energy on one direction -> effective #dirs = 1
    u[1, :] = 1.0 / (r**0.5)  # uniform spread -> effective #dirs = r
    nov = token_novelty_ipr(u, singular_values)
    assert abs(nov[0].item() - 1.0) < 1e-5
    assert abs(nov[1].item() - r) < 1e-4


def test_ipr_decouples_novelty_from_importance():
    # Two steps BOTH fully inside the top-k* consensus (strength ~ 1, Div ~ 0), yet different IPR:
    # step A concentrates on one dominant mode (redundant, low IPR); step B spreads across k* modes
    # (novel combination, high IPR). This is the decoupling the leverage/residual axis cannot see.
    r, k = 6, 3
    singular_values = torch.ones(r)
    u = torch.zeros(8, r)
    u[0:4, 0] = 1.0  # step A: one top mode
    for row in range(4, 8):  # step B: spread across the 3 top modes
        u[row, :k] = 1.0 / (k**0.5)
    spans = [(0, 4), (4, 8)]
    div = step_diversity_scores(u, singular_values, k, spans)
    nov = step_novelty_ipr(u, singular_values, spans)
    assert div[0] < 1e-6 and div[1] < 1e-6  # both inside consensus -> residual ~ 0
    assert abs(nov[0] - 1.0) < 1e-5  # A redundant
    assert nov[1] > 2.5  # B spreads across ~3 modes


def test_analyze_gradient_matrix_emits_step_novelty():
    torch.manual_seed(10)
    result = analyze_gradient_matrix(torch.randn(12, 6), [(0, 4), (4, 12)])
    assert len(result.step_novelty) == 2
    assert all(n >= 1.0 - 1e-4 for n in result.step_novelty)


def test_token_distribution_stats_matches_direct_logits():
    # Chunked entropy/nll/logprob must equal a direct full-logits computation.
    from gradient_utils import token_distribution_stats

    torch.manual_seed(11)
    hidden = torch.randn(7, 8)
    unembedding = torch.randn(20, 8)
    targets = torch.randint(0, 20, (7,))
    entropy, nll, logprob = token_distribution_stats(hidden, targets, unembedding, chunk_size=3)

    log_p = torch.log_softmax(hidden @ unembedding.T, dim=-1)
    ref_entropy = -(log_p.exp() * log_p).sum(dim=-1)
    ref_logprob = log_p[torch.arange(7), targets]
    assert torch.allclose(entropy, ref_entropy, atol=1e-5)
    assert torch.allclose(logprob, ref_logprob, atol=1e-5)
    assert torch.allclose(nll, -ref_logprob, atol=1e-5)
    assert (entropy >= 0).all()


def test_baseline_scorers_are_well_formed():
    torch.manual_seed(8)
    logits = torch.randn(6, 100)
    targets = torch.randint(0, 100, (6,))
    spans = [(0, 3), (3, 6)]
    entropy = step_mean_entropy(logits, spans)
    ppl = step_perplexity(logits, targets, spans)
    logprob = step_local_logprob(logits, targets, spans)
    assert all(e >= 0.0 for e in entropy)
    assert all(p > 0.0 for p in ppl)
    assert all(lp <= 0.0 for lp in logprob)
