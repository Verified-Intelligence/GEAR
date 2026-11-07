"""Tests for the PDHG lower bound on  min c^T x + c0  s.t.  G x <= h,  l <= x <= u."""
import os
import sys

import numpy as np
import pytest
import torch
from scipy.optimize import linprog

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nlp_solver.dual_ascent import lp_bound_pdhg  # noqa: E402


def _random_lp(seed, B=4, m=12, n=8):
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(B, m, n, generator=g, dtype=torch.float64) * torch.logspace(
        -2, 3, m, dtype=torch.float64).view(1, m, 1)
    l = -torch.rand(B, n, generator=g, dtype=torch.float64) * 5
    u = torch.rand(B, n, generator=g, dtype=torch.float64) * 5
    x_feas = l + (u - l) * torch.rand(B, n, generator=g, dtype=torch.float64)
    # Rows pass through a known feasible point with random slack.
    h = torch.einsum('bmn,bn->bm', G, x_feas) + torch.rand(B, m, generator=g, dtype=torch.float64)
    c = torch.randn(B, n, generator=g, dtype=torch.float64) * 10
    c0 = torch.randn(B, generator=g, dtype=torch.float64)
    return c, c0, G, h, l, u


def _exact(c, c0, G, h, l, u):
    out = []
    for b in range(G.shape[0]):
        r = linprog(c[b].numpy(), A_ub=G[b].numpy(), b_ub=h[b].numpy(),
                    bounds=list(zip(l[b].numpy(), u[b].numpy())), method='highs')
        assert r.status == 0
        out.append(r.fun + c0[b].item())
    return torch.tensor(out, dtype=torch.float64)


@pytest.mark.parametrize('seed', range(6))
def test_bound_is_valid_at_every_budget(seed):
    c, c0, G, h, l, u = _random_lp(seed)
    exact = _exact(c, c0, G, h, l, u)
    for iters in (0, 10, 200, 2000):
        bound, y = lp_bound_pdhg(c, c0, G, h, l, u, iters=iters)
        assert torch.all(bound <= exact + 1e-7 * (1 + exact.abs())), (iters, bound, exact)
        assert torch.all(y >= 0)


def test_converges_on_badly_scaled_rows():
    c, c0, G, h, l, u = _random_lp(11, B=2, m=10, n=6)
    exact = _exact(c, c0, G, h, l, u)
    bound, _ = lp_bound_pdhg(c, c0, G, h, l, u, iters=5000)
    gap = (exact - bound) / (1 + exact.abs())
    assert torch.all(gap <= 1e-4), gap


def test_infinite_rhs_rows_are_ignored():
    c, c0, G, h, l, u = _random_lp(3, B=2, m=6, n=4)
    exact = _exact(c, c0, G[:, :4], h[:, :4], l, u)
    h_inf = h.clone()
    h_inf[:, 4:] = float('inf')
    bound, y = lp_bound_pdhg(c, c0, G, h_inf, l, u, iters=5000)
    assert torch.isfinite(bound).all()
    assert torch.all(y[:, 4:] == 0)
    assert torch.all(bound <= exact + 1e-7 * (1 + exact.abs()))
    assert torch.all((exact - bound) / (1 + exact.abs()) <= 1e-4)


def test_chunked_batch_matches_unchunked():
    c, c0, G, h, l, u = _random_lp(5, B=5, m=7, n=5)
    full, y_full = lp_bound_pdhg(c, c0, G, h, l, u, iters=300)
    chunked, y_chunk = lp_bound_pdhg(c, c0, G, h, l, u, iters=300, max_elems=2 * 7 * 5)
    assert torch.allclose(full, chunked) and torch.allclose(y_full, y_chunk)


def test_exact_refinement_reaches_the_lp_and_stays_valid():
    from nlp_solver.dual_ascent import lp_bound
    c, c0, G, h, l, u = _random_lp(13, B=6, m=12, n=8)
    exact = _exact(c, c0, G, h, l, u)
    bound, y = lp_bound(c, c0, G, h, l, u, pdhg_iters=5, exact_max=3)
    assert torch.all(bound <= exact + 1e-7 * (1 + exact.abs()))
    assert torch.all(y >= 0)
    refined = (exact - bound).abs() <= 1e-6 * (1 + exact.abs())
    assert int(refined.sum()) >= 3


def test_parallel_exact_refinement_equals_sequential_solves():
    # The exact HiGHS re-solves run in a thread pool; every returned multiplier must be
    # the one the same linprog call gives when run alone.
    from nlp_solver.dual_ascent import lp_bound, lp_bound_pdhg, _lp_dual_value
    c, c0, G, h, l, u = _random_lp(21, B=12, m=15, n=9)
    bound, y = lp_bound(c, c0, G, h, l, u, pdhg_iters=5, exact_max=10)
    b0, _ = lp_bound_pdhg(c, c0, G, h, l, u, iters=5)
    for b in torch.argsort(b0)[:10].tolist():
        r = linprog(c[b].numpy(), A_ub=G[b].numpy(), b_ub=h[b].numpy(),
                    bounds=list(zip(l[b].numpy(), u[b].numpy())), method='highs')
        y_ref = torch.from_numpy(np.maximum(-r.ineqlin.marginals, 0.0))
        d_ref = _lp_dual_value(c[b:b + 1], c0[b:b + 1], G[b:b + 1], h[b:b + 1],
                               l[b:b + 1], u[b:b + 1], y_ref.unsqueeze(0))[0]
        if d_ref > b0[b]:
            assert torch.equal(y[b], y_ref)
            assert bound[b] == d_ref


def _zero_objective_lp():
    """A feasible one-row LP whose zero multiplier already gives the optimum."""
    c = torch.tensor([[0.0]], dtype=torch.float64)
    c0 = torch.tensor([0.0], dtype=torch.float64)
    G = torch.tensor([[[1.0]]], dtype=torch.float64)
    h = torch.tensor([[1.0]], dtype=torch.float64)
    l = torch.tensor([[0.0]], dtype=torch.float64)
    u = torch.tensor([[1.0]], dtype=torch.float64)
    return c, c0, G, h, l, u


@pytest.mark.parametrize('iters', [0, 50])
def test_pdhg_warm_start_cannot_worsen_zero_multiplier_bound(iters):
    c, c0, G, h, l, u = _zero_objective_lp()
    baseline, _ = lp_bound_pdhg(c, c0, G, h, l, u, iters=iters)
    bound, y = lp_bound_pdhg(
        c, c0, G, h, l, u, iters=iters,
        y0=torch.tensor([[100.0]], dtype=torch.float64))

    assert torch.isfinite(bound).all()
    assert torch.all(bound >= baseline)
    assert torch.equal(y, torch.zeros_like(y))


@pytest.mark.parametrize('bad_y0', [float('nan'), float('inf'), -float('inf')])
def test_pdhg_sanitizes_nonfinite_warm_start(bad_y0):
    c, c0, G, h, l, u = _zero_objective_lp()
    bound, y = lp_bound_pdhg(
        c, c0, G, h, l, u, iters=10,
        y0=torch.tensor([[bad_y0]], dtype=torch.float64))

    assert torch.isfinite(bound).all()
    assert torch.isfinite(y).all()
    assert torch.equal(y, torch.zeros_like(y))
    assert torch.allclose(bound, torch.zeros_like(bound), rtol=0, atol=1e-300)     # 0 less the underflow term


def test_pdhg_keeps_only_better_warm_bounds_per_domain():
    c = torch.tensor([[-1.0], [0.0], [-1.0]], dtype=torch.float64)
    c0 = torch.zeros(3, dtype=torch.float64)
    G = torch.tensor([[[1.0], [float('nan')]]], dtype=torch.float64).repeat(3, 1, 1)
    h = torch.tensor([[0.5, 1.0], [1.0, float('inf')], [0.5, 1.0]], dtype=torch.float64)
    l = torch.zeros(3, 1, dtype=torch.float64)
    u = torch.ones(3, 1, dtype=torch.float64)
    y0 = torch.tensor([[1.0, 9.0], [2.0, float('nan')], [-1.0, float('inf')]])

    bound, y = lp_bound_pdhg(c, c0, G, h, l, u, iters=0, y0=y0)

    assert torch.allclose(bound, torch.tensor([-0.5, 0.0, -1.0], dtype=torch.float64))
    assert torch.equal(y, torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64))
