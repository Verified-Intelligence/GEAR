"""Exact LP fallbacks keep an infeasibility proof (A45): a box whose rows a checked Farkas certificate proves empty
gets +inf, and an LP that is feasible, even only on the box boundary, never does."""
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nlp_solver.dual_ascent import lp_bound  # noqa: E402
from nlp_solver.lifted_lp import LiftedLP  # noqa: E402


def _batch(rhs):
    """min x1 - x2 over [0, 1]^2 with x1 + x2 <= rhs[b] and -x1 <= 1 (B = len(rhs))."""
    B = len(rhs)
    G = torch.tensor([[1., 1.], [-1., 0.]], dtype=torch.float64).expand(B, 2, 2).clone()
    h = torch.stack([torch.tensor(rhs, dtype=torch.float64), torch.ones(B, dtype=torch.float64)], 1)
    c = torch.tensor([1., -1.], dtype=torch.float64).expand(B, 2).clone()
    return c, torch.zeros(B, dtype=torch.float64), G, h, torch.zeros(B, 2, dtype=torch.float64), \
        torch.ones(B, 2, dtype=torch.float64)


def _one(rhs):
    c, c0, G, h, l, u = _batch([rhs])
    return c[0].numpy(), 0.0, sp.csr_matrix(G[0].numpy()), h[0].numpy(), l[0].numpy(), u[0].numpy()


def test_solve_certified_proves_an_empty_box_only_on_request():
    data = _one(-0.5)
    assert LiftedLP.solve_certified(*data) is None
    assert LiftedLP.solve_certified(*data, prove_empty=True) == np.inf
    assert LiftedLP.solve_certified(*data, return_x=True, prove_empty=True) == (np.inf, None)
    assert LiftedLP.solve_certified(*data, presolve=False, prove_empty=True) == np.inf


def test_solve_certified_never_proves_a_feasible_box_empty():
    for rhs in (0.0, 1e-12, 0.3):           # feasible only at the corner (0, 0) for rhs = 0
        bound = LiftedLP.solve_certified(*_one(rhs), prove_empty=True)
        assert bound is not None and np.isfinite(bound) and bound <= -min(rhs, 1.0) + 1e-9


def test_lp_bound_exact_path_gives_inf_only_for_the_empty_boxes():
    c, c0, G, h, l, u = _batch([-0.5, 0.0, 0.3, -1e-3])
    bound, y = lp_bound(c, c0, G, h, l, u, pdhg_iters=1, exact_max=4)
    assert torch.isinf(bound[0]) and bound[0] > 0 and torch.isinf(bound[3]) and bound[3] > 0
    assert torch.isfinite(bound[1:3]).all()
    assert bound[1] <= 1e-9 and bound[2] <= -0.3 + 1e-9
    assert torch.all(y >= 0)


def test_solve_node_still_proves_an_empty_box():
    assert LiftedLP.solve_node(*_one(-0.5)) == (np.inf, None)
    bound, point = LiftedLP.solve_node(*_one(0.3))
    assert np.isfinite(bound) and point is not None


def test_lp_bound_keeps_the_pdhg_bounds_when_the_deadline_passes_before_a_solve(monkeypatch):
    # The deadline check before the exact solves passes; it has passed when each solve starts.
    import nlp_solver.dual_ascent as da
    c, c0, G, h, l, u = _batch([0.3, 0.5])
    stub = (torch.tensor([-1.25, -0.75], dtype=torch.float64), torch.zeros(2, 2, dtype=torch.float64))
    monkeypatch.setattr(da, 'lp_bound_pdhg', lambda *a, **k: stub)
    calls = iter(range(10**6))
    monkeypatch.setattr(da.time, 'time', lambda: 0.0 if next(calls) == 0 else 2.0)
    bound, y = lp_bound(c, c0, G, h, l, u, pdhg_iters=1, exact_max=2, deadline=1.0)
    # No ValueError (the skipped solve returned 3 values); the PDHG bounds survive and stay valid
    # (LP optima: -0.3 and -0.5).
    assert torch.all(bound >= stub[0]) and torch.all(bound <= torch.tensor([-0.3, -0.5], dtype=torch.float64) + 1e-9)
