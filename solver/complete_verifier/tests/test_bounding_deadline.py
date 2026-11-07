"""Regression checks for the cooperative bounding deadline prototype."""
import time
import torch
from auto_LiRPA import concretize_func as cf
from nlp_solver.dual_ascent import lp_bound, lp_bound_pdhg
from utils import Timer


def test_expired_pdhg_keeps_a_certified_box_bound_and_finite_point():
    c = torch.tensor([[-1., 2.]], dtype=torch.float64)
    l, u = torch.zeros_like(c), torch.ones_like(c)
    G = torch.tensor([[[1., 1.]]], dtype=torch.float64)
    h, c0 = torch.ones(1, 1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
    bound, y, x = lp_bound_pdhg(c, c0, G, h, l, u, iters=100000,
                                return_primal=True, deadline=time.time()-1)
    assert -1.000001 < bound.item() <= -1.
    assert torch.equal(y, torch.zeros_like(y))
    assert torch.isfinite(x).all() and ((x >= l) & (x <= u)).all()


def test_expired_exact_lp_does_not_enter_highs(monkeypatch):
    import scipy.optimize
    def forbidden(*a, **kw):
        raise AssertionError('HiGHS started after the deadline')
    monkeypatch.setattr(scipy.optimize, 'linprog', forbidden)
    c = torch.tensor([[-1.]], dtype=torch.float64)
    bound, _ = lp_bound(c, torch.zeros(1, dtype=c.dtype), torch.ones(1, 1, 1, dtype=c.dtype),
                        torch.ones(1, 1, dtype=c.dtype), torch.zeros_like(c), torch.ones_like(c),
                        exact_max=1, deadline=time.time()-1)
    assert bound.item() <= -1.


def test_expired_concretization_retains_enclosed_box_bound(monkeypatch):
    l = torch.zeros(1, 2, dtype=torch.float64)
    u = torch.ones_like(l)
    c = torch.tensor([[[-1., 2.]]], dtype=l.dtype)
    constraints, batches = cf.sort_out_constr_batches(
        l, u, (torch.ones(1, 1, 2, dtype=l.dtype), -torch.ones(1, 1, dtype=l.dtype)))
    def forbidden(*a, **kw):
        raise AssertionError('dual sweep started after the deadline')
    monkeypatch.setattr(cf, '_solve_dual_var', forbidden)
    timer = Timer()
    timer.deadline = time.time()-1
    bound = cf.constraints_solving(l, u, c, constraints, -1.,
                                  sorted_out_batches=batches, timer=timer, use_grad=False)
    assert -1.000001 < bound.item() <= -1.


def test_non_rlt_exact_solve_receives_remaining_budget(monkeypatch):
    from tests.test_lifted_lp import _setup
    from nlp_solver.lifted_lp import build_lifted_lp
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    assert lp.rlt is None
    limits = []
    def solve(*args, **kwargs):
        limits.append(kwargs['time_limit'])
        return None, None
    monkeypatch.setattr(lp, 'solve_certified', solve)
    lp.bound(l, u, rhs[:, 1:], pdhg_iters=0, exact_max=1,
             exact_time=100, deadline=time.time() + 0.25)
    assert len(limits) == 1 and 0 < limits[0] <= 0.25


def test_expired_warm_retry_does_not_start_cold_solve(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from nlp_solver import lifted_lp as ll
    clock = [0.0]
    calls = []
    def solve(*args):
        calls.append(1)
        clock[0] = 2.0
        return SimpleNamespace(status=1)
    monkeypatch.setattr(ll.time, 'time', lambda: clock[0])
    monkeypatch.setattr(ll, '_highs_solve', solve)
    result = ll._highs_proposal(None, None, None, None, None,
        {'presolve': True, 'time_limit': 1.0, 'warm': {'basis': np.ones(1)}})
    assert result.status == 1 and len(calls) == 1
