"""An early incumbent must not strand unfinished root bound tightening."""
from types import SimpleNamespace

import pytest
import torch

from input_split.lifted_bound import RootLifted, reobbt_on_improvement
from nlp_solver.lifted_lp import LiftedLP


def problem():
    # min x+y, xy >= 1, x,y in [.1,3]. (1,1) is feasible and optimal.
    dt = torch.float64
    obj = (0., torch.ones(2, dtype=dt), torch.zeros(1, dtype=dt))
    rows = (torch.zeros(1, dtype=dt), torch.zeros(1, 2, dtype=dt),
            -torch.ones(1, 1, dtype=dt))
    lp = LiftedLP(2, torch.tensor([0]), torch.tensor([1]), obj,
                  tuple(abs(v) for v in obj), rows, torch.tensor([1]),
                  tuple(abs(v) for v in rows), 'cpu')
    lo, hi = torch.full((1, 2), .1, dtype=dt), torch.full((1, 2), 3., dtype=dt)
    rhs = torch.tensor([[2.05, -1.]], dtype=dt)
    cfg = dict(lifted_lp_obbt_time=10., lifted_lp_obbt_rounds=1)
    history = SimpleNamespace(lifted=lp)
    setup = RootLifted(history, lp, lo, hi, rhs, rhs[:, 1:], None, cfg, True)
    setup.add_rlt()
    return lp, lo, hi, rhs, cfg, history, setup


def test_same_incumbent_continues_paying_root_work_and_preserves_optimum():
    lp, lo, hi, rhs, cfg, history, setup = problem()
    setup.obbt()
    assert history.lifted_obbt_pending
    first_l, first_u = history.lifted_box
    assert reobbt_on_improvement(history, rhs[:, 0], cfg, 100.)
    next_l, next_u = history.lifted_box
    assert (next_l >= first_l).all() and (next_u <= first_u).all()
    assert ((next_u-next_l) < .98*(first_u-first_l)).any()
    optimum = torch.ones_like(lo)
    assert ((optimum >= next_l) & (optimum <= next_u)).all()
    # The unchanged cutoff cannot defeat the existing total work limit.
    history.lifted_reobbt_time = 20.
    assert not reobbt_on_improvement(history, rhs[:, 0], cfg, 100.)


def test_converged_pass_does_not_repeat_at_same_cutoff():
    lp, lo, hi, rhs, cfg, history, setup = problem()
    setup.dm_l = setup.dm_u = setup.l0 = setup.u0 = setup.lo = setup.hi = torch.ones_like(lo)
    setup.obbt()
    assert not history.lifted_obbt_pending
    assert not reobbt_on_improvement(history, rhs[:, 0], cfg, 100.)


@pytest.mark.parametrize('gain', [False, True])
def test_time_limited_pass_needs_measured_gain(monkeypatch, gain):
    from nlp_solver import lifted_lp as ll
    lp, lo, hi, rhs, cfg, history, setup = problem()
    clock = [0.]
    monkeypatch.setattr(ll.time, 'time', lambda: clock[0])
    original = lp.solve_certified
    def solve(c, *a, **kw):
        value = original(c, *a, **kw)
        clock[0] += 1.
        if not gain:
            j = int(c.abs().argmax()) if torch.is_tensor(c) else int(abs(c).argmax())
            return float(lo[0, j]) if c[j] > 0 else -float(hi[0, j])
        return value
    monkeypatch.setattr(lp, 'solve_certified', solve)
    lp.obbt(lo, hi, rhs[:, 1:], cutoff=2.05, rounds=5, time_budget=.5)
    assert lp.obbt_pending == gain
