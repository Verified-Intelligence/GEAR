"""Continuation admission uses certified/primal progress and charged service."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from input_split.lifted_bound import _continuation_pays, run_lifted_bab
from input_split.batch_branch_and_bound import OpLedger


@pytest.mark.parametrize('progress,competing,expected', [
    (None, (10., 1.), True),
    ((0., 4.), (0., 8.), False),
    ((2., 4.), (4., 8.), True),
    ((2., 4.), (5., 8.), False),
    ((2., 4.), (0., 0.), True),
])
def test_measured_continuation_rate(progress, competing, expected):
    assert _continuation_pays(progress, competing) is expected


def test_main_rate_excludes_noted_operations():
    ledger = OpLedger()
    timer = SimpleNamespace(time_sum={'bounding': 1., 'alm_attack': 2.})
    ledger.P, ledger.D = 10., 0.
    ledger.start_bab(timer)
    timer.time_sum = {'bounding': 4., 'alm_attack': 3.}
    ledger.note('lifted_bab', 100., dP=1.)
    assert ledger.competing_progress(timer, 8., 1.) == (2., 4.)
    ledger.iteration(timer, 8, 8., 1.)
    timer.time_sum['bounding'] += 2.
    assert ledger.competing_progress(timer, 8., 2.) == (3., 6.)


@pytest.mark.parametrize('gain', [0., 1.])
def test_production_continuation_observes_open_frontier(monkeypatch, gain):
    from nlp_solver.lifted_lp import LiftedLP
    import input_split.lifted_bound as module
    # A real LP object satisfies the public producer contract; only its tree
    # solve is replaced so elapsed-time budgets cannot change the test.
    lp = object.__new__(LiftedLP)
    lp.obj = object()
    lp.n = 1
    calls = []
    leaves = lambda D: (np.zeros((1, 1)), np.ones((1, 1)), np.array([D]), np.zeros((1, 1)))
    def tree(*a, **kw):
        calls.append(kw)
        D = gain if len(calls) >= 2 else 0.
        return D, 1, False, leaves(D)
    lp.branch_and_bound = tree
    monkeypatch.setattr(module, '_relaxes_problem', lambda *a: True)
    h = SimpleNamespace(lifted=lp)
    cfg = {'lifted_lp_bab_time': 30.}
    l, u = torch.zeros(1, 1), torch.ones(1, 1)
    assert run_lifted_bab(h, l, u, 10., cfg, 100.) is None
    assert run_lifted_bab(h, l, u, 9., cfg, 100., bab_lb=0.) is None
    assert h.lifted_bab_progress[0] == gain
    run_lifted_bab(h, l, u, 8., cfg, 100., bab_lb=0.)
    assert len(calls) == (3 if gain else 2)
