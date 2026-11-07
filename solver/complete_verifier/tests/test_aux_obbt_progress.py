"""Auxiliary transfer must not change the meaning of root/node progress."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nlp_solver import lifted_lp as ll
from tests.test_lifted_lp_rlt import make_lp


@pytest.mark.parametrize('transfer_aux', [False, True])
def test_zero_gain_does_not_buy_more_probes(monkeypatch, transfer_aux):
    lp, rh = make_lp([1, 1, 1], 1, [(0, 1), (1, 2)], [1, 1])
    lp.transfer_aux = transfer_aux
    point = np.r_[np.full(3, .5), np.zeros(2)]
    monkeypatch.setattr(lp, 'solve_node', lambda *a, **k: (1., point.copy()))
    calls = []
    def probe(c, *a, **k):
        calls.append(c.copy())
        return .2 if c.max() > 0 else -.8
    monkeypatch.setattr(lp, 'solve_certified', probe)
    progress = [0.] * 4
    result = lp.solve_bab_node(np.zeros(3), np.ones(3), rh[0].numpy(), None, None,
                               10., progress=progress, inherited=1.)
    assert len(calls) == 2  # one measured min/max probe, despite equal zero gain rates
    assert result[0] == 1. and progress[0] == progress[2] == 0
    assert progress[1] > 0 and not ll._probe_pays(progress)
    assert ll._probe_pays(progress, require_gain=False)  # diagnostic legacy comparison
    assert ll._probe_pays([0., 0., 0., 1.])  # the initial measurement remains allowed
    assert ll._probe_pays([2., 1., 4., 2.])  # positive gain at equal rates still pays
    assert not ll._probe_pays([2., 1., 5., 2.])  # a slower positive rate does not


@pytest.mark.parametrize('aux_columns', [1, 100])
@pytest.mark.parametrize('propagate', [False, True])
def test_root_progress_counts_requested_inputs_and_fbbt(monkeypatch, capsys, aux_columns, propagate):
    # Input 0 has 20% real progress per round; auxiliary columns cannot move.
    # Neither their number nor whether FBBT/LP provides the progress may stop it.
    n = aux_columns + 1
    scope = []
    lp = SimpleNamespace(n=1, nb=1, branch_K=aux_columns, N=n,
                         qvars=torch.tensor([0]), transfer_aux=True, rlt=None,
                         certificate_registry=SimpleNamespace(add_auxiliary=lambda *a, **k: scope.append(a)))
    def sparse(l, u, *a):
        import scipy.sparse as sp
        return np.zeros(n), 0., sp.csr_matrix((1, n)), np.ones(1), np.r_[l, np.zeros(aux_columns)], np.r_[u, np.ones(aux_columns)]
    lp.sparse_lp = sparse
    probe_counts = [0]
    def solve(c, c0, G, h, l, u, **kw):
        if c[0] > 0:
            probe_counts[0] += 1
            return l[0] if propagate else l[0] + .2 * (u[0] - l[0])
        return -u[np.argmax(abs(c))] if c.min() < 0 else l[np.argmax(c)]
    lp.solve_certified = solve
    seen = [False]
    def fbbt(G, h, l, u):
        # Once per round, before the input's LP; later aux solves do not shrink it.
        if propagate and not seen[0]:
            l[0] += .2 * (u[0] - l[0]); seen[0] = True
        return l, u, None
    # For FBBT, one first-round change is enough to require a second round.
    monkeypatch.setattr(ll, '_fbbt', fbbt)
    lo, hi = torch.zeros(1, 1, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64)
    nl, nu = ll.LiftedLP.obbt(lp, lo, hi, torch.zeros(1, 1), rounds=2, min_shrink=.1)
    assert probe_counts[0] == 2
    assert 'OBBT round 1' in capsys.readouterr().out
    assert scope and float(nl[0, 0]) >= .2 and float(nu[0, 0]) == 1.
