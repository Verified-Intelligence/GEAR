"""Root OBBT buys fresh proposals by observed gain/work under budget pressure."""
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch

from nlp_solver import lifted_lp as ll


def test_refresh_admission_compares_gain_per_lp_and_reserves_inputs():
    # First measurement and spare-budget work stay eligible.
    assert ll._root_obbt_refresh(None, [10., 2.], 0., 1., 1.)
    assert ll._root_obbt_refresh(0., [10., 2.], 2., 1., 1.)
    assert ll._root_obbt_refresh(0., [10., 2.], 0., 0., 1.)
    # With competing work pending, zero ties and slower positive returns lose.
    assert not ll._root_obbt_refresh(0., [0., 2.], 1., 1., 1.)
    assert not ll._root_obbt_refresh(.25, [1., 2.], 1., 1., 1.)
    assert ll._root_obbt_refresh(.5, [1., 2.], 1., 1., 1.)


def test_low_return_auxiliary_replays_duals_on_changed_rows(monkeypatch):
    scopes, events = [], []
    lp = SimpleNamespace(n=2, nb=2, branch_K=1, N=3, qvars=torch.tensor([0, 1]),
                         transfer_aux=True, rlt=None,
                         certificate_registry=SimpleNamespace(add_auxiliary=lambda *a, **k: scopes.append(a)))
    bounds = [.2, .8]
    def sparse(l, u, *a):
        G = sp.csr_matrix([[0., 0., 1.], [0., 0., -1.]])
        return np.zeros(3), 0., G, np.array([bounds[1], -bounds[0]]), np.r_[l, 0.], np.r_[u, 1.]
    lp.sparse_lp = sparse
    def solve(c, *a, **kw):
        events.append(('lp', int(np.argmax(abs(c)))))
        return ll.LiftedLP.solve_certified(c, *a, **kw)
    lp.solve_certified = solve
    # Expose LP-derived bounds rather than letting free propagation supply them.
    monkeypatch.setattr(ll, '_fbbt', lambda G, h, l, u: (l.copy(), u.copy(), None))
    lo, hi = torch.zeros(1, 2, dtype=torch.float64), torch.ones(1, 2, dtype=torch.float64)
    ll.LiftedLP.obbt(lp, lo, hi, torch.zeros(1, 1), rounds=1, verbose=False)
    assert [j for kind, j in events].count(2) == 2
    assert lp._obbt_target_duals[2][0] is not None
    bounds[:] = [.3, .7]
    events.clear()
    monkeypatch.setattr(ll, '_root_obbt_refresh', lambda rate, *a: rate is None)
    ll.LiftedLP.obbt(lp, lo, hi, torch.zeros(1, 1), rounds=1, verbose=False)
    assert all(j < 2 for kind, j in events)  # no fresh auxiliary LP
    # Re-evaluation uses the new h; copying the old scalar would leave [.2,.8].
    assert .3 - 1e-12 <= scopes[-1][2][2] <= .3
    assert .7 <= scopes[-1][3][2] <= .7 + 1e-12
    # A changed row count invalidates the proposal's shape and forces measurement.
    saved = lp._obbt_target_duals[2]
    lp._obbt_target_duals[2] = [np.r_[y, 0.] for y in saved]
    events.clear()
    ll.LiftedLP.obbt(lp, lo, hi, torch.zeros(1, 1), rounds=1, verbose=False)
    assert [j for kind, j in events].count(2) == 2


def test_cold_class_measurements_are_deferred_by_return_not_population(monkeypatch):
    import pytest
    for aux_columns in (3, 100):
        n = 2 + aux_columns
        lp = SimpleNamespace(n=2, nb=2, branch_K=aux_columns, N=n,
                             qvars=torch.tensor([0, 1]), transfer_aux=True, rlt=None,
                             certificate_registry=SimpleNamespace(add_auxiliary=lambda *a, **k: None))
        G = np.zeros((2 * aux_columns, n))
        for k in range(aux_columns):
            G[2*k, k+2], G[2*k+1, k+2] = 1., -1.
        def sparse(l, u, *a):
            return (np.zeros(n), 0., sp.csr_matrix(G), np.tile([.8, -.2], aux_columns),
                    np.r_[l, np.zeros(aux_columns)], np.r_[u, np.ones(aux_columns)])
        lp.sparse_lp = sparse
        calls = []
        def solve(c, *a, **kw):
            calls.append(int(np.argmax(abs(c))))
            return ll.LiftedLP.solve_certified(c, *a, **kw)
        lp.solve_certified = solve
        monkeypatch.setattr(ll, '_fbbt', lambda G, h, l, u: (l.copy(), u.copy(), None))
        # Emulate a budget under pressure with a slower measured auxiliary rate.
        monkeypatch.setattr(ll, '_root_obbt_refresh', lambda rate, *a: rate is None)
        lo, hi = torch.zeros(1, 2, dtype=torch.float64), torch.ones(1, 2, dtype=torch.float64)
        ll.LiftedLP.obbt(lp, lo, hi, torch.zeros(1, 1), rounds=1, verbose=False)
        assert [j for j in calls if j >= 2] == [2, 2]
        # Next pass samples the next unseen target, independently of population.
        calls.clear()
        ll.LiftedLP.obbt(lp, lo, hi, torch.zeros(1, 1), rounds=1, verbose=False)
        assert [j for j in calls if j >= 2] == [3, 3]
        # A new pass with spare budget admits every remaining target.
        calls.clear()
        monkeypatch.setattr(ll, '_root_obbt_refresh', lambda *a: True)
        ll.LiftedLP.obbt(lp, lo, hi, torch.zeros(1, 1), rounds=1, verbose=False)
        assert len([j for j in calls if j >= 2]) == 2 * aux_columns


def test_reserved_input_work_continues_while_paying(monkeypatch):
    # A prior phase deferred auxiliaries to fund original-variable passes.
    # Small positive LP gains must not release that reservation prematurely.
    for active, gain, expected in ((False, .005, 1), (True, .005, 3), (True, 0., 1)):
        lp = SimpleNamespace(n=1, nb=1, branch_K=1, N=2, qvars=torch.tensor([0]),
                             transfer_aux=True, rlt=None, _obbt_adaptive_work=active,
                             certificate_registry=SimpleNamespace(add_auxiliary=lambda *a, **k: None))
        def sparse(l, u, *a):
            return np.zeros(2), 0., sp.csr_matrix((1, 2)), np.ones(1), np.r_[l, 0.], np.r_[u, 1.]
        lp.sparse_lp = sparse
        calls = []
        def solve(c, c0, G, h, l, u, **kw):
            j = int(np.argmax(abs(c)))
            if j == 0 and c[j] > 0:
                calls.append(j)
                return l[j] + gain * (u[j] - l[j])
            return -u[j] if c[j] < 0 else l[j]
        lp.solve_certified = solve
        monkeypatch.setattr(ll, '_fbbt', lambda G, h, l, u: (l.copy(), u.copy(), None))
        lo, hi = torch.zeros(1, 1, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64)
        ll.LiftedLP.obbt(lp, lo, hi, torch.zeros(1, 1), rounds=3, verbose=False)
        assert len(calls) == expected
