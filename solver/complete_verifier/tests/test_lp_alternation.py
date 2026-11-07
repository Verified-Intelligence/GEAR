"""The root primal for bilinear structure (nlp_solver.lp_alternation): block LPs with the elastic phase, the joint
trust-region step, the verifier gate, eligibility, the root merge, and the flag-off root path."""
import os
import sys
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nlp_solver import lp_alternation as la  # noqa: E402
from nlp_solver import ipopt_solver  # noqa: E402

pytestmark = pytest.mark.skipif(not ipopt_solver.CYIPOPT_AVAILABLE, reason='cyipopt not available')


class Haverly(nn.Module):
    """Haverly's pooling problem 1 with a minimum demand on product Y. x = (fA, fB, pX, pY, cX, cY, q): crude A (3 %
    sulfur) and B (1 %) into the pool of quality q, pool flows pX, pY and crude C (2 %) cX, cY to the products X (<= 2.5 %,
    <= 100) and Y (<= 1.5 %, 50 ... 200). Optimum -400 (fB = pY = cY = 100, q = 1). At q = 3 no flow can make Y: the
    first conditional LP of the flows is infeasible."""
    def forward(self, x):
        fA, fB, pX, pY, cX, cY, q = (x[:, k:k + 1] for k in range(7))
        obj = 6 * fA + 16 * fB + 10 * (cX + cY) - 9 * (pX + cX) - 15 * (pY + cY)
        bal = fA + fB - pX - pY
        qual = q * (pX + pY) - 3 * fA - fB
        return torch.cat([obj, bal, -bal, qual, -qual, q * pX + 2 * cX - 2.5 * (pX + cX),
                          q * pY + 2 * cY - 1.5 * (pY + cY), pX + cX - 100, pY + cY - 200, 50 - (pY + cY)], 1)


LO = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
HI = [300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 3.0]


class Transcendental(nn.Module):
    def forward(self, x):
        return torch.cat([x[:, :1] * x[:, 1:2], torch.exp(x[:, :1]) - 2.0], 1)


def state_of(model, lo, hi, **config):
    lo = torch.tensor(lo, dtype=torch.float64)
    hi = torch.tensor(hi, dtype=torch.float64)
    with torch.no_grad():
        K = model.double()(lo[None]).shape[1]
    return SimpleNamespace(net=SimpleNamespace(model_ori=model.double()), dm_l=lo[None], dm_u=hi[None],
                           c=torch.eye(K, dtype=torch.float64)[None], rhs=torch.zeros(1, K, dtype=torch.float64),
                           or_spec_size=None, base_input_linear_constraints=None,
                           nlp_config=dict({'lp_alternation': True, 'constr_viol_tol': 1e-6,
                                            'ipopt_constr_viol_tol': 1e-6}, **config),
                           primal_solution=None, best_primal_x=None)


def test_eligibility_is_structural():
    assert la.eligible(state_of(Haverly(), LO, HI))
    assert not la.eligible(state_of(Haverly(), LO, HI, lp_alternation=False))
    assert not la.eligible(state_of(Transcendental(), [0.0, 0.0], [1.0, 1.0]))
    space = la._setup(state_of(Haverly(), LO, HI))[2].space
    assert space.bipartite and space.colours == 2
    assert space.colour[6] != space.colour[2] and space.colour[2] == space.colour[3]   # q against the pool flows
    assert all(space.colour[k] == -1 for k in (0, 1, 4, 5))                          # product-free: every block


def test_block_lps_run_after_another_highs_user_in_the_process():
    """HiGHS's thread scheduler is global: a solve that sets a thread count other than the first solve's fails, which
    turned every LP of the lane into "no point" once another test (or a root stage) had run HiGHS first."""
    import highspy
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = 1, 0
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = np.array([1.0]), np.array([0.0]), np.array([1.0])
    h.passModel(lp)
    h.run()
    import scipy.sparse as sp
    res = la._lp(np.array([1.0, 1.0]), sp.csr_matrix([[1.0, 1.0]]), np.array([1.0]), np.array([np.inf]),
                 np.zeros(2), np.ones(2), None, time.time() + 10, 1e-9)
    assert res is not None and res[0].sum() == pytest.approx(1.0)


def test_infeasible_first_lp_is_repaired_by_the_elastic_phase():
    alt = la._setup(state_of(Haverly(), LO, HI))[2]
    seed = np.array([0, 0, 0, 0, 0, 0, 3.0])
    flows = int(alt.space.colour[2])
    deadline = time.time() + 30
    free = alt.space.block(flows)
    A, const, cost, _, idx = alt.affine(seed, free)
    # the plain conditional LP of the flows at q = 3 has no feasible point ...
    assert la._lp(cost, A, alt.cl - const, alt.cu - const, alt.lb[idx], alt.ub[idx], None, deadline, 1e-9) is None
    # ... the elastic phase moves anyway, and the alternation ends feasible
    x, feasible = alt.block_step(seed, flows, False, deadline)
    assert not feasible and np.any(x != seed)
    found = []
    last = alt.alternate(seed, flows, deadline, found.append)
    assert last is not None and found
    assert np.max(alt.sparse.raw_rows(last)) <= 1e-6


def test_trust_region_steps_keep_feasibility_and_never_increase_the_objective():
    alt = la._setup(state_of(Haverly(), LO, HI))[2]
    deadline = time.time() + 30
    x = alt.alternate(np.array([0, 0, 0, 0, 0, 0, 3.0]), int(alt.space.colour[2]), deadline, lambda z: None)
    f = alt.sparse.objective(x)
    seen = []

    def offer(z):
        if np.max(alt.sparse.raw_rows(z)) <= 1e-6:
            seen.append(alt.sparse.objective(z))
            return seen[-1]
        return None
    y, g = alt.trust_region(x, f, deadline, offer)
    assert g <= f + 1e-9 and np.max(alt.sparse.raw_rows(y)) <= 1e-6


def test_run_commits_only_verified_points_and_finds_the_optimum():
    state = state_of(Haverly(), LO, HI)
    la.run(state, time.time() + 6.0)
    assert state.primal_solution is not None
    assert float(state.primal_solution[0]) == pytest.approx(-400.0, abs=1e-3)
    from nlp_solver.statistics import verify_counterexample
    C = state.c[0]
    ok, obj, _ = verify_counterexample(state.net.model_ori, state.best_primal_x, C[:1], torch.zeros(1), C[1:],
                                       torch.zeros(C.shape[0] - 1), tol=1e-6)
    assert ok and obj == pytest.approx(float(state.primal_solution[0]))


def test_a_point_the_verifier_rejects_never_becomes_the_incumbent():
    state = state_of(Haverly(), LO, HI)
    with mock.patch('nlp_solver.statistics.verify_counterexample', return_value=(False, -1e9, 1.0)):
        la.run(state, time.time() + 2.0)
    assert state.primal_solution is None and state.best_primal_x is None


def test_merge_keeps_the_better_root_point():
    state = state_of(Haverly(), LO, HI)
    lane = la.start(state, time.time() + 5.0)
    time.sleep(5.0)
    worse = SimpleNamespace(best_method='alm', best_obj=-10.0, best_x=None)
    lane.merge(worse)
    assert worse.best_method == 'lp_alternation' and worse.best_obj == pytest.approx(-400.0, abs=1e-3)
    better = SimpleNamespace(best_method='ipopt', best_obj=-1e4, best_x='kept')
    state.primal_solution = torch.tensor([-400.0])
    lane.merge(better)
    assert better.best_method == 'ipopt' and better.best_x == 'kept'


def test_the_lane_ends_when_the_root_collects_it():
    """the lane never lengthens the root: merge() stops the child well before its deadline and keeps its points"""
    state = state_of(Haverly(), LO, HI)
    t0 = time.time()
    lane = la.start(state, t0 + 120.0)
    time.sleep(6.0)
    comparison = SimpleNamespace(best_method='none', best_obj=None, best_x=None)
    lane.merge(comparison)
    assert time.time() - t0 < 30.0
    assert comparison.best_method == 'lp_alternation' and comparison.best_obj == pytest.approx(-400.0, abs=1e-3)


def test_root_without_the_flag_starts_no_lane():
    from input_split import nlp_in_bab
    comparison = SimpleNamespace(best_method='ipopt', best_obj=1.0, best_x=None, local_method='IPOPT',
                                 local_verified=True, local_obj=1.0, local_time=0.0, alm_verified=False,
                                 alm_obj=None, alm_time=0.0)
    config = {'enabled': True, 'enable_alm': True, 'enable_local_solver': True, 'timeout': 1.0}
    net = SimpleNamespace(model_ori=Haverly().double())
    dm = torch.zeros(1, 7, dtype=torch.float64)
    with mock.patch('nlp_solver.compare_solvers', return_value=comparison), \
            mock.patch.object(la, 'start', side_effect=AssertionError('lane started')):
        res = nlp_in_bab.run_root_nlp_and_ipopt(net, dm, dm + 1, torch.eye(10)[None], torch.zeros(1, 10), None,
                                                None, config, False)
    assert res.nlp_obj == 1.0


def test_child_publishes_verified_incumbent_before_join():
    import threading
    state = state_of(Haverly(), LO, HI)
    published, release = threading.Event(), threading.Event()
    original = la._publish

    def publish(*args, **kwargs):
        original(*args, **kwargs)
        if state.primal_solution is not None and not published.is_set():
            published.set()
            release.wait()  # inspect the first point, not a later optimum

    with mock.patch.object(la, '_publish', publish):
        lane = la.start(state, time.time() + 120.0)
        try:
            assert published.wait(15.0)
            assert lane.thread.is_alive()  # publication precedes collection
            assert state.primal_solution_found
            assert state.primal_solution.device == state.dm_l.device
            # The first verified point may be -350 while the worker is still
            # searching toward -400. Publication does not promise optimality.
            point = state.best_primal_x.reshape_as(state.dm_l)
            value = state.net.model_ori(point)[0]
            assert torch.isfinite(value).all()
            assert float(value[0]) == pytest.approx(float(state.primal_solution[0]))
            tol = state.nlp_config['constr_viol_tol']
            assert bool((value[1:] <= tol).all())
            assert bool((point >= state.dm_l - tol).all())
            assert bool((point <= state.dm_u + tol).all())
        finally:
            release.set()
            lane.join()
    assert not lane.thread.is_alive() and lane.error is None


def test_child_never_publishes_a_rejected_point():
    import threading
    state = state_of(Haverly(), LO, HI)
    offered = threading.Event()

    def reject(*args, **kwargs):
        offered.set()
        return False, -1e9, 1.0

    with mock.patch('nlp_solver.statistics.verify_counterexample', side_effect=reject):
        lane = la.start(state, time.time() + 120.0)
        try:
            assert offered.wait(15.0)
            assert state.primal_solution is None and state.best_primal_x is None
        finally:
            lane.join()
    assert lane.error is None


def test_native_root_operation_closes_on_lane_point_and_reaps_on_error():
    from input_split import batch_branch_and_bound as bbb
    from tests.test_unified_root import _root
    op = next(op for op in bbb._ROOT_OPERATIONS if op.name == 'lp_alternation')
    assert bbb._ROOT_OPERATIONS.index(op) + 1 == next(
        i for i, item in enumerate(bbb._ROOT_OPERATIONS) if item.name == 'nlp')
    root = _root(True, deadline=time.time() + 30, D=-400.0)
    state = state_of(Haverly(), LO, HI)
    for key in ('net', 'dm_l', 'dm_u', 'c', 'rhs', 'or_spec_size', 'base_input_linear_constraints'):
        setattr(root, key, getattr(state, key))
    root.nlp_config.update(state.nlp_config, enabled=True)
    root.solving_mode, root.initial_verified, root.ret = True, False, {}
    root.rowpool = None
    assert op.eligible(root)
    root.nlp_config['lp_alternation'] = False
    assert not op.eligible(root)
    root.nlp_config['lp_alternation'] = True

    seed = 0.5 * (root.dm_l + root.dm_u)

    def nlp(*args, **kwargs):
        from input_split.nlp_in_bab import RootNLPResult
        assert kwargs['local_extra_x0'] is seed
        until = time.time() + 15
        while root.primal_solution is None and time.time() < until:
            time.sleep(0.05)
        assert root.primal_solution_found
        return RootNLPResult(dm_l=root.dm_l, dm_u=root.dm_u)

    later = mock.Mock(side_effect=AssertionError('gap did not close'))
    with mock.patch('input_split.nlp_in_bab.compute_root_dual_warmstart', return_value=None) as dual, \
            mock.patch('input_split.nlp_in_bab.compute_root_lp_point', return_value=seed) as lp, \
            mock.patch.object(bbb, 'run_root_nlp_and_ipopt', side_effect=nlp), \
            mock.patch.object(bbb, 'run_root_diagnostics_and_budget'), \
            mock.patch.object(bbb, '_ROOT_OPERATIONS', (op, bbb.RootOp('nlp', bbb._root_nlp),
                                                       bbb.RootOp('later', later))), \
            mock.patch.object(bbb, '_finish_with_lifted_dual', return_value='closed'):
        assert bbb._run_root(root) == 'closed'
        assert lp.call_count == dual.call_count == 1
    assert not root.primal_lane.thread.is_alive()
    assert float(root.primal_solution[0]) == pytest.approx(-400.0, abs=1e-3)

    def fail(root, deadline):
        raise RuntimeError('NLP failed')

    root.primal_solution, root.primal_solution_found = None, False
    with mock.patch('input_split.nlp_in_bab.compute_root_dual_warmstart', return_value=None), \
            mock.patch('input_split.nlp_in_bab.compute_root_lp_point', return_value=None), \
            mock.patch.object(bbb, '_ROOT_OPERATIONS', (op, bbb.RootOp('nlp', fail))):
        with pytest.raises(RuntimeError, match='NLP failed'):
            bbb._run_root(root)
    assert not root.primal_lane.thread.is_alive()
