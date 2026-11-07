"""RLT soundness at relaxed equality boundaries, including large signed multipliers."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nlp_solver.lifted_lp import LiftedLP  # noqa: E402


def make_lp(g, b, pairs, q):
    n, K = len(g), len(pairs)
    a, bb = (torch.tensor(v, dtype=torch.long) for v in zip(*pairs))
    dt = torch.float64
    g = torch.as_tensor(g, dtype=dt)
    # Opposite variable coefficients, but constants need not be opposite.
    k = torch.tensor([1.25, 0.75], dtype=dt)
    rows = (k, torch.stack([g, -g]), torch.zeros(2, K, dtype=dt))
    obj = (0.0, torch.zeros(n, dtype=dt), torch.as_tensor(q, dtype=dt))
    mag = lambda rep: tuple(abs(v) for v in rep)
    lp = LiftedLP(n, a, bb, obj, mag(obj), rows, torch.tensor([1, 2]), mag(rows), 'cpu')
    rhs = torch.tensor([[b + 1.25, -b + 0.75]], dtype=dt)
    return lp, rhs


@pytest.mark.parametrize('seed', range(6))
@pytest.mark.parametrize('magnitude', [1.0, 1e5])
def test_random_relaxed_points_and_certified_minimum(seed, magnitude):
    rng = np.random.default_rng(seed)
    g = np.r_[rng.uniform(-1, 1, 4), 1.0]
    pairs = [(0, 1), (1, 2), (2, 3), (0, 0)]
    lp, rhs = make_lp(g, 0.0, pairs, rng.normal(size=len(pairs)))
    lp.add_rlt(rhs)
    lo = torch.tensor([[-magnitude] * 4 + [-5 * magnitude]], dtype=torch.float64)
    hi = -lo
    tol = 1e-5
    points = rng.uniform(-magnitude, magnitude, (2000, 5))
    noise = rng.uniform(-tol, tol, len(points))
    noise[:2] = [-tol, tol]
    points[:, 4] = -points[:, :4] @ g[:4] + noise
    z = np.c_[points, points[:, lp.a] * points[:, lp.b]]
    data = lp.sparse_lp(lo[0].numpy(), hi[0].numpy(), (rhs[0] + tol).numpy())
    c, c0, G, h, lz, uz = data
    assert np.all(G @ z.T <= h[:, None] + 2e-5)
    assert np.all(z >= lz) and np.all(z <= uz)
    for presolve in (True, False):
        D, _ = lp.solve_node(*data, presolve=presolve)
        assert np.isfinite(D)
        assert D <= (z @ c + c0).min() + 1e-7
    # At this scale a point at the tolerance boundary violates an unrelaxed RLT
    # equality by order one. Ensure the test really exercises amplified noise.
    if magnitude > 1:
        assert np.max(np.abs(points[:, :4] * noise[:, None])) > 0.5


def test_rlt_closes_balance_gap_and_agrees_in_dense_sparse_and_nodes():
    lp, rhs = make_lp([1, 1, 0], 1, [(0, 2), (1, 2)], [-1, -1])
    lp.obj[1][2] = lp.obj_mag[1][2] = 1.0
    lo, hi = torch.zeros(1, 3, dtype=torch.float64), torch.ones(1, 3, dtype=torch.float64)
    rr = rhs + 1e-5
    old, _ = lp.solve_node(*lp.sparse_lp(lo[0].numpy(), hi[0].numpy(), rr[0].numpy()))
    assert old < -0.49
    lp.add_rlt(rhs)
    assert lp.K == len(set(zip(lp.a.tolist(), lp.b.tolist())))
    for l, u in [(lo, hi), (lo + 0.1, hi - 0.1)]:
        dense = lp.assemble(l, u, rr, cutoff=torch.tensor([0.1]))
        sparse = lp.sparse_lp(l[0].numpy(), u[0].numpy(), rr[0].numpy(), cutoff=0.1)
        for i, (d, s) in enumerate(zip(dense, sparse)):
            assert np.allclose(d[0].numpy(), s.toarray() if i == 2 else s)
        D, _ = lp.solve_node(*sparse)
        assert -3e-5 < D <= 0
    D, _, closed = lp.branch_and_bound(lo, hi, rr, cutoff=0.0, tol_abs=1e-4,
                                      time_budget=2.0, max_nodes=10)
    assert closed and -3e-5 < D <= 0
    nl, nu = lp.obbt(lo, hi, rr, cutoff=0.0, rounds=1, verbose=False)
    points = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    assert ((points >= nl - 1e-9) & (points <= nu + 1e-9)).all()


@pytest.mark.parametrize('cap', [0, 1, 2, 7, 10000])
def test_cap_and_option_off_preserve_old_data(cap):
    lp, rhs = make_lp([1, 1, 0], 1, [(0, 2), (1, 2)], [-1, -1])
    l, u = torch.zeros(1, 3, dtype=torch.float64), torch.ones(1, 3, dtype=torch.float64)
    old = lp.assemble(l, u, rhs + 1e-5)
    original_pairs = (lp.a.clone(), lp.b.clone())
    lp.add_rlt(rhs, max_rows=cap)
    assert lp.n_rlt <= cap and lp.n_rlt % 2 == 0
    if cap < 2:
        assert all(torch.equal(a, b) for a, b in zip(original_pairs, (lp.a, lp.b)))
        assert all(torch.equal(a, b) for a, b in zip(old, lp.assemble(l, u, rhs + 1e-5)))
    if cap == 2:
        # The centre of this star is a complete vertex cover; prefer it under the cap.
        assert lp.rlt[3].tolist() == [2]


def test_rlt_multiplies_only_by_partners_of_the_equalitys_support():
    # x0 + x1 = 1; the original products are x0 x2, x1 x3 and x4^2. x2 and x3 are the
    # only variables that already share a product with the support {0, 1}.
    g = [1, 1, 0, 0, 0]
    lp, rhs = make_lp(g, 1, [(0, 2), (1, 3), (4, 4)], [1, 1, 1])
    lp.add_rlt(rhs)
    assert sorted(lp.rlt[3].tolist()) == [2, 3]
    # x2 (x0 + x1 - 1) = x0 x2 + x1 x2 - x2 creates the product x1 x2, x3 the product x0 x3.
    assert lp.K == 3 + 2
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (500, 5))
    x[:, 1] = 1 - x[:, 0]
    z = np.c_[x, x[:, lp.a] * x[:, lp.b]]
    l, u = np.zeros(5), np.ones(5)
    c, c0, G, h, lz, uz = lp.sparse_lp(l, u, (rhs[0] + 1e-6).numpy())
    assert np.all(G @ z.T <= h[:, None] + 1e-9)


def test_rlt_cap_is_all_or_nothing_over_the_selected_rows():
    g = [1, 1, 0, 0, 0]
    pairs = [(0, 2), (1, 3), (4, 4)]
    lp, rhs = make_lp(g, 1, pairs, [1, 1, 1])
    lp.add_rlt(rhs, max_rows=3)
    assert lp.n_rlt == 0 and lp.K == 3
    lp.add_rlt(rhs, max_rows=4)
    assert lp.n_rlt == 4
    lp, rhs = make_lp(g, 1, pairs, [1, 1, 1])
    lp.add_rlt(rhs, max_columns=7)
    assert lp.n_rlt == 0 and lp.K == 3


def test_presolve_infeasible_or_failed_is_retried_without_presolve(monkeypatch):
    from types import SimpleNamespace
    import nlp_solver.lifted_lp as module
    calls = []

    def fake(c, G, h, lz, uz, options):
        calls.append(options['presolve'])
        return SimpleNamespace(status=2 if options['presolve'] else 0)
    monkeypatch.setattr(module, '_highs_solve', fake)
    args = (None, None, None, None, None)
    assert module._highs_proposal(*args, {'presolve': True, 'retry': True}).status == 0
    assert calls == [True, False]
    calls.clear()
    assert module._highs_proposal(*args, {'presolve': True}).status == 2
    assert calls == [True]


def test_base_equalities_asymmetric_tolerance_and_missing_sources():
    lp, rhs = make_lp([1, 0, 0], 0, [(0, 1)], [1])
    # Remove the opposite spec coefficients, leaving only the base equality.
    lp.rows[1][1, 0] = -0.9
    A = torch.tensor([[[0., 1., 1.], [0., -1., -1.]]], dtype=torch.float64)
    b = torch.tensor([[1., -1.]], dtype=torch.float64)
    lp.add_rlt(rhs, A, b)
    assert lp.n_rlt == 2
    l = torch.tensor([[-1e5, -1., -1.]], dtype=torch.float64)
    u = torch.tensor([[1e5, 2., 2.]], dtype=torch.float64)
    bh = b + torch.tensor([[1e-5, 2e-5]])
    rr = rhs + 1e-5
    data = lp.assemble(l, u, rr, A, bh)
    sparse = lp.sparse_lp(l[0].numpy(), u[0].numpy(), rr[0].numpy(),
                          A[0].numpy(), bh[0].numpy())
    assert np.allclose(data[2][0].numpy(), sparse[2].toarray())
    assert np.allclose(data[3][0].numpy(), sparse[3])
    x = torch.tensor([[-1e5, 0.5, 0.5 + 1e-5], [1e5, 0.5, 0.5 - 2e-5]], dtype=torch.float64)
    z = torch.cat([x, x[:, lp.a] * x[:, lp.b]], 1)
    rlt = slice(-lp.n_rlt, None)
    assert (z @ data[2][0, rlt].T <= data[3][0, rlt] + 1e-9).all()
    # A node without these base rows must not retain their derived cuts.
    missing = lp.sparse_lp(l[0].numpy(), u[0].numpy(), rr[0].numpy())
    assert missing[2].shape[0] == lp.n_rows() - lp.n_rlt


def test_near_opposites_and_inactive_rows_are_not_equalities():
    lp, rhs = make_lp([1, 1, 0], 1, [(0, 2)], [1])
    lp.rows[1][1, 0] += 1e-12
    lp.add_rlt(rhs)
    assert lp.n_rlt == 0
    lp, rhs = make_lp([1, 1, 0], 1, [(0, 2)], [1])
    rhs[0, 1] = torch.inf
    lp.add_rlt(rhs)
    assert lp.n_rlt == 0


def test_preexisting_equality_slab_is_preserved():
    lp, rhs = make_lp([1, 1, 0], 1, [(0, 2)], [1])
    # The benchmark's VNNLIB already relaxes the equality by 1e-5, before the
    # solver adds constr_viol_tol. It is the entire slab that must be multiplied.
    rhs += 1e-5
    lp.add_rlt(rhs)
    assert lp.n_rlt == 2
    l = torch.tensor([[0., 0., -1e5]], dtype=torch.float64)
    u = torch.tensor([[1., 1., 1e5]], dtype=torch.float64)
    data = lp.assemble(l, u, rhs + 1e-4)
    x = torch.tensor([[0.5, 0.50011, -1e5], [0.5, 0.49989, 1e5]], dtype=torch.float64)
    z = torch.cat([x, x[:, lp.a] * x[:, lp.b]], 1)
    assert (z @ data[2][0].T <= data[3][0] + 1e-8).all()


def test_duplicate_base_equalities_do_not_spend_the_cap_twice():
    lp, rhs = make_lp([1, 1, 0], 1, [(0, 2)], [1])
    A = lp.rows[1].unsqueeze(0)
    b = rhs - lp.rows[0]
    lp.add_rlt(rhs, A, b)
    assert lp.n_rlt == 2


def test_dense_rlt_memory_limit_uses_sparse_certificate(monkeypatch):
    lp, rhs = make_lp([1, 1, 0], 1, [(0, 2), (1, 2)], [-1, -1])
    lp.add_rlt(rhs)
    l, u = torch.zeros(1, 3, dtype=torch.float64), torch.ones(1, 3, dtype=torch.float64)
    def too_large(*args, **kwargs):
        pytest.fail('dense assembly exceeds the configured budget')
    monkeypatch.setattr(lp, 'assemble', too_large)
    D, _ = lp.bound(l, u, rhs + 1e-5, max_elems=1, pdhg_iters=50, exact_max=1)
    assert torch.isfinite(D).all()


def test_rlt_auxiliary_products_do_not_change_branch_scores():
    lp, rhs = make_lp([1, 1, 0], 1, [(0, 2)], [1])
    z = torch.tensor([[0.5, 0.5, 0.5, 0.3]], dtype=torch.float64)
    old_score = lp.violation_score(z)
    lp.add_rlt(rhs)
    z = torch.cat([z, torch.full((1, lp.K - 1), 1e10)], 1)
    assert torch.equal(old_score, lp.violation_score(z))


def test_solver_objective_is_never_used_as_a_certificate(monkeypatch):
    from types import SimpleNamespace
    import scipy.optimize
    import scipy.sparse as sp
    result = SimpleNamespace(status=0, fun=12345.0, x=np.array([0.5]),
                             ineqlin=SimpleNamespace(marginals=np.array([0.0])))
    monkeypatch.setattr(scipy.optimize, 'linprog', lambda *a, **kw: result)
    data = (np.array([1.0]), 0.0, sp.csr_matrix([[1.0]]), np.array([1.0]),
            np.array([0.0]), np.array([1.0]))
    zero = pytest.approx(0.0, abs=1e-300)        # 0 less the evaluator's underflow term
    assert LiftedLP.solve_certified(*data) == zero
    assert LiftedLP.solve_node(*data)[0] == zero
    result.status = 1  # A time-limited or otherwise unfinished solve supplies no bound.
    assert LiftedLP.solve_certified(*data) is None
    assert LiftedLP.solve_node(*data) == (-np.inf, None)


@pytest.mark.parametrize('enabled', [False, True])
def test_root_setup_keeps_relaxed_feasible_points_with_a_cutoff(monkeypatch, enabled):
    from types import SimpleNamespace
    import input_split.lifted_bound as glue
    import nlp_solver.lifted_lp as module
    lp, rr = make_lp([1, 1, 0], 1, [(0, 2), (1, 2)], [-1, -1])
    lp.obj[1][2] = lp.obj_mag[1][2] = 1.0
    monkeypatch.setattr(module, 'build_lifted_lp', lambda *a, **kw: lp)
    def relaxed(rhs):
        out = rhs.clone()
        out[:, 1:] += 1e-5
        return out
    rhs = torch.cat([torch.zeros(1, 1, dtype=torch.float64), rr], 1)
    l, u = torch.zeros(1, 3, dtype=torch.float64), torch.ones(1, 3, dtype=torch.float64)
    history = SimpleNamespace(qtb=None, rowpool=SimpleNamespace(thresholds=relaxed))
    nl, nu = glue.setup_lifted_lp(SimpleNamespace(net=None, model_ori=None), history,
        l, u, torch.zeros(1, 3, 3), rhs, None,
        {'lifted_lp_bound': True, 'lifted_lp_obbt': True, 'lifted_lp_rlt': enabled,
         'lifted_lp_obbt_time': 2.0, 'lifted_lp_obbt_rounds': 1})
    assert (history.lifted.n_rlt > 0) == enabled
    x = torch.tensor([[0., 1., 1.], [1., 0., 0.], [0.4, 0.60001, 0.8]], dtype=torch.float64)
    assert ((x >= nl - 1e-9) & (x <= nu + 1e-9)).all()
    data = lp.sparse_lp(nl[0].numpy(), nu[0].numpy(), (rr[0] + 1e-5).numpy())
    D, _ = lp.solve_node(*data, presolve=not enabled)
    f = -(x[:, 0] + x[:, 1]) * x[:, 2] + x[:, 2]
    assert D <= float(f.min()) + 1e-9


def test_known_sign_tolerance_is_pointwise_on_a_wide_box():
    tol = 1e-5
    lp, rhs = make_lp([1, 1, 0], 1, [(0, 2), (1, 2)], [-1, -1])
    lp.obj[1][2] = lp.obj_mag[1][2] = 1 + 2 * tol
    lp.add_rlt(rhs)
    l, u = np.zeros(3), np.array([1., 1., 1e5])
    D, _ = lp.solve_node(*lp.sparse_lp(l, u, (rhs[0] + tol).numpy()), presolve=False)
    # On the relaxed equality x0+x1 <= 1+tol, f >= tol*x2 >= 0.
    # A constant tol*1e5 allowance loses this even at a small x2.
    assert -1e-5 < D <= 0


def test_expired_leaf_budget_skips_primal_work():
    from input_split.lifted_bound import _leaf_incumbent
    import time
    leaves = (np.zeros((1, 2)), np.ones((1, 2)), np.array([0.]), np.array([[0.5, 0.5]]))
    def expired(*args):
        pytest.fail('primal work started after the deadline')
    assert _leaf_incumbent(leaves, None, expired, 2.0, tight_point=expired,
                           deadline=time.time() - 1.0) is None


def test_rlt_is_skipped_not_truncated_when_rows_exceed_the_cap():
    # Every variable shares a product with the equality's support: 4 multipliers, 8 rows.
    pairs = [(0, 1), (1, 2), (2, 3), (0, 0)]
    lp, rhs = make_lp([0.5, -1.0, 2.0, 1.0, 1.0], 0.0, pairs, [1.0] * len(pairs))
    original = (lp.a.clone(), lp.b.clone())
    lp.add_rlt(rhs, max_rows=7)
    assert lp.rlt is None and lp.n_rlt == 0
    assert torch.equal(original[0], lp.a) and torch.equal(original[1], lp.b)
    lp.add_rlt(rhs, max_rows=8)
    assert lp.n_rlt == 8


def test_rlt_is_skipped_when_the_lifted_lp_would_be_too_wide():
    pairs = [(0, 1), (1, 2), (2, 3), (0, 0)]
    lp, rhs = make_lp([0.5, -1.0, 2.0, 1.0, 1.0], 0.0, pairs, [1.0] * len(pairs))
    lp.add_rlt(rhs, max_columns=lp.n + len(pairs))
    assert lp.rlt is None and lp.K == len(pairs)


def test_full_budget_charges_new_square_and_bilinear_envelopes():
    g, pairs = [1, 1, 0], [(0, 1), (1, 2)]
    lp, rhs = make_lp(g, 1, pairs, [1, 1])
    old = (lp.n_mc, lp.N)
    lp.add_rlt(rhs, max_rows=10000, full_budget=True)
    cost = lp.rlt_budget
    assert cost['rows'] == lp.n_rlt + lp.n_mc - old[0]
    assert cost['columns'] == lp.N - old[1]
    data = lp.sparse_lp(-np.ones(3), np.ones(3), (rhs[0] + 1e-5).numpy())
    bare, _ = make_lp(g, 1, pairs, [1, 1])
    base = bare.sparse_lp(-np.ones(3), np.ones(3), (rhs[0] + 1e-5).numpy())
    assert cost['nnz'] >= data[2].nnz - base[2].nnz
    # All-or-nothing admission, with the same cap and no mutated products on rejection.
    bare.add_rlt(rhs, max_rows=sum(cost.values()) - 1, full_budget=True)
    assert bare.rlt is None and bare.N == old[1]
    bare.add_rlt(rhs, max_rows=sum(cost.values()), full_budget=True)
    assert bare.rlt is not None and bare.N == lp.N
