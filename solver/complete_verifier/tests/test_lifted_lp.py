"""Tests for the lifted McCormick LP bound and root OBBT (nlp_solver/lifted_lp.py).

Validity is checked by sampling: no point of a box that satisfies the relaxed rows may have
an objective below the certified bound, and OBBT may not cut off any such point whose
objective is at most the cutoff.
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from auto_LiRPA import BoundedModule  # noqa: E402
from nlp_solver.lifted_lp import build_lifted_lp, extract_quadratic  # noqa: E402

TOL = 1e-4


class PoolModel(nn.Module):
    """[obj, three quadratic constraints, one tanh constraint] on 4 variables."""

    def forward(self, x):
        x0, x1, x2, x3 = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        obj = -2.0 * x0 * x1 + 0.5 * x2 * x2 - x3 + 3.0 * x0 * x3
        c1 = x0 * x2 - 0.7 * x1 + 0.2                  # bilinear
        c2 = 1.0 - x0 * x0 - x1 * x1                   # reverse convex
        c3 = x1 * x3 + x2 - 1.5
        c4 = torch.tanh(x2) - 0.5                       # not quadratic: must be dropped
        return torch.cat([obj, c1, c2, c3, c4], dim=-1)


def _setup():
    torch.set_default_dtype(torch.float64)
    m = PoolModel()
    x = torch.zeros(1, 4)
    bm = BoundedModule(m, x)
    # Specs: objective, c1 <= 0, c2 <= 0, c3 <= 0, c4 <= 0, and a combined row c1 - c3 <= 0.3.
    C = torch.zeros(1, 6, 5)
    for s in range(5):
        C[0, s, s] = 1.0
    C[0, 5, 1], C[0, 5, 3] = 1.0, -1.0
    rhs = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.3]])
    l = torch.tensor([[-1.0, -0.5, 0.0, -2.0]])
    u = torch.tensor([[1.5, 2.0, 1.0, 1.0]])
    return m, bm, x, C, rhs, l, u


def _feasible(m, C, rhs, z, rows):
    y = m(z) @ C[0].T                                    # (P, S)
    return (y[:, rows] <= rhs[0, rows] + TOL).all(1), y[:, 0]


def test_extraction_reproduces_quadratic_outputs_and_flags_others():
    m, bm, x, C, rhs, l, u = _setup()
    g = torch.Generator().manual_seed(1)
    pts = l + (u - l) * torch.rand(5, 4, generator=g)
    k, gr, a, b, q, ok = extract_quadratic(lambda z: m(z.reshape(1, -1)).reshape(-1), 4, pts)
    assert ok.tolist() == [True, True, True, True, False]
    z = l + (u - l) * torch.rand(200, 4, generator=g)
    rep = k[:4] + z @ gr[:4].T + (z[:, a] * z[:, b]) @ q[:4].T
    assert torch.allclose(rep, m(z)[:, :4], atol=1e-12)


def test_non_quadratic_rows_are_not_lifted():
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    assert lp is not None and lp.obj is not None
    assert lp.row_spec.tolist() == [1, 2, 3, 5]


@pytest.mark.parametrize('seed', range(4))
def test_bound_is_below_every_feasible_point(seed):
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    g = torch.Generator().manual_seed(seed)
    B = 6
    lo = l + (u - l) * torch.rand(B, 4, generator=g) * 0.6
    hi = lo + (u - lo) * (0.2 + 0.8 * torch.rand(B, 4, generator=g))
    rhs_rows = (rhs[:, 1:] + TOL).expand(B, -1)
    for exact in (0, B):
        for iters in (0, 50, 1000):
            bnd, _ = lp.bound(lo, hi, rhs_rows, pdhg_iters=iters, exact_max=exact)
            for i in range(B):
                z = lo[i] + (hi[i] - lo[i]) * torch.rand(20000, 4, generator=g)
                # Only the lifted rows restrict the LP; feasibility for all rows is stronger.
                feas, f = _feasible(m, C, rhs, z, [1, 2, 3, 5])
                if feas.any():
                    assert float(f[feas].min()) >= float(bnd[i]) - 1e-9, (iters, exact, i)


def test_cushion_follows_the_box_and_bounds_stay_valid_on_wide_boxes():
    m, bm, x, C, rhs, l, u = _setup()
    big_l = torch.tensor([[-1e3, -5e2, 0.0, -2e3]])
    big_u = torch.tensor([[1.5e3, 2e3, 1e3, 1e3]])
    lp = build_lifted_lp(bm, m, x, C, big_l, big_u)
    cr_big, co_big = lp._cushions(big_l, big_u)
    cr_small, co_small = lp._cushions(l, u)
    assert torch.all(cr_small < cr_big) and torch.all(co_small < co_big)
    g = torch.Generator().manual_seed(11)
    lo = big_l + (big_u - big_l) * torch.rand(4, 4, generator=g) * 0.5
    hi = lo + (big_u - lo) * torch.rand(4, 4, generator=g)
    rr = (rhs[:, 1:] + TOL).expand(4, -1)
    bnd, _ = lp.bound(lo, hi, rr, pdhg_iters=200, exact_max=4)
    for i in range(4):
        z = lo[i] + (hi[i] - lo[i]) * torch.rand(50000, 4, generator=g)
        feas, f = _feasible(m, C, rhs, z, [1, 2, 3, 5])
        if feas.any():
            assert float(f[feas].min()) >= float(bnd[i]) - 1e-6 * (1 + abs(float(bnd[i])))


def test_exact_bound_matches_the_lp_and_beats_nothing_feasible():
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    rhs_rows = rhs[:, 1:] + TOL
    b_pdhg, _ = lp.bound(l, u, rhs_rows, pdhg_iters=3000, exact_max=0)
    b_ex, _ = lp.bound(l, u, rhs_rows, pdhg_iters=0, exact_max=1)
    assert torch.isfinite(b_ex).all()
    assert float(b_ex[0]) >= float(b_pdhg[0]) - 1e-3 * (1 + abs(float(b_ex[0])))


def test_extra_rows_and_objective_plane_path():
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    lp.obj = None       # pretend the objective is not quadratic: use a valid lower plane
    # A valid lower plane of obj on the box: obj >= a^T x + b with b from interval bounds
    # of the remaining terms (here crude: evaluate a very low constant).
    a = torch.zeros(1, 4)
    b = torch.tensor([-50.0])
    eg = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])     # x0 + x1 <= 2 (true for sampled pts)
    eh = torch.tensor([[2.0]])
    bnd, _ = lp.bound(l, u, rhs[:, 1:] + TOL, eg, eh, (a, b), pdhg_iters=100, exact_max=1)
    assert float(bnd[0]) <= -50.0 + 1e-9


@pytest.mark.parametrize('seed', range(3))
def test_obbt_keeps_every_feasible_point_below_the_cutoff(seed):
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    g = torch.Generator().manual_seed(seed)
    z = l + (u - l) * torch.rand(200000, 4, generator=g)
    feas, f = _feasible(m, C, rhs, z, [1, 2, 3, 5])
    assert feas.sum() > 100
    cutoff = float(torch.quantile(f[feas], 0.3))
    nl, nu = lp.obbt(l, u, rhs[:, 1:] + TOL, cutoff=cutoff, rounds=4, verbose=False)
    keep = feas & (f <= cutoff)
    inside = ((z >= nl) & (z <= nu)).all(1)
    assert inside[keep].all()
    assert float((nu - nl).sum()) < float((u - l).sum())     # it did tighten something


def test_dense_and_sparse_assembly_agree():
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    rhs_rows = rhs[:, 1:] + TOL
    c, c0, G, h, lz, uz = lp.assemble(l, u, rhs_rows, cutoff=torch.tensor([1.0]))
    cs, c0s, Gs, hs, lzs, uzs = lp.sparse_lp(l[0].numpy(), u[0].numpy(), rhs_rows[0].numpy(),
                                             cutoff=1.0)
    assert np.allclose(G[0].numpy(), Gs.toarray())
    assert np.allclose(h[0].numpy(), hs)
    assert np.allclose(c[0].numpy(), cs) and abs(float(c0[0]) - c0s) < 1e-12
    assert np.allclose(lz[0].numpy(), lzs) and np.allclose(uz[0].numpy(), uzs)


def test_mccormick_rows_hold_at_lifted_points():
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    g = torch.Generator().manual_seed(3)
    z = l + (u - l) * torch.rand(5000, 4, generator=g)
    w = z[:, lp.a] * z[:, lp.b]
    zz = torch.cat([z, w], 1)
    _, _, G, h, lz, uz = lp.assemble(l, u, rhs[:, 1:] + 1e9)
    mc = slice(int(lp.row_spec.numel()), int(lp.row_spec.numel()) + lp.n_mc)
    assert torch.all(zz @ G[0, mc].T <= h[0, mc] + 1e-12)
    assert torch.all((zz >= lz - 1e-12) & (zz <= uz + 1e-12))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_bound_and_obbt_on_cuda():
    m, bm, x, C, rhs, l, u = _setup()
    dev = torch.device('cuda')
    m, x, C, rhs, l, u = m.to(dev), x.to(dev), C.to(dev), rhs.to(dev), l.to(dev), u.to(dev)
    bm = BoundedModule(m, x, device=dev)
    lp = build_lifted_lp(bm, m, x, C, l, u)
    assert lp is not None
    lo = l.expand(3, -1).contiguous()
    hi = (l + 0.5 * (u - l)).expand(3, -1).contiguous()
    bnd, y = lp.bound(lo, hi, (rhs[:, 1:] + TOL).expand(3, -1), pdhg_iters=100, exact_max=2)
    m2, bm2, x2, C2, rhs2, l2, u2 = _setup()
    bnd_cpu, _ = build_lifted_lp(bm2, m2, x2, C2, l2, u2).bound(
        lo.cpu(), hi.cpu(), (rhs2[:, 1:] + TOL).expand(3, -1), pdhg_iters=100, exact_max=2)
    # identical boxes: which one gets the exact re-solve is a tie, compare as multisets
    assert torch.allclose(bnd.cpu().sort().values, bnd_cpu.sort().values, atol=1e-6, rtol=1e-6)
    nl, nu = lp.obbt(l, u, rhs[:, 1:] + TOL, cutoff=0.0, rounds=2, verbose=False)
    assert nl.device.type == 'cuda' and bool((nl <= nu).all())


def test_points_and_violation_branching():
    from types import SimpleNamespace
    from input_split.branching_heuristics import _lifted_violation_choice
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    lo = l.expand(3, -1).contiguous()
    hi = u.expand(3, -1).contiguous()
    bnd, _, Z = lp.bound(lo, hi, (rhs[:, 1:] + TOL).expand(3, -1), pdhg_iters=200,
                         exact_max=1, return_points=True)
    assert Z.shape == (3, lp.N) and torch.isfinite(Z).all()
    assert bool(((Z[:, :4] >= lo - 1e-9) & (Z[:, :4] <= hi + 1e-9)).all())
    s = lp.violation_score(Z)
    # by hand for one row
    xz, wz = Z[0, :4], Z[0, 4:]
    ref = torch.zeros(4)
    for k in range(lp.K):
        v = abs(float(wz[k] - xz[lp.a[k]] * xz[lp.b[k]]))
        ref[lp.a[k]] += v
        ref[lp.b[k]] += v
    assert torch.allclose(s[0], ref)
    h = SimpleNamespace(lifted_branch_score=torch.tensor([[0.0, 3.0, 1.0, 0.0],
                                                           [float('nan')] * 4]))
    idx = torch.tensor([[0], [2]])
    out = _lifted_violation_choice(idx, lo[:2], hi[:2], h)
    assert out.tolist() == [[1], [2]]           # row 1 has no LP point: SB choice kept


def test_split_at_lp_point_keeps_a_partition():
    from types import SimpleNamespace
    from input_split.lifted_bound import apply_lifted_split_points
    from input_split.split import input_split_parallel
    torch.set_default_dtype(torch.float64)
    pl = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    pu = torch.tensor([[1.0, 2.0, 4.0], [3.0, 3.0, 3.0]])
    idx = torch.tensor([[2], [0]])
    nl, nu, _ = input_split_parallel(pl, pu, pl.shape, 1, idx, 2)
    h = SimpleNamespace(
        lifted_branch_score=torch.tensor([[0.0, 1.0, 5.0], [2.0, 1.0, 0.0]]),
        lifted_split_point=torch.tensor([[0.5, 0.5, 3.9], [1.5, 2.0, 2.0]]))
    al, au = apply_lifted_split_points(h, pl, pu, idx, nl, nu)
    # domain 0 on dim 2: point 3.9 clamped to 0.9 * 4 = 3.6; domain 1 on dim 0: 1.5
    assert float(al[0, 2]) == pytest.approx(3.6) and float(au[2, 2]) == pytest.approx(3.6)
    assert float(al[1, 0]) == pytest.approx(1.5) and float(au[3, 0]) == pytest.approx(1.5)
    for d in range(2):
        up, lo = d, 2 + d
        # the two children cover the parent exactly
        assert torch.equal(torch.minimum(al[up], al[lo]), pl[d])
        assert torch.equal(torch.maximum(au[up], au[lo]), pu[d])
    # a domain whose chosen dim is not the lifted argmax keeps the midpoint
    h.lifted_branch_score = torch.tensor([[0.0, 9.0, 5.0], [2.0, 1.0, 0.0]])
    al, au = apply_lifted_split_points(h, pl, pu, idx, nl, nu)
    assert float(al[0, 2]) == 2.0


@pytest.mark.parametrize('seed', range(3))
def test_lifted_branch_and_bound_is_valid_and_converges(seed):
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    g = torch.Generator().manual_seed(seed)
    z = l + (u - l) * torch.rand(400000, 4, generator=g)
    feas, f = _feasible(m, C, rhs, z, [1, 2, 3, 5])
    best = float(f[feas].min())
    # With a cutoff above the sampled optimum, D may never exceed a feasible objective.
    for cutoff in (best + 0.5, best + 1e-3):
        D, nodes, closed = lp.branch_and_bound(l, u, rhs[:, 1:] + TOL, cutoff=cutoff,
                                               time_budget=20.0, max_nodes=3000)
        assert D <= best + 1e-9, (D, best)
    # A cutoff far below every feasible point is proved by pruning everything: D = cutoff.
    D, nodes, closed = lp.branch_and_bound(l, u, rhs[:, 1:] + TOL, cutoff=best - 5.0,
                                           time_budget=20.0, max_nodes=3000)
    assert D <= best - 5.0 + 1e-12


def test_run_lifted_bab_gating():
    from types import SimpleNamespace
    from input_split.lifted_bound import run_lifted_bab
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    # The spec has a non-lifted finite row (tanh): the certifier must refuse.
    h = SimpleNamespace(qtb=None, lifted=lp, lifted_obbt_rows=(rhs[:, 1:] + TOL, None, None))
    cfg = {"lifted_lp_bab_time": 5.0}
    assert run_lifted_bab(h, l, u, torch.tensor([0.0]), cfg, 100.0) is None
    # With that row inactive it runs once per incumbent improvement, and its open leaves get one
    # continued measurement at the same incumbent; without measured gain nothing more runs.
    rr = (rhs[:, 1:] + TOL).clone()
    rr[0, 3] = float('inf')
    h = SimpleNamespace(qtb=None, lifted=lp, lifted_obbt_rows=(rr, None, None))
    run_lifted_bab(h, l, u, torch.tensor([5.0]), cfg, 100.0)
    assert h.lifted_bab_cutoff == 5.0 and h.lifted_bab_spent > 0
    assert len(h.lifted_bab_open[2]) > 0 and getattr(h, 'lifted_bab_progress', None) is None
    spent = h.lifted_bab_spent
    run_lifted_bab(h, l, u, torch.tensor([5.0]), cfg, 100.0)
    assert h.lifted_bab_spent > spent and h.lifted_bab_progress is not None
    h.lifted_bab_progress, h.lifted_bab_paid = (0.0, 1.0), False
    spent = h.lifted_bab_spent
    assert run_lifted_bab(h, l, u, torch.tensor([5.0]), cfg, 100.0) is None
    assert h.lifted_bab_spent == spent


def test_leaf_incumbent_takes_only_verified_points():
    import numpy as np
    from input_split.lifted_bound import _leaf_incumbent
    # "Feasible" iff x0 >= 0.5 (like a row at its tolerance); objective = sum(x).
    verify = lambda x: (bool(x[0] >= 0.5), float(x.sum()), x)
    L_lo = np.zeros((2, 2)); L_hi = np.ones((2, 2))
    L_bd = np.array([0.1, 0.2])
    L_x = np.array([[0.4999, 0.1], [0.6, 0.3]])
    inc_x = torch.tensor([1.0, 1.0])
    x, obj = _leaf_incumbent((L_lo, L_hi, L_bd, L_x), inc_x, verify, inc=2.0)
    # leaf 0's point fails, its blend 0.999 towards the incumbent passes and is best
    assert x[0] >= 0.5 and obj < 0.9 + 1e-9
    assert _leaf_incumbent((L_lo, L_hi, L_bd, L_x), inc_x, verify, inc=0.5) is None


def test_verifier_reports_raw_objective_when_rhs_holds_the_incumbent():
    # Regression: verify_counterexample returns C_obj f(x) - rhs_obj; with the incumbent
    # in rhs[:, 0] that shifted value was compared with the raw incumbent and a leaf
    # point "beat" it (kall_diffcircles_5a: -0.00104 vs 5.1158, a false certificate).
    import numpy as np
    from input_split.lifted_bound import make_verifier
    m, bm, x, C, rhs, l, u = _setup()
    rhs2 = rhs.clone()
    rhs2[0, 0] = 123.0
    verify = make_verifier(m, l, u, C, rhs2, None, None, {"constr_viol_tol": TOL})
    g = torch.Generator().manual_seed(7)
    for _ in range(20):
        z = (l + (u - l) * torch.rand(1, 4, generator=g))[0]
        ok, obj, xz = verify(z.numpy())
        assert abs(obj - float(m(z.reshape(1, -1))[0, 0])) < 1e-9
        assert np.all(xz >= l[0].numpy()) and np.all(xz <= u[0].numpy())


def test_parallel_branch_and_bound_is_valid():
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    g = torch.Generator().manual_seed(4)
    z = l + (u - l) * torch.rand(400000, 4, generator=g)
    feas, f = _feasible(m, C, rhs, z, [1, 2, 3, 5])
    best = float(f[feas].min())
    D1, n1, c1 = lp.branch_and_bound(l, u, rhs[:, 1:] + TOL, cutoff=best + 1e-3,
                                     time_budget=10.0, max_nodes=400)
    D4, n4, c4, leaves = lp.branch_and_bound(l, u, rhs[:, 1:] + TOL, cutoff=best + 1e-3,
                                             time_budget=10.0, max_nodes=400, workers=4,
                                             return_leaves=True)
    assert D1 <= best + 1e-9 and D4 <= best + 1e-9
    assert n4 > 0
    # every sampled feasible point below the cutoff lies in an open leaf with bound <= f
    lo, hi, bd = (torch.as_tensor(t) for t in leaves[:3])
    keep = feas & (f <= best + 1e-3)
    zk, fk = z[keep][:2000], f[keep][:2000]
    inside = ((zk.unsqueeze(1) >= lo - 1e-12) & (zk.unsqueeze(1) <= hi + 1e-12)).all(-1)
    assert inside.any(1).all()
    ok = torch.where(inside, bd.unsqueeze(0) <= fk.unsqueeze(1) + 1e-9, torch.ones_like(inside))
    assert ok.all()


def test_infeasible_node_is_certified():
    import numpy as np
    import scipy.sparse as sp
    from nlp_solver.lifted_lp import LiftedLP
    # x0 + x1 <= -1 on the box [0, 1]^2 has no point.
    G = sp.csr_matrix(np.array([[1.0, 1.0]]))
    b, xz = LiftedLP.solve_node(np.array([1.0, 0.0]), 0.0, G, np.array([-1.0]),
                                np.zeros(2), np.ones(2))
    assert b == float('inf') and xz is None
    b, xz = LiftedLP.solve_node(np.array([1.0, 0.0]), 0.0, G, np.array([1.0]),
                                np.zeros(2), np.ones(2))
    assert abs(b) < 1e-9 and xz is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_parallel_branch_and_bound_from_a_cuda_process():
    # The production process holds a CUDA context when it forks the B&B workers; the
    # workers must only use numpy/scipy (sparse_lp, solve_node).
    m, bm, x, C, rhs, l, u = _setup()
    dev = torch.device('cuda')
    m, x, C, rhs, l, u = m.to(dev), x.to(dev), C.to(dev), rhs.to(dev), l.to(dev), u.to(dev)
    lp = build_lifted_lp(BoundedModule(m, x, device=dev), m, x, C, l, u)
    D1, _, _ = lp.branch_and_bound(l, u, rhs[:, 1:] + TOL, cutoff=5.0, time_budget=5.0,
                                   max_nodes=200)
    D4, n4, _ = lp.branch_and_bound(l, u, rhs[:, 1:] + TOL, cutoff=5.0, time_budget=5.0,
                                    max_nodes=200, workers=4)
    assert n4 > 0 and abs(D1 - D4) < 1.0 + 1e-3 * abs(D1)


def test_clamp_to_lifted_box_intersects_and_degenerates_empty():
    from types import SimpleNamespace
    from input_split.lifted_bound import clamp_to_lifted_box
    torch.set_default_dtype(torch.float64)
    h = SimpleNamespace(lifted_box=(torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0, 1.0]])),
                        lifted_cutoff=1.0)
    xl = torch.tensor([[-1.0, 0.5], [2.0, 0.2]])
    xu = torch.tensor([[0.5, 3.0], [3.0, 0.4]])
    nl, nu = clamp_to_lifted_box(h, xl, xu)
    assert nl.tolist() == [[0.0, 0.5], [1.0, 0.2]]
    assert nu.tolist() == [[0.5, 1.0], [1.0, 0.4]]
    assert bool((nl <= nu).all())


def test_reobbt_runs_only_on_improvement_and_keeps_feasible_points():
    from types import SimpleNamespace
    from input_split.lifted_bound import reobbt_on_improvement
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    g = torch.Generator().manual_seed(5)
    z = l + (u - l) * torch.rand(200000, 4, generator=g)
    feas, f = _feasible(m, C, rhs, z, [1, 2, 3, 5])
    c1 = float(torch.quantile(f[feas], 0.6))
    c2 = float(torch.quantile(f[feas], 0.2))
    h = SimpleNamespace(lifted=lp, lifted_box=(l.clone(), u.clone()), lifted_cutoff=c1,
                        lifted_obbt_rows=(rhs[:, 1:] + TOL, None, None),
                        lifted_reobbt_time=0.0)
    cfg = {"lifted_lp_obbt_time": 10.0, "lifted_lp_obbt_rounds": 3}
    assert not reobbt_on_improvement(h, torch.tensor([c1]), cfg, 100.0)
    assert reobbt_on_improvement(h, torch.tensor([c2]), cfg, 100.0)
    nl, nu = h.lifted_box
    keep = feas & (f <= c2)
    assert ((z >= nl) & (z <= nu)).all(1)[keep].all()
    assert h.lifted_cutoff == c2
    # an unproductive re-OBBT (mean width shrunk by under 2%) exhausts the re-OBBT budget
    ratio = float(((nu - nl) / (u - l)).mean())
    assert (h.lifted_reobbt_time == 2 * cfg["lifted_lp_obbt_time"]) == (ratio >= 0.98)
    h.lifted_reobbt_time = 20.0
    assert not reobbt_on_improvement(h, torch.tensor([c2 - 1.0]), cfg, 100.0)


def test_fbbt_keeps_points_on_a_row_boundary():
    """Rows with coefficients over 18 orders of magnitude, rhs the exact value (rounded up) at the
    minimising corner: every operation rounds outward, so that corner survives and no row is
    claimed empty. An unrounded propagation cuts it off (the wastewater14m1 failure)."""
    import scipy.sparse as sp
    from fractions import Fraction
    from nlp_solver.lifted_lp import _fbbt
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = 6
        coef = rng.choice([-1.0, 1.0], n) * 10.0 ** rng.uniform(-8, 10, n)
        lo = rng.uniform(-3, 3, n)
        hi = lo + rng.uniform(0.01, 2, n)
        corner = np.where(coef > 0, lo, hi)
        exact = sum(Fraction(c) * Fraction(z) for c, z in zip(coef, corner))
        h = float(exact)
        if Fraction(h) < exact:
            h = np.nextafter(h, np.inf)
        new_lo, new_hi, row = _fbbt(sp.csr_matrix(coef.reshape(1, -1)), np.array([h]), lo, hi)
        assert row is None
        assert (new_lo <= corner).all() and (corner <= new_hi).all()


def test_dual_value_is_never_nan_on_unbounded_columns():
    """A column with an infinite bound and no reduced cost adds 0; an unbounded direction gives
    -inf. NaN (from 0 * inf) used to close the node without a proof."""
    import scipy.sparse as sp
    from nlp_solver.lifted_lp import _dual_value_sparse
    G, h, y = sp.csr_matrix(np.zeros((1, 2))), np.zeros(1), np.zeros(1)
    lz, uz = np.array([0.0, -np.inf]), np.array([1.0, np.inf])
    assert _dual_value_sparse(np.array([1.0, 0.0]), 0.0, G, h, lz, uz, y) == pytest.approx(0.0)
    assert _dual_value_sparse(np.array([1.0, 1.0]), 0.0, G, h, lz, uz, y) == -np.inf


def test_pool_workers_evaluate_large_rows_after_the_parent_used_torch_threads():
    """The dual evaluator is torch: a forked worker that enters the parent's thread pool (an op on more than 32768
    entries) hangs, so _pool_solve runs it on one thread."""
    import scipy.sparse as sp
    import nlp_solver.lifted_lp as module
    G = sp.random(2000, 3000, density=0.02, random_state=0, format='csr')
    n, m = G.shape[1], G.shape[0]
    args = (np.zeros(n), 0.0, G, np.ones(m), -np.ones(n), np.ones(n), np.ones(m))

    class FakeLP:
        def solve_bab_node(self, *_, **kwargs):
            return module._dual_value_sparse(*args)

    torch.randn(2_000_000).add_(1).sum()                 # the parent has started its thread pool
    pool = module._make_pool(FakeLP(), None, None, None, 0.0, 2)
    try:
        out = pool.map_async(module._pool_solve, [(None, None, None)] * 2, chunksize=1).get(timeout=60)
    finally:
        pool.terminate()
    assert out == [module._dual_value_sparse(*args)] * 2


def test_fbbt_row_claims_are_proved_by_multipliers():
    import scipy.sparse as sp
    from nlp_solver.lifted_lp import _fbbt, _certified_infeasible
    G = sp.csr_matrix(np.array([[1.0, 1.0]]))
    lo, hi = np.zeros(2), np.ones(2)
    for rhs in (-1.0, -1e-12):      # empty: the certificate of the rigorous evaluator needs no margin
        h = np.array([rhs])
        _, _, row = _fbbt(G, h, lo, hi)
        assert row == 0
        assert _certified_infeasible(G, h, lo, hi, np.ones(1))
    assert not _certified_infeasible(G, np.zeros(1), lo, hi, np.ones(1))        # the corner (0, 0) satisfies the row


def test_node_box_keeps_every_feasible_point_below_the_cutoff():
    """solve_bab_node (cutoff row, propagation, OBBT) on random sub-boxes: no sampled point
    that satisfies the relaxed rows with objective <= cutoff leaves the returned box or falls
    below the returned bound; a cutoff under every feasible objective empties the node."""
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    rr = (rhs[:, 1:] + TOL).numpy().reshape(-1)
    g = torch.Generator().manual_seed(5)
    z = l + (u - l) * torch.rand(400000, 4, generator=g)
    feas, f = _feasible(m, C, rhs, z, [1, 2, 3, 5])
    best = float(f[feas].min())
    for cutoff in (best + 0.05, best + 1.0):
        keep = feas & (f <= cutoff)
        for _ in range(6):
            a = l + (u - l) * torch.rand(1, 4, generator=g) * 0.6
            b = a + (u - a) * (0.3 + 0.7 * torch.rand(1, 4, generator=g))
            bound, _, nl, nu, _ = lp.solve_bab_node(a[0].numpy(), b[0].numpy(), rr, None, None,
                                                 cutoff)
            inside = ((z >= a) & (z <= b)).all(1) & keep
            if not bool(inside.any()):
                continue
            assert np.isfinite(bound) or bound < 0
            zi = z[inside].numpy()
            assert (zi >= nl - 1e-12).all() and (zi <= nu + 1e-12).all()
            assert bound <= f[inside].min() + 1e-9
    bound, *_ = lp.solve_bab_node(l[0].numpy(), u[0].numpy(), rr, None, None, best - 5.0)
    assert bound == float('inf')


def test_warm_started_node_bound_equals_the_cold_one():
    """A child's LP started from its parent's basis (HiGHS dual simplex, no presolve) gives the
    same certified bound as the cold solve, on random nested boxes; a basis that does not fit
    the rows is dropped."""
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    rr = (rhs[:, 1:] + TOL).numpy().reshape(-1)
    g = torch.Generator().manual_seed(7)
    parent = {'basis': None}
    lp.solve_node(*lp.sparse_lp(l[0].numpy(), u[0].numpy(), rr), warm=parent)
    assert parent['basis'] is not None
    for _ in range(8):
        a = l + (u - l) * torch.rand(1, 4, generator=g) * 0.5
        b = a + (u - a) * (0.2 + 0.8 * torch.rand(1, 4, generator=g))
        data = lp.sparse_lp(a[0].numpy(), b[0].numpy(), rr)
        cold, _ = lp.solve_node(*data)
        warm = {'basis': parent['basis']}
        hot, _ = lp.solve_node(*data, warm=warm)
        assert warm['basis'] is not None or not np.isfinite(hot)
        assert hot == pytest.approx(cold, rel=1e-9, abs=1e-9)
    bad = {'basis': (parent['basis'][0][:-1], parent['basis'][1])}
    assert lp.solve_node(*data, warm=bad)[0] == pytest.approx(cold, rel=1e-9, abs=1e-9)


def test_probing_stops_when_probes_do_not_pay_and_counts_those_that_do():
    """probes = [probed nodes, probes, paying probes]: a run that has probed _PROBE_NODES nodes
    with fewer than _PROBE_PAY paying skips the probes (no counter moves, the box is the
    propagated one); otherwise the node is probed and counted."""
    from nlp_solver import lifted_lp
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    rr = (rhs[:, 1:] + TOL).numpy().reshape(-1)
    args = (l[0].numpy(), u[0].numpy(), rr, None, None, 3.0)
    K = lifted_lp._PROBE_NODES
    fresh = [0, 0, 0]
    lp.solve_bab_node(*args, probes=fresh)
    assert fresh[0] == 1 and fresh[1] > 0 and 0 <= fresh[2] <= fresh[1]
    barren = [K, 100, 10]
    _, _, lo, hi, _ = lp.solve_bab_node(*args, probes=barren)
    assert barren == [K, 100, 10]
    old, lifted_lp._NODE_OBBT_VARS = lifted_lp._NODE_OBBT_VARS, 0
    try:
        _, _, lo0, hi0, _ = lp.solve_bab_node(*args)
    finally:
        lifted_lp._NODE_OBBT_VARS = old
    assert np.array_equal(lo, lo0) and np.array_equal(hi, hi0)
    paying = [K, 100, 30]
    lp.solve_bab_node(*args, probes=paying)
    assert paying[0] == K + 1 and paying[1] > 100


def test_continued_branch_and_bound_covers_every_point_below_the_new_cutoff():
    """Leaves of a run at cutoff c1, continued at c2 < c1 (leaves of bound >= c2 dropped), still
    hold every sampled relaxed-feasible point of objective <= c2, each with a bound below its
    objective, and never fall below the first run's D."""
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    rr = rhs[:, 1:] + TOL
    g = torch.Generator().manual_seed(11)
    z = l + (u - l) * torch.rand(400000, 4, generator=g)
    feas, f = _feasible(m, C, rhs, z, [1, 2, 3, 5])
    best = float(f[feas].min())
    D1, n1, _, first = lp.branch_and_bound(l, u, rr, cutoff=best + 1.0, time_budget=10.0,
                                           max_nodes=12, return_leaves=True)
    c2 = best + 1e-3
    D2, n2, _, second = lp.branch_and_bound(l, u, rr, cutoff=c2, time_budget=10.0, max_nodes=60,
                                            return_leaves=True, resume=first)
    assert D1 - 1e-9 <= D2 <= best + 1e-9
    lo, hi, bd = (torch.as_tensor(t) for t in second[:3])
    keep = feas & (f <= c2)
    zk, fk = z[keep][:2000], f[keep][:2000]
    inside = ((zk.unsqueeze(1) >= lo - 1e-12) & (zk.unsqueeze(1) <= hi + 1e-12)).all(-1)
    assert inside.any(1).all()
    assert torch.where(inside, bd.unsqueeze(0) <= fk.unsqueeze(1) + 1e-9,
                       torch.ones_like(inside)).all()
    # all leaves above the cutoff: nothing below it is left, D is the cutoff
    D3, n3, closed, _ = lp.branch_and_bound(l, u, rr, cutoff=float(first[2].min()) - 1.0,
                                            time_budget=10.0, resume=first, return_leaves=True)
    assert n3 == 0 and closed


def test_run_lifted_bab_continues_from_the_previous_leaves(capsys):
    from types import SimpleNamespace
    from input_split.lifted_bound import run_lifted_bab
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    rr = (rhs[:, 1:] + TOL).clone()
    rr[0, 3] = float('inf')
    h = SimpleNamespace(qtb=None, lifted=lp, lifted_obbt_rows=(rr, None, None))
    cfg = {"lifted_lp_bab_time": 5.0, "lifted_lp_bab_max_nodes": 6}
    run_lifted_bab(h, l, u, torch.tensor([5.0]), cfg, 100.0)
    first = h.lifted_bab_open
    assert first is not None and 'continuing' not in capsys.readouterr().out
    run_lifted_bab(h, l, u, torch.tensor([4.0]), cfg, 100.0)
    assert f"continuing from {int((first[2] < 4.0).sum())} of {len(first[2])} open leaves" \
        in capsys.readouterr().out


def test_later_lifted_call_runs_only_while_the_lifted_search_is_the_frontier(capsys):
    from types import SimpleNamespace
    from input_split.lifted_bound import run_lifted_bab
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    rr = (rhs[:, 1:] + TOL).clone()
    rr[0, 3] = float('inf')
    h = SimpleNamespace(qtb=None, lifted=lp, lifted_obbt_rows=(rr, None, None))
    cfg = {"lifted_lp_bab_time": 5.0, "lifted_lp_bab_max_nodes": 6}
    run_lifted_bab(h, l, u, torch.tensor([5.0]), cfg, 100.0)
    D = h.lifted_bab_D
    capsys.readouterr()
    # the BaB's bound passed the lifted D: no call, nothing consumed
    assert run_lifted_bab(h, l, u, torch.tensor([4.0]), cfg, 100.0, bab_lb=D + 0.1) is None
    assert h.lifted_bab_cutoff == 5.0 and 'lifted-LP B&B:' not in capsys.readouterr().out
    # level with it (within the certification tolerance) or behind it: the call runs
    run_lifted_bab(h, l, u, torch.tensor([4.0]), cfg, 100.0, bab_lb=D + 1e-6)
    assert 'lifted-LP B&B:' in capsys.readouterr().out
    # Isolate the frontier guard from the separate measured-progress guard.
    h.lifted_bab_progress = (1.0, 1.0)
    run_lifted_bab(h, l, u, torch.tensor([3.0]), cfg, 100.0, bab_lb=D - 1.0)
    assert 'lifted-LP B&B:' in capsys.readouterr().out


class TouchingModel(nn.Module):
    """min x0 x1 - 2 x1 s.t. x0 >= 1, x0 x1 <= 0.5 on [0, 1]^2: the feasible set touches the box
    at x0 = 1 and the optimum is -0.5 at (1, 0.5)."""

    def forward(self, x):
        x0, x1 = x[:, 0:1], x[:, 1:2]
        return torch.cat([x0 * x1 - 2.0 * x1, 1.0 - x0, x0 * x1 - 0.5], dim=-1)


def test_node_that_touches_its_box_is_not_declared_empty():
    torch.set_default_dtype(torch.float64)
    m = TouchingModel()
    x = torch.zeros(1, 2)
    C = torch.eye(3).unsqueeze(0)
    l, u = torch.zeros(1, 2), torch.ones(1, 2)
    lp = build_lifted_lp(BoundedModule(m, x), m, x, C, l, u)
    rr = np.full(2, TOL)
    for cutoff in (0.0, -0.4, -0.5):
        bound, z, nl, nu, _ = lp.solve_bab_node(np.zeros(2), np.ones(2), rr, None, None, cutoff)
        assert bound < float('inf') and nl[0] <= 1.0 <= nu[0] and nl[1] <= 0.5 <= nu[1]
    bound, *_ = lp.solve_bab_node(np.zeros(2), np.ones(2), rr, None, None, -0.51)
    assert bound == float('inf')        # nothing is that good: the cutoff row empties it


def test_primal_fn_lowers_the_cutoff_on_the_fly():
    """A verified point from a local solve (offered the best node's point at node 1) drops the
    cutoff: open leaves of bound >= it are pruned and D stays below every feasible objective."""
    from nlp_solver import lifted_lp
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    rr = rhs[:, 1:] + TOL
    g = torch.Generator().manual_seed(5)
    z = l + (u - l) * torch.rand(400000, 4, generator=g)
    feas, f = _feasible(m, C, rhs, z, [1, 2, 3, 5])
    best = float(f[feas].min())
    offered = []

    def primal_fn(p):
        offered.append(p)
        return best

    old, lifted_lp._LOCAL_NODES = lifted_lp._LOCAL_NODES, (1, 8)
    try:
        D, n, closed, leaves = lp.branch_and_bound(l, u, rr, cutoff=best + 1.0, primal_fn=primal_fn,
                                                   time_budget=10.0, max_nodes=40, return_leaves=True)
    finally:
        lifted_lp._LOCAL_NODES = old
    assert len(offered) == 1 and offered[0].shape == (4,)       # one improving local start only
    assert (l.numpy() - 1e-9 <= offered[0]).all() and (offered[0] <= u.numpy() + 1e-9).all()
    assert D <= best + 1e-9 and (leaves[2] < best).all()
