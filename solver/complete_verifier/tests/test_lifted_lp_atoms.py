"""Atoms (exp, log, real powers, 1/x of an affine argument) enter the lifted LP as columns with
tangent and secant rows (nlp_solver/lifted_atoms.py). Validity is checked directly (every true
lifted point satisfies every atom row; the column boxes enclose mpmath values) and through the
LP bound against dense sampling."""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from auto_LiRPA import BoundedModule  # noqa: E402
from nlp_solver import lifted_atoms as la  # noqa: E402
from nlp_solver.lifted_lp import _column_boxes, build_lifted_lp  # noqa: E402

TOL = 1e-4


class AtomModel(nn.Module):
    """[objective, row, linear row, tanh row, non-affine exp row] on 4 variables."""

    def forward(self, x):
        x0, x1, x2, x3 = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        obj = (x0 * torch.exp(x1) + x2 * (x0 + x3 + 2.0) ** -0.3 - torch.log(x1 + 2.0)
               + 1.0 / (x0 - 3.0) + (x2 + 1.0) ** 0.5 * x3 + torch.exp(-2.0 / (x2 + x3 + 1.0))
               + 0.1 * x1 * x2 ** -1.5 + 0.05 * x3 ** -1.0)
        r1 = 1.0 / (x2 + x3 + 1.0) + x0 * x3 - 0.3
        r2 = x0 + x1 + x2 + x3 - 2.0
        return torch.cat([obj, r1, r2, torch.tanh(x2) - 0.5, torch.exp(x0 * x1) - 3.0], dim=-1)


def _setup():
    torch.set_default_dtype(torch.float64)
    m = AtomModel()
    x = torch.zeros(1, 4)
    C = torch.eye(5).unsqueeze(0)
    l = torch.tensor([[-1.0, -1.0, 0.5, 0.2]])
    u = torch.tensor([[1.0, 1.0, 1.5, 1.5]])
    return m, BoundedModule(m, x), x, C, l, u


def _lifted_point(lp, x):
    """The true column values (inputs, atoms, products) at the input points x (P, n)."""
    atoms, n = lp.atoms, lp.n
    c = atoms.arrays()
    z = torch.cat([x, torch.zeros(x.shape[0], lp.N - n)], 1).numpy()
    for j in range(atoms.T):
        arg = z[:, :n + j] @ c['A'][j, :n + j] + c['a0'][j]
        z[:, n + j] = la._phi(c['code'][j:j + 1], c['expo'][j:j + 1], arg[:, None, None], np)[0][:, 0, 0]
    for k in range(lp.K):       # factors are inputs, atoms or earlier products
        z[:, lp.nb + k] = z[:, int(lp.a[k])] * z[:, int(lp.b[k])]
    return torch.as_tensor(z)


def test_atoms_are_lifted_with_the_right_curvature_and_unliftable_rows_are_not():
    m, bm, x, C, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    assert lp is not None and lp.obj is not None
    assert lp.row_spec.tolist() == [1, 2]           # tanh and exp(x0 x1) are not liftable
    c = lp.atoms.arrays()
    # exp(x1), (x0+x3+2)^-0.3, log(x1+2), 1/(x0-3) < 0, sqrt, 1/(x2+x3+1), exp(-2 s), x2^-1.5, 1/x3
    assert sorted(zip(c['code'].tolist(), c['expo'].tolist(), c['kappa'].tolist())) == sorted([
        (la.EXP, 0.0, 1), (la.POW, -0.3, 1), (la.LOG, 0.0, -1), (la.POW, -1.0, -1),
        (la.POW, 0.5, -1), (la.POW, -1.0, 1), (la.EXP, 0.0, 1), (la.POW, -1.5, 1),
        (la.POW, -1.0, 1)])
    assert lp.nb == 4 + 9 and lp.N == lp.nb + lp.K


def test_a_domain_that_is_not_proved_is_not_lifted():
    class Model(nn.Module):
        def forward(self, x):
            return torch.cat([x[:, 0:1] * x[:, 1:2], torch.log(x[:, 0:1] + 0.5) + x[:, 1:2],
                              torch.log(x[:, 0:1] + 1.5) * x[:, 1:2],
                              (x[:, 0:1] * x[:, 1:2] + 3.0) ** 0.5], dim=-1)
    torch.set_default_dtype(torch.float64)
    x = torch.zeros(1, 2)
    C = torch.eye(4).unsqueeze(0)
    lp = build_lifted_lp(BoundedModule(Model(), x), Model(), x, C, -torch.ones(1, 2), torch.ones(1, 2))
    assert lp is not None
    assert lp.row_spec.tolist() == [2]      # log(x0 + 0.5) has no positive range, the sqrt a product


@pytest.mark.parametrize('code, expo, lo, hi', [(la.EXP, 0.0, -30.0, 25.0), (la.LOG, 0.0, 1e-8, 1e6),
                                                (la.POW, -0.1, 1e-6, 1e4), (la.POW, 0.3, 1e-9, 1e3),
                                                (la.POW, -1.0, -100.0, -1e-3), (la.POW, 1.8, 0.01, 50.0)])
def test_value_boxes_enclose_the_exact_values(code, expo, lo, hi):
    import mpmath
    mpmath.mp.dps = 60
    rng = np.random.default_rng(0)
    xs = np.sort(rng.uniform(lo, hi, (2, 300)), 0) if lo < 0 else np.exp(np.sort(
        rng.uniform(np.log(lo), np.log(hi), (2, 300)), 0))
    tl, tu = la._value_box(np.array([code]), np.array([expo]), xs[0][:, None], xs[1][:, None], np)
    f = {la.EXP: mpmath.exp, la.LOG: mpmath.log}.get(code, lambda v: mpmath.power(v, mpmath.mpf(expo)))
    for i in range(300):
        for end in (0, 1):
            exact = f(mpmath.mpf(float(xs[end][i])))
            assert mpmath.mpf(float(tl[i, 0])) <= exact <= mpmath.mpf(float(tu[i, 0]))


def test_column_boxes_and_atom_rows_hold_at_every_true_lifted_point():
    m, bm, x, C, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    g = torch.Generator().manual_seed(0)
    B = 60
    lo = l + (u - l) * torch.rand(B, 4, generator=g) * 0.7
    hi = lo + (u - lo) * (0.05 + 0.95 * torch.rand(B, 4, generator=g))
    lz, uz = _column_boxes(lo, hi, lp.a, lp.b, lp.levels, torch, lp.atoms)
    G, h = lp.atoms.dense(lz, uz, lp.N)
    assert G.shape == (B, lp.atoms.n_rows, lp.N)
    for i in range(B):
        pts = lo[i] + (hi[i] - lo[i]) * torch.rand(300, 4, generator=g)
        z = _lifted_point(lp, pts)
        assert (z >= lz[i] - 1e-12).all() and (z <= uz[i] + 1e-12).all()
        assert (z @ G[i].T <= h[i] + 1e-12).all()
        # the sparse single-box assembly is the same rows
        Gs, hs = lp.atoms.sparse(lz[i].numpy(), uz[i].numpy(), lp.N)
        assert np.allclose(Gs.toarray(), G[i].numpy()) and np.allclose(hs, h[i].numpy())


def test_narrow_boxes_give_narrow_atom_columns():
    m, bm, x, C, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    lo = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
    hi = lo + 1e-3
    lz, uz = _column_boxes(lo, hi, lp.a, lp.b, lp.levels, torch, lp.atoms)
    assert (uz - lz)[0, 4:4 + lp.atoms.T].max() < 1e-2


@pytest.mark.parametrize('pdhg', [False, True])
def test_bound_is_below_the_sampled_minimum_on_random_boxes(pdhg):
    m, bm, x, C, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    g = torch.Generator().manual_seed(2)
    B = 50
    lo = l + (u - l) * torch.rand(B, 4, generator=g) * 0.7
    hi = lo + (u - lo) * (0.1 + 0.9 * torch.rand(B, 4, generator=g))
    rhs = torch.tensor([[0.0, 0.5]]) + TOL
    bnd, _ = lp.bound(lo, hi, rhs.expand(B, -1), pdhg_iters=800 if pdhg else 0,
                      exact_max=0 if pdhg else B)
    feasible = 0
    for i in range(B):
        z = lo[i] + (hi[i] - lo[i]) * torch.rand(20000, 4, generator=g)
        y = m(z)
        ok = (y[:, 1] <= rhs[0, 0]) & (y[:, 2] <= rhs[0, 1])
        if ok.any():
            feasible += 1
            assert float(y[ok, 0].min()) >= float(bnd[i]) - 1e-9, i
    assert feasible >= 25


def test_lifted_branch_and_bound_and_branching_score():
    m, bm, x, C, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    rhs = torch.tensor([[0.0, 0.5]]) + TOL
    D, nodes, _ = lp.branch_and_bound(l, u, rhs, time_budget=5.0, max_nodes=100)
    g = torch.Generator().manual_seed(5)
    z = l + (u - l) * torch.rand(400000, 4, generator=g)
    y = m(z)
    ok = (y[:, 1] <= rhs[0, 0]) & (y[:, 2] <= rhs[0, 1])
    assert nodes > 1 and D <= float(y[ok, 0].min()) + 1e-9
    _, _, Z = lp.bound(l, u, rhs, pdhg_iters=200, exact_max=1, return_points=True)
    score = lp.violation_score(Z)
    assert score.shape == (1, 4) and bool(torch.isfinite(score).all())


def test_atom_rows_are_not_mistaken_for_linear_rows():
    from input_split.lifted_bound import _qtb_covers_linear_rows
    m, bm, x, C, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)

    class History:
        qtb = type('Q', (), {'lam_lb': 0.0})()
    assert not _qtb_covers_linear_rows(History, lp)


def test_rlt_with_atoms_stays_valid():
    m, bm, x, C, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    rhs = torch.tensor([[1.0, 5.0]]) + TOL          # the sum row is slack: the equality below binds
    eg = torch.tensor([[1.0] * 4, [-1.0] * 4])
    eh = torch.tensor([2.5, -2.5])
    lp.add_rlt(rhs, eg, eh)
    assert lp.n_rlt > 0 and lp.N == lp.nb + lp.K
    g = torch.Generator().manual_seed(4)
    B = 20
    lo = l + (u - l) * torch.rand(B, 4, generator=g) * 0.5
    hi = lo + (u - lo) * (0.3 + 0.7 * torch.rand(B, 4, generator=g))
    bnd, _ = lp.bound(lo, hi, rhs.expand(B, -1), eg.expand(B, -1, -1), eh.expand(B, -1),
                      pdhg_iters=0, exact_max=B)
    feasible = 0
    for i in range(B):
        z = lo[i] + (hi[i] - lo[i]) * torch.rand(40000, 4, generator=g)
        z = z - (z.sum(1, keepdim=True) - 2.5) / 4      # onto the hyperplane, then back into the box
        z = z[((z >= lo[i]) & (z <= hi[i])).all(1)]
        if len(z):
            y = m(z)
            ok = (y[:, 1] <= rhs[0, 0]) & (y[:, 2] <= rhs[0, 1])
            if ok.any():
                feasible += 1
                assert float(y[ok, 0].min()) >= float(bnd[i]) - 1e-9, i
    assert feasible >= 5


# ---- sin and cos: lines on both sides over any range --------------------------------------
class TrigModel(nn.Module):
    """[objective, row] on 3 variables: sin and cos of affine arguments, and an atom times inputs."""

    def forward(self, x):
        x0, x1, x2 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        obj = (torch.sin(x0 - x1) + 0.5 * torch.cos(2 * x2 + x0 + 0.3) + x0 * torch.sin(x1)
               - 0.2 * x2 * torch.cos(x0 + x2))
        return torch.cat([obj, torch.cos(x1 - 2 * x2) - 0.2], dim=-1)


def _trig_setup():
    torch.set_default_dtype(torch.float64)
    m, x = TrigModel(), torch.zeros(1, 3)
    l, u = -5 * torch.ones(1, 3), 5 * torch.ones(1, 3)
    return m, x, l, u, build_lifted_lp(BoundedModule(m, x), m, x, torch.eye(2).unsqueeze(0), l, u)


@pytest.mark.parametrize('xp', [np, torch])
def test_support_lines_are_valid_and_tight_on_any_range(xp):
    """sigma (sin u - m u) <= H(m) on [a, b] for every slope m, side, end slope included:
    widths below 1e-9 and above 2 pi, ranges across inflection points and maxima."""
    rng = np.random.default_rng(1)
    asx = np.asarray if xp is np else torch.as_tensor
    for width in [1e-9, 1e-4, 0.5, 3.0, 3.2, 6.3, 12.0, 40.0]:
        for a in rng.uniform(-30, 30, 25):
            b = a + width
            lo, hi = asx(np.array([a])), asx(np.array([b]))
            u = np.linspace(a, b, 6001)
            for sigma in (1.0, -1.0):
                ends = xp.stack([la._end_slope(lo, hi, sigma, True, xp), la._end_slope(lo, hi, sigma, False, xp)], -1)
                m = np.concatenate([rng.uniform(-1, 1, 6), np.asarray(ends)[0], [-1.0, 1.0, 0.0]])
                H = np.asarray(la._support(asx(m)[None], sigma, lo[..., None], hi[..., None], xp))[0]
                for mi, Hi in zip(m, H):
                    peak = (sigma * (np.sin(u) - mi * u)).max()
                    assert peak <= Hi + 1e-12 * (1 + abs(mi) * abs(u).max())      # valid
                    assert Hi - peak <= 1e-6 * (1 + width)                         # and the tightest of its slope


def test_trig_rows_hold_at_every_true_lifted_point():
    m, x, l, u, lp = _trig_setup()
    assert lp.atoms.T == 5 and lp.atoms.n_rows == 5 * 2 * (len(la._TRIG_TANGENTS) + 2)
    g = torch.Generator().manual_seed(0)
    B = 80
    lo = l + (u - l) * torch.rand(B, 3, generator=g) * 0.9
    width = torch.tensor([1e-6, 0.01, 1.0, 3.0, 7.0, 10.0])[torch.randint(0, 6, (B, 3), generator=g)]
    hi = torch.minimum(lo + width, u)
    lz, uz = _column_boxes(lo, hi, lp.a, lp.b, lp.levels, torch, lp.atoms)
    G, h = lp.atoms.dense(lz, uz, lp.N)
    for i in range(B):
        z = _lifted_point(lp, lo[i] + (hi[i] - lo[i]) * torch.rand(300, 3, generator=g))
        assert (z >= lz[i] - 1e-12).all() and (z <= uz[i] + 1e-12).all()
        assert (z @ G[i].T <= h[i] + 1e-12).all()
        Gs, hs = lp.atoms.sparse(lz[i].numpy(), uz[i].numpy(), lp.N)
        assert np.allclose(Gs.toarray(), G[i].numpy()) and np.allclose(hs, h[i].numpy())


def test_trig_lp_bound_is_below_the_sampled_minimum_on_random_boxes():
    m, x, l, u, lp = _trig_setup()
    g = torch.Generator().manual_seed(3)
    B = 40
    lo = l + (u - l) * torch.rand(B, 3, generator=g) * 0.9
    hi = torch.minimum(lo + 0.05 + 8 * torch.rand(B, 3, generator=g) ** 2, u)
    rhs = torch.tensor([[0.0]]) + TOL
    bnd, _ = lp.bound(lo, hi, rhs.expand(B, -1), pdhg_iters=0, exact_max=B)
    feasible = 0
    for i in range(B):
        y = m(lo[i] + (hi[i] - lo[i]) * torch.rand(30000, 3, generator=g))
        ok = y[:, 1] <= rhs[0, 0]
        if ok.any():
            feasible += 1
            assert float(y[ok, 0].min()) >= float(bnd[i]) - 1e-9, i
    assert feasible >= 20


def test_root_refinement_reaches_the_regular_polygon_bound():
    """maximize sum sin(x_{i+1} - x_i) with the closing arc, x increasing in [0, 2 pi]: the
    interval range of an argument is [-2 pi, 2 pi]; the rows and the LP point tangents give
    n sin(2 pi / n)."""
    n = 8

    class Polygon(nn.Module):
        def forward(self, x):
            d = torch.cat([x[:, 1:] - x[:, :-1], 2 * np.pi - x[:, -1:]], 1)
            return -torch.sin(d).sum(1, keepdim=True) @ torch.ones(1, 1)

    torch.set_default_dtype(torch.float64)
    m, x = Polygon(), torch.zeros(1, n)
    l, u = torch.zeros(1, n), torch.full((1, n), 2 * np.pi)
    lp = build_lifted_lp(BoundedModule(m, x), m, x, torch.ones(1, 1, 1), l, u)
    eg = torch.zeros(1, n - 1, n)
    for i in range(n - 1):
        eg[0, i, i], eg[0, i, i + 1] = 1.0, -1.0
    eh, rhs = torch.zeros(1, n - 1), torch.zeros(1, 0)
    lp.tighten_atoms(l, u, rhs, eg, eh)
    args = (l[0].numpy(), u[0].numpy(), rhs[0].numpy(), eg[0].numpy(), eh[0].numpy())
    c, c0, G, h, lz, uz = lp.sparse_lp(*args)
    bound = lp.solve_certified(c, c0, G, h, lz, uz)
    exact = -n * np.sin(2 * np.pi / n)
    assert bound <= exact + 1e-9 and bound >= exact * (1 + 1e-4)
