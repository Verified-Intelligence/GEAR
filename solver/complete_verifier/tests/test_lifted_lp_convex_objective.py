"""Structural recognition, outward epigraph cuts and independent primal checks."""
import os
import sys
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from auto_LiRPA import BoundedModule
from nlp_solver.convex_objective import Atom, recognize_convex_objective
from nlp_solver.lifted_lp import build_lifted_lp


class Model(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        f = self.fn(x).reshape(x.shape[0], -1)
        row = (x[:, :1] * x[:, 1:2] - 1.)
        return torch.cat([f, row, -row], -1)


def graph(fn, lower=.5, upper=2.):
    l, u = torch.full((1, 2), lower, dtype=torch.float64), torch.full((1, 2), upper, dtype=torch.float64)
    x = (l+u)/2
    model = Model(fn)
    bm = BoundedModule(model, x, bound_opts={'pow_negative_exponent': True})
    C = torch.eye(3, dtype=torch.float64)[None]
    return model, bm, x, C, l, u


Q = lambda x: 1. + x[:, 0] * x[:, 1]
ACCEPT = [
    lambda x: Q(x).pow(-.6),
    lambda x: (7. / Q(x)).pow(.6) * 1300.,
    lambda x: Q(x).pow(2.7) + x[:, 0].square() - 3. * x[:, 1],
    lambda x: Q(x).exp(),
    lambda x: -Q(x).log(),
    lambda x: Q(x).reciprocal(),
    lambda x: (Q(x).pow(-.6) + (x[:, 0]+2.).exp()),
    lambda x: (x*x).pow(2).mean(-1),
    lambda x: F.linear(x.pow(-.6), torch.tensor([[2., 3.]], dtype=x.dtype)),
]


@pytest.mark.parametrize('fn', ACCEPT)
def test_recognition_accepts_structural_convex_atoms(fn):
    model, bm, x, C, l, u = graph(fn)
    assert recognize_convex_objective(bm, x, C, l, u) is not None
    lp = build_lifted_lp(bm, model, x, C, l, u, convex_objective=True)
    assert lp is not None and lp.epigraph


@pytest.mark.parametrize('fn,lower,upper', [
    (lambda x: -Q(x).exp(), .5, 2.),
    (lambda x: Q(x).pow(.6), .5, 2.),
    (lambda x: Q(x).log(), .5, 2.),
    (lambda x: (x[:, 0]*x[:, 1]*x[:, 0]).exp(), .5, 2.),
    (lambda x: Q(x).exp() * x[:, 0], .5, 2.),
    (lambda x: x[:, 0].reciprocal(), -1., 1.),
    (lambda x: x[:, 0].pow(-.6), -1., 1.),
    (lambda x: -x[:, 0].log(), -1., 1.),
    (lambda x: Q(x).sin(), .5, 2.),
    (lambda x: (1000.*Q(x)).exp(), .5, 2.),
])
def test_recognition_rejects_unproved_functions_and_domains(fn, lower, upper):
    model, bm, x, C, l, u = graph(fn, lower, upper)
    assert recognize_convex_objective(bm, x, C, l, u) is None


def test_default_is_off_and_signed_spec_is_checked():
    model, bm, x, C, l, u = graph(ACCEPT[0])
    lp = build_lifted_lp(bm, model, x, C, l, u)
    assert lp is not None and lp.obj is None and not lp.epigraph
    C[:, 0] *= -1
    assert recognize_convex_objective(bm, x, C, l, u) is None


def test_float32_keeps_the_existing_path_without_narrowing_cuts():
    model = Model(ACCEPT[0])
    l, u = torch.full((1, 2), .5), torch.full((1, 2), 2.)
    x = (l+u)/2
    bm = BoundedModule(model, x, bound_opts={'pow_negative_exponent': True})
    lp = build_lifted_lp(bm, model, x, torch.eye(3)[None], l, u, convex_objective=True)
    assert lp is None or not lp.epigraph


@pytest.mark.parametrize('fn', ACCEPT[:6])
def test_tangent_rows_stay_below_sampled_function(fn):
    model, bm, x, C, l, u = graph(fn)
    lp = build_lifted_lp(bm, model, x, C, l, u, convex_objective=True)
    data = lp.sparse_lp(l[0].numpy(), u[0].numpy(), np.array([10., 10.]))
    _, _, _, _, lz, uz = data
    generator = torch.Generator().manual_seed(54)
    points = l + (u-l) * torch.rand(1000, 2, generator=generator, dtype=torch.float64)
    for j, term in enumerate(lp.epigraph):
        column = lp.n + lp.K + j
        z = np.zeros((len(points), lp.N))
        z[:, :2] = points.numpy()
        z[:, 2:2+lp.K] = (points[:, lp.a] * points[:, lp.b]).numpy()
        q = term.constant + z[:, :len(term.coefficients)] @ term.coefficients
        z[:, column] = np.array([sum(term.atom.intervals(float(v), float(v))[0])/2 for v in q])
        for anchor in term.points():
            row, rhs = term.cut(float(anchor), lz, uz, column)
            assert np.max(z @ row - rhs) <= 0


@pytest.mark.parametrize('seed', range(5))
def test_random_lifted_bounds_and_obbt_preserve_feasible_points(seed):
    generator = torch.Generator().manual_seed(seed)
    a, b, c = (.1 + torch.rand(3, generator=generator, dtype=torch.float64)).tolist()
    fn = lambda x: c*(a + b*x[:, 0]*x[:, 1]).pow(-.6) + .2*x[:, 0].square()
    model, bm, x, C, l, u = graph(fn)
    lp = build_lifted_lp(bm, model, x, C, l, u, convex_objective=True)
    points = l + (u-l)*torch.rand(500, 2, generator=generator, dtype=torch.float64)
    points = points[(points.prod(1)-1.).abs() <= .6]
    rhs = torch.tensor([[.6, .6]], dtype=torch.float64)
    P = float(model(points)[:, 0].min())
    D, z = lp.solve_node(*lp.sparse_lp(l[0].numpy(), u[0].numpy(), rhs[0].numpy()))
    assert np.isfinite(D) and z is not None and D <= P
    bnd, _, _ = lp.bound(l, u, rhs, pdhg_iters=20, return_points=True)
    assert float(bnd[0]) <= P
    keep = points[model(points)[:, 0] <= P + .1]
    nl, nu = lp.obbt(l, u, rhs, cutoff=P+.1, rounds=1, time_budget=2.)
    assert ((keep >= nl) & (keep <= nu)).all()


def test_dense_sparse_rlt_cutoff_and_adaptation():
    model, bm, x, C, l, u = graph(lambda x: (7./Q(x)).pow(.6)*1300.)
    lp = build_lifted_lp(bm, model, x, C, l, u, convex_objective=True)
    extra_G = torch.tensor([[[1., 1.], [-1., -1.]]], dtype=torch.float64)
    extra_h = torch.tensor([[2., -2.]], dtype=torch.float64)
    rhs = torch.tensor([[.1, .1]], dtype=torch.float64)
    lp.add_rlt(rhs, extra_G, extra_h)
    dense = lp.assemble(l, u, rhs, extra_G, extra_h, cutoff=torch.tensor([10000.]))
    sparse = lp.sparse_lp(l[0].numpy(), u[0].numpy(), rhs[0].numpy(),
                          extra_G[0].numpy(), extra_h[0].numpy(), cutoff=10000.)
    for j in (0, 1, 3, 4, 5):
        np.testing.assert_allclose(dense[j][0].numpy(), sparse[j], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(dense[2][0].numpy(), sparse[2].toarray(), rtol=1e-12, atol=1e-10)
    D, z = lp.solve_node(*sparse, presolve=False)
    assert D <= float(model(torch.ones_like(x))[0, 0])
    term = lp.epigraph[0]
    q = term.constant + term.coefficients @ z[:len(term.coefficients)]
    assert term.atom.intervals(q, q)[0][0] - z[lp.n+lp.K] < 1e-3
    D, nodes, closed = lp.branch_and_bound(l, u, rhs, extra_G, extra_h, cutoff=10000.,
                                         time_budget=2., max_nodes=10)
    assert D <= float(model(torch.ones_like(x))[0, 0]) and nodes <= 10


def test_interval_slope_and_value_enclosures():
    import mpmath as mp
    # Independently evaluated analytic derivative of the reciprocal-power chain.
    atom = Atom((), (('pow', Fraction(-1)), ('scale', Fraction(7)), ('pow', Fraction(.6))))
    for z in np.geomspace(.01, 100., 100):
        value, derivative, _ = atom.intervals(float(z), float(z))
        with mp.workdps(80):
            f = (mp.mpf(7)/mp.mpf(float(z)))**mp.mpf(.6)
            slope = -mp.mpf(.6)*f/mp.mpf(float(z))
            assert mp.mpf(float(value[0])) <= f <= mp.mpf(float(value[1]))
            assert mp.mpf(float(derivative[0])) <= slope <= mp.mpf(float(derivative[1]))


def test_affine_inner_objective_without_products_and_retained_cut_cap():
    model = Model(lambda x: x[:, 0].exp())
    l, u = torch.zeros(1, 2, dtype=torch.float64), torch.ones(1, 2, dtype=torch.float64)
    x = (l+u)/2
    bm = BoundedModule(model, x)
    C = torch.tensor([[[1., 0., 0.]]], dtype=torch.float64)
    lp = build_lifted_lp(bm, model, x, C, l, u, convex_objective=True)
    assert lp.K == 0 and lp.N == 3
    lp.add_rlt(torch.empty(1, 0, dtype=torch.float64),
               torch.tensor([[[1., 1.], [-1., -1.]]], dtype=torch.float64),
               torch.tensor([[1., -1.]], dtype=torch.float64))
    assert lp.rlt is None
    for point in np.linspace(.01, .99, 100):
        lp.epigraph[0].remember(float(point))
    assert len(lp.epigraph[0].points()) == 44 and lp.n_rows() == 44
    D, _ = lp.solve_node(*lp.sparse_lp(l[0].numpy(), u[0].numpy(), np.empty(0)))
    assert .99999 <= D <= 1.


def test_convex_objective_preserves_denominator_clearing():
    class RationalModel(torch.nn.Module):
        def forward(self, x):
            return torch.stack([(1+x[:, 0]*x[:, 1]).pow(-.6), x[:, 0]/x[:, 1]-1], -1)
    model = RationalModel()
    l, u = torch.full((1, 2), .5, dtype=torch.float64), torch.full((1, 2), 2., dtype=torch.float64)
    x = (l+u)/2
    bm = BoundedModule(model, x, bound_opts={'pow_negative_exponent': True})
    C = torch.eye(2, dtype=torch.float64)[None]
    rhs = torch.tensor([[.1]], dtype=torch.float64)
    lp = build_lifted_lp(bm, model, x, C, l, u, convex_objective=True,
                         clear_denominators=True, relaxed_rhs_rows=rhs)
    assert lp.epigraph and lp.row_spec.tolist() == [1]
    assert float(lp.cleared_rhs[0]) == .1
    points = torch.tensor([[.5, .5], [1., 1.], [.5, 2.], [1.1, 1.]], dtype=torch.float64)
    D, _ = lp.solve_node(*lp.sparse_lp(l[0].numpy(), u[0].numpy(), rhs[0].numpy()))
    assert D <= float(model(points)[:, 0].min())


def test_epigraph_preserves_an_existing_tight_objective_plane():
    import scipy.sparse as sp
    model, bm, x, C, l, u = graph(lambda x: x.exp().sum(-1))
    lp = build_lifted_lp(bm, model, x, C, l, u, convex_objective=True)
    eg = torch.tensor([[[1., 1.], [-1., -1.]]], dtype=torch.float64)
    eh = torch.tensor([[2., -2.]], dtype=torch.float64)
    op = (torch.full_like(l, np.e), torch.zeros(1, dtype=torch.float64))
    rhs = torch.full((1, 2), 100., dtype=torch.float64)
    dense = lp.assemble(l, u, rhs, eg, eh, op)
    # Certify only the initial cuts, without the sparse solver's adaptive loop.
    data = [v[0].numpy() for v in dense]
    data[2] = sp.csr_matrix(data[2])
    D = lp.solve_certified(*data)
    assert 2*np.e-1e-8 <= D <= 2*np.e
    sparse = lp.sparse_lp(l[0].numpy(), u[0].numpy(), rhs[0].numpy(),
                          eg[0].numpy(), eh[0].numpy(), (op[0][0].numpy(), 0.))
    np.testing.assert_allclose(dense[2][0].numpy(), sparse[2].toarray())
    np.testing.assert_allclose(dense[3][0].numpy(), sparse[3])
