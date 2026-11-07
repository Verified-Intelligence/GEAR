"""Outputs of degree > 2 enter the lifted LP through their exact polynomial (structure/forms.py)
and auxiliary product columns (nlp_solver/lifted_lp.py). Validity is checked by sampling: no point
of a box that satisfies the relaxed rows may have an objective below the certified bound."""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from auto_LiRPA import BoundedModule  # noqa: E402
from nlp_solver.lifted_lp import _column_boxes, _levels, _liftable_forms, build_lifted_lp  # noqa: E402

TOL = 1e-4


class QuarticModel(nn.Module):
    """[quartic objective, cubic row, linear row, tanh row] on 4 variables."""

    def forward(self, x):
        x0, x1, x2, x3 = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        q = x0 * x0 + x0 + 1.0
        obj = q * q - 2.0 * x0 * x1 * x1 * x2 + x3 ** 4 + x1 * x2 * x3 - 0.5 * x0 * x3
        cubic = x0 ** 3 + x1 * x2 - x3 * x3 * x0 - 0.5
        linear = x0 + x1 + x2 + x3 - 2.0
        return torch.cat([obj, cubic, linear, torch.tanh(x2) - 0.5], dim=-1)


def _setup():
    torch.set_default_dtype(torch.float64)
    m = QuarticModel()
    x = torch.zeros(1, 4)
    bm = BoundedModule(m, x)
    C = torch.eye(4).unsqueeze(0)
    rhs = torch.tensor([[0.0, 0.2, 0.5, 0.0]])
    l = torch.tensor([[-1.0, -1.5, -0.5, -1.0]])
    u = torch.tensor([[1.5, 1.0, 1.5, 1.5]])
    return m, bm, x, C, rhs, l, u


def test_high_degree_outputs_get_product_columns_and_exact_rows():
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    assert lp is not None and lp.obj is not None
    assert lp.row_spec.tolist() == [1, 2]           # the tanh row is not liftable
    assert int(lp.a.max()) >= 4 and (lp.a < lp.n + torch.arange(lp.K)).all()   # product factors
    g = torch.Generator().manual_seed(0)
    z = l + (u - l) * torch.rand(300, 4, generator=g)
    lz, _ = _column_boxes(z, z, lp.a, lp.b, lp.levels, torch)
    k0, g0, q0 = lp.obj
    rep = k0 + z @ g0 + lz[:, 4:] @ q0
    assert torch.allclose(rep, m(z)[:, 0], rtol=1e-11, atol=1e-11)
    for r, s in enumerate(lp.row_spec.tolist()):
        k, gr, q = (t[r] for t in lp.rows)
        assert torch.allclose(k + z @ gr + lz[:, 4:] @ q, m(z)[:, s], rtol=1e-11, atol=1e-11)


def test_column_boxes_enclose_every_monomial():
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    g = torch.Generator().manual_seed(1)
    lo = l + (u - l) * torch.rand(8, 4, generator=g) * 0.6
    hi = lo + (u - lo) * torch.rand(8, 4, generator=g)
    lz, uz = _column_boxes(lo, hi, lp.a, lp.b, lp.levels, torch)
    for i in range(8):
        pts = lo[i] + (hi[i] - lo[i]) * torch.rand(2000, 4, generator=g)
        cols = torch.cat([pts, torch.zeros(2000, lp.K)], 1)
        for ks in _levels(lp.a, lp.b, 4):
            cols[:, 4 + ks] = cols[:, lp.a[ks]] * cols[:, lp.b[ks]]
        assert (cols >= lz[i] - 1e-12).all() and (cols <= uz[i] + 1e-12).all()


@pytest.mark.parametrize('rlt', [False, True])
def test_bound_is_below_every_feasible_point_on_random_boxes(rlt):
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    rr = rhs[:, 1:3] + TOL
    if rlt:
        lp.add_rlt(rr, torch.tensor([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]]),
                   torch.tensor([2.5, -2.5]))
    g = torch.Generator().manual_seed(3)
    B = 60
    lo = l + (u - l) * torch.rand(B, 4, generator=g) * 0.7
    hi = lo + (u - lo) * (0.1 + 0.9 * torch.rand(B, 4, generator=g))
    if rlt:
        eg = torch.tensor([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]]).expand(B, -1, -1)
        eh = torch.tensor([2.5, -2.5]).expand(B, -1)
    else:
        eg = eh = None
    bnd, _ = lp.bound(lo, hi, rr.expand(B, -1), eg, eh, pdhg_iters=0, exact_max=B)
    feasible = 0
    for i in range(B):
        z = lo[i] + (hi[i] - lo[i]) * torch.rand(20000, 4, generator=g)
        if rlt:             # onto the hyperplane sum = 2.5, keeping the points still in the box
            z = z - (z.sum(1, keepdim=True) - 2.5) / 4
            z = z[((z >= lo[i]) & (z <= hi[i])).all(1)]
        y = m(z)
        ok = (y[:, 1] <= rhs[0, 1] + TOL) & (y[:, 2] <= rhs[0, 2] + TOL)
        if ok.any():
            feasible += 1
            assert float(y[ok, 0].min()) >= float(bnd[i]) - 1e-9, i
    assert feasible >= (10 if rlt else 30)


def test_lifted_branch_and_bound_stays_below_the_sampled_minimum():
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    rr = rhs[:, 1:3] + TOL
    D, nodes, _ = lp.branch_and_bound(l, u, rr, time_budget=5.0, max_nodes=200)
    root, _ = lp.bound(l, u, rr, pdhg_iters=0, exact_max=1)
    g = torch.Generator().manual_seed(5)
    z = l + (u - l) * torch.rand(400000, 4, generator=g)
    y = m(z)
    ok = (y[:, 1] <= rhs[0, 1] + TOL) & (y[:, 2] <= rhs[0, 2] + TOL)
    assert nodes > 1 and D >= float(root[0]) - 1e-9
    assert D <= float(y[ok, 0].min()) + 1e-9


def test_degree_two_models_keep_input_factors_only():
    torch.set_default_dtype(torch.float64)

    class Quad(nn.Module):
        def forward(self, x):
            return torch.cat([x[:, 0:1] * x[:, 1:2] + x[:, 2:3] ** 2, x[:, 0:1] + x[:, 1:2]], 1)

    x = torch.zeros(1, 3)
    lp = build_lifted_lp(BoundedModule(Quad(), x), Quad(), x, torch.eye(2).unsqueeze(0),
                         -torch.ones(1, 3), torch.ones(1, 3))
    assert int(lp.a.max()) < 3 and int(lp.b.max()) < 3 and len(lp.levels) == 1
    assert lp.row_error is None and lp.obj_error is None


class RoundedConstant(nn.Module):
    """[x0, sqrt(2) x0 + (x0 x1 + x2)^2]: a rounded constant, and a pow atom whose expansion is a polynomial"""

    def forward(self, x):
        s = (x[:, :1] * 0 + 2.0) ** 0.5
        return torch.cat([x[:, :1], s * x[:, :1] + (x[:, :1] * x[:, 1:2] + x[:, 2:3]) ** 2], 1)


def test_a_rounded_constant_is_not_lifted_as_an_exact_coefficient():
    torch.set_default_dtype(torch.float64)
    x = torch.full((1, 3), 0.5)
    bm = BoundedModule(RoundedConstant(), x)
    found, _ = _liftable_forms(bm, x, torch.zeros(1, 3), torch.ones(1, 3), [1], [False, False])
    assert found == {}          # expand() keeps the rounded flag, and a rounded form without atoms is not lifted
