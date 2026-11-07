"""Structural extraction, outward coefficient errors, and cleared-row certificates."""
import os
import sys
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from auto_LiRPA import BoundedModule
from nlp_solver.structure import RowPool
from nlp_solver.rational_rows import (RationalForm, clear_positive_denominators,
                                      extract_rational_rows, rounded_coefficients)
from nlp_solver.lifted_lp import build_lifted_lp


def _graph(model, n):
    x = torch.ones(1, n, dtype=torch.float64)
    bm = BoundedModule(model, x, bound_opts={'pow_negative_exponent': True})
    C = torch.eye(model(x).numel(), dtype=x.dtype).unsqueeze(0)
    return bm, x, C


class ExtractionModel(nn.Module):
    def forward(self, x):
        a, b, c = x[:, :1], x[:, 1:2], x[:, 2:3]
        return torch.cat([a + b, .3 * a / c + .7 / c - 1,
                          (a * b + .9) / (b * c), -a / c,
                          (a + b) ** 2 / c, c.reciprocal(), c ** -2,
                          a / (b + c), torch.tanh(a), c ** .5, a ** b], -1)


def test_extraction_accepts_laurent_operations_and_rejects_other_operators():
    model = ExtractionModel()
    bm, x, C = _graph(model, 3)
    points = torch.rand(8, 3, dtype=x.dtype) + .5
    forms = extract_rational_rows(bm, model, x, C, points)
    assert [f.terms is not None for f in forms] == [True] * 7 + [False] * 4
    for r in range(7):
        assert torch.allclose(forms[r].evaluate(points)[0], model(points)[:, r])
    # Signed combinations of outputs use C, rather than assuming positive row sense.
    Cs = C[:, :2].clone()
    Cs[0, 1] = 0
    Cs[0, 1, 1], Cs[0, 1, 3] = -2, .1
    signed = extract_rational_rows(bm, model, x, Cs, points)[1]
    assert torch.allclose(signed.evaluate(points)[0], (model(points) @ Cs[0].T)[:, 1])


def test_numerical_mismatch_rejects_the_row():
    model = ExtractionModel()
    bm, x, C = _graph(model, 3)
    points = torch.rand(4, 3, dtype=x.dtype) + .5
    forms = extract_rational_rows(bm, lambda x: model(x) + 1, x, C, points)
    assert all(f.terms is None for f in forms)


def test_supported_rows_survive_unsupported_neighboring_rows():
    model = ExtractionModel()
    bm, x, C = _graph(model, 3)
    rhs = torch.full((1, C.shape[1] - 1), .01, dtype=x.dtype)
    lp = build_lifted_lp(bm, model, x, C, x * .5, x * 2,
                         clear_denominators=True, relaxed_rhs_rows=rhs)
    assert lp.row_spec.tolist() == [1, 2, 3, 4, 5, 6]


def test_generated_graph_operations_and_value_preserving_casts():
    class Model(nn.Module):
        def forward(self, x):
            x = x.to(torch.float64)
            weights = torch.tensor([[.3, .7, 0.]], dtype=x.dtype)
            linear = torch.nn.functional.linear(x, weights)
            indices = torch.tensor([1, 1], dtype=torch.long)
            source = torch.cat([x[:, :1] / x[:, 2:], .1 / x[:, 2:]], 1)
            out = torch.zeros(x.shape[0], 3, dtype=x.dtype).index_add(1, indices, source,
                                                                   alpha=.3)
            return torch.cat([linear, out[:, 1:2]], 1)
    model = Model()
    bm, x, C = _graph(model, 3)
    points = torch.rand(8, 3, dtype=x.dtype) + .5
    forms = extract_rational_rows(bm, model, x, C, points)
    assert all(f.terms is not None for f in forms)
    assert torch.allclose(forms[1].evaluate(points)[0], model(points)[:, 1])


def test_concat_with_a_constant_operand():
    class Model(nn.Module):
        def forward(self, x):
            const = torch.full((x.shape[0], 1), 2.5, dtype=x.dtype)
            return torch.cat([x[:, :1] / x[:, 1:2], const, x[:, 1:2] ** 2], 1)
    model = Model()
    bm, x, C = _graph(model, 2)
    points = torch.rand(8, 2, dtype=x.dtype) + .5
    forms = extract_rational_rows(bm, model, x, C, points)
    assert all(f.terms is not None for f in forms)
    for r in range(3):
        assert torch.allclose(forms[r].evaluate(points)[0], model(points)[:, r])


def test_narrowing_cast_is_rejected(monkeypatch):
    class Model(nn.Module):
        def forward(self, x):
            return (x.to(torch.float32).to(torch.float64))[:, :1]
    model = Model()
    # BoundedModule refuses a float64 model with a float32 node (WP-H); the
    # extraction keeps its own rejection for a graph built without that check.
    with pytest.raises(TypeError, match='lower precision'):
        _graph(model, 2)
    monkeypatch.setattr(BoundedModule, '_check_float64_graph', lambda self: None)
    bm, x, C = _graph(model, 2)
    forms = extract_rational_rows(bm, model, x, C, torch.rand(4, 2, dtype=x.dtype))
    assert forms[0].terms is None


@pytest.mark.parametrize('lower', [-1., 0., float('-inf')])
def test_denominator_with_unproven_positive_sign_is_rejected(lower):
    form = RationalForm({((0, -1),): Fraction(1)})
    assert clear_positive_denominators(form, 0., [lower]) is None


def test_high_degree_or_nonfinite_threshold_is_rejected():
    form = RationalForm({((0, 3), (1, -1)): Fraction(1)})
    assert clear_positive_denominators(form, 0., [1., 1.]) is None
    assert clear_positive_denominators(form, float('inf'), [1., 1.]) is None
    assert clear_positive_denominators(RationalForm(), 0., [1., 1.]) is None


def test_outward_errors_enclose_exact_products_cancellation_and_underflow():
    coeff = Fraction(.1) * Fraction(.3) / Fraction(.7)
    tiny = Fraction(1, 2 ** 1100)
    form = RationalForm({(): coeff, ((0, 1),): tiny})
    for key, (value, error) in rounded_coefficients(form).items():
        assert Fraction(error) >= abs(Fraction(value) - form.terms[key])
    assert rounded_coefficients(form)[((0, 1),)][1] > 0


class RandomRationalModel(nn.Module):
    def __init__(self, seed):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.coeffs = rng.normal(size=(3, 7))

    def forward(self, x):
        a, b, c, d = [x[:, i:i+1] for i in range(4)]
        terms = [torch.ones_like(a), a, b, a * b, a * c, b * d, c * d]
        rows = [sum(float(k) * term for k, term in zip(co, terms)) / (c * d)
                for co in self.coeffs]
        return torch.cat([a + b + c + d, *rows, a * b - 4], -1)


@pytest.mark.parametrize('seed', range(6))
def test_random_cleared_rows_hold_and_bounds_do_not_exceed_feasible_objectives(seed):
    model = RandomRationalModel(seed)
    bm, x, C = _graph(model, 4)
    lo, hi = torch.full_like(x, .5), torch.full_like(x, 2.)
    gen = torch.Generator().manual_seed(seed)
    points = lo + (hi - lo) * torch.rand(4000, 4, generator=gen, dtype=x.dtype)
    values = model(points)
    # Nonzero thresholds and a nonzero original-row tolerance are cleared together.
    rhs = values[:, 1:4].quantile(.8, dim=0).reshape(1, -1)
    rhs = torch.cat([rhs + .17, torch.full((1, 1), .17, dtype=x.dtype)], 1)
    lp = build_lifted_lp(bm, model, x, C, lo, hi, clear_denominators=True,
                         relaxed_rhs_rows=rhs)
    assert lp is not None and lp.row_spec.tolist() == [1, 2, 3, 4]
    assert lp.cleared_rhs[:3].isfinite().all()
    _, _, G, h, _, _ = lp.assemble(lo, hi, rhs)
    lifted_points = torch.cat([points, points[:, lp.a] * points[:, lp.b]], 1)
    feasible = (values[:, 1:] <= rhs).all(1)
    assert feasible.sum() > 50
    assert (lifted_points[feasible] @ G[0].T <= h[0] + 1e-11).all()
    # The cleared polynomial itself equals m*(g-relaxed_rhs) at all sampled points.
    k, g, q = lp.rows
    rep = k + points @ g.T + (points[:, lp.a] * points[:, lp.b]) @ q.T
    assert torch.allclose(rep[:, :3], (values[:, 1:4] - rhs[:, :3]) *
                          (points[:, 2] * points[:, 3]).unsqueeze(1), atol=1e-11)
    bound, _ = lp.bound(lo, hi, rhs, pdhg_iters=0, exact_max=1)
    assert float(bound[0]) <= float(values[feasible, 0].min())
    dense = lp.assemble(lo, hi, rhs)
    sparse = lp.sparse_lp(lo[0].numpy(), hi[0].numpy(), rhs[0].numpy())
    assert np.allclose(dense[2][0].numpy(), sparse[2].toarray())
    assert np.allclose(dense[3][0].numpy(), sparse[3])
    # A looser RHS cannot accidentally reuse the root cleared inequality.
    loose = rhs + 1
    assert torch.isposinf(lp._row_rhs(loose)[0, :3]).all()


class ProductConstraint(nn.Module):
    def forward(self, x):
        a, b = x[:, :1], x[:, 1:2]
        return torch.cat([a + b, 3. / (a * b) - 1.], -1)


def test_cleared_rows_enable_obbt_bab_and_handoff_without_altering_original_model():
    from input_split.lifted_bound import run_lifted_bab
    model = ProductConstraint()
    bm, x, C = _graph(model, 2)
    lo, hi = torch.full_like(x, .5), torch.full_like(x, 4.)
    rhs = torch.tensor([[.05]], dtype=x.dtype)
    # Without clearing, the reciprocals a^-1 b^-1 are atom columns (lifted_atoms) of a product.
    assert build_lifted_lp(bm, model, x, C, lo, hi).atoms.T == 2
    lp = build_lifted_lp(bm, model, x, C, lo, hi, clear_denominators=True,
                         relaxed_rhs_rows=rhs)
    points = lo + (hi - lo) * torch.rand(10000, 2, dtype=x.dtype)
    values = model(points)
    feasible = (values[:, 1] <= .05) & (values[:, 0] <= 4)
    nl, nu = lp.obbt(lo, hi, rhs, cutoff=4., rounds=2, verbose=False)
    assert ((points[feasible] >= nl) & (points[feasible] <= nu)).all()
    history = SimpleNamespace(qtb=None, lifted=lp, lifted_obbt_rows=(rhs, None, None))
    # An open B&B returns leaves for the existing handoff path when every row is usable.
    config = {'lifted_lp_bab_time': 2., 'lifted_lp_bab_max_nodes': 3}
    result = run_lifted_bab(history, lo, hi, 4., config, 20., want_leaves=True)
    assert result is None
    assert history.lifted_bab_leaves[2].size > 0
    exact_opt = 2 * np.sqrt(3 / 1.05)
    D, _, closed = lp.branch_and_bound(lo, hi, rhs, cutoff=exact_opt + 1e-8,
                                      tol_abs=1e-3, tol_rel=0., time_budget=5.)
    assert D <= exact_opt + 1e-8 and closed


def test_nonquadratic_objective_does_not_enable_denominator_clearing():
    class Model(nn.Module):
        def forward(self, x):
            base = ProductConstraint()(x)
            return torch.cat([base[:, :1].exp(), base[:, 1:]], 1)
    model = Model()
    bm, x, C = _graph(model, 2)
    lp = build_lifted_lp(bm, model, x, C, x * .5, x * 4, clear_denominators=True,
                         relaxed_rhs_rows=x[:, :1] * .05)
    assert lp is None or lp.cleared_rhs is None     # the atom columns are no cleared denominator


def test_linear_cleared_rows_work_with_qtb_obbt_and_bab():
    import arguments
    from input_split.lifted_bound import setup_lifted_lp, run_lifted_bab
    class Model(nn.Module):
        def forward(self, x):
            return torch.cat([x, 3. / x - 1.], 1)
    model = Model()
    bm, x, C = _graph(model, 1)
    lo, hi = x * .5, x * 4
    rhs = torch.tensor([[4., 0.]], dtype=x.dtype)
    config = {'clear_denominators': True, 'lifted_lp_bound': True,
              'lifted_lp_obbt': True, 'lifted_lp_bab_time': 1.}
    solving = arguments.Config['solving']['solving_mode']
    tolerance = arguments.Config['solver']['nlp']['constr_viol_tol']
    try:
        arguments.Config['solving']['solving_mode'] = True
        arguments.Config['solver']['nlp']['constr_viol_tol'] = .05
        history = SimpleNamespace(qtb=SimpleNamespace(lam_lb=0.), rowpool=RowPool(rhs))
        nl, nu = setup_lifted_lp(SimpleNamespace(net=bm, model_ori=model), history,
                                 lo, hi, C, rhs, None, config)
        assert history.lifted is not None and history.lifted.K == 0
        optimum = 3 / 1.05
        assert optimum - 1e-6 <= float(nl[0, 0]) <= optimum
        assert float(nu[0, 0]) >= optimum
        D = run_lifted_bab(history, nl, nu, optimum + 1e-8, config, 10.)
        assert D is not None and D <= optimum + 1e-8
    finally:
        arguments.Config['solving']['solving_mode'] = solving
        arguments.Config['solver']['nlp']['constr_viol_tol'] = tolerance
