"""Structural DCP/domain gates and global tangent/cut validity on random boxes."""
import math
import os
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import arguments
from auto_LiRPA import BoundedModule
from nlp_solver.convex_tangent import (
    AFFINE, CONVEX, CONCAVE, UNKNOWN, ConvexityProof, signed_curvature,
    build_convex_tangent, build_convex_tangent_rows,
)
from nlp_solver.quadratic_tangent import build_quadratic_tangent
from nlp_solver.structure import RowPool


@pytest.fixture(autouse=True)
def config(monkeypatch):
    torch.set_default_dtype(torch.float64)
    cfg = arguments.ConfigHandler()
    cfg.construct_config_dict(cfg.default_args)
    cfg['solving']['solving_mode'] = True
    cfg['solver']['nlp']['constr_viol_tol'] = 1e-5
    monkeypatch.setattr(arguments, 'Config', cfg)
    return cfg


class Function(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x).reshape(x.shape[0], -1)


def graph(fn, lower, upper):
    model = Function(fn)
    lower, upper = torch.tensor([lower]), torch.tensor([upper])
    x = (lower + upper) / 2
    bounded = BoundedModule(model, x, device='cpu', bound_opts={
        'hidden_square_rewrite': True, 'pow_negative_exponent': True,
        'op_budget': {'hidden_squares': 1 << 20, 'complete_squares': 1 << 24, 'prove_rows': 1 << 23,
                      'commit': 'partial'}})
    return model, bounded, x, lower, upper


@pytest.mark.parametrize('fn,lo,hi,expected', [
    (lambda x: x + 2, [-1.], [1.], AFFINE),
    (lambda x: torch.exp(x), [-2.], [2.], CONVEX),
    (lambda x: torch.exp(x.square()), [-2.], [2.], CONVEX),
    (lambda x: x.square(), [-2.], [2.], CONVEX),
    (lambda x: x.pow(4), [-2.], [2.], CONVEX),
    (lambda x: x.pow(3), [0.], [2.], CONVEX),
    (lambda x: x.pow(1.5), [0.], [2.], CONVEX),
    (lambda x: x.pow(3), [-2.], [2.], UNKNOWN),
    (lambda x: x.pow(1.5), [-1.], [2.], UNKNOWN),
    (lambda x: torch.reciprocal(x), [.1], [2.], CONVEX),
    (lambda x: torch.reciprocal(x), [-2.], [-.1], CONCAVE),
    (lambda x: torch.reciprocal(x), [-1.], [2.], UNKNOWN),
    (lambda x: torch.reciprocal(x), [0.], [2.], UNKNOWN),
    (lambda x: -torch.log(x), [.1], [2.], CONVEX),
    (lambda x: torch.log(x), [.1], [2.], CONCAVE),
    (lambda x: -torch.log(x), [0.], [2.], UNKNOWN),
    (lambda x: torch.sqrt(x), [0.], [2.], CONCAVE),
    (lambda x: torch.sqrt(x), [-1.], [2.], UNKNOWN),
    (lambda x: -torch.sqrt(x), [0.], [2.], CONVEX),
    (lambda x: -torch.exp(x), [-1.], [1.], CONCAVE),
    (lambda x: torch.exp(-torch.sqrt(x)), [.1], [2.], CONVEX),
    (lambda x: torch.log(torch.exp(x)), [-1.], [1.], UNKNOWN),
    (lambda x: torch.sin(x), [-1.], [1.], UNKNOWN),
    (lambda x: x[:, :1] * x[:, 1:2], [1., 1.], [2., 2.], UNKNOWN),
    (lambda x: 3*x / (1-.2*x), [0.], [4.], CONVEX),
    (lambda x: -3*x / (1-.2*x), [0.], [4.], CONCAVE),
    (lambda x: 3*x / (1-.2*x), [0.], [6.], UNKNOWN),
    (lambda x: x[:, :1] / (1-.2*x[:, 1:2]), [0., 0.], [4., 4.], UNKNOWN),
])
def test_composition_and_root_domain_rules(fn, lo, hi, expected):
    _, bounded, x, lower, upper = graph(fn, lo, hi)
    assert ConvexityProof(bounded, x, lower, upper).output.tolist() == [expected]


def test_elementwise_signed_combinations_and_movement():
    def fn(x):
        values = torch.cat((torch.exp(x[:, :1]), torch.log(x[:, 1:2]), x[:, 2:3]), 1)
        return torch.nn.functional.linear(values, torch.tensor([[2., -3., 1.], [-1., 2., 0.], [1., 1., 0.], [0., 0., 2.]]))
    _, bounded, x, lower, upper = graph(fn, [-1., .1, -2.], [1., 2., 2.])
    curves = ConvexityProof(bounded, x, lower, upper).output
    assert curves.tolist() == [CONVEX, CONCAVE, UNKNOWN, AFFINE]
    rows = torch.tensor([[1., -1., 0., 0.], [1., 1., 0., 0.], [0., 0., 0., -2.]])
    assert signed_curvature(curves, rows).tolist() == [CONVEX, UNKNOWN, AFFINE]


def test_affine_operations_scatter_mean_and_negative_constants():
    class Scatter(torch.nn.Module):
        def forward(self, x):
            a = -torch.exp(x).reshape(1, -1).transpose(0, 1).transpose(0, 1)
            data = torch.zeros(1, 2).index_add(1, torch.tensor([0, 1]), a, alpha=-2)
            return data.mean(1, keepdim=True)
    m = Scatter()
    x = torch.zeros(1, 2)
    b = BoundedModule(m, x, device='cpu')
    assert ConvexityProof(b, x, x-1, x+1).output.tolist() == [CONVEX]


def test_gemm_alpha_sign_is_part_of_the_proof():
    fn = lambda x: torch.nn.functional.linear(torch.exp(x), torch.ones(1, 1), torch.zeros(1))
    _, bounded, x, lo, hi = graph(fn, [-1.], [1.])
    linear = next(node for node in bounded.nodes() if type(node).__name__ == 'BoundLinear')
    linear.alpha_linear = -2.
    assert ConvexityProof(bounded, x, lo, hi).output.tolist() == [CONCAVE]


def test_gemm_zero_alpha_does_not_hide_a_pole():
    fn = lambda x: torch.nn.functional.linear(torch.reciprocal(x), torch.ones(1, 1), torch.zeros(1))
    _, bounded, x, lo, hi = graph(fn, [-1.], [2.])
    linear = next(node for node in bounded.nodes() if type(node).__name__ == 'BoundLinear')
    linear.alpha_linear = 0.
    assert ConvexityProof(bounded, x, lo, hi).output.tolist() == [UNKNOWN]


def test_domain_comparisons_do_not_round_a_zero_lower_bound_positive():
    # Endpoint cancellation has roundoff uncertainty and cannot prove positivity.
    _, b, x, lo, hi = graph(lambda x: -torch.log(x - .1), [.1], [1.])
    assert ConvexityProof(b, x, lo, hi).output.tolist() == [UNKNOWN]


@pytest.mark.parametrize('fn', [lambda x: 0*torch.reciprocal(x)+torch.exp(x),
                              lambda x: torch.reciprocal(x).pow(0)+torch.exp(x),
                              lambda x: (0*torch.reciprocal(x)).pow(0)+torch.exp(x)])
def test_zero_coefficients_do_not_hide_an_unproved_operator_domain(fn):
    _, b, x, lo, hi = graph(fn, [-1.], [2.])
    assert ConvexityProof(b, x, lo, hi).output.tolist() == [UNKNOWN]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('fn', [
    lambda x: torch.exp(x.square()).sum(1),
    lambda x: -torch.log(x+3).sum(1),
    lambda x: (2*x/(1-.2*x)).sum(1),
    lambda x: x.pow(4).sum(1),
    lambda x: -torch.sqrt(x+3).sum(1),
])
def test_tangent_below_values_on_random_subboxes(fn, dtype):
    model, bounded, x, lower, upper = graph(fn, [-1., -.5], [1., 1.5])
    model = model.to(dtype)
    x, lower, upper = x.to(dtype), lower.to(dtype), upper.to(dtype)
    bounded = BoundedModule(model, x, device='cpu')
    generator = torch.Generator().manual_seed(28)
    anchor = (lower + .37*(upper-lower)).flatten()
    tangent = build_convex_tangent(bounded, model, x, 0, anchor, lower, upper, enabled=True)
    assert tangent is not None
    assert tangent.lam_lb == 0
    for _ in range(6):
        lo = lower + .4*(upper-lower)*torch.rand(lower.shape, generator=generator, dtype=dtype)
        hi = upper - .4*(upper-lower)*torch.rand(lower.shape, generator=generator, dtype=dtype)
        a, b = tangent.plane(lo, hi)
        points = lo + (hi-lo)*torch.rand(500, 2, generator=generator, dtype=dtype)
        values = model.double()(points.double()).flatten()
        assert (points.double() @ a[0].double() + b[0].double() <= values).all()


def test_quadratics_keep_qtb_and_disabled_options_do_no_proof_work(config):
    model, bounded, x, lower, upper = graph(lambda x: x.square().sum(1), [-1.], [1.])
    assert build_quadratic_tangent(bounded, model, x, 0, x.flatten()) is not None
    assert build_convex_tangent(bounded, model, x, 0, x.flatten(), lower, upper, enabled=True) is None
    assert config['solver']['nlp']['convex_tangent_objective'] is False
    assert config['solver']['nlp']['convex_tangent_rows'] is False
    with patch('nlp_solver.convex_tangent.ConvexityProof', side_effect=AssertionError):
        assert build_convex_tangent(None, None, x, 0, x.flatten(), lower, upper) is None
        assert build_convex_tangent_rows(None, None, x, torch.eye(2)[None], torch.zeros(1, 2), lower, upper, x.flatten(),
                                         rowpool=RowPool(torch.zeros(1, 2))) is None


def test_anchor_requires_differentiability_and_root_membership():
    model, bounded, x, lo, hi = graph(lambda x: -torch.sqrt(x), [0.], [2.])
    assert build_convex_tangent(bounded, model, x, 0, torch.zeros(1), lo, hi, enabled=True) is None
    tangent = build_convex_tangent(bounded, model, x, 0, torch.ones(1), lo, hi, enabled=True)
    assert tangent.update(torch.tensor([-1.])) is False
    assert tangent.update(torch.tensor([3.])) is False
    torch.testing.assert_close(tangent.x_hat, torch.ones(1))
    tangent.offer(torch.tensor([.5]), 0.)
    torch.testing.assert_close(tangent.x_hat, torch.tensor([.5]))
    with pytest.raises(ValueError, match='root box'):
        tangent.plane(lo-1, hi)
    m, b, x, lo, hi = graph(lambda x: x.pow(1.5), [0.], [2.])
    assert build_convex_tangent(b, m, x, 0, torch.zeros(1), lo, hi, enabled=True) is not None


def test_signed_row_cuts_relaxed_feasible_points_and_round_cap():
    fn = lambda x: torch.cat((x[:, :1], torch.exp(x[:, :1]), torch.log(x[:, 1:2])), 1)
    model, bounded, x, lo, hi = graph(fn, [-1., .2], [1., 2.])
    # The -log side of an equality is convex; its +log side is concave.
    c = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., -1.], [0., 0., 1.]]])
    rhs = torch.tensor([[0., 1., 0., 0.]])
    cuts = build_convex_tangent_rows(bounded, model, x, c, rhs, lo, hi, torch.tensor([0., 1.]), enabled=True,
                                     rowpool=RowPool(rhs))
    assert len(cuts.tangents) == 2
    A, bias = cuts.offer(torch.tensor([0., 1.]), 10.)
    generator = torch.Generator().manual_seed(28)
    points = lo + (hi-lo)*torch.rand(5000, 2, generator=generator)
    # Test each cut against points satisfying that relaxed row.
    values = model(points)
    for row, spec in enumerate([1, 2]):
        feasible = (values @ c[0, spec] <= rhs[0, spec]+1e-5)
        assert feasible.any()
        assert (points[feasible] @ A[0, row] + bias[0, row] <= 0).all()
    relaxed = torch.tensor([[torch.log(torch.tensor(1.+1e-5)), 1.]])
    assert (relaxed @ A[0].T + bias[0] <= 0).all()
    for k in range(1, 6):
        result = cuts.offer(torch.tensor([-.1*k, 1.]), 10.-k)
        assert (result is not None) == (k < 4)
    assert cuts.rounds == 4


def test_fraction_rewrite_never_guesses_proportionality():
    fn = lambda x: (x[:, :1]+1e-14*x[:, 1:2])/(1-.2*x[:, :1])
    _, bounded, x, lo, hi = graph(fn, [0., 0.], [4., 4.])
    assert ConvexityProof(bounded, x, lo, hi).output.tolist() == [UNKNOWN]


def test_rows_use_root_hull_and_require_identical_specs():
    model, bounded, x, lo, hi = graph(lambda x: torch.cat((x, torch.reciprocal(x)), 1), [.2], [2.])
    c = torch.eye(2)[None].expand(2, -1, -1)
    rhs = torch.zeros(2, 2)
    root_lower, root_upper = torch.tensor([[-1.], [.2]]), hi.expand(2, -1)
    # A positive child box cannot establish a domain fact for the root hull.
    assert build_convex_tangent_rows(bounded, model, x, c, rhs, root_lower, root_upper, x.flatten(), enabled=True,
                                     rowpool=RowPool(rhs)) is None
    rhs[1, 1] = 1.
    assert build_convex_tangent_rows(bounded, model, x, c, rhs, lo.expand(2, -1), root_upper, x.flatten(), enabled=True,
                                     rowpool=RowPool(rhs)) is None
    assert build_convex_tangent(bounded, model, x, 1, x.flatten(), lo, hi*torch.inf, enabled=True) is None


def test_combined_row_options_do_not_duplicate_dcp_rows():
    fn = lambda x: torch.cat((x, x.square()), 1)
    model, bounded, x, lo, hi = graph(fn, [-1.], [1.])
    manager = build_convex_tangent_rows(bounded, model, x, torch.eye(2)[None], torch.zeros(1, 2),
                                       lo, hi, x.flatten(), enabled=True, quadratic_rows=True,
                                       rowpool=RowPool(torch.zeros(1, 2)))
    assert len(manager.tangents) == 1


def _second_derivative(fn, lo, hi, points=2001):
    z = torch.linspace(lo, hi, points).reshape(-1, 1).requires_grad_(True)
    g, = torch.autograd.grad(fn(z).sum(), z, create_graph=True)
    return torch.autograd.grad(g.sum(), z)[0]


@pytest.mark.parametrize('fn,lo,hi,expected', [
    (lambda x: x * torch.exp(-x.square()), 0., 1., CONCAVE),    # f'' <= 0 up to x = sqrt(1.5) = 1.22
    (lambda x: x * torch.exp(-x.square()), 0., 2., UNKNOWN),
    (lambda x: -x * torch.exp(-x.square()), 0., 1., CONVEX),
    (lambda x: 3 * x * torch.exp(-.1 * x.pow(3)), 0., 10., UNKNOWN),   # inflection at 2.4 for c=.1
    (lambda x: x * torch.exp(-.001 * x.pow(3)), 0., 10., CONCAVE),    # pricing050 term, inflection at 11
    (lambda x: x * torch.log(x), .1, 2., CONVEX),
    (lambda x: torch.sin(x), 0., math.pi, CONCAVE),
    (lambda x: torch.sin(x), 0., 2 * math.pi, UNKNOWN),
])
def test_univariate_curvature_matches_dense_second_derivative(fn, lo, hi, expected):
    _, bounded, x, lower, upper = graph(fn, [lo], [hi])
    assert ConvexityProof(bounded, x, lower, upper).output.tolist() == [expected]
    f2 = _second_derivative(fn, lo, hi)
    if expected == CONVEX:
        assert (f2 >= -1e-9).all()
    elif expected == CONCAVE:
        assert (f2 <= 1e-9).all()
    elif lo > 0 or hi < 3:  # the rejected intervals really are indefinite
        assert (f2 > 0).any() and (f2 < 0).any()


def test_univariate_terms_sum_over_different_variables():
    term = lambda z: z * torch.exp(-z.square())
    _, bounded, x, lower, upper = graph(lambda x: term(x[:, :1]) + 2 * term(x[:, 1:2]), [0., 0.], [1., 1.])
    assert ConvexityProof(bounded, x, lower, upper).output.tolist() == [CONCAVE]
    _, bounded, x, lower, upper = graph(lambda x: term(x[:, :1]) + x[:, 1:2].square(), [0., 0.], [1., 1.])
    assert ConvexityProof(bounded, x, lower, upper).output.tolist() == [UNKNOWN]


def test_multivariate_products_are_not_given_a_univariate_certificate():
    _, bounded, x, lower, upper = graph(lambda x: x[:, :1] * torch.exp(-x[:, 1:2]), [1., 1.], [2., 2.])
    assert ConvexityProof(bounded, x, lower, upper).output.tolist() == [UNKNOWN]
    _, bounded, x, lower, upper = graph(lambda x: (x[:, :1] + x[:, 1:2]) * torch.exp(-x[:, 1:2]), [1., 1.], [2., 2.])
    assert ConvexityProof(bounded, x, lower, upper).output.tolist() == [UNKNOWN]


def test_jet_proved_term_with_a_deficit_keeps_its_cut_below_the_function():
    fn = lambda x: -x * torch.exp(-.001 * x.pow(3))   # convex on [0, 10]; f'' touches 0 at x = 0
    model, bounded, x, lower, upper = graph(fn, [0.], [10.])
    proof = ConvexityProof(bounded, x, lower, upper)
    assert proof.output.tolist() == [CONVEX]
    error = proof.cut_error(torch.ones(1))
    assert 0 < error < 1e-20
    zs = torch.linspace(0, 10, 2001).reshape(-1, 1)
    for anchor in (0., .7, 4., 10.):
        tangent = build_convex_tangent(bounded, model, x, 0, torch.tensor([anchor]), lower, upper, enabled=True)
        assert tangent.cut_error == error
        a, b = tangent.plane(lower, upper)
        assert (zs @ a[0].reshape(-1, 1) + b[0] <= model(zs.reshape(-1, 1)) ).all()


def test_exact_proofs_have_no_cut_error_and_the_old_cut():
    model, bounded, x, lower, upper = graph(lambda x: torch.exp(x), [-1.], [2.])
    assert ConvexityProof(bounded, x, lower, upper).cut_error(torch.ones(1)) == 0.
    tangent = build_convex_tangent(bounded, model, x, 0, torch.tensor([.5]), lower, upper, enabled=True)
    assert tangent.cut_error == 0.
    _, b = tangent.plane(lower, upper)
    tangent.cut_error = 1e-3
    assert tangent.plane(lower, upper)[1] < b


def test_deficit_terms_do_not_feed_nonlinear_rules():
    term = lambda z: -z * torch.exp(-.001 * z.pow(3))
    _, bounded, x, lower, upper = graph(lambda x: torch.exp(term(x)), [0.], [10.])
    assert ConvexityProof(bounded, x, lower, upper).output.tolist() == [CONVEX]  # exp(convex) is the jet's own proof
    _, bounded, x, lower, upper = graph(lambda x: term(x) + 3 * term(x), [0.], [10.])
    proof = ConvexityProof(bounded, x, lower, upper)
    assert proof.output.tolist() == [CONVEX] and proof.cut_error(torch.ones(1)) > 0
