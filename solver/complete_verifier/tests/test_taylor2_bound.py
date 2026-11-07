"""Interval AD and Taylor certificates against independently sampled derivatives."""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from auto_LiRPA import BoundedModule  # noqa: E402
from nlp_solver.interval_hessian import IntervalHessian  # noqa: E402
from nlp_solver.quadratic_tangent import build_quadratic_tangent, output_degrees  # noqa: E402
from nlp_solver.taylor2_tangent import Taylor2Tangent  # noqa: E402


class Model(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x).reshape(x.shape[0], -1)


def graph(fn, n=4, dtype=torch.float64):
    model = Model(fn)
    x = torch.full((1, n), 1.2, dtype=dtype)
    bm = BoundedModule(model, x, bound_opts={'pow_negative_exponent': True})
    output_degrees(bm, x)
    return model, bm, x


OPS = [
    ('add_sub_neg', lambda x: (-(x + x[:, :1]) - 2.3).sum(-1)),
    ('both_factors_perturbed', lambda x: (x * x.flip(-1)).sum(-1)),
    ('sqr', lambda x: x.square().sum(-1)),
    ('pow4', lambda x: x.pow(4).sum(-1)),
    ('pow3', lambda x: x.pow(3).sum(-1)),
    ('pow0', lambda x: x.pow(0).sum(-1) + x[:, 0]),
    ('pow1', lambda x: x.pow(1).sum(-1)),
    ('pow_fraction', lambda x: x.pow(2.7).sum(-1)),
    ('pow_negative', lambda x: x.pow(-3).sum(-1)),
    ('reciprocal', lambda x: x.reciprocal().sum(-1)),
    ('division', lambda x: (x / (x[:, :1] + 1.)).sum(-1)),
    ('exp', lambda x: x.exp().sum(-1)),
    ('log', lambda x: x.log().sum(-1)),
    ('sqrt', lambda x: x.sqrt().sum(-1)),
    ('sin', lambda x: x.sin().sum(-1)),
    ('cos', lambda x: x.cos().sum(-1)),
    ('tanh', lambda x: x.tanh().sum(-1)),
    ('reduce_mean', lambda x: x.pow(4).mean(-1)),
    ('matmul', lambda x: (x @ torch.tensor([[1., -2.], [-3., .2], [4., 1.], [-.1, 2.]], dtype=x.dtype)).square().sum(-1)),
    ('left_matmul', lambda x: (torch.tensor([[-1., 2.], [3., -.5]], dtype=x.dtype) @ x.reshape(-1, 2, 2)).pow(3).sum((-1, -2))),
    ('linear', lambda x: F.linear(x, torch.tensor([[1., -3., 2., .5]], dtype=x.dtype), torch.tensor([2.], dtype=x.dtype)).pow(4)),
    ('movement', lambda x: torch.cat((x[:, 1:3].unsqueeze(-1), x[:, :2].unsqueeze(-1)), -1)
     .transpose(1, 2).reshape(-1, 4).pow(4).sum(-1)),
    ('squeeze_flatten', lambda x: x.unsqueeze(1).squeeze(1).flatten(1).pow(3).sum(-1)),
    ('slice', lambda x: x[:, 1:3].pow(3).sum(-1)),
    ('split', lambda x: (torch.split(x, 2, dim=1)[1].pow(4)).sum(-1)),
    ('gather_expand', lambda x: x[:, [3, 0]].unsqueeze(1).expand(-1, 2, -1).squeeze(1).pow(3).sum((1, 2))),
    ('composition', lambda x: ((x.sin() * x.pow(3) + (x + 2.).log()).exp() / (x + 3.).sqrt()).sum(-1)),
]


@pytest.mark.parametrize('name,fn', OPS, ids=[item[0] for item in OPS])
def test_interval_jet_encloses_autograd(name, fn):
    model, bm, x = graph(fn)
    evaluator = IntervalHessian(bm, x, 0)
    generator = torch.Generator().manual_seed(123)
    l = .3 + torch.rand(4, 4, dtype=x.dtype, generator=generator)
    u = l + .2 + torch.rand(4, 4, dtype=x.dtype, generator=generator)
    if name in ('pow3', 'pow4', 'sin', 'cos', 'both_factors_perturbed'):
        l -= 2.  # Include negative intervals, zero crossings and trig extrema.
    jet = evaluator.evaluate(l, u)
    assert jet.valid.all(), [(type(node).__name__, evaluator.rules.get(node.name)) for node in evaluator.order]
    for box in range(len(l)):
        for _ in range(12):
            z = (l[box] + (u[box] - l[box]) * torch.rand(4, dtype=x.dtype, generator=generator)).requires_grad_()
            scalar = lambda t: model(t[None])[0, 0]
            value = scalar(z)
            gradient, = torch.autograd.grad(value, z)
            hessian = torch.autograd.functional.hessian(scalar, z)
            for interval, sampled in ((jet.value, value), (jet.gradient, gradient), (jet.hessian, hessian)):
                assert (interval[0][box, 0] <= sampled).all(), (name, interval, sampled)
                assert (sampled <= interval[1][box, 0]).all(), (name, interval, sampled)


def test_index_add_and_buffers():
    class Buffered(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('indices', torch.tensor([0, 1, 0, 2]))
            self.register_buffer('weights', torch.tensor([2., -1., 3., .5], dtype=torch.float64))
        def forward(self, x):
            data = torch.zeros(x.shape[0], 3, dtype=x.dtype)
            return data.index_add(1, self.indices, (x * self.weights).pow(4)).sum(-1, keepdim=True)
    model = Buffered()
    x = torch.ones(1, 4, dtype=torch.float64)
    bm = BoundedModule(model, x)
    output_degrees(bm, x)
    evaluator = IntervalHessian(bm, x, 0)
    l, u = -.5 * x, 1.5 * x
    jet = evaluator.evaluate(l, u)
    assert jet.valid.all()
    z = torch.tensor([.1, .2, .3, .4], dtype=x.dtype)
    h = torch.autograd.functional.hessian(lambda t: model(t[None])[0, 0], z)
    assert (jet.hessian[0][0, 0] <= h).all() and (h <= jet.hessian[1][0, 0]).all()


@pytest.mark.parametrize('fn', [lambda x: x.reciprocal().sum(-1), lambda x: x.log().sum(-1),
                               lambda x: x.sqrt().sum(-1), lambda x: x.pow(-2).sum(-1),
                               lambda x: x.pow(.7).sum(-1)])
def test_invalid_domains_are_vacuous_per_box(fn):
    model, bm, x = graph(fn)
    t = build_quadratic_tangent(bm, model, x, 0, x[0], taylor2_bound=True)
    l = torch.tensor([[.5] * 4, [-1.] * 4, [0.] * 4], dtype=x.dtype)
    u = torch.tensor([[2.] * 4, [1.] * 4, [2.] * 4], dtype=x.dtype)
    assert t.evaluator.evaluate(l, u).valid[:, 0].tolist() == [True, False, False]
    a, b = t.plane(l, u)
    assert torch.isfinite(a).all() and torch.isfinite(b[0]) and torch.isneginf(b[1:]).all()


def test_invalid_unrelated_output_does_not_disable_objective():
    model, bm, x = graph(lambda x: torch.cat((x.pow(4).sum(-1, keepdim=True), x[:, :1].log()), -1))
    t = build_quadratic_tangent(bm, model, x, 0, x[0], taylor2_bound=True)
    assert torch.isfinite(t.plane(-x, x)[1]).all()


def test_unsupported_and_nonfinite_are_vacuous():
    for fn, l, u in ((lambda x: x.relu().sum(-1), -1., 2.),
                     (lambda x: x.exp().sum(-1), 700., 1000.)):
        model, bm, x = graph(fn)
        t = build_quadratic_tangent(bm, model, x, 0, x[0], taylor2_bound=True)
        assert torch.isneginf(t.plane(x * l, x * u)[1]).all()


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_taylor_plane_on_random_boxes_and_external_anchor(dtype):
    model, bm, x = graph(lambda x: (x.pow(4) - 3. * x.square() + (x[:, :1] * x).sin()).sum(-1), dtype=dtype)
    generator = torch.Generator().manual_seed(92)
    anchor = torch.tensor([.2, -.4, 1.2, -.9], dtype=dtype)
    t = build_quadratic_tangent(bm, model, x, 0, anchor, taylor2_bound=True)
    assert isinstance(t, Taylor2Tangent) and t.g.dtype == torch.float64
    l = -2. + 3. * torch.rand(8, 4, generator=generator, dtype=dtype)
    u = l + .1 + torch.rand(8, 4, generator=generator, dtype=dtype)
    a, b = t.plane(l, u)
    assert torch.isfinite(b).all()
    for k in range(len(l)):
        z = l[k].double() + (u[k] - l[k]).double() * torch.rand(300, 4, generator=generator, dtype=torch.float64)
        assert (z @ a[k].double() + b[k].double() <= model.double()(z)[:, 0]).all()


def test_hull_domain_check_includes_external_anchor():
    model, bm, x = graph(lambda x: x.reciprocal().sum(-1))
    t = build_quadratic_tangent(bm, model, x, 0, -x[0], taylor2_bound=True)
    assert torch.isneginf(t.plane(x, 2. * x)[1]).all()


def test_invalid_anchor_has_finite_coefficients_and_keeps_crown():
    from nlp_solver.dual_ascent import lp_bound
    model, bm, x = graph(lambda x: x.log().sum(-1))
    t = build_quadratic_tangent(bm, model, x, 0, -x[0], taylor2_bound=True)
    a, b = t.plane(x, 2. * x)
    assert torch.isfinite(a).all() and torch.isneginf(b).all()
    bound, _ = lp_bound(a, b, a[:, None], torch.ones(1, 1, dtype=x.dtype), x, 2. * x)
    assert torch.equal(torch.maximum(bound, torch.tensor([1.], dtype=x.dtype)), torch.tensor([1.], dtype=x.dtype))


def test_fixed_coordinates_and_scaled_gershgorin():
    model, bm, x = graph(lambda x: (x[:, 0] + 2. * x[:, 1]).square()
                        + 5. * x[:, 1].square() - 100. * x[:, 2].pow(4), n=3)
    l, u = torch.tensor([[-1., -1., 2.]], dtype=x.dtype), torch.tensor([[1., 1., 2.]], dtype=x.dtype)
    t = Taylor2Tangent(bm, x, 0, lambda z: model(z[None])[0, 0], torch.zeros(3), l, u)
    mu = t.evaluator.lower_eigenvalue(l, u, t.fixed_mask)
    h = torch.tensor([[2., 4.], [4., 18.]], dtype=x.dtype)
    assert 0. < mu <= torch.linalg.eigvalsh(h).min()
    assert t.x_hat[2] == 2.
    a, b = t.plane(l, u)
    assert abs(float(a[0] @ t.x_hat + b[0]) - t.f) < 1e-4
    bad = l.clone(); bad[0, 2] = 1.
    with pytest.raises(ValueError, match='fixed-coordinate subspace'):
        t.plane(bad, u)
    t.polish(torch.ones(3))
    assert not t.polished
    t.offer(torch.zeros(3), margin=1e6)
    assert t.x_hat.tolist() == [0., 0., 2.]


def test_domain_cap_selects_lowest_bounds():
    model, bm, x = graph(lambda x: x.pow(4).sum(-1))
    t = build_quadratic_tangent(bm, model, x, 0, x[0], taylor2_bound=True, taylor2_max_domains=2)
    l, u = x.repeat(5, 1), 2. * x.repeat(5, 1)
    _, b = t.plane(l, u, bound_scores=torch.tensor([4., 0., 3., 2., -1.]))
    assert torch.isfinite(b).tolist() == [False, True, False, False, True]
    t.max_domains = 0
    assert torch.isneginf(t.plane(l, u)[1]).all()


def test_chunked_planes_match_whole_batch():
    model, bm, x = graph(lambda x: (x.pow(4) + x.cos()).sum(-1))
    t = build_quadratic_tangent(bm, model, x, 0, x[0], taylor2_bound=True)
    l, u = .5 * x.repeat(7, 1), 2. * x.repeat(7, 1)
    whole = t.plane(l, u)
    t.chunk_size = 2
    chunked = t.plane(l, u)
    for a, b in zip(whole, chunked):
        torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-13)


def test_configuration_defaults_are_opt_in():
    from arguments import ConfigHandler
    config = ConfigHandler()
    assert config.default_args['nlp_taylor2_bound'] is False
    assert config.default_args['nlp_taylor2_max_vars'] == 40
    assert config.default_args['nlp_taylor2_max_domains'] == 1024


def test_dot_product_underflow_is_enclosed():
    # Each product rounds to zero in float64; their exact sum is several ulps
    # of the smallest subnormal. Check against decimal, not rounded autograd.
    from decimal import Decimal, localcontext
    weights = torch.full((4, 1), .1, dtype=torch.float64)
    model, bm, x = graph(lambda x: (x @ weights).sum(-1))
    evaluator = IntervalHessian(bm, x, 0)
    z = torch.full_like(x, torch.finfo(x.dtype).tiny * torch.finfo(x.dtype).eps)
    jet = evaluator.evaluate(z, z)
    with localcontext() as context:
        context.prec = 400
        exact = Decimal.from_float(float(z[0, 0])) * Decimal.from_float(.1) * 4
        assert Decimal.from_float(float(jet.value[0][0, 0])) <= exact
        assert exact <= Decimal.from_float(float(jet.value[1][0, 0]))


def test_option_off_and_size_limit_keep_existing_behavior():
    model, bm, x = graph(lambda x: x.pow(4).sum(-1))
    assert build_quadratic_tangent(bm, model, x, 0, x[0]) is None
    assert build_quadratic_tangent(bm, model, x, 0, x[0], taylor2_bound=True, taylor2_max_vars=3) is None
    quadratic, qb, xx = graph(lambda x: x.square().sum(-1))
    old = build_quadratic_tangent(qb, quadratic, xx, 0, xx[0])
    enabled = build_quadratic_tangent(qb, quadratic, xx, 0, xx[0], taylor2_bound=True)
    assert not isinstance(enabled, Taylor2Tangent)
    assert all(torch.equal(a, b) for a, b in zip(old.plane(-xx, xx), enabled.plane(-xx, xx)))
    assert build_quadratic_tangent(qb, quadratic, xx, 0, xx[0], taylor2_bound=True, quadratic_bound=False) is None


def test_evaluator_preserves_bound_graph_caches():
    model, bm, x = graph(lambda x: x[:, [2, 0]].reshape(1, 2).pow(4).sum(-1))
    before = model(x)
    evaluator = IntervalHessian(bm, x, 0)
    evaluator.evaluate(-x.repeat(3, 1), x.repeat(3, 1))
    torch.testing.assert_close(bm(x), before, rtol=0, atol=0)
