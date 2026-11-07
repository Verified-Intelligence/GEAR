"""Pole gaps tighten square intervals without losing other CROWN planes."""
from fractions import Fraction
from unittest.mock import patch

import pytest
import torch
from torch import nn

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from auto_LiRPA.interval_bound import _interval_propagate
from auto_LiRPA.operators.base import Interval
from auto_LiRPA.operators.activations import BoundSqr
from auto_LiRPA.operators.add_sub import BoundAdd, BoundSub
from auto_LiRPA.operators.bivariate import BoundMul
from auto_LiRPA.operators.convex_concave import BoundReciprocal
from auto_LiRPA.operators.linear import BoundNeg
from auto_LiRPA.range_gap import reciprocal_gap, scale_gap, shift_gap, square_gap_lower


OPTS = {'undefined_relaxation_rows': True,
        'undefined_relaxation_interval': True,
        'compare_crown_with_ibp_intermediate': True}


def _box(lower, upper, dtype=torch.float64):
    return Interval(torch.tensor([[lower]], dtype=dtype),
                    torch.tensor([[upper]], dtype=dtype))


def _mul(x, y):
    node = BoundMul()
    node.is_linear_op = False
    return node.interval_propagate(x, y)


def _quotient(n, d):
    return _mul(n, BoundReciprocal().interval_propagate(d))


def _exact_min(nl, nu, dl, du, b):
    """Independent sign-split oracle, with exact rational finite endpoints."""
    nl, nu, dl, du, b = map(Fraction, (nl, nu, dl, du, b))
    numerators = []
    if nu > 0:
        numerators.append((max(nl, 0), nu))
    if nl < 0:
        numerators.append((nl, min(nu, 0)))
    denominators = []
    if du > 0:
        denominators.append((max(dl, 0), du, 1))
    if dl < 0:
        denominators.append((dl, min(du, 0), -1))
    distances = []
    if nl <= 0 <= nu and denominators:
        distances.append(abs(b))
    for al, au in numerators:
        for bl, bu, side in denominators:
            corners = [a / d if d else (0 if a == 0 else
                       (torch.inf if (a > 0) == (side > 0) else -torch.inf))
                       for a in (al, au) for d in (bl, bu)]
            distances.append(max(min(corners) - b, b - max(corners), 0))
    distance = min(distances) if distances else 0
    return distance ** 2


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('nl,nu', [(2., 5.), (-5., -2.), (-2., 3.),
                                  (0., 3.), (-3., 0.), (0., 0.)])
@pytest.mark.parametrize('dl,du', [(-2., 3.), (0., 3.), (-2., 0.),
                                  (-0., 3.), (-2., -0.)])
def test_quotient_gap_all_signs_dense_and_exact(dtype, nl, nu, dl, du):
    quotient = _quotient(_box(nl, nu, dtype), _box(dl, du, dtype))
    ns = torch.linspace(nl, nu, 129, dtype=dtype)
    ds = torch.linspace(dl, du, 129, dtype=dtype)
    ds = torch.cat((ds, torch.tensor([dl * 1e-6, du * 1e-6], dtype=dtype)))
    ds = ds[ds != 0]
    values = ns[:, None] / ds[None, :]
    for b in (-100., -2., -.25, 0., .25, 2., 100.):
        shifted = _interval_propagate(BoundSub(options={}),
                                      [quotient, _box(b, b, dtype)], [])
        result = BoundSqr().interval_propagate(shifted)
        lb = float(result.lower.item())
        assert not torch.isnan(result.lower).any()
        assert lb <= float(((values - b) ** 2).min())
        assert Fraction(lb) <= _exact_min(nl, nu, dl, du, b)
        if shifted.range_gap is not None:
            assert Fraction(square_gap_lower(shifted.range_gap).item()) <= _exact_min(
                nl, nu, dl, du, b)
        if nl * nu > 0 and b == 0:
            assert lb > 0


@pytest.mark.parametrize('factor', [(2., 4.), (-4., -2.), (0., 4.), (-4., 0.),
                                   (-2., 4.), (0., 0.)])
def test_shifted_gap_scaling_intersects_all_factor_gaps(factor):
    q = BoundReciprocal().interval_propagate(_box(-.5, .25))
    # Also exercise constant - q, negation, and shifts in both operand orders.
    q = BoundSub(options={}).interval_propagate(_box(3., 3.), q)
    q = BoundNeg().interval_propagate(q)
    q = BoundAdd().interval_propagate(_box(1., 1.), q)
    q = _mul(q, _box(*factor))
    ds = torch.linspace(-.5, .25, 1001, dtype=torch.float64)
    ds = ds[ds != 0]
    fs = torch.linspace(*factor, 81, dtype=torch.float64)
    values = (1 / ds - 2)[:, None] * fs[None, :]
    if q.range_gap is not None:
        left, right = q.range_gap
        assert not ((values > left.item()) & (values < right.item())).any()
    for b in (-10., 0., 10.):
        out = BoundSqr().interval_propagate(
            BoundAdd().interval_propagate(q, _box(b, b)))
        assert float(out.lower.item()) <= float(((values + b) ** 2).min())


def test_nonconstant_shift_and_unsupported_operator_drop_gap():
    q = BoundReciprocal().interval_propagate(_box(-.5, .25))
    result = BoundAdd().interval_propagate(q, _box(-1., 1.))
    assert square_gap_lower(result.range_gap).item() == 0

    class Model(nn.Module):
        def forward(self, x):
            return torch.sin(torch.reciprocal(x)) ** 2

    l, u = torch.tensor([[-.5]]), torch.tensor([[.25]])
    net = BoundedModule(Model(), l, device='cpu', bound_opts=OPTS)
    lb, _ = net.compute_bounds(x=(_input(l, u),), method='IBP')
    assert lb.item() == 0
    sine = next(n for n in net.nodes() if type(n).__name__ == 'BoundSin')
    assert sine._range_gap is None


def test_gap_endpoint_rounding_extremes():
    for dl, du in [(-3., 7.), (0., 1e-300), (-1e-300, 0.),
                   (-1e300, 1e300), (-1e-300, 1e-300)]:
        gap = reciprocal_gap(_box(dl, du).lower, _box(dl, du).upper)
        for nl, nu in [(1.1, 2.3), (-2.3, -1.1)]:
            scaled = scale_gap(gap, _box(nl, nu).lower, _box(nl, nu).upper)
            for b in (-.3, 0., .3):
                shifted = shift_gap(scaled, _box(-b, -b).lower, _box(-b, -b).upper)
                lb = square_gap_lower(shifted).item()
                assert Fraction(lb) <= _exact_min(nl, nu, dl, du, b)


class PoleResidual(nn.Module):
    def forward(self, x):
        y = x[:, 1:2]
        return (2 / (1 + .1 * x[:, :1]) - .3) ** 2 + (y - .5) ** 2 + y - y


def _input(l, u):
    return BoundedTensor((l + u) / 2, PerturbationLpNorm(x_L=l, x_U=u))


@pytest.mark.parametrize('cached_hulls', [False, True])
def test_crown_stops_at_square_and_preserves_other_plane(cached_hulls):
    l = torch.tensor([[-10., 2.]], dtype=torch.float64)
    u = torch.tensor([[-9.99, 3.]], dtype=torch.float64)
    model = PoleResidual()
    net = BoundedModule(model, (l + u) / 2, device='cpu', bound_opts=OPTS)
    bx = _input(l, u)
    ibp, _ = net.compute_bounds(x=(bx,), method='IBP')
    intermediates = None
    if cached_hulls:
        # Deliberately strip all gap metadata and the square's improved bound.
        intermediates = {n.name: (n.lower, n.upper) for n in net.nodes()
                         if n.is_lower_bound_current() and n.is_upper_bound_current()
                         and not isinstance(n, BoundSqr)}
    lb, ub, A = net.compute_bounds(x=(bx,), method='CROWN', return_A=True,
                                   interm_bounds=intermediates)
    entry = A[net.output_name[0]][net.input_name[0]]
    assert lb.item() > 3.99e6 and lb.item() > ibp.item() + .9
    assert torch.isposinf(ub).all()
    torch.testing.assert_close(entry['lA'], l.new_tensor([[[0., 3.]]]))
    gen = torch.Generator().manual_seed(10)
    pts = l + (u - l) * torch.rand(10000, 2, generator=gen, dtype=l.dtype)
    pts = torch.cat((pts, l, u))
    values = model(pts)
    assert (lb <= values).all()
    plane = pts @ entry['lA'][0].T + entry['lbias'][0]
    assert (plane <= values).all()


def test_gap_reset_on_new_box_and_batch_and_undefined_rows():
    model = PoleResidual()
    l = torch.tensor([[-10., 2.]], dtype=torch.float64)
    net = BoundedModule(model, l, device='cpu', bound_opts=OPTS)
    lb, _ = net.compute_bounds(x=(_input(l, l + l.new_tensor([[.01, 1.]])),),
                               method='CROWN', bound_upper=False)
    assert lb.item() > 3.99e6
    l = l.new_tensor([[-210., .5], [-10., 2.], [-210., .5]])
    u = l.new_tensor([[190., 1.], [-9.99, 3.], [190., 1.]])
    lb, _ = net.compute_bounds(x=(_input(l, u),), method='CROWN',
                               bound_upper=False, cache_bounds=True)
    assert lb[1, 0] > 3.99e6
    assert (lb[[0, 2]] < 1e-9).all()

    # A row that needs the unbounded upper side of the square stays discarded.
    lb, _ = net.compute_bounds(x=(_input(l, u),), method='CROWN',
                               C=-torch.ones(3, 1, 1, dtype=l.dtype), bound_upper=False)
    assert torch.isneginf(lb).all()


@pytest.mark.parametrize('method', ['IBP', 'CROWN'])
def test_without_pole_gap_bounds_are_unchanged(method):
    l = torch.tensor([[0., 2.]], dtype=torch.float64)
    u = torch.tensor([[1., 3.]], dtype=torch.float64)
    net = BoundedModule(PoleResidual(), l, device='cpu', bound_opts=OPTS)
    normal = net.compute_bounds(x=(_input(l, u),), method=method)
    with patch('auto_LiRPA.range_gap.reciprocal_gap', return_value=None):
        without = net.compute_bounds(x=(_input(l, u),), method=method)
    for actual, expected in zip(normal, without):
        assert torch.equal(actual, expected)


def test_gap_stop_preserves_finite_term_parameter_gradients():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(2., dtype=torch.float64))
            self.shift = nn.Parameter(torch.tensor(.5, dtype=torch.float64))

        def forward(self, x):
            return (self.a / (1 + .1 * x[:, :1]) - .3) ** 2 + (x[:, 1:2] - self.shift) ** 2

    l = torch.tensor([[-10., 2.]], dtype=torch.float64)
    u = torch.tensor([[-9.99, 3.]], dtype=torch.float64)
    net = BoundedModule(Model(), (l + u) / 2, device='cpu', bound_opts=OPTS)
    lb, _ = net.compute_bounds(x=(_input(l, u),), method='CROWN', bound_upper=False)
    lb.sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and (p.grad != 0).any()
               for p in net.parameters())


def test_random_quotient_boxes_dense_sampling():
    gen = torch.Generator().manual_seed(42)
    bounds = torch.randn(600, 4, generator=gen, dtype=torch.float64) * 5
    nl, nu = torch.minimum(bounds[:, 0], bounds[:, 1]), torch.maximum(bounds[:, 0], bounds[:, 1])
    dl, du = torch.minimum(bounds[:, 2], bounds[:, 3]), torch.maximum(bounds[:, 2], bounds[:, 3])
    dl = torch.where(torch.arange(600) % 3 == 0, 0., dl.clamp(max=0))
    du = torch.where(torch.arange(600) % 3 == 1, 0., du.clamp(min=0))
    b = torch.randn(600, generator=gen, dtype=torch.float64) * 3
    q = _quotient(Interval(nl[:, None], nu[:, None]),
                  Interval(dl[:, None], du[:, None]))
    shifted = _interval_propagate(BoundSub(options={}),
                                  [q, Interval(b[:, None], b[:, None])], [])
    out = BoundSqr().interval_propagate(shifted)
    grid = torch.linspace(0., 1., 81, dtype=torch.float64)
    ns = nl[:, None] + (nu - nl)[:, None] * grid
    ds = dl[:, None] + (du - dl)[:, None] * grid
    # The rounded interpolation at t=1 can land just outside the box.
    ns = torch.maximum(nl[:, None], torch.minimum(nu[:, None], ns))
    ds = torch.maximum(dl[:, None], torch.minimum(du[:, None], ds))
    ys = (ns[:, :, None] / ds[:, None, :] - b[:, None, None]) ** 2
    ys = torch.where(torch.isfinite(ys), ys, torch.inf)
    assert (out.lower[:, 0] <= ys.flatten(1).min(1).values).all()
    gap_lower = square_gap_lower(shifted.range_gap)[:, 0]
    for i in range(600):
        assert Fraction(gap_lower[i].item()) <= _exact_min(
            *[v[i].item() for v in (nl, nu, dl, du, b)])
