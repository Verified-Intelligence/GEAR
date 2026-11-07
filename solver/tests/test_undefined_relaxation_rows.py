"""A relaxation undefined on some domains' intervals discards only the rows using it.

With the bound option `undefined_relaxation_rows`, a pole inside an interval (1/x
across 0, tan across pi/2) or an infinite endpoint no longer aborts the whole
batch: backward_general sets the bounds of exactly the (row, domain) pairs whose
coefficients select an undefined relaxation side to -inf / +inf (interval
arithmetic may then tighten them), and every other bound is unchanged.
"""

import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.operators.base import OperatorDomainError
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.fp_enclosure import box_rounding_error, rounding_error
from enclosure_test_utils import assert_enclosed

OPTS = {'undefined_relaxation_rows': True,
        'compare_crown_with_ibp_intermediate': True}
INTERVAL_OPTS = {**OPTS, 'undefined_relaxation_interval': True}


class RecipRows(nn.Module):
    """Row 0 uses 1/x0 with a positive sign, row 1 with a negative one, row 2 not."""

    def forward(self, x):
        r = torch.reciprocal(x[:, 0:1])
        return torch.cat([r + x[:, 1:2], x[:, 1:2] - r, 2 * x[:, 1:2]], dim=1)


class TanRow(nn.Module):
    def forward(self, x):
        return torch.cat([torch.tan(3 * x[:, 0:1]) + x[:, 1:2], x[:, 1:2] ** 3], dim=1)


class SquareOfPole(nn.Module):
    def forward(self, x):
        r = torch.reciprocal(x[:, 0:1])
        return r * r + x[:, 1:2]


class SquarePoleResidual(nn.Module):
    def forward(self, x):
        return (torch.square(torch.reciprocal(x[:, 0:1]) - 2)
                + (x[:, 1:2] - 0.5 * x[:, 2:3]) ** 2)


class TanhPoleResidual(nn.Module):
    def forward(self, x):
        r = torch.tanh(torch.reciprocal(x[:, 0:1])) + 2
        return torch.square(r) + x[:, 1:2] - x[:, 1:2] + 0.25 * x[:, 2:3]


class InfiniteInputResidual(nn.Module):
    def forward(self, x):
        r = torch.tanh(torch.exp(x[:, 0:1])) + 2
        return torch.square(r) + x[:, 1:2] - 0.5 * x[:, 2:3]


class ProductPoleResidual(nn.Module):
    def forward(self, x):
        a = torch.reciprocal(x[:, :2])
        b = -torch.reciprocal(x[:, 2:4])
        return -(a * b).sum(dim=1, keepdim=True) + x[:, 4:5] - x[:, 4:5]


def _bounds(model, lower, upper, opts=OPTS, return_A=False, method='CROWN',
            store_intervals=False):
    dtype = torch.float64
    lower, upper = lower.to(dtype), upper.to(dtype)
    net = BoundedModule(model, lower, bound_opts=dict(opts))
    x = BoundedTensor(lower, PerturbationLpNorm(x_L=lower, x_U=upper))
    interm_bounds = None
    if store_intervals:
        net.compute_bounds(x=(x,), method='IBP')
        interm_bounds = {n.name: (n.lower, n.upper) for n in net.nodes()
                         if n.is_lower_bound_current() and n.is_upper_bound_current()}
    if return_A:
        needed = {net.output_name[0]: [net.input_name[0]]}
        return net, net.compute_bounds(
            x=(x,), method=method, return_A=True, needed_A_dict=needed,
            interm_bounds=interm_bounds)
    return net.compute_bounds(x=(x,), method=method, interm_bounds=interm_bounds)


def _sampled_sound(model, lower, upper, lb, ub, n=20000):
    torch.manual_seed(0)
    for b in range(lower.shape[0]):
        pts = lower[b] + (upper[b] - lower[b]) * torch.rand(
            n, lower.shape[1], dtype=torch.float64)
        y = model(pts)
        finite = torch.isfinite(y)
        assert (~finite | (y >= lb[b] - 1e-9)).all()
        assert (~finite | (y <= ub[b] + 1e-9)).all()


def test_reciprocal_pole_discards_only_the_crossing_domain_rows():
    lower = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    upper = torch.tensor([[2.0, 1.0], [2.0, 1.0]], dtype=torch.float64)
    lb, ub = _bounds(RecipRows(), lower, upper)
    # Domain 0 crosses the pole: rows 0 and 1 are unknown, row 2 is not.
    assert torch.isneginf(lb[0, :2]).all() and torch.isposinf(ub[0, :2]).all()
    margin = 10 * box_rounding_error(
        lower.new_tensor([[[0., 2.]]]), lower[:1], upper[:1]).squeeze()
    assert_enclosed(lb[0, 2], 0, margin, -1)
    assert_enclosed(ub[0, 2], 2, margin, 1)
    # Domain 1 is exactly what it is on its own, without the option.
    lb1, ub1 = _bounds(RecipRows(), lower[1:], upper[1:], opts={})
    assert torch.equal(lb[1], lb1[0]) and torch.equal(ub[1], ub1[0])
    _sampled_sound(RecipRows(), lower, upper, lb, ub)


def test_touching_zero_keeps_the_bounded_side():
    # On (0, 2] 1/x is bounded below, unbounded above; -1/x the other way round.
    lower = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    upper = torch.tensor([[2.0, 1.0]], dtype=torch.float64)
    lb, ub = _bounds(RecipRows(), lower, upper)
    assert torch.isfinite(lb[0, 0]) and torch.isposinf(ub[0, 0])
    assert torch.isneginf(lb[0, 1]) and torch.isfinite(ub[0, 1])
    assert lb[0, 0] <= 0.5 and ub[0, 1] >= 1.0 - 0.5
    _sampled_sound(RecipRows(), lower, upper, lb, ub)


@pytest.mark.parametrize('store_intervals', [False, True])
@pytest.mark.parametrize('sign', [1., -1.])
def test_interval_stop_reciprocal_sum_keeps_the_defined_side(sign, store_intervals):
    class ReciprocalSum(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('weights', torch.tensor([1., 2., 3.], dtype=torch.float64))

        def forward(self, x):
            return sign * (self.weights * torch.reciprocal(x)).sum(dim=1, keepdim=True)

    lower = torch.zeros(2, 3, dtype=torch.float64)
    upper = torch.tensor([[1., 2., 4.], [2., 3., 5.]], dtype=torch.float64)
    model = ReciprocalSum()
    net, (lb, ub, planes) = _bounds(
        model, lower, upper, opts=INTERVAL_OPTS, return_A=True,
        store_intervals=store_intervals)
    entry = planes[net.output_name[0]][net.input_name[0]]
    defined, undefined = ('l', 'u') if sign > 0 else ('u', 'l')
    # A positive lower coefficient (negative upper coefficient) uses the valid
    # reciprocal tangent. The opposite direction uses the undefined secant.
    assert (sign * entry[defined + 'A'] < 0).all()
    assert torch.isfinite(entry[defined + 'bias']).all()
    assert (entry[undefined + 'A'] == 0).all()
    if sign > 0:
        assert torch.isfinite(lb).all() and torch.isposinf(ub).all()
        assert torch.isposinf(entry['ubias']).all()
    else:
        assert torch.isneginf(lb).all() and torch.isfinite(ub).all()
        assert torch.isneginf(entry['lbias']).all()
    _sampled_sound(model, lower, upper, lb, ub, n=2000)
    for fraction in torch.linspace(.01, 1, 50, dtype=lower.dtype):
        points = fraction * upper
        values = model(points)
        assert ((entry['lA'] * points.unsqueeze(1)).sum(-1)
                + entry['lbias'] <= values + 1e-9).all()
        assert ((entry['uA'] * points.unsqueeze(1)).sum(-1)
                + entry['ubias'] >= values - 1e-9).all()


@pytest.mark.parametrize('opts', [{}, {'undefined_relaxation_interval': True}])
def test_default_still_raises(opts):
    lower = torch.tensor([[-1.0, 0.0]], dtype=torch.float64)
    upper = torch.tensor([[2.0, 1.0]], dtype=torch.float64)
    with pytest.raises(OperatorDomainError):
        _bounds(RecipRows(), lower, upper, opts=opts)


def test_tan_pole_rows_and_interval():
    lower = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    upper = torch.tensor([[1.0, 1.0], [0.3, 1.0]], dtype=torch.float64)
    lb, ub = _bounds(TanRow(), lower, upper)
    # 3 x0 in [0, 3] contains pi/2: row 0 unknown in domain 0 only.
    assert torch.isneginf(lb[0, 0]) and torch.isposinf(ub[0, 0])
    assert torch.isfinite(lb[1]).all() and torch.isfinite(ub[1]).all()
    assert torch.isfinite(lb[0, 1]) and torch.isfinite(ub[0, 1])
    _sampled_sound(TanRow(), lower, upper, lb, ub)
    ilb, iub = _bounds(TanRow(), lower, upper, method='IBP')
    assert torch.isneginf(ilb[0, 0]) and torch.isposinf(iub[0, 0])
    _sampled_sound(TanRow(), lower, upper, ilb, iub)


def test_square_of_unbounded_input_keeps_its_tangent_lower_plane():
    # 1/x on [-1, 1] is all of R; (1/x)^2 >= 0 is still a valid lower plane.
    lower = torch.tensor([[-1.0, 0.0]], dtype=torch.float64)
    upper = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    lb, ub = _bounds(SquareOfPole(), lower, upper)
    margin = 10 * box_rounding_error(
        lower.new_tensor([[[0., 1.]]]), lower, upper).squeeze()
    assert_enclosed(lb[0, 0], 0, margin, -1)
    assert torch.isposinf(ub[0, 0])


def test_returned_plane_of_a_discarded_row_is_vacuous():
    lower = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    upper = torch.tensor([[2.0, 1.0], [2.0, 1.0]], dtype=torch.float64)
    net, (lb, ub, A) = _bounds(RecipRows(), lower, upper, return_A=True)
    entry = A[net.output_name[0]][net.input_name[0]]
    assert (entry['lA'][0, :2] == 0).all()
    assert torch.isneginf(entry['lbias'][0, :2]).all()
    assert torch.isfinite(entry['lbias'][1]).all()
    assert torch.isfinite(entry['lbias'][0, 2])


def test_early_returned_intermediate_plane_of_a_discarded_row_is_vacuous():
    lower = torch.tensor([[-1., 0.], [1., 0.]], dtype=torch.float64)
    upper = torch.tensor([[2., 1.], [2., 1.]], dtype=torch.float64)
    net = BoundedModule(RecipRows(), lower, bound_opts=dict(OPTS))
    reciprocal = next(n for n in net.nodes() if type(n).__name__ == 'BoundReciprocal')
    x = BoundedTensor(lower, PerturbationLpNorm(x_L=lower, x_U=upper))
    lb, ub, planes = net.compute_bounds(
        x=(x,), method='CROWN', return_A=True, need_A_only=True,
        needed_A_dict={net.output_name[0]: [reciprocal.name]})
    assert lb is None and ub is None
    entry = planes[net.output_name[0]][reciprocal.name]
    assert (entry['lA'][0, :2] == 0).all()
    assert (entry['uA'][0, :2] == 0).all()
    assert torch.isneginf(entry['lbias'][0, :2]).all()
    assert torch.isposinf(entry['ubias'][0, :2]).all()
    assert torch.isfinite(entry['lbias'][1]).all()


def test_nonfinite_rows_are_zeroed_and_reported():
    from auto_LiRPA.backward_bound import _drop_nonfinite_rows
    last_A = torch.ones(2, 3, 4, dtype=torch.float64)
    A_x = torch.ones(2, 3, 4, dtype=torch.float64)
    A_x[1, 2, 0] = torch.nan
    bias = torch.zeros(2, 3, dtype=torch.float64)
    bias[0, 1] = -torch.inf
    A, bias_out, rows = _drop_nonfinite_rows([(A_x, None)], 0, bias, last_A)
    expected = torch.zeros(2, 3, dtype=torch.bool)
    expected[1, 2] = expected[0, 1] = True
    assert torch.equal(rows, expected)
    assert torch.isfinite(A[0][0]).all() and torch.isfinite(bias_out).all()
    assert (A[0][0][~expected] == 1).all() and (A[0][0][expected] == 0).all()
    # All finite: nothing is reported or changed.
    A2, b2, rows2 = _drop_nonfinite_rows(
        [(torch.ones(2, 3, 4), None)], 0, torch.zeros(2, 3), last_A)
    assert rows2 is None


@pytest.mark.parametrize('interval_stop', [False, True])
def test_cancelling_denominator_keeps_stored_and_returned_bounds_sound(interval_stop):
    class CancellingDenominator(nn.Module):
        def forward(self, x):
            a = x[:, :1] - 200
            denominator = torch.abs(a) + 1e-10 + a
            return torch.cat([
                torch.reciprocal(denominator) * x[:, 1:2],
                2 * x[:, :1]], dim=1)

    # For negative a the real-arithmetic denominator is 1e-10, but forward
    # rounding depends on a. CROWN first cancels O(200) biases, then cancels
    # O(1e22) biases in the reciprocal. Its false point bounds are off by ~1e6.
    # The positive domain and the row independent of the reciprocal stay usable.
    lower = torch.tensor([[-50., 1.], [220., 1.], [-50., 1.], [-50., 1.]], dtype=torch.float64)
    upper = torch.tensor([[-40., 2.], [240., 2.], [-50., 1.],
                          [-50. + 1e-13, 1.]], dtype=torch.float64)
    opts = {'undefined_relaxation_rows': True,
            'undefined_relaxation_interval': interval_stop,
            'compare_crown_with_ibp_intermediate': interval_stop}
    model = CancellingDenominator()
    net, (lb, ub, planes) = _bounds(model, lower, upper, opts=opts, return_A=True)
    stored = {n.name: (n.lower.clone(), n.upper.clone()) for n in net.nodes()
              if n.perturbed and isinstance(n.lower, torch.Tensor)
              and isinstance(n.upper, torch.Tensor)}
    entry = planes[net.output_name[0]][net.input_name[0]]
    for fraction in torch.linspace(0, 1, 33, dtype=lower.dtype):
        points = lower + fraction * (upper - lower)
        values = net(points)
        assert (values >= lb - 1e-9).all()
        assert (values <= ub + 1e-9).all()
        for node in net.nodes():
            if node.name in stored:
                node_lb, node_ub = stored[node.name]
                assert (node.forward_value >= node_lb - 1e-9).all(), node.name
                assert (node.forward_value <= node_ub + 1e-9).all(), node.name
        for side, sign in [('l', 1), ('u', -1)]:
            plane = (entry[side + 'A'] * points.unsqueeze(1)).sum(-1)
            plane = plane + entry[side + 'bias']
            assert (sign * (values - plane) >= -1e-9).all()
    assert lb[0, 1] <= -100 and ub[0, 1] >= -80
    torch.testing.assert_close(lb[0, 1], lb.new_tensor(-100), rtol=0, atol=1e-11)
    torch.testing.assert_close(ub[0, 1], ub.new_tensor(-80), rtol=0, atol=1e-11)
    assert torch.isfinite(lb[1]).all() and torch.isfinite(ub[1]).all()
    denominator = next(n for n in net.nodes() if type(n).__name__ == 'BoundReciprocal').inputs[0]
    denominator_lb, denominator_ub = stored[denominator.name]
    assert (denominator_lb[[0, 2, 3]] < 1e-10).all()
    assert (denominator_ub[[0, 2, 3]] > 1e-10).all()


@pytest.mark.parametrize('method', ['IBP', 'CROWN'])
def test_missing_relaxation_descriptor_keeps_unrelated_rows(method):
    class VariablePower(nn.Module):
        def forward(self, x):
            power = torch.pow(x[:, :1], x[:, 1:2])
            return torch.cat([power * x[:, 2:3], 2 * x[:, 2:3]], dim=1)

    # Both sides of row 0 use the unsupported node (a zero multiplier could
    # legitimately keep one side finite without ever using its relaxation).
    lower = torch.tensor([[1., 1., 1.]], dtype=torch.float64)
    upper = torch.tensor([[2., 2., 2.]], dtype=torch.float64)
    lb, ub = _bounds(VariablePower(), lower, upper,
                     opts={'undefined_relaxation_rows': True}, method=method)
    assert torch.isneginf(lb[0, 0]) and torch.isposinf(ub[0, 0])
    assert lb[0, 1] <= 2 and ub[0, 1] >= 4
    torch.testing.assert_close(lb[0, 1], lb.new_tensor(2), rtol=0, atol=1e-11)
    torch.testing.assert_close(ub[0, 1], ub.new_tensor(4), rtol=0, atol=1e-11)
    _sampled_sound(VariablePower(), lower, upper, lb, ub, n=1000)


def test_tan_inverse_masks_use_values_without_period_reduction():
    # tan(upper) > pi/2: reducing this atan input modulo pi used to select
    # incompatible convex/concave masks and produce an invalid upper bound.
    class TanSquare(nn.Module):
        def forward(self, x):
            return torch.tan(x) ** 2

    lower = torch.tensor([[0.6234595931993638],
                          [0.6234595931993638 - torch.pi]], dtype=torch.float64)
    upper = torch.tensor([[1.1951670796895981],
                          [1.1951670796895981 - torch.pi]], dtype=torch.float64)
    lb, ub = _bounds(TanSquare(), lower, upper,
                     opts={'undefined_relaxation_rows': True})
    _sampled_sound(TanSquare(), lower, upper, lb, ub, n=1000)


def test_constant_product_interval_with_an_infinite_endpoint():
    # c * [l, u] picks the endpoint by the sign of c; 0 * (anything real) = 0.
    # The masked sum it replaces gave 0 * inf = NaN, widened to (-inf, inf).
    from auto_LiRPA.operators.base import Interval
    from auto_LiRPA.operators.bivariate import BoundMul
    inf = torch.inf
    c = torch.tensor([[2.0, 0.0, -3.0, 1.0]], dtype=torch.float64)
    y = Interval(torch.tensor([[1.0, -inf, 2.0, -1.0]], dtype=torch.float64),
                 torch.tensor([[inf, inf, inf, 3.0]], dtype=torch.float64))
    out = BoundMul.interval_propagate_constant(Interval(c, c), y)
    assert torch.equal(out.lower, torch.tensor([[2.0, 0.0, -inf, -1.0]], dtype=torch.float64))
    assert torch.equal(out.upper, torch.tensor([[inf, 0.0, -6.0, 3.0]], dtype=torch.float64))


def test_affine_interval_with_an_infinite_endpoint():
    # The center/radius form gave inf - inf = NaN in every output row (and the
    # final IBP bound with C lost its objective row whenever any other output
    # element was unbounded); endpoints are now chosen by the weight sign.
    from auto_LiRPA.operators.base import Interval
    from auto_LiRPA.operators.linear import BoundLinear
    inf = torch.inf
    x = Interval(torch.tensor([[-10.0, 2.7, -inf]], dtype=torch.float64),
                 torch.tensor([[0.0, inf, -2.7]], dtype=torch.float64))
    W = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [1.0, 1.0, 1.0]],
                     dtype=torch.float64)
    for weight in (W, W.unsqueeze(0)):
        out = BoundLinear._bound_box(x, weight)
        margin = box_rounding_error(weight, x.lower, x.upper)
        assert_enclosed(out.lower, [[-10.0, 2.7, -inf]], margin, -1)
        assert_enclosed(out.upper, [[0.0, inf, inf]], margin, 1)


def test_even_power_interval_of_an_unbounded_input_is_nonnegative():
    # x^2 and x^4 of (-inf, inf) or [-inf, 3] are >= 0; a masked product used to
    # give 0 * inf = NaN, widened to -inf.
    from auto_LiRPA.operators.base import Interval
    from auto_LiRPA.operators.activations import BoundSqr
    inf = torch.inf
    h = Interval(torch.tensor([[-inf, -inf, 2.0, -3.0]], dtype=torch.float64),
                 torch.tensor([[inf, 3.0, inf, -1.0]], dtype=torch.float64))
    out = BoundSqr.interval_propagate(BoundSqr(), h)
    assert torch.equal(out.lower, torch.tensor([[0.0, 0.0, 4.0, 1.0]], dtype=torch.float64))
    assert torch.equal(out.upper, torch.tensor([[inf, inf, inf, 9.0]], dtype=torch.float64))

    class Fourth(nn.Module):
        def forward(self, x):
            return torch.pow(torch.reciprocal(x), 4)

    lower = torch.tensor([[-1.0]], dtype=torch.float64)
    upper = torch.tensor([[2.0]], dtype=torch.float64)
    lb, ub = _bounds(Fourth(), lower, upper, method='IBP')
    assert lb[0, 0] == 0 and torch.isposinf(ub[0, 0])


@pytest.mark.parametrize('store_intervals', [False, True])
def test_interval_stop_square_pole_plus_finite_residual(store_intervals):
    lower = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float64)
    upper = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    lb, ub = _bounds(SquarePoleResidual(), lower, upper, opts=INTERVAL_OPTS,
                     store_intervals=store_intervals)
    assert torch.isfinite(lb).all() and (lb >= -1e-9).all()
    assert torch.isposinf(ub).all()
    _sampled_sound(SquarePoleResidual(), lower, upper, lb, ub, n=2000)


def test_interval_stop_preserves_other_terms_and_returned_plane():
    lower = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float64)
    upper = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    model = TanhPoleResidual()
    lb_old, ub_old = _bounds(model, lower, upper, store_intervals=True)
    net, (lb, ub, A) = _bounds(model, lower, upper, opts=INTERVAL_OPTS,
                               store_intervals=True, return_A=True)
    # Interval arithmetic loses x1 - x1; the stopped CROWN row retains it.
    # The final sum combines r^2 in [1, 9], x1, -x1, and .25*x2.
    # Include those operands before cancellation when measuring its margin.
    margin = 10 * box_rounding_error(
        lower.new_tensor([[[1., 1., -1., .25]]]),
        lower.new_tensor([[1., 0., 0., 0.]]),
        upper.new_tensor([[9., 1., 1., 1.]])).squeeze()
    assert_enclosed(lb_old, 0, margin, -1)
    assert_enclosed(lb, 1, margin, -1)
    assert ub.item() < ub_old.item()
    entry = A[net.output_name[0]][net.input_name[0]]
    assert torch.equal(entry['lA'], torch.tensor([[[0., 0., 0.25]]], dtype=lb.dtype))
    assert torch.isfinite(entry['lbias']).all()
    _sampled_sound(model, lower, upper, lb, ub, n=2000)
    points = lower + (upper - lower) * torch.rand(2000, 3, dtype=lb.dtype)
    values = model(points)
    assert (points @ entry['lA'][0].T + entry['lbias'][0] <= values + 1e-9).all()
    assert (points @ entry['uA'][0].T + entry['ubias'][0] >= values - 1e-9).all()


def test_interval_stop_multiple_undefined_binary_inputs():
    lower = torch.tensor([[0., 1., 0., 1., 0.]], dtype=torch.float64)
    upper = torch.tensor([[1., 2., 1., 2., 1.]], dtype=torch.float64)
    lb_old, _ = _bounds(ProductPoleResidual(), lower, upper, store_intervals=True)
    net, (lb, ub, _) = _bounds(ProductPoleResidual(), lower, upper, opts=INTERVAL_OPTS,
                               store_intervals=True, return_A=True)
    assert_enclosed(lb_old, .25, 10 * rounding_error(lb_old.new_tensor(.25), 5), -1)
    assert lb.item() >= 1.
    product = next(n for n in net.nodes() if type(n).__name__ == 'BoundMul')
    assert torch.isposinf(product.inputs[0].upper[0, 0])
    assert torch.isneginf(product.inputs[1].lower[0, 0])
    assert all(n.is_lower_bound_current() and n.is_upper_bound_current()
               for n in product.inputs)
    _sampled_sound(ProductPoleResidual(), lower, upper, lb, ub, n=2000)


def test_interval_stop_without_stored_interval_uses_input_intervals():
    # A node that feeds only linear consumers has no stored bounds; its interval is
    # then computed from its inputs' bounds, never looser than the row fallback.
    lower = torch.tensor([[-1., 0., 0.]], dtype=torch.float64)
    upper = torch.tensor([[1., 1., 1.]], dtype=torch.float64)
    lb, ub = _bounds(TanhPoleResidual(), lower, upper, opts=INTERVAL_OPTS)
    lb_old, ub_old = _bounds(TanhPoleResidual(), lower, upper)
    assert (lb >= lb_old).all() and (ub <= ub_old).all()
    _sampled_sound(TanhPoleResidual(), lower, upper, lb, ub, n=2000)


@pytest.mark.parametrize('model,lower,upper', [
    (RecipRows(), [[-1., 0.], [0., 0.], [1., 0.]],
     [[1., 1.], [2., 1.], [2., 1.]]),
    (TanRow(), [[0., 0.], [0., 0.]], [[1., 1.], [0.3, 1.]]),
    (SquarePoleResidual(), [[-1., 0., 0.], [0., 0., 0.]],
     [[1., 1., 1.], [2., 1., 1.]]),
    (TanhPoleResidual(), [[-1., 0., 0.], [1., 0., 0.]],
     [[1., 1., 1.], [2., 1., 1.]]),
    (InfiniteInputResidual(), [[-1000., 0., 0.], [0., 0., 0.]],
     [[1000., 1., 1.], [800., 1., 1.]]),
])
def test_interval_stop_sampled_soundness(model, lower, upper):
    lower, upper = torch.tensor(lower, dtype=torch.float64), torch.tensor(upper, dtype=torch.float64)
    lb, ub = _bounds(model, lower, upper, opts=INTERVAL_OPTS, store_intervals=True)
    assert not torch.isnan(lb).any() and not torch.isnan(ub).any()
    _sampled_sound(model, lower, upper, lb, ub, n=2000)


def test_interval_stop_needed_infinite_endpoint_still_discards_row():
    lower = torch.tensor([[-1., 0.]], dtype=torch.float64)
    upper = torch.tensor([[1., 1.]], dtype=torch.float64)
    net, (lb, ub, A) = _bounds(RecipRows(), lower, upper, opts=INTERVAL_OPTS,
                               store_intervals=True, return_A=True)
    assert torch.isneginf(lb[0, 0]) and torch.isposinf(ub[0, 0])
    margin = 10 * box_rounding_error(
        lower.new_tensor([[[0., 2.]]]), lower, upper).squeeze()
    assert_enclosed(lb[0, 2], 0, margin, -1)
    assert_enclosed(ub[0, 2], 2, margin, 1)
    entry = A[net.output_name[0]][net.input_name[0]]
    assert (entry['lA'][0, 0] == 0).all() and torch.isneginf(entry['lbias'][0, 0])


def _stop_test_node():
    from auto_LiRPA.operators.activations import BoundSqr
    from auto_LiRPA.operators.base import Bound
    inp = Bound()
    inp.output_shape = torch.Size([2, 3])
    inp.lower = torch.zeros(2, 3, dtype=torch.float64)
    inp.upper = torch.ones(2, 3, dtype=torch.float64)
    node = BoundSqr(inputs=[inp])
    node.output_shape = inp.output_shape
    node.lower = torch.tensor([[0., -torch.inf, 2.], [1., 2., 3.]], dtype=torch.float64)
    node.upper = torch.tensor([[torch.inf, 4., 5.], [4., 6., 7.]], dtype=torch.float64)
    return node


@pytest.mark.parametrize('traced_batch', [1, 2])
def test_interval_stop_signs_masks_and_zero_times_infinity(traced_batch):
    from auto_LiRPA.backward_bound import _stop_undefined_interval
    node = _stop_test_node()
    node.output_shape = node.inputs[0].output_shape = torch.Size([traced_batch, 3])
    no_lower = torch.tensor([[True, True, False], [False, True, False]])
    no_upper = torch.tensor([[False, True, True], [False, True, False]])
    A = torch.tensor([[[2., -3., 4.], [5., -2., 3.]],
                      [[0., 0., 1.], [2., 3., -1.]]], dtype=torch.float64)
    lA, uA, lb, ub = _stop_undefined_interval(
        node, A, A, no_lower, no_upper, (node.lower, node.upper))
    assert torch.equal(lA, A.new_tensor([[[0., 0., 4.], [5., 0., 3.]],
                                        [[0., 0., 1.], [2., 0., -1.]]]))
    assert torch.equal(uA, A.new_tensor([[[2., 0., 0.], [5., 0., 3.]],
                                        [[0., 0., 0.], [2., 0., -1.]]]))
    assert torch.equal(lb, torch.tensor([[-12., -12.], [0., 6.]], dtype=A.dtype))
    assert torch.equal(ub, torch.tensor([[torch.inf, -4.], [5., 18.]], dtype=A.dtype))


def test_interval_stop_unary_power_with_constant_exponent():
    class Fourth(nn.Module):
        def forward(self, x):
            return torch.pow(torch.reciprocal(x[:, :1]), 4) + x[:, 1:2] - x[:, 1:2]

    lower = torch.tensor([[-1., 0.]], dtype=torch.float64)
    upper = torch.tensor([[1., 1.]], dtype=torch.float64)
    lb, ub = _bounds(Fourth(), lower, upper, opts=INTERVAL_OPTS, store_intervals=True)
    # Keep the defined zero-slope tangent, including its power relaxation margin.
    lb_without_stop, _ = _bounds(Fourth(), lower, upper, store_intervals=True)
    margin = 10 * rounding_error(upper[:, 1:2] * 2, 2)
    torch.testing.assert_close(lb, lb_without_stop, rtol=0, atol=margin.item())
    assert torch.isposinf(ub).all()
    _sampled_sound(Fourth(), lower, upper, lb, ub, n=2000)


@pytest.mark.parametrize('unusable', [
    'stale_lower', 'stale_upper', 'wrong_lower_batch', 'wrong_upper_batch',
    'output_shape', 'broadcast_input', 'mixed_elements', 'non_tensor_A',
    'nan_interval', 'inverted_interval', 'wrong_binary_input_batch',
])
def test_interval_stop_requires_current_identity_intervals(unusable):
    from auto_LiRPA.backward_bound import _stop_undefined_interval
    from auto_LiRPA.operators.linear import BoundLinear
    node = _stop_test_node()
    A = torch.ones(1, 2, 3, dtype=torch.float64)
    if unusable == 'stale_lower':
        node._is_lower_bound_current = False
    elif unusable == 'stale_upper':
        node._is_upper_bound_current = False
    elif unusable == 'wrong_lower_batch':
        node.lower = node.lower[:1]
    elif unusable == 'wrong_upper_batch':
        node.upper = node.upper[:1]
    elif unusable == 'output_shape':
        node.output_shape = torch.Size([2, 1, 3])
    elif unusable == 'broadcast_input':
        node.inputs.append(node.inputs[0])
        node.inputs[0].output_shape = torch.Size([2, 1])
    elif unusable == 'mixed_elements':
        node = BoundLinear(inputs=node.inputs)
    elif unusable == 'non_tensor_A':
        A = 1
    elif unusable == 'nan_interval':
        node.lower[0, 0] = torch.nan
    elif unusable == 'inverted_interval':
        node.lower[0, 2] = 6.
    elif unusable == 'wrong_binary_input_batch':
        from auto_LiRPA.operators.base import Bound
        from auto_LiRPA.operators.bivariate import BoundMul
        other = Bound()
        other.output_shape = node.output_shape
        other.lower = torch.zeros(1, 3, dtype=torch.float64)
        product = BoundMul(inputs=[node.inputs[0], other])
        product.requires_input_bounds = [0, 1]
        product.output_shape = node.output_shape
        product.lower, product.upper = node.lower, node.upper
        node = product
    mask = torch.ones(2, 3, dtype=torch.bool)
    assert _stop_undefined_interval(node, A, None, mask, mask) is None
