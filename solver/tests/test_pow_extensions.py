"""Powers that BoundPow rejected: negative exponents and variable exponents.

* `pow_negative_exponent`: x ** y with a constant y < 0 is convex and decreasing on
  (0, inf); lower plane = tangent, upper plane = secant, IBP = [u^y, l^y]. A base
  interval reaching 0 keeps only its lower side (with undefined_relaxation_rows).
* `pow_variable_exponent_exp_log`: x ** y with a non-constant y is rewritten as
  exp(y * log(x)), exact for the positive base on which a real power is defined.
* A non-integer power's base interval is intersected with [0, inf) under
  domain_aware_ops in the relaxation too (it used to produce NaN planes).
"""

import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.operators import BoundExp, BoundLog, BoundPow
from auto_LiRPA.operators.base import OperatorCapabilityError
from auto_LiRPA.perturbations import PerturbationLpNorm


class NegPow(nn.Module):
    def forward(self, x):
        return torch.cat([torch.pow(5 * x[:, 0:1], -0.1),
                          x[:, 1:2] * torch.pow(x[:, 0:1], -0.3)], dim=1)


class VarPow(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('c', torch.tensor(0.157))
        self.register_buffer('d', torch.tensor(62.5))

    def forward(self, x):
        return torch.cat([torch.pow(self.d - x[:, 0:1], x[:, 1:2]) / 1000,
                          x[:, 0:1] / torch.pow(self.c, x[:, 1:2])], dim=1)


class SqrtPow(nn.Module):
    def forward(self, x):
        return torch.pow(x, 0.5) + torch.pow(x, 1.5)


def _net(model, lower, opts):
    return BoundedModule(model, lower, bound_opts=dict(opts))


def _bounds(net, lower, upper, method):
    x = BoundedTensor(lower, PerturbationLpNorm(x_L=lower, x_U=upper))
    return net.compute_bounds(x=(x,), method=method)


def _check_sound(model, lower, upper, lb, ub, n=20000):
    torch.manual_seed(0)
    for b in range(lower.shape[0]):
        pts = lower[b] + (upper[b] - lower[b]) * torch.rand(
            n, lower.shape[1], dtype=lower.dtype)
        y = model(pts)
        finite = torch.isfinite(y)
        assert (~finite | (y >= lb[b] - 1e-9)).all()
        assert (~finite | (y <= ub[b] + 1e-9)).all()


@pytest.mark.parametrize('method', ['IBP', 'CROWN'])
def test_negative_fractional_exponent(method):
    lower = torch.tensor([[0.1, -1.0], [2.0, 0.5]], dtype=torch.float64)
    upper = torch.tensor([[10.0, 2.0], [3.0, 1.0]], dtype=torch.float64)
    net = _net(NegPow(), lower, {'pow_negative_exponent': True})
    lb, ub = _bounds(net, lower, upper, method)
    assert torch.isfinite(lb).all() and torch.isfinite(ub).all()
    _check_sound(NegPow(), lower, upper, lb, ub)
    # Exact monotone interval for the first row.
    assert torch.allclose(lb[:, 0], torch.pow(5 * upper[:, 0], -0.1)) or method == 'CROWN'


def test_negative_exponent_touching_zero_keeps_lower_side():
    lower = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    upper = torch.tensor([[4.0, 2.0]], dtype=torch.float64)
    net = _net(NegPow(), lower, {'pow_negative_exponent': True,
                                 'undefined_relaxation_rows': True})
    lb, ub = _bounds(net, lower, upper, 'CROWN')
    assert torch.isfinite(lb).all() and torch.isposinf(ub).all()
    _check_sound(NegPow(), lower, upper, lb, ub)


def test_negative_exponent_off_by_default():
    lower = torch.tensor([[0.1, -1.0]], dtype=torch.float64)
    upper = torch.tensor([[10.0, 2.0]], dtype=torch.float64)
    net = _net(NegPow(), lower, {})
    with pytest.raises(OperatorCapabilityError):
        _bounds(net, lower, upper, 'CROWN')


@pytest.mark.parametrize('method', ['IBP', 'CROWN'])
def test_variable_exponent_becomes_exp_log(method):
    lower = torch.tensor([[0.0, 0.0], [20.0, 1.0]], dtype=torch.float64)
    upper = torch.tensor([[25.6, 5.0], [21.0, 1.2]], dtype=torch.float64)
    opts = {'pow_variable_exponent_exp_log': True, 'domain_aware_ops': True,
            'compare_crown_with_ibp_intermediate': True}
    net = _net(VarPow(), lower, opts)
    types = [type(n) for n in net.nodes()]
    assert BoundPow not in types and BoundLog in types and BoundExp in types
    lb, ub = _bounds(net, lower, upper, method)
    assert torch.isfinite(lb).all() and torch.isfinite(ub).all()
    _check_sound(VarPow(), lower, upper, lb, ub)


def test_variable_exponent_off_by_default_keeps_pow():
    lower = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    net = _net(VarPow(), lower, {})
    assert any(type(n) == BoundPow for n in net.nodes())


def test_noninteger_power_relaxation_uses_the_restricted_domain():
    # The base interval [-1, 4] reaches outside the domain of x ** 0.5 / x ** 1.5;
    # the relaxation is built on [0, 4] (it used to be NaN).
    lower = torch.tensor([[-1.0]], dtype=torch.float64)
    upper = torch.tensor([[4.0]], dtype=torch.float64)
    net = _net(SqrtPow(), lower, {'domain_aware_ops': True})
    lb, ub = _bounds(net, lower, upper, 'CROWN')
    assert torch.isfinite(lb).all() and torch.isfinite(ub).all()
    _check_sound(SqrtPow(), torch.zeros_like(lower), upper, lb, ub)
