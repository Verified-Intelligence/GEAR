"""log is defined on all of (0, inf): bounds must hold for arguments below 1e-6.

hs025: the argument of a log node ranged over [<= 0, 4.4e-16] and a feasible point had
argument 3.3e-16. Restricting the interval to >= 1e-6 inverted it; the resulting
relaxation (and, with `undefined_relaxation_rows`, the interval fallback on the
inverted interval) gave an objective lower bound of 0.693 above the feasible 0.1528.
"""

import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


class LogOfTiny(nn.Module):
    """log(x0) and a square through exp(log(x0)), like hs025's residuals."""

    def forward(self, x):
        y = torch.log(x[:, 0:1])
        r = torch.exp(y) - 0.01
        return torch.cat([y, r * r], dim=1)


@pytest.mark.parametrize("undefined_rows", [False, True])
@pytest.mark.parametrize("method", ["IBP", "CROWN"])
def test_log_bounds_hold_below_one_micro(undefined_rows, method):
    opts = {'domain_aware_ops': True, 'compare_crown_with_ibp_intermediate': True,
            'undefined_relaxation_rows': undefined_rows}
    lo = torch.tensor([[-1e-16]], dtype=torch.float64)
    hi = torch.tensor([[4.4e-16]], dtype=torch.float64)
    model = BoundedModule(LogOfTiny(), torch.full((1, 1), 3e-16, dtype=torch.float64),
                          bound_opts=opts, device='cpu')
    x = BoundedTensor(torch.full((1, 1), 3e-16, dtype=torch.float64),
                      PerturbationLpNorm(x_L=lo, x_U=hi))
    try:
        lb, _ = model.compute_bounds(x=(x,), method=method, bound_upper=False)
    except Exception:
        return      # refusing to bound is sound; a wrong finite bound is not
    point = torch.tensor([[3.3e-16]], dtype=torch.float64)
    value = LogOfTiny()(point)
    assert not torch.isnan(lb).any()
    assert (lb <= value + 1e-12).all(), (lb, value)


@pytest.mark.parametrize('scale', [1e-16, 1e-27, 1e-300])
def test_negative_log_lower_plane_for_tiny_positive_inputs(scale):
    class NegativeLog(nn.Module):
        def forward(self, x):
            return -torch.log(x)

    lower = torch.tensor([[scale]], dtype=torch.float64)
    upper = 2 * lower
    model = NegativeLog()
    net = BoundedModule(model, lower, bound_opts={
        'domain_aware_ops': True, 'compare_crown_with_ibp_intermediate': True,
        'undefined_relaxation_rows': True, 'undefined_relaxation_interval': True}, device='cpu')
    x = BoundedTensor(lower, PerturbationLpNorm(x_L=lower, x_U=upper))
    lb, ub, planes = net.compute_bounds(
        x=(x,), method='CROWN', return_A=True,
        needed_A_dict={net.output_name[0]: [net.input_name[0]]})
    entry = planes[net.output_name[0]][net.input_name[0]]
    points = torch.linspace(1, 2, 101, dtype=lower.dtype).unsqueeze(1) * scale
    values = model(points)
    assert (lb <= values + 1e-12).all()
    assert (values <= ub + 1e-12).all()
    assert (points @ entry['lA'][0].T + entry['lbias'][0] <= values + 1e-12).all()
    assert (points @ entry['uA'][0].T + entry['ubias'][0] >= values - 1e-12).all()


def test_fallback_interval_ignores_inverted_input_bounds():
    # The undefined-row fallback evaluates a node by interval arithmetic from its
    # inputs' stored bounds; an inverted stored interval must not yield a finite bound.
    from auto_LiRPA.backward_bound import _interval_from_inputs

    class Square(nn.Module):
        def forward(self, x):
            return (x - 0.0) * (x - 0.0)

    model = BoundedModule(Square(), torch.zeros(1, 1, dtype=torch.float64),
                          bound_opts={'domain_aware_ops': True}, device='cpu')
    x = BoundedTensor(torch.zeros(1, 1, dtype=torch.float64),
                      PerturbationLpNorm(x_L=torch.full((1, 1), -1.0, dtype=torch.float64),
                                         x_U=torch.full((1, 1), 1.0, dtype=torch.float64)))
    model.compute_bounds(x=(x,), method='IBP')
    final = model[model.final_name]
    src = final.inputs[0]
    src.lower = torch.full_like(src.lower, torch.inf)
    src.upper = torch.full_like(src.upper, -0.01)
    box = _interval_from_inputs(model, final, 1)
    assert box is None or not (box.lower > 0).any(), box.lower
