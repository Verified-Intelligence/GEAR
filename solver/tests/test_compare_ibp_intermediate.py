"""bound_opts['compare_crown_with_ibp_intermediate']: intermediate CROWN bounds are
intersected with IBP, the final bound is left alone (so return_A still works)."""
import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.operators.base import OperatorDomainError
from auto_LiRPA.perturbations import PerturbationLpNorm


class InvCube(nn.Module):
    def forward(self, x):
        return 1.0 / x ** 3


def _bounds(opts):
    torch.set_default_dtype(torch.float64)
    x = torch.full((1, 1), 1.0)
    lo, hi = torch.full((1, 1), 1e-3), torch.full((1, 1), 5.0)
    bm = BoundedModule(InvCube(), x, device='cpu', bound_opts=opts)
    ptb = BoundedTensor(x, PerturbationLpNorm(norm=float('inf'), x_L=lo, x_U=hi))
    final = bm.final_name
    input_name = bm.input_name[0]
    return bm.compute_bounds(x=(ptb,), method='CROWN', return_A=True,
                             needed_A_dict={final: [input_name]})


def test_intermediate_ibp_keeps_a_positive_denominator_positive():
    # CROWN's lower plane for x^3 on [1e-3, 5] dips below 0, so without IBP the
    # reciprocal sees an interval containing 0 (schitt_s330). IBP gives [1e-9, 125].
    with pytest.raises(OperatorDomainError):
        _bounds({})
    lb, ub, A = _bounds({'compare_crown_with_ibp_intermediate': True})
    # Sound (1/x^3 >= 1/125 on the box) and finite; the final bound itself is CROWN's.
    assert torch.isfinite(lb).all() and 0.0 <= float(lb) <= 1.0 / 125.0
    assert float(ub) >= 1e9
    assert A


def test_nan_from_ibp_does_not_poison_the_crown_bound():
    # hs102 (GPU): a NaN interval endpoint (e.g. 0 * inf) taken through torch.max
    # replaced a finite CROWN bound, and the next relaxation rejected the NaN.
    from types import SimpleNamespace
    from auto_LiRPA.interval_bound import compare_with_IBP
    nan = float('nan')
    fake = SimpleNamespace(IBP_general=lambda node, **kw: SimpleNamespace(
        lower=torch.tensor([nan, 0.5]), upper=torch.tensor([nan, 2.0])))
    lo, up = compare_with_IBP(fake, None, torch.tensor([-1.0, 0.0]), torch.tensor([3.0, 4.0]))
    assert torch.equal(lo, torch.tensor([-1.0, 0.5]))
    assert torch.equal(up, torch.tensor([3.0, 2.0]))
