"""Graph pass merge_identical_gather: identical gathers become one node, so the
product of two gathers of the same entries is a square, not a bilinear term."""
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.bound_ops import BoundGather, BoundMul, BoundSqr
from auto_LiRPA.perturbations import PerturbationLpNorm


class TwoGatherSquare(nn.Module):
    """x_i * x_i written as the product of two separate gathers (as generated models do),
    plus a gather of different entries that must stay separate."""

    def __init__(self):
        super().__init__()
        self.register_buffer('i1', torch.tensor([0, 1]))
        self.register_buffer('i2', torch.tensor([0, 1]))
        self.register_buffer('j', torch.tensor([1, 0]))

    def forward(self, x):
        sq = torch.index_select(x, -1, self.i1) * torch.index_select(x, -1, self.i2)
        cross = torch.index_select(x, -1, self.i1) * torch.index_select(x, -1, self.j)
        return torch.cat([sq, cross], dim=-1)


def _bounds(model):
    x = torch.zeros(1, 2)
    lo, hi = -torch.full((1, 2), 100.0), torch.full((1, 2), 100.0)
    bm = BoundedModule(model, x, device='cpu')
    ptb = BoundedTensor(x, PerturbationLpNorm(norm=float('inf'), x_L=lo, x_U=hi))
    lb, ub = bm.compute_bounds(x=(ptb,), method='CROWN')
    return bm, lb, ub


def test_identical_gathers_merge_into_a_square():
    bm, lb, ub = _bounds(TwoGatherSquare())
    types = [type(n) for n in bm.nodes()]
    assert types.count(BoundSqr) == 1
    assert types.count(BoundMul) == 1          # the cross term stays bilinear
    assert types.count(BoundGather) == 2       # i1/i2 merged; j kept
    # x_i^2 on [-100, 100] is >= 0; as a bilinear term of independent copies it was -1e4.
    assert torch.all(lb[0, :2] >= -1e-9)
    assert torch.all(ub[0, :2] >= 1e4 - 1e-6)
    x = torch.tensor([[3.0, -7.0]])
    assert torch.equal(bm(x), TwoGatherSquare()(x))
