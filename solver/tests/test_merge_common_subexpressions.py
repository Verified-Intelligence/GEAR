"""Graph pass merge_common_subexpressions: structurally identical nodes (same op, same
inputs, same attributes; constants with equal values) become one node, so e.g.
(x1 - x0*x0) * (x1 - x0*x0) built twice is a square, not a bilinear term."""
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.bound_ops import BoundMul, BoundSqr
from auto_LiRPA.perturbations import PerturbationLpNorm


class _Gather(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.register_buffer('idx', torch.tensor(idx))

    def forward(self, x):
        return torch.index_select(x, -1, self.idx)


class _Diff(nn.Module):
    """x1 - x0 * x0 with its own copies of every constant, as each generated module has."""

    def __init__(self):
        super().__init__()
        self.g0a, self.g0b, self.g1 = _Gather([0]), _Gather([0]), _Gather([1])

    def forward(self, x):
        return self.g1(x) - self.g0a(x) * self.g0b(x)


class RosenbrockTerm(nn.Module):
    """100 (x1 - x0^2)^2 with the inner expression built twice from separate modules,
    next to x0 * x1 which must stay a product."""

    def __init__(self):
        super().__init__()
        self.a, self.b = _Diff(), _Diff()
        self.p0, self.p1 = _Gather([0]), _Gather([1])
        self.register_buffer('scale', torch.tensor(100.0))

    def forward(self, x):
        return torch.cat([self.scale * (self.a(x) * self.b(x)), self.p0(x) * self.p1(x)], dim=-1)


def _bounds(model, lo=-2.0, hi=2.0):
    x = torch.zeros(1, 2)
    # The test checks that the square rewrite gives lb >= 0 (to 1e-6); the A67 rounding enclosure (default on) widens
    # that by ~4e-4 on this graph, so it is off here.
    bm = BoundedModule(model, x, bound_opts={'fp_enclosure': {'affine_backward': False}}, device='cpu')
    ptb = BoundedTensor(x, PerturbationLpNorm(norm=float('inf'), x_L=torch.full((1, 2), lo),
                                              x_U=torch.full((1, 2), hi)))
    lb, ub = bm.compute_bounds(x=(ptb,), method='CROWN')
    return bm, lb, ub


def test_repeated_subexpression_becomes_a_square():
    bm, lb, ub = _bounds(RosenbrockTerm())
    types = [type(n) for n in bm.nodes()]
    assert types.count(BoundSqr) == 2          # x0*x0 and a*a
    assert types.count(BoundMul) >= 1          # the scale and x0*x1 stay products
    # A square is >= 0; as a product of two independent copies it went far below.
    assert float(lb[0, 0]) >= -1e-6
    x = torch.tensor([[0.7, -1.3]])
    assert torch.allclose(bm(x), RosenbrockTerm()(x))
