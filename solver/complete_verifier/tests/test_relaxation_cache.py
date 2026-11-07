"""The relaxation of a non-optimised activation depends only on its input bounds.

CROWN rebuilds it on every backward pass, and one compute_bounds call runs a backward
pass per intermediate node, so on a deep graph the same relaxation is rebuilt hundreds
of times (Case57: 186 s of a 250 s bound pass). Reusing it must not change any bound.
"""
import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.operators import BoundMul, BoundTanh
from auto_LiRPA.perturbations import PerturbationLpNorm


@pytest.fixture(autouse=True)
def _double_default():
    # The solver runs with double_fp (torch.set_default_dtype(float64)); the tanh
    # relaxation tables are built in the default dtype.
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


class Deep(nn.Module):
    """tanh MLP unrolled over steps with a product, so every relax is revisited often."""

    def __init__(self, steps=4, width=6):
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.ModuleList([nn.Linear(width if i else 2, width) for i in range(steps)])
        self.out = nn.Linear(width, 3)

    def forward(self, x):
        h = x
        for lin in self.lin:
            h = torch.tanh(lin(h))
            h = h * (h + 0.5)
        return self.out(h)


def _bounds(cache, count=None):
    model = Deep().double()
    net = BoundedModule(model, torch.zeros(1, 2, dtype=torch.float64), device="cpu",
                        bound_opts={"relaxation_cache": cache})
    if count is not None:
        for node in net.nodes():
            if isinstance(node, (BoundTanh, BoundMul)):
                orig = node.bound_relax

                def counted(*a, _orig=orig, _n=type(node).__name__, **k):
                    count[_n] = count.get(_n, 0) + 1
                    return _orig(*a, **k)
                node.bound_relax = counted
    x = BoundedTensor(torch.zeros(3, 2, dtype=torch.float64),
                      PerturbationLpNorm(x_L=-torch.ones(3, 2, dtype=torch.float64),
                                         x_U=torch.ones(3, 2, dtype=torch.float64)))
    return net, x, net.compute_bounds(x=(x,), method="CROWN")


def test_cached_crown_bounds_are_bitwise_identical():
    _, _, (lb0, ub0) = _bounds(cache=False)
    _, _, (lb1, ub1) = _bounds(cache=True)
    assert torch.equal(lb0, lb1) and torch.equal(ub0, ub1)


def test_cache_removes_repeated_relaxations():
    off, on = {}, {}
    _bounds(cache=False, count=off)
    _bounds(cache=True, count=on)
    # Every activation is relaxed at least once per distinct set of input bounds.
    assert on["BoundTanh"] >= 4 and on["BoundMul"] >= 4
    assert on["BoundTanh"] * 3 <= off["BoundTanh"]
    assert on["BoundMul"] * 3 <= off["BoundMul"]


def test_cache_follows_new_and_in_place_updated_bounds():
    net, x, (lb0, _) = _bounds(cache=True)
    # Same input again: identical bounds (cache hits are harmless).
    lb1, _ = net.compute_bounds(x=(x,), method="CROWN")
    assert torch.equal(lb0, lb1)
    # Tighter box built in place on the same tensor objects: bounds must move.
    x.ptb.x_L.mul_(0.5)
    x.ptb.x_U.mul_(0.5)
    lb2, _ = net.compute_bounds(x=(x,), method="CROWN")
    assert bool((lb2 >= lb0).all()) and not torch.equal(lb2, lb0)
    fresh_net, _, _ = _bounds(cache=False)
    lb_ref, _ = fresh_net.compute_bounds(x=(x,), method="CROWN")
    assert torch.equal(lb2, lb_ref)


def test_cache_is_bypassed_while_alpha_is_optimised():
    _, _, (lb_off, _) = _bounds(cache=False)
    model = Deep().double()
    for cache in (False, True):
        net = BoundedModule(model, torch.zeros(1, 2, dtype=torch.float64), device="cpu",
                            bound_opts={"relaxation_cache": cache,
                                        "optimize_bound_args": {"iteration": 3, "lr_alpha": 0.1}})
        x = BoundedTensor(torch.zeros(3, 2, dtype=torch.float64),
                          PerturbationLpNorm(x_L=-torch.ones(3, 2, dtype=torch.float64),
                                             x_U=torch.ones(3, 2, dtype=torch.float64)))
        lb, _ = net.compute_bounds(x=(x,), method="CROWN-Optimized")
        if not cache:
            lb_ref = lb
    assert torch.allclose(lb, lb_ref)
    assert bool((lb >= lb_off - 1e-12).all())
