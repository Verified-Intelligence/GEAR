"""NN smoke on the CPU (G5 WP12a; DESIGN section 7 'NN smoke' and the WP12a parity column).

Three NN-verification models: ``direct:torch`` (``compile_torch`` + the program through the ``__gear_program__`` seam)
vs today's ``parse_module`` path. The two ``BoundedModule``s must give bitwise identical forward values and IBP,
CROWN and alpha-CROWN bounds, on one box and on a batch of boxes. ``parse_module`` is patched to raise on the direct
side. Also: direct programs are isomorphic to ``parse_module``'s tuples on the NN models of ``tests/`` and
``model_defs.py`` that use only DIRECT_TABLE ops, and the 11-of-64 rule admits all of them (DESIGN 'Pending gate').
The CUDA half is the orchestrator's kit (wp12a/nn_smoke_cuda.py).
"""
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

import model_defs
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from gear.io import torch_direct as td
from tests.gear_wp12a_helpers import bits, isomorphic, parse_program


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


class GatedResidual(nn.Module):
    """A residual MLP with a sigmoid gate: h = x W1 + b1; y = W3 (relu(h) * sigmoid(W2 h) + h)."""

    def __init__(self, n=20, width=32, k=5):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(n, width), nn.Linear(width, width), nn.Linear(width, k)

    def forward(self, x):
        h = self.l1(x)
        return self.l3(torch.relu(h) * torch.sigmoid(self.l2(h)) + h)


def _sigmoid_tanh_mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 100), nn.Sigmoid(), nn.Linear(100, 50), nn.Tanh(),
                         nn.Linear(50, 10))


def smoke_models():
    """(name, constructor, input width): the three NN-verification models of the smoke."""
    return [("mnist_6_100", model_defs.mnist_6_100, 784), ("mnist_sigmoid_tanh", _sigmoid_tanh_mlp, 784),
            ("gated_residual", GatedResidual, 20)]


def _make(ctor, seed=0):
    torch.manual_seed(seed)
    return ctor().double().eval()


def direct_and_reference(model, n, box=None):
    lo, hi = box if box is not None else (np.zeros(n), np.ones(n))
    k = model(torch.zeros(1, n)).shape[1]
    tc = td.compile_torch(model, box=(lo, hi), objective=0, le=list(range(1, k)))
    x0 = torch.zeros(1, n)
    ref = BoundedModule(model, (x0,), device="cpu")
    with patch("auto_LiRPA.bound_general.parse_module", side_effect=AssertionError("parse_module called")):
        direct = BoundedModule(td.attach_program(td.computing_graph(tc), tc.program), (x0,), device="cpu")
    return tc, ref, direct


def _bounds(bm, x_l, x_u, method):
    ptb = PerturbationLpNorm(norm=np.inf, x_L=x_l, x_U=x_u)
    x = BoundedTensor((x_l + x_u) / 2, ptb)
    if method == "alpha-CROWN":
        bm.set_bound_opts({"optimize_bound_args": {"iteration": 5, "lr_alpha": 0.1}})
        return bm.compute_bounds(x=(x,), method="CROWN-Optimized")
    return bm.compute_bounds(x=(x,), method=method)


@pytest.mark.parametrize("name, ctor, n", smoke_models(), ids=[m[0] for m in smoke_models()])
def test_nn_smoke_identical_bounded_module_results(name, ctor, n):
    model = _make(ctor)
    tc, ref, direct = direct_and_reference(model, n)
    assert tc.path == "direct:torch" and tc.objective_row is None and not tc.structural_outputs
    assert [type(m).__name__ for m in ref.nodes()] == [type(m).__name__ for m in direct.nodes()]
    gen = torch.Generator().manual_seed(1)
    X = torch.rand(4, 1, n, generator=gen)
    for x in X:
        assert torch.equal(bits(ref(x)), bits(direct(x)))
    for batch in (1, 3):
        centers = torch.rand(batch, n, generator=gen)
        x_l, x_u = (centers - 0.02).clamp(0, 1), (centers + 0.02).clamp(0, 1)
        for method in ("IBP", "CROWN", "alpha-CROWN"):
            lb_r, ub_r = _bounds(ref, x_l, x_u, method)
            lb_d, ub_d = _bounds(direct, x_l, x_u, method)
            assert torch.equal(bits(lb_r), bits(lb_d)) and torch.equal(bits(ub_r), bits(ub_d)), (name, method, batch)


# ---- isomorphism with parse_module and the 11-of-64 gate on the NN models -----------------------------------------
class LinearModel(nn.Module):
    """The model of tests/test_linear_model.py (repository root)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 3)

    def forward(self, x):
        return self.fc(x)


def iso_models():
    out = [(name, ctor, n) for name, ctor, n in smoke_models()]
    out += [("tests.linear_model", LinearModel, 256),
            ("tests.interval_bound.linear", lambda: nn.Linear(3, 2), 3),
            ("tests.interval_bound.relu_linear", lambda: nn.Sequential(nn.ReLU(), nn.Linear(2, 1, bias=False)), 2)]
    for name in ("mnist_tiny_mlp", "mnist_fc", "mnist_fc_2_200", "mnist_9_100", "mnist_6_200", "mnist_9_200",
                 "mnist_fc1", "mnist_fc_3_512", "mnist_fc_4_512", "mnist_fc_5_512"):
        out.append((f"model_defs.{name}", getattr(model_defs, name), 784))
    return out


@pytest.mark.parametrize("name, ctor, n", iso_models(), ids=[m[0] for m in iso_models()])
def test_direct_tuples_isomorphic_to_parse_module(name, ctor, n):
    model = _make(ctor)
    lo, hi = np.zeros(n), np.ones(n)
    valid = td.domain_valid(model, td.candidates(lo, hi))
    assert len(valid) >= td.N_POINTS  # the 11-of-64 gate
    k = model(torch.zeros(1, n)).shape[1]
    tc = td.compile_torch(model, box=(lo, hi), objective=0, le=list(range(1, k)))
    assert isomorphic(tc.program, parse_program(model, torch.zeros(1, n)))
