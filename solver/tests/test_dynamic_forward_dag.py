"""Dynamic-forward regressions for shared coefficient columns on DAGs."""
import pytest
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.operators.relu import BoundRelu
from auto_LiRPA.perturbations import PerturbationLpNorm

_DT = torch.float64


class _Diamond(nn.Module):
    """Two branches whose independent ReLU errors must use separate columns."""

    def __init__(self, sub):
        super().__init__()
        self.l = nn.Linear(1, 1, dtype=_DT)
        self.r = nn.Linear(1, 1, dtype=_DT)
        self.lo = nn.Linear(1, 1, dtype=_DT)
        self.ro = nn.Linear(1, 1, dtype=_DT)
        self.sub = sub
        with torch.no_grad():
            self.l.weight.fill_(1.0);  self.l.bias.fill_(0.5)
            self.r.weight.fill_(1.0);  self.r.bias.fill_(-0.5)
            self.lo.weight.fill_(-2.0 if sub else 2.0); self.lo.bias.zero_()
            self.ro.weight.fill_(-3.0); self.ro.bias.zero_()

    def forward(self, x):
        a = self.lo(torch.relu(self.l(x)))
        b = self.ro(torch.relu(self.r(x)))
        return a - b if self.sub else a + b


class _Residual(nn.Module):
    def __init__(self, blocks=3):
        super().__init__()
        torch.manual_seed(1)
        self.b = nn.ModuleList([nn.Linear(4, 4, dtype=_DT) for _ in range(blocks)])
        self.s = nn.ModuleList([nn.Linear(4, 4, dtype=_DT) for _ in range(blocks)])
        self.o = nn.Linear(4, 3, dtype=_DT)

    def forward(self, x):
        for block, skip in zip(self.b, self.s):
            x = torch.relu(block(x)) + skip(x)
        return self.o(x)


def _diamond_bounds(model, max_dim=int(1e9), c_dtype=None):
    x_L = torch.tensor([[-1.]], dtype=_DT)
    x_U = torch.tensor([[1.]], dtype=_DT)
    bm = BoundedModule(model, (x_L + x_U) / 2, device='cpu',
                       bound_opts={'forward_max_dim': max_dim})
    bx = BoundedTensor((x_L + x_U) / 2, PerturbationLpNorm(
        norm=float('inf'), x_L=x_L, x_U=x_U))
    C = torch.ones(1, 1, 1, dtype=c_dtype or _DT)
    return bm, bm.compute_bounds(x=(bx,), method='dynamic-forward', C=C)


def _bounds(model, max_dim=int(1e9), eps=1.0):
    x_L = torch.full((1, 4), -eps, dtype=_DT)
    x_U = torch.full((1, 4), eps, dtype=_DT)
    bm = BoundedModule(model, (x_L + x_U) / 2, device='cpu',
                       bound_opts={'forward_max_dim': max_dim})
    bx = BoundedTensor((x_L + x_U) / 2, PerturbationLpNorm(
        norm=float('inf'), x_L=x_L, x_U=x_U))
    C = torch.eye(3, dtype=_DT).unsqueeze(0)
    return bm, bm.compute_bounds(x=(bx,), method='dynamic-forward', C=C)


@pytest.mark.parametrize('sub', [False, True], ids=['add', 'sub'])
def test_a_diamond_keeps_a_point_independent_variables_would_cancel_away(sub):
    model = _Diamond(sub)
    _, (lower, upper) = _diamond_bounds(model)
    with torch.no_grad():
        value = model(torch.tensor([[0.5]], dtype=_DT))
    expected = -2.0 if sub else 2.0
    torch.testing.assert_close(value, torch.tensor([[expected]], dtype=_DT))
    assert (lower <= value + 1e-9).all(), (lower, value)
    assert (upper >= value - 1e-9).all(), (upper, value)


def test_a_dag_survives_windowing():
    """A window is a slice of the one column layout, not of each branch's own."""
    model = _Residual()
    _, (want_l, want_u) = _bounds(model)
    _, (lower, upper) = _bounds(model, max_dim=1)
    torch.testing.assert_close(lower, want_l, rtol=0, atol=1e-12)
    torch.testing.assert_close(upper, want_u, rtol=0, atol=1e-12)


def test_reusing_input_rebuilds_the_variable_layout():
    model = _Diamond(sub=True)
    x_L = torch.tensor([[-1.]], dtype=_DT)
    x_U = torch.tensor([[1.]], dtype=_DT)
    x = (x_L + x_U) / 2
    bm = BoundedModule(model, x, device='cpu')
    bx = BoundedTensor(x, PerturbationLpNorm(
        norm=float('inf'), x_L=x_L, x_U=x_U))

    # First a propagation over one branch, so its layout covers only one ReLU;
    # then the whole graph with x=None, which must rebuild rather than reuse it.
    bm.compute_bounds(
        x=(bx,), method='dynamic-forward',
        final_node_name=bm.relus[0].name, C=torch.ones(1, 1, 1, dtype=_DT))
    reused = bm.compute_bounds(
        x=None, method='dynamic-forward', C=torch.ones(1, 1, 1, dtype=_DT))
    _, fresh = _diamond_bounds(model)
    torch.testing.assert_close(reused, fresh, rtol=0, atol=1e-12)


def test_a_shared_ancestor_is_propagated_once(monkeypatch):
    """A shared ancestor is propagated once per traversal."""
    model = _Residual(blocks=5)
    x_L = torch.full((1, 4), -1., dtype=_DT)
    x_U = torch.full((1, 4), 1., dtype=_DT)
    bm = BoundedModule(model, (x_L + x_U) / 2, device='cpu')
    bx = BoundedTensor((x_L + x_U) / 2, PerturbationLpNorm(
        norm=float('inf'), x_L=x_L, x_U=x_U))
    # Supplying intermediate bounds keeps the propagation from caching them
    # per node, which would otherwise hide the repeated traversal.
    bm.compute_bounds(x=(bx,), method='backward')
    interm = {r.inputs[0].name: (r.inputs[0].lower.detach(),
                                 r.inputs[0].upper.detach()) for r in bm.relus}

    calls = {}
    real = BoundRelu.bound_dynamic_forward

    def counted(self, x, max_dim=None, offset=0):
        calls[self.name] = calls.get(self.name, 0) + 1
        return real(self, x, max_dim=max_dim, offset=offset)

    monkeypatch.setattr(BoundRelu, 'bound_dynamic_forward', counted)
    bm.compute_bounds(x=(bx,), method='dynamic-forward', interm_bounds=interm)
    assert calls and max(calls.values()) == 1, calls


class _UnperturbedBranch(nn.Module):
    """Merge a perturbed branch with a coefficient-free branch."""

    def __init__(self, n_in, sub=False):
        super().__init__()
        torch.manual_seed(3)
        self.sub = sub
        self.perturbed = nn.Linear(n_in, 3, dtype=_DT)
        self.fixed = nn.Linear(3, 3, dtype=_DT)
        self.o = nn.Linear(3, 2, dtype=_DT)

    def forward(self, x, c):
        a, b = self.perturbed(x), self.fixed(c)
        return self.o(b - a if self.sub else a + b)


@pytest.mark.parametrize('subtract', [False, True])
@pytest.mark.parametrize('n_in', [1, 3])
def test_a_coefficient_free_branch_is_widened_over_the_batch(n_in, subtract):
    """Broadcasting keeps the batch and coefficient axes distinct."""
    model = _UnperturbedBranch(n_in, sub=subtract)
    eps = 0.5
    x = torch.arange(4 * n_in, dtype=_DT).reshape(4, n_in) / 10
    c = torch.arange(12, dtype=_DT).reshape(4, 3)

    def run(x, c):
        bm = BoundedModule(model, (x, c), device='cpu')
        bx = BoundedTensor(x, PerturbationLpNorm(
            norm=float('inf'), x_L=x - eps, x_U=x + eps))
        return bm.compute_bounds(
            x=(bx, c), method='dynamic-forward',
            C=torch.eye(2, dtype=_DT).expand(x.shape[0], 2, 2))

    lower, upper = run(x, c)
    for i in range(x.shape[0]):
        want_l, want_u = run(x[i:i + 1], c[i:i + 1])
        torch.testing.assert_close(lower[i:i + 1], want_l, rtol=0, atol=1e-12)
        torch.testing.assert_close(upper[i:i + 1], want_u, rtol=0, atol=1e-12)


def test_a_float32_specification_is_applied_in_the_bound_dtype():
    """A specification is converted to the propagated coefficient dtype."""
    model = _Diamond(sub=False)
    _, (lower, upper) = _diamond_bounds(model, c_dtype=torch.float32)
    assert lower.dtype is _DT and upper.dtype is _DT
    _, (want_l, want_u) = _diamond_bounds(model)
    torch.testing.assert_close(lower, want_l, rtol=0, atol=1e-12)
    torch.testing.assert_close(upper, want_u, rtol=0, atol=1e-12)


class _ConvResidual(nn.Module):
    def __init__(self, blocks=2):
        super().__init__()
        torch.manual_seed(5)
        self.b = nn.ModuleList([nn.Conv2d(3, 3, 3, padding=1, dtype=_DT)
                                for _ in range(blocks)])
        self.s = nn.ModuleList([nn.Conv2d(3, 3, 1, dtype=_DT)
                                for _ in range(blocks)])
        self.o = nn.Linear(3 * 5 * 5, 3, dtype=_DT)

    def forward(self, x):
        for block, skip in zip(self.b, self.s):
            x = torch.relu(block(x)) + skip(x)
        return self.o(x.flatten(1))


def test_a_conv_dag_is_bounded_soundly_and_survives_windowing():
    """DAG merging and windowing support 4-D convolutional coefficients."""
    model, eps = _ConvResidual(), 0.15

    def bounds(max_dim):
        x0 = torch.zeros(1, 3, 5, 5, dtype=_DT)
        bm = BoundedModule(model, x0, device='cpu',
                           bound_opts={'forward_max_dim': max_dim})
        bx = BoundedTensor(x0, PerturbationLpNorm(
            norm=float('inf'), x_L=x0 - eps, x_U=x0 + eps))
        return bm.compute_bounds(
            x=(bx,), method='dynamic-forward',
            C=torch.eye(3, dtype=_DT).unsqueeze(0))

    lower, upper = bounds(int(1e9))
    x0 = torch.zeros(3, 5, 5, dtype=_DT)
    pattern = torch.linspace(-eps, eps, x0.numel(), dtype=_DT).reshape_as(x0)
    points = torch.stack([
        x0, torch.full_like(x0, -eps), torch.full_like(x0, eps),
        pattern, -pattern,
    ])
    with torch.no_grad():
        value = model(points)
    assert (lower <= value + 1e-9).all()
    assert (upper >= value - 1e-9).all()

    windowed_l, windowed_u = bounds(3)
    torch.testing.assert_close(windowed_l, lower, rtol=0, atol=1e-12)
    torch.testing.assert_close(windowed_u, upper, rtol=0, atol=1e-12)
