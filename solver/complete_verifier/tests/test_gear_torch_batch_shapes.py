"""Direct torch programs must preserve pointwise semantics for split domains."""
import pytest
import torch
from torch import nn

from auto_LiRPA import BoundedTensor, PerturbationLpNorm
from tests.gear_wp12a_helpers import evaluators


class ScalarReshape(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.ones(2, 1, dtype=torch.float64))

    def forward(self, x):
        total = (x @ self.weight).reshape(1)
        return torch.cat((total.unsqueeze(1), x[:, :1]), 1)


class ConstantConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('constant', torch.tensor([[.25, -.5]], dtype=torch.float64))

    def forward(self, x):
        return torch.cat((self.constant, x * x), 1)


@pytest.mark.parametrize('model_type', [ScalarReshape, ConstantConcat])
@pytest.mark.parametrize('batch', [1, 2, 4])
def test_batched_forward_and_bounds_enclose_pointwise_source(model_type, batch):
    x = torch.linspace(-.4, .6, batch * 2, dtype=torch.float64).reshape(batch, 2)
    graph, reference, _ = evaluators(model_type(), x[:1])
    expected = reference(x)
    torch.testing.assert_close(graph(x), expected, rtol=0, atol=0)
    lower, upper = x - .05, x + .05
    bounded = BoundedTensor(x, PerturbationLpNorm(x_L=lower, x_U=upper))
    for method in ('IBP', 'CROWN'):
        lb, ub = graph.compute_bounds(x=(bounded,), method=method)
        assert lb.shape == ub.shape == expected.shape
        serial_lower, serial_upper = [], []
        for row in range(batch):
            one = BoundedTensor(x[row:row+1], PerturbationLpNorm(
                x_L=lower[row:row+1], x_U=upper[row:row+1]))
            one_lb, one_ub = graph.compute_bounds(x=(one,), method=method)
            serial_lower.append(one_lb); serial_upper.append(one_ub)
        torch.testing.assert_close(lb, torch.cat(serial_lower), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(ub, torch.cat(serial_upper), atol=1e-12, rtol=1e-12)
        # Probe each coordinate extreme, as well as the box centre.
        for sample in (lower, upper, x, torch.stack((lower[:, 0], upper[:, 1]), 1)):
            values = reference(sample)
            assert (lb <= values + 1e-12).all() and (ub >= values - 1e-12).all()
