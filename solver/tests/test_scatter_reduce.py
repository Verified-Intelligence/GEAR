# pylint: disable=wrong-import-position
"""Test BoundScatterElements (scatter_reduce with reduction='sum'/'add')."""
import pytest
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE


class ScatterAddModel(nn.Module):
    """y = data.scatter_add(dim=1, index, src).

    Both data and src are model inputs; index is a fixed buffer.
    Requires batch_size=1 to avoid dynamic-shape ONNX ops.
    """
    def __init__(self, index):
        super().__init__()
        self.register_buffer('index', index)

    def forward(self, data, src):
        return data.scatter_add(1, self.index, src)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verify_bounds_sampling(model, data_center, src_center, eps, lb, ub,
                            n_samples=100_000, atol=1e-5):
    """Check soundness by random sampling within the perturbation region."""
    outputs = []
    for _ in range(n_samples):
        d = data_center + (2 * torch.rand_like(data_center) - 1) * eps
        s = src_center + (2 * torch.rand_like(src_center) - 1) * eps
        outputs.append(model(d, s))
    outputs = torch.stack(outputs)
    sampled_min = outputs.min(dim=0).values
    sampled_max = outputs.max(dim=0).values
    assert (lb <= sampled_min + atol).all(), (
        f"Lower bound violated! lb={lb}, sampled_min={sampled_min}, "
        f"max violation={(lb - sampled_min).max().item()}"
    )
    assert (ub >= sampled_max - atol).all(), (
        f"Upper bound violated! ub={ub}, sampled_max={sampled_max}, "
        f"max violation={(sampled_max - ub).max().item()}"
    )


def _compute_bounds(data_center, src_center, index, eps, method):
    """Create a BoundedModule, compute bounds, and return (lb, ub)."""
    model = ScatterAddModel(index)
    bounded_model = BoundedModule(model,
                                  (data_center.clone(), src_center.clone()))
    ptb_data = PerturbationLpNorm(norm=float('inf'), eps=eps)
    ptb_src = PerturbationLpNorm(norm=float('inf'), eps=eps)
    bd = BoundedTensor(data_center, ptb_data)
    bs = BoundedTensor(src_center, ptb_src)
    lb, ub = bounded_model.compute_bounds(x=(bd, bs), method=method)
    return model, lb, ub


# ---------------------------------------------------------------------------
# Parametrized test cases for sampling-based soundness
# ---------------------------------------------------------------------------

SCATTER_TEST_CASES = [
    # (name, data, src, index, eps)
    ("basic",
     torch.tensor([[1.0, 2.0, 3.0]]),
     torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
     torch.tensor([[0, 1, 0, 2]]),
     0.1),
    ("duplicate_indices",
     torch.tensor([[1.0, 2.0]]),
     torch.tensor([[0.5, 0.5, 0.5]]),
     torch.tensor([[0, 0, 1]]),
     0.2),
    ("single_src",
     torch.tensor([[1.0, 2.0, 3.0]]),
     torch.tensor([[0.5]]),
     torch.tensor([[1]]),
     0.3),
    ("larger",
     torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
     torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
     torch.tensor([[0, 2, 4, 1, 3, 0]]),
     0.05),
    ("all_to_one",
     torch.tensor([[1.0, 2.0, 3.0]]),
     torch.tensor([[0.1, 0.2, 0.3]]),
     torch.tensor([[0, 0, 0]]),
     0.15),
]

METHODS = ['IBP', 'CROWN', 'Forward', 'CROWN-Optimized']


@pytest.mark.parametrize(
    "name,data,src,index,eps",
    SCATTER_TEST_CASES,
    ids=[c[0] for c in SCATTER_TEST_CASES],
)
@pytest.mark.parametrize("method", METHODS)
def test_scatter_reduce_soundness(name, data, src, index, eps, method):
    """Verify that computed bounds are sound (contain all true outputs)."""
    model, lb, ub = _compute_bounds(data, src, index, eps, method)
    _verify_bounds_sampling(model, data, src, eps, lb, ub)


@pytest.mark.parametrize(
    "name,data,src,index,eps",
    SCATTER_TEST_CASES,
    ids=[c[0] for c in SCATTER_TEST_CASES],
)
@pytest.mark.parametrize("method", METHODS)
def test_scatter_reduce_tight(name, data, src, index, eps, method):
    """For a purely linear op, bounds should match analytical extremes."""
    model, lb, ub = _compute_bounds(data, src, index, eps, method)
    expected_lb = model(data - eps, src - eps)
    expected_ub = model(data + eps, src + eps)
    assert torch.allclose(lb, expected_lb, atol=1e-4), (
        f"Lower bound not tight: lb={lb}, expected={expected_lb}, "
        f"diff={(lb - expected_lb).abs().max().item()}"
    )
    assert torch.allclose(ub, expected_ub, atol=1e-4), (
        f"Upper bound not tight: ub={ub}, expected={expected_ub}, "
        f"diff={(ub - expected_ub).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Reference-based regression test using TestCase
# ---------------------------------------------------------------------------

class TestScatterReduceRef(TestCase):
    """Reference-based regression test for BoundScatterElements.

    Saves/compares bound values to detect regressions where bounds
    become looser even if they remain sound.
    """
    def __init__(self, methodName='runTest', generate=False,
                 device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(
            methodName, seed=1,
            ref_name='scatter_reduce_test_data',
            generate=generate,
            device=device, dtype=dtype)

    def test(self):
        self.result = []

        # Representative subset of test cases x methods
        ref_cases = [
            # (data, src, index, eps)
            (torch.tensor([[1.0, 2.0, 3.0]]),
             torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
             torch.tensor([[0, 1, 0, 2]]),
             0.1),
            (torch.tensor([[1.0, 2.0]]),
             torch.tensor([[0.5, 0.5, 0.5]]),
             torch.tensor([[0, 0, 1]]),
             0.2),
            (torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
             torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
             torch.tensor([[0, 2, 4, 1, 3, 0]]),
             0.05),
        ]

        ref_methods = ['IBP', 'CROWN', 'Forward']

        for data, src, index, eps in ref_cases:
            data = data.to(device=self.default_device, dtype=self.default_dtype)
            src = src.to(device=self.default_device, dtype=self.default_dtype)
            index = index.to(device=self.default_device)
            for method in ref_methods:
                _, lb, ub = _compute_bounds(data, src, index, eps, method)
                self.result.append((lb, ub))

        self.check()


if __name__ == '__main__':
    # Change to generate=True when generating reference results
    testcase = TestScatterReduceRef(generate=False)
    testcase.setUp()
    testcase.test()
