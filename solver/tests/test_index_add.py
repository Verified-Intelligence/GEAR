# pylint: disable=wrong-import-position
"""Tests for BoundATenIndex_add (torch.Tensor.index_add).

ONNX export preserves index_add as aten::ATen only when the index has
duplicate values AND src.shape[dim] != data.shape[dim] (specifically when
src is larger). Other cases get decomposed into ScatterElements+Add.
Tests are written to exercise the aten::ATen path.
"""
import pytest
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class IndexAddModel(nn.Module):
    """y = data.index_add(dim, index, src). Both inputs from caller."""
    def __init__(self, index, dim=1):
        super().__init__()
        self.register_buffer('index', index)
        self.dim = dim

    def forward(self, data, src):
        return data.index_add(self.dim, self.index, src)


class IndexAddDataOnly(nn.Module):
    """Only data is perturbed; src is a fixed buffer."""
    def __init__(self, index, src, dim=1):
        super().__init__()
        self.register_buffer('index', index)
        self.register_buffer('src', src)
        self.dim = dim

    def forward(self, data):
        return data.index_add(self.dim, self.index, self.src)


class IndexAddSrcOnly(nn.Module):
    """Only src is perturbed; data is a fixed buffer."""
    def __init__(self, index, data, dim=1):
        super().__init__()
        self.register_buffer('index', index)
        self.register_buffer('data_buf', data)
        self.dim = dim

    def forward(self, src):
        return self.data_buf.index_add(self.dim, self.index, src)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verify_bounds_sampling(model, inputs, eps, lb, ub,
                            n_samples=50_000, atol=1e-5):
    """Check soundness: bounds must contain all sampled outputs."""
    outputs = []
    for _ in range(n_samples):
        perturbed = tuple(
            x + (2 * torch.rand_like(x) - 1) * eps for x in inputs
        )
        outputs.append(model(*perturbed))
    outputs = torch.stack(outputs)
    # Flatten to match compute_bounds output (which may flatten spatial dims)
    B = outputs.shape[1]
    outputs = outputs.reshape(outputs.shape[0], B, -1)
    lb = lb.reshape(B, -1)
    ub = ub.reshape(B, -1)
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


def _compute_bounds_two_input(data_center, src_center, index, eps, method,
                               dim=1):
    """Both data and src perturbed."""
    model = IndexAddModel(index, dim=dim)
    bounded_model = BoundedModule(model,
                                  (data_center.clone(), src_center.clone()))
    ptb_data = PerturbationLpNorm(norm=float('inf'), eps=eps)
    ptb_src = PerturbationLpNorm(norm=float('inf'), eps=eps)
    bd = BoundedTensor(data_center, ptb_data)
    bs = BoundedTensor(src_center, ptb_src)
    lb, ub = bounded_model.compute_bounds(x=(bd, bs), method=method)
    return model, lb, ub


def _compute_bounds_data_only(data_center, src_fixed, index, eps, method,
                               dim=1):
    """Only data perturbed."""
    model = IndexAddDataOnly(index, src_fixed, dim=dim)
    bounded_model = BoundedModule(model, (data_center.clone(),))
    ptb = PerturbationLpNorm(norm=float('inf'), eps=eps)
    bd = BoundedTensor(data_center, ptb)
    lb, ub = bounded_model.compute_bounds(x=(bd,), method=method)
    return model, lb, ub


def _compute_bounds_src_only(src_center, data_fixed, index, eps, method,
                              dim=1):
    """Only src perturbed."""
    model = IndexAddSrcOnly(index, data_fixed, dim=dim)
    bounded_model = BoundedModule(model, (src_center.clone(),))
    ptb = PerturbationLpNorm(norm=float('inf'), eps=eps)
    bs = BoundedTensor(src_center, ptb)
    lb, ub = bounded_model.compute_bounds(x=(bs,), method=method)
    return model, lb, ub


# ---------------------------------------------------------------------------
# Test cases
# ONNX keeps aten::ATen when: src.shape[dim] > data.shape[dim] AND index
# has duplicate values.
# ---------------------------------------------------------------------------

TWO_INPUT_CASES = [
    # (name, data, src, index, eps)
    ("basic",
     torch.tensor([[1.0, 2.0, 3.0]]),
     torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
     torch.tensor([0, 1, 0, 2]),
     0.1),
    ("duplicate_indices",
     torch.tensor([[1.0, 2.0]]),
     torch.tensor([[0.5, 0.5, 0.5]]),
     torch.tensor([0, 0, 1]),
     0.2),
    ("larger",
     torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
     torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
     torch.tensor([0, 2, 4, 1, 3, 0]),
     0.05),
    ("three_to_two",
     torch.tensor([[1.0, 2.0]]),
     torch.tensor([[0.3, 0.4, 0.5]]),
     torch.tensor([1, 0, 1]),
     0.15),
    ("large_eps",
     torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
     torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]]),
     torch.tensor([0, 2, 0, 3, 1]),
     1.0),
]

METHODS = ['IBP', 'CROWN', 'Forward', 'CROWN-Optimized']


# ---------------------------------------------------------------------------
# Soundness tests (both inputs perturbed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,data,src,index,eps",
    TWO_INPUT_CASES,
    ids=[c[0] for c in TWO_INPUT_CASES],
)
@pytest.mark.parametrize("method", METHODS)
def test_two_input_soundness(name, data, src, index, eps, method):
    """Bounds must contain all sampled outputs (both inputs perturbed)."""
    model, lb, ub = _compute_bounds_two_input(data, src, index, eps, method)
    _verify_bounds_sampling(model, (data, src), eps, lb, ub)


@pytest.mark.parametrize(
    "name,data,src,index,eps",
    TWO_INPUT_CASES,
    ids=[c[0] for c in TWO_INPUT_CASES],
)
@pytest.mark.parametrize("method", METHODS)
def test_two_input_tight(name, data, src, index, eps, method):
    """For a linear op, bounds should match analytical extremes."""
    model, lb, ub = _compute_bounds_two_input(data, src, index, eps, method)
    expected_lb = model(data - eps, src - eps)
    expected_ub = model(data + eps, src + eps)
    assert torch.allclose(lb, expected_lb, atol=1e-4), (
        f"Lower bound not tight: lb={lb}, expected={expected_lb}")
    assert torch.allclose(ub, expected_ub, atol=1e-4), (
        f"Upper bound not tight: ub={ub}, expected={expected_ub}")


# ---------------------------------------------------------------------------
# Only data perturbed
# ---------------------------------------------------------------------------

DATA_ONLY_CASES = [
    # For data-only, src is a buffer. The model is traced with batch=1, and
    # bounds are computed with batch=1. Data shape must differ from src on
    # scatter dim for aten::ATen routing.
    ("basic",
     torch.tensor([[1.0, 2.0, 3.0]]),
     torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
     torch.tensor([0, 1, 0, 2]),
     0.1),
    ("dup",
     torch.tensor([[1.0, 2.0]]),
     torch.tensor([[0.5, 0.5, 0.5]]),
     torch.tensor([0, 0, 1]),
     0.2),
]

@pytest.mark.parametrize(
    "name,data,src,index,eps",
    DATA_ONLY_CASES,
    ids=[c[0] for c in DATA_ONLY_CASES],
)
@pytest.mark.parametrize("method", METHODS)
def test_data_only_soundness(name, data, src, index, eps, method):
    """Soundness when only data is perturbed."""
    model, lb, ub = _compute_bounds_data_only(data, src, index, eps, method)
    _verify_bounds_sampling(model, (data,), eps, lb, ub)


@pytest.mark.parametrize(
    "name,data,src,index,eps",
    DATA_ONLY_CASES,
    ids=[c[0] for c in DATA_ONLY_CASES],
)
@pytest.mark.parametrize("method", METHODS)
def test_data_only_tight(name, data, src, index, eps, method):
    """Tightness when only data is perturbed."""
    model, lb, ub = _compute_bounds_data_only(data, src, index, eps, method)
    expected_lb = model(data - eps)
    expected_ub = model(data + eps)
    assert torch.allclose(lb, expected_lb, atol=1e-4), (
        f"Lower bound not tight: lb={lb}, expected={expected_lb}")
    assert torch.allclose(ub, expected_ub, atol=1e-4), (
        f"Upper bound not tight: ub={ub}, expected={expected_ub}")


# ---------------------------------------------------------------------------
# Only src perturbed
# ---------------------------------------------------------------------------

SRC_ONLY_CASES = [
    # For src-only, data is a buffer. src.shape[dim] > data.shape[dim] for
    # aten::ATen routing.
    ("basic",
     torch.tensor([[1.0, 2.0, 3.0]]),   # data (fixed)
     torch.tensor([[0.1, 0.2, 0.3, 0.4]]),  # src (perturbed)
     torch.tensor([0, 1, 0, 2]),
     0.1),
    ("dup",
     torch.tensor([[1.0, 2.0]]),
     torch.tensor([[0.5, 0.5, 0.5]]),
     torch.tensor([0, 0, 1]),
     0.2),
]

@pytest.mark.parametrize(
    "name,data,src,index,eps",
    SRC_ONLY_CASES,
    ids=[c[0] for c in SRC_ONLY_CASES],
)
@pytest.mark.parametrize("method", METHODS)
def test_src_only_soundness(name, data, src, index, eps, method):
    """Soundness when only src is perturbed."""
    model, lb, ub = _compute_bounds_src_only(src, data, index, eps, method)
    _verify_bounds_sampling(model, (src,), eps, lb, ub)


@pytest.mark.parametrize(
    "name,data,src,index,eps",
    SRC_ONLY_CASES,
    ids=[c[0] for c in SRC_ONLY_CASES],
)
@pytest.mark.parametrize("method", METHODS)
def test_src_only_tight(name, data, src, index, eps, method):
    """Tightness when only src is perturbed."""
    model, lb, ub = _compute_bounds_src_only(src, data, index, eps, method)
    expected_lb = model(src - eps)
    expected_ub = model(src + eps)
    assert torch.allclose(lb, expected_lb, atol=1e-4), (
        f"Lower bound not tight: lb={lb}, expected={expected_lb}")
    assert torch.allclose(ub, expected_ub, atol=1e-4), (
        f"Upper bound not tight: ub={ub}, expected={expected_ub}")


# ---------------------------------------------------------------------------
# Batch dimension tests — core motivation for index_add over scatter_add.
# We trace with batch=1, then compute bounds with batch>1.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_batch_two_input(method):
    """Both data and src with batch B>1."""
    index = torch.tensor([0, 1, 0, 2])
    data = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]])  # B=2
    src = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                         [0.5, 0.6, 0.7, 0.8]])  # B=2
    eps = 0.1
    model, lb, ub = _compute_bounds_two_input(data, src, index, eps, method)
    _verify_bounds_sampling(model, (data, src), eps, lb, ub)
    expected_lb = model(data - eps, src - eps)
    expected_ub = model(data + eps, src + eps)
    assert torch.allclose(lb, expected_lb, atol=1e-4)
    assert torch.allclose(ub, expected_ub, atol=1e-4)


# ---------------------------------------------------------------------------
# Higher dimensional test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_higher_dim(method):
    """2D spatial: data (1,3,3), src (1,4,3), scatter along channel dim."""
    torch.manual_seed(42)
    data = torch.randn(1, 3, 3)
    src = torch.randn(1, 4, 3)
    index = torch.tensor([0, 1, 0, 2])
    eps = 0.05

    class IndexAdd2D(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.register_buffer('index', idx)
        def forward(self, data, src):
            return data.index_add(1, self.index, src)

    model = IndexAdd2D(index)
    bounded_model = BoundedModule(model, (data.clone(), src.clone()))
    ptb_d = PerturbationLpNorm(norm=float('inf'), eps=eps)
    ptb_s = PerturbationLpNorm(norm=float('inf'), eps=eps)
    bd = BoundedTensor(data, ptb_d)
    bs = BoundedTensor(src, ptb_s)
    lb, ub = bounded_model.compute_bounds(x=(bd, bs), method=method)

    _verify_bounds_sampling(model, (data, src), eps, lb, ub)


# ---------------------------------------------------------------------------
# Consistency with scatter_add
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ['IBP', 'CROWN', 'Forward'])
def test_consistency_with_scatter_add(method):
    """index_add and scatter_add should give the same bounds."""
    data = torch.tensor([[1.0, 2.0, 3.0]])
    src = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    index_1d = torch.tensor([0, 1, 0, 2])
    eps = 0.1

    _, lb_ia, ub_ia = _compute_bounds_two_input(
        data, src, index_1d, eps, method)

    class ScatterAddModel(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.register_buffer('index', idx)
        def forward(self, data, src):
            return data.scatter_add(1, self.index, src)

    index_2d = index_1d.unsqueeze(0)
    sa_model = ScatterAddModel(index_2d)
    bm = BoundedModule(sa_model, (data.clone(), src.clone()))
    ptb_d = PerturbationLpNorm(norm=float('inf'), eps=eps)
    ptb_s = PerturbationLpNorm(norm=float('inf'), eps=eps)
    bd = BoundedTensor(data, ptb_d)
    bs = BoundedTensor(src, ptb_s)
    lb_sa, ub_sa = bm.compute_bounds(x=(bd, bs), method=method)

    assert torch.allclose(lb_ia, lb_sa, atol=1e-5), (
        f"index_add lb={lb_ia} != scatter_add lb={lb_sa}")
    assert torch.allclose(ub_ia, ub_sa, atol=1e-5), (
        f"index_add ub={ub_ia} != scatter_add ub={ub_sa}")


# ---------------------------------------------------------------------------
# Op routing check
# ---------------------------------------------------------------------------

def test_op_routing():
    """Verify that index_add routes to BoundATenIndex_add."""
    from auto_LiRPA.operators.indexing import BoundATenIndex_add
    index = torch.tensor([0, 1, 0, 2])
    data = torch.randn(1, 3)
    src = torch.randn(1, 4)
    model = IndexAddModel(index)
    bm = BoundedModule(model, (data.clone(), src.clone()))
    found = any(
        isinstance(m, BoundATenIndex_add)
        for m in bm._modules.values()
    )
    assert found, (
        "BoundATenIndex_add not found in BoundedModule — "
        "ONNX export may have decomposed the op.")


# ---------------------------------------------------------------------------
# Reference-based regression test using TestCase
# ---------------------------------------------------------------------------

from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE


class TestIndexAddRef(TestCase):
    """Reference-based regression test for BoundATenIndex_add.

    Saves/compares bound values to detect regressions where bounds
    become looser even if they remain sound.
    """
    def __init__(self, methodName='runTest', generate=False,
                 device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(
            methodName, seed=1,
            ref_name='index_add_test_data',
            generate=generate,
            device=device, dtype=dtype)

    def test(self):
        self.result = []

        # Representative subset of test cases x methods.
        # All cases must route to aten::ATen (src.shape[dim] > data.shape[dim]
        # with duplicate indices).
        ref_cases = [
            # (data, src, index, eps)
            (torch.tensor([[1.0, 2.0, 3.0]]),
             torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
             torch.tensor([0, 1, 0, 2]),
             0.1),
            (torch.tensor([[1.0, 2.0]]),
             torch.tensor([[0.5, 0.5, 0.5]]),
             torch.tensor([0, 0, 1]),
             0.2),
            (torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
             torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
             torch.tensor([0, 2, 4, 1, 3, 0]),
             0.05),
        ]

        ref_methods = ['IBP', 'CROWN', 'Forward']

        for data, src, index, eps in ref_cases:
            data = data.to(device=self.default_device, dtype=self.default_dtype)
            src = src.to(device=self.default_device, dtype=self.default_dtype)
            for method in ref_methods:
                _, lb, ub = _compute_bounds_two_input(
                    data, src, index, eps, method)
                self.result.append((lb, ub))

        self.check()


if __name__ == '__main__':
    # Change to generate=True when generating reference results
    testcase = TestIndexAddRef(generate=False)
    testcase.setUp()
    testcase.test()
