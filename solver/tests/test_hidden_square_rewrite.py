"""Exact hidden-square proofs and the opt-in graph rewrite (CPU)."""
import importlib
import os
from pathlib import Path
import re
import sys
from auto_LiRPA.op_budget import OpBudget
from fractions import Fraction
from unittest.mock import patch

import pytest
import torch
from torch import nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.bound_ops import BoundConstant, BoundExp, BoundMul, BoundSqr
from auto_LiRPA.optimize_graph import (
    _HiddenSquarePolynomial, _HiddenSquareUnsupported, rewrite_hidden_squares,
)
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.fp_enclosure import finite_abs, rounding_error
from enclosure_test_utils import assert_enclosed


OP_BUDGET = {'hidden_squares': 1 << 20, 'complete_squares': 1 << 24, 'prove_rows': 1 << 23,
             'commit': 'partial'}
OPTIONS = {
    'op_budget': dict(OP_BUDGET),
    'domain_aware_ops': True,
    'compare_crown_with_ibp_intermediate': True,
    'undefined_relaxation_rows': True,
    'hidden_square_rewrite': True,
}
BENCHMARK_DIR = Path(os.environ.get(
    'NLP_BENCHMARK_DIR',
    str(Path(__file__).resolve().parents[2] / 'benchmarks'),
)) / 'gams_world'


def _hidden_products(bm):
    return [n for n in bm.nodes()
            if type(n) is BoundMul and n.name.endswith('/hidden_scaled_square')]


class HiddenSquare(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.tensor(100.0))

    def forward(self, x):
        x0, x1 = x[:, :1], x[:, 1:2]
        u = x0 * x0 - x1
        v = x0 * (x0 * self.scale) - x1 * self.scale
        return (u * v).sum(dim=-1, keepdim=True)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('enabled', [False, True, None])
def test_hidden_square_graph_bounds_and_forward(dtype, enabled):
    model = HiddenSquare().to(dtype=dtype)
    x = torch.zeros(1, 2, dtype=dtype)
    # lower >= 0 / < 0 below asserts the square rewrite; the A67 rounding enclosure (default on) puts the
    # rewritten bound slightly below 0, so it is off here.
    opts = dict(OPTIONS, fp_enclosure={'affine_backward': False})
    if enabled is None:
        opts.pop('hidden_square_rewrite')  # The default must stay off.
    else:
        opts['hidden_square_rewrite'] = enabled
    bm = BoundedModule(model, x, bound_opts=opts, device='cpu')
    products = _hidden_products(bm)
    assert bool(products) == bool(enabled)
    for product in products:
        assert type(product.inputs[0]) is BoundSqr
        assert type(product.inputs[1]) is BoundConstant
        assert product.inputs[1].value.dtype == dtype
        assert product.inputs[1].value.device.type == 'cpu'
    box = BoundedTensor(x, PerturbationLpNorm(x_L=x - 2, x_U=x + 2))
    for method in ('IBP', 'CROWN'):
        lower, _ = bm.compute_bounds(x=(box,), method=method)
        assert bool((lower >= 0).all()) if enabled else bool((lower < 0).all())
    sample = torch.rand(64, 2, generator=torch.Generator().manual_seed(42), dtype=dtype) * 4 - 2
    torch.testing.assert_close(bm(sample), model(sample))


class ExpAtomSquare(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared
        self.register_buffer('y', torch.tensor([1.0, -3.0], dtype=torch.float64))

    def forward(self, x):
        x0, x1 = x[:, :2], x[:, 2:4]
        exp = torch.exp(x1)
        # The identity add keeps a separate exp node through graph CSE.
        other_exp = exp if self.shared else torch.exp(x1 + 0.0)
        u = self.y - x0 * exp
        v = 2 * self.y - (x0 * 2) * other_exp
        return u * v


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_exp_atoms_match_only_when_shared(shared, enabled):
    model = ExpAtomSquare(shared)
    opts = dict(OPTIONS, hidden_square_rewrite=enabled)
    bm = BoundedModule(model, torch.zeros(1, 4, dtype=torch.float64),
                       bound_opts=opts, device='cpu')
    assert sum(type(n) is BoundExp for n in bm.nodes()) == (1 if shared else 2)
    products = _hidden_products(bm)
    assert len(products) == int(shared and enabled)
    if products:
        torch.testing.assert_close(products[0].inputs[1].value,
                                   torch.full((1, 2), 2.0, dtype=torch.float64))
    sample = torch.rand(64, 4, generator=torch.Generator().manual_seed(42),
                        dtype=torch.float64) * 4 - 2
    torch.testing.assert_close(bm(sample), model(sample))


class RejectedSquare(nn.Module):
    def __init__(self, case):
        super().__init__()
        self.case = case
        self.register_buffer('scale', torch.tensor(100.0, dtype=torch.float64))
        self.register_buffer('offset', torch.tensor(1e-9, dtype=torch.float64))
        self.register_buffer('reverse', torch.tensor([1, 0]))

    def forward(self, x):
        u = x * x - x
        if self.case == 'offset':
            v = self.scale * u + self.offset
        elif self.case == 'reordered':
            v = torch.index_select(u, -1, self.reverse)
        elif self.case == 'exp':
            v = torch.exp(u)
        elif self.case == 'broadcast':
            v = self.scale * u[:, :1]
        elif self.case == 'zero':
            u = x - x
            v = self.scale * u
        elif self.case == 'zero_element':
            u = torch.cat((x[:, :1] - x[:, :1], x[:, 1:2]), dim=-1)
            v = self.scale * u
        elif self.case == 'inexact_scale':
            u = x * 3
            v = x
        elif self.case == 'inexact_dtype':
            u = x
            v = x + x * (2.0 ** -24)
        elif self.case == 'degree':
            u = x.pow(5)
            v = self.scale * u
        elif self.case == 'terms':
            u = x.sum(-1, keepdim=True).pow(2)
            v = self.scale * u
        else:
            raise AssertionError(self.case)
        return (u * v).sum(dim=-1, keepdim=True)


@pytest.mark.parametrize('case', [
    'offset', 'reordered', 'exp', 'broadcast', 'zero', 'zero_element',
    'inexact_scale', 'degree', 'terms',
])
def test_rejected_hidden_squares(case):
    model = RejectedSquare(case)
    n = 64 if case == 'terms' else 2
    bm = BoundedModule(model, torch.zeros(1, n, dtype=torch.float64),
                       bound_opts=dict(OPTIONS), device='cpu')
    assert not _hidden_products(bm)
    sample = torch.rand(64, n, dtype=torch.float64) * 4 - 2
    torch.testing.assert_close(bm(sample), model(sample))


def test_scale_must_be_exact_in_the_model_dtype():
    model = RejectedSquare('inexact_dtype').float()
    bm = BoundedModule(model, torch.zeros(1, 2), bound_opts=dict(OPTIONS), device='cpu')
    assert not _hidden_products(bm)  # 1 + 2**-24 is exact in float64, but not float32.


def test_scalar_hidden_square():
    class ScalarSquare(nn.Module):
        def forward(self, x):
            u = -x.sum(dim=(0, 1))
            v = -(x * 4).sum(dim=(0, 1))
            return u * v

    model = ScalarSquare()
    bm = BoundedModule(model, torch.zeros(1, 2),
                       bound_opts=dict(OPTIONS), device='cpu')
    assert len(_hidden_products(bm)) == 1
    sample = torch.rand(64, 2)
    torch.testing.assert_close(bm(sample), model(sample))


class PolynomialOps(nn.Module):
    def __init__(self, case):
        super().__init__()
        self.case = case
        self.register_buffer('scale', torch.tensor([4.0, -0.25], dtype=torch.float64))
        self.register_buffer('indices', torch.tensor([1, 0]))
        self.register_buffer('weight', torch.tensor([[0.5, 2.0], [-1.0, 0.25]], dtype=torch.float64))
        self.register_buffer('bias', torch.tensor([0.25, -2.0], dtype=torch.float64))

    def forward(self, x):
        if self.case == 'matmul':
            u = x @ self.weight + self.bias
            v = x @ (self.weight * self.scale) + self.bias * self.scale
        elif self.case == 'matmul_left':
            u = self.weight @ x.unsqueeze(-1)
            v = (self.weight * self.scale.unsqueeze(-1)) @ x.unsqueeze(-1)
            u, v = u.squeeze(-1), v.squeeze(-1)
        elif self.case == 'linear':
            u = torch.nn.functional.linear(x, self.weight, self.bias)
            v = torch.nn.functional.linear(
                x, self.weight * self.scale.unsqueeze(-1), self.bias * self.scale)
        elif self.case == 'reduce':
            u = x.sum(-1, keepdim=True)
            v = (x * 4).sum(-1, keepdim=True)
        elif self.case == 'pow':
            u = x.pow(3) + x.pow(0)
            v = x.pow(3) * self.scale + self.scale
        elif self.case == 'div_neg':
            u = -x / 2
            v = x * (-self.scale / 2)
        elif self.case == 'selections':
            selected = torch.index_select(x, -1, self.indices)
            u = torch.cat((selected[:, :1], selected[:, 1:2]), -1).unsqueeze(-1).squeeze(-1)
            v = (selected * self.scale).reshape(-1, 2)
        elif self.case == 'expand':
            u = x[:, :1].expand(-1, 2)
            v = x[:, :1].expand(-1, 2) * self.scale
        else:
            raise AssertionError(self.case)
        return u * v


@pytest.mark.parametrize('case', [
    'matmul', 'matmul_left', 'linear', 'reduce', 'pow', 'div_neg', 'selections', 'expand',
])
def test_supported_polynomial_operations(case):
    model = PolynomialOps(case)
    bm = BoundedModule(model, torch.zeros(1, 2, dtype=torch.float64),
                       bound_opts=dict(OPTIONS), device='cpu')
    products = _hidden_products(bm)
    assert len(products) == 1
    sample = torch.rand(64, 2, dtype=torch.float64) * 4 - 2
    torch.testing.assert_close(bm(sample), model(sample))
    if case != 'reduce':
        torch.testing.assert_close(products[0].inputs[1].value,
                                   model.scale.reshape(1, 2))


def test_polynomial_limits_and_exact_cancellation():
    deadline = OpBudget("t", 1 << 40)
    # 45 linear terms square to 1035 distinct monomials, so it is allowed
    # despite the 2025 pairwise products. 64 terms produce 2080 and are refused.
    def linear(n):
        return _HiddenSquarePolynomial({(('x', i),): Fraction(1) for i in range(n)}, deadline)
    assert len((linear(45) * linear(45)).terms) == 1035
    with pytest.raises(_HiddenSquareUnsupported):
        linear(64) * linear(64)
    with pytest.raises(_HiddenSquareUnsupported):
        linear(1) ** 5
    assert not (linear(2) - linear(2)).terms
    one = _HiddenSquarePolynomial({(): Fraction(1)}, deadline)
    assert (one ** 100).terms == {(): Fraction(1)}


def test_pass_stops_at_op_budget():
    bm = BoundedModule(HiddenSquare().double(), torch.zeros(1, 2, dtype=torch.float64),
                       device='cpu')
    bm.bound_opts['hidden_square_rewrite'] = True
    bm.bound_opts['op_budget'] = dict(OP_BUDGET, hidden_squares=1)
    rewrite_hidden_squares(bm)
    assert not _hidden_products(bm)


def _benchmark_case(name):
    sys.path.insert(0, str(BENCHMARK_DIR))
    module = importlib.import_module(f'mod_files_gams.{name}_model')
    cls = next(getattr(module, c) for c in dir(module) if c.endswith('Model') and c != 'Model')
    model = cls().double().eval()
    text = (BENCHMARK_DIR / f'vnnlib_gams/{name}.vnnlib').read_text()
    n = len(re.findall(r'declare-const X_\d+', text))
    lo, hi = torch.zeros(1, n, dtype=torch.float64), torch.zeros(1, n, dtype=torch.float64)
    for sign, target in (('>=', lo), ('<=', hi)):
        bounds = re.findall(r'\(' + sign + r' X_(\d+) ([-\d.eE+]+)\)', text)
        assert len({i for i, _ in bounds}) == n
        for i, v in bounds:
            target[0, int(i)] = float(v)
    assert torch.isfinite(lo).all() and torch.isfinite(hi).all() and (lo <= hi).all()
    return model, lo, hi


@pytest.mark.skipif(not (BENCHMARK_DIR / 'mod_files_gams').is_dir(),
                    reason='Slow real-model tests require the NLP benchmark directory')
@pytest.mark.parametrize('name', ['Rosenbrock', 'cute_obstclal'])
def test_real_benchmark_hidden_square_bounds(name):
    model, lo, hi = _benchmark_case(name)
    x = (lo + hi) / 2
    rewrites = []

    def counted_rewrite(bm):
        rewrites.append(rewrite_hidden_squares(bm))

    with patch('auto_LiRPA.optimize_graph.rewrite_hidden_squares', counted_rewrite):
        bm = BoundedModule(model, x, bound_opts=dict(OPTIONS), device='cpu')
    products = _hidden_products(bm)
    assert rewrites == [2]
    # CSE and orphan cleanup can remove an inner rewrite after the outer one.
    assert products
    box = BoundedTensor(x, PerturbationLpNorm(x_L=lo, x_U=hi))
    C = torch.zeros(1, 1, model(x).shape[-1], dtype=torch.float64)
    C[0, 0, 0] = 1
    bounds = {}
    margins = {}
    for method in ('IBP', 'CROWN'):
        lower, _ = bm.compute_bounds(x=(box,), C=C, method=method)
        bounds[method] = lower.item()
        if name == 'Rosenbrock':
            # Its final add combines large nonnegative square sums. Measure
            # the enclosure on those actual operand boxes before cancellation.
            final = bm.final_node()
            intervals = [bm.IBP_general(node=inp) for inp in final.inputs]
            magnitude = sum(finite_abs(torch.maximum(v.lower.abs(), v.upper.abs()))
                            for v in intervals)
            margins[method] = 10 * rounding_error(magnitude, len(intervals))
    print(f'{name}: rewrites={rewrites[0]}, root output 0 lower bounds={bounds}')
    if name == 'Rosenbrock':
        for method, lb in bounds.items():
            assert_enclosed(lo.new_tensor([[lb]]), 0, margins[method], -1)
    else:
        assert bounds['IBP'] == pytest.approx(-255901866.59428)
        assert bounds['CROWN'] == pytest.approx(-888.5555668524, abs=1.0)
    sample = lo + (hi - lo) * torch.rand(64, lo.numel(), dtype=torch.float64)
    torch.testing.assert_close(bm(sample), model(sample))
