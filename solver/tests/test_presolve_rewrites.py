"""CPU presolve proofs, refusal cases, and sampled CROWN/IBP enclosures."""
from fractions import Fraction
from auto_LiRPA.op_budget import OpBudget

import pytest
import torch
from torch import nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.optimize_graph import _HiddenSquareProof, _HiddenSquareUnsupported
from auto_LiRPA.presolve import (
    _PresolveProof, split_exact, directed_float, nearest_float, outward_sqrt,
    objective_alias_floor, prove_rows, square_row_cuts,
)
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.fp_enclosure import rounding_error

OPTIONS = dict(op_budget=dict(hidden_squares=1 << 20, complete_squares=1 << 24, prove_rows=1 << 23,
                              commit='partial'),
               domain_aware_ops=True, compare_crown_with_ibp_intermediate=True,
               undefined_relaxation_rows=True, hidden_square_rewrite=True)


class Quadratic(nn.Module):
    def __init__(self, inner=False, product=False, c=8.6, k=-70.0):
        super().__init__()
        self.inner, self.product, self.c, self.k = inner, product, c, k

    def forward(self, x):
        z = torch.exp(x[:, :1]) if self.inner else x[:, :1]
        sq = z*z if self.product else z.pow(2)
        return self.c*sq + self.k*z + x[:, 1:2]


def bounded(model, opts):
    return BoundedModule(model.double(), torch.zeros(1, 2, dtype=torch.float64),
                         bound_opts=dict(OPTIONS, **opts), device='cpu')


def sample_bounds(bm, model, lo=-2, hi=2):
    x = torch.zeros(1, 2, dtype=torch.float64)
    box = BoundedTensor(x, PerturbationLpNorm(x_L=x+lo, x_U=x+hi))
    axis = torch.linspace(lo, hi, 121, dtype=x.dtype)
    samples = torch.cartesian_prod(axis, axis)
    y = model(samples)
    torch.testing.assert_close(bm(samples), y, rtol=2e-13, atol=2e-13)
    random = lo + (hi-lo)*torch.rand(256, 2, dtype=x.dtype,
                                    generator=torch.Generator().manual_seed(27))
    torch.testing.assert_close(bm(random), model(random), rtol=2e-13, atol=2e-13)
    for method in ('IBP', 'CROWN'):
        l, u = bm.compute_bounds(x=(box,), method=method)
        assert (l <= y.min(0).values + 1e-10).all()
        assert (u >= y.max(0).values - 1e-10).all()
    return box


@pytest.mark.parametrize('inner', [False, True])
@pytest.mark.parametrize('product', [False, True])
def test_complete_square(inner, product):
    model = Quadratic(inner, product)
    bm = bounded(model, dict(complete_square_rewrite=True))
    assert bm.presolve_rewrite_counts['complete_square_rewrite'] == 1
    sample_bounds(bm, model)
    original = bounded(model, {})
    a = _HiddenSquareProof(original, OpBudget("t", 1 << 40)).build(original.final_node())
    b = _HiddenSquareProof(bm, OpBudget("t", 1 << 40)).build(bm.final_node())
    assert [p.terms for p in a.flat] == [p.terms for p in b.flat]


class NearProduct(nn.Module):
    def __init__(self, unrelated=False):
        super().__init__()
        self.unrelated = unrelated

    def forward(self, x):
        z = x[:, :1] + 0.1*x[:, 1:2]
        u = 8.6*x[:, :1] + (0.86 if not self.unrelated else 0.9)*x[:, 1:2]
        return u*z - 3*z


def test_inexact_product_retains_exact_residual():
    model = NearProduct()
    bm = bounded(model, dict(complete_square_rewrite=True))
    assert bm.presolve_rewrite_counts['product_residual'] == 1
    assert any('presolve_product_residual' in n.name for n in bm.nodes())
    original = bounded(model, {})
    a = _HiddenSquareProof(original, OpBudget("t", 1 << 40)).build(original.final_node())
    b = _HiddenSquareProof(bm, OpBudget("t", 1 << 40)).build(bm.final_node())
    assert [p.terms for p in a.flat] == [p.terms for p in b.flat]
    sample_bounds(bm, model)


@pytest.mark.parametrize('model', [Quadratic(c=0), Quadratic(k=0), NearProduct(True)])
def test_complete_square_refusals(model):
    bm = bounded(model, dict(complete_square_rewrite=True))
    assert bm.presolve_rewrite_counts['complete_square_rewrite'] == 0
    assert bm.presolve_rewrite_counts['product_residual'] == 0
    sample_bounds(bm, model)


def test_default_options_are_off():
    bm = bounded(Quadratic(), {})
    assert all(bm.bound_opts[k] is False for k in (
        'complete_square_rewrite', 'square_row_to_abs', 'objective_row_alias'))
    assert not any('completed_square' in n.name for n in bm.nodes())


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_rational_split_and_directed_rounding(dtype):
    like = torch.zeros((), dtype=dtype)
    c, s = Fraction(8.6), Fraction(-70.0/(2*8.6))
    for q in (-c*s*s, Fraction(-70)-2*c*s):
        assert sum(map(Fraction, split_exact(q, like)), Fraction(0)) == q
        assert Fraction(float(directed_float(q, like))) <= q
        assert Fraction(float(directed_float(q, like, upward=True))) >= q
    with pytest.raises(_HiddenSquareUnsupported):
        split_exact(Fraction(1, 3), like)
    with pytest.raises(_HiddenSquareUnsupported):
        split_exact(Fraction(1, 2**2000), like)


class Rows(nn.Module):
    def __init__(self, nonlinear=False, alias=-2.0, offset=0.3, changed=False):
        super().__init__()
        self.nonlinear, self.alias, self.offset, self.changed = nonlinear, alias, offset, changed

    def forward(self, x):
        h = x[:, :1] - 0.2*x[:, 1:2]
        if self.nonlinear:
            h = torch.exp(h)-1
        obj = torch.sin(x[:, :1]) + x[:, 1:2]
        row = self.alias*obj + self.offset
        if self.changed:
            row = row + 1e-12*x[:, :1]
        sq = h.pow(2) - 1e-5
        # Opposing equality rows must not turn a negative square into a strip.
        return torch.cat((obj, row, sq, -sq), -1)


@pytest.mark.parametrize('nonlinear', [False, True])
def test_square_rows_original_semantics_and_crown_cuts(nonlinear):
    model = Rows(nonlinear)
    bm = bounded(model, dict(square_row_to_abs=True))
    C = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    rhs = torch.tensor([[0, 2, 0, 0.00001]], dtype=torch.float64)
    tol = 1e-4
    assert len(prove_rows(bm, C[0])[0]) == 1
    box = sample_bounds(bm, model)
    A, b = square_row_cuts(bm, C, rhs, box, tol)
    assert A.shape == (1, 2, 2)
    axis = torch.linspace(-2, 2, 601, dtype=torch.float64)
    points = torch.cartesian_prod(axis, axis)
    original = model(points)
    feasible = (original[:, 2] <= rhs[0, 2]+tol)
    assert feasible.any()
    assert ((points[feasible] @ A[0].T) <= b[0]+1e-12).all()
    torch.testing.assert_close(bm(points), original)
    local = square_row_cuts(bm, C, rhs, box, tol, nonlinear_only=True)
    assert (local is not None) == nonlinear
    if not nonlinear:
        radius = b[0, 0]
        assert Fraction(float(radius))**2 >= Fraction(1e-5)+Fraction(tol)
        assert feasible.any()


def test_square_row_refusals():
    bm = bounded(Rows(), dict(square_row_to_abs=True))
    C = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    rhs = torch.tensor([[0, 2, -1, 0]], dtype=torch.float64)
    x = torch.zeros(1, 2, dtype=torch.float64)
    box = BoundedTensor(x, PerturbationLpNorm(x_L=x-1, x_U=x+1))
    assert square_row_cuts(bm, C, rhs, box, 1e-4) is None
    negative_square = C[:, :3].clone()
    negative_square[:, 2] *= -1
    assert square_row_cuts(bm, negative_square, rhs[:, :3], box, 1e-4) is None
    assert square_row_cuts(bounded(Rows(), {}), C, rhs, box, 1e-4) is None


def test_square_strip_concretization_keeps_original_tolerance_boundary():
    model = Rows()
    bm = bounded(model, dict(square_row_to_abs=True))
    C = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    rhs = torch.tensor([[0, 2, 0, 0.00001]], dtype=torch.float64)
    lo = torch.tensor([[-2.0, 0.0]], dtype=torch.float64)
    hi = torch.tensor([[2.0, 0.0]], dtype=torch.float64)
    box = BoundedTensor(lo, PerturbationLpNorm(x_L=lo, x_U=hi))
    A, b = square_row_cuts(bm, C, rhs, box, 1e-4)
    # Base rows are consumed directly by constrained concretization. The RHS
    # must retain the tolerance already included in the square-row radius.
    constrained = BoundedTensor(lo, PerturbationLpNorm(
        x_L=lo, x_U=hi, constraints=(A, -b)))
    radius = outward_sqrt(Fraction(1e-5)+Fraction(1e-4), rhs)
    inside = torch.nextafter(radius, radius.new_tensor(0.0))
    points = torch.tensor([[-inside, 0.0], [inside, 0.0]], dtype=lo.dtype)
    values = model(points)
    assert (values[:, 2] <= rhs[0, 2]+1e-4).all()
    for method in ('IBP', 'CROWN'):
        lower, upper = bm.compute_bounds(x=(constrained,), method=method)
        assert (lower <= values.min(0).values+1e-12).all()
        assert (upper >= values.max(0).values-1e-12).all()


@pytest.mark.parametrize('alpha', [-2.0, 2.0])
def test_objective_alias_exact_and_floor(alpha):
    model = Rows(alias=alpha)
    bm = bounded(model, dict(objective_row_alias=True))
    C = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    rhs = torch.tensor([[100, 0.2, 0, 0]], dtype=torch.float64)
    tol = 1e-4
    aliases = prove_rows(bm, C[0])[1]
    assert aliases == [(1, Fraction(alpha), Fraction(0.3))]
    floor = objective_alias_floor(bm, C, rhs, tol)
    if alpha > 0:
        assert torch.isneginf(floor).all()
    else:
        exact = (Fraction(0.2)+Fraction(tol)-Fraction(0.3))/Fraction(alpha)
        assert Fraction(float(floor)) <= exact
        axis = torch.linspace(-2, 2, 301, dtype=torch.float64)
        points = torch.cartesian_prod(axis, axis)
        y = model(points)
        feasible = y[:, 1] <= rhs[0, 1]+tol
        assert (floor <= y[feasible, 0]+1e-12).all()
    sample_bounds(bm, model)


@pytest.mark.parametrize('alpha,changed', [(0, False), (-2, True)])
def test_alias_refusals(alpha, changed):
    bm = bounded(Rows(alias=alpha, changed=changed), dict(objective_row_alias=True))
    C = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    rhs = torch.zeros(1, 4, dtype=torch.float64)
    assert torch.isneginf(objective_alias_floor(bm, C, rhs, 1e-4)).all()


class VectorizedAlias(nn.Module):
    def __init__(self, changed=False):
        super().__init__()
        self.changed = changed
        self.register_buffer('index', torch.tensor([0, 0, 1, 1]))

    def forward(self, x):
        a = torch.cat((x[:, :1], x[:, 1:2], x[:, :1], x[:, 1:2]), -1)
        b = torch.cat((x[:, 1:2], x[:, :1], -x[:, 1:2], -x[:, :1]), -1)
        if self.changed:
            b = b + torch.tensor([0, 0, 1e-12, 0], dtype=x.dtype)
        return x.new_zeros((x.shape[0], 2)).index_add(1, self.index, a*b)


@pytest.mark.parametrize('changed', [False, True])
def test_vectorized_structural_alias_and_repeated_index_add(changed):
    model = VectorizedAlias(changed)
    bm = bounded(model, dict(objective_row_alias=True))
    C = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    rhs = torch.tensor([[100, 0.25]], dtype=torch.float64)
    floor = objective_alias_floor(bm, C, rhs, 1e-4)
    if changed:
        assert torch.isneginf(floor).all()
    else:
        assert prove_rows(bm, C[0])[1] == [(1, Fraction(-1), Fraction(0))]
        assert float(floor) <= -0.2501
    poly = _PresolveProof(bm, OpBudget("t", 1 << 40)).build(bm.final_node())
    input_name = bm.input_name[0]
    assert poly.flat[0].terms == {
        ((input_name, 0), (input_name, 1)): Fraction(2)}
    sample_bounds(bm, model)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('c,k', [(8.6, -70.0), (-3.0, 0.7)])
def test_completion_dtype_and_negative_curvature(dtype, c, k):
    model = Quadratic(c=c, k=k).to(dtype)
    x = torch.zeros(1, 2, dtype=dtype)
    bm = BoundedModule(model, x, bound_opts=dict(OPTIONS, complete_square_rewrite=True),
                       device='cpu')
    original = BoundedModule(model, x, bound_opts=OPTIONS, device='cpu')
    a = _PresolveProof(original, OpBudget("t", 1 << 40)).build(original.final_node())
    b = _PresolveProof(bm, OpBudget("t", 1 << 40)).build(bm.final_node())
    assert [p.terms for p in a.flat] == [p.terms for p in b.flat]
    pts = torch.rand(1024, 2, dtype=dtype,
                     generator=torch.Generator().manual_seed(27))*4-2
    s = x.new_tensor(k/(2*c))
    magnitude = (abs(c)*pts[:, :1].square() + abs(k)*pts[:, :1].abs()
                 + 2*pts[:, 1:2].abs() + abs(c)*(pts[:, :1]+s).square()
                 + abs(c)*s.square())
    assert ((bm(pts)-model(pts)).abs() <= rounding_error(magnitude, 12)).all()
    box = BoundedTensor(x, PerturbationLpNorm(x_L=x-2, x_U=x+2))
    for method in ('IBP', 'CROWN'):
        lo, hi = bm.compute_bounds(x=(box,), method=method)
        assert (lo <= model(pts).min(0).values + 1e-5).all()
        assert (hi >= model(pts).max(0).values - 1e-5).all()


@pytest.mark.parametrize('q', [Fraction(0), Fraction(1, 2**1200),
                             Fraction(2**1600), Fraction(1, 3)])
def test_outward_sqrt_extreme_scales(q):
    radius = outward_sqrt(q, torch.zeros((), dtype=torch.float64))
    assert Fraction(float(radius))**2 >= q
    if radius > 0:
        previous = torch.nextafter(radius, radius.new_tensor(-torch.inf))
        assert Fraction(float(previous))**2 < q


def test_shift_rounds_directly_to_float32():
    like = torch.zeros((), dtype=torch.float32)
    midpoint = Fraction(1) + Fraction(1, 2**24)
    assert float(nearest_float(midpoint, like)) == 1
    assert float(nearest_float(midpoint+Fraction(1, 2**80), like)) == 1+2**-23


class BufferedRows(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.tensor([[1.0, -0.2]], dtype=torch.float64))
        self.register_buffer('bias', torch.tensor([0.5], dtype=torch.float64))

    def forward(self, x):
        h = x @ self.weight.T + self.bias
        return torch.cat((x[:, :1], h.square()-1e-5), -1)


def test_square_strip_with_traced_buffers():
    model = BufferedRows()
    bm = bounded(model, dict(square_row_to_abs=True))
    assert len(bm.input_name) > 1
    C = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    rhs = torch.zeros(1, 2, dtype=torch.float64)
    box = sample_bounds(bm, model)
    A, b = square_row_cuts(bm, C, rhs, box, 1e-4)
    assert A.shape == (1, 2, 2)
    points = torch.rand(10000, 2, dtype=rhs.dtype,
                        generator=torch.Generator().manual_seed(27))*4-2
    feasible = model(points)[:, 1] <= 1e-4
    assert feasible.any()
    assert (points[feasible] @ A[0].T <= b[0]).all()


def test_unsupported_constant_function_is_an_atom():
    class ConstantFunction(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('value', torch.tensor(0.7, dtype=torch.float64))

        def forward(self, x):
            q = torch.exp(self.value)
            obj = x[:, :1].square()+x[:, 1:2]
            return torch.cat((obj, -obj+q, (x[:, :1]+x[:, 1:2]).square()+q), -1)

    model = ConstantFunction()
    bm = bounded(model, dict(complete_square_rewrite=True, square_row_to_abs=True,
                             objective_row_alias=True))
    C = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    rhs = torch.zeros(1, 3, dtype=torch.float64)
    assert torch.isneginf(objective_alias_floor(bm, C, rhs, 1e-4)).all()
    box = sample_bounds(bm, model)
    assert square_row_cuts(bm, C, rhs, box, 1e-4) is None


@pytest.mark.parametrize('scaled_product', [False, True])
def test_subtraction_and_scaled_product(scaled_product):
    class SubtractQuadratic(nn.Module):
        def forward(self, x):
            z = x[:, :1]
            square = (8.6*z)*z if scaled_product else 8.6*(z*z)
            return square - 70*z + x[:, 1:2]

    model = SubtractQuadratic()
    bm = bounded(model, dict(complete_square_rewrite=True))
    assert bm.presolve_rewrite_counts['complete_square_rewrite'] == 1
    original = bounded(model, {})
    a = _PresolveProof(original, OpBudget("t", 1 << 40)).build(original.final_node())
    b = _PresolveProof(bm, OpBudget("t", 1 << 40)).build(bm.final_node())
    assert [p.terms for p in a.flat] == [p.terms for p in b.flat]
    sample_bounds(bm, model)
