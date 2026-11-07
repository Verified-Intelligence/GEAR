"""Soundness and parity check for BoundExp.bound_forward.

Background:
    BoundExp.bound_forward used to compute m = min((l+u)/2, l+0.99) (the
    midpoint tangent). bound_relax (used by backward CROWN) instead uses
    the parallel-tangent point m* = log((exp(u)-exp(l))/(u-l)), where
    exp'(m*) = (exp(u)-exp(l))/(u-l) = secant slope. This is the
    convention used throughout backward CROWN. The forward-mode override
    has been removed so BoundExp.bound_forward now inherits
    BoundActivation.bound_forward, which calls bound_relax(x, init=True)
    and composes the relaxation with the input's forward-linear form.
    Forward and backward modes are now in parity.

Tests:
  1. Soundness: on random boxes, the concretized forward upper bound is
     >= exp(x) >= concretized forward lower bound at sample points inside
     the input box. Primary correctness guarantee.
  2. Structural: the lower-bound slope equals the secant slope -- the
     defining property of the parallel-tangent point.
  3. Parity vs. bound_relax: bexp.lw / lb / uw / ub populated by
     bound_forward equal those produced by a direct bound_relax call.
     This is the explicit forward-vs-backward parity check.
  4. The case where x has no forward-linear form (lw = uw = None) still
     works via the fallback path.

Note: pointwise and integrated (L1) tightness dominance vs. midpoint do
NOT hold in general -- the midpoint tangent is L1-optimal on the box.
The parallel tangent is the convention for backward-CROWN parity; the
win shows up in symbolic compositions with downstream coefficients, not
in standalone box-concretization metrics. This test asserts only the
soundness and structural properties.
"""
import pytest
import torch

from auto_LiRPA.linear_bound import LinearBound
from auto_LiRPA.operators.base import Interval
from auto_LiRPA.operators.bivariate import BoundMul
from auto_LiRPA.operators.convex_concave import BoundExp


def _make_bexp():
    bexp = BoundExp.__new__(BoundExp)
    bexp.options = {}
    bexp.max_input = 0
    bexp.has_constraint = True
    bexp.loss_fusion = False
    bexp.opt_stage = None
    bexp.alpha = {}
    bexp._start = None
    bexp.lw = torch.zeros(1)
    bexp.uw = torch.zeros(1)
    bexp.lb = torch.zeros(1)
    bexp.ub = torch.zeros(1)
    return bexp


def _make_linear_bound(lower, upper, with_linear=True):
    """Build a LinearBound where the forward-linear form is the identity:
    lw = uw = I_n, lb = ub = 0, so concretizing recovers exp at sample x.
    """
    if with_linear:
        n = lower.numel()
        eye = torch.eye(n).reshape(*lower.shape, n)
        lw = eye.clone()
        uw = eye.clone()
        lb = torch.zeros_like(lower)
        ub = torch.zeros_like(upper)
    else:
        lw = uw = None
        lb = lower.clone()
        ub = upper.clone()
    return LinearBound(lw=lw, lb=lb, uw=uw, ub=ub, lower=lower, upper=upper)


def _concretize(lin, x):
    """Concretize a LinearBound (identity forward form) at point x.
    Returns (lower_at_x, upper_at_x).
    """
    if lin.lw is None:
        return lin.lb, lin.ub
    n = x.numel()
    lw_flat = lin.lw.reshape(-1, n)
    uw_flat = lin.uw.reshape(-1, n)
    x_flat = x.reshape(-1)
    lower = (lw_flat @ x_flat).reshape(x.shape) + lin.lb
    upper = (uw_flat @ x_flat).reshape(x.shape) + lin.ub
    return lower, upper


def _run_soundness(x_l, x_u, with_linear=True, n_samples=20):
    bexp = _make_bexp()
    dim_in = x_l.numel() if with_linear else 0
    x_in = _make_linear_bound(x_l, x_u, with_linear=with_linear)
    out = bexp.bound_forward(dim_in, x_in)
    if with_linear:
        assert out.lw is not None
    for _ in range(n_samples):
        t = torch.rand_like(x_l)
        x = x_l + t * (x_u - x_l)
        if with_linear:
            lo, hi = _concretize(out, x)
        else:
            lo, hi = out.lb, out.ub
        true_val = torch.exp(x)
        assert torch.all(lo <= true_val + 1e-4), (
            f"Lower violated: max viol {(lo - true_val).max().item():.3e}")
        assert torch.all(hi + 1e-4 >= true_val), (
            f"Upper violated: max viol {(true_val - hi).max().item():.3e}")


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_soundness_random_boxes(seed):
    torch.manual_seed(seed)
    x_l = -2.0 + 4.0 * torch.rand(8)
    x_u = x_l + 0.1 + 2.0 * torch.rand(8)
    _run_soundness(x_l, x_u)


def test_soundness_no_forward_linear():
    """x.lw = x.uw = None -> constant forward bounds. Still must be sound."""
    x_l = torch.tensor([-1.0, 0.0, 0.5, 1.5])
    x_u = torch.tensor([-0.5, 0.3, 1.0, 2.0])
    bexp = _make_bexp()
    x_in = _make_linear_bound(x_l, x_u, with_linear=False)
    out = bexp.bound_forward(0, x_in)
    assert torch.all(out.lb <= torch.exp(x_l) + 1e-4)
    assert torch.all(out.ub + 1e-4 >= torch.exp(x_u))


def test_soundness_narrow_box():
    """Near-degenerate u ~= l should fall back to midpoint cleanly."""
    x_l = torch.tensor([0.0, -1.0, 2.5])
    x_u = x_l + 1e-7
    _run_soundness(x_l, x_u, n_samples=5)


def test_soundness_wide_box():
    """Wide box still sound."""
    x_l = torch.tensor([-3.0, -1.0, 0.5])
    x_u = torch.tensor([-1.0, 1.0, 3.0])
    _run_soundness(x_l, x_u)


def test_exp_check_nan_local_bias_keeps_global_bound():
    """The original local 0*inf case remains a finite global exp bound."""
    bexp = _make_bexp()
    A = torch.tensor([[[float("nan"), 2.0]], [[3.0, 4.0]]])
    bias = torch.tensor([[float("nan")], [7.0]])
    last_A = torch.tensor([[[1.0, 0.0]], [[2.0, 3.0]]])

    A_out, bias_out = bexp._check_nan(
        A, bias, last_A, const_bound=0., invalid_bound=-torch.inf)

    assert torch.all(A_out[0] == 0)
    assert bias_out[0, 0] == 0
    assert torch.allclose(A_out[1], A[1])
    assert bias_out[1, 0] == 7


def test_exp_check_nan_upstream_nan_fails_closed():
    """NaN from upstream coefficients must not become a finite bound."""
    bexp = _make_bexp()
    A = torch.tensor([[[float("nan"), 1.0]]])
    bias = torch.tensor([[5.0]])
    last_A = torch.tensor([[[float("nan"), 1.0]]])

    A_out, bias_out = bexp._check_nan(
        A, bias, last_A, const_bound=0., invalid_bound=-torch.inf)

    assert torch.all(A_out == 0)
    assert torch.isneginf(bias_out).all()


def test_mul_interval_zero_times_inf_fails_closed():
    x = Interval(torch.tensor([0.0]), torch.tensor([1.0]))
    y = Interval(torch.tensor([float("inf")]), torch.tensor([float("inf")]))

    result = BoundMul.interval_propagate_both_perturbed(x, y)
    lower, upper = result.lower, result.upper

    assert torch.isneginf(lower).all()
    assert torch.isposinf(upper).all()


def test_square_interval_zero_times_inf_has_finite_lower():
    x = Interval(torch.tensor([0.0]), torch.tensor([float("inf")]))

    result = BoundMul.interval_propagate_both_perturbed(x, x)
    lower, upper = result.lower, result.upper

    assert lower.item() == 0
    assert torch.isposinf(upper).all()


@pytest.mark.parametrize("seed", [10, 11, 12, 13])
def test_parallel_tangent_slope_matches_secant(seed):
    """The defining property of the parallel-tangent point: lower-bound
    slope = (exp(u) - exp(l)) / (u - l). This is what backward CROWN uses
    and what the forward path now inherits.
    """
    torch.manual_seed(seed)
    x_l = -1.5 + 2.0 * torch.rand(6)
    x_u = x_l + 0.2 + 1.5 * torch.rand(6)
    bexp = _make_bexp()
    x_in = _make_linear_bound(x_l, x_u, with_linear=True)
    bexp.bound_forward(x_l.numel(), x_in)
    secant = (torch.exp(x_u) - torch.exp(x_l)) / (x_u - x_l).clamp(min=1e-8)
    assert torch.allclose(bexp.lw, secant, atol=1e-5, rtol=1e-4), (
        f"Parallel-tangent slope != secant slope: max diff "
        f"{(bexp.lw - secant).abs().max().item():.3e}")
    # Upper-bound slope is also the secant (chord between (l, exp l) and
    # (u, exp u)).
    assert torch.allclose(bexp.uw, secant, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("seed", [20, 21, 22])
def test_parity_with_bound_relax(seed):
    """The inherited bound_forward should produce identical lw, lb, uw, ub
    to a direct bound_relax(x, init=True) call. This confirms parity
    between forward CROWN and the relaxation used by backward CROWN.
    """
    torch.manual_seed(seed)
    x_l = -1.5 + 2.0 * torch.rand(5)
    x_u = x_l + 0.2 + 1.5 * torch.rand(5)

    # Path A: bound_forward (which calls bound_relax internally).
    bexp_a = _make_bexp()
    x_in = _make_linear_bound(x_l, x_u, with_linear=True)
    bexp_a.bound_forward(x_l.numel(), x_in)

    # Path B: bound_relax directly.
    bexp_b = _make_bexp()
    bexp_b.bound_relax(x_in, init=True)

    assert torch.allclose(bexp_a.lw, bexp_b.lw, atol=1e-6)
    assert torch.allclose(bexp_a.lb, bexp_b.lb, atol=1e-6)
    assert torch.allclose(bexp_a.uw, bexp_b.uw, atol=1e-6)
    assert torch.allclose(bexp_a.ub, bexp_b.ub, atol=1e-6)
