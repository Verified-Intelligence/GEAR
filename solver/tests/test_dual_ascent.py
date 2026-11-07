"""
Tests for complete_verifier/nlp_solver/dual_ascent.py

Following TDD: these tests are written BEFORE the implementation.
"""
import sys
import os

import torch
import pytest

# Make sure the complete_verifier module can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'complete_verifier'))

from nlp_solver.dual_ascent import (
    DualAscentConfig,
    crown_lb_from_lA,
    dual_ascent_tighten,
)


def test_crown_lb_from_lA_shape():
    """crown_lb_from_lA should return a (B,) tensor given (B, d_in) inputs."""
    B, d_in = 3, 5
    lA = torch.randn(B, d_in)
    lbias = torch.randn(B)
    dm_l = torch.full((B, d_in), -1.0)
    dm_u = torch.full((B, d_in), 1.0)

    out = crown_lb_from_lA(lA, lbias, dm_l, dm_u)

    assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"


def test_crown_lb_from_lA_correctness():
    """Manually verify the CROWN lower bound formula on a simple example.

    lA = [[1, -1, 2, -0.5]]
    dm_l = [[-1, -1, -1, -1]]
    dm_u = [[1, 1, 1, 1]]
    lbias = [0.3]

    x_worst selects dm_l for positive coefficients, dm_u for negative:
      1*(-1) + (-1)*1 + 2*(-1) + (-0.5)*1 + 0.3 = -4.2
    """
    lA = torch.tensor([[1.0, -1.0, 2.0, -0.5]])
    dm_l = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
    dm_u = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    lbias = torch.tensor([0.3])

    out = crown_lb_from_lA(lA, lbias, dm_l, dm_u)

    expected = torch.tensor([-4.2])
    assert out.shape == (1,), f"Expected shape (1,), got {out.shape}"
    assert torch.allclose(out, expected, atol=1e-5), (
        f"Expected {expected.item():.6f}, got {out.item():.6f}"
    )


def test_dual_ascent_improves_or_equals_lb():
    """Dual ascent should never decrease the lower bound (d(0) ≤ d(λ*))."""
    torch.manual_seed(42)
    B, m, d_in = 4, 3, 8

    # Random problem: lA (B, m, d_in), lbias (B, m), dm_l/dm_u (B, d_in)
    lA = torch.randn(B, m, d_in)
    lbias = torch.randn(B, m)
    dm_l = -torch.rand(B, d_in) - 0.5   # in [-1.5, -0.5]
    dm_u = torch.rand(B, d_in) + 0.5    # in [0.5, 1.5]
    rhs = torch.zeros(B, m)             # zero thresholds for simplicity

    cfg = DualAscentConfig(num_steps=50, lr=0.05, verbose=False)
    lb_initial, lb_dual, lam_star = dual_ascent_tighten(lA, lbias, dm_l, dm_u, rhs, cfg)

    assert lb_dual.shape == (B,), f"Expected shape ({B},), got {lb_dual.shape}"
    assert lb_initial.shape == (B,), f"Expected shape ({B},), got {lb_initial.shape}"

    improvement = lb_dual - lb_initial
    assert (improvement >= -1e-6).all(), (
        f"Dual ascent decreased lower bound! improvements: {improvement}"
    )


def test_dual_ascent_lambda_nonneg():
    """Dual variables lam_star must be non-negative (projected onto λ ≥ 0)."""
    torch.manual_seed(7)
    B, m, d_in = 5, 4, 6

    lA = torch.randn(B, m, d_in)
    lbias = torch.randn(B, m)
    dm_l = -torch.ones(B, d_in)
    dm_u = torch.ones(B, d_in)
    rhs = torch.randn(B, m) * 0.1

    cfg = DualAscentConfig(num_steps=30, lr=0.05, verbose=False)
    _, _, lam_star = dual_ascent_tighten(lA, lbias, dm_l, dm_u, rhs, cfg)

    assert lam_star.shape == (B, m - 1), (
        f"Expected lam_star shape ({B}, {m-1}), got {lam_star.shape}"
    )
    assert (lam_star >= -1e-7).all(), (
        f"lam_star has negative values: min={lam_star.min().item()}"
    )


def test_dual_ascent_trivial_m1():
    """When m=1 (only objective spec, no constraints), lam_star should be empty
    and lb_dual should equal lb_initial."""
    torch.manual_seed(99)
    B, m, d_in = 6, 1, 10

    lA = torch.randn(B, m, d_in)
    lbias = torch.randn(B, m)
    dm_l = -torch.ones(B, d_in)
    dm_u = torch.ones(B, d_in)
    rhs = torch.randn(B, m) * 0.5

    cfg = DualAscentConfig(num_steps=20, lr=0.05, verbose=False)
    lb_initial, lb_dual, lam_star = dual_ascent_tighten(lA, lbias, dm_l, dm_u, rhs, cfg)

    assert lam_star.shape == (B, 0), (
        f"Expected lam_star shape ({B}, 0) for m=1, got {lam_star.shape}"
    )
    assert torch.allclose(lb_dual, lb_initial, atol=1e-6), (
        f"lb_dual != lb_initial for m=1: max diff = {(lb_dual - lb_initial).abs().max().item()}"
    )


def test_dual_ascent_returns_best_iterate_not_overshot_final():
    """The certificate and returned lambda must come from the same iterate."""
    lA = torch.tensor([[[-0.01], [0.2]]], dtype=torch.float64)
    lbias = torch.zeros(1, 2, dtype=torch.float64)
    dm_l = torch.tensor([[-1.0]], dtype=torch.float64)
    dm_u = torch.tensor([[1.0]], dtype=torch.float64)
    rhs = torch.zeros(1, 2, dtype=torch.float64)

    # The first Adam update lands near the dual optimum at lambda=0.05. Its
    # momentum then overshoots; the last iterate has a worse dual value.
    lb_initial, lb_dual, lam_star = dual_ascent_tighten(
        lA, lbias, dm_l, dm_u, rhs,
        DualAscentConfig(num_steps=2, lr=0.05))

    assert torch.allclose(lam_star, torch.tensor([[0.05]], dtype=torch.float64), atol=1e-5)
    assert lb_dual.item() > lb_initial.item() + 0.005

    lA_eff = lA[:, 0, :] + (lam_star.unsqueeze(-1) * lA[:, 1:, :]).sum(1)
    lbias_eff = lbias[:, 0] + (lam_star * lbias[:, 1:]).sum(1)
    expected = crown_lb_from_lA(lA_eff, lbias_eff, dm_l, dm_u) - rhs[:, 0] \
        - (lam_star * rhs[:, 1:]).sum(1)
    assert torch.allclose(lb_dual, expected, atol=1e-10)


def test_dual_ascent_sanitizes_nonfinite_warmstart():
    lA = torch.tensor([[[1.0], [-1.0]]], dtype=torch.float64)
    lbias = torch.zeros(1, 2, dtype=torch.float64)
    dm_l = torch.tensor([[-1.0]], dtype=torch.float64)
    dm_u = torch.tensor([[1.0]], dtype=torch.float64)
    rhs = torch.zeros(1, 2, dtype=torch.float64)

    lb_initial, lb_dual, lam_star = dual_ascent_tighten(
        lA, lbias, dm_l, dm_u, rhs,
        DualAscentConfig(num_steps=0),
        lambda_init=torch.tensor([[float('nan')]], dtype=torch.float64))

    assert torch.isfinite(lam_star).all()
    assert torch.equal(lam_star, torch.zeros_like(lam_star))
    assert torch.isfinite(lb_dual).all()
    assert torch.allclose(lb_dual, lb_initial)


@pytest.mark.parametrize('bad_row', [float('nan'), float('inf'), -float('inf')])
def test_dual_ascent_nonfinite_row_does_not_affect_other_domains(bad_row, monkeypatch):
    lA = torch.tensor([[[1.0], [bad_row]], [[1.0], [-1.0]]], dtype=torch.float64)
    lbias = torch.zeros(2, 2, dtype=torch.float64)
    dm_l = -torch.ones(2, 1, dtype=torch.float64)
    dm_u = torch.ones(2, 1, dtype=torch.float64)
    rhs = torch.zeros(2, 2, dtype=torch.float64)
    cfg = DualAscentConfig(num_steps=5)
    _, expected_bound, expected_lam = dual_ascent_tighten(
        lA[1:], lbias[1:], dm_l[1:], dm_u[1:], rhs[1:], cfg)

    adam_step = torch.optim.Adam.step
    gradients = []

    def checked_step(optimizer, *args, **kwargs):
        lam = optimizer.param_groups[0]['params'][0]
        gradients.append(lam.grad.detach().clone())
        return adam_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, 'step', checked_step)
    initial, bound, lam = dual_ascent_tighten(lA, lbias, dm_l, dm_u, rhs, cfg)

    assert gradients
    assert all(torch.isfinite(grad).all() and (grad[0] == 0).all() for grad in gradients)
    assert bound[0] == initial[0]
    assert lam[0].item() == 0
    assert torch.allclose(bound[1:], expected_bound)
    assert torch.allclose(lam[1:], expected_lam)


# ---- TH2: the returned values are rigorous, not plain floating point ----
from fractions import Fraction as _Fr


def _exact_dual(lA, lbias, rhs, dm_l, dm_u, inA, inb, lam, b=0):
    """Exact rational Lagrangian value of the plane LP at lam, from the float data."""
    n = lA.shape[-1]
    m1 = lA.shape[1] - 1
    f = lambda t: _Fr(float(t))
    c = [f(lA[b, 0, j]) for j in range(n)]
    val = f(lbias[b, 0]) - f(rhs[b, 0])
    for k in range(m1):
        y = f(lam[b, k])
        val -= y * (f(rhs[b, 1 + k]) - f(lbias[b, 1 + k]))
        for j in range(n):
            c[j] += y * f(lA[b, 1 + k, j])
    if inA is not None:
        for k in range(inA.shape[1]):
            y = f(lam[b, m1 + k])
            val -= y * f(inb[b, k])
            for j in range(n):
                c[j] += y * f(inA[b, k, j])
    for j in range(n):
        val += min(c[j] * f(dm_l[b, j]), c[j] * f(dm_u[b, j]))
    return val


def _cancelling_case(dtype):
    # objective plane +1e8*x and a constraint plane -1e8*x + tiny: the plain sum cancels to garbage.
    lA = torch.tensor([[[1e8, 0.3], [-1e8, 0.1]]], dtype=dtype)
    lbias = torch.tensor([[0.1, 1e-3]], dtype=dtype)
    rhs = torch.tensor([[0.0, 0.2]], dtype=dtype)
    return lA, lbias, rhs, torch.full((1, 2), -1.0, dtype=dtype), torch.full((1, 2), 1.0, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_adam_return_is_rigorous_on_cancelling_plane(dtype):
    lA, lbias, rhs, dm_l, dm_u = _cancelling_case(dtype)
    cfg = DualAscentConfig(num_steps=30, lr=0.5)
    lb0, lb, lam = dual_ascent_tighten(lA, lbias, dm_l, dm_u, rhs, cfg)
    assert lb.item() >= lb0.item()
    for val, y in ((lb0, torch.zeros_like(lam)), (lb, lam)):
        assert _Fr(float(val)) <= _exact_dual(lA, lbias, rhs, dm_l, dm_u, None, None, y)
    # the plain floating-point value at lam overshoots the exact one here
    plain = crown_lb_from_lA(lA[:, 0] + (lam.unsqueeze(-1) * lA[:, 1:]).sum(1),
                             lbias[:, 0] + (lam * lbias[:, 1:]).sum(1), dm_l, dm_u) \
        - (lam * rhs[:, 1:]).sum(1) - rhs[:, 0]
    assert _Fr(float(plain)) > _exact_dual(lA, lbias, rhs, dm_l, dm_u, None, None, lam) or dtype == torch.float64


def test_adam_d0_not_above_native_lb():
    # d(0) is the native plane value minus the outward margin: never above the native lb.
    torch.manual_seed(1)
    for dtype in (torch.float32, torch.float64):
        lA = torch.randn(4, 1, 6, dtype=dtype)
        dm_l, dm_u = -torch.rand(4, 6, dtype=dtype), torch.rand(4, 6, dtype=dtype)
        lbias = torch.randn(4, 1, dtype=dtype)
        native = crown_lb_from_lA(lA[:, 0], lbias[:, 0], dm_l, dm_u)
        lb0, _, _ = dual_ascent_tighten(lA, lbias, dm_l, dm_u, torch.zeros(4, 1, dtype=dtype),
                                        DualAscentConfig(num_steps=0))
        assert (lb0 <= native).all()
        assert ((native - lb0) <= 1e-5 * (1 + native.abs())).all()


def test_adam_rigorous_fuzz():
    torch.manual_seed(7)
    bad = 0
    for trial in range(60):
        dtype = torch.float32 if trial % 2 else torch.float64
        B, m, n, k = 2, 1 + int(torch.randint(1, 4, ())), int(torch.randint(2, 6, ())), int(torch.randint(0, 3, ()))
        scale = 10 ** float(torch.randint(-2, 9, ()))
        lA = torch.randn(B, m, n, dtype=dtype) * scale
        lbias = torch.randn(B, m, dtype=dtype)
        rhs = torch.randn(B, m, dtype=dtype)
        lo = -torch.rand(B, n, dtype=dtype)
        hi = torch.rand(B, n, dtype=dtype)
        inA = torch.randn(B, k, n, dtype=dtype) if k else None
        inb = torch.randn(B, k, dtype=dtype) if k else None
        lb0, lb, lam = dual_ascent_tighten(lA, lbias, lo, hi, rhs, DualAscentConfig(num_steps=15, lr=0.2),
                                           input_constr_A=inA, input_constr_b=inb)
        for b in range(B):
            if _Fr(float(lb[b])) > _exact_dual(lA, lbias, rhs, lo, hi, inA, inb, lam, b):
                bad += 1
            if _Fr(float(lb0[b])) > _exact_dual(lA, lbias, rhs, lo, hi, inA, inb, torch.zeros_like(lam), b):
                bad += 1
    assert bad == 0
