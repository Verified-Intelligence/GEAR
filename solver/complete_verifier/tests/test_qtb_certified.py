"""QTB plane soundness under cancellation in f(x_hat) (THEORY.md G2 / E3): checked in exact rationals."""
import math
import os
import sys
from fractions import Fraction as F

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from auto_LiRPA import BoundedModule  # noqa: E402
from nlp_solver.quadratic_tangent import build_quadratic_tangent  # noqa: E402


class Shifted(nn.Module):
    """(z0 - A)^2 + z1^2 written expanded: z0^2 - 2 A z0 + A^2 + z1^2 (catastrophic cancellation near z0 = A)."""

    def __init__(self, A):
        super().__init__()
        self.register_buffer('A', torch.tensor(float(A), dtype=torch.float64))

    def forward(self, z):
        return (z[:, :1] * z[:, :1] - 2 * self.A * z[:, :1] + self.A * self.A + z[:, 1:2] * z[:, 1:2])


def max_violation(a, d, half=1.0, rewrite=False):
    """max over the box of plane - f in exact rationals (<= 0 means valid)."""
    torch.set_default_dtype(torch.float64)
    m, x = Shifted(a), torch.zeros(1, 2, dtype=torch.float64)
    xh = torch.tensor([a + d, 0.0], dtype=torch.float64)
    qt = build_quadratic_tangent(BoundedModule(m, x), m, x, 0, xh)
    lo = torch.tensor([[a - half, -half]], dtype=torch.float64)
    hi = torch.tensor([[a + half, half]], dtype=torch.float64)
    if rewrite:
        from nlp_solver.crown_rewrite import QuadraticRewrite
        g, b = QuadraticRewrite(qt, lo.device).plane(lo, hi)
    else:
        g, b = qt.plane(lo, hi)
    g0, g1, bb, Fa = F(float(g[0, 0])), F(float(g[0, 1])), F(float(b[0])), F(float(a))
    worst = None
    # plane - f is concave and separable: the maximum per coordinate is at the clipped stationary point
    x0 = min(max(Fa + g0 / 2, F(float(lo[0, 0]))), F(float(hi[0, 0])))
    x1 = min(max(g1 / 2, F(float(lo[0, 1]))), F(float(hi[0, 1])))
    fx = (x0 - Fa) ** 2 + x1 ** 2
    return g0 * x0 + g1 * x1 + bb - fx


E3 = [(a, d) for a in (1e3, 1e5, 1e7) for d in (1e-6, 1e-3, 0.3)]


@pytest.mark.parametrize('a,d', E3)
def test_e3_plane_is_valid(a, d):
    assert max_violation(a, d) <= 0


def test_cancellation_fuzz_1000():
    g = torch.Generator().manual_seed(0)
    bad = 0
    for _ in range(1000):
        a = 10 ** float(torch.rand(1, generator=g) * 8)
        # the model holds the constant a*a: keep it exactly representable (26 significant bits) so that the checked
        # function (x - a)^2 is the model's own function
        e = math.frexp(a)[1]
        a = math.ldexp(round(math.ldexp(a, 26 - e)), e - 26)
        d = 10 ** float(-torch.rand(1, generator=g) * 8) * (1 if float(torch.rand(1, generator=g)) < .5 else -1)
        bad += max_violation(a, d) > 0
    assert bad == 0


@pytest.mark.parametrize('a,d', E3)
def test_e3_crown_rewrite_plane_is_valid(a, d):
    assert max_violation(a, d, rewrite=True) <= 0


def test_exact_fallback_on_a_recursion_where_interval_ad_is_hopeless():
    """x_{k+1} = x_k + h (x_k^2 - c x_k) unrolled: interval AD widens exponentially (dependency), the exact rational
    evaluation does not; the plane stays finite and valid."""
    torch.set_default_dtype(torch.float64)

    class Roll(nn.Module):
        def forward(self, z):
            s = z[:, :1]
            for _ in range(100):
                s = 1.5 * s - 0.5 * s          # equals s; the interval widths double per step
            return s * s + (z[:, 1:2] - 1.0) * (z[:, 1:2] - 1.0)
    m, x = Roll(), torch.zeros(1, 2, dtype=torch.float64)
    xh = torch.tensor([0.3, 0.7], dtype=torch.float64)
    qt = build_quadratic_tangent(BoundedModule(m, x), m, x, 0, xh)
    assert qt is not None and qt.exact_eval is not None
    assert qt.f_qtb < 1e-12 and bool((qt.g_qtb < 1e-12).all())
    assert qt.qtb_certificate is not None   # the interval bound alone is not usable here
    lo, hi = torch.full((1, 2), -2.), torch.full((1, 2), 2.)
    a, b = qt.plane(lo, hi)
    assert torch.isfinite(b).all()


class Cancelling(nn.Module):
    """(A z0)^2 - C z0^2 + z1^2 with C = fl(A*A) < A^2: the exact curvature in z0 is 2 (A^2 - C) < 0 while the Hessian
    entry is computed as 0 (cancellation of rounded constant products)."""

    def __init__(self, A):
        super().__init__()
        self.register_buffer('A', torch.tensor(A)); self.register_buffer('C', torch.tensor(A * A))

    def forward(self, z):
        a = self.A * z[:, :1]
        return a * a - self.C * z[:, :1] * z[:, :1] + z[:, 1:2] * z[:, 1:2]


@pytest.mark.parametrize('rewrite', [False, True])
def test_curvature_is_certified_when_the_hessian_cancels(rewrite):
    import random
    torch.set_default_dtype(torch.float64)
    rnd = random.Random(1)
    while True:           # A whose square rounds up: the exact A^2 - fl(A^2) < 0
        A = rnd.uniform(1e5, 2e5)
        if F(A) ** 2 < F(A * A):
            break
    m, x = Cancelling(A), torch.zeros(1, 2)
    lo, hi = torch.tensor([[-10., -1.]]), torch.tensor([[10., 1.]])
    qt = build_quadratic_tangent(BoundedModule(m, x), m, x, 0, torch.tensor([0.25, 0.5]), root_lower=lo, root_upper=hi)
    q0 = F(A) ** 2 - F(A * A)
    assert qt.lam_lb <= float(2 * q0)
    if rewrite:
        from nlp_solver.crown_rewrite import QuadraticRewrite
        a, b = QuadraticRewrite(qt, lo.device).plane(lo, hi)
    else:
        a, b = qt.plane(lo, hi)
    g0, g1, bb = F(float(a[0, 0])), F(float(a[0, 1])), F(float(b[0]))
    z1 = min(max(g1 / 2, F(-1)), F(1))
    assert max(g0 * z0 + g1 * z1 + bb - (q0 * z0 * z0 + z1 * z1) for z0 in (F(-10), F(10))) <= 0
