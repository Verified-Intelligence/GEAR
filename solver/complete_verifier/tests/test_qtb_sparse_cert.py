"""QTB ||H - H_hat|| certificate: the O(nnz) exact-Hessian bound and the deadline-aware jet fallback."""
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from auto_LiRPA import BoundedModule  # noqa: E402
from nlp_solver.quadratic_tangent import build_quadratic_tangent  # noqa: E402


class Quad(nn.Module):
    """f = z^T Q z / 2 + c.z with sparse symmetric Q (constants are binary64)."""

    def __init__(self, n, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        Q = torch.randn(n, n, generator=g, dtype=torch.float64) * (torch.rand(n, n, generator=g) < 0.4)
        self.register_buffer('Q', (Q + Q.T) / 2)
        self.register_buffer('c', torch.randn(n, generator=g, dtype=torch.float64))

    def forward(self, z):
        return ((z @ self.Q) * z).sum(-1, keepdim=True) / 2 + (z * self.c).sum(-1, keepdim=True)


def build(n, seed, deadline=math.inf):
    torch.set_default_dtype(torch.float64)
    m, x = Quad(n, seed), torch.zeros(1, n, dtype=torch.float64)
    xh = torch.randn(n, dtype=torch.float64)
    qt = build_quadratic_tangent(BoundedModule(m, x), m, x, 0, xh, deadline=deadline)
    return m, qt


def test_sparse_bound_dominates_true_error():
    for n, seed in ((4, 0), (7, 1), (12, 2)):
        m, qt = build(n, seed)
        assert qt is not None and qt.exact_hessian is not None
        H_hat = torch.autograd.functional.hessian(lambda z: m(z.reshape(1, -1))[0, 0], torch.zeros(n))
        H_hat = (H_hat + H_hat.T) / 2
        true = torch.linalg.matrix_norm(m.Q - H_hat, ord=2)       # the Hessian of f is Q exactly
        assert qt.hess_err is not None and math.isfinite(qt.hess_err)
        assert qt.hess_err >= float(true)
        assert qt.hess_err < 1e-9                                   # tight: the sparse path is not the jet slack


def test_deadline_returns_inf_and_no_plane():
    m, qt = build(6, 3, deadline=time.time() - 1)
    qt.exact_hessian = None                                         # force the jet fallback
    free = (~qt.fixed_mask).nonzero().flatten()
    H = torch.autograd.functional.hessian(lambda z: m(z.reshape(1, -1))[0, 0], torch.zeros(6))
    qt.deadline = time.time() - 1
    assert qt._probe_error(H, qt.x_hat.clone(), free) == math.inf


def test_extrapolation_gives_up_after_first_chunk():
    m, qt = build(24, 4)
    qt.exact_hessian = None
    free = (~qt.fixed_mask).nonzero().flatten()
    H = torch.autograd.functional.hessian(lambda z: m(z.reshape(1, -1))[0, 0], torch.zeros(24))
    qt.deadline = time.time() + 1e-4          # the first chunk alone outlasts this
    assert qt._probe_error(H, qt.x_hat.clone(), free, chunk=2) == math.inf
    qt.deadline = math.inf
    assert math.isfinite(qt._probe_error(H, qt.x_hat.clone(), free, chunk=2))


def test_expired_deadline_gives_no_curvature_certificate():
    m, qt = build(6, 5)
    qt.exact_hessian = None
    qt.deadline = time.time() - 1
    H = torch.autograd.functional.hessian(lambda z: m(z.reshape(1, -1))[0, 0], torch.zeros(6))
    err = qt._hessian_error(H, qt.x_hat)
    assert err == math.inf


# --- review regressions (opus_review_qtb) -----------------------------------------------------------------------------
class AlphaQuad(nn.Module):
    """f = a x^T Wt x through addmm(alpha=a): BoundLinear with alpha != 1; H_01 = a (Wt01 + Wt10) in exact rationals."""

    def __init__(self, Wt, a):
        super().__init__()
        self.register_buffer('Wt', Wt)
        self.a = a

    def forward(self, x):
        return (torch.addmm(torch.zeros(1, 2), x, self.Wt, alpha=self.a) * x).sum(-1, keepdim=True)


def test_alpha_linear_is_exact_not_rounded():
    from fractions import Fraction as Fr
    torch.set_default_dtype(torch.float64)
    a, w, eps = 1 / 3, 0.9, 2.0 ** -12
    Wt = torch.tensor([[0.0, w], [-w * (1 + eps), 0.0]])
    h = Fr(a) * (Fr(float(Wt[0, 1])) + Fr(float(Wt[1, 0])))
    m, x0 = AlphaQuad(Wt, a), torch.zeros(1, 2)
    qt = build_quadratic_tangent(BoundedModule(m, x0, device='cpu'), m, x0, 0, torch.tensor([0.3, -0.2]))
    assert qt is not None
    assert Fr(qt.lam_lb) <= -abs(h)          # lambda_min = -|h|; the rounded alpha*W certified a larger value


def test_tiny_scale_backward_term_does_not_underflow():
    import pytest
    mp = pytest.importorskip('mpmath')
    from fractions import Fraction as Fr
    mp.mp.dps = 60
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(700)
    n, s = 4, 1e-300
    Q = torch.randn(n, n, generator=g)
    Q = (Q + Q.T) / 2 * s
    m = Quad(n, 0)
    m.Q.copy_(Q)
    m.c.zero_()
    qt = build_quadratic_tangent(BoundedModule(m, torch.zeros(1, n), device='cpu'), m, torch.zeros(1, n), 0,
                                 torch.randn(n))
    M = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            q = Fr(float(Q[i, j]))
            M[i, j] = mp.mpf(q.numerator) / q.denominator
    lam = min(mp.eigsy(M, eigvals_only=True))
    assert mp.mpf(qt.lam_lb) <= lam


def test_rewrite_uses_the_certified_hessian():
    from nlp_solver.crown_rewrite import QuadraticRewrite
    m, qt = build(5, 9)
    qt.objective_fn = lambda z: m(z.reshape(1, -1)).reshape(-1)[0] + 1e-3 * (z * z).sum()   # a recomputed H differs
    rw = QuadraticRewrite(qt, 'cpu')
    assert torch.equal(rw.lam, torch.linalg.eigh(qt.H_free)[0])


def test_merge_sumsq_nan_chunk_is_inf():
    from nlp_solver.quadratic_tangent import _merge_sumsq
    assert _merge_sumsq((0.0, 1.0), torch.tensor([float('nan'), 1.0]))[0] == math.inf


class _Cancel(nn.Module):
    """s * [(x0 + B x1)^2 - (B x1)^2 - 2B x0 x1 + t x1^2] = s x0^2 + s t x1^2 exactly; autograd loses the t term."""

    def __init__(self, s, B, t):
        super().__init__()
        self.s, self.B, self.t = s, B, t

    def forward(self, x):
        B, t = self.B, self.t
        x0, x1 = x[:, :1], x[:, 1:2]
        return self.s * ((x0 + B * x1) ** 2 - (B * x1) ** 2 - 2 * B * x0 * x1 + t * x1 ** 2)


def test_exact_rational_path_does_not_underflow(monkeypatch):
    from fractions import Fraction as Fr
    import nlp_solver.quadratic_tangent as QT
    torch.set_default_dtype(torch.float64)
    monkeypatch.setattr(QT, '_exact_hessian', lambda *a, **k: None)
    monkeypatch.setattr(QT.QuadraticTangent, '_probe_error', lambda self, *a, **k: math.inf)
    s, B, t = 1e-170, 1e8, -1e-8
    m, x0 = _Cancel(s, B, t), torch.zeros(1, 2)
    qt = build_quadratic_tangent(BoundedModule(m, x0, device='cpu'), m, x0, 0, torch.tensor([0.3, -0.2]))
    assert Fr(qt.lam_lb) <= 2 * Fr(s) * Fr(t)
