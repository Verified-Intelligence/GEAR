"""Native CROWN rewrites of the quadratic tangent bound and the GP certificate (nlp_solver/crown_rewrite.py)."""
from fractions import Fraction

import numpy as np
import pytest
import torch

from nlp_solver.crown_rewrite import PosynomialPlanes, QuadraticRewrite
from nlp_solver.gp_certificate import Form
from nlp_solver.quadratic_tangent import QuadraticTangent
from tests.test_gp_certificate import RandomGP, config, setup  # noqa: F401  (config is an autouse fixture there)


def _quadratic(H, c):
    return lambda x: 0.5 * (x @ H @ x) + c @ x


def _problem(seed, n=5, negative=0):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(n, n, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(M)
    lam = torch.rand(n, generator=g, dtype=torch.float64) + 0.1
    lam[:negative] = -lam[:negative]
    H = Q @ torch.diag(lam) @ Q.T
    c = torch.randn(n, generator=g, dtype=torch.float64)
    xh = 0.3 * torch.randn(n, generator=g, dtype=torch.float64)
    f = _quadratic(H, c)
    qtb = QuadraticTangent(f, n, xh, root_lower=-torch.full((1, n), 3.), root_upper=torch.full((1, n), 3.))
    return f, qtb, xh


def _below(f, a, b, l, u, k=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = l + (u - l) * torch.rand(k, l.numel(), generator=g, dtype=torch.float64)
    values = torch.stack([f(p) for p in pts])
    return bool((pts @ a + b <= values + 1e-9 * (1 + values.abs())).all())


@pytest.mark.parametrize('seed', range(4))
def test_rewrite_is_qtb_plane_on_the_anchor_box_and_valid_everywhere(seed):
    f, qtb, xh = _problem(seed)
    rw = QuadraticRewrite(qtb, 'cpu')
    l, u = (xh - 0.5).reshape(1, -1), (xh + 0.7).reshape(1, -1)
    a, b = rw.plane(l, u)
    aq, bq = qtb.plane(l, u)
    assert torch.allclose(a, aq, atol=1e-9)
    assert abs(float(b - bq)) <= 1e-6 * (1 + abs(float(bq)))
    assert _below(f, a[0], b[0], l[0], u[0])
    # away from the anchor: still valid, and its box minimum is at least QTB's
    l2, u2 = (xh + 0.8).reshape(1, -1), (xh + 1.5).reshape(1, -1)
    a2, b2 = rw.plane(l2, u2)
    aq2, bq2 = qtb.plane(l2, u2)
    assert _below(f, a2[0], b2[0], l2[0], u2[0])
    box_min = lambda a_, b_: float(torch.where(a_ >= 0, a_ * l2, a_ * u2).sum() + b_)
    assert box_min(a2, b2) >= box_min(aq2, bq2) - 1e-9


@pytest.mark.parametrize('seed', range(3))
def test_rewrite_of_an_indefinite_quadratic_is_valid(seed):
    f, qtb, xh = _problem(seed, negative=2)
    rw = QuadraticRewrite(qtb, 'cpu')
    for shift, width in ((-0.4, 0.9), (0.6, 1.0), (-1.5, 0.5)):
        l, u = (xh + shift).reshape(1, -1), (xh + shift + width).reshape(1, -1)
        a, b = rw.plane(l, u)
        assert _below(f, a[0], b[0], l[0], u[0], seed=seed)


def test_rewrite_handles_batches_and_fixed_coordinates():
    f, _, xh = _problem(5)
    lower, upper = -torch.full((1, 5), 3.), torch.full((1, 5), 3.)
    lower[0, 2] = upper[0, 2] = 0.25
    qtb = QuadraticTangent(f, 5, xh, root_lower=lower, root_upper=upper)
    rw = QuadraticRewrite(qtb, 'cpu')
    anchor = qtb.x_hat
    l = torch.stack([anchor - 0.2, anchor + 0.4])
    u = torch.stack([anchor + 0.3, anchor + 0.9])
    l[:, 2] = u[:, 2] = 0.25
    a, b = rw.plane(l, u)
    assert a.shape == (2, 5) and b.shape == (2,)
    for i in range(2):
        assert _below(f, a[i], b[i], l[i], u[i])


def _posynomial(form, y):
    return sum(float(c) * np.exp(y @ np.array([float(e) for e in a])) for a, c in form.terms.items())


def test_posynomial_planes_are_valid_and_touch_at_the_point():
    form = Form('poly', {(Fraction(1, 3), Fraction(-7, 5)): Fraction(11, 3),
                         (Fraction(-8, 9), Fraction(5, 3)): Fraction(2, 7),
                         (Fraction(2), Fraction(1, 2)): Fraction(1, 5)})
    lo, hi = np.array([-3., -2.]), np.array([2., 3.])
    planes = PosynomialPlanes([form], lo, hi)
    for point in (np.array([1.2, -.7]), np.array([-2.5, 2.9]), (lo + hi) / 2):
        (slope, intercept), = planes.planes(point)
        pts = np.random.default_rng(3).uniform(lo, hi, (10000, 2))
        assert np.all(pts @ slope + intercept <= _posynomial(form, pts))
        value = _posynomial(form, point[None])[0]
        assert point @ slope + intercept == pytest.approx(value, rel=1e-9)


@pytest.mark.parametrize('seed', range(5))
def test_gp_crown_planes_certify_the_same_bound(seed):
    model = RandomGP(seed)
    gp, diag, *_ = setup(model, n=3)
    assert gp is not None, diag
    rng = np.random.default_rng(seed + 12)
    points = torch.tensor(rng.uniform(.2, 3., (20000, 3)))
    values = model(points)
    feasible = values[:, 1] <= 1e-4
    best = int(torch.argmin(torch.where(feasible, values[:, 0], torch.inf)))
    entropy, _ = gp.solve(points[best], time_budget=2.)
    gp.crown = True
    native, _ = gp.solve(points[best], time_budget=2.)
    assert native <= float(values[feasible, 0].min())
    assert native == pytest.approx(entropy, rel=1e-3, abs=1e-6)
