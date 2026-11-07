"""Tests for the quadratic tangent bound (nlp_solver/quadratic_tangent.py)."""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from auto_LiRPA import BoundedModule  # noqa: E402
from nlp_solver.quadratic_tangent import (  # noqa: E402
    QuadraticTangent, build_quadratic_tangent, output_degrees,
)


class QuadModel(nn.Module):
    """[x^T Q x + c^T x, a^T x - 1, exp(x0), x0 * x1 * x0]."""

    def __init__(self, Q, nonlinear_constraints=True):
        super().__init__()
        self.register_buffer('Q', Q)
        self.register_buffer('c', torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64))
        self.nonlinear_constraints = nonlinear_constraints

    def forward(self, x):
        obj = ((x @ self.Q) * x).sum(-1, keepdim=True) + (x * self.c).sum(-1, keepdim=True)
        lin = x.sum(-1, keepdim=True) - 1.0
        if not self.nonlinear_constraints:
            return torch.cat([obj, lin], dim=-1)
        ex = torch.exp(x[:, :1])
        cub = x[:, :1] * x[:, 1:2] * x[:, :1]
        return torch.cat([obj, lin, ex, cub], dim=-1)


def _model(Q, nonlinear_constraints=True):
    torch.set_default_dtype(torch.float64)
    m = QuadModel(Q, nonlinear_constraints)
    x = torch.zeros(1, 3, dtype=torch.float64)
    return m, BoundedModule(m, x), x


def test_linear_layer_outputs_have_degree_one():
    # api._StructuralConstraintOutputWrapper appends F.linear(x, A, -b) outputs.
    torch.set_default_dtype(torch.float64)

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('A', torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]]))
            self.register_buffer('b', torch.tensor([1.0, 2.0]))

        def forward(self, x):
            obj = (x[:, :1] * x[:, 1:2])
            lin = torch.nn.functional.linear(x, self.A, -self.b)
            return torch.cat([obj, lin, torch.exp(x[:, 2:3])], dim=1)

    m = Wrapped()
    x = torch.zeros(1, 3)
    d = output_degrees(BoundedModule(m, x), x)
    assert d.tolist()[:3] == [2.0, 1.0, 0.0]
    assert d[3] == float('inf')


def test_degrees_are_exact_per_output():
    m, bm, x = _model(torch.eye(3, dtype=torch.float64))
    d = output_degrees(bm, x)
    assert d.tolist()[:2] == [2.0, 1.0]
    assert d[2] == float('inf')
    assert d[3] == 3.0


def test_non_quadratic_objective_is_rejected():
    m, bm, x = _model(torch.eye(3, dtype=torch.float64))
    assert build_quadratic_tangent(bm, m, x, 2, x.reshape(-1)) is None
    assert build_quadratic_tangent(bm, m, x, 3, x.reshape(-1)) is None


def test_plane_is_valid_on_boxes_convex_and_nonconvex():
    g = torch.Generator().manual_seed(0)
    for Q in (torch.eye(3, dtype=torch.float64) * 2,
              torch.tensor([[1.0, 3.0, 0.0], [3.0, -2.0, 1.0], [0.0, 1.0, 0.5]],
                           dtype=torch.float64)):
        m, bm, x = _model(Q)
        x_hat = torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64)
        qt = build_quadratic_tangent(bm, m, x, 0, x_hat)
        assert qt is not None
        l = torch.tensor([[-1.0, -2.0, 0.0], [0.5, -0.5, -3.0]], dtype=torch.float64)
        u = torch.tensor([[1.0, 0.0, 2.0], [0.9, 0.5, -1.0]], dtype=torch.float64)
        a, b = qt.plane(l, u)
        for k in range(2):
            z = l[k] + (u[k] - l[k]) * torch.rand(2000, 3, generator=g, dtype=torch.float64)
            f = m(z)[:, 0]
            assert torch.all(z @ a[k] + b[k] <= f + 1e-12)


def test_nan_curvature_gives_no_plane():
    m, bm, x = _model(torch.tensor([[1.0, 3.0, 0.0], [3.0, -2.0, 1.0], [0.0, 1.0, 0.5]],
                                   dtype=torch.float64))
    qt = build_quadratic_tangent(bm, m, x, 0, torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64))
    qt.lam_lb = float('nan')        # an eigvalsh of a Hessian with NaN entries
    _, b = qt.plane(torch.full((1, 3), -1.0), torch.full((1, 3), 1.0))
    assert b.item() == -float('inf')


def test_convex_tangent_is_tight_at_the_point():
    m, bm, x = _model(torch.eye(3, dtype=torch.float64))
    x_hat = torch.tensor([0.2, 0.4, -0.1], dtype=torch.float64)
    qt = build_quadratic_tangent(bm, m, x, 0, x_hat)
    a, b = qt.plane(x_hat.view(1, -1) - 1, x_hat.view(1, -1) + 1)
    f_hat = m(x_hat.view(1, -1))[0, 0]
    assert abs(float(a[0] @ x_hat + b[0] - f_hat)) < 1e-7


@pytest.mark.parametrize('fixed_curvature', [-5.0, 0.0])
def test_fixed_coordinate_uses_only_free_curvature(fixed_curvature):
    Q = torch.tensor([[fixed_curvature, 3.0, 0.0], [3.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
                     dtype=torch.float64)
    m, bm, x = _model(Q, nonlinear_constraints=False)
    lower = torch.tensor([[2.0, -10.0, -20.0]])
    upper = torch.tensor([[2.0, 10.0, 20.0]])
    # The supplied anchor can disagree with the fixed coordinate: it must be projected.
    qt = build_quadratic_tangent(bm, m, x, 0, torch.tensor([9.0, 0.3, -0.5]),
                                 root_lower=lower, root_upper=upper)
    assert qt.fixed_mask.tolist() == [True, False, False]
    assert qt.x_hat[0] == 2.0
    assert 1.999999 < qt.lam_lb < 2.0
    l = torch.tensor([[2.0, -1.0, -2.0], [2.0, 0.0, 1.0]])
    u = torch.tensor([[2.0, 2.0, 3.0], [2.0, 1.0, 2.0]])
    a, b = qt.plane(l, u)
    assert abs(float(a[0] @ qt.x_hat + b[0]) - qt.f) < 1e-6
    gen = torch.Generator().manual_seed(12)
    for k in range(2):
        points = l[k] + (u[k] - l[k]) * torch.rand(500, 3, generator=gen)
        assert torch.all(points @ a[k] + b[k] <= m(points)[:, 0] + 1e-12)
    qt.polish(torch.tensor([9.0, 0.3, -0.5]))
    assert qt.polished and qt.x_hat[0] == 2.0
    qt.update(torch.tensor([-9.0, 0.0, 0.0]))
    assert qt.x_hat[0] == 2.0
    for endpoint in (l, u):
        bad = endpoint.clone()
        bad[1, 0] = 2.1
        with pytest.raises(ValueError, match='fixed-coordinate subspace'):
            qt.plane(bad if endpoint is l else l, bad if endpoint is u else u)


def test_all_coordinates_fixed_have_zero_curvature_correction():
    m, bm, x = _model(-torch.eye(3, dtype=torch.float64))
    point = torch.tensor([[0.2, 0.4, -0.1]])
    qt = build_quadratic_tangent(bm, m, x, 0, torch.zeros(3),
                                 root_lower=point, root_upper=point)
    assert qt.lam_lb == 0.0
    a, b = qt.plane(point, point)
    assert abs(float(a[0] @ point[0] + b[0]) - qt.f) < 1e-7


@pytest.mark.parametrize('convex,affine', [(True, False), (False, True)])
def test_local_kkt_anchor_does_not_lock_without_convex_affine_problem(convex, affine):
    Q = torch.eye(3, dtype=torch.float64) * (1 if convex else -1)
    m, bm, x = _model(Q, nonlinear_constraints=not affine)
    qt = build_quadratic_tangent(bm, m, x, 0, torch.zeros(3))
    assert qt.linear_constraint_outputs == affine
    qt.polish(torch.zeros(3))
    assert not qt.polished
    offered = torch.tensor([2.0, -2.0, 1.0])
    qt.offer(offered, margin=1e6)
    assert torch.equal(qt.x_hat, offered)


def test_quadratic_dual_bound_on_random_convex_qps():
    from nlp_solver.dual_ascent import lp_bound
    torch.set_default_dtype(torch.float64)
    gen = torch.Generator().manual_seed(24)
    n, m = 4, 3
    for k in range(12):
        R = torch.randn(n, n, generator=gen)
        H = R.T @ R + 0.5 * torch.eye(n)
        l = -1.0 - torch.rand(1, n, generator=gen)
        u = 1.0 + torch.rand(1, n, generator=gen)
        optimum = l[0] + (u[0] - l[0]) * (0.2 + 0.6 * torch.rand(n, generator=gen))
        G = torch.randn(1, m, n, generator=gen)
        y_star = torch.rand(1, m, generator=gen)
        y_star[0, -1] = 0.0
        box_gradient = torch.zeros(n)
        # Include minima on both lower and upper box faces.
        if k % 3 == 1:
            optimum[0] = l[0, 0]
            box_gradient[0] = 0.7
        elif k % 3 == 2:
            optimum[0] = u[0, 0]
            box_gradient[0] = -0.7
        h = (G @ optimum).reshape(1, m)
        h[0, -1] += 0.5
        c = -H @ optimum - G[0].T @ y_star[0] + box_gradient

        def objective(z):
            return 0.5 * z @ H @ z + c @ z + 0.3

        # By construction, optimum is feasible and satisfies the convex QP KKT
        # conditions with y_star and the signed box multipliers above.
        true_minimum = objective(optimum)
        qt = QuadraticTangent(objective, n, torch.randn(n, generator=gen))
        a, b = qt.plane(l, u)
        plane_lb, y = lp_bound(a, b, G, h, l, u, pdhg_iters=50, exact_max=1)
        quadratic_lb = qt.quadratic_dual_bound(l, u, G, h, y)
        assert quadratic_lb <= true_minimum + 1e-10
        assert quadratic_lb >= plane_lb - 1e-10
        arbitrary_y = 10.0 * torch.rand(1, m, generator=gen)
        assert qt.quadratic_dual_bound(l, u, G, h, arbitrary_y) <= true_minimum + 1e-10


def _exact_model_value(qt, l, u, G, h, y):
    """Exact (rational) box minimum of the quadratic model plus y^T (G x - h), the larger of the value at y and at 0."""
    from fractions import Fraction as F
    a, b = qt.plane(l.double(), u.double())
    m, n = h.shape[-1], l.shape[-1]
    lam, xh = F(qt.lam_lb), [F(float(v)) for v in qt.x_hat]

    def exact(y):
        total = F(float(b[0])) - sum(F(float(h[0, i])) * F(float(y[0, i])) for i in range(m))
        for j in range(n):
            q = F(float(a[0, j])) + sum(F(float(G[0, i, j])) * F(float(y[0, i])) for i in range(m))
            p = min(max(xh[j] - q / lam, F(float(l[0, j]))), F(float(u[0, j])))
            total += q * p + lam / 2 * (p - xh[j]) ** 2
        return total

    return max(exact(y), exact(torch.zeros_like(y)))


def test_quadratic_dual_bound_is_rigorous_for_the_model():
    """Never above the exact (rational) box minimum of the quadratic model plus y^T (G x - h) at y or at 0, and
    within 1e-9 of the larger of the two."""
    from fractions import Fraction as F
    torch.set_default_dtype(torch.float64)
    gen = torch.Generator().manual_seed(7)
    n, m = 5, 4
    worst = 0.0
    for k in range(20):
        R = torch.randn(n, n, generator=gen)
        H = R.T @ R * 10.0 ** torch.randint(-3, 4, (1,), generator=gen).item() + 0.1 * torch.eye(n)
        c = torch.randn(n, generator=gen)
        l, u = -torch.rand(1, n, generator=gen) * 3, torch.rand(1, n, generator=gen) * 3
        G = torch.randn(1, m, n, generator=gen) * 10.0 ** torch.randint(-2, 3, (1, m, 1), generator=gen)
        h = torch.randn(1, m, generator=gen)
        y = torch.rand(1, m, generator=gen) * (torch.rand(1, m, generator=gen) < 0.7) * 10.0 ** (3 * torch.rand(1))
        qt = QuadraticTangent(lambda z: 0.5 * z @ H @ z + c @ z, n, torch.randn(n, generator=gen))
        best = _exact_model_value(qt, l, u, G, h, y)
        bound = F(float(qt.quadratic_dual_bound(l, u, G, h, y)[0]))
        assert bound <= best
        worst = max(worst, float(best - bound) / (1.0 + abs(float(best))))
    assert worst < 1e-9, worst


def test_quadratic_dual_bound_of_float32_data_is_rounded_toward_minus_inf():
    """float32 l, u, G, h, y: the returned float32 bound is not above the exact model value of those float32 numbers
    (a cast to nearest exceeded it on about half of these)."""
    from fractions import Fraction as F
    gen = torch.Generator().manual_seed(1)
    n, m = 5, 4
    for k in range(60):
        R = torch.randn(n, n, generator=gen, dtype=torch.float64)
        H = R.T @ R + 0.1 * torch.eye(n, dtype=torch.float64)
        c = torch.randn(n, generator=gen, dtype=torch.float64)
        qt = QuadraticTangent(lambda z: 0.5 * z @ H @ z + c @ z, n, torch.randn(n, generator=gen, dtype=torch.float64))
        f32 = dict(generator=gen, dtype=torch.float32)
        l, u = -torch.rand(1, n, **f32) * 3, torch.rand(1, n, **f32) * 3
        G, h, y = torch.randn(1, m, n, **f32), torch.randn(1, m, **f32), torch.rand(1, m, **f32)
        bound = qt.quadratic_dual_bound(l, u, G, h, y)
        assert bound.dtype == torch.float32
        assert F(float(bound[0])) <= _exact_model_value(qt, l, u, G, h, y)


def test_quadratic_dual_bound_without_rows_and_a_large_rounding_term():
    """No rows and a rounding term above tol/10 (the model value is about 0 while b and a x are about 1e10): the
    evaluator's double-double step has no entry to recompute and the bound is still returned."""
    from fractions import Fraction as F
    torch.set_default_dtype(torch.float64)
    qt = QuadraticTangent(lambda z: (z ** 2).sum() + 45.0, 1, torch.tensor([1e5]))
    l, u = -torch.ones(1, 1), torch.ones(1, 1)
    G, h = torch.zeros(1, 0, 1), torch.zeros(1, 0)
    bound = qt.quadratic_dual_bound(l, u, G, h, h)
    best = _exact_model_value(qt, l, u, G, h, h)
    assert F(float(bound[0])) <= best and float(best - F(float(bound[0]))) < 1e-3, (bound, float(best))


def test_quadratic_dual_bound_zero_multipliers_and_fixed_coordinate():
    torch.set_default_dtype(torch.float64)
    l, u = torch.tensor([[2.0, -3.0]]), torch.tensor([[2.0, 3.0]])
    qt = QuadraticTangent(lambda z: -5 * z[0] ** 2 + z[1] ** 2, 2,
                         torch.tensor([9.0, 1.0]), l, u)
    # A poor multiplier must not erase the unconstrained box bound (-20).
    G, h = torch.tensor([[[0.0, 1.0]]]), torch.tensor([[2.0]])
    bound = qt.quadratic_dual_bound(l, u, G, h, torch.tensor([[100.0]]))
    assert -20.000001 < bound <= -20.0
    empty_bound = qt.quadratic_dual_bound(l, u, G[:, :0], h[:, :0], h[:, :0])
    torch.testing.assert_close(bound, empty_bound)
    qt.lam_lb = 0.0
    assert qt.quadratic_dual_bound(l, u, G, h, h) is None
    qt.lam_lb = -1.0
    assert qt.quadratic_dual_bound(l, u, G, h, h) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_degrees_and_plane_on_cuda():
    # Graph constants live on the model's device; the degree proof must not
    # fail (and silently disable the bound) when that device is not the CPU.
    torch.set_default_dtype(torch.float64)
    m = QuadModel(torch.eye(3, dtype=torch.float64)).cuda()
    x = torch.zeros(1, 3, dtype=torch.float64, device='cuda')
    bm = BoundedModule(m, x, device='cuda')
    d = output_degrees(bm, x)
    assert d.tolist() == [2.0, 1.0, float('inf'), 3.0]
    x_hat = torch.tensor([0.2, 0.4, -0.1], dtype=torch.float64, device='cuda')
    qt = build_quadratic_tangent(bm, m, x, 0, x_hat)
    assert qt is not None
    a, b = qt.plane(x_hat.view(1, -1) - 1, x_hat.view(1, -1) + 1)
    assert a.device.type == 'cuda'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_host_incumbent_is_offered_on_the_tangent_device():
    # history.alm_best_primal_x lives on the host; offering it as is raised a device
    # error inside the dual-bound block, which then skipped dual ascent, the LP and the
    # tangent bound on every later iteration.
    from input_split.bounding import incumbent_for_tangent
    torch.set_default_dtype(torch.float64)
    m = QuadModel(torch.eye(3, dtype=torch.float64), nonlinear_constraints=False).cuda()
    x = torch.zeros(1, 3, dtype=torch.float64, device='cuda')
    bm = BoundedModule(m, x, device='cuda')
    x_kkt = torch.tensor([0.2, 0.4, -0.1], dtype=torch.float64, device='cuda')
    qt = build_quadratic_tangent(bm, m, x, 0, x_kkt)
    qt.polish(x_kkt)
    host_incumbent = torch.tensor([[-0.5, 1.0, -1.0]], dtype=torch.float32)
    with pytest.raises(RuntimeError):
        qt.offer(host_incumbent, margin=1e-5)        # the old call
    x_inc = incumbent_for_tangent(host_incumbent, qt)
    assert x_inc.device == qt.x_hat.device and x_inc.dtype == qt.x_hat.dtype
    qt.offer(x_inc, margin=1e-5)
    assert torch.equal(qt.x_hat, x_inc)
    assert incumbent_for_tangent(torch.zeros(4), qt) is None
    assert incumbent_for_tangent(None, qt) is None


def test_polished_tangent_moves_only_for_a_clearly_better_point():
    # A polished tangent (the minimizer of the relaxed problem) must not be replaced by
    # an incumbent that is no better: that point is generally not a KKT point, and its
    # tangent plane is loose over the feasible set.
    m, bm, x = _model(torch.eye(3, dtype=torch.float64), nonlinear_constraints=False)
    x_kkt = torch.tensor([0.2, 0.4, -0.1], dtype=torch.float64)
    qt = build_quadratic_tangent(bm, m, x, 0, x_kkt)
    qt.polish(x_kkt)
    f_kkt = qt.f
    worse = torch.tensor([0.3, 0.5, 0.0], dtype=torch.float64)
    assert float(m(worse.view(1, -1))[0, 0]) > f_kkt
    qt.offer(worse, margin=1e-5)
    assert torch.equal(qt.x_hat, x_kkt)
    better = torch.tensor([-0.5, 1.0, -1.0], dtype=torch.float64)
    assert float(m(better.view(1, -1))[0, 0]) < f_kkt - 1e-5
    qt.offer(better, margin=1e-5)
    assert torch.equal(qt.x_hat, better)
    assert not qt.polished
    # Unpolished tangents follow every new incumbent, as before.
    qt.offer(worse, margin=1e-5)
    assert torch.equal(qt.x_hat, worse)
