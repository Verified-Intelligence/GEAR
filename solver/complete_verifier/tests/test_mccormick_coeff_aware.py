"""Soundness check for coeff-aware McCormick plane selection in BoundMul.

Verifies that for a variety of random and structured boxes, the
coefficient-aware lower plane is <= x*y for all sample points in the box, and
the upper plane is >= x*y. Also exercises positive, negative, and crossing
boxes explicitly.
"""
import torch

from auto_LiRPA.operators.bivariate import BoundMul, MulHelper


def _build_planes(x_l, x_u, y_l, y_u):
    options = {'mul': {'coeff_aware': True}}
    bm = BoundMul.__new__(BoundMul)
    # Bypass the heavy __init__; we only need a few attributes.
    bm.coeff_aware = True
    bm.middle = False
    bm.adaptive = False
    bm.mul_helper = MulHelper()
    bm._start = None
    bm.opt_stage = None
    r_l, r_u = bm._compute_coeff_aware_r(x_l, x_u, y_l, y_u)
    (alpha_l, beta_l, gamma_l, alpha_u, beta_u,
     gamma_u) = MulHelper.interpolated_relaxation(
         x_l, x_u, y_l, y_u, r_l, r_u)
    return (alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u, r_l, r_u)


def _check(x_l, x_u, y_l, y_u, n_samples=20, atol=1e-5):
    (alpha_l, beta_l, gamma_l,
     alpha_u, beta_u, gamma_u, _, _) = _build_planes(x_l, x_u, y_l, y_u)
    # Sample points uniformly inside the box.
    shape = torch.broadcast_shapes(x_l.shape, y_l.shape)
    x_l_b = x_l.expand(shape)
    x_u_b = x_u.expand(shape)
    y_l_b = y_l.expand(shape)
    y_u_b = y_u.expand(shape)
    for _ in range(n_samples):
        tx = torch.rand(shape)
        ty = torch.rand(shape)
        x = x_l_b + tx * (x_u_b - x_l_b)
        y = y_l_b + ty * (y_u_b - y_l_b)
        xy = x * y
        lower = alpha_l * x + beta_l * y + gamma_l
        upper = alpha_u * x + beta_u * y + gamma_u
        assert torch.all(lower <= xy + atol), (
            f"Lower plane violates xy on some point: max viol "
            f"{(lower - xy).max().item()}")
        assert torch.all(upper >= xy - atol), (
            f"Upper plane violates xy on some point: max viol "
            f"{(xy - upper).max().item()}")
    # Also check the corners exactly.
    for cx in (x_l_b, x_u_b):
        for cy in (y_l_b, y_u_b):
            xy = cx * cy
            lower = alpha_l * cx + beta_l * cy + gamma_l
            upper = alpha_u * cx + beta_u * cy + gamma_u
            assert torch.all(lower <= xy + atol)
            assert torch.all(upper >= xy - atol)


def test_positive_box():
    x_l = torch.tensor([1.0]); x_u = torch.tensor([3.0])
    y_l = torch.tensor([1.0]); y_u = torch.tensor([4.0])
    _check(x_l, x_u, y_l, y_u)


def test_negative_box():
    x_l = torch.tensor([-3.0]); x_u = torch.tensor([-1.0])
    y_l = torch.tensor([-4.0]); y_u = torch.tensor([-1.0])
    _check(x_l, x_u, y_l, y_u)


def test_crossing_box_x():
    x_l = torch.tensor([-2.0]); x_u = torch.tensor([3.0])
    y_l = torch.tensor([1.0]); y_u = torch.tensor([4.0])
    _check(x_l, x_u, y_l, y_u)


def test_crossing_box_y():
    x_l = torch.tensor([1.0]); x_u = torch.tensor([3.0])
    y_l = torch.tensor([-2.0]); y_u = torch.tensor([4.0])
    _check(x_l, x_u, y_l, y_u)


def test_both_crossing_box():
    x_l = torch.tensor([-2.0]); x_u = torch.tensor([3.0])
    y_l = torch.tensor([-1.0]); y_u = torch.tensor([2.0])
    _check(x_l, x_u, y_l, y_u)


def test_random_boxes_batched():
    torch.manual_seed(0)
    n = 200
    x_c = torch.randn(n) * 2.0
    y_c = torch.randn(n) * 2.0
    x_r = torch.rand(n) * 3.0 + 0.05
    y_r = torch.rand(n) * 3.0 + 0.05
    x_l = x_c - x_r
    x_u = x_c + x_r
    y_l = y_c - y_r
    y_u = y_c + y_r
    _check(x_l, x_u, y_l, y_u, n_samples=50)


def test_broadcast_shapes():
    # x has shape (3,1), y has shape (1,4). r_l, r_u should broadcast to (3,4)
    x_l = torch.tensor([[-1.0], [0.5], [-0.2]])
    x_u = torch.tensor([[1.0], [2.0], [0.3]])
    y_l = torch.tensor([[-1.5, 0.0, 0.5, -0.1]])
    y_u = torch.tensor([[1.5, 1.0, 2.0, 0.2]])
    _check(x_l, x_u, y_l, y_u, n_samples=30)


if __name__ == "__main__":
    test_positive_box()
    test_negative_box()
    test_crossing_box_x()
    test_crossing_box_y()
    test_both_crossing_box()
    test_random_boxes_batched()
    test_broadcast_shapes()
    print("All coeff-aware McCormick soundness checks passed.")
