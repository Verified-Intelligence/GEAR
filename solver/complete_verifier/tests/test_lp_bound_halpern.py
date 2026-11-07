"""exact_max < 0 runs the plane LP of each box as a restarted-Halpern lane: valid, near the LP optimum, no HiGHS."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from nlp_solver.dual_ascent import lp_bound  # noqa: E402
from test_exact_lp_infeasible import _batch  # noqa: E402
from test_lp_bound_pdhg import _exact, _random_lp  # noqa: E402


def test_halpern_lane_is_valid_and_reaches_the_lp():
    c, c0, G, h, l, u = _random_lp(13, B=6, m=12, n=8)
    exact = _exact(c, c0, G, h, l, u)
    bound, y = lp_bound(c, c0, G, h, l, u, pdhg_iters=500, halpern=True)
    assert torch.all(bound <= exact + 1e-9 * (1 + exact.abs())) and torch.all(y >= 0)
    assert torch.all(exact - bound <= 1e-4 * (1 + exact.abs()))


def test_halpern_lane_returns_inf_only_for_empty_boxes_and_honours_the_cutoff():
    c, c0, G, h, l, u = _batch([-0.5, 0.0, 0.3, -1e-3])
    bound, _ = lp_bound(c, c0, G, h, l, u, pdhg_iters=500, halpern=True)
    assert torch.isinf(bound[0]) and bound[0] > 0 and torch.isinf(bound[3]) and bound[3] > 0
    assert torch.isfinite(bound[1:3]).all() and bound[1] <= 1e-9 and bound[2] <= -0.3 + 1e-9
    bound0, _ = lp_bound(c, c0, G, h, l, u, pdhg_iters=500, halpern=True, stop=torch.full((4,), -1e9, dtype=c.dtype))
    assert torch.all(bound0[1:3] <= bound[1:3] + 1e-9)


def test_flag_off_still_uses_pdhg():
    c, c0, G, h, l, u = _random_lp(5, B=3, m=7, n=5)
    a, _ = lp_bound(c, c0, G, h, l, u, pdhg_iters=50)
    b, _ = lp_bound(c, c0, G, h, l, u, pdhg_iters=50, exact_max=0)
    assert torch.equal(a, b)
