"""compute_qtb_tangent_point: the KKT point of the tolerance-relaxed root problem."""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from input_split.nlp_in_bab import compute_qtb_tangent_point  # noqa: E402
from nlp_solver.ipopt_solver import IPOPT_AVAILABLE  # noqa: E402


class ConvexQP(nn.Module):
    """[(x0 - 1)^2 + (x1 - 1)^2, x0 + x1 - 1]: min obj s.t. row 1 <= 0."""

    def forward(self, x):
        obj = ((x - 1.0) ** 2).sum(-1, keepdim=True)
        return torch.cat([obj, x.sum(-1, keepdim=True) - 1.0], dim=-1)


@pytest.mark.skipif(not IPOPT_AVAILABLE, reason='needs cyipopt')
def test_tangent_point_is_the_relaxed_minimizer():
    tol = 1e-4
    dm_l = torch.full((1, 2), -2.0, dtype=torch.float64)
    dm_u = torch.full((1, 2), 2.0, dtype=torch.float64)
    c = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    rhs = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    cfg = {'constr_viol_tol': tol, 'ipopt_constr_viol_tol': tol, 'ipopt_timeout': 10.0,
           'ipopt_tol': 1e-10, 'ipopt_hessian_approximation': 'limited-memory'}
    x, accepted = compute_qtb_tangent_point(ConvexQP(), dm_l, dm_u, c, rhs, None, cfg,
                                            x0=torch.zeros(1, 2, dtype=torch.float64))
    assert x is not None
    # The dual LP sees x0 + x1 <= 1 + tol, whose minimizer is 0.5 + tol / 2 per axis;
    # the exact problem's minimizer (0.5, 0.5) would leave its plane tol/2 * 2 loose.
    assert torch.allclose(x.reshape(-1), torch.full((2,), 0.5 + tol / 2, dtype=torch.float64),
                          atol=1e-7)


@pytest.mark.skipif(not IPOPT_AVAILABLE, reason='needs cyipopt')
def test_failed_solve_gives_no_tangent_point():
    # An objective Ipopt cannot evaluate: the start point comes back unchanged and
    # must not be passed off as a KKT point.
    dm_l = torch.full((1, 2), -2.0, dtype=torch.float64)
    dm_u = torch.full((1, 2), 2.0, dtype=torch.float64)
    c = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    rhs = torch.tensor([[float('inf'), 0.0]], dtype=torch.float64)
    cfg = {'constr_viol_tol': 1e-4, 'ipopt_timeout': 10.0}
    assert compute_qtb_tangent_point(ConvexQP(), dm_l, dm_u, c, rhs, None, cfg,
                                     x0=torch.zeros(1, 2, dtype=torch.float64)) == (None, None)


@pytest.mark.skipif(not IPOPT_AVAILABLE, reason='needs cyipopt')
def test_tangent_solve_also_yields_an_accepted_incumbent():
    # The relaxed KKT point itself may sit at g = tol (+ Ipopt's tolerance), outside
    # acceptance; the finished point must pass the primal's own check and be optimal.
    # pooling_bental5tp: the root Ipopt stalled at -2900 while this solve reached -3500.
    tol = 1e-4
    dm_l = torch.full((1, 2), -2.0, dtype=torch.float64)
    dm_u = torch.full((1, 2), 2.0, dtype=torch.float64)
    c = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    rhs = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    cfg = {'constr_viol_tol': tol, 'ipopt_constr_viol_tol': tol, 'ipopt_timeout': 10.0,
           'ipopt_constraint_slack_frac': 0.5}
    _, accepted = compute_qtb_tangent_point(ConvexQP(), dm_l, dm_u, c, rhs, None, cfg,
                                            x0=torch.full((1, 2), -1.5, dtype=torch.float64))
    assert accepted is not None
    x_acc, obj = accepted
    assert float(x_acc.sum()) - 1.0 <= tol
    assert abs(obj - 0.5) < 1e-4


@pytest.mark.skipif(not IPOPT_AVAILABLE, reason='needs cyipopt')
def test_blend_accepts_a_verified_point_near_the_kkt_point():
    # With a finishing solve that cannot use the whole tolerance (slack 0), the blend
    # towards the feasible warm start keeps a verified point within ~tol of the KKT one.
    tol = 1e-4
    dm_l = torch.full((1, 2), -2.0, dtype=torch.float64)
    dm_u = torch.full((1, 2), 2.0, dtype=torch.float64)
    c = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    rhs = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    cfg = {'constr_viol_tol': tol, 'ipopt_constr_viol_tol': tol, 'ipopt_timeout': 10.0,
           'ipopt_constraint_slack_frac': 0.0, 'qtb_accept_blend': True}
    x_kkt, accepted = compute_qtb_tangent_point(
        ConvexQP(), dm_l, dm_u, c, rhs, None, cfg,
        x0=torch.full((1, 2), -1.5, dtype=torch.float64))
    assert x_kkt is not None and accepted is not None
    x_acc, obj = accepted
    assert float(x_acc.sum()) - 1.0 <= tol                      # verified feasible
    assert bool(((x_acc >= dm_l.reshape(-1)) & (x_acc <= dm_u.reshape(-1))).all())
    f_kkt = float(((x_kkt - 1.0) ** 2).sum())
    assert obj <= f_kkt + 5 * tol
