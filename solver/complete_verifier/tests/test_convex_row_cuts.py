"""Global validity, degree/sign gates, and shared-domain propagation of row cuts."""
import os
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import arguments
from auto_LiRPA import BoundedModule
from nlp_solver.convex_row_cuts import build_convex_row_cuts
from nlp_solver.structure import RowPool


@pytest.fixture(autouse=True)
def config(monkeypatch):
    torch.set_default_dtype(torch.float64)
    cfg = arguments.ConfigHandler()
    cfg.construct_config_dict(cfg.default_args)
    cfg['solving']['solving_mode'] = True
    cfg['solver']['nlp']['constr_viol_tol'] = 1e-5
    monkeypatch.setattr(arguments, 'Config', cfg)
    return cfg


class Rows(torch.nn.Module):
    def __init__(self, H, linear, bias):
        super().__init__()
        self.register_buffer('H', H)
        self.register_buffer('linear', linear)
        self.register_buffer('bias', bias)

    def forward(self, x):
        rows = 0.5 * ((x @ self.H) * x).sum(-1, keepdim=True)
        rows = rows + (x * self.linear).sum(-1, keepdim=True) + self.bias
        return torch.cat((x.sum(-1, keepdim=True), rows), dim=1)


def build(model, lower, upper, x_hat, rhs, c=None, **kwargs):
    centre = (lower + upper) / 2
    bounded = BoundedModule(model, centre[:1], device='cpu')
    if c is None:
        c = torch.eye(rhs.shape[-1]).unsqueeze(0).expand(rhs.shape[0], -1, -1)
    return build_convex_row_cuts(bounded, model, centre, c, rhs, lower, upper,
                                x_hat, enabled=True, rowpool=RowPool(rhs), allow_uncertified=True, **kwargs)


@pytest.mark.parametrize('outside', [False, True])
def test_random_psd_rows_preserve_every_sampled_feasible_point(outside):
    gen = torch.Generator().manual_seed(14)
    for k in range(12):
        # Include rank-deficient PSD rows whose conservative eigenvalue lower
        # bound is below -delta solely because of the rounding allowance.
        R = torch.randn(3 if k % 2 else 5, 5, generator=gen)
        model = Rows(R.T @ R, torch.randn(5, generator=gen), torch.randn((), generator=gen))
        lower, upper = -2 * torch.ones(1, 5), 3 * torch.ones(1, 5)
        x_hat = lower[0] + (upper[0] - lower[0]) * torch.rand(5, generator=gen)
        if outside:
            x_hat += 20
        samples = lower + (upper - lower) * torch.rand(2000, 5, generator=gen)
        values = model(samples)[:, 1]
        threshold = values.quantile(0.4)
        rhs = torch.tensor([[0., threshold]])
        manager = build(model, lower, upper, x_hat, rhs)
        assert len(manager.tangents) == 1
        A, b = manager.offer(x_hat, 10.)
        feasible = samples[values <= threshold + 1e-5]
        assert feasible.shape[0] >= 800
        assert (feasible @ A[0].T + b[0] <= 0).all()
        # The affine cut lies below the row everywhere, regardless of feasibility.
        assert (samples @ A[0, 0] + b[0, 0] <= values - threshold - 1e-5).all()


def test_tiny_negative_curvature_uses_root_distance_even_with_outside_anchor():
    model = Rows(torch.diag(torch.tensor([-5e-10, 2.])), torch.zeros(2), torch.tensor(0.))
    lower, upper = torch.tensor([[-100., -2.]]), torch.tensor([[100., 2.]])
    anchor = torch.tensor([1000., 0.])
    manager = build(model, lower, upper, anchor, torch.zeros(1, 2))
    assert manager is not None and manager.tangents[0].lam_lb < 0
    A, b = manager.offer(anchor, 1.)
    points = torch.tensor([[-100., 0.], [100., 0.], [0., 0.]])
    assert (points @ A[0].T + b[0] <= 0).all()
    # A correction from a descendant's small box would not be globally valid.
    t = manager.tangents[0]
    assert b[0, 0] <= t.f - (t.g * anchor).sum() + 0.5 * t.lam_lb * 1100 ** 2


def test_nonconvex_row_is_rejected():
    model = Rows(torch.diag(torch.tensor([-1., 2.])), torch.zeros(2), torch.tensor(0.))
    assert build(model, -torch.ones(1, 2), torch.ones(1, 2), torch.zeros(2),
                 torch.zeros(1, 2)) is None


def test_signed_equality_only_admits_convex_side_and_uses_threshold_tolerance():
    model = Rows(torch.eye(2) * 2, torch.zeros(2), torch.tensor(0.))
    # x^2 = 1: signed specs -g <= -1, g <= 1. Only the latter is convex.
    c = torch.tensor([[[1., 0.], [0., -1.], [0., 1.]]])
    rhs = torch.tensor([[0., -1., 1.]])
    anchor = torch.tensor([1., 0.])
    manager = build(model, -torch.ones(1, 2) * 2, torch.ones(1, 2) * 2, anchor, rhs, c)
    assert len(manager.tangents) == 1
    A, b = manager.offer(anchor, 2.)
    # This point exceeds the exact threshold but belongs to the relaxed row.
    relaxed_point = torch.tensor([[(1. + 1e-5) ** 0.5, 0.]])
    assert (relaxed_point @ A[0].T + b[0] <= 0).all()
    assert abs(float(b[0, 0]) + 2. + 1e-5) < 1e-7


def test_degree_is_proved_for_all_outputs_used_by_a_row():
    class Nonquadratic(torch.nn.Module):
        def forward(self, x):
            return torch.cat((x[:, :1], x[:, :1] ** 2, x[:, :1] ** 3,
                              torch.exp(x[:, :1])), dim=1)

    c = torch.tensor([[[1., 0., 0., 0.], [0., 1., 1., 0.], [0., 1., 0., 1.]]])
    assert build(Nonquadratic(), -torch.ones(1, 1), torch.ones(1, 1), torch.zeros(1),
                 torch.zeros(1, 3), c) is None


def test_option_defaults_off_and_does_no_graph_or_hessian_work(config):
    assert config['solver']['nlp']['convex_row_cuts'] is False
    with patch('nlp_solver.convex_row_cuts.output_degrees', side_effect=AssertionError):
        assert build_convex_row_cuts(None, None, torch.zeros(1, 2), torch.eye(2)[None],
                                     torch.zeros(1, 2), -torch.ones(1, 2), torch.ones(1, 2),
                                     torch.zeros(2), rowpool=RowPool(torch.zeros(1, 2))) is None


def test_missing_anchor_nonfinite_root_and_different_specs_are_skipped():
    model = Rows(torch.eye(2), torch.zeros(2), torch.tensor(0.))
    lower, upper = -torch.ones(2, 2), torch.ones(2, 2)
    rhs = torch.zeros(2, 2)
    assert build(model, lower, upper, None, rhs) is None
    bad_upper = upper.clone()
    bad_upper[0, 0] = torch.inf
    assert build(model, lower, bad_upper, torch.zeros(2), rhs) is None
    rhs[1, 1] = 1
    assert build(model, lower, upper, torch.zeros(2), rhs) is None


def test_improvement_rounds_preserve_old_rows_and_reach_queued_domains():
    from input_split.batch_branch_and_bound import _offer_convex_row_cuts
    from input_split.branching_domains import UnsortedInputDomainList
    from history import SplitHistory

    model = Rows(torch.eye(2) * 2, torch.zeros(2), torch.tensor(0.))
    lower, upper = -torch.ones(2, 2), torch.ones(2, 2)
    manager = build(model, lower, upper, torch.zeros(2), torch.ones(2, 2))
    history = SplitHistory({})
    history.set_usage_flags(False, False)
    history.convex_row_cuts = manager
    domains = UnsortedInputDomainList(1, 'cpu', history=history, use_split_idx=False)
    old_base = (torch.tensor([[[1., 0.]]]).expand(2, -1, -1),
                torch.tensor([[-1.]]).expand(2, -1))
    c = torch.eye(2)[None].expand(2, -1, -1)
    domains.add(-torch.ones(2, 2), lower, upper, c, torch.ones(2, 2),
                base_constraints=old_base, check_dm_lbs=False, check_input_boxes=False)
    base, constraints = old_base, old_base
    for k in range(6):
        base, constraints = _offer_convex_row_cuts(
            domains, base, constraints, torch.tensor([k / 10., 0.]), 10. - k, 2)
        assert base[0].shape[1] == 1 + min(k + 1, 4)
        torch.testing.assert_close(base[0][:, :1], old_base[0])
    assert manager.rounds == 4
    picked = domains.pick_out_batch(2, device='cpu')
    torch.testing.assert_close(picked[6][0], base[0])
    torch.testing.assert_close(picked[6][1], base[1])
    assert manager.offer(torch.zeros(2), 0.) is None


def test_same_or_worse_incumbent_does_not_consume_a_round():
    model = Rows(torch.eye(2), torch.zeros(2), torch.tensor(0.))
    manager = build(model, -torch.ones(1, 2), torch.ones(1, 2), torch.zeros(2),
                    torch.zeros(1, 2))
    assert manager.offer(torch.zeros(2), torch.inf) is None
    assert manager.offer(torch.zeros(2), 10.) is not None
    assert manager.offer(torch.ones(2), 10.) is None
    assert manager.offer(torch.ones(2), 11.) is None
    assert manager.offer(torch.ones(2), 9.) is not None
    assert manager.rounds == 2


def test_affine_rows_get_no_cut():
    # A linear row's tangent is the row itself; duplicating it as a base row only adds
    # constrained-concretization work (cute_gouldqp2/3).
    import torch
    from auto_LiRPA import BoundedModule
    from nlp_solver.convex_row_cuts import build_convex_row_cuts

    class Lin(torch.nn.Module):
        def forward(self, x):
            return torch.cat([x[:, :1] ** 2 + x[:, 1:] ** 2, x[:, :1] + 2 * x[:, 1:] - 1], dim=1)

    x = torch.zeros(1, 2, dtype=torch.float64)
    model = Lin()
    bm = BoundedModule(model, x, device='cpu')
    c = torch.tensor([[[1., 0.], [0., 1.]]], dtype=torch.float64)
    rhs = torch.tensor([[float('inf'), 0.]], dtype=torch.float64)
    lo, hi = -torch.ones(1, 2, dtype=torch.float64), torch.ones(1, 2, dtype=torch.float64)
    assert build_convex_row_cuts(bm, model, x, c, rhs, lo, hi, torch.zeros(2, dtype=torch.float64),
                                 enabled=True, rowpool=RowPool(rhs), allow_uncertified=True) is None


class ThreeConvexRows(torch.nn.Module):
    def forward(self, x):
        return torch.cat((x.sum(-1, keepdim=True), (x * x).sum(-1, keepdim=True),
                          x[:, :1] * x[:, :1], x[:, 1:2] * x[:, 1:2]), dim=1)


def test_more_quadratic_rows_than_the_cap_get_no_cuts():
    lower, upper = -torch.ones(1, 2), torch.ones(1, 2)
    rhs = torch.tensor([[0., 1., 0.5, 0.5]])
    x_hat = torch.tensor([0.3, -0.2])
    assert build(ThreeConvexRows(), lower, upper, x_hat, rhs, max_rows=2) is None
    assert len(build(ThreeConvexRows(), lower, upper, x_hat, rhs, max_rows=3).tangents) == 3


class ShiftedRow(torch.nn.Module):
    """Row (z0 - a)^2 + z1^2 written expanded, with catastrophic cancellation near z0 = a (TH1's fuzz graph)."""

    def __init__(self, a):
        super().__init__()
        self.register_buffer('a', torch.tensor(float(a), dtype=torch.float64))

    def forward(self, z):
        row = z[:, :1] * z[:, :1] - 2 * self.a * z[:, :1] + self.a * self.a + z[:, 1:2] * z[:, 1:2]
        return torch.cat((z[:, :1], row), dim=1)


def test_uncertified_row_cuts_are_never_built_on_cancelling_graphs():
    # build_convex_row_cuts has no rounding certificate for f, grad f or the curvature: it builds nothing
    # (convex_tangent_rows is the certified path), whatever the incumbent.
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        a = 10 ** float(torch.rand(1, generator=g) * 8)
        d = 10 ** float(-torch.rand(1, generator=g) * 8)
        lower = torch.tensor([[a - 1., -1.]], dtype=torch.float64)
        upper = torch.tensor([[a + 1., 1.]], dtype=torch.float64)
        x_hat = torch.tensor([a + d, 0.], dtype=torch.float64)
        model = ShiftedRow(a).double()
        centre = (lower + upper) / 2
        rhs = torch.tensor([[0., 0.5]], dtype=torch.float64)
        c = torch.eye(2, dtype=torch.float64).unsqueeze(0)
        bounded = BoundedModule(model, centre[:1], device='cpu')
        assert build_convex_row_cuts(bounded, model, centre, c, rhs, lower, upper, x_hat,
                                     enabled=True, rowpool=RowPool(rhs)) is None
