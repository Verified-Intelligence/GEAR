"""Tests for batched Gauss-Newton feasibility restoration (nlp_solver/alm_feas.py)."""

import torch

from nlp_solver.alm import ALMConfig, ALMPGDSolver
from nlp_solver.alm_feas import (
    batched_gauss_newton_restoration,
    gauss_newton_step,
    make_domain_evaluator,
    residual_jacobian,
    row_norms,
)
from nlp_solver.problem import NNProblemSingleDomain

DT = torch.float64


class _CircleProduct(torch.nn.Module):
    """Outputs [x0^2 + x1^2, x0 * x1, x0 + x1]."""

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return torch.stack(
            [x[:, 0].square() + x[:, 1].square(), x[:, 0] * x[:, 1], x[:, 0] + x[:, 1]],
            dim=1,
        )


def _circle_problem(lbs, ubs):
    """x0^2 + x1^2 = 1 (paired slab), x0 * x1 >= 0.3, objective x0 + x1."""
    B = len(lbs)
    c_con = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=DT
    ).expand(B, 3, 3)
    rhs_con = torch.tensor([1.0, -1.0, -0.3], dtype=DT).expand(B, 3)
    return NNProblemSingleDomain(
        model=_CircleProduct(),
        dm_l=torch.tensor(lbs, dtype=DT),
        dm_u=torch.tensor(ubs, dtype=DT),
        C_obj=torch.tensor([[0.0, 0.0, 1.0]], dtype=DT).expand(B, 1, 3),
        rhs_obj=torch.zeros(B, 1, dtype=DT),
        C_constr=c_con,
        rhs_constr=rhs_con,
        device=torch.device("cpu"),
        dtype=DT,
    )


def test_jacobian_matches_autograd_per_candidate():
    problem = _circle_problem([[0.0, 0.0], [-1.0, 0.0]], [[1.0, 1.0], [1.0, 2.0]])
    evaluate = make_domain_evaluator(problem, torch.arange(2))
    z = torch.rand(2, 3, 2, dtype=DT)
    _, g, jac = residual_jacobian(evaluate, z, row_chunk=2)
    for b in range(2):
        for q in range(3):
            def single(zz, b=b):
                return evaluate(zz.view(1, 1, 2).expand(2, 1, 2))[1][b, 0]
            ref = torch.autograd.functional.jacobian(single, z[b, q])
            assert torch.allclose(jac[b, q], ref, atol=1e-12)


def test_batched_restoration_reaches_raw_feasibility_in_every_domain():
    # Two domains with different boxes; both contain part of the feasible arc.
    problem = _circle_problem([[0.0, 0.0], [0.2, 0.1]], [[1.0, 1.0], [2.0, 1.0]])
    evaluate = make_domain_evaluator(problem, torch.arange(2))
    z0 = torch.tensor([[[0.05, 0.9], [0.9, 0.05]], [[0.0, 0.0], [1.0, 1.0]]], dtype=DT)
    z, phi, viol, iters = batched_gauss_newton_restoration(
        evaluate, z0, steps=30, feas_tol=1e-4
    )
    assert iters > 0
    assert torch.all((z >= 0.0) & (z <= 1.0))
    assert torch.all(viol <= 1e-4)
    # Raw re-evaluation agrees with the archived violation and objective.
    phi_re, g_re = evaluate(z)
    assert torch.allclose(torch.relu(g_re).amax(dim=2), viol)
    assert torch.allclose(phi_re, phi)


def test_step_stops_at_the_box_and_moves_the_free_coordinate():
    # One row: 1.5 - x0 - x1 <= 0 on the unit box, seed x0 at its upper bound.
    z = torch.tensor([[1.0, 0.2]], dtype=DT)
    g = torch.tensor([[1.5 - 1.2]], dtype=DT)
    jac = torch.tensor([[[-1.0, -1.0]]], dtype=DT)
    step = gauss_newton_step(z, g, jac, torch.tensor([1e-12], dtype=DT), 0.0, 3)
    assert abs(step[0, 0].item()) < 1e-9
    assert abs((z + step)[0, 1].item() - 0.5) < 1e-6


def test_active_set_grows_to_rows_the_step_would_violate():
    # Violated row x0 >= 0.8; satisfied row x0 + x1 <= 1 becomes violated if x1
    # stays at 0.9.  The growing active set moves x1 as well.
    z = torch.tensor([[0.5, 0.45]], dtype=DT)
    g = torch.tensor([[0.8 - 0.5, 0.5 + 0.45 - 1.0]], dtype=DT)
    jac = torch.tensor([[[-1.0, 0.0], [1.0, 1.0]]], dtype=DT)
    step = gauss_newton_step(z, g, jac, torch.tensor([1e-12], dtype=DT), 0.0, 3)
    new = z + step
    assert new[0, 0].item() >= 0.8 - 1e-8
    assert new[0, 0].item() + new[0, 1].item() <= 1.0 + 1e-8


def test_row_scaling_keeps_small_rows_moving_next_to_huge_rows():
    # Row 0 is violated with an O(1) gradient; row 1 is violated with a 1e7
    # gradient.  Unscaled relative damping (1e-6 * 1e14) would freeze row 0.
    z = torch.tensor([[0.5, 0.5]], dtype=DT)
    g = torch.tensor([[0.3, 1.0e3]], dtype=DT)
    jac = torch.tensor([[[-1.0, 0.0], [0.0, 1.0e7]]], dtype=DT)
    lam = torch.tensor([1e-6], dtype=DT)
    scaled = gauss_newton_step(z, g, jac, lam, 0.0, 3, row_norm=row_norms(jac))
    new = z + scaled
    assert new[0, 0].item() >= 0.8 - 1e-4
    assert abs(new[0, 1].item() - (0.5 - 1e-4)) < 1e-6
    unscaled = gauss_newton_step(z, g, jac, lam, 0.0, 3)
    assert abs(unscaled[0, 0].item()) < 1e-3


def test_solver_hook_restores_infeasible_domains_and_never_accepts_raw_infeasible():
    problem = _circle_problem(
        [[0.0, 0.0], [0.2, 0.1], [0.0, 0.0]],
        [[1.0, 1.0], [2.0, 1.0], [0.3, 0.3]],  # third box: no feasible point
    )
    cfg = ALMConfig(
        outer_max_iter=0,
        inner_steps=0,
        num_restarts=4,
        feas_tol=1e-4,
        batched_gn_steps=30,
        batched_gn_seeds=4,
    )
    torch.manual_seed(0)
    result = ALMPGDSolver(cfg).solve_with_restarts(problem, num_restarts=4)
    assert result.success.tolist() == [True, True, False]
    out = _CircleProduct()(result.x_solution)
    g = torch.stack([out[:, 0] - 1.0, 1.0 - out[:, 0], 0.3 - out[:, 1]], dim=1)
    raw = torch.relu(g).amax(dim=1)
    assert torch.allclose(raw, result.constraint_violation)
    assert torch.all(raw[:2] <= 1e-4)
    assert raw[2] > 1e-4
    lb = problem.dm_l
    ub = problem.dm_u
    assert torch.all((result.x_solution >= lb) & (result.x_solution <= ub))


def test_improve_mode_finds_a_feasible_point_below_the_incumbent():
    # Feasible arc objective x0 + x1 ranges over [sqrt(1.6), sqrt(2)] = [1.265, 1.414].
    problem = _circle_problem([[0.0, 0.0]], [[1.0, 1.0]])
    import math
    c40, s40 = math.cos(math.radians(40.0)), math.sin(math.radians(40.0))
    start = c40 + s40
    cfg = ALMConfig(outer_max_iter=0, inner_steps=0, num_restarts=1, feas_tol=1e-4,
                    batched_gn_steps=40, batched_gn_seeds=2,
                    batched_gn_improve=True, batched_gn_improve_rel=1e-3, verbose=True)
    x0 = torch.tensor([[c40, s40]], dtype=DT)
    plain = ALMPGDSolver(ALMConfig(outer_max_iter=0, inner_steps=0, num_restarts=1,
                                   feas_tol=1e-4, batched_gn_steps=40, batched_gn_seeds=2)
                         ).solve_with_restarts(problem, num_restarts=1, x_init=x0,
                                               incumbent=torch.tensor([1.35], dtype=DT))
    assert plain.success and abs(plain.objective_value - start) < 1e-9  # untouched
    result = ALMPGDSolver(cfg).solve_with_restarts(
        problem, num_restarts=1, x_init=x0, incumbent=torch.tensor([1.35], dtype=DT))
    assert result.success
    assert result.objective_value <= 1.35 - 1e-3 + 1e-9
    assert result.objective_value >= 1.6 ** 0.5 - 1e-3
    assert result.constraint_violation <= 1e-4
    # Without an incumbent the improve flag does nothing to a feasible domain.
    same = ALMPGDSolver(cfg).solve_with_restarts(problem, num_restarts=1, x_init=x0)
    assert abs(same.objective_value - start) < 1e-9


def test_unattainable_level_never_costs_the_feasible_restoration():
    # Objective on the feasible arc is >= 1.265; an incumbent of 1.0 cannot be beaten.
    problem = _circle_problem([[0.0, 0.0], [0.2, 0.1]], [[1.0, 1.0], [2.0, 1.0]])
    x0 = torch.tensor([[0.05, 0.9], [1.9, 0.15]], dtype=DT)  # infeasible starts
    cfg = dict(outer_max_iter=0, inner_steps=0, num_restarts=1, feas_tol=1e-4,
               batched_gn_steps=30, batched_gn_seeds=2)
    inc = torch.tensor([1.0], dtype=DT)
    plain = ALMPGDSolver(ALMConfig(**cfg)).solve_with_restarts(
        problem, num_restarts=1, x_init=x0, incumbent=inc)
    level = ALMPGDSolver(ALMConfig(batched_gn_improve=True, **cfg)).solve_with_restarts(
        problem, num_restarts=1, x_init=x0, incumbent=inc)
    assert plain.success.all() and level.success.all()
    assert torch.all(level.objective_value <= plain.objective_value + 1e-12)


class _SqrtRow(torch.nn.Module):
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return torch.stack([torch.sqrt(x[:, 0]) + x[:, 1], x[:, 1]], dim=1)


def test_interior_margin_moves_off_a_singular_bound():
    # Row: 0.5 - sqrt(x0) - x1 <= 0 on [0,1]^2 with the seed at x0 = 0, x1 = 0:
    # d sqrt(x0)/dx0 is infinite on the bound, so the plain step cannot move x0.
    problem = NNProblemSingleDomain(
        model=_SqrtRow(), dm_l=torch.zeros(2, dtype=DT), dm_u=torch.ones(2, dtype=DT),
        C_obj=torch.tensor([[0.0, 1.0]], dtype=DT), rhs_obj=torch.zeros(1, dtype=DT),
        C_constr=torch.tensor([[-1.0, 0.0]], dtype=DT), rhs_constr=torch.tensor([-0.5], dtype=DT),
        device=torch.device("cpu"), dtype=DT)
    evaluate = make_domain_evaluator(problem, torch.arange(1))
    z0 = torch.zeros(1, 1, 2, dtype=DT)
    _, _, viol_plain, _ = batched_gauss_newton_restoration(
        evaluate, z0, steps=20, feas_tol=1e-4)
    z, _, viol, _ = batched_gauss_newton_restoration(
        evaluate, z0, steps=20, feas_tol=1e-4, interior=1e-12)
    assert viol.item() <= 1e-4
    assert torch.all((z > 0) & (z < 1))
    assert viol_plain.item() > 1e-4  # the whole row is dropped (non-finite Jacobian)


class _Bowl(torch.nn.Module):
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return ((x[:, 0] - 0.3).square() + (x[:, 1] - 0.7).square()).reshape(-1, 1)


def test_improve_mode_runs_without_constraints():
    problem = NNProblemSingleDomain(
        model=_Bowl(), dm_l=torch.zeros(2, dtype=DT), dm_u=torch.ones(2, dtype=DT),
        C_obj=torch.ones(1, 1, dtype=DT), rhs_obj=torch.zeros(1, dtype=DT),
        C_constr=torch.empty(0, 1, dtype=DT), rhs_constr=torch.empty(0, dtype=DT),
        device=torch.device("cpu"), dtype=DT)
    x0 = torch.tensor([[0.9, 0.2]], dtype=DT)  # f = 0.61
    cfg = dict(outer_max_iter=0, inner_steps=0, num_restarts=1, feas_tol=1e-4,
               batched_gn_steps=20, batched_gn_seeds=1)
    off = ALMPGDSolver(ALMConfig(**cfg)).solve_with_restarts(
        problem, num_restarts=1, x_init=x0, incumbent=torch.tensor([0.2], dtype=DT))
    assert abs(off.objective_value - 0.61) < 1e-12
    on = ALMPGDSolver(ALMConfig(batched_gn_improve=True, batched_gn_improve_rel=1e-3,
                                **cfg)).solve_with_restarts(
        problem, num_restarts=1, x_init=x0, incumbent=torch.tensor([0.2], dtype=DT))
    assert on.success and on.objective_value <= 0.2 - 1e-3 + 1e-12


def test_origin_scaled_restarts_stay_in_box_and_only_touch_the_tail():
    solver = ALMPGDSolver(ALMConfig(init_scaled_fraction=0.5, init_scaled_min=1e-3))
    lb = torch.tensor([[-100.0, 2.0], [-1.0, -3.0]], dtype=DT)
    ub = torch.tensor([[100.0, 5.0], [1.0, -2.0]], dtype=DT)
    x_rand = lb.unsqueeze(1) + torch.rand(2, 8, 2, dtype=DT) * (ub - lb).unsqueeze(1)
    torch.manual_seed(0)
    out = solver._origin_scaled_restarts(x_rand, lb, ub)
    assert torch.equal(out[:, :4], x_rand[:, :4])
    assert torch.all((out >= lb.unsqueeze(1)) & (out <= ub.unsqueeze(1)))
    # Anchored at the projection of the origin: coordinate 0 of domain 0 is
    # centred on 0 with scales down to 1e-3 of the width.
    assert out[0, 4:, 0].abs().min() < 100.0
    # Off by default: unchanged.
    assert torch.equal(
        ALMPGDSolver(ALMConfig())._origin_scaled_restarts(x_rand, lb, ub), x_rand
    )


def test_disabled_by_default_and_parsed_from_config():
    assert ALMConfig().batched_gn_steps == 0
    parsed = ALMConfig.from_dict({
        "alm_batched_gn_steps": 7,
        "alm_batched_gn_seeds": 3,
        "alm_batched_gn_damping": 1e-3,
        "alm_batched_gn_trust_radius": 0.5,
        "alm_batched_gn_max_viol": 10.0,
        "alm_batched_gn_max_elems": 1000,
        "alm_init_scaled_fraction": 0.25,
        "alm_init_scaled_min": 1e-2,
        "alm_batched_gn_max_time": 2.5,
        "alm_batched_gn_interior": 1e-12,
    })
    assert parsed.batched_gn_steps == 7
    assert parsed.batched_gn_seeds == 3
    assert parsed.batched_gn_damping == 1e-3
    assert parsed.batched_gn_trust_radius == 0.5
    assert parsed.batched_gn_max_viol == 10.0
    assert parsed.batched_gn_max_elems == 1000
    assert parsed.init_scaled_fraction == 0.25
    assert parsed.init_scaled_min == 1e-2
    assert parsed.batched_gn_max_time == 2.5
    assert parsed.batched_gn_interior == 1e-12


def test_root_alm_overrides_only_touch_the_root_copy():
    from input_split.nlp_in_bab import apply_root_alm_overrides

    cfg = {"alm_inner_lr": 1.0, "root_alm_inner_lr": 0.01,
           "root_alm_batched_gn_steps": 5, "timeout": 60.0}
    root = apply_root_alm_overrides(cfg)
    assert root["alm_inner_lr"] == 0.01
    assert root["alm_batched_gn_steps"] == 5
    assert cfg["alm_inner_lr"] == 1.0 and "alm_batched_gn_steps" not in cfg
    assert ALMConfig.from_dict(root).inner_lr == 0.01
    assert ALMConfig.from_dict(cfg).inner_lr == 1.0
