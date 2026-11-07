"""Soundness-focused tests for ALM primal-restoration options."""

from types import SimpleNamespace

import torch

from nlp_solver.alm import ALMConfig, ALMPGDSolver
from nlp_solver.problem import NNProblemSingleDomain


class _ScaledIdentity(torch.nn.Module):
    def forward(self, x):
        return 1.0e6 * x.reshape(x.shape[0], 1)


class _MixedAffineNonlinearPairs(torch.nn.Module):
    def forward(self, x):
        affine = x[:, 0] + x[:, 1] - 1.0
        nonlinear = x[:, 0] * x[:, 1] - 0.2
        tolerance = 1.0e-5
        return torch.stack(
            [
                affine - tolerance,
                -affine - tolerance,
                nonlinear - tolerance,
                -nonlinear - tolerance,
            ],
            dim=1,
        )


def _scaled_problem() -> NNProblemSingleDomain:
    dtype = torch.float64
    return NNProblemSingleDomain(
        model=_ScaledIdentity().to(dtype=dtype),
        dm_l=torch.tensor([1.0], dtype=dtype),
        dm_u=torch.tensor([1.0], dtype=dtype),
        C_obj=torch.tensor([[0.0]], dtype=dtype),
        rhs_obj=torch.tensor([0.0], dtype=dtype),
        C_constr=torch.tensor([[1.0]], dtype=dtype),
        rhs_constr=torch.tensor([0.0], dtype=dtype),
        device=torch.device("cpu"),
        dtype=dtype,
    )


def test_normalization_never_changes_raw_feasibility_decision():
    """A small normalized residual must not make a raw-infeasible point valid."""
    cfg = ALMConfig(
        outer_max_iter=0,
        inner_steps=0,
        num_restarts=2,
        constraint_normalize=True,
        constraint_scale_floor=1.0,
        feas_tol=1.0e-4,
    )
    result = ALMPGDSolver(cfg).solve_with_restarts(
        _scaled_problem(), num_restarts=2
    )

    assert result.success is False
    assert result.constraint_violation == 1.0e6


def test_normalized_multiplier_round_trip_returns_raw_space():
    """Dual warm starts and returned multipliers retain raw-constraint units."""
    cfg = ALMConfig(
        outer_max_iter=0,
        inner_steps=0,
        num_restarts=2,
        constraint_normalize=True,
        constraint_scale_floor=1.0,
    )
    raw_mu = torch.tensor([7.0], dtype=torch.float64)
    result = ALMPGDSolver(cfg).solve_with_restarts(
        _scaled_problem(), num_restarts=2, mu_init=raw_mu
    )

    assert result.final_mu is not None
    assert torch.allclose(result.final_mu, raw_mu)


def test_normalized_search_still_archives_smallest_raw_violation():
    """Preconditioned merit must not replace the authoritative raw fallback."""
    dtype = torch.float64
    problem = NNProblemSingleDomain(
        model=torch.nn.Identity(),
        dm_l=torch.zeros(1, dtype=dtype),
        dm_u=torch.ones(1, dtype=dtype),
        C_obj=torch.zeros(1, 1, dtype=dtype),
        rhs_obj=torch.zeros(1, dtype=dtype),
        # At x=0, raw max violation is 10. Random restarts usually make
        # the second row hundreds large, but its RMS normalization can make
        # normalized merit look smaller.
        C_constr=torch.tensor([[-100.0], [1000.0]], dtype=dtype),
        rhs_constr=torch.tensor([-10.0, 100.0], dtype=dtype),
        device=torch.device("cpu"),
        dtype=dtype,
    )
    cfg = ALMConfig(
        outer_max_iter=0,
        inner_steps=0,
        num_restarts=4,
        constraint_normalize=True,
        constraint_scale_floor=1.0,
    )
    torch.manual_seed(0)
    result = ALMPGDSolver(cfg).solve_with_restarts(
        problem,
        num_restarts=4,
        x_init=torch.zeros(1, dtype=dtype),
    )

    assert result.success is False
    assert result.constraint_violation == 10.0
    assert result.x_solution is not None
    assert torch.equal(result.x_solution, torch.zeros(1, dtype=dtype))


def test_raw_minimax_polish_converts_near_feasible_archive_to_primal():
    """The local phase must meet the unchanged raw tolerance, not a scaled proxy."""
    dtype = torch.float64
    problem = NNProblemSingleDomain(
        model=torch.nn.Identity(),
        dm_l=torch.zeros(1, dtype=dtype),
        dm_u=torch.ones(1, dtype=dtype),
        C_obj=torch.zeros(1, 1, dtype=dtype),
        rhs_obj=torch.zeros(1, dtype=dtype),
        # g(x) = 0.25 - x <= 0.
        C_constr=torch.tensor([[-1.0]], dtype=dtype),
        rhs_constr=torch.tensor([-0.25], dtype=dtype),
        device=torch.device("cpu"),
        dtype=dtype,
    )
    cfg = ALMConfig(
        outer_max_iter=0,
        inner_steps=0,
        num_restarts=1,
        raw_polish_steps=100,
        raw_polish_lr=0.02,
        raw_polish_restarts=2,
        raw_polish_jitter=0.0,
        raw_polish_max_viol=1.0,
        feas_tol=1.0e-4,
    )
    result = ALMPGDSolver(cfg).solve_with_restarts(
        problem,
        num_restarts=1,
        x_init=torch.zeros(1, dtype=dtype),
    )

    assert result.success is True
    assert result.constraint_violation <= cfg.feas_tol
    assert result.x_solution is not None
    assert result.x_solution.item() >= 0.25 - cfg.feas_tol


def test_raw_minimax_polish_respects_activation_threshold():
    """Far-away archives are left unchanged instead of consuming local budget."""
    cfg = ALMConfig(
        outer_max_iter=0,
        inner_steps=0,
        num_restarts=1,
        raw_polish_steps=100,
        raw_polish_lr=0.02,
        raw_polish_max_viol=1.0,
    )
    result = ALMPGDSolver(cfg).solve_with_restarts(
        _scaled_problem(), num_restarts=1
    )

    assert result.success is False
    assert result.constraint_violation == 1.0e6
    assert result.total_inner_steps == 0


def test_raw_minimax_polish_skips_oversized_bab_batch():
    """A huge BaB domain batch must not be multiplied by local clone count."""
    dtype = torch.float64
    cfg = ALMConfig(
        raw_polish_steps=100,
        raw_polish_restarts=64,
        raw_polish_max_parallel=1,
    )
    problem = SimpleNamespace(batch_size=2, n_constraints=1)
    best_x = torch.zeros(2, 1, dtype=dtype)
    best_obj = torch.zeros(2, dtype=dtype)
    best_viol = torch.ones(2, dtype=dtype)
    best_is_feas = torch.zeros(2, dtype=torch.bool)

    returned = ALMPGDSolver(cfg)._polish_raw_feasibility(
        problem,
        best_x,
        best_obj,
        best_viol,
        best_is_feas,
        deadline=None,
    )

    assert returned[-1] == 0
    assert torch.equal(returned[0], best_x)
    assert torch.equal(returned[2], best_viol)


def test_output_affine_projection_excludes_nonlinear_equality_pair():
    """Constant-Jacobian detection should recover only the affine manifold."""
    dtype = torch.float64
    problem = NNProblemSingleDomain(
        model=_MixedAffineNonlinearPairs().to(dtype=dtype),
        dm_l=torch.zeros(2, dtype=dtype),
        dm_u=torch.ones(2, dtype=dtype),
        C_obj=torch.zeros(1, 4, dtype=dtype),
        rhs_obj=torch.zeros(1, dtype=dtype),
        C_constr=torch.eye(4, dtype=dtype),
        rhs_constr=torch.zeros(4, dtype=dtype),
        device=torch.device("cpu"),
        dtype=dtype,
    )
    cfg = ALMConfig(
        output_affine_project_init=True,
        output_affine_project_sweeps=3,
        output_affine_detection_tol=1.0e-10,
    )
    seeds = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [0.1, 0.2]], dtype=dtype
    )
    projected = ALMPGDSolver(cfg)._project_output_affine_equalities(
        seeds.clone(), problem, B=1, R=3
    )

    assert torch.all(projected >= 0.0)
    assert torch.all(projected <= 1.0)
    assert torch.allclose(
        projected.sum(dim=1), torch.ones(3, dtype=dtype), atol=1.0e-9
    )
    # The nonlinear equality x0*x1=0.2 is not projected automatically.
    assert not torch.allclose(
        projected[:, 0] * projected[:, 1],
        torch.full((3,), 0.2, dtype=dtype),
        atol=1.0e-3,
    )


def test_output_affine_projection_can_preserve_unprojected_restart_pool():
    """Only the configured prefix should be committed to the hard manifold."""
    dtype = torch.float64
    problem = NNProblemSingleDomain(
        model=_MixedAffineNonlinearPairs().to(dtype=dtype),
        dm_l=torch.zeros(2, dtype=dtype),
        dm_u=torch.ones(2, dtype=dtype),
        C_obj=torch.zeros(1, 4, dtype=dtype),
        rhs_obj=torch.zeros(1, dtype=dtype),
        C_constr=torch.eye(4, dtype=dtype),
        rhs_constr=torch.zeros(4, dtype=dtype),
        device=torch.device("cpu"),
        dtype=dtype,
    )
    seeds = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [0.1, 0.2]], dtype=dtype
    )
    cfg = ALMConfig(
        output_affine_project_init=True,
        output_affine_project_fraction=0.5,
        output_affine_project_sweeps=3,
        output_affine_detection_tol=1.0e-10,
    )
    projected = ALMPGDSolver(cfg)._project_output_affine_equalities(
        seeds.clone(), problem, B=1, R=3
    )

    assert torch.allclose(
        projected[:2].sum(dim=1), torch.ones(2, dtype=dtype), atol=1.0e-9
    )
    assert torch.equal(projected[2], seeds[2])

