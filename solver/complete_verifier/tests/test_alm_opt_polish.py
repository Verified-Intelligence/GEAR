"""Tests for the batched SQP optimality polish of ALM candidates."""

import math

import pytest
import torch

from nlp_solver.alm import ALMConfig, ALMPGDSolver
from nlp_solver.alm_polish import CandidateEvaluator, select_candidates, sqp_polish
from nlp_solver.problem import NNProblemSingleDomain

DT = torch.float64


class _DiskLinear(torch.nn.Module):
    """out = [-(x0 + x1), x0^2 + x1^2 - 1]: maximize x0 + x1 on the unit disk."""

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return torch.stack([-(x[:, 0] + x[:, 1]), x[:, 0] ** 2 + x[:, 1] ** 2 - 1.0], dim=1)


class _EqualityPair(torch.nn.Module):
    """min (x0-2)^2 + (x1-2)^2  s.t.  x0 * x1 = 1 as two opposite rows."""

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        h = x[:, 0] * x[:, 1] - 1.0
        f = (x[:, 0] - 2.0) ** 2 + (x[:, 1] - 2.0) ** 2
        return torch.stack([f, h, -h], dim=1)


def _disk_problem(batch=1):
    lb = torch.full((batch, 2), -2.0, dtype=DT)
    ub = torch.full((batch, 2), 2.0, dtype=DT)
    return NNProblemSingleDomain(
        _DiskLinear(), lb if batch > 1 else lb[0], ub if batch > 1 else ub[0],
        torch.tensor([[[1.0, 0.0]]], dtype=DT).expand(batch, 1, 2),
        torch.zeros(batch, 1, dtype=DT),
        torch.tensor([[[0.0, 1.0]]], dtype=DT).expand(batch, 1, 2),
        torch.zeros(batch, 1, dtype=DT),
        device=torch.device("cpu"), dtype=DT,
    )


def _evaluator(problem, count):
    dom = torch.zeros(count, dtype=torch.long)
    lb = problem.dm_l.reshape(1, -1).expand(count, -1)
    ub = problem.dm_u.reshape(1, -1).expand(count, -1)
    return CandidateEvaluator(problem, dom, lb, ub), lb, ub


def test_polish_reaches_the_constrained_optimum():
    problem = _disk_problem()
    x0 = torch.tensor([[0.0, 0.0], [-0.5, 0.3], [0.9, -0.1], [1.5, 1.5]], dtype=DT)
    ev, lb, ub = _evaluator(problem, 4)
    z0 = (x0 - lb) / (ub - lb)
    z, phi, viol, found, _ = sqp_polish(ev, z0, tol=1e-4, max_iters=60)
    assert bool(found.all())
    assert bool((viol <= 1e-4).all())
    # The tolerance may be used: the optimum of the relaxed disk is slightly better.
    relaxed_opt = -math.sqrt(2.0 * (1.0 + 0.9e-4))
    assert torch.allclose(phi, torch.full_like(phi, relaxed_opt), atol=1e-6)
    # The reported point reproduces the reported objective and stays in the box.
    x = lb + z * (ub - lb)
    assert bool(((x >= lb) & (x <= ub)).all())
    phi_check, g_check, _ = ev(z)
    assert torch.allclose(phi_check, phi)
    assert bool((g_check.max(dim=1).values <= 1e-4).all())


def test_polish_never_reports_an_infeasible_point():
    problem = _disk_problem()
    ev, lb, ub = _evaluator(problem, 1)
    # Far outside the disk; one iteration cannot restore feasibility.
    z0 = (torch.tensor([[2.0, 2.0]], dtype=DT) - lb) / (ub - lb)
    _, phi, viol, found, _ = sqp_polish(ev, z0, tol=1e-4, max_iters=1)
    if bool(found[0]):
        assert viol[0] <= 1e-4
    else:
        assert viol[0] > 1e-4


def test_polish_respects_box_faces():
    problem = _disk_problem()
    # Shrink the box so the disk optimum (0.707, 0.707) is cut off by x0 <= 0.5.
    problem.dm_u = torch.tensor([[0.5, 2.0]], dtype=DT)
    ev, lb, ub = _evaluator(problem, 2)
    z0 = (torch.tensor([[0.0, 0.0], [0.4, -0.5]], dtype=DT) - lb) / (ub - lb)
    z, phi, viol, found, _ = sqp_polish(ev, z0, tol=1e-4, max_iters=60)
    x = lb + z * (ub - lb)
    assert bool(found.all())
    assert bool((x[:, 0] <= 0.5 + 1e-12).all())
    expected = -(0.5 + math.sqrt(1.0 + 0.9e-4 - 0.25))
    assert torch.allclose(phi, torch.full_like(phi, expected), atol=1e-6)


def test_polish_handles_equality_pairs():
    problem = NNProblemSingleDomain(
        _EqualityPair(), torch.tensor([0.1, 0.1], dtype=DT), torch.tensor([4.0, 4.0], dtype=DT),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=DT), torch.tensor([0.0], dtype=DT),
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=DT), torch.zeros(2, dtype=DT),
        device=torch.device("cpu"), dtype=DT,
    )
    ev, lb, ub = _evaluator(problem, 2)
    z0 = (torch.tensor([[0.5, 2.0], [3.0, 0.4]], dtype=DT) - lb) / (ub - lb)
    z, phi, viol, found, _ = sqp_polish(ev, z0, tol=1e-4, max_iters=80)
    assert bool(found.all())
    # Optimum at x0 = x1 = sqrt(1 + slack): f = 2 (2 - sqrt(1+s))^2 with |h| <= 0.9e-4.
    expected = 2.0 * (2.0 - math.sqrt(1.0 + 0.9e-4)) ** 2
    assert torch.allclose(phi, torch.full_like(phi, expected), atol=1e-6)


class _Wiggly(torch.nn.Module):
    """min sin(5x) + 0.1 x^2 on [-3, 3]: many local minima, global near x = -0.31."""

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return (torch.sin(5.0 * x[:, :1]) + 0.1 * x[:, :1] ** 2)


def test_select_candidates_prefers_distinct_feasible_points():
    x = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.5, 0.5], [0.9, 0.9], [0.2, 0.1]], dtype=DT)
    phis = torch.tensor([-1.0, -1.0, -0.5, -3.0, -2.0], dtype=DT)
    viols = torch.tensor([0.0, 0.0, 0.0, 5.0, 1e-3], dtype=DT)
    lb = torch.zeros(5, 2, dtype=DT)
    ub = torch.ones(5, 2, dtype=DT)
    pick = select_candidates(x, phis, viols, lb, ub, 4, 1e-4, near_quantile=0.8)
    picked = pick.tolist()
    assert picked[:2] == [0, 2]          # best feasible first, duplicate 1 dropped
    assert 4 in picked                   # near-feasible, low objective
    assert 3 not in picked               # badly violated
    assert 1 not in picked


def test_solver_polish_is_off_by_default_and_improves_when_on():
    base = dict(outer_max_iter=3, inner_steps=20, num_restarts=64, feas_tol=1e-4,
                first_feas_patience=1, inner_lr=0.01, rho0=10.0)
    off = ALMPGDSolver(ALMConfig(**base))
    torch.manual_seed(0)
    r_off = off.solve_with_restarts(_disk_problem(), timeout=30.0)
    on = ALMPGDSolver(ALMConfig(**base, opt_polish_candidates=8, opt_polish_iters=60))
    torch.manual_seed(0)
    r_on = on.solve_with_restarts(_disk_problem(), timeout=30.0)
    assert ALMConfig().opt_polish_candidates == 0
    assert r_on.success
    assert r_on.constraint_violation <= 1e-4
    assert r_on.objective_value <= -math.sqrt(2.0) + 1e-6
    if r_off.success:
        assert r_on.objective_value <= r_off.objective_value + 1e-12


def test_polish_uses_each_candidates_own_domain():
    problem = _disk_problem(batch=2)
    # Domain 1 excludes the optimum: x0 <= 0.
    problem.dm_u = torch.tensor([[2.0, 2.0], [0.0, 2.0]], dtype=DT)
    dom = torch.tensor([0, 1])
    lb = problem.dm_l.reshape(2, -1)[dom]
    ub = problem.dm_u.reshape(2, -1)[dom]
    ev = CandidateEvaluator(problem, dom, lb, ub)
    z0 = (torch.tensor([[0.1, 0.1], [-0.5, 0.2]], dtype=DT) - lb) / (ub - lb)
    z, phi, viol, found, _ = sqp_polish(ev, z0, tol=1e-4, max_iters=60)
    x = lb + z * (ub - lb)
    assert bool(found.all())
    assert x[1, 0] <= 1e-12
    assert phi[0] < phi[1]


def test_batched_domains_polish_within_their_own_boxes():
    # Two BaB-style domains; domain 1 excludes the disk optimum (x0 <= 0).
    problem = _disk_problem(batch=2)
    problem.dm_u = torch.tensor([[2.0, 2.0], [0.0, 2.0]], dtype=DT)
    cfg = ALMConfig(outer_max_iter=2, inner_steps=10, num_restarts=16, feas_tol=1e-4,
                    inner_lr=0.01, opt_polish_candidates=8, opt_polish_seeds=2,
                    opt_polish_iters=60)
    torch.manual_seed(0)
    result = ALMPGDSolver(cfg).solve_with_restarts(problem, timeout=30.0)
    assert bool(result.success.all())
    assert bool((result.constraint_violation <= 1e-4).all())
    x = result.x_solution
    assert bool(((x >= problem.dm_l) & (x <= problem.dm_u)).all())
    # Optimal up to how much of the 1e-4 tolerance the final point uses.
    obj0, obj1 = result.objective_value.tolist()
    assert -math.sqrt(2.0 * (1.0 + 1e-4)) - 1e-9 <= obj0 <= -math.sqrt(2.0 * (1.0 + 0.9e-4)) + 1e-6
    # Best point of domain 1: x0 = 0, x1 = sqrt(1 + slack).
    assert -math.sqrt(1.0 + 1e-4) - 1e-9 <= obj1 <= -math.sqrt(1.0 + 0.9e-4) + 1e-6


class _SumModel(torch.nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1).sum(dim=1, keepdim=True)


def test_polish_skips_problems_too_large_for_dense_matrices():
    n = 6000                                  # n^2 alone exceeds the dense budget
    problem = NNProblemSingleDomain(
        _SumModel(), torch.zeros(n, dtype=DT), torch.ones(n, dtype=DT),
        torch.tensor([[1.0]], dtype=DT), torch.tensor([0.0], dtype=DT),
        torch.zeros(0, 1, dtype=DT), torch.zeros(0, dtype=DT),
        device=torch.device("cpu"), dtype=DT,
    )
    cfg = ALMConfig(outer_max_iter=0, inner_steps=0, num_restarts=2, feas_tol=1e-4,
                    opt_polish_candidates=4, opt_polish_seeds=2)
    solver = ALMPGDSolver(cfg)
    best = torch.full((1, n), 0.5, dtype=DT)
    out = solver._optimality_polish(
        problem, best.clone(), torch.tensor([n * 0.5], dtype=DT), torch.zeros(1, dtype=DT),
        torch.ones(1, dtype=torch.bool), best.repeat(2, 1), torch.full((2,), n * 0.5, dtype=DT),
        torch.zeros(2, dtype=DT), problem.dm_l.reshape(1, -1).repeat(2, 1),
        problem.dm_u.reshape(1, -1).repeat(2, 1), 2, deadline=None,
    )
    assert out[4] == 0 and torch.equal(out[0], best)


def test_polish_keeps_its_time_share_when_the_outer_loop_runs_to_the_deadline():
    # An outer loop that never stops early would consume the whole budget and
    # leave the polish nothing; the reserve keeps a quarter of it.
    cfg = ALMConfig(outer_max_iter=10 ** 9, inner_steps=1, num_restarts=4, feas_tol=1e-4,
                    first_feas_patience=-1, obj_patience=-1,
                    inner_lr=1e-6, opt_polish_candidates=4, opt_polish_seeds=1,
                    opt_polish_time=0.5, opt_polish_iters=60)
    torch.manual_seed(0)
    result = ALMPGDSolver(cfg).solve_with_restarts(_disk_problem(), timeout=2.0)
    assert result.success
    assert result.objective_value <= -math.sqrt(2.0 * (1.0 + 0.9e-4)) + 1e-6
    assert result.solve_time <= 2.5


def test_polish_seeds_start_from_the_restart_start_points():
    # No ALM iterations at all: only the seed (box midpoint) can reach the optimum.
    cfg = ALMConfig(outer_max_iter=0, inner_steps=0, num_restarts=4, feas_tol=1e-4,
                    opt_polish_candidates=4, opt_polish_seeds=1, opt_polish_iters=60)
    solver = ALMPGDSolver(cfg)
    x = torch.arange(8, dtype=DT).reshape(4, 2)
    seed_x, seed_dom = solver._polish_seed_points(x, 1, 4)
    assert seed_x.tolist() == [[0.0, 1.0]] and seed_dom.tolist() == [0]
    seed_x, seed_dom = solver._polish_seed_points(x, 2, 2)
    assert seed_x.tolist() == [[0.0, 1.0]] and seed_dom.tolist() == [0]
    result = solver.solve_with_restarts(_disk_problem(), timeout=30.0)
    assert result.success
    assert abs(result.objective_value + math.sqrt(2.0 * (1.0 + 0.9e-4))) < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chol_solve_avoids_magma_on_large_batched_systems():
    # cute_gouldqp2/3 (n = 699): a batched triangular solve with more than 8 systems
    # larger than 512 dispatches to MAGMA, whose raw cudaMalloc of a pointer array
    # aborts the process (assert) when the caching allocator holds the memory.
    from torch.profiler import profile, ProfilerActivity
    from nlp_solver.alm_polish import _chol_solve
    n, batch = 600, 12
    a = torch.randn(batch, n, n, dtype=torch.float64, device="cuda")
    mat = a @ a.transpose(1, 2) + n * torch.eye(n, dtype=torch.float64, device="cuda")
    factor = torch.linalg.cholesky(mat)
    rhs = torch.randn(batch, n, dtype=torch.float64, device="cuda")
    _chol_solve(factor, rhs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        x = _chol_solve(factor, rhs)
        torch.cuda.synchronize()
    keys = {e.key for e in prof.key_averages()}
    assert not any("magma" in k.lower() or "template_batched" in k for k in keys), keys
    assert torch.allclose(torch.bmm(mat, x.unsqueeze(-1)).squeeze(-1), rhs, rtol=1e-8, atol=1e-8)
