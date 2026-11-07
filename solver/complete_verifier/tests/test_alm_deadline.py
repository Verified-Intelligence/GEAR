"""Cancellation must discard partial derivatives and stop root OOM retries."""
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from nlp_solver import alm_feas
from nlp_solver.alm import ALMConfig, ALMPGDSolver
from nlp_solver.utils import DeadlineExceeded
from tests.test_alm_batched_gn import _circle_problem
from tests.test_root_alm_oom import _compare, _alm_result


def test_initialization_expiry_never_enters_outer_loop(monkeypatch):
    problem = _circle_problem([[0., 0.]], [[1., 1.]])
    solver = ALMPGDSolver(ALMConfig(output_affine_project_init=True))
    def expired(*args, **kwargs):
        raise DeadlineExceeded
    monkeypatch.setattr(solver, '_project_output_affine_equalities', expired)
    result = solver.solve_with_restarts(problem, timeout=1)
    assert result.outer_iterations == 0 and not result.success
    assert result.x_solution is None and 'deadline' in result.status


def test_expiry_in_jacobian_chunk_discards_partial_jacobian(monkeypatch):
    clock = [0.]
    calls = []
    orig = torch.autograd.grad
    def grad(*args, **kwargs):
        calls.append(1)
        result = orig(*args, **kwargs)
        clock[0] += 1
        return result
    monkeypatch.setattr(alm_feas.time, 'time', lambda: clock[0])
    monkeypatch.setattr(torch.autograd, 'grad', grad)
    def evaluate(z):
        return z[..., 0], torch.cat([z.square(), -z.square()], -1)
    with pytest.raises(DeadlineExceeded):
        alm_feas.residual_jacobian(evaluate, torch.ones(1, 1, 2), 1, deadline=.5)
    assert len(calls) == 1


def test_oom_after_root_deadline_does_not_restart(monkeypatch):
    from nlp_solver import falsification
    clock = [0.]
    monkeypatch.setattr(falsification.time, 'time', lambda: clock[0])
    calls = []
    def oom(*args, **kwargs):
        calls.append(1)
        clock[0] = 2.
        raise torch.OutOfMemoryError('CUDA out of memory')
    with patch.object(falsification, 'ALMPGDSolver') as cls:
        cls.return_value.solve_with_restarts.side_effect = oom
        cls.return_value._build_invalid_result.return_value = _alm_result()
        _compare({'enable_alm': True, 'verbose': False, 'root_parallel': False,
                  'timeout': 1., 'bab_br_budget': 16})
    assert len(calls) == 1
