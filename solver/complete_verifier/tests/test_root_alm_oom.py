"""The root ALM retries with half the restarts when the GPU runs out of memory."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class IdentityModel(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


def _alm_result():
    return SimpleNamespace(
        success=False, x_solution=None, objective_value=2.0,
        constraint_violation=1.0, outer_iterations=1, final_rho=1.0)


def _compare(config):
    from nlp_solver.falsification import compare_solvers
    return compare_solvers(
        None, IdentityModel(),
        torch.tensor([1.0]), torch.tensor([1.0]),
        torch.tensor([[[1.0]]]), torch.tensor([[0.0]]),
        torch.tensor([1], dtype=torch.int32), config=config)


class TestRootAlmOom(unittest.TestCase):
    CONFIG = {"enable_alm": True, "verbose": False, "root_parallel": False,
              "alm_bab_br_budget": 0, "alm_num_restarts": 16, "alm_timeout": 30.0}

    def test_halves_restarts_until_it_fits(self):
        calls = []

        def fake_solve(problem, num_restarts=None, timeout=None, **kwargs):
            del problem, kwargs
            calls.append((num_restarts, timeout))
            if num_restarts is None or num_restarts > 4:
                raise torch.OutOfMemoryError("CUDA out of memory")
            return _alm_result()

        with mock.patch("nlp_solver.falsification.ALMPGDSolver") as cls, \
                mock.patch("nlp_solver.falsification.resolve_alm_restart_count",
                           return_value=16):
            cls.return_value.solve_with_restarts.side_effect = fake_solve
            result = _compare(dict(self.CONFIG))
        self.assertEqual([c[0] for c in calls], [None, 8, 4])
        self.assertIsNone(calls[0][1])
        self.assertTrue(all(c[1] is not None and c[1] > 0 for c in calls[1:]))
        self.assertEqual(result.best_method, "none")

    def test_single_restart_oom_still_raises(self):
        with mock.patch("nlp_solver.falsification.ALMPGDSolver") as cls, \
                mock.patch("nlp_solver.falsification.resolve_alm_restart_count",
                           return_value=1):
            cls.return_value.solve_with_restarts.side_effect = \
                torch.OutOfMemoryError("CUDA out of memory")
            with self.assertRaises(torch.OutOfMemoryError):
                _compare(dict(self.CONFIG))



class TestBabAlmOom(unittest.TestCase):
    def setUp(self):
        import input_split.attack_in_input_split as mod
        self.mod = mod
        mod._OOM_RESTART_CAP = None

    def tearDown(self):
        self.mod._OOM_RESTART_CAP = None

    def test_cap_halves_and_persists(self):
        calls = []

        class Solver:
            def solve_with_restarts(self, problem, num_restarts=None, **kwargs):
                calls.append(num_restarts)
                if num_restarts > 300:
                    raise torch.OutOfMemoryError("CUDA out of memory")
                return "ok"

        self.assertEqual(self.mod._solve_within_memory(Solver(), None, 1000), "ok")
        self.assertEqual(calls, [1000, 500, 250])
        self.assertEqual(self.mod._OOM_RESTART_CAP, 250)
        calls.clear()
        self.mod._solve_within_memory(Solver(), None, 1000)
        self.assertEqual(calls, [250])
        calls.clear()
        self.mod._solve_within_memory(Solver(), None, 100)
        self.assertEqual(calls, [100])

    def test_single_restart_oom_raises(self):
        class Solver:
            def solve_with_restarts(self, problem, num_restarts=None, **kwargs):
                raise torch.OutOfMemoryError("CUDA out of memory")

        with self.assertRaises(torch.OutOfMemoryError):
            self.mod._solve_within_memory(Solver(), None, 1)


if __name__ == "__main__":
    unittest.main()
