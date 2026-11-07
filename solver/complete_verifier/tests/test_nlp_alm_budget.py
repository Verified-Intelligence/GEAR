"""Tests for ALM restart resolution from the shared B*R budget."""

import os
import sys
import types
import unittest
from unittest.mock import patch

import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAlmRestartBudget(unittest.TestCase):
    """Tests for root-side ALM restart budgeting."""

    def test_resolve_alm_restart_count_uses_shared_budget(self):
        """Shared B*R budget should determine root ALM restarts."""
        from nlp_solver.alm import ALMConfig, resolve_alm_restart_count

        cfg = ALMConfig(num_restarts=99, br_budget=18)

        self.assertEqual(resolve_alm_restart_count(batch_size=4, cfg=cfg), 4)
        self.assertEqual(
            resolve_alm_restart_count(batch_size=4, cfg=cfg, explicit_num_restarts=7),
            7,
        )

    def test_solver_uses_budget_when_num_restarts_is_omitted(self):
        """solve_with_restarts() should derive root restarts from B*R budget."""
        from nlp_solver.alm import ALMConfig, ALMPGDSolver
        from nlp_solver.problem import NNProblemSingleDomain

        class Identity(torch.nn.Module):
            def forward(self, x):
                return x.view(x.shape[0], -1)

        problem = NNProblemSingleDomain(
            model=Identity(),
            dm_l=torch.zeros(2, 1),
            dm_u=torch.ones(2, 1),
            C_obj=torch.tensor([[[1.0]], [[1.0]]], dtype=torch.float32),
            rhs_obj=torch.zeros(2, 1, dtype=torch.float32),
            C_constr=torch.empty((2, 0, 1), dtype=torch.float32),
            rhs_constr=torch.empty((2, 0), dtype=torch.float32),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        solver = ALMPGDSolver(ALMConfig(
            num_restarts=99,
            br_budget=6,
            outer_max_iter=1,
            inner_steps=1,
        ))

        seen = {}

        def fake_solve_subproblem(
            self, x, mu_constr, rho, eta_k, problem, lb, ub,
            constraint_scale=None, inner_lr_scale=1.0, deadline=None,
        ):
            del self, mu_constr, rho, eta_k, lb, ub
            del constraint_scale, inner_lr_scale, deadline
            seen["total_parallel"] = x.shape[0]
            seen["batch_size"] = problem.batch_size
            return x.detach(), torch.zeros(x.shape[0], device=x.device, dtype=x.dtype), 0

        with patch.object(
            ALMPGDSolver,
            "_solve_subproblem_adam",
            autospec=True,
            side_effect=fake_solve_subproblem,
        ):
            result = solver.solve_with_restarts(problem)

        self.assertEqual(seen["batch_size"], 2)
        self.assertEqual(seen["total_parallel"], 6)
        self.assertEqual(tuple(result.x_solution.shape), (2, 1))

    def test_from_dict_ignores_legacy_num_restarts(self):
        """ALMConfig should no longer inherit root restarts from legacy num_restarts."""
        from nlp_solver.alm import ALMConfig

        cfg = ALMConfig.from_dict({
            "num_restarts": 5,
            "bab_br_budget": 12,
        })

        self.assertEqual(cfg.num_restarts, ALMConfig.num_restarts)
        self.assertEqual(cfg.br_budget, 12)


class TestPreIncompleteRootPrimal(unittest.TestCase):
    """Tests for the root-primal route that runs before LiRPA construction."""

    def test_runner_caps_budget_and_serializes_verified_result(self):
        from input_split.nlp_in_bab import run_pre_incomplete_primal

        comparison = types.SimpleNamespace(
            alm_verified=True,
            alm_obj=-0.25,
            alm_viol=4.0e-6,
            alm_time=7.0,
            best_method="alm",
            best_obj=-0.25,
            best_x=torch.tensor([0.75]),
        )
        config = {
            "enabled": True,
            "enable_alm": True,
            "timeout": 30.0,
            "root_budget_frac": 0.5,
        }

        with patch(
            "nlp_solver.falsification.compare_solvers",
            return_value=comparison,
        ) as mock_compare:
            payload = run_pre_incomplete_primal(
                model_ori=torch.nn.Identity(),
                dm_l=torch.tensor([0.0]),
                dm_u=torch.tensor([1.0]),
                c=torch.tensor([[1.0]]),
                rhs=torch.tensor([0.0]),
                or_spec_size=torch.tensor([1]),
                nlp_config=config,
                remaining_global=15.0,
            )

        used_config = mock_compare.call_args.kwargs["config"]
        self.assertEqual(used_config["timeout"], 7.5)
        self.assertTrue(payload["verified"])
        self.assertTrue(payload["nlp_found"])
        self.assertEqual(payload["objective"], -0.25)
        self.assertEqual(payload["violation"], 4.0e-6)
        self.assertEqual(payload["x"].device.type, "cpu")
        self.assertTrue(torch.equal(payload["x"], torch.tensor([0.75])))

    def test_only_verified_payload_is_reused_at_post_incomplete_root(self):
        from input_split.nlp_in_bab import pre_incomplete_to_root_result

        dm_l = torch.tensor([[0.0]])
        dm_u = torch.tensor([[1.0]])
        verified = {
            "pre_incomplete_primal": {
                "verified": True,
                "nlp_found": False,
                "objective": 0.5,
                "x": torch.tensor([0.25]),
                "peak_mem_bytes": 123,
                "total_mem_bytes": 456,
            }
        }
        root = pre_incomplete_to_root_result(verified, dm_l, dm_u)

        self.assertIsNotNone(root)
        self.assertFalse(root.nlp_found)
        self.assertEqual(root.nlp_obj, 0.5)
        self.assertTrue(torch.equal(root.nlp_best_x, torch.tensor([[0.25]])))
        self.assertEqual(root.root_peak_mem_bytes, 123)
        self.assertEqual(root.root_total_mem_bytes, 456)

        unverified = {
            "pre_incomplete_primal": {
                "verified": False,
                "objective": None,
                "x": None,
            }
        }
        self.assertIsNone(
            pre_incomplete_to_root_result(unverified, dm_l, dm_u)
        )
