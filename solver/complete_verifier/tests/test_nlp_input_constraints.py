"""Focused tests for native input linear constraints."""

import os
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVerificationSpecInputConstraints(unittest.TestCase):
    """Tests for VerificationSpec input-side linear constraints."""

    def test_input_spec_broadcasts_constraints(self):
        from api import VerificationSpec

        lower = torch.zeros(3, 2)
        upper = torch.ones(3, 2)
        clauses = [[(torch.tensor([[1.0, 0.0]]), torch.tensor([0.0]))]]
        A = torch.tensor([[1.0, 1.0]])
        b = torch.tensor([1.0])

        spec = VerificationSpec.build_from_input_bounds(
            lower,
            upper,
            clauses,
            input_linear_constraints=(A, b),
        )

        constr_A, constr_b = spec.input_linear_constraints
        self.assertEqual(constr_A.shape, (3, 1, 2))
        self.assertEqual(constr_b.shape, (3, 1))
        self.assertTrue(torch.allclose(constr_A[0], A))
        self.assertTrue(torch.allclose(constr_b[2], b))

    def test_input_spec_rejects_bad_constraint_dim(self):
        from api import VerificationSpec

        lower = torch.zeros(1, 2)
        upper = torch.ones(1, 2)
        with self.assertRaises(ValueError):
            VerificationSpec.InputSpec(
                lower,
                upper,
                input_linear_constraints=(torch.ones(1, 3), torch.zeros(1)),
            )



class TestBoundedTensorInputConstraintCopies(unittest.TestCase):
    """Tests that BoundedTensor helpers preserve immutable base input constraints."""

    def test_take_batch_preserves_base_input_constraints(self):
        from auto_LiRPA import BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm
        from utils import take_batch

        data = torch.randn(4, 2)
        x_L = data - 0.1
        x_U = data + 0.1
        base_linear = (
            torch.arange(8, dtype=torch.float32).view(4, 1, 2),
            torch.arange(4, dtype=torch.float32).view(4, 1),
        )
        base_standard = (base_linear[0].clone(), -base_linear[1].clone())
        ptb = PerturbationLpNorm(
            norm=float("inf"),
            x_L=x_L,
            x_U=x_U,
            constraints=base_standard,
            base_input_linear_constraints=base_linear,
            base_input_constraints=base_standard,
        )
        bounded = BoundedTensor(data, ptb)

        batch = take_batch(bounded, batch_size=2, batch_idx=1)

        self.assertTrue(torch.equal(batch.ptb.base_input_linear_constraints[0], base_linear[0][2:4]))
        self.assertTrue(torch.equal(batch.ptb.base_input_linear_constraints[1], base_linear[1][2:4]))
        self.assertTrue(torch.equal(batch.ptb.base_input_constraints[0], base_standard[0][2:4]))
        self.assertTrue(torch.equal(batch.ptb.base_input_constraints[1], base_standard[1][2:4]))


class TestNLPProblemInputConstraints(unittest.TestCase):
    """Tests that NLP problems append input constraints after output constraints."""

    def test_constraints_and_jacobian_include_input_constraints(self):
        from nlp_solver.problem import NNProblemSingleDomain

        class Toy(torch.nn.Module):
            def forward(self, x):
                x = x.view(x.shape[0], -1)
                return torch.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], dim=1)

        problem = NNProblemSingleDomain(
            model=Toy(),
            dm_l=torch.zeros(1, 2),
            dm_u=torch.ones(1, 2),
            C_obj=torch.tensor([[1.0, 0.0]]),
            rhs_obj=torch.tensor([0.0]),
            C_constr=torch.tensor([[0.0, 1.0]]),
            rhs_constr=torch.tensor([0.25]),
            input_constr_A=torch.tensor([[1.0, 1.0]]),
            input_constr_b=torch.tensor([1.2]),
        )

        x = np.array([0.5, 0.25], dtype=np.float64)
        constr = problem.constraints(x)
        jac = problem.jacobian(x)

        self.assertEqual(problem.n_constraints, 2)
        self.assertTrue(np.allclose(constr, np.array([0.0, -0.45]), atol=1e-6))
        self.assertTrue(np.allclose(jac[0], np.array([1.0, -1.0]), atol=1e-6))
        self.assertTrue(np.allclose(jac[1], np.array([1.0, 1.0]), atol=1e-6))


class TestIPOPTProblemCallbacks(unittest.TestCase):
    """Regression tests for IPOPT callback semantics."""

    def test_ipopt_problem_matches_single_domain_callbacks(self):
        from nlp_solver.ipopt_solver import IPOPTSolver, NNProblemIPOPT

        class Toy(torch.nn.Module):
            def forward(self, x):
                x = x.view(x.shape[0], -1)
                return torch.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], dim=1)

        problem = NNProblemIPOPT(
            model=Toy(),
            dm_l=torch.zeros(1, 2),
            dm_u=torch.ones(1, 2),
            C_obj=torch.tensor([[1.0, 0.0]]),
            rhs_obj=torch.tensor([0.0]),
            C_constr=torch.tensor([[0.0, 1.0]]),
            rhs_constr=torch.tensor([0.25]),
            input_constr_A=torch.tensor([[1.0, 1.0]]),
            input_constr_b=torch.tensor([1.2]),
        )
        solver = IPOPTSolver({"verbose": False})

        x = np.array([0.5, 0.25], dtype=np.float64)
        obj, constr = solver._pytorch_objective_and_constraints(problem, x)
        grad, jac = solver._pytorch_gradients(problem, x)

        self.assertAlmostEqual(obj, 0.75, places=6)
        self.assertTrue(np.allclose(constr, np.array([0.0, -0.45]), atol=1e-6))
        self.assertTrue(np.allclose(grad, np.array([1.0, 1.0]), atol=1e-6))
        self.assertTrue(np.allclose(jac[0], np.array([1.0, -1.0]), atol=1e-6))
        self.assertTrue(np.allclose(jac[1], np.array([1.0, 1.0]), atol=1e-6))


class TestDualAscentInputConstraints(unittest.TestCase):
    """Tests dual ascent with appended input-side linear constraints."""

    def test_dual_ascent_appends_input_constraint_duals(self):
        from nlp_solver.dual_ascent import DualAscentConfig, dual_ascent_tighten

        lA = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        lbias = torch.tensor([[0.0, 0.0]])
        dm_l = torch.tensor([[0.0, 0.0]])
        dm_u = torch.tensor([[1.0, 1.0]])
        rhs = torch.tensor([[0.0, 0.25]])
        input_A = torch.tensor([[[1.0, 1.0]]])
        input_b = torch.tensor([[1.2]])

        lb_initial, lb_dual, lam_star = dual_ascent_tighten(
            lA,
            lbias,
            dm_l,
            dm_u,
            rhs,
            DualAscentConfig(num_steps=5, lr=0.1),
            input_constr_A=input_A,
            input_constr_b=input_b,
        )

        self.assertEqual(lb_initial.shape, (1,))
        self.assertEqual(lb_dual.shape, (1,))
        self.assertEqual(lam_star.shape, (1, 2))
        self.assertGreaterEqual(lb_dual.item(), lb_initial.item())


class TestCounterexampleVerificationDType(unittest.TestCase):
    """Regression tests for dtype inference in NLP counterexample checks."""

    def test_verify_counterexample_checks_input_linear_constraints(self):
        from nlp_solver.statistics import verify_counterexample

        class IdentityModel(torch.nn.Module):
            def forward(self, x):
                return x[:, :1]

        valid, obj, viol = verify_counterexample(
            model=IdentityModel(),
            x=torch.tensor([0.75], dtype=torch.float32),
            C_obj=torch.tensor([[1.0]], dtype=torch.float32),
            rhs_obj=torch.tensor([0.0], dtype=torch.float32),
            C_constr=torch.empty((0, 1), dtype=torch.float32),
            rhs_constr=torch.empty((0,), dtype=torch.float32),
            input_constr_A=torch.tensor([[[1.0]]], dtype=torch.float32),
            input_constr_b=torch.tensor([[0.5]], dtype=torch.float32),
            tol=1e-6,
        )

        self.assertFalse(valid)
        self.assertAlmostEqual(obj, 0.75, places=6)
        self.assertAlmostEqual(viol, 0.25, places=6)

    def test_verify_counterexample_prefers_floating_buffer_dtype(self):
        from nlp_solver.statistics import verify_counterexample

        class BufferOnlyDoubleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("indices", torch.tensor([0], dtype=torch.int64))
                self.register_buffer("scale", torch.tensor(2.0, dtype=torch.float64))

            def forward(self, x):
                x = x.to(dtype=self.scale.dtype)
                return (x[:, :1] * self.scale)

        model = BufferOnlyDoubleModel()
        valid, obj, viol = verify_counterexample(
            model=model,
            x=torch.tensor([1.0], dtype=torch.float32),
            C_obj=torch.tensor([[1.0]], dtype=torch.float32),
            rhs_obj=torch.tensor([1.0], dtype=torch.float32),
            C_constr=torch.empty((0, 1), dtype=torch.float32),
            rhs_constr=torch.empty((0,), dtype=torch.float32),
            tol=1e-6,
        )

        self.assertTrue(valid)
        self.assertAlmostEqual(obj, 1.0, places=6)
        self.assertAlmostEqual(viol, 0.0, places=6)


class TestCompareSolversFeasibleIncumbent(unittest.TestCase):
    """Regression tests for feasible incumbent propagation in compare_solvers."""

    def test_compare_solvers_warmstarts_alm_from_verified_ipopt_point(self):
        from nlp_solver.falsification import compare_solvers

        class IdentityModel(torch.nn.Module):
            def forward(self, x):
                return x.view(x.shape[0], -1)

        ipopt_result = SimpleNamespace(
            success=False,
            x_solution=np.array([0.25], dtype=np.float64),
            objective_value=0.25,
            constraint_violation=0.0,
            solve_time=0.01,
            status="cyipopt: solved",
            num_iterations=3,
        )
        alm_result = SimpleNamespace(
            success=False,
            x_solution=None,
            objective_value=2.0,
            constraint_violation=1.0,
            outer_iterations=1,
            final_rho=1.0,
        )

        with mock.patch(
            "nlp_solver.falsification.IPOPT_AVAILABLE",
            True,
        ), mock.patch(
            "nlp_solver.falsification.IPOPTSolver"
        ) as mock_ipopt_cls, mock.patch(
            "nlp_solver.falsification.ALMPGDSolver"
        ) as mock_alm_cls:
            mock_ipopt_cls.return_value.solve_single.return_value = ipopt_result
            mock_alm_cls.return_value.solve_with_restarts.return_value = alm_result

            result = compare_solvers(
                None,
                IdentityModel(),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([1.0], dtype=torch.float32),
                torch.tensor([[[1.0]]], dtype=torch.float32),
                torch.tensor([[0.5]], dtype=torch.float32),
                torch.tensor([1], dtype=torch.int32),
                config={
                    "enable_local_solver": True,
                    "method": "ipopt",
                    "verbose": False,
                    "root_parallel": False,
                    "constr_viol_tol": 1e-8,
                },
            )

        mock_ipopt_cls.return_value.solve_single.assert_called_once()
        mock_ipopt_cls.return_value.solve_with_restarts.assert_not_called()
        _, kwargs = mock_alm_cls.return_value.solve_with_restarts.call_args
        self.assertIsNotNone(kwargs["x_init"])
        self.assertTrue(torch.allclose(kwargs["x_init"].reshape(-1), torch.tensor([0.25])))
        self.assertEqual(result.local_method, "IPOPT")
        self.assertEqual(result.local_status, "cyipopt: solved")
        self.assertEqual(result.alm_warmstart_source, "IPOPT")
        self.assertEqual(result.best_method, "ipopt")

    def test_compare_solvers_parallel_ipopt_runs_alm_without_waiting_for_warmstart(self):
        from nlp_solver.falsification import compare_solvers

        class IdentityModel(torch.nn.Module):
            def forward(self, x):
                return x.view(x.shape[0], -1)

        local_started = threading.Event()
        alm_started = threading.Event()
        order = []

        def fake_local_solver(*args, **kwargs):
            del args, kwargs
            order.append("local_start")
            local_started.set()
            alm_started.wait(timeout=1.0)
            order.append("local_finish")
            return SimpleNamespace(
                method_key="ipopt",
                method_label="IPOPT",
                success=False,
                objective=None,
                violation=None,
                time_sec=0.01,
                verified=False,
                status="ipopt done",
                x=None,
            )

        alm_result = SimpleNamespace(
            success=False,
            x_solution=None,
            objective_value=2.0,
            constraint_violation=1.0,
            outer_iterations=1,
            final_rho=1.0,
        )

        def fake_alm_solve(*args, **kwargs):
            del args
            order.append("alm_start")
            alm_started.set()
            self.assertIsNone(kwargs.get("x_init"))
            return alm_result

        with mock.patch(
            "nlp_solver.falsification.IPOPT_AVAILABLE",
            True,
        ), mock.patch(
            "nlp_solver.falsification._run_local_solver",
            side_effect=fake_local_solver,
        ), mock.patch(
            "nlp_solver.falsification.ALMPGDSolver"
        ) as mock_alm_cls:
            mock_alm_cls.return_value.solve_with_restarts.side_effect = fake_alm_solve

            result = compare_solvers(
                None,
                IdentityModel(),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([1.0], dtype=torch.float32),
                torch.tensor([[[1.0]]], dtype=torch.float32),
                torch.tensor([[0.5]], dtype=torch.float32),
                torch.tensor([1], dtype=torch.int32),
                config={
                    "enable_local_solver": True,
                    "method": "ipopt",
                    "verbose": False,
                    "root_parallel": True,
                    "constr_viol_tol": 1e-8,
                },
            )

        self.assertTrue(local_started.is_set())
        self.assertTrue(alm_started.is_set())
        self.assertLess(order.index("alm_start"), order.index("local_finish"))
        self.assertEqual(result.local_method, "IPOPT")
        self.assertIsNone(result.alm_warmstart_source)


class TestAffineConstraintUpperBounds(unittest.TestCase):
    """Tests for affine upper-bound helpers used by ALM dual-cut selection."""

    def test_domain_upper_bound_respects_base_constraints(self):
        from input_constraints import affine_constraints_domain_upper_bound

        candidate = (
            torch.tensor([[[1.0]]]),
            torch.tensor([[-0.9]]),
        )
        base_constraints = (
            torch.tensor([[[1.0]]]),
            torch.tensor([[-0.8]]),
        )
        x_L = torch.tensor([[0.0]])
        x_U = torch.tensor([[1.0]])

        upper = affine_constraints_domain_upper_bound(
            candidate,
            x_L,
            x_U,
            domain_constraints=base_constraints,
            rearrange_constraints=False,
        )

        # Sound: never below the exact value -0.1. Close: the float32 result carries the outward rounding enclosure
        # of every concretization (7d928238d), here 3.4e-5 = a few (n + 16) * 4 * eps_fp32 terms; in float64 it is
        # ~6e-14.
        self.assertTrue(bool((upper >= -0.1 - 1e-9).all()))
        self.assertTrue(torch.allclose(upper, torch.tensor([[-0.1]]), atol=1e-4))


if __name__ == '__main__':
    unittest.main()
