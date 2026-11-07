import unittest
from unittest import mock

import numpy as np


class DummyProblem:
    n_vars = 1
    n_constraints = 0
    lb = np.array([0.0], dtype=np.float64)
    ub = np.array([1.0], dtype=np.float64)

    def objective(self, x):
        return float(x[0] ** 2)

    def constraints(self, x):
        return np.empty((0,), dtype=np.float64)

    def gradient(self, x):
        return np.array([2.0 * x[0]], dtype=np.float64)

    def jacobian(self, x):
        return np.empty((0, 1), dtype=np.float64)


class TestIPOPTSolverFallback(unittest.TestCase):
    def test_solve_single_uses_scipy_fallback_when_ipopt_bindings_are_missing(self):
        from nlp_solver import ipopt_solver

        solver = ipopt_solver.IPOPTSolver({"max_iter": 5, "verbose": False})

        with mock.patch.object(ipopt_solver, "CYIPOPT_AVAILABLE", False):
            result = solver.solve_single(DummyProblem(), timeout=0.1)

        self.assertIsNotNone(result.objective_value)
        self.assertNotIn("No IPOPT backend available", result.status)
        self.assertIn("trust-constr fallback", result.status)


class TestIPOPTWallClockBudget(unittest.TestCase):
    def test_cpu_time_limit_does_not_undercut_the_wall_budget(self):
        """max_cpu_time is process CPU time: the ALM thread running beside IPOPT at the
        root, plus BLAS threads, burn it several times faster than the wall clock, so
        IPOPT stopped with "Maximum CPU time exceeded" after ~2.6 s of a 10 s budget
        (pooling_bental5tp, at -3096.7 with violation 8e-3). The wall deadline is
        enforced by the intermediate callback; the CPU limit must not bind first."""
        from nlp_solver import ipopt_solver

        if not ipopt_solver.CYIPOPT_AVAILABLE:
            self.skipTest("cyipopt not available")
        options = {}

        class FakeProblem:
            def __init__(self, *args, **kwargs):
                pass

            def add_option(self, key, value):
                options[key] = value

            def solve(self, x0):
                return np.asarray(x0, dtype=np.float64), {"status": 0, "status_msg": b"ok",
                                                           "obj_val": 0.0}

        solver = ipopt_solver.IPOPTSolver({"max_iter": 5, "verbose": False})
        with mock.patch.object(ipopt_solver.cyipopt, "Problem", FakeProblem):
            try:
                solver.solve_single(DummyProblem(), timeout=10.0)
            except Exception:  # noqa: BLE001 - only the options matter here
                pass
        self.assertIn("max_cpu_time", options)
        self.assertGreaterEqual(options["max_cpu_time"], 10.0 * 10.0)


class TestIPOPTKeepInBoxRetry(unittest.TestCase):
    def test_rejected_point_is_finished_inside_the_box(self):
        """Ipopt's default bound relaxation lets iterates leave the box; clipping the
        returned point back moved waterund01's constraints by ~3e-4 and it was rejected.
        Keeping iterates in the box from the start stalled Ipopt on wastewater04m2 (1000
        iterations at 124.4 against 89.8 in 1.8 s), so it is only the retry."""
        from nlp_solver import ipopt_solver

        if not ipopt_solver.CYIPOPT_AVAILABLE:
            self.skipTest("cyipopt not available")
        calls = []

        class FakeProblem:
            def __init__(self, *args, **kwargs):
                calls.append({})

            def add_option(self, key, value):
                calls[-1][key] = value

            def solve(self, x0):
                # First solve: a point outside the constraint; the retry: a feasible one.
                x = 0.9 if len(calls) == 1 else 0.4
                return np.array([x], dtype=np.float64), {"status": 0, "status_msg": b"ok"}

        class OneConstraint(DummyProblem):
            n_constraints = 1

            def constraints(self, x):
                return np.array([x[0] - 0.5], dtype=np.float64)

            def jacobian(self, x):
                return np.ones((1, 1), dtype=np.float64)

        solver = ipopt_solver.IPOPTSolver({"constr_viol_tol": 1e-4})
        with mock.patch.object(ipopt_solver.cyipopt, "Problem", FakeProblem):
            result = solver.solve_single(OneConstraint(), timeout=10.0)
        self.assertNotIn("bound_relax_factor", calls[0])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1].get("bound_relax_factor"), 0.0)
        self.assertIsNotNone(result.x_solution)
        self.assertAlmostEqual(float(result.x_solution[0]), 0.4)


class TestIPOPTConstraintSlack(unittest.TestCase):
    def test_slack_moves_cu_and_keeps_result_inside_acceptance(self):
        from nlp_solver import ipopt_solver

        if not ipopt_solver.CYIPOPT_AVAILABLE:
            self.skipTest("cyipopt not available")
        seen = {}

        class FakeProblem:
            def __init__(self, *args, **kwargs):
                seen['cu'] = kwargs.get('cu')

            def add_option(self, key, value):
                seen[key] = value

            def solve(self, x0):
                return np.asarray(x0, dtype=np.float64), {"status": 0, "status_msg": b"ok"}

        class OneConstraint(DummyProblem):
            n_constraints = 1

        solver = ipopt_solver.IPOPTSolver({"constr_viol_tol": 1e-4,
                                           "ipopt_constraint_slack_frac": 0.5})
        with mock.patch.object(ipopt_solver.cyipopt, "Problem", FakeProblem):
            try:
                solver.solve_single(OneConstraint(), timeout=1.0)
            except Exception:  # noqa: BLE001 - only the options matter here
                pass
        self.assertTrue(np.allclose(seen['cu'], [5e-5]))
        # cu + Ipopt's own violation tolerance stays inside the acceptance check.
        self.assertLessEqual(5e-5 + seen['constr_viol_tol'], 1e-4)


class TestIPOPTSlackFallback(unittest.TestCase):
    def test_rejected_slack_solution_is_resolved_without_slack(self):
        """Ipopt's own violation measure is not a bound on max g: waterund18 came back
        at 1.13e-4 with slack 0.5 and was rejected. Re-solve once without slack."""
        from nlp_solver import ipopt_solver

        if not ipopt_solver.CYIPOPT_AVAILABLE:
            self.skipTest("cyipopt not available")
        calls = []

        class OneConstraint(DummyProblem):
            n_constraints = 1

            def constraints(self, x):
                return np.array([x[0] - 0.5], dtype=np.float64)

        class FakeProblem:
            def __init__(self, *args, **kwargs):
                calls.append(float(kwargs['cu'][0]))

            def add_option(self, key, value):
                pass

            def solve(self, x0):
                # First (slack) solve lands just outside the tolerance, second inside.
                x = np.array([0.5 + (2e-4 if len(calls) == 1 else 0.0)])
                return x, {"status": 0, "status_msg": b"ok"}

        solver = ipopt_solver.IPOPTSolver({"constr_viol_tol": 1e-4,
                                           "ipopt_constraint_slack_frac": 0.5})
        with mock.patch.object(ipopt_solver.cyipopt, "Problem", FakeProblem), \
                mock.patch.object(solver, "_pytorch_objective_and_constraints",
                                  lambda prob, x: (float(x[0] ** 2),
                                                   np.array([x[0] - 0.5]))):
            result = solver.solve_single(OneConstraint(), timeout=10.0)
        self.assertEqual(len(calls), 2)
        self.assertTrue(np.isclose(calls[0], 5e-5) and calls[1] == 0.0)
        self.assertIsNotNone(result.x_solution)


class TestIPOPTVectorizedJacobian(unittest.TestCase):
    def test_vectorized_jacobian_matches_the_row_loop(self):
        """One backward pass (plus a device sync) per constraint row cost 0.13-0.23 s
        per Jacobian on GPU for 50-100 rows; the root IPOPT then ran out of time on
        81 of the 96 paper instances without a primal. The vectorized Jacobian must be
        identical to the row loop."""
        import torch
        from nlp_solver.ipopt_solver import NNProblemIPOPT
        from nlp_solver.problem import NNProblemSingleDomain

        class Net(torch.nn.Module):
            def forward(self, x):
                return torch.cat([(x ** 2).sum(-1, keepdim=True), x[:, :1] * x[:, 1:2],
                                  torch.sin(x), x.sum(-1, keepdim=True) ** 3], dim=-1)

        torch.manual_seed(0)
        dm_l = -torch.ones(1, 3, dtype=torch.float64)
        dm_u = torch.ones(1, 3, dtype=torch.float64)
        C = torch.eye(6, dtype=torch.float64)
        rhs = torch.zeros(6, dtype=torch.float64)
        A = torch.randn(1, 2, 3, dtype=torch.float64)
        b = torch.randn(1, 2, dtype=torch.float64)
        args = (Net(), dm_l, dm_u, C[:1], rhs[:1], C[1:], rhs[1:])
        prob = NNProblemIPOPT(*args, input_constr_A=A, input_constr_b=b, device='cpu',
                              dtype=torch.float64)
        ref = NNProblemSingleDomain(*args, input_constr_A=A, input_constr_b=b,
                                    device='cpu', dtype=torch.float64)
        x = np.array([0.3, -0.7, 0.5])
        np.testing.assert_allclose(prob.jacobian(x), ref.jacobian(x), rtol=0, atol=1e-14)
        np.testing.assert_allclose(prob.constraints(x), ref.constraints(x), rtol=0, atol=1e-14)
        self.assertAlmostEqual(prob.objective(x), ref.objective(x), places=14)


if __name__ == "__main__":
    unittest.main()


class TestIPOPTSlackPolish(unittest.TestCase):
    """A high slack target is reached by polishing an accepted half-slack point: an
    interrupted solve aimed straight at 0.99 * tol can stop just outside the
    tolerance and be rejected (pooling_foulds5tp lost its incumbent that way)."""

    def _run(self, second_x):
        from nlp_solver import ipopt_solver
        if not ipopt_solver.CYIPOPT_AVAILABLE:
            self.skipTest("cyipopt not available")
        calls = []

        class FakeProblem:
            def __init__(self, *args, **kwargs):
                calls.append(float(kwargs['cu'][0]))

            def add_option(self, key, value):
                pass

            def solve(self, x0):
                x = 0.50004 if len(calls) == 1 else second_x
                return np.array([x], dtype=np.float64), {"status": 0, "status_msg": b"ok"}

        class OneConstraint(DummyProblem):
            n_constraints = 1

            def objective(self, x):
                return float(-x[0])

            def constraints(self, x):
                return np.array([x[0] - 0.5], dtype=np.float64)

            def jacobian(self, x):
                return np.ones((1, 1), dtype=np.float64)

        solver = ipopt_solver.IPOPTSolver({"constr_viol_tol": 1e-4,
                                           "ipopt_constraint_slack_frac": 0.99})
        with mock.patch.object(ipopt_solver.cyipopt, "Problem", FakeProblem):
            result = solver.solve_single(OneConstraint(), timeout=10.0)
        return calls, result

    def test_polish_to_the_high_target_is_kept_when_accepted(self):
        calls, result = self._run(second_x=0.500098)
        self.assertTrue(np.isclose(calls[0], 5e-5) and np.isclose(calls[1], 9.9e-5))
        self.assertAlmostEqual(float(result.x_solution[0]), 0.500098)

    def test_rejected_polish_keeps_the_half_slack_point(self):
        calls, result = self._run(second_x=0.50012)
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(float(result.x_solution[0]), 0.50004)


class TestIPOPTSparseJacobian(unittest.TestCase):
    """Ipopt gets the Jacobian's structural nonzeros instead of a dense pattern. With a
    dense pattern MUMPS factors a dense KKT system every iteration; on pooling_foulds3stp
    (832 vars, 2214 rows, a few variables per row) 60 s gave no feasible point."""

    def test_structure_is_the_nonzero_pattern_and_values_follow_it(self):
        from nlp_solver import ipopt_solver
        if not ipopt_solver.CYIPOPT_AVAILABLE:
            self.skipTest("cyipopt not available")
        captured = {}

        class FakeProblem:
            def __init__(self, *args, **kwargs):
                captured['obj'] = kwargs['problem_obj']

            def add_option(self, key, value):
                pass

            def solve(self, x0):
                return np.asarray(x0, dtype=np.float64), {"status": 0, "status_msg": b"ok"}

        class Sparse(DummyProblem):
            n_vars = 4
            n_constraints = 3
            lb = np.full(4, -1.0)
            ub = np.full(4, 2.0)

            def objective(self, x):
                return float((x ** 2).sum())

            def gradient(self, x):
                return 2 * x

            def constraints(self, x):
                return np.array([x[0] * x[1] - 5, x[2] ** 2 - 5, x[3] + x[0] - 5])

            def jacobian(self, x):
                J = np.zeros((3, 4))
                J[0, 0], J[0, 1] = x[1], x[0]
                J[1, 2] = 2 * x[2]
                J[2, 3], J[2, 0] = 1.0, 1.0
                return J

        solver = ipopt_solver.IPOPTSolver({"constr_viol_tol": 1e-4})
        with mock.patch.object(ipopt_solver.cyipopt, "Problem", FakeProblem):
            solver.solve_single(Sparse(), timeout=5.0)
        rows, cols = captured['obj'].jacobianstructure()
        self.assertEqual(sorted(zip(rows.tolist(), cols.tolist())),
                         [(0, 0), (0, 1), (1, 2), (2, 0), (2, 3)])
        x = np.array([0.3, -0.7, 1.5, 0.2])
        vals = captured['obj'].jacobian(x)
        dense = Sparse().jacobian(x)
        np.testing.assert_allclose(vals, dense[rows, cols])


class TestCompressedJacobian(unittest.TestCase):
    """With a known sparsity pattern the Jacobian is formed from one forward-mode
    product per column colour instead of one reverse pass per row (foulds3stp: 568 ms
    per dense Jacobian, 0.48% nonzeros). Values must equal the dense Jacobian."""

    def test_compressed_equals_dense(self):
        import torch
        from nlp_solver.ipopt_solver import NNProblemIPOPT

        class Net(torch.nn.Module):
            def forward(self, x):
                return torch.cat([(x ** 2).sum(-1, keepdim=True), x[:, :1] * x[:, 1:2],
                                  x[:, 2:3] ** 2, x[:, 3:4] + x[:, :1],
                                  torch.sin(x[:, 1:2]) * x[:, 2:3], x[:, 4:5] * x[:, 5:6]], dim=-1)

        n = 6
        dm_l, dm_u = -torch.ones(1, n, dtype=torch.float64), torch.ones(1, n, dtype=torch.float64)
        C = torch.eye(6, dtype=torch.float64)
        A = torch.tensor([[[1.0, 0, 0, 0, 0, 2.0]]], dtype=torch.float64)
        b = torch.tensor([[0.5]], dtype=torch.float64)
        args = (Net(), dm_l, dm_u, C[:1], torch.zeros(1, dtype=torch.float64), C[1:],
                torch.zeros(5, dtype=torch.float64))
        prob = NNProblemIPOPT(*args, input_constr_A=A, input_constr_b=b, device='cpu',
                              dtype=torch.float64)
        x = np.array([0.3, -0.7, 0.5, 0.1, -0.2, 0.9])
        dense = prob.jacobian(x)
        rows, cols = np.nonzero(dense != 0)
        prob.set_jacobian_sparsity(rows, cols)
        self.assertLess(prob._jac_ncolors, n)
        np.testing.assert_allclose(prob.jacobian(x), dense, rtol=0, atol=1e-14)
        x2 = np.array([-0.4, 0.2, -0.9, 0.6, 0.8, -0.1])
        prob_ref = NNProblemIPOPT(*args, input_constr_A=A, input_constr_b=b, device='cpu',
                                  dtype=torch.float64)
        np.testing.assert_allclose(prob.jacobian(x2), prob_ref.jacobian(x2), rtol=0, atol=1e-14)


class TestStructuralJacobianPattern(unittest.TestCase):
    """The Jacobian pattern comes from the structure walk's supports instead of dense
    probe Jacobians (polygon100: 374 s of probes against a 30 s cap; 0.3 s from the
    walk). It must contain every nonzero the Jacobian has anywhere."""

    @staticmethod
    def _problem(net, n=6, rows=5, with_input_rows=True):
        import torch
        from nlp_solver.ipopt_solver import NNProblemIPOPT
        C = torch.eye(rows + 1, dtype=torch.float64)
        A = torch.tensor([[[1.0, 0, 0, 0, 0, 2.0]]], dtype=torch.float64) if with_input_rows else None
        b = torch.tensor([[0.5]], dtype=torch.float64) if with_input_rows else None
        return NNProblemIPOPT(net, -torch.ones(1, n, dtype=torch.float64),
                              torch.ones(1, n, dtype=torch.float64), C[:1],
                              torch.zeros(1, dtype=torch.float64), C[1:],
                              torch.zeros(rows, dtype=torch.float64),
                              input_constr_A=A, input_constr_b=b, device='cpu',
                              dtype=torch.float64)

    class Net(__import__('torch').nn.Module):
        def forward(self, x):
            import torch
            return torch.cat([(x ** 2).sum(-1, keepdim=True), x[:, :1] * x[:, 1:2],
                              x[:, 2:3] ** 2, x[:, 3:4] + x[:, :1],
                              torch.sin(x[:, 1:2]) * x[:, 2:3], x[:, 4:5] * x[:, 5:6]], dim=-1)

    def test_pattern_covers_the_jacobian_without_dense_probes(self):
        from nlp_solver.ipopt_solver import IPOPTSolver
        prob = self._problem(self.Net())
        probes = []
        dense_jacobian = prob.jacobian
        prob.jacobian = lambda x: probes.append(1) or dense_jacobian(x)
        x0 = np.zeros(6)
        rows, cols = IPOPTSolver._jacobian_pattern(prob, x0, float('inf'))
        self.assertEqual(probes, [])
        found = set(zip(rows.tolist(), cols.tolist()))
        rng = np.random.default_rng(1)
        for _ in range(5):
            jac = dense_jacobian(rng.uniform(-1, 1, 6))
            self.assertLessEqual(set(zip(*np.nonzero(jac))), found)
        # constraint row 1 is x2^2 -> {2}; the input row (row 5) touches columns 0 and 5
        self.assertIn((1, 2), found)
        self.assertNotIn((1, 0), found)
        self.assertEqual({c for r, c in found if r == 5}, {0, 5})

    def test_the_support_matrix_is_cached_on_the_model(self):
        net = self.Net()
        self._problem(net)
        first = self._problem(net)
        first.structural_jacobian_pattern(np.zeros(6))
        cached = net._ipopt_output_supports
        second = self._problem(net)
        second.structural_jacobian_pattern(np.zeros(6))
        self.assertIs(net._ipopt_output_supports, cached)

    def test_unfollowable_model_falls_back_to_probing(self):
        import torch
        from nlp_solver.ipopt_solver import IPOPTSolver

        class Opaque(torch.nn.Module):
            def forward(self, x):
                return torch.cat([x.sum(-1, keepdim=True), torch.floor(x[:, :1] + 1.5) * x[:, 1:2],
                                  x[:, 2:3] ** 2, x[:, 3:4], x[:, 4:5], x[:, 5:6]], dim=-1)

        prob = self._problem(Opaque(), with_input_rows=False)
        self.assertIsNone(prob.structural_jacobian_pattern(np.zeros(6)))
        rows, cols = IPOPTSolver._jacobian_pattern(prob, np.full(6, 0.3), float('inf'))
        self.assertIn((0, 1), set(zip(rows.tolist(), cols.tolist())))

    def test_probing_stops_at_the_deadline(self):
        from nlp_solver.ipopt_solver import IPOPTSolver
        prob = self._problem(self.Net())
        prob.structural_jacobian_pattern = lambda x: None
        self.assertIsNone(IPOPTSolver._jacobian_pattern(prob, np.zeros(6), 0.0))
        self.assertIsNone(getattr(prob, '_ipopt_jac_pattern', None))

    def test_setup_past_the_deadline_returns_no_solution(self):
        from nlp_solver import ipopt_solver
        if not ipopt_solver.CYIPOPT_AVAILABLE:
            self.skipTest("cyipopt not available")
        prob = self._problem(self.Net())
        prob.structural_jacobian_pattern = lambda x: None
        result = ipopt_solver.IPOPTSolver({}).solve_single(prob, x_init=np.zeros(6), timeout=1e-3)
        self.assertFalse(result.success)
        self.assertIsNone(result.x_solution)
        self.assertIn("setup exceeded the deadline", result.status)
