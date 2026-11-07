"""Unit tests for auto_LiRPA/concretize_func.py."""
import os
import sys
import unittest

import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_LiRPA.concretize_func import (
    constraints_solving,
    drop_redundant_constraints,
    sort_out_constr_batches,
)


class TestSortOutConstrBatchesDiscardAbnormal(unittest.TestCase):
    """Tests for solving-mode abnormal-constraint sanitization."""

    def test_only_abnormal_objective_rows_are_neutralized(self):
        x_L = torch.zeros(2, 2)
        x_U = torch.ones(2, 2)
        constraints_A = torch.tensor([
            [[float("nan"), 1.0], [2.0, 3.0]],
            [[4.0, 5.0], [6.0, 7.0]],
        ])
        constraints_b = torch.zeros(2, 2)

        constraints, sorted_out_batches = sort_out_constr_batches(
            x_L,
            x_U,
            (constraints_A, constraints_b),
            discard_abnormal=True,
        )

        self.assertIsNotNone(sorted_out_batches)
        self.assertIsNotNone(constraints)

        sanitized_A, sanitized_d = constraints
        self.assertEqual(sanitized_A.shape, constraints_A.shape)
        self.assertTrue(torch.equal(sanitized_A[0, 0], torch.zeros(2)))
        self.assertEqual(sanitized_d[0, 0].item(), -1.0)
        self.assertTrue(torch.equal(sanitized_A[0, 1], constraints_A[0, 1]))
        self.assertTrue(torch.equal(sanitized_A[1], constraints_A[1]))


class TestDropRedundantConstraints(unittest.TestCase):
    """Rows that hold on the whole box are dropped without changing any bound."""

    def test_keeps_binding_rows_in_order(self):
        # Centred form a^T y + d <= 0 on |y| <= 1: a row binds iff d + |a|_1 > 0.
        A = torch.tensor([[[1., 0.], [1., 1.], [0., 1.]],
                          [[0., 1.], [1., 1.], [1., 0.]]])
        d = torch.tensor([[0.5, -5., 0.5],
                          [-5., 0.5, -5.]])
        A_kept, d_kept = drop_redundant_constraints(A, d, torch.ones(2, 2, 1))

        self.assertEqual(A_kept.shape, (2, 2, 2))
        self.assertTrue(torch.equal(A_kept[0], A[0, [0, 2]]))
        self.assertTrue(torch.equal(d_kept[0], d[0, [0, 2]]))
        self.assertTrue(torch.equal(A_kept[1, 0], A[1, 1]))
        # The second batch is padded with one of its own redundant rows.
        self.assertLessEqual((d_kept[1, 1] + A_kept[1, 1].abs().sum()).item(), 0.)

    def test_uncentred_rows_use_x0(self):
        A = torch.tensor([[[1., 0.], [1., 0.]]])
        b = torch.tensor([[-3.5, -1.5]])        # x_0 <= 3.5 holds on [1, 3], x_0 <= 1.5 does not
        x0 = torch.full((1, 2, 1), 2.)
        A_kept, b_kept = drop_redundant_constraints(A, b, torch.ones(1, 2, 1), x0)
        self.assertTrue(torch.equal(b_kept, torch.tensor([[-1.5]])))

    def test_bounds_unchanged(self):
        torch.manual_seed(0)
        B, X, H = 4, 5, 3
        x_L, x_U = -torch.rand(B, X), torch.rand(B, X)
        x0, eps = (x_U + x_L) / 2, (x_U - x_L) / 2
        binding_A = torch.randn(B, 2, X)
        binding_b = -(binding_A @ x0.unsqueeze(-1)).squeeze(-1)     # hyperplanes through the centre
        slack_A = torch.randn(B, 6, X)
        slack_b = (-(slack_A @ x0.unsqueeze(-1)).squeeze(-1)
                   - (slack_A.abs() * eps.unsqueeze(1)).sum(-1) - 0.1)
        objective = torch.randn(B, H, X)

        def solve(A, b):
            constraints, sorted_out = sort_out_constr_batches(x_L, x_U, (A, b))
            return constraints_solving(x_L, x_U, objective, constraints, -1.0,
                                       sorted_out_batches=sorted_out, use_grad=False)

        with_slack = solve(torch.cat([slack_A[:, :3], binding_A, slack_A[:, 3:]], 1),
                           torch.cat([slack_b[:, :3], binding_b, slack_b[:, 3:]], 1))
        self.assertTrue(torch.equal(with_slack, solve(binding_A, binding_b)))




class TestConstraintRowCap(unittest.TestCase):
    """At most k binding rows per domain, the most binding first. Any subset of the
    constraints gives a valid (possibly weaker) bound; the sweep cost is linear in the
    number of rows kept (wastewater04m2: 212 rows, 90% of BaB time in this sweep)."""

    def test_keeps_the_most_binding_rows(self):
        A = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]], dtype=torch.float64)
        d = torch.tensor([[0.2, 0.5, 0.1]], dtype=torch.float64)   # worst case 1.2, 1.5, 2.1
        A_kept, d_kept = drop_redundant_constraints(
            A, d, torch.ones(1, 2, 1, dtype=torch.float64), max_rows=2)
        self.assertEqual(A_kept.shape, (1, 2, 2))
        self.assertEqual(sorted(d_kept[0].tolist()), [0.1, 0.5])

    def test_capped_bound_is_valid_and_between_naive_and_full(self):
        torch.manual_seed(1)
        B, X, H = 6, 5, 3
        x_L, x_U = -torch.rand(B, X), torch.rand(B, X)
        x0 = (x_U + x_L) / 2
        A = torch.randn(B, 8, X)
        b = -(A @ x0.unsqueeze(-1)).squeeze(-1) + 0.05 * torch.randn(B, 8)
        objective = torch.randn(B, H, X)

        def solve(constraints_on, cap=None):
            import auto_LiRPA.concretize_func as cf
            old = cf.MAX_CONSTRAINT_ROWS
            cf.MAX_CONSTRAINT_ROWS = cap
            try:
                constraints, sorted_out = sort_out_constr_batches(x_L, x_U, (A, b))
                return constraints_solving(x_L, x_U, objective, constraints, -1.0,
                                           sorted_out_batches=sorted_out, use_grad=False,
                                           constraints_enable=constraints_on)
            finally:
                cf.MAX_CONSTRAINT_ROWS = old

        full, capped, naive = solve(True), solve(True, cap=2), solve(False)
        finite = torch.isfinite(full) & torch.isfinite(capped)
        self.assertTrue(torch.all(capped[finite] >= naive[finite] - 1e-9))
        # Validity: no feasible sample beats the capped lower bound.
        for i in range(B):
            z = x_L[i] + (x_U[i] - x_L[i]) * torch.rand(20000, X)
            feas = ((z @ A[i].T + b[i]) <= 0).all(1)
            if feas.any():
                vals = z[feas] @ objective[i].T
                self.assertTrue(torch.all(vals.min(0).values >= capped[i, :, 0] - 1e-9))


class TestRowBlockedDenseProducts(unittest.TestCase):
    """|A| @ v is formed in blocks of rows under DENSE_BLOCK_ELEMS, with the same result."""

    def setUp(self):
        import auto_LiRPA.fp_enclosure as fp
        self.fp = fp
        self.saved = fp.DENSE_BLOCK_ELEMS

    def tearDown(self):
        self.fp.DENSE_BLOCK_ELEMS = self.saved

    def _matrix(self):
        torch.manual_seed(3)
        A = torch.randn(2, 11, 5, dtype=torch.float64)
        A[0, 3, 1] = float("inf")           # excluded by finite_abs, kept by abs
        A[1, 7, 2] = float("nan")
        return A, torch.rand(2, 5, 1, dtype=torch.float64)

    def test_blocks_equal_the_single_product(self):
        A, v = self._matrix()
        whole = {f: self.fp.abs_matvec(A, v, finite=f) for f in (True, False)}
        self.fp.DENSE_BLOCK_ELEMS = 3 * 2 * 5               # 3 rows of the batch per block
        for finite in (True, False):
            blocked = self.fp.abs_matvec(A, v, finite=finite)
            self.assertEqual(blocked.shape, (2, 11, 1))
            torch.testing.assert_close(blocked, whole[finite], rtol=0, atol=0, equal_nan=True)

    def test_a_block_never_exceeds_the_budget(self):
        A, v = self._matrix()
        self.fp.DENSE_BLOCK_ELEMS = 4 * 2 * 5
        seen = []
        real = self.fp.finite_abs
        self.fp.finite_abs = lambda value: (seen.append(value.shape), real(value))[1]
        try:
            self.fp.abs_matvec(A, v)
        finally:
            self.fp.finite_abs = real
        self.assertEqual([shape[1] for shape in seen], [4, 4, 3])
        self.assertTrue(all(shape[0] * shape[1] * shape[2] <= 4 * 2 * 5 for shape in seen))

    def test_sort_out_is_unchanged_by_blocking(self):
        torch.manual_seed(5)
        B, N, X = 3, 13, 6
        x_L, x_U = -torch.rand(B, X, dtype=torch.float64), torch.rand(B, X, dtype=torch.float64)
        A = torch.randn(B, N, X, dtype=torch.float64)
        b = 0.3 * torch.randn(B, N, dtype=torch.float64)
        whole, whole_sorted = sort_out_constr_batches(x_L, x_U, (A, b))
        self.fp.DENSE_BLOCK_ELEMS = 2 * B * X
        blocked, blocked_sorted = sort_out_constr_batches(x_L, x_U, (A, b))
        self.assertTrue(torch.equal(whole[0], blocked[0]))
        self.assertTrue(torch.equal(whole[1], blocked[1]))
        self.assertTrue(torch.equal(whole_sorted["active_indices"], blocked_sorted["active_indices"]))


if __name__ == "__main__":
    unittest.main()
