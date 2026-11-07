"""The root ALM drops the input rows only when the proof-time expansion appended them."""
import unittest

import torch

from nlp_solver.falsification import _ends_with_structural_rows


class TestEndsWithStructuralRows(unittest.TestCase):
    A = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    b = torch.tensor([[1.0, 2.0]])

    def test_no_input_rows(self):
        self.assertTrue(_ends_with_structural_rows(torch.zeros(0, 3), torch.zeros(0), None))

    def test_spec_without_output_rows_is_not_expanded(self):
        # linear-only instances (ex2_1_1, st_e*, ex6_2_*): C_constr is empty
        self.assertFalse(_ends_with_structural_rows(torch.zeros(0, 3), torch.zeros(0), (self.A, self.b)))

    def test_expanded_rows_follow_the_output_rows(self):
        c = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertTrue(_ends_with_structural_rows(c, torch.zeros(3), (self.A, self.b)))
        self.assertFalse(_ends_with_structural_rows(c, torch.tensor([0.0, 0.0, 0.5]), (self.A, self.b)))

    def test_nonlinear_rows_only(self):
        c = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        self.assertFalse(_ends_with_structural_rows(c, torch.zeros(2), (self.A, self.b)))


if __name__ == "__main__":
    unittest.main()
