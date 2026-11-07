"""Tests for selecting batch rows without advanced indexing."""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_LiRPA.utils import select_rows


class TestSelectRows(unittest.TestCase):
    def test_index_tensor(self):
        tensor = torch.arange(20, dtype=torch.float64).reshape(5, 4)
        index = torch.tensor([3, 1, 4])
        self.assertTrue(torch.equal(select_rows(tensor, index), tensor[index]))

    def test_boolean_mask(self):
        tensor = torch.arange(12).reshape(4, 3)
        mask = torch.tensor([False, True, False, True])
        self.assertTrue(torch.equal(select_rows(tensor, mask), tensor[mask]))

    def test_empty_selection(self):
        tensor = torch.arange(12).reshape(4, 3)
        index = torch.empty(0, dtype=torch.long)
        selected = select_rows(tensor, index)
        self.assertTrue(torch.equal(selected, tensor[index]))
        self.assertEqual(selected.shape, (0, 3))

    def test_three_dimensional_input(self):
        tensor = torch.arange(60, dtype=torch.float64).reshape(5, 3, 4)
        index = torch.tensor([4, 0, 2])
        selected = select_rows(tensor, index)
        self.assertTrue(torch.equal(selected, tensor[index]))
        self.assertEqual(selected.dtype, tensor.dtype)
        self.assertEqual(selected.shape, (3, 3, 4))


if __name__ == "__main__":
    unittest.main()
