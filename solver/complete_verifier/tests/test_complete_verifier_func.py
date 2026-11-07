#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""
Unit tests for complete_verifier_func.py

This module tests the complete verifier main interface functions:
- bab: Branch-and-bound verification main function
- complete_verifier: Complete verification interface
- prepare_for_act_bab: Prepare batches for activation BaB
- _format_result_act_bab: Format BaB results

To run tests: python -m pytest complete_verifier/tests/test_complete_verifier_func.py -v
"""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import torch
import sys
import os
import tempfile
from contextlib import redirect_stdout
from io import StringIO

# Add the complete_verifier to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def setup_module():
    """Setup arguments.Config for all tests."""
    import arguments
    global original_config
    original_config = arguments.Config
    new_config = arguments.ConfigHandler()
    new_config.construct_config_dict(new_config.default_args)
    new_config.file = None
    arguments.Config = new_config


def teardown_module():
    """Restore original config."""
    import arguments
    arguments.Config = original_config


class TestFixedPrimalObjective(unittest.TestCase):
    """Tests for the activation-BaB fixed-incumbent loader."""

    def setUp(self):
        import arguments

        arguments.Config["solving"]["enable_solver_res"] = False
        arguments.Config["solving"]["solver_res"] = ""

    def test_loads_scalar_with_reference_dtype(self):
        import arguments
        from solving_utils import load_fixed_primal

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            torch.save(3.25, tmp.name)
            arguments.Config["solving"]["enable_solver_res"] = True
            arguments.Config["solving"]["solver_res"] = tmp.name

            result = load_fixed_primal(
                torch.zeros(1, dtype=torch.float64)
            )

        self.assertEqual(result.dtype, torch.float64)
        self.assertEqual(result.item(), 3.25)

    def test_disabled_loader_returns_none(self):
        from solving_utils import load_fixed_primal

        result = load_fixed_primal(torch.zeros(1))

        self.assertIsNone(result)

    def test_rejects_non_scalar_result(self):
        import arguments
        from solving_utils import load_fixed_primal

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            torch.save(torch.tensor([1.0, 2.0]), tmp.name)
            arguments.Config["solving"]["enable_solver_res"] = True
            arguments.Config["solving"]["solver_res"] = tmp.name

            with self.assertRaisesRegex(ValueError, "scalar"):
                load_fixed_primal(torch.zeros(1))

    def test_rejects_nonfinite_result(self):
        import arguments
        from solving_utils import load_fixed_primal

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            torch.save(float("inf"), tmp.name)
            arguments.Config["solving"]["enable_solver_res"] = True
            arguments.Config["solving"]["solver_res"] = tmp.name

            with self.assertRaisesRegex(ValueError, "finite"):
                load_fixed_primal(torch.zeros(1))

    def test_requires_configured_path(self):
        import arguments
        from solving_utils import load_fixed_primal

        arguments.Config["solving"]["enable_solver_res"] = True

        with self.assertRaisesRegex(ValueError, "solver_res is required"):
            load_fixed_primal(torch.zeros(1))






class TestCompleteVerifier(unittest.TestCase):
    """Tests for the complete_verifier function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_abcrown = MagicMock()
        self.mock_all_specs = MagicMock()
        self.mock_abcrown.vnnlib_handler.all_specs = self.mock_all_specs

    @patch('complete_verifier_func.arguments.Config', {
        'solver': {'beta-crown': {'enable_opt_interm_bounds': False}},
    })
    @patch('complete_verifier_func.time.time')
    @patch('complete_verifier_func.torch.cuda.empty_cache')
    @patch('complete_verifier_func.gc.collect')
    def test_complete_verifier_safe_result(self, mock_gc, mock_empty_cache, mock_time):
        """Test complete_verifier with safe result."""
        from complete_verifier_func import complete_verifier

        mock_time.return_value = 1  # Fixed time for simplicity

        self.mock_abcrown.bab = MagicMock(return_value=(0.5, 100, 'safe'))
        model_ori = MagicMock()
        bab_ret = []

        result = complete_verifier(
            self.mock_abcrown, model_ori,
            index=0, timeout_threshold=100,
            bab_ret=bab_ret, reference_dict={}
        )

        self.assertEqual(result, 'safe')
        self.assertEqual(len(bab_ret), 1)
        mock_empty_cache.assert_called_once()
        mock_gc.assert_called_once()

    @patch('complete_verifier_func.arguments.Config', {
        'solver': {'beta-crown': {'enable_opt_interm_bounds': False}},
    })
    @patch('complete_verifier_func.time.time')
    @patch('complete_verifier_func.torch.cuda.empty_cache')
    @patch('complete_verifier_func.gc.collect')
    def test_complete_verifier_unsafe_result(self, mock_gc, mock_empty_cache, mock_time):
        """Test complete_verifier with unsafe result."""
        from complete_verifier_func import complete_verifier

        mock_time.return_value = 1

        self.mock_abcrown.bab = MagicMock(return_value=(-1.0, 50, 'unsafe'))
        model_ori = MagicMock()
        bab_ret = []

        result = complete_verifier(
            self.mock_abcrown, model_ori,
            index=0, timeout_threshold=100,
            bab_ret=bab_ret, reference_dict={}
        )

        self.assertEqual(result, 'unsafe-bab')

    @patch('complete_verifier_func.arguments.Config', {
        'solver': {'beta-crown': {'enable_opt_interm_bounds': False}},
    })
    @patch('complete_verifier_func.time.time')
    @patch('complete_verifier_func.torch.cuda.empty_cache')
    @patch('complete_verifier_func.gc.collect')
    def test_complete_verifier_unknown_result(self, mock_gc, mock_empty_cache, mock_time):
        """Test complete_verifier with unknown result."""
        from complete_verifier_func import complete_verifier

        mock_time.return_value = 1

        self.mock_abcrown.bab = MagicMock(return_value=(-0.5, 200, 'unknown'))
        model_ori = MagicMock()
        bab_ret = []

        result = complete_verifier(
            self.mock_abcrown, model_ori,
            index=0, timeout_threshold=100,
            bab_ret=bab_ret, reference_dict={}
        )

        self.assertEqual(result, 'unknown')

    @patch('complete_verifier_func.arguments.Config', {
        'solver': {'beta-crown': {'enable_opt_interm_bounds': False}},
    })
    @patch('complete_verifier_func.time.time')
    @patch('complete_verifier_func.torch.cuda.empty_cache')
    @patch('complete_verifier_func.gc.collect')
    def test_complete_verifier_timeout(self, mock_gc, mock_empty_cache, mock_time):
        """Test complete_verifier when timeout is exceeded."""
        from complete_verifier_func import complete_verifier

        # Simulate that bab takes longer than timeout - use side_effect with enough values
        mock_time.side_effect = [0, 1, 102, 103, 104, 105]  # Last time > timeout_threshold

        self.mock_abcrown.bab = MagicMock(return_value=(0.1, 500, 'safe'))
        model_ori = MagicMock()
        bab_ret = []

        result = complete_verifier(
            self.mock_abcrown, model_ori,
            index=0, timeout_threshold=100,
            bab_ret=bab_ret, reference_dict={}
        )

        # Should return unknown due to timeout
        self.assertEqual(result, 'unknown')

    @patch('complete_verifier_func.arguments.Config', {
        'solver': {'beta-crown': {'enable_opt_interm_bounds': False}},
    })
    @patch('complete_verifier_func.time.time')
    @patch('complete_verifier_func.torch.cuda.empty_cache')
    @patch('complete_verifier_func.gc.collect')
    def test_complete_verifier_invalid_return_raises(self, mock_gc, mock_empty_cache, mock_time):
        """Test complete_verifier raises on invalid bab return value."""
        from complete_verifier_func import complete_verifier

        mock_time.return_value = 1

        self.mock_abcrown.bab = MagicMock(return_value=(0.5, 100, 'invalid_status'))
        model_ori = MagicMock()
        bab_ret = []

        with self.assertRaises(ValueError) as context:
            complete_verifier(
                self.mock_abcrown, model_ori,
                index=0, timeout_threshold=100,
                bab_ret=bab_ret, reference_dict={}
            )

        self.assertIn('invalid_status', str(context.exception))

    @patch('complete_verifier_func.arguments.Config', {
        'solver': {'beta-crown': {'enable_opt_interm_bounds': False}},
    })
    @patch('complete_verifier_func.time.time')
    @patch('complete_verifier_func.torch.cuda.empty_cache')
    @patch('complete_verifier_func.gc.collect')
    def test_complete_verifier_records_bab_stats(self, mock_gc, mock_empty_cache, mock_time):
        """Test complete_verifier records stats in bab_ret."""
        from complete_verifier_func import complete_verifier

        mock_time.return_value = 1

        self.mock_abcrown.bab = MagicMock(return_value=(0.5, 100, 'safe'))
        model_ori = MagicMock()
        bab_ret = []

        complete_verifier(
            self.mock_abcrown, model_ori,
            index=5, timeout_threshold=100,
            bab_ret=bab_ret, reference_dict={}
        )

        # Check bab_ret contains [index, lb, num_domains, time]
        self.assertEqual(len(bab_ret), 1)
        self.assertEqual(bab_ret[0][0], 5)  # index
        self.assertEqual(bab_ret[0][1], 0.5)  # lb
        self.assertEqual(bab_ret[0][2], 100)  # num_domains
        # time is 0 because mock_time.return_value is constant
        self.assertIsInstance(bab_ret[0][3], (int, float))  # time

    @patch('complete_verifier_func.arguments.Config', {
        'solver': {'beta-crown': {'enable_opt_interm_bounds': True}},  # Will trigger assertion
    })
    @patch('complete_verifier_func.time.time')
    @patch('complete_verifier_func.torch.cuda.empty_cache')
    @patch('complete_verifier_func.gc.collect')
    def test_complete_verifier_asserts_opt_interm_bounds(self, mock_gc, mock_empty_cache, mock_time):
        """Test complete_verifier asserts enable_opt_interm_bounds is False."""
        from complete_verifier_func import complete_verifier

        mock_time.return_value = 1

        self.mock_abcrown.bab = MagicMock(return_value=(0.5, 100, 'safe'))
        model_ori = MagicMock()
        bab_ret = []

        with self.assertRaises(AssertionError):
            complete_verifier(
                self.mock_abcrown, model_ori,
                index=0, timeout_threshold=100,
                bab_ret=bab_ret, reference_dict={}
            )

    @patch('complete_verifier_func.arguments.Config', {
        'solver': {'beta-crown': {'enable_opt_interm_bounds': False}},
    })
    @patch('complete_verifier_func.time.time')
    @patch('complete_verifier_func.torch.cuda.empty_cache')
    @patch('complete_verifier_func.gc.collect')
    @patch('builtins.print')
    def test_complete_verifier_prints_progress(self, mock_print, mock_gc, mock_empty_cache, mock_time):
        """Test complete_verifier prints verification progress."""
        from complete_verifier_func import complete_verifier

        mock_time.return_value = 1

        self.mock_abcrown.bab = MagicMock(return_value=(0.5, 100, 'safe'))
        model_ori = MagicMock()
        bab_ret = []

        complete_verifier(
            self.mock_abcrown, model_ori,
            index=3, timeout_threshold=100,
            bab_ret=bab_ret, reference_dict={}
        )

        # Check that progress was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        found_instance_print = any('Instance 3' in call for call in print_calls)
        self.assertTrue(found_instance_print)




class TestShareSpecMatrix(unittest.TestCase):
    """bab() takes the incomplete verifier's copy of the spec matrix when it is the same."""

    def _matrix(self):
        torch.manual_seed(0)
        return torch.randn(1, 4, 3, dtype=torch.float64)

    def test_equal_matrix_is_shared(self):
        from complete_verifier_func import _share_spec_matrix
        c = self._matrix()
        model = MagicMock(c=c.clone())
        shared = _share_spec_matrix(model, c)
        self.assertIs(shared, model.c)
        self.assertTrue(torch.equal(shared, c))

    def test_different_values_shape_or_dtype_keep_the_own_matrix(self):
        from complete_verifier_func import _share_spec_matrix
        c = self._matrix()
        other = c.clone()
        other[0, 0, 0] += 1.0
        for model_c in (other, c[:, :3].clone(), c.float(), None):
            self.assertIs(_share_spec_matrix(MagicMock(c=model_c), c), c)


if __name__ == '__main__':
    unittest.main()
