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
"""Unit tests for input_split/batch_branch_and_bound.py"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import math

import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arguments
from nlp_solver.structure import RowPool


def _setup_mock_config():
    """Create a mock config for testing."""
    new_config = arguments.ConfigHandler()
    new_config.construct_config_dict(new_config.default_args)
    new_config.file = None

    # Set up test-specific values
    new_config['bab']['branching']['input_split']['split_hint'] = None
    new_config['bab']['branching']['input_split']['update_rhs_with_attack'] = False
    new_config['bab']['branching']['input_split']['compare_with_old_bounds'] = False
    new_config['bab']['branching']['input_split']['skip_getting_worst_domain'] = False
    new_config['bab']['branching']['input_split']['show_progress'] = False
    new_config['bab']['branching']['input_split']['reorder_bab'] = False
    new_config['bab']['branching']['input_split']['adv_check'] = -1
    new_config['bab']['branching']['input_split']['split_partitions'] = 2
    new_config['bab']['branching']['input_split']['presplit_domains'] = None
    new_config['bab']['branching']['input_split']['sort_index'] = None
    new_config['bab']['branching']['input_split']['sort_descending'] = True
    new_config['bab']['branching']['input_split']['bf_iters'] = 0
    new_config['bab']['branching']['input_split']['bf_batch_size'] = 1
    new_config['bab']['clip_n_verify']['clip_input_domain']['enabled'] = False
    new_config['bab']['clip_n_verify']['clip_input_domain']['clip_iterations'] = 1
    new_config['bab']['clip_n_verify']['clip_input_domain']['clip_calculate_volume_metrics'] = False
    new_config['bab']['clip_n_verify']['clip_input_domain']['clip_type'] = 'simple'
    new_config['bab']['clip_n_verify']['clip_input_domain']['clip_neuron_selection_value'] = 1.0
    new_config['bab']['clip_n_verify']['clip_input_domain']['clip_neuron_selection_type'] = 'ratio'
    new_config['bab']['timeout'] = 100
    new_config['bab']['max_iterations'] = 10
    new_config['bab']['sort_domain_interval'] = 0
    new_config['bab']['branching']['method'] = 'sb'
    new_config['solver']['batch_size'] = 64
    new_config['solver']['bound_prop_method'] = 'crown'
    new_config['solver']['init_bound_prop_method'] = 'same'
    new_config['solver']['min_batch_size_ratio'] = 0.1
    new_config['solver']['auto_enlarge_batch_size'] = False
    new_config['attack']['pgd_order'] = 'skip'
    new_config['attack']['input_split_check_adv']['enabled'] = 'false'
    new_config['model']['with_jacobian'] = False
    new_config['general']['device'] = 'cpu'

    return new_config


# ============================================================================
# _print_final_results Tests
# ============================================================================

class TestPrintFinalResults(unittest.TestCase):
    """Tests for _print_final_results function."""

    def setUp(self):
        """Set up mock config."""
        self.original_config = arguments.Config
        arguments.Config = _setup_mock_config()

    def tearDown(self):
        """Restore original config."""
        arguments.Config = self.original_config

    @patch('builtins.print')
    def test_print_final_results_basic(self, mock_print):
        """Test _print_final_results outputs correctly."""
        from input_split.batch_branch_and_bound import _print_final_results
        from input_split.utils import Stats

        stats = Stats()
        stats.visited = 100
        stats.timer.start('summary')

        _print_final_results(stats, 50, torch.zeros(1, 1))

        # Verify print was called with expected domain info
        print_calls = [str(call) for call in mock_print.call_args_list]
        # Should print domain length and visited count
        self.assertTrue(any('50' in call for call in print_calls), "Should print domain length")
        self.assertTrue(any('100' in call and 'visited' in call for call in print_calls), "Should print visited count")

    @patch('builtins.print')
    def test_print_final_results_zero_domains(self, mock_print):
        """Test _print_final_results with zero domains."""
        from input_split.batch_branch_and_bound import _print_final_results
        from input_split.utils import Stats

        stats = Stats()
        stats.visited = 0
        stats.timer.start('summary')

        _print_final_results(stats, 0, torch.zeros(1, 1))

        # Verify print was called with zero values
        print_calls = [str(call) for call in mock_print.call_args_list]
        # Should print "Length of domains: 0" and "0 domains visited"
        self.assertTrue(any('0' in call for call in print_calls), "Should print domain length of 0")
        self.assertTrue(any('0' in call and 'visited' in call for call in print_calls), "Should print 0 visited")


class TestInitialVerification(unittest.TestCase):
    """Tests for initial verification logic."""

    def test_initial_verified_true(self):
        """Test initial verification when all bounds verified."""
        from input_split.utils import initial_verify_criterion

        global_lb = torch.tensor([[0.5, 0.3], [0.1, 0.2]])
        rhs = torch.zeros(2, 2)

        initial_verified, remaining_index = initial_verify_criterion(global_lb, rhs)

        self.assertTrue(initial_verified)

    def test_initial_verified_false(self):
        """Test initial verification when not all bounds verified."""
        from input_split.utils import initial_verify_criterion

        # For a sample to be verified, ANY of its lbs - rhs must be > 0
        # First sample: [-0.5, -0.3] - all <= 0, NOT verified
        # Second sample: [0.1, 0.2] - both > 0, verified
        # Since not all samples are verified, initial_verified = False
        global_lb = torch.tensor([[-0.5, -0.3], [0.1, 0.2]])
        rhs = torch.zeros(2, 2)

        initial_verified, remaining_index = initial_verify_criterion(global_lb, rhs)

        self.assertFalse(initial_verified)
        # remaining_index should contain index 0 (unverified sample)
        self.assertEqual(len(remaining_index), 1)
        self.assertEqual(remaining_index[0].item(), 0)

    def test_return_early_when_verified(self):
        """Test return logic when initially verified."""
        initial_verified = True
        global_lb = torch.tensor([[0.5], [0.3]])
        visited = 10

        if initial_verified:
            result = (global_lb.max(), visited, "safe")

        self.assertEqual(result[0].item(), 0.5)
        self.assertEqual(result[1], 10)
        self.assertEqual(result[2], "safe")


class TestPrimalDualConvergence(unittest.TestCase):
    """Tests for primal-dual convergence criterion in solving mode."""

    def test_rejects_relative_allclose_false_positive(self):
        """Should not accept convergence when absolute gap is too large."""
        from input_split.batch_branch_and_bound import _is_primal_dual_converged

        primal = torch.tensor([-268.01516724])
        dual = torch.tensor(-268.0148620605469)

        self.assertFalse(_is_primal_dual_converged(primal, dual, abs_tol=1e-5))

    def test_accepts_small_gap_within_tolerance(self):
        """Should accept convergence when primal-dual gap is tiny."""
        from input_split.batch_branch_and_bound import _is_primal_dual_converged

        primal = torch.tensor([1.000004])
        dual = torch.tensor(1.0)

        self.assertTrue(_is_primal_dual_converged(primal, dual, abs_tol=1e-5))

    def test_rejects_primal_below_dual_beyond_tolerance(self):
        """Should reject unsound case where primal is clearly below dual."""
        from input_split.batch_branch_and_bound import _is_primal_dual_converged

        primal = torch.tensor([-1.0002])
        dual = torch.tensor(-1.0)

        self.assertFalse(_is_primal_dual_converged(primal, dual, abs_tol=1e-5))

    def test_accepts_configured_relative_gap(self):
        """Large objectives may close through the configured relative gap."""
        from input_split.batch_branch_and_bound import _is_primal_dual_converged

        primal = torch.tensor([10000.0])
        dual = torch.tensor([9999.5])

        self.assertTrue(_is_primal_dual_converged(
            primal, dual, abs_tol=1e-5, rel_tol=1e-4
        ))

    def test_rejects_nonfinite_or_vector_values(self):
        from input_split.batch_branch_and_bound import _is_primal_dual_converged

        self.assertFalse(_is_primal_dual_converged(
            torch.tensor(float("inf")), torch.tensor(1.0)
        ))
        with self.assertRaisesRegex(ValueError, "scalar"):
            _is_primal_dual_converged(
                torch.tensor([1.0, 2.0]), torch.tensor(1.0)
            )

    def test_initial_inconsistent_incumbent_is_detected(self):
        from input_split.batch_branch_and_bound import (
            _is_initial_primal_dual_inconsistent,
        )

        self.assertTrue(_is_initial_primal_dual_inconsistent(
            torch.tensor([-675.0]), torch.tensor(-263.7522), abs_tol=1e-5
        ))
        self.assertFalse(_is_initial_primal_dual_inconsistent(
            torch.tensor([1.000004]), torch.tensor(1.0), abs_tol=1e-5
        ))
        self.assertFalse(_is_initial_primal_dual_inconsistent(
            torch.tensor(float("inf")), torch.tensor(1.0), abs_tol=1e-5
        ))


class TestApplyPrimalObjectiveThreshold(unittest.TestCase):
    """Tests for applying finite primal incumbents to objective thresholds."""

    def test_updates_objective_column_with_finite_primal(self):
        """Finite incumbents should overwrite only the objective RHS."""
        from input_split.batch_branch_and_bound import _apply_primal_objective_threshold

        thresholds = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
        updated = _apply_primal_objective_threshold(thresholds, torch.tensor([-4.5]))

        self.assertTrue(torch.equal(updated[:, 0], torch.tensor([-4.5, -4.5])))
        self.assertTrue(torch.equal(updated[:, 1], thresholds[:, 1]))
        self.assertNotEqual(updated.data_ptr(), thresholds.data_ptr())

    def test_nonfinite_primal_disables_objective_cutoff(self):
        """Without an incumbent the objective row must not prune anything.

        The objective RHS starts at 0 (verification-mode heritage). Leaving it there
        prunes every box whose objective lower bound is positive, and an emptied queue
        is then reported as solved: ex5_3_3 was declared solved with no primal while a
        point with violation 1.5e-13 and objective 3.23401 exists.
        """
        from input_split.batch_branch_and_bound import _apply_primal_objective_threshold

        thresholds = torch.tensor([[0.0, 1.0], [0.0, 3.0]])
        updated = _apply_primal_objective_threshold(thresholds, torch.tensor([float("inf")]))

        self.assertTrue(torch.isposinf(updated[:, 0]).all())
        self.assertTrue(torch.equal(updated[:, 1], thresholds[:, 1]))
        self.assertEqual(float(thresholds[0, 0]), 0.0)


class TestPickedObjectiveLowerBounds(unittest.TestCase):
    """The level-set metadata extractor must preserve domain/objective axes."""

    def test_extracts_first_objective_column_for_batched_bounds(self):
        from input_split.batch_branch_and_bound import (
            _extract_picked_objective_lower_bounds,
        )

        bounds = torch.tensor([
            [1.0, -10.0, -20.0],
            [2.0, -30.0, -40.0],
        ])
        extracted = _extract_picked_objective_lower_bounds(
            bounds, batch_size=2
        )

        self.assertTrue(torch.equal(extracted, torch.tensor([1.0, 2.0])))

    def test_single_domain_vector_uses_only_first_objective(self):
        from input_split.batch_branch_and_bound import (
            _extract_picked_objective_lower_bounds,
        )

        extracted = _extract_picked_objective_lower_bounds(
            torch.tensor([3.0, -5.0, -7.0]), batch_size=1
        )

        self.assertTrue(torch.equal(extracted, torch.tensor([3.0])))

    def test_unrecognized_shape_disables_level_metadata(self):
        from input_split.batch_branch_and_bound import (
            _extract_picked_objective_lower_bounds,
        )

        self.assertIsNone(_extract_picked_objective_lower_bounds(
            torch.zeros(3, 2), batch_size=2
        ))


class TestSolvingModeRequeueFiltering(unittest.TestCase):
    """Tests for solving-mode add-domain filtering after clipping."""

    def setUp(self):
        self.original_config = arguments.Config
        arguments.Config = _setup_mock_config()
        arguments.Config['solving']['solving_mode'] = True
        arguments.Config['bab']['branching']['input_split']['reorder_bab'] = True
        arguments.Config['bab']['clip_n_verify']['clip_input_domain']['enabled'] = True
        arguments.Config['bab']['clip_n_verify']['clip_input_domain']['clip_type'] = 'complete'

    def tearDown(self):
        arguments.Config = self.original_config

    @patch('input_split.batch_branch_and_bound.clip_domains')
    @patch('input_split.batch_branch_and_bound.input_split_branching')
    @patch('input_split.batch_branch_and_bound.input_split_and_repeat')
    @patch('input_split.batch_branch_and_bound.UnsortedInputDomainList.filter_verified_domains')
    @patch('input_split.batch_branch_and_bound.get_bounds')
    def test_rechecks_dm_lbs_when_readding_domains(
        self,
        mock_get_bounds,
        mock_filter_verified,
        mock_split_repeat,
        mock_branching,
        mock_clip_domains,
    ):
        """Solving mode should not requeue domains already resolved by +inf objective bounds."""
        from input_split.batch_branch_and_bound import batch_verification_input_split
        from utils import Stats

        dm_lb = torch.tensor([[float('inf')]])
        x_L = torch.zeros(1, 2)
        x_U = torch.ones(1, 2)
        cs = torch.ones(1, 1, 1)
        thresholds = torch.zeros(1, 1)
        spec_sizes = torch.ones(1, dtype=torch.int32)
        split_idx = torch.zeros(1, 10, dtype=torch.long)
        lA = torch.zeros(1, 1, 2)
        lbias = torch.zeros(1, 1)

        mock_get_bounds.return_value = (dm_lb, lA, lbias, None)
        mock_filter_verified.return_value = (
            1, dm_lb, x_L, x_U, cs, thresholds, lA, lbias,
            None, None, None, spec_sizes, None, None, None
        )
        mock_branching.return_value = split_idx
        mock_split_repeat.return_value = (
            x_L, x_U, cs, thresholds, dm_lb, spec_sizes, lA, lbias,
            None, None, None, None
        )
        mock_clip_domains.side_effect = lambda *args, **kwargs: (args[0], args[1])

        domain_list = MagicMock()
        domain_list.history = MagicMock()
        domain_list.history.rowpool = RowPool(thresholds)
        domain_list.pick_out_batch.return_value = (
            dm_lb, x_L, x_U, cs, thresholds, None, None,
            spec_sizes, split_idx, None, None
        )
        domain_list.__len__.return_value = 0

        net = MagicMock()
        net.device = 'cpu'
        stats = Stats()
        stats.storage_depth = 10

        batch_verification_input_split(
            domain_list,
            net,
            batch=1,
            num_iter=1,
            decision_thresh=thresholds,
            shape=(1, 2),
            use_reordered_bab=True,
            stats=stats,
            primal_solution=torch.tensor([0.0]),
        )

        self.assertTrue(domain_list.add.called)
        self.assertTrue(domain_list.add.call_args.kwargs['check_dm_lbs'])

    @patch('input_split.batch_branch_and_bound.input_split_branching')
    @patch('input_split.batch_branch_and_bound.UnsortedInputDomainList.filter_verified_domains')
    @patch('input_split.batch_branch_and_bound.get_bounds')
    def test_empty_feasible_region_returns_infinite_sentinel_in_solving_mode(
        self,
        mock_get_bounds,
        mock_filter_verified,
        mock_branching,
    ):
        """Filtering all domains by constraints should not become an uncaught exception."""
        from input_split.batch_branch_and_bound import batch_verification_input_split
        from utils import Stats

        dm_lb = torch.tensor([[1.0, 2.0]])
        x_L = torch.zeros(1, 2)
        x_U = torch.ones(1, 2)
        cs = torch.ones(1, 2, 2)
        thresholds = torch.zeros(1, 2)
        spec_sizes = torch.tensor([2], dtype=torch.int32)
        split_idx = torch.zeros(1, 10, dtype=torch.long)
        lA = torch.zeros(1, 2, 2)
        lbias = torch.zeros(1, 2)

        mock_get_bounds.return_value = (dm_lb, lA, lbias, None)
        mock_filter_verified.return_value = (
            0, dm_lb[:0], x_L[:0], x_U[:0], cs[:0], thresholds[:0], lA[:0], lbias[:0],
            None, None, None, spec_sizes[:0], None, None, None
        )
        mock_branching.return_value = split_idx

        domain_list = MagicMock()
        domain_list.history = MagicMock()
        domain_list.history.rowpool = RowPool(thresholds)
        domain_list.pick_out_batch.return_value = (
            dm_lb, x_L, x_U, cs, thresholds, None, None,
            spec_sizes, split_idx, None, None
        )
        domain_list.__len__.return_value = 0

        net = MagicMock()
        net.device = 'cpu'
        stats = Stats()
        stats.storage_depth = 10

        dual, primal, primal_found = batch_verification_input_split(
            domain_list,
            net,
            batch=1,
            num_iter=1,
            decision_thresh=thresholds,
            shape=(1, 2),
            use_reordered_bab=True,
            stats=stats,
            primal_solution=torch.tensor([float('inf')]),
            primal_solution_found=False,
        )

        self.assertTrue(torch.isposinf(dual).all())
        self.assertTrue(torch.isposinf(primal).all())
        self.assertFalse(primal_found)


class TestInitialSplitCsPropagation(unittest.TestCase):
    """Tests for passing spec matrices into initial input-split branching."""

    def setUp(self):
        self.original_config = arguments.Config
        arguments.Config = _setup_mock_config()

    def tearDown(self):
        arguments.Config = self.original_config

    @patch('input_split.batch_branch_and_bound.input_split_branching')
    @patch('input_split.batch_branch_and_bound.UnsortedInputDomainList')
    @patch('input_split.batch_branch_and_bound.run_root_diagnostics_and_budget')
    @patch('input_split.batch_branch_and_bound.run_root_nlp_and_ipopt')
    def test_verified_pre_incomplete_primal_skips_second_root_solve(
        self,
        mock_root_nlp,
        _mock_root_diag,
        mock_domain_list,
        mock_input_split_branching,
    ):
        """A verified pre-incomplete candidate should be handed into BaB once."""
        from input_split.batch_branch_and_bound import input_bab_parallel

        arguments.Config['solving']['solving_mode'] = True
        x_L = torch.zeros(1, 3)
        x_U = torch.ones(1, 3)
        x = types.SimpleNamespace(
            ptb=types.SimpleNamespace(x_L=x_L, x_U=x_U),
            shape=(1, 3),
        )
        c = torch.randn(1, 2, 2)
        rhs = torch.zeros(1, 2)
        or_spec_size = torch.tensor([2])

        domains = MagicMock()
        domains.add.side_effect = StopIteration
        mock_domain_list.return_value = domains
        mock_input_split_branching.return_value = torch.zeros((1, 1), dtype=torch.long)

        net = types.SimpleNamespace(
            net=MagicMock(),
            final_name='final',
        )
        net.net.input_name = ['input']
        net.net.get_splittable_activations.return_value = []
        net.net.bound_opts = {'op_budget': {'hidden_squares': 1 << 20, 'complete_squares': 1 << 24,
                                             'prove_rows': 1 << 23, 'commit': 'partial'}}

        reference_dict = {
            'global_lb': torch.tensor([[-1.0, -1.0]]),
            'alphas': {},
            'betas': None,
            'lA': {'input': torch.ones(1, 2, 3)},
            'pre_incomplete_primal': {
                'verified': True,
                'nlp_found': False,
                'objective': 0.25,
                'x': torch.tensor([0.1, 0.2, 0.3]),
                'peak_mem_bytes': None,
                'total_mem_bytes': None,
            },
        }

        with self.assertRaises(StopIteration):
            input_bab_parallel(
                net,
                x,
                c,
                rhs,
                or_spec_size,
                reference_dict=reference_dict,
            )

        mock_root_nlp.assert_not_called()


class TestLiftedHandoffRootQTB(unittest.TestCase):
    """Lifted leaves must inherit the regular root QTB bound."""

    def setUp(self):
        self.original_config = arguments.Config
        arguments.Config = _setup_mock_config()
        arguments.Config['solving']['solving_mode'] = True
        arguments.Config['bab']['branching']['input_split']['reorder_bab'] = True
        arguments.Config['bab']['clip_n_verify']['clip_input_domain']['enabled'] = True
        arguments.Config['bab']['clip_n_verify']['clip_input_domain']['clip_type'] = 'complete'
        arguments.Config['solver']['nlp'].update({
            'quadratic_tangent_bound': True,
            'dual_lp_pdhg_iters': 500,
            'lifted_lp_bab_time': 20.0,
            'lifted_lp_bab_handoff': True,
        })

    def tearDown(self):
        arguments.Config = self.original_config

    def _capture_initial_branching(self, root_bound=25.0, *, qtb_enabled=True, error=None,
                                   root_boxes=None):
        from input_split.batch_branch_and_bound import input_bab_parallel

        x_L, x_U = ((torch.zeros(1, 3), torch.ones(1, 3))
                    if root_boxes is None else root_boxes)
        batch_size = x_L.shape[0]
        x = types.SimpleNamespace(
            ptb=types.SimpleNamespace(x_L=x_L, x_U=x_U), shape=x_L.shape)
        c = torch.eye(2).unsqueeze(0).expand(batch_size, -1, -1)
        rhs = torch.zeros(batch_size, 2)
        base_A, base_b = torch.tensor([[[1.0, 0.0, 0.0]]]), torch.tensor([[0.5]])
        base_A, base_b = base_A.expand(batch_size, -1, -1), base_b.expand(batch_size, -1)
        reference_dict = {
            'global_lb': torch.tensor([[10.0, -1.0]]).expand(batch_size, -1).clone(),
            'lA': {'input': torch.ones(batch_size, 2, 3)},
            'base_input_linear_constraints': (base_A, base_b),
            'base_input_constraints': (base_A, -base_b),
            'pre_incomplete_primal': {
                'verified': True, 'nlp_found': False, 'objective': 100.0,
                'x': torch.full((3,), 0.1),
                'peak_mem_bytes': None, 'total_mem_bytes': None,
            },
        }
        net = types.SimpleNamespace(net=MagicMock(), model_ori=MagicMock(), final_name='final')
        net.net.input_name = ['input']
        net.net.get_splittable_activations.return_value = []
        net.net.bound_opts = {}
        qtb = types.SimpleNamespace(x_hat=torch.full((3,), 0.1), lam_lb=0.0, f=100.0,
                                    fixed_mask=x_L[0] == x_U[0], fixed_values=x_L[0])
        if not qtb_enabled:
            arguments.Config['solver']['nlp']['quadratic_tangent_bound'] = False

        def lifted_bab(history, *args, **kwargs):
            history.lifted_bab_leaves = (
                torch.zeros(3, 3), torch.full((3, 3), 0.5), torch.tensor([5.0, 20.0, 35.0]))
            return None

        with patch('input_split.batch_branch_and_bound.run_root_diagnostics_and_budget'), \
             patch('input_split.batch_branch_and_bound.UnsortedInputDomainList',
                   side_effect=lambda *args, **kwargs: MagicMock(history=kwargs['history'])), \
             patch('nlp_solver.quadratic_tangent.build_quadratic_tangent',
                   return_value=qtb if qtb_enabled else None) as build_qtb, \
             patch('input_split.nlp_in_bab.compute_qtb_tangent_point', return_value=(None, None)), \
             patch('input_split.lifted_bound.RootLifted.build', return_value=None), \
             patch('input_split.lifted_bound.make_verifier', return_value=None), \
             patch('input_split.lifted_bound.run_lifted_bab', side_effect=lifted_bab), \
             patch('input_split.batch_branch_and_bound.get_bounds', side_effect=error,
                   return_value=(torch.tensor([[root_bound, 99.0]]), None, None, None)) as bounds, \
             patch('input_split.batch_branch_and_bound.input_split_branching',
                   side_effect=StopIteration) as branching:
            with self.assertRaises(StopIteration):
                input_bab_parallel(net, x, c, rhs, torch.tensor([2]),
                                   reference_dict=reference_dict)
        if qtb_enabled:
            torch.testing.assert_close(build_qtb.call_args.kwargs['root_lower'], x_L[:1])
            torch.testing.assert_close(build_qtb.call_args.kwargs['root_upper'], x_U[:1])
        self.captured_history = branching.call_args.kwargs['history']

        return bounds, branching.call_args.args, reference_dict

    def test_leaves_inherit_root_objective_bound_and_keep_stronger_leaf_bounds(self):
        bounds, branching, reference = self._capture_initial_branching()
        torch.testing.assert_close(branching[1], torch.tensor([
            [25.0, -1.0], [25.0, -1.0], [35.0, -1.0]]))
        # Only one root box is bounded, with the current incumbent and both
        # original input constraints and the root planes' concretization cuts.
        args = bounds.call_args.args
        torch.testing.assert_close(args[2], torch.zeros(1, 3))
        torch.testing.assert_close(args[3], torch.ones(1, 3))
        torch.testing.assert_close(args[5], torch.eye(2).unsqueeze(0))
        torch.testing.assert_close(args[6], torch.tensor([[100.0, 0.0]]))
        self.assertIsNotNone(args[4].qtb)
        self.assertGreater(args[7][0].shape[1], 1)
        torch.testing.assert_close(args[7][0][:, :1], reference['base_input_constraints'][0])
        torch.testing.assert_close(args[7][1][:, :1], reference['base_input_constraints'][1])
        # The root bound reads the standard-form rows (the relaxed ones, see relax_input_constraints)
        # as A x <= b, not the primal's own tuple; this reference holds the same rows twice.
        torch.testing.assert_close(args[8][0], reference['base_input_linear_constraints'][0])
        torch.testing.assert_close(args[8][1], reference['base_input_linear_constraints'][1])
        # Scalar strengthening must not change the original planes' offsets.
        torch.testing.assert_close(reference['global_lb'], torch.tensor([[10.0, -1.0]]))

    def test_convex_row_cuts_enter_root_lp_and_survive_handoff(self):
        arguments.Config['solver']['nlp']['convex_row_cuts'] = True
        cut_A = torch.tensor([[[0.0, 1.0, 0.0]]])
        cut_b = torch.tensor([[-0.9]])
        manager = types.SimpleNamespace(rounds=1, offer=MagicMock(return_value=(cut_A, cut_b)))
        with patch('nlp_solver.convex_row_cuts.build_convex_row_cuts',
                   return_value=manager) as builder:
            bounds, _, reference = self._capture_initial_branching()
        args = bounds.call_args.args
        torch.testing.assert_close(args[8][0][:, :1], reference['base_input_constraints'][0])
        torch.testing.assert_close(args[8][0][:, 1:], cut_A)
        torch.testing.assert_close(args[8][1][:, 1:], -cut_b)
        torch.testing.assert_close(args[7][0][:, :2], args[8][0])
        self.assertIs(self.captured_history.convex_row_cuts, manager)
        torch.testing.assert_close(builder.call_args.args[5], torch.zeros(1, 3))
        torch.testing.assert_close(builder.call_args.args[6], torch.ones(1, 3))
        manager.offer.assert_called_once()

    def test_dcp_row_cuts_enter_root_lp_and_survive_handoff(self):
        arguments.Config['solver']['nlp']['convex_tangent_rows'] = True
        cut_A, cut_b = torch.tensor([[[0., 1., 0.]]]), torch.tensor([[-.9]])
        manager = types.SimpleNamespace(rounds=1, offer=MagicMock(return_value=(cut_A, cut_b)))
        with patch('nlp_solver.convex_tangent.build_convex_tangent_rows', return_value=manager) as builder:
            bounds, _, reference = self._capture_initial_branching()
        args = bounds.call_args.args
        torch.testing.assert_close(args[8][0][:, :1], reference['base_input_constraints'][0])
        torch.testing.assert_close(args[8][0][:, 1:], cut_A)
        self.assertIs(self.captured_history.convex_row_cuts, manager)
        self.assertFalse(builder.call_args.kwargs['quadratic_rows'])
        manager.offer.assert_called_once()

    def test_dcp_objective_uses_root_box_and_existing_tangent_lp(self):
        arguments.Config['solver']['nlp']['convex_tangent_objective'] = True
        tangent = types.SimpleNamespace(x_hat=torch.full((3,), .1), lam_lb=0., f=100., n=3,
                                        fixed_mask=torch.zeros(3, dtype=torch.bool),
                                        fixed_values=torch.zeros(3), certificate=object())
        with patch('nlp_solver.convex_tangent.build_convex_tangent', return_value=tangent) as builder:
            bounds, _, _ = self._capture_initial_branching(qtb_enabled=False)
        self.assertIs(self.captured_history.qtb, tangent)
        self.assertIs(bounds.call_args.args[4].qtb, tangent)
        torch.testing.assert_close(builder.call_args.args[5], torch.zeros(1, 3))
        torch.testing.assert_close(builder.call_args.args[6], torch.ones(1, 3))

    def test_converged_root_skips_handoff_using_configured_gap_tolerance(self):
        arguments.Config['solver']['nlp']['bab_primal_dual_abs_tol'] = 1e-5
        arguments.Config['solver']['nlp']['bab_primal_dual_rel_tol'] = 1e-4
        bounds, branching, _ = self._capture_initial_branching(root_bound=99.995)
        bounds.assert_called_once()
        torch.testing.assert_close(branching[1], torch.tensor([[99.995, -1.0]]))
        torch.testing.assert_close(branching[2], torch.zeros(1, 3))
        torch.testing.assert_close(branching[3], torch.ones(1, 3))

    def test_failed_root_bound_keeps_existing_handoff(self):
        with self.assertWarnsRegex(UserWarning, 'Root QTB bound skipped.*failed root LP'):
            bounds, branching, _ = self._capture_initial_branching(error=RuntimeError('failed root LP'))
        bounds.assert_called_once()
        torch.testing.assert_close(branching[1], torch.tensor([
            [10.0, -1.0], [20.0, -1.0], [35.0, -1.0]]))

    def test_no_qtb_keeps_existing_handoff_without_extra_bound(self):
        bounds, branching, _ = self._capture_initial_branching(qtb_enabled=False)
        bounds.assert_not_called()
        torch.testing.assert_close(branching[1], torch.tensor([
            [10.0, -1.0], [20.0, -1.0], [35.0, -1.0]]))

    def test_disabled_handoff_does_not_rebound_root(self):
        arguments.Config['solver']['nlp']['lifted_lp_bab_handoff'] = False
        bounds, branching, _ = self._capture_initial_branching()
        bounds.assert_not_called()
        torch.testing.assert_close(branching[1], torch.tensor([[10.0, -1.0]]))

    def test_differing_fixed_coordinates_disable_qtb(self):
        arguments.Config['solver']['nlp']['lifted_lp_bab_time'] = 0.0
        lower = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        upper = torch.tensor([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        bounds, branching, _ = self._capture_initial_branching(root_boxes=(lower, upper))
        bounds.assert_not_called()
        self.assertIsNone(self.captured_history.qtb)


class TestClipBaseConstraintForwarding(unittest.TestCase):
    """Tests forwarding original input constraints into clip call sites."""

    def setUp(self):
        self.original_config = arguments.Config
        arguments.Config = _setup_mock_config()
        arguments.Config['bab']['clip_n_verify']['clip_input_domain']['enabled'] = True
        arguments.Config['bab']['branching']['input_split']['reorder_bab'] = False

    def tearDown(self):
        arguments.Config = self.original_config

    def _run_root_clip_and_capture(self, clip_type: str, clip_error=None):
        from input_split.batch_branch_and_bound import input_bab_parallel

        arguments.Config['bab']['clip_n_verify']['clip_input_domain']['clip_type'] = clip_type

        x_L = torch.zeros(1, 3)
        x_U = torch.ones(1, 3)
        x = types.SimpleNamespace(
            ptb=types.SimpleNamespace(x_L=x_L, x_U=x_U),
            shape=(1, 3),
        )
        c = torch.randn(1, 1, 2)
        rhs = torch.zeros(1, 1)
        or_spec_size = torch.tensor([1])
        base_A = torch.tensor([[[1.0, 0.0, 0.0]]])
        base_b = torch.tensor([[0.5]])
        captured = {}

        def _clip_side_effect(*args, **kwargs):
            captured['kwargs'] = kwargs
            raise clip_error or StopIteration

        domain_list = MagicMock()

        def _add_side_effect(lb, dm_l, dm_u, *args, **kwargs):
            captured['boxes'] = (dm_l, dm_u)
            raise StopIteration

        domain_list.add.side_effect = _add_side_effect

        net = types.SimpleNamespace(
            net=MagicMock(),
            final_name='final',
        )
        net.net.input_name = ['input']
        net.net.get_splittable_activations.return_value = []
        net.net.bound_opts = {}

        reference_dict = {
            'global_lb': torch.tensor([[-1.0]]),
            'alphas': {},
            'betas': None,
            'lA': {'input': torch.ones(1, 1, 3)},
            'base_input_linear_constraints': (base_A, base_b),
            'base_input_constraints': (base_A, -base_b),
        }

        root_nlp_result = types.SimpleNamespace(
            dm_l=x_L,
            dm_u=x_U,
            ipopt_found=False,
            nlp_found=False,
            nlp_obj=None,
            ipopt_obj=None,
            nlp_best_x=None,
            ipopt_x=None,
            root_peak_mem_bytes=None,
            root_total_mem_bytes=None,
        )

        with patch('input_split.batch_branch_and_bound.clip_domains', side_effect=_clip_side_effect), \
             patch('input_split.batch_branch_and_bound.input_split_branching', return_value=torch.zeros((1, 1), dtype=torch.long)), \
             patch('input_split.batch_branch_and_bound.UnsortedInputDomainList', return_value=domain_list), \
             patch('input_split.batch_branch_and_bound.run_root_nlp_and_ipopt', return_value=root_nlp_result), \
             patch('input_split.batch_branch_and_bound.run_root_diagnostics_and_budget'):
            with self.assertRaises(StopIteration):
                input_bab_parallel(
                    net,
                    x,
                    c,
                    rhs,
                    or_spec_size,
                    reference_dict=reference_dict,
                )

        return captured

    def test_relaxed_clip_receives_original_base_constraints(self):
        """Relaxed clip should receive original problem constraints explicitly."""
        forwarded = self._run_root_clip_and_capture('relaxed')['kwargs']
        self.assertIsNotNone(forwarded['base_input_linear_constraints'])
        self.assertTrue(torch.equal(
            forwarded['base_input_linear_constraints'][0],
            torch.tensor([[[1.0, 0.0, 0.0]]]),
        ))
        self.assertTrue(torch.equal(
            forwarded['base_input_linear_constraints'][1],
            torch.tensor([[0.5]]),
        ))
        self.assertTrue(torch.equal(
            forwarded['base_input_constraints'][0],
            torch.tensor([[[1.0, 0.0, 0.0]]]),
        ))
        self.assertTrue(torch.equal(
            forwarded['base_input_constraints'][1],
            torch.tensor([[-0.5]]),
        ))
        self.assertTrue(torch.equal(
            forwarded['domain_constraints'][0],
            torch.tensor([[[1.0, 0.0, 0.0]]]),
        ))
        self.assertTrue(torch.equal(
            forwarded['domain_constraints'][1],
            torch.tensor([[-0.5]]),
        ))

    def test_complete_clip_also_receives_original_base_constraints(self):
        """Complete clip should forward base constraints into the clip kernel for parity."""
        forwarded = self._run_root_clip_and_capture('complete')['kwargs']
        self.assertIsNotNone(forwarded['base_input_linear_constraints'])
        self.assertTrue(torch.equal(
            forwarded['base_input_linear_constraints'][0],
            torch.tensor([[[1.0, 0.0, 0.0]]]),
        ))
        self.assertTrue(torch.equal(
            forwarded['base_input_linear_constraints'][1],
            torch.tensor([[0.5]]),
        ))
        self.assertTrue(torch.equal(
            forwarded['base_input_constraints'][0],
            torch.tensor([[[1.0, 0.0, 0.0]]]),
        ))
        self.assertTrue(torch.equal(
            forwarded['base_input_constraints'][1],
            torch.tensor([[-0.5]]),
        ))
        self.assertTrue(torch.equal(
            forwarded['domain_constraints'][0][:, :1],
            torch.tensor([[[1.0, 0.0, 0.0]]]),
        ))
        self.assertTrue(torch.equal(
            forwarded['domain_constraints'][1][:, :1],
            torch.tensor([[-0.5]]),
        ))

    def test_root_clip_oom_keeps_the_unclipped_boxes(self):
        """Clipping is optional: an OOM in the root clip must not crash the run."""
        captured = self._run_root_clip_and_capture(
            'complete', clip_error=torch.cuda.OutOfMemoryError("CUDA out of memory"))
        dm_l, dm_u = captured['boxes']
        self.assertTrue(torch.equal(dm_l, torch.zeros(1, 3)))
        self.assertTrue(torch.equal(dm_u, torch.ones(1, 3)))


class TestReorderedSplitClipping(unittest.TestCase):
    """Tests clipping children after reordered input splitting."""

    def setUp(self):
        self.original_config = arguments.Config
        arguments.Config = _setup_mock_config()
        arguments.Config['bab']['branching']['input_split']['reorder_bab'] = True
        arguments.Config['bab']['clip_n_verify']['clip_input_domain']['enabled'] = True

    def tearDown(self):
        arguments.Config = self.original_config

    @patch('input_split.batch_branch_and_bound.input_split_and_repeat')
    @patch('input_split.batch_branch_and_bound.input_split_branching')
    @patch('input_split.batch_branch_and_bound.get_bounds')
    def test_does_not_queue_children_certified_by_affine_bounds(
            self, mock_get_bounds, mock_branching, mock_split):
        from input_split.batch_branch_and_bound import batch_verification_input_split
        from input_split.branching_domains import UnsortedInputDomainList
        from input_split.utils import Stats

        domains = MagicMock()
        domains.history.rowpool = None
        domains.use_alpha = False
        domains.__len__.return_value = 0
        stored_domains = UnsortedInputDomainList(
            storage_depth=1, output_device='cpu', use_split_idx=False)
        domains.add.side_effect = stored_domains.add
        # pick_out_batch returns (dm_lb, x_L, x_U, cs, thresholds, constraints,
        # base_constraints, spec_sizes, split_idx, attack_counter, split_counter);
        # alphas travel on the split history instead.
        domains.pick_out_batch.return_value = (
            torch.tensor([[-0.25, -0.25]]),
            torch.tensor([[0.0]]), torch.tensor([[1.0]]),
            torch.zeros(1, 2, 2), torch.zeros(1, 2), None, None,
            torch.ones(1, dtype=torch.int32), None, None, None,
        )

        # Neither affine bound verifies the parent interval. After splitting,
        # the first bound verifies the left child and the second verifies the
        # right child.
        parent_lA = torch.tensor([[[-1.0], [1.0]]])
        parent_lbias = torch.tensor([[0.75, -0.25]])
        # get_bounds returns (dm_lb, lA, lbias, x_star).
        mock_get_bounds.return_value = (
            torch.tensor([[-0.25, -0.25]]), parent_lA, parent_lbias, None)
        mock_branching.return_value = torch.tensor([[0]])

        child_x_L = torch.tensor([[0.0], [0.5]])
        child_x_U = torch.tensor([[0.5], [1.0]])
        # input_split_and_repeat returns (x_L, x_U, cs, thresholds, dm_lb,
        # spec_sizes, lA, lbias, constraints, base_constraints, attack_counter,
        # split_counter).
        mock_split.return_value = (
            child_x_L, child_x_U, torch.zeros(2, 2, 2),
            torch.zeros(2, 2), torch.tensor([[-0.25, -0.25]]).repeat(2, 1),
            torch.ones(2, dtype=torch.int32), parent_lA.repeat(2, 1, 1),
            parent_lbias.repeat(2, 1), None, None, None, None,
        )

        net = MagicMock()
        net.device = 'cpu'
        stats = Stats()

        batch_verification_input_split(
            domains, net, batch=1, num_iter=1,
            decision_thresh=torch.zeros(1, 2), shape=(1, 1),
            bounding_method='crown', branching_method='naive',
            use_reordered_bab=True, stats=stats)

        self.assertEqual(len(stored_domains), 0)


class TestInitialDomainClipping(unittest.TestCase):
    """Tests filtering domains emptied by initial input clipping."""

    def setUp(self):
        self.original_config = arguments.Config
        arguments.Config = _setup_mock_config()
        arguments.Config['bab']['clip_n_verify']['clip_input_domain']['enabled'] = True

    def tearDown(self):
        arguments.Config = self.original_config

    @patch(
        'input_split.batch_branch_and_bound.batch_verification_input_split',
        side_effect=AssertionError('an empty domain entered the BaB loop'))
    @patch(
        'input_split.batch_branch_and_bound.input_split_branching',
        return_value=torch.tensor([[0]]))
    def test_filters_domain_emptied_by_multiple_constraints(
            self, mock_branching, mock_batch_verification):
        from input_split.batch_branch_and_bound import input_bab_parallel

        x = MagicMock()
        x.shape = (1, 1)
        x.ptb.x_L = torch.tensor([[0.0]])
        x.ptb.x_U = torch.tensor([[1.0]])
        # No problem-level input constraints on this input.
        x.ptb.constraints = None
        x.ptb.base_input_constraints = None
        x.ptb.base_input_linear_constraints = None

        net = MagicMock()
        net.device = 'cpu'
        net.net.input_name = ['input']
        net.net.get_splittable_activations.return_value = []
        net.net.bound_opts = {}

        # Neither affine constraint certifies [0, 1] by itself. Together they
        # clip it to x <= 0.4 and x >= 0.6, leaving no region to search.
        reference_dict = {
            'global_lb': torch.tensor([[-0.4, -0.4]]),
            'lA': {'input': torch.tensor([[[1.0], [-1.0]]])},
        }

        _, visited, result = input_bab_parallel(
            net, x, c=torch.zeros(1, 2, 1), rhs=torch.zeros(1, 2),
            or_spec_size=torch.tensor([2]), reference_dict=reference_dict)

        self.assertEqual(result, 'safe')
        self.assertEqual(visited, 1)
        mock_batch_verification.assert_not_called()


if __name__ == '__main__':
    unittest.main()


class TestBranchingWithoutLA(unittest.TestCase):
    def test_sb_family_falls_back_to_naive_without_lA(self):
        # Neumaier2 / schitt_s358: the root bound came back without lA and the run died
        # on an assertion; input_split_branching already splits the longest edge then.
        from input_split.batch_branch_and_bound import _branching_method_without_lA
        self.assertEqual(_branching_method_without_lA("sb"), "naive")
        self.assertEqual(_branching_method_without_lA("naive"), "naive")
        self.assertEqual(_branching_method_without_lA("brute-force"), "brute-force")


class TestPickoutRestoreOnOOM(unittest.TestCase):
    def test_restored_pickout_puts_the_domains_back_unchanged(self):
        # cute_airport / cute_batch / cute_explin: a batch that no longer fits killed the
        # run; the picked domains must go back to the queue before retrying smaller.
        from history import SplitHistory
        from input_split.branching_domains import UnsortedInputDomainList
        from input_split.batch_branch_and_bound import _remember_pickout, _restore_pickout
        history = SplitHistory()
        history.set_usage_flags(False, False)
        d = UnsortedInputDomainList(1, 'cpu', history=history, sort_index=None,
                                    sort_heuristic='lb_only', sort_descending=True,
                                    use_split_idx=False)
        lb = torch.tensor([[-4.0], [-3.0], [-2.0], [-1.0]])
        dm_l = torch.arange(4.0).view(4, 1)
        dm_u = dm_l + 1
        d.add(lb, dm_l, dm_u, torch.ones(4, 1, 1), torch.zeros(4, 1),
              constraints=None, base_constraints=None, check_dm_lbs=False,
              check_input_boxes=False, spec_sizes=torch.ones(4, dtype=torch.int32),
              attack_counter=torch.zeros(4, 1, dtype=torch.int),
              split_counter=torch.zeros(4, 1, dtype=torch.int))
        ret = d.pick_out_batch(2, device='cpu')
        _remember_pickout(d, ret)
        ret[1].add_(100.0)          # processing may modify the picked tensors in place
        self.assertEqual(len(d), 2)
        self.assertTrue(_restore_pickout(d))
        self.assertEqual(len(d), 4)
        self.assertFalse(_restore_pickout(d))   # a restore happens at most once
        back = d.pick_out_batch(4, device='cpu')
        self.assertEqual(sorted(back[1].reshape(-1).tolist()), [0.0, 1.0, 2.0, 3.0])

    def test_restore_undoes_a_partial_add(self):
        # Hartman6 (GPU domain storage): the out-of-memory error struck inside d.add
        # after the bounds and boxes had been appended but before the cut rows were.
        # Without undoing that, the storages stayed misaligned: every later pop paired
        # boxes with other domains' rows (and a lazy sort indexed past the short one).
        from unittest import mock
        from history import SplitHistory
        from input_split.branching_domains import UnsortedInputDomainList
        from input_split.batch_branch_and_bound import _remember_pickout, _restore_pickout
        history = SplitHistory()
        history.set_usage_flags(False, False)
        d = UnsortedInputDomainList(1, 'cpu', history=history, sort_index=None,
                                    sort_heuristic='lb_only', sort_descending=True,
                                    use_split_idx=False)

        def add(tags, dl=d):
            n = len(tags)
            t = torch.tensor(tags, dtype=torch.float64)
            dl.add(t.view(n, 1) - 100, t.view(n, 1), t.view(n, 1) + 1, torch.ones(n, 1, 1),
                   torch.zeros(n, 1), constraints=(torch.ones(n, 1, 1), (t - 1e6).view(n, 1)),
                   base_constraints=None, check_dm_lbs=False, check_input_boxes=False,
                   spec_sizes=torch.ones(n, dtype=torch.int32),
                   attack_counter=torch.zeros(n, 1, dtype=torch.int),
                   split_counter=torch.zeros(n, 1, dtype=torch.int))
        add([0.0, 1.0, 2.0, 3.0])
        _remember_pickout(d, d.pick_out_batch(2, device='cpu'))
        with mock.patch.object(UnsortedInputDomainList, '_append_constraints',
                               side_effect=torch.OutOfMemoryError('cut rows')):
            with self.assertRaises(torch.OutOfMemoryError):
                add([10.0, 11.0, 12.0])            # children of the failed batch
        self.assertTrue(_restore_pickout(d))
        sizes = {n for _, n in d.storage_sizes()}
        self.assertEqual(sizes, {4})               # all storages aligned again
        back = d.pick_out_batch(4, device='cpu')
        tags = back[1].reshape(-1)
        self.assertEqual(sorted(tags.tolist()), [0.0, 1.0, 2.0, 3.0])
        self.assertTrue(torch.equal(back[5][1].reshape(-1) + 1e6, tags))   # rows paired

    def test_remembered_pickout_keeps_the_shared_spec_broadcast(self):
        # The spec matrix is one row broadcast over the batch; remembering the pickout
        # must not materialize a copy per domain.
        from input_split.batch_branch_and_bound import _clone_tree
        cs = torch.randn(1, 5, 5).expand(1000, 5, 5)
        saved = _clone_tree((cs, torch.arange(3.0)))
        self.assertEqual(saved[0].stride(0), 0)
        self.assertTrue(torch.equal(saved[0], cs))
        self.assertNotEqual(saved[0].data_ptr(), cs.data_ptr())
        self.assertTrue(torch.equal(saved[1], torch.arange(3.0)))

    def test_restore_does_not_refilter_the_domains(self):
        # kall_circlespolygons_c1p13: re-filtering the restored pickout (base-constraint
        # lower bounds) ran out of memory again inside the OOM handler. The domains were
        # already in the queue, so they go back without any filtering.
        from unittest import mock
        from history import SplitHistory
        from input_split.branching_domains import UnsortedInputDomainList
        from input_split.batch_branch_and_bound import _remember_pickout, _restore_pickout
        history = SplitHistory()
        history.set_usage_flags(False, False)
        d = UnsortedInputDomainList(1, 'cpu', history=history, sort_index=None,
                                    sort_heuristic='lb_only', sort_descending=True,
                                    use_split_idx=False)
        dm_l = torch.arange(3.0).view(3, 1)
        d.add(torch.tensor([[-3.0], [-2.0], [-1.0]]), dm_l, dm_l + 1, torch.ones(3, 1, 1),
              torch.zeros(3, 1), constraints=None, base_constraints=None, check_dm_lbs=False,
              check_input_boxes=False, spec_sizes=torch.ones(3, dtype=torch.int32),
              attack_counter=torch.zeros(3, 1, dtype=torch.int),
              split_counter=torch.zeros(3, 1, dtype=torch.int))
        _remember_pickout(d, d.pick_out_batch(3, device='cpu'))
        with mock.patch.object(UnsortedInputDomainList, 'get_remaining_index',
                               side_effect=AssertionError('restore re-filtered')):
            self.assertTrue(_restore_pickout(d))
        self.assertEqual(len(d), 3)

    def test_restore_after_the_shared_base_grew(self):
        # kriging_peaks-full030 (A87): convex row cuts widened the shared base rows from
        # 38 to 68 after the pickout; restoring the pickout with its old base failed the
        # shared-base check, and the OOM handler stopped the search.
        from history import SplitHistory
        from input_split.branching_domains import UnsortedInputDomainList
        from input_split.batch_branch_and_bound import _remember_pickout, _restore_pickout
        history = SplitHistory()
        history.set_usage_flags(False, False)
        d = UnsortedInputDomainList(1, 'cpu', history=history, sort_index=None,
                                    sort_heuristic='lb_only', sort_descending=True,
                                    use_split_idx=False)
        dm_l = torch.arange(3.0).view(3, 1)
        base = (torch.ones(1, 2, 1).expand(3, 2, 1), torch.full((1, 2), -10.).expand(3, 2))
        d.add(torch.tensor([[-3.0], [-2.0], [-1.0]]), dm_l, dm_l + 1, torch.ones(3, 1, 1),
              torch.zeros(3, 1), constraints=None, base_constraints=base, check_dm_lbs=False,
              check_input_boxes=False, spec_sizes=torch.ones(3, dtype=torch.int32),
              attack_counter=torch.zeros(3, 1, dtype=torch.int),
              split_counter=torch.zeros(3, 1, dtype=torch.int))
        _remember_pickout(d, d.pick_out_batch(2, device='cpu'))
        d.base_constraint_A = torch.ones(1, 3, 1)          # what _offer_convex_row_cuts does
        d.base_constraint_b = torch.full((1, 3), -10.)
        self.assertTrue(_restore_pickout(d))
        self.assertEqual(len(d), 3)
        back = d.pick_out_batch(3, device='cpu')
        self.assertEqual(tuple(back[6][0].shape), (3, 3, 1))   # the grown base reaches them

    def test_restore_after_a_widen_ran_out_of_memory(self):
        # kriging_peaks-full200 (A90): the children's wider cut rows widened the A storage,
        # then widening b ran out of memory; the restore must still append the pickout and
        # later batches must see A and b at one width.
        from unittest import mock
        from history import SplitHistory
        from tensor_storage import StackTensorStorage
        from input_split.branching_domains import UnsortedInputDomainList
        from input_split.batch_branch_and_bound import _remember_pickout, _restore_pickout
        history = SplitHistory()
        history.set_usage_flags(False, False)
        d = UnsortedInputDomainList(1, 'cpu', history=history, sort_index=None,
                                    sort_heuristic='lb_only', sort_descending=True,
                                    use_split_idx=False)

        def add(tags, rows):
            n = len(tags)
            t = torch.tensor(tags, dtype=torch.float64)
            d.add(t.view(n, 1) - 100, t.view(n, 1), t.view(n, 1) + 1, torch.ones(n, 1, 1),
                  torch.zeros(n, 1), constraints=(torch.ones(n, rows, 1), torch.full((n, rows), -1e6)),
                  base_constraints=None, check_dm_lbs=False, check_input_boxes=False,
                  spec_sizes=torch.ones(n, dtype=torch.int32),
                  attack_counter=torch.zeros(n, 1, dtype=torch.int),
                  split_counter=torch.zeros(n, 1, dtype=torch.int))
        add([0.0, 1.0, 2.0, 3.0], rows=1)
        _remember_pickout(d, d.pick_out_batch(2, device='cpu'))
        real_widen = StackTensorStorage.widen
        calls = []

        def widen_then_fail(self, dim, new_size, fill_value):
            calls.append(new_size)
            if len(calls) == 2:
                raise torch.OutOfMemoryError('widen b')
            return real_widen(self, dim, new_size, fill_value)
        with mock.patch.object(StackTensorStorage, 'widen', widen_then_fail):
            with self.assertRaises(torch.OutOfMemoryError):
                add([10.0, 11.0], rows=3)          # children with wider cut rows
        self.assertTrue(_restore_pickout(d))
        self.assertEqual(len(d), 4)
        add([20.0], rows=3)
        back = d.pick_out_batch(5, device='cpu')
        self.assertEqual(back[5][0].shape[1], back[5][1].shape[1])


class TestRootDomainFilterOnOOM(unittest.TestCase):
    def _domains(self):
        from history import SplitHistory
        from input_split.branching_domains import UnsortedInputDomainList
        history = SplitHistory()
        history.set_usage_flags(False, False)
        return UnsortedInputDomainList(1, 'cpu', history=history, sort_index=None,
                                       sort_heuristic='lb_only', sort_descending=True,
                                       use_split_idx=False)

    def _add_kwargs(self, n):
        # Domains 0 and 2 are verified (lb > threshold); the input constraints x <= -1 on
        # domain boxes [i, i + 1] would verify all of them if the filter ran.
        return dict(constraints=None,
                    base_constraints=(torch.ones(n, 1, 1), torch.ones(n, 1)),
                    check_dm_lbs=True, check_input_boxes=False,
                    spec_sizes=torch.ones(n, dtype=torch.int32),
                    attack_counter=torch.zeros(n, 1, dtype=torch.int),
                    split_counter=torch.zeros(n, 1, dtype=torch.int))

    def test_input_constraint_filter_that_runs_out_of_memory_is_skipped(self):
        # pooling_sppb5pq/b5tp/c1stp: the dense constraint bound of the first domains.add
        # ran out of memory after the root had finished, and the run died with rc=1.
        from unittest import mock
        from input_split.branching_domains import UnsortedInputDomainList
        from input_split.batch_branch_and_bound import _add_root_domains
        real = UnsortedInputDomainList.get_remaining_index

        def get_remaining_index(*args, **kwargs):
            if kwargs.get('base_constraints') is not None:
                raise torch.OutOfMemoryError('input-constraint bound')
            return real(*args, **kwargs)

        d = self._domains()
        dm_l = torch.arange(4.0).view(4, 1)
        lb = torch.tensor([[5.0], [-3.0], [5.0], [-1.0]])
        with mock.patch.object(UnsortedInputDomainList, 'get_remaining_index',
                               staticmethod(get_remaining_index)):
            _add_root_domains(d, lb, dm_l, dm_l + 1, torch.ones(4, 1, 1), torch.zeros(4, 1),
                              remaining_index=None, **self._add_kwargs(4))
        self.assertEqual(len(d), 2)                 # the bound filter still ran
        self.assertEqual({n for _, n in d.storage_sizes()}, {2})
        back = d.pick_out_batch(2, device='cpu')
        self.assertEqual(sorted(back[1].reshape(-1).tolist()), [1.0, 3.0])

    def test_a_filter_that_fits_is_the_plain_add(self):
        from input_split.batch_branch_and_bound import _add_root_domains
        d = self._domains()
        dm_l = torch.arange(4.0).view(4, 1)
        lb = torch.tensor([[5.0], [-3.0], [5.0], [-1.0]])
        _add_root_domains(d, lb, dm_l, dm_l + 1, torch.ones(4, 1, 1), torch.zeros(4, 1),
                          remaining_index=None, **dict(self._add_kwargs(4), base_constraints=None))
        self.assertEqual(len(d), 2)


class TestBatchAfterOOM(unittest.TestCase):
    def _domains_with_pickout(self, n_pick):
        from history import SplitHistory
        from input_split.branching_domains import UnsortedInputDomainList
        from input_split.batch_branch_and_bound import _remember_pickout
        history = SplitHistory()
        history.set_usage_flags(False, False)
        d = UnsortedInputDomainList(1, 'cpu', history=history, sort_index=None,
                                    sort_heuristic='lb_only', sort_descending=True,
                                    use_split_idx=False)
        d.add(torch.tensor([[-2.0], [-1.0]]), torch.zeros(2, 1), torch.ones(2, 1),
              torch.ones(2, 1, 1), torch.zeros(2, 1), constraints=None,
              base_constraints=None, check_dm_lbs=False, check_input_boxes=False,
              spec_sizes=torch.ones(2, dtype=torch.int32),
              attack_counter=torch.zeros(2, 1, dtype=torch.int),
              split_counter=torch.zeros(2, 1, dtype=torch.int))
        _remember_pickout(d, d.pick_out_batch(n_pick, device='cpu'))
        return d

    def test_halves_a_larger_batch(self):
        from types import SimpleNamespace
        from input_split.batch_branch_and_bound import _batch_after_oom
        d = self._domains_with_pickout(2)
        auto = SimpleNamespace(batch_size=2, enable=True)
        self.assertEqual(_batch_after_oom(d, 2, auto), 1)
        self.assertEqual(len(d), 2)
        self.assertFalse(auto.enable)

    def test_single_domain_stops_with_the_domain_kept(self):
        # A single domain that does not fit ends the search open (unknown); it
        # must be back in the list, never dropped.
        from types import SimpleNamespace
        from input_split.batch_branch_and_bound import _batch_after_oom
        d = self._domains_with_pickout(1)
        self.assertEqual(len(d), 1)
        auto = SimpleNamespace(batch_size=1, enable=True)
        self.assertIsNone(_batch_after_oom(d, 1, auto))
        self.assertEqual(len(d), 2)

    def test_oom_stop_reports_the_open_domain_dual_and_primal(self):
        # pooling_sppc1pq: the search stopped after an OOM at batch size 1 with neither line.
        from unittest import mock
        import input_split.batch_branch_and_bound as bbb
        d = self._domains_with_pickout(1)
        stats = types.SimpleNamespace(visited=3, timer=MagicMock())
        with patch.object(arguments, 'Config', _setup_mock_config()), \
             patch.object(arguments, 'Globals', {'discarded_threshold_min': -1.5}), \
             mock.patch('builtins.print') as pr:
            self.assertIsNone(bbb._batch_after_oom(
                d, 1, types.SimpleNamespace(batch_size=1, enable=True)))
            lb = bbb._report_after_oom_stop(d, stats, torch.tensor([4.0]))
        self.assertEqual(float(lb[0, 0]), -2.0)       # lowest of -2, -1
        text = "\n".join(str(c.args[0]) for c in pr.call_args_list if c.args)
        self.assertIn("Dual Solution", text)
        self.assertIn("Primal Solution", text)

    def test_oom_stop_caps_the_dual_by_the_discarded_threshold(self):
        import input_split.batch_branch_and_bound as bbb
        d = self._domains_with_pickout(1)
        stats = types.SimpleNamespace(visited=3, timer=MagicMock())
        with patch.object(arguments, 'Config', _setup_mock_config()), \
             patch.object(arguments, 'Globals', {'discarded_threshold_min': -3.0}):
            lb = bbb._report_after_oom_stop(d, stats, torch.tensor([4.0]))
        self.assertEqual(float(lb[0, 0]), -3.0)

    def test_oom_stop_reports_nothing_when_domains_were_lost(self):
        import input_split.batch_branch_and_bound as bbb
        d = self._domains_with_pickout(1)
        d._domains_lost = True
        self.assertIsNone(bbb._report_after_oom_stop(d, None, torch.tensor([4.0])))

    def test_remembered_pickout_is_on_the_host(self):
        d = self._domains_with_pickout(2)
        saved = [t for t in d._last_pickout if isinstance(t, torch.Tensor)]
        self.assertTrue(saved and all(t.device.type == 'cpu' for t in saved))

    def test_failed_restore_stops_and_marks_the_list(self):
        # If the picked domains cannot go back (MeyerRoth: the GPU was still full),
        # the search is not exhaustive any more: stop, and never call it safe.
        from types import SimpleNamespace
        from unittest import mock
        import input_split.batch_branch_and_bound as bbb
        d = self._domains_with_pickout(2)
        with mock.patch.object(bbb, '_restore_pickout',
                               side_effect=torch.OutOfMemoryError('full')):
            self.assertIsNone(bbb._batch_after_oom(
                d, 2, SimpleNamespace(batch_size=2, enable=True)))
        self.assertTrue(d._domains_lost)


class TestRaiseRootLbToOpenLeaves(unittest.TestCase):
    """The lifted-LP B&B's proven bound survives a skipped BaB handoff (digabel18)."""

    def test_raises_to_the_lowest_open_leaf(self):
        from input_split.batch_branch_and_bound import _raise_root_lb_to_open_leaves
        lb = torch.tensor([[-781.0, 0.0]])
        out = _raise_root_lb_to_open_leaves(
            lb, torch.tensor([-720.6, -500.0], dtype=torch.float64), torch.tensor([-100.0]))
        self.assertGreater(float(out[0, 0]), -720.6 - 1e-3)
        self.assertLessEqual(float(out[0, 0]), -720.6)       # never rounded above the leaf bound
        self.assertEqual(float(out[0, 1]), 0.0)              # constraint columns untouched
        self.assertEqual(float(lb[0, 0]), -781.0)            # input not modified

    def test_never_exceeds_the_incumbent(self):
        from input_split.batch_branch_and_bound import _raise_root_lb_to_open_leaves
        out = _raise_root_lb_to_open_leaves(
            torch.tensor([[-5.0]]), torch.tensor([7.0]), torch.tensor([2.0]))
        self.assertLessEqual(float(out[0, 0]), 2.0)

    def test_never_lowers_a_stronger_root_bound(self):
        from input_split.batch_branch_and_bound import _raise_root_lb_to_open_leaves
        out = _raise_root_lb_to_open_leaves(
            torch.tensor([[-1.0]]), torch.tensor([-9.0]), torch.tensor([2.0]))
        self.assertEqual(float(out[0, 0]), -1.0)

    def test_nonfinite_bound_changes_nothing(self):
        from input_split.batch_branch_and_bound import _raise_root_lb_to_open_leaves
        out = _raise_root_lb_to_open_leaves(
            torch.tensor([[-1.0]]), torch.tensor([float('nan')]), torch.tensor([2.0]))
        self.assertEqual(float(out[0, 0]), -1.0)


class TestIsCudaOom(unittest.TestCase):
    def test_torchscript_wrapped_oom_counts(self):
        from nlp_solver.utils import is_cuda_oom
        self.assertTrue(is_cuda_oom(torch.OutOfMemoryError("CUDA out of memory")))
        # TorchScript re-raises the OOM as a plain RuntimeError (MeyerRoth).
        self.assertTrue(is_cuda_oom(RuntimeError(
            "The following operation failed in the TorchScript interpreter.\n"
            "RuntimeError: CUDA out of memory. Tried to allocate 20.00 MiB.")))
        self.assertFalse(is_cuda_oom(RuntimeError("NaN detected in backward")))
        self.assertFalse(is_cuda_oom(ValueError("out of memory")))
