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
"""Branch and bound for input space split."""
from typing import Tuple, Union, Callable, List, NamedTuple, Optional, Dict, Any
import collections
import importlib
import time
import torch
from torch import Tensor
import math
import sys
import os
import warnings

import arguments
from history import SplitHistory
from beta_CROWN_solver import LiRPANet
from auto_LiRPA.utils import (stop_criterion_batch_any, AutoBatchSize)
from auto_LiRPA import BoundedTensor
from input_constraints import (
    merge_standard_constraints,
    repeat_input_constraints,
    standard_form_to_input_constraints,
)

from utils import check_auto_enlarge_batch_size, Stats
from gear import report as _report

from input_split.attack_in_input_split import (pgd_attack_on_domains,
                                            update_rhs_with_attack)
from nlp_solver.continuation import Atom, retained, invoke
from input_split.nlp_in_bab import (run_root_nlp_and_ipopt, RootNLPResult,
                                    run_root_diagnostics_and_budget,
                                    pre_incomplete_to_root_result,
                                    init_alm_bab_state,
                                    PickedDomainBatch,
                                    run_bab_alm_step)
from input_split.branching_heuristics import input_split_branching, _exhaustive_subdivision_guard
from input_split.clip import clip_domains, deconstruct_bias
from input_split.split import input_split_and_repeat
from input_split.utils import (initial_verify_criterion, prefix_print_output, no_op_context,
                               construct_cut_constraints)
from input_split.branching_domains import UnsortedInputDomainList, UnsortedMultiSpecInputDomainList
from input_split.bounding import set_instance_index, extract_relaxation_minimizer  # lp sanity check
from nlp_solver.utils import is_cuda_oom
from nlp_solver.structure import RowPool
from solving_utils import (
    is_primal_dual_converged as _is_primal_dual_converged,
    load_fixed_primal,
)


def _remaining_root_wall_budget(timeout, start_time, fraction, now=None):
    """Return the root solver's true share of the remaining global budget.

    Negative values are intentional: the root solver entry point uses them to
    skip all heuristics after bound propagation has exhausted the deadline.
    """
    fraction = float(fraction)
    if fraction <= 0:
        return None
    now = time.time() if now is None else float(now)
    remaining_global = float(timeout) - (now - float(start_time))
    return fraction * remaining_global


class _DeadlineBatchSizer:
    """Keep native batches until recent measured work would cross the deadline.

    The last two completed batches give seconds per picked parent. Predict with
    their maximum plus their observed range, rather than a fitted safety factor.
    A cap is local to this call; memory admission and enlargement keep their own
    batch size. With no measurements the first batch remains unchanged.
    """
    def __init__(self, deadline, clock=None):
        self.deadline = deadline
        self.clock = time.monotonic if clock is None else clock
        self.rates = collections.deque(maxlen=2)

    def record(self, parents, seconds):
        if parents > 0 and seconds > 0:
            self.rates.append(seconds / parents)

    def cap(self, requested, available):
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            return 0
        if not self.rates:
            return requested
        upper_rate = max(self.rates) + (max(self.rates) - min(self.rates))
        actual = min(requested, available)
        if actual * upper_rate < remaining:
            return requested
        # Strictly before the deadline, including an exactly integral quotient.
        return min(actual, max(0, math.ceil(remaining / upper_rate) - 1))


def _apply_primal_objective_threshold(
    thresholds: Tensor,
    primal_solution: Optional[Tensor],
) -> Tensor:
    """Return thresholds with the objective RHS set to the incumbent.

    Without a finite incumbent the objective RHS becomes +inf: nothing may be pruned
    against the objective. The RHS otherwise stays at its initial 0, which prunes every
    box whose objective lower bound is positive, and an emptied queue is then reported
    as solved with no primal at all.

    We clone before the in-place write because pickout tensors may be views created
    under `no_grad`, and PyTorch forbids mutating those views later with grad mode
    enabled.
    """
    if primal_solution is None:
        return thresholds

    primal_scalar = torch.as_tensor(
        primal_solution, device=thresholds.device, dtype=thresholds.dtype
    ).reshape(-1)[0]
    if not torch.isfinite(primal_scalar):
        primal_scalar = torch.full_like(primal_scalar, float("inf"))

    updated_thresholds = thresholds.clone()
    updated_thresholds[..., 0] = primal_scalar
    return updated_thresholds


def _m1_floor(dm_lb):
    """Minimum objective bound of the picked parents (scheduling only); 1-D dm_lb is one bound per domain."""
    if dm_lb is None or not dm_lb.numel():
        return None
    return (dm_lb if dm_lb.ndim < 2 else dm_lb[:, 0]).min()


def _extract_picked_objective_lower_bounds(
    domain_lower_bounds: Optional[Tensor],
    *,
    batch_size: int,
) -> Optional[Tensor]:
    """Extract one objective lower bound per picked domain, or disable safely."""
    if domain_lower_bounds is None or int(batch_size) <= 0:
        return None
    bounds = torch.as_tensor(domain_lower_bounds)
    if bounds.ndim == 0:
        return bounds.reshape(1) if batch_size == 1 else None
    if bounds.ndim == 1:
        if bounds.numel() == batch_size:
            return bounds
        if batch_size == 1 and bounds.numel() > 0:
            return bounds[:1]
        return None
    if bounds.shape[0] != batch_size:
        return None
    flattened = bounds.reshape(batch_size, -1)
    if flattened.shape[1] == 0:
        return None
    return flattened[:, 0]


def _is_initial_primal_dual_inconsistent(
    primal_solution: Tensor,
    dual_solution: Tensor,
    *,
    abs_tol: float,
) -> bool:
    """Return whether a finite root incumbent contradicts the root lower bound."""
    primal = torch.as_tensor(primal_solution).reshape(-1)
    dual = torch.as_tensor(
        dual_solution, device=primal.device, dtype=primal.dtype
    ).reshape(-1)
    if primal.numel() != 1 or dual.numel() != 1:
        raise ValueError("initial primal and dual solutions must both be scalar")
    if not (torch.isfinite(primal[0]) and torch.isfinite(dual[0])):
        return False
    return float((primal[0] - dual[0]).item()) < -abs_tol


def _plane_minimum_down(lA, lbias, x_L, x_U):
    """Minimum of each parent's objective plane (row 0 of lA, lbias) over its box, never above the exact value, barring
    underflow (a product in the subnormal range has an absolute error the relative margin does not cover).

    With c the centre and r the half-width of the box, the minimum is sum_i (a_i c_i - |a_i| r_i) + b. Its float value is off
    by at most ((1 + u)^(n + 3) - 1) S, about (n + 3) u S, with S = sum_i |a_i| (|c_i| + r_i) + |b| and u the unit roundoff:
    each term rounds three times (centre, product, difference) and the sum n times. The margin is (n + 4) u S, one unit more
    than the first-order count (E1 and results.md R3.3 had n + 2, which is one short): the extra unit covers the second-order
    terms and the rounding of the margin itself while 2 (n + 4)^2 u <= 1, that is n^2 u < 1 up to the factor 2 (n <= 2,900 in
    fp32, 6.7e7 in fp64). One step toward -inf follows for the subtraction. Of 5,000 random planes and boxes in each of fp32
    and fp64 (tests/test_child_key.py) none is above the exact minimum, against 2,443 / 2,515 for nearest rounding. The result
    is NaN or infinite where the plane is, and the caller skips those.
    """
    xl, xu = x_L.flatten(1), x_U.flatten(1)
    a = lA.flatten(2)[:, 0]
    b = lbias.reshape(lbias.shape[0], -1)[:, 0]
    c, r = (xl + xu) / 2, (xu - xl) / 2
    plane_min = (a * c - a.abs() * r).sum(1) + b
    margin = (xl.shape[1] + 4) * torch.finfo(xl.dtype).eps / 2 * ((a.abs() * (c.abs() + r)).sum(1) + b.abs())
    return torch.nextafter(plane_min - margin, torch.full_like(plane_min, -math.inf))


def _raise_child_keys(dm_lb, x_L, x_U, lA, lbias):
    """Objective bound of the children of a split: max(the parent's bound, the parent's objective plane minimised over the
    clipped child box). The plane is a lower bound on the points of the parent box that satisfy the rows it was bounded with
    (not on the whole box), and the parent's bound holds on the same points, so both terms bound the child from below there,
    and the reported dual, which reads the keys, stays a bound. The child on the far side of the plane's minimiser now ranks
    above its parent's bound instead of inheriting it. With the sort every iteration (gear.yaml) the loop closes in 0.73-1.00x
    of the old loop's iterations at the optimal incumbent (median 0.94, the 12 CPU instances where both close, E1a), and a
    wrong incumbent costs a median 1.13x of those iterations (+100 eps and far, E1a) instead of 2.1x and 2.9x (E1, old
    loop)."""
    plane_min = _plane_minimum_down(lA, lbias, x_L, x_U)
    assert plane_min.dtype == dm_lb.dtype, "a key stored in a narrower dtype would round to nearest, possibly upward"
    keys = dm_lb.reshape(dm_lb.shape[0], -1).clone()
    usable = (x_L.flatten(1) <= x_U.flatten(1)).all(1) & torch.isfinite(plane_min)
    keys[:, 0] = torch.where(usable, torch.maximum(keys[:, 0], plane_min), keys[:, 0])
    return keys.reshape(dm_lb.shape)


def _get_clip_base_constraints(
    base_constraints: Optional[tuple],
) -> Optional[tuple]:
    """Return original input constraints for the clipping kernel.

    The box-clipping kernel is used in both top-level "relaxed" and "complete"
    modes. Legacy explicit-output NLP instances implicitly included structural
    linear rows in this kernel through lA; compact instances must forward the
    original problem constraints explicitly to preserve parity.
    """
    return standard_form_to_input_constraints(base_constraints)


def _primal_input_rows(base_constraints, input_rows_raw):
    """The base rows as A x <= b for the primal (ALM, IPOPT, verify). The standard-form
    `base_constraints` hold the problem's own rows relaxed by the feasibility tolerance
    (relax_input_constraints); the primal applies that tolerance itself, so those rows,
    always the first ones, go back to their raw rhs (`input_rows_raw`, A x <= b). Rows
    appended after them (cuts) are shared by both."""
    rows = standard_form_to_input_constraints(base_constraints)
    if rows is None or input_rows_raw is None:
        return rows
    A, b = rows
    raw_b = input_rows_raw[1][:1].to(b)
    return A, torch.cat([raw_b.expand(b.shape[0], -1), b[:, raw_b.shape[1]:]], dim=1)


def _offer_convex_row_cuts(domains, base_constraints, constraints, x_hat, primal, batch_size):
    """Append new global cuts to the shared base and the currently picked batch."""
    row_cuts = getattr(domains.history, 'convex_row_cuts', None)
    if row_cuts is None:
        return base_constraints, constraints
    cuts = row_cuts.offer(x_hat, primal)
    if cuts is None:
        return base_constraints, constraints
    cuts = tuple(t.expand(batch_size, *t.shape[1:]) for t in cuts)
    base_constraints = merge_standard_constraints(base_constraints, cuts)
    constraints = merge_standard_constraints(constraints, cuts)
    # Queued domains receive the enlarged base on their next pickout. Their old
    # domain-specific cuts remain valid and need no rewriting.
    domains.base_constraint_A = base_constraints[0][:1].detach().clone().cpu()
    domains.base_constraint_b = base_constraints[1][:1].detach().clone().cpu()
    print(f"[ConvexRowCuts] round {row_cuts.rounds}: added {cuts[0].shape[1]} global rows")
    return base_constraints, constraints


@retained
def batch_verification_input_split(
        d, net, batch, num_iter, decision_thresh, shape=None,
        bounding_method="crown", branching_method="sb",
        stop_criterion=stop_criterion_batch_any, split_partitions=2,
        use_reordered_bab=False, stats=None, primal_solution=None,
        primal_solution_found=False, nlp_config=None, alm_state=None, _sliced=False):
    """
    General function of the batch_verification_input_split method.

    @param d:                   Domain list
    @param net:                 Bounded neural network
    @param batch:               Number of effective batches to evaluate
    @param num_iter:            The current iteration number of the input BaB run
    @param decision_thresh:     The specification threshold to verify against
    @param shape:               The shape of the network's input
    @param bounding_method:     The method to use when bounding the subdomains of the network
    @param branching_method:    The branching heuristic to use when splitting on input dimensions
    @param stop_criterion:      Criterion to stop naive lower bound of network
    @param split_partitions:    The number of partitions to create for subdomains, currently is always 2 for input split
    @param use_reordered_bab:   Whether to use the reordered BaB order
        - If True, the order of operations is:
            - Reorder:
                pickout -> (attack) -> bounding -> filter -> split -> decision -> (clip) -> (child key) -> add_domain
            - Regular:
                pickout -> (attack) -> split -> bounding -> filter -> (clip) -> decision -> add_domain
    @param stats:               Statistics object to record the time and visited domains

    @return:                   Tuple of (global lower bound, primal solution, primal_solution_found)
    """
    # --- INITIALIZATION STEP --- #
    input_split_args = arguments.Config["bab"]["branching"]["input_split"]
    split_hint = input_split_args['split_hint']
    clip_input_args = arguments.Config["bab"]["clip_n_verify"]["clip_input_domain"]
    enable_clip_domains = clip_input_args['enabled']
    clip_iters = clip_input_args['clip_iterations']
    clip_calculate_volume_metrics = clip_input_args['clip_calculate_volume_metrics']
    enable_constrained_concretize = enable_clip_domains and (clip_input_args["clip_type"] == "complete")
    solving_mode = arguments.Config["solving"]["solving_mode"]
    if nlp_config is None:
        nlp_config = arguments.Config["solver"]["nlp"]

    # --- PICKOUT STEP --- #
    stats.timer.start("pickout")
    rowpool = getattr(d.history, 'rowpool', None)
    ret = d.pick_out_batch(batch, device=net.device)
    _remember_pickout(d, ret)
    if (num_iter > 1 and getattr(d.history, 'controller_primal_ready', False)
            and getattr(d.history, 'controller_primal_batch', None) is None):
        d.history.controller_primal_batch = d._last_pickout
    dm_lb, x_L, x_U, cs, thresholds, constraints, base_constraints, spec_sizes, split_idx, attack_counter, split_counter = ret
    # M1's controller measures termination progress from the parents' minimum (scheduling only).
    d.history.m1_floor = _m1_floor(dm_lb)
    if isinstance(getattr(d.history, "lifted_box", None), tuple):
        from input_split.lifted_bound import clamp_to_lifted_box
        x_L, x_U = clamp_to_lifted_box(d.history, x_L, x_U)
    #TODO: Essentially, attack_counter and split_counter and solving mode should be distangled.
    #       But for now, we would only enable attack_counter and split_counter in solving mode.
    pickout_batch = len(x_L)
    print(f"Current pickout batch: {pickout_batch}")
    if os.environ.get("SPLIT_DIAG") == "1":
        _diag_note_pickout(x_L, x_U)
    stats.timer.add('pickout')

    # --- ATTACK STEP (If required) --- #
    if input_split_args["update_rhs_with_attack"]:
        assert spec_sizes.unique().numel() == 1, "Currently only support same number of AND specs for update_rhs_with_attack"
        stats.timer.start('update_rhs_with_attack')
        if arguments.Config['model']['with_jacobian']:
            model_to_attack = net.net
        else:
            model_to_attack = net.model_ori
        thresholds = update_rhs_with_attack(x_L, x_U, cs, thresholds, dm_lb, model_to_attack)
        stats.timer.add('update_rhs_with_attack')

    if alm_state is not None:
        thresholds = _apply_primal_objective_threshold(
            thresholds,
            1e30 if os.environ.get("NLP_OBJ_RHS_PROBE") == "1" else 0.0,
        )
        objective_lower_bound = _extract_picked_objective_lower_bounds(
            dm_lb, batch_size=pickout_batch
        )
        picked_batch = PickedDomainBatch(
            dm_l=x_L,
            dm_u=x_U,
            c=cs,
            rhs=thresholds,
            spec_size=spec_sizes,
            objective_lower_bound=objective_lower_bound,
            attack_counter=attack_counter,
            base_input_linear_constraints=_primal_input_rows(
                base_constraints, getattr(d.history, 'input_rows_raw', None)),
            base_input_constraints=base_constraints,
            domain_constraints=constraints,
        )
        primal_solution, primal_solution_found = run_bab_alm_step(
            net,
            d,
            picked_batch,
            nlp_config,
            primal_solution,
            primal_solution_found,
            alm_state,
            stats,
            num_iter,
            solving_mode,
        )
    if solving_mode and primal_solution is not None:
        thresholds = _apply_primal_objective_threshold(thresholds, primal_solution)
        if ((nlp_config.get("convex_row_cuts", False) or nlp_config.get("convex_tangent_rows", False))
                and getattr(d.history, 'convex_row_cuts', None) is not None):
            base_constraints, constraints = _offer_convex_row_cuts(
                d, base_constraints, constraints,
                getattr(d.history, 'alm_best_primal_x', None), primal_solution, pickout_batch)
    addon = getattr(net, 'm1_addon', None)
    sigma_main = addon is not None and addon.sigma_main and addon.select_main() == 'sigma'
    # M3 commits children without bounding them, bypassing M1 (A161): it yields while M1 is on.
    if (solving_mode and nlp_config.get('m3_reduce', False) and getattr(d.history, 'm3_pattern', None) is not None and not sigma_main
            and (addon is None or not addon.m3_yield())):
        # --- M3 COVER STEP: propagate both children of each picked box, commit them (input_split/m3_reduce.py) --- #
        from input_split.m3_reduce import continue_cover
        stats.timer.start('m3')
        global_lb = continue_cover(d, (dm_lb, x_L, x_U, cs, thresholds, constraints, base_constraints, spec_sizes,
                                       split_idx, attack_counter, split_counter), primal_solution, stats, nlp_config)
        stats.timer.add('m3')
        return global_lb, primal_solution, primal_solution_found
    # --- SPLIT STEP (Normal) --- #
    if not use_reordered_bab:
        stats.timer.start('split')
        if not input_split_args["compare_with_old_bounds"] and not (addon is not None and addon.sigma_main):
            dm_lb = None
        x_L, x_U, cs, thresholds, dm_lb, spec_sizes, _, _, constraints, base_constraints, attack_counter, split_counter = input_split_and_repeat(
            x_L, x_U, pickout_batch, shape, split_idx, split_partitions, split_hint,
            cs, thresholds, dm_lb, d.history, spec_sizes, None, None, constraints, base_constraints, attack_counter, split_counter)
        stats.timer.add('split')

    # --- BOUNDING STEP --- #
    dm_lb, lA, lbias, x_star = yield from invoke(get_bounds,
        net, dm_lb, x_L, x_U, d.history, cs, thresholds, constraints,
        standard_form_to_input_constraints(base_constraints),
        bounding_method, branching_method, stop_criterion, stats, sliced=_sliced,
        **({"_sliced": True} if _sliced else {})
    )
    sigma_main = addon is not None and addon.sigma_main and lA is None

    assert not dm_lb.isnan().any()
    # When the clip type is complete, linear constraints will be applied when concretizing.
    if enable_constrained_concretize and not sigma_main:
        # --- Initialize constraints --- #
        batch_size = x_L.shape[0]
        x_dim = x_L.flatten(1).shape[1]
        concretize_constraints = construct_cut_constraints(lA, lbias, thresholds, batch_size, x_dim, rowpool)
        constraints = merge_standard_constraints(base_constraints, concretize_constraints)
    else:
        constraints = base_constraints

    # --- Filter out verified subdomains --- #
    stats.timer.start('filtering')
    # Since we have only bounded the domains and not clipped them, we only need to check thresholds
    if solving_mode:
        print("================= Dual Filter Report ================= ")
    ret_filt = UnsortedInputDomainList.filter_verified_domains(len(x_L), dm_lb, x_L, x_U,
                                        cs, thresholds, d.history, lA, lbias, constraints, base_constraints,
                                        spec_sizes=spec_sizes,
                                        check_dm_lbs=True, check_input_boxes=False,
                                        attack_counter=attack_counter, split_counter=split_counter,
                                        x_star=x_star)
    num_unverified_domains, dm_lb, x_L, x_U, cs, thresholds, lA, lbias, constraints, base_constraints, _, spec_sizes, attack_counter, split_counter, x_star = ret_filt
    assert not dm_lb.isnan().any()
    stats.timer.add('filtering')

    # When num_unverified_domains > 0, there are still unverified subdomains after filtering.
    # It will be processed to decision and add to domain in the following steps.
    if num_unverified_domains > 0:

        # --- CLIP STEP (Normal) (If required) --- #
        if not use_reordered_bab and enable_clip_domains and not sigma_main:
            stats.timer.start("clip")
            x_L, x_U = clip_domains(x_L, x_U, thresholds, lA, lbias,
                                num_iters=clip_iters,
                                calculate_volume=clip_calculate_volume_metrics,
                                base_input_linear_constraints=_get_clip_base_constraints(
                                    base_constraints
                                ),
                                base_input_constraints=base_constraints,
                                domain_constraints=constraints, rowpool=rowpool)
            stats.timer.add("clip")

        # --- DECISION STEP --- #
        stats.timer.start("decision")
        split_idx = input_split_branching(
                net, dm_lb, x_L, x_U, lA, thresholds,
                branching_method, stats.storage_depth, num_iter=num_iter,
                history=d.history,
            )
        stats.timer.add("decision")
        sb_unc = thresholds.shape[1] == 1 and base_constraints is None
        sb_k = _strong_candidates(input_split_args, thresholds.shape[1],
                                  base_constraints is not None, x_L.flatten(1).shape[1])
        sb_n = _strong_domain_count(input_split_args, x_L.shape[0], sb_unc)
        sb_max_dom = -1 if sb_n >= x_L.shape[0] else sb_n
        if (sb_k != 0 and branching_method == "sb" and lA is not None
                and torch.isfinite(lA).all() and torch.isfinite(dm_lb).all()):
            stats.timer.start("strong_branch")
            global _SB_STRONG_CHILD_CAP
            max_children = int(input_split_args.get("sb_strong_max_children", 65536))
            if _SB_STRONG_CHILD_CAP is not None:
                max_children = min(max_children, _SB_STRONG_CHILD_CAP)
            try:
                split_idx = _strong_branching_split(
                    net, d.history, dm_lb, x_L, x_U, cs, thresholds, constraints,
                    base_constraints, lA, lbias, split_idx, sb_k, bounding_method,
                    stop_criterion, stats, max_children, max_domains=sb_max_dom)
            except torch.OutOfMemoryError:
                # The lookahead is optional: keep the sb choice for this batch and
                # bound fewer candidate children per call from now on.
                torch.cuda.empty_cache()
                _SB_STRONG_CHILD_CAP = max(1, max_children // 2)
                print(f"[strong-branch] OOM; sb choice kept, chunk -> {_SB_STRONG_CHILD_CAP}")
            except Exception as e:  # pylint: disable=broad-except
                # e.g. a bound-propagation domain error on a candidate child that the
                # loop itself never bounds: fall back to sb for this batch.
                print(f"[strong-branch] skipped ({type(e).__name__}: {e}); sb choice kept")
            stats.timer.add("strong_branch")

        # Last word on the choice, so it holds over the lifted, angle-first and strong-branching choices.
        if split_counter is not None:
            split_idx, split_counter = _exhaustive_subdivision_guard(split_idx, split_counter, x_L, x_U, d.history)

        if use_reordered_bab:

            # --- SPLIT STEP (Reordered) --- #
            stats.timer.start('split')
            # This workload consists of all pending items (num_unverified_domains + len(d))
            # but it will not exceed the system's batch limit.
            if not input_split_args["compare_with_old_bounds"] and not (addon is not None and addon.sigma_main):
                dm_lb = None
            parent_l, parent_u = x_L, x_U
            x_L, x_U, cs, thresholds, dm_lb, spec_sizes, lA, lbias, constraints, base_constraints, attack_counter, split_counter = input_split_and_repeat(
                    x_L, x_U, num_unverified_domains, shape, split_idx, split_partitions,
                    split_hint, cs, thresholds, dm_lb, d.history, spec_sizes, lA, lbias, constraints, base_constraints,
                    attack_counter=attack_counter, split_counter=split_counter)
            if ((nlp_config.get("lifted_lp_split_at_point", False) or getattr(d.history, "node_pool", False))
                    and torch.is_tensor(getattr(d.history, "lifted_split_point", None))):
                from input_split.lifted_bound import apply_lifted_split_points
                x_L, x_U = apply_lifted_split_points(d.history, parent_l, parent_u, split_idx,
                                                     x_L, x_U)
            stats.timer.add('split')

            # --- CLIP STEP (Reordered) (If required) --- #
            if enable_clip_domains and not sigma_main:
                stats.timer.start('clip')
                x_L, x_U = clip_domains(x_L, x_U, thresholds, lA, lbias,
                                        num_iters=clip_iters,
                                        calculate_volume=clip_calculate_volume_metrics,
                                        base_input_linear_constraints=_get_clip_base_constraints(
                                            base_constraints
                                        ),
                                        base_input_constraints=base_constraints,
                                        domain_constraints=constraints, rowpool=rowpool)
                stats.timer.add('clip')

            # --- CHILD KEY STEP (Reordered) (Solving mode, the objective row) --- #
            if solving_mode and dm_lb is not None and lA is not None and lbias is not None:
                dm_lb = _raise_child_keys(dm_lb, x_L, x_U, lA, lbias)

        # --- ADD DOMAIN STEP --- #
        stats.timer.start('add_domain')
        # Clipping only updates the input bounds but not the dm_lb
        split_idx = None if use_reordered_bab else split_idx
        d.add(dm_lb, x_L.detach(), x_U.detach(),
              cs, thresholds,
              constraints=None if addon is not None and addon.sigma_main else constraints,
              base_constraints=base_constraints, split_idx=split_idx,
              check_dm_lbs=solving_mode, check_input_boxes=enable_clip_domains,
              spec_sizes=spec_sizes, attack_counter=attack_counter, split_counter=split_counter,
              x_star=x_star)
        stats.timer.add('add_domain')

    # --- SUMMARIZATION STEP --- #
    stats.timer.start('summary')
    len_domains = len(d)
    if (os.environ.get("BRANCH_DIAG") == "1" and len_domains and solving_mode
            and primal_solution is not None):
        _n = d.lb.num_used
        _q = d.lb._storage[:_n].reshape(_n, -1)[:, 0].double()
        _P = float(torch.as_tensor(primal_solution).reshape(-1)[0])
        _tol = max(1e-5, 1e-4 * abs(_P))
        print(f"[branch-diag] queue={_n} lb0>=P-tol: {int((_q >= _P - _tol).sum())} "
              f"lb0>=P-tol/2: {int((_q >= _P - _tol / 2).sum())} picked={pickout_batch}")
    if os.environ.get("SPLIT_DIAG") == "1":
        try:
            _print_split_diagnostics(d, num_iter, pickout_batch, dm_lb, x_L, x_U, split_idx,
                                     num_unverified_domains)
        except Exception as e:  # pylint: disable=broad-except
            print(f"[split-diag] skipped: {e!r}")

    if len_domains == 0:
        if solving_mode:
            if primal_solution is not None and torch.isfinite(primal_solution).all():
                print("No domains left, optimal solution found!")
                return primal_solution, primal_solution, primal_solution_found
            print("No domains left and no feasible primal solution was found.")
            infeasible_optimum = torch.tensor(
                float("inf"),
                device=decision_thresh.device,
                dtype=decision_thresh.dtype,
            )
            if primal_solution is None:
                primal_solution = infeasible_optimum.reshape(1)
            return infeasible_optimum, primal_solution, primal_solution_found
        else:
            print("No domains left, verification finished!")
            if dm_lb is not None and len(dm_lb) > 0:
                dm_lb_min = dm_lb.min().item()
                print(f"The lower bound of last batch is {dm_lb_min}")
            _print_final_results(stats, 0, thresholds)
            return decision_thresh.max() + 1e-7, primal_solution, primal_solution_found
    else:
        # TODO: Here, we'll need to set sort_index and sort_heuristic for the get_topk_indices()
        #       But setting an member attribute outside of a class is undesirable. Need to change it. 
        if solving_mode:
            sort_index = d.sort_index
            sort_heuristic = d.sort_heuristic
            d.sort_index = 0
            d.sort_heuristic = "lb_only"
        if input_split_args["skip_getting_worst_domain"]:
            # It can be costly to call get_topk_indices when the domain list is long
            worst_idx = 0
        else:
            worst_idx = d.get_topk_indices()
        worst_val = d[worst_idx]
        # `d[idx]` returns (lb, dm_l, dm_u, cs, threshold, spec_size, x_star).
        if solving_mode:
            # The minimum over the SURVIVING domains is not a bound on the problem.
            # Domains are discarded once `lb > threshold`, so everything discarded has a
            # true minimum above that threshold, and the valid global bound is
            #     min(min over survivors, threshold).
            # Reporting the survivors' minimum alone lets the dual climb past the true
            # optimum the moment the domain holding it is discarded -- and that domain is
            # discarded as soon as its bound reaches the incumbent, which for an exact
            # incumbent happens at a one-ulp margin. On ns_vorticity the objective row of
            # the domain containing a verified feasible point read
            # lb=-0.03899354541694431 against threshold=-0.03899354541694432, one ulp
            # apart; after it went, the survivors put the reported dual 6.889e-4 ABOVE a
            # point that is actually attainable.
            # Cap the objective row only, and keep the tensor's shape: callers index
            # global_lb[0, 0]. The constraint rows are not bounds on the objective.
            # Cap against the CURRENT filtering threshold, not the one each domain
            # happens to have stored: a domain keeps whatever rhs was in force when it
            # was added, which early on can still be +inf, and min(lb, inf) = lb caps
            # nothing. `decision_thresh` is the threshold domains are actually being
            # discarded against right now.
            global_lb = worst_val[0].clone()
            _cap = arguments.Globals.get("discarded_threshold_min")
            if _cap is not None and math.isfinite(_cap):
                if global_lb.dim() >= 2:
                    global_lb[..., 0] = torch.clamp(global_lb[..., 0], max=_cap)
                else:
                    global_lb = torch.clamp(global_lb, max=_cap)
        else:
            global_lb = worst_val[0] - worst_val[-2]
        _print_final_results(stats, len_domains, global_lb)
        # TODO: Should set this in the config. Make "solving_mode" and "print (lb-rhs)" mutually exclusive.
        if not input_split_args["skip_getting_worst_domain"] and not solving_mode:
            if 1 <= global_lb.numel() <= 5:
                print(f"Current (lb-rhs): {global_lb}")
            else:
                print(f"Current (lb-rhs): {global_lb.max().item()}")
        if solving_mode:
            d.sort_index = sort_index
            d.sort_heuristic = sort_heuristic

    if input_split_args["show_progress"]:
        print(f"Progress: {d.get_progess():.10f}")
    sys.stdout.flush()

    return global_lb, primal_solution, primal_solution_found

_SB_STRONG_CHILD_CAP = None


def _strong_domain_count(input_split_args, n_domains, unconstrained):
    """How many domains of a batch (the lowest objective bounds) get the lookahead.

    ceil(frac * n) clamped to [sb_strong_min_domains, sb_strong_max_domains]; frac is
    sb_strong_domain_frac, or sb_strong_domain_frac_unconstrained on objective-only
    problems. On elena_s383 strong-branching only the 10k lowest of 262k domains per
    batch (4%) was what closed the gap; at small batches a fraction keeps the lookahead
    from multiplying the bounding work (6x with top-3 when every domain gets it)."""
    key = "sb_strong_domain_frac_unconstrained" if unconstrained else "sb_strong_domain_frac"
    frac = float(input_split_args.get(key, 1.0))
    n = n_domains if frac >= 1.0 else int(math.ceil(frac * n_domains))
    n = max(n, int(input_split_args.get("sb_strong_min_domains", 0)))
    max_d = int(input_split_args.get("sb_strong_max_domains", -1))
    if max_d >= 0:
        n = min(n, max_d)
    return max(0, min(n, n_domains))


def _strong_candidates(input_split_args, n_spec, has_base_constraints, n_inputs):
    """Lookahead candidates per domain for this problem (0 = no strong branching).

    Objective-only problems (one spec row, no input linear constraints) use their own
    count, since nothing couples the inputs through an LP there and the first-order sb
    ranking is a poor filter for curvature-driven gains; they can also be limited to
    few inputs."""
    k = int(input_split_args.get("sb_strong_candidates", 0))
    if n_spec == 1 and not has_base_constraints:
        k = int(input_split_args.get("sb_strong_candidates_unconstrained", k))
        max_in = int(input_split_args.get("sb_strong_max_inputs_unconstrained", 0))
        if 0 < max_in < n_inputs:
            k = 0
    return k


def _strong_branching_split(net, history, dm_lb, x_L, x_U, cs, thresholds, constraints,
                            base_constraints, lA, lbias, split_idx, n_candidates,
                            bounding_method, stop_criterion, stats, max_children,
                            max_domains=-1):
    """Replace the first split choice by objective-aware strong branching over the top
    sb candidates (input_split/strong_branching.py). Only the `max_domains` domains with
    the lowest objective bound are strong-branched (all if negative); the others keep
    the sb choice. The exhaustiveness guard still applies to the result."""
    from input_split.branching_heuristics import _compute_sb_score, _exhaustive_guard
    from input_split.strong_branching import strong_branching_decision, _repeat_rows
    xl, xu = x_L.flatten(1), x_U.flatten(1)
    base = _compute_sb_score(xl, xu, dm_lb, thresholds, lA)
    B = xl.shape[0]
    sel = None
    if 0 <= max_domains < B:
        if max_domains == 0:
            return split_idx
        sel = torch.topk(dm_lb[:, 0], max_domains, largest=False).indices

    def take(t):
        if t is None or sel is None:
            return t
        if isinstance(t, tuple):
            return tuple(take(v) for v in t)
        if t.dim() > 0 and t.stride(0) == 0:
            return t.narrow(0, 0, 1).expand(sel.numel(), *t.shape[1:])
        return t[sel.to(t.device)]

    s_lb, s_L, s_U, s_cs, s_thr = take(dm_lb), take(x_L), take(x_U), take(cs), take(thresholds)
    s_cons, s_base, s_lA, s_lbias = take(constraints), take(base_constraints), take(lA), take(lbias)
    clip = None
    rowpool = getattr(history, 'rowpool', None)
    clip_args = arguments.Config["bab"]["clip_n_verify"]["clip_input_domain"]
    if clip_args["enabled"] and lbias is not None:
        def clip(cl, cu, s, e, reps):
            rep = lambda t: None if t is None else _repeat_rows(t[s:e], reps)
            base_rep = None if s_base is None else tuple(rep(t) for t in s_base)
            cons_rep = None if s_cons is None else tuple(rep(t) for t in s_cons)
            return clip_domains(cl, cu, rep(s_thr), rep(s_lA), rep(s_lbias),
                                num_iters=clip_args["clip_iterations"],
                                base_input_linear_constraints=_get_clip_base_constraints(base_rep),
                                base_input_constraints=base_rep, domain_constraints=cons_rep,
                                rowpool=rowpool)
    chosen = strong_branching_decision(
        net, history, s_lb, s_L, s_U, s_cs, s_thr, s_cons,
        standard_form_to_input_constraints(s_base), take(base), take(split_idx)[:, :1],
        n_candidates, bounding_method, stop_criterion, stats, max_children, clip=clip)
    split_idx = split_idx.clone()
    if sel is None:
        split_idx[:, :1] = chosen
    else:
        split_idx[sel.to(split_idx.device), :1] = chosen.to(split_idx)
    return _exhaustive_guard(split_idx, xl, xu, history)


_DIAG_ROOT_WIDTH = None


def _diag_note_pickout(x_L, x_U):
    """Track the widest box ever picked out per input dim (the root, from iteration 1)."""
    global _DIAG_ROOT_WIDTH
    with torch.no_grad():
        w = (x_U - x_L).reshape(x_L.shape[0], -1).to(torch.float64).max(0).values
        _DIAG_ROOT_WIDTH = w.clone() if _DIAG_ROOT_WIDTH is None else torch.maximum(_DIAG_ROOT_WIDTH, w)


def _print_split_diagnostics(d, num_iter, pickout_batch, dm_lb, x_L, x_U, split_idx,
                             num_unverified):
    """SPLIT_DIAG=1: one line per BaB iteration with the branching / bound-quality
    facts the log does not otherwise show -- which input dims the survivors will be split
    on next, how many of the batch's objective bounds are non-finite or astronomically
    loose (|lb| > 1e6), and the queue's halving
    depth per input dim (relative to the widest box ever picked out, i.e. the root)."""
    global _DIAG_ROOT_WIDTH
    with torch.no_grad():
        x_L_f = x_L.reshape(x_L.shape[0], -1) if x_L is not None else None
        parts = [f"[split-diag] it={num_iter} picked={pickout_batch} survivors={num_unverified} queue={len(d)}"]
        if dm_lb is not None and dm_lb.numel():
            lb0 = dm_lb[:, 0]
            parts.append(f"lb0: nonfinite={int((~torch.isfinite(lb0)).sum())} huge={int((torch.isfinite(lb0) & (lb0.abs() > 1e6)).sum())} "
                         f"min={lb0[torch.isfinite(lb0)].min().item() if torch.isfinite(lb0).any() else float('nan'):.3e}")
        if split_idx is not None and split_idx.numel() and x_L_f is not None:
            D = x_L_f.shape[1]
            hist = torch.bincount(split_idx[:, 0].reshape(-1).long().cpu(), minlength=D).tolist()
            parts.append(f"next_split_dim_hist={hist}")
            if os.environ.get("SPLIT_DIAG_WORST") == "1" and dm_lb is not None and dm_lb.shape[0] == x_L_f.shape[0]:
                j = int(dm_lb[:, 0].argmin())
                x_U_f = x_U.reshape(x_U.shape[0], -1)
                w = (x_U_f[j] - x_L_f[j]).tolist()
                parts.append(f"WORST j={j} lb0={dm_lb[j, 0].item():.10f} split={split_idx[j].tolist()} "
                             f"widths={[f'{v:.2e}' for v in w]} "
                             f"L={[f'{v:.6f}' for v in x_L_f[j].tolist()]}")
        if len(d) and d.dm_l is not None:
            n = d.dm_l.num_used
            q_L = d.dm_l._storage[:n].reshape(n, -1)
            q_U = d.dm_u._storage[:n].reshape(n, -1)
            w = (q_U - q_L).to(torch.float64)
            if _DIAG_ROOT_WIDTH is None or _DIAG_ROOT_WIDTH.shape != w[0].shape:
                _DIAG_ROOT_WIDTH = w.max(0).values.clone()
            root_w = _DIAG_ROOT_WIDTH.to(w.device)                      # the domain list may live on CPU
            halvings = torch.log2(root_w.unsqueeze(0) / w.clamp(min=1e-300)).round()
            depth = halvings.sum(1)
            parts.append(f"queue depth min/med/max={int(depth.min())}/{int(depth.median())}/{int(depth.max())} "
                         f"mean_halvings_per_dim={[round(v, 2) for v in halvings.mean(0).tolist()]} "
                         f"never_halved_frac_per_dim={[round(v, 2) for v in (halvings == 0).double().mean(0).tolist()]}")
            if d.lb is not None and d.lb.num_used == n:
                qlb = d.lb._storage[:n].reshape(n, -1)[:, 0].to(torch.float64)
                parts.append(f"queue lb0: huge_frac={(torch.isfinite(qlb) & (qlb.abs() > 1e6)).double().mean().item():.2f} "
                             f"nonfinite_frac={(~torch.isfinite(qlb)).double().mean().item():.2f}")
        print(" ".join(parts))


def _clone_tree(v, device=None):
    if isinstance(v, Tensor):
        v = v.detach()
        def copy(t):
            return t.clone() if device is None else t.to(device, copy=True)
        if v.dim() > 0 and v.shape[0] > 1 and v.stride(0) == 0:
            # A broadcast over the batch (the shared spec matrix of pick_out_batch):
            # one row is all it holds. clone() would materialize every copy -- 862 MiB
            # per iteration on kall_circlespolygons_c1p13, which then ran out of memory.
            return copy(v[:1]).expand_as(v)
        return copy(v)
    if isinstance(v, tuple):
        return tuple(_clone_tree(x, device) for x in v)
    return v


def _remember_pickout(d, ret):
    """Keep an untouched copy of a pickout so it can go back to the queue if the
    batch runs out of memory (the processing may modify the picked tensors).

    The copy lives on the host (where UnsortedInputDomainList stores domains):
    putting it back then allocates little or nothing on the GPU, which may still be
    full when the retry runs (MeyerRoth: 27.7 GiB still held and a 24 MiB index
    allocation failed).

    Also record how many rows every per-domain storage holds right after the pop: an
    out-of-memory error inside `d.add` can strike after some storages were appended
    and before others were, and the restore must undo such a partial add."""
    d._last_pickout = _clone_tree(tuple(ret), device="cpu")
    sizes = getattr(d, "storage_sizes", None)
    d._last_pickout_sizes = sizes() if sizes is not None else None


def _restore_pickout(d) -> bool:
    """Put the remembered pickout back into the domain list, once."""
    saved = getattr(d, "_last_pickout", None)
    d._last_pickout = None
    if saved is None:
        return False
    # Failed insertion may have changed only part of the incremental measure.
    # Rebuild from the restored committed cover at its next safe boundary.
    if hasattr(d, 'controller_measure'):
        d.controller_measure = None
    (dm_lb, x_L, x_U, cs, thresholds, constraints, base_constraints, spec_sizes,
     split_idx, attack_counter, split_counter) = saved
    sizes = getattr(d, "_last_pickout_sizes", None)
    d._last_pickout_sizes = None
    if sizes is not None:
        # Drop whatever a failed add appended to some storages but not to others;
        # otherwise the storages stay misaligned (boxes paired with other domains'
        # bounds and cut rows) for the rest of the run.
        d.truncate_storages(sizes)
    # The domains were already in the queue: no re-filtering (its base-constraint
    # bounds ran out of memory again on kall_circlespolygons_c1p13).
    # The base rows are problem-level and already stored; convex row cuts may have
    # widened them since the pickout (A87), and the domains get the current base when
    # they are picked again.
    if getattr(d, 'base_constraint_A', None) is not None:
        base_constraints = None
    d.add(dm_lb, x_L, x_U, cs, thresholds, constraints=constraints,
          base_constraints=base_constraints, split_idx=split_idx,
          remaining_index=torch.arange(dm_lb.shape[0], device=dm_lb.device),
          check_dm_lbs=False, check_input_boxes=False, spec_sizes=spec_sizes,
          attack_counter=attack_counter, split_counter=split_counter)
    return True


def _batch_after_oom(domains, batch_, auto_batch_size, m2=None, addon=None):
    """React to a batch that ran out of GPU memory; returns the next batch size.

    Call it outside the except block (its traceback keeps the failed batch's
    tensors alive). The cache is freed, the remembered pickout goes back to the
    queue, the batch is halved and enlargement stops. A single domain that does
    not fit switches to sparse main when opted in and an engine is available.
    Otherwise None stops the search open ("unknown") with its cover restored.
    """
    evicted_m2 = m2 is not None and m2.release()
    if evicted_m2:
        m2.pause_memory()
    torch.cuda.empty_cache()
    try:
        restored = _restore_pickout(domains)
    except (torch.OutOfMemoryError, RuntimeError) as error:
        # The picked domains could not go back: the search is no longer
        # exhaustive, so it must end open and never certify (see the caller).
        print(f"[OOM] the picked domains could not be put back ({error}); "
              "stopping the search")
        if not isinstance(error, torch.OutOfMemoryError):
            import traceback
            traceback.print_exc()   # a shape error here is a bug (A87, A90), not memory
        domains._domains_lost = True
        return None
    if not restored:
        raise RuntimeError("OOM retry without a remembered pickout")
    if batch_ <= 1 and addon is not None and addon.sigma_main and not addon.native_oom:
        addon.native_oom = True
        addon.select_main()
        batch = _sigma_batch_hint(addon, auto_batch_size)
        print(f"[OOM] a single native domain did not fit; switching to sigma-main batch={batch}")
        return batch
    if evicted_m2:
        print(f"[OOM] evicted M2 residency; retrying native batch {batch_}")
        return batch_
    if batch_ <= 1:
        print("[OOM] a single domain did not fit; stopping the search")
        return None
    batch = max(1, batch_ // 2)
    auto_batch_size.batch_size = batch
    auto_batch_size.enable = False
    print(f"[OOM] batch {batch_} did not fit; retrying with {batch}")
    return batch


def _clip_root_domains(dm_l, dm_u, *args, **kwargs):
    """clip_domains at the root; the unclipped boxes on OOM. Clipping only tightens the
    boxes (pooling_sppb5pq/b5tp/c1stp crashed in its (B, S, N) einsum)."""
    try:
        return clip_domains(dm_l, dm_u, *args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        print("[OOM] root clip_domains did not fit; keeping the unclipped boxes")
        torch.cuda.empty_cache()
        return dm_l, dm_u


def _add_root_domains(domains, lb, dm_l, dm_u, c, rhs, **kwargs):
    """domains.add for the root's domains. Its filter bounds the input constraints over
    every domain with a dense (batch, rows, vars) product; when that does not fit, the
    domains are filtered by their bounds and boxes alone -- the constraint filter only
    drops domains that are already verified, so keeping them is sound. Without it the
    root OOM ended the run (pooling_sppb5pq/b5tp/c1stp) after the root bound and the
    primal search had finished."""
    storage_sizes = getattr(domains, "storage_sizes", None)
    sizes = storage_sizes() if storage_sizes is not None else None
    try:
        return domains.add(lb, dm_l, dm_u, c, rhs, **kwargs)
    except torch.cuda.OutOfMemoryError:
        if sizes is None:
            raise
    print("[OOM] root input-constraint filter did not fit; "
          "filtering the domains by their bounds and boxes only")
    domains.truncate_storages(sizes)
    torch.cuda.empty_cache()
    remaining_index = UnsortedInputDomainList.get_remaining_index(
        lb.shape[0], lb, rhs, dm_l, dm_u,
        kwargs.get("check_dm_lbs", True), kwargs.get("check_input_boxes", True),
        rowpool=getattr(domains.history, "rowpool", None))
    return domains.add(lb, dm_l, dm_u, c, rhs, **dict(kwargs, remaining_index=remaining_index))


def _report_after_oom_stop(domains, stats, primal_solution):
    """Print the dual (the lowest objective bound over the open domains, capped by the
    discarded-domain threshold like batch_verification_input_split) and the primal when
    the search stops after an OOM at batch size 1: pooling_sppc1pq/c1tp/b2stp/c0stp
    ended with neither line. Returns the dual tensor, None when the domains were lost
    (then no bound is valid)."""
    if getattr(domains, "_domains_lost", False) or len(domains) == 0:
        return None
    domains.sort_index = 0
    domains.sort_heuristic = "lb_only"
    global_lb = domains[domains.get_topk_indices()][0].clone()
    cap = arguments.Globals.get("discarded_threshold_min")
    if cap is not None and math.isfinite(cap):
        global_lb[..., 0] = torch.clamp(global_lb[..., 0], max=cap)
    stats.timer.start('summary')    # the OOM left the iteration's summary timer unstarted
    _print_final_results(stats, len(domains), global_lb)
    sign = -1 if arguments.Config["solving"]["is_maximization"] else 1
    print(f"Primal Solution {sign * primal_solution}, Dual Solution {sign * global_lb[0, 0]}")
    _report.record_pd(sign * primal_solution, sign * global_lb[0, 0])
    _report.record_event("memory_limit")
    return global_lb


def _branching_method_without_lA(branching_method):
    """The sb score needs lA; without it (e.g. a root bound that returned no A
    matrix, Neumaier2) split the longest edge, as input_split_branching does for
    non-finite inputs, instead of aborting the run."""
    return "naive" if branching_method == "sb" else branching_method


def _root_structure(net, x, c, rhs, nlp_config, rowpool, algebra=None):
    """The shared structure report of the root problem, logged once. It merges the parallel rows of `rowpool`.
    algebra: the FormAlgebra class of its forms (structure/facts.build_structure)."""
    if not nlp_config.get("structure_report", True):
        return None
    from nlp_solver.structure import build_structure
    t0 = time.time()
    center = ((x.ptb.x_L + x.ptb.x_U) / 2).detach()
    try:
        report = build_structure(net.net, center[:1], c, rhs, x.ptb.x_L, x.ptb.x_U, pool=rowpool, algebra=algebra)
    except Exception as error:  # advisory analysis must never abort a solve; the reason stays visible
        print(f"[Structure] skipped: analysis failed ({type(error).__name__}: {error})")
        return None
    if report.skipped:
        print(f"[Structure] skipped: {report.skipped}")
        return report
    counts = collections.Counter(r.cls for r in report.rows[1:])
    print(f"[Structure] objective={report.objective.cls} rows={dict(counts)} "
          f"duplicates={sum(r.duplicate_of is not None for r in report.rows)} "
          f"opposites={sum(r.opposite_of is not None for r in report.rows)} "
          f"redundant={sum(r.redundant for r in report.rows)} "
          f"aliases={sum(r.alias is not None for r in report.rows)} "
          f"pool={report.pool.stats()} spent={report.spent} time={time.time() - t0:.3f}s")
    return report


def _add_symmetry_rows(domains, structure, rhs, x, base_constraints, incumbents):
    """Order interchangeable variable blocks (see nlp_solver/structure/symmetry.py): add x[B_t[0]] <=
    x[B_t+1[0]] as global rows, which complete clipping applies at every node. The optimal value is
    unchanged. Returns the enlarged base constraints and the incumbents sorted into the ordered region
    (they stay feasible with the same objective; the ordered region need not contain the originals)."""
    if structure is None or structure.skipped:
        return base_constraints, incumbents
    from nlp_solver.structure.symmetry import ordering_pairs, sort_blocks, symmetry_chains
    t0 = time.time()
    lo = x.ptb.x_L.reshape(x.ptb.x_L.shape[0], -1).amin(0).detach().cpu().double().numpy()
    hi = x.ptb.x_U.reshape(x.ptb.x_U.shape[0], -1).amax(0).detach().cpu().double().numpy()
    A = b = None
    if base_constraints is not None:
        if not (torch.equal(base_constraints[0], base_constraints[0][:1].expand_as(base_constraints[0]))
                and torch.equal(base_constraints[1], base_constraints[1][:1].expand_as(base_constraints[1]))):
            return base_constraints, incumbents          # per-domain rows: no detection
        A = base_constraints[0][0].detach().cpu().double().numpy()
        b = base_constraints[1][0].detach().cpu().double().numpy()
    chains = symmetry_chains(structure, rhs[0, 1:].detach().cpu().double().tolist(), lo, hi, A, b)
    if not chains:
        return base_constraints, incumbents
    pairs = ordering_pairs(chains)
    batch = x.ptb.x_L.shape[0]
    rows = torch.zeros(batch, len(pairs), lo.size, dtype=x.ptb.x_L.dtype, device=x.ptb.x_L.device)
    for k, (a, c) in enumerate(pairs):
        rows[:, k, a], rows[:, k, c] = 1, -1
    print(f"[Symmetry] {len(chains)} chains of interchangeable blocks "
          f"(blocks x size: {[(len(c), len(c[0])) for c in chains]}), {len(pairs)} ordering rows, "
          f"time={time.time() - t0:.3f}s")
    base_constraints = merge_standard_constraints(base_constraints, (rows, rows.new_zeros(batch, len(pairs))))
    domains.base_constraint_A = base_constraints[0][:1].detach().clone().cpu()
    domains.base_constraint_b = base_constraints[1][:1].detach().clone().cpu()
    return base_constraints, [None if v is None else sort_blocks(v.reshape(-1), chains).reshape(v.shape)
                              for v in incumbents]


def _root_certificate_dual(label, certificate, time_key, nlp_config, best_primal_x,
                           primal_solution, timeout, start):
    """Solve a root certificate against the incumbent; its dual, rounded down, when it
    closes the gap, else None (an inconclusive certificate changes nothing)."""
    primal = float(primal_solution.reshape(-1)[0])
    tol = max(float(nlp_config.get("bab_primal_dual_abs_tol", 1e-5)),
              float(nlp_config.get("bab_primal_dual_rel_tol", 1e-4)) * abs(primal))
    budget = min(float(nlp_config.get(time_key, 20.0)), max(0., timeout - (time.time() - start)))
    t0 = time.time()
    dual, rounds = certificate.solve(best_primal_x, budget, primal, tol)
    # Preserve downward rounding if the solver uses a narrower output dtype.
    dual = float(torch.nextafter(primal_solution.new_tensor(dual),
                                 primal_solution.new_tensor(-torch.inf)))
    print(f"[{label} certificate] D={dual:.12g} P={primal:.12g} "
          f"rounds={rounds} time={time.time() - t0:.3f}s")
    return dual if math.isfinite(dual) and 0 <= primal - dual <= tol else None


def _raise_root_lb_to_open_leaves(global_lb, leaf_bounds, primal_solution):
    """Root objective bound raised to the lifted-LP B&B's proven bound. The open leaves
    hold every relaxed-feasible point with objective below the incumbent and closed
    leaves lie at or above it, so min(leaf bounds, incumbent) bounds the problem. Kept
    even when the BaB handoff is skipped for size: digabel18 proved -720.6 and reported
    -781.0. Children inherit it through compare_with_old_bounds."""
    bound = min(float(torch.as_tensor(leaf_bounds).min()),
                float(primal_solution.reshape(-1)[0]))
    if not math.isfinite(bound):
        return global_lb
    global_lb = global_lb.clone()
    # one ulp down: a narrower global_lb dtype must not round the bound up
    bound = torch.nextafter(global_lb.new_tensor(bound), global_lb.new_tensor(-math.inf))
    global_lb[..., 0] = torch.maximum(global_lb[..., 0], bound)
    return global_lb


def _finish_with_lifted_dual(net, domains, best_primal_x, primal_solution,
                             primal_solution_found, lifted_dual, stats):
    """End the solve with the lifted-LP B&B's certified dual (input_split/lifted_bound.py
    run_lifted_bab), reporting it like a converged BaB."""
    dual_t = torch.tensor([lifted_dual], device=primal_solution.device,
                          dtype=primal_solution.dtype)
    sign = -1 if arguments.Config["solving"]["is_maximization"] else 1
    print(f"Primal Solution {sign * primal_solution}, Dual Solution {sign * float(lifted_dual)}")
    print(f"Solution gap: {float((primal_solution - dual_t).reshape(-1)[0])}")
    print(f"Found optimal object: {sign * primal_solution}")
    _report.record_pd(sign * primal_solution, sign * float(lifted_dual))
    export_best_primal(net, domains, best_primal_x, primal_solution, primal_solution_found)
    return dual_t.reshape(-1)[0], stats.visited, "safe"


def export_best_primal(net, domains, root_best_x, primal_solution, primal_solution_found):
    """Hand the incumbent's coordinates to the caller and to ABCROWN_BEST_PRIMAL_PATH.

    The runners replay this point against the exact constraints at zero tolerance
    before calling a solve certified; without the file every closed gap was reported
    as `solver_claim_replay_infeasible`. The in-BaB ALM/IPOPT incumbent lives on
    `domains.history.alm_best_primal_x`; the root NLP point is the fallback.
    """
    x = None
    if primal_solution_found:
        history_x = getattr(getattr(domains, "history", None), "alm_best_primal_x", None)
        x = history_x if history_x is not None else root_best_x
    if x is not None:
        x = torch.as_tensor(x).detach().reshape(-1).cpu().clone()
        obj = torch.as_tensor(primal_solution).detach().reshape(-1)[0].item()
        print(f"[best-primal] objective={obj!r} x={x.tolist()!r}")
        _report.record_best_primal(obj, x)
        path = os.environ.get("ABCROWN_BEST_PRIMAL_PATH")
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            torch.save(x, path)
    net._best_primal_x = x
    return x


def _print_final_results(stats: Stats, len_domains: int, global_lb: torch.Tensor):
    stats.timer.add('summary')
    context_manager = prefix_print_output if arguments.Config["solving"]["solving_mode"] else no_op_context
    if arguments.Config["solving"]["solving_mode"]:
        print("================== Dual Veri Report ================== ")
    with context_manager():
        stats.timer.print()
        print("Length of domains:", len_domains)
        print(f"{stats.visited} domains visited")
        if arguments.Config["solving"]["solving_mode"]:
            print(f"Dual Solution This Iter: {global_lb[0, 0]}")

def _apply_engine_box(x_L, x_U, history):
    """Consume the optional certified FBBT/DBBT box before filtering or splitting."""
    box = getattr(history, 'engine_box', None)
    if box is None: return
    from nlp_solver.rigorous_dual import round_down
    lo = torch.maximum(x_L, round_down(box[0].reshape_as(x_L), x_L.dtype).to(x_L.device))
    hi = torch.minimum(x_U, -round_down(-box[1].reshape_as(x_U), x_U.dtype).to(x_U.device))
    valid = torch.isfinite(lo).flatten(1).all(1) & torch.isfinite(hi).flatten(1).all(1)
    valid &= (lo <= hi).flatten(1).all(1)
    mask = valid.reshape(-1, *([1]*(x_L.ndim-1)))
    # Empty boxes alone are hints, never new pruning certificates.
    x_L.copy_(torch.where(mask, lo, x_L))
    x_U.copy_(torch.where(mask, hi, x_U))
    history.engine_box = None


def _sigma_batch_hint(addon, auto_batch_size):
    if addon is None or not addon.sigma_main: return None
    if addon.select_main() == 'sigma':
        if not getattr(auto_batch_size, '_sigma_active', False):
            auto_batch_size._native_batch_size = auto_batch_size.batch_size
        auto_batch_size._sigma_active = True
        hint = addon.batch_hint
    else:
        # A switch back to D cannot carry over the much wider sparse batch.
        if getattr(auto_batch_size, '_sigma_active', False):
            auto_batch_size.batch_size = auto_batch_size._native_batch_size
        auto_batch_size._sigma_active = False
        memory = getattr(addon.engine, 'memory', None)
        room = memory.available() if memory is not None else math.inf
        if not addon.native_peak_bytes or not math.isfinite(room): return None
        width = max(1, int(room//addon.native_peak_bytes))
        hint = 1 << (width.bit_length()-1)
        if not auto_batch_size.enable: hint = min(hint, auto_batch_size.batch_size)
    auto_batch_size.batch_size = hint
    return auto_batch_size.batch_size


def _drop_sigma_planes(domains):
    # Optional native planes live for one batch when roles can change. The
    # scalar floors remain valid; the input rows already have shared storage.
    domains.constraint_A = domains.constraint_b = None


@retained
def get_bounds(net: LiRPANet, dm_lb: Tensor, x_L: Tensor, x_U: Tensor, history: SplitHistory,
                cs: Tensor, thresholds: Tensor, constraints: tuple,
                base_input_linear_constraints: Optional[tuple],
                bounding_method: str,
                branching_method: str, stop_func: Callable, stats: Stats, _sliced=False
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    clip_input_args = arguments.Config["bab"]["clip_n_verify"]["clip_input_domain"]
    input_split_args = arguments.Config["bab"]["branching"]["input_split"]
    enable_clip_domains = clip_input_args["enabled"]
    # need lA for branching/clipping/dual-ascent; need lbias for clip domains and dual ascent
    nlp_da_enabled = arguments.Config["solver"]["nlp"].get("dual_ascent_enabled", False)
    return_A = branching_method != 'naive' or enable_clip_domains or nlp_da_enabled
    return_b = enable_clip_domains or nlp_da_enabled
    stats.timer.start("bounding")

    stats.visited += len(x_L)
    addon = getattr(net, 'm1_addon', None)

    from input_split.tree_slice import checkpoint
    checkpoint()
    ret = yield from invoke(net.get_lower_bound_naive, sliced=_sliced,
        **({"_sliced": True} if _sliced else {}),
        dm_lb=dm_lb if input_split_args["compare_with_old_bounds"] or (addon is not None and addon.sigma_main) else None,
        dm_l=x_L, dm_u=x_U, history=history,
        bounding_method=bounding_method,
        C=cs, stop_criterion=stop_func, thresholds=thresholds,
        return_A=return_A, return_b=return_b,
        constraints=constraints,
        base_input_linear_constraints=base_input_linear_constraints,
        stats=stats)

    checkpoint()
    new_dm_lb, lA, lbias, x_star = ret
    addon = getattr(net, 'm1_addon', None)
    if addon is not None and addon.sigma_main:
        _apply_engine_box(x_L, x_U, history)
    new_dm_lb = new_dm_lb.to(device=thresholds.device)  # ensures it is on the same device as it may be different
    if not torch.isfinite(new_dm_lb).all():
        warnings.warn(
            "Non-finite lower bounds encountered in input BaB; replacing NaN with -inf and keeping infinities."
        )
        new_dm_lb = torch.nan_to_num(
            new_dm_lb, nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf
        )
    stats.timer.add("bounding")
    # Visited domain is now incremented based on the input of get_lower_bound_naive.

    return new_dm_lb, lA, lbias, x_star


def _reset_instance_state():
    """Process-wide state that must not leak from a previous instance in this process."""
    arguments.Globals["discarded_threshold_min"] = None   # caps the reported dual


def _process_start_time():
    """time.time() at which this process started (Linux /proc), so the ledger's clock includes interpreter startup,
    imports and model loading; the current time where /proc is unavailable."""
    try:
        with open("/proc/self/stat") as f:
            start_ticks = int(f.read().rsplit(")", 1)[1].split()[19])
    except OSError:
        return time.time()
    age = time.clock_gettime(time.CLOCK_BOOTTIME) - start_ticks / os.sysconf("SC_CLK_TCK")
    return time.time() - age


def _gain(old, new):
    """new - old, and 0 when they are equal (both infinite included)."""
    return 0.0 if old == new else new - old


# The BaB loop's own steps (stats.timer keys); the bounding sub-timers (init, concretize) run inside them.
_BAB_STEPS = ("sort_domains", "adv_check", "pickout", "update_rhs_with_attack", "alm_attack", "m3", "split",
              "bounding", "filtering", "clip", "decision", "strong_branch", "add_domain", "summary")


class OpLedger:
    """The run's operation ledger: one `[Op]` line per root operation and per BaB step of each iteration, printed at
    those boundaries only, from host values:

        [Op] <name> t=<s since process start> s=<seconds> work=<units> dD=<bound gain> dP=<incumbent gain>

    Gains are in the internal minimization sense (dP = inf for a first incumbent) and count from the last line: a
    root operation gets the change it committed; in an iteration the bounding step (or the M3 cover step, m3, that
    replaces it) gets the bound's gain and the primal step (alm_attack, with IPOPT-in-BaB) the incumbent's, less what
    the steps noted on the way gained. Work is one call per root operation and the domains bounded per iteration for
    the BaB steps."""

    def __init__(self):
        self.origin = _process_start_time()
        self.P, self.D = math.inf, -math.inf      # incumbent and certified bound at the last line
        self.sums = {}                            # the steps' seconds (stats.timer.time_sum) at the last iteration
        self.notes = []                           # (name, seconds, dD, dP) of the steps noted in this iteration
        self.bab_gain, self.bab_seconds = 0.0, 0.0

    def line(self, name, seconds, work=1, dD=0.0, dP=0.0):
        text = f"[Op] {name} t={time.time() - self.origin:.3f} s={seconds:.4f} work={work} dD={dD:.6g} dP={dP:.6g}"
        print(text)
        _report.record_op(text)

    def commit(self, name, seconds, P, D, work=1):
        """An operation that left incumbent P and certified bound D."""
        self.line(name, seconds, work, _gain(self.D, D), _gain(P, self.P))
        self.P, self.D = P, D

    def start_bab(self, timer):
        """The BaB loop starts: its steps count from here (root operations also run timed steps)."""
        self.sums = dict(timer.time_sum)

    def note(self, name, seconds, dD=0.0, dP=0.0):
        """A step of this iteration timed outside stats.timer, with what it gained, for the iteration's lines."""
        self.notes.append((name, seconds, dD, dP))

    def competing_progress(self, timer, P, D):
        """Completed and current main-BaB gap gain/service, excluding noted ops."""
        seconds = sum(max(0.0, timer.time_sum.get(name, 0.0) - self.sums.get(name, 0.0))
                      for name in _BAB_STEPS)
        gains = (max(0.0, _gain(self.D, D)), max(0.0, _gain(P, self.P)))
        gain = sum(v for v in gains if math.isfinite(v))
        noted = sum(n[2] + n[3] for n in self.notes if math.isfinite(n[2] + n[3]))
        return self.bab_gain + max(0.0, gain - noted), self.bab_seconds + seconds

    def iteration(self, timer, work, P, D):
        """The lines of one BaB iteration that left incumbent P and certified bound D."""
        self.bab_gain, self.bab_seconds = self.competing_progress(timer, P, D)
        noted_dD = sum(n[2] for n in self.notes)
        noted_dP = sum(n[3] for n in self.notes)
        dD, dP = _gain(self.D, D), _gain(P, self.P)
        dD = dD - noted_dD if math.isfinite(noted_dD) else 0.0
        dP = dP - noted_dP if math.isfinite(noted_dP) else 0.0
        for name in _BAB_STEPS:
            seconds = timer.time_sum.get(name, 0.0) - self.sums.get(name, 0.0)
            if seconds > 0:
                self.line(name, seconds, work, dD if name in ("bounding", "m3") else 0.0,
                          dP if name == "alm_attack" else 0.0)
        for name, seconds, n_dD, n_dP in self.notes:
            self.line(name, seconds, 1, n_dD, n_dP)
        self.sums, self.notes = dict(timer.time_sum), []
        self.P, self.D = P, D


def clamp_to_solve_deadline(root, solve_deadline):
    """GEAR facade, timing="solve" (G5 DESIGN section 1 Timing): the one deadline (wall clock) clamps the root's
    deadline fields, which every root operation and BaB timer reads."""
    root.solve_deadline = solve_deadline
    root.deadline = min(root.deadline, solve_deadline)
    root.timeout = max(0.0, root.deadline - root.start)
    root.stats.timer.deadline = root.deadline
    if root.controller:
        root.controller_monotonic_deadline = time.monotonic() + max(0.0, root.deadline - time.time())
        root.ledger.deadline = root.controller_monotonic_deadline


class RootState:
    """The root of one input BaB run: its options, the root problem, and what the root operations commit.

    The incumbent is `primal_solution` (the internal minimization sense) at `best_primal_x`; only points verified on
    the original model enter it. The certified root bound is the objective column of `global_lb`. The operations of
    _ROOT_OPERATIONS read and replace these fields and prepare the boxes, rows and search state the BaB starts from.
    The run has one deadline, start + timeout."""
    solve_deadline = None   # the GEAR facade's one deadline (timing="solve"), set by clamp_to_solve_deadline

    def __init__(self, net, x, c, rhs, or_spec_size, reference_dict, timeout, max_iterations):
        self.net, self.x, self.c, self.rhs, self.or_spec_size = net, x, c, rhs, or_spec_size
        self.reference_dict = reference_dict
        self.stats = Stats()
        self.start = time.time()

        bab_args = arguments.Config["bab"]
        clip_n_verify_args = bab_args["clip_n_verify"]
        self.clip_input_args = clip_n_verify_args["clip_input_domain"]
        branching_args = bab_args["branching"]
        self.input_split_args = branching_args["input_split"]

        self.batch = arguments.Config["solver"]["batch_size"]
        self.bounding_method = arguments.Config["solver"]["bound_prop_method"]
        self.init_bounding_method = arguments.Config["solver"]["init_bound_prop_method"]

        self.timeout = timeout or bab_args["timeout"]
        self.stats.timer.deadline = self.start + self.timeout
        self.max_iterations = max_iterations or bab_args["max_iterations"]
        self.sort_domain_iter = bab_args["sort_domain_interval"]

        self.branching_method = branching_args["method"]

        self.adv_check = self.input_split_args["adv_check"]
        self.split_partitions = self.input_split_args["split_partitions"]
        self.presplit_domains = self.input_split_args["presplit_domains"]
        self.use_reordered_bab = self.input_split_args["reorder_bab"]
        self.split_hint = self.input_split_args["split_hint"]
        self.sort_index = self.input_split_args["sort_index"]
        self.sort_heuristic = self.input_split_args["sort_heuristic"]

        self.use_clip_domains = self.clip_input_args["enabled"]
        self.clip_calculate_volume_metrics = self.clip_input_args["clip_calculate_volume_metrics"]
        self.constrained_concretize = self.use_clip_domains and self.clip_input_args["clip_type"] == "complete"
        self.clip_neuron_selection_value = self.clip_input_args["clip_neuron_selection_value"]
        self.clip_neuron_selection_type = self.clip_input_args["clip_neuron_selection_type"]

        self.pgd_order = arguments.Config["attack"]["pgd_order"]
        self.solving_mode = arguments.Config["solving"]["solving_mode"]
        self.enable_attack_counter = arguments.Config["solving"]["enable_attack_counter"]

        self.enable_check_adv = arguments.Config["attack"]["input_split_check_adv"]["enabled"]
        self.enable_check_adv = (
            self.pgd_order != "skip" if self.enable_check_adv == "auto"
            else self.enable_check_adv == "true"
        )

        # In regular input BaB, the total number of subquestions equals batch_size * 2 after splitting.
        # In reordered BaB, we also ensure the total subquestions equal batch_size * 2,
        # but the input domains are split after the bounding step.
        self.batch = 2 * self.batch if self.use_reordered_bab else self.batch

        if self.init_bounding_method == "same":
            self.init_bounding_method = self.bounding_method

        self.use_alpha = self.init_bounding_method.lower() == "alpha-crown" or self.bounding_method == "alpha-crown"
        self.use_beta = self.input_split_args["beta"] or self.init_bounding_method.lower() == "beta-crown" or self.bounding_method == "beta-crown"
        if self.use_beta:
            self.net.net.get_output_nodes()

        self.stop_criterion = stop_criterion_batch_any

        # get input domain lower/upper limits
        self.dm_l = self.x.ptb.x_L
        self.dm_u = self.x.ptb.x_U

        def _broadcast_dm(dm):
            if dm.shape[0] == 1 and self.c.shape[0] > 1:
                dm = dm.expand(self.c.shape[0], *[-1] * (dm.ndim - 1))
            assert dm.shape[0] == self.c.shape[0]
            return dm

        # c is expanded in build(). dm_l(u) must be expanded as well
        self.dm_l = _broadcast_dm(self.dm_l)
        self.dm_u = _broadcast_dm(self.dm_u)
        if (self.dm_u - self.dm_l > 0).int().sum() == 1:
            self.branching_method = "naive"
        self.deadline = self.start + self.timeout
        self.nlp_config = arguments.Config["solver"]["nlp"]
        # One deadline, the gap checked after each operation (solver.nlp.unified_root; arch2 S1).
        self.unified = bool(self.solving_mode and self.nlp_config.get("unified_root", False))

        self.rowpool = None                   # the problem's rows (solving mode)
        self.structure = None                 # the root structure report (struct_forms or symmetry)
        self.hybrid = None                    # RootHybrid's bound
        self.primal_solution = None           # the incumbent: objective (1,) and point
        self.primal_solution_found = False
        self.best_primal_x = None
        self.warmstart_x = None
        self.report_only_primal = None
        self.dual_solution = None
        self.root_lbias = None                # the root planes' offsets on the box global_lb was computed on
        self.lifted_setup = None              # the root's lifted-LP setup (lifted_bound.RootLifted)
        self.lifted_verify_loop = None
        self.lifted_local_loop = None
        self.lifted_handoff_leaves = 0
        self.lifted_bound_version = 0         # the lifted setup's version the unified root last bounded
        self.controller = bool(self.solving_mode and self.nlp_config.get("controller", False))
        self.unified = self.unified or self.controller
        import queue
        self.controller_inbox = queue.Queue()
        self.certified_dual = -math.inf
        from input_split.root_controller import ReceiptLedger
        self.ledger = ReceiptLedger(self.nlp_config) if self.controller else OpLedger()
        if self.controller:
            # The API's remaining timeout excludes interpreter/model startup.
            # S4's single deadline includes that work, using the user limit.
            user_limit = bab_args.get('override_timeout')
            if user_limit is None:
                user_limit = float(bab_args['timeout']) * float(bab_args.get('timeout_scale', 1.0))
            self.deadline = min(self.deadline, self.ledger.origin + float(user_limit))
            self.timeout = max(0.0, self.deadline - self.start)
            self.stats.timer.deadline = self.deadline
            self.controller_monotonic_deadline = time.monotonic() + max(0.0, self.deadline - time.time())
            self.ledger.deadline = self.controller_monotonic_deadline
        self.primal_lane = None
        if bab_args.get('solve_deadline') is not None:
            clamp_to_solve_deadline(self, float(bab_args['solve_deadline']))

    def values(self):
        """(P, D): the incumbent and the certified root bound as host floats, internal minimization sense."""
        P = math.inf if self.primal_solution is None else float(self.primal_solution.reshape(-1)[0])
        global_lb = getattr(self, "global_lb", None)
        D = -math.inf if global_lb is None else float(global_lb.reshape(global_lb.shape[0], -1)[:, 0].min())
        return P, max(D, getattr(self, "certified_dual", -math.inf))

    def time_left(self):
        """Seconds to the run's deadline, which every operation of the unified root ends by; inf without it (the
        operations keep their own budgets)."""
        if getattr(self, 'controller_monotonic_deadline', None) is not None:
            return self.controller_monotonic_deadline - time.monotonic()
        return self.deadline - time.time() if self.unified else math.inf

    def closes(self, dual):
        """Whether the GEAR gap rule closes the incumbent (as reported) against a certified bound `dual`."""
        if getattr(self, 'controller', False) and self.time_left() <= 0:
            return False
        if not (self.primal_solution_found or self.report_only_primal is not None):
            return False
        reported = self.primal_solution if self.report_only_primal is None else self.report_only_primal
        return _is_primal_dual_converged(
            reported, dual,
            abs_tol=float(self.nlp_config.get("bab_primal_dual_abs_tol", 1e-5)),
            rel_tol=float(self.nlp_config.get("bab_primal_dual_rel_tol", 1e-4)))

    def finish(self, after, dual):
        """End the solve at the root with the certified bound `dual` that closes the gap."""
        print(f"[UnifiedRoot] closed after {after} at {time.time() - self.start:.2f}s: "
              f"P={float(self.primal_solution.reshape(-1)[0]):.12g} D={dual:.12g}")
        return _finish_with_lifted_dual(self.net, getattr(self, "domains", None), self.best_primal_x,
                                        self.primal_solution, self.primal_solution_found, dual, self.stats)


class RootOp(NamedTuple):
    """One root operation: `run(root, deadline)` commits to the RootState (an incumbent, a certified bound, rows,
    boxes, search state) and returns the solve's outcome when it ends the solve, else None. It runs when
    `eligible(root)`; a `timed` one spends a time budget, which the unified root admits only before the deadline."""
    name: str
    run: Callable[[RootState, float], Optional[tuple]]
    eligible: Callable[[RootState], bool] = lambda root: True
    timed: bool = False


def _root_reference_bounds(root, deadline):
    """The root bound and alphas of incomplete verification, and the problem's input rows."""
    # Since we always enable incomplete verification by default,
    # it always has reference_dict. When incomplete verification is disabled,
    # reference_dict may still carry input-constraint entries without bounds,
    # so key off 'global_lb' to decide whether usable reference bounds exist.
    if root.reference_dict and 'global_lb' in root.reference_dict:
        root.ret = root.reference_dict
        root.global_lb = root.ret['global_lb']
        root.base_input_linear_constraints = root.ret.get('base_input_linear_constraints')
        root.base_input_constraints = root.ret.get('base_input_constraints')
        # format alphas. extract only alpha values from the complete alpha dict
        root.alphas = {}
        if root.use_alpha:
            root.alphas = {k: {kk: vv for kk, vv in v['alpha'].items()} for k, v in root.ret['alphas'].items()}
            print("Using reference alphas to initialize input BaB.")
        if root.use_beta:
            root.betas = root.ret.get('betas')
            if root.betas is None:
                root.betas = [{} for _ in range(root.global_lb.shape[0])]

    else:
        # Backup path: input-split BaB normally receives bounds produced by
        # incomplete verification (upstream asserts reference_dict). Keep the
        # standalone build for callers that arrive without reference bounds.
        root.global_lb, root.ret = root.net.build(
            root.x, root.c, root.rhs, root.stop_criterion,
            bounding_method=root.init_bounding_method, return_A=False)
        root.base_input_linear_constraints = root.ret.get('base_input_linear_constraints')
        root.base_input_constraints = root.ret.get('base_input_constraints')
        root.alphas = root.ret['alphas'] if root.use_alpha else {}
        root.betas = root.ret.get('betas')
        if root.betas is None:
            root.betas = [{} for _ in range(root.global_lb.shape[0])]

    if root.base_input_constraints is None and getattr(root.x.ptb, 'base_input_constraints', None) is not None:
        root.base_input_constraints = root.x.ptb.base_input_constraints
    if root.base_input_linear_constraints is None and getattr(root.x.ptb, 'base_input_linear_constraints', None) is not None:
        root.base_input_linear_constraints = root.x.ptb.base_input_linear_constraints
    if root.base_input_constraints is None and getattr(root.x.ptb, 'constraints', None) is not None:
        root.base_input_constraints = root.x.ptb.constraints
        root.base_input_linear_constraints = standard_form_to_input_constraints(root.base_input_constraints)
    # base_input_constraints (standard form) feed every bound and carry the relaxed rows;
    # base_input_linear_constraints (A x <= b) feed the primal and carry the raw ones.
    root.input_rows_raw = root.base_input_linear_constraints

    root.stats.visited += len(root.global_lb)


def _root_rows(root, deadline):
    """The problem's rows (RowPool), the square-row cuts and the objective alias floor."""
    from auto_LiRPA.presolve import objective_alias_floor, square_row_cuts
    # The one place the verify predicate's tolerance is applied to the problem's rows (the pool also holds the
    # raw base rows A x <= b); every bound below reads the pool's relaxed thresholds.
    root.rowpool = RowPool(root.rhs, None if root.input_rows_raw is None else (root.input_rows_raw[0][0], root.input_rows_raw[1][0]))
    tol = root.rowpool.tolerance
    cuts = square_row_cuts(root.net.net, root.c, root.rhs, root.x, tol)
    if cuts is not None:
        cut_A, cut_b = cuts
        # These rows already include the original square-row tolerance.
        # Clipping/concretization use base rows directly; keep the full
        # proved RHS. No consumer relaxes the base rows again.
        root.base_input_constraints = merge_standard_constraints(
            root.base_input_constraints, (cut_A, -cut_b))
        root.base_input_linear_constraints = _primal_input_rows(root.base_input_constraints, root.input_rows_raw)
    floor = objective_alias_floor(root.net.net, root.c, root.rhs, tol)
    root.net.net.objective_alias_floor_value = None
    if floor is not None and torch.equal(root.c, root.c[:1].expand_as(root.c)) and torch.equal(
            root.rhs[:, 1:], root.rhs[:1, 1:].expand_as(root.rhs[:, 1:])):
        # Same rows and row thresholds in every domain: one global floor, applied by
        # the bounding step without re-proving it per domain.
        value = float(floor.min())
        if math.isfinite(value):
            root.net.net.objective_alias_floor_value = value
    if floor is not None:
        improved = floor > root.global_lb[:, 0]
        root.global_lb = root.global_lb.clone()
        root.global_lb[:, 0] = torch.maximum(root.global_lb[:, 0], floor)
        # The alias floor is a valid constant plane, not a shift of the
        # previous CROWN plane. This matters when clipping reconstructs b.
        root.ret = dict(root.ret)
        root.ret['global_lb'] = root.global_lb
        root.ret['lA'] = {name: None if A is None else A.clone()
                          for name, A in (root.ret.get('lA') or {}).items()}
        for A in root.ret['lA'].values():
            if A is not None:
                A[improved, 0] = 0
    counts = getattr(root.net.net, 'presolve_rewrite_counts', {})
    if counts:
        print(f"[Presolve] rows/identities={counts}")


STRUCT_CERT_KINDS = ("diameter", "assignment", "eigenblock")


def _struct_cert_kinds(root):
    """The root structure certificates whose flag solver.nlp.struct_cert_<kind> is on (solving mode)."""
    return [k for k in STRUCT_CERT_KINDS if root.solving_mode and root.nlp_config.get(f"struct_cert_{k}", False)]


def _root_struct_forms(root, deadline):
    """The root structure report, built with the square-recording algebra whose exact forms the structure
    certificates read; the symmetry step reuses it."""
    from nlp_solver.structure.certificates import SquareForms
    root.structure = _root_structure(root.net, root.x, root.c, root.rhs, root.nlp_config, root.rowpool,
                                     algebra=SquareForms)


def _struct_cert_due(root):
    """The structure certificates run when one of their flags is on and the root has exact forms to detect them in."""
    structure = root.structure
    return (bool(_struct_cert_kinds(root)) and structure is not None and not structure.skipped
            and bool(structure.forms))


def _raise_root_floor(root, floor):
    """Commit a global objective floor proved at the root: raise the root bound to it, with the constant plane on the
    raised objective rows (a raised bias under the old CROWN slope would make clipping unsound), and give it to every
    later bound through the root objective floor that bounding applies per domain."""
    global_lb = root.global_lb
    floor = float(torch.nextafter(global_lb.new_tensor(floor), global_lb.new_tensor(-math.inf)))
    improved = floor > global_lb[:, 0]
    root.global_lb = global_lb.clone()
    root.global_lb[:, 0] = torch.maximum(global_lb[:, 0], global_lb.new_tensor(floor))
    root.ret = dict(root.ret)
    root.ret['global_lb'] = root.global_lb
    root.ret['lA'] = {name: None if A is None else A.clone() for name, A in (root.ret.get('lA') or {}).items()}
    for A in root.ret['lA'].values():
        if A is not None:
            A[improved, 0] = 0
    previous = root.net.net.objective_alias_floor_value
    root.net.net.objective_alias_floor_value = floor if previous is None else max(previous, floor)


def _root_struct_cert(root, deadline):
    """The root structure certificates (nlp_solver/structure/certificates.py): the best certified floor on the
    objective over the root hull raises the root bound and every later bound, and the best candidate verified on the
    original model at constr_viol_tol becomes the incumbent. The unified root's gap rule closes the solve on them;
    without it, this operation does."""
    from nlp_solver.structure.certificates import root_certificates
    from input_split.lifted_bound import make_verifier
    certificates, _ = root_certificates(root.structure, _struct_cert_kinds(root), root.x.ptb.x_L, root.x.ptb.x_U,
                                        deadline)
    floor = max((cert.lower_bound for cert in certificates), default=None)
    if floor is not None:
        _raise_root_floor(root, floor)
    verify = make_verifier(root.net.model_ori, root.x.ptb.x_L, root.x.ptb.x_U, root.c, root.rhs, root.or_spec_size,
                           root.input_rows_raw, root.nlp_config)
    for cert in certificates:
        if cert.candidate is None or verify is None:
            continue
        ok, obj, point = verify(cert.candidate)
        print(f"[StructCert] {cert.kind} candidate verified={ok} objective={obj:.12g}")
        if ok and (not root.primal_solution_found or obj < float(root.primal_solution.reshape(-1)[0])):
            root.primal_solution = torch.tensor([obj], device=root.global_lb.device)
            root.primal_solution_found = True
            root.best_primal_x = torch.as_tensor(point, dtype=root.x.ptb.x_L.dtype, device=root.x.ptb.x_L.device
                                                 ).reshape(root.x.ptb.x_L[:1].shape)
    dual = root.values()[1]
    if not root.unified and root.closes(dual):
        return _finish_with_lifted_dual(root.net, None, root.best_primal_x, root.primal_solution, True, dual,
                                        root.stats)
    return None


def _root_hybrid(root, deadline):
    """RootHybrid: a certified bound on the objective from its separable terms."""
    from input_split.hybrid_bound import root_hybrid_objective_lb
    hybrid_info = {}
    root.hybrid = root_hybrid_objective_lb(
        root.net.net, root.x.ptb.x_L, root.x.ptb.x_U,
        max_terms=root.nlp_config.get("root_hybrid_max_terms", 64),
        diagnostics=hybrid_info)
    if root.hybrid is not None:
        root_lb = root.global_lb[..., 0].clone()
        improved = root.hybrid > root_lb
        root.global_lb = root.global_lb.clone()
        root.global_lb[..., 0] = torch.maximum(root_lb, root_lb.new_tensor(root.hybrid))
        # A stronger scalar bound does not shift the existing CROWN plane.
        # Use the valid constant plane on improved objective rows so bias
        # reconstruction and clipping remain sound; constraint rows stay intact.
        root.ret = dict(root.ret)
        root.ret['lA'] = dict(root.ret['lA'])
        for name, root_A in root.ret['lA'].items():
            if root_A is not None:
                root_A = root_A.clone()
                root_A[improved, 0] = 0
                root.ret['lA'][name] = root_A
        print(f"[RootHybrid] terms={hybrid_info['terms']} lb={root.hybrid:.12g} "
              f"(root lb was {root_lb.min().item():.12g})")


def _root_initial_verify(root, deadline):
    """Whether the root bound already proves the constraints infeasible (outside solving mode: verifies the
    specification)."""
    # check if the initial global lower bound is verified
    if root.solving_mode:
        # The first dimension of global_lb corresponds to the objective value
        # so we only check the remaining dimensions as constraints.
        # Add the primal feasibility tolerance: equality constraints are encoded
        # as paired inequalities whose true bound sits exactly on the rhs, so a
        # zero-tolerance check turns FP rounding into a false infeasibility proof.
        root.initial_verified, root.remaining_index = initial_verify_criterion(
            root.global_lb[:, 1:], root.rowpool.thresholds(root.rhs)[:, 1:])
        if root.initial_verified:
            print("Constraints infeasible!")
    else:
        root.initial_verified, root.remaining_index = initial_verify_criterion(root.global_lb, root.rhs)
        if root.initial_verified:
            return root.global_lb.max(), root.stats.visited, "safe"


def _prepare_root_nlp(root):
    """Prepare the shared root dual warm start and LP seed once, before launching either primal engine."""
    # Optional: run dual ascent at root to derive a dual warm-start (mu)
    # for ALM. Requires lA in `ret` (CROWN-based init_bounding_method).
    from input_split.nlp_in_bab import compute_root_dual_warmstart, compute_root_lp_point
    root.root_dual_mu = compute_root_dual_warmstart(
        root.net, root.ret, root.dm_l, root.dm_u, root.c, root.rhs, root.global_lb,
        base_input_linear_constraints=root.base_input_linear_constraints,
    )
    root.root_lp_x = compute_root_lp_point(
        root.net, root.ret, root.dm_l, root.dm_u, root.rhs, root.global_lb,
        base_input_linear_constraints=root.base_input_linear_constraints,
        rowpool=root.rowpool,
    )


def _root_lp_alternation(root, deadline):
    """Start the native primal lane beside the root NLP; it publishes only verified incumbents."""
    from nlp_solver import lp_alternation
    # A reused pre-incomplete candidate has no root NLP window to run beside.
    if (root.nlp_config.get("enabled", False) and not root.initial_verified
            and pre_incomplete_to_root_result(root.ret, root.dm_l, root.dm_u) is None):
        _prepare_root_nlp(root)
        root.seeds = list(getattr(root, "seeds", None) or []) + [root.root_lp_x]
        root.primal_lane = lp_alternation.start(root, deadline)


def _root_nlp(root, deadline):
    """The root NLP: IPOPT and the GPU ALM, or the verified pre-incomplete candidate."""
    # --- NLP/IPOPT Falsification at Root Node ---
    root.root_nlp = pre_incomplete_to_root_result(root.ret, root.dm_l, root.dm_u)
    if root.root_nlp is not None:
        print(
            "[Pre-Incomplete Primal] Reusing verified root candidate; "
            "skipping duplicate post-incomplete NLP solve."
        )
    elif root.solve_deadline is not None and root.time_left() <= 0:
        # GEAR timing="solve" (G5 A293): the one deadline also binds the root NLP. Past it the NLP is not admitted
        # (the result of a disabled NLP): its budget cap applies only to a positive time left, so it would run on
        # its configured timeouts (ALM 60 s, IPOPT 30 s) and overrun the deadline by them.
        print("[Root] NLP not admitted: the solve deadline has passed")
        root.root_nlp = RootNLPResult(dm_l=root.dm_l, dm_u=root.dm_u)
        if root.primal_lane is not None:
            root.primal_lane.join()
    else:
        if root.primal_lane is None:
            _prepare_root_nlp(root)
        if root.unified:
            root_wall_budget = root.time_left()
        else:
            # Cap root solver wall time by a fraction of the remaining global budget
            # so the root stage cannot starve the BaB loop on expensive graphs.
            root_budget_frac = float(root.nlp_config.get("root_budget_frac", 0.5))
            root_wall_budget = _remaining_root_wall_budget(
                root.timeout, root.start, root_budget_frac
            )
        try:
            runner = run_root_nlp_and_ipopt
            if getattr(root, "controller", False):
                from input_split.root_controller import run_nlp
                runner = lambda *args, **kwargs: run_nlp(root, run_root_nlp_and_ipopt, *args, **kwargs)
            root.root_nlp = runner(
                root.net, root.dm_l, root.dm_u, root.c, root.rhs, root.or_spec_size,
                reference_dict=root.ret, nlp_config=root.nlp_config,
                initial_verified=root.initial_verified,
                base_input_linear_constraints=root.base_input_linear_constraints,
                dual_mu_init=root.root_dual_mu,
                root_wall_budget=root_wall_budget,
                local_extra_x0=root.root_lp_x,
            )
        finally:
            if root.primal_lane is not None:
                root.primal_lane.join()
    root.dm_l, root.dm_u = root.root_nlp.dm_l, root.root_nlp.dm_u
    if root.root_nlp.ipopt_found:
        return root.global_lb.max(), root.stats.visited, "unsafe-ipopt"

    root.nlp_obj = torch.tensor([torch.inf], device=root.rhs.device)
    if root.root_nlp.nlp_obj is not None:
        root.nlp_obj = torch.tensor([float(root.root_nlp.nlp_obj)], device=root.rhs.device)
    root.ipopt_obj = torch.tensor([torch.inf], device=root.rhs.device)
    root.ipopt_found = root.root_nlp.ipopt_found
    if root.root_nlp.ipopt_obj is not None:
        root.ipopt_obj = torch.tensor([float(root.root_nlp.ipopt_obj)], device=root.rhs.device)

    # Root diagnostics and BaB budget estimation
    run_root_diagnostics_and_budget(
        root.net, root.dm_l, root.dm_u, root.c, root.rhs, root.or_spec_size, root.nlp_config,
        nlp_best_x=root.root_nlp.nlp_best_x,
        root_peak_mem_bytes=root.root_nlp.root_peak_mem_bytes,
        root_total_mem_bytes=root.root_nlp.root_total_mem_bytes,
        base_input_linear_constraints=root.base_input_linear_constraints,
    )

    # Release the allocator cache built up by the root NLP stage (its restart
    # batches can reserve >10GB). AutoBatchSize gates enlargement on
    # torch.cuda.memory_reserved, so a stale cache permanently blocks the BaB
    # pickout batch from growing.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _root_search_state(root, deadline):
    """The search state: clipping options, storage depth, the root planes and the split history."""
    root.result = "unknown"

    # Set the output activations for heuristic driven constrained concretize.
    # Please check auto_LiRPA/concretize_func.py
    for node in root.net.net.get_splittable_activations():
        for preact_node in node.inputs:
            if not preact_node.perturbed:
                continue
            if preact_node.output_activations is not None:
                preact_node.output_activations.append(node)
            else:
                preact_node.output_activations = [node]

    # Set the neuron_selection_value for complete clipping
    root.net.net.clip_neuron_selection_value = root.clip_neuron_selection_value
    root.net.net.clip_neuron_selection_type = root.clip_neuron_selection_type
    if root.clip_neuron_selection_type == "ratio":
        assert root.clip_neuron_selection_value <= 1.0, "Neuron selection ratio should be smaller than 1.0! "
    else:
        assert isinstance(root.clip_neuron_selection_value, int), "Neuron selection number should be an integer! "

    # compute storage depth
    min_batch_size = (
            arguments.Config["solver"]["min_batch_size_ratio"]
            * arguments.Config["solver"]["batch_size"]
    )
    max_depth = max(int(math.log(max(min_batch_size, 1)) // math.log(root.split_partitions)), 1)
    root.stats.storage_depth = min(max_depth, root.dm_l.shape[-1])

    # get lA if needed
    root.lA = root.ret['lA'].get(root.net.net.input_name[0], None) # IF bound_prop_method is IBP, lA is None
    if root.lA is None:
        fallback = _branching_method_without_lA(root.branching_method)
        if fallback != root.branching_method:
            print(f"[branching] no lA from the root bound; {root.branching_method} -> {fallback}")
            root.branching_method = fallback
        if root.use_clip_domains:
            # IBP and other linear-coefficient-free bounding methods provide no
            # lA; clipping is a tightening step, so disabling it is sound.
            # Also flip the config entry: batch_verification_input_split and the
            # constrained-concretize path re-read it on every call.
            print("[clip] lA unavailable under the current bound_prop_method; "
                  "input-domain clipping disabled for this run.")
            root.use_clip_domains = False
            root.constrained_concretize = False
            root.clip_input_args["enabled"] = False

    import auto_LiRPA.concretize_func as _concretize_func
    _max_rows = int(arguments.Config["solver"]["nlp"].get("concretize_max_constraint_rows", 0))
    _concretize_func.MAX_CONSTRAINT_ROWS = _max_rows if _max_rows > 0 else None
    _concretize_func.FUSED_SWEEP = bool(
        arguments.Config["solver"]["nlp"].get("concretize_fused_kernel", False))

    root.history = SplitHistory(root.alphas, initial_beta=root.betas if root.use_beta else None, final_name=root.net.final_name)
    root.history.set_usage_flags(root.use_alpha, root.use_beta)
    # Scale for the SB exhaustiveness guard (branching_heuristics._exhaustive_guard).
    root.history.root_width = (root.dm_u[:1] - root.dm_l[:1]).detach().flatten()
    root.history.input_rows_raw = root.input_rows_raw      # read by batch_verification_input_split
    root.history.rowpool = root.rowpool                    # None outside solving mode; read by every bound


def _root_incumbent(root, deadline):
    """The incumbent from the root NLP (or a fixed primal), unless one an earlier root operation verified is lower."""
    # initialize the input domain list
    root.same_or_spec_size = (root.or_spec_size == root.or_spec_size[0]).all()
    if getattr(root, "controller", False):
        from input_split.root_controller import offer, poll
        for point in (root.root_nlp.nlp_best_x, root.root_nlp.ipopt_x):
            if point is not None:
                offer(root, 'incumbent', point, 0.0)
        poll(root)
        if root.primal_solution is None:
            root.primal_solution = root.rhs.new_full((1,), math.inf)
        root.warmstart_x = root.best_primal_x
        return

    if root.solving_mode:
        earlier = (root.primal_solution, root.best_primal_x) if root.primal_solution_found else None
        root.primal_solution = torch.tensor([torch.inf], device=root.global_lb.device)
        if root.nlp_obj < root.primal_solution:
            root.primal_solution = torch.tensor([root.nlp_obj], device=root.global_lb.device)
            root.primal_solution_found = True
        if root.ipopt_obj < root.primal_solution:
            root.primal_solution = torch.tensor([root.ipopt_obj], device=root.global_lb.device)
            root.primal_solution_found = True
        fixed_primal = load_fixed_primal(root.rhs)
        if fixed_primal is not None:
            if arguments.Config["solving"]["fixed_primal_report_only"]:
                # Reporting bound only. It must NOT reach the objective RHS: pruning at a
                # value the model cannot actually attain discards the region holding the
                # true optimum, after which the empty-domain branch reports the injected
                # value as a certified dual.
                root.report_only_primal = fixed_primal.reshape(1)
                print("[fixed-primal] report-only incumbent "
                      f"{float(root.report_only_primal[0].item()):.12g}; pruning unaffected.")
            else:
                root.primal_solution = fixed_primal.reshape(1)
                root.primal_solution_found = True

        # Initialize best_primal_x from the best source
        if root.primal_solution_found:
             if root.nlp_obj is not None and not torch.isinf(torch.as_tensor(root.nlp_obj)) and (root.ipopt_obj is None or torch.isinf(root.ipopt_obj) or root.nlp_obj <= root.ipopt_obj):
                 if root.root_nlp.nlp_best_x is not None:
                     root.best_primal_x = root.root_nlp.nlp_best_x
             elif root.ipopt_found and root.root_nlp.ipopt_x is not None:
                 root.best_primal_x = root.root_nlp.ipopt_x
        if earlier is not None and float(earlier[0].reshape(-1)[0]) < float(root.primal_solution.reshape(-1)[0]):
            root.primal_solution, root.best_primal_x = earlier
            root.primal_solution_found = True
        root.warmstart_x = root.best_primal_x


def _root_tangent(root, deadline):
    """The quadratic, convex or Taylor tangent bound of the objective (QTB), with its tangent solve, which may improve
    the incumbent."""
    from nlp_solver.quadratic_tangent import build_quadratic_tangent
    root_center = ((root.dm_l[:1] + root.dm_u[:1]) / 2).detach()
    # Any point gives a valid tangent plane; the incumbent gives a tight one.
    x_tan = root.best_primal_x if root.best_primal_x is not None else root.root_nlp.nlp_best_x
    tangent_args = (
        root.net.net, root.net.model_ori, root_center,
        int(arguments.Config["solving"]["obj_index"]),
        root_center.reshape(-1) if x_tan is None else x_tan)
    # Keep QTB for quadratics. A DCP certificate takes priority over
    # interval-Hessian Taylor bounds only for nonquadratic objectives.
    root.history.qtb = build_quadratic_tangent(
        *tangent_args,
        root_lower=root.dm_l[:1], root_upper=root.dm_u[:1],
        quadratic_bound=root.nlp_config.get("quadratic_tangent_bound", False), deadline=deadline)
    if root.history.qtb is None and root.nlp_config.get("convex_tangent_objective", False):
        from nlp_solver.convex_tangent import build_convex_tangent
        root.history.qtb = build_convex_tangent(
            *tangent_args, root.x.ptb.x_L, root.x.ptb.x_U, enabled=True)
    if root.history.qtb is None and root.nlp_config.get("taylor2_bound", False):
        root.history.qtb = build_quadratic_tangent(
            *tangent_args, root_lower=root.dm_l[:1], root_upper=root.dm_u[:1],
            taylor2_bound=True,
            taylor2_max_vars=int(root.nlp_config.get("taylor2_max_vars", 40)),
            taylor2_max_domains=int(root.nlp_config.get("taylor2_max_domains", 1024)),
            quadratic_bound=root.nlp_config.get("quadratic_tangent_bound", False), deadline=deadline)
    if root.history.qtb is not None:
        fixed = root.history.qtb.fixed_mask.to(root.dm_l.device)
        anchor = root.history.qtb.fixed_values.to(root.dm_l)
        lower_flat, upper_flat = root.dm_l.flatten(1), root.dm_u.flatten(1)
        if not ((lower_flat[:, fixed] == anchor[fixed]).all()
                and (upper_flat[:, fixed] == anchor[fixed]).all()):
            print("[QTB] Root boxes have different fixed coordinates; bound disabled.")
            root.history.qtb = None
    if root.history.qtb is not None:
        from input_split.nlp_in_bab import compute_qtb_tangent_point
        # A Taylor anchor needs no extra local solve or global convexity.
        is_taylor = hasattr(root.history.qtb, 'evaluator')
        x_kkt, accepted = (None, None) if is_taylor else compute_qtb_tangent_point(
            root.net.model_ori, root.dm_l, root.dm_u, root.c, root.rhs, root.or_spec_size, root.nlp_config,
            root.history.qtb.x_hat, root.base_input_linear_constraints, time_left=root.time_left())
        if x_kkt is not None:
            root.history.qtb.polish(x_kkt)
        if accepted is not None and (
                root.primal_solution is None
                or accepted[1] < float(root.primal_solution.reshape(-1)[0])):
            x_acc, obj_acc = accepted
            print(f"[QTB] tangent solve improves the incumbent: {obj_acc:.10g}"
                  + (f" (was {float(root.primal_solution.reshape(-1)[0]):.10g})"
                     if root.primal_solution is not None else ""))
            root.primal_solution = torch.tensor([obj_acc], device=root.global_lb.device)
            root.primal_solution_found = True
            root.best_primal_x = (x_acc.reshape(root.best_primal_x.shape)
                              if root.best_primal_x is not None else x_acc)
            root.warmstart_x = root.best_primal_x
        if is_taylor:
            print(f"[Taylor2] objective degree={root.history.qtb.objective_degree:g}; "
                  f"interval-Hessian tangent bound enabled (n={root.history.qtb.n}, "
                  f"max_domains={root.history.qtb.max_domains}, f={root.history.qtb.f:.10g})")
        elif hasattr(root.history.qtb, 'certificate'):
            print(f"[ConvexTangent] objective proved convex on root box; "
                  f"tangent bound enabled (n={root.history.qtb.n}, f={root.history.qtb.f:.10g})")
        else:
            print(f"[QTB] objective is quadratic; tangent bound enabled "
                  f"(lambda_min >= {root.history.qtb.lam_lb:.4g}, tangent at "
                  f"{'local KKT point' if x_kkt is not None else 'incumbent'} "
                  f"f={root.history.qtb.f:.10g})")


def _root_consistency(root, deadline):
    """The root dual, the guard against an incumbent below it, and the objective threshold."""
    if root.solving_mode:
        root.dual_solution = root.global_lb[0, 0]
        primal_dual_abs_tol = float(
            root.nlp_config.get("bab_primal_dual_abs_tol", 1e-5)
        )
        if (
            root.primal_solution_found
            and _is_initial_primal_dual_inconsistent(
                root.primal_solution,
                root.dual_solution,
                abs_tol=primal_dual_abs_tol,
            )
        ):
            initial_gap = float(
                (
                    root.primal_solution.reshape(-1)[0]
                    - root.dual_solution.reshape(-1)[0]
                ).item()
            )
            print(
                "[Optimality Guard] Refusing to prune or certify with an "
                "incumbent below the root lower bound: "
                f"primal={float(root.primal_solution.reshape(-1)[0].item()):.12g} "
                f"dual={float(root.dual_solution.reshape(-1)[0].item()):.12g} "
                f"gap={initial_gap:.12g}."
            )
            return root.global_lb.max(), root.stats.visited, "unknown"
    if root.solving_mode and root.primal_solution is not None:
        root.rhs = _apply_primal_objective_threshold(root.rhs, root.primal_solution)


def _root_domain_list(root, deadline):
    """The domain list the BaB searches."""
    if root.same_or_spec_size:
        root.domains = UnsortedInputDomainList(
            root.stats.storage_depth,
            arguments.Config["general"]["device"],
            history=root.history,
            sort_index=root.input_split_args["sort_index"],
            sort_heuristic=root.sort_heuristic,
            sort_descending=root.input_split_args["sort_descending"],
            use_split_idx=not root.use_reordered_bab
        )
    else:
        print(f"Handling general specifications in input BaB.")
        assert not root.constrained_concretize, "Currently constrained concretization is not supported for general specifications."
        assert root.base_input_constraints is None, (
            "Native input linear constraints with variable OR spec sizes are not yet supported in input BaB."
        )
        # the remaining properties have different number of AND clauses
        root.domains = UnsortedMultiSpecInputDomainList(
            root.or_spec_size,
            root.x.shape[1:],  # input shape of the network
            root.c.shape[2],  # output dimension of the network before applying C
            root.stats.storage_depth,
            arguments.Config["general"]["device"],
            history=root.history,
            sort_index=root.sort_index,
            sort_heuristic=root.sort_heuristic,
            sort_descending=root.input_split_args["sort_descending"],
            use_split_idx=not root.use_reordered_bab,
            alpha_final_name=root.net.final_name if root.history.use_alpha else None,
        )
    if root.solving_mode:
        root.domains.sort_index = 0
        root.domains.sort_heuristic = "lb_only"


def _root_symmetry(root, deadline):
    """The structure report and its symmetry rows."""
    root.structure = structure = root.structure or _root_structure(root.net, root.x, root.c, root.rhs, root.nlp_config,
                                                                   root.rowpool)
    enlarged, (root.best_primal_x, root.warmstart_x) = _add_symmetry_rows(
        root.domains, structure, root.rhs, root.x, root.base_input_constraints, [root.best_primal_x, root.warmstart_x])
    if enlarged is not root.base_input_constraints:
        root.base_input_constraints = enlarged
        root.base_input_linear_constraints = _primal_input_rows(root.base_input_constraints, root.input_rows_raw)
    if structure is not None and structure.rows:
        circles = sorted({I.circle for I in structure.rows if I.circle})
        root.history.angle_pairs = torch.tensor(circles, dtype=torch.long) if circles else None


def _root_certificate(label, key, module, builder):
    """The GP or exp certificate (solver.nlp.<key>): built on the root box and solved against the incumbent; one
    that closes the gap ends the solve."""
    def run(root, deadline):
        build = getattr(importlib.import_module(module), builder)  # lazy: mpmath import
        diagnostics = {}
        center = ((root.x.ptb.x_L + root.x.ptb.x_U) / 2).detach()
        certificate = build(root.net.net, root.net.model_ori, center, root.c, root.rhs, root.x.ptb.x_L, root.x.ptb.x_U,
                            diagnostics, rowpool=root.rowpool)
        print(f"[{label} certificate] {diagnostics}")
        if certificate is None or not root.primal_solution_found:
            return None
        if key == "gp_certificate":
            # Native CROWN planes of the log-coordinate rewrite instead of the entropy planes (crown_rewrite.py).
            certificate.crown = bool(root.nlp_config.get("gp_crown", False))
            if certificate.crown:
                print(f"[{label} certificate] planes: native CROWN on the log-coordinate rewrite")
        dual = _root_certificate_dual(label, certificate, key + "_time", root.nlp_config,
                                      root.best_primal_x, root.primal_solution, root.timeout, root.start)
        if dual is not None:
            return _finish_with_lifted_dual(root.net, root.domains, root.best_primal_x, root.primal_solution,
                                            root.primal_solution_found, dual, root.stats)
        return None
    return run


def _root_convex_cuts(root, deadline):
    """The convex row cuts and tangent rows, offered at the incumbent."""
    from nlp_solver.convex_row_cuts import build_convex_row_cuts
    qtb = getattr(root.history, 'qtb', None)
    root_center = ((root.dm_l[:1] + root.dm_u[:1]) / 2).detach()
    x_tan = qtb.x_hat if qtb is not None else (
        root.best_primal_x if root.best_primal_x is not None else root_center.reshape(-1))
    if root.nlp_config.get("convex_tangent_rows", False):
        from nlp_solver.convex_tangent import build_convex_tangent_rows
        root.history.convex_row_cuts = build_convex_tangent_rows(
            root.net.net, root.net.model_ori, root_center, root.c, root.rhs,
            root.x.ptb.x_L, root.x.ptb.x_U, x_tan, enabled=True,
            quadratic_rows=root.nlp_config.get("convex_row_cuts", False), rowpool=root.rowpool)
    else:
        root.history.convex_row_cuts = build_convex_row_cuts(
            root.net.net, root.net.model_ori, root_center, root.c, root.rhs,
            root.x.ptb.x_L, root.x.ptb.x_U, x_tan, enabled=True, rowpool=root.rowpool)
    if root.history.convex_row_cuts is not None:
        root.base_input_constraints, _ = _offer_convex_row_cuts(
            root.domains, root.base_input_constraints, None, x_tan, root.primal_solution, root.dm_l.shape[0])
        root.base_input_linear_constraints = _primal_input_rows(root.base_input_constraints, root.input_rows_raw)


def _root_lifted_lp(root, deadline):
    """The lifted McCormick LP of the root box (lifted_bound.RootLifted)."""
    from input_split.lifted_bound import RootLifted
    if root.use_reordered_bab and root.lA is not None:
        # The root planes' offsets belong to the box global_lb was computed on; a
        # plane valid there stays valid on any sub-box OBBT leaves.
        root.root_lbias = deconstruct_bias(root.dm_l, root.dm_u, root.lA, root.global_lb)
    root.lifted_setup = RootLifted.build(root.net, root.history, root.dm_l, root.dm_u, root.c, root.rhs,
                                         standard_form_to_input_constraints(root.base_input_constraints),
                                         root.nlp_config, allow_obbt=root.use_reordered_bab)


def _root_lifted_bound(root, deadline):
    """The lifted root bound of the relaxation built so far (RootLifted.bound), before the OBBT and the lifted-LP
    B&B spend their budgets (unified root). A bound that closes the gap ends the solve; one that does not is left
    to the B&B, which bounds the OBBT box again."""
    root.lifted_bound_version = root.lifted_setup.version
    bound = root.lifted_setup.bound(root.primal_solution, deadline - time.time())
    if bound is None:
        return None
    print(f"[LiftedLP] root bound before OBBT: {bound:.12g} (incumbent "
          f"{float(root.primal_solution.reshape(-1)[0]):.12g}, version {root.lifted_bound_version})")
    if getattr(root, "controller", False) and math.isfinite(bound):
        root.certified_dual = max(root.certified_dual, bound)
    return root.finish("lifted_bound", bound) if root.closes(bound) else None


def _root_lifted_warm_up(root, deadline):
    """The certified McCormick OBBT warm-up (RootLifted.warm_up)."""
    root.lifted_setup.warm_up(root.time_left())


def _root_lifted_rlt(root, deadline):
    """The RLT rows of the lifted LP (RootLifted.add_rlt)."""
    root.lifted_setup.add_rlt()


def _root_lifted_obbt(root, deadline):
    """The root OBBT over the lifted LP (RootLifted.obbt); the BaB starts from its box."""
    if (root.nlp_config.get('crown_engine_m1', False)
            or root.nlp_config.get('crown_engine_sigma_main', False)
            or root.nlp_config.get('crown_engine_m2', False)):
        from input_split.m1_addon import root_addon
        root_addon(root, deadline)
    if root.lifted_setup is not None:
        root.dm_l, root.dm_u = root.lifted_setup.obbt(incumbent_x=root.best_primal_x, time_left=root.time_left())
    root.history.root_width = (root.dm_u[:1] - root.dm_l[:1]).detach().flatten()
    root.history.lifted_root_box = (root.dm_l.detach().clone(), root.dm_u.detach().clone())


def _root_lifted_bab(root, deadline):
    """The lifted-LP B&B with its leaf incumbents. A closing one ends the solve; otherwise its open leaves raise the
    root bound and may become the BaB's first domains."""
    from input_split.lifted_bound import run_lifted_bab
    from input_split.lifted_bound import make_verifier
    handoff = bool(root.nlp_config.get("lifted_lp_bab_handoff", False)) and root.use_reordered_bab
    lifted_verify = make_verifier(root.net.model_ori, root.x.ptb.x_L, root.x.ptb.x_U, root.c, root.rhs,
                                  root.or_spec_size, root.base_input_linear_constraints,
                                  root.nlp_config)
    root.lifted_verify_loop = lifted_verify
    if lifted_verify is not None and bool(root.nlp_config.get("lifted_lp_leaf_ipopt", False)):
        from input_split.lifted_bound import make_local_solver
        lifted_local = make_local_solver(root.net.model_ori, root.x.ptb.x_L, root.x.ptb.x_U, root.c, root.rhs,
                                         root.or_spec_size, root.base_input_linear_constraints,
                                         root.nlp_config, lifted_verify)
    else:
        lifted_local = None
    root.lifted_local_loop = lifted_local
    lifted_dual = run_lifted_bab(root.history, root.dm_l, root.dm_u,
                                 root.rhs.reshape(root.rhs.shape[0], -1)[0, 0], root.nlp_config,
                                 root.timeout - (time.time() - root.start), want_leaves=handoff,
                                 verify_fn=lifted_verify, incumbent_x=root.best_primal_x,
                                 local_fn=lifted_local,
                                 floor=float(root.global_lb.reshape(root.global_lb.shape[0], -1)[:, 0].min()),
                                 time_share=1.0 if root.unified else 0.3)
    new_inc = getattr(root.history, "lifted_bab_new_incumbent", None)
    if new_inc is not None:
        x_new, obj_new = new_inc
        root.primal_solution = torch.tensor([obj_new], device=root.global_lb.device,
                                            dtype=root.rhs.dtype)
        root.best_primal_x = torch.as_tensor(x_new, dtype=root.dm_l.dtype, device=root.dm_l.device
                                             ).reshape(root.dm_l[:1].shape)
        root.warmstart_x = root.best_primal_x
        root.rhs = _apply_primal_objective_threshold(root.rhs, root.primal_solution)
        if getattr(root.history, 'convex_row_cuts', None) is not None:
            root.history.alm_best_primal_x = root.best_primal_x.detach().cpu()
            root.base_input_constraints, _ = _offer_convex_row_cuts(
                root.domains, root.base_input_constraints, None, root.best_primal_x,
                root.primal_solution, root.dm_l.shape[0])
            root.base_input_linear_constraints = _primal_input_rows(
                root.base_input_constraints, root.input_rows_raw)
    if lifted_dual is not None:
        return _finish_with_lifted_dual(root.net, root.domains, root.best_primal_x, root.primal_solution,
                                        root.primal_solution_found, lifted_dual, root.stats)
    leaves = getattr(root.history, "lifted_bab_leaves", None)
    root.history.lifted_bab_leaves = None
    if leaves is not None and len(leaves[2]) > 0:
        root.global_lb = _raise_root_lb_to_open_leaves(root.global_lb, leaves[2], root.primal_solution)
    if handoff and leaves is not None and root.lA is not None and root.root_lbias is not None:
        n_leaf = int(leaves[2].shape[0])
        budget_elems = n_leaf * root.lA[0].numel()
        if 1 <= n_leaf <= 8192 and budget_elems <= 50_000_000:
            if getattr(root.history, "qtb", None) is not None:
                try:
                    root_constraints = root.base_input_constraints
                    if root.constrained_concretize:
                        root_cuts = construct_cut_constraints(
                            root.lA, root.root_lbias, root.rhs, root.dm_l.shape[0], root.dm_l.flatten(1).shape[1], root.rowpool)
                        root_constraints = merge_standard_constraints(
                            root.base_input_constraints, root_cuts)
                    root_bounds, _, _, _ = get_bounds(
                        root.net, root.global_lb, root.dm_l, root.dm_u, root.history, root.c, root.rhs, root_constraints,
                        standard_form_to_input_constraints(root.base_input_constraints),
                        root.bounding_method, root.branching_method, root.stop_criterion, root.stats)
                    root_qtb_lb = root_bounds[:, 0]
                    root_converged = _is_primal_dual_converged(
                        root.primal_solution, root_qtb_lb,
                        abs_tol=float(root.nlp_config.get("bab_primal_dual_abs_tol", 1e-5)),
                        rel_tol=float(root.nlp_config.get("bab_primal_dual_rel_tol", 1e-4)))
                    # The regular root bound includes the QTB LP and its exact
                    # re-solve. A lower bound on the whole root box is also a
                    # lower bound on every lifted leaf, which is a sub-box.
                    root.global_lb = root.global_lb.clone()
                    root.global_lb[:, 0] = torch.maximum(root.global_lb[:, 0], root_qtb_lb)
                    if root_converged:
                        handoff = False
                except Exception as e:  # pylint: disable=broad-except
                    warnings.warn(f"[LiftedLP] Root QTB bound skipped due to error: {e}")
        if (handoff and 1 <= n_leaf <= 8192
                and budget_elems <= 50_000_000):
            # Start BaB from the open leaves of the lifted-LP B&B instead of the
            # root: they hold every relaxed-feasible point with objective below
            # the incumbent, each with its certified bound; the root planes and
            # constraint bounds stay valid on these sub-boxes.
            def _rep(t):
                return t[:1].expand(n_leaf, *t.shape[1:]).clone()
            root.dm_l = torch.as_tensor(leaves[0], dtype=root.dm_l.dtype, device=root.dm_l.device
                                        ).reshape(n_leaf, *root.dm_l.shape[1:])
            root.dm_u = torch.as_tensor(leaves[1], dtype=root.dm_u.dtype, device=root.dm_u.device
                                        ).reshape(n_leaf, *root.dm_u.shape[1:])
            root.c, root.rhs, root.lA, root.root_lbias = _rep(root.c), _rep(root.rhs), _rep(root.lA), _rep(root.root_lbias)
            root.global_lb = _rep(root.global_lb)
            root.global_lb[:, 0] = torch.maximum(
                root.global_lb[:, 0], torch.as_tensor(leaves[2]).to(root.global_lb))
            if root.base_input_constraints is not None:
                root.base_input_constraints = (_rep(root.base_input_constraints[0]),
                                               _rep(root.base_input_constraints[1]))
            root.or_spec_size = _rep(root.or_spec_size.reshape(-1))
            root.remaining_index = torch.arange(n_leaf, device=root.dm_l.device)
            print(f"[LiftedLP] BaB starts from {n_leaf} lifted-LP B&B leaves")
            root.lifted_handoff_leaves = n_leaf


def _root_initial_domains(root, deadline):
    """The initial split of the root boxes, clipped, as the BaB's first domains."""
    addon = getattr(root.net, 'm1_addon', None)
    if addon is not None and addon.sigma_main and addon.select_main() == 'sigma':
        root.global_lb = addon.bound_main(root.dm_l, root.dm_u, root.global_lb,
            root.rowpool.thresholds(root.rhs).to(root.global_lb), deadline, root.history)
        _apply_engine_box(root.dm_l, root.dm_u, root.history)
        split_idx = None if root.use_reordered_bab else input_split_branching(
            root.net, root.global_lb, root.dm_l, root.dm_u, None, root.rhs,
            'naive', root.stats.storage_depth, history=root.history)
        counters = torch.zeros_like(root.rhs[..., :1], dtype=torch.int)
        _add_root_domains(root.domains, root.global_lb, root.dm_l.detach(), root.dm_u.detach(),
            root.c, root.rhs, constraints=None,
            base_constraints=root.base_input_constraints, split_idx=split_idx,
            check_dm_lbs=True, check_input_boxes=False, spec_sizes=root.or_spec_size,
            attack_counter=counters, split_counter=counters.clone())
        root.lA = root.root_lbias = root.ret = None
        return
    if root.use_reordered_bab:
        # filter out verified subdomains
        lbias = root.root_lbias if root.root_lbias is not None else deconstruct_bias(root.dm_l, root.dm_u, root.lA, root.global_lb)

        if getattr(root, "controller", False) and math.isfinite(root.certified_dual):
            root.global_lb = root.global_lb.clone()
            root.global_lb[:, 0].clamp_(min=root.certified_dual)
        constraints = root.base_input_constraints
        # Construct constraints matrices if use complete clip.
        if root.constrained_concretize:
            # Constraints matrices will come from the output node A matrices and b matrices from last CROWN call
            current_batch_size = root.dm_l.shape[0]
            current_x_dim = root.dm_l.view((current_batch_size, -1)).shape[1]
            concretize_constraints = construct_cut_constraints(
                root.lA, lbias, root.rhs, current_batch_size, current_x_dim, root.rowpool)
            constraints = merge_standard_constraints(root.base_input_constraints, concretize_constraints)

        # TODO: still, in the future, attack_counter/split_counter and solving mode should be distangled
        attack_counter = None
        split_counter = None
        if root.solving_mode:
            attack_counter = torch.zeros_like(root.rhs[...,0:1], device=root.rhs.device, dtype=torch.int)
            split_counter = torch.zeros_like(root.rhs[...,0:1], device=root.rhs.device, dtype=torch.int)

        num_unverified_domains = len(root.remaining_index)
        spec_sizes = root.or_spec_size
        # compute initial split idx
        split_idx = input_split_branching(
            root.net, root.global_lb, root.dm_l, root.dm_u, root.lA, root.rhs, root.branching_method, root.stats.storage_depth, num_iter=0,
            history=root.history)

        # perform the initial split on the domains
        root.dm_l, root.dm_u, root.c, root.rhs, root.global_lb, spec_sizes, root.lA, lbias, constraints, root.base_input_constraints, attack_counter, split_counter = input_split_and_repeat(
            root.dm_l, root.dm_u, num_unverified_domains, root.x.shape, split_idx, root.split_partitions, root.split_hint,
            root.c, root.rhs, root.global_lb, root.domains.history, spec_sizes, root.lA, lbias, constraints, root.base_input_constraints,
            attack_counter=attack_counter, split_counter=split_counter)
        # shrink the initial dm_l and dm_u
        if root.use_clip_domains:
            root.dm_l, root.dm_u = _clip_root_domains(
                root.dm_l,
                root.dm_u,
                root.rhs,
                root.lA,
                lbias,
                calculate_volume=root.clip_calculate_volume_metrics,
                base_input_linear_constraints=_get_clip_base_constraints(
                    root.base_input_constraints
                ),
                base_input_constraints=root.base_input_constraints,
                domain_constraints=constraints,
                rowpool=root.rowpool,
            )
        # Compute x_star for initial domains if relaxation warmstart is enabled
        x_star = None
        if root.input_split_args.get("use_relaxation_warmstart", False):
            x_star = extract_relaxation_minimizer(root.lA, lbias, root.dm_l, root.dm_u, target_spec_idx=0)
        _add_root_domains(root.domains, root.global_lb, root.dm_l.detach(), root.dm_u.detach(),
                    root.c, root.rhs, constraints=constraints, base_constraints=root.base_input_constraints,
                    split_idx=None, remaining_index=None,
                    check_dm_lbs=root.solving_mode, check_input_boxes=root.use_clip_domains,
                    spec_sizes=spec_sizes, attack_counter=attack_counter, split_counter=split_counter,
                    x_star=x_star
                    )
    else:
        # compute initial split idx
        split_idx = input_split_branching(
            root.net, root.global_lb, root.dm_l, root.dm_u, root.lA, root.rhs, root.branching_method, root.stats.storage_depth, num_iter=0,
            history=root.history)

        constraints = root.base_input_constraints
        # Construct constraints matrices if constrained concretize is enabled.
        if root.constrained_concretize:
            # Constraints matrices will come from the output node A matrices and b matrices from last CROWN call
            lbias = deconstruct_bias(root.dm_l, root.dm_u, root.lA, root.global_lb)
            current_batch_size = root.dm_l.shape[0]
            current_x_dim = root.dm_l.view((current_batch_size, -1)).shape[1]
            concretize_constraints = construct_cut_constraints(
                root.lA, lbias, root.rhs, current_batch_size, current_x_dim, root.rowpool)
            constraints = merge_standard_constraints(root.base_input_constraints, concretize_constraints)

        # TODO: still, in the future, attack_counter/split_counter and solving mode should be distangled
        attack_counter = None
        split_counter = None
        if root.solving_mode:
            attack_counter = torch.zeros_like(root.rhs[...,0:1], device=root.rhs.device, dtype=torch.int)
            split_counter = torch.zeros_like(root.rhs[...,0:1], device=root.rhs.device, dtype=torch.int)

        if root.use_clip_domains and not root.use_reordered_bab:
            root.dm_l, root.dm_u = _clip_root_domains(
                root.dm_l,
                root.dm_u,
                root.rhs,
                root.lA,
                None,
                root.global_lb,
                calculate_volume=root.clip_calculate_volume_metrics,
                base_input_linear_constraints=_get_clip_base_constraints(
                    root.base_input_constraints
                ),
                base_input_constraints=root.base_input_constraints,
                domain_constraints=constraints,
                rowpool=root.rowpool,
            )
        # Compute x_star for initial domains if relaxation warmstart is enabled
        x_star = None
        if root.input_split_args.get("use_relaxation_warmstart", False):
            # For non-reordered bab, lbias might not be computed yet, so compute it if needed
            lbias_init = deconstruct_bias(root.dm_l, root.dm_u, root.lA, root.global_lb) if root.lA is not None else None
            x_star = extract_relaxation_minimizer(root.lA, lbias_init, root.dm_l, root.dm_u, target_spec_idx=0)
        _add_root_domains(root.domains, root.global_lb, root.dm_l.detach(), root.dm_u.detach(),
                    root.c, root.rhs, constraints=constraints, base_constraints=root.base_input_constraints, split_idx=split_idx,
                    remaining_index=None, check_dm_lbs=True,
                    check_input_boxes=root.use_clip_domains,
                    spec_sizes=root.or_spec_size, attack_counter=attack_counter, split_counter=split_counter,
                    x_star=x_star
                    )

    # The root's A matrices, cut rows and biases are in `domains` now and the search below
    # never reads these names again; at (batch x rows x vars) each is gigabytes on the
    # large pooling instances (5.7 GiB on pooling_sppb5pq), held for the whole search.
    root.lA = root.root_lbias = root.ret = None
    if addon is not None and addon.sigma_main:
        _drop_sigma_planes(root.domains)


def _lifted_bound_due(root):
    """The unified root bounds the lifted LP once an incumbent exists, after each step that strengthened it."""
    return (root.unified and root.primal_solution_found and root.lifted_setup is not None
            and root.lifted_setup.version > root.lifted_bound_version)


_ROOT_OPERATIONS = (
    RootOp("reference_bounds", _root_reference_bounds),
    RootOp("rows", _root_rows, lambda root: root.solving_mode),
    RootOp("struct_forms", _root_struct_forms, lambda root: bool(_struct_cert_kinds(root))),
    RootOp("struct_cert", _root_struct_cert, _struct_cert_due),
    RootOp("hybrid", _root_hybrid,
           lambda root: (root.solving_mode and root.nlp_config.get("root_hybrid_term_bound", False)
                         and arguments.Config["solving"]["obj_index"] == 0)),
    RootOp("initial_verify", _root_initial_verify),
    RootOp("lp_alternation", _root_lp_alternation,
           lambda root: root.solving_mode and root.nlp_config.get("lp_alternation", False), timed=True),
    RootOp("nlp", _root_nlp),   # the root primal window, with the deadline ahead
    RootOp("search_state", _root_search_state),
    RootOp("incumbent", _root_incumbent),
    RootOp("tangent", _root_tangent,
           lambda root: (root.solving_mode
                         and (root.nlp_config.get("quadratic_tangent_bound", False)
                              or root.nlp_config.get("convex_tangent_objective", False)
                              or root.nlp_config.get("taylor2_bound", False))
                         and int(root.nlp_config.get("dual_lp_pdhg_iters", 0)) > 0),
           timed=True),
    RootOp("consistency", _root_consistency),
    RootOp("domain_list", _root_domain_list),
    RootOp("symmetry", _root_symmetry, lambda root: root.solving_mode),
    *(RootOp(key, _root_certificate(label, key, module, builder),
             lambda root, key=key: root.solving_mode and root.same_or_spec_size and root.nlp_config.get(key, False),
             timed=True)
      for label, key, module, builder in (
          ("GP", "gp_certificate", "nlp_solver.gp_certificate", "build_gp_certificate"),
          ("exp", "exp_certificate", "nlp_solver.exp_certificate", "build_exp_certificate"))),
    RootOp("convex_cuts", _root_convex_cuts,
           lambda root: (root.solving_mode and root.same_or_spec_size
                         and (root.nlp_config.get("convex_row_cuts", False)
                              or root.nlp_config.get("convex_tangent_rows", False)))),
    RootOp("lifted_lp", _root_lifted_lp, lambda root: root.solving_mode),
    RootOp("lifted_bound", _root_lifted_bound, _lifted_bound_due, timed=True),
    RootOp("lifted_warm_up", _root_lifted_warm_up, lambda root: root.lifted_setup is not None, timed=True),
    RootOp("lifted_rlt", _root_lifted_rlt, lambda root: root.lifted_setup is not None),
    RootOp("lifted_bound", _root_lifted_bound, _lifted_bound_due, timed=True),
    RootOp("lifted_obbt", _root_lifted_obbt, lambda root: root.solving_mode, timed=True),
    RootOp("lifted_bab", _root_lifted_bab,
           lambda root: (root.solving_mode and root.primal_solution_found
                         and float(root.nlp_config.get("lifted_lp_bab_time", 0.0)) > 0),
           timed=True),
    RootOp("initial_domains", _root_initial_domains),
)


def _run_root(root):
    """Run the root operations in order, one ledger line each; the outcome of the one that ends the solve, else None.
    The unified root admits a timed operation only before the deadline and ends the solve as soon as an operation
    leaves the incumbent and the certified root bound closed under the gap rule."""
    try:
        for op in _ROOT_OPERATIONS:
            if getattr(root, 'controller', False) and root.time_left() <= 0:
                from input_split.root_controller import finish_timeout
                return finish_timeout(root)
            if not op.eligible(root) or (root.unified and op.timed and time.time() >= root.deadline):
                continue
            t0 = time.time()
            try:
                outcome = op.run(root, root.deadline)
            except BaseException as error:
                from input_split.root_controller import RootClosed, RootDeadline, finish_timeout
                if not isinstance(error, (RootClosed, RootDeadline, GeneratorExit)):
                    raise
                P, D = root.values()
                seconds = time.time() - t0
                if op.name == 'nlp' and getattr(root, 'controller', False):
                    seconds = max(0.0, seconds - getattr(root, 'controller_nested_s', 0.0))
                root.ledger.commit(op.name, seconds, P, D)
                if isinstance(error, RootDeadline):
                    return finish_timeout(root)
                return root.finish(error.source, D)
            P, D = root.values()
            if isinstance(outcome, tuple) and outcome[2] == "safe":
                D = max(D, float(outcome[0]))     # an operation that ends the solve returns its certified dual
            seconds = time.time() - t0
            if op.name == 'nlp' and getattr(root, 'controller', False):
                seconds = max(0.0, seconds - getattr(root, 'controller_nested_s', 0.0))
            root.ledger.commit(op.name, seconds, P, D)
            if outcome is not None:
                return outcome
            if root.unified and root.closes(D):
                return root.finish(op.name, D)
        return None
    finally:
        # Also reap our child if an operation raises or closes the gap before the NLP returns.
        lane = getattr(root, "primal_lane", None)
        if lane is not None:
            lane.join()
        if getattr(root, "controller", False):
            from input_split.root_controller import settle_cancelled
            settle_cancelled(root)




@retained
def _input_bab_loop(root, vnnlib, return_domains, index):
    """The BaB loop from the root's first domains until the gap closes, the frontier empties or time runs out;
    then the incumbent's export."""
    net, x, rhs, stats, nlp_config, solving_mode = (
        root.net, root.x, root.rhs, root.stats, root.nlp_config, root.solving_mode)
    domains, history, dm_l, global_lb, dual_solution = (
        root.domains, root.history, root.dm_l, root.global_lb, root.dual_solution)
    primal_solution, primal_solution_found, best_primal_x, warmstart_x = (
        root.primal_solution, root.primal_solution_found, root.best_primal_x, root.warmstart_x)
    report_only_primal, hybrid, result = root.report_only_primal, root.hybrid, root.result
    start, timeout, max_iterations, batch = root.start, root.timeout, root.max_iterations, root.batch
    bounding_method, branching_method, stop_criterion = root.bounding_method, root.branching_method, root.stop_criterion
    split_partitions, use_reordered_bab, input_split_args = (
        root.split_partitions, root.use_reordered_bab, root.input_split_args)
    sort_domain_iter, enable_attack_counter, enable_check_adv, adv_check = (
        root.sort_domain_iter, root.enable_attack_counter, root.enable_check_adv, root.adv_check)
    pgd_order, presplit_domains, same_or_spec_size = root.pgd_order, root.presplit_domains, root.same_or_spec_size
    lifted_verify_loop, lifted_local_loop, lifted_handoff_leaves = (
        root.lifted_verify_loop, root.lifted_local_loop, root.lifted_handoff_leaves)
    ledger = root.ledger
    if pgd_order == "after":
        if arguments.Config['model']['with_jacobian']:
            model_to_attack = net.net
        else:
            model_to_attack = net.model_ori

        # The domain list currently only has the original unverified domains, we can perform
        # pgd attacks on them directly.
        if pgd_attack_on_domains(model_to_attack, domains, 'init', vnnlib):
            print("pgd attack succeed in input_bab_parallel")
            result = "unsafe"

    if presplit_domains:
        assert not history.use_alpha
        assert same_or_spec_size, "Currently only support same number of AND specs"
        load_presplit_domains(
            domains, net, bounding_method, branching_method, stop_criterion, stats, index=index,
        )

    if solving_mode and nlp_config.get("m3_reduce", False):
        # M3: the exact pattern of the root rows and the base rows the BaB starts with (input_split/m3_reduce.py)
        from input_split.m3_reduce import setup as m3_setup
        m3_setup(history, root.structure, dm_l.flatten(1).shape[1], root.base_input_constraints, dm_l.device,
                 root.deadline)

    if root.controller:
        from input_split.tree_slice import BatchCap, active, region, service_clock
        from input_split.tree_progress import domain_snapshot
        from input_split.root_controller import RootClosed, RootDeadline
        tree_cap = BatchCap(batch, int(input_split_args.get('sb_strong_max_children', 65536)))

    auto_batch_size = AutoBatchSize(
        batch, net.device,
        enable=arguments.Config["solver"]["auto_enlarge_batch_size"])
    sigma_addon = getattr(net, 'm1_addon', None)
    deadline_batches = _DeadlineBatchSizer(
        time.monotonic() + max(0.0, start + timeout - time.time()))

    alm_state = init_alm_bab_state(nlp_config, warmstart_x, best_primal_x)
    alm_state.deadline = start + timeout
    if root.controller:
        alm_state = None

    if (result == "unknown" and hybrid is not None and primal_solution_found
            and _is_primal_dual_converged(
                primal_solution, dual_solution,
                abs_tol=float(nlp_config.get("bab_primal_dual_abs_tol", 1e-5)),
                rel_tol=float(nlp_config.get("bab_primal_dual_rel_tol", 1e-4)))):
        print(f"Found optimal object at root: {primal_solution}")
        result = "safe"

    num_iter = 1
    ledger.start_bab(stats.timer)
    m2 = getattr(net,'m2_interleave',None)
    lane_local = vars(net).get('lane_local')
    lane_tree_seconds = 0.
    m2_expired = False
    m2_unconverged = False
    if m2 is not None:
        if root.controller: m2.tree_clock = lambda: service_clock(root)
        m2.start()
    while (result == "unknown" and len(domains) > 0
           and (max_iterations == -1 or num_iter <= max_iterations)):
        if lane_local is not None:
            primal_solution, primal_solution_found = lane_local.service(
                domains, primal_solution, primal_solution_found, alm_state, start+timeout,
                lane_tree_seconds, m2.m2_seconds if m2 is not None else 0.)
        if m2 is not None:
            cutoff = m2.service(domains,primal_solution,primal_solution_found,start+timeout)
            m2_expired = m2.expired
            if m2_expired: break
            if cutoff is not None:
                from nlp_solver.rigorous_dual import round_down
                discarded = arguments.Globals.get('discarded_threshold_min')
                cutoff = cutoff if discarded is None else min(cutoff,discarded)
                certified = round_down(torch.tensor(cutoff,device=global_lb.device,dtype=torch.float64),global_lb.dtype)
                m2_unconverged = not _is_primal_dual_converged(
                    primal_solution, certified,
                    abs_tol=float(nlp_config.get("bab_primal_dual_abs_tol", 1e-5)),
                    rel_tol=float(nlp_config.get("bab_primal_dual_rel_tol", 1e-4)))
                if not m2_unconverged:
                    dual_solution = torch.maximum(torch.as_tensor(dual_solution,device=global_lb.device,dtype=global_lb.dtype),
                                                  certified)
                    global_lb = dual_solution
                    reported_primal = primal_solution if report_only_primal is None else report_only_primal
                    sign = -1 if arguments.Config["solving"]["is_maximization"] else 1
                    print(f"Primal Solution {sign*reported_primal}, Dual Solution {sign*dual_solution}")
                    _report.record_pd(sign*reported_primal, sign*dual_solution)
                    ledger.iteration(stats.timer, 0, float(primal_solution.reshape(-1)[0]), float(dual_solution))
                    result = 'safe'
                    break
                if not len(domains): break
        if root.controller:
            yield Atom('gpu_tree', bound=max(root.values()[1], float(dual_solution)),
                       progress=domain_snapshot(domains, root))
            if root.time_left() <= 0:
                break
            primal_solution = root.primal_solution
            primal_solution_found = root.primal_solution_found
            best_primal_x = root.best_primal_x
            dual_solution = global_lb.new_tensor(root.values()[1])
            domains.history.alm_best_primal_x = best_primal_x
        print(f"Iteration {num_iter}")
        visited = stats.visited
        # sort the domains every certain number of iterations
        if sort_domain_iter > 0 and num_iter % sort_domain_iter == 0:
            stats.timer.start("sort_domains")
            domains.sort(enable_attack_counter=enable_attack_counter)
            stats.timer.add("sort_domains")


        context_manager = prefix_print_output if solving_mode else no_op_context
        if solving_mode:
            print("####################################################################################")
            print("##                                  Primal Solving                                ##")
            print("####################################################################################")
        with context_manager():
            if enable_check_adv:
                if adv_check != -1 and stats.visited >= adv_check:
                    stats.timer.start("adv_check")
                    # check whether adv example found
                    if arguments.Config['model']['with_jacobian']:
                        model_to_attack = net.net
                    else:
                        model_to_attack = net.model_ori
                    if pgd_attack_on_domains(model_to_attack, domains, 'bab', vnnlib):
                        return global_lb.max(), stats.visited, "unsafe"
                    stats.timer.add("adv_check")

        if solving_mode:
            print("####################################################################################")
            print("##                                   Dual Solving                                 ##")
            print("####################################################################################")
        with context_manager():
            hint = _sigma_batch_hint(sigma_addon, auto_batch_size)
            if hint is not None: batch = hint
            batch_ = batch
            if hint is None and branching_method == "brute-force" and num_iter <= input_split_args["bf_iters"]:
                batch_ = input_split_args["bf_batch_size"]
            if root.controller:
                # AutoBatchSize's VRAM test is CUDA-only. On CPU, let the same
                # measured work cap grow a batch, within the existing child cap.
                if torch.device(net.device).type == 'cpu' and auto_batch_size.enable:
                    batch_ = tree_cap.limit
                batch_ = tree_cap.batch(batch_, getattr(root, 'controller_tree_quantum', None), root.time_left())
            requested_batch = batch_
            batch_ = deadline_batches.cap(batch_, len(domains))
            if batch_ < min(requested_batch, len(domains)):
                print(f"[DeadlineBatch] requested={requested_batch} capped={batch_} "
                      f"remaining={deadline_batches.deadline - deadline_batches.clock():.6f}s")
            if batch_ == 0:
                print("Time out! No predicted batch fits before the deadline.")
                break
            auto_batch_size.record_actual_batch_size(min(batch_, len(domains)))
            oom = False
            try:
                batch_start = time.monotonic()
                tree_start = service_clock(root) if root.controller else time.monotonic()
                parents = min(batch_, len(domains))
                if root.controller:
                    with active(root), region('batch', parents):
                        global_lb, primal_solution, primal_solution_found = yield from batch_verification_input_split(
                            domains, net, batch_,
                            num_iter=num_iter, decision_thresh=rhs, shape=x.shape,
                            bounding_method=bounding_method, branching_method=branching_method,
                            stop_criterion=stop_criterion, split_partitions=split_partitions,
                            use_reordered_bab=use_reordered_bab, stats=stats,
                            primal_solution=primal_solution,
                            primal_solution_found=primal_solution_found,
                            nlp_config=nlp_config,
                            alm_state=alm_state, _sliced=True, _iterate=True)
                    tree_cap.account(parents, service_clock(root)-tree_start)
                else:
                    global_lb, primal_solution, primal_solution_found = batch_verification_input_split(
                        domains, net, batch_,
                        num_iter=num_iter, decision_thresh=rhs, shape=x.shape,
                        bounding_method=bounding_method, branching_method=branching_method,
                        stop_criterion=stop_criterion, split_partitions=split_partitions,
                        use_reordered_bab=use_reordered_bab, stats=stats,
                        primal_solution=primal_solution,
                        primal_solution_found=primal_solution_found,
                        nlp_config=nlp_config,
                        alm_state=alm_state)
            except BaseException as error:
                if root.controller and isinstance(error, (RootClosed, RootDeadline, GeneratorExit)):
                    _restore_pickout(domains)
                    raise
                if not isinstance(error, RuntimeError) and not is_cuda_oom(error):
                    raise
                # Enlargement assumes memory grows linearly with the batch; some ops
                # spike (cute_airport: an 8 GiB allocation at batch 2^k). Put the batch
                # back, halve it, stop enlarging and retry the iteration. TorchScript
                # kernels re-raise the OOM as a plain RuntimeError (MeyerRoth).
                if (not is_cuda_oom(error)
                        or getattr(domains, "_last_pickout", None) is None):
                    raise
                oom = True
            if oom:
                # Outside the handler: its traceback keeps the failed batch's tensors alive.
                batch = _batch_after_oom(domains, batch_, auto_batch_size, m2=m2, addon=sigma_addon)
                if batch is None:
                    stop_lb = _report_after_oom_stop(domains, stats, primal_solution)
                    if stop_lb is not None:
                        global_lb = stop_lb
                    break
                continue
            deadline_batches.record(parents, time.monotonic() - batch_start)
            domains._last_pickout = None
            if lane_local is not None:
                lane_tree_seconds += max(0., (service_clock(root) if root.controller else time.monotonic())-tree_start)
            if not root.controller and solving_mode and isinstance(getattr(domains.history, "lifted_box", None), tuple):
                from input_split.lifted_bound import reobbt_on_improvement
                t_op = time.time()
                if reobbt_on_improvement(domains.history, primal_solution, nlp_config,
                                         timeout - (time.time() - start)):
                    ledger.note("lifted_reobbt", time.time() - t_op)
            if (not root.controller and solving_mode and primal_solution_found
                    and float(nlp_config.get("lifted_lp_bab_time", 0.0)) > 0
                    and isinstance(getattr(domains.history, "lifted_root_box", None), tuple)):
                from input_split.lifted_bound import run_lifted_bab
                t_op, P_op = time.time(), float(primal_solution.reshape(-1)[0])
                box = getattr(domains.history, "lifted_box", None)
                if not isinstance(box, tuple):
                    box = domains.history.lifted_root_box
                inc_x = getattr(domains.history, "alm_best_primal_x", None)
                # The reported bound is the one at the start of the iteration: until the BaB has visited the
                # leaves it started from and their children it cannot show whether it is level or ahead
                # (kall_diffcircles_8 iteration 1: 4 domains visited, bound = D); count it as ahead.
                lifted_dual = run_lifted_bab(domains.history, box[0], box[1], primal_solution,
                                             nlp_config, timeout - (time.time() - start),
                                             verify_fn=lifted_verify_loop,
                                             incumbent_x=inc_x if inc_x is not None
                                             else best_primal_x,
                                             local_fn=lifted_local_loop,
                                             bab_lb=float(global_lb.reshape(-1)[0])
                                             if stats.visited >= 2 * lifted_handoff_leaves
                                             else float("inf"),
                                             competing_progress=ledger.competing_progress(
                                                 stats.timer, float(primal_solution.reshape(-1)[0]),
                                                 float(global_lb.reshape(-1)[0])))
                new_inc = getattr(domains.history, "lifted_bab_new_incumbent", None)
                if new_inc is not None:
                    x_new, obj_new = new_inc
                    primal_solution = torch.tensor([obj_new], device=primal_solution.device,
                                                   dtype=primal_solution.dtype)
                    domains.history.alm_best_primal_x = torch.as_tensor(
                        x_new, dtype=dm_l.dtype, device=dm_l.device).reshape(dm_l[:1].shape)
                    domains.history.lifted_bab_new_incumbent = None
                ledger.note("lifted_bab", time.time() - t_op, dP=_gain(float(primal_solution.reshape(-1)[0]), P_op),
                            dD=0.0 if lifted_dual is None else _gain(ledger.D, float(lifted_dual)))
                if lifted_dual is not None:
                    ledger.iteration(stats.timer, stats.visited - visited, float(primal_solution.reshape(-1)[0]),
                                     float(lifted_dual))
                    return _finish_with_lifted_dual(net, domains, best_primal_x,
                                                    primal_solution, primal_solution_found,
                                                    lifted_dual, stats)

        if solving_mode:
            print("####################################################################################")
            print("##                                 Solving Results                                ##")
            print("####################################################################################")
        with context_manager():
            if solving_mode:
                if global_lb.ndim == 0:
                    dual_solution = max(global_lb, dual_solution)
                elif global_lb.ndim == 1:
                    dual_solution = max(global_lb[0], dual_solution)
                else:
                    dual_solution = max(global_lb[0, 0], dual_solution)

                # Read optimization direction from config (set at problem
                # construction time) rather than from model_ori, which may
                # have been wrapped by _StructuralConstraintOutputWrapper.
                reported_primal = (primal_solution if report_only_primal is None
                                   else report_only_primal)
                if arguments.Config["solving"]["is_maximization"]:
                    primal_solution_print = -reported_primal
                    dual_solution_print = -dual_solution
                else:
                    primal_solution_print = reported_primal
                    dual_solution_print = dual_solution
                dual_report = dual_solution
                if m2 is not None:
                    # Report only: the tree's D does not see M2's root bound; pruning is unchanged.
                    dual_report = torch.maximum(dual_solution, torch.as_tensor(m2.root_floor(), dtype=dual_solution.dtype, device=dual_solution.device))
                    dual_solution_print = -dual_report if arguments.Config["solving"]["is_maximization"] else dual_report
                print(f"Primal Solution {primal_solution_print}, Dual Solution {dual_solution_print}")
                _report.record_pd(primal_solution_print, dual_solution_print)
                solution_gap = (reported_primal - dual_report).item()
                print(f"Solution gap: {solution_gap}")
                if not root.controller:
                    ledger.iteration(stats.timer, stats.visited - visited, float(primal_solution.reshape(-1)[0]),
                                     float(dual_solution))
                primal_dual_abs_tol = float(nlp_config.get("bab_primal_dual_abs_tol", 1e-5))
                primal_dual_rel_tol = float(nlp_config.get("bab_primal_dual_rel_tol", 1e-4))
                if deadline_batches.clock() >= deadline_batches.deadline:
                    print(f"Time out! Current sub-optimal object: {primal_solution_print}")
                    _report.record_event("time_limit")
                    break
                if _is_primal_dual_converged(
                        reported_primal, dual_solution,
                        abs_tol=primal_dual_abs_tol, rel_tol=primal_dual_rel_tol):
                    # TODO: Maybe return?
                    print(f"Found optimal object: {primal_solution_print}")
                    result = "safe"
                    break
                if solution_gap < -primal_dual_abs_tol:
                    # This is the soundness invariant in solving mode: a certified
                    # dual must never exceed the objective of a feasible primal.
                    # --sanity_check turns the warning into a hard failure, using
                    # --sanity_check_delta as the allowed slack on the primal.
                    _debug_cfg = arguments.Config["debug"]
                    if _debug_cfg["sanity_check"]:
                        _slack = float(_debug_cfg["sanity_check_delta"])
                        if solution_gap < -_slack:
                            raise AssertionError(
                                "Sanity check failed: certified dual exceeds a feasible "
                                f"primal. primal={primal_solution_print}, "
                                f"dual={dual_solution_print}, gap={solution_gap:.6e}, "
                                f"allowed slack={_slack:.3e}."
                            )
                    print(
                        "Warning: primal is below dual beyond tolerance "
                        f"({solution_gap:.3e} < -{primal_dual_abs_tol:.3e}); continue BaB."
                    )
                elif time.time() - start > timeout:
                    print(f"Time out! Current sub-optimal object: {primal_solution_print}")
                    _report.record_event("time_limit")
                    break
            hint = _sigma_batch_hint(sigma_addon, auto_batch_size)
            batch = check_auto_enlarge_batch_size(auto_batch_size) if hint is None else hint
            if time.time() - start > timeout:
                print("Time out!")
                _report.record_event("time_limit")
                break
            print(f"Cumulative time: {time.time() - start}\n")
            num_iter += 1

    if lane_local is not None:
        lane_local.close()
        primal_solution, primal_solution_found = lane_local.service(
            domains, primal_solution, primal_solution_found, alm_state, time.time(),
            lane_tree_seconds, m2.m2_seconds if m2 is not None else 0.)
    if m2 is not None:
        m2.disabled = True
        m2.release()

    if (result == "unknown" and len(domains) == 0
            and not m2_expired and not m2_unconverged
            and deadline_batches.clock() < deadline_batches.deadline
            and not getattr(domains, "_domains_lost", False)):
        result = "safe"

    if root.controller:
        yield Atom('gpu_tree', bound=max(root.values()[1], float(dual_solution)),
                       progress=domain_snapshot(domains, root))
    if solving_mode:
        export_best_primal(net, domains, best_primal_x, primal_solution, primal_solution_found)
    if return_domains:
        assert same_or_spec_size, "Currently only support same number of AND specs"
        # Thresholds may have been updated by PGD attack so that different
        # domains may have different thresholds. Restore thresholds to the
        # default RHS for the sorting.
        # FIXME:
        #   We cannot return domains when using UnsortedMultiSpecInputDomainList with this
        #   sort of indexing.
        domains.threshold._storage.data[:] = rhs
        domains.sort()
        if return_domains == -1:
            return_domains = len(domains)
        lower_bound, x_L, x_U = domains.pick_out_batch(
            return_domains, device="cpu")[1:4]
        return lower_bound, x_L, x_U
    else:
        del domains
        return global_lb.max(), stats.visited, result


def input_bab_parallel(net: LiRPANet, x: BoundedTensor, c: Tensor, rhs: Tensor,
                       or_spec_size: Tensor,
                       reference_dict:Optional[dict]=None,
                       timeout:Optional[float]=None, max_iterations: Optional[int]=None,
                       vnnlib=None, return_domains:bool=False, index: Optional[int] = None,
                       ):
    """Run input split bab: the root operations (_ROOT_OPERATIONS), then the BaB loop.

    """
    # lp sanity check
    # Set instance index for LP sanity check CSV file naming
    if index is not None:
        set_instance_index(index)
    # lp sanity check end

    _reset_instance_state()
    root = RootState(net, x, c, rhs, or_spec_size, reference_dict, timeout, max_iterations)
    root.ledger.line("setup", root.start - root.ledger.origin)   # process start to here: load, presolve, incomplete
    try:
        if root.controller:
            from input_split.slice_controller import run
            return run(root, vnnlib, return_domains, index)
        outcome = _run_root(root)
        if outcome is not None:
            return outcome
        return _input_bab_loop(root, vnnlib, return_domains, index)
    finally:
        lane = vars(net).get('lane_local')
        if lane is not None: lane.close()


def load_presplit_domains(domains: Union[UnsortedInputDomainList, UnsortedMultiSpecInputDomainList],
                          net: LiRPANet, bounding_method: str, branching_method: str,
                          stop_criterion: Callable, stats: Stats, index: int):
    """

    :param domains:             Input domain list for storing subdomains during BaB.
    :param net:                 AutoLiRPA net object. This is a mutable object.
    :param bounding_method:     Bounding method to be used the original domain has been bounded and split.
    :param branching_method:    Input BaB branching method i.e. naive, sb, etc.
    :param stop_criterion:           lambda function defining stopping (verification) criteria
    """
    input_split_args = arguments.Config["bab"]["branching"]["input_split"]
    use_reordered_bab = input_split_args["reorder_bab"]

    clip_input_args = arguments.Config["bab"]["clip_n_verify"]["clip_input_domain"]
    enable_clip_domains = clip_input_args["enabled"]
    batch_size = arguments.Config["solver"]["batch_size"]
    batch_size = batch_size*2 if use_reordered_bab else batch_size
    ret = domains.pick_out_batch(len(domains))
    dm_lb, x_L, x_U, cs, thresholds, _, base_constraints, split_idx, _, _ = ret
    batch_size = max(1, batch_size // dm_lb.shape[0])

    if input_split_args["presplit_domains"].endswith(".pt"):
        # There should be only a single instance if the presplit domains is a single file
        assert index == 0
        load_path = input_split_args["presplit_domains"]
    else:
        # One presplit domain list for each instance
        load_path = os.path.join(input_split_args["presplit_domains"], f"{index}.pt")
    print("Loading pre-split domains from", load_path)
    presplit_dm_l, presplit_dm_u = torch.load(load_path)
    presplit_dm_l = presplit_dm_l.to(dm_lb)
    presplit_dm_u = presplit_dm_u.to(dm_lb)
    num_presplit_domains = presplit_dm_l.shape[0]
    print(f"Loaded {num_presplit_domains} pre-split domains")

    num_batches = (num_presplit_domains + batch_size - 1) // batch_size
    for i in range(num_batches):
        print(f"Pre-split domains batch {i+1}/{num_batches}:")
        x_L = presplit_dm_l[i*batch_size:(i+1)*batch_size]
        x_U = presplit_dm_u[i*batch_size:(i+1)*batch_size]
        size = x_L.shape[0]

        x_L = x_L.view(x_L.shape[0], -1, x_L.shape[-1]).repeat(1, dm_lb.shape[0], 1).view(-1, x_L.shape[-1])
        x_U = x_U.view(x_U.shape[0], -1, x_U.shape[-1]).repeat(1, dm_lb.shape[0], 1).view(-1, x_U.shape[-1])
        dm_lb_ = dm_lb.repeat(size, 1)
        cs_ = cs.repeat(size, 1, 1)
        thresholds_ = thresholds.repeat(size, 1)
        clip_base_constraints = _get_clip_base_constraints(base_constraints)
        if clip_base_constraints is not None:
            clip_base_constraints = repeat_input_constraints(
                clip_base_constraints,
                size,
            )
        clip_base_constraints_std = base_constraints
        if clip_base_constraints_std is not None:
            clip_base_constraints_std = repeat_input_constraints(
                clip_base_constraints_std,
                size,
            )

        new_dm_lb, lA, lbias, _ = get_bounds(
            net, dm_lb_, x_L, x_U, domains.history, cs_, thresholds_, None,
            None, bounding_method, branching_method, stop_criterion, stats=stats
        )

        if enable_clip_domains:
            clip_iters = clip_input_args["clip_iterations"]
            clip_calculate_volume_metrics = clip_input_args["clip_calculate_volume_metrics"]
            stats.timer.start("clip")
            ret = clip_domains(x_L, x_U, thresholds_, lA, lbias,
                               dm_lb=new_dm_lb, num_iters=clip_iters,
                               calculate_volume=clip_calculate_volume_metrics,
                               base_input_linear_constraints=clip_base_constraints,
                               base_input_constraints=clip_base_constraints_std,
                               domain_constraints=None,
                               rowpool=getattr(domains.history, 'rowpool', None))
            x_L, x_U = ret
            stats.timer.add("clip")

        stats.timer.start("decision")
        split_idx = input_split_branching(
            net, new_dm_lb, x_L, x_U, lA, thresholds_,
            branching_method, stats.storage_depth, num_iter=1,
            history=domains.history,
        )
        stats.timer.add("decision")

        num_domains_pre = len(domains)
        domains.add(new_dm_lb, x_L, x_U, cs_, thresholds_,
                    split_idx = None if use_reordered_bab else split_idx)
        print(f"{len(domains) - num_domains_pre} domains added, "
              f"{len(domains)} in total")
        print()

    print(f"{len(domains)} pre-split domains added out of {presplit_dm_l.shape[0]}")
    verified_ratio = 1 - len(domains) * 1. / presplit_dm_l.shape[0]
    print(f"Verified ratio: {verified_ratio}")
