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
import torch
import time
import copy
from torch import Tensor
from collections import defaultdict
import warnings

import arguments
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import select_rows, stop_criterion_batch_any
from crown_spec_batching import compute_bounds_in_spec_batches
from nlp_solver.continuation import retained, invoke
from input_split.tree_slice import atom_work_cap

from typing import TYPE_CHECKING, Optional, Tuple, Callable
from state.alpha import resolve_alpha_storage_dtype
from history import ensure_split_history
from input_split.utils import bound_thresholds
from input_constraints import merge_standard_constraints

if TYPE_CHECKING:
    from beta_CROWN_solver import LiRPANet


def _get_crown_spec_batch_size():
    try:
        return arguments.Config['solver']['crown']['spec_batch_size']
    except KeyError:
        try:
            return arguments.Config['solver']['crown']['batch_size']
        except KeyError:
            return 2048


def extract_relaxation_minimizer(
    lA: Tensor,
    lbias: Tensor,
    dm_l: Tensor,
    dm_u: Tensor,
    target_spec_idx: int = 0
) -> Tensor:
    """
    Extract the minimizer of the CROWN linear relaxation for primal warm-start.

    Given the linear relaxation from CROWN: f_relaxed(x) = lA @ x + lbias,
    subject to box constraints dm_l <= x <= dm_u, find the minimizer x*.

    For a linear function over a box, the minimum is achieved at a vertex:
    x*[i] = dm_u[i] if lA[i] > 0 else dm_l[i]

    This minimizer can be used to warm-start the PGD attack, as it lies in
    the "basin of attraction" of the global minimum.

    Args:
        lA: Linear coefficients from CROWN, shape (batch, m, d_in)
             where m = number of specs, d_in = input dimension
        lbias: Bias terms from CROWN, shape (batch, m)
        dm_l: Domain lower bounds, shape (batch, d_in) or (batch, 1, d_in)
        dm_u: Domain upper bounds, shape (batch, d_in) or (batch, 1, d_in)
        target_spec_idx: Which specification to minimize (default: 0, the objective)

    Returns:
        x_star: Tensor of shape (batch, d_in), the relaxation minimizer for each domain
    """
    if lA is None:
        return None

    batch_size = lA.shape[0]

    # Flatten dm_l and dm_u if they have extra dimensions
    dm_l_flat = dm_l.view(batch_size, -1)  # (batch, d_in)
    dm_u_flat = dm_u.view(batch_size, -1)  # (batch, d_in)
    d_in = dm_l_flat.shape[-1]

    # Extract target spec coefficients
    # lA shape: (batch, m, d_in) -> A_target shape: (batch, d_in)
    A_target = lA[:, target_spec_idx, :].view(batch_size, d_in)

    # Compute minimizer: for minimization, choose x_U when coeff > 0, x_L when coeff < 0
    # (since we want to minimize, positive coeffs should use lower x, negative should use higher x)
    # Wait - this depends on the sign convention. Let's be careful:
    # minimize lA @ x + lbias
    # For coefficient lA[i] > 0: minimize by using smallest x[i] = dm_l[i]
    # For coefficient lA[i] < 0: minimize by using largest x[i] = dm_u[i]
    x_star = torch.where(
        A_target > 0,
        dm_l_flat,
        dm_u_flat
    )

    return x_star


def set_instance_index(index):
    """Set the current instance index for LP sanity check CSV file naming."""
    global _current_instance_index
    _current_instance_index = index

def transfer(tensor, device=None, dtype=None, non_blocking=False):
    """Transfer a tensor to a specific device or dtype."""
    if device:
        tensor = tensor.to(device, non_blocking=non_blocking)
    if dtype:
        tensor = tensor.to(dtype)

    return tensor

@retained
def get_lower_bound_naive(
        self: 'LiRPANet',
        dm_lb: Optional[Tensor] = None,
        dm_l: Optional[Tensor] = None,
        dm_u: Optional[Tensor] = None,
        history = None,
        bounding_method: str = 'crown',
        C: Optional[Tensor] = None,
        stop_criterion=None,
        thresholds: Optional[Tensor] = None,
        reference_interm_bounds = None,
        return_A = False,
        return_b = False,
        constraints: Optional[tuple] = None,
        base_input_linear_constraints: Optional[tuple] = None,
        stats=None, _sliced=False, _native_only=False
) -> Tuple[Tensor, dict, Optional[Tensor], Optional[Tensor], Tensor]:
    """

    Bounds a batch of subdomains.

    :param self:                        Reference to the LiRPANet object
    :param dm_lb:                       The lower bound on the outputs of the domains
    :param dm_l:                        The lower bound on the inputs of the subdomains
    :param dm_u:                        The upper bound on the inputs of the subdomains
    :param history:                     Split history carrying the alphas, betas and
                                        alpha/beta usage flags for each subdomain.
                                        Defaults to an empty history.
    :param bounding_method:             crown, alpha-crown, etc.
    :param C:                           Output specification matrix
    :param stop_criterion:              Callable function which returns True for domains that have been verified
    :param thresholds:                  The specification threshold where dm_lb > thresholds implies the subdomain is verified
    :param reference_interm_bounds:     If given, these bounds are used as a reference, and the bounds
                                        for intermediate layers will still be computed (e.g., using CROWN,
                                        or other specified methods). The computed bounds will be
                                        compared to "reference_bounds" and the tighter one between the two
                                        will be used.
    :param return_A, return_b:          If true, the lA and lbias matrices of the final layer will be returned.
    :param constraints:                 A tuple of (constraints_A, constraints_b).
    :param stats:                       Statistics recorder.
    :return lb:                         The lower bound of the network output
    :return ret_alphas:                      Returns the updated alphas for each subdomain
    :return lA:                         If available, the bounding hyperplane coefficients
    :return lbias:                      If available, the bounding hyperplane offsets
    """

    native_started = time.perf_counter()
    history = ensure_split_history(history)
    deadline = getattr(stats.timer, 'deadline', None) if stats is not None else None
    rowpool = getattr(history, 'rowpool', None)
    addon = getattr(self, 'm1_addon', None)
    if (not _native_only and addon is not None and addon.sigma_main
            and addon.select_main() == 'sigma'):
        thr = bound_thresholds(rowpool, thresholds)
        lb = addon.bound_main(dm_l, dm_u, dm_lb, thr, deadline, history)
        def native(count):
            def take(value):
                if isinstance(value, tuple): return tuple(take(t) for t in value)
                return value[:count] if torch.is_tensor(value) and len(value) == len(dm_l) else value
            return get_lower_bound_naive(self, dm_lb=take(dm_lb), dm_l=dm_l[:count], dm_u=dm_u[:count],
                history=copy.copy(history), bounding_method=bounding_method, C=take(C),
                thresholds=thresholds[:count], stop_criterion=stop_criterion, return_A=True, return_b=True,
                constraints=take(constraints), base_input_linear_constraints=take(base_input_linear_constraints),
                stats=stats, _native_only=True)
        lb = addon.native_addon(native, lb, dm_lb, thr)
        point = getattr(history, 'engine_witness', None)
        x_star = None if point is None else point[:, :addon.engine.n].reshape_as(dm_l).to(dm_l)
        return lb, None, None, x_star
    if not _native_only and addon is not None and addon.sigma_main:
        history.engine_box = None  # the root probe or previous batch is out of scope
    if getattr(history, 'lifted_branch_score', None) is not None:
        history.lifted_branch_score = None     # belongs to the previous batch
        history.lifted_split_point = None
    clip_args = arguments.Config["bab"]["clip_n_verify"]
    enable_constrained_concretize = clip_args["clip_input_domain"]["enabled"] and (clip_args['clip_input_domain']['clip_type'] == "complete")
    rearrange_constraints = clip_args["rearrange_constraints"]
    try:
        solving_mode = arguments.Config["solving"]["solving_mode"]
    except KeyError:
        solving_mode = False

    if bounding_method == 'alpha-forward':
        raise ValueError("Should not use alpha-forward.")

    # Initialize the infeasible_bounds_constraints tensor
    if enable_constrained_concretize:
        batchsize = len(dm_l)
        device = dm_l.device
        self.net.init_infeasible_bounds_constraints(batchsize, device)

    auxiliary_cap = None
    if solving_mode and arguments.Config['solver']['nlp'].get('aux_reference_bounds', False):
        registry = getattr(history, 'certificate_registry', None)
        if registry is not None:
            from nlp_solver.domain_certificate import merge_reference_bounds
            example = ((dm_l[:1] + dm_u[:1]) / 2).detach()
            mapped, auxiliary_cap = registry.graph_references(self.net, example, dm_l, dm_u, thresholds[:, 0])
            reference_interm_bounds = merge_reference_bounds(reference_interm_bounds, mapped)

    presolve_cuts = None
    if (solving_mode and rowpool is not None and C is not None and thresholds is not None
            and self.net.bound_opts.get('square_row_to_abs', False)):
        from auto_LiRPA.presolve import square_row_cuts
        probe = BoundedTensor(dm_l, PerturbationLpNorm(x_L=dm_l, x_U=dm_u))
        presolve_cuts = square_row_cuts(
            self.net, C, thresholds, probe, rowpool.tolerance, nonlinear_only=True)
        if presolve_cuts is not None:
            # Non-affine h needs fresh planes for +/-h on every domain, just
            # as appended nonlinear rows would. Affine strips are already base
            # rows. These thresholds include tolerance exactly once.
            constraints = merge_standard_constraints(
                constraints, (presolve_cuts[0], -presolve_cuts[1]))

    if (solving_mode and rowpool is not None
            and arguments.Config['solver']['nlp'].get('input_cutoff_cuts', False)):
        derived = rowpool.derived_cut_view(dm_l, dm_u, thresholds[:, 0])
        if derived is not None:
            constraints = merge_standard_constraints(constraints, (derived[0], -derived[1]))

    ptb = PerturbationLpNorm(x_L=dm_l, x_U=dm_u,
        constraints=constraints, rearrange_constraints=rearrange_constraints, no_return_inf=enable_constrained_concretize,
        timer=stats.timer, discard_abnormal=solving_mode)
    new_x = BoundedTensor(dm_l, ptb)  # the value of new_x doesn't matter, only ptb matters
    # set alpha here again
    self.net.set_alpha(history.alphas, set_all=True)
    self.net.set_bound_opts({'optimize_bound_args': {
        'enable_beta_crown': history.use_beta,
        'enable_alpha_crown': history.use_alpha,
        'fix_interm_bounds': True,
        'iteration': arguments.Config['solver']['alpha-crown']['input_split_alpha_iteration'],
        'lr_alpha': arguments.Config['solver']['alpha-crown']['input_split_lr_alpha'],
        'stop_criterion_func': stop_criterion(thresholds),
    }})
    if history.use_beta:
        # Simplified output beta: bound target spec using others as Lagrangian relaxations
        # Beta shape is (batch, m) instead of (batch, m, m)
        # We still propagate all m specs, but the beta modification in backward_bound
        # effectively computes the Lagrangian-relaxed bound for the target spec
        target_idx = 0  # We bound spec[0] by default
        splits_per_example = self.net.set_output_beta(
            batch=len(dm_l), spec=C, bias=bound_thresholds(rowpool, thresholds), warm_betas=history.betas,
            target_idx=target_idx)
        self.set_crown_bound_opts('beta')
    if return_A:
        needed_A_dict = defaultdict(set)
        needed_A_dict[self.net.output_name[0]].add(self.net.input_name[0])
    else:
        needed_A_dict = None

    def _get_lA(ret):
        if return_A:
            A_dict = ret[-1]
            lA = A_dict[self.net.output_name[0]][self.net.input_name[0]]['lA']
            if lA is not None:
                lA = lA.detach()
        else:
            lA = None
        return lA

    def _get_lbias(ret):
        if return_b:
            A_dict = ret[-1]
            lbias = A_dict[self.net.output_name[0]][self.net.input_name[0]]['lbias']
            if lbias is not None:
                lbias = lbias.detach()
        else:
            lbias = None
        return lbias

    if bounding_method == "alpha-crown" and not history.use_beta:
        ret = self.net.compute_bounds(
                x=(new_x,), C=C, method='CROWN-Optimized', bound_upper=False,
                return_A=return_A, needed_A_dict=needed_A_dict,
                reference_bounds=reference_interm_bounds)
        lb = ret[0]
        lA = _get_lA(ret)
        lbias = _get_lbias(ret)
    elif bounding_method == 'ibp':
        lb = self.net.compute_bounds(x=(new_x,), C=C, method='ibp')[0]
        lA = None
        lbias = None
    else:
        if arguments.Config['bab']['branching']['input_split']['ibp_enhancement']:
            lb, lA, lbias = yield from invoke(get_lower_bound_with_ibp_enhancement,
                self, new_x, C, bounding_method,
                thresholds, return_A=return_A, needed_A_dict=needed_A_dict,
                stop_criterion=stop_criterion, constraints=constraints, timer=stats.timer,
                rearrange_constraints=rearrange_constraints, no_return_inf=enable_constrained_concretize,
                rowpool=rowpool, external_reference_bounds=reference_interm_bounds,
                sliced=_sliced, **({'_sliced': True} if _sliced else {}))
            ret = None
        else:
            with torch.no_grad():
                ret = yield from compute_bounds_in_spec_batches(
                    self.net,
                    spec_batch_size=_get_crown_spec_batch_size(), oom_retry=True,
                    _sliced=_sliced, _iterate=True,
                    _work_cap=atom_work_cap,
                    x=(new_x,), C=C, method=bounding_method,
                    bound_upper=False, return_A=return_A, needed_A_dict=needed_A_dict,
                    reuse_alpha=bounding_method.lower() == 'crown' and len(history.alphas) > 0,
                    reference_bounds=reference_interm_bounds
                )
            ret = _optimize_objective_row(self, new_x, C, thresholds, stop_criterion, ret, needed_A_dict)
        if history.use_beta:
            interm_bounds = get_intermediate_bounds(self.net.nodes(), dm_l.device)
            ptb_wo_constraint = PerturbationLpNorm(x_L=dm_l, x_U=dm_u, constraints=None)
            x_wo_constraint = BoundedTensor(dm_l, ptb_wo_constraint)
            # # For beta pass: use iteration=1 since we only need a single CROWN pass with beta adjustment.
            # # Alpha is already optimized from the first pass, intermediate bounds are fixed.
            # orig_iteration = self.net.bound_opts['optimize_bound_args']['iteration']
            # # self.net.bound_opts['optimize_bound_args']['iteration'] = 1
            # # Use C_target (single spec) for beta-enabled compute_bounds - m× speedup
            ret = self.net.compute_bounds(
                    x=(x_wo_constraint, ), C=C, method='CROWN-Optimized', bound_upper=False,
                    interm_bounds=interm_bounds, return_A=return_A, needed_A_dict=needed_A_dict)

        if ret is not None:
            with torch.no_grad():
                lb = ret[0]
                lA = _get_lA(ret)
                lbias = _get_lbias(ret)
                # For infeasible batches, CROWN will always return naive results out of computational stability.
                # But BoundedModule instance will keep track of these batches with "infeasible_bounds_constraints"
                infeasible_batch = self.net.infeasible_bounds_constraints
                if infeasible_batch is not None:
                    lb[infeasible_batch] = torch.inf

    if auxiliary_cap is not None:
        lb[:, 0] = torch.minimum(lb[:, 0], auxiliary_cap.to(lb))

    if solving_mode and arguments.Config['solver']['nlp'].get('lp_certificate_transfer', False):
        registry = getattr(history, 'certificate_registry', None)
        if registry is not None:
            transferred, history.certificate_scope_refs = registry.bound(dm_l, dm_u, thresholds[:, 0])
            lb[:, 0] = torch.maximum(lb[:, 0], transferred.to(lb))

    # dm_lb is not None if compare_with_old_bounds is True, 
    if dm_lb is not None:
        lb = torch.max(lb, dm_lb)

    if not solving_mode:
        worst_idx = (lb - thresholds).amax(dim=-1).argmin()
        print('Worst bound:', (lb - thresholds)[worst_idx])

    with torch.no_grad():
        alphas = None
        if bounding_method == 'alpha-crown':
            alpha_storage_dtype = resolve_alpha_storage_dtype(
                arguments.Config["solver"]["alpha-crown"]["alpha_dtype"],
                reproduce_alpha_bounds=arguments.Config[
                    "solver"]["reproduce_alpha_bounds"],
            )
            alphas = self.net.get_alpha(
                get_all=not history.use_beta,
                half=alpha_storage_dtype is torch.float16)
        betas = self.net.get_beta(splits_per_example, device='cpu') if history.use_beta else None
        history.update_bounds_optimizers(alphas, betas)

    # Compute x* (relaxation minimizer) for warm-starting PGD attacks
    # This is done here so that x* has the same batch size as the domains,
    # and can be correctly indexed when a subset of domains is selected for attack
    x_star = None
    input_split_config = arguments.Config["bab"]["branching"]["input_split"]
    if input_split_config.get("use_relaxation_warmstart", False):
        x_star = extract_relaxation_minimizer(lA, lbias, dm_l, dm_u, target_spec_idx=0)

    # --- Dual Ascent Lower Bound Tightening ---
    # When there are AND-constraints (m > 1) and lA/lbias are available,
    # tighten the CROWN lower bound on the objective spec by optimizing λ ≥ 0.
    # Uses linearity of CROWN in the spec matrix C (with fixed α/intermediate bounds).
    # d(λ*) = lb_crown(C_obj + λ*ᵀ C_constr) - λ*ᵀ rhs_constr is a valid dual LB.
    # `solver.nlp` is an optional feature block, so tolerate its absence here.
    nlp_da_cfg = arguments.Config["solver"].get("nlp", {})
    record_planes = (not _native_only and addon is not None and addon.sigma_main)
    plane_value = lb[:, 0].clone() if record_planes else None
    plane_exact = lb[:, 0].clone() if record_planes else None
    diagnostic_seconds = 0.
    if (nlp_da_cfg.get("dual_ascent_enabled", False)
            and lA is not None
            and lbias is not None
            and C is not None
            and (C.shape[1] > 1 or base_input_linear_constraints is not None
                 or getattr(history, 'qtb', None) is not None)):
        try:
            from nlp_solver.dual_ascent import DualAscentConfig, dual_ascent_tighten, lp_bound
            da_cfg = DualAscentConfig(
                num_steps=int(nlp_da_cfg.get("dual_ascent_steps", 20)),
                lr=float(nlp_da_cfg.get("dual_ascent_lr", 0.05)),
                verbose=bool(nlp_da_cfg.get("dual_ascent_verbose", False)),
            )
            B_da = lA.shape[0]
            d_in = lA.shape[2]
            # Only run dual ascent on batches where lA/lbias are finite.
            # Infeasible domains have lA=inf/nan (set by infeasible_bounds_constraints),
            # which would poison the Adam optimizer and produce NaN lb_dual.
            finite_mask = (
                torch.isfinite(lA).all(dim=(1, 2))   # (B,)
                & torch.isfinite(lbias).all(dim=1)   # (B,)
            )
            if finite_mask.any():
                dm_l_flat = dm_l.reshape(B_da, d_in)
                dm_u_flat = dm_u.reshape(B_da, d_in)
                n_finite = int(finite_mask.sum().item())
                n_output_constr_da = C.shape[1] - 1  # number of output constraint specs
                n_input_constr_da = 0
                input_constr_A = None
                input_constr_b = None
                if base_input_linear_constraints is not None:
                    input_constr_A, input_constr_b = base_input_linear_constraints
                    input_constr_A = input_constr_A.to(device=lA.device, dtype=lA.dtype)
                    input_constr_b = input_constr_b.to(device=lA.device, dtype=lA.dtype)
                    n_input_constr_da = input_constr_A.shape[1]
                if presolve_cuts is not None:
                    cut_A, cut_b = presolve_cuts
                    input_constr_A = (cut_A if input_constr_A is None else
                                      torch.cat((input_constr_A, cut_A), dim=1))
                    input_constr_b = (cut_b if input_constr_b is None else
                                      torch.cat((input_constr_b, cut_b), dim=1))
                    n_input_constr_da = input_constr_A.shape[1]
                n_constr_da = n_output_constr_da + n_input_constr_da

                # Warm-start lambda from ALM mu if available and shape-compatible.
                # ALM mu_constr and dual-ascent lambda are Lagrange multipliers for
                # the same constraints, so using ALM mu as initialization can
                # reduce the number of Adam steps needed to reach the dual optimum.
                lambda_init_da = None
                if (nlp_da_cfg.get("dual_ascent_lam_warmstart", False)
                        and history is not None
                        and getattr(history, 'alm_mean_mu', None) is not None
                        and history.alm_mean_mu.shape == (n_constr_da,)):
                    lambda_init_da = (
                        history.alm_mean_mu
                        .to(device=lA.device, dtype=lA.dtype)
                        .unsqueeze(0)
                        .expand(n_finite, -1)
                    )

                # thresholds shape: (B, m) — contains rhs for each AND spec
                # constraint rows relaxed by the pool like every other cut
                rhs_da = bound_thresholds(rowpool, thresholds).reshape(B_da, -1)[finite_mask].clone()
                lp_stop = torch.where(torch.isfinite(rhs_da[:, 0]), rhs_da[:, 0],
                                      torch.full_like(rhs_da[:, 0], torch.inf))
                # dual_ascent_tighten returns margins d(lambda) - rhs_obj. Give it a finite
                # objective RHS (0 when there is no incumbent yet) and add it back below,
                # so lb_dual_finite is in the objective's own units, like `lb`.
                obj_shift_da = torch.where(torch.isfinite(rhs_da[:, 0]), rhs_da[:, 0],
                                           torch.zeros_like(rhs_da[:, 0]))
                rhs_da[:, 0] = obj_shift_da
                _, lb_dual_finite, lam_star = dual_ascent_tighten(
                    lA=lA[finite_mask],
                    lbias=lbias[finite_mask],
                    dm_l=dm_l_flat[finite_mask],
                    dm_u=dm_u_flat[finite_mask],
                    rhs=rhs_da,
                    cfg=da_cfg,
                    lambda_init=lambda_init_da,
                    input_constr_A=None if input_constr_A is None else input_constr_A[finite_mask],
                    input_constr_b=None if input_constr_b is None else input_constr_b[finite_mask],
                    deadline=deadline,
                )
                lb_dual_finite = lb_dual_finite + obj_shift_da

                # The same Lagrangian bound, solved as the LP it is: Adam with a fixed
                # step stays far from the LP optimum (pooling_foulds2tp root: -9400 vs
                # the LP's -1100 over the very same rows). Any y >= 0 PDHG returns is a
                # valid bound, so the two are combined with max.
                pdhg_iters = int(nlp_da_cfg.get("dual_lp_pdhg_iters", 0))
                if pdhg_iters > 0:
                    lA_f, lbias_f = lA[finite_mask], lbias[finite_mask]
                    G_lp = lA_f[:, 1:, :]
                    h_lp = rhs_da[:, 1:] - lbias_f[:, 1:]
                    if input_constr_A is not None:
                        G_lp = torch.cat([G_lp, input_constr_A[finite_mask]], dim=1)
                        h_lp = torch.cat([h_lp, input_constr_b[finite_mask]], dim=1)
                    y0_lp = getattr(history, 'lp_mean_y', None) if history is not None else None
                    if y0_lp is not None and y0_lp.shape == (G_lp.shape[1],):
                        y0_lp = y0_lp.to(G_lp).unsqueeze(0).expand(G_lp.shape[0], -1)
                    else:
                        y0_lp = None
                    exact_max = int(nlp_da_cfg.get("dual_lp_exact_max", 0))
                    halpern_lp = bool(nlp_da_cfg.get("dual_lp_halpern", False))
                    lb_lp, y_lp = lp_bound(
                        lA_f[:, 0, :], lbias_f[:, 0], G_lp, h_lp,
                        dm_l_flat[finite_mask], dm_u_flat[finite_mask],
                        pdhg_iters=pdhg_iters, y0=y0_lp, exact_max=exact_max, deadline=deadline,
                        halpern=halpern_lp, stop=lp_stop if halpern_lp else None)
                    if record_planes:
                        # Diagnostic replay only: it never enters a certificate,
                        # a warm start, or the production plane/exact result.
                        tick = time.perf_counter()
                        plain = (lp_bound(lA_f[:, 0, :], lbias_f[:, 0], G_lp, h_lp,
                            dm_l_flat[finite_mask], dm_u_flat[finite_mask], pdhg_iters=pdhg_iters,
                            y0=y0_lp, exact_max=0, deadline=deadline)[0] if exact_max > 0 and not halpern_lp else lb_lp)
                        diagnostic_seconds += time.perf_counter()-tick
                        plane_value[finite_mask] = torch.fmax(plane_value[finite_mask], plain)
                        plane_exact[finite_mask] = torch.fmax(plane_exact[finite_mask], lb_lp)
                    lb_dual_finite = torch.max(lb_dual_finite, lb_lp)
                    if history is not None and y_lp.numel() > 0:
                        history.lp_mean_y = y_lp.mean(0).detach().cpu()
                    qtb = getattr(history, 'qtb', None) if history is not None else None
                    if qtb is not None:
                        # Quadratic objective: its tangent plane at the incumbent is a
                        # global lower plane, tight where CROWN's single plane is not.
                        x_inc = incumbent_for_tangent(
                            getattr(history, 'alm_best_primal_x', None), qtb)
                        if x_inc is not None and not torch.equal(x_inc, qtb.x_hat):
                            qtb.offer(x_inc, margin=max(
                                float(nlp_da_cfg.get("bab_primal_dual_abs_tol", 1e-5)),
                                float(nlp_da_cfg.get("bab_primal_dual_rel_tol", 1e-4))
                                * abs(qtb.f)))
                        plane = None
                        if nlp_da_cfg.get("qtb_crown_rewrite", False):
                            # Native CROWN on the objective's eigenbasis rewrite: QTB's plane on a box
                            # holding the anchor, and per-direction tangents/chords elsewhere.
                            from nlp_solver.crown_rewrite import quadratic_rewrite_plane
                            plane = quadratic_rewrite_plane(
                                history, qtb, dm_l_flat[finite_mask], dm_u_flat[finite_mask],
                                int(nlp_da_cfg.get("qtb_crown_rewrite_max_vars", 2000)))
                        a_tb, b_tb = plane if plane is not None else qtb.plane(
                            dm_l_flat[finite_mask], dm_u_flat[finite_mask])
                        y0_tb = getattr(history, 'qtb_mean_y', None)
                        if y0_tb is not None and y0_tb.shape == (G_lp.shape[1],):
                            y0_tb = y0_tb.to(G_lp).unsqueeze(0).expand(G_lp.shape[0], -1)
                        else:
                            y0_tb = None
                        lb_tb, y_tb = lp_bound(
                            a_tb, b_tb, G_lp, h_lp,
                            dm_l_flat[finite_mask], dm_u_flat[finite_mask],
                            pdhg_iters=pdhg_iters, y0=y0_tb, exact_max=exact_max, deadline=deadline,
                            halpern=halpern_lp, stop=lp_stop if halpern_lp else None)
                        if record_planes:
                            tick = time.perf_counter()
                            plain = (lp_bound(a_tb, b_tb, G_lp, h_lp,
                                dm_l_flat[finite_mask], dm_u_flat[finite_mask], pdhg_iters=pdhg_iters,
                                y0=y0_tb, exact_max=0, deadline=deadline)[0] if exact_max > 0 and not halpern_lp else lb_tb)
                            diagnostic_seconds += time.perf_counter()-tick
                            plane_value[finite_mask] = torch.fmax(plane_value[finite_mask], plain)
                            plane_exact[finite_mask] = torch.fmax(plane_exact[finite_mask], lb_tb)
                        lb_dual_finite = torch.max(lb_dual_finite, lb_tb)
                        if nlp_da_cfg.get("qtb_quadratic_dual", False):
                            lb_quadratic = qtb.quadratic_dual_bound(
                                dm_l_flat[finite_mask], dm_u_flat[finite_mask],
                                G_lp, h_lp, y_tb)
                            if lb_quadratic is not None:
                                lb_dual_finite = torch.max(lb_dual_finite, lb_quadratic)
                        if y_tb.numel() > 0:
                            history.qtb_mean_y = y_tb.mean(0).detach().cpu()
                    if getattr(history, 'lifted', None) is not None:
                        # Quadratic rows lifted with both McCormick planes of every
                        # product (nlp_solver/lifted_lp.py).
                        from input_split.lifted_bound import lifted_node_bound
                        lf = lifted_node_bound(
                            history, dm_l_flat[finite_mask], dm_u_flat[finite_mask],
                            lA_f, lbias_f, rhs_da[:, 1:],
                            None if input_constr_A is None else input_constr_A[finite_mask],
                            None if input_constr_b is None else input_constr_b[finite_mask],
                            lb_dual_finite, nlp_da_cfg, deadline=deadline)
                        history.lifted_branch_score = None
                        history.lifted_split_point = None
                        if lf is not None:
                            if lf[1] is not None:
                                # Branch on the lifted relaxation only where it is the
                                # one that sets the domain's bound.
                                sc = torch.where((lf[0] >= lb_dual_finite).unsqueeze(-1),
                                                 lf[1], torch.full_like(lf[1], torch.nan))
                                score = torch.full((B_da, d_in), torch.nan,
                                                   dtype=lf[1].dtype, device=lf[1].device)
                                score[finite_mask] = sc
                                history.lifted_branch_score = score
                                pts = torch.full_like(score, torch.nan)
                                pts[finite_mask] = lf[2].to(pts)
                                history.lifted_split_point = pts
                            lb_dual_finite = torch.max(lb_dual_finite, lf[0])

                # Update alm_mean_mu with the dual ascent solution for future warm-starts.
                if (nlp_da_cfg.get("dual_ascent_lam_warmstart", False)
                        and history is not None
                        and lam_star.numel() > 0):
                    history.alm_mean_mu = lam_star.mean(0).detach().cpu()
                # Store dual_lam_star for bidirectional warm-start (dual → ALM μ)
                if history is not None and lam_star.numel() > 0:
                    history.dual_lam_star = lam_star.mean(0).detach().cpu()

                # Compute dual-guided relaxation minimizer x* from lA_eff(λ*)
                # The effective linear bound lA_eff accounts for both objective and
                # constraint structure, directing ALM to start near points that
                # minimize the Lagrangian relaxation.
                if history is not None and lam_star.numel() > 0 and lA is not None:
                    with torch.no_grad():
                        lA_obj_f = lA[finite_mask, 0, :]        # (F, d_in)
                        lA_constr_f = lA[finite_mask, 1:, :]    # (F, n_constr, d_in)
                        lam_out = lam_star[:, :n_output_constr_da]
                        lam_in = lam_star[:, n_output_constr_da:]
                        lA_eff_star = lA_obj_f
                        if n_output_constr_da > 0:
                            lA_eff_star = lA_eff_star + (lam_out.unsqueeze(-1) * lA_constr_f).sum(1)
                        if n_input_constr_da > 0:
                            lA_eff_star = lA_eff_star + (
                                lam_in.unsqueeze(-1) * input_constr_A[finite_mask]
                            ).sum(1)
                        # Minimizer of linear function over box:
                        # For lA_eff[i] > 0: min at dm_l[i]; for lA_eff[i] < 0: min at dm_u[i]
                        x_star_dual_f = torch.where(
                            lA_eff_star > 0,
                            dm_l_flat[finite_mask],
                            dm_u_flat[finite_mask],
                        )
                        # Store full-batch version (non-finite entries get zeros)
                        x_star_dual = torch.zeros(B_da, d_in, device=lA.device, dtype=lA.dtype)
                        x_star_dual[finite_mask] = x_star_dual_f
                        history.dual_x_star = x_star_dual.detach().cpu()

                # lb_dual_finite: (F,) — d(λ*), a valid lower bound for the CONSTRAINED
                # minimum of the objective in the domain (weak duality), in the same
                # units as lb[:, 0] (the raw objective bound, compared to rhs[:, 0]).
                obj_thr = thresholds.reshape(B_da, -1)[:, 0]
                # Half the certification tolerance of is_primal_dual_converged.
                near_tol = 0.5 * torch.maximum(
                    torch.full_like(obj_thr, float(nlp_da_cfg.get("bab_primal_dual_abs_tol", 1e-5))),
                    float(nlp_da_cfg.get("bab_primal_dual_rel_tol", 1e-4))
                    * obj_thr.abs().clamp(min=1.0, max=1e300))
                if nlp_da_cfg.get("dual_bound_record_all", False):
                    # Every domain keeps its tightest valid bound (best-first on it).
                    obj_thr = torch.full_like(obj_thr, float("-inf"))
                    near_tol = torch.zeros_like(obj_thr)
                n_pruned = _merge_dual_ascent_bound(
                    lb, finite_mask, lb_dual_finite, obj_thr, solving_mode,
                    near_tol=near_tol[finite_mask])
                if solving_mode and da_cfg.verbose:
                    print(f"  [DualAscent] Gap pruning: {n_pruned}/{n_finite} domains "
                          f"(lb_dual >= rhs_obj=primal_solution)")
        except Exception as e:  # pylint: disable=broad-except
            # Dual ascent is optional; never let it break the main bounding loop
            import warnings
            warnings.warn(f"[DualAscent] Skipped due to error: {e}")

    alias_floor = getattr(self.net, 'objective_alias_floor_value', None)
    if solving_mode and alias_floor is not None:
        # A global bound proved once at the root (input_bab_parallel); the rows and their
        # thresholds are the same for every domain. Re-proving it per domain looped in
        # Python over the whole batch on every call (1M domains on the GPU).
        improved = lb[:, 0] < alias_floor
        if improved.any():
            lb = lb.clone()
            lb[improved, 0] = alias_floor
            if lA is not None and lbias is not None:
                lA = lA.clone()
                lbias = lbias.clone()
                lA[improved, 0] = 0
                lbias[improved, 0] = alias_floor

    if auxiliary_cap is not None:
        lb[:, 0] = torch.minimum(lb[:, 0], auxiliary_cap.to(lb))
    addon = getattr(self, 'm1_addon', None)
    if solving_mode and addon is not None and not _native_only:
        lb = addon.bound(dm_l, dm_u, lb, bound_thresholds(rowpool, thresholds), deadline, inherited=dm_lb,
                         native_seconds=time.perf_counter()-native_started-diagnostic_seconds, history=history,
                         **({'planes': (plane_value, plane_exact)} if record_planes else {}))
    return lb, lA, lbias, x_star

def incumbent_for_tangent(x_inc, qtb):
    """The incumbent on the tangent's device and dtype, or None if it does not fit.

    history.alm_best_primal_x is kept on the host, while the tangent evaluates the
    objective with the model on the GPU. Offering the host tensor raised a device error
    inside the dual-bound block, whose broad except then skipped dual ascent, the LP
    and the tangent bound on every later BaB iteration (cute_hs106: from iteration 5).
    """
    if x_inc is None or x_inc.numel() != qtb.n:
        return None
    return x_inc.reshape(-1).to(qtb.x_hat)


def _merge_dual_ascent_bound(lb, finite_mask, lb_dual_finite, obj_threshold, solving_mode,
                             near_tol=None):
    """Fold the dual-ascent objective bound `lb_dual_finite` (raw units) into `lb[:, 0]`.

    `lb[:, 0]` is the raw objective bound that the filter compares with the raw
    incumbent, so the constrained bound must enter in the same units. In solving mode a
    domain is raised when its constrained bound reaches the incumbent `obj_threshold`
    (+inf before any incumbent, so nothing is raised) less `near_tol`: the domain that
    holds the optimum can never reach the incumbent itself, and unless its bound is
    recorded the reported dual cannot close the gap. Other domains keep their bound,
    since raising it reorders the queue away from the optimum. `near_tol` (B,) or scalar
    defaults to 0. Returns the number of domains raised.
    """
    if not solving_mode:
        lb[finite_mask, 0] = torch.max(lb[finite_mask, 0], lb_dual_finite)
        return int(finite_mask.sum())
    finite_indices = torch.where(finite_mask)[0]
    target = obj_threshold.to(lb_dual_finite)[finite_mask]
    if near_tol is not None:
        target = target - torch.as_tensor(near_tol).to(lb_dual_finite)
    prune_in_finite = lb_dual_finite >= target
    if prune_in_finite.any():
        prune_indices = finite_indices[prune_in_finite]
        lb[prune_indices, 0] = torch.max(lb[prune_indices, 0],
                                         lb_dual_finite[prune_in_finite])
    return int(prune_in_finite.sum())


def get_intermediate_bounds(layers, device):
    interm_bounds = {}
    for layer in layers:
        if layer.lower is None or layer.upper is None:
            continue
        interm_bounds[layer.name] = [
                transfer(layer.lower.detach(), device),
                transfer(layer.upper.detach(), device)
            ]
    return interm_bounds

def _optimize_objective_row(self, x, C, thresholds, stop_criterion, ret, needed_A_dict=None):
    """Re-bound the objective row (spec 0) of a CROWN result `ret` with alpha-CROWN.

    Solving mode only; the other rows keep CROWN. The CROWN pass that produced `ret` has
    fixed every intermediate bound, so this only optimizes the relaxations along the
    objective's own backward path, for a single spec, and covers every optimizable
    activation on it at once. On KS50 it lifts the mean domain bound by ~0.09: sqr gives
    most of it (its default lower tangent is the line 0 on an interval straddling zero),
    tanh the rest.
    """
    try:
        iteration = arguments.Config['solver']['alpha-crown']['input_split_objective_iteration']
        solving_mode = arguments.Config['solving']['solving_mode']
    except KeyError:    # partial configs, e.g. in unit tests
        return ret
    if iteration <= 0 or not solving_mode:
        return ret
    opts = self.net.bound_opts['optimize_bound_args']
    saved = {k: opts[k] for k in ('enable_alpha_crown', 'init_alpha', 'iteration', 'stop_criterion_func')}
    self.net.set_bound_opts({'optimize_bound_args': {
        'enable_alpha_crown': True, 'init_alpha': False, 'iteration': iteration,
        'stop_criterion_func': stop_criterion(thresholds[:, :1])}})
    return_A = needed_A_dict is not None
    interm_bounds = get_intermediate_bounds(self.net.nodes(), x.device)
    x.ptb.use_grad = True   # the optimizer differentiates through constrained concretization
    try:
        # Fresh alphas for the objective row only; BaB otherwise keeps no alphas for it.
        self.net.init_alpha((x,), c=C[:, :1], bound_upper=False,
                            interm_bounds=interm_bounds, final_layer_alpha_only=True)
        ret_obj = self.net.compute_bounds(
            x=(x,), C=C[:, :1], method='CROWN-Optimized', bound_upper=False,
            interm_bounds=interm_bounds, return_A=return_A, needed_A_dict=needed_A_dict)
    finally:
        x.ptb.use_grad = False
        self.net.set_bound_opts({'optimize_bound_args': saved})
    with torch.no_grad():
        lb = ret[0].clone()
        improved = ret_obj[0][:, 0] > lb[:, 0]
        lb[:, 0] = torch.where(improved, ret_obj[0][:, 0], lb[:, 0])
        if not return_A:
            return (lb, *ret[1:])
        out, inp = self.net.output_name[0], self.net.input_name[0]
        entry, entry_obj = ret[-1][out][inp], ret_obj[-1][out][inp]
        lA, lbias = entry['lA'].clone(), entry['lbias'].clone()
        lA[:, 0] = torch.where(improved.view(-1, *[1] * (lA.dim() - 2)),
                               entry_obj['lA'].reshape(lA[:, 0].shape), lA[:, 0])
        lbias[:, 0] = torch.where(improved, entry_obj['lbias'].reshape(-1), lbias[:, 0])
        A_dict = {**ret[-1], out: {**ret[-1][out], inp: {**entry, 'lA': lA, 'lbias': lbias}}}
    return (lb, *ret[1:-1], A_dict)


@retained
def get_lower_bound_with_ibp_enhancement(
        self: 'LiRPANet', new_x: Tensor, C: Tensor, bounding_method: str, thresholds: Tensor,
        return_A:bool=False, needed_A_dict:Optional[dict]=None,
        stop_criterion: Callable=stop_criterion_batch_any,
        constraints:tuple=None, timer=None,
        rearrange_constraints:bool=False, no_return_inf:bool=False, rowpool=None,
        external_reference_bounds=None, _sliced=False
)->Tuple[Tensor, Optional[Tensor], Optional[Tensor], Tensor]:
    """

    Bounds a batch of domains using interval bound propagation (IBP). If a subset of domains could not be verified
    using IBP, their lower bounds are refined using a specified bounding method e.g. crown.

    :param self:                Reference to the LiRPANet object
    :param new_x:               Updated input box for each subdomain
    :param C:                   Output specification matrix
    :param bounding_method:     crown, alpha-crown, etc.
    :param thresholds:          The specification threshold where dom_lb > thresholds implies the subdomain is verified
    :param return_A:            If True, must return parameters specified in 'needed_A_dict'
    :param needed_A_dict:       Nodes whose CROWN lA and lbias parameters should be returned
    :param stop_criterion:      Callable function which returns True for domains that have been verified
    :param constraints:         A tuple of (constraints_A, constraints_b)
    :param timer:               A time recorder.
    :param rearrange_constraints:
                                Whether to rearrange the constraints based on their distances to input region centroids.
    :param no_return_inf:       Whether inf value is allowed in intermediate bound.
    :param rowpool:             The instance's RowPool (solving mode), None without one.

    :return lb:                 The lower bound of the network output, a mixture of bounds from IBP and 'bounding_method'
    :return lA:                 CROWN hyperplane coefficients mapping the input to the network output
    :return lbias:              CROWN hyperplane offsets mapping the input to the network output
    :return lb_crown:           The lower bound of the network output, solely from 'bounding_method', not IBP
    """
    # Holds the intermediate bounds achieved using IBP to record as reference. When 'bounding_method' is subsequently
    # invoked on the remaining unverified domains, the tighter intermediate bounds between IBP and 'bounding_method' are used.
    reference_interm_bounds = {}
    # get the lower bound using interval bound propagation (IBP)
    try:
        lb_ibp = self.net.compute_bounds(
            x=(new_x,), C=C, method='ibp', bound_upper=False, return_A=False)[0]
    except Exception as exc:  # pylint: disable=broad-except
        warnings.warn(
            f"[IBP enhancement] Falling back to {bounding_method} because IBP failed: {exc}"
        )
        clip_args = arguments.Config["bab"]["clip_n_verify"]
        enable_constrained_concretize = (
            clip_args["clip_input_domain"]["enabled"]
            and clip_args["clip_input_domain"]["clip_type"] == "complete"
        )
        if enable_constrained_concretize:
            self.net.init_infeasible_bounds_constraints(new_x.shape[0], new_x.ptb.x_L.device)
        ret = yield from compute_bounds_in_spec_batches(
            self.net,
            spec_batch_size=_get_crown_spec_batch_size(), oom_retry=True,
            _sliced=_sliced, _iterate=True, _work_cap=atom_work_cap,
            x=(new_x,), C=C, method=bounding_method,
            bound_upper=False, return_A=return_A, needed_A_dict=needed_A_dict,
            reference_bounds=external_reference_bounds,
        )
        ret = _optimize_objective_row(self, new_x, C, thresholds, stop_criterion, ret, needed_A_dict)
        lb = ret[0]
        infeasible_batch = getattr(self.net, "infeasible_bounds_constraints", None)
        if infeasible_batch is not None:
            lb[infeasible_batch] = torch.inf
        if return_A:
            A_dict = ret[-1]
            lA = A_dict[self.net.output_name[0]][self.net.input_name[0]]['lA']
            lbias = A_dict[self.net.output_name[0]][self.net.input_name[0]]['lbias']
        else:
            lA = None
            lbias = None
        return lb, lA, lbias
    lb = lb_ibp.clone()
    for node in self.net.nodes():
        if (node.perturbed
            and isinstance(node.lower, torch.Tensor)
            and isinstance(node.upper, torch.Tensor)):
            reference_interm_bounds[node.name] = (node.lower, node.upper)
    if external_reference_bounds:
        from nlp_solver.domain_certificate import merge_reference_bounds
        reference_interm_bounds = merge_reference_bounds(reference_interm_bounds, external_reference_bounds)
    # The domain filter accepts constraint rows up to the pool's relaxed thresholds; IBP must
    # not settle (and skip CROWN for) a domain the filter would still keep.
    unverified = torch.logical_not(
        stop_criterion(bound_thresholds(rowpool, thresholds))(lb_ibp).any(dim=-1))
    if not unverified.any():
        # The code below assumes that there is at least one unverified domain
        unverified[0] = True
    unverified_constraints = None
    if constraints is not None:
        constraints_A, constraints_b = constraints
        unverified_A = select_rows(constraints_A, unverified)
        unverified_b = select_rows(constraints_b, unverified)
        unverified_constraints = (unverified_A, unverified_b)
    try:
        solving_mode = arguments.Config["solving"]["solving_mode"]
    except KeyError:
        solving_mode = False
    x_unverified = BoundedTensor(
        select_rows(new_x, unverified),
        ptb=PerturbationLpNorm(
            x_L=select_rows(new_x.ptb.x_L, unverified),
            x_U=select_rows(new_x.ptb.x_U, unverified),
            constraints=unverified_constraints,
            timer=timer,
            rearrange_constraints=rearrange_constraints,
            no_return_inf=no_return_inf,
            discard_abnormal=solving_mode))
    # Initialize the infeasible_bounds_constraints tensor
    clip_args = arguments.Config["bab"]["clip_n_verify"]
    enable_constrained_concretize = clip_args["clip_input_domain"]["enabled"] and (clip_args['clip_input_domain']['clip_type'] == "complete")
    if enable_constrained_concretize:
        batchsize = unverified.sum()
        device = new_x.ptb.x_L.device
        self.net.init_infeasible_bounds_constraints(batchsize, device)

    ret = yield from compute_bounds_in_spec_batches(
        self.net,
        spec_batch_size=_get_crown_spec_batch_size(), oom_retry=True,
        _sliced=_sliced, _iterate=True, _work_cap=atom_work_cap,
        x=(x_unverified,), C=select_rows(C, unverified), method=bounding_method,
        bound_upper=False, return_A=return_A, needed_A_dict=needed_A_dict,
        reference_bounds={
            k: (select_rows(v[0], unverified), select_rows(v[1], unverified))
            for k, v in reference_interm_bounds.items()}
    )
    ret = _optimize_objective_row(self, x_unverified, select_rows(C, unverified),
                                  select_rows(thresholds, unverified),
                                  stop_criterion, ret, needed_A_dict)
    lb_crown = ret[0]
    A_dict = ret[-1] if return_A else None
    infeasible_batch = self.net.infeasible_bounds_constraints
    if infeasible_batch is not None:
        lb_crown[infeasible_batch] = torch.inf
    lb[unverified] = torch.max(select_rows(lb, unverified), lb_crown)

    # format lA and lbias if required
    if return_A:
        entry = (A_dict or {}).get(self.net.output_name[0], {}).get(self.net.input_name[0])
        if entry is None or entry.get('lA') is None:
            # No A matrix for the output (its bounds came from interval propagation,
            # Neumaier2): the constant plane lA = 0, lbias = lb is a valid linear bound.
            entry = {'lA': torch.zeros(lb_crown.shape[0], lb_crown.shape[1],
                                       *new_x.shape[1:], device=lb_crown.device,
                                       dtype=lb_crown.dtype),
                     'lbias': lb_crown}
        lA_ = entry['lA']
        # Domains settled by IBP skip CROWN but still reach branching, clipping and
        # dual ascent: their valid plane is the constant IBP bound (lA = 0), never an
        # uninitialized row that may look finite.
        lA = torch.zeros(new_x.shape[0], *lA_.shape[1:], device=lA_.device, dtype=lA_.dtype)
        lA[unverified] = lA_
        lbias_ = entry['lbias']
        lbias = lb_ibp.to(device=lbias_.device, dtype=lbias_.dtype).clone()
        lbias[unverified] = lbias_
    else:
        lA = None
        lbias = None

    return lb, lA, lbias
