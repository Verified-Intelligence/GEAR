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
from torch import Tensor
from typing import Tuple, Optional, Union

import arguments
from input_constraints import (
    normalize_input_linear_constraints,
)
from input_split.utils import bound_thresholds
from auto_LiRPA.fp_enclosure import box_rounding_error, finite_abs

log_underflow = torch.log(torch.tensor([torch.finfo(torch.float64).tiny], dtype=torch.float64)).item()

def _in_depth_volume_metrics(original_x_L: Tensor, original_x_U: Tensor, new_x_L: Tensor, new_x_U: Tensor,
                                  nv_mask: Tensor, lA:Tensor, thresholds: Tensor,
                                  dm_lb: Tensor, is_lower: int):
    """

    Performs in depth volume calculations in order to better gauge how effective clipping was on this batch of
    subdomains. In particular, the total volume of this batch of domains is calculated before and after clipping.
    The log-sum-exp trick is used in order to make the calculations more stable.

    When numerical underflow is encountered, each constraint is applied one at a time and evaluated in order to
    analyze if there is a particular constraint that is very valuable to the clipping procedure or if it is a
    combination of constraints that allow the input to be clipped a lot.

    These metrics are optional in the 'clip_domains' method because it faces two major limitations:

    1)  When the input space is high-dimensional and/or small, these calculations will consistently run into
        numerical underflow. Even the log-sum-exp trick cannot salvage this scenario.
    2)  This metric is not insightful in every instance, therefore it can add unnecessary computational overhead
        that can compound over many rounds of BaB.

    Despite these limitations, these metrics may be insightful in certain scenarios. For low dimensional inputs,
    it can consistently provide accurate volume calculations and help gauge which constraints are the most
    valuable.

    Note that the Tensors are convert to float64 for higher precision as to mitigate numerical underflow
    as much as possible.

    :param original_x_L:    The original lower bound on the inputs of the subdomains
    :param original_x_U:    The original upper bound on the inputs of the subdomains
    :param new_x_L:         The lower bound on the inputs of the subdomains after clipping
    :param new_x_U:         The upper bound on the inputs of the subdomains after clipping
    :param nv_mask:         Mask of unverified domains post-clipping
    :param lA:              CROWN lA for subdomains
    :param thresholds:      The specification threshold where dom_lb > thresholds implies the subdomain is verified
    :param dm_lb:           The lower bound on the outputs of the domains
    :param is_lower:        If set to true, we are currently lower bounding the network output, else we are upper
    """

    batches, num_spec, input_dim = lA.shape

    # convert tensors to float64 for better precision in order to try
    # and mitigate numerical underflow as much as possible
    lA = lA.to(dtype=torch.float64)
    thresholds = thresholds.to(dtype=torch.float64)
    dm_lb = dm_lb.to(dtype=torch.float64)
    nv_x_L = new_x_L[nv_mask].to(dtype=torch.float64)
    nv_x_U = new_x_U[nv_mask].to(dtype=torch.float64)
    nv_ori_x_L = original_x_L[nv_mask].to(dtype=torch.float64)
    nv_ori_x_U = original_x_U[nv_mask].to(dtype=torch.float64)

    # get the log sum of the areas of the original (unverified) domains before and after clipping
    log_original_areas = torch.log((nv_ori_x_U - nv_ori_x_L)).sum(1)
    log_new_areas = torch.log((nv_x_U - nv_x_L)).sum(1)

    # Uses log-sum-exp trick to calculate the ratio (total new area / total original area) in a numerically
    # stable manner; Smaller ratios are better as this indicates more shrinkage
    lse_new_total_area = torch.logsumexp(log_new_areas, dim=0)
    lse_original_total_area = torch.logsumexp(log_original_areas, dim=0)

    if lse_new_total_area > log_underflow:
        # lse_new_total_area is never greater than lse_original_total_area, only need to check underflow on this
        # variable.
        new_total_area = f"{torch.exp(lse_new_total_area).item():.2e}"
        original_total_area = f"{torch.exp(lse_original_total_area).item():.2e}"
    else:
        new_total_area = original_total_area = "numerical_underflow"

    if new_total_area == "numerical_underflow":
        # Look more closely, is a single constraint causing this numerical underflow? Or is it a combination
        # of constraints?
        print(f"Clipping got numerical underflow, looking closer at constraints one at a time...")
        for i in range(num_spec):

            c_lA = lA[:, i:i + 1, :]
            c_thresholds = thresholds[:, i:i + 1]
            c_dm_lb = dm_lb[:, i:i + 1]

            test_x_L, test_x_U = _clip_main_fn(original_x_L, original_x_U, c_lA, None, c_thresholds, c_dm_lb,
                                               1, is_lower)

            test_nv_x_U = test_x_U[nv_mask]
            test_nv_x_L = test_x_L[nv_mask]

            test_log_new_areas = torch.log((test_nv_x_U - test_nv_x_L)).sum(1)

            # Uses log-sum-exp trick to calculate the ratio (total new area / total original area) in a numerically
            # stable manner; Smaller ratios are better as this indicates more shrinkage
            test_lse_new_total_area = torch.logsumexp(test_log_new_areas, dim=0)
            test_shrunken_ratio = torch.exp(test_lse_new_total_area - lse_original_total_area).item()
            test_new_area = torch.exp(test_lse_new_total_area).item()
            print(
                f"\t{i}: area new/original {test_new_area}/{torch.exp(lse_original_total_area).item():.2e} "
                f"({100 * test_shrunken_ratio:.2f}%)")

    shrunken_ratio = torch.exp(lse_new_total_area - lse_original_total_area).item()

    area_str = (f"Domain clipping (input dim {input_dim}, spec dim {num_spec}): "
                f"area new/original {new_total_area}/{original_total_area} ({100 * shrunken_ratio:.2f}%), ")
    log_area_str = f"log area new: {lse_new_total_area:2f} log area original: {lse_original_total_area:2f}, "
    domain_str = area_str + log_area_str
    print(domain_str)

def _underflow(t: Tensor, n: int) -> float:
    """Absolute rounding allowance of (n+16) elementwise products/sums in the subnormal range: each is below half
    the smallest subnormal (tiny*eps); the relative margins do not cover it. Below one ulp of any normal value."""
    fi = torch.finfo(t.dtype)
    return (n + 16) * fi.tiny * fi.eps


def _concretize_error(lA: Tensor, x_L: Tensor, x_U: Tensor, lbias: Tensor) -> Tensor:
    """Bound on the rounding error of concretize_bounds. box_rounding_error is (n+16)*4 eps * M with
    M = |A| max|x| + |bias|; this evaluation is two n-term dot products, a sum and the rounding of
    xhat/eps, i.e. below (n+4) eps/2 * M, so (n+4) eps * M has 2x slack and keeps clipped boxes close
    to the plain computation."""
    n = lA.shape[-1]
    return box_rounding_error(lA, x_L, x_U, lbias) * ((n + 4) / (4 * (n + 16))) + _underflow(lA, n)


def _clip_main_fn(x_L: Tensor, x_U: Tensor, lA:Tensor, lbias: Union[Tensor, None], thresholds: Tensor, dm_lb: Tensor,
                  num_iters: int, is_lower: int, dm_lb_err: Union[Tensor, None] = None,
)->Tuple[Tensor, Tensor]:
    """

    The main clipping algorithm. Applies constraints one at a time in order to use a closed-form solution that can
    potentially clip off axis-aligned regions of the input box.

    :param x_L:         The lower bound on the inputs of the subdomains
    :param x_U:         The upper bound on the inputs of the subdomains
    :param lA:          CROWN lA for subdomains
    :param lbias:       CROWN lbias for subdomains
    :param thresholds:  The specification threshold where dom_lb > thresholds implies the subdomain is verified
    :param dm_lb:       The lower bound on the outputs of the domains
    :param num_iters:   Number of times to perform clipping as this may potentially improve clipping. Requires
                        lbias to be given.
    :param is_lower:    If set to true, we are currently lower bounding the network output, else we are upper
    :param dm_lb_err:   Rounding error already made in the given dm_lb (batch, num_spec); 0 if None.
    :return:            The new x_L, x_U. Every candidate bound is rounded outward and widened by the rounding error
                        of its own formula, so no exactly feasible point is removed.
    """
    sign = 1 if is_lower else -1
    batches, num_spec, _ = lA.shape
    # adds singleton dimension so that we can later broadcast multiplication/addition/subtraction operations along
    # this dimension
    thresholds = thresholds.reshape(batches, num_spec, 1)
    dm_lb = dm_lb.reshape(batches, num_spec, 1)
    for i in range(num_iters):
        # The order in which the constraints are applied affect clipping. We can repeatedly perform clipping to
        # take advantage of this sequential dependency.
        xhat = (x_U + x_L) / 2
        eps = (x_U - x_L) / 2
        if i > 0:
            # after each clip, we need to update the domain lower bound
            dm_lb = concretize_bounds(xhat, eps, lA, lbias, is_lower).reshape(batches, num_spec, 1)
            dm_lb_err = _concretize_error(lA, x_L, x_U, lbias)

        # Solve for x across all batches, all specifications, and all input dimensions; The following three lines greatly
        # simplify and parallelize an operation that naively require 3 nested for-loops.
        eop = 'bsn,bn->bsn'  # specify multiplication broadcasting
        concrete_minus_one = dm_lb - torch.einsum(eop, lA, xhat) + sign * torch.einsum(eop, lA.abs(), eps)
        # Error of the numerator: xhat, eps, two products, three add/sub are each below eps_mach/2 of
        # the magnitude of the terms involved, so 4 eps_mach * magnitude encloses them (2x slack).
        numerator = thresholds - concrete_minus_one
        magnitude = (finite_abs(thresholds) + finite_abs(dm_lb) + finite_abs(torch.einsum(eop, lA, xhat))
                     + finite_abs(torch.einsum(eop, lA.abs(), eps)))
        num_err = magnitude * (4 * torch.finfo(magnitude.dtype).eps) + _underflow(magnitude, lA.shape[-1])
        if dm_lb_err is not None:      # the plane value dm_lb itself is only known to dm_lb_err
            num_err = num_err + dm_lb_err.reshape(batches, num_spec, 1)
        finite = torch.isfinite(numerator)
        num_lo = torch.where(finite, numerator - num_err, numerator)
        num_hi = torch.where(finite, numerator + num_err, numerator)
        q_lo, q_hi = num_lo / lA, num_hi / lA
        # Division rounds to nearest: step one ulp outward on each side.
        inf = torch.full_like(q_hi, torch.inf)
        curr_x_U = torch.nextafter(torch.maximum(q_lo, q_hi), inf)
        curr_x_L = torch.nextafter(torch.minimum(q_lo, q_hi), -inf)
        # concrete_minus_one and curr_x: (b=batch, s=num_spec, n=input_dim)

        # Sort solutions appropriately
        x_U_candidates = torch.where(lA > 0, curr_x_U, torch.inf)
        x_L_candidates = torch.where(lA < 0, curr_x_L, -torch.inf)

        # Update new_x_U(L)
        x_U = torch.min(x_U_candidates.amin(dim=1), x_U)
        x_L = torch.max(x_L_candidates.amax(dim=1), x_L)

    return x_L, x_U


def _append_base_input_constraints(
    x_L: Tensor,
    x_U: Tensor,
    lA: Tensor,
    lbias: Optional[Tensor],
    thresholds: Tensor,
    dm_lb: Optional[Tensor],
    base_input_linear_constraints: Optional[Tuple[Tensor, Tensor]],
) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
    """Append problem-side input constraints A x <= b as extra clip rows."""
    if base_input_linear_constraints is None:
        return lA, lbias, thresholds, dm_lb

    base_A, base_b = base_input_linear_constraints
    if base_A.shape[1] == 0:
        return lA, lbias, thresholds, dm_lb

    batch_size = x_L.shape[0]
    base_lbias = torch.zeros(
        (batch_size, base_A.shape[1]),
        device=x_L.device,
        dtype=lA.dtype,
    )
    xhat = (x_U + x_L) / 2
    eps = (x_U - x_L) / 2
    base_dm_lb = concretize_bounds(xhat, eps, base_A, base_lbias, is_lower=True)

    lA = torch.cat([lA, base_A], dim=1)
    thresholds = torch.cat([thresholds, base_b], dim=1)
    lbias = base_lbias if lbias is None else torch.cat([lbias, base_lbias], dim=1)
    dm_lb = base_dm_lb if dm_lb is None else torch.cat([dm_lb, base_dm_lb], dim=1)
    return lA, lbias, thresholds, dm_lb


def _slice_constraints(
    constraints: Optional[Tuple[Tensor, Tensor]],
    mask: Tensor,
) -> Optional[Tuple[Tensor, Tensor]]:
    if constraints is None:
        return None
    return constraints[0][mask], constraints[1][mask]

def clip_domains(
        x_L: Tensor,
        x_U: Tensor,
        thresholds: Tensor,
        lA: Tensor,
        lbias: Tensor,
        dm_lb: Optional[Tensor] = None,
        is_lower: bool = True,
        num_iters: int = 1,
        calculate_volume: bool = False,
        base_input_linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        base_input_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        domain_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        rowpool=None,
) -> Tuple[Tensor, Tensor]:
    """
    Takes a batch of subdomains and shrinks along dimensions to remove verified portions of the input domain
    to remove redundancy and allow for more effective splits. Domains already verified before clipping are
    represented as empty input boxes in the returned tensors.
    @param x_L:                 The lower bound on the inputs of the subdomains
    @param x_U:                 The upper bound on the inputs of the subdomains
    @param thresholds:          The specification threshold where dom_lb > thresholds implies the subdomain is verified
    @param lA:                  CROWN lA for subdomains
    @param dm_lb:               The lower bound on the outputs of the domains
    @param lbias:               CROWN lbias for subdomains. Needed to concretize dm_lb if dm_lb is not given/incorrect
    @param is_lower:            If set to true, we are currently lower bounding the network output, else we are upper
    @param num_iters:           Number of times to perform clipping as this may potentially improve clipping. Requires
                                lbias to be given
    @param base_input_linear_constraints:
                                Optional original problem constraints A x <= b appended as extra clip rows.
    @param rowpool:             The instance's RowPool (solving mode): the constraint rows are clipped at its relaxed
                                thresholds. None: `thresholds` are used as given.
    @return:                    The new x_L, x_U
    """
    # save original shapes
    x_L_shape = x_L.shape
    x_U_shape = x_U.shape
    # Get initial variables and correct views
    lA = lA.flatten(2)                 # (batch, num_spec, input_dim)
    # lbias shape: (batch, num_spec)
    batches, num_spec, input_dim = lA.shape

    thresholds = thresholds.reshape(batches, -1)
    # Constraint rows are clipped at rhs + constr_viol_tol, the feasibility tolerance the
    # filter and the primal already use (the pool applies it, see RowPool.thresholds);
    # the objective row keeps the raw incumbent. The recursive call below receives the
    # raw thresholds so they are relaxed exactly once.
    thresholds_raw = thresholds
    thresholds = bound_thresholds(rowpool, thresholds)
    if lbias is not None:
        lbias = lbias.reshape(batches, -1)
    if dm_lb is not None:
        dm_lb = dm_lb.reshape(batches, -1)

    x_L = x_L.clone().view(batches, input_dim) # (batch, input_dim)
    x_U = x_U.clone().view(batches, input_dim) # (batch, input_dim)
    original_x_L = x_L.clone()         # (batch, input_dim)
    original_x_U = x_U.clone()         # (batch, input_dim)

    normalized_base_input_linear_constraints = normalize_input_linear_constraints(
        base_input_linear_constraints,
        batch_size=batches,
        input_shape=(input_dim,),
        device=x_L.device,
        dtype=lA.dtype,
    )

    if (rowpool is not None and getattr(rowpool, 'derived_cut_registry', None) is not None
            and arguments.Config['solver']['nlp'].get('input_cutoff_cuts', False)):
        derived = rowpool.derived_cut_view(x_L, x_U, thresholds_raw[:, 0])
        if derived is not None:
            if normalized_base_input_linear_constraints is None:
                normalized_base_input_linear_constraints = derived
            else:
                normalized_base_input_linear_constraints = tuple(
                    torch.cat((old, new), dim=1)
                    for old, new in zip(normalized_base_input_linear_constraints, derived))
            base_input_linear_constraints = normalized_base_input_linear_constraints

    # The rows as given: a recursive call appends the base constraints itself.
    lA_given, lbias_given, dm_lb_given = lA, lbias, dm_lb
    if base_input_linear_constraints is not None:
        if not is_lower:
            raise ValueError(
                "base_input_linear_constraints are only supported for lower-bound relaxed clipping."
            )
        lA, lbias, thresholds, dm_lb = _append_base_input_constraints(
            x_L,
            x_U,
            lA,
            lbias,
            thresholds,
            dm_lb,
            normalized_base_input_linear_constraints,
        )
        num_spec = lA.shape[1]

    # A +inf threshold is an inactive row (no incumbent yet): it can neither certify
    # (dm_lb > inf) nor clip ((inf - c) / a is an infinite candidate), so it must not
    # disable clipping on the other rows.
    finite_mask = (torch.isfinite(lA).all(dim=(1, 2))
                   & (torch.isfinite(thresholds) | torch.isposinf(thresholds)).all(dim=1))
    if lbias is not None:
        finite_mask = finite_mask & torch.isfinite(lbias).all(dim=1)
    if dm_lb is not None:
        finite_mask = finite_mask & torch.isfinite(dm_lb).all(dim=1)
    if not finite_mask.all():
        num_skipped = (~finite_mask).sum().item()
        print(
            f"Domain clipping: skipping {num_skipped}/{batches} domains due to non-finite clipping data."
        )
        if not finite_mask.any():
            return x_L.view(x_L_shape), x_U.view(x_U_shape)
        clipped_x_L, clipped_x_U = clip_domains(
            x_L[finite_mask],
            x_U[finite_mask],
            thresholds_raw[finite_mask],
            lA_given[finite_mask],
            lbias_given[finite_mask] if lbias_given is not None else None,
            dm_lb=dm_lb_given[finite_mask] if dm_lb_given is not None else None,
            is_lower=is_lower,
            num_iters=num_iters,
            calculate_volume=calculate_volume,
            base_input_linear_constraints=_slice_constraints(
                normalized_base_input_linear_constraints, finite_mask
            ),
            rowpool=rowpool,
        )
        x_L[finite_mask] = clipped_x_L.view(-1, input_dim)
        x_U[finite_mask] = clipped_x_U.view(-1, input_dim)
        return x_L.view(x_L_shape), x_U.view(x_U_shape)

    if lbias is not None:
        xhat = (x_U + x_L) / 2             # (batch, input_dim)
        eps = (x_U - x_L) / 2              # (batch, input_dim)
        dm_lb = concretize_bounds(xhat, eps, lA, lbias, is_lower) # (batch, num_spec)
    elif dm_lb is None:
        # if lbias is not given, we cannot concretize dm_lb, so we assume it is already correct
        assert lbias is not None, "lbias must be given if dm_lb is not given"

    dm_lb_err = None
    certified_lb = dm_lb
    if lbias is not None:
        # a domain is emptied only when the lb clears the threshold by the library's standard margin
        certified_lb = dm_lb - box_rounding_error(lA, x_L, x_U, lbias) - _underflow(lA, lA.shape[-1])  # (batch, num_spec)
        dm_lb_err = _concretize_error(lA, x_L, x_U, lbias)
    certified_mask = (certified_lb > thresholds).any(1)
    num_verified = certified_mask.sum().item()
    unresolved_mask = torch.logical_not(certified_mask)
    num_unverified = batches - num_verified

    # A certified domain has no region left to search, so encode it directly as
    # an empty input box for domain-list filtering. Do not rely on
    # _clip_main_fn to make it empty: zero affine coefficients cannot shrink
    # the box, and finite-precision rounding can leave a boundary point even
    # when the relevant coefficient is nonzero.
    if num_verified:
        certified_x_U = x_U[certified_mask, 0]
        x_L[certified_mask, 0] = torch.nextafter(
            certified_x_U, torch.full_like(certified_x_U, torch.inf))

    if num_unverified == 0:
        print(
            f"All {batches} domains are already verified; "
            "returning empty input boxes.")
        return x_L.view(x_L_shape), x_U.view(x_U_shape)

    # Avoid gather/scatter overhead on the common path where every domain still
    # needs clipping. Mixed batches only clip their unresolved residuals.
    if num_verified == 0:
        x_L, x_U = _clip_main_fn(
            x_L, x_U, lA, lbias, thresholds, dm_lb, num_iters, is_lower, dm_lb_err)
    else:
        unresolved_lbias = (
            lbias[unresolved_mask] if lbias is not None else None)
        x_L[unresolved_mask], x_U[unresolved_mask] = _clip_main_fn(
            x_L[unresolved_mask], x_U[unresolved_mask],
            lA[unresolved_mask], unresolved_lbias,
            thresholds[unresolved_mask], dm_lb[unresolved_mask],
            num_iters, is_lower,
            None if dm_lb_err is None else dm_lb_err[unresolved_mask])

    # Extract domains that were unresolved before clipping for statistics.
    unresolved_x_L = x_L[unresolved_mask]
    unresolved_x_U = x_U[unresolved_mask]
    original_unresolved_x_L = original_x_L[unresolved_mask]
    original_unresolved_x_U = original_x_U[unresolved_mask]
    # Each true entry selects a nonempty residual for optional volume metrics.
    volume_mask = unresolved_mask

    # on these unverified domains, calculate the number of domains that became verified after clipping
    # (this scenario is only possible if there are at least 2 constraints, otherwise it will always be 0)
    if num_spec > 1:
        clipped_empty_mask = (unresolved_x_L > unresolved_x_U).any(1)
        shrunken_and_verified = clipped_empty_mask.sum().item()
        if calculate_volume:
            # Volume metrics include only domains with a nonempty residual box.
            volume_mask = unresolved_mask.clone()
            volume_mask[unresolved_mask] = torch.logical_not(
                clipped_empty_mask)
    else:
        shrunken_and_verified = 0

    # on these unverified domains, calculate the number that were clipped along any dimension
    num_shrunken = torch.logical_or(
        (unresolved_x_L > original_unresolved_x_L).any(1),
        (unresolved_x_U < original_unresolved_x_U).any(1)).sum(0).item()

    shrunken_str = (f"Domain clipping ({num_iters} iterations): shrunken {num_shrunken}/{num_unverified} "
                    f"({100 * (num_shrunken / num_unverified):.2f}%), ")
    batch_str = (f"Of {batches} batches, {num_verified} were verified before clipping, {num_shrunken} domains "
                 f"were shrunken, and {shrunken_and_verified} were shrunken and additionally verified.")
    domain_str = shrunken_str + batch_str
    if not arguments.Config["solving"]["solving_mode"]:
        print(domain_str)

    if calculate_volume:
        if volume_mask.sum() > 0:
            # calculate the total volume of this batch of subdomains before and after clipping
            _in_depth_volume_metrics(
                original_x_L, original_x_U, x_L, x_U,
                volume_mask, lA, thresholds, dm_lb, is_lower)
        else:
            print(f"Domain clipping (input dim {input_dim}, spec dim {num_spec}): No volume calculations to perform; "
                  f"Some (potentially none) domains were verified before clipping, but all domains became verified "
                  f"after clipping.")

    return x_L.view(x_L_shape), x_U.view(x_U_shape)

def concretize_bounds(xhat: Tensor, eps: Tensor, lA: Tensor, lbias: Tensor, is_lower: bool = True):
    """
    Given a batch of domains and hyperplanes, uses Hölder's inequality to concretize the lower bounds for L-inf norm.
    :param xhat:        Domain center
    :param eps:         Domain half-widths
    :param lA:          Hyperplane coefficients
    :param lbias:       Hyperplane offsets
    :param is_lower:    If true, these parameters are for the domain lower bound, else upper bound.
    :return:
    """
    sign = 1 if is_lower else -1  # sign based on if the network is being lower/upper bounded
    # use lbias to concretize dm_lb for the subdomains
    eop = 'bsn,bn->bs'  # specify batch matrix multiplication
    # b = batch, s = num_spec, n = input_dim
    dm_lb = torch.einsum(eop, lA, xhat) - sign * torch.einsum(eop, lA.abs(), eps) + lbias
    return dm_lb

def deconstruct_bias(
    x_L: Tensor,
    x_U: Tensor,
    A: Optional[Tensor],
    dm_ob: Tensor,
    is_lower: bool = True,
) -> Optional[Tensor]:
    """
    Deconstructs the bias parameter given the A matrix and domain lower/upper bounds.
    :param x_L:         Domain input lower bound
    :param x_U:         Domain input upper bound
    :param A:           Hyperplane coefficients
    :param dm_ob:       Domain output lower/upper bounds
    :param is_lower:    If true, these parameters are for the domain lower bound, else upper bound.
    :return:
    """
    if A is None:
        return None
    sign = 1 if is_lower else -1  # sign based on if the network was being lower/upper bounded
    A = A.flatten(2) # (batch, num_spec, in_dim)
    xhat_vect = ((x_U + x_L) / 2).flatten(1).unsqueeze(2) # (batch, in_dim, 1)
    eps_vect = ((x_U - x_L) / 2).flatten(1).unsqueeze(2) # (batch, in_dim, 1)
    dm_ob_vect = dm_ob.unsqueeze(2) # (batch, num_spec, 1)
    bias = dm_ob_vect - (A.bmm(xhat_vect) - sign*A.abs().bmm(eps_vect))
    return bias.squeeze(2) # (batch, num_spec)

def check_lbias(x_L: Tensor, x_U: Tensor, lA: Tensor, lbias: Union[Tensor, None], dm_lb: Tensor, thresholds: Tensor):
    """

    For development purposes. If it is imperative that the lA and lbias from CROWN correspond to the dm_lb,
    this function can check this (e.g. for clipping to be sound, if lA and/or lbias is incorrect, clipping will
    be unsound and remove valid regions, so this can be a valuable check).

    :param x_L:         Domain input lower bound
    :param x_U:         Domain input upper bound
    :param lA:          Hyperplane coefficients
    :param lbias:       Hyperplane offsets
    :param dm_lb:       Domain lower bound
    :param thresholds:  The specification threshold where dom_lb > thresholds implies the subdomain is verified
    """

    if lbias is None:
        # cannot check that lA and lbias is consistent with dm_lb if lbias is not given
        return

    # flatten shapes
    lbias = lbias.flatten(1)
    lA = lA.flatten(2)
    x_L = x_L.flatten(1)
    x_U = x_U.flatten(1)
    dm_lb = dm_lb.flatten(1)
    thresholds = thresholds.flatten(1)

    # Only check lbias for unverified domains. lA and lbias are allowed to be incorrect for verified subdomains
    unverified_mask = (dm_lb <= thresholds).all(1)

    # reconstruct lbias to compare with the lbias returned from CROWN
    lbias_check = deconstruct_bias(x_L, x_U, lA, dm_lb)
    # from the unverified domains, get a mask of all the domains whose lbias do not match 'lbias_check'
    same = torch.isclose(lbias[unverified_mask], lbias_check[unverified_mask], atol=1e-4).all(1)
    check = same.all().item()

    if not check:
        # print some additional info before throwing an error

        # print the lbias and lbias_check for the first 10 subdomains that have a mismatch
        not_same = torch.logical_not(same)
        num_mismatch = not_same.sum().item()
        print(f"{num_mismatch} domains have a mismatch in lbias. First 10 domains with mismatches: "
              f"\nlbias:\n{lbias[unverified_mask][not_same][:10]}"
              f"\nlbias_check:\n{lbias_check[unverified_mask][not_same][:10]}")

        # if lbias from CROWN and deconstructed lbias do not match, check that the dm_lb can be recomputed using
        # lA and lbias from CROWN
        xl_nv = x_L[unverified_mask]
        xu_nv = x_U[unverified_mask]
        xhat = (xu_nv + xl_nv) / 2
        eps = (xu_nv - xl_nv) / 2
        lA_nv = lA[unverified_mask]
        lbias_nv = lbias[unverified_mask]
        # for the unverified domains, concretize the dm_lb using lbias from CROWN
        lb_with_lbias = concretize_bounds(xhat, eps, lA_nv, lbias_nv)
        dm_lb_nv = dm_lb[unverified_mask]
        # check if the given dm_lb matches the dm_lb we calculted
        same_with_lbias = torch.allclose(dm_lb_nv, lb_with_lbias, atol=1e-4)
        print(f"Can we recover dm_lb using lA and lbias from CROWN? {same_with_lbias}")

        assert check, "The lbias returned from bounding does not match the computed lbias"
