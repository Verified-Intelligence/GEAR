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
Old branching heuristics, must be removed very soon (assigned to Kaidi).
"""

import torch
from auto_LiRPA import BoundedTensor, PerturbationLpNorm
import arguments
from typing import Optional


def _compute_sb_score(x_L, x_U, dom_lb, thresholds, lA) -> torch.Tensor:
    """Return SB scores before the final top-k selection."""
    input_split_args = arguments.Config['bab']['branching']['input_split']
    lA_clamping_thresh = input_split_args['sb_coeff_thresh']
    sb_margin_weight = input_split_args['sb_margin_weight']
    sb_sum = input_split_args['sb_sum']
    sb_primary_spec = input_split_args['sb_primary_spec']
    touch_zero_score = input_split_args['touch_zero_score']

    lA = lA.flatten(2)
    perturb = (x_U - x_L).unsqueeze(-2)
    assert lA_clamping_thresh >= 0

    # Per-spec, per-dim raw contributions (B, n_spec, d_in).
    raw = lA.abs().clamp(min=lA_clamping_thresh) * perturb / 2
    if sb_sum:
        score = raw.sum(dim=-2)
        if touch_zero_score:
            touch_zero = torch.logical_or(x_L == 0, x_U == 0)
            score = score + touch_zero * (x_U - x_L) * touch_zero_score
    else:
        score = (
            raw
            + (dom_lb.to(lA.device).unsqueeze(-1) - thresholds.unsqueeze(-1))
            * sb_margin_weight
        )
        if sb_primary_spec is not None:
            assert score.ndim == 3
            score = score[:, sb_primary_spec, :]
        else:
            score = score.amax(dim=-2)
    return score


def input_split_branching(net, dom_lb, x_L, x_U, lA, thresholds,
                          branching_method, split_depth=1, num_iter=0,
                          history: Optional[object] = None):
    """
    Produce input split according to branching methods.
    """
    x_L = x_L.flatten(1)
    x_U = x_U.flatten(1)
    if (thresholds is not None and dom_lb is not None
            and torch.isposinf(thresholds).any()):
        # +inf is an inactive row (no incumbent yet), not bad data: give it a zero
        # margin instead of letting it force the longest-edge fallback below.
        thresholds = torch.where(torch.isposinf(thresholds),
                                 dom_lb.to(thresholds), thresholds)
    nonfinite_sb_inputs = (
        branching_method in ('sb', 'brute-force')
        and (
            lA is None
            or not torch.isfinite(lA).all()
            or not torch.isfinite(dom_lb).all()
            or not torch.isfinite(thresholds).all()
        )
    )

    if branching_method == 'naive' or nonfinite_sb_inputs:
        # we just select the longest edge
        indices = torch.topk(x_U - x_L, split_depth, -1).indices
        if getattr(history, 'node_pool', False):
            indices = _lifted_violation_choice(indices, x_L, x_U, history)
        return indices
    elif branching_method == 'sb':
        return input_split_heuristic_sb(
            x_L, x_U, dom_lb, thresholds, lA, split_depth, history=history)
    elif branching_method == 'brute-force':
        assert split_depth == 1
        if num_iter <= arguments.Config['bab']['branching']['input_split']['bf_iters']:
            return input_split_heuristic_bf(
                net, x_L, x_U, dom_lb, thresholds, lA)
        else:
            return input_split_heuristic_sb(
                x_L, x_U, dom_lb, thresholds, lA, split_depth, history=history)
    else:
        raise NameError(f'Unsupported branching method "{branching_method}" for input splits.')


def input_split_heuristic_sb(x_L, x_U, dom_lb, thresholds, lA, split_depth=1,
                             history: Optional[object] = None) -> torch.Tensor:
    """
    Smart branching where the sensitivities for each input is calculated as a score. More sensitive inputs are split.
    @param x_L:             The lower bound on the inputs of the subdomains
    @param x_U:             The upper bound on the inputs of the subdomains
    @param dom_lb:          The lower bound on the outputs of the subdomains
    @param thresholds:      The specification threshold where dom_lb > thresholds implies the subdomain is verified
    @param lA:              CROWN lA for subdomains
    @param split_depth:     How many splits we wish to consider for all subdomains where split_depth <= input_dim
    @param history:         SplitHistory carrying the root width and the lifted-LP branching score
    @return:                Input indices to split on for each batch
    """
    score = _compute_sb_score(x_L, x_U, dom_lb, thresholds, lA)
    # note: the k (split_depth) in topk <= # inputs, because split_depth is computed as
    # min(max split depth, # inputs).
    # 1) If max split depth <= # inputs, then split_depth <= # inputs.
    # 2) If max split depth > # inputs, then split_depth = # inputs.
    indices = torch.topk(score, split_depth, -1).indices
    indices = _lifted_violation_choice(indices, x_L, x_U, history)
    indices = _exhaustive_guard(indices, x_L, x_U, history)
    return _angle_first_choice(indices, x_L, x_U, history)


def _lifted_violation_choice(indices, x_L, x_U, history):
    """First split choice from the lifted McCormick LP (solver.nlp.lifted_lp_branching):
    the variable whose products are the most violated at the domain's LP point, as in
    spatial branch-and-bound for bilinear programs. Domains without an LP point keep SB."""
    score = getattr(history, 'lifted_branch_score', None)
    if not torch.is_tensor(score) or indices.shape[-1] != 1:
        return indices
    width = (x_U - x_L).flatten(1)
    if score.shape != width.shape:
        return indices
    score = score.to(width)
    score = torch.where(width > 0, score, torch.zeros_like(score))
    best, arg = torch.nan_to_num(score, nan=-1.0).max(dim=1)
    use = best > 0
    if use.any():
        indices = indices.clone()
        indices[use, 0] = arg[use].to(indices.dtype)
    return indices


def _angle_first_choice(indices, x_L, x_U, history):
    """Split the widest variable of a circle equality x_i^2 + x_j^2 = const (cos/sin of a
    rotation) while it is wider than `angle_first_ratio` of its root width. The disk is
    the hull of a circle and the secant of x^2 on [-1, 1] is the constant 1, so the
    relaxation only becomes informative on angle sectors; SB and the lifted violation
    never pick c or s. Last in the chain, it overrides the lifted choice and the
    width guard. Emulated LP B&B: kall_circlesrectangles_c1r13 closes in 9.4k nodes
    instead of 19.6k (D = 0 at the cutoff). Non-finite widths keep the incoming choice."""
    ratio = float(arguments.Config['bab']['branching']['input_split'].get('angle_first_ratio', 0.0))
    pairs = getattr(history, 'angle_pairs', None)
    root_width = getattr(history, 'root_width', None)
    if ratio <= 0 or pairs is None or root_width is None or indices.shape[-1] != 1:
        return indices
    width = x_U.flatten(1) - x_L.flatten(1)
    root_width = root_width.to(width).reshape(1, -1)
    if root_width.shape[1] != width.shape[1]:
        return indices
    pairs = pairs.to(width.device).flatten().unique()
    rel = width[:, pairs] / root_width[:, pairs].clamp(min=1e-300)
    rel = torch.where(torch.isfinite(rel) & (root_width[:, pairs] > 0), rel, torch.zeros_like(rel))
    best, arg = rel.max(dim=1)
    use = best > ratio
    if use.any():
        indices = indices.clone()
        indices[use, 0] = pairs[arg[use]].to(indices.dtype)
    return indices


def _exhaustive_guard(indices, x_L, x_U, history):
    """Replace a first choice already `sb_min_width_ratio` times narrower (relative to
    the root box) than the widest input by that widest input.

    |a_i| w_i / 2 is the first-order change of the linear bound; it is blind to the
    relaxation's curvature, so an input whose coefficient is 0 -- x^2 on a symmetric box
    has a flat secant -- is never split. On DekkersAarts SB then bisected x1 down to
    width 6e-4 while x2 stayed at [-20, 20] and the dual stalled.
    """
    ratio = float(arguments.Config['bab']['branching']['input_split'].get(
        'sb_min_width_ratio', 0.0))
    root_width = getattr(history, 'root_width', None)
    if ratio <= 0 or root_width is None or indices.shape[-1] < 1:
        return indices
    width = x_U.flatten(1) - x_L.flatten(1)
    root_width = root_width.to(width).reshape(1, -1)
    if root_width.shape[1] != width.shape[1]:
        return indices
    rel = torch.where(root_width > 0, width / root_width.clamp(min=1e-300),
                      torch.zeros_like(width))
    rel_max, widest = rel.max(dim=1)
    chosen = rel.gather(1, indices[:, :1].long()).squeeze(1)
    degenerate = chosen < ratio * rel_max
    if degenerate.any():
        indices = indices.clone()
        indices[degenerate, 0] = widest[degenerate].to(indices.dtype)
    return indices


def _exhaustive_subdivision_guard(split_idx, split_counter, x_L, x_U, history):
    """Exhaustive subdivision: split a box's widest input once as many splits as it has open inputs (width > 0) have gone
    elsewhere since its last widest split.

    With midpoint splits a widest split halves a coordinate at the maximum relative width, so n of them halve the maximum
    and one comes at least every n + 1 splits: along any infinite path every relative width goes to 0, whatever the
    heuristics chose (the first-order SB score never picks a coordinate whose coefficient is 0, see _exhaustive_guard).
    Measured (E1a, CPU, B = 2, optimal incumbent, 16 instances): without it MultiGauss needs 177 iterations instead of 59 and
    kall_congruentcircles_c51 stays at D = 0 for 6.3k-6.7k iterations where it leaves at ~2k; on the other 14 the iterations
    change by -6% to +2%, although it forces up to 14% of the splits (Hartman3: 38 of 277).

    split_counter holds each box's splits since its last widest split (widest = largest width relative to the root width, as
    _exhaustive_guard measures it). input_split_and_repeat adds 1 to the counter of the children (once per step, also when a
    short queue makes the step split several inputs), so this returns the counter before that step: -1 after a widest split,
    the current count otherwise.
    """
    width = (x_U - x_L).flatten(1)
    root_width = getattr(history, 'root_width', None)
    if torch.is_tensor(root_width) and root_width.numel() == width.shape[1]:
        root_width = root_width.to(width).reshape(1, -1)
        rel = torch.where(root_width > 0, width / root_width.clamp(min=1e-300), torch.zeros_like(width))
    else:
        rel = width
    rel_max, widest = rel.max(dim=1)
    count = split_counter.reshape(-1)
    forced = count >= (width > 0).sum(1).clamp(min=1)
    split_idx = split_idx.clone()
    split_idx[forced, 0] = widest[forced].to(split_idx.dtype)
    is_widest = rel.gather(1, split_idx[:, :1].long()).squeeze(1) >= rel_max
    count = torch.where(is_widest, torch.full_like(count, -1), count)
    return split_idx, count.reshape(split_counter.shape)


def input_split_heuristic_bf(net, x_L, x_U, dom_lb, thresholds, lA):
    branching_args = arguments.Config['bab']['branching']
    input_split_args = branching_args['input_split']
    lA_clamping_thresh = input_split_args['sb_coeff_thresh']
    sb_margin_weight = input_split_args['sb_margin_weight']
    bf_backup_thresh = input_split_args['bf_backup_thresh']
    bf_rhs_offset = input_split_args['bf_rhs_offset']
    zero_crossing_score = input_split_args['bf_zero_crossing_score']
    touch_zero_score = input_split_args['touch_zero_score']

    assert x_L.ndim == 2
    input_dim = x_L.shape[1]
    x_M = (x_L + x_U) / 2
    new_x_L = x_L.expand(2, input_dim, -1, -1).clone()
    new_x_U = x_U.expand(2, input_dim, -1, -1).clone()
    for i in range(input_dim):
        new_x_U[0, i, :, i] = x_M[:, i]
        new_x_L[1, i, :, i] = x_M[:, i]
    new_x_L = new_x_L.view(-1, new_x_L.shape[-1])
    new_x_U = new_x_U.view(-1, new_x_U.shape[-1])
    new_x = BoundedTensor(
        new_x_L,
        ptb=PerturbationLpNorm(x_L=new_x_L, x_U=new_x_U))
    C = net.c.expand(new_x.shape[0], -1, -1)
    lb_ibp = net.net.compute_bounds(
        x=(new_x,), C=C, method='ibp', bound_upper=False)[0]
    reference_interm_bounds = {}
    for node in net.net.nodes():
        if (node.perturbed
                and isinstance(node.lower, torch.Tensor)
                and isinstance(node.upper, torch.Tensor)):
            reference_interm_bounds[node.name] = (node.lower, node.upper)
    lb_crown = net.net.compute_bounds(
        x=(new_x,), C=C, method='crown', bound_upper=False,
        reference_bounds=reference_interm_bounds
    )[0]
    lb = torch.max(lb_ibp, lb_crown)

    margin = (lb - thresholds[0]).view(2, input_dim, -1, lb.shape[-1])
    lb_base = dom_lb.cuda() - thresholds[0]
    verified = margin.amax(dim=-1) > 0

    assert bf_rhs_offset >= 0
    objective = (
        (margin - lb_base).clamp(min=0)
        / (lb_base - bf_rhs_offset).abs().clamp(min=1e-8)
        * (1 - verified.unsqueeze(-1).int())
    ).clamp(max=2e8).sum(dim=0)

    objective = objective.sum(dim=-1)
    objective = objective + 1e9 * verified.sum(dim=0)
    too_bad = objective.amax(dim=0) < bf_backup_thresh

    # TODO branch at zero rather than midpoint
    if zero_crossing_score:
        cross_zero = torch.logical_and(x_L < 0, x_U > 0)
        objective = objective + (cross_zero * (x_U - x_L) * 50000).t()
    if touch_zero_score:
        touch_zero = torch.logical_or(x_L == 0, x_U == 0)
        objective = objective + (touch_zero * (x_U - x_L) * touch_zero_score).t()

    lA = lA.view(lA.shape[0], lA.shape[1], -1)
    perturb = (x_U - x_L).unsqueeze(-2)
    sb_score = (lA.abs().clamp(min=lA_clamping_thresh) * perturb / 2
            + (dom_lb.to(lA.device).unsqueeze(-1)
                - thresholds.unsqueeze(-1)) * sb_margin_weight)
    sb_score = sb_score.sum(dim=-2)
    objective[:, too_bad] = sb_score[too_bad].t()

    index = objective.argmax(0).unsqueeze(-1)

    worst_idx = margin.amax(dim=-1).amin(dim=0).amax(dim=0).argmin()
    print('Worst idx:', worst_idx)
    print('Before', lb_base[worst_idx])
    print('Left branch:', margin[0, :, worst_idx])
    print('Right branch:', margin[1, :, worst_idx])
    print('Selected index:', index[worst_idx])
    print('Objective', objective[:, worst_idx])
    print('x_L', x_L[worst_idx])
    print('x_U', x_U[worst_idx])
    if too_bad[worst_idx]:
        print('Bad objective. Using SB.')

    if torch.isnan(margin).any():
        pass

    return index
