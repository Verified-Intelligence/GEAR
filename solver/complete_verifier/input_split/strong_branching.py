"""Objective-aware strong branching for input-split BaB.

The SB score |a_i| w_i / 2 is the first-order change of the linear bound, and only for
the child the linear function points away from: the child containing the minimizing
corner keeps the parent's plane minimum, so its gain -- which is what moves the global
bound -- comes entirely from the relaxation getting tighter. Strong branching measures
that gain directly: bound both midpoint children for a few candidate inputs with the
same bounding call the BaB loop uses, and split the input whose children improve most.

Only the branching DECISION uses these bounds; the children are bounded again when they
are picked, so soundness does not depend on anything here.
"""
from typing import Optional, Tuple

import torch
from torch import Tensor

import arguments
from input_split.utils import bound_thresholds

# Candidate children per bounding call that fitted in memory, learned on OOM (per run).
_CHILD_CAP = [None]

# History attributes written by `get_lower_bound_naive` as warm starts for the next
# call. Candidate children are not domains of the tree, so they must not move them.
_HISTORY_WARM_STARTS = ("lp_mean_y", "qtb_mean_y", "dual_lam_star", "dual_x_star",
                        "alm_mean_mu")


def candidate_children(x_L: Tensor, x_U: Tensor, cand: Tensor) -> Tuple[Tensor, Tensor]:
    """Midpoint children of every (domain, candidate input) pair.

    x_L, x_U: (B, D); cand: (B, K) input indices. Returns (B*K*2, D) boxes ordered
    [b][k][side] with side 0 the lower half and side 1 the upper half.
    """
    B, D = x_L.shape
    K = cand.shape[1]
    cand = cand.long()
    lo = x_L.unsqueeze(1).unsqueeze(2).expand(B, K, 2, D).clone()
    hi = x_U.unsqueeze(1).unsqueeze(2).expand(B, K, 2, D).clone()
    mid = (x_L.gather(1, cand) + x_U.gather(1, cand)) / 2           # (B, K)
    idx = cand.view(B, K, 1, 1)
    hi[:, :, 0:1, :].scatter_(3, idx, mid.view(B, K, 1, 1))           # lower half: u = mid
    lo[:, :, 1:2, :].scatter_(3, idx, mid.view(B, K, 1, 1))           # upper half: l = mid
    return lo.reshape(B * K * 2, D), hi.reshape(B * K * 2, D)


def child_objective_gains(parent_lb0: Tensor, child_lb: Tensor, thresholds: Tensor) -> Tensor:
    """Objective-row improvement of each child over its parent.

    parent_lb0: (B,); child_lb: (B, K, 2, S); thresholds: (B, S) (row 0 the incumbent, the
    constraint rows as the pool relaxes them). A child that would be discarded -- objective
    bound above the incumbent, +inf, or a constraint row above its threshold, the tests of
    `get_remaining_index` -- closes its whole gap, so its gain is the parent's distance
    to the incumbent (+inf without an incumbent).
    """
    thr = thresholds.unsqueeze(1).unsqueeze(1)                      # (B, 1, 1, S)
    lb0 = child_lb[..., 0]
    thr0 = thr[..., 0]
    pruned = torch.isposinf(lb0) | (torch.isfinite(lb0) & torch.isfinite(thr0) & (lb0 > thr0))
    if child_lb.shape[-1] > 1:
        c_lb, c_thr = child_lb[..., 1:], thr[..., 1:]
        pruned = pruned | (torch.isfinite(c_lb) & torch.isfinite(c_thr) & (c_lb > c_thr)).any(-1)
    p = parent_lb0.view(-1, 1, 1)
    gain = torch.where(torch.isfinite(lb0), lb0 - p, torch.zeros_like(lb0)).clamp(min=0)
    gap = torch.where(torch.isfinite(thr0), thr0 - p, torch.full_like(lb0, float("inf")))
    gap = torch.maximum(gap.clamp(min=0), gain)
    return torch.where(pruned, gap, gain)


def product_score(gains: Tensor, eps: float = 1e-12) -> Tensor:
    """SCIP-style product score of the two children's gains, (B, K, 2) -> (B, K).

    Each gain is floored at eps; +inf gains (both sides pruned without an incumbent) are
    clamped to a finite large value so ties can still be broken by the other side.
    """
    g = torch.nan_to_num(gains, nan=0.0, posinf=torch.finfo(gains.dtype).max ** 0.25)
    return torch.clamp(g[..., 0], min=eps) * torch.clamp(g[..., 1], min=eps)


def _repeat_rows(t: Optional[Tensor], reps: int) -> Optional[Tensor]:
    if t is None:
        return None
    if t.dim() > 0 and t.stride(0) == 0:
        return t.narrow(0, 0, 1).expand(t.shape[0] * reps, *t.shape[1:])
    return t.repeat_interleave(reps, dim=0)


def evaluate_candidates(net, history, dm_lb: Tensor, x_L: Tensor, x_U: Tensor, cs: Tensor,
                        thresholds: Tensor, constraints: Optional[tuple],
                        base_input_linear_constraints: Optional[tuple], cand: Tensor,
                        bounding_method: str, stop_criterion, stats,
                        max_children: int, clip=None) -> Tensor:
    """Bound both midpoint children of every candidate; return the gains (B, K, 2).

    `clip(child_L, child_U, s, e, reps)`, when given, shrinks the children of parent
    domains s:e (each repeated `reps` times) the way the loop clips real children before
    they are queued; a child clipped to an empty box is discarded (+inf bound)."""
    B, D = x_L.shape[0], x_L.flatten(1).shape[1]
    K = cand.shape[1]
    x_L = x_L.flatten(1)
    x_U = x_U.flatten(1)
    saved = {k: getattr(history, k, None) for k in _HISTORY_WARM_STARTS}
    saved_visited = stats.visited
    nlp_cfg = arguments.Config["solver"]["nlp"]
    saved_da = nlp_cfg.get("dual_ascent_enabled", False)
    saved_exact = nlp_cfg.get("dual_lp_exact_max", 0)
    # Rank candidates by the CROWN bound alone (no dual ascent / node LP / QTB), and keep
    # the batched PDHG node LP but skip the exact HiGHS re-solves (CPU, per call).
    nlp_cfg["dual_ascent_enabled"] = False
    nlp_cfg["dual_lp_exact_max"] = 0
    if bounding_method == "crown" and thresholds.shape[1] == 1 and base_input_linear_constraints is None:
        # Objective-only: no constraint rows tighten the intermediate bounds, and the
        # lookahead only ranks splits, so take them from IBP (one backward pass, not
        # one per layer). hs025 (99 exp/log terms): lookahead 4.5x -> 0.1x the real
        # bounding time at the same D per domain; Kowalik / cute_hart6: 2-4x cheaper
        # for 8-13% more domains.
        bounding_method = "crown-ibp"
    per_domain = 2 * K
    max_children = int(max_children)
    if _CHILD_CAP[0] is not None:
        max_children = min(max_children, _CHILD_CAP[0])
    chunk = max(1, max_children // per_domain)

    def bound_chunk(s, e):
        from input_split.tree_slice import checkpoint
        checkpoint()
        cl, cu = candidate_children(x_L[s:e], x_U[s:e], cand[s:e])
        n = cl.shape[0]
        keep = torch.ones(n, dtype=torch.bool, device=cl.device)
        if clip is not None:
            cl, cu = clip(cl, cu, s, e, per_domain)
            cl, cu = cl.flatten(1), cu.flatten(1)
            keep = ~(cl > cu).any(1)      # clipped to an empty box: discarded

        def rep(t):
            t = _repeat_rows(t[s:e], per_domain)
            if bool(keep.all()):
                return t
            if t.stride(0) == 0:     # shared row (the spec matrix): never materialise
                return t.narrow(0, 0, 1).expand(int(keep.sum()), *t.shape[1:])
            return t[keep]

        lb = torch.full((n, thresholds.shape[1]), torch.inf,
                        device=thresholds.device, dtype=thresholds.dtype)
        if bool(keep.any()):
            cons = None if constraints is None else tuple(rep(t) for t in constraints)
            base = (None if base_input_linear_constraints is None
                    else tuple(rep(t) for t in base_input_linear_constraints))
            lb_k, _, _, _ = net.get_lower_bound_naive(
                dm_lb=rep(dm_lb), dm_l=cl[keep], dm_u=cu[keep], history=history,
                bounding_method=bounding_method, C=rep(cs),
                stop_criterion=stop_criterion, thresholds=rep(thresholds),
                return_A=True, return_b=True, constraints=cons,
                base_input_linear_constraints=base, stats=stats)
            lb[keep] = torch.nan_to_num(lb_k.to(thresholds), nan=-torch.inf)
        checkpoint()
        return lb

    lbs = []
    try:
        s = 0
        while s < B:
            e = min(B, s + chunk)
            try:
                lbs.append(bound_chunk(s, e))
            except torch.OutOfMemoryError:
                # Memory-adaptive: retry this chunk with half the domains, and start
                # every later lookahead of the run at the size that fitted.
                if e - s <= 1:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                chunk = max(1, (e - s) // 2)
                _CHILD_CAP[0] = chunk * per_domain
                continue
            s = e
    finally:
        for k, v in saved.items():
            setattr(history, k, v)
        stats.visited = saved_visited
        nlp_cfg["dual_ascent_enabled"] = saved_da
        nlp_cfg["dual_lp_exact_max"] = saved_exact
    child_lb = torch.cat(lbs).view(B, K, 2, -1)
    return child_objective_gains(dm_lb[:, 0].to(thresholds), child_lb,
                                 bound_thresholds(getattr(history, 'rowpool', None), thresholds))


def strong_branching_decision(net, history, dm_lb: Tensor, x_L: Tensor, x_U: Tensor,
                              cs: Tensor, thresholds: Tensor, constraints: Optional[tuple],
                              base_input_linear_constraints: Optional[tuple],
                              base_score: Tensor, fallback: Tensor, n_candidates: int,
                              bounding_method: str, stop_criterion, stats,
                              max_children: int, clip=None) -> Tensor:
    """Split index (B, 1) chosen by strong branching over the top-`n_candidates` inputs of
    `base_score` (all inputs of nonzero width if n_candidates < 0). Domains where no
    candidate improves either child keep `fallback`."""
    width = (x_U - x_L).flatten(1)
    D = width.shape[1]
    k = D if n_candidates < 0 else min(int(n_candidates), D)
    score = torch.where(width > 0, base_score.to(width), torch.full_like(width, -float("inf")))
    cand = torch.topk(score, k, dim=-1).indices
    gains = evaluate_candidates(net, history, dm_lb, x_L, x_U, cs, thresholds, constraints,
                                base_input_linear_constraints, cand, bounding_method,
                                stop_criterion, stats, max_children, clip=clip)
    # A zero-width candidate (only when fewer than k inputs are open) never wins.
    open_ = width.gather(1, cand) > 0
    prod = torch.where(open_, product_score(gains), torch.zeros_like(gains[..., 0]))
    best = prod.argmax(dim=1, keepdim=True)
    chosen = cand.gather(1, best)
    improved = (gains.amax(dim=(1, 2)) > 0) & open_.any(dim=1)
    chosen = torch.where(improved.unsqueeze(1), chosen, fallback[:, :1].to(chosen))
    return chosen.to(fallback.dtype)
