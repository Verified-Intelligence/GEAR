"""
Differentiable Dual Ascent for CROWN Lower Bound Tightening.

After alpha-CROWN computes a lower plane for each spec, nonnegative
combinations of those planes give valid lower bounds for the Lagrangian.
Any nonnegative multipliers therefore give a bound by weak duality:

  lA_eff(lambda) = lA_obj + lambda^T lA_constr
  lbias_eff(lambda) = lbias_obj + (lambda * lbias_constr).sum(1)
  lb_crown(lA_eff, lbias_eff) = (lA_eff * dm_l).sum(-1) - (relu(-lA_eff) * (dm_u-dm_l)).sum(-1) + lbias_eff
  d(lambda) = lb_crown(lA_eff) - (lambda * rhs_constr).sum(1) - rhs_obj

Projected Adam searches for stronger multipliers. Its iterates need not
improve monotonically, so we retain the best bound and its multipliers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import time

import torch
import torch.nn.functional as F
from torch import Tensor

from auto_LiRPA.cap_log import cap_log

from .rigorous_dual import Rows, certify_infeasible, dual_lower_bound, round_down


@dataclass
class DualAscentConfig:
    """Configuration for the dual ascent optimizer.

    Attributes:
        num_steps: Number of projected gradient ascent steps.
        lr: Adam learning rate.
        lambda_init: Initialization strategy for lambda. Currently only
            "zero" is supported (lambda starts at all zeros).
        verbose: If True, prints debug information every 5 steps.
    """
    num_steps: int = 20
    lr: float = 0.05
    lambda_init: str = "zero"
    verbose: bool = False


def crown_lb_from_lA(
    lA: Tensor,
    lbias: Tensor,
    dm_l: Tensor,
    dm_u: Tensor,
) -> Tensor:
    """Compute CROWN lower bound given per-spec input sensitivities.

    Given the linear bound lb(x) = lA @ x + lbias for all x in [dm_l, dm_u],
    the tightest (worst-case) lower bound is achieved by choosing:
      x_worst[i] = dm_l[i]  if lA[i] >= 0
      x_worst[i] = dm_u[i]  if lA[i] < 0

    This is equivalent to:
      lb = (lA * dm_l).sum(-1) - relu(-lA) * (dm_u - dm_l)).sum(-1) + lbias

    Args:
        lA: (B, d_in) — linear coefficients of the CROWN bound.
        lbias: (B,) — constant bias of the CROWN bound.
        dm_l: (B, d_in) — lower bounds of the input domain.
        dm_u: (B, d_in) — upper bounds of the input domain.

    Returns:
        Tensor of shape (B,) — the CROWN lower bound for each batch element.
    """
    # (lA * dm_l).sum(-1): contribution when all inputs at lower bound
    # - (relu(-lA) * (dm_u - dm_l)).sum(-1): correction for negative coefficients
    return (
        (lA * dm_l).sum(-1)
        - (F.relu(-lA) * (dm_u - dm_l)).sum(-1)
        + lbias
    )


def dual_ascent_tighten(
    lA: Tensor,
    lbias: Tensor,
    dm_l: Tensor,
    dm_u: Tensor,
    rhs: Tensor,
    cfg: DualAscentConfig,
    lambda_init: Optional[Tensor] = None,
    input_constr_A: Optional[Tensor] = None,
    input_constr_b: Optional[Tensor] = None,
    deadline: Optional[float] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Tighten CROWN lower bounds via projected Adam dual ascent.

    Combines valid CROWN lower planes for the objective and constraints. Every
    nonnegative multiplier vector gives a valid lower bound by weak duality.

    The dual function is:
      d(lambda) = lb_crown(lA_eff(lambda)) - (lambda * rhs_constr).sum(1) - rhs_obj

    where lA_eff(lambda) = lA_obj + lambda @ lA_constr
    and   lbias_eff(lambda) = lbias_obj + (lambda * lbias_constr).sum(1)

    d is concave in lambda (minimum of linear functions), but projected Adam
    does not guarantee finding its global maximum. Return the best finite
    certificate seen, including the zero-multiplier baseline.

    Args:
        lA: (B, m, d_in) — CROWN per-spec input sensitivities.
        lbias: (B, m) — CROWN per-spec biases.
        dm_l: (B, d_in) or compatible shape — domain lower bounds.
        dm_u: (B, d_in) or compatible shape — domain upper bounds.
        rhs: (B, m) — per-spec thresholds. rhs[:,0] is the objective RHS,
            rhs[:,1:] are the constraint RHS values.
        cfg: DualAscentConfig controlling optimization hyperparameters.
        lambda_init: Optional (B, m-1) tensor for warm-starting lambda.
            If None, uses the strategy specified in cfg.lambda_init.

    Returns:
        lb_initial: (B,) — initial lower bound d(0) = crown_lb(lA_obj) - rhs_obj.
        lb_dual: (B,) — best lower bound d(lambda*) seen.
        lam_star: (B, n_total_constr) — non-negative multipliers for lb_dual.
    """
    B, m, d_in = lA.shape

    # Reshape domain bounds to (B, d_in) in case they were passed in
    # a different compatible shape (e.g., flattened).
    dm_l = dm_l.reshape(B, d_in)
    dm_u = dm_u.reshape(B, d_in)

    # Split into objective (col 0) and output-constraint (cols 1:) components.
    lA_obj = lA[:, 0, :]       # (B, d_in)
    lbias_obj = lbias[:, 0]    # (B,)
    rhs_obj = rhs[:, 0]        # (B,)

    # Compute the initial lower bound d(0) = crown_lb(lA_obj) - rhs_obj.
    lb_initial = crown_lb_from_lA(lA_obj, lbias_obj, dm_l, dm_u) - rhs_obj

    lA_constr = lA[:, 1:, :] if m > 1 else lA.new_zeros((B, 0, d_in))
    lbias_constr = lbias[:, 1:] if m > 1 else lbias.new_zeros((B, 0))
    rhs_constr = rhs[:, 1:] if m > 1 else rhs.new_zeros((B, 0))
    n_output_constr = lA_constr.shape[1]

    if input_constr_A is not None or input_constr_b is not None:
        if input_constr_A is None or input_constr_b is None:
            raise ValueError("input_constr_A and input_constr_b must be provided together.")
        input_constr_A = input_constr_A.reshape(B, -1, d_in).to(device=lA.device, dtype=lA.dtype)
        input_constr_b = input_constr_b.reshape(B, -1).to(device=lA.device, dtype=lA.dtype)
    else:
        input_constr_A = lA.new_zeros((B, 0, d_in))
        input_constr_b = lA.new_zeros((B, 0))
    n_input_constr = input_constr_A.shape[1]
    n_total_constr = n_output_constr + n_input_constr

    # Returned values are the rigorous (outward rounded) Lagrangian bound over the same rows,
    # not the plain floating-point d(lambda): Adam only proposes multipliers.
    f64 = torch.float64      # exact for fp32 data
    G_rig = torch.cat([lA_constr, input_constr_A], dim=1).to(f64)
    rhs_rows = torch.cat([rhs_constr, input_constr_b], dim=1).to(f64)
    shift_rows = torch.cat([lbias_constr, lbias_constr.new_zeros(B, n_input_constr)], dim=1).to(f64)
    h_rig = rhs_rows - shift_rows
    c_rig, c0_rig = lA_obj.to(f64), lbias_obj.to(f64) - rhs_obj.to(f64)
    u2 = 2 * torch.finfo(f64).eps    # f64 inputs: rounding of h and c0 (exact for fp32), 2 ulp margin

    def fin(t):      # |t| with non-finite entries dropped: those make the bound -inf by themselves
        return torch.nan_to_num(t.to(f64).abs(), nan=0.0, posinf=0.0, neginf=0.0)

    def rigorous(lam_value: Tensor) -> Tensor:
        y = lam_value.to(f64)
        c0_err = u2 * (fin(lbias_obj) + fin(rhs_obj) + (y * (fin(rhs_rows) + fin(shift_rows))).sum(1))
        lb, _, _ = dual_lower_bound(Rows.from_dense(G_rig), c_rig, c0_rig, h_rig,
                                    dm_l.to(f64), dm_u.to(f64), y, c0_err=c0_err)
        return round_down(lb, lA.dtype)

    lb_initial = rigorous(torch.zeros(B, n_total_constr, device=lA.device, dtype=lA.dtype))

    # Trivial case: no constraints beyond the objective, nothing to optimize.
    if n_total_constr == 0:
        lam_star = torch.zeros(B, 0, device=lA.device, dtype=lA.dtype)
        return lb_initial, lb_initial.clone(), lam_star

    # Initialize lambda. Non-finite warm starts must not poison the optimizer.
    if lambda_init is not None:
        lam = lambda_init.clone().to(device=lA.device, dtype=lA.dtype)
        lam = torch.where(torch.isfinite(lam), lam, torch.zeros_like(lam))
    else:
        lam = torch.zeros(B, n_total_constr, device=lA.device, dtype=lA.dtype)

    lam = lam.clamp(min=0.0).requires_grad_(True)

    # Retain each domain's bound and matching multiplier, starting from d(0).
    best_bound = lb_initial.detach().clone()
    best_lam = torch.zeros_like(lam)

    def dual_value(lam_value: Tensor) -> Tensor:
        lam_out = lam_value[:, :n_output_constr]
        lam_in = lam_value[:, n_output_constr:]

        lA_eff = lA_obj
        lbias_eff = lbias_obj
        dual_penalty = rhs_obj
        if n_output_constr > 0:
            lA_eff = lA_eff + (lam_out.unsqueeze(-1) * lA_constr).sum(1)
            lbias_eff = lbias_eff + (lam_out * lbias_constr).sum(1)
            dual_penalty = dual_penalty + (lam_out * rhs_constr).sum(1)
        if n_input_constr > 0:
            lA_eff = lA_eff + (lam_in.unsqueeze(-1) * input_constr_A).sum(1)
            dual_penalty = dual_penalty + (lam_in * input_constr_b).sum(1)

        return crown_lb_from_lA(lA_eff, lbias_eff, dm_l, dm_u) - dual_penalty

    def retain_best(candidate_bound: Tensor, candidate_lam: Tensor) -> None:
        # Invalid iterates cannot replace a finite certificate.
        better = torch.isfinite(candidate_bound) & (
            ~torch.isfinite(best_bound) | (candidate_bound > best_bound))
        best_bound.copy_(torch.where(better, candidate_bound, best_bound))
        best_lam.copy_(torch.where(better.unsqueeze(-1), candidate_lam, best_lam))

    optimizer = torch.optim.Adam([lam], lr=cfg.lr)

    # Early stopping state
    plateau_count = 0
    prev_d_mean = None

    for step in range(cfg.num_steps):
        if deadline is not None and time.time() >= deadline:
            break
        optimizer.zero_grad()

        # Dual objective: d(lambda) = lb_eff - lam_out^T rhs_constr - lam_in^T b_in - rhs_obj
        d_lam = dual_value(lam)
        with torch.no_grad():
            retain_best(d_lam, lam)

        # Early stop 1: all domains pruned, with finite certificates.
        finite_d = torch.isfinite(d_lam)
        if (finite_d & (d_lam >= 0)).all():
            if cfg.verbose:
                print(f"  [DualAscent] Early stop at step {step}: all finite domains pruned (d >= 0)")
            break

        # Maximize d, excluding non-finite domains from the batch loss.
        loss = -torch.where(finite_d, d_lam, torch.zeros_like(d_lam)).mean()
        loss.backward()
        # A masked loss can still produce NaN gradients through 0 * inf.
        # Domains are independent, so clear the invalid rows after backward.
        lam.grad = torch.where(finite_d.unsqueeze(-1), lam.grad, torch.zeros_like(lam.grad))
        optimizer.step()

        # Project lambda onto the non-negative orthant (lambda >= 0).
        with torch.no_grad():
            lam.clamp_(min=0.0)

            # Early stop 2: improvement plateau (only over non-pruned domains)
            unpruned = finite_d & (d_lam < 0)
            if unpruned.any():
                current_d_mean = d_lam[unpruned].mean()
            else:
                # All finite domains were pruned; invalid rows cannot guide ascent.
                break
            if prev_d_mean is not None:
                improvement = current_d_mean - prev_d_mean
                if improvement.abs() < 1e-6:
                    plateau_count += 1
                    if plateau_count >= 3:
                        if cfg.verbose:
                            print(f"  [DualAscent] Early stop at step {step}: plateau "
                                  f"(d_mean={current_d_mean.item():.6f})")
                        break
                else:
                    plateau_count = 0
            prev_d_mean = current_d_mean.clone()

        if cfg.verbose and (step % 5 == 0 or step == cfg.num_steps - 1):
            with torch.no_grad():
                print(
                    f"  [DualAscent] step={step:3d} | "
                    f"d_min={d_lam.min().item():.4f} | "
                    f"d_mean={d_lam.mean().item():.4f} | "
                    f"norm(lam)={lam.norm().item():.4f}"
                )

    # Score the post-update iterate too, including when the budget is zero.
    with torch.no_grad():
        retain_best(dual_value(lam), lam)

    # Certify the proposed multipliers; zero multipliers remain the fallback.
    with torch.no_grad():
        rig_best = rigorous(best_lam)
        better = torch.isfinite(rig_best) & (~torch.isfinite(lb_initial) | (rig_best > lb_initial))
        best_bound = torch.where(better, rig_best, lb_initial)
        best_lam = torch.where(better.unsqueeze(-1), best_lam, torch.zeros_like(best_lam))
    return lb_initial, best_bound, best_lam


def _lp_dual_value(c: Tensor, c0: Tensor, G: Tensor, h: Tensor,
                   l: Tensor, u: Tensor, y: Tensor) -> Tensor:
    """Lagrangian bound for min c^T x + c0 s.t. G x <= h, l <= x <= u at y >= 0.

    d(y) = c0 - h^T y + sum_j min(r_j l_j, r_j u_j) with r = c + G^T y is a valid
    lower bound for every y >= 0 (weak duality). The rigorous evaluator subtracts the
    rounding error of this very evaluation (-inf where a value is not finite).
    """
    f64 = torch.float64      # exact for fp32 data
    lb, _, _ = dual_lower_bound(Rows.from_dense(G.to(f64)), c.to(f64), c0.to(f64), h.to(f64),
                                l.to(f64), u.to(f64), y.to(f64))
    return round_down(lb, c.dtype)


@torch.no_grad()
def lp_bound_pdhg(
    c: Tensor,
    c0: Tensor,
    G: Tensor,
    h: Tensor,
    l: Tensor,
    u: Tensor,
    iters: int = 1000,
    ruiz_iters: int = 10,
    check_every: int = 50,
    y0: Optional[Tensor] = None,
    max_elems: int = 25_000_000,
    stall_rtol: float = 0.0,
    return_primal: bool = False,
    deadline: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """Batched lower bound for  min c^T x + c0  s.t.  G x <= h,  l <= x <= u.

    Solves the LP approximately with diagonally preconditioned PDHG (Pock and
    Chambolle) after rescaling the box to [0, 1] and Ruiz-equilibrating G, and
    returns the best Lagrangian bound seen at the checkpoints together with its
    multipliers. Every returned value is a valid lower bound whatever the
    iteration budget, because it is the closed-form dual value of a y >= 0.

    Rows with a non-finite rhs are inactive and keep y = 0.

    Args:
        c: (B, n) objective plane coefficients; c0: (B,) its constant.
        G: (B, m, n) constraint rows; h: (B, m) right-hand sides.
        l, u: (B, n) box. y0: optional (B, m) warm start.
    Returns:
        (bound (B,), y (B, m)).
    Stops early once no element's best bound improved by more than
    stall_rtol * (1 + |bound|) between two checkpoints (0 disables).
    With return_primal, also returns the last primal iterate (B, n) in the original
    coordinates (a heuristic point in the box, e.g. for branching; not a bound).
    """
    B, m, n = G.shape
    if B > 1 and B * m * n > max_elems:
        # The solver holds several (B, m, n) temporaries; bound them by chunking.
        step = max(1, max_elems // max(m * n, 1))
        parts = [lp_bound_pdhg(c[i:i + step], c0[i:i + step], G[i:i + step], h[i:i + step],
                               l[i:i + step], u[i:i + step], iters, ruiz_iters, check_every,
                               None if y0 is None else y0[i:i + step], max_elems,
                               stall_rtol, return_primal, deadline)
                 for i in range(0, B, step)]
        return tuple(torch.cat([p[k] for p in parts]) for k in range(len(parts[0])))
    active = torch.isfinite(h) & torch.isfinite(G).all(-1)
    G = torch.where(active.unsqueeze(-1), G, torch.zeros_like(G))
    h = torch.where(active, h, torch.ones_like(h))
    # Always score the zero multiplier; a warm start may give a weaker bound.
    y = torch.zeros_like(h)
    best = _lp_dual_value(c, c0, G, h, l, u, y)
    best_y = y.clone()
    if y0 is not None:
        y = y0.to(device=h.device, dtype=h.dtype)
        y = torch.where(torch.isfinite(y), y, torch.zeros_like(y)).clamp(min=0.0)
        y = torch.where(active, y, torch.zeros_like(y))
        warm_bound = _lp_dual_value(c, c0, G, h, l, u, y)
        better = torch.isfinite(warm_bound) & (
            ~torch.isfinite(best) | (warm_bound > best))
        best = torch.where(better, warm_bound, best)
        best_y = torch.where(better.unsqueeze(-1), y, best_y)
    if iters <= 0 or m == 0 or (deadline is not None and time.time() >= deadline):
        return (best, best_y, (l + u) / 2) if return_primal else (best, best_y)

    w = (u - l).clamp(min=0)
    Gs = G * w.unsqueeze(1)                          # x = l + w z, z in [0, 1]
    cs = c * w
    hs = h - torch.einsum('bmn,bn->bm', G, l)
    # Ruiz equilibration Gs <- Dr Gs Dc; z = Dc zs, y = Dr ys.
    Dr = torch.ones_like(h)
    Dc = torch.ones_like(c)
    for _ in range(ruiz_iters):
        if deadline is not None and time.time() >= deadline:
            break
        rn = Gs.abs().amax(2)
        cn = Gs.abs().amax(1)
        rn = torch.where(rn > 0, rn.sqrt(), torch.ones_like(rn))
        cn = torch.where(cn > 0, cn.sqrt(), torch.ones_like(cn))
        Gs = Gs / rn.unsqueeze(2) / cn.unsqueeze(1)
        Dr = Dr / rn
        Dc = Dc / cn
    cs = cs * Dc
    hs = hs * Dr
    zub = 1.0 / Dc
    absG = Gs.abs()
    tau = 1.0 / absG.sum(1).clamp(min=1e-12)
    sig = 1.0 / absG.sum(2).clamp(min=1e-12)
    zs = 0.5 * zub
    ys = y / Dr
    for k in range(iters):
        if deadline is not None and time.time() >= deadline:
            break
        grad = cs + torch.einsum('bmn,bm->bn', Gs, ys)
        zn = torch.minimum((zs - tau * grad).clamp(min=0.0), zub)
        ys = (ys + sig * (torch.einsum('bmn,bn->bm', Gs, 2 * zn - zs) - hs)).clamp(min=0.0)
        zs = zn
        if (k + 1) % check_every == 0 or k + 1 == iters:
            y = ys * Dr
            d = _lp_dual_value(c, c0, G, h, l, u, y)
            better = torch.isfinite(d) & (~torch.isfinite(best) | (d > best))
            prev = best
            best = torch.where(better, d, best)
            best_y = torch.where(better.unsqueeze(-1), y, best_y)
            if stall_rtol > 0 and k + 1 >= 2 * check_every and bool(
                    ((best - prev) <= stall_rtol * (1 + best.abs())).all()):
                break
    if return_primal:
        x = torch.minimum(torch.maximum(l + w * (Dc * zs), l), u)
        return best, best_y, x
    return best, best_y


@torch.no_grad()
def lp_bound_halpern(
    c: Tensor,
    c0: Tensor,
    G: Tensor,
    h: Tensor,
    l: Tensor,
    u: Tensor,
    budget: int,
    y0: Optional[Tensor] = None,
    stop: Optional[Tensor] = None,
    accuracy: float = 1e-6,
    deadline: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """Plane LP of every box as one restarted-Halpern lane (auto_LiRPA.node_lanes.solve_pool).

    Same contract as lp_bound_pdhg: min c^T x + c0 s.t. G x <= h, l <= x <= u, rows with a
    non-finite rhs inactive. Each lane runs on its own COO table (values per box, the
    union pattern of the batch) and stops at KKT error <= `accuracy`, at the cutoff `stop`
    (default +inf: never), or after `budget` steps. The bound is the rigorous outward-rounded
    dual value d_B of the lane's best multipliers, so it is valid whatever the stopping
    point; the zero multiplier and the warm start y0 are also scored and the maximum is
    returned. A lane whose Farkas ray certifies the box empty returns +inf.
    """
    from auto_LiRPA.node_lanes import solve_pool
    B, m, n = G.shape
    active = torch.isfinite(h) & torch.isfinite(G).all(-1)
    Gz = torch.where(active.unsqueeze(-1), G, torch.zeros_like(G))
    hz = torch.where(active, h, torch.ones_like(h))
    y = torch.zeros_like(hz)
    best = _lp_dual_value(c, c0, Gz, hz, l, u, y)
    best_y = y
    if y0 is not None:
        y = torch.where(torch.isfinite(y0), y0, torch.zeros_like(y0)).clamp(min=0.0).to(hz)
        y = torch.where(active, y, torch.zeros_like(y))
        d = _lp_dual_value(c, c0, Gz, hz, l, u, y)
        better = torch.isfinite(d) & (~torch.isfinite(best) | (d > best))
        best = torch.where(better, d, best)
        best_y = torch.where(better.unsqueeze(-1), y, best_y)
    if budget <= 0 or m == 0 or (deadline is not None and time.time() >= deadline):
        return best, best_y
    f64 = torch.float64
    rows = Rows.from_dense(Gz.to(f64))
    cd = c.to(f64).contiguous()
    data = (rows, cd, c0.to(f64).contiguous(), hz.to(f64).contiguous(), l.to(f64).contiguous(),
            u.to(f64).contiguous(), torch.zeros_like(cd))
    cutoff = torch.full((B,), torch.inf, dtype=f64, device=c.device) if stop is None else stop.to(f64)
    warm = None if y0 is None else ((l.to(f64) + u.to(f64)) / 2, best_y.to(f64))
    out = solve_pool(data, capacity=B, budget=int(budget), stop=cutoff, warm=warm, accuracy=accuracy,
                     capture=False, deadline=deadline, iteration_scheme='halpern', implied_box=True)
    lane = round_down(out.lower, c.dtype)
    better = torch.isfinite(lane) & (~torch.isfinite(best) | (lane > best))
    best = torch.where(better, lane, best)
    best_y = torch.where(better.unsqueeze(-1), out.best_y.to(best_y), best_y)
    best = torch.where(out.empty, torch.full_like(best, torch.inf), best)
    return best, best_y


_HIGHS_POOL = None


def _highs_pool():
    global _HIGHS_POOL
    if _HIGHS_POOL is None:
        import concurrent.futures
        import os
        workers = max(1, min(8, (os.cpu_count() or 2) // 2))
        _HIGHS_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=workers,
                                                            thread_name_prefix="highs")
    return _HIGHS_POOL


@torch.no_grad()
def lp_bound(
    c: Tensor,
    c0: Tensor,
    G: Tensor,
    h: Tensor,
    l: Tensor,
    u: Tensor,
    pdhg_iters: int = 500,
    y0: Optional[Tensor] = None,
    exact_max: int = 0,
    exact_max_size: int = 200_000,
    deadline: Optional[float] = None,
    stop: Optional[Tensor] = None,
    halpern: bool = False,
) -> Tuple[Tensor, Tensor]:
    """halpern=True: Halpern lanes instead (lp_bound_halpern, no CPU LP, exact_max ignored). Each box
    stops at KKT convergence or at its cutoff `stop` (objective threshold, default never); the step
    budget 20 * pdhg_iters is only a backstop (hs109 needs <= 5150 steps = 10 * 500, so 2x headroom).
    Otherwise: lp_bound_pdhg on the whole batch, then an exact HiGHS solve for the `exact_max`
    elements with the lowest bound when each LP has at most `exact_max_size` = m * n
    coefficients. HiGHS only proposes multipliers: the bound is the closed-form
    Lagrangian value of y = max(-marginals, 0), so its accuracy cannot affect validity.
    """
    if halpern:
        return lp_bound_halpern(c, c0, G, h, l, u, 20 * pdhg_iters, y0=y0, stop=stop, deadline=deadline)
    bound, y = lp_bound_pdhg(c, c0, G, h, l, u, iters=pdhg_iters, y0=y0,
                           deadline=deadline)
    B, m, n = G.shape
    if exact_max > 0 and m > 0 and m * n > exact_max_size:
        cap_log('dual_ascent.exact_lp', 'm*n', exact_max_size, m * n, 'PDHG bound only, no exact LP')
    if exact_max <= 0 or m == 0 or m * n > exact_max_size or \
            (deadline is not None and time.time() >= deadline):
        return bound, y
    try:
        from scipy.optimize import linprog
    except ImportError:
        return bound, y
    import numpy as np
    active = torch.isfinite(h) & torch.isfinite(G).all(-1)
    idx = torch.argsort(bound)[:exact_max].tolist()
    Gc, hc, cc = G[idx].cpu().double(), h[idx].cpu().double(), c[idx].cpu().double()
    lc, uc, ac = l[idx].cpu().double(), u[idx].cpu().double(), active[idx].cpu()
    y_ex = torch.zeros(len(idx), m, dtype=torch.float64)
    jobs = []
    for k in range(len(idx)):
        rows = ac[k].nonzero().reshape(-1)
        if rows.numel() == 0:
            continue
        jobs.append((k, rows, cc[k].numpy(), Gc[k, rows].numpy(), hc[k, rows].numpy(),
                     list(zip(lc[k].numpy(), uc[k].numpy()))))

    def _solve(job):
        k, rows, ck, Gk, hk, bk = job
        left = float('inf') if deadline is None else deadline - time.time()
        if left <= 0:
            return k, rows, None, False
        options = {} if deadline is None else {'time_limit': left}
        res = linprog(ck, A_ub=Gk, b_ub=hk, bounds=bk, method='highs', options=options)
        if res.status == 0 and getattr(res, 'ineqlin', None) is not None:
            return k, rows, np.maximum(-res.ineqlin.marginals, 0.0), False
        left = float('inf') if deadline is None else deadline - time.time()
        if res.status != 2 or left <= 0:
            return k, rows, None, False
        # HiGHS calls it infeasible: the multipliers of min{t : G z - t <= h, t >= 0} are
        # a Farkas proposal, checked below (scipy's result carries no ray).
        options = {} if deadline is None else {'time_limit': left}
        ct = np.zeros(Gk.shape[1] + 1)
        ct[-1] = 1.0
        r2 = linprog(ct, A_ub=np.hstack([Gk, -np.ones((Gk.shape[0], 1))]), b_ub=hk,
                     bounds=bk + [(0.0, None)], method='highs', options=options)
        if r2.status == 0 and getattr(r2, 'ineqlin', None) is not None:
            return k, rows, np.maximum(-r2.ineqlin.marginals, 0.0), True
        return k, rows, None, False

    # The LPs are independent and HiGHS runs outside the GIL for most of a solve, so
    # they are solved in parallel threads; each solve is unchanged.
    y_far = torch.zeros(len(idx), m, dtype=torch.float64)
    farkas = torch.zeros(len(idx), dtype=torch.bool)
    for k, rows, yk, empty in _highs_pool().map(_solve, jobs):
        if yk is None:
            continue
        if empty:
            y_far[k, rows] = torch.from_numpy(yk)
            farkas[k] = True
        else:
            y_ex[k, rows] = torch.from_numpy(yk)
    y_ex = y_ex.to(G)
    Gs = torch.where(active[idx].unsqueeze(-1), G[idx], torch.zeros_like(G[idx]))
    hs = torch.where(active[idx], h[idx], torch.ones_like(h[idx]))
    b_ex = _lp_dual_value(c[idx], c0[idx], Gs, hs, l[idx], u[idx], y_ex)
    if farkas.any():
        # A box whose rows the multipliers prove empty holds no feasible point: +inf.
        f = farkas.nonzero().reshape(-1).to(bound.device)
        empty = certify_infeasible(Rows.from_dense(Gs[f]), hs[f], l[idx][f], u[idx][f], y_far.to(G)[f])
        b_ex = b_ex.clone()
        b_ex[f[empty]] = torch.inf
    idx_t = torch.tensor(idx, device=bound.device)
    better = b_ex > bound[idx_t]
    bound = bound.clone()
    y = y.clone()
    bound[idx_t] = torch.where(better, b_ex, bound[idx_t])
    y[idx_t] = torch.where(better.unsqueeze(-1), y_ex, y[idx_t])
    return bound, y
