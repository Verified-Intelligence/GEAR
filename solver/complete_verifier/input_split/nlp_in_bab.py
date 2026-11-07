"""NLP/ALM-PGD solver integration for input-split Branch-and-Bound.

Centralizes all NLP/ALM-PGD related logic used within input_bab_parallel
(batch_branch_and_bound.py), including:
  - Root-node NLP falsification (ALM-PGD, IPOPT)
  - BaB GPU budget estimation
  - BaB-loop periodic ALM-PGD attacks with adaptive interval
  - Primal solution bookkeeping
"""

import math
from auto_LiRPA.cap_log import cap_log
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

import torch
import concurrent.futures

import arguments


# ================================================================
# Data classes
# ================================================================

@dataclass
class RootNLPResult:
    """Result container for root-node NLP/IPOPT falsification."""
    dm_l: torch.Tensor                          # possibly tightened lower bounds
    dm_u: torch.Tensor                          # possibly tightened upper bounds
    nlp_found: bool = False
    nlp_obj: Any = None                         # float or None
    nlp_best_x: Optional[torch.Tensor] = None
    ipopt_found: bool = False
    ipopt_obj: Any = None                       # float or None
    ipopt_x: Optional[torch.Tensor] = None
    root_peak_mem_bytes: Optional[int] = None
    root_total_mem_bytes: Optional[int] = None


@dataclass
class ALMBaBState:
    """Mutable state for ALM-PGD during BaB loop."""
    current_interval: int = 5
    base_interval: int = 5
    no_improve_count: int = 0
    warmstart_x: Optional[torch.Tensor] = None
    best_primal_x: Optional[torch.Tensor] = None
    ipopt_bab_fails: int = 0            # consecutive IPOPT-in-BaB calls without improvement
    # `bab_alm_max_time_share` accounting: the window opens at the BaB start and reopens
    # at every incumbent improvement; share_alm_time is the ALM time spent inside it.
    share_start: Optional[float] = None
    share_alm_time: float = 0.0
    fail_streak: int = 0                # consecutive in-BaB calls without a better incumbent
    deadline: float = math.inf          # absolute end of the global time limit


@dataclass
class PickedDomainBatch:
    """Picked input domains for one BaB dual-verification step."""
    dm_l: torch.Tensor
    dm_u: torch.Tensor
    c: torch.Tensor
    rhs: torch.Tensor
    spec_size: torch.Tensor
    objective_lower_bound: Optional[torch.Tensor] = None
    base_input_linear_constraints: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    base_input_constraints: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    domain_constraints: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    attack_counter: Optional[torch.Tensor] = None


# ================================================================
# Internal helpers (moved from batch_branch_and_bound.py)
# ================================================================

def run_pre_incomplete_primal(
    model_ori,
    dm_l: torch.Tensor,
    dm_u: torch.Tensor,
    c: torch.Tensor,
    rhs: torch.Tensor,
    or_spec_size: torch.Tensor,
    nlp_config: Dict[str, Any],
    remaining_global: float,
    base_input_linear_constraints: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,
) -> Dict[str, Any]:
    """Run the existing verified root primal search before LiRPA construction.

    The returned payload is deliberately serialization-friendly and keeps
    primal feasibility separate from proof status. Callers may terminate only
    when ``nlp_found`` is true; a verified feasible point with a nonnegative
    objective is a primal incumbent, not a proof.
    """
    if remaining_global <= 0:
        raise ValueError("pre-incomplete primal search requires positive remaining time")

    root_config = dict(nlp_config)
    configured_timeout = max(0.0, float(root_config.get("timeout", 60.0)))
    root_budget_frac = float(root_config.get("root_budget_frac", 0.5))
    allotted = (
        remaining_global * root_budget_frac
        if root_budget_frac > 0
        else remaining_global
    )
    capped_timeout = min(configured_timeout, remaining_global, allotted)
    if capped_timeout <= 0:
        raise ValueError("pre-incomplete primal search received a zero time budget")
    root_config["timeout"] = capped_timeout
    root_config["ipopt_timeout"] = min(
        float(root_config.get("ipopt_timeout", configured_timeout)),
        capped_timeout,
    )

    cuda_mem_device = None
    if torch.cuda.is_available() and dm_l.is_cuda:
        cuda_mem_device = (
            dm_l.device.index
            if dm_l.device.index is not None
            else torch.cuda.current_device()
        )
        torch.cuda.reset_peak_memory_stats(cuda_mem_device)

    from nlp_solver.falsification import compare_solvers

    print(
        "[Pre-Incomplete Primal] Running root search before bound construction "
        f"(budget={capped_timeout:.1f}s)"
    )
    comparison = compare_solvers(
        None,
        model_ori,
        dm_l,
        dm_u,
        c,
        rhs,
        or_spec_size,
        reference_dict={},
        config=root_config,
        base_input_linear_constraints=base_input_linear_constraints,
    )

    alm_verified = bool(getattr(comparison, "alm_verified", False))
    local_verified = bool(getattr(comparison, "local_verified", False))
    verified = bool(
        (alm_verified or local_verified)
        and getattr(comparison, "best_x", None) is not None
        and getattr(comparison, "best_obj", None) is not None
    )
    best_obj = getattr(comparison, "best_obj", None)
    best_x = getattr(comparison, "best_x", None)

    peak_mem_bytes = None
    total_mem_bytes = None
    if cuda_mem_device is not None:
        peak_mem_bytes = int(torch.cuda.max_memory_allocated(cuda_mem_device))
        total_mem_bytes = int(
            torch.cuda.get_device_properties(cuda_mem_device).total_memory
        )

    payload = {
        "verified": verified,
        "nlp_found": bool(verified and float(best_obj) < 0.0),
        "objective": None if best_obj is None else float(best_obj),
        "violation": (
            None
            if getattr(comparison, "alm_viol", None) is None
            else float(comparison.alm_viol)
        ),
        "x": (
            None
            if best_x is None
            else best_x.detach().clone().to(device="cpu")
        ),
        "best_method": str(getattr(comparison, "best_method", "none")),
        "alm_verified": alm_verified,
        "alm_objective": (
            None
            if getattr(comparison, "alm_obj", None) is None
            else float(comparison.alm_obj)
        ),
        "alm_violation": (
            None
            if getattr(comparison, "alm_viol", None) is None
            else float(comparison.alm_viol)
        ),
        "alm_time": float(getattr(comparison, "alm_time", 0.0)),
        "budget": float(capped_timeout),
        "peak_mem_bytes": peak_mem_bytes,
        "total_mem_bytes": total_mem_bytes,
    }
    print(
        "[Pre-Incomplete Primal] "
        f"verified={payload['verified']}, alm_verified={alm_verified}, "
        f"objective={payload['objective']}, violation={payload['violation']}"
    )
    return payload


def pre_incomplete_to_root_result(
    reference_dict: Optional[Dict[str, Any]],
    dm_l: torch.Tensor,
    dm_u: torch.Tensor,
) -> Optional[RootNLPResult]:
    """Convert a verified pre-incomplete payload into root-BaB state."""
    if not reference_dict:
        return None
    payload = reference_dict.get("pre_incomplete_primal")
    if not isinstance(payload, dict) or not payload.get("verified", False):
        return None

    x = payload.get("x")
    objective = payload.get("objective")
    if x is None or objective is None:
        return None
    x = torch.as_tensor(x, device=dm_l.device, dtype=dm_l.dtype)
    if x.numel() != dm_l[0].numel():
        print(
            "[Pre-Incomplete Primal] Ignoring candidate with incompatible "
            f"shape {tuple(x.shape)} for root shape {tuple(dm_l[0].shape)}"
        )
        return None

    return RootNLPResult(
        dm_l=dm_l,
        dm_u=dm_u,
        nlp_found=bool(payload.get("nlp_found", float(objective) < 0.0)),
        nlp_obj=float(objective),
        nlp_best_x=x.reshape_as(dm_l[0:1]),
        root_peak_mem_bytes=payload.get("peak_mem_bytes"),
        root_total_mem_bytes=payload.get("total_mem_bytes"),
    )


def compute_root_dual_warmstart(
    net,
    ret: Dict[str, Any],
    dm_l: torch.Tensor,
    dm_u: torch.Tensor,
    c: torch.Tensor,
    rhs: torch.Tensor,
    global_lb: torch.Tensor,
    base_input_linear_constraints: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Optional[torch.Tensor]:
    """Run dual ascent at the root node and return lam_star flattened to (m,).

    Returns None when disabled or prerequisites are missing (lA unavailable,
    no constraints to ascend on, etc.).
    """
    nlp_cfg = arguments.Config["solver"]["nlp"]
    if not bool(nlp_cfg.get("alm_root_mu_warmstart", False)):
        return None
    if not bool(nlp_cfg.get("dual_ascent_enabled", False)):
        print("[Root DualAscent] skipped: dual_ascent_enabled=False")
        return None

    input_name = net.net.input_name[0]
    lA = ret.get("lA", {}).get(input_name, None) if isinstance(ret, dict) else None
    if lA is None:
        print("[Root DualAscent] skipped: lA not in reference_dict")
        return None
    print(f"[Root DualAscent] running (lA.shape={tuple(lA.shape)})")

    from input_split.clip import deconstruct_bias
    from nlp_solver.dual_ascent import DualAscentConfig, dual_ascent_tighten

    lbias = deconstruct_bias(dm_l, dm_u, lA, global_lb)

    B_da = lA.shape[0]
    d_in = dm_l.reshape(B_da, -1).shape[1]
    lA_flat = lA.reshape(B_da, lA.shape[1], d_in)
    dm_l_flat = dm_l.reshape(B_da, d_in)
    dm_u_flat = dm_u.reshape(B_da, d_in)

    input_constr_A = None
    input_constr_b = None
    if base_input_linear_constraints is not None:
        input_constr_A = base_input_linear_constraints[0].to(lA.device, lA.dtype)
        input_constr_b = base_input_linear_constraints[1].to(lA.device, lA.dtype)

    try:
        da_cfg = DualAscentConfig(
            num_steps=int(nlp_cfg.get("dual_ascent_steps", 20)),
            lr=float(nlp_cfg.get("dual_ascent_lr", 0.05)),
            verbose=bool(nlp_cfg.get("dual_ascent_verbose", False)),
        )
        _, _, lam_star = dual_ascent_tighten(
            lA=lA_flat,
            lbias=lbias,
            dm_l=dm_l_flat,
            dm_u=dm_u_flat,
            rhs=rhs,
            cfg=da_cfg,
            input_constr_A=input_constr_A,
            input_constr_b=input_constr_b,
        )
    except Exception as e:  # pylint: disable=broad-except
        import warnings
        warnings.warn(f"[Root DualAscent] Skipped due to error: {e}")
        return None

    if lam_star.numel() == 0:
        return None

    # Squeeze batch dim: root has a single domain (B=1 for root NLP).
    return lam_star[0].detach()


# Share of the root primal window (nlp `timeout`) the first-order root start may use: one tenth,
# so that the start can never take the window from the local solvers it seeds.
_ROOT_START_WINDOW_SHARE = 0.1


def _first_order_root_start(lA, lbias, rhs_row, dm_l, dm_u, base_input_linear_constraints, nlp_config):
    """Primal iterate of the root plane LP from PDHG, all on the device; None when the dense table
    plus the PDHG workspace does not fit in the available memory. A heuristic start only: the local
    solvers repair it and every incumbent passes the unchanged feasibility gate.

    PDHG stops when the dual bound improves by less than `constr_viol_tol` (relative) between two
    checkpoints, or at the time slice; the iteration limit is only a backstop.
    """
    from nlp_solver.dual_ascent import lp_bound_pdhg
    dev = dm_l.device
    n = dm_l[:1].reshape(1, -1).shape[1]
    m = lA.shape[1] - 1 + (0 if base_input_linear_constraints is None
                           else base_input_linear_constraints[0][:1].reshape(-1, n).shape[0])
    # G and its concatenation with the input rows, plus the four copies lp_bound_pdhg holds (masked rows,
    # scaled rows, their absolute values, one transient).
    need = 6 * m * n * 8
    if dev.type == 'cuda':
        free = torch.cuda.mem_get_info(dev)[0]
    else:
        import psutil
        free = psutil.virtual_memory().available
    if need > free:
        cap_log('root_lp_start', 'bytes', free, need, 'no first-order LP start')
        return None
    lo = dm_l[:1].reshape(1, -1).detach().double()
    hi = dm_u[:1].reshape(1, -1).detach().double()
    G, h = lA[:1].reshape(1, lA.shape[1], n)[0, 1:].detach().double(), rhs_row[1:] - lbias.reshape(-1)[1:].double()
    if base_input_linear_constraints is not None:
        G = torch.cat([G, base_input_linear_constraints[0][:1].reshape(-1, n).double()])
        h = torch.cat([h, base_input_linear_constraints[1][:1].reshape(-1).double()])
    c = lA[:1].reshape(1, lA.shape[1], n)[:, 0].detach().double()
    window = float(nlp_config.get("timeout", 60.0))
    t0 = time.time()
    _, _, x = lp_bound_pdhg(
        c, torch.zeros(1, dtype=c.dtype, device=dev), G.unsqueeze(0), h.unsqueeze(0), lo, hi,
        iters=10 ** 6, stall_rtol=float(nlp_config["constr_viol_tol"]), return_primal=True,
        deadline=time.time() + _ROOT_START_WINDOW_SHARE * window)
    print(f'[root-lp-start] first-order rows={m} cols={n} seconds={time.time() - t0:.2f}', flush=True)
    x = torch.where(torch.isfinite(x), x, (lo + hi) / 2).clamp(lo, hi)
    return x.to(dtype=dm_l.dtype).reshape(dm_l[:1].shape)


def compute_root_lp_point(
    net,
    ret: Dict[str, Any],
    dm_l: torch.Tensor,
    dm_u: torch.Tensor,
    rhs: torch.Tensor,
    global_lb: torch.Tensor,
    base_input_linear_constraints: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    rowpool=None,
) -> Optional[torch.Tensor]:
    """Minimizer of the root objective plane over the box and the root rows (HiGHS), the constraint
    rows at the thresholds of `rowpool` (None: as given).

    A start point for the root local solver: on pooling-type problems the LP
    relaxation's solution sits near the global optimum while the box centre leads to
    a local one. Heuristic only; returns None when unavailable.
    """
    if not bool(arguments.Config["solver"]["nlp"].get("root_lp_local_start", False)):
        return None
    input_name = net.net.input_name[0]
    lA = ret.get("lA", {}).get(input_name, None) if isinstance(ret, dict) else None
    if lA is None:
        return None
    try:
        import numpy as np
        from scipy.optimize import linprog
        from input_split.clip import deconstruct_bias
        from input_split.utils import bound_thresholds

        n = dm_l[:1].reshape(1, -1).shape[1]
        nlp_config = arguments.Config["solver"]["nlp"]
        first_order = bool(nlp_config.get("root_primal_first_order", False))
        if first_order:
            lbias = deconstruct_bias(dm_l[:1], dm_u[:1], lA[:1], global_lb[:1])
            r = bound_thresholds(rowpool, rhs[:1].reshape(1, -1))[0].detach().double()
            return _first_order_root_start(lA, lbias, r, dm_l, dm_u, base_input_linear_constraints, nlp_config)
        if lA.shape[1] * n > 2_000_000:
            cap_log('root_lp_start', 'rows*cols', 2_000_000, lA.shape[1] * n, 'no LP start')
            return None     # dense LP too large to be a cheap heuristic
        lbias = deconstruct_bias(dm_l[:1], dm_u[:1], lA[:1], global_lb[:1])
        A = lA[:1].reshape(1, lA.shape[1], n)[0].detach().double().cpu()
        b = lbias.reshape(1, -1)[0].detach().double().cpu()
        r = bound_thresholds(rowpool, rhs[:1].reshape(1, -1))[0].detach().double().cpu()
        G, h = A[1:], r[1:] - b[1:]
        if base_input_linear_constraints is not None:
            G = torch.cat([G, base_input_linear_constraints[0][:1].reshape(-1, n).double().cpu()])
            h = torch.cat([h, base_input_linear_constraints[1][:1].reshape(-1).double().cpu()])
        keep = torch.isfinite(h) & torch.isfinite(G).all(1)
        G, h = G[keep], h[keep]
        lo = dm_l[:1].reshape(-1).detach().double().cpu().numpy()
        hi = dm_u[:1].reshape(-1).detach().double().cpu().numpy()
        res = linprog(A[0].numpy(), A_ub=G.numpy() if G.numel() else None,
                      b_ub=h.numpy() if G.numel() else None,
                      bounds=list(zip(lo, hi)), method="highs")
        if res.status != 0:
            return None
        x = np.clip(res.x, lo, hi)
        return torch.as_tensor(x, dtype=dm_l.dtype, device=dm_l.device).reshape(dm_l[:1].shape)
    except Exception as e:  # pylint: disable=broad-except
        print(f"[Root LP point] skipped: {e}")
        return None


def compute_qtb_tangent_point(model_ori, dm_l, dm_u, c, rhs, or_spec_size, nlp_config,
                              x0, base_input_linear_constraints=None, time_left=math.inf):
    """Tangent point for the quadratic tangent bound: a KKT point of the root problem
    with every constraint row relaxed by `constr_viol_tol`, i.e. of exactly the set the
    dual LP bounds over (the RowPool's relaxed rows).

    The incumbent is no substitute: it is feasible only to within the tolerance and
    not a KKT point, so its tangent plane is tilted and loose across the feasible set
    (cute_dual1: 0.03359 against an optimum of 0.03501). IPOPT is warm-started at `x0`.

    Returns (x_kkt, accepted). x_kkt is None when IPOPT is unavailable or does not reach
    the relaxed set. The KKT point may sit at g = tol (plus Ipopt's own tolerance), so
    it is finished into a point the primal accepts: `accepted` is (x, objective), or
    None. Being a second, warm-started solve it can beat the root local solver
    (pooling_bental5tp: the root Ipopt stalled at -2900 at its iteration cap; this
    solve reached -3500).
    """
    import numpy as np
    from nlp_solver.ipopt_solver import IPOPT_AVAILABLE, NNProblemIPOPT, IPOPTSolver
    from nlp_solver.statistics import verify_counterexample
    from nlp_solver.utils import _split_or_clauses

    if not IPOPT_AVAILABLE or x0 is None:
        return None, None
    clauses = list(_split_or_clauses(c.squeeze(0), rhs.squeeze(0), or_spec_size))
    if not clauses:
        return None, None
    C_obj, rhs_obj, C_constr, rhs_constr = clauses[0]
    A = None if base_input_linear_constraints is None else base_input_linear_constraints[0]
    b = None if base_input_linear_constraints is None else base_input_linear_constraints[1]
    problem = NNProblemIPOPT(
        model_ori, dm_l.squeeze(0), dm_u.squeeze(0),
        C_obj, rhs_obj, C_constr, rhs_constr,
        input_constr_A=A, input_constr_b=b, device=None, dtype=None,
    )
    solver = IPOPTSolver(nlp_config)
    tol = float(nlp_config.get("constr_viol_tol", solver.constr_viol_tol))
    # Its own budget: the root Ipopt budget can be long, and this runs after it. It ends at the run's deadline
    # (`time_left`, the unified root) at the latest.
    timeout = (time_left if nlp_config.get("controller", False) else
               min(float(nlp_config.get("qtb_tangent_timeout", 10.0)), time_left))
    x0_np = np.clip(x0.detach().reshape(-1).cpu().double().numpy(), problem.lb, problem.ub)
    res = solver._solve_single_cyipopt(problem, x0_np, timeout, slack=tol)
    if res.exit_code not in (0, 1) or res.x_last is None \
            or res.constraint_violation is None or res.constraint_violation > tol + 1e-6:
        return None, None
    x_kkt = torch.as_tensor(res.x_last, dtype=dm_l.dtype, device=dm_l.device)

    accepted = None
    remaining = timeout - res.solve_time
    if remaining > 0.1 * timeout:
        fin = solver.solve_single(problem, x_init=res.x_last, timeout=remaining)
        if fin.x_solution is not None:
            x_acc = torch.as_tensor(fin.x_solution, dtype=dm_l.dtype, device=dm_l.device)
            ok, obj, _ = verify_counterexample(
                model_ori, x_acc.view(dm_l.squeeze(0).shape),
                C_obj, rhs_obj, C_constr, rhs_constr,
                input_constr_A=A, input_constr_b=b, tol=tol)
            if ok:
                accepted = (x_acc, float(obj))
    if bool(nlp_config.get("qtb_accept_blend", False)):
        # The KKT point sits at g = tol up to Ipopt's own tolerance, and the finishing
        # solve may run out of time (cute_gouldqp3: KKT 2.062677, finished 2.063803,
        # tolerance 2.1e-4). Points on the segment towards the warm start (the
        # incumbent, feasible) are tried as they are; only verified points count.
        x_start = torch.as_tensor(x0_np, dtype=dm_l.dtype, device=dm_l.device)
        for lam in (1.0, 0.999, 0.99, 0.95, 0.8):
            x_b = torch.minimum(torch.maximum(lam * x_kkt + (1.0 - lam) * x_start,
                                              dm_l.reshape(-1)), dm_u.reshape(-1))
            ok, obj, _ = verify_counterexample(
                model_ori, x_b.view(dm_l.squeeze(0).shape),
                C_obj, rhs_obj, C_constr, rhs_constr,       # same units as `accepted`
                input_constr_A=A, input_constr_b=b, tol=tol)
            if ok and (accepted is None or float(obj) < accepted[1]):
                accepted = (x_b, float(obj))
                break
    return x_kkt, accepted


def _build_root_single_problem(model_ori, dm_l, dm_u, c, rhs, or_spec_size,
                               base_input_linear_constraints=None):
    """Build NNProblemSingleDomain from the first OR clause at root."""
    from nlp_solver import NNProblemSingleDomain
    from nlp_solver.utils import _split_or_clauses

    clauses = list(_split_or_clauses(c.squeeze(0), rhs.squeeze(0), or_spec_size))
    if not clauses:
        return None, None, None, None, None

    C_obj, rhs_obj, C_constr, rhs_constr = clauses[0]
    problem = NNProblemSingleDomain(
        model=model_ori,
        dm_l=dm_l.squeeze(0),
        dm_u=dm_u.squeeze(0),
        C_obj=C_obj,
        rhs_obj=rhs_obj,
        C_constr=C_constr,
        rhs_constr=rhs_constr,
        input_constr_A=None if base_input_linear_constraints is None else base_input_linear_constraints[0],
        input_constr_b=None if base_input_linear_constraints is None else base_input_linear_constraints[1],
        device=None,
        dtype=None,
    )
    return problem, C_obj, rhs_obj, C_constr, rhs_constr


def _run_alm_problem(problem, nlp_config: Dict[str, Any], x_init=None, use_bab_budget=False):
    """Run ALM solver on a prepared single-domain problem."""
    from nlp_solver import ALMConfig, ALMPGDSolver

    cfg = ALMConfig.from_dict(nlp_config)
    num_restarts = None
    if use_bab_budget:
        cfg.num_restarts = int(nlp_config.get("bab_num_restarts", max(1, cfg.num_restarts // 10)))
        cfg.timeout = float(nlp_config.get("bab_timeout", min(5.0, cfg.timeout)))
        cfg.outer_max_iter = int(nlp_config.get("bab_outer_max_iter", min(10, cfg.outer_max_iter)))
        num_restarts = cfg.num_restarts
    solver = ALMPGDSolver(cfg)
    return solver.solve_with_restarts(problem, num_restarts=num_restarts, x_init=x_init)


def _update_primal_and_warmstart(
    primal_solution: torch.Tensor,
    best_primal_x: Optional[torch.Tensor],
    warmstart_x: Optional[torch.Tensor],
    alm_shifted_obj: Optional[torch.Tensor],
    alm_raw_obj: Optional[torch.Tensor],
    alm_x: Optional[torch.Tensor],
):
    """Update primal incumbent with raw objective and propagate warm-start."""
    del alm_shifted_obj  # Kept for debug call-site readability.

    improved = False
    new_primal = primal_solution
    new_best_primal_x = best_primal_x
    new_warmstart_x = warmstart_x

    if alm_x is not None:
        new_warmstart_x = alm_x.detach().clone()

    if alm_raw_obj is not None and not alm_raw_obj.isinf().all() and alm_raw_obj < primal_solution:
        new_primal = alm_raw_obj.to(primal_solution.device)
        improved = True
        if alm_x is not None:
            new_best_primal_x = alm_x.detach().clone()

    return new_primal, new_best_primal_x, new_warmstart_x, improved


def _estimate_bab_br_budget(
    nlp_config: Dict[str, Any],
    root_domains: int,
    root_restarts: int,
    root_peak_mem_bytes: Optional[int] = None,
    total_mem_bytes: Optional[int] = None,
) -> int:
    """Estimate shared BaB budget on B*R from root-node GPU profile."""
    root_br = max(1, int(root_domains) * int(root_restarts))

    if root_peak_mem_bytes is None or total_mem_bytes is None or root_peak_mem_bytes <= 0:
        return root_br

    utilization = float(nlp_config.get("bab_budget_utilization", 0.9))
    min_scale = float(nlp_config.get("bab_budget_min_scale", 1.0))
    max_scale = float(nlp_config.get("bab_budget_max_scale", 4.0))

    scale = utilization * float(total_mem_bytes) / float(root_peak_mem_bytes)
    scale = max(min_scale, min(max_scale, scale))
    return max(1, int(root_br * scale))


def _gather_constraint_rows(
    constraints: Tuple[torch.Tensor, torch.Tensor],
    row_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather per-batch constraint rows along dim=1."""
    A, b = constraints
    gathered_A = A.gather(1, row_indices.unsqueeze(-1).expand(-1, -1, A.shape[-1]))
    gathered_b = b.gather(1, row_indices)
    return gathered_A, gathered_b


def _select_alm_dual_cut_constraints(
    picked_batch: PickedDomainBatch,
    nlp_config: Dict[str, Any],
    history: Optional[object] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Select useful dual-derived cuts to append as temporary ALM input constraints.

    Candidate rows come from the stored domain constraints for the picked domains.
    We drop the immutable base-input-constraint prefix and the objective row, then
    keep only rows whose max slack over the current domain is not clearly inactive.
    The retained rows are padded with no-op constraints so the batch has a uniform
    `(B, K, d)` shape for ALM.
    """
    if not bool(nlp_config.get("alm_dual_cuts_enabled", False)):
        return None
    if picked_batch.domain_constraints is None:
        return None
    if picked_batch.domain_constraints[0].numel() == 0:
        return None

    same_spec_size = (picked_batch.spec_size == picked_batch.spec_size[0]).all()
    if not same_spec_size:
        return None

    from input_constraints import (
        affine_constraints_domain_upper_bound,
        drop_prefix_constraints,
        standard_form_to_input_constraints,
    )

    base_prefix = 0
    derived_constraints = picked_batch.domain_constraints
    if picked_batch.base_input_constraints is not None:
        base_prefix = int(picked_batch.base_input_constraints[0].shape[1])
        domain_A, domain_b = picked_batch.domain_constraints
        base_A, base_b = picked_batch.base_input_constraints
        if (
            domain_A.shape[1] >= base_prefix
            and torch.equal(domain_A[:, :base_prefix], base_A)
            and torch.equal(domain_b[:, :base_prefix], base_b)
        ):
            derived_constraints = drop_prefix_constraints(
                picked_batch.domain_constraints,
                base_prefix,
            )
    if derived_constraints is None or derived_constraints[0].shape[1] <= 1:
        return None

    # Stored derived rows are ordered as [objective, output constraints...].
    candidate_constraints = drop_prefix_constraints(derived_constraints, 1)
    if candidate_constraints is None or candidate_constraints[0].shape[1] == 0:
        return None

    eval_domain = str(nlp_config.get("alm_dual_cut_eval_domain", "complete")).strip().lower()
    if eval_domain in ("base", "complete"):
        domain_constraints = picked_batch.base_input_constraints
    elif eval_domain in ("box", "relaxed"):
        domain_constraints = None
    else:
        raise ValueError(
            f"Unsupported ALM dual-cut evaluation domain: {eval_domain}"
        )

    ub = affine_constraints_domain_upper_bound(
        candidate_constraints,
        picked_batch.dm_l,
        picked_batch.dm_u,
        domain_constraints=domain_constraints,
        rearrange_constraints=False,
    )
    if ub is None:
        return None

    finite_mask = torch.isfinite(candidate_constraints[0]).all(dim=-1)
    finite_mask = finite_mask & torch.isfinite(candidate_constraints[1])
    finite_mask = finite_mask & torch.isfinite(ub)

    inactive_tol = float(nlp_config.get("alm_dual_cut_inactive_tol", 1e-6))
    useful_mask = finite_mask & (ub > -inactive_tol)
    if not useful_mask.any():
        return None

    candidate_count = int(candidate_constraints[0].shape[1])
    topk = int(nlp_config.get("alm_dual_cut_topk", 0))
    if topk <= 0:
        return None
    topk = min(candidate_count, topk)

    score = (ub + inactive_tol).clamp(min=0)
    score_mode = str(nlp_config.get("alm_dual_cut_score", "ub")).strip().lower()
    if score_mode == "lambda_ub" and history is not None:
        lam_star = getattr(history, "dual_lam_star", None)
        if lam_star is not None and lam_star.numel() >= candidate_count:
            lam_weight = lam_star[:candidate_count].to(device=score.device, dtype=score.dtype)
            lam_floor = float(nlp_config.get("alm_dual_cut_lambda_floor", 0.0))
            score = score * lam_weight.clamp(min=lam_floor).view(1, -1)
    elif score_mode != "ub":
        raise ValueError(f"Unsupported ALM dual-cut score: {score_mode}")

    masked_score = torch.where(
        useful_mask,
        score,
        torch.full_like(score, float("-inf")),
    )
    topk_idx = masked_score.topk(topk, dim=1).indices
    selected_A, selected_b = _gather_constraint_rows(candidate_constraints, topk_idx)
    selected_mask = useful_mask.gather(1, topk_idx)
    selected_mask = selected_mask & torch.isfinite(masked_score.gather(1, topk_idx))

    selected_A = torch.where(
        selected_mask.unsqueeze(-1),
        selected_A,
        torch.zeros_like(selected_A),
    )
    selected_b = torch.where(
        selected_mask,
        selected_b,
        torch.zeros_like(selected_b),
    )

    if not selected_mask.any():
        return None

    if bool(nlp_config.get("verbose", False)):
        useful_counts = useful_mask.sum(dim=1).float()
        kept_counts = selected_mask.sum(dim=1).float()
        print(
            "  [ALM-BaB] Dual cuts: "
            f"candidates={candidate_count}, "
            f"useful_mean={useful_counts.mean().item():.2f}, "
            f"kept_mean={kept_counts.mean().item():.2f}, "
            f"eval_domain={eval_domain}"
        )

    return standard_form_to_input_constraints((selected_A, selected_b))


# ================================================================
# Root-node NLP/IPOPT integration
# ================================================================

def apply_root_alm_overrides(nlp_config) -> Dict[str, Any]:
    """Copy of ``nlp_config`` where ``root_alm_<key>`` replaces ``alm_<key>``.

    Lets the root ALM use restoration-oriented settings while the in-BaB ALM
    keeps its own (the in-BaB ALM reads ``alm_<key>`` directly).
    """
    root_config = dict(nlp_config)
    for key, value in nlp_config.items():
        if isinstance(key, str) and key.startswith("root_alm_"):
            root_config["alm_" + key[len("root_alm_"):]] = value
    return root_config


def run_root_nlp_and_ipopt(net, dm_l, dm_u, c, rhs, or_spec_size,
                           reference_dict, nlp_config, initial_verified,
                           base_input_linear_constraints=None,
                           dual_mu_init=None,
                           root_wall_budget=None,
                           local_extra_x0=None):
    """Run the root NLP (IPOPT + ALM-PGD) and optional standalone IPOPT falsification.

    Args:
        root_wall_budget: optional wall-clock cap (seconds) for the root
            solvers, derived from the remaining global timeout. Caps the
            configured ALM/IPOPT timeouts so the root stage cannot starve BaB.

    Returns:
        RootNLPResult with solver results and possibly tightened bounds.
        Caller must check result.ipopt_found for early return.
    """
    result = RootNLPResult(dm_l=dm_l, dm_u=dm_u)
    device = rhs.device
    model_ori = net.model_ori
    local_solver_used_in_nlp = False

    # --- NLP Falsification ---
    if nlp_config["enabled"] and not initial_verified:
        print("Running NLP falsification at root node...")

        root_nlp_config = apply_root_alm_overrides(nlp_config)
        if root_wall_budget is not None and root_wall_budget > 0:
            if nlp_config.get("controller", False):
                root_nlp_config['_controller_parallel_ipopt_budget'] = min(root_wall_budget, max(
                    float(root_nlp_config.get('timeout', 60.0)),
                    float(root_nlp_config.get('ipopt_timeout', 60.0))))
            capped_timeout = (root_wall_budget if nlp_config.get("controller", False) else
                              min(float(root_nlp_config.get("timeout", 60.0)), root_wall_budget))
            capped_ipopt = min(float(root_nlp_config.get(
                "ipopt_timeout", root_nlp_config.get("timeout", 60.0))), root_wall_budget)
            root_nlp_config["timeout"] = capped_timeout
            root_nlp_config["ipopt_timeout"] = capped_ipopt
            print(f"  [Root] Wall budget cap: ALM/NLP {capped_timeout:.1f}s, "
                  f"IPOPT {capped_ipopt:.1f}s (remaining-global-based)")
        enable_local_solver = bool(root_nlp_config.get("enable_local_solver", False))
        enable_alm = bool(nlp_config.get("enable_alm", False))
        root_local_solver_warmstart = bool(root_nlp_config.get("root_local_solver_warmstart", True))

        # Auto-enable the configured local solver at root to warm-start ALM.
        if enable_alm and root_local_solver_warmstart and not enable_local_solver:
            enable_local_solver = True
            root_nlp_config["enable_local_solver"] = True
            print("  [Root] Auto-enabled IPOPT warm-start for ALM")

        # CUDA memory tracking for budget estimation
        cuda_mem_device = None
        if torch.cuda.is_available() and rhs.is_cuda:
            cuda_mem_device = rhs.device.index if rhs.device.index is not None else torch.cuda.current_device()

        if enable_alm:
            if cuda_mem_device is not None:
                torch.cuda.reset_peak_memory_stats(cuda_mem_device)

            from nlp_solver import compare_solvers
            print("=" * 70)
            if not enable_local_solver:
                print("Running NLP Solver (ALM-PGD only)")
            else:
                print("Running NLP Solver Comparison (IPOPT + ALM-PGD)")
            print("=" * 70)

            comparison_result = compare_solvers(
                net, model_ori,
                result.dm_l.squeeze(0), result.dm_u.squeeze(0),
                c.squeeze(0), rhs.squeeze(0),
                or_spec_size,
                reference_dict=reference_dict,
                config=root_nlp_config,
                base_input_linear_constraints=base_input_linear_constraints,
                dual_mu_init=dual_mu_init,
                local_extra_x0=local_extra_x0,
            )
            local_solver_used_in_nlp = enable_local_solver

            if comparison_result.best_method != 'none' and comparison_result.best_obj is not None:
                if comparison_result.best_obj < 0:
                    result.nlp_found = True
                    result.nlp_obj = comparison_result.best_obj
                    print(f"NLP Comparison: Best method = {comparison_result.best_method}")
                    print(f"NLP Comparison: Found optimal! objective = {result.nlp_obj:.6f}")
                else:
                    result.nlp_obj = comparison_result.best_obj
                    print(f"NLP Comparison: Best objective = {result.nlp_obj:.6f} (not counterexample)")
            else:
                print("NLP Comparison: No feasible solution found by either solver")

            result.nlp_best_x = comparison_result.best_x

            if enable_local_solver:
                print(f"  Local ({comparison_result.local_method}): verified={comparison_result.local_verified}, obj={comparison_result.local_obj}, time={comparison_result.local_time:.2f}s")
            print(f"  ALM:   verified={comparison_result.alm_verified}, obj={comparison_result.alm_obj}, time={comparison_result.alm_time:.2f}s")
            print("=" * 70)

            if cuda_mem_device is not None:
                result.root_peak_mem_bytes = int(torch.cuda.max_memory_allocated(cuda_mem_device))
                result.root_total_mem_bytes = int(torch.cuda.get_device_properties(cuda_mem_device).total_memory)

    # --- IPOPT Falsification ---
    if (
        nlp_config.get("ipopt_enabled", False)
        and not initial_verified
        and not local_solver_used_in_nlp
    ):
        print("Running IPOPT falsification at root node...")
        from nlp_solver import ipopt_falsification

        ipopt_found, ipopt_x, ipopt_obj_val = ipopt_falsification(
            net, model_ori,
            result.dm_l.squeeze(0), result.dm_u.squeeze(0),
            c.squeeze(0), rhs.squeeze(0),
            or_spec_size,
            config=nlp_config,
            base_input_linear_constraints=base_input_linear_constraints,
        )

        if ipopt_found:
            print(f"IPOPT found counterexample! objective = {ipopt_obj_val:.6f}")
            result.ipopt_found = True
            result.ipopt_obj = ipopt_obj_val
            result.ipopt_x = ipopt_x
        else:
            if ipopt_obj_val is not None:
                result.ipopt_obj = ipopt_obj_val
                print(f"IPOPT did not find counterexample. Best objective = {ipopt_obj_val}")
            else:
                print("IPOPT did not find any feasible solution.")

    return result


# ================================================================
# Root diagnostics and BaB budget estimation
# ================================================================

def run_root_diagnostics_and_budget(net, dm_l, dm_u, c, rhs, or_spec_size,
                                    nlp_config, nlp_best_x,
                                    root_peak_mem_bytes, root_total_mem_bytes,
                                    base_input_linear_constraints=None):
    """Estimate the BaB ALM budget.

    Mutates nlp_config to set 'bab_br_budget' when ALM-in-BaB is enabled.
    """
    # BaB budget estimation
    if nlp_config.get("in_bab", False) and bool(nlp_config.get("enable_alm", False)):
        from nlp_solver.alm import ALMConfig, resolve_alm_restart_count

        root_domains = int(dm_l.shape[0]) if dm_l is not None else 1
        root_cfg = ALMConfig.from_dict(nlp_config)
        root_restarts = resolve_alm_restart_count(root_domains, root_cfg)
        manual_br_budget = int(nlp_config.get("bab_br_budget", 0))
        if manual_br_budget > 0:
            br_budget = manual_br_budget
        else:
            br_budget = _estimate_bab_br_budget(
                nlp_config=nlp_config,
                root_domains=root_domains,
                root_restarts=root_restarts,
                root_peak_mem_bytes=root_peak_mem_bytes,
                total_mem_bytes=root_total_mem_bytes,
            )
        nlp_config["bab_br_budget"] = br_budget

        if manual_br_budget > 0:
            print(
                "[ALM-BaB] Shared B*R budget overridden by user: "
                f"budget={br_budget}"
            )
        elif root_peak_mem_bytes is not None and root_total_mem_bytes is not None:
            print(
                "[ALM-BaB] Shared B*R budget estimated from root GPU profile: "
                f"budget={br_budget}, root_br={root_domains * root_restarts}, "
                f"root_peak={root_peak_mem_bytes / (1024**3):.2f}GB, "
                f"total={root_total_mem_bytes / (1024**3):.2f}GB"
            )
        else:
            print(
                "[ALM-BaB] Shared B*R budget fallback from root restarts: "
                f"budget={br_budget}, root_br={root_domains * root_restarts}"
            )


# ================================================================
# BaB-loop ALM-PGD integration
# ================================================================

def init_alm_bab_state(nlp_config, warmstart_x=None, best_primal_x=None):
    """Create initial ALM BaB state from config."""
    interval = nlp_config.get("bab_interval", 5)
    return ALMBaBState(
        current_interval=interval,
        base_interval=interval,
        no_improve_count=0,
        warmstart_x=warmstart_x,
        best_primal_x=best_primal_x,
        share_start=time.time(),
    )


_MIN_ATTACK_TIME = 1.0  # s; an in-BaB attack with less left cannot finish a useful solve


def _clip_to_deadline(nlp_config, key, default, deadline):
    """nlp_config with its in-BaB budget `key` cut to the time left before `deadline`, or
    None when less than _MIN_ATTACK_TIME is left. An attack iteration (ALM ~12 s + IPOPT
    10-18 s) otherwise ignores the global limit: pooling_sppc0pq ended at 212 s of 180 s
    with its last iteration started at 159 s."""
    left = deadline - time.time()
    if left < _MIN_ATTACK_TIME:
        return None
    return dict(nlp_config, **{key: min(float(nlp_config.get(key, default)), left)})


def _scheduled_ipopt_in_bab(net, picked_batch, nlp_config, primal_solution, alm_state):
    """run_ipopt_in_bab, skipped after `ipopt_bab_max_fails` consecutive calls that found
    no better incumbent (pooling_adhya4tp: the first call found the optimum, the next
    ten cost most of the BaB budget). The count resets when a call improves."""
    if not nlp_config.get("enable_ipopt_in_bab", False):
        return None, None
    max_fails = int(nlp_config.get("ipopt_bab_max_fails", 2))
    if max_fails > 0 and alm_state.ipopt_bab_fails >= max_fails:
        return None, None
    nlp_config = _clip_to_deadline(nlp_config, "ipopt_bab_timeout", 3.0, alm_state.deadline)
    if nlp_config is None:
        return None, None
    x, obj = run_ipopt_in_bab(net, picked_batch, nlp_config, primal_solution)
    alm_state.ipopt_bab_fails = 0 if obj is not None else alm_state.ipopt_bab_fails + 1
    return x, obj


def run_ipopt_in_bab(net, picked_batch: PickedDomainBatch, nlp_config: Dict[str, Any], primal_solution: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[float]]:
    """Run IPOPT falsification in parallel on top k domains from the given batch."""
    from nlp_solver.ipopt_solver import ipopt_falsification

    enable_ipopt = nlp_config.get("enable_ipopt_in_bab", False)
    if not enable_ipopt:
        return None, None

    model_ori = net.model_ori
    batch_size = picked_batch.dm_l.shape[0]
    k = min(batch_size, int(nlp_config.get("ipopt_bab_topk", 2)))
    timeout = float(nlp_config.get("ipopt_bab_timeout", 3.0))

    if k <= 0:
        return None, None

    print(f"  Running IPOPT-in-BaB parallelly on top {k} domains (timeout={timeout}s)")

    # Temporarily override config for the inside-BaB run
    bab_ipopt_config = dict(nlp_config)
    bab_ipopt_config["ipopt_timeout"] = timeout
    bab_ipopt_config["ipopt_verbose"] = False
    
    # We will pick top k from the batch. The batch is typically sorted, so [0:k] are the worst (most promising) domains.
    futures = []
    # ipopt_falsification minimizes (objective row - rhs) and returns a point only when
    # that is negative, i.e. when the point beats the level rhs[0] (the incumbent, set
    # by run_bab_alm_step); the objective it reports is in those shifted units. Taken
    # as raw, a point 2.6e-5 below the incumbent 89.835 became incumbent -2.6e-5 and the
    # run ended "verified" (wastewater04m2; also hs065, cute_hs074, elena_gold, ex3_1_1,
    # st_e16). The shift is added back per point, and only returned points count.
    shift = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=k) as executor:
        for i in range(k):
            dm_l_i = picked_batch.dm_l[i]
            dm_u_i = picked_batch.dm_u[i]
            c_i = picked_batch.c[i]
            rhs_i = picked_batch.rhs[i]
            level = float(rhs_i.reshape(-1)[0])
            if not math.isfinite(level):
                continue    # no incumbent level: nothing to beat, and obj - inf is no problem
            
            # Slice the base input constraints for the i-th domain if present
            base_input_linear_constraints = None
            if picked_batch.base_input_linear_constraints is not None:
                A, b = picked_batch.base_input_linear_constraints
                base_input_linear_constraints = (A[i], b[i])

            fut = executor.submit(
                ipopt_falsification,
                net, model_ori, dm_l_i, dm_u_i, c_i, rhs_i,
                picked_batch.spec_size,
                bab_ipopt_config,
                base_input_linear_constraints
            )
            shift[fut] = level
            futures.append(fut)

    best_obj = float('inf')
    best_x = None

    for f in concurrent.futures.as_completed(futures):
        try:
            found_optimal, x_sol, obj_val = f.result()
            if not found_optimal or x_sol is None or obj_val is None:
                continue
            raw_obj = float(obj_val) + shift[f]
            if raw_obj < best_obj:
                best_obj = raw_obj
                best_x = x_sol
        except Exception as e:
            print(f"  [IPOPT-in-BaB] Task Exception: {e}")

    # Check if we improved over the current primal solution globally
    # Primal solution might be a tensor like tensor([3.14])
    primal_scalar = primal_solution.item() if isinstance(primal_solution, torch.Tensor) else float(primal_solution)
    if best_obj < primal_scalar:
        print(f"  [IPOPT-in-BaB] Found new primal incumbent: {best_obj:.6f} < {primal_scalar:.6f}")
        return best_x, best_obj
    else:
        print(f"  [IPOPT-in-BaB] No improvement found. Best local obj: {best_obj:.6f}")
        return None, None


def run_bab_alm_step(net, domains, picked_batch: PickedDomainBatch, nlp_config,
                     primal_solution, primal_solution_found,
                     alm_state, stats, num_iter, solving_mode):
    """Run ALM-PGD attack on the picked domains if it is time.

    Checks interval timing internally.  Returns immediately if not scheduled.
    Mutates *alm_state* in-place (warmstart_x, best_primal_x, interval state).

    Returns:
        (primal_solution, primal_solution_found) — possibly updated.
    """
    if (
            not nlp_config.get("in_bab", False)
            or not bool(nlp_config.get("enable_alm", False))
            or not solving_mode):
        return primal_solution, primal_solution_found
    if num_iter % alm_state.current_interval != 0:
        return primal_solution, primal_solution_found
    if _alm_over_time_share(alm_state, nlp_config):
        return primal_solution, primal_solution_found
    if _alm_given_up(alm_state, nlp_config, primal_solution, picked_batch.objective_lower_bound):
        return primal_solution, primal_solution_found

    nlp_config = _clip_to_deadline(
        nlp_config, "bab_timeout", min(5.0, float(nlp_config.get("timeout", 5.0))),
        alm_state.deadline)
    if nlp_config is None:
        return primal_solution, primal_solution_found

    from .attack_in_input_split import alm_attack_on_batch

    stats.timer.start("alm_attack")
    t_alm_start = time.time()
    print(f"  Running ALM-PGD attack (interval={alm_state.current_interval})")

    if arguments.Config['model']['with_jacobian']:
        model_to_attack = net.net
    else:
        model_to_attack = net.model_ori

    if picked_batch.attack_counter is not None:
        picked_batch.attack_counter.add_(1)

    dual_cut_linear_constraints = _select_alm_dual_cut_constraints(
        picked_batch,
        nlp_config,
        history=domains.history if hasattr(domains, 'history') else None,
    )

    alm_obj, alm_raw_obj, alm_x = alm_attack_on_batch(
        model_to_attack,
        picked_batch.dm_l,
        picked_batch.dm_u,
        picked_batch.c,
        picked_batch.rhs,
        picked_batch.spec_size,
        history=domains.history if hasattr(domains, 'history') else None,
        nlp_config=nlp_config,
        x_warm=alm_state.warmstart_x,
        base_input_linear_constraints=picked_batch.base_input_linear_constraints,
        dual_cut_linear_constraints=dual_cut_linear_constraints,
        incumbent=primal_solution,
    )

    alm_improved = False
    if alm_obj is not None and not alm_obj.isinf():
        print(f"  ALM-PGD found obj={alm_obj.item():.6f}")
        print(f"  ALM-PGD raw obj={alm_raw_obj.item():.6f}")

        primal_solution, alm_state.best_primal_x, alm_state.warmstart_x, improved = _update_primal_and_warmstart(
            primal_solution,
            alm_state.best_primal_x,
            alm_state.warmstart_x,
            alm_shifted_obj=alm_obj,
            alm_raw_obj=alm_raw_obj,
            alm_x=alm_x,
        )
        alm_improved = improved

        # Write best primal x* to history for gradient-guided branching (LGB).
        if (alm_state.best_primal_x is not None
                and hasattr(domains, 'history')
                and domains.history is not None):
            domains.history.alm_best_primal_x = alm_state.best_primal_x.detach().cpu()

        if improved and not primal_solution_found:
            domains.sort_heuristic = "lb_only"
            domains.sort_index = 0
            primal_solution_found = True
        primal_scalar = torch.as_tensor(
            primal_solution,
            device=picked_batch.rhs.device,
            dtype=picked_batch.rhs.dtype,
        ).reshape(-1)[0]
        if torch.isfinite(primal_scalar):
            picked_batch.rhs[:, 0] = primal_scalar
    else:
        print(f"  ALM-PGD did not find feasible solution")

    # Run parallel IPOPT heuristic right after ALM
    ipopt_x, ipopt_obj = _scheduled_ipopt_in_bab(
        net, picked_batch, nlp_config, primal_solution, alm_state)
    if ipopt_obj is not None:
        alm_improved = True
        primal_solution = torch.tensor([ipopt_obj], device=primal_solution.device, dtype=primal_solution.dtype)
        if ipopt_x is not None:
            alm_state.best_primal_x = ipopt_x.detach().clone()
            
            # Write best primal x* to history for LGB.
            if hasattr(domains, 'history') and domains.history is not None:
                domains.history.alm_best_primal_x = alm_state.best_primal_x.cpu()

        if not primal_solution_found:
            domains.sort_heuristic = "lb_only"
            domains.sort_index = 0
            primal_solution_found = True
        
        primal_scalar = primal_solution.reshape(-1)[0]
        if torch.isfinite(primal_scalar):
            picked_batch.rhs[:, 0] = primal_scalar

    # Adaptive interval: increase on no improvement, reset on improvement
    if nlp_config.get("bab_adaptive_interval", True):
        bab_interval_patience = nlp_config.get("bab_interval_patience", 3)
        bab_max_interval = nlp_config.get("bab_max_interval", 20)
        if alm_improved:
            alm_state.current_interval = alm_state.base_interval
            alm_state.no_improve_count = 0
        else:
            alm_state.no_improve_count += 1
            if alm_state.no_improve_count >= bab_interval_patience:
                old_interval = alm_state.current_interval
                alm_state.current_interval = min(alm_state.current_interval * 2, bab_max_interval)
                alm_state.no_improve_count = 0
                if alm_state.current_interval != old_interval:
                    print(f"  [ALM-adaptive] No improvement for {bab_interval_patience} calls, "
                          f"interval {old_interval} -> {alm_state.current_interval}")

    alm_state.fail_streak = 0 if alm_improved else alm_state.fail_streak + 1
    alm_state.share_alm_time += time.time() - t_alm_start
    if alm_improved:
        alm_state.share_start = time.time()
        alm_state.share_alm_time = 0.0

    if nlp_config.get("bab_alm_release_cache", False) and torch.cuda.is_available():
        # AutoBatchSize enlarges the BaB batch only while torch.cuda.memory_reserved
        # leaves room, and the ALM's restart batches stay reserved in the allocator
        # cache after the call: on cute_hs119 the reserved memory went 0.3 -> 11.9 GB
        # over four ALM calls while the BaB itself needed ~2 GB, and the batch stopped
        # at 16384. Release the cache, as is already done after the root NLP stage.
        torch.cuda.empty_cache()

    stats.timer.add("alm_attack")
    return primal_solution, primal_solution_found


def _alm_given_up(alm_state, nlp_config, primal_solution, objective_lower_bound) -> bool:
    """True when a scheduled in-BaB ALM call is skipped by `bab_alm_stop_after_failures`.

    After that many consecutive calls without a better incumbent, calls run only while the
    relative gap (P - D) / (1 + |P|) is below `bab_alm_stop_gap`, with D the lowest bound of
    the picked domains (>= the global bound, so the gap is never overstated). Over the
    final_596a871 / final_51a2c25 / migr3 logs no in-BaB improvement came after 4
    consecutive failures (0 of 245 calls), while the calls took a median 30-37% of the BaB
    time. Without an incumbent nothing is skipped. Only delays calls, like the time share.
    """
    k = int(nlp_config.get("bab_alm_stop_after_failures", 0) or 0)
    if k <= 0 or alm_state.fail_streak < k or objective_lower_bound is None:
        return False
    p = float(torch.as_tensor(primal_solution).reshape(-1)[0])
    lb = torch.as_tensor(objective_lower_bound)
    if not math.isfinite(p) or lb.numel() == 0:
        return False
    d = float(lb.min())
    gap = (p - d) / (1.0 + abs(p)) if math.isfinite(d) else math.inf
    return not gap < float(nlp_config.get("bab_alm_stop_gap", 0.01))


def _alm_over_time_share(alm_state, nlp_config) -> bool:
    """True when a scheduled in-BaB ALM call must be skipped by `bab_alm_max_time_share`.

    The iteration-count cadence alone lets one ALM call (up to `bab_timeout`, 10 s) come
    every `bab_max_interval` iterations however fast the iterations are: on
    kriging_peaks-full010 the ALM took 57 of the 67 s of BaB and never improved. The
    cap only delays calls; it never changes what a call computes.
    """
    share = float(nlp_config.get("bab_alm_max_time_share", 0.0) or 0.0)
    if not 0.0 < share < 1.0 or alm_state.share_start is None:
        return False
    window = time.time() - alm_state.share_start
    return alm_state.share_alm_time > share * window
