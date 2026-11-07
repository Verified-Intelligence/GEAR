"""Root-stage primal search: IPOPT and the GPU ALM, compared on one NLP."""

import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Callable
import torch

from .utils import _eval_mode, _split_or_clauses, is_cuda_oom
from .problem import NNProblemSingleDomain
from .alm import ALMConfig, ALMPGDSolver, resolve_alm_restart_count
from .statistics import verify_counterexample
from .ipopt_solver import IPOPT_AVAILABLE, NNProblemIPOPT, IPOPTSolver


_LOCAL_SOLVER_LABEL = 'IPOPT'


@dataclass
class LocalSolverRunResult:
    """Result from the selected root local solver."""
    success: bool
    objective: Optional[float]
    violation: Optional[float]
    time_sec: float
    verified: bool
    status: str
    x: Optional[torch.Tensor]


def _run_local_solver(
    model_ori: Callable,
    dm_l: torch.Tensor,
    dm_u: torch.Tensor,
    C_obj: torch.Tensor,
    rhs_obj: torch.Tensor,
    C_constr: torch.Tensor,
    rhs_constr: torch.Tensor,
    config: Dict[str, Any],
    verbose: bool,
    feas_tol: float,
    base_input_linear_constraints: Optional[Tuple[torch.Tensor, torch.Tensor]],
    extra_x0: Optional[torch.Tensor] = None,
) -> LocalSolverRunResult:
    """Run the configured root local solver and verify its point.

    With `extra_x0`, IPOPT is also started from that point and the better result is
    kept (feasible before infeasible, then lower objective, then lower violation).
    """
    if str(config.get('method', 'ipopt')).strip().lower() != 'ipopt':
        raise ValueError(f"Unsupported root local solver: {config.get('method')}")
    if not IPOPT_AVAILABLE:
        return LocalSolverRunResult(
            success=False,
            objective=None,
            violation=None,
            time_sec=0.0,
            verified=False,
            status="No IPOPT backend available",
            x=None,
        )

    ipopt_problem = NNProblemIPOPT(
        model_ori, dm_l, dm_u,
        C_obj, rhs_obj, C_constr, rhs_constr,
        input_constr_A=None if base_input_linear_constraints is None else base_input_linear_constraints[0],
        input_constr_b=None if base_input_linear_constraints is None else base_input_linear_constraints[1],
        device=None, dtype=None,
    )

    local_config = dict(config)
    ipopt_solver = IPOPTSolver(local_config)

    start_time = time.time()
    ipopt_timeout = local_config.get('ipopt_timeout', local_config.get('timeout', 60.0))
    result = ipopt_solver.solve_single(ipopt_problem, timeout=ipopt_timeout)
    publish = config.get('_controller_publish')
    published_seconds = 0.0
    if publish is not None:
        point = None
        if result.x_solution is not None:
            point = torch.as_tensor(result.x_solution, device=dm_l.device, dtype=dm_l.dtype).view(dm_l.shape)
            ok, _, _ = verify_counterexample(
                model_ori, point, C_obj, rhs_obj, C_constr, rhs_constr,
                input_constr_A=None if base_input_linear_constraints is None else base_input_linear_constraints[0],
                input_constr_b=None if base_input_linear_constraints is None else base_input_linear_constraints[1],
                tol=1e-6)
            if not ok:
                point = None
        published_seconds = time.time() - start_time
        publish('ipopt', point, published_seconds)
    # The second start shares the one IPOPT wall budget: a single Ipopt iteration
    # cannot be interrupted, and on large dense problems one takes tens of seconds.
    remaining = float(ipopt_timeout) - (time.time() - start_time)
    if (extra_x0 is not None and remaining > 0.2 * float(ipopt_timeout)
            and not (config.get('_controller_stop') is not None and config['_controller_stop'].is_set())):
        alt = ipopt_solver.solve_single(
            ipopt_problem,
            x_init=extra_x0.detach().reshape(-1).cpu().double().numpy(),
            timeout=remaining)

        def _rank(r):
            feasible = r.x_solution is not None
            obj = r.objective_value if r.objective_value is not None else float('inf')
            viol = r.constraint_violation if r.constraint_violation is not None else float('inf')
            return (0 if feasible else 1, obj if feasible else viol)

        if _rank(alt) < _rank(result):
            if verbose:
                print(f"  [{_LOCAL_SOLVER_LABEL}] LP-point start kept: obj={alt.objective_value} "
                      f"viol={alt.constraint_violation} (center start: "
                      f"obj={result.objective_value} viol={result.constraint_violation})")
            result = alt
    elapsed = time.time() - start_time
    x = None
    verified = False
    if result.x_solution is not None:
        x = torch.from_numpy(result.x_solution).to(
            device=dm_l.device, dtype=dm_l.dtype
        ).view(dm_l.shape)
    success = x is not None
    obj = result.objective_value
    viol = result.constraint_violation
    status = result.status

    if verbose:
        print(f"  [{_LOCAL_SOLVER_LABEL}] Solver status: {status}")

    if x is not None:
        verified, v_obj, v_viol = verify_counterexample(
            model_ori, x,
            C_obj, rhs_obj, C_constr, rhs_constr,
            input_constr_A=None if base_input_linear_constraints is None else base_input_linear_constraints[0],
            input_constr_b=None if base_input_linear_constraints is None else base_input_linear_constraints[1],
            tol=feas_tol,
        )
        if obj is None:
            obj = v_obj
        if viol is None:
            viol = v_viol
        if verbose:
            print(f"  [{_LOCAL_SOLVER_LABEL}] Verification: valid={verified}, obj={v_obj:.6f}, viol={v_viol:.2e}")
        if not verified:
            obj = v_obj
            viol = v_viol

    if publish is not None:
        publish('ipopt', x if verified else None, max(0.0, time.time() - start_time - published_seconds))
    return LocalSolverRunResult(
        success=success,
        objective=obj,
        violation=viol,
        time_sec=elapsed,
        verified=verified,
        status=status,
        x=x,
    )



@dataclass
class ComparisonResult:
    """Result of comparing the root local solver (IPOPT) and ALM-PGD."""
    local_method: str                      # Human-readable local solver name
    local_status: Optional[str]            # Root local solver status string
    alm_warmstart_source: Optional[str]    # Warm-start source used by ALM

    # Local solver results
    local_success: bool
    local_obj: Optional[float]
    local_viol: Optional[float]
    local_time: float
    local_verified: bool                    # Did the local solution pass verification?

    # ALM-PGD results
    alm_success: bool
    alm_obj: Optional[float]
    alm_viol: Optional[float]
    alm_time: float
    alm_verified: bool                      # Did ALM solution pass verification?
    alm_outer_iter: int
    alm_final_rho: float

    # Best overall
    best_method: str                        # 'ipopt', 'alm', or 'none'
    best_obj: Optional[float]
    best_x: Optional[torch.Tensor]

    def __str__(self) -> str:
        """Pretty print comparison results."""
        lines = [
            "=" * 70,
            "NLP Solver Comparison Results",
            "=" * 70,
            "",
            f"Local Solver ({self.local_method}):",
            f"  Success: {self.local_success}",
            f"  Objective: {self.local_obj:.6f}" if self.local_obj is not None else "  Objective: None",
            f"  Violation: {self.local_viol:.2e}" if self.local_viol is not None else "  Violation: None",
            f"  Time: {self.local_time:.2f}s",
            f"  Verified: {self.local_verified}",
            f"  Status: {self.local_status}" if self.local_status is not None else "  Status: None",
            "",
            "ALM-PGD:",
            f"  Success: {self.alm_success}",
            f"  Objective: {self.alm_obj:.6f}" if self.alm_obj is not None else "  Objective: None",
            f"  Violation: {self.alm_viol:.2e}" if self.alm_viol is not None else "  Violation: None",
            f"  Time: {self.alm_time:.2f}s",
            f"  Verified: {self.alm_verified}",
            f"  Outer iterations: {self.alm_outer_iter}",
            f"  Final rho: {self.alm_final_rho:.1f}",
            f"  Warm-start source: {self.alm_warmstart_source or 'None'}",
            "",
            f"Best method: {self.best_method}",
            f"Best objective: {self.best_obj:.6f}" if self.best_obj is not None else "Best objective: None",
            "=" * 70,
        ]
        return "\n".join(lines)


def _ends_with_structural_rows(C_constr, rhs_constr, input_constraints) -> bool:
    """Whether C_constr ends with the rows y_{1+j} <= 0 that the proof-time expansion
    appends for the input constraints. A spec without output rows is never expanded
    (28 linear-only instances, e.g. ex2_1_*, ex6_2_*, st_*, ran the root ALM with 0
    constraints and returned points that violate the linear rows by up to 1e3)."""
    n = 0 if input_constraints is None else int(input_constraints[0].shape[-2])
    if n == 0:
        return True
    tail = C_constr[-n:]
    return (
        tuple(tail.shape) == (n, C_constr.shape[-1])
        and bool(torch.equal(tail[:, 1:1 + n], torch.eye(n, dtype=tail.dtype, device=tail.device)))
        and not tail[:, 1 + n:].any()
        and not tail[:, :1].any()
        and not rhs_constr[-n:].any()
    )


def compare_solvers(
    _net,  # Unused but kept for API compatibility
    model_ori: Callable,
    dm_l: torch.Tensor,
    dm_u: torch.Tensor,
    C_mat: torch.Tensor,
    rhs_mat: torch.Tensor,
    or_spec_size: torch.Tensor,
    reference_dict: Optional[Dict] = None,  # pylint: disable=unused-argument
    config: Optional[Dict] = None,
    base_input_linear_constraints: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    dual_mu_init: Optional[torch.Tensor] = None,
    local_extra_x0: Optional[torch.Tensor] = None,
) -> ComparisonResult:
    """
    Run the local solver (IPOPT) and ALM-PGD and compare results.

    This function:
    1. Runs the local IPOPT solver
    2. Runs ALM-PGD solver
    3. Verifies both solutions against the actual network
    4. Returns detailed comparison

    Returns:
        ComparisonResult with results from both solvers
    """
    if config is None:
        config = {}

    verbose = config.get('verbose', True)  # Default to verbose for comparison
    feas_tol = config.get('constr_viol_tol', 1e-8)
    enable_local_solver = bool(config.get('enable_local_solver', False))
    local_method_label = _LOCAL_SOLVER_LABEL

    # Use first OR clause for comparison (comparison is for debugging, not full OR-spec support)
    clauses = list(_split_or_clauses(C_mat, rhs_mat, or_spec_size))
    C_obj, rhs_obj, C_constr, rhs_constr = clauses[0]

    # Proof-time structural augmentation already inserts exact A x - b rows
    # into C_constr. Optionally avoid appending the same input constraints a
    # second time inside the primal heuristic.
    alm_input_constraints = base_input_linear_constraints
    if (
        config.get("expand_input_constraints_as_outputs", False)
        and config.get("alm_deduplicate_expanded_input_constraints", False)
        and _ends_with_structural_rows(C_constr, rhs_constr, base_input_linear_constraints)
    ):
        alm_input_constraints = None

    # Create single problem with dtype inference from model
    single_problem = NNProblemSingleDomain(
        model_ori, dm_l, dm_u,
        C_obj, rhs_obj, C_constr, rhs_constr,
        input_constr_A=None if alm_input_constraints is None else alm_input_constraints[0],
        input_constr_b=None if alm_input_constraints is None else alm_input_constraints[1],
        initialization_constr_A=(
            None if base_input_linear_constraints is None
            else base_input_linear_constraints[0]
        ),
        initialization_constr_b=(
            None if base_input_linear_constraints is None
            else base_input_linear_constraints[1]
        ),
        device=None, dtype=None  # Infer from model
    )

    print("=" * 70)
    print("Running NLP Solver Comparison")
    print(f"Problem: {dm_l.numel()} vars, {single_problem.n_constraints} constraints")
    print(f"Local solver: {local_method_label}")
    print("=" * 70)

    parallel = config.get('root_parallel', False) and enable_local_solver and IPOPT_AVAILABLE
    # The verified local point defines ALM's improvement target and progress test.
    unified = bool(config.get('unified_root', False)) and not config.get('controller', False)
    publish = config.get('_controller_publish')

    def _run_alm(x_init=None, rival=None):
        """Run ALM-PGD with the root's target-level progress test."""
        alm_cfg = ALMConfig.from_dict(config)
        alm_cfg.verbose = verbose
        # Disable early stopping at root — use full budget for best solution
        alm_cfg.obj_patience = -1
        alm_cfg.inner_plateau_check = 0
        solver = ALMPGDSolver(alm_cfg)
        # Root-level dual→ALM mu warm-start. Only used when the first-OR clause's
        # constraint count matches the dual-ascent lam_star length (output constraints
        # followed by optional input constraints).
        mu_init_to_use = None
        if dual_mu_init is not None:
            expected = single_problem.n_constraints
            if dual_mu_init.shape == (expected,):
                mu_init_to_use = dual_mu_init.to(device=dm_l.device, dtype=single_problem.dtype)
            elif verbose:
                print(f"  [Root] dual_mu_init shape {tuple(dual_mu_init.shape)} != "
                      f"n_constraints {expected}; skipping mu warm-start")
        start = time.time()
        if config.get('_controller_alm_start') is not None:
            config['_controller_alm_start']()
        num_restarts = None
        timeout = None
        while True:
            if timeout is not None and timeout <= 0:
                result = solver._build_invalid_result(single_problem, start, "ALM retry deadline")
                break
            retry = False
            try:
                result = solver.solve_with_restarts(
                    single_problem, num_restarts=num_restarts, timeout=timeout,
                    x_init=x_init, mu_init=mu_init_to_use, rival=rival,
                )
                break
            except RuntimeError as error:
                # Large models (arki*, pooling_spp*) cannot hold every restart on
                # the GPU. Fewer restarts only narrow the primal search; retry
                # with half of them in the remaining time.
                current = num_restarts or resolve_alm_restart_count(
                    single_problem.batch_size, alm_cfg)
                if not is_cuda_oom(error) or current <= 1:
                    raise
                num_restarts = current // 2
                timeout = max(0.0, float(alm_cfg.timeout) - (time.time() - start))
                retry = True
                print(f"  [ALM] {current} restarts ran out of GPU memory; "
                      f"retrying with {num_restarts}")
            if retry:
                # The exception traceback owns the failed restart tensors.
                # Release it before returning cached memory to the allocator.
                torch.cuda.empty_cache()
        elapsed = time.time() - start
        verified = False
        x = None
        obj = result.objective_value
        viol = result.constraint_violation
        if result.x_solution is not None:
            x = result.x_solution.view(dm_l.shape)
            verified, v_obj, v_viol = verify_counterexample(
                model_ori, x,
                C_obj, rhs_obj, C_constr, rhs_constr,
                input_constr_A=None if base_input_linear_constraints is None else base_input_linear_constraints[0],
                input_constr_b=None if base_input_linear_constraints is None else base_input_linear_constraints[1],
                tol=feas_tol,
            )
            if verbose:
                print(f"  [ALM] Verification: valid={verified}, "
                      f"obj={v_obj:.6f}, viol={v_viol:.2e}")
            if not verified:
                obj = v_obj
                viol = v_viol
        if publish is not None:
            elapsed = time.time() - start
        return result, x, elapsed, obj, viol, verified

    # Wrap with eval mode for stable BN/Dropout
    with _eval_mode(model_ori):
        local_x = None
        local_time = 0.0
        local_success = False
        local_obj = None
        local_viol = None
        local_verified = False
        local_status = None
        alm_warmstart_source = None

        alm_result = None
        alm_x = None
        alm_time = 0.0
        alm_success = False
        alm_obj = None
        alm_viol = None
        alm_verified = False
        alm_outer_iter = None
        alm_final_rho = None

        if parallel:
            # =================================================================
            # Parallel mode: selected local solver in a CPU thread + ALM on GPU.
            # IPOPT native work releases the GIL, so it can overlap with
            # PyTorch GPU ops.
            # =================================================================
            from concurrent.futures import ThreadPoolExecutor
            print(f"\n[Root] Running {local_method_label} + ALM-PGD in parallel...")

            # The local solver runs beside ALM and the root waits for ALM anyway, so it
            # gets ALM's whole wall budget instead of its own shorter cap. On
            # pooling_foulds3stp Ipopt needs ~30.2 s: its 30 s cap decided between a
            # point that certifies (-8.0166) and one that never closes (-8.0084).
            local_config = dict(config, ipopt_timeout=max(
                float(config.get('ipopt_timeout', 0.0)), float(config.get(
                    '_controller_parallel_ipopt_budget', config.get('timeout', 60.0)))))
            with ThreadPoolExecutor(max_workers=1) as executor:
                local_future = executor.submit(
                    _run_local_solver,
                    model_ori, dm_l, dm_u,
                    C_obj, rhs_obj, C_constr, rhs_constr,
                    local_config, verbose, feas_tol,
                    base_input_linear_constraints,
                    local_extra_x0,
                )
                def rival():
                    """The local solver's verified objective once it has returned, else None."""
                    if not local_future.done() or local_future.exception() is not None:
                        return None
                    local = local_future.result()
                    return local.objective if local.verified else None

                # ALM starts immediately (no warm start in parallel mode)
                (alm_result, alm_x, alm_time, alm_obj,
                 alm_viol, alm_verified) = _run_alm(x_init=None, rival=rival if unified else None)
                # Collect the local solver result
                local_result = local_future.result()
                local_x = local_result.x
                local_time = local_result.time_sec
                local_success = local_result.success
                local_obj = local_result.objective
                local_viol = local_result.violation
                local_verified = local_result.verified
                local_status = local_result.status

            alm_success = alm_result.success
            alm_outer_iter = alm_result.outer_iterations
            alm_final_rho = alm_result.final_rho
            print(f"  [Root] Parallel done: {local_method_label} {local_time:.2f}s, ALM {alm_time:.2f}s")

        else:
            # =================================================================
            # Sequential mode: local solver first -> warm-start ALM
            # =================================================================
            if enable_local_solver:
                print(f"\n[1/2] Running Local Solver ({local_method_label})...")
                local_result = _run_local_solver(
                    model_ori, dm_l, dm_u,
                    C_obj, rhs_obj, C_constr, rhs_constr,
                    config, verbose, feas_tol,
                    base_input_linear_constraints,
                    local_extra_x0,
                )
                local_x = local_result.x
                local_time = local_result.time_sec
                local_success = local_result.success
                local_obj = local_result.objective
                local_viol = local_result.violation
                local_verified = local_result.verified
                local_status = local_result.status

            if not enable_local_solver:
                print("\n[1/1] Running ALM-PGD...")
            else:
                print("\n[2/2] Running ALM-PGD...")
            # Warm-start ALM from a verified local solver point when available.
            alm_x_init = local_x if local_verified else None
            alm_warmstart_source = local_method_label if local_verified else None
            if alm_warmstart_source is not None:
                print(f"  [Root] ALM warm-start source: {alm_warmstart_source}")
            (alm_result, alm_x, alm_time, alm_obj,
             alm_viol, alm_verified) = _run_alm(
                x_init=alm_x_init, rival=(lambda: local_obj) if unified and local_verified else None)

            alm_success = alm_result.success
            alm_outer_iter = alm_result.outer_iterations
            alm_final_rho = alm_result.final_rho

    # =========================================================================
    # Determine best method
    # =========================================================================
    best_method = 'none'
    best_obj = None
    best_x = None

    # Prefer verified solutions
    if local_verified and alm_verified:
        if local_obj <= alm_obj:
            best_method = 'ipopt'
            best_obj = local_obj
            best_x = local_x
        else:
            best_method = 'alm'
            best_obj = alm_obj
            best_x = alm_x
    elif local_verified:
        best_method = 'ipopt'
        best_obj = local_obj
        best_x = local_x
    elif alm_verified:
        best_method = 'alm'
        best_obj = alm_obj
        best_x = alm_x
    elif not config.get('controller', False):
        # Neither verified - report best feasible
        if local_viol is not None and alm_viol is not None:
            if local_viol <= feas_tol and (alm_viol > feas_tol or local_obj <= alm_obj):
                best_method = 'ipopt'
                best_obj = local_obj
                best_x = local_x
            elif alm_viol <= feas_tol:
                best_method = 'alm'
                best_obj = alm_obj
                best_x = alm_x

    if publish is not None:
        publish('alm', alm_x if alm_verified else None, alm_time)

    result = ComparisonResult(
        local_method=local_method_label,
        local_status=local_status,
        alm_warmstart_source=alm_warmstart_source,
        local_success=local_success,
        local_obj=local_obj,
        local_viol=local_viol,
        local_time=local_time,
        local_verified=local_verified,
        alm_success=alm_success,
        alm_obj=alm_obj,
        alm_viol=alm_viol,
        alm_time=alm_time,
        alm_verified=alm_verified,
        alm_outer_iter=alm_outer_iter,
        alm_final_rho=alm_final_rho,
        best_method=best_method,
        best_obj=best_obj,
        best_x=best_x,
    )

    print("\n" + str(result))

    return result
