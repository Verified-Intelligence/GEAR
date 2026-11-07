"""ALM-PGD Solver for neural network optimization.

Based on Hestenes-Powell-Rockafellar (HPR) Augmented Lagrangian formulation.
Reference: arXiv:2510.16827v1
"""

import time
import math
from dataclasses import dataclass, fields
from typing import Callable, Optional, Dict, Any, Tuple, Union
import torch

from .utils import _eval_mode, VectorLinearLR, check_deadline, DeadlineExceeded
from .problem import NNProblemSingleDomain
from .continuation import Atom, candidate, retained, invoke
from attack.attack_utils import AdamClipping


def _phi_and_residuals(problem, x, out, B, R):
    """Objective (N,) and raw constraint residuals (N, m) of the N = B*R points ``x``.

    Without input rows one batched matmul gives both; with input rows the residuals
    are stacked [output, input] by the problem.
    """
    if problem.n_input_constraints == 0:
        return problem.compute_phi_and_output_residuals(out, B, R)
    return (problem.compute_phi(x, out, B, R),
            problem.compute_constraint_residuals(x, out, B, R))


def _certificate_residual(phi, viol, rival, abs_tol, rel_tol, feas_tol):
    """Joint residual for a feasible improvement at certificate resolution.

    Invalid restarts have no proposal; they must not poison valid restarts' minimum.
    """
    tol = max(abs_tol, rel_tol * max(1., abs(rival)), torch.finfo(phi.dtype).eps)
    residual = torch.maximum(
        viol / max(feas_tol, torch.finfo(phi.dtype).tiny),
        1. + torch.relu(phi - (rival - tol)) / tol).clamp(min=1.)
    return torch.where(torch.isfinite(phi) & torch.isfinite(viol), residual,
                       torch.full_like(residual, float('inf')))


class _RivalProgress:
    """Amortize stalled search against work that improved a target-level residual.

    A residual of one means feasible and better than the rival by the root gap
    tolerance. Infeasible points can earn progress, including before the first
    feasible point and through temporary regressions in continuation.
    Work is the number of completed inner steps, not host wall seconds. A restored
    feasible point within the certificate tolerance exhausts this basin's credit.
    """
    def __init__(self):
        self.rival = None

    def stop(self, rival, residual, work, remaining, initial_residual=None):
        if not math.isfinite(residual):
            return False  # invalid values cannot establish a contraction rate
        score = math.log(max(1., residual))
        if initial_residual is not None and not math.isfinite(initial_residual):
            initial_residual = None
        if rival != self.rival:
            self.rival = rival
            self.initial = (score if initial_residual is None else
                            math.log(max(1., initial_residual)))
            self.best = self.initial
            self.origin = work if initial_residual is None else 0.
            self.productive = self.origin
            if initial_residual is None:
                return False  # one sample cannot establish a rate
        improved = score < self.best - math.ulp(max(1., abs(self.best)))
        if improved:
            self.best, self.productive = score, work
        if self.best == 0.:
            return False  # ALM has a better basin; its own progress rules apply
        elapsed = max(0., work - self.origin)
        gain = self.initial - self.best
        # A full productive episode's service is available for a temporary
        # regression; records accumulate credit rather than requiring each
        # iteration to beat the incumbent anew.
        stalled = work - self.productive
        productive = self.productive - self.origin
        if stalled > productive:
            return True
        return gain <= 0. or elapsed * self.best > remaining * gain


def _format_alm_verbose_status(
    k,
    local_min_viol,
    local_obj_at_min_viol,
    local_domain,
    local_restart,
    rho,
    eps,
    eta,
    grad,
    mu_norm,
    feas_count,
    total_count,
    good_count,
    bad_count,
    iter_best_feas_obj=None,
    global_best_feas_obj=None,
):
    """Build an unambiguous ALM iteration log message."""
    iter_best_text = "NA" if iter_best_feas_obj is None else f"{iter_best_feas_obj:.4f}"
    global_best_text = "NA" if global_best_feas_obj is None else f"{global_best_feas_obj:.4f}"
    return (
        f"    [ALM-PGD] k={k}: local_min_viol={local_min_viol:.2e} "
        f"local_obj_at_min_viol={local_obj_at_min_viol:.4f} "
        f"local_domain={local_domain} local_restart={local_restart} "
        f"rho={rho:.3g} eps={eps:.2e} eta={eta:.2e} "
        f"grad={grad:.2e} ||mu||={mu_norm:.2e} "
        f"feas={feas_count}/{total_count} good={good_count} bad={bad_count} "
        f"iter_best_feas_obj={iter_best_text} global_best_feas_obj={global_best_text}"
    )

class AdamClipping_NLP(AdamClipping):
    """Adam for the ALM inner solve: the update ``lr * m / denom`` is clipped per
    coordinate to ``lr_vec`` (a fraction of the box width) and projected onto the box.

    Unlike ``AdamClipping`` it keeps the second-moment information (an elementwise
    clip instead of ``sign``) and uses the raw ``lr`` rather than the bias-corrected
    step, which would make the first steps ~10x larger than intended.
    """

    def __init__(self, params, lr, lr_vec, scale_eps_with_bias=False):
        """``scale_eps_with_bias``: divide the clip cap by the first-moment bias
        correction, allowing larger steps early on."""
        super().__init__(params, lr=lr, lr_vec=lr_vec)
        self.scale_eps_with_bias = scale_eps_with_bias

    @torch.no_grad()
    def step(self, lower_limit, upper_limit):
        """One descent step on the single parameter, clipped and projected onto [lower, upper]."""
        (group,) = self.param_groups
        (p,) = group['params']
        grad = p.grad
        state = self.state[p]
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']
        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

        max_step = group['lr_vec']
        if self.scale_eps_with_bias:
            max_step = max_step / bias_correction1
        update = -(exp_avg / denom) * group['lr']
        update = update.clamp(min=-max_step, max=max_step)
        p.copy_((p.data + update).clamp(min=lower_limit, max=upper_limit))


@dataclass
class ALMConfig:
    """Configuration for ALM-PGD solver.

    ``from_dict`` reads each field from ``alm_<field>`` (see the exceptions there).
    """
    # Outer loop parameters (Practical ALM - Algorithm 1)
    rho0: float = 1.0                      # Initial penalty parameter (higher for faster feasibility)
    alpha: float = 0.5                      # Alpha parameter for tolerance schedule
    beta: float = 0.8                       # Beta parameter for tolerance schedule
    kappa: float = 1.1                      # Penalty increase factor
    max_rho: float = 1e6                    # Maximum penalty parameter
    outer_max_iter: int = 200               # Maximum outer iterations (Practical ALM needs more)
    eps_final: float = 1e-5      # Final epsilon (1/rho)
    eta_final: float = 1e-5      # Final eta
    feas_tol: float = 1e-5       # Feasibility tolerance (tightened from 1e-5 to fix unsoundness) for counterexample
    obj_patience: int = 3        # Early stop if best_obj doesn't improve for n iters (-1 to disable)
    obj_improvement_tol: float = 1e-4  # Absolute tolerance for objective improvement detection
    obj_gap_threshold: float = 0.0     # Min gap between best and 2nd-best domain obj for early stop (0=disabled)
    first_feas_patience: int = 1       # Max extra outer iters after ALL batch items first become feasible (-1=disable)

    # Inner loop parameters (PGD/Adam)
    inner_steps: int = 100                  # Steps per inner solve
    inner_lr: float = 0.01                  # Learning rate for Adam optimizer
    inner_lr_outer_decay: float = 1.0       # Per-outer step-resolution continuation
    inner_lr_min_factor: float = 1e-6       # Minimum factor applied to inner_lr
    inner_plateau_check: int = 10           # Check L_val plateau every N inner steps (0=disabled)
    inner_plateau_rtol: float = 1e-3        # Relative tolerance for L_val plateau
    inner_plateau_patience: int = 2         # Consecutive plateau checks before inner early stop
    scale_eps_with_bias: bool = True         # Legacy first-step clipping-cap expansion

    # Primal-restoration improvements.
    constraint_normalize: bool = False       # Optimize positive row-scaled residuals
    constraint_scale_floor: float = 1.0      # Minimum RMS residual scale per row
    constraint_scale_decay: float = 1.0      # Per-outer annealing toward the scale floor
    update_multipliers_always: bool = False  # Standard ALM update instead of global epsilon gate
    output_affine_project_init: bool = False  # Detect/project affine generated equality pairs
    output_affine_project_fraction: float = 1.0 # Fraction of root restarts using the manifold
    output_affine_project_sweeps: int = 3     # Generated-equality projection / clamp sweeps
    output_affine_inner_sweeps: int = 0       # Hard-manifold reprojection after each PGD step
    output_affine_detection_tol: float = 1e-9 # Jacobian constancy/pairing tolerance
    output_affine_pair_slack_tol: float = 1e-4 # Maximum normalized paired-slab width
    raw_polish_steps: int = 0                 # Local raw-minimax steps after global restoration
    raw_polish_lr: float = 1e-3               # Initial trust radius in unit-box coordinates
    raw_polish_restarts: int = 8               # Local clones of each archived raw-best point
    raw_polish_max_parallel: int = 65536        # Cap B*local_clones to bound BaB memory
    raw_polish_jitter: float = 1e-3            # Clone jitter as a fraction of box width
    raw_polish_max_viol: float = 10.0          # Skip polish outside the near-feasible basin

    # Wall-clock shares of the endgame.
    endgame_time_frac: float = 0.0            # Wall-clock share reserved for polish+GN
    endgame_polish_frac: float = 1.0          # Share of the endgame reserve polish may use

    # Batched Gauss-Newton feasibility restoration (nlp_solver/alm_feas.py).
    # Runs for every infeasible domain (root or BaB batch) from several seeds.
    batched_gn_steps: int = 0                 # GN iterations (0 = off)
    batched_gn_seeds: int = 16                # Seeds per infeasible domain
    batched_gn_damping: float = 1e-6          # LM damping relative to max diag(J^T J)
    batched_gn_trust_radius: float = 1.0      # Max unit-box step (inf-norm)
    batched_gn_max_viol: float = 1e12         # Skip seeds with a larger raw violation
    batched_gn_max_elems: int = 1 << 24       # Cap on candidates * rows * vars (Jacobian)
    # Multi-scale restarts around the projected origin (trailing share of the
    # random restarts; 0 = off).
    init_scaled_fraction: float = 0.0
    init_scaled_min: float = 1e-3
    # Objective-level batched GN: with an incumbent U, also search for
    # objective <= U - rel * max(1, |U|) in every domain lacking such a point.
    batched_gn_improve: bool = False
    batched_gn_improve_rel: float = 1e-4
    # Wall-clock cap of one batched GN call in seconds (0 = up to the ALM deadline).
    batched_gn_max_time: float = 0.0
    # Keep GN iterates this absolute distance inside the box: derivatives of
    # sqrt-like terms are infinite exactly on a bound (0 = off).
    batched_gn_interior: float = 0.0
    # Local optimality phase: batched SQP on distinct final restart points.
    opt_polish_candidates: int = 0            # Points polished per solve (0 = off)
    opt_polish_iters: int = 50                # SQP iterations
    opt_polish_time: float = 5.0              # Wall-clock cap (s) for the phase
    opt_polish_target_frac: float = 0.9       # Drive g to <= frac * feas_tol
    opt_polish_near_quantile: float = 0.05    # Near-feasible pool: least-violated share
    opt_polish_seeds: int = 0                 # Also polish this many restart *start* points

    controller: bool = False                 # Yield retained count-defined atoms
    rival_abs_tol: float = 1e-5
    rival_rel_tol: float = 1e-4

    # General
    num_restarts: int = 10                  # Fallback restart count when no shared B*R budget is set
    br_budget: int = 0                      # Shared B*R budget; root restarts resolve to floor(budget / batch_size)
    timeout: float = 60.0                   # Total timeout
    verbose: bool = False                   # Print detailed output

    # Fields not read from ``alm_<field>``: ``num_restarts`` comes from the caller's budget.
    _KEYS = {'feas_tol': 'constr_viol_tol', 'br_budget': 'bab_br_budget',
             'timeout': 'timeout', 'verbose': 'verbose', 'controller': 'controller',
             'rival_abs_tol': 'bab_primal_dual_abs_tol', 'rival_rel_tol': 'bab_primal_dual_rel_tol'}

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ALMConfig':
        """Create ALMConfig from dictionary."""
        values = {}
        for f in fields(cls):
            if f.name != 'num_restarts':
                key = cls._KEYS.get(f.name, 'alm_' + f.name)
                values[f.name] = f.type(config.get(key, getattr(cls, f.name)))
        return cls(**values)


def resolve_alm_restart_count(
    batch_size: int,
    cfg: ALMConfig,
    explicit_num_restarts: Optional[int] = None,
) -> int:
    """Resolve ALM restart count from an explicit override or a shared B*R budget."""
    if explicit_num_restarts is not None:
        return max(1, int(explicit_num_restarts))

    effective_batch = max(1, int(batch_size))
    br_budget = max(0, int(cfg.br_budget))
    if br_budget > 0:
        return max(1, br_budget // effective_batch)

    return max(1, int(cfg.num_restarts))


@dataclass
@dataclass
class ALMResult:
    """Result container for ALM-PGD solver."""
    success: Union[bool, torch.Tensor]                           # Whether optimal/KKT point found (feasible)
    x_solution: Optional[torch.Tensor]      # Input point (optimal or counterexample)
    objective_value: Optional[Union[float, torch.Tensor]]        # Objective at solution
    constraint_violation: Optional[Union[float, torch.Tensor]]   # Max constraint violation (||[g]_+||_inf)
    solve_time: float                       # Wall-clock time
    status: str                             # Solver status string
    outer_iterations: int                   # ALM outer iterations
    total_inner_steps: int                  # Total inner PGD steps
    final_rho: Union[float, torch.Tensor]                        # Final penalty parameter
    final_mu: Optional[torch.Tensor] = None                     # Best-restart mu_constr: (B, m) or (m,) for B=1


class ALMPGDSolver:
    """
    ALM-PGD solver for constrained neural network optimization.

    Implements Practical ALM (Algorithm 1) from arXiv:2510.16827v1 with:
    - HPR Augmented Lagrangian: L_rho(x,λ) = φ(x) + (ρ/2)||[g(x) + λ/ρ]_+||² - (1/2ρ)||λ||²
    - Projected Adam for the inner subproblem (box constraints)
    - Adaptive ρ scheduling based on feasibility progress
    """

    def __init__(self, config: ALMConfig):
        self.cfg = config
        # (E, d, gram_pinv, project_count) of the affine equality manifold of the current
        # solve: the first ``project_count`` restarts live on it.
        self._output_affine_projection = None

    @retained
    def _project_output_affine_equalities(
        self,
        x: torch.Tensor,
        problem: NNProblemSingleDomain,
        B: int,
        R: int,
        deadline: Optional[float] = None,
    ) -> torch.Tensor:
        """Project seeds onto affine equality pairs embedded in model outputs.

        Conversion represents ``h(x) = 0`` as the thin slab
        ``h(x)-tol <= 0`` and ``-h(x)-tol <= 0``.  For affine ``h``, the two
        residual Jacobians are constant and opposite.  Detecting that exact
        structure recovers the native linear equality manifold without relying
        on benchmark-specific metadata.  Raw residual verification is unchanged.
        """
        if (
            not self.cfg.output_affine_project_init
            or B != 1
            or problem.n_constraints < 2
        ):
            return x
        fraction = min(
            max(float(self.cfg.output_affine_project_fraction), 0.0), 1.0
        )
        if fraction <= 0.0:
            return x

        lb = problem.dm_l
        ub = problem.dm_u
        if lb.ndim == len(problem.input_shape):
            lb = lb.unsqueeze(0)
            ub = ub.unsqueeze(0)
        x_ref = (0.5 * (lb[0] + ub[0])).detach()
        flat_count = x_ref.numel()
        fractions = torch.frac(
            torch.arange(
                flat_count, device=x.device, dtype=x.dtype
            ) * 0.6180339887498949
        ).reshape_as(x_ref)
        x_probe = (lb[0] + (0.2 + 0.6 * fractions) * (ub[0] - lb[0])).detach()

        def residual_fn(x_single: torch.Tensor) -> torch.Tensor:
            x_flat = x_single.unsqueeze(0)
            out = problem.model(x_flat).reshape(1, 1, -1)
            return _phi_and_residuals(problem, x_flat, out, 1, 1)[1][0]

        try:
            check_deadline(deadline)
            g_ref = residual_fn(x_ref)
            # Reuse GN's bounded reverse sweeps instead of materializing the
            # full m x m identity seed in functional.jacobian(vectorize=True).
            from .alm_feas import residual_jacobian
            def evaluate(z):
                g = residual_fn(z.reshape_as(x_ref)).reshape(1, 1, -1)
                return g.new_zeros((1, 1)), g
            jac_ref = (yield from residual_jacobian(
                evaluate, x_ref.reshape(1, 1, -1), deadline=deadline,
                sliced=self.cfg.controller, _iterate=True))[2].reshape(problem.n_constraints, -1)
            jac_probe = (yield from residual_jacobian(
                evaluate, x_probe.reshape(1, 1, -1), deadline=deadline,
                sliced=self.cfg.controller, _iterate=True))[2].reshape(problem.n_constraints, -1)
            check_deadline(deadline)
        except RuntimeError as exc:
            if self.cfg.verbose:
                print(
                    "    [ALM-PGD] Output affine projection skipped: "
                    f"Jacobian evaluation failed ({exc})"
                )
            return x

        tol = max(float(self.cfg.output_affine_detection_tol), 0.0)
        row_norm = torch.linalg.vector_norm(jac_ref, dim=1)
        probe_norm = torch.linalg.vector_norm(jac_probe, dim=1)
        jac_change = torch.linalg.vector_norm(jac_ref - jac_probe, dim=1)
        affine = (
            torch.isfinite(jac_ref).all(dim=1)
            & torch.isfinite(jac_probe).all(dim=1)
            & (row_norm > torch.finfo(x.dtype).eps)
            & (
                jac_change
                <= tol * (1.0 + torch.maximum(row_norm, probe_norm))
            )
        )
        affine_idx = torch.where(affine)[0]
        if affine_idx.numel() < 2:
            return x

        A = jac_ref[affine_idx]
        A_norm = A / row_norm[affine_idx].unsqueeze(1)
        x_ref_flat = x_ref.reshape(-1)
        intercept = g_ref[affine_idx] - A @ x_ref_flat
        b_norm = -intercept / row_norm[affine_idx]
        opposite_cos = A_norm @ A_norm.T
        opposite_rhs = b_norm.unsqueeze(1) + b_norm.unsqueeze(0)
        rhs_scale = 1.0 + torch.maximum(
            b_norm.abs().unsqueeze(1), b_norm.abs().unsqueeze(0)
        )
        pair_slack_tol = max(
            float(self.cfg.output_affine_pair_slack_tol), 0.0
        )
        candidates = torch.triu(
            (opposite_cos + 1.0).abs() <= tol,
            diagonal=1,
        ) & torch.triu(
            opposite_rhs.abs() <= pair_slack_tol * rhs_scale,
            diagonal=1,
        )

        used = set()
        equality_rows = []
        equality_rhs = []
        for left, right in candidates.nonzero(as_tuple=False).tolist():
            check_deadline(deadline)
            if left in used or right in used:
                continue
            used.add(left)
            used.add(right)
            equality_rows.append(A_norm[left])
            # Midpoint of A x <= b_left and A x >= -b_right.
            equality_rhs.append(0.5 * (b_norm[left] - b_norm[right]))
        if not equality_rows:
            return x

        E = torch.stack(equality_rows)
        d = torch.stack(equality_rhs)
        check_deadline(deadline)
        gram_pinv = torch.linalg.pinv(E @ E.T)
        check_deadline(deadline)
        project_count = min(R, int(math.ceil(fraction * R)))
        self._output_affine_projection = (E, d, gram_pinv, project_count)
        x_flat = x.reshape(R, -1)
        projected_flat = x_flat[:project_count]
        lb_flat = lb[0].reshape(-1)
        ub_flat = ub[0].reshape(-1)
        before = (projected_flat @ E.T - d).abs().max()
        sweeps = max(int(self.cfg.output_affine_project_sweeps), 1)
        for _ in range(sweeps):
            check_deadline(deadline)
            residual = projected_flat @ E.T - d
            projected_flat = projected_flat - residual @ gram_pinv @ E
            projected_flat = torch.maximum(
                torch.minimum(projected_flat, ub_flat), lb_flat
            )
            if self.cfg.controller:
                yield Atom("alm_init")
        x_flat[:project_count] = projected_flat
        after = (projected_flat @ E.T - d).abs().max()
        if self.cfg.verbose:
            print(
                "    [ALM-PGD] Output affine equality projection: "
                f"pairs={len(equality_rows)} restarts={project_count}/{R} "
                f"sweeps={sweeps} "
                f"max_residual={before.item():.3e}->{after.item():.3e}"
            )
        return x_flat.reshape_as(x)

    @staticmethod
    def _is_finite_or_empty(tensor: Optional[torch.Tensor]) -> bool:
        """True when tensor has no non-finite entries (or is empty)."""
        if tensor is None or tensor.numel() == 0:
            return True
        return bool(torch.isfinite(tensor).all().item())

    def _validate_problem_data(self, problem: NNProblemSingleDomain) -> Optional[str]:
        """Return error string for invalid numerical inputs, else None."""
        checks = (
            ("dm_l", problem.dm_l),
            ("dm_u", problem.dm_u),
            ("C_obj", problem.C_obj),
            ("rhs_obj", problem.rhs_obj),
            ("C_constr", problem.C_constr),
            ("rhs_constr", problem.rhs_constr),
            ("input_constr_A", getattr(problem, "input_constr_A", None)),
            ("input_constr_b", getattr(problem, "input_constr_b", None)),
            (
                "initialization_constr_A",
                getattr(problem, "initialization_constr_A", None),
            ),
            (
                "initialization_constr_b",
                getattr(problem, "initialization_constr_b", None),
            ),
        )
        for name, tensor in checks:
            if not self._is_finite_or_empty(tensor):
                return f"Invalid problem data: non-finite {name}"
        if problem.dm_l.shape != problem.dm_u.shape:
            return "Invalid problem data: dm_l/dm_u shape mismatch"
        if bool((problem.dm_l > problem.dm_u).any().item()):
            return "Invalid problem data: lower bound exceeds upper bound"
        return None

    def _build_invalid_result(self, problem: NNProblemSingleDomain,
                              start_time: float, status: str) -> ALMResult:
        """Build a consistent failure result for invalid input data."""
        B = int(problem.batch_size)
        m = int(problem.n_constraints)
        device = problem.C_obj.device
        dtype = problem.dtype
        solve_time = time.time() - start_time

        if B == 1:
            return ALMResult(
                success=False,
                x_solution=None,
                objective_value=float('inf'),
                constraint_violation=float('inf'),
                solve_time=solve_time,
                status=status,
                outer_iterations=0,
                total_inner_steps=0,
                final_rho=float(self.cfg.rho0),
                final_mu=torch.zeros(m, device=device, dtype=dtype) if m > 0 else None,
            )

        return ALMResult(
            success=torch.zeros(B, dtype=torch.bool, device=device),
            x_solution=None,
            objective_value=torch.full((B,), float('inf'), device=device, dtype=dtype),
            constraint_violation=torch.full((B,), float('inf'), device=device, dtype=dtype),
            solve_time=solve_time,
            status=status,
            outer_iterations=0,
            total_inner_steps=0,
            final_rho=torch.full((B,), self.cfg.rho0, device=device, dtype=dtype),
            final_mu=torch.zeros(B, m, device=device, dtype=dtype) if m > 0 else None,
        )

    @retained
    def _solve_subproblem_adam(
        self,
        x: torch.Tensor,            # (N, D)
        mu_constr: torch.Tensor,    # (N, m)
        rho: torch.Tensor,          # (N,)
        eta_k: torch.Tensor,        # (N,)
        problem: NNProblemSingleDomain,
        lb: torch.Tensor,           # (D,)
        ub: torch.Tensor,           # (D,)
        constraint_scale: Optional[torch.Tensor] = None,  # (N, m), positive
        inner_lr_scale: float = 1.0,
        deadline: Optional[float] = None,  # absolute wall time (time.time()) to stop at
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Solve ALM subproblem using Adam-PGD (User Requested).

        Key features:
        - Uses torch.optim.Adam for updates (better than manual PGD).
        - Projects (clamps) x after every step.
        - Checks stationarity (grad_norm <= eta_k) for early stopping.
        """
        # (A) x must be a leaf tensor for optimizer.
        # For ALM, we intentionally RESET the Adam state each outer iteration
        # (optimizer is recreated per call). The augmented Lagrangian changes when
        # rho/mu update, and stale Adam moments can destabilize the inner solve.
        x = x.detach().clone().requires_grad_(True)
        N = x.shape[0]

        # Adaptive LR: lr_b = inner_lr * max_d(u_bd - l_bd) per instance.
        # Scales Adam base LR linearly with each domain's longest edge,
        # so BaB child domains automatically get proportionally smaller steps.
        # Adam's v-normalization already handles gradient scaling from ρ
        # changes, so no ρ-dependent adjustment is needed.
        domain_width = (ub - lb).detach()  # (N, *input_shape)
        max_edge = domain_width.flatten(1).max(dim=1).values  # (N,)
        lr_shape = [N] + [1] * len(problem.input_shape)
        effective_inner_lr = self.cfg.inner_lr * max(float(inner_lr_scale), 0.0)
        lr = effective_inner_lr * max_edge.clamp(min=1e-10).view(*lr_shape)

        # Per-coordinate clipping: cap coordinate i's step to inner_lr * width_i.
        lr_vec = effective_inner_lr * domain_width

        # Pass scalar lr to satisfy Adam.__init__ validation
        # (``not 0.0 <= tensor`` raises RuntimeError), then replace
        # with the per-instance tensor lr.
        optimizer = AdamClipping_NLP(
            [x],
            lr=float(effective_inner_lr),
            lr_vec=lr_vec,
            scale_eps_with_bias=self.cfg.scale_eps_with_bias,
        )
        optimizer.param_groups[0]['lr'] = lr

        scheduler = VectorLinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=max(int(self.cfg.inner_steps), 1),
        )
        steps_taken = 0
        m = problem.n_constraints

        grad_norm = torch.full((N,), float('inf'), device=x.device, dtype=x.dtype)
        last_bad_grad_mask = torch.zeros(
            N, dtype=torch.bool, device=x.device
        )

        # Per-instance L_val plateau detection state
        prev_L_checkpoint = None
        instance_plateau_count = torch.zeros(N, device=x.device, dtype=torch.int64)
        converged_mask = torch.zeros(N, device=x.device, dtype=torch.bool)
        plateau_check = self.cfg.inner_plateau_check  # 0 = disabled

        # Helper to reshape x to input shape
        # But for batched PGD, we need to know B and R to broadcast correctly against C_obj (B, 1, out).
        # We know N = B * R.
        B = problem.batch_size
        R = N // B

        # Stationarity check cadence (1 = every step, like the original).
        # We tried K=4 to amortize the per-step `(grad_norm <= eta_k).all()`
        # CPU sync, but on borderline cases (e.g. cute_hs083 at rho=10) the
        # 1–3 extra inner steps shifted x* enough to lose one outer iter via
        # `first_feas_patience`, with d_obj ≈ 4e-5 relative.  K=1 keeps
        # bit-for-bit obj parity at the cost of ~7% wall savings.
        stationarity_check_every = 1

        for t in range(self.cfg.inner_steps):
            # Wall-clock deadline: without this check a single inner solve on a
            # large graph can overshoot the outer-loop timeout by minutes
            # (the outer loop only checks between iterations).
            if deadline is not None and time.time() > deadline:
                break
            optimizer.zero_grad(set_to_none=True)

            # 1. Forward Pass with Batch+Restart structure
            x_in = x.reshape(N, *problem.input_shape)
            out = problem.model(x_in) # (N, out_dim)
            out = out.reshape(B, R, -1)  # (B, R, out_dim)

            # 2. Objective + constraint residuals
            phi, g = _phi_and_residuals(problem, x, out, B, R)

            if self.cfg.controller:
                raw_viol = torch.relu(g).amax(dim=1) if m else torch.zeros_like(phi)
                point = candidate(x, phi, raw_viol, self.cfg.feas_tol)

            # 3. Augmented Lagrangian.  Dual term 0.5/rho * ||mu||^2 has zero
            # gradient w.r.t. x, so we drop it from the backward path.  It is
            # added back (detached) only for the plateau check so the relative
            # change denominator matches the original L_val (tolerance
            # behavior is preserved).
            if m > 0:
                g_alm = (
                    g if constraint_scale is None
                    else g / constraint_scale
                )
                # Broadcasting: g is (N, m), rho is (N,) -> rho[:, None]
                shifted = g_alm + mu_constr / rho[:, None]
                # Penalty: 0.5 * rho * ||relu(shifted)||^2, shape (N,)
                penalty = 0.5 * rho * (
                    torch.relu(shifted) ** 2
                ).sum(dim=1)
                L_val = phi + penalty
            else:
                L_val = phi

            forward_finite = torch.isfinite(phi) & torch.isfinite(L_val)
            if m > 0:
                forward_finite &= torch.isfinite(g).all(dim=1)

            # Per-instance L_val plateau detection
            if plateau_check > 0 and t > 0 and t % plateau_check == 0:
                with torch.no_grad():
                    if m > 0:
                        dual_const = 0.5 / rho * (mu_constr ** 2).sum(dim=1)
                        current_L = L_val.detach() - dual_const
                    else:
                        current_L = L_val.detach()
                    if prev_L_checkpoint is not None:
                        rel_change = (current_L - prev_L_checkpoint).abs() / (prev_L_checkpoint.abs() + 1e-8)
                        plateaued = rel_change < self.cfg.inner_plateau_rtol
                        instance_plateau_count[plateaued] += 1
                        instance_plateau_count[~plateaued] = 0
                        converged_mask = instance_plateau_count >= self.cfg.inner_plateau_patience
                        # All instances converged → break
                        if converged_mask.all():
                            break
                    prev_L_checkpoint = current_L.clone()

            # Backward (sum to scalar)
            L_val.sum().backward()

            if x.grad is None:
                break

            grad_view = x.grad.reshape(N, -1)
            # Sanitize bad gradients in one shot (no host sync); any row that
            # had NaN/inf gets zeroed here.  Previously we detected & randomly
            # reinitialized such rows, but R>=128 parallel restarts means
            # losing one is not material and the branch required two CPU syncs
            # per inner step.  We track `bad_grad_mask` as a tensor (no sync)
            # so the stationarity check does not mistake a zeroed-grad row
            # for a genuinely stationary one.
            bad_grad_mask = (
                ~torch.isfinite(grad_view).all(dim=1)
                | ~forward_finite
            )
            last_bad_grad_mask = bad_grad_mask
            torch.nan_to_num_(grad_view, nan=0.0, posinf=0.0, neginf=0.0)

            # Zero gradients for converged instances (skip their optimizer updates)
            if plateau_check > 0 and converged_mask.any():
                grad_view[converged_mask] = 0

            # Adam + PGD-style projected step (elementwise clipped update + projection).
            optimizer.step(lower_limit=lb, upper_limit=ub)

            output_projection = self._output_affine_projection
            projection_sweeps = max(
                int(self.cfg.output_affine_inner_sweeps), 0
            )
            if output_projection is not None and projection_sweeps > 0:
                E, d, gram_pinv, project_count = output_projection
                with torch.no_grad():
                    x_flat = x.reshape(N, -1)
                    lb_flat = lb.reshape(N, -1)
                    ub_flat = ub.reshape(N, -1)
                    projected_flat = x_flat[:project_count]
                    projected_lb = lb_flat[:project_count]
                    projected_ub = ub_flat[:project_count]
                    for _ in range(projection_sweeps):
                        affine_residual = projected_flat @ E.T - d
                        projected_flat = (
                            projected_flat
                            - affine_residual @ gram_pinv @ E
                        )
                        projected_flat = torch.maximum(
                            torch.minimum(projected_flat, projected_ub),
                            projected_lb,
                        )
                    x_flat[:project_count] = projected_flat
                    x.copy_(x_flat.reshape_as(x))

            scheduler.step()
            steps_taken += 1
            if self.cfg.controller:
                yield Atom("alm", point)

            # Stationarity check.  Check on step 0 (so trivial early-exit
            # cases still exit after 1 step as in the original), then gate
            # to every K=stationarity_check_every steps afterwards to avoid
            # per-step CPU sync on `.all()`.
            if t == 0 or (t + 1) % stationarity_check_every == 0:
                with torch.no_grad():
                    grad_norm = torch.norm(grad_view, dim=1)
                    # Rows with NaN/inf grads were zeroed above; flag them as
                    # non-stationary so they don't spuriously trigger early
                    # exit.
                    grad_norm = torch.where(
                        bad_grad_mask,
                        torch.full_like(grad_norm, float('inf')),
                        grad_norm,
                    )
                if (grad_norm <= eta_k).all():
                    break

        # Always compute final grad_norm for the returned tensor contract.
        if x.grad is not None:
            with torch.no_grad():
                grad_norm = torch.norm(x.grad.reshape(N, -1), dim=1)
                grad_norm = torch.where(
                    last_bad_grad_mask,
                    torch.full_like(grad_norm, float('inf')),
                    grad_norm,
                )

        return x.detach(), grad_norm.detach(), steps_taken

    @retained
    def _polish_raw_feasibility(
        self,
        problem: NNProblemSingleDomain,
        best_x: torch.Tensor,
        best_obj: torch.Tensor,
        best_viol: torch.Tensor,
        best_is_feas: torch.Tensor,
        deadline: Optional[float],
        candidate_x: Optional[torch.Tensor] = None,
        candidate_viols: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        """Locally minimize the exact raw feasibility acceptance metric.

        The global ALM stage is intentionally preconditioned and uses a summed
        HPR merit to find promising basins.  This bounded final stage instead
        starts from the archived minimum-raw-violation point and directly
        minimizes ``max(relu(g_raw))``.  Variables are optimized in unit-box
        coordinates so the step size has the same meaning for every input
        width.  Raw feasibility and the original tolerance remain authoritative.
        """
        steps_limit = max(int(self.cfg.raw_polish_steps), 0)
        if steps_limit == 0 or problem.n_constraints == 0:
            return best_x, best_obj, best_viol, best_is_feas, 0
        if deadline is not None and time.time() >= deadline:
            return best_x, best_obj, best_viol, best_is_feas, 0

        B = problem.batch_size
        max_parallel = max(int(self.cfg.raw_polish_max_parallel), 1)
        if B > max_parallel:
            if self.cfg.verbose:
                print(
                    "    [ALM-PGD] Raw minimax polish skipped: "
                    f"domains={B} exceeds parallel cap={max_parallel}"
                )
            return best_x, best_obj, best_viol, best_is_feas, 0
        P = min(
            max(int(self.cfg.raw_polish_restarts), 1),
            max_parallel // B,
        )
        device = best_x.device
        dtype = best_x.dtype
        active = (
            (~best_is_feas)
            & torch.isfinite(best_viol)
            & (best_viol <= float(self.cfg.raw_polish_max_viol))
        )
        if not bool(active.any().item()):
            return best_x, best_obj, best_viol, best_is_feas, 0

        lb = problem.dm_l
        ub = problem.dm_u
        if lb.ndim == len(problem.input_shape):
            lb = lb.unsqueeze(0)
            ub = ub.unsqueeze(0)
        width = ub - lb
        safe_width = width.clamp(min=torch.finfo(dtype).eps)
        seed_x = best_x.unsqueeze(1).expand(
            B, P, *problem.input_shape
        ).clone()
        # Preserve basin diversity from the global phase.  Slot zero is the
        # all-iteration archive; remaining slots receive the best distinct
        # current restart states before any optional local jitter.
        if (
            P > 1
            and candidate_x is not None
            and candidate_viols is not None
            and candidate_x.shape[0] % B == 0
            and candidate_viols.numel() == candidate_x.shape[0]
        ):
            global_R = candidate_x.shape[0] // B
            take = min(P - 1, global_R)
            current_viols = candidate_viols.reshape(B, global_R)
            top_values, top_indices = torch.topk(
                current_viols, k=take, dim=1, largest=False
            )
            current_x = candidate_x.reshape(
                B, global_R, *problem.input_shape
            )
            gather_shape = [B, take] + [1] * len(problem.input_shape)
            gather_idx = top_indices.reshape(*gather_shape).expand(
                B, take, *problem.input_shape
            )
            top_x = torch.gather(current_x, 1, gather_idx)
            finite_top = torch.isfinite(top_values)
            finite_shape = [B, take] + [1] * len(problem.input_shape)
            seed_x[:, 1:1 + take] = torch.where(
                finite_top.reshape(*finite_shape),
                top_x,
                seed_x[:, 1:1 + take],
            )

        z_bp = ((seed_x - lb.unsqueeze(1)) / safe_width.unsqueeze(1)).clamp(
            0.0, 1.0
        )
        if P > 1 and self.cfg.raw_polish_jitter > 0:
            jitter = (
                2.0
                * torch.rand(
                    B,
                    P - 1,
                    *problem.input_shape,
                    device=device,
                    dtype=dtype,
                )
                - 1.0
            ) * float(self.cfg.raw_polish_jitter)
            z_bp[:, 1:] = (z_bp[:, 1:] + jitter).clamp(0.0, 1.0)

        z = z_bp.reshape(B * P, *problem.input_shape).detach().requires_grad_(True)
        initial_z = z.detach().clone()
        lb_p = lb.repeat_interleave(P, dim=0)
        width_p = width.repeat_interleave(P, dim=0)
        initial_radius = max(float(self.cfg.raw_polish_lr), 1e-12)
        step_radius = torch.full(
            (B * P,), initial_radius, device=device, dtype=dtype
        )
        def evaluate(z_eval: torch.Tensor):
            x_eval = lb_p + z_eval * width_p
            out_eval = problem.model(
                x_eval.reshape(B * P, *problem.input_shape)
            ).reshape(B, P, -1)
            phi_eval, g_eval = _phi_and_residuals(problem, x_eval, out_eval, B, P)
            viol_eval = torch.relu(g_eval).max(dim=1).values
            return x_eval, phi_eval, viol_eval

        def archive(
            x_eval: torch.Tensor,
            phi_eval: torch.Tensor,
            viol_eval: torch.Tensor,
        ) -> None:
            nonlocal best_x, best_obj, best_viol, best_is_feas
            phi_bp = phi_eval.reshape(B, P)
            viol_bp = viol_eval.reshape(B, P)
            feasible_bp = viol_bp <= self.cfg.feas_tol
            feasible_phi = phi_bp.masked_fill(~feasible_bp, float("inf"))
            min_feas_phi, min_feas_idx = feasible_phi.min(dim=1)
            min_viol, min_viol_idx = viol_bp.min(dim=1)
            found_feas = torch.isfinite(min_feas_phi)
            improved_feas = found_feas & (
                (~best_is_feas) | (min_feas_phi < best_obj)
            )
            better_viol = (
                (~found_feas)
                & (~best_is_feas)
                & (min_viol < best_viol)
            )
            update = active & (improved_feas | better_viol)
            if bool(update.any().item()):
                select_p = torch.where(
                    improved_feas, min_feas_idx, min_viol_idx
                )
                batch_idx = torch.arange(B, device=device)
                flat_idx = batch_idx * P + select_p
                dst = torch.where(update)[0]
                src = flat_idx[dst]
                best_x[dst] = x_eval[src].detach()
                best_obj[dst] = phi_eval[src].detach()
                best_viol[dst] = viol_eval[src].detach()
                best_is_feas[dst] |= improved_feas[dst]

        start_viol = best_viol.detach().clone()
        steps_taken = 0
        for polish_iter in range(steps_limit):
            if deadline is not None and time.time() >= deadline:
                break
            z.grad = None
            x_polish, phi_polish, viol_polish = evaluate(z)
            with torch.no_grad():
                archive(x_polish, phi_polish, viol_polish)
            remaining = active & (~best_is_feas)
            if not bool(remaining.any().item()):
                break

            remaining_flat = remaining.repeat_interleave(P)
            viol_polish[remaining_flat].sum().backward()
            if z.grad is None:
                break
            grad_view = z.grad.reshape(B * P, -1)
            bad_grad = ~torch.isfinite(grad_view).all(dim=1)
            torch.nan_to_num_(grad_view, nan=0.0, posinf=0.0, neginf=0.0)
            grad_view[~remaining_flat | bad_grad] = 0.0
            direction_scale = grad_view.abs().max(dim=1).values.clamp(
                min=torch.finfo(dtype).eps
            )
            direction = (
                grad_view / direction_scale.unsqueeze(1)
            ).reshape_as(z)
            radius_shape = [B * P] + [1] * len(problem.input_shape)
            with torch.no_grad():
                proposal_z = (
                    z - step_radius.reshape(*radius_shape) * direction
                ).clamp(0.0, 1.0)
                proposed_x, proposed_phi, proposed_viol = evaluate(proposal_z)
                accept = (
                    remaining_flat
                    & torch.isfinite(proposed_viol)
                    & (proposed_viol < viol_polish)
                )
                if bool(accept.any().item()):
                    accept_shape = [B * P] + [1] * len(problem.input_shape)
                    z.copy_(
                        torch.where(
                            accept.reshape(*accept_shape), proposal_z, z
                        )
                    )
                step_radius[accept] = torch.clamp(
                    step_radius[accept] * 1.2, max=initial_radius
                )
                step_radius[remaining_flat & (~accept)] *= 0.5
                archive(proposed_x, proposed_phi, proposed_viol)
            if self.cfg.verbose and (
                polish_iter == 0
                or polish_iter + 1 == steps_limit
                or (polish_iter + 1) % 250 == 0
            ):
                local_grad_norm = torch.linalg.vector_norm(
                    grad_view[remaining_flat], dim=1
                )
                movement = (
                    z.detach() - initial_z
                ).reshape(B * P, -1).abs().max()
                print(
                    "    [ALM-PGD] Raw polish iter: "
                    f"t={polish_iter} "
                    f"current_min={viol_polish[remaining_flat].min().item():.3e} "
                    f"grad_med={local_grad_norm.median().item():.3e} "
                    f"grad_max={local_grad_norm.max().item():.3e} "
                    f"unit_box_move={movement.item():.3e} "
                    f"accepted={int(accept.sum().item())}/{int(remaining_flat.sum().item())} "
                    f"radius_med={step_radius[remaining_flat].median().item():.3e}"
                )
            steps_taken += 1
            if self.cfg.controller:
                yield Atom("polish", candidate(best_x, best_obj, best_viol, self.cfg.feas_tol))

        if deadline is None or time.time() < deadline:
            with torch.no_grad():
                archive(*evaluate(z))

        if self.cfg.verbose:
            active_idx = torch.where(active)[0]
            before = start_viol[active_idx].min().item()
            after = best_viol[active_idx].min().item()
            successes = int((best_is_feas & active).sum().item())
            print(
                "    [ALM-PGD] Raw minimax polish: "
                f"steps={steps_taken}/{steps_limit} clones={P} "
                f"min_viol={before:.3e}->{after:.3e} "
                f"new_feasible={successes}/{int(active.sum().item())}"
            )
        return best_x, best_obj, best_viol, best_is_feas, steps_taken

    def _origin_scaled_restarts(
        self,
        x_rand: torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
    ) -> torch.Tensor:
        """Replace the trailing share of random restarts by multi-scale seeds.

        Seeds are ``clip(a + (2u - 1) * s * (ub - lb))`` around the anchor ``a``,
        the projection of the origin onto the box (the usual modelling start),
        with a log-uniform scale ``s`` in ``[init_scaled_min, 1]``.  Uniform
        sampling of wide boxes (e.g. exp(x) on [-100, 100]) almost never lands in
        the region where the residuals have a usable scale.
        """
        fraction = min(max(float(self.cfg.init_scaled_fraction), 0.0), 1.0)
        count = int(math.floor(fraction * x_rand.shape[1]))
        if count <= 0:
            return x_rand
        B = x_rand.shape[0]
        shape = (B, count) + tuple(x_rand.shape[2:])
        lb_b = lb.unsqueeze(1)
        ub_b = ub.unsqueeze(1)
        anchor = torch.zeros_like(lb_b).clamp(min=lb_b, max=ub_b)
        low = math.log(max(min(float(self.cfg.init_scaled_min), 1.0), 1e-12))
        scale_shape = (B, count) + (1,) * (x_rand.ndim - 2)
        scale = torch.exp(
            low * torch.rand(scale_shape, device=x_rand.device, dtype=x_rand.dtype)
        )
        unit = 2.0 * torch.rand(shape, device=x_rand.device, dtype=x_rand.dtype) - 1.0
        seeds = anchor + unit * scale * (ub_b - lb_b)
        seeds = torch.maximum(torch.minimum(seeds, ub_b), lb_b)
        x_rand = x_rand.clone()
        x_rand[:, x_rand.shape[1] - count:] = seeds
        return x_rand

    @retained
    def _batched_gn_restore(
        self,
        problem: NNProblemSingleDomain,
        best_x: torch.Tensor,
        best_obj: torch.Tensor,
        best_viol: torch.Tensor,
        best_is_feas: torch.Tensor,
        candidate_x: torch.Tensor,
        candidate_viols: torch.Tensor,
        deadline: Optional[float],
        incumbent: Optional[torch.Tensor] = None,
        _level_phase: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Batched Gauss-Newton restoration of every infeasible domain.

        Seeds per domain: the archived best point plus the lowest-violation
        current restart states.  See ``nlp_solver/alm_feas.py``.

        With ``batched_gn_improve`` and a finite raw ``incumbent`` U, the
        search also carries the row ``objective <= U - rel * max(1, |U|)`` and
        runs for every domain without such a point yet (feasible or not).  The
        row only steers the search: points are accepted on raw feasibility and
        replace the domain's best only if feasible with a lower objective (or,
        while the domain has no feasible point, with a lower violation).
        The level phase runs after a plain restoration phase, so it can only
        add to what feasibility restoration alone finds: an unattainable level
        (incumbent already optimal) must not cost the domain its feasible point.
        """
        from .alm_feas import (
            batched_gauss_newton_restoration,
            make_domain_evaluator,
        )

        steps = max(int(self.cfg.batched_gn_steps), 0)
        m = problem.n_constraints
        improve = bool(self.cfg.batched_gn_improve) and incumbent is not None
        max_time = float(self.cfg.batched_gn_max_time)
        if improve and not _level_phase:
            if not self.cfg.controller and max_time > 0:
                cap_deadline = time.time() + max_time
                deadline = cap_deadline if deadline is None else min(deadline, cap_deadline)
            best_x, best_obj, best_viol, best_is_feas, first_iters = (
                (yield from invoke(self._batched_gn_restore,
                    problem, best_x, best_obj, best_viol, best_is_feas,
                    candidate_x, candidate_viols, deadline, incumbent=None,
                    _level_phase=True,
                    sliced=self.cfg.controller))
            )
            best_x, best_obj, best_viol, best_is_feas, second_iters = (
                (yield from invoke(self._batched_gn_restore,
                    problem, best_x, best_obj, best_viol, best_is_feas,
                    candidate_x, candidate_viols, deadline, incumbent=incumbent,
                    _level_phase=True,
                    sliced=self.cfg.controller))
            )
            return best_x, best_obj, best_viol, best_is_feas, first_iters + second_iters
        if (
            steps == 0
            or (m == 0 and not improve)
            or (deadline is not None and time.time() >= deadline)
        ):
            return best_x, best_obj, best_viol, best_is_feas, 0
        B = problem.batch_size
        n = problem.n_vars
        level_all = None
        if self.cfg.batched_gn_improve and incumbent is not None:
            u_raw = torch.as_tensor(
                incumbent, device=best_obj.device, dtype=best_obj.dtype
            ).reshape(-1)[:1]
            if u_raw.numel() == 1 and bool(torch.isfinite(u_raw).all().item()):
                margin = float(self.cfg.batched_gn_improve_rel) * max(
                    1.0, abs(float(u_raw.item()))
                )
                level_all = (u_raw - margin) - problem.rhs_obj.reshape(B, -1)[:, 0]
        if level_all is None and m == 0:
            return best_x, best_obj, best_viol, best_is_feas, 0
        m_rows = m + (0 if level_all is None else 1)
        if level_all is not None:
            improving = best_is_feas & (best_obj <= level_all)
            active = (~improving) & torch.isfinite(best_viol) & (
                best_viol <= float(self.cfg.batched_gn_max_viol)
            )
        else:
            active = (
                (~best_is_feas)
                & torch.isfinite(best_viol)
                & (best_viol <= float(self.cfg.batched_gn_max_viol))
            )
        if not bool(active.any().item()):
            return best_x, best_obj, best_viol, best_is_feas, 0
        R_cur = candidate_x.shape[0] // B
        per = max(1, min(int(self.cfg.batched_gn_seeds), R_cur + 1))
        cap = max(int(self.cfg.batched_gn_max_elems), 1)
        active_idx = torch.where(active)[0]
        # Keep the memory of the Jacobian (candidates * m * n) bounded: fewer
        # seeds first, then the domains closest to feasibility.
        while per > 1 and active_idx.numel() * per * m_rows * n > cap:
            per //= 2
        max_domains = max(1, cap // max(per * m_rows * n, 1))
        if active_idx.numel() > max_domains:
            order = torch.argsort(best_viol.index_select(0, active_idx))
            active_idx = active_idx.index_select(0, order[:max_domains])
        Ba = int(active_idx.numel())

        dtype = best_x.dtype
        cand = candidate_x.reshape(B, R_cur, -1).index_select(0, active_idx)
        cand_v = candidate_viols.reshape(B, R_cur).index_select(0, active_idx)
        cand_v = torch.nan_to_num(cand_v, nan=float("inf"), posinf=float("inf"))
        seeds = best_x.reshape(B, 1, -1).index_select(0, active_idx).expand(
            Ba, per, n
        ).clone()
        if per > 1:
            take = min(per - 1, R_cur)
            top_v, top_i = torch.topk(cand_v, k=take, dim=1, largest=False)
            top_x = torch.gather(
                cand, 1, top_i.unsqueeze(2).expand(Ba, take, n)
            )
            usable = torch.isfinite(top_v).unsqueeze(2)
            seeds[:, 1:1 + take] = torch.where(
                usable, top_x, seeds[:, 1:1 + take]
            )
        if not self.cfg.controller and max_time > 0 and not _level_phase:
            cap_deadline = time.time() + max_time
            deadline = cap_deadline if deadline is None else min(deadline, cap_deadline)
        evaluate = make_domain_evaluator(
            problem,
            active_idx,
            level=None if level_all is None else level_all.index_select(0, active_idx),
        )
        safe_width = evaluate.width.clamp(min=torch.finfo(dtype).eps)
        z0 = ((seeds - evaluate.lb) / safe_width).clamp(0.0, 1.0)
        start_viol = best_viol.index_select(0, active_idx).clone()
        z_best, phi_best, viol_best, iters = yield from batched_gauss_newton_restoration(
            evaluate,
            z0,
            steps=steps,
            feas_tol=float(self.cfg.feas_tol),
            damping=float(self.cfg.batched_gn_damping),
            trust_radius=float(self.cfg.batched_gn_trust_radius),
            deadline=deadline,
            verbose=self.cfg.verbose,
            n_raw_rows=evaluate.n_raw,
            # Absolute x-distance from the bounds, in each domain's unit box.
            interior=float(self.cfg.batched_gn_interior) / safe_width,
            sliced=self.cfg.controller, _iterate=True,
        )
        with torch.no_grad():
            feas = viol_best <= self.cfg.feas_tol
            phi_f = torch.where(feas, phi_best, torch.full_like(phi_best, float("inf")))
            f_val, f_idx = phi_f.min(dim=1)
            v_val, v_idx = viol_best.min(dim=1)
            got_feas = torch.isfinite(f_val)
            pick = torch.where(got_feas, f_idx, v_idx)
            rows = torch.arange(Ba, device=best_x.device)
            x_pick = (evaluate.lb[:, 0] + z_best[rows, pick] * evaluate.width[:, 0])
            was_feas = best_is_feas.index_select(0, active_idx)
            old_obj = best_obj.index_select(0, active_idx)
            better = (got_feas & ((~was_feas) | (f_val < old_obj))) | (
                (~got_feas) & (~was_feas) & (v_val < start_viol)
            )
            if bool(better.any().item()):
                dst = active_idx[better]
                best_x[dst] = x_pick[better].reshape(
                    int(better.sum().item()), *problem.input_shape
                ).to(best_x.dtype)
                best_obj[dst] = phi_best[rows, pick][better]
                best_viol[dst] = viol_best[rows, pick][better]
                best_is_feas[dst] = best_is_feas[dst] | got_feas[better]
        if self.cfg.verbose:
            print(
                "    [ALM-FEAS] Batched GN restoration: "
                f"domains={Ba}/{B} seeds={per} iters={iters}/{steps} "
                f"min_viol={start_viol.min().item():.3e}->"
                f"{best_viol.index_select(0, active_idx).min().item():.3e} "
                f"feasible={int(got_feas.sum().item())}/{Ba} "
                f"improved={int(better.sum().item())} "
                f"newly_feasible={int((got_feas & ~was_feas).sum().item())}"
                + ("" if level_all is None else
                   f" below_level={int((got_feas & (f_val <= level_all.index_select(0, active_idx))).sum().item())}")
            )
        return best_x, best_obj, best_viol, best_is_feas, iters

    @retained
    def solve_with_restarts(
        self,
        problem: NNProblemSingleDomain,
        num_restarts: Optional[int] = None,
        timeout: Optional[float] = None,
        x_init: Optional[torch.Tensor] = None,
        mu_init: Optional[torch.Tensor] = None,
        incumbent: Optional[torch.Tensor] = None,
        rival: Optional[Callable[[], Optional[float]]] = None,
    ) -> ALMResult:
        """Solve with multiple random restarts using parallel ALM-PGD.

        rival() -> the objective of a verified point another solver holds, or None while it has none (the unified
        root, solver.nlp.unified_root). Target-level residual progress and measured
        work govern continuation; a stopped search still runs its restoration endgame.
        """
        resolved_from_budget = num_restarts is None and int(self.cfg.br_budget) > 0
        num_restarts = resolve_alm_restart_count(
            problem.batch_size,
            self.cfg,
            explicit_num_restarts=num_restarts,
        )
        if timeout is None:
            timeout = self.cfg.timeout

        with _eval_mode(problem.model):
            start_time = time.time()
            deadline = start_time + timeout
            device = problem.C_obj.device
            dtype = problem.dtype

            invalid_reason = self._validate_problem_data(problem)
            if invalid_reason is not None:
                if self.cfg.verbose:
                    print(f"    [ALM-PGD] {invalid_reason}")
                return self._build_invalid_result(problem, start_time, invalid_reason)
            if time.time() >= deadline:
                return self._build_invalid_result(problem, start_time, "ALM initialization deadline")

            # Projection metadata is solve-local.  Batched BaB calls and later
            # problems must never inherit a root manifold from an earlier solve.
            self._output_affine_projection = None

            if resolved_from_budget and self.cfg.verbose:
                print(
                    "    [ALM-PGD] Shared B*R budget: "
                    f"domains={problem.batch_size}, restarts={num_restarts}, "
                    f"B*R={problem.batch_size * num_restarts}/{int(self.cfg.br_budget)}"
                )

            # 1. Initialize Batch x0
            # N = Total parallel instances = batch_size * num_restarts
            B = problem.batch_size
            R = num_restarts
            N = B * R

            # Bounds: (B, *input_shape) -> expand to (N, *input_shape) if needed for clamping
            # But clamping usually broadcasts fine if we shape x as (B, R, *input) or (N, *input).
            # Let's keep x flattened as (N, *input_shape) for optimizer, but be careful with broadcasting.

            # lb, ub are (B, *input_shape)
            lb = problem.dm_l
            ub = problem.dm_u

            # Ensure lb/ub have batch dimension (B, *input_shape)
            # If lb has same ndim as input_shape, it's missing batch dim (B=1)
            if lb.ndim == len(problem.input_shape):
                lb = lb.unsqueeze(0)
                ub = ub.unsqueeze(0)
            # Initial point
            if x_init is not None:
                # x_init: (B, *input_shape)
                x0 = x_init.to(device=device, dtype=dtype)
                x0 = torch.clamp(x0, lb, ub)
            else:
                x0 = 0.5 * (lb + ub)

            # Create N initial points
            # Structure: [Prob1_R1, Prob1_R2... Prob2_R1...]
            # Shape x: (B, R, *input_shape) -> reshape to (N, *input_shape)

            # First restart: from x0 (warm start or center)
            # shape: (B, 1, *input_shape)
            x_r0 = x0.unsqueeze(1)

            init_list = [x_r0]
            if R > 1:
                # Random restarts: (B, R-1, *input_shape)
                delta = ub - lb
                x_rand = torch.rand(B, R-1, *problem.input_shape, device=device, dtype=dtype)
                x_rand = lb.unsqueeze(1) + x_rand * delta.unsqueeze(1)
                x_rand = self._origin_scaled_restarts(x_rand, lb, ub)
                init_list.append(x_rand)

            x = torch.cat(init_list, dim=1) # (B, R, *input_shape)
            x = x.reshape(N, *problem.input_shape) # Flatten batch and restarts

            # Expand bounds to (N, *input_shape) for PGD clamping
            if lb.ndim > 0 and lb.shape[0] == B and R > 1:
                lb = lb.repeat_interleave(R, dim=0)
                ub = ub.repeat_interleave(R, dim=0)

            try:
                x = yield from invoke(self._project_output_affine_equalities, x, problem, B, R,
                                      deadline=deadline, sliced=self.cfg.controller)
                check_deadline(deadline)
            except DeadlineExceeded:
                return self._build_invalid_result(problem, start_time, "ALM initialization deadline")
            polish_seeds = self._polish_seed_points(x, B, R)
            # 2. State
            m = problem.n_constraints
            mu_constr = torch.zeros(N, m, device=device, dtype=dtype)

            # Warm-start mu_constr from dual ascent lambda* if provided
            if mu_init is not None and m > 0:
                # mu_init: (m,) -> broadcast to (N, m)
                mu_constr = mu_init.unsqueeze(0).expand(N, m).clone()

            rho = torch.full((N,), self.cfg.rho0, device=device, dtype=dtype)
            eps_k = 1.0 / (rho ** self.cfg.alpha)
            eta_k = 1.0 / (rho ** self.cfg.beta)

            total_steps = 0
            global_steps = 0

            # Track best feasible point (prefer feasibility, then objective).
            # If B=1, we can use scalar logic or tensor logic. Let's use tensor logic for consistency.
            best_obj = torch.full((B,), float('inf'), device=device, dtype=dtype)
            best_viol = torch.full((B,), float('inf'), device=device, dtype=dtype)
            # Use 'best_x' as the store for feasible points (or best viol if none feas)
            # Initialize with first restart points (B, *input)
            # x is (N, *input), N=B*R. First restart for each batch is at indices 0, R, 2R...
            # But simpler: x was constructed as [x_r0, x_rand...].
            # Actually x structure is [Batch1_R1..Rn, Batch2_R1..Rn...]?
            # Wait, Initialization was:
            # x_r0 = x0.unsqueeze(1) (B, 1, ...)
            # x = torch.cat(..., dim=1).reshape(N, ...)
            # So x structure is:
            # B=2, R=2.
            # dim 1 cat: [[b1r1, b1r2], [b2r1, b2r2]]
            # view(N): [b1r1, b1r2, b2r1, b2r2]
            # So batch b is at indices [b*R : (b+1)*R]

            # Initial best_x can be the first restart for each batch
            best_x = x.reshape(B, R, *problem.input_shape)[:, 0].clone()

            best_is_feas = torch.zeros(B, dtype=torch.bool, device=device)

            best_rho = torch.full((B,), float(self.cfg.rho0), device=device, dtype=dtype)
            best_mu = torch.zeros(B, m, device=device, dtype=dtype)
            best_mu_scale = torch.ones(B, m, device=device, dtype=dtype)

            # Preserve a feasible warm start (e.g. IPOPT/SciPy) before ALM's
            # first inner step has a chance to move it off the feasible set.
            with torch.no_grad():
                x_init_in = x.reshape(N, *problem.input_shape)
                out_init = problem.model(x_init_in).reshape(B, R, -1)
                phis_init, gs_init = _phi_and_residuals(problem, x, out_init, B, R)
                if m > 0:
                    viols_init = torch.max(torch.relu(gs_init), dim=1).values
                else:
                    viols_init = torch.zeros(N, device=device, dtype=dtype)

                phis_init_batch = phis_init.reshape(B, R)
                viols_init_batch = viols_init.reshape(B, R)

                # Positive row scaling is used only inside ALM. Raw residuals
                # remain authoritative for feasibility and returned violations.
                if m > 0 and self.cfg.constraint_normalize:
                    gs_init_batch = gs_init.reshape(B, R, m)
                    finite_gs = torch.nan_to_num(
                        gs_init_batch, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    # Scale by the magnitude that actually enters the HPR
                    # penalty. Large negative slack must not weaken a row if
                    # it later crosses the feasibility boundary.
                    positive_gs = torch.relu(finite_gs)
                    scale_floor = max(
                        float(self.cfg.constraint_scale_floor), 1e-12
                    )
                    output_projection = getattr(
                        self, "_output_affine_projection", None
                    )
                    if (
                        B == 1
                        and output_projection is not None
                        and output_projection[3] < R
                    ):
                        project_count = output_projection[3]
                        projected_scale = (
                            positive_gs[:, :project_count]
                            .square()
                            .mean(dim=1)
                            .sqrt()
                            .clamp(min=scale_floor)
                        )
                        unprojected_scale = (
                            positive_gs[:, project_count:]
                            .square()
                            .mean(dim=1)
                            .sqrt()
                            .clamp(min=scale_floor)
                        )
                        constraint_scale = torch.cat(
                            [
                                projected_scale.repeat(project_count, 1),
                                unprojected_scale.repeat(
                                    R - project_count, 1
                                ),
                            ],
                            dim=0,
                        )
                    else:
                        scale_batch = (
                            positive_gs.square().mean(dim=1).sqrt()
                        ).clamp(min=scale_floor)
                        constraint_scale = scale_batch.repeat_interleave(
                            R, dim=0
                        )
                    # A raw multiplier lambda on g is represented as
                    # mu=lambda*scale on g/scale.
                    if mu_init is not None:
                        mu_constr.mul_(constraint_scale)
                    if self.cfg.verbose:
                        print(
                            "    [ALM-PGD] Positive-residual RMS scale: "
                            f"min={constraint_scale.min().item():.3e} "
                            f"median={constraint_scale.median().item():.3e} "
                            f"max={constraint_scale.max().item():.3e}"
                        )
                else:
                    constraint_scale = torch.ones(
                        N, m, device=device, dtype=dtype
                    )

                init_is_feas_batch = viols_init_batch <= self.cfg.feas_tol
                init_feas_phis = phis_init_batch.clone()
                init_feas_phis[~init_is_feas_batch] = float('inf')
                init_min_feas_phi, init_min_feas_idx = init_feas_phis.min(dim=1)
                init_min_viol, init_min_viol_idx = viols_init_batch.min(dim=1)
                init_found_feas = init_min_feas_phi != float('inf')

                init_select_r = torch.where(init_found_feas, init_min_feas_idx, init_min_viol_idx)
                init_batch_indices = torch.arange(B, device=device)
                init_flat_indices = init_batch_indices * R + init_select_r

                best_x = x[init_flat_indices].detach().clone()
                best_obj = phis_init[init_flat_indices]
                best_viol = viols_init[init_flat_indices]
                best_is_feas = init_found_feas
                best_rho = rho[init_flat_indices].detach().clone()
                if m > 0:
                    best_mu = mu_constr[init_flat_indices].detach().clone()
                    best_mu_scale = constraint_scale[init_flat_indices].detach().clone()

            # Track consecutive iterations without best_obj improvement (for early stopping)
            no_obj_improve = torch.zeros(B, device=device, dtype=torch.int64)
            prev_best_obj = torch.full((B,), float('inf'), device=device, dtype=dtype)

            outer_iters_done = 0
            rival_progress = _RivalProgress()
            progress_stop = False
            restoration_trial = False
            trial_obj = trial_viol = trial_feas = None
            restored_archive = None
            certificate_incumbent = None
            trial_exhausted = False
            feas_plateau_start = None  # Track when global best feasible obj last improved
            best_global_feas_obj = torch.tensor(float('inf'), device=device, dtype=dtype)
            last_grad_norms = torch.full((N,), float('inf'), device=device, dtype=dtype)
            # Define these so they exist even if the loop terminates early.
            phis = torch.full((N,), float('inf'), device=device, dtype=dtype)
            viols = torch.full((N,), float('inf'), device=device, dtype=dtype)
            # Previous-iteration violation cache for the penalty growth test.
            prev_viols = torch.full((N,), float('inf'), device=device, dtype=dtype)

            # The restoration endgame (raw minimax polish + damped Gauss-Newton)
            # is the only stage that reliably reaches 1e-9, but it used to share
            # the outer loop's deadline.  Whenever the outer loop ran to the wall
            # clock the endgame was entered with an already-expired deadline and
            # skipped entirely.  Reserve its share up front.
            endgame_frac = (0.0 if self.cfg.controller else
                            min(max(float(self.cfg.endgame_time_frac), 0.0), 0.9))
            global_timeout = timeout * (1.0 - endgame_frac)
            if int(self.cfg.opt_polish_candidates) > 0:
                # The optimality polish runs last; keep its share (at most a
                # quarter of the budget) out of the outer loop's deadline.
                polish_reserve = min(max(float(self.cfg.opt_polish_time), 0.0), 0.25 * timeout)
                global_timeout = min(global_timeout, timeout - polish_reserve)

            if self.cfg.controller:
                global_timeout = timeout

            outer_limit = max(int(self.cfg.outer_max_iter), 0)

            if self.cfg.controller:
                yield Atom("alm", candidate(best_x, best_obj, best_viol, self.cfg.feas_tol))

            for k in range(outer_limit):
                if time.time() - start_time > global_timeout:
                    break

                outer_iters_done = k + 1

                # Use normalization as a continuation preconditioner, not as
                # the final accuracy metric. Anneal positive row scales toward
                # the configured floor and preserve raw multiplier units:
                # mu_internal = lambda_raw * scale.
                if (
                    k > 0
                    and m > 0
                    and self.cfg.constraint_normalize
                    and self.cfg.constraint_scale_decay < 1.0
                ):
                    decay = max(float(self.cfg.constraint_scale_decay), 0.0)
                    scale_floor = max(
                        float(self.cfg.constraint_scale_floor), 1e-12
                    )
                    previous_scale = constraint_scale
                    constraint_scale = torch.clamp(
                        previous_scale * decay, min=scale_floor
                    )
                    mu_constr.mul_(constraint_scale / previous_scale)
                    if self.cfg.verbose and (
                        k == 1
                        or float(constraint_scale.max().item())
                        <= scale_floor * (1.0 + 1e-6)
                    ):
                        print(
                            "    [ALM-PGD] Constraint scale continuation: "
                            f"k={k} max={constraint_scale.max().item():.3e}"
                        )

                # Inner Solve
                inner_lr_scale = max(
                    float(self.cfg.inner_lr_outer_decay) ** k,
                    float(self.cfg.inner_lr_min_factor),
                )
                if self.cfg.verbose and (
                    k == 0
                    or k == self.cfg.outer_max_iter - 1
                    or (k > 0 and k % 5 == 0)
                ):
                    print(
                        "    [ALM-PGD] Inner LR continuation: "
                        f"k={k} lr={self.cfg.inner_lr * inner_lr_scale:.3e}"
                    )
                x, grad_norms, steps = (yield from invoke(self._solve_subproblem_adam,
                    x, mu_constr, rho, eta_k, problem, lb, ub,
                    constraint_scale=constraint_scale,
                    inner_lr_scale=inner_lr_scale,
                    deadline=start_time + global_timeout,
                    sliced=self.cfg.controller))
                last_grad_norms = grad_norms
                total_steps += steps
                global_steps += steps

                # Evaluate
                with torch.no_grad():
                    # 1. Forward Pass with Batch+Restart structure
                    x_in = x.reshape(N, *problem.input_shape)
                    out = problem.model(x_in) # (N, out_dim)
                    out = out.reshape(B, R, -1)  # (B, R, out_dim)

                    # 2. Objective and constraint residuals
                    phis, gs = _phi_and_residuals(problem, x, out, B, R)
                    if m > 0:
                        viols = torch.max(torch.relu(gs), dim=1).values
                        gs_alm = gs / constraint_scale
                        merit_viols = torch.max(
                            torch.relu(gs_alm), dim=1
                        ).values
                    else:
                        viols = torch.zeros(N, device=device, dtype=dtype)
                        gs_alm = gs
                        merit_viols = viols

                # Track best feasible objective and optionally stop after plateau.
                phis_batch = phis.reshape(B, R)
                viols_batch = viols.reshape(B, R)
                is_feas_batch = viols_batch <= self.cfg.feas_tol

                # Check feasibility per random restart
                # Best Feasible in current iter
                iter_feas_phis = phis_batch.clone()
                iter_feas_phis[~is_feas_batch] = float('inf')
                min_feas_phi, min_feas_idx = iter_feas_phis.min(dim=1)

                # Best Violation in current iter (fallback if best_is_feas is false)
                min_viol, min_viol_idx = viols_batch.min(dim=1)

                # Calculate masks
                # Improved Feasible: Found feas in current iter AND (never found feas OR new obj < best obj)
                found_feas = min_feas_phi != float('inf')
                improved_feas = found_feas & (~best_is_feas | (min_feas_phi < best_obj))

                # Better Violation: Not found feas in current iter (for this batch item) AND not found earlier AND new viol < best viol
                # This ensures we store the minimum violation point until we find a feasible point.
                # If we found feasible, we don't care about violation points anymore.
                better_viol = (
                    (~found_feas)
                    & (~best_is_feas)
                    & (min_viol < best_viol)
                )

                update_mask = improved_feas | better_viol

                if update_mask.any():
                    # Select best restart index for update
                    select_r = torch.where(improved_feas, min_feas_idx, min_viol_idx)

                    # Convert to flat index
                    # flat = b * R + r
                    batch_indices = torch.arange(B, device=device)
                    flat_indices = batch_indices * R + select_r

                    idx_upd = torch.where(update_mask)[0]
                    src_upd = flat_indices[idx_upd]

                    best_x[idx_upd] = x[src_upd].detach().clone()
                    # If improved_feas -> update obj with phi
                    # If better_viol -> update obj with phi (but it's not feas)

                    best_obj[idx_upd] = phis[src_upd]
                    best_viol[idx_upd] = viols[src_upd]
                    best_rho[idx_upd] = rho[src_upd]

                    if mu_constr.numel() > 0:
                        best_mu[idx_upd] = mu_constr[src_upd].detach().clone()
                        best_mu_scale[idx_upd] = (
                            constraint_scale[src_upd].detach().clone()
                        )

                    best_is_feas[improved_feas] = True

                if rival is not None:
                    rival_obj = rival()
                    if rival_obj is not None and math.isfinite(rival_obj):
                        # Test the basin with the existing count-limited endgame before
                        # buying more global search. Infeasible PGD contraction alone
                        # does not say whether restoration can change the certificate.
                        gap_tol = max(self.cfg.rival_abs_tol,
                                      self.cfg.rival_rel_tol * max(1., abs(rival_obj)))
                        if not restoration_trial:
                            # The measurement must not replace the PGD archive:
                            # a feasible trial would otherwise suppress GN/raw
                            # restoration of later, better infeasible basins.
                            global_archive = tuple(t.clone() for t in
                                (best_x, best_obj, best_viol, best_is_feas))
                            # Moving the endgame earlier must not enlarge its
                            # configured reserve or steal raw-polish time from SQP.
                            trial_start = time.time()
                            trial_reserve = timeout - global_timeout
                            trial_deadline = (min(deadline, trial_start + trial_reserve)
                                              if trial_reserve > 0 else deadline)
                            trial_polish_deadline = (min(trial_deadline, trial_start +
                                min(max(float(self.cfg.endgame_polish_frac), 0.), 1.) * trial_reserve)
                                if trial_reserve > 0 else trial_deadline)
                            best_x, best_obj, best_viol, best_is_feas, trial_steps = (
                                yield from invoke(self._batched_gn_restore,
                                    problem, best_x, best_obj, best_viol, best_is_feas,
                                    x, viols, deadline=trial_deadline, incumbent=incumbent,
                                    sliced=self.cfg.controller))
                            total_steps += trial_steps
                            best_x, best_obj, best_viol, best_is_feas, trial_steps = (
                                yield from invoke(self._polish_raw_feasibility,
                                    problem, best_x, best_obj, best_viol, best_is_feas,
                                    deadline=trial_polish_deadline, candidate_x=x, candidate_viols=viols,
                                    sliced=self.cfg.controller))
                            total_steps += trial_steps
                            best_x, best_obj, best_viol, best_is_feas, trial_steps = (
                                yield from invoke(self._optimality_polish,
                                    problem, best_x, best_obj, best_viol, best_is_feas,
                                    x, phis, viols, lb, ub, R, deadline=trial_deadline,
                                    seeds=polish_seeds, sliced=self.cfg.controller))
                            total_steps += trial_steps
                            restoration_trial = True
                            restored_archive = tuple(t.clone() for t in
                                (best_x, best_obj, best_viol, best_is_feas))
                            _, trial_obj, trial_viol, trial_feas = restored_archive
                            # A restored point at the rival's certificate resolution
                            # has spent this basin's residual credit. Retain every
                            # verified improvement, including sub-tolerance ones.
                            trial_exhausted = bool((trial_feas &
                                ((trial_obj - rival_obj).abs() <= gap_tol)).all())
                            best_x, best_obj, best_viol, best_is_feas = global_archive
                            if trial_exhausted:
                                print(f"    [ALM-PGD] Unified root: restored basin matches "
                                      f"the certificate after outer iteration {k}; "
                                      f"inner work={total_steps}")
                                break

                        # A material improvement moves the target too: reaching an
                        # old rival once cannot buy the rest of the root budget.
                        feasible_best = best_obj.masked_fill(~best_is_feas, float('inf')).min()
                        trial_best = trial_obj.masked_fill(~trial_feas, float('inf')).min()
                        target_inc = min(rival_obj, float(feasible_best), float(trial_best))
                        residual = _certificate_residual(
                            torch.cat((phis, best_obj, trial_obj)),
                            torch.cat((viols, best_viol, trial_viol)),
                            target_inc, self.cfg.rival_abs_tol, self.cfg.rival_rel_tol,
                            self.cfg.feas_tol).min()
                        initial = _certificate_residual(
                            phis_init, viols_init, target_inc,
                            self.cfg.rival_abs_tol, self.cfg.rival_rel_tol,
                            self.cfg.feas_tol).min()
                        # Restoration is a measured competitor for global search.
                        # Its residual floor replaces startup contraction credit:
                        # reducing infeasibility that restoration already removes
                        # cannot purchase another global episode.
                        restored = _certificate_residual(
                            trial_obj, trial_viol, target_inc,
                            self.cfg.rival_abs_tol, self.cfg.rival_rel_tol,
                            self.cfg.feas_tol).min()
                        initial = torch.minimum(initial, restored)
                        # Compare measured gain per completed inner step against the
                        # configured remaining work, independently of CPU/GPU speed.
                        remaining_work = (outer_limit - k - 1) * max(self.cfg.inner_steps, 0)
                        material_gain = (certificate_incumbent is None or
                                         target_inc < certificate_incumbent - gap_tol)
                        # A material primal gain is productive work too. It buys
                        # a fresh measurement against the new target, rather than
                        # being mistaken for a stall immediately after restoration.
                        if certificate_incumbent is None:
                            material_gain = target_inc < rival_obj - gap_tol
                        if material_gain:
                            rival_progress = _RivalProgress()
                        certificate_incumbent = target_inc
                        if rival_progress.stop(target_inc, float(residual), global_steps,
                                               remaining_work,
                                               initial_residual=None if material_gain else float(initial)):
                            print(f"    [ALM-PGD] Unified root: progress exhausted after outer iteration {k}: "
                                  f"target residual={float(residual):.6g}, work={global_steps} inner steps")
                            progress_stop = True
                            break

                # Early stopping based on best_obj improvement + cross-domain gap
                if self.cfg.obj_patience >= 0:
                    obj_improved = best_obj < prev_best_obj - self.cfg.obj_improvement_tol
                    no_obj_improve[obj_improved] = 0
                    no_obj_improve[~obj_improved] += 1
                    prev_best_obj = torch.min(prev_best_obj, best_obj)

                    if (no_obj_improve >= self.cfg.obj_patience).all() and best_is_feas.all():
                        # Check cross-domain gap: best domain obj vs second-best
                        gap_ok = True
                        if B > 1 and self.cfg.obj_gap_threshold > 0:
                            sorted_obj, _ = best_obj.sort()
                            gap = sorted_obj[1] - sorted_obj[0]
                            gap_ok = gap.item() >= self.cfg.obj_gap_threshold
                        if gap_ok:
                            if self.cfg.verbose:
                                msg = f"    [ALM-PGD] Early stop: best_obj unchanged for {self.cfg.obj_patience} iters"
                                if B > 1 and self.cfg.obj_gap_threshold > 0:
                                    msg += f" (domain gap={gap.item():.4f} >= thresh={self.cfg.obj_gap_threshold})"
                                print(msg)
                            break

                # First-feasibility fast exit: once ANY domain finds a feasible
                # solution, track the global best feasible objective.  If it does
                # not improve for `first_feas_patience` consecutive outer iters,
                # stop early.  This is critical in deep-BaB where B is huge but
                # only a handful of domains are feasible and the objective
                # plateaus from the very first inner solve.
                if self.cfg.first_feas_patience >= 0 and best_is_feas.any():
                    current_feas_obj = best_obj[best_is_feas].min()
                    if (feas_plateau_start is None
                            or current_feas_obj < best_global_feas_obj - self.cfg.obj_improvement_tol):
                        # First feasible found, or improvement detected — reset.
                        feas_plateau_start = k
                        best_global_feas_obj = current_feas_obj.clone()
                    elif k - feas_plateau_start >= self.cfg.first_feas_patience:
                        if self.cfg.verbose:
                            print(f"    [ALM-PGD] First-feas fast exit at k={k} "
                                  f"(best feas obj={best_global_feas_obj.item():.4f} "
                                  f"unchanged since k={feas_plateau_start}, "
                                  f"patience={self.cfg.first_feas_patience})")
                        break

                if self.cfg.controller:
                    yield Atom("alm", candidate(best_x, best_obj, best_viol, self.cfg.feas_tol))

                # ALM Update (Alg 1)
                cond = merit_viols <= eps_k
                good = cond
                bad = ~cond

                if self.cfg.update_multipliers_always:
                    # Standard inequality-ALM update. Unlike the legacy
                    # restart-wide epsilon gate, one stubborn row no longer
                    # freezes every multiplier in the restart.
                    if m > 0:
                        mu_constr = torch.clamp(
                            mu_constr + rho.unsqueeze(1) * gs_alm,
                            min=0.0,
                        )

                    if k >= 1:
                        improvement_ratio = (
                            prev_viols / viols.clamp(min=1e-12)
                        )
                        # Grow rho only when feasibility failed to improve by
                        # at least 25%; multiplier updates do the primary work.
                        inc = (
                            (improvement_ratio < 1.25)
                            & (viols > self.cfg.feas_tol)
                        )
                        if inc.any():
                            rho[inc] = torch.min(
                                self.cfg.kappa * rho[inc],
                                torch.tensor(
                                    self.cfg.max_rho, device=device, dtype=dtype
                                ),
                            )
                    eps_k = 1.0 / (rho ** self.cfg.alpha)
                    eta_k = 1.0 / (rho ** self.cfg.beta)
                else:
                    if m > 0 and good.any():
                        mu_constr[good] = torch.clamp(
                            mu_constr[good]
                            + rho[good].unsqueeze(1) * gs_alm[good],
                            min=0.0,
                        )

                    if good.any():
                        eps_k[good] /= rho[good]
                        eta_k[good] /= rho[good]

                    if bad.any():
                        # Follow Practical ALM: only increase rho when violation is still large.
                        # When violation is already small but fails the tightened eps_k, keep rho
                        # and reset eps_k/eta_k based on the current rho.
                        inc = bad & (viols > 1e-3)
                        if inc.any():
                            rho[inc] = torch.min(
                                self.cfg.kappa * rho[inc],
                                torch.tensor(self.cfg.max_rho, device=device, dtype=dtype),
                            )
                        eps_k[bad] = 1.0 / (rho[bad] ** self.cfg.alpha)
                        eta_k[bad] = 1.0 / (rho[bad] ** self.cfg.beta)

                # Cache optimization-space violations for the next progress check.
                if self.cfg.update_multipliers_always:
                    prev_viols = viols.detach().clone()

                eps_k = torch.max(eps_k, torch.tensor(self.cfg.eps_final, device=device, dtype=dtype))
                eta_k = torch.max(eta_k, torch.tensor(self.cfg.eta_final, device=device, dtype=dtype))

                if self.cfg.verbose:
                    best_idx_print = int(torch.argmin(viols).item())
                    b_idx = best_idx_print // R
                    r_idx = best_idx_print % R
                    iter_best_feas = None
                    if found_feas.any():
                        iter_best_feas = float(min_feas_phi[found_feas].min().item())
                    global_best_feas = None
                    if best_is_feas.any():
                        global_best_feas = float(best_obj[best_is_feas].min().item())
                    print(_format_alm_verbose_status(
                        k=k,
                        local_min_viol=float(viols[best_idx_print].item()),
                        local_obj_at_min_viol=float(phis[best_idx_print].item()),
                        local_domain=int(b_idx),
                        local_restart=int(r_idx),
                        rho=float(rho[best_idx_print].item()),
                        eps=float(eps_k[best_idx_print].item()),
                        eta=float(eta_k[best_idx_print].item()),
                        grad=float(last_grad_norms[best_idx_print].item()),
                        mu_norm=float(torch.norm(
                            mu_constr[best_idx_print]
                            / constraint_scale[best_idx_print]
                        ).item()),
                        feas_count=int(is_feas_batch.sum().item()),
                        total_count=int(N),
                        good_count=int(good.sum().item()),
                        bad_count=int(bad.sum().item()),
                        iter_best_feas_obj=iter_best_feas,
                        global_best_feas_obj=global_best_feas,
                    ))

            if self.cfg.controller:
                yield Atom("alm", candidate(best_x, best_obj, best_viol, self.cfg.feas_tol))

            # The global ALM phase locates a basin; an optional bounded local
            # phase targets the exact raw max-residual acceptance metric.
            #
            # Polish and Gauss-Newton previously shared one deadline, so a
            # 250-step / 64-clone polish could consume the whole endgame reserve
            # and leave GN with none.  Give polish a bounded share and keep the
            # rest for GN. A rival's point does not invalidate an infeasible basin.
            if (not trial_exhausted and (not restoration_trial or outer_iters_done > 1)
                    and time.time() < deadline):
                endgame_budget = timeout - global_timeout
                endgame_deadline = deadline
                endgame_start = start_time + global_timeout
                if progress_stop and endgame_budget > 0.:
                    # Stopping the global search hands over the configured reserve,
                    # not every second it chose to leave unused. Keep restoration
                    # affordable without discarding an infeasible promising basin.
                    endgame_start = time.time()
                    endgame_deadline = min(deadline, endgame_start + endgame_budget)
                polish_share = min(
                    max(float(self.cfg.endgame_polish_frac), 0.0), 1.0
                )
                polish_deadline = (
                    min(endgame_deadline, endgame_start + polish_share * endgame_budget)
                    if endgame_budget > 0.0
                    else start_time + timeout
                )
                if self.cfg.controller:
                    polish_deadline = deadline
                best_x, best_obj, best_viol, best_is_feas, bgn_steps = (
                    (yield from invoke(self._batched_gn_restore,
                        problem, best_x, best_obj, best_viol, best_is_feas,
                        x, viols, deadline=endgame_deadline,
                        incumbent=incumbent,
                        sliced=self.cfg.controller))
                )
                total_steps += bgn_steps
                if self.cfg.controller:
                    yield Atom("restore", candidate(best_x, best_obj, best_viol, self.cfg.feas_tol))
                best_x, best_obj, best_viol, best_is_feas, polish_steps = (
                    (yield from invoke(self._polish_raw_feasibility,
                        problem,
                        best_x,
                        best_obj,
                        best_viol,
                        best_is_feas,
                        deadline=polish_deadline,
                        candidate_x=x,
                        candidate_viols=viols,
                        sliced=self.cfg.controller))
                )
                total_steps += polish_steps
                if self.cfg.controller:
                    yield Atom("polish", candidate(best_x, best_obj, best_viol, self.cfg.feas_tol))
                best_x, best_obj, best_viol, best_is_feas, sqp_steps = (
                    (yield from invoke(self._optimality_polish,
                        problem, best_x, best_obj, best_viol, best_is_feas,
                        x, phis, viols, lb, ub, R,
                        deadline=endgame_deadline,
                        seeds=polish_seeds,
                        sliced=self.cfg.controller))
                )
                total_steps += sqp_steps

            if restored_archive is not None:
                trial_x, trial_obj, trial_viol, trial_feas = restored_archive
                take = ((trial_feas & (~best_is_feas | (trial_obj < best_obj))) |
                        (~trial_feas & ~best_is_feas & (trial_viol < best_viol)))
                best_x[take], best_obj[take] = trial_x[take], trial_obj[take]
                best_viol[take], best_is_feas[take] = trial_viol[take], trial_feas[take]

            # Return best: prefer feasible point with smallest objective; otherwise smallest violation.
            # Return best known results (batch-wise)
            success = best_is_feas
            if m > 0 and self.cfg.constraint_normalize:
                return_mu = best_mu / best_mu_scale
            else:
                return_mu = best_mu

            if success.all():
                status = "Optimal point found (ALM-PGD)"
            elif success.any():
                status = f"Partial Success ({success.sum()}/{B}) (ALM-PGD)"
            else:
                status = "ALM-PGD Completed (infeasible)"

            # Return ALMResult
            # If B=1, convert tensors to scalars for backward compatibility
            if B == 1:
                return ALMResult(
                    success=bool(success.item()),
                    x_solution=best_x[0], # (1, *input) -> (*input)
                    objective_value=float(best_obj.item()),
                    constraint_violation=float(best_viol.item()),
                    solve_time=time.time() - start_time,
                    status=status,
                    outer_iterations=outer_iters_done,
                    total_inner_steps=total_steps,
                    final_rho=float(best_rho.item()),
                    final_mu=return_mu[0].clone() if m > 0 else None,  # raw-space (m,)
                )
            else:
                return ALMResult(
                    success=success,
                    x_solution=best_x,
                    objective_value=best_obj,
                    constraint_violation=best_viol,
                    solve_time=time.time() - start_time,
                    status=status,
                    outer_iterations=outer_iters_done,
                    total_inner_steps=total_steps,
                    final_rho=best_rho,
                    final_mu=return_mu.clone() if m > 0 else None,  # raw-space (B, m)
                )

    @retained
    def _optimality_polish(
        self,
        problem: NNProblemSingleDomain,
        best_x: torch.Tensor,
        best_obj: torch.Tensor,
        best_viol: torch.Tensor,
        best_is_feas: torch.Tensor,
        x: torch.Tensor,
        phis: torch.Tensor,
        viols: torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
        restarts: int,
        deadline: Optional[float],
        seeds=None,
    ):
        """Batched SQP from distinct final restart points (see alm_polish).

        The archive only changes when a polished point is raw-feasible
        (``max(g, 0) <= feas_tol``) and improves its domain's objective.
        """
        count = int(self.cfg.opt_polish_candidates)
        if count <= 0 or x.shape[0] == 0:
            return best_x, best_obj, best_viol, best_is_feas, 0
        from .alm_polish import CandidateEvaluator, select_candidates, sqp_polish
        local_deadline = time.time() + max(float(self.cfg.opt_polish_time), 0.0)
        if self.cfg.controller:
            local_deadline = deadline
        elif deadline is not None:
            local_deadline = min(local_deadline, deadline)
        if time.time() >= local_deadline:
            return best_x, best_obj, best_viol, best_is_feas, 0

        B = problem.batch_size
        n_total = x.shape[0]
        n = problem.n_vars
        m = problem.n_constraints
        dom_lb = problem.dm_l.reshape(B, -1)
        dom_ub = problem.dm_u.reshape(B, -1)
        # The archived per-domain best joins the final restart points.
        pool_x = torch.cat([best_x.reshape(B, -1), x.reshape(n_total, -1)])
        pool_phi = torch.cat([best_obj, phis.reshape(-1)])
        pool_viol = torch.cat([best_viol, viols.reshape(-1)])
        pool_dom = torch.cat([
            torch.arange(B, device=x.device),
            torch.arange(n_total, device=x.device) // max(int(restarts), 1),
        ])
        pool_lb = torch.cat([dom_lb, lb.reshape(n_total, -1)])
        pool_ub = torch.cat([dom_ub, ub.reshape(n_total, -1)])
        # Per candidate the SQP keeps an n x n Hessian and Newton matrix and
        # m x n Jacobian products (dense); bound the memory, and skip problems
        # where even two candidates would not fit.
        per_candidate = max(n * n + 2 * m * n, 1)
        cap = (1 << 25) // per_candidate
        if cap < 2:
            if self.cfg.verbose:
                print(f"    [ALM-SQP] Optimality polish skipped: n={n}, m={m} too large")
            return best_x, best_obj, best_viol, best_is_feas, 0
        if seeds is not None:
            keep = max(cap // 2, 1)
            seeds = (seeds[0][:keep], seeds[1][:keep])
            cap = max(cap - seeds[0].shape[0], 1)
        count = min(count, cap)
        pick = select_candidates(
            pool_x, pool_phi, pool_viol, pool_lb, pool_ub, count,
            float(self.cfg.feas_tol),
            near_quantile=float(self.cfg.opt_polish_near_quantile),
        )
        dom = pool_dom[pick]
        start_x = pool_x[pick]
        if seeds is not None:
            # Restart start points: local solves from the same starts a
            # classical solver would use (box midpoint / warm start first).
            seed_x, seed_dom = seeds
            start_x = torch.cat([start_x, seed_x.reshape(seed_x.shape[0], -1)])
            dom = torch.cat([dom, seed_dom])
        if dom.numel() == 0:
            return best_x, best_obj, best_viol, best_is_feas, 0
        c_lb = dom_lb[dom]
        c_ub = dom_ub[dom]
        width = (c_ub - c_lb).clamp(min=torch.finfo(x.dtype).tiny)
        z0 = ((start_x - c_lb) / width).clamp(0.0, 1.0)
        before = best_obj[best_is_feas].min().item() if bool(best_is_feas.any()) else float("inf")
        t0 = time.time()
        with _eval_mode(problem.model):
            evaluator = CandidateEvaluator(problem, dom, c_lb, c_ub)
            z_best, phi_best, viol_best, found, iters = yield from sqp_polish(
                evaluator, z0,
                tol=float(self.cfg.feas_tol),
                target_frac=float(self.cfg.opt_polish_target_frac),
                sliced=self.cfg.controller, _iterate=True,
                max_iters=int(self.cfg.opt_polish_iters),
                deadline=local_deadline,
                verbose=self.cfg.verbose,
            )
        # Re-evaluate exactly the points that would be returned (clamped into
        # their domain box); acceptance is decided on these values only.
        x_new = torch.minimum(torch.maximum(c_lb + z_best * (c_ub - c_lb), c_lb), c_ub)
        with torch.no_grad(), _eval_mode(problem.model):
            phi_best, g_new, _ = evaluator.at(x_new)
        viol_best = torch.relu(g_new).amax(dim=1) if g_new.shape[1] else torch.zeros_like(phi_best)
        found = found & torch.isfinite(phi_best) & (viol_best <= float(self.cfg.feas_tol))
        x_new = x_new.reshape(-1, *problem.input_shape)
        improved = 0
        for k in torch.where(found)[0].tolist():
            d = int(dom[k].item())
            if (not bool(best_is_feas[d].item())) or phi_best[k] < best_obj[d]:
                best_x[d] = x_new[k]
                best_obj[d] = phi_best[k]
                best_viol[d] = viol_best[k]
                best_is_feas[d] = True
                improved += 1
        if self.cfg.verbose:
            after = best_obj[best_is_feas].min().item() if bool(best_is_feas.any()) else float("inf")
            print(
                "    [ALM-SQP] Optimality polish: "
                f"candidates={int(pick.numel())}+{int(dom.numel() - pick.numel())} iters={iters} "
                f"feasible={int(found.sum().item())} improved={improved} "
                f"best_feas_obj={before:.6g}->{after:.6g} "
                f"time={time.time() - t0:.2f}s"
            )
        return best_x, best_obj, best_viol, best_is_feas, iters

    def _polish_seed_points(self, x: torch.Tensor, batch: int, restarts: int):
        """Start points kept for the optimality polish: ``opt_polish_seeds``
        restarts of a single domain (restart 0 is the midpoint or warm start),
        or restart 0 of the first domains of a batch."""
        count = int(self.cfg.opt_polish_seeds)
        if count <= 0 or int(self.cfg.opt_polish_candidates) <= 0:
            return None
        x_br = x.reshape(batch, restarts, -1)
        if batch == 1:
            seed_x = x_br[0, :min(count, restarts)]
            seed_dom = torch.zeros(seed_x.shape[0], dtype=torch.long, device=x.device)
        else:
            seed_x = x_br[:min(count, batch), 0]
            seed_dom = torch.arange(seed_x.shape[0], device=x.device)
        return seed_x.detach().clone(), seed_dom

