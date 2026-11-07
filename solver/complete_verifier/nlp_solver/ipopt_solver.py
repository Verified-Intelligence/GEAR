"""IPOPT-based falsification solver.

This module provides IPOPT solver integration to solve
the root primal problem for finding counterexamples before Branch-and-Bound starts.

Backend preference:
  1. ``cyipopt`` for a real IPOPT solve.
  2. ``scipy.optimize.trust-constr`` as a local-solver fallback when IPOPT
     bindings are unavailable.

Requirements:
    pip install cyipopt
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Callable
import torch
from nlp_solver.utils import locked_jvp
import numpy as np
from input_constraints import normalize_input_linear_constraints

# Import shared utilities from nlp_solver package
from nlp_solver.utils import _eval_mode, _split_or_clauses
from nlp_solver.problem import NNProblemSingleDomain
from nlp_solver.sparse_derivatives import output_walk, sparse_problem

try:
    import cyipopt
    CYIPOPT_AVAILABLE = True
    CYIPOPT_IMPORT_ERROR = None
except ImportError as exc:
    CYIPOPT_AVAILABLE = False
    CYIPOPT_IMPORT_ERROR = str(exc)

try:
    from scipy.optimize import NonlinearConstraint, minimize
    SCIPY_AVAILABLE = True
    SCIPY_IMPORT_ERROR = None
except ImportError as exc:
    NonlinearConstraint = None
    minimize = None
    SCIPY_AVAILABLE = False
    SCIPY_IMPORT_ERROR = str(exc)

IPOPT_AVAILABLE = CYIPOPT_AVAILABLE or SCIPY_AVAILABLE


@contextmanager
def _suppress_ipopt_info_logs(enabled: bool):
    """Mute cyipopt INFO spam unless verbose logging is explicitly requested."""
    if enabled:
        yield
        return

    root_logger = logging.getLogger()
    cyipopt_logger = logging.getLogger('cyipopt')
    root_level = root_logger.level
    cyipopt_level = cyipopt_logger.level
    try:
        if root_level == logging.NOTSET or root_level < logging.WARNING:
            root_logger.setLevel(logging.WARNING)
        if cyipopt_level == logging.NOTSET or cyipopt_level < logging.WARNING:
            cyipopt_logger.setLevel(logging.WARNING)
        yield
    finally:
        root_logger.setLevel(root_level)
        cyipopt_logger.setLevel(cyipopt_level)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class IPOPTResult:
    """Result container for IPOPT solver."""
    success: bool                           # Whether counterexample found
    x_solution: Optional[np.ndarray]        # Input point (counterexample, numpy)
    objective_value: Optional[float]        # Objective at solution
    constraint_violation: Optional[float]   # Max constraint violation
    solve_time: float                       # Wall-clock time
    status: str                             # Solver status string
    num_iterations: int                     # Solver iterations
    x_last: Optional[np.ndarray] = None     # Returned point even when rejected
    exit_code: Optional[int] = None         # Ipopt ApplicationReturnStatus (0 solved, 1 acceptable)


def _output_supports(model, x0: np.ndarray, lower: np.ndarray, input_shape):
    """Sparse (outputs x inputs) matrix of the inputs each model output depends on, from
    the structure walk (None when some output's support is unknown). Cached on the model:
    the in-BaB IPOPT builds a new problem per call. The walk is bounded by its operation
    budget, not by time."""
    if hasattr(model, "_ipopt_output_supports"):
        return model._ipopt_output_supports
    supports = None
    try:
        import scipy.sparse as sp
        forms, algebra = output_walk(model, x0, lower, input_shape)
        found = [algebra.support(f) for f in forms]
        if all(s is not None for s in found):
            sizes = [len(s) for s in found]
            supports = sp.csr_matrix(
                (np.ones(sum(sizes), dtype=bool),
                 np.fromiter((v for s in found for v in sorted(s)), dtype=np.int64, count=sum(sizes)),
                 np.concatenate([[0], np.cumsum(sizes)])),
                shape=(len(found), int(np.prod(input_shape))))
    except Exception:  # noqa: BLE001 - an op the walk cannot follow: probe the Jacobian instead
        pass
    try:
        model._ipopt_output_supports = supports
    except AttributeError:
        pass
    return supports


# ============================================================================
# IPOPT Problem
# ============================================================================

class NNProblemIPOPT(NNProblemSingleDomain):
    """IPOPT uses the same single-domain problem semantics as SciPy/ALM.

    Ipopt calls these callbacks one point at a time. On a GPU every call pays launch
    and synchronization latency, and the row-by-row Jacobian of the base class paid it
    once per constraint: 0.13-0.23 s per Jacobian for 50-100 rows, so the root IPOPT
    ran out of time on 81 of the 96 paper instances without a primal. Small problems
    are therefore evaluated on a CPU copy of the model, and the Jacobian is formed in
    one vectorized reverse pass (0.4-0.6 ms, identical values).
    """

    CPU_EVAL_MAX_ENTRIES = 4_000_000     # n_vars * n_constraints

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self._source_model = model      # the cache key of the structure walk
        if (torch.device(self.device).type == 'cuda'
                and self.n_vars * max(self.n_constraints, 1) <= self.CPU_EVAL_MAX_ENTRIES):
            import copy
            self.model = copy.deepcopy(self.model).to('cpu')
            self.device = torch.device('cpu')
            for name in ('C_obj', 'rhs_obj', 'C_constr', 'rhs_constr', 'dm_l', 'dm_u',
                         'input_constr_A', 'input_constr_b'):
                value = getattr(self, name, None)
                if isinstance(value, torch.Tensor):
                    setattr(self, name, value.to('cpu'))

    _jac_color = None
    _jac_ncolors = 0

    def structural_jacobian_pattern(self, x0: np.ndarray):
        """(rows, cols) of the constraint Jacobian from the structure walk: the union of the
        input supports of the output elements each row combines, plus the nonzeros of the
        input rows. A superset of the true nonzeros. None when the walk cannot give every
        output's support. Replaces four dense probe Jacobians (polygon75: 90 s, polygon100:
        370 s, pooling_sppa5pq: 35-457 s at m*n of 26M-150M) by ~0.2 s."""
        import scipy.sparse as sp
        supports = _output_supports(self._source_model, np.clip(x0, self.lb, self.ub),
                                    self.lb, self.input_shape)
        if supports is None or supports.shape[0] != self.C_constr.shape[-1]:
            return None
        pairs = []
        for matrix in (self.C_constr, self.input_constr_A):
            # one batch entry; a problem without output rows keeps an unbatched (0, m) C_constr
            # (ex6_2_11, st_e24 crashed indexing it)
            matrix = matrix[0] if matrix.ndim == 3 else matrix.reshape(-1, matrix.shape[-1])
            r, c = (matrix != 0).nonzero().T.cpu().numpy()
            pairs.append(sp.csr_matrix((np.ones(r.size, dtype=bool), (r, c)), shape=matrix.shape))
        pattern = sp.vstack([pairs[0] @ supports, pairs[1]]).tocoo()
        order = np.lexsort((pattern.col, pattern.row))
        return pattern.row[order], pattern.col[order]

    def set_jacobian_sparsity(self, rows: np.ndarray, cols: np.ndarray) -> None:
        """Colour the columns of the output-constraint Jacobian so that no two columns
        of one colour share a row; the Jacobian then takes one forward-mode product per
        colour instead of one reverse pass per row (pooling_foulds3stp: 568 ms per
        dense Jacobian for 2131 x 832 entries at 0.48% density). Forward-mode products
        also beat the reverse pass when there are far more rows than colours: polygon75
        (2849 rows, 150 columns, every column coupled to every other) takes 0.41 s per
        Jacobian with one product per column against 5.6 s for the reverse pass."""
        keep = rows < self.n_output_constraints
        rows, cols = np.asarray(rows)[keep], np.asarray(cols)[keep]
        if rows.size == 0:
            return
        by_col = [[] for _ in range(self.n_vars)]
        by_row = [[] for _ in range(self.n_output_constraints)]
        for r, c in zip(rows.tolist(), cols.tolist()):
            by_col[c].append(r)
            by_row[r].append(c)
        color = np.full(self.n_vars, -1, dtype=np.int64)
        for c in range(self.n_vars):
            if not by_col[c]:
                continue
            used = {color[o] for r in by_col[c] for o in by_row[r] if color[o] >= 0}
            k = 0
            while k in used:
                k += 1
            color[c] = k
        ncolors = int(color.max()) + 1
        if ncolors >= max(self.n_vars, self.n_output_constraints) // 2:
            return              # no saving over the dense reverse pass
        self._jac_color = torch.as_tensor(np.maximum(color, 0), device=self.device)
        self._jac_active = torch.as_tensor(color >= 0, device=self.device)
        self._jac_ncolors = ncolors
        self._jac_rows = torch.as_tensor(rows, device=self.device)
        self._jac_cols = torch.as_tensor(cols, device=self.device)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        if self.n_constraints == 0 or self.n_output_constraints == 0:
            return super().jacobian(x)
        x_flat = self._to_torch(x).reshape(-1)

        def output_constraints(z):
            out = self.model(z.view(1, *self.input_shape))
            return ((self.C_constr @ out.T).squeeze(-1) - self.rhs_constr).reshape(-1)

        jac = None
        if self._jac_color is not None:
            try:
                seeds = torch.nn.functional.one_hot(
                    self._jac_color, self._jac_ncolors).T.to(x_flat)
                seeds = seeds * self._jac_active.to(x_flat)
                compressed = torch.func.vmap(
                    lambda v: locked_jvp(output_constraints, (x_flat,), (v,))[1])(seeds)
                jac = torch.zeros(self.n_output_constraints, self.n_vars, dtype=x_flat.dtype,
                                  device=x_flat.device)
                jac[self._jac_rows, self._jac_cols] = compressed[
                    self._jac_color[self._jac_cols], self._jac_rows]
            except Exception:  # noqa: BLE001 - an op without forward-mode support
                self._jac_color = None
                jac = None
        if jac is None:
            try:
                jac = torch.func.jacrev(output_constraints)(x_flat).reshape(-1, self.n_vars)
            except Exception:  # noqa: BLE001 - an op without vmap support: row loop
                return super().jacobian(x)
        if self.n_input_constraints > 0:
            jac = torch.cat([jac, self.input_constr_A.reshape(-1, self.n_vars).to(jac)], dim=0)
        return jac.detach().cpu().numpy().astype(np.float64)


# ============================================================================
# IPOPT Solver
# ============================================================================

class IPOPTSolver:
    """
    IPOPT solver wrapper (cyipopt, with a SciPy fallback) for neural network falsification.

    The PyTorch network supplies objective, constraints and Jacobians through callbacks.
    """

    def __init__(self, config: Dict[str, Any]):
        self.controller = bool(config.get('controller', False))
        self.controller_stop = config.get('_controller_stop')
        self.controller_deadline = config.get('_controller_deadline')
        self.controller_monotonic_deadline = config.get('_controller_monotonic_deadline')
        self.controller_point = config.get('_controller_point') if config.get('controller', False) else None
        self.max_iter = config.get('ipopt_max_iter', config.get('max_iter', 1000))
        self.tol = config.get('ipopt_tol', config.get('tol', 1e-6))
        self.constr_viol_tol = config.get('ipopt_constr_viol_tol', config.get('constr_viol_tol', 1e-4))
        # Aim at g <= slack instead of g <= 0: feasibility is g <= constr_viol_tol for the
        # acceptance check and for every dual cut, so an incumbent that stops at exact
        # feasibility leaves a gap the dual can never close.
        self.constraint_slack = float(config.get('ipopt_constraint_slack_frac', 0.0)) * float(
            self.constr_viol_tol)
        self.timeout = config.get('ipopt_timeout', config.get('timeout', 60.0))
        self.verbose = config.get('ipopt_verbose', config.get('verbose', False))
        self.linear_solver = config.get('ipopt_linear_solver', 'mumps')
        self.hessian_approximation = config.get('ipopt_hessian_approximation', 'limited-memory')
        # Callbacks from the exact forms of the structure walk (sparse_derivatives), with the
        # exact Hessian where every atom is twice differentiable: the torch callbacks cost
        # 12-96 ms per Jacobian on the large pooling/wastewater models, and the quasi-Newton
        # Hessian ran IPOPT into its wall cap there.
        self.sparse_derivatives = bool(config.get('sparse_derivatives', False))

    def _solve_single_cyipopt(
        self,
        problem: NNProblemIPOPT,
        x0: np.ndarray,
        timeout: float,
        slack: Optional[float] = None,
        keep_in_box: bool = False,
        merge_rows: bool = True,
    ) -> IPOPTResult:
        """Solve one IPOPT call using a real IPOPT backend via cyipopt."""
        if slack is None:
            slack = self.constraint_slack
        start_time = time.time()
        n_vars = problem.n_vars
        n_constraints = problem.n_constraints

        wall_deadline = start_time + float(max(timeout, 1e-3))
        # A solve that is nearly feasible at the deadline may finish past it (at most
        # half the budget more): pooling_foulds5stp converged in 29.8 s of a 30 s budget
        # in one run and was cut off at 30.1 s in the next, leaving no incumbent.
        grace_deadline = wall_deadline + 0.5 * float(max(timeout, 1e-3))
        grace_viol = 10.0 * float(self.constr_viol_tol)

        class _CyIpoptProblem:
            def __init__(self, outer, prob):
                self.outer = outer
                self.prob = prob
                self.num_iterations = 0

            def intermediate(self, alg_mod, iter_count, obj_value, inf_pr,
                             inf_du, mu, d_norm, regularization_size,
                             alpha_du, alpha_pr, ls_trials):
                # Ipopt's max_cpu_time counts CPU seconds only; with GPU-bound
                # function evaluations that can be ~30x wall time. Returning
                # False stops Ipopt at the current iterate.
                self.num_iterations = int(iter_count)
                if self.outer.controller_stop is not None and self.outer.controller_stop.is_set():
                    return False
                if (self.outer.controller_monotonic_deadline is not None
                        and time.monotonic() >= self.outer.controller_monotonic_deadline):
                    return False
                now = time.time()
                if self.outer.controller_deadline is not None and now >= self.outer.controller_deadline:
                    return False
                return now <= wall_deadline or (now <= grace_deadline and inf_pr <= grace_viol)

            def objective(self, x):
                obj_val, _ = self.outer._pytorch_objective_and_constraints(self.prob, x)
                return obj_val

            def gradient(self, x):
                grad_obj, _ = self.outer._pytorch_gradients(self.prob, x)
                return grad_obj

            def constraints(self, x):
                _, constr_vals = self.outer._pytorch_objective_and_constraints(self.prob, x)
                return constr_vals

            def jacobian(self, x):
                _, jac = self.outer._pytorch_gradients(self.prob, x)
                jac = np.asarray(jac)
                if jac.size != n_constraints * n_vars:
                    return jac.reshape(-1)
                return jac.reshape(n_constraints, n_vars)[jac_rows, jac_cols]

            def jacobianstructure(self):
                return jac_rows, jac_cols

        class _SparseCyIpoptProblem(_CyIpoptProblem):
            # the exact forms: rows merged by exact direction, an equality is one ranged row
            def objective(self, x):
                return sparse.objective(x)

            def gradient(self, x):
                return sparse.gradient(x)

            def constraints(self, x):
                return sparse.constraints(x)

            def jacobian(self, x):
                return sparse.jacobian(x)

            def jacobianstructure(self):
                return sparse.jacobianstructure()

            def hessianstructure(self):
                return sparse.hessianstructure()

            def hessian(self, x, lam, obj_factor):
                return sparse.hessian(x, lam, obj_factor)

        sparse = sparse_problem(problem) if self.sparse_derivatives else None
        if sparse is not None and not merge_rows:
            sparse = sparse.unmerged()
        if sparse is None:
            pattern = self._jacobian_pattern(problem, x0, wall_deadline)
            if pattern is None or time.time() > wall_deadline:
                return IPOPTResult(
                    success=False, x_solution=None, objective_value=None,
                    constraint_violation=None, solve_time=time.time() - start_time,
                    status="cyipopt: setup exceeded the deadline", num_iterations=0)
            jac_rows, jac_cols = pattern

        try:
            if sparse is None:
                cl = np.full(n_constraints, -np.inf, dtype=np.float64)
                cu = np.full(n_constraints, slack, dtype=np.float64)
                m, callbacks = n_constraints, _CyIpoptProblem(self, problem)
            else:
                (cl, cu), m = sparse.bounds(slack), sparse.m
                callbacks = _SparseCyIpoptProblem(self, problem)
            with _suppress_ipopt_info_logs(self.verbose):
                nlp = cyipopt.Problem(
                    n=n_vars,
                    m=m,
                    problem_obj=callbacks,
                    lb=problem.lb,
                    ub=problem.ub,
                    cl=cl,
                    cu=cu,
                )
            nlp.add_option('tol', float(self.tol))
            # Ipopt measures violation against cu; slack + its tolerance stays inside ours.
            ipopt_cvt = float(self.constr_viol_tol) if slack <= 0 else \
                max(0.5 * (float(self.constr_viol_tol) - slack), 1e-9)
            nlp.add_option('constr_viol_tol', ipopt_cvt)
            nlp.add_option('acceptable_constr_viol_tol', ipopt_cvt)
            nlp.add_option('max_iter', int(self.max_iter))
            # The wall deadline is enforced by `intermediate`. max_cpu_time counts the
            # whole process's CPU time -- including the ALM thread running beside IPOPT
            # at the root and BLAS threads -- and cut IPOPT off at a fraction of its
            # wall budget, so it is only a far backstop here.
            nlp.add_option('max_cpu_time', 64.0 * float(max(timeout, 1e-3)))
            nlp.add_option('print_level', 5 if self.verbose else 0)
            nlp.add_option('sb', 'yes')
            nlp.add_option('hessian_approximation',
                           'exact' if sparse is not None and sparse.hessian_available
                           else self.hessian_approximation)
            nlp.add_option('linear_solver', self.linear_solver)
            # Exact derivatives can be non-finite at a point (x^k, k < 2, at x = 0); unchecked,
            # they reach the linear solver, which crashed the process on eq6_1 (A49).
            nlp.add_option('check_derivatives_for_naninf', 'yes')
            if keep_in_box:
                # With Ipopt's default relaxation (1e-8 |bound|) the returned point is
                # clipped back, which on large-coefficient models moved the constraints
                # by ~3e-4 and the point failed acceptance (waterund01/08).
                nlp.add_option('bound_relax_factor', 0.0)

            with _suppress_ipopt_info_logs(self.verbose):
                x_sol, info = nlp.solve(x0.astype(np.float64))
            solve_time = time.time() - start_time
            x_sol = np.clip(np.asarray(x_sol, dtype=np.float64), problem.lb, problem.ub)

            obj_val, constr_vals = self._pytorch_objective_and_constraints(problem, x_sol)
            max_violation = float(max(0.0, constr_vals.max())) if len(constr_vals) > 0 else 0.0
            is_feasible = max_violation <= self.constr_viol_tol
            success = is_feasible and (obj_val < 0)

            status_msg = info.get('status_msg', '') if isinstance(info, dict) else ''
            if isinstance(status_msg, bytes):
                status_msg = status_msg.decode('utf-8', errors='replace')
            if not status_msg:
                status_msg = (
                    f"cyipopt status={info.get('status')}"
                    if isinstance(info, dict) else "cyipopt"
                )
            status_msg = f"cyipopt: {status_msg}"
            num_iterations = callbacks.num_iterations

            return IPOPTResult(
                success=success,
                x_solution=x_sol if is_feasible else None,
                objective_value=obj_val,
                constraint_violation=max_violation,
                solve_time=solve_time,
                status=status_msg,
                num_iterations=num_iterations,
                x_last=x_sol,
                exit_code=info.get('status') if isinstance(info, dict) else None,
            )
        except Exception as e:
            return IPOPTResult(
                success=False, x_solution=None, objective_value=None,
                constraint_violation=None, solve_time=time.time() - start_time,
                status=f"cyipopt exception: {str(e)}", num_iterations=0
            )

    @staticmethod
    def _jacobian_pattern(problem, x0: np.ndarray, deadline: float):
        """Structural nonzeros of the constraint Jacobian, cached on the problem: the
        structure walk's supports (NNProblemIPOPT), else the union over the start point and
        three random box points. None when probing ran past `deadline`.

        With a dense pattern Ipopt's linear solver factors a dense KKT system every
        iteration: on pooling_foulds3stp (832 vars, 2214 rows, a few variables per row)
        60 s gave no feasible point. A missed entry can only slow Ipopt down; its final
        point is verified exactly.
        """
        cached = getattr(problem, "_ipopt_jac_pattern", None)
        if cached is not None:
            return cached
        m, n = problem.n_constraints, problem.n_vars
        if m == 0:
            rows = cols = np.array([], dtype=np.int32)
        else:
            pattern = getattr(problem, "structural_jacobian_pattern", lambda x: None)(x0)
            if pattern is None:
                lb = np.where(np.isfinite(problem.lb), problem.lb, x0 - 1.0)
                ub = np.where(np.isfinite(problem.ub), problem.ub, x0 + 1.0)
                rng = np.random.default_rng(0)
                points = [np.clip(x0, problem.lb, problem.ub)]
                points += [lb + (ub - lb) * rng.random(n) for _ in range(3)]
                mask = np.zeros((m, n), dtype=bool)
                for pt in points:
                    if time.time() > deadline:
                        return None
                    jac = np.asarray(problem.jacobian(pt))
                    if jac.size != m * n:       # unexpected layout: keep the dense pattern
                        mask[:] = True
                        break
                    jac = jac.reshape(m, n)
                    mask |= (jac != 0) | ~np.isfinite(jac)
                pattern = np.nonzero(mask)
            rows, cols = (a.astype(np.int32) for a in pattern)
            if hasattr(problem, "set_jacobian_sparsity"):
                problem.set_jacobian_sparsity(rows, cols)
        try:
            problem._ipopt_jac_pattern = (rows, cols)
        except AttributeError:
            pass
        return rows, cols

    def _pytorch_objective_and_constraints(
        self,
        problem: NNProblemIPOPT,
        x: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Evaluate objective and constraints with the shared single-domain logic."""
        return problem.objective(x), problem.constraints(x)

    def _pytorch_gradients(
        self,
        problem: NNProblemIPOPT,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients with the shared single-domain logic."""
        return problem.gradient(x), problem.jacobian(x)

    def solve_single(
        self,
        problem: NNProblemIPOPT,
        x_init: Optional[np.ndarray] = None,
        timeout: Optional[float] = None,
    ) -> IPOPTResult:
        """Solve the NLP with IPOPT from a single initial point."""
        if timeout is None:
            timeout = self.timeout

        if x_init is None:
            x0 = 0.5 * (problem.lb + problem.ub)
        else:
            x0 = np.clip(x_init, problem.lb, problem.ub)

        if CYIPOPT_AVAILABLE:
            start = time.time()
            result = self._solve_with_slacks(problem, x0, timeout)
            remaining = float(timeout) - (time.time() - start)
            if (result.x_solution is None and self.sparse_derivatives and remaining > 0
                    and sparse_problem(problem) is not None):
                if self.verbose:
                    print('[IPOPT] Merged sparse solve rejected; retrying original inequality rows')
                # Restart at the caller's seed: the failed merged barrier solve may
                # already have moved to a different basin. Share the original budget.
                raw = self._solve_with_slacks(problem, x0, remaining, merge_rows=False)
                if (raw.x_solution is not None or result.constraint_violation is None
                        or (raw.constraint_violation is not None
                            and raw.constraint_violation < result.constraint_violation)):
                    result = raw
            result.solve_time = time.time() - start
            return result

        return self._solve_scipy(problem, x0, timeout)

    def _solve_with_slacks(self, problem, x0, timeout, merge_rows=True):
        """One row representation, with the existing acceptance/polish schedule."""
        def solve(x, budget, **kwargs):
            if not merge_rows:
                kwargs['merge_rows'] = False
            return self._solve_single_cyipopt(problem, x, budget, **kwargs)

        if CYIPOPT_AVAILABLE:
            # Aim at half the tolerance first: an interrupted solve aimed straight at a
            # high slack can stop just outside the tolerance and be rejected
            # (pooling_foulds5tp at 0.99). A higher target is reached by polishing the
            # accepted point below.
            target = self.constraint_slack
            first = min(target, 0.5 * float(self.constr_viol_tol))
            result = solve(x0, timeout, slack=first)
            if self.controller_point is not None and result.x_solution is not None:
                self.controller_point(result.x_solution)
            # A rejected point is finished from where Ipopt stopped, inside the box (the
            # clipped point of a relaxed-bound solve can miss acceptance; keeping iterates
            # in the box from the start instead stalled Ipopt on wastewater04m2, 1000
            # iterations at 124.4 against 89.8 in 1.8 s) and without slack (Ipopt's own
            # violation measure is not a bound on max g).
            if result.x_solution is None and result.x_last is not None:
                remaining = float(timeout) - result.solve_time
                if remaining > (0.0 if self.controller else 0.1 * float(timeout)):
                    retry = solve(result.x_last, remaining, slack=0.0, keep_in_box=True)
                    if self.controller_point is not None and retry.x_solution is not None:
                        self.controller_point(retry.x_solution)
                    if retry.x_solution is not None:
                        retry.solve_time += result.solve_time
                        return retry
            elif result.x_solution is not None and target > first:
                remaining = float(timeout) - result.solve_time
                if remaining > (0.0 if self.controller else 0.1 * float(timeout)):
                    polish = solve(result.x_solution, remaining, slack=target)
                    if self.controller_point is not None and polish.x_solution is not None:
                        self.controller_point(polish.x_solution)
                    polish.solve_time += result.solve_time
                    if (polish.x_solution is not None and polish.objective_value is not None
                            and result.objective_value is not None
                            and polish.objective_value < result.objective_value):
                        return polish
                    result.solve_time = polish.solve_time
            return result

    def _solve_scipy(self, problem, x0, timeout):
        if not SCIPY_AVAILABLE:
            no_backend_reason = "No local solver backend available (need cyipopt or scipy)"
            if CYIPOPT_IMPORT_ERROR:
                no_backend_reason += f"; cyipopt import failed: {CYIPOPT_IMPORT_ERROR}"
            if SCIPY_IMPORT_ERROR:
                no_backend_reason += f"; scipy import failed: {SCIPY_IMPORT_ERROR}"
            return IPOPTResult(
                success=False, x_solution=None, objective_value=None,
                constraint_violation=None, solve_time=0.0,
                status=no_backend_reason, num_iterations=0
            )

        n_vars = problem.n_vars
        n_constraints = problem.n_constraints

        start_time = time.time()

        try:
            # Use scipy.optimize.minimize as a local-solver fallback when the
            # real IPOPT Python binding is unavailable.
            def objective(x):
                obj_val, _ = self._pytorch_objective_and_constraints(problem, x)
                return obj_val

            def objective_grad(x):
                grad_obj, _ = self._pytorch_gradients(problem, x)
                return grad_obj

            constraints = []
            if n_constraints > 0:
                def constraint_func(x):
                    _, constr_vals = self._pytorch_objective_and_constraints(problem, x)
                    return constr_vals

                def constraint_jac(x):
                    _, jac = self._pytorch_gradients(problem, x)
                    return jac

                constraints.append(NonlinearConstraint(
                    constraint_func,
                    -np.inf, 0.0,  # g(x) <= 0
                    jac=constraint_jac
                ))

            # Fallback when cyipopt is not installed.
            result = minimize(
                objective,
                x0,
                method='trust-constr',
                jac=objective_grad,
                bounds=list(zip(problem.lb, problem.ub)),
                constraints=constraints,
                options={
                    'maxiter': self.max_iter,
                    'gtol': self.tol,
                    'verbose': 2 if self.verbose else 0,
                }
            )

            solve_time = time.time() - start_time

            # Evaluate solution
            obj_val, constr_vals = self._pytorch_objective_and_constraints(problem, result.x)
            max_violation = float(max(0.0, constr_vals.max())) if len(constr_vals) > 0 else 0.0

            # Check feasibility and success
            is_feasible = max_violation <= self.constr_viol_tol
            success = is_feasible and (obj_val < 0)

            return IPOPTResult(
                success=success,
                x_solution=result.x if is_feasible else None,
                objective_value=obj_val,
                constraint_violation=max_violation,
                solve_time=solve_time,
                status=(
                    "trust-constr fallback: "
                    + (result.message if hasattr(result, 'message') else str(result.success))
                ),
                num_iterations=result.nit if hasattr(result, 'nit') else 0,
            )

        except Exception as e:
            return IPOPTResult(
                success=False, x_solution=None, objective_value=None,
                constraint_violation=None, solve_time=time.time() - start_time,
                status=f"Exception: {str(e)}", num_iterations=0
            )


# ============================================================================
# Helper Functions
# ============================================================================

def _solve_ipopt_single_clause(
    problem: NNProblemIPOPT,
    solver: IPOPTSolver,
    config: Dict[str, Any],
    verbose: bool = False,
    phase_name: str = "IPOPT",
) -> Tuple[Optional[IPOPTResult], Optional[np.ndarray]]:
    """
    Solve a single OR clause using one IPOPT call.

    Returns:
        Tuple of (best_result, x_solution) where x_solution is set
        if a counterexample was found.
    """
    timeout = config.get('ipopt_timeout', config.get('timeout', 60.0))

    result = solver.solve_single(
        problem,
        timeout=timeout,
    )

    if verbose:
        obj_str = f"{result.objective_value:.6f}" if result.objective_value is not None else "None"
        viol_str = f"{result.constraint_violation:.2e}" if result.constraint_violation is not None else "None"
        print(f"  [{phase_name}] obj={obj_str}, viol={viol_str}, "
              f"iters={result.num_iterations}, time={result.solve_time:.2f}s, "
              f"status={result.status}")

    if result.success:
        return result, result.x_solution

    return result, None


# ============================================================================
# Main Entry Point
# ============================================================================

def ipopt_falsification(
    _net,  # Unused but kept for API compatibility
    model_ori: Callable,
    dm_l: torch.Tensor,
    dm_u: torch.Tensor,
    C_mat: torch.Tensor,
    rhs_mat: torch.Tensor,
    or_spec_size: torch.Tensor,
    config: Optional[Dict] = None,
    base_input_linear_constraints: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[bool, Optional[torch.Tensor], Optional[float]]:
    """
    Main entry point for IPOPT-based falsification at root node.

    Properly handles OR-specs by iterating over all OR clauses. Finding a
    counterexample for ANY clause means falsification succeeded.

    Config options:
        ipopt_verbose: bool - Print detailed output
        ipopt_max_iter: int - Maximum IPOPT iterations
        ipopt_timeout: float - IPOPT solve timeout
        ipopt_tol: float - Optimality tolerance
        ipopt_constr_viol_tol: float - Constraint violation tolerance

    Returns:
        Tuple of (found_optimal, x_solution, objective_value)
        - found_optimal: True if counterexample found (obj < 0 and feasible)
        - x_solution: The optimal input point (if found), as torch.Tensor
        - objective_value: Best objective value found (even if not optimal)
    """
    if not IPOPT_AVAILABLE:
        print("  [IPOPT] Warning: no IPOPT backend available, skipping IPOPT falsification")
        return False, None, None

    if config is None:
        config = {}

    verbose = config.get('ipopt_verbose', config.get('verbose', False))
    constr_viol_tol = config.get('ipopt_constr_viol_tol', config.get('constr_viol_tol', 1e-4))

    solver = IPOPTSolver(config)

    # Split OR clauses - need to check each one
    clauses = list(_split_or_clauses(C_mat, rhs_mat, or_spec_size))
    num_clauses = len(clauses)

    if verbose:
        print(f"  [IPOPT] Problem: {dm_l.numel()} vars, {num_clauses} OR clause(s)")

    best_obj_overall = None
    # Wrap with eval mode for stable BN/Dropout
    with _eval_mode(model_ori):
        for clause_id, (C_obj, rhs_obj, C_constr, rhs_constr) in enumerate(clauses):
            if verbose and num_clauses > 1:
                print(f"  [IPOPT] Solving clause {clause_id + 1}/{num_clauses}...")

            # Create problem for this clause
            problem = NNProblemIPOPT(
                model_ori, dm_l, dm_u,
                C_obj, rhs_obj, C_constr, rhs_constr,
                input_constr_A=None if base_input_linear_constraints is None else base_input_linear_constraints[0],
                input_constr_b=None if base_input_linear_constraints is None else base_input_linear_constraints[1],
                device=None, dtype=None  # Infer from model
            )

            # Solve this clause
            result, x_solution = _solve_ipopt_single_clause(
                problem, solver, config,
                verbose=verbose,
                phase_name=f"IPOPT-Clause{clause_id}"
            )

            # Found valid counterexample for this clause - return immediately
            if x_solution is not None:
                print(f"  [IPOPT] Counterexample found in clause {clause_id}! "
                      f"obj={result.objective_value:.6f}")
                # Convert numpy solution to torch tensor
                x_tensor = torch.from_numpy(x_solution).to(
                    device=dm_l.device, dtype=dm_l.dtype
                ).view(dm_l.shape)
                return True, x_tensor, result.objective_value

            # Track best feasible result across clauses
            if result is not None and result.constraint_violation is not None:
                if result.constraint_violation <= constr_viol_tol:
                    if best_obj_overall is None or (
                        result.objective_value is not None and
                        result.objective_value < best_obj_overall
                    ):
                        best_obj_overall = result.objective_value
    # No counterexample found in any clause
    final_obj = best_obj_overall
    if verbose:
        obj_str = f"{final_obj:.6f}" if final_obj is not None else "None"
        print(f"  [IPOPT] Final Results: optimal=False, best_UB={obj_str}")

    return False, None, final_obj
