"""Glue between input BaB and the lifted McCormick LP (nlp_solver/lifted_lp.py).

- `RootLifted` builds the LP once at the root (solving mode, when
  solver.nlp.lifted_lp_bound or solver.nlp.lifted_lp_obbt is on) and, with lifted_lp_obbt,
  tightens the root box by OBBT over it before the first split; `setup_lifted_lp` runs its steps in one call.
- `lifted_node_bound` bounds the objective of a batch of domains over the lifted rows,
  the CROWN planes of the rows that are not lifted, and the base linear rows.
"""
from __future__ import annotations

import time
from typing import Optional

import torch

import arguments
from input_split.utils import bound_thresholds


def _qtb_covers_linear_rows(history, lp):
    """Convex quadratic objective over linear rows: QTB's tangent at the relaxed KKT
    point is exact there, McCormick envelopes of its products are not (cute_portfl4)."""
    qtb = history.qtb
    rows_linear = lp.atoms is None and (lp.rows is None or not bool((lp.rows[2] != 0).any()))
    return (rows_linear and lp.cleared_rhs is None and qtb is not None
            and float(getattr(qtb, "lam_lb", -1.0)) >= -1e-6)


def _relaxes_problem(history, lp, rhs_rows):
    """Whether the lifted LP relaxes the whole problem: the objective is lifted and every constraint row with a
    finite threshold is lifted (base linear rows enter as extra rows), and QTB is not exact over linear rows instead.
    Only then does a lifted-LP bound bound the problem."""
    if lp.obj is None or rhs_rows is None:
        return False
    finite = torch.isfinite(rhs_rows.reshape(-1))
    lifted = torch.zeros_like(finite)
    lifted[(lp.row_spec - 1).to(finite.device)] = True
    return not bool((finite & ~lifted).any()) and not _qtb_covers_linear_rows(history, lp)


class RootLifted:
    """The root's lifted-LP setup, in the steps the root runs (batch_branch_and_bound._ROOT_OPERATIONS): `build`, the
    certified McCormick OBBT `warm_up` (with RLT only), `add_rlt`, then the root `obbt`. Between steps the unified
    root (solver.nlp.unified_root) takes the lifted root `bound` of the relaxation so far; `version` counts the steps
    that strengthened it.

    OBBT rows: the lifted spec rows (relaxed by the pool, history.rowpool), the base linear rows
    (as given: the caller passes them relaxed, see relax_input_constraints), and, when the
    objective is lifted and rhs[:, 0] holds a finite incumbent, objective <= incumbent. The
    incumbent is feasible for the relaxed problem, so the relaxed problem's minimiser
    satisfies all of these rows and stays in the box.
    """

    @classmethod
    def build(cls, net, history, dm_l, dm_u, c, rhs, base_input_linear_constraints, nlp_config, allow_obbt=True):
        """history.lifted := LiftedLP (or None); the setup, or None when there is no lifted LP."""
        history.lifted = None
        history.lifted_mean_y = None
        use_bound = bool(nlp_config.get("lifted_lp_bound", False))
        use_obbt = bool(nlp_config.get("lifted_lp_obbt", False)) and allow_obbt
        use_bab = float(nlp_config.get("lifted_lp_bab_time", 0.0)) > 0
        if not (use_bound or use_obbt or use_bab) or dm_l.shape[0] != 1:
            return None
        from nlp_solver.lifted_lp import build_lifted_lp
        t0 = time.time()
        center = ((dm_l + dm_u) / 2).detach()
        qtb = getattr(history, "qtb", None)
        qtb_convex = qtb is not None and float(getattr(qtb, "lam_lb", -1.0)) >= -1e-6
        # A plain QTB proved the objective quadratic: it has no convex epigraph atoms, and
        # recognizing them anyway cost 5 s on cute_gouldqp2 (699 variables).
        from nlp_solver.quadratic_tangent import QuadraticTangent
        quadratic_objective = type(qtb) is QuadraticTangent
        if nlp_config.get("controller", False) and not quadratic_objective:
            # The early builder precedes QTB. Prove the objective's degree here
            # so a polynomial never enters the expensive convex-atom walk over
            # all constraint outputs merely because QTB has not run yet.
            from nlp_solver.quadratic_tangent import output_degrees
            degrees = output_degrees(net.net, center).to(c.device)
            used = c[0, 0] != 0
            quadratic_objective = bool((torch.isfinite(degrees[used]) &
                                        (degrees[used] <= 2)).all())
        rhs_rows = bound_thresholds(getattr(history, "rowpool", None), rhs[:1].reshape(1, -1).double())[:, 1:]
        lp = build_lifted_lp(net.net, net.model_ori, center, c, dm_l.flatten(1), dm_u.flatten(1),
                             max_vars=int(nlp_config.get("lifted_lp_max_vars", 2000)),
                             skip_if_linear_rows=qtb_convex,
                             clear_denominators=bool(nlp_config.get("clear_denominators", False)),
                             relaxed_rhs_rows=rhs_rows,
                             convex_objective=(bool(nlp_config.get("lifted_lp_convex_objective", False))
                                               and not quadratic_objective))
        if lp is None:
            print(f"[LiftedLP] not applicable ({time.time() - t0:.2f}s)")
            return None
        if _qtb_covers_linear_rows(history, lp):
            print("[LiftedLP] skipped: convex quadratic objective over linear rows (QTB)")
            return None
        history.lifted = lp
        if any(nlp_config.get(key, False) for key in
               ('lp_certificate_transfer', 'aux_reference_bounds', 'input_cutoff_cuts')):
            from nlp_solver.domain_certificate import CertificateRegistry
            lp.certificate_registry = CertificateRegistry(
                lp, input_cutoff_cuts=bool(nlp_config.get('input_cutoff_cuts', False)))
            history.certificate_registry = lp.certificate_registry
            lp.transfer_aux = bool(nlp_config.get('aux_reference_bounds', False))
            if lp.certificate_registry.input_cutoff_cuts and getattr(history, 'rowpool', None) is not None:
                history.rowpool.derived_cut_registry = lp.certificate_registry
        print(f"[LiftedLP] built in {time.time() - t0:.2f}s")
        return cls(history, lp, dm_l, dm_u, rhs, rhs_rows, base_input_linear_constraints, nlp_config, use_obbt)

    def __init__(self, history, lp, dm_l, dm_u, rhs, rhs_rows, base_input_linear_constraints, nlp_config, use_obbt):
        self.history, self.lp, self.rhs, self.rhs_rows = history, lp, rhs, rhs_rows
        self.nlp_config, self.use_obbt = nlp_config, use_obbt
        self.dm_l, self.dm_u = dm_l, dm_u
        self.l0, self.u0 = dm_l.reshape(1, -1).double(), dm_u.reshape(1, -1).double()
        self.eg = self.eh = None
        if base_input_linear_constraints is not None:
            A, b = base_input_linear_constraints
            self.eg = A[:1].reshape(1, -1, self.l0.shape[1]).to(self.l0)
            self.eh = b[:1].reshape(1, -1).to(self.l0)
        if lp.atoms is not None:
            lp.tighten_atoms(self.l0, self.u0, rhs_rows, self.eg, self.eh)
        self.use_rlt = bool(nlp_config.get("lifted_lp_rlt", False))
        inc = float(rhs.reshape(rhs.shape[0], -1)[0, 0])
        self.cutoff = inc if lp.obj is not None and inc < float("inf") else None
        self.obbt_budget = float(nlp_config.get("lifted_lp_obbt_time", 10.0))
        self.rounds = int(nlp_config.get("lifted_lp_obbt_rounds", 5))
        self.obbt_spent = 0.0              # seconds of the OBBT budget the warm-up and the RLT rows took
        self.lo, self.hi = self.l0, self.u0  # the box OBBT has reached
        self.version = 1

    def warm_up(self, time_left=float("inf")):
        """Cheap certified McCormick tightening first: building thousands of new
        products on 1e6 boxes otherwise consumes OBBT's budget before it can shrink
        the original products. The strengthened pass shares the same total budget."""
        if not (self.use_obbt and self.use_rlt and self.obbt_budget > 0):
            return
        t1 = time.time()
        self.lo, self.hi = self.lp.obbt(self.lo, self.hi, self.rhs_rows, self.eg, self.eh, cutoff=self.cutoff,
                                        rounds=self.rounds, time_budget=min(0.5 * self.obbt_budget, time_left),
                                        **self._checkpoint_args())
        self.obbt_spent += time.time() - t1
        self.version += 1
        print(f"[LiftedLP] McCormick OBBT warm-up in {time.time() - t1:.2f}s")

    def _checkpoint_args(self):
        checkpoint = getattr(self, 'checkpoint', None)
        return {} if checkpoint is None else {'checkpoint': checkpoint}

    def add_rlt(self):
        """The RLT rows (with solver.nlp.lifted_lp_rlt), then the rows every later lifted bound uses."""
        if self.use_rlt:
            t1 = time.time()
            self.lp.add_rlt(self.rhs[:1].reshape(1, -1).double()[:, 1:], self.eg, self.eh,
                            max_rows=int(self.nlp_config.get("lifted_lp_rlt_max_rows", 1200)),
                            max_columns=int(self.nlp_config.get("lifted_lp_rlt_max_columns", 7000)),
                            full_budget=bool(self.nlp_config.get("lifted_lp_rlt_full_budget", False)))
            self.obbt_spent += time.time() - t1
            self.version += self.lp.rlt is not None
            if self.lp.rlt is not None and hasattr(self.lp, 'certificate_registry'):
                from nlp_solver.domain_certificate import LiftedMap
                self.lp.certificate_registry.columns = LiftedMap(self.lp)
        self.history.lifted_obbt_rows = (self.rhs_rows, self.eg, self.eh)
        self.history.lifted_reobbt_time = 0.0

    def obbt(self, incumbent_x=None, time_left=float("inf")):
        """The root OBBT (with solver.nlp.lifted_lp_obbt): the tightened root box, or the root box as given."""
        if not self.use_obbt:
            return self.dm_l, self.dm_u
        # Without a lifted objective there is no valid root plane to cut with (the root
        # planes are rebuilt from a bound that may come from constrained concretization).
        remaining = max(0.0, self.obbt_budget - self.obbt_spent) if self.use_rlt else self.obbt_budget
        t1 = time.time()
        nl, nu = self.lp.obbt(self.lo, self.hi, self.rhs_rows, self.eg, self.eh, cutoff=self.cutoff,
                              rounds=self.rounds, time_budget=min(remaining, time_left), **self._checkpoint_args())
        shape = self.dm_l.shape
        nl = torch.maximum(nl, self.l0).to(self.dm_l).reshape(shape)
        nu = torch.minimum(nu, self.u0).to(self.dm_u).reshape(shape)
        _report_box("root OBBT", self.dm_l, self.dm_u, nl, nu, self.obbt_spent + time.time() - t1, self.cutoff)
        if incumbent_x is not None and incumbent_x.numel() == nl.numel():
            # Audit: the incumbent satisfies every relaxed row and its own cutoff, so it
            # must lie in the certified box; False would mean an invalid tightening.
            xi = incumbent_x.reshape(nl.shape).to(nl)
            slack = torch.minimum(xi - nl, nu - xi)
            print(f"[LiftedLP] audit: incumbent inside the OBBT box: {bool((slack >= 0).all())} "
                  f"(min slack {float(slack.min()):.3g})")
        self.history.lifted_box = (nl.clone(), nu.clone())
        self.history.lifted_cutoff = self.cutoff
        self.history.lifted_obbt_pending = getattr(self.lp, 'obbt_pending', False)
        return nl, nu

    def bound(self, incumbent, time_left):
        """The lifted root bound over the box OBBT has reached: a certified lower bound on the objective of every
        relaxed-feasible point there with objective <= the incumbent (min(node bound, incumbent)), from the same
        node solve as the lifted-LP B&B's root (LiftedLP.solve_bab_node). None when the lifted LP does not relax
        the whole problem."""
        from nlp_solver.lifted_lp import _NODE_TIME
        if not _relaxes_problem(self.history, self.lp, self.rhs_rows):
            return None
        inc = float("inf") if incumbent is None else float(torch.as_tensor(incumbent).reshape(-1)[0])
        f64 = lambda t: None if t is None else t.reshape(-1).cpu().double().numpy()
        eg = None if self.eg is None else self.eg.reshape(-1, self.lp.n).cpu().double().numpy()
        bd = self.lp.solve_bab_node(f64(self.lo), f64(self.hi), f64(self.rhs_rows), eg, f64(self.eh), inc,
                                    (max(1e-3, time_left) if self.nlp_config.get("controller", False)
                                     else min(_NODE_TIME, max(1e-3, time_left))),
                                    **({'probe': False} if self.nlp_config.get('controller', False) else {}))[0]
        return float(min(bd, inc))


def setup_lifted_lp(net, history, dm_l, dm_u, c, rhs, base_input_linear_constraints,
                    nlp_config, allow_obbt=True, incumbent_x=None):
    """history.lifted := LiftedLP (or None); returns the (possibly OBBT-tightened) box: the steps of RootLifted, in
    the order the root runs them."""
    lifted = RootLifted.build(net, history, dm_l, dm_u, c, rhs, base_input_linear_constraints, nlp_config,
                              allow_obbt)
    if lifted is None:
        return dm_l, dm_u
    lifted.warm_up()
    lifted.add_rlt()
    return lifted.obbt(incumbent_x)


def make_verifier(model_ori, dm_l, dm_u, c, rhs, or_spec_size,
                  base_input_linear_constraints, nlp_config):
    """x (n,) numpy -> (feasible, objective, x clipped to the box) with the primal's own
    check (nlp_solver.statistics.verify_counterexample at constr_viol_tol)."""
    import numpy as np
    from nlp_solver.statistics import verify_counterexample
    from nlp_solver.utils import _split_or_clauses
    clauses = list(_split_or_clauses(c[:1].squeeze(0), rhs[:1].squeeze(0),
                                     None if or_spec_size is None else or_spec_size[:1]))
    if not clauses:
        return None
    C_obj, rhs_obj, C_constr, rhs_constr = clauses[0]
    # verify_counterexample returns C_obj f(x) - rhs_obj; rhs[:, 0] holds the incumbent
    # here, so a zero rhs keeps the objective in raw units (it must be compared with the
    # raw incumbent).
    rhs_obj = torch.zeros_like(rhs_obj)
    A = None if base_input_linear_constraints is None else base_input_linear_constraints[0]
    b = None if base_input_linear_constraints is None else base_input_linear_constraints[1]
    tol = float(nlp_config.get("constr_viol_tol", 1e-4))
    lo = dm_l[:1].reshape(-1).detach().double().cpu().numpy()
    hi = dm_u[:1].reshape(-1).detach().double().cpu().numpy()
    shape = dm_l[:1].squeeze(0).shape

    def verify(x):
        x = np.minimum(np.maximum(x, lo), hi)      # LP solvers stray from bounds by ~1e-10
        xt = torch.as_tensor(x, dtype=dm_l.dtype, device=dm_l.device).view(shape)
        ok, obj, _ = verify_counterexample(model_ori, xt, C_obj, rhs_obj, C_constr,
                                           rhs_constr, input_constr_A=A, input_constr_b=b,
                                           tol=tol)
        return bool(ok), float(obj), x
    return verify


def make_local_solver(model_ori, dm_l, dm_u, c, rhs, or_spec_size,
                      base_input_linear_constraints, nlp_config, verify_fn):
    """x0 (n,) numpy -> verified (x, objective) from the root local solver (Ipopt, with
    the production slack settings) started at x0, or None. Like root_lp_local_start, a
    start point taken from an LP relaxation; only verified points are returned."""
    import numpy as np
    from nlp_solver.ipopt_solver import IPOPT_AVAILABLE, NNProblemIPOPT, IPOPTSolver
    from nlp_solver.utils import _split_or_clauses
    if not IPOPT_AVAILABLE:
        return None
    clauses = list(_split_or_clauses(c[:1].squeeze(0), rhs[:1].squeeze(0),
                                     None if or_spec_size is None else or_spec_size[:1]))
    if not clauses:
        return None
    C_obj, rhs_obj, C_constr, rhs_constr = clauses[0]
    rhs_obj = torch.zeros_like(rhs_obj)     # objective in raw units (see make_verifier)
    A = None if base_input_linear_constraints is None else base_input_linear_constraints[0]
    b = None if base_input_linear_constraints is None else base_input_linear_constraints[1]
    problem = NNProblemIPOPT(model_ori, dm_l[:1].squeeze(0), dm_u[:1].squeeze(0), C_obj,
                             rhs_obj, C_constr, rhs_constr, input_constr_A=A,
                             input_constr_b=b, device=None, dtype=None)
    solver = IPOPTSolver(nlp_config)
    timeout = float(nlp_config.get("lifted_lp_leaf_ipopt_time", 5.0))

    def solve(x0):
        x0 = np.clip(np.asarray(x0, dtype=np.float64), problem.lb, problem.ub)
        try:
            res = solver.solve_single(problem, x_init=x0, timeout=timeout)
        except Exception:  # noqa: BLE001
            return None
        if res.x_solution is None:
            return None
        ok, obj, x_used = verify_fn(np.asarray(res.x_solution, dtype=np.float64))
        return (x_used, obj) if ok else None
    return solve


# A B&B whose leaves go to the GPU BaB stops at this many open leaves per input (at least
# _HANDOFF_MIN): the lifted search sets up good starting boxes, the GPU BaB is the better engine
# beyond, and how fine a useful decomposition is grows with the dimension. schitt_s368 (8 inputs)
# verified from 244-256 leaves but not from 1024 or 4148; kall_diffcircles_10 (24) from 1024 but
# not from 256 or 2935; kall_diffcircles_8 (20) took 41 s from ~600 and 100 s from 3190.
_HANDOFF_LEAVES_PER_INPUT = 32
_HANDOFF_MIN = 256


def _continuation_pays(progress, competing):
    """One continued measurement, then positive gap gain per unit of work.

    The lifted frontier may improve before it closes: use its certified D, not
    run_lifted_bab's closed-only return value. Verified primal gain counts too.
    """
    if progress is None:
        return True
    gain, seconds = progress
    other_gain, other_seconds = competing
    return gain > 0 and (other_seconds <= 0 or gain * other_seconds >= other_gain * seconds)


def run_lifted_bab(history, dm_l, dm_u, incumbent, nlp_config, time_left,
                   want_leaves=False, verify_fn=None, incumbent_x=None, local_fn=None,
                   floor=-float("inf"), bab_lb=None, time_share=0.3, checkpoint=None,
                   competing_progress=(0.0, 0.0)):
    """B&B over the lifted LP alone (solver.nlp.lifted_lp_bab_time > 0), at the root and
    again when an improved incumbent, the first measurement of open leaves, or a paying
    continuation admits more work.
    After one continued measurement, require positive measured gap gain at least
    as fast as the main BaB (at most twice the option's budget unless paid).

    Only when every constraint row with a finite threshold is lifted and the objective is
    quadratic (then the lifted LP relaxes the whole problem; base linear rows enter as
    extra rows). Returns the certified dual D when it proves incumbent - D within the
    certification tolerance, else None. dm_l, dm_u must hold every relaxed-feasible point
    with objective <= the incumbent (the root box, or a later OBBT box); D then bounds the
    relaxed problem. A later call continues from the open leaves the previous call left
    (history.lifted_bab_open) and uses the box for the reference widths only; it runs only while
    the lifted search is the frontier: the BaB's bound `bab_lb` has not passed its last D.
    A call takes at most `time_share` of `time_left` (the unified root admits by its deadline alone: 1).
    """
    from nlp_solver.lifted_lp import LiftedLP
    lp = getattr(history, "lifted", None)
    budget = float(nlp_config.get("lifted_lp_bab_time", 0.0))
    if not isinstance(lp, LiftedLP) or budget <= 0 or lp.obj is None:
        return None
    rhs_rows, eg, eh = getattr(history, "lifted_obbt_rows", (None, None, None))
    if not _relaxes_problem(history, lp, rhs_rows):
        return None
    inc = float(torch.as_tensor(incumbent).reshape(-1)[0])
    if not inc < float("inf"):
        return None
    last = getattr(history, "lifted_bab_cutoff", None)
    paid = getattr(history, "lifted_bab_paid", False)
    resume = getattr(history, "lifted_bab_open", None)
    progress = getattr(history, "lifted_bab_progress", None)
    # Open leaves earn one continued measurement at the same incumbent (wastewater14m1: the root
    # incumbent was already optimal, so no new cutoff ever came); then its measured gain decides.
    measure = resume is not None and len(resume[2]) > 0 and progress is None
    if last is not None and not paid and not measure and not inc < last - 1e-4 * abs(last) - 1e-9:
        return None         # nothing new since the last attempt
    last_D = getattr(history, "lifted_bab_D", None)
    if bab_lb is not None and last_D is not None and bab_lb > last_D + max(
            float(nlp_config.get("bab_primal_dual_abs_tol", 1e-5)),
            float(nlp_config.get("bab_primal_dual_rel_tol", 1e-4)) * abs(inc)):
        # The BaB passed the lifted search (kall_diffcircles_8/9/10: within 1 s of the handoff, while
        # the continuations cost 30 s each); where it stays level (wastewater11m1) they are the progress.
        return None
    if resume is not None and not _continuation_pays(progress, competing_progress):
        return None
    spent = getattr(history, "lifted_bab_spent", 0.0)
    # 2x the option in all, unless the last continued call halved its gap (see below)
    budget = min(budget, time_share * time_left, float("inf") if paid else 2 * budget - spent)
    if budget <= 0.2:
        return None
    history.lifted_bab_cutoff = inc
    t0 = time.time()
    best, local_time = [], [0.0]    # the local solves' best verified (point, objective); seconds

    def local_start(x):
        # The B&B offers its best node's point at nodes 128, 256, ...: an improving solve
        # lowers its cutoff at once. Called from the B&B's no_grad.
        t = time.time()
        with torch.enable_grad():
            r = local_fn(x)
        local_time[0] += time.time() - t
        if r is not None and r[1] < (best[0][1] if best else inc) - 1e-9 * (1.0 + abs(inc)):
            best[:] = [r]
            return r[1]
        return None

    res = lp.branch_and_bound(
        dm_l.reshape(1, -1).double(), dm_u.reshape(1, -1).double(), rhs_rows, eg, eh,
        cutoff=inc, tol_abs=float(nlp_config.get("bab_primal_dual_abs_tol", 1e-5)),
        tol_rel=float(nlp_config.get("bab_primal_dual_rel_tol", 1e-4)),
        time_budget=budget, max_nodes=int(nlp_config.get("lifted_lp_bab_max_nodes", 200000)),
        return_leaves=True, workers=int(nlp_config.get("lifted_lp_bab_workers", 1)),
        node_cutoff=bool(nlp_config.get("lifted_lp_node_cutoff", True)),
        node_obbt_gain=bool(nlp_config.get("lifted_lp_node_obbt_gain", False)),
        max_leaves=(max(_HANDOFF_MIN, _HANDOFF_LEAVES_PER_INPUT * lp.n)
                    if want_leaves and resume is None else None),
        floor=floor, resume=resume, primal_fn=None if local_fn is None else local_start,
        **({} if checkpoint is None else {'checkpoint': checkpoint}))
    D, nodes, closed, leaves = res
    history.lifted_bab_open = leaves
    history.lifted_bab_D = D
    history.lifted_bab_spent = spent + (time.time() - t0) - local_time[0]    # the budget is for the LPs
    if nodes == 0 and not closed and resume is None:
        # The root bound did not improve `floor` (the BaB's own root bound): no later call either.
        history.lifted_bab_spent = 2 * float(nlp_config.get("lifted_lp_bab_time", 0.0))
    print(f"[LiftedLP] lifted-LP B&B: {nodes} nodes in {time.time() - t0:.2f}s, "
          f"D={D:.12g}, incumbent {inc:.12g}, {'CLOSED' if closed else 'open'}"
          + ("" if closed else f", {leaves[2].shape[0]} open leaves"))
    history.lifted_bab_new_incumbent = None
    if best:
        history.lifted_bab_new_incumbent = best[0]
        inc = best[0][1]
    # Steady progress, not a plateau: a continued call that halved the gap between its starting
    # bound (the lowest resumed leaf) and the incumbent earns the next call. wastewater11m1
    # (incumbent 2127.10) needs ~2200 nodes (85 s) over three calls: gap 6.4 -> 2.7 -> closed.
    history.lifted_bab_paid = (resume is not None and len(resume[2]) > 0 and not closed
                               and inc - D <= 0.5 * (inc - float(resume[2].min())))
    # The leaf phase ends with the global limit; without RLT it had no deadline at all.
    deadline = t0 + time_left
    if not closed and verify_fn is not None and leaves[2].shape[0] > 0:
        # The leaves' LP points are (nearly) feasible for the relaxed problem once the
        # McCormick envelopes are tight; offer them, and blends towards the incumbent,
        # to the primal's own check. Only verified points are taken.
        ctol = float(nlp_config.get("constr_viol_tol", 1e-4))
        # Epigraph leaves approach the tolerance-relaxed optimum. Use the root
        # local solver's configured interior fraction, then the ordinary primal
        # check, so a half-tolerance incumbent does not mask a closed dual gap.
        interior = (float(nlp_config.get("ipopt_constraint_slack_frac", 0.99))
                    if lp.epigraph else 0.5)
        interior = min(1.0, max(0.0, interior))
        rh_t = (rhs_rows.reshape(-1) - (1-interior) * ctol).cpu().double().numpy()
        eg_t = None if eg is None else eg.reshape(-1, lp.n).cpu().double().numpy()
        eh_t = None if eh is None else (eh.reshape(-1) - (1-interior) * ctol).cpu().double().numpy()

        def tight_point(lo, hi):
            # The leaf LP with every row inside the tolerance: where the envelopes are
            # tight its point is feasible with room to spare. A heuristic point only.
            from scipy.optimize import linprog
            import numpy as np
            if checkpoint is not None:
                checkpoint()
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            c_, _, G_, h_, lz_, uz_ = lp.sparse_lp(lo, hi, rh_t, eg_t, eh_t)
            try:
                r = linprog(c_, A_ub=G_, b_ub=h_, bounds=np.stack([lz_, uz_], 1),
                            method="highs", options={'time_limit': max(1e-3, remaining)})
            except Exception:  # noqa: BLE001
                return None
            return r.x[:lp.n] if r.status == 0 else None

        found = _leaf_incumbent(leaves, incumbent_x, verify_fn, inc, tight_point,
                                deadline=deadline)
        if local_fn is not None:
            # Local solves from the lowest-bound leaves' LP points: at the relaxation's
            # plateau these sit next to the relaxed problem's minimiser.
            import numpy as np
            L_x = leaves[3]
            with_pt = np.nonzero(np.isfinite(L_x).all(1))[0]
            # the lowest leaves are often tiny boxes whose LP HiGHS calls infeasible at
            # its tolerance (no point): take the lowest three that have one
            for i in with_pt[np.argsort(leaves[2][with_pt])][:3]:
                if checkpoint is not None:
                    checkpoint()
                if time.time() >= deadline:
                    break
                r = local_fn(L_x[i])
                if r is not None and r[1] < inc - 1e-9 * (1.0 + abs(inc)) and (found is None or r[1] < found[1]):
                    found = r
        if found is not None:
            x_new, obj_new = found
            history.lifted_bab_new_incumbent = (x_new, obj_new)
            tol = max(float(nlp_config.get("bab_primal_dual_abs_tol", 1e-5)),
                      float(nlp_config.get("bab_primal_dual_rel_tol", 1e-4)) * abs(obj_new))
            # D still bounds every relaxed-feasible point (the leaves cover all points
            # with objective below the old incumbent, which is above the new one).
            closed = obj_new - D <= tol
            print(f"[LiftedLP] leaf LP point verified as incumbent: {obj_new:.12g} "
                  f"(was {inc:.12g}); {'CLOSED' if closed else 'open'} with D={D:.12g}")
    if resume is not None:
        import math
        start_D = float(resume[2].min()) if len(resume[2]) else float('-inf')
        # Only progress beyond the competing frontier can help close its gap.
        baseline = max(start_D, bab_lb) if bab_lb is not None and math.isfinite(bab_lb) else start_D
        dual_gain = max(0.0, D - baseline) if math.isfinite(baseline) else 0.0
        new_inc = getattr(history, 'lifted_bab_new_incumbent', None)
        final_inc = new_inc[1] if new_inc is not None else inc
        primal_gain = max(0.0, float(torch.as_tensor(incumbent).reshape(-1)[0]) - final_inc)
        history.lifted_bab_progress = (dual_gain + primal_gain, time.time() - t0)
        print(f"[LiftedLP] continuation gain={dual_gain + primal_gain:.12g} "
              f"s={history.lifted_bab_progress[1]:.6g}")
    if closed:
        return D
    if want_leaves:
        history.lifted_bab_leaves = leaves
    return None


def _leaf_incumbent(leaves, incumbent_x, verify_fn, inc, tight_point=None, max_leaves=32,
                    deadline=None):
    """Best verified point among the lowest-bound leaves' LP points, their LP points with
    the rows at half the tolerance (`tight_point`), and blends lam * x + (1 - lam) *
    incumbent_x; (x, objective) if it beats `inc`, else None."""
    import numpy as np
    L_lo, L_hi, L_bd, L_x = leaves[:4]
    with_pt = np.nonzero(np.isfinite(L_x).all(1))[0]
    order = list(with_pt[np.argsort(L_bd[with_pt])][:max_leaves])
    order += [i for i in np.argsort(L_bd)[:max_leaves // 4] if i not in set(order)]
    lams = (1.0,) if incumbent_x is None else (1.0, 0.999, 0.99, 0.9)
    best = None
    x_inc = None if incumbent_x is None else \
        incumbent_x.reshape(-1).detach().double().cpu().numpy()
    for i in order:
        if deadline is not None and time.time() >= deadline:
            break
        cands = [L_x[i]] if np.isfinite(L_x[i]).all() else []
        if tight_point is not None:
            xt = tight_point(L_lo[i], L_hi[i])
            if xt is not None:
                cands.append(xt)
        for xc in cands:
            for lam in lams:
                x = xc if lam == 1.0 else lam * xc + (1 - lam) * x_inc
                ok, obj, x_used = verify_fn(x)
                if ok and obj < inc - 1e-9 * (1.0 + abs(inc)) and (best is None or obj < best[1]):
                    best = (x_used, obj)
                if ok:
                    break
    return best


def _report_box(what, l0, u0, l1, u1, secs, cutoff):
    w0 = (u0 - l0).flatten()
    w1 = (u1 - l1).flatten()
    rel = torch.where(w0 > 0, w1 / w0.clamp(min=1e-300), torch.ones_like(w0))
    print(f"[LiftedLP] {what} in {secs:.2f}s: mean width ratio {float(rel.mean()):.4f}, "
          f"min {float(rel.min()):.3g}"
          + (f", cutoff {cutoff:.10g}" if cutoff is not None else ", no cutoff"))
    return float(rel.mean())


def reobbt_on_improvement(history, incumbent, nlp_config, time_left, checkpoint=None):
    """Re-run OBBT after a cutoff improvement or an unfinished paying pass.

    A root time/round limit can leave useful tightening at the same cutoff.
    Requiring another primal improvement would deny that work precisely when a
    good incumbent arrived early. The existing total budget and gain stop still
    bound continuations. The result replaces history.lifted_box,
    which clamps every picked domain (`clamp_to_lifted_box`). Returns True if it ran.

    Valid for the same reason as the root OBBT: the removed points have objective above
    the (new) incumbent or violate a relaxed row, and the incumbent only decreases.
    """
    lp = getattr(history, "lifted", None)
    box = getattr(history, "lifted_box", None)
    if lp is None or box is None or lp.obj is None or incumbent is None:
        return False
    inc = float(torch.as_tensor(incumbent).reshape(-1)[0])
    if not inc < float("inf"):
        return False
    last = getattr(history, "lifted_cutoff", None)
    if (last is not None and not inc < last - 1e-4 * abs(last) - 1e-9
            and not getattr(history, 'lifted_obbt_pending', False)):
        return False
    total = float(nlp_config.get("lifted_lp_obbt_time", 10.0))
    budget = min(0.5 * total, 2 * total - history.lifted_reobbt_time, 0.1 * time_left)
    if budget <= 0.2:
        return False
    rhs_rows, eg, eh = history.lifted_obbt_rows
    l0, u0 = box
    t1 = time.time()
    nl, nu = lp.obbt(l0.reshape(1, -1).double(), u0.reshape(1, -1).double(), rhs_rows, eg,
                     eh, cutoff=inc, rounds=int(nlp_config.get("lifted_lp_obbt_rounds", 5)),
                     time_budget=budget, verbose=False,
                     **({} if checkpoint is None else {'checkpoint': checkpoint}))
    nl = torch.maximum(nl.to(l0).reshape(l0.shape), l0)
    nu = torch.minimum(nu.to(u0).reshape(u0.shape), u0)
    history.lifted_reobbt_time += time.time() - t1
    ratio = _report_box("re-OBBT", l0, u0, nl, nu, time.time() - t1, inc)
    if ratio >= 0.98:
        # 326 of 359 logged re-OBBTs (355 s in all) shrank the mean width by under 2%, and after one that
        # did, 129 of the 136 later ones did too: one such call ends them (wastewater11m1: 5 s for 0.9969).
        history.lifted_reobbt_time = 2 * total
    history.lifted_box = (nl, nu)
    history.lifted_cutoff = inc
    history.lifted_obbt_pending = getattr(lp, 'obbt_pending', False) and ratio < 0.98
    return True


def clamp_to_lifted_box(history, x_L, x_U):
    """Intersect picked domains with the global OBBT box. Every relaxed-feasible point
    with objective <= the OBBT cutoff (>= the current incumbent) lies in that box, so
    the part cut away can hold no better point. A domain whose intersection is empty
    becomes a degenerate box on the global box's boundary: any bound computed on it is
    a valid bound for the empty set it stands for."""
    box = getattr(history, "lifted_box", None)
    if box is None:
        return x_L, x_U
    gl = box[0].reshape(1, *x_L.shape[1:]).to(x_L)
    gu = box[1].reshape(1, *x_U.shape[1:]).to(x_U)
    x_L = torch.minimum(torch.maximum(x_L, gl), gu)
    x_U = torch.maximum(torch.minimum(x_U, gu), x_L)
    return x_L, x_U


def lifted_node_bound(history, dm_l, dm_u, lA, lbias, rhs_rows, input_constr_A,
                      input_constr_b, prev, nlp_config, deadline=None):
    """(bound (F,), score) or None when the lifted LP is off. bound: lower bound on the
    objective of each domain from the lifted LP (-inf where not computed). score: with
    solver.nlp.lifted_lp_branching, the McCormick-violation branching score (F, n) at
    the domains' LP points (NaN rows where none), else None.

    dm_l, dm_u (F, n); lA (F, S, n), lbias (F, S): the domains' CROWN planes, valid on
    their boxes; rhs_rows (F, S-1) relaxed constraint thresholds; input_constr_A/b the
    relaxed base linear rows (F, M, n)/(F, M) or None; prev (F,) the current bound.
    """
    from nlp_solver.lifted_lp import LiftedLP
    lp = getattr(history, "lifted", None)
    if not isinstance(lp, LiftedLP) or not bool(nlp_config.get("lifted_lp_bound", False)):
        return None
    S = lA.shape[1]
    lifted = torch.zeros(S, dtype=torch.bool, device=lA.device)
    lifted[lp.row_spec.to(lA.device)] = True
    other = (~lifted).nonzero().reshape(-1)
    other = other[other >= 1]
    eg = lA[:, other, :]
    eh = rhs_rows[:, other - 1] - lbias[:, other]
    if input_constr_A is not None:
        eg = torch.cat([eg, input_constr_A.to(eg)], 1)
        eh = torch.cat([eh, input_constr_b.to(eh)], 1)
    obj_plane = ((lA[:, 0, :], lbias[:, 0])
                 if lp.obj is None or lp.epigraph else None)
    y0 = getattr(history, "lifted_mean_y", None)
    branching = bool(nlp_config.get("lifted_lp_branching", False))
    res = lp.bound(
        dm_l, dm_u, rhs_rows, eg, eh, obj_plane,
        pdhg_iters=int(nlp_config.get("lifted_lp_pdhg_iters",
                                      nlp_config.get("dual_lp_pdhg_iters", 500))),
        exact_max=int(nlp_config.get("lifted_lp_exact_max", 16)),
        y0=None if y0 is None else y0.to(dm_l),
        max_elems=int(nlp_config.get("lifted_lp_max_elems", 25_000_000)),
        prev=prev, return_points=branching,
        exact_time=float(nlp_config.get("lifted_lp_exact_time", 0.5)), deadline=deadline)
    bnd, y_mean = res[0], res[1]
    if y_mean is not None:
        history.lifted_mean_y = y_mean.detach().cpu()
    rel_width = None
    root_width = getattr(history, "root_width", None)
    if branching and root_width is not None and root_width.numel() == dm_l.shape[1]:
        rel_width = (dm_u - dm_l) / root_width.to(dm_l).reshape(1, -1).clamp(min=1e-300)
    score = lp.violation_score(res[2], rel_width) if branching else None
    points = res[2][:, :lp.n] if branching else None
    return bnd, score, points


def apply_lifted_split_points(history, parent_l, parent_u, split_idx, new_l, new_u,
                              frac=0.1):
    """Move the split of every domain branched by the lifted rule from the midpoint to its
    LP point (clamped to the middle 1 - 2 frac of the interval), as spatial B&B does.
    Children layout of input_split_parallel (split depth 1): rows [0, B) take the upper
    part [s, u], rows [B, 2B) the lower part [l, s]. Any split point keeps the two
    children a partition of the parent, so this changes no bound's validity."""
    pts = getattr(history, "lifted_split_point", None)
    score = getattr(history, "lifted_branch_score", None)
    B = parent_l.shape[0]
    if (not torch.is_tensor(pts) or not torch.is_tensor(score) or pts.shape[0] != B
            or score.shape != pts.shape or new_l.shape[0] != 2 * B
            or split_idx is None or split_idx.dim() != 2 or split_idx.shape[1] != 1):
        return new_l, new_u
    pl, pu = parent_l.flatten(1), parent_u.flatten(1)
    nl, nu = new_l.flatten(1).clone(), new_u.flatten(1).clone()
    ar = torch.arange(B, device=pl.device)
    j = split_idx[:, 0].long().to(pl.device)
    best, arg = torch.nan_to_num(score.to(pl), nan=-1.0).max(dim=1)
    v = pts.to(pl)[ar, j]
    lo, hi = pl[ar, j], pu[ar, j]
    mid = (lo + hi) / 2
    layout_ok = (nl[:B][ar, j] == mid) & (nu[B:][ar, j] == mid)
    use = (best > 0) & (arg == j) & torch.isfinite(v) & (hi > lo) & layout_ok
    if not bool(use.any()):
        return new_l, new_u
    w = hi - lo
    v = torch.minimum(torch.maximum(v, lo + frac * w), hi - frac * w)
    iu = ar[use]
    nl[iu, j[use]] = v[use]
    nu[B + iu, j[use]] = v[use]
    return nl.reshape(new_l.shape), nu.reshape(new_u.shape)
