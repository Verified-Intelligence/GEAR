"""Root primal for bilinear structure: block alternation of exact LPs, joint trust-region LP steps, and the sparse
exact-Hessian IPOPT, on the exact forms (arch2 spec 2.2 `primal`, stage S5).

On a QCQP, holding every product variable fixed except one block (a colour class of the product graph, from the
compiler's branching space) leaves every row and the objective affine in the block, so each block step is an exact LP
(HiGHS dual simplex, warm basis per block):
  - phase 1, from an infeasible point: the elastic LP minimising the scaled L1 row violation. The first conditional LP
    of a water or wastewater network is infeasible at a fixed block, and without this phase nothing moves;
  - phase 2, from a feasible point: the objective LP on the same block, which decreases monotonically.
A joint trust-region LP on all rows, linearised at a verified point, moves both blocks at once; block LPs then restore
exact feasibility. Every feasible point also seeds GEAR's IPOPT with sparse exact derivatives (sparse_derivatives),
where each equality is a band of half the verify tolerance. The optima of the large network models are degenerate LP
vertices (on wastewater13m2, 992 of 1039 variables are at a bound). The LP steps land on such vertices exactly, while
penalty and interior methods have to approach them through a band 1e-11 wide.

Seeds: the McCormick LP point of the box and the lifted-ratio fits of its blocks, the box centre, the caller's points
(ALM, IPOPT, LP planes), then McCormick vertices of perturbed costs until the deadline; IPOPT points that improve on the
best one go back to the alternation. Time comes from the deadline alone: a job starts while time remains, and no job
(an alternation with its trust-region steps, an IPOPT call) gets more than half of what is left. Only points that pass
GEAR's verifier on the original model become incumbents.

Operation interface (S1's root list): `eligible(state)` and `run(state, deadline)` on a state with the RootState fields
net, dm_l, dm_u, c, rhs, or_spec_size, base_input_linear_constraints, nlp_config, primal_solution and best_primal_x
(plus optional `seeds`, a list of points); `start(state, deadline)` runs the same search in a child process beside the
GPU work until the deadline or until the caller collects it, and this process verifies what it returns.
"""
import math
import threading
import time
from collections import deque

import numpy as np
import scipy.sparse as sp
import torch
from .continuation import Atom, retained

_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def _lp(c, A, rl, ru, cl, cu, basis, deadline, tol):
    """HiGHS dual simplex on min c x s.t. rl <= A x <= ru, cl <= x <= cu, warm from `basis` (an earlier solve's of the
    same pattern) when given. Feasibility tolerance `tol`. Returns (x, basis) or None."""
    import highspy
    left = deadline - time.time()
    if left <= 0:
        return None
    A = sp.csc_matrix(A)
    inf = highspy.kHighsInf
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = np.asarray(c, dtype=np.float64)
    lp.col_lower_ = np.where(np.isfinite(cl), cl, -inf).astype(np.float64)
    lp.col_upper_ = np.where(np.isfinite(cu), cu, inf).astype(np.float64)
    lp.row_lower_ = np.where(np.isfinite(rl), rl, -inf).astype(np.float64)
    lp.row_upper_ = np.where(np.isfinite(ru), ru, inf).astype(np.float64)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_ = A.indptr.astype(np.int32), A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    # no 'threads' option: HiGHS's scheduler is global to the process, and a solve that asks for another thread count
    # than the first one in the process fails (kError); the dual simplex is serial anyway
    h.setOptionValue('solver', 'simplex')
    h.setOptionValue('simplex_strategy', 1)
    h.setOptionValue('primal_feasibility_tolerance', tol)
    h.setOptionValue('dual_feasibility_tolerance', tol)
    h.setOptionValue('time_limit', float(left))
    h.passModel(lp)
    if basis is not None:
        h.setOptionValue('presolve', 'off')
        try:
            h.setBasis(basis)
        except Exception:  # noqa: BLE001 - a stale basis only costs a cold start
            pass
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return None
    return np.array(h.getSolution().col_value), h.getBasis()


class Alternation:
    """The exact QCQP of one problem (a SparseProblem without atoms) and its LP steps.

    rows: the tape rows q_r with bounds cl <= q_r <= cu of the exact problem (the raw rows at zero slack); blocks: the
    compiler's branching space."""

    def __init__(self, sparse, lb, ub, space, tol):
        self.sparse, self.poly = sparse, sparse.nlp.tape.out
        self.n, self.m = sparse.nlp.n, sparse.m
        self.lb, self.ub = lb, ub
        self.cl, self.cu = sparse.bounds(0.0)
        self.space, self.tol = space, tol
        self.lp_tol = 1e-3 * tol        # LP feasibility well inside the verify tolerance
        self.basis = {}
        self.lps = 0

    # -------------------------------------------------------------------------------------------- block LPs
    def affine(self, x, free):
        """rows and objective affine in the free variables at x: (A (m x |F|), const (m,), cost (|F|,), f0, F)"""
        r, c, v, const = self.poly.affine_in(np.asarray(x, dtype=np.float64), free)
        idx = np.flatnonzero(free)
        pos = np.full(self.n, -1)
        pos[idx] = np.arange(len(idx))
        obj = r == 0
        cost = np.bincount(pos[c[obj]], v[obj], minlength=len(idx))
        A = sp.csr_matrix((v[~obj], (r[~obj] - 1, pos[c[~obj]])), shape=(self.m, len(idx)))
        return A, const[1:], cost, const[0], idx

    def _solve(self, key, *args, deadline):
        res = _lp(*args, self.basis.get(key), deadline, self.lp_tol)
        self.lps += 1
        if res is None:
            return None
        self.basis[key] = res[1]
        return res[0]

    def block_step(self, x, k, feasible, deadline):
        """One block k from x: the objective LP when x is feasible; else the elastic LP, followed by the objective LP
        when the block reaches zero violation. Returns (x_new, feasible)."""
        free = self.space.block(k)
        if not free.any():
            return x, feasible
        A, const, cost, _, idx = self.affine(x, free)
        rl, ru, cl, cu = self.cl - const, self.cu - const, self.lb[idx], self.ub[idx]
        out = np.array(x, dtype=np.float64)
        if feasible:
            z = self._solve(('obj', k), cost, A, rl, ru, cl, cu, deadline=deadline)
            if z is not None:
                out[idx] = np.clip(z, cl, cu)
                return out, True
        norm = np.asarray(abs(A).max(1).todense()).ravel()
        live = norm > 0                  # a row without a free variable is a constant of this block
        mag = 1.0 + np.abs(np.where(np.isfinite(rl), rl, 0.0)) + np.abs(np.where(np.isfinite(ru), ru, 0.0))
        dead = ~live & ((rl > self.lp_tol * mag) | (ru < -self.lp_tol * mag))
        Al, m = A[live], int(live.sum())
        w = 1.0 / norm[live]
        I = sp.identity(m, format='csr')
        z = self._solve(('elastic', k), np.r_[np.zeros(len(idx)), w, w], sp.hstack([Al, I, -I]), rl[live], ru[live],
                        np.r_[cl, np.zeros(2 * m)], np.r_[cu, np.full(2 * m, np.inf)], deadline=deadline)
        if z is None:
            return x, False
        out[idx] = np.clip(z[:len(idx)], cl, cu)
        s = z[len(idx):len(idx) + m] + z[len(idx) + m:]
        if dead.any() or np.any(s > self.lp_tol * mag[live]):
            return out, False
        z = self._solve(('obj', k), cost, A, rl, ru, cl, cu, deadline=deadline)
        if z is not None:
            out[idx] = np.clip(z, cl, cu)
        return out, True

    @retained
    def alternate(self, x0, first, deadline, offer, sliced=False):
        """Blocks first, first + 1, ... from x0 while a whole cycle makes progress (a lower objective once feasible,
        else a lower largest row violation); `offer` gets every feasible point. Returns the last feasible point or
        None."""
        x = np.clip(np.where(np.isfinite(x0), x0, 0.0), self.lb, self.ub)
        K = max(self.space.colours, 1)
        feasible, last, before = False, None, None
        while time.time() < deadline:
            for k in [(first + j) % K for j in range(K)]:
                x, feasible = self.block_step(x, k, feasible, deadline)
                if sliced:
                    yield Atom("lp_alternation", x.copy() if feasible else None)
                if feasible:
                    last = x
                    offer(x)
            now = (0, self.sparse.objective(x)) if feasible else (1, float(np.max(self.sparse.raw_rows(x), initial=0)))
            if before is not None and not (now[0] < before[0] or now[1] < before[1] - self.lp_tol * (1 + abs(before[1]))):
                break
            before = now
        return last

    # -------------------------------------------------------------------------------------------- joint step
    @retained
    def trust_region(self, x, f, deadline, offer, sliced=False):
        """Joint moves from a feasible x of objective f: the LP on all rows linearised at x inside a trust region
        (both blocks move), then exact restoration by each block's LPs from the LP point. The region doubles on a
        feasible decrease and halves otherwise, until the LP predicts no decrease beyond its own precision."""
        sp_ = self.sparse
        width = np.where(np.isfinite(self.ub - self.lb), self.ub - self.lb, 1.0 + np.abs(x))
        radius = 1.0
        while time.time() < deadline:
            g = sp_.gradient(x)
            J = sp.csr_matrix((sp_.jacobian(x), sp_.jacobianstructure()), shape=(self.m, self.n))
            q = sp_.constraints(x)
            lo, hi = np.maximum(self.lb, x - radius * width), np.minimum(self.ub, x + radius * width)
            z = self._solve(('joint',), g, J, self.cl - q + J @ x, self.cu - q + J @ x, lo, hi, deadline=deadline)
            if sliced:
                yield Atom('lp_alternation')
            if z is None or g @ (z - x) >= -self.lp_tol * (1.0 + abs(f)):
                break
            best = None
            for k in range(max(self.space.colours, 1)):
                xn, ok = self.block_step(np.clip(z, self.lb, self.ub), k, False, deadline)
                fn = offer(xn) if ok else None
                if sliced:
                    yield Atom('lp_alternation', xn.copy() if ok else None)
                if fn is not None and fn < f - self.lp_tol * (1.0 + abs(f)) and (best is None or fn < best[1]):
                    best = (xn, fn)
            if best is None:
                radius *= 0.5
            else:
                (x, f), radius = best, min(1.0, 2.0 * radius)
        return x, f

    # -------------------------------------------------------------------------------------------- relaxation
    def mccormick(self, cost, deadline):
        """The McCormick LP over the box (one lifted variable per distinct product or square) with x-cost `cost` and
        the objective's own product terms. Returns (x, products {(i, j): w}) or None."""
        const = self.poly.const
        pairs, rows, cols, vals = {}, [], [], []
        lin_r, lin_c, lin_v = [], [], []
        for b in self.poly.buckets:
            for t in range(len(b.own)):
                o, coef = int(b.own[t]), float(b.coef[t])
                if b.d == 1 and b.ex[t, 0] == 1:
                    lin_r.append(o), lin_c.append(int(b.node[t, 0])), lin_v.append(coef)
                    continue
                i, j = (int(b.node[t, 0]),) * 2 if b.d == 1 else sorted((int(b.node[t, 0]), int(b.node[t, 1])))
                key = pairs.setdefault((i, j), len(pairs))
                rows.append(o), cols.append(key), vals.append(coef)
        K, n = len(pairs), self.n
        I = np.array([p[0] for p in pairs], dtype=np.int64)
        Jx = np.array([p[1] for p in pairs], dtype=np.int64)
        lin = sp.csr_matrix((lin_v, (lin_r, lin_c)), shape=(self.m + 1, n))
        W = sp.csr_matrix((vals, (rows, cols)), shape=(self.m + 1, K))
        li, ui, lj, uj = self.lb[I], self.ub[I], self.lb[Jx], self.ub[Jx]
        env, rl, ru = [], [], []
        k = np.arange(K)
        # w - lj xi - li xj >= -li lj ; w - uj xi - ui xj >= -ui uj ; w - uj xi - li xj <= -li uj ; w - lj xi - ui xj <= -ui lj
        for a, bb, rhs, lower in ((lj, li, -li * lj, True), (uj, ui, -ui * uj, True),
                                  (uj, li, -li * uj, False), (lj, ui, -ui * lj, False)):
            ok = np.isfinite(a) & np.isfinite(bb) & np.isfinite(rhs)
            Mx = sp.csr_matrix((-a[ok], (k[ok], I[ok])), shape=(K, n)) \
                + sp.csr_matrix((-bb[ok], (k[ok], Jx[ok])), shape=(K, n))
            Mw = sp.csr_matrix((np.ones(int(ok.sum())), (k[ok], k[ok])), shape=(K, K))
            env.append(sp.hstack([Mx, Mw]))
            rl.append(np.where(ok, rhs, -np.inf) if lower else np.full(K, -np.inf))
            ru.append(np.full(K, np.inf) if lower else np.where(ok, rhs, np.inf))
        A = sp.vstack([sp.hstack([lin[1:], W[1:]])] + env)
        with np.errstate(invalid='ignore'):
            cand = np.stack([li * lj, li * uj, ui * lj, ui * uj])
        fin = np.isfinite(cand).all(0)                  # an unbounded factor: no product bound
        wl = np.where(fin, np.where(fin, cand, 0.0).min(0), -np.inf)
        wu = np.where(fin, np.where(fin, cand, 0.0).max(0), np.inf)
        wl = np.where(I == Jx, np.maximum(wl, 0.0), wl)
        obj = np.r_[cost, np.asarray(W[0].todense()).ravel()]
        z = _lp(obj, A, np.r_[self.cl - const[1:], np.concatenate(rl)], np.r_[self.cu - const[1:], np.concatenate(ru)],
                np.r_[self.lb, wl], np.r_[self.ub, wu], None, deadline, self.lp_tol)
        self.lps += 1
        if z is None:
            return None
        x = np.clip(z[0][:n], self.lb, self.ub)
        return x, {p: z[0][n + q] for p, q in pairs.items()}

    def fit(self, x, products, k):
        """block k set consistently with the lifted point: x_j = argmin_t sum_i (w_ij - x_i t)^2 over j's product
        partners i (the least-squares ratio), clipped to the box; j keeps its value without a nonzero partner"""
        num, den = np.zeros(self.n), np.zeros(self.n)
        for (i, j), w in products.items():
            if i == j:
                continue
            for a, b in ((i, j), (j, i)):
                if self.space.colour[b] == k:
                    num[b] += w * x[a]
                    den[b] += x[a] * x[a]
        out = np.array(x, dtype=np.float64)
        ok = (self.space.colour == k) & (den > 0)
        out[ok] = np.clip(num[ok] / den[ok], self.lb[ok], self.ub[ok])
        return out


def _problem(state):
    """the root problem of the first OR clause as an NNProblemIPOPT, and the clause"""
    from nlp_solver.ipopt_solver import NNProblemIPOPT
    from nlp_solver.utils import _split_or_clauses
    C, rhs = state.c, state.rhs
    C, rhs = (C.squeeze(0), rhs.squeeze(0)) if C.ndim == 3 else (C, rhs)
    clauses = list(_split_or_clauses(C, rhs, state.or_spec_size))
    if not clauses:
        return None, None
    rows = state.base_input_linear_constraints
    C_obj, rhs_obj, C_constr, rhs_constr = clauses[0]
    dm_l, dm_u = state.dm_l.reshape(-1, *state.dm_l.shape[1:])[0], state.dm_u.reshape(-1, *state.dm_u.shape[1:])[0]
    problem = NNProblemIPOPT(state.net.model_ori, dm_l, dm_u, C_obj, rhs_obj, C_constr, rhs_constr,
                             input_constr_A=None if rows is None else rows[0],
                             input_constr_b=None if rows is None else rows[1], device=None, dtype=None)
    return problem, (C_obj, rhs_obj, C_constr, rhs_constr, dm_l)


def _setup(state):
    """(problem, clause, Alternation) when the root problem is a QCQP with a product, else None; cached on the
    state"""
    cached = getattr(state, '_lp_alternation', False)
    if cached is not False:
        return cached
    out = None
    try:
        from nlp_solver.sparse_derivatives import sparse_problem
        from nlp_solver.structure.branching import branching_space
        problem, clause = _problem(state)
        sparse = None if problem is None else sparse_problem(problem)
        if sparse is not None and not sparse.nlp.tape.levels and _quadratic(sparse.nlp.polys):
            polys, n = sparse.nlp.polys, sparse.nlp.n
            cl, cu = sparse.bounds(0.0)
            eq = [r for r in range(sparse.m) if all(len(m) == 1 and m[0][1] == 1 for m in polys[1 + r])
                  and np.isfinite(cl[r]) and cl[r] == cu[r]]
            A_eq = sp.csr_matrix(([float(c) for r in eq for c in polys[1 + r].values()],
                                  ([i for i, r in enumerate(eq) for _ in polys[1 + r]],
                                   [m[0][0] for r in eq for m in polys[1 + r]])), shape=(len(eq), n))
            space = branching_space(polys, n, None, A_eq, cl[eq])
            if len(space.edges):
                tol = float(state.nlp_config.get('constr_viol_tol', 1e-6))
                out = (problem, clause, Alternation(sparse, problem.lb.copy(), problem.ub.copy(), space, tol))
    except Exception as e:  # noqa: BLE001 - the lane is a heuristic; a failure only means no lane
        print(f"[LPAlternation] not run: {type(e).__name__}: {e}")
    try:
        state._lp_alternation = out
    except AttributeError:
        pass
    return out


def _quadratic(polys):
    """every monomial a product of at most two input factors with positive integer exponents (callers have no
    atoms), and at least one of degree two"""
    degrees = [sum(e for _, e in m) for p in polys for m in p
               if all(type(e) is int and e > 0 for _, e in m)] + [0]
    total = sum(len(p) for p in polys)
    return len(degrees) - 1 == total and max(degrees) == 2


def eligible(state):
    """the root problem's exact forms are a QCQP with at least one product of two variables (no atoms), so every
    block step is an exact LP"""
    return bool(state.nlp_config.get('lp_alternation', False)) and _setup(state) is not None


@retained
def _search(alt, config, seeds, deadline, offer, progress=lambda stats: None):
    """The search on the exact problem until `deadline`: the alternation and IPOPT families served in turn.
    offer(x, f, source) gets every point within the verify tolerance on the tape that improves on the best one so
    far (f: its tape objective); progress(stats) the statistics after every job. Returns the statistics."""
    from nlp_solver.ipopt_solver import IPOPTSolver
    from nlp_solver.sparse_derivatives import TapeProblem
    tol = alt.tol
    sliced = bool(config.get("controller", False))
    best, stats = [math.inf], {'sources': {}, 'perturbed': 0}
    problem = TapeProblem(alt.sparse, alt.lb, alt.ub)
    # GEAR's IPOPT (solve_single) aimed at every row's band of half the verify tolerance: a point that ends outside
    # the box's relaxed bounds is finished inside the box, and there is no polish beyond the band
    solver = IPOPTSolver(dict(config, sparse_derivatives=True, ipopt_verbose=False, ipopt_constraint_slack_frac=0.5))

    def feasible(x, source):
        """tape objective of x when every raw row is within the verify tolerance (else None); offers improvements"""
        x = np.clip(np.asarray(x, dtype=np.float64), alt.lb, alt.ub)
        raw = alt.sparse.raw_rows(x)
        if raw.size and np.max(raw) > tol:
            return None
        f = alt.sparse.objective(x)
        if f < best[0]:
            best[0] = f
            stats['sources'][source] = stats['sources'].get(source, 0) + 1
            offer(x, f, source)
        return f

    def ipopt(x0, source):
        """IPOPT on the exact problem from x0 (at most half of the time left); a point that improves on the best one
        seeds the alternation, whose objective LPs from a feasible point move it to a vertex"""
        if config.get('_controller_native_submit') is not None:
            config['_controller_native_submit'](np.clip(x0, alt.lb, alt.ub))
            return
        before = best[0]
        res = solver.solve_single(problem, x_init=np.clip(x0, alt.lb, alt.ub), timeout=(1.0 if sliced else 0.5) * (deadline - time.time()))
        if res.x_solution is not None and feasible(res.x_solution, source) is not None and best[0] < before:
            queue.append((np.clip(res.x_solution, alt.lb, alt.ub), 0, source + '/alt'))

    def alternation(x0, first, source):
        """seed -> block alternation -> joint trust-region steps (at most half of the time left); the feasible end
        point seeds IPOPT"""
        until = deadline if sliced else time.time() + 0.5 * (deadline - time.time())
        x = yield from alt.alternate(x0, first, until, lambda z: feasible(z, source + '/alt'),
                                     sliced=sliced, _iterate=True)
        f = None if x is None else feasible(x, source + '/alt')
        if f is not None:
            x, f = yield from alt.trust_region(x, f, until, lambda z: feasible(z, source + '/tr'),
                                               sliced=sliced, _iterate=True)
            starts.append((x, source + '/alt/ipopt'))

    K = max(alt.space.colours, 1)
    g = _cost(alt)
    relax = alt.mccormick(g, deadline)
    lo = np.where(np.isfinite(alt.lb), alt.lb, np.minimum(alt.ub, 0.0) - 1.0)
    hi = np.where(np.isfinite(alt.ub), alt.ub, np.maximum(lo, 0.0) + 1.0)
    # two job families served in turn: block alternation (feasibility) and IPOPT (quality); alternation results
    # join IPOPT's queue as warm starts
    queue, starts = deque(), deque([(0.5 * (lo + hi), 'centre/ipopt')])
    if relax is not None:
        xr, products = relax
        queue.extend((xr, k, 'mccormick') for k in range(K))
        queue.extend((alt.fit(xr, products, k), (k + 1) % K, f'fit{k}') for k in range(K))
        starts.append((xr, 'mccormick/ipopt'))
    starts.extend((np.asarray(x0, dtype=np.float64), 'seed/ipopt') for x0 in seeds)
    rng = np.random.default_rng(0)
    scale = float(np.max(np.abs(g), initial=0.0)) or 1.0
    while time.time() < deadline:
        if not queue and not starts:
            # a McCormick vertex of a perturbed cost: the k-th of a scale-free sequence of perturbation sizes
            stats['perturbed'] += 1
            k = stats['perturbed']
            vertex = alt.mccormick(g + ((k * _GOLDEN) % 1.0) * scale * rng.standard_normal(alt.n), deadline)
            if vertex is not None:
                queue.append((vertex[0], k % K, 'perturbed'))
                starts.append((vertex[0], 'perturbed/ipopt'))
            if sliced:
                yield Atom('lp_alternation')
            continue
        if queue:
            yield from alternation(*queue.popleft())
        if starts and time.time() < deadline:
            if config.get("_controller_native_wait") is not None:
                yield from config["_controller_native_wait"]()
            ipopt(*starts.popleft())
            if sliced:
                yield Atom('lp_alternation')
        stats['lps'] = alt.lps
        progress(stats)
        if sliced:
            yield Atom('lp_alternation', settled=True)
    stats['lps'] = alt.lps
    return stats


def _seeds(state):
    """the caller's points as numpy vectors"""
    pts = list(getattr(state, 'seeds', None) or []) + [state.best_primal_x]
    return [np.asarray(s.detach().reshape(-1).cpu().double().numpy() if torch.is_tensor(s) else s) for s in pts
            if s is not None]


def _verifier(state, setup):
    """x -> objective when GEAR's verifier accepts x on the original model, else None"""
    from nlp_solver.statistics import verify_counterexample
    _, (C_obj, rhs_obj, C_constr, rhs_constr, dm_l), alt = setup
    rows = state.base_input_linear_constraints

    def verify(x):
        xt = torch.as_tensor(x, dtype=dm_l.dtype, device=dm_l.device).view(dm_l.shape)
        ok, obj, _ = verify_counterexample(state.net.model_ori, xt, C_obj, rhs_obj, C_constr, rhs_constr,
                                           input_constr_A=None if rows is None else rows[0],
                                           input_constr_b=None if rows is None else rows[1], tol=alt.tol)
        return float(obj) if ok else None
    return verify


def _publish(state, setup, x, obj, work=0.0):
    """Publish a point already accepted on the original model, in the root's device and objective sense."""
    if getattr(state, 'controller', False):
        from input_split.root_controller import offer
        offer(state, 'lp_alternation', x, work)
        return
    current = state.primal_solution
    if x is not None and (current is None or obj < float(torch.as_tensor(current).reshape(-1)[0])):
        dm_l = setup[1][4]
        point = torch.as_tensor(x, dtype=dm_l.dtype, device=dm_l.device).view(dm_l.shape)
        value = torch.tensor([obj], dtype=dm_l.dtype, device=dm_l.device)
        state.best_primal_x = point
        state.primal_solution = value
        state.primal_solution_found = True


def _commit(state, setup, x, obj, stats, t0, offered=None, rejected=None, tape_best=None, charged=0.0):
    alt = setup[2]
    print(f"[Mech] lp_alternation P={obj:.10g} found={x is not None} offered={offered} rejected={rejected} "
          f"tape_best={tape_best} sources={stats.get('sources')} "
          f"perturbed={stats.get('perturbed')} lps={stats.get('lps')} colours={max(alt.space.colours, 1)} "
          f"bipartite={alt.space.bipartite} |S|={len(alt.space.cover)} |B|={len(alt.space.basis)} "
          f"t={time.time() - t0:.2f}s reason=ok")
    if getattr(state, 'controller', False):
        from input_split.root_controller import offer
        offer(state, 'lp_alternation_done', x, max(0.0, time.time() - t0 - charged))
    else:
        _publish(state, setup, x, obj)


def run(state, deadline):
    """The search on the root problem in this process until `deadline`. Commits the best point that GEAR's verifier
    accepts to state.primal_solution and state.best_primal_x when it beats them; returns None (it never ends the solve
    by itself)."""
    t0 = time.time()
    setup = _setup(state)
    if setup is None:
        return None
    verify, best, count = _verifier(state, setup), [None, math.inf], [0, 0, math.inf]

    def offer(x, f, source):
        obj = verify(x)
        count[0], count[1], count[2] = count[0] + 1, count[1] + (obj is None), min(count[2], f)
        if obj is not None and obj < best[1]:
            best[0], best[1] = x, obj
    stats = _search(setup[2], state.nlp_config, _seeds(state), deadline, offer)
    _commit(state, setup, best[0], best[1], stats, t0, *count)
    return None


def _cost(alt):
    """the objective's linear part"""
    g = np.zeros(alt.n)
    for m, c in alt.sparse.nlp.polys[0].items():
        if len(m) == 1 and m[0][1] == 1:
            g[m[0][0]] += float(c)
    return g


class Lane:
    """The search in a child process beside the GPU work, until `deadline` or until the root's own primal work is
    done (`merge`/`join`), whichever is first: the lane never lengthens the root. A thread lane is starved by the GIL
    that the GPU ALM's main thread holds (1 LP/s instead of 20 with a process; arch2 spec 2.4 asks for processes). This
    process compiles the problem and keeps the judge: the child gets the exact problem (tape, box, blocks), streams
    every improving point back, and each one goes through GEAR's verifier on the original model here."""

    def __init__(self, state, deadline):
        self.state, self.deadline, self.error = state, deadline, None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name='lp_alternation', daemon=True)
        self.thread.start()

    def _run(self):
        import os
        import pickle
        import subprocess
        import sys
        t0 = time.time()
        child, reader = None, None
        try:
            if not eligible(self.state) or self.stop.is_set():
                return
            setup = _setup(self.state)
            payload = pickle.dumps((setup[2], dict(self.state.nlp_config), _seeds(self.state), self.deadline),
                                   protocol=pickle.HIGHEST_PROTOCOL)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', HIGHS_THREADS='1', NUMEXPR_NUM_THREADS='1',
                       PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
            child = subprocess.Popen([sys.executable, '-c', 'from nlp_solver.lp_alternation import _child_main; '
                                      '_child_main()'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
            points, stats = deque(), {}
            reader = threading.Thread(target=_read, args=(child.stdout, points, stats), daemon=True)
            reader.start()
            child.stdin.write(payload)
            child.stdin.close()
            verify, best = _verifier(self.state, setup), (None, math.inf)
            offered, rejected, tape_best = 0, 0, math.inf
            charged = 0.0

            def publish():
                nonlocal best, offered, rejected, tape_best, charged
                while points:
                    x, f, source = points.popleft()
                    obj = verify(x)
                    offered, tape_best = offered + 1, min(tape_best, f)
                    if obj is None:
                        rejected += 1
                    elif obj < best[1]:
                        best = (x, obj)
                        # Root NLP does not write the incumbent; the next root op waits for join().
                        # Publish now so RootState always holds the verified incumbent, not a tape claim.
                        if getattr(self.state, 'controller', False):
                            elapsed = time.time() - t0
                            _publish(self.state, setup, x, obj, max(0.0, elapsed - charged))
                            charged = elapsed
                        else:
                            _publish(self.state, setup, x, obj)

            while reader.is_alive() and not self.stop.is_set() and time.time() < self.deadline:
                reader.join(0.05)
                publish()
            if child.poll() is None:
                child.kill()                    # our own child, by PID; its points so far are already here
            child.wait()
            reader.join()
            publish()
            _commit(self.state, setup, best[0], best[1], dict(stats), t0, offered, rejected,
                    tape_best if offered else None, charged=charged)
        except Exception as e:  # noqa: BLE001 - the lane is a heuristic: a failure only means no point
            self.error = e
            print(f"[LPAlternation] failed: {type(e).__name__}: {e}")
        finally:
            if child is not None:
                if child.poll() is None:
                    child.kill()
                child.wait()
                if reader is not None:
                    reader.join()
                child.stdout.close()
                child.stdin.close()

    def join(self):
        """stops the search (the caller's primal work is done) and waits for the lane; (objective, point) of the
        state's incumbent"""
        self.stop.set()
        self.thread.join()
        return self.state.primal_solution, self.state.best_primal_x

    def merge(self, comparison):
        """stops the lane and makes its verified point the root comparison's best when it is better"""
        obj, x = self.join()
        if obj is None:
            return
        obj = float(obj.reshape(-1)[0])
        better = comparison.best_obj is None or comparison.best_method == 'none' or obj < comparison.best_obj
        print(f"[Root] LP alternation: P={obj:.10g} {'kept' if better else 'not better'} "
              f"(root best {comparison.best_method}={comparison.best_obj})")
        if better:
            comparison.best_method, comparison.best_obj, comparison.best_x = 'lp_alternation', obj, x


def _read(stream, points, stats):
    """the child's messages: ('point', x, f, source) and ('stats', statistics so far)"""
    import pickle
    try:
        while True:
            kind, *body = pickle.load(stream)
            if kind == 'point':
                points.append(tuple(body))
            else:
                stats.update(body[0])
    except (EOFError, pickle.UnpicklingError, OSError):
        pass


def _child_main():
    """The child of a Lane: the pickled (Alternation, config, seeds, deadline) on stdin; every improving point and
    the statistics after every job on stdout (pickled; prints go to stderr)."""
    import os
    import pickle
    import sys
    out = os.fdopen(os.dup(1), 'wb')
    os.dup2(2, 1)
    alt, config, seeds, deadline = pickle.load(sys.stdin.buffer)

    def send(*message):
        pickle.dump(message, out)
        out.flush()
    try:
        stats = _search(alt, config, seeds, deadline, lambda x, f, source: send('point', x, f, source),
                        lambda stats: send('stats', stats))
        send('stats', stats)
    finally:
        out.close()


def start(state, deadline):
    """the search in a child process; `merge(comparison)` or `join()` collects it"""
    return Lane(state, deadline)
