"""Lifted McCormick LP (RLT level 1) over the spec rows that are provably quadratic.

A spec row r is a combination C_r . y(x) of model outputs. When every output it uses is a
polynomial of degree <= 2 (proved on the bound graph by quadratic_tangent.output_degrees,
never guessed), the row is represented exactly as

    s_r(x) = k_r + g_r^T x + sum_k q_rk w_k,      w_k = x_{a_k} x_{b_k},

with one lifted variable per distinct product (a_k <= b_k). On a box [l, u] every product
obeys its McCormick inequalities (a square: its secant above and tangent lines below) and
lies in the interval hull of the product. So for every x in the box that satisfies the
relaxed rows s_r(x) <= rhs_r + tol, the point (x, w(x)) is feasible for the LP

    min  s_0 (lifted)   s.t.  lifted rows <= rhs + tol + cushion,  McCormick rows,
                              extra x-space rows (CROWN planes, base linear rows), box on (x, w),

and by weak duality, for ANY multipliers y >= 0, the closed-form Lagrangian value
c0 - h^T y + sum_j min(r_j lz_j, r_j uz_j), r = c + G^T y, bounds s_0(x) from below.
HiGHS or PDHG only propose y; their accuracy never affects validity.

The opt-in float64 convex objective path proves f=q0+sum c_k*phi_k(q_k),
degree(q_k)<=2 and c_k>0, with convex phi_k on finite root domains. It appends
bounded epigraph variables and outward-relaxed tangent rows, retaining a capped
set of adaptive points across nodes. These rows use the same dual certificate.

Rows of degree > 2 or with exp, log and real powers (1/x included) enter through their exact
forms.py form: an atom of an affine argument is a column t = phi(arg) between the inputs and
the products (lifted_atoms.py: tangents and secant of phi over the argument's range on each
box), and monomials multiply atoms by inputs, atoms or products like any other column.

Unlike the node LP over one CROWN plane per row, one lifted w_k is shared by every row that
uses the product, and both McCormick planes of every product are present.

With clear_denominators, rational_rows proves Laurent forms on the graph and adds
m(x) * (s_r(x) - relaxed_rhs_r) <= 0 when m is a monomial of strictly positive root
variables and the resulting polynomial is quadratic. These cuts leave the original
rows intact. Their coefficients use exact rational arithmetic followed by outward
conversion-error enclosures, added to each box's rounding cushion.

Root OBBT: minimising and maximising x_j over the same LP, with the objective row cut off
at the incumbent (whose point is feasible for the relaxed problem, so the minimiser of the
relaxed problem stays inside), gives certified bounds on x_j by the same weak duality.

Ordinary coefficients come from autograd in float64 at x = 0 (exact for a polynomial of degree
<= 2 up to rounding), are checked against the model at random points of the box, and every
row carries a cushion 1e-12 (1 + |k| + sum |g_j| M_j + sum |q_k| MW_k), M_j = max(|l_j|, |u_j|)
and MW_k = M_a M_b on the box the LP is built for (coefficient rounding is ~1e-16 relative per
operation; a root-box cushion made rows with 1e6 bounds useless after OBBT); the McCormick
rows carry 1e-12 of their own magnitude against the rounding of their data.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from .continuation import Atom, retained
from nlp_solver.utils import locked_jvp
from torch import Tensor

from .rigorous_dual import Rows, certify_infeasible, dual_lower_bound

# Tangent points of a square w = x^2 on [l, u], as fractions of the interval.
_SQ_TANGENTS = (0.0, 0.25, 0.5, 0.75, 1.0)
_CUSHION = 1e-12
# Lifted-LP branching: only variables at least this fraction as wide (relative to the root box)
# as the widest one compete on McCormick violation (see violation_score).
_BRANCH_MIN_WIDTH_RATIO = 0.05
_MC_CUSHION = 1e-12
# Problems with more atoms (lifted_atoms) than this keep the unlifted path (see _liftable_forms);
# pgon_polygon has 20 sin atoms, the polygon family hundreds to thousands.
MAX_ATOMS = 64
# Lifted-LP B&B nodes: passes of propagate-then-rebuild, and OBBT probes (two LPs per variable, on the
# node's own LP) of the most violated variables, only when that LP has at most _NODE_OBBT_ROWS rows.
# At 51 s, probes of 4 variables helped the LPs of 374-1060 rows and hurt those of 1644-8694 rows
# (wastewater15m1 closes in 408 nodes without probes, not in 65 nodes with them; 11m1, 12m1, waterund08
# likewise); with 32 variables ex8_4_3 (P 0.00449) closes in 16 nodes (D 0.0022 without probes) and
# gsg_0001 in 147 (D 2375.0 without).
_NODE_PASSES = 3
_NODE_OBBT_VARS = 32
_NODE_OBBT_ROWS = 1200
# Probing stops for the rest of a run when, after _PROBE_NODES probed nodes, fewer than _PROBE_PAY of the
# probes tightened their variable by more than _PAY_SHRINK of its width (or emptied the node). Paying
# fraction of the first 10 probed nodes (CPU): kall_diffcircles_9 0.14, _10 0.14, kall_congruentcircles_c51
# 0.13 (with probes 80-100 nodes in 7.5 s and D stays 0; without, _10 reaches D = 8.2 in 7000 nodes);
# kall_diffcircles_8 0.15-0.18 looks the same early but gains from probes (D 14.48 of P 14.78 in 600 nodes,
# without 12.63); kall_congruentcircles_c52/_c62 0.32, kall_diffcircles_5b/6/7 0.44-0.50, gsg_0001 and
# ex8_4_3 0.87 (whole runs).
_PROBE_NODES = 10
_PROBE_PAY = 0.2
_PAY_SHRINK = 0.01
# Simplex iterations per (rows + columns) a node LP or probe may take. Node LPs need 0.15 of that at
# the median; the one that stalls on waterund08 (5221 rows) ran 172k iterations (10 s) and made the
# progress rule stop the search at D 164.407, and ex8_4_3 at a near-optimal cutoff has a probe that
# runs to the time limit.
_NODE_ITERS = 5
# The bound counts as frozen only after this many nodes without progress as well as the time window: with
# probes a node costs 10-100 times more, and kall_diffcircles_5a (D flat for 41 nodes in 3 s) closes at 180.
_PLATEAU_NODES = 200
# The B&B offers its best node's LP point to the local solver at nodes _LOCAL_NODES[0], 2x, 4x, ... _LOCAL_NODES[1]
# (primal_fn) until one improves the cutoff materially. A solve costs 0.3-5 s: from node 1 on, the instances that
# close within 100 nodes paid 3-9 s for nothing (ex8_4_3 7.7 -> 16.7 s, waterund08 3.6 -> 11.8 s, wastewater15m1
# 6.8 -> 11.4 s; the root's local optimum was their optimum). wastewater11m1: 6 of 11 solves from the best node's
# point reach the optimum 2127.10 below the root's local optimum 2149.01.
_LOCAL_NODES = (128, 2048)
# Seconds one node (LP and probes) may take: a stalled LP only loses its node's bound.
_NODE_TIME = 10.0
# Boxes of columns that are products of product columns are widened by this relative amount
# (several roundings of an interval product); first-level products keep the plain corners.
_BOX_WIDEN = 4e-16
# Relative error assumed for the coefficients of a form with rounded transcendental constants.
_INEXACT = 1e-13


def _levels(a, b, n):
    """Product indices grouped by depth: level d holds the products whose factors a, b (column
    indices; < n are inputs) are inputs or products of lower levels, so every level can be
    evaluated from the previous ones. A degree <= 2 model has the single level of all products."""
    depth = torch.zeros(n + a.numel(), dtype=torch.long, device=a.device)
    while True:
        new = 1 + torch.maximum(depth[a], depth[b])
        if torch.equal(new, depth[n:]):
            break
        depth[n:] = new
    top = int(depth.max()) if a.numel() else 0
    return [(depth[n:] == d).nonzero().reshape(-1) for d in range(1, top + 1)]


def _monomial_vars(a, b, n, atoms=None):
    """(k, v) pairs: input variable v occurs in definition k once per factor occurrence. The
    definitions are the atoms (k < T, each over the inputs of its argument), then the products."""
    T = 0 if atoms is None else atoms.T
    k = torch.arange(a.numel(), device=a.device) + T
    ks, cols = torch.cat([k, k]), torch.cat([a, b])
    out_k, out_v = [k[:0]], [a[:0]]
    if T:
        support = torch.as_tensor(atoms.arrays()['support'], device=a.device)
        out_k.append(support.nonzero()[:, 0]); out_v.append(support.nonzero()[:, 1])
    while ks.numel():
        inp, atom = cols < n, (cols >= n) & (cols < n + T)
        out_k.append(ks[inp]); out_v.append(cols[inp])
        if bool(atom.any()):
            j, v = support[cols[atom] - n].nonzero(as_tuple=True)
            out_k.append(ks[atom][j]); out_v.append(v)
        ks, cols = ks[cols >= n + T], cols[cols >= n + T] - n - T
        ks, cols = torch.cat([ks, ks]), torch.cat([a[cols], b[cols]])
    return torch.cat(out_k), torch.cat(out_v)


def _column_boxes(l, u, a, b, levels, xp, atoms=None):
    """Boxes (lz, uz) of all columns (inputs, atoms, then products w_k = z_{a_k} z_{b_k}) over the
    input box l, u: one box (xp=np, 1-D) or a batch (xp=torch, (B, n)). Atoms from their argument's
    range (lifted_atoms), then interval products in level order; a square is nonnegative."""
    n = l.shape[-1] + (0 if atoms is None else atoms.T)
    pad = (lambda t: np.concatenate([t, np.zeros(t.shape[:-1] + (n - t.shape[-1] + len(a),))], -1)) if xp is np \
        else (lambda t: torch.cat([t, t.new_zeros(t.shape[:-1] + (n - t.shape[-1] + len(a),))], -1))
    lz, uz = pad(l), pad(u)
    if atoms is not None:
        atoms.fill_boxes(lz, uz, xp)
    amin, amax = (np.amin, np.amax) if xp is np else (torch.amin, torch.amax)
    for depth, ks in enumerate(levels, 1):
        ia, ib = a[ks], b[ks]
        la, ua, lb, ub = lz[..., ia], uz[..., ia], lz[..., ib], uz[..., ib]
        cand = xp.stack([la * lb, la * ub, ua * lb, ua * ub], -1)
        wl, wu = amin(cand, -1), amax(cand, -1)
        sq = ia == ib
        wl = xp.where(sq & (wl < 0), 0.0, wl)
        wl = xp.where(sq & (la <= 0) & (ua >= 0), 0.0, wl)
        if depth > 1:
            wl, wu = wl - _BOX_WIDEN * abs(wl), wu + _BOX_WIDEN * abs(wu)
        lz[..., n + ks], uz[..., n + ks] = wl, wu
    return lz, uz


def _vectors(*arrays):
    """float64 batches of one, the evaluator's input (rigorous_dual)."""
    return [torch.from_numpy(np.asarray(a, dtype=np.float64)).reshape(1, -1) for a in arrays]


def _dual_value_sparse(c, c0, G, h, lz, uz, y):
    """Weak-duality bound of min c^T z + c0 s.t. G z <= h, lz <= z <= uz at y >= 0 (scipy sparse G), rigorous
    for the data as given; -inf when it is not finite: the B&B treats a NaN bound as closed (nan < cutoff is False)."""
    c, h, lz, uz, y = _vectors(c, h, lz, uz, y)
    lb, _, _ = dual_lower_bound(Rows.from_scipy(G), c, torch.tensor([float(c0)], dtype=torch.float64), h, lz, uz, y)
    return float(lb[0])


def _certified_infeasible(G, h, lz, uz, y):
    """True when the multipliers y >= 0 prove G z <= h empty over the box: a Farkas certificate, min over the
    box of y^T (G z - h) is > 0 after the rounding error of its evaluation."""
    h, lz, uz, y = _vectors(h, lz, uz, y)
    return bool(certify_infeasible(Rows.from_scipy(G), h, lz, uz, y)[0])


def _ray_proves_empty(res, G, h, lz, uz):
    """True when the Farkas ray HiGHS returned with an infeasible status proves G z <= h empty over the box."""
    ray = getattr(res, 'ray', None)
    return ray is not None and _certified_infeasible(G, h, lz, uz, np.maximum(-ray, 0.0))


def _phase1_proves_empty(G, h, lz, uz, options):
    """True when the multipliers of the phase-I LP min{t : G z - t <= h, lz <= z <= uz, t >= 0} prove
    G z <= h empty over the box (options as for _highs_proposal, without a warm start)."""
    import scipy.sparse as sp
    m = G.shape[0]
    Gt = sp.hstack([G, -np.ones((m, 1))]).tocsr()
    ct = np.zeros(G.shape[1] + 1)
    ct[-1] = 1.0
    try:
        r2 = _highs_proposal(ct, Gt, h, np.append(lz, 0.0), np.append(uz, np.inf), options)
    except Exception:  # noqa: BLE001
        return False
    if r2.status != 0 or getattr(r2, 'ineqlin', None) is None:
        return False
    return _certified_infeasible(G, h, lz, uz, np.maximum(-r2.ineqlin.marginals, 0.0))


def _fbbt(G, h, lz, uz, rounds=3):
    """Interval propagation of G z <= h over the box lz <= z <= uz, every operation rounded
    outward: a term is rounded down, the row sum is lowered by its summation error, and the
    bound of z_j is rounded away from the box, so only points that violate a row (as stored,
    cushions included) are removed. An unrounded version declared wastewater14m1 empty
    (cancellation among 1e10 coefficients). Returns (lo, hi, row): the tightened box (the
    input one where a bound is not finite) and, when the lowest value of row `row` over it
    exceeds its rhs, that row's index, else None. The caller proves such a claim."""
    lo, hi = lz.copy(), uz.copy()
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        return lo, hi, None
    sizes = np.diff(G.indptr)
    rows = np.repeat(np.arange(G.shape[0]), sizes)
    cols, coef = G.indices, G.data
    pos, nonzero = coef > 0, coef != 0
    eps = np.finfo(np.float64).eps
    gamma = (sizes + 2) * eps / (1 - (sizes + 2) * eps)
    for _ in range(rounds):
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            terms = np.nextafter(coef * np.where(pos, lo[cols], hi[cols]), -np.inf)
            total = np.bincount(rows, weights=terms, minlength=len(h))
            magnitude = np.nextafter(np.bincount(rows, weights=abs(terms), minlength=len(h))
                                     / (1 - gamma), np.inf)
            least = np.nextafter(total - np.nextafter(gamma * magnitude, np.inf), -np.inf)
            if np.any(least > h):
                return lo, hi, int(np.argmax(least - h))
            rest = np.nextafter(least[rows] - terms, -np.inf)
            bound = np.nextafter(np.nextafter(h[rows] - rest, np.inf) / coef,
                                 np.where(pos, np.inf, -np.inf))
        ok = np.isfinite(bound) & nonzero      # overflow or 0/0 only means no tightening
        new_lo, new_hi = lo.copy(), hi.copy()
        np.minimum.at(new_hi, cols[pos & ok], bound[pos & ok])
        np.maximum.at(new_lo, cols[~pos & ok], bound[~pos & ok])
        if np.any(new_lo > new_hi):
            break       # crossing is no single-row proof: the caller's LP decides
        moved = np.any(new_lo > lo + 1e-3 * (hi - lo)) or np.any(new_hi < hi - 1e-3 * (hi - lo))
        lo, hi = new_lo, new_hi
        if not moved:
            break
    return lo, hi, None


def _highs_proposal(c, G, h, lz, uz, options):
    """HiGHS proposal; with options['retry'] a presolved solve that HiGHS calls infeasible
    or numerically failed is repeated without presolve, which can falsely reject wide,
    redundant RLT systems (waterund08) but is 10-20x faster on the rest (ex8_3_2)."""
    options = dict(options)
    retry = options.pop('retry', False)
    start = time.time()
    warm = options.get('warm')
    if warm is not None and warm['basis'] is not None:
        # Unpresolved, from the basis. An infeasible node is proved by the caller (multipliers),
        # so only a failed solve is repeated cold.
        res = _highs_solve(c, G, h, lz, uz, {**options, 'presolve': False})
        if res.status in (0, 2):
            return res
        warm['basis'] = None
        if 'time_limit' in options:
            left = options['time_limit'] - (time.time() - start)
            if left <= 0:
                return res
            options['time_limit'] = left
        start = time.time()
    res = _highs_solve(c, G, h, lz, uz, options)
    if retry and options['presolve'] and res.status in (2, 4):
        if 'time_limit' in options:
            left = options['time_limit'] - (time.time() - start)
            if left <= 0:
                return res
            options['time_limit'] = left
        res = _highs_solve(c, G, h, lz, uz, {**options, 'presolve': False})
    return res


def _highs_lp(c, A, h, bounds, options, warm, rows):
    """linprog-like result of HiGHS on min c z s.t. A z <= h, bounds, started from warm['basis']
    (the (column, row) statuses of an earlier solve of the same LP structure, rows indexed as
    `rows` = (ids of A's rows among all rows, their count)); warm['basis'] is replaced by this
    solve's. A basis whose basic count no longer fits (rows that went inactive) or that HiGHS
    rejects is dropped: a cold start. Only speed and the proposed multipliers change."""
    import highspy
    H = highspy.Highs()
    H.setOptionValue('output_flag', False)
    H.setOptionValue('presolve', 'on' if options['presolve'] else 'off')
    # Devex pricing: steepest-edge weights are recomputed from scratch at a given basis. 25 node LPs
    # of wastewater12m1 from their parents' bases: 20434 iterations, 1.62 s -> 12393, 0.71 s.
    H.setOptionValue('simplex_dual_edge_weight_strategy', 1)
    if 'time_limit' in options:
        H.setOptionValue('time_limit', float(options['time_limit']))
    if 'maxiter' in options:
        H.setOptionValue('simplex_iteration_limit', int(options['maxiter']))
    lp = highspy.HighsLp()
    lp.num_row_, lp.num_col_ = A.shape
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = c, bounds[:, 0], bounds[:, 1]
    lp.row_lower_, lp.row_upper_ = np.full(len(h), -np.inf), h
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    H.passModel(lp)
    status = highspy.HighsBasisStatus
    status_table = tuple(status(i) for i in range(5))
    if warm.get('basis') is not None:
        col, row = warm['basis'][0], warm['basis'][1][rows[0]]
        if len(col) == A.shape[1] and (col == 1).sum() + (row == 1).sum() == A.shape[0]:
            basis = highspy.HighsBasis()
            basis.col_status = [status_table[int(s)] for s in col]
            basis.row_status = [status_table[int(s)] for s in row]
            basis.valid = True
            H.setBasis(basis)
    H.run()
    model = H.getModelStatus()
    code = {'kOptimal': 0, 'kIterationLimit': 1, 'kTimeLimit': 1, 'kInfeasible': 2,
            'kUnbounded': 3}.get(model.name, 4)
    res = SimpleNamespace(status=code, nit=H.getInfo().simplex_iteration_count,
                          x=None, ineqlin=None)
    if code == 0:
        solution, basis = H.getSolution(), H.getBasis()
        res.x = np.array(solution.col_value)
        res.ineqlin = SimpleNamespace(marginals=np.array(solution.row_dual))
        row = np.full(rows[1], 1, dtype=np.int8)       # rows not in this LP count as basic
        row[rows[0]] = [int(s) for s in basis.row_status]
        warm['basis'] = (np.array([int(s) for s in basis.col_status], dtype=np.int8), row)
    else:
        warm['basis'] = None
        ray = H.getDualRay()
        if code == 2 and ray[1]:
            res.ray = np.array(ray[2])
    return res


def _highs_solve(c, G, h, lz, uz, options):
    """Equilibrate wide/narrow RLT boxes for HiGHS, then map its proposal back.

    The certificate always evaluates the ORIGINAL G, h and box. Even inaccurate
    translated data or a poor solver proposal cannot invalidate that dual bound.
    Presolve-enabled (ordinary McCormick) solves retain their original data exactly.
    """
    from scipy.optimize import linprog
    import scipy.sparse as sp
    options = dict(options)
    warm = options.pop('warm', None)
    rows = getattr(G, '_rows', None)
    if options['presolve'] or not (np.isfinite(lz).all() and np.isfinite(uz).all()):
        if warm is not None and rows is not None:
            return _highs_lp(c, G, h, np.stack([lz, uz], 1), options, warm, rows)
        return linprog(c, A_ub=G, b_ub=h, bounds=np.stack([lz, uz], 1),
                       method='highs', options=options)
    radius = 0.5 * (uz - lz)
    center = lz + radius
    bounds = np.zeros((len(c), 2))
    nonzero = radius > 0
    bounds[nonzero, 0] = np.nextafter((lz[nonzero] - center[nonzero]) / radius[nonzero], -np.inf)
    bounds[nonzero, 1] = np.nextafter((uz[nonzero] - center[nonzero]) / radius[nonzero], np.inf)
    A = (G @ sp.diags(radius)).tocsr()
    scale = abs(A).max(axis=1).toarray().reshape(-1)
    scale = np.where(scale > 0, scale, 1.0)
    A = sp.diags(1.0 / scale) @ A
    # Translation can cancel large constants on almost fixed flow variables.
    hp = np.asarray(h.astype(np.longdouble) - G.astype(np.longdouble) @ center.astype(np.longdouble),
                    dtype=np.float64) / scale
    cp = c * radius
    objective_scale = max(float(np.abs(cp).max()), 1.0)
    if warm is not None and rows is not None:
        res = _highs_lp(cp / objective_scale, A, hp, bounds, options, warm, rows)
    else:
        res = linprog(cp / objective_scale, A_ub=A, b_ub=hp, bounds=bounds,
                      method='highs', options=options)
    if res.x is not None:
        res.x = center + radius * res.x
    if getattr(res, 'ineqlin', None) is not None and res.ineqlin.marginals is not None:
        res.ineqlin.marginals = res.ineqlin.marginals * (objective_scale / scale)
    if getattr(res, 'ray', None) is not None:
        res.ray = res.ray / scale
    return res


def _epigraph_proposal(c, G, h, lz, uz, options):
    """Refine a sparse epigraph LP at its proposed solution, with a fixed cut cap.

    Only the solver proposal changes. The caller certifies the final, outward
    relaxed rows using the same weak-duality expression as every other lifted LP.
    Metadata lives on the original sparse matrix, including in forked workers.
    """
    import scipy.sparse as sp
    metadata = getattr(G, '_epigraph', None)
    start = time.time()
    res, previous_G, previous_h = None, G, h
    # Each LP gets the grid and at most 32 retained points, plus 8 fresh cuts/term.
    for iteration in range(9):
        current_options = options.copy()
        if 'time_limit' in options:
            left = options['time_limit'] - (time.time() - start)
            if left <= 0:
                break
            current_options['time_limit'] = left
        previous_G, previous_h = G, h
        res = _highs_proposal(c, G, h, lz, uz, current_options)
        if metadata is None or iteration == 8 or res.status != 0 or res.x is None:
            return res, G, h
        terms, n_products = metadata
        cuts, rhs = [], []
        for j, term in enumerate(terms):
            point = float(np.clip(term.constant + term.coefficients @ res.x[:len(term.coefficients)],
                                  *term.domain))
            value = term.atom.intervals(point, point)[0][0]
            if value - res.x[n_products+j] > 1e-7 * (1 + abs(value)):
                row, bound = term.cut(point, lz, uz, n_products+j)
                cuts.append(row); rhs.append(bound)
                term.remember(point)
        if not cuts:
            return res, G, h
        G = sp.vstack([G, sp.csr_matrix(np.stack(cuts))]).tocsr()
        h = np.concatenate([h, rhs])
    # A solve of the previous rows still supplies a valid bound at the deadline.
    return res, previous_G, previous_h


def _mccormick_terms(l, u, a, b, sq, n, xp):
    """McCormick rows of every product as (row, col, val) triplets and rhs, for one box
    (xp=np, 1-D l, u) or a batch (xp=torch, (B, N) l, u), l, u the boxes of the columns
    (see _column_boxes) and a, b the factor columns of each product. Rows per bilinear product:

        -w + l_b x_a + l_a x_b <= l_a l_b        (x_a - l_a)(x_b - l_b) >= 0
        -w + u_b x_a + u_a x_b <= u_a u_b        (u_a - x_a)(u_b - x_b) >= 0
         w - l_b x_a - u_a x_b <= -u_a l_b       (u_a - x_a)(x_b - l_b) >= 0
         w - u_b x_a - l_a x_b <= -l_a u_b       (x_a - l_a)(u_b - x_b) >= 0

    and per square w = x^2: its secant w - (l+u) x <= -l u, and tangents -w + 2 t x <= t^2.
    Each rhs is relaxed by _MC_CUSHION times the row's magnitude over the box.
    Returns a list of (row_index (k,), w_col (k,), x_col (k,) or (k,2) pairs, coefs, rhs).
    """
    out = []
    kb = xp.nonzero(~sq)[0] if xp is np else (~sq).nonzero().reshape(-1)
    ks = xp.nonzero(sq)[0] if xp is np else sq.nonzero().reshape(-1)
    absf = np.abs if xp is np else torch.abs
    maxf = np.maximum if xp is np else torch.maximum
    r0 = 0
    if len(kb):
        ia, ib = a[kb], b[kb]
        la, ua, lb, ub = l[..., ia], u[..., ia], l[..., ib], u[..., ib]
        Ma, Mb = maxf(absf(la), absf(ua)), maxf(absf(lb), absf(ub))
        base = 4 * (np.arange(len(kb)) if xp is np else torch.arange(len(kb), device=a.device))
        for r, (sw, ca, cb, rh) in enumerate((
                (-1.0, lb, la, la * lb), (-1.0, ub, ua, ua * ub),
                (1.0, -lb, -ua, -ua * lb), (1.0, -ub, -la, -la * ub))):
            scale = absf(rh) + absf(ca) * Ma + absf(cb) * Mb + Ma * Mb
            out.append((base + r, n + kb, sw, (ia, ca), (ib, cb), rh + _MC_CUSHION * scale))
        r0 = 4 * len(kb)
    if len(ks):
        ia = a[ks]
        la, ua = l[..., ia], u[..., ia]
        M = maxf(absf(la), absf(ua))
        per = 1 + len(_SQ_TANGENTS)
        base = r0 + per * (np.arange(len(ks)) if xp is np else torch.arange(len(ks), device=a.device))
        out.append((base, n + ks, 1.0, (ia, -(la + ua)), None,
                    -la * ua + _MC_CUSHION * (absf(la * ua) + absf(la + ua) * M + M * M)))
        for i, fr in enumerate(_SQ_TANGENTS):
            t = la + fr * (ua - la)
            out.append((base + 1 + i, n + ks, -1.0, (ia, 2 * t), None,
                        t * t + _MC_CUSHION * (t * t + 2 * absf(t) * M + M * M)))
    return out


def extract_quadratic(fn, n: int, x_check: Tensor, device=None, chunk: int = 64,
                      linear: Optional[Tensor] = None):
    """Exact (k, g, pairs, q) of every output of fn: R^n -> R^m that is a polynomial of
    degree <= 2:  y_o(x) = k_o + g_o^T x + sum_k q_ok x_{a_k} x_{b_k}.

    Returns (k (m,), g (m, n), a (K,), b (K,), q (m, K), ok (m,) bool) as float64 tensors on
    `device`; ok[o] is False where the representation does not reproduce fn at the check
    points `x_check` (P, n), i.e. the output is not quadratic after all. Outputs flagged
    in `linear` (m,) get no Hessian (the check still covers them).
    """
    from input_split.tree_slice import checkpoint
    checkpoint()
    dt = torch.float64
    x0 = torch.zeros(n, dtype=dt, device=device)
    k = fn(x0).detach()
    m = k.numel()
    jac = torch.func.jacrev(fn)
    g = jac(x0).detach().reshape(m, n)
    # Supports: an output's Hessian lives in supp x supp, supp from gradients at 0 and at
    # the check points (a quadratic's gradient is affine, so this is its full support
    # with probability one; a missed variable only makes the check below fail).
    supp = g != 0
    for p in range(x_check.shape[0]):
        supp |= jac(x_check[p].to(device=device, dtype=dt)).detach().reshape(m, n) != 0
    if linear is not None:
        supp[linear.to(supp.device)] = False
    rows, ia, ib, vals = [], [], [], []
    order = torch.argsort(supp.to(torch.int64).argmax(1))     # group outputs by first var
    order = order[supp[order].any(1)]
    for s in range(0, order.numel(), chunk):
        checkpoint()
        idx = order[s:s + chunk]
        U = supp[idx].any(0).nonzero().reshape(-1)
        if U.numel() == 0:
            continue
        grad = torch.func.jacrev(lambda x, idx=idx: fn(x)[idx])
        E = torch.zeros(U.numel(), n, dtype=dt, device=device)
        E[torch.arange(U.numel(), device=device), U] = 1.0
        # H[:, :, j] for j in U: directional derivative of the Jacobian along e_j.
        Hc = torch.func.vmap(lambda v: locked_jvp(grad, (x0,), (v,))[1])(E).detach()
        checkpoint()
        Hc = Hc.reshape(U.numel(), idx.numel(), n).permute(1, 2, 0)   # [o, i, j in U]
        nz = (Hc != 0).nonzero()
        o_l, i_l, j_l = nz[:, 0], nz[:, 1], U[nz[:, 2]]
        keep = i_l <= j_l
        v = Hc[nz[keep, 0], nz[keep, 1], nz[keep, 2]]
        o_l, i_l, j_l = o_l[keep], i_l[keep], j_l[keep]
        v = torch.where(i_l == j_l, 0.5 * v, v)       # x^T H x / 2 in upper-triangular pairs
        rows.append(idx[o_l]); ia.append(i_l); ib.append(j_l); vals.append(v)
    if rows:
        rows = torch.cat(rows); ia = torch.cat(ia); ib = torch.cat(ib); vals = torch.cat(vals)
    if len(rows):
        uniq, inv = torch.unique(ia * n + ib, return_inverse=True)
        a, b = uniq // n, uniq % n
        q = torch.zeros(m, uniq.numel(), dtype=dt, device=device)
        q.index_put_((rows, inv), vals, accumulate=True)
    else:
        a = b = torch.zeros(0, dtype=torch.long, device=device)
        q = torch.zeros(m, 0, dtype=dt, device=device)
    ok = torch.ones(m, dtype=torch.bool, device=device)
    for p in range(x_check.shape[0]):
        x = x_check[p].to(device=device, dtype=dt)
        y = fn(x).detach()
        w = x[a] * x[b]
        rep = k + g @ x + q @ w
        scale = 1.0 + k.abs() + (g.abs() @ x.abs()) + (q.abs() @ w.abs())
        ok &= (rep - y).abs() <= 1e-9 * scale
    return k, g, a, b, q, ok


_POOL_CTX = None


def _root_obbt_refresh(rate, input_progress, time_left, reserve, auxiliary_seconds):
    """Use spare root budget, otherwise compare measured tightening per LP.

    Reserve the quoted original-variable sweeps only while their observed
    progress calls for another round. An unseen target gets a measurement.
    Low-return targets can instead replay a saved dual on the current LP.
    """
    return (rate is None or reserve <= 0 or time_left >= reserve + auxiliary_seconds
            or _probe_pays([rate, 1.0, *input_progress]))


def _probe_pays(progress, require_gain=True):
    """[probe gain, probe seconds, node gain, node seconds], with one initial measurement.

    Node gain is the certified improvement over the inherited bound; root uses
    the caller's floor (or its certified box bound). Probe gain is the additional
    certified objective bound after tightening and rebuilding the relaxation.
    Require a positive measured probe gain after the initial measurement, so
    equal zero rates do not buy unlimited probes, with or without auxiliary
    transfer. `require_gain=False` supports diagnostic policy comparisons only.
    """
    return progress[1] == 0 or ((not require_gain or progress[0] > 0)
                              and progress[0] * progress[3] >= progress[2] * progress[1])


def _pool_solve(box):
    """Worker: `LiftedLP.solve_bab_node` of one box (forked; numpy/scipy, and torch on one thread)."""
    torch.set_num_threads(1)        # the evaluator is torch: a forked child must not use the parent's thread pool
    lp, rh, eg, eh, cutoff = _POOL_CTX
    progress = box[4].copy() if len(box) > 4 else None
    registry = getattr(lp, 'certificate_registry', None)
    if registry is not None:
        from nlp_solver.domain_certificate import LeafCertificates
        certificates = LeafCertificates()
    else:
        certificates = None
    result = lp.solve_bab_node(box[0], box[1], rh, eg, eh, cutoff, _NODE_TIME, box[2],
                               cutoff_row=getattr(lp, 'node_cutoff', True),
                               progress=progress, inherited=box[3] if len(box) > 3 else -np.inf,
                               certificates=certificates)
    result = result if progress is None else (*result, np.array(progress) - box[4])
    if registry is not None:
        return (*result, certificates)
    return result


def _make_pool(lp, rh, eg, eh, cutoff, workers):
    """A fork pool whose workers inherit (lp, rows, cutoff); None if it cannot be created."""
    import multiprocessing as mp
    global _POOL_CTX
    _POOL_CTX = (lp, rh, eg, eh, cutoff)
    try:
        return mp.get_context("fork").Pool(int(workers))
    except Exception:  # noqa: BLE001
        return None


class LiftedLP:
    """Spec rows as lifted linear rows over z = (x, w); see the module docstring.

    obj: (k0, g0 (n,), q0 (K,)) or None without a lifted objective. With epigraphs,
    this is the quadratic part; the positive epigraph coefficients are appended.
    rows: (k (R,), g (R, n), q (R, K)) for the constraint spec rows `row_spec` (spec
    indices >= 1). obj_mag / row_mag: the same with absolute values (|C| |coefficients|),
    from which each box gets its rounding cushion (see `_cushions`).
    """

    def __init__(self, n, a, b, obj, obj_mag, rows, row_spec, row_mag, device, atoms=None):
        self.n = n
        self.atoms = atoms          # atom columns n .. nb (lifted_atoms), before the products
        self.nb = n + (0 if atoms is None else atoms.T)
        self.a, self.b = a, b
        self.K = int(a.numel())
        self.branch_K = self.K
        self.N = self.nb + self.K
        self.sq = a == b
        self.obj = obj
        self.obj_mag = obj_mag
        self.rows = rows
        self.row_spec = row_spec
        self.row_mag = row_mag
        self.device = device
        nb, ns = int((~self.sq).sum()), int(self.sq.sum())
        self.n_mc = 4 * nb + (1 + len(_SQ_TANGENTS)) * ns
        self.n_sq = ns
        # (product, input variable) incidence of the original products, for branching
        self.mono = _monomial_vars(a, b, n, atoms)
        self.qvars = torch.unique(self.mono[1]) if self.nb > n or self.K else a
        self._set_levels()
        self._np = None
        # Optional denominator-cleared rows include their root relaxed threshold in
        # their polynomial. NaN marks an ordinary row with a runtime RHS.
        self.cleared_rhs = None
        self.row_error = None
        self.obj_error = None
        self.rlt = None
        self.n_rlt = 0
        self.epigraph = []

    def _set_levels(self):
        self.levels = _levels(self.a, self.b, self.nb)
        self._levels_np = [t.cpu().numpy() for t in self.levels]

    def _row_rhs(self, rhs_rows):
        rhs = rhs_rows[:, self.row_spec - 1]
        if self.cleared_rhs is not None:
            root = self.cleared_rhs.to(rhs)
            # A later looser threshold cannot use a polynomial cleared at the root
            # threshold. Drop that cut. Tighter thresholds can safely reuse it.
            rhs = torch.where(torch.isnan(root), rhs,
                              torch.where(rhs <= root, 0.0, float('inf')))
        return rhs

    def add_rlt(self, rhs_rows, extra_G=None, extra_h=None, max_rows=1200, max_columns=7000,
                full_budget=False):
        """Add equality x variable rows, recognising EXACT opposite linear coefficients.
        Equality specs may already encode a tolerance slab in their constants or RHS;
        use its midpoint b and preserve its full width. This algebra also holds for
        any pair of opposite inequalities. `extra_G/h` are the base linear rows;
        subsequent assemblies must put these last among their extra rows.

        RLT is all or nothing: when the selected rows exceed max_rows, or the lifted LP would
        exceed max_columns, no RLT rows are added (pooling_foulds3stp: 383 s vs 86 s per node
        solve without presolve, then). With warm-started node LPs the caps 1200 / 7000 admit
        wastewater13m1 (1080 rows, 5580 columns: root 1017.2 -> 1506.5, closed in 5 nodes) and ex8_3_8
        (690 rows: root -10 -> -5.70). The pairs are chosen by one rule, reduced RLT: an
        equality is multiplied by x_i iff x_i x_j is an original product for some j in its support.

        If g x = b is relaxed to -eps_minus <= g x - b <= eps_plus, then
        sum_j g_j w_ij - b x_i = x_i (g x - b) lies in the interval product
        [l_i, u_i] * [-eps_minus, eps_plus]. In particular symmetric tolerance eps
        requires eps * max(|l_i|, |u_i|) on BOTH sides, including negative multipliers.
        For a multiplier of known sign, multiplying the relaxed inequalities directly
        retains eps * x_i as a linear term: stronger than the interval-product bound,
        and still satisfied by every relaxed-feasible point.
        The eps values below come from the actual relaxed RHS and source cushions at
        each box, so tighter heuristic rows and node boxes also have the right semantics.
        """
        import scipy.sparse as sp
        if self.rlt is not None:
            raise ValueError("RLT rows already configured")
        if max_rows < 2:
            return
        R = 0 if self.rows is None else len(self.row_spec)
        gs, hs, linear = [], [], []
        if R:
            k, g, q = self.rows
            gs.append(g[:, :self.n].detach().cpu().double().numpy())
            hs.append((rhs_rows.reshape(-1)[self.row_spec - 1] - k).detach().cpu().numpy())
            linear.extend(((q == 0).all(1) & (g[:, self.n:] == 0).all(1)).cpu().tolist())
        base = None
        if extra_G is not None:
            base = extra_G.reshape(-1, self.n).detach().cpu().double().numpy()
            gs.append(base)
            hs.append(extra_h.reshape(-1).detach().cpu().double().numpy())
            linear.extend([True] * len(base))
        if not gs:
            return
        g, h = np.concatenate(gs), np.concatenate(hs)
        seen, matched, eqs = {}, set(), []
        for r in range(len(h)):
            if not linear[r] or not np.isfinite(h[r]) or not np.any(g[r]):
                continue
            key = tuple(g[r])
            opposite = tuple(-g[r])
            if key in matched:
                continue
            if opposite in seen:
                eqs.append((seen.pop(opposite), r))
                matched.update([key, opposite])
            elif key not in seen:
                seen[key] = r
        if not eqs or not self.qvars.numel():
            return
        pairs = list(zip(self.a.cpu().tolist(), self.b.cpu().tolist()))
        # Reduced RLT (Liberti and Pantelides 2006): multiply an equality by x_i only if
        # x_i x_j is already a product column for some j in its support, so the rows mostly
        # reuse existing products. Root bounds (McCormick OBBT + RLT), rows/new products:
        #   ex8_3_2: no RLT under the cap (full: 1674 rows) -> -0.58 with 578/945, 12 ms per LP
        #   wastewater11m1: 1024.8 -> 1491.5 with 280/805 (full: 4158/6097 gives 1491.7)
        #   wastewater14m1: 213.3 -> 437.5 with 320/610 (full: 2640/2515 gives 442.0)
        #   wastewater13m1: 1496.0 with 1080/5580 (greedy cover 3120/10770: 1017.2, or 1506.2
        #   with the balance-size tie-break, 4x the LP time); waterund22: 256.4 -> 295.6.
        # It loses on wastewater05m1 (119.45 with 216 rows against the cover's 119.88 with 456).
        partners = {}
        for i, j in pairs:
            if max(i, j) >= self.n:
                continue    # product columns of higher monomials: RLT stays over input products
            partners.setdefault(i, set()).add(j)
            partners.setdefault(j, set()).add(i)
        selected = sorted((i, e) for e, (p, _) in enumerate(eqs)
                          for i in set().union(*(partners.get(int(j), ()) for j in np.flatnonzero(g[p]))))
        if not selected:
            return
        if 2 * len(selected) > max_rows:
            print(f"[LiftedLP] RLT skipped: {2 * len(selected)} rows exceed the cap {max_rows}")
            return
        pair_col = {p: self.nb + k for k, p in enumerate(pairs)}
        rr, cc, vv, pos, neg, mult, bs = [], [], [], [], [], [], []
        for i, e in selected:
            p, m = eqs[e]
            r = len(pos)
            for j in np.flatnonzero(g[p]):
                pair = (min(i, int(j)), max(i, int(j)))
                if pair not in pair_col:
                    pair_col[pair] = self.nb + len(pairs)
                    pairs.append(pair)
                rr.append(r); cc.append(pair_col[pair]); vv.append(g[p, j])
            midpoint = 0.5 * (h[p] - h[m])
            rr.append(r); cc.append(i); vv.append(-midpoint)
            pos.append(p); neg.append(m); mult.append(i); bs.append(midpoint)
        if self.nb + len(pairs) > max_columns:
            print(f"[LiftedLP] RLT skipped: {self.nb + len(pairs)} lifted columns "
                  f"exceed the cap {max_columns}")
            return
        old_K = self.K
        H = sp.csr_matrix((vv, (rr, cc)),
                          shape=(len(pos), self.nb + len(pairs) + len(self.epigraph)))
        H.eliminate_zeros()
        if full_budget:
            new_sq = sum(i == j for i, j in pairs[old_K:])
            new_bilinear = len(pairs) - old_K - new_sq
            square_rows = (1 + len(_SQ_TANGENTS)) * new_sq
            envelope_rows = 4 * new_bilinear + square_rows
            # The multiplier column can occur even if the equality midpoint
            # is zero. Count its structural slot on both sides of every RLT row.
            structure = abs(H) + sp.csr_matrix(
                (np.ones(len(mult)), (np.arange(len(mult)), mult)), shape=H.shape)
            cost = dict(rows=2 * len(pos) + envelope_rows,
                        columns=len(pairs) - old_K,
                        nnz=2 * structure.nnz + 12 * new_bilinear + 2 * square_rows)
            self.rlt_budget = cost
            if sum(cost.values()) > max_rows:
                print(f"[LiftedLP] RLT skipped: full generated cost {cost} "
                      f"exceeds the cap {max_rows}")
                return
        self.a = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=self.device)
        self.b = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=self.device)
        self.K, self.N = len(pairs), self.nb + len(pairs) + len(self.epigraph)
        self.sq = self.a == self.b
        self.n_sq = int(self.sq.sum())
        self.n_mc = 4 * (self.K - self.n_sq) + (1 + len(_SQ_TANGENTS)) * self.n_sq
        self.qvars = torch.unique(_monomial_vars(self.a, self.b, self.n, self.atoms)[1])
        self._set_levels()
        def pad(rep):
            return None if rep is None else (*rep[:2], torch.nn.functional.pad(rep[2], (0, self.K - old_K)))
        self.rows, self.row_mag = pad(self.rows), pad(self.row_mag)
        self.obj, self.obj_mag = pad(self.obj), pad(self.obj_mag)
        self.row_error, self.obj_error = pad(self.row_error), pad(self.obj_error)
        self.rlt = (H, np.array(pos), np.array(neg), np.array(mult), np.array(bs), base,
                    self.a.cpu().numpy(), self.b.cpu().numpy())
        self.n_rlt = 2 * len(pos)
        self._np = None
        print(f"[LiftedLP] RLT: {len(eqs)} linear equalities, {len(set(mult))} multipliers, "
              f"{self.n_rlt} rows, {self.K - old_K} new products (cap {max_rows})")

    def _rlt_block(self, l, u, source_h, extra_G, extra_h):
        """Sparse RLT block and RHS for one box; numpy only, including in fork workers."""
        import scipy.sparse as sp
        H, pos, neg, mult, bs, base, a, b = self.rlt
        if base is not None:
            valid = extra_G is not None and extra_G.shape[0] >= len(base) \
                and np.array_equal(extra_G[-len(base):], base)
            source_h = np.concatenate([source_h, extra_h[-len(base):] if valid
                                       else np.full(len(base), np.inf)])
        el, eu = -source_h[neg] - bs, source_h[pos] - bs
        # Inactive source rows produce inactive cuts (avoid 0 * inf NaNs).
        active = np.isfinite(el) & np.isfinite(eu)
        el, eu = np.where(active, el, 0.0), np.where(active, eu, 0.0)
        products = np.stack([l[mult] * el, l[mult] * eu, u[mult] * el, u[mult] * eu])
        M = np.maximum(np.abs(l), np.abs(u))
        lz, uz = _column_boxes(l, u, a, b, self._levels_np, np, self.atoms)
        zmag = np.concatenate([np.maximum(np.abs(lz), np.abs(uz)), np.zeros(len(self.epigraph))])
        cushion = _CUSHION * (1.0 + abs(H) @ zmag
                              + np.maximum(np.abs(el), np.abs(eu)) * M[mult])
        hp, hm = products.max(0) + cushion, -products.min(0) + cushion
        positive, negative = l[mult] >= 0, u[mult] <= 0
        fixed_sign = positive | negative
        # q - b*x_i <= eu*x_i and -(q - b*x_i) <= -el*x_i for x_i >= 0;
        # the endpoints exchange roles for x_i <= 0. This multiplies the actual
        # relaxed source inequalities by a nonnegative factor, including their
        # rounding cushions. For multipliers spanning zero use tol * max(|l|, |u|).
        pc = np.where(positive, -eu, np.where(negative, -el, 0.0))
        mc = np.where(positive, el, np.where(negative, eu, 0.0))
        idx = np.arange(len(mult))
        GP = H + sp.csr_matrix((pc, (idx, mult)), shape=H.shape)
        GM = -H + sp.csr_matrix((mc, (idx, mult)), shape=H.shape)
        hp[fixed_sign] = hm[fixed_sign] = cushion[fixed_sign]
        hp[~active] = hm[~active] = np.inf
        return sp.vstack([GP, GM]).tocsr(), np.concatenate([hp, hm])

    def _cushions(self, l, u):
        """Rounding cushions for boxes l, u: (rows (B, R) or None, objective (B,) or
        None), _CUSHION (1 + |k| + |g| . M + |q| . MW) with M = max(|l|, |u|)."""
        lz, uz = _column_boxes(l, u, self.a, self.b, self.levels, torch, self.atoms)
        Mz = torch.maximum(lz.abs(), uz.abs())
        M, MW = Mz[:, :self.nb], Mz[:, self.nb:]
        rows = obj = None
        if self.row_mag is not None:
            ka, ga, qa = (t.to(l) for t in self.row_mag)
            rows = _CUSHION * (1.0 + ka.unsqueeze(0) + M @ ga.T + MW @ qa.T)
        if self.obj_mag is not None:
            ka, ga, qa = self.obj_mag
            obj = _CUSHION * (1.0 + float(ka) + M @ ga.to(l) + MW @ qa.to(l))
        if self.row_error is not None:
            ke, ge, qe = (t.to(l) for t in self.row_error)
            error = ke.unsqueeze(0) + M @ ge.T + MW @ qe.T
            rows = rows + torch.nextafter(error, torch.full_like(error, float('inf')))
        if self.obj_error is not None:
            ke, ge, qe = self.obj_error
            error = float(ke) + M @ ge.to(l) + MW @ qe.to(l)
            obj = obj + torch.nextafter(error, torch.full_like(error, float('inf')))
        return rows, obj

    def n_rows(self, n_extra=0):
        return ((0 if self.rows is None else int(self.row_spec.numel())) + self.n_mc
                + self.n_rlt + sum(len(t.points()) for t in self.epigraph) + n_extra
                + (0 if self.atoms is None else self.atoms.n_rows))

    def _epigraph_data(self, lz, uz):
        """Initial tangent rows, finite t bounds, and positive objective weights."""
        T = len(self.epigraph)
        tl, tu, weights = np.empty(T), np.empty(T), np.empty(T)
        rows, rhs = [], []
        for j, term in enumerate(self.epigraph):
            tl[j], tu[j] = term.bounds()
            weights[j] = term.weight
        lower, upper = np.concatenate([lz, tl]), np.concatenate([uz, tu])
        for j, term in enumerate(self.epigraph):
            for point in term.points():
                row, bound = term.cut(float(point), lower, upper, self.nb+self.K+j)
                rows.append(row); rhs.append(bound)
        # The exact positive coefficient can differ from its stored float.
        error = sum(t.weight_error * max(abs(tl[j]), abs(tu[j]))
                    for j, t in enumerate(self.epigraph))
        return np.stack(rows), np.array(rhs), lower, upper, weights, np.nextafter(error, np.inf)

    # ------------------------------------------------------------------ dense (batched)
    def assemble(self, l, u, rhs_rows, extra_G=None, extra_h=None, obj_plane=None, cutoff=None):
        """Dense batched LP data (c, c0, G, h, lz, uz) for boxes l, u (B, n).

        rhs_rows (B, S-1): relaxed thresholds of the constraint spec rows (index s-1 for
        spec s). extra_G/extra_h (B, E, n)/(B, E): further valid x-space rows.
        obj_plane: (a (B, n), b (B,)) lower plane of the objective, used when the
        objective is not lifted. cutoff (B,): lifted objective <= cutoff.
        """
        B, n = l.shape
        dt, dev = l.dtype, l.device
        blocks_G, blocks_h = [], []
        cush_rows, cush_obj = self._cushions(l, u)
        lz, uz = _column_boxes(l, u, self.a, self.b, self.levels, torch, self.atoms)
        if self.rows is not None and self.row_spec.numel():
            kr, gr, qr = self.rows
            blocks_G.append(torch.nn.functional.pad(torch.cat([gr, qr], 1),
                            (0, len(self.epigraph))).to(dt).unsqueeze(0).expand(B, -1, -1))
            blocks_h.append(self._row_rhs(rhs_rows) - kr.to(dt) + cush_rows)
        if self.n_mc:
            G = torch.zeros(B, self.n_mc, self.N, dtype=dt, device=dev)
            h = torch.zeros(B, self.n_mc, dtype=dt, device=dev)
            ar = torch.arange(B, device=dev).unsqueeze(1)
            for rows, wcol, sw, xa, xb, rh in _mccormick_terms(lz, uz, self.a, self.b, self.sq,
                                                               self.nb, torch):
                rr = rows.unsqueeze(0).expand(B, -1)
                G[ar, rr, wcol.unsqueeze(0).expand(B, -1)] = sw
                G[ar, rr, xa[0].unsqueeze(0).expand(B, -1)] += xa[1]
                if xb is not None:
                    G[ar, rr, xb[0].unsqueeze(0).expand(B, -1)] += xb[1]
                h[ar, rr] = rh
            blocks_G.append(G)
            blocks_h.append(h)
        if self.atoms is not None:
            G, h = self.atoms.dense(lz, uz, self.N)
            blocks_G.append(G); blocks_h.append(h)
        if extra_G is not None and extra_G.shape[1] > 0:
            blocks_G.append(torch.cat([extra_G, extra_G.new_zeros(B, extra_G.shape[1], self.N-n)], 2))
            blocks_h.append(extra_h)
        if self.rlt is not None:
            # RLT is stored sparsely; materialise only for selected PDHG boxes.
            source = blocks_h[0] if self.rows is not None else l.new_empty(B, 0)
            rg, rh = [], []
            for i in range(B):
                G_, h_ = self._rlt_block(l[i].cpu().double().numpy(), u[i].cpu().double().numpy(),
                    source[i].cpu().double().numpy(),
                    None if extra_G is None else extra_G[i].cpu().double().numpy(),
                    None if extra_h is None else extra_h[i].cpu().double().numpy())
                rg.append(torch.as_tensor(G_.toarray(), dtype=dt, device=dev))
                rh.append(torch.as_tensor(h_, dtype=dt, device=dev))
            blocks_G.append(torch.stack(rg)); blocks_h.append(torch.stack(rh))
        if self.obj is not None:
            k0, g0, q0 = self.obj
            c = torch.cat([g0, q0]).to(dt).unsqueeze(0).expand(B, -1).contiguous()
            c0 = float(k0) - cush_obj
        else:
            a_o, b_o = obj_plane
            c = torch.cat([a_o, a_o.new_zeros(B, self.nb - n + self.K)], 1)
            c0 = b_o
        if self.epigraph:
            data = [self._epigraph_data(lz[i].cpu().double().numpy(), uz[i].cpu().double().numpy())
                    for i in range(B)]
            tensor = lambda j: torch.as_tensor(np.stack([d[j] for d in data]), dtype=dt, device=dev)
            blocks_G.append(tensor(0)); blocks_h.append(tensor(1))
            lz, uz = tensor(2), tensor(3)
            c = torch.cat([c, tensor(4)], 1)
            c0 = c0 - tensor(5)
            if obj_plane is not None:
                # Preserve the existing global objective lower plane alongside
                # the epigraph. At t=phi(q), f-L <= 2*(k_float-c0), including
                # the polynomial and positive-weight conversion enclosures.
                ao, bo = obj_plane
                row = -c.clone()
                row[:, :n] += ao
                M = torch.maximum(lz.abs(), uz.abs())
                margin = _CUSHION * (1 + c0.abs() + bo.abs()
                                    + (ao.abs()*M[:, :n]).sum(1) + (c.abs()*M).sum(1))
                blocks_G.append(row.unsqueeze(1))
                blocks_h.append((2*float(self.obj[0])-c0-bo+margin).unsqueeze(1))
        if cutoff is not None:
            blocks_G.append(c.unsqueeze(1))
            blocks_h.append((cutoff - c0).unsqueeze(1))
        G, h = torch.cat(blocks_G, 1), torch.cat(blocks_h, 1)
        if self.rlt is not None:
            scale = G.abs().amax(-1).clamp(min=1.0)
            G, h = G / scale.unsqueeze(-1), h / scale
        return c, c0, G, h, lz, uz

    # ------------------------------------------------------------------ sparse (one box)
    def sparse_lp(self, l, u, rhs_rows, extra_G=None, extra_h=None, obj_plane=None,
                  cutoff=None):
        """(c, c0, G csr, h, lz, uz) as numpy/scipy for ONE box l, u (n,) (numpy arrays):
        the LP of `assemble`, without its inactive (non-finite) rows."""
        import scipy.sparse as sp
        n, nb, K, N = self.n, self.nb, self.K, self.N
        if self._np is None:
            fun = None
            if self.rows is not None and self.row_spec.numel():
                kr, gr, qr = self.rows
                fun = (sp.csr_matrix(torch.nn.functional.pad(torch.cat([gr, qr], 1),
                        (0, len(self.epigraph))).cpu().numpy()), kr.cpu().numpy(),
                       None, self.row_spec.cpu().numpy() - 1)
            obj = None
            if self.obj is not None:
                obj = (torch.cat([self.obj[1], self.obj[2]]).cpu().numpy(), float(self.obj[0]))
            rmag = None if self.row_mag is None else \
                tuple(t.detach().cpu().double().numpy() for t in self.row_mag)
            omag = None if self.obj_mag is None else \
                (float(self.obj_mag[0]), self.obj_mag[1].detach().cpu().double().numpy(),
                 self.obj_mag[2].detach().cpu().double().numpy())
            self._np = (self.a.cpu().numpy(), self.b.cpu().numpy(), self.sq.cpu().numpy(),
                        fun, obj, rmag, omag,
                        None if self.cleared_rhs is None else self.cleared_rhs.cpu().numpy(),
                        None if self.row_error is None else
                        tuple(t.cpu().numpy() for t in self.row_error),
                        None if self.obj_error is None else
                        tuple(torch.as_tensor(t, dtype=torch.float64).cpu().numpy()
                              for t in self.obj_error))
        # numpy only from here on: safe in forked worker processes (no CUDA calls).
        a, b, sq, fun, obj, rmag, omag, cleared_rhs, rerr, oerr = self._np
        lz, uz = _column_boxes(l, u, a, b, self._levels_np, np, self.atoms)
        Mz = np.maximum(np.abs(lz), np.abs(uz))
        M, MW = Mz[:nb], Mz[nb:]
        blocks, hs = [], []
        if fun is not None:
            cr = _CUSHION * (1.0 + rmag[0] + rmag[1] @ M + rmag[2] @ MW)
            if rerr is not None:
                cr += np.nextafter(rerr[0] + rerr[1] @ M + rerr[2] @ MW, np.inf)
            rh = rhs_rows[fun[3]]
            if cleared_rhs is not None:
                rh = np.where(np.isnan(cleared_rhs), rh,
                              np.where(rh <= cleared_rhs, 0.0, np.inf))
            blocks.append(fun[0])
            hs.append(rh - fun[1] + cr)
        if self.n_mc:
            rr, cc, vv = [], [], []
            hm = np.zeros(self.n_mc)
            for rows, wcol, sw, xa, xb, rh in _mccormick_terms(lz, uz, a, b, sq, nb, np):
                rr += [rows, rows]
                cc += [wcol, xa[0]]
                vv += [np.full(len(rows), sw), xa[1]]
                if xb is not None:
                    rr.append(rows); cc.append(xb[0]); vv.append(xb[1])
                hm[rows] = rh
            blocks.append(sp.csr_matrix(
                (np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                shape=(self.n_mc, N)))
            hs.append(hm)
        if self.atoms is not None:
            G_, h_ = self.atoms.sparse(lz, uz, N)
            blocks.append(G_); hs.append(h_)
        if extra_G is not None and extra_G.shape[0] > 0:
            blocks.append(sp.hstack([sp.csr_matrix(extra_G),
                                     sp.csr_matrix((extra_G.shape[0], N-n))]))
            hs.append(extra_h)
        if self.rlt is not None:
            source = hs[0] if fun is not None else np.empty(0)
            rg, rh = self._rlt_block(l, u, source, extra_G, extra_h)
            blocks.append(rg); hs.append(rh)
        if obj is not None:
            co = _CUSHION * (1.0 + omag[0] + float(omag[1] @ M) + float(omag[2] @ MW))
            if oerr is not None:
                co += np.nextafter(oerr[0] + oerr[1] @ M + oerr[2] @ MW, np.inf)
            c, c0 = obj[0], obj[1] - co
        elif obj_plane is not None:
            c, c0 = np.concatenate([obj_plane[0], np.zeros(nb - n + K)]), float(obj_plane[1])
        else:       # no objective at all (OBBT without a cutoff)
            assert cutoff is None
            c, c0 = np.zeros(N), 0.0
        if self.epigraph:
            epi_G, epi_h, lz, uz, weights, error = self._epigraph_data(lz, uz)
            blocks.append(sp.csr_matrix(epi_G)); hs.append(epi_h)
            c, c0 = np.concatenate([c, weights]), c0 - error
            if obj_plane is not None:
                ao, bo = obj_plane
                row = -c.copy()
                row[:n] += ao
                Mz = np.maximum(abs(lz), abs(uz))
                margin = _CUSHION * (1 + abs(c0) + abs(bo) + abs(ao) @ Mz[:n] + abs(c) @ Mz)
                blocks.append(sp.csr_matrix(row.reshape(1, -1)))
                hs.append(np.array([2*obj[1]-c0-bo+margin]))
        if cutoff is not None:
            blocks.append(sp.csr_matrix(c.reshape(1, -1)))
            hs.append(np.array([cutoff - c0]))
        G = sp.vstack(blocks).tocsr()
        h = np.concatenate(hs)
        act = np.isfinite(h)
        rows = np.flatnonzero(act)
        if not act.all():
            G, h = G[act], h[act]
        if self.rlt is not None:
            scale = np.maximum(1.0, abs(G).max(axis=1).toarray().reshape(-1))
            G, h = sp.diags(1.0 / scale) @ G, h / scale
        if self.epigraph:
            G._epigraph = (self.epigraph, self.nb+self.K)
        G._rows = (rows, len(act))      # which of the node-independent rows G holds (warm starts)
        return c, c0, G, h, lz, uz

    @staticmethod
    def solve_certified(c, c0, G, h, lz, uz, return_x=False, presolve=True,
                        retry=False, time_limit=float('inf'), max_iter=None, warm=None,
                        certificate_sink=None, prove_empty=False):
        """HiGHS proposes multipliers; returns their weak-duality bound, or None (and, with
        return_x, HiGHS's primal point or None). `warm`: see _highs_lp. With prove_empty an
        LP HiGHS calls infeasible gets +inf when a Farkas ray or the phase-I multipliers
        prove its rows empty over the box (as solve_node)."""
        t0 = time.time()
        options = {'presolve': presolve, 'retry': retry}
        if warm is not None:
            options['warm'] = warm
        if max_iter is not None:
            options['maxiter'] = max_iter
        if np.isfinite(time_limit):
            options['time_limit'] = max(0.0, time_limit)
        try:
            res, G, h = _epigraph_proposal(c, G, h, lz, uz, options)
        except Exception:  # noqa: BLE001 - numerical trouble: no bound
            res = None
        if prove_empty and res is not None and res.status == 2:
            left = time_limit - (time.time() - t0)
            phase1 = {k: v for k, v in options.items() if k != 'warm'}
            if np.isfinite(time_limit):
                phase1['time_limit'] = max(0.0, left)
            if _ray_proves_empty(res, G, h, lz, uz) or \
                    (left > 0 and _phase1_proves_empty(G, h, lz, uz, phase1)):
                return (np.inf, None) if return_x else np.inf
        if res is None or res.status != 0 or getattr(res, 'ineqlin', None) is None:
            return (None, None) if return_x else None
        y = np.maximum(-res.ineqlin.marginals, 0.0)
        if certificate_sink is not None:
            certificate_sink((c, c0, G, h, lz, uz), y)
        bnd = _dual_value_sparse(c, c0, G, h, lz, uz, y)
        return (bnd, res.x) if return_x else bnd

    def _residuals(self, z, xp):
        """(..., T + branch_K): |t_j - phi(arg_j)| of the atoms, then |w_k - z_a z_b| of the
        original products, at lifted points z (..., N), matching the definitions of `mono`."""
        absf = np.abs if xp is np else torch.abs
        a, b = self.a[:self.branch_K], self.b[:self.branch_K]
        if xp is np:
            a, b = a.cpu().numpy(), b.cpu().numpy()
        else:
            a, b = a.to(z.device), b.to(z.device)
        viol = absf(z[..., self.nb:self.nb + self.branch_K] - z[..., a] * z[..., b])
        if self.atoms is None:
            return viol
        cat = np.concatenate if xp is np else torch.cat
        return cat([self.atoms.residual(z, xp), viol], -1)

    def violation_score(self, z, rel_width=None):
        """Branching score (B, n) from lifted points z (B, N): for each variable, the sum
        of the residuals of the atoms and original products it occurs in (see _residuals).
        RLT auxiliary products can have huge residuals without affecting the original
        objective or constraints.

        With rel_width (B, n), the width relative to the root box, only variables at least
        _BRANCH_MIN_WIDTH_RATIO as wide as the widest compete: the violation grows with a
        variable's magnitude, and otherwise the large-valued variables are split down to
        slivers while the others stay whole (ex7_2_3, hs106)."""
        viol = self._residuals(z, torch)
        score = torch.zeros_like(z[:, :self.n])
        score.index_add_(1, self.mono[1].to(z.device), viol[:, self.mono[0].to(z.device)])
        if rel_width is not None:
            rel = rel_width.to(score)
            score = score * (rel >= _BRANCH_MIN_WIDTH_RATIO * rel.amax(1, keepdim=True))
        return score

    # ------------------------------------------------------------------ bounds
    @torch.no_grad()
    def bound(self, l, u, rhs_rows, extra_G=None, extra_h=None, obj_plane=None,
              pdhg_iters=500, exact_max=16, y0=None, max_elems=25_000_000, prev=None,
              return_points=False, exact_time=float("inf"), deadline=None):
        """Certified lower bound (B,) on the objective row over each box (-inf where not
        computed), and the mean PDHG multipliers (m,) (or None); with return_points also
        the LP points (B, N) (NaN where none: PDHG's last iterate, or HiGHS's solution).

        PDHG runs on the boxes with the lowest `prev` bound that fit in `max_elems` dense
        coefficients; HiGHS re-solves the `exact_max` boxes with the lowest bound.
        """
        from nlp_solver.dual_ascent import lp_bound_pdhg
        B = l.shape[0]
        m = self.n_rows(0 if extra_G is None else extra_G.shape[1])
        m += int(bool(self.epigraph) and obj_plane is not None)
        out = torch.full((B,), -torch.inf, dtype=l.dtype, device=l.device)
        Z = torch.full((B, self.N), torch.nan, dtype=l.dtype, device=l.device) \
            if return_points else None
        key = prev if prev is not None else torch.zeros(B, dtype=l.dtype, device=l.device)
        order = torch.argsort(key)
        y_mean = None
        # Added products can make even one dense RLT LP exceed the memory budget.
        # In that case use the sparse exact path; uncomputed boxes retain -inf.
        if pdhg_iters > 0 and (self.rlt is None or m * self.N <= max_elems) and \
                (deadline is None or time.time() < deadline):
            sel = order[:max(1, max_elems // max(m * self.N, 1))]
            sub = (lambda t: None if t is None else t[sel])
            op = None if obj_plane is None else (obj_plane[0][sel], obj_plane[1][sel])
            c, c0, G, h, lz, uz = self.assemble(l[sel], u[sel], rhs_rows[sel], sub(extra_G),
                                                sub(extra_h), op)
            y0s = None
            if y0 is not None and y0.numel() == G.shape[1]:
                y0s = y0.to(G).reshape(1, -1).expand(G.shape[0], -1)
            res = lp_bound_pdhg(c, c0, G, h, lz, uz, iters=pdhg_iters, y0=y0s,
                                max_elems=max_elems, return_primal=return_points, deadline=deadline)
            out[sel] = res[0]
            y_mean = res[1].mean(0)
            if return_points:
                Z[sel] = res[2]
            del c, c0, G, h, lz, uz, res
        if exact_max > 0:
            rank = torch.where(torch.isfinite(out), out, key)
            t_ex = time.time()
            if deadline is not None:
                exact_time = min(exact_time, max(0.0, deadline - t_ex))
            for i in torch.argsort(rank)[:exact_max].tolist():
                if time.time() - t_ex >= exact_time:
                    break           # the rest keep their PDHG bound
                eg = None if extra_G is None else extra_G[i].cpu().double().numpy()
                eh = None if extra_h is None else extra_h[i].cpu().double().numpy()
                op = None if obj_plane is None else (obj_plane[0][i].cpu().double().numpy(),
                                                     float(obj_plane[1][i]))
                lp = self.sparse_lp(l[i].cpu().double().numpy(), u[i].cpu().double().numpy(),
                                    rhs_rows[i].cpu().double().numpy(), eg, eh, op)
                left = exact_time - (time.time() - t_ex)
                if left <= 0:
                    break
                b_ex, x_ex = self.solve_certified(*lp, return_x=True,
                                                 retry=self.rlt is not None,
                                                 time_limit=left, prove_empty=True)
                if b_ex is not None and b_ex > float(out[i]):
                    out[i] = b_ex
                if return_points and x_ex is not None:
                    Z[i] = torch.as_tensor(x_ex, dtype=Z.dtype, device=Z.device)
        if return_points:
            return out, y_mean, Z
        return out, y_mean

    @staticmethod
    def solve_node(c, c0, G, h, lz, uz, presolve=True, retry=False, time_limit=float('inf'),
                   max_iter=None, warm=None, certificate_sink=None):
        """(certified bound, LP point) for one box. HiGHS proposes multipliers. An LP HiGHS
        calls infeasible is certified by the multipliers of min{t : G z - t <= h}: if
        min over the box of y^T (G z - h) > 0 no point satisfies the rows, and the bound
        is +inf. Anything unproved gives (-inf, None). `warm`: see _highs_lp (not for the
        phase-I solve)."""
        t0 = time.time()
        options = {'presolve': presolve, 'retry': retry}
        if warm is not None:
            options['warm'] = warm
        if max_iter is not None:
            options['maxiter'] = max_iter
        if np.isfinite(time_limit):
            options['time_limit'] = max(0.0, time_limit)
        try:
            res, G, h = _epigraph_proposal(c, G, h, lz, uz, options)
        except Exception:  # noqa: BLE001
            return -np.inf, None
        if res is None:
            return -np.inf, None
        if res.status == 0 and getattr(res, 'ineqlin', None) is not None:
            y = np.maximum(-res.ineqlin.marginals, 0.0)
            if certificate_sink is not None:
                certificate_sink((c, c0, G, h, lz, uz), y)
            return _dual_value_sparse(c, c0, G, h, lz, uz, y), res.x
        if res.status != 2:
            return -np.inf, None
        if _ray_proves_empty(res, G, h, lz, uz):
            return np.inf, None
        if presolve and not retry:
            # Presolve can falsely reject a nearly fixed feasible McCormick
            # system, especially with a tight objective cutoff. RLT already
            # retries in _highs_proposal; ordinary LPs need the same recovery.
            # Disabling presolve also enables the existing box equilibration.
            left = time_limit - (time.time() - t0)
            if left > 0:
                return LiftedLP.solve_node(
                    c, c0, G, h, lz, uz, presolve=False, time_limit=left,
                    max_iter=max_iter, warm=warm, certificate_sink=certificate_sink)
        if np.isfinite(time_limit):
            left = time_limit - (time.time() - t0)
            if left <= 0:
                return -np.inf, None
            options['time_limit'] = left
        options.pop('warm', None)
        return (np.inf, None) if _phase1_proves_empty(G, h, lz, uz, options) else (-np.inf, None)

    def _variable_violation(self, z):
        """(n,) numpy: for each input, the sum of the residuals (see _residuals) of the atoms
        and original products it occurs in, at the lifted point z (N,)."""
        if getattr(self, '_mono_np', None) is None:
            self._mono_np = (self.mono[0].cpu().numpy(), self.mono[1].cpu().numpy())
        mono_k, mono_v = self._mono_np
        score = np.zeros(self.n)
        np.add.at(score, mono_v, self._residuals(z, np)[mono_k])
        return score

    def solve_bab_node(self, lo, hi, rh, eg, eh, cutoff, limit=float('inf'), basis=None,
                       probes=None, cutoff_row=True, progress=None, inherited=-np.inf,
                       certificates=None, probe=True):
        """(bound, LP point, lo, hi, basis) of the node box lo, hi (n,) numpy; the returned box
        holds every point of the node that satisfies the rows with objective <= cutoff.

        With cutoff_row the LP carries the objective <= cutoff row.
        Propagation over its rows tightens the box,
        the envelopes are rebuilt on it (at most _NODE_PASSES times), the LP is solved
        (solve_node's weak-duality bound), and OBBT probes the _NODE_OBBT_VARS variables with
        the largest violation x width at the LP point: min and max of x_j over this LP are
        certified like the bound and rounded outward. Children inherit the box.

        Bound +inf: the node holds no such point, proved by multipliers evaluated on the LP
        rows (one propagation row, or solve_node's), or by an OBBT interval that misses the
        box by more than rounding. `limit` seconds cap the LP and the probes.

        `basis` (a parent's, returned as the last item) warm-starts the LP and the probes: the
        child's LP differs in bounds and envelope rows only (see _highs_lp). Epigraph LPs
        change their rows during the solve and start cold.

        `probes` ([probed nodes, probes, paying probes], updated here) makes the probing adaptive:
        see _PROBE_NODES. None: always probe."""
        started = time.time()
        deadline = started + limit
        cut = cutoff if cutoff_row and np.isfinite(cutoff) else None
        lo, hi = lo.copy(), hi.copy()
        for _ in range(_NODE_PASSES):
            c, c0, G, h, lz, uz = self.sparse_lp(lo, hi, rh, eg, eh, cutoff=cut)
            if self.epigraph:       # the epigraph rows change during the solve
                break
            lz, uz, row = _fbbt(G, h, lz, uz)
            if row is not None and _certified_infeasible(G, h, lz, uz,
                                                         np.eye(1, G.shape[0], row)[0]):
                if progress is not None:
                    progress[3] += time.time() - started
                return np.inf, None, lo, hi, None
            slack = 1e-3 * (hi - lo)
            moved = np.any(lz[:self.n] > lo + slack) or np.any(uz[:self.n] < hi - slack)
            lo, hi = lz[:self.n].copy(), uz[:self.n].copy()
            if not moved:
                break
        def export(data, y):
            certificates.capture(lo, hi, data, y, cut)
        sink = export if certificates is not None else None
        retry = self.rlt is not None
        warm = None if self.epigraph else {'basis': basis}
        if progress is not None and not np.isfinite(inherited):
            inherited = _dual_value_sparse(c, c0, G, h, lz, uz, np.zeros(len(h)))
        bound, z = self.solve_node(c, c0, G, h, lz, uz, retry=retry,
                                   time_limit=max(0.0, deadline - time.time()),
                                   max_iter=_NODE_ITERS * sum(G.shape), warm=warm, certificate_sink=sink)
        if certificates is not None and not certificates.records and not np.isposinf(bound):
            # A timed-out proposal still hands off a complete (zero-dual)
            # certificate; the domain also keeps its inherited scalar maximum.
            export((c, c0, G, h, lz, uz), np.zeros(len(h)))
        basis = warm and warm['basis']
        if progress is not None:
            if np.isfinite(bound) and np.isfinite(inherited):
                progress[2] += max(0.0, bound - inherited)
            progress[3] += time.time() - started
        if not probe or z is None or not bound < cutoff or G.shape[0] > _NODE_OBBT_ROWS or \
                (progress is None and probes and probes[0] >= _PROBE_NODES
                 and probes[2] < _PROBE_PAY * probes[1]):
            return bound, z, lo, hi, basis
        if probes:
            probes[0] += 1
        score = self._variable_violation(z) * (hi - lo)
        for j in np.argsort(-score)[:_NODE_OBBT_VARS]:
            if progress is not None and not _probe_pays(progress):
                break
            if score[j] <= 0 or time.time() > deadline:
                break
            probe_started, before_bound = time.time(), bound
            e = np.zeros(G.shape[1])
            e[j] = 1.0
            ends, width = [], hi[j] - lo[j]
            for sign in (1.0, -1.0):
                left = deadline - time.time()
                if left <= 0:
                    ends.append(None)
                    continue
                b = self.solve_certified(sign * e, 0.0, G, h, lz, uz, retry=retry,
                                         time_limit=left,
                                         max_iter=_NODE_ITERS * sum(G.shape),
                                         warm=warm and {'basis': basis})
                ends.append(None if b is None else np.nextafter(sign * b, -sign * np.inf))
            new_lo, new_hi = ends
            if new_lo is not None and new_lo > hi[j] + 1e-9 * (1 + abs(hi[j])) or \
                    new_hi is not None and new_hi < lo[j] - 1e-9 * (1 + abs(lo[j])):
                if probes:
                    probes[1:] = probes[1] + 1, probes[2] + 1
                if progress is not None:
                    progress[1] += time.time() - probe_started
                return np.inf, None, lo, hi, None
            if new_lo is not None and lo[j] < new_lo <= hi[j]:
                lo[j] = lz[j] = new_lo
            if new_hi is not None and lo[j] <= new_hi < hi[j]:
                hi[j] = uz[j] = new_hi
            if probes:
                probes[1] += 1
                probes[2] += width - (hi[j] - lo[j]) > _PAY_SHRINK * width
            if progress is not None:
                # Width alone earns no credit. Pay for rebuilding and certifying
                # the objective relaxation before measuring the probe's gain.
                if hi[j] - lo[j] < width and time.time() < deadline:
                    c, c0, G, h, lz, uz = self.sparse_lp(lo, hi, rh, eg, eh, cutoff=cut)
                    new_bound, new_z = self.solve_node(
                        c, c0, G, h, lz, uz, retry=retry,
                        time_limit=max(0.0, deadline - time.time()),
                        max_iter=_NODE_ITERS * sum(G.shape), warm=warm, certificate_sink=sink)
                    bound = max(bound, new_bound)
                    if new_z is not None:
                        z = new_z
                    basis = warm and warm['basis']
                if np.isfinite(bound) and np.isfinite(before_bound):
                    progress[0] += max(0.0, bound - before_bound)
                progress[1] += time.time() - probe_started
                if bound >= cutoff:
                    break
        return bound, z, lo, hi, basis

    @torch.no_grad()
    @retained
    def branch_and_bound(self, l, u, rhs_rows, extra_G=None, extra_h=None, cutoff=None,
                         tol_abs=1e-5, tol_rel=1e-4, time_budget=10.0, max_nodes=100000,
                         frac=0.1, return_leaves=False, workers=1, plateau_time=2.0,
                         floor=-np.inf, resume=None, primal_fn=None, max_leaves=None,
                         node_cutoff=True, node_obbt_gain=False, checkpoint=None,
                         sliced=False, deadline=None, cutoff_fn=None):
        """Best-first B&B over the lifted LP alone on the box l, u (1, n): every node gets
        its certified bound and tightened box (solve_bab_node), the split variable is the one
        whose products are most violated at the node's LP point, split at that point (clamped
        to the middle 1 - 2 frac). Needs a lifted objective. Returns (D, nodes, closed): D is a
        certified lower bound on the objective over every point of the box that
        satisfies the lifted and extra rows with objective < cutoff (min of the open
        leaves' bounds, or the cutoff when none is left); closed: cutoff - D <= tol.
        With return_leaves, also (lo (L, n), hi (L, n), bound (L,), x (L, n), warm) of the open
        leaves: together they hold every such point, each with its certified bound; x is
        the leaf's LP point (NaN if none), a heuristic primal candidate only; warm lists each
        leaf's (full LP point, basis) for `resume`.
        resume: the leaves of an earlier run on this LP and rows with a cutoff at least as high.
        The search continues from those leaves of bound < cutoff instead of the root box:
        they hold every point below the old cutoff, so also every point below the new one.
        primal_fn(x (n,) numpy) -> objective of a verified feasible point, found by a local solve
        from x, that improved the caller's incumbent, else None; it is offered the best node's
        point at nodes _LOCAL_NODES[0], 2x, 4x, ..., and an improvement lowers the cutoff on the
        fly (the open nodes of bound >= it are dropped: they hold no point below a feasible
        point's objective). Only verified points enter, so D is unchanged in meaning.
        workers > 1 expands that many best nodes per step and solves their children in
        forked worker processes (numpy/scipy, torch on one thread); the certificate is unchanged."""
        import heapq
        assert self.obj is not None
        t0 = time.time()
        if sliced:
            time_budget = float("inf")
        self.node_cutoff = bool(node_cutoff)  # inherited by fork workers
        lo0 = l.reshape(-1).cpu().double().numpy().copy()
        hi0 = u.reshape(-1).cpu().double().numpy().copy()
        rh = rhs_rows.reshape(-1).cpu().double().numpy()
        eg = None if extra_G is None else extra_G.reshape(-1, self.n).cpu().double().numpy()
        eh = None if extra_h is None else extra_h.reshape(-1).cpu().double().numpy()
        P = float("inf") if cutoff is None else float(cutoff)
        tol = max(tol_abs, tol_rel * abs(P)) if P < float("inf") else 0.0
        n = self.n

        def offer(z):
            """Lower the cutoff to the objective of a verified point found from z (primal_fn);
            whether it dropped by more than the certification tolerance."""
            nonlocal P, tol, heap
            obj = None if primal_fn is None or z is None else primal_fn(z[:n])
            if obj is not None and obj < P:
                drop = P - obj
                P, tol = obj, max(tol_abs, tol_rel * abs(obj))
                print(f"[LiftedLP] local solve from node {nodes}: incumbent {obj:.12g}")
                heap = [e for e in heap if e[0] < P]
                heapq.heapify(heap)
                return drop > tol
            return False

        # see _PROBE_NODES (the pool's workers always probe). A continuation keeps the earlier call's
        # count: restarted, its first nodes (the lowest leaves, whose probes pay) re-enabled probing for
        # good: kall_diffcircles_10 475 nodes in 30 s with D 7.97, 6166 nodes and D 8.91 without.
        probes = getattr(self, "probe_count", [0, 0, 0]) if resume is not None else [0, 0, 0]
        self.probe_count = probes
        progress = ((getattr(self, 'probe_progress', None) or [0.0] * 4) if resume is not None
                    else [0.0] * 4) if node_obbt_gain else None
        self.probe_progress = progress
        registry = getattr(self, 'certificate_registry', None)

        def solve(lo, hi, basis=None, inherited=-np.inf):
            if checkpoint is not None:
                checkpoint()  # cancellation discards this uncommitted batch, never its caller's frontier
            certificates = registry.new_leaf() if registry is not None else None
            result = self.solve_bab_node(lo, hi, rh, eg, eh, P,
                                       max(1e-3, deadline - time.time()) if sliced else
                                       min(_NODE_TIME, max(1e-3, time_budget - (time.time() - t0))),
                                       basis, probes, cutoff_row=node_cutoff,
                                       progress=progress, inherited=inherited, certificates=certificates)
            return (*result, certificates) if registry is not None else result

        if resume is None:
            # also caches the numpy LP data before any fork; the propagated root box sets the
            # reference widths of the width guard
            result = solve(lo0, hi0, inherited=floor)
            bd, z, lo0, hi0, basis = result[:5]
            records = result[5:]  # empty on the flag-off path
            print(f"[LiftedLP] root bound: {bd:.12g}")
            root_bd = bd
            if not bd > floor - (P - floor):
                # Hopeless: the lifted root lies further below the BaB's root bound (`floor`) than
                # that lies below the incumbent (GoldPrice: 635 nodes in 20 s to D = -6535 with
                # P = 3, which the BaB solves in 13 s; cute_spanhyd: root -inf). A root merely level
                # with the BaB's is no reason to stop: st_e30 (-3 vs -3) closes in 43 nodes and
                # kall_congruentcircles_c52 in 1267, far sooner than the BaB alone.
                print(f"[LiftedLP] no lifted-LP B&B: its root bound is far below {floor:.12g}")
                max_nodes = 0
            heap = [(bd, 0, lo0, hi0, z, basis, *records)] if bd < P else []
        else:
            # wastewater11m1: the restart from the root box spent its 30 s on the same nodes again
            r_lo, r_hi, r_bd, _, r_warm = resume
            heap = [(r_bd[i], i, r_lo[i], r_hi[i], *r_warm[i]) for i in np.nonzero(r_bd < P)[0]]
            heapq.heapify(heap)
            root_bd = float(r_bd.min()) if len(r_bd) else P
            print(f"[LiftedLP] continuing from {len(heap)} of {len(r_bd)} open leaves")
        # The same certified open-cover potential applies to fresh and resumed
        # lifted trees; it never substitutes for the minimum-frontier bound.
        if sliced:
            from input_split.tree_progress import GapMass
            measure = GapMass(l, u, P)
            for e in heap:
                measure.add(e[2][None], e[3][None], [e[0]])
        stuck = []          # (bound, lo, hi) of leaves that cannot be split any further
        def frontier():
            return [e[6] for e in heap] + [s[3] for s in stuck]
        if registry is not None:
            registry.defer(frontier)
        cnt, nodes = len(heap) if resume is None else len(r_bd), 0
        checked = False
        pool = _make_pool(self, rh, eg, eh, P, workers) if workers > 1 else None

        def solve_many(boxes):
            nonlocal pool
            if pool is not None:
                try:
                    left = max(1.0, time_budget - (time.time() - t0)) + 30.0
                    jobs = [(*box, progress.copy()) for box in boxes] if progress is not None else boxes
                    results = pool.map_async(_pool_solve, jobs, chunksize=1).get(timeout=left)
                    if registry is not None:
                        records = [registry.new_leaf(result[-1]) for result in results]
                        results = [result[:-1] for result in results]
                    if progress is not None:
                        for result in results:
                            progress[:] = np.array(progress) + result[5]
                        results = [result[:5] for result in results]
                    if registry is not None:
                        results = [(*result, record) for result, record in zip(results, records)]
                    return results
                except Exception:  # noqa: BLE001 - a stuck or broken pool: go serial
                    pool.terminate()
                    pool = None
            return [solve(*box) for box in boxes]

        local_starts = primal_fn is not None
        best_cur, t_best, n_best = -np.inf, time.time(), 0
        try:
            while heap and nodes < max_nodes and time.time() - t0 < time_budget:
                if sliced:
                    D = min([P, heap[0][0]] + [s[0] for s in stuck])
                    z_first = heap[0][4]
                    if registry is not None:
                        registry.defer(frontier)
                    mass_snapshot = measure.snapshot()
                    yield Atom('lp_tree', None if z_first is None else z_first[:n].copy(), D,
                               progress=(P, None if mass_snapshot is None else mass_snapshot[1], D))
                    if time.time() >= deadline:
                        break
                    if cutoff_fn is not None:
                        old_P = P
                        P = min(P, cutoff_fn())
                        tol = max(tol_abs, tol_rel * abs(P)) if np.isfinite(P) else 0.0
                        heap = [e for e in heap if e[0] < P]
                        heapq.heapify(heap)
                        if P != old_P:
                            measure = GapMass(l, u, P)
                            for e in heap:
                                measure.add(e[2][None], e[3][None], [e[0]])
                            for e in stuck:
                                measure.add(e[1][None], e[2][None], [e[0]])
                        if not heap:
                            break
                if checkpoint is not None:
                    checkpoint()
                if max_leaves is not None and len(heap) + len(stuck) >= max_leaves:
                    break       # enough starting boxes for the BaB (see lifted_bound._HANDOFF_LEAVES)
                cur = min([heap[0][0]] + [s[0] for s in stuck])
                if P - cur <= tol:
                    break
                now = time.time()
                if not sliced and not checked and now - t0 > 0.25 * time_budget:
                    checked = True
                    if np.isfinite(root_bd) and np.isfinite(P) and \
                            cur - root_bd < 1e-3 * (P - root_bd):
                        break   # a quarter of the budget moved the bound by < 0.1% of the gap
                step = 1e-9 * (1.0 + abs(cur))
                if np.isfinite(P) and np.isfinite(cur):
                    step = max(step, 1e-3 * (P - cur))     # 0.1% of the remaining gap
                if cur > best_cur + step:
                    best_cur, t_best, n_best = cur, now, nodes
                elif not sliced and now - t_best > max(plateau_time, 0.15 * time_budget) \
                        and nodes - n_best > _PLATEAU_NODES:
                    break       # the bound is frozen (poor cutoff or unprovable leaves)
                # Expand the `workers` best nodes at once (one node when serial).
                parents, boxes, ancestors = [], [], []
                while heap and len(parents) < max(1, workers):
                    entry = heapq.heappop(heap)
                    bd, _, lo, hi, z, basis = entry[:6]
                    records = entry[6:]
                    nodes += 1
                    if local_starts and nodes & (nodes - 1) == 0 and _LOCAL_NODES[0] <= nodes <= _LOCAL_NODES[1]:
                        t_local = time.time()
                        local_starts = not offer(z)     # the best node's point
                        t0 += time.time() - t_local     # the budget is for the LPs
                        t_best += time.time() - t_local
                    width = hi - lo
                    rel = width / np.maximum(hi0 - lo0, 1e-300)
                    if z is not None:
                        score = self._variable_violation(z)
                        # Penalize slivers without excluding their nonlinear error.
                        # A nearly fixed root variable can keep rel.max() at one
                        # while all materially violated products are narrower.
                        # A hard mask then starves the variables that can raise D.
                        weight = np.minimum(1.0, rel / max(_BRANCH_MIN_WIDTH_RATIO * rel.max(), 1e-300))
                        score = score * (rel > 1e-9) * weight
                    else:
                        score = np.zeros(n)
                    if score.max() > 1e-12:
                        j = int(np.argmax(score))
                        v = float(np.clip(z[j], lo[j] + frac * width[j],
                                          hi[j] - frac * width[j]))
                    else:
                        j = int(np.argmax(rel))
                        v = 0.5 * (lo[j] + hi[j])
                    if not (lo[j] < v < hi[j]):
                        stuck.append((bd, lo, hi, *records))    # nothing left to split: keeps its bound
                        continue
                    hi1 = hi.copy()
                    hi1[j] = v
                    lo2 = lo.copy()
                    lo2[j] = v
                    if sliced:
                        measure.add(lo[None], hi[None], [bd], sign=-1)
                    parents.append(bd)
                    if registry is not None:
                        ancestors.append(records[0].last_id)
                    boxes += [(lo, hi1, basis, bd), (lo2, hi, basis, bd)] if node_obbt_gain \
                        else [(lo, hi1, basis), (lo2, hi, basis)]
                if not boxes:
                    continue
                for k, result in enumerate(solve_many(boxes)):
                    b2, z2, lo2, hi2, basis2 = result[:5]
                    records = result[5:]
                    if registry is not None:
                        records[0].ancestor_id = records[0].last_id = ancestors[k // 2]
                    b2 = max(b2, parents[k // 2])  # a child's region lies inside its parent's
                    if b2 < P:
                        cnt += 1
                        heapq.heappush(heap, (b2, cnt, lo2, hi2, z2, basis2, *records))
                        if sliced:
                            measure.add(lo2[None], hi2[None], [b2])
        finally:
            if pool is not None:
                pool.terminate()
            if registry is not None:
                registry.defer(frontier())
        D = min([P] + ([heap[0][0]] if heap else []) + [s[0] for s in stuck])
        if not return_leaves:
            return D, nodes, bool(P - D <= tol)
        nan = np.full(n, np.nan)
        leaves = [(e[0], e[2], e[3], nan if e[4] is None else e[4][:n]) for e in heap] + \
                 [(s[0], s[1], s[2], nan) for s in stuck]
        warm = [e[4:] for e in heap] + [(None, None, *s[3:]) for s in stuck]
        if leaves:
            L_bd = np.array([e[0] for e in leaves])
            L_lo = np.stack([e[1] for e in leaves])
            L_hi = np.stack([e[2] for e in leaves])
            L_x = np.stack([e[3] for e in leaves])
        else:
            L_bd, L_lo, L_hi, L_x = np.zeros(0), np.zeros((0, n)), np.zeros((0, n)), \
                np.zeros((0, n))
        return D, nodes, bool(P - D <= tol), (L_lo, L_hi, L_bd, L_x, warm)

    # ------------------------------------------------------------------ OBBT
    @torch.no_grad()
    def tighten_atoms(self, l, u, rhs_rows, extra_G=None, extra_h=None, rounds=16):
        """Root setup of the sin and cos atoms (lifted_atoms), over the lifted LP with its linear
        rows. (1) The certified range of each argument, cut into the atom's root range: the lines
        only see the interval range of an argument from the box, which ignores the rows
        (pgon_polygon: x_{i+1} - x_i in [-2 pi, 2 pi] by intervals, [0, 2 pi] by x_{i+1} >= x_i).
        (2) `rounds` cutting-plane rounds: each adds a tangent slope at the argument of the LP's
        point. pgon_polygon needs the slope at 2 pi / 20 for its bound (20 sin(pi / 10) / 2, not
        pi): 16 rounds bring the root bound within 2e-4. 2 LPs per atom (at most MAX_ATOMS) and
        `rounds` more."""
        from nlp_solver.lifted_atoms import SIN
        trig = [j for j, k in enumerate(self.atoms.kind) if k == SIN]
        if not trig:
            return
        f64 = lambda t: None if t is None else t.reshape(-1).cpu().double().numpy()
        args = (f64(l), f64(u), f64(rhs_rows), None if extra_G is None else
                extra_G.reshape(-1, self.n).cpu().double().numpy(), f64(extra_h))
        c, c0, G, h, lz, uz = self.sparse_lp(*args)
        for j in trig:
            cols, coefs, a0 = self.atoms.args[j]
            e = np.zeros(self.N)
            e[cols] = coefs
            lo = self.solve_certified(e, a0, G, h, lz, uz)
            hi = self.solve_certified(-e, -a0, G, h, lz, uz)
            if lo is not None and hi is not None:
                slack = 1e-12 * (1 + abs(lo) + abs(hi))
                self.atoms.restrict(j, lo - slack, -hi + slack)
        for _ in range(rounds):
            c, c0, G, h, lz, uz = self.sparse_lp(*args)
            z = self.solve_certified(c, c0, G, h, lz, uz, return_x=True)[1]
            if z is None:
                break
            self.atoms.add_points(z)

    @torch.no_grad()
    @retained
    def obbt(self, l, u, rhs_rows, extra_G=None, extra_h=None, obj_plane=None, cutoff=None,
             rounds=3, time_budget=10.0, min_shrink=0.01, verbose=True, checkpoint=None,
             sliced=False, deadline=None, cutoff_fn=None):
        """Certified tightening of the box l, u (1, n) over the lifted LP (with the
        objective row <= cutoff when given), for the variables of the products.
        Returns (l, u) shaped like the inputs."""
        t0 = time.time()
        # A budget/round limit is not a fixed point. Let the caller resume a
        # paying pass even when its objective cutoff has not changed.
        self.obbt_pending = False
        ln = l.reshape(-1).cpu().double().numpy().copy()
        un = u.reshape(-1).cpu().double().numpy().copy()
        rh = rhs_rows.reshape(-1).cpu().double().numpy()
        eg = None if extra_G is None else extra_G.reshape(-1, self.n).cpu().double().numpy()
        eh = None if extra_h is None else extra_h.reshape(-1).cpu().double().numpy()
        op = None if obj_plane is None else (obj_plane[0].reshape(-1).cpu().double().numpy(),
                                             float(obj_plane[1].reshape(-1)[0]))
        vars_ = self.qvars.tolist()
        input_vars = vars_.copy()
        transfer_aux = getattr(self, 'transfer_aux', False)
        aux_lo = aux_hi = None
        if transfer_aux:
            vars_ += list(range(self.n, self.nb + self.branch_K))
            # Quotes and nonnegative multipliers are proposals, not scoped
            # bounds: every replay certifies the CURRENT rows and column box.
            target_rates = getattr(self, '_obbt_target_rates', {})
            self._obbt_target_rates = target_rates
            target_duals = getattr(self, '_obbt_target_duals', {})
            self._obbt_target_duals = target_duals
            adaptive_work = getattr(self, '_obbt_adaptive_work', False)
        if sliced:
            time_budget = float("inf")
        n_lp = n_reuse = n_deferred = 0
        for rd in range(rounds):
            completed = True
            round_started = time.time() if transfer_aux else None
            # Auxiliary propagation can tighten inputs before their own LPs run.
            # Judge the requested input variables over the whole round, rather
            # than averaging LP-only changes over the additional aux columns.
            if transfer_aux:
                input_width = (un - ln)[input_vars].copy()
                gain_width = (un - ln).copy()
                progress = [0.0] * 4  # aux gain/work, input gain/work
            shrink = 0.0
            for i_j, j in enumerate(vars_):
                if checkpoint is not None:
                    checkpoint()
                if time.time() - t0 > time_budget or (sliced and time.time() >= deadline):
                    completed = False
                    break
                if transfer_aux:
                    count = min(i_j, len(input_vars))
                    if (not sliced and rd == 0 and count >= max(8, len(input_vars) // 4)
                            and time.time() - t0 > 1.0):
                        widths = (un - ln)[input_vars[:count]]
                        active = input_width[:count] > 0
                        early_shrink = float(np.sum(
                            (input_width[:count][active] - widths[active]) / input_width[:count][active]))
                        if early_shrink < 1e-3 * count:
                            break
                elif (not sliced and rd == 0 and i_j >= max(8, len(vars_) // 4) and time.time() - t0 > 1.0
                        and shrink < 1e-3 * i_j):
                    break       # a quarter of the first round tightened nothing: give up
                if not transfer_aux and un[j] - ln[j] <= 0:
                    continue
                if cutoff_fn is not None:
                    cutoff = cutoff_fn()
                c, c0, G, h, lz, uz = self.sparse_lp(ln, un, rh, eg, eh, op, cutoff)
                if transfer_aux:
                    if aux_lo is not None:
                        lz, uz = np.maximum(lz, aux_lo), np.minimum(uz, aux_hi)
                    lz, uz, _ = _fbbt(G, h, lz, uz)
                    if np.any(lz > uz):
                        break
                    aux_lo, aux_hi = lz.copy(), uz.copy()
                    before_width = (aux_hi - aux_lo)[:self.n].copy()
                    ln, un = aux_lo[:self.n].copy(), aux_hi[:self.n].copy()
                    rate = target_rates.get(j)
                    cached = target_duals.get(j)
                    if cached is not None and not all(
                            y is not None and len(y) == len(h) for y in cached):
                        # A changed row bank (e.g. RLT) needs a new class
                        # measurement; a failed proposal is not a saved dual.
                        rate = cached = None
                    if j >= self.n:
                        admission_rate = rate
                        if admission_rate is None and progress[1] > 0:
                            admission_rate = progress[0] / progress[1]
                        if j == self.n:
                            input_seconds = time.time() - round_started
                            active = input_width > 0
                            input_mean = float(np.mean(
                                (input_width[active] - (un - ln)[input_vars][active])
                                / input_width[active])) if active.any() else 0.0
                        pending_inputs = (input_mean >= min_shrink or
                                          (adaptive_work and progress[2] > 0))
                        reserve = input_seconds * (rounds - rd - 1) if pending_inputs else 0.0
                        remaining_work = (self.nb + self.branch_K - j) / max(len(input_vars), 1)
                        refresh = _root_obbt_refresh(
                            admission_rate, progress[2:], time_budget - (time.time() - t0),
                            0.0 if sliced else reserve, input_seconds * remaining_work)
                        if not refresh and cached is None:
                            n_deferred += 2
                            adaptive_work = self._obbt_adaptive_work = True
                            continue
                    reuse = (j >= self.n and rate is not None and cached is not None and
                             not refresh)
                ends_lo, ends_hi = (aux_lo, aux_hi) if transfer_aux else (ln, un)
                w0 = ends_hi[j] - ends_lo[j]
                if w0 <= 0:
                    continue
                e = np.zeros(self.N)
                e[j] = 1.0
                # Capped like the B&B nodes, with or without RLT: cute_spanhyd's 9122-row
                # McCormick LPs took 42 s for 46 OBBT LPs against the 10 s budget.
                cap = _NODE_ITERS * sum(G.shape)
                left = deadline - time.time() if sliced else time_budget - (time.time() - t0)
                if transfer_aux and left <= 0:
                    completed = False
                    break
                duals = [None, None]
                def solve_end(sign, index, left):
                    if transfer_aux and reuse:
                        return _dual_value_sparse(sign * e, 0.0, G, h, lz, uz, cached[index])
                    kwargs = {}
                    if transfer_aux and j >= self.n:
                        def capture(data, y):
                            duals[index] = y.copy()
                        kwargs['certificate_sink'] = capture
                    return self.solve_certified(
                        sign * e, 0.0, G, h, lz, uz, retry=self.rlt is not None,
                        time_limit=left, max_iter=cap, **kwargs)
                lo = solve_end(1.0, 0, left)
                if checkpoint is not None:
                    checkpoint()
                if sliced:
                    yield Atom("obbt")
                left = deadline - time.time() if sliced else time_budget - (time.time() - t0)
                hi = None if left <= 0 else solve_end(-1.0, 1, left)
                if transfer_aux and reuse:
                    n_reuse += 1 + (hi is not None)
                    adaptive_work = self._obbt_adaptive_work = True
                else:
                    n_lp += 2
                if lo is not None and lo > ends_lo[j]:
                    ends_lo[j] = lo
                if hi is not None and -hi < ends_hi[j]:
                    ends_hi[j] = -hi
                if ends_hi[j] < ends_lo[j]:
                    # Empty up to rounding: keep the single point in between.
                    ends_lo[j] = ends_hi[j] = 0.5 * (ends_lo[j] + ends_hi[j])
                if transfer_aux:
                    ln, un = aux_lo[:self.n].copy(), aux_hi[:self.n].copy()
                    active = gain_width > 0
                    gain = float(np.sum((before_width[active] -
                                         (aux_hi - aux_lo)[:self.n][active])
                                        / gain_width[active]))
                    if j >= self.n:
                        gain += max(0.0, (w0 - (ends_hi[j] - ends_lo[j])) / w0)
                    cls = 2 if j < self.n else 0
                    if not reuse:
                        progress[cls] += max(0.0, gain)
                        progress[cls + 1] += 2.0
                    if j >= self.n and not reuse and any(y is not None for y in duals):
                        target_rates[j] = max(0.0, gain) / 2.0
                        target_duals[j] = duals
                shrink += (w0 - (ends_hi[j] - ends_lo[j])) / w0
                if sliced:
                    yield Atom("obbt", box=(torch.from_numpy(ln.copy()).to(l).reshape(l.shape),
                                            torch.from_numpy(un.copy()).to(u).reshape(u.shape)))
            if transfer_aux:
                width = (un - ln)[input_vars]
                active = input_width > 0
                mean = float(np.mean((input_width[active] - width[active]) / input_width[active])) \
                    if active.any() else 0.0
            else:
                mean = shrink / max(len(vars_), 1)
            if verbose:
                print(f"[LiftedLP] OBBT round {rd}: mean relative shrink {mean:.3f} "
                      f"({n_lp} LPs, {time.time() - t0:.1f}s)"
                      + (f" [{n_reuse} dual replays, {n_deferred} deferred]" if transfer_aux else ""))
            paying_inputs = transfer_aux and adaptive_work and progress[2] > 0
            self.obbt_pending = (mean >= min_shrink or paying_inputs or
                                 (not completed and mean > 0))
            spent = time.time() - t0
            if (mean < min_shrink and not paying_inputs) or spent > time_budget or (transfer_aux and spent >= time_budget):
                break
        if transfer_aux and aux_lo is not None:
            self.certificate_registry.add_auxiliary(ln, un, aux_lo, aux_hi, cutoff,
                                                    proof='outward FBBT and certified OBBT')
        return (torch.from_numpy(ln).to(l).reshape(l.shape),
                torch.from_numpy(un).to(u).reshape(u.shape))


def _build_with_cleared_rows(bounded_module, model_ori, x_center, C, l, u, spec_ok,
                             relaxed_rhs_rows, max_pairs, n_check):
    """Build the opt-in extension using exact symbolic coefficients for every LP row.

    The original model/spec is untouched. Cleared rows are additional implied cuts in
    the relaxation; their spec indices also count as usable for lifted-LP B&B/handoff.
    """
    from nlp_solver.rational_rows import (extract_rational_rows,
                                          clear_positive_denominators,
                                          rounded_coefficients)
    n, dev = x_center.numel(), x_center.device
    lo, hi = l.reshape(-1).double().cpu(), u.reshape(-1).double().cpu()
    gen = torch.Generator().manual_seed(0)
    points = (lo + (hi - lo) * torch.rand(n_check, n, generator=gen,
                                        dtype=torch.float64)).to(dev)
    forms = extract_rational_rows(bounded_module, model_ori, x_center, C, points)
    rhs = relaxed_rhs_rows.reshape(-1).detach().cpu().double()
    selected, cleared = {}, {}
    for r, form in enumerate(forms):
        if bool(spec_ok[r]):
            # Every existing quadratic row must be preserved. If the symbolic walker
            # cannot handle one, fall back to the unchanged autograd builder.
            if form.terms is None or any(any(k < 0 for _, k in key) or
                                        sum(k for _, k in key) > 2 for key in form.terms):
                return None
            selected[r] = form
        elif r > 0:
            polynomial = clear_positive_denominators(form, float(rhs[r - 1]), lo)
            if polynomial is not None:
                selected[r] = polynomial
                cleared[r] = float(rhs[r - 1])
    if not cleared:
        return None
    try:
        coeffs = {r: rounded_coefficients(form) for r, form in selected.items()}
    except (ValueError, OverflowError):
        return None
    pairs = set()
    for row in coeffs.values():
        for key in row:
            if sum(k for _, k in key) == 2:
                ij = tuple(i for i, k in key for _ in range(k))
                pairs.add(ij)
    pairs = sorted(pairs)
    if len(pairs) > max_pairs:
        return None
    pair_idx = {pair: j for j, pair in enumerate(pairs)}
    S, K = C.shape[1], len(pairs)
    ks = torch.zeros(S, dtype=torch.float64, device=dev)
    gs = ks.new_zeros(S, n)
    qs = ks.new_zeros(S, K)
    ke, ge, qe = torch.zeros_like(ks), torch.zeros_like(gs), torch.zeros_like(qs)
    for r, row in coeffs.items():
        for key, (value, error) in row.items():
            degree = sum(k for _, k in key)
            if degree == 0:
                ks[r], ke[r] = value, error
            elif degree == 1:
                i = key[0][0]
                gs[r, i], ge[r, i] = value, error
            else:
                ij = tuple(i for i, k in key for _ in range(k))
                j = pair_idx[ij]
                qs[r, j], qe[r, j] = value, error
    row_spec = torch.tensor(sorted(r for r in selected if r > 0), device=dev, dtype=torch.long)
    a = torch.tensor([pair[0] for pair in pairs], device=dev, dtype=torch.long)
    b = torch.tensor([pair[1] for pair in pairs], device=dev, dtype=torch.long)
    obj = (float(ks[0]), gs[0], qs[0])
    obj_mag = (abs(float(ks[0])), gs[0].abs(), qs[0].abs())
    rows = (ks[row_spec], gs[row_spec], qs[row_spec])
    row_mag = tuple(t.abs() for t in rows)
    lp = LiftedLP(n, a, b, obj, obj_mag, rows, row_spec, row_mag, dev)
    lp.cleared_rhs = ks.new_tensor([cleared.get(r, float('nan')) for r in row_spec.tolist()])
    lp.row_error = (ke[row_spec], ge[row_spec], qe[row_spec])
    lp.obj_error = (float(ke[0]), ge[0], qe[0])
    # Clearing can also produce a purely linear cut. Include its variables in OBBT,
    # even when there are no lifted products at all.
    linear_vars = (gs[list(cleared)] != 0).any(0).nonzero().reshape(-1)
    lp.qvars = torch.unique(torch.cat([lp.qvars, linear_vars]))
    print(f"[LiftedLP] cleared denominators for {len(cleared)} constraint rows "
          "(strictly positive root variables; relaxed RHS)")
    print(f"[LiftedLP] {row_spec.numel()}/{S - 1} constraint rows and a quadratic "
          f"objective lifted: n={n}, {K} products ({lp.n_sq} squares), "
          f"{lp.n_mc} McCormick rows")
    return lp


def _liftable_forms(bounded_module, x_center, l, u, outputs, quad_out):
    """({output: Form}, algebra) for the outputs whose forms.py form is a polynomial in the inputs
    and liftable atoms (lifted_atoms): finite coefficients, atoms only of an affine argument over
    columns with a finite range (float-representable coefficients), and each monomial's
    variables inputs or atoms. Anything else, an exhausted operation budget or a failed walk,
    is simply absent. A form with atoms that is flagged inexact (a constant rounded from c**k
    or exp(c)) is kept: its coefficients get _INEXACT relative error (see _polynomial_rows).
    Without atoms only the outputs that are not quadratic are returned (the autograd extraction
    serves the others); with atoms all of them are, as extracting at x = 0 meets the poles of
    the atoms (0 * inf)."""
    from nlp_solver.lifted_atoms import LiftedAtoms, NotLifted
    from nlp_solver.structure.forms import FormAlgebra
    from nlp_solver.structure.walk import build_forms, output_forms
    n = x_center.numel()
    lo, hi = (t.reshape(-1).double().cpu().numpy() for t in (l, u))
    algebra = FormAlgebra(n, lo)
    try:
        forms = output_forms(build_forms(bounded_module, x_center, algebra))
        found, scratch = {}, LiftedAtoms(n, lo, hi, algebra.atom_info)
        for o in outputs:
            form = algebra.expand(forms[o])
            mark = scratch.T
            try:
                if form.poly is None:
                    continue
                columns = [c for mono in form.poly for c in scratch.columns(mono)]
                has_atoms = any(c >= n for c in columns)
                if has_atoms and x_center.dtype != torch.float64:
                    raise NotLifted('atoms need float64 enclosures')
                if (form.inexact and not has_atoms) or not all(
                        np.isfinite(float(c)) for c in form.poly.values()):
                    raise NotLifted('rounded constants or infinite coefficients')
                found[o] = form
            except NotLifted:
                scratch.rollback(mark)
    except Exception:  # noqa: BLE001  (BudgetExceeded, an unsupported node, float overflow)
        return {}, algebra
    if scratch.T > MAX_ATOMS:
        # Many atoms never certified but cost the root: polygon100 (4950 sin atoms times
        # radius products) took 53 s to build for a bound of -43.9 against P = -0.27.
        return {}, algebra
    if not scratch.T:
        found = {o: f for o, f in found.items() if not quad_out[o]}
    return found, algebra


def _polynomial_rows(forms, n, a, b, atoms):
    """Lifted rows (k, g, q) of the polynomials of forms.py Forms, with the outward conversion
    errors (k, g, q shaped) of their float coefficients (an inexact form's constants are
    rounded from transcendental values: _INEXACT of each coefficient is added, an assumed few
    ulp per rounded constant without cancellation between them, not a proof), and the products a, b extended by
    one column per further monomial of degree >= 2. `atoms` (lifted_atoms) registers the
    atoms the polynomials use (columns n .. nb, g covers them). A monomial's columns (sorted,
    with multiplicity) are split into a lower and an upper half, each a column (an input, an atom
    or a product): x^4 = x^2 x^2 and x_i x_j^2 = x_i (x_j x_j) reuse the square relaxation, and
    every product column is defined by two columns before it."""
    from types import SimpleNamespace
    from nlp_solver.rational_rows import rounded_coefficients
    terms = [[(tuple(sorted(atoms.columns(mono))), value, error + _INEXACT * abs(value) * f.inexact)
              for mono, (value, error) in
              rounded_coefficients(SimpleNamespace(terms=f.poly)).items()] for f in forms]
    nb = n + atoms.T
    pairs = list(zip(a.tolist(), b.tolist()))
    column_of = {p: nb + k for k, p in enumerate(pairs)}

    def column(vs):
        if len(vs) == 1:
            return vs[0]
        h = len(vs) // 2
        key = tuple(sorted((column(vs[:h]), column(vs[h:]))))
        if key not in column_of:
            column_of[key] = nb + len(pairs)
            pairs.append(key)
        return column_of[key]

    dt, dev = torch.float64, a.device
    m = len(forms)
    k, ek = torch.zeros(m, dtype=dt), torch.zeros(m, dtype=dt)
    g, eg = torch.zeros(m, nb, dtype=dt), torch.zeros(m, nb, dtype=dt)
    products = []
    for o, row in enumerate(terms):
        for vs, value, error in row:
            if not vs:
                k[o], ek[o] = value, error
            elif len(vs) == 1:
                g[o, vs[0]] += value; eg[o, vs[0]] += error
            else:
                products.append((o, column(vs) - nb, value, error))
    q, eq = torch.zeros(m, len(pairs), dtype=dt), torch.zeros(m, len(pairs), dtype=dt)
    for o, j, value, error in products:
        q[o, j] += value; eq[o, j] += error
    a = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=dev)
    b = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=dev)
    return a, b, (k.to(dev), g.to(dev), q.to(dev)), (ek.to(dev), eg.to(dev), eq.to(dev))


def build_lifted_lp(bounded_module, model_ori, x_center: Tensor, C: Tensor, l: Tensor,
                    u: Tensor, max_vars: int = 2000, max_pairs: int = 20000,
                    n_check: int = 4, skip_if_linear_rows: bool = False,
                    clear_denominators: bool = False,
                    relaxed_rhs_rows: Optional[Tensor] = None,
                    convex_objective: bool = False) -> Optional[LiftedLP]:
    """LiftedLP for the spec matrix C (1, S, m_out) on the root box l, u (1, n), or None
    when no products or proved convex objective epigraphs can be lifted."""
    from nlp_solver.quadratic_tangent import output_degrees
    n = int(x_center.reshape(-1).numel())
    if n > max_vars or not (torch.isfinite(l).all() and torch.isfinite(u).all()):
        return None
    # Epigraph coefficients and directed scalar enclosures are float64. Avoid
    # narrowing them in the dense path (whose ordinary-row cushions assume that
    # precision too); other dtypes retain the existing lifted LP behaviour.
    if convex_objective and x_center.dtype == torch.float64:
        from .convex_objective import build_convex_lifted_lp
        try:
            lp = build_convex_lifted_lp(bounded_module, model_ori, x_center, C, l, u,
                                      max_pairs, clear_denominators, relaxed_rhs_rows)
        except (ValueError, OverflowError, ZeroDivisionError):
            lp = None
        if lp is not None:
            return lp
    try:
        degs = output_degrees(bounded_module, x_center)
    except Exception:  # noqa: BLE001
        return None
    dev = x_center.device
    Cm = C[0].detach().to(device=dev, dtype=torch.float64)          # (S, m_out)
    if degs.numel() != Cm.shape[1]:
        return None
    quad_out = (torch.isfinite(degs) & (degs <= 2)).to(dev)
    uses = Cm != 0
    spec_ok = ~(uses & ~quad_out.unsqueeze(0)).any(1)
    if (clear_denominators and relaxed_rhs_rows is not None and bool(spec_ok[0])
            and bool((~spec_ok[1:]).any())):
        lp = _build_with_cleared_rows(bounded_module, model_ori, x_center, C, l, u,
                                     spec_ok, relaxed_rhs_rows, max_pairs, n_check)
        if lp is not None:
            return lp
    # The other outputs enter through their exact form (degree > 2, or atoms), with product columns.
    found, algebra = _liftable_forms(bounded_module, x_center, l, u,
                                        uses.any(0).nonzero().reshape(-1).tolist(), quad_out) \
        if bool((uses.any(0) & ~quad_out).any()) else ({}, None)
    lift_out, from_forms = quad_out.clone(), torch.zeros_like(quad_out)
    if found:
        lift_out[list(found)] = from_forms[list(found)] = True
    spec_ok = ~(uses & ~lift_out.unsqueeze(0)).any(1)
    if not bool(spec_ok.any()):
        return None
    if skip_if_linear_rows:
        lin_out = (torch.isfinite(degs) & (degs <= 1)).to(dev)
        if not bool((uses[1:] & ~lin_out.unsqueeze(0)).any()):
            print("[LiftedLP] skipped: every constraint row is linear (QTB covers the "
                  "convex objective)")
            return None
    out_idx = (uses[spec_ok].any(0) & lift_out).nonzero().reshape(-1)
    quad_idx, hi_idx = out_idx[~from_forms[out_idx]], out_idx[from_forms[out_idx]]
    shape = x_center.shape

    def outputs(z):
        return model_ori(z.reshape(shape).to(x_center.dtype)).reshape(-1).to(torch.float64)

    def fn(z):
        return outputs(z)[quad_idx]

    g_ = torch.Generator().manual_seed(0)
    lo, hi = l.reshape(-1).double().cpu(), u.reshape(-1).double().cpu()
    x_check = lo + (hi - lo) * torch.rand(n_check, n, generator=g_, dtype=torch.float64)
    try:
        if quad_idx.numel():
            k, g, a, b, q, ok = extract_quadratic(fn, n, x_check.to(dev), device=dev,
                                                  linear=(degs.to(dev)[quad_idx] <= 1))
        else:
            k, g, q, ok = (torch.zeros(0, dtype=torch.float64, device=dev),
                           torch.zeros(0, n, dtype=torch.float64, device=dev),
                           torch.zeros(0, 0, dtype=torch.float64, device=dev),
                           torch.zeros(0, dtype=torch.bool, device=dev))
            a = b = torch.zeros(0, dtype=torch.long, device=dev)
    except Exception as e:  # noqa: BLE001
        print(f"[LiftedLP] extraction failed: {e!r}")
        return None
    errors, atoms, nb = None, None, n
    if hi_idx.numel():
        from nlp_solver.lifted_atoms import LiftedAtoms
        atoms = LiftedAtoms(n, lo.numpy(), hi.numpy(), algebra.atom_info)
        a, b, hi_rows, hi_err = _polynomial_rows([found[o] for o in hi_idx.tolist()], n, a, b, atoms)
        nb = n + atoms.T
        kh, gh, qh = hi_rows
        ok_h = torch.ones(hi_idx.numel(), dtype=torch.bool, device=dev)
        levels = _levels(a, b, nb)
        for p in range(x_check.shape[0]):       # the model at the check points, as for quadratics
            x = x_check[p].to(device=dev, dtype=torch.float64)
            lz, uz = _column_boxes(x, x, a, b, levels, torch, atoms if atoms.T else None)
            z = 0.5 * (lz + uz)         # the columns' values: their boxes are points up to rounding
            rep = kh + gh @ z[:nb] + qh @ z[nb:]
            scale = 1.0 + kh.abs() + gh.abs() @ z[:nb].abs() + qh.abs() @ z[nb:].abs()
            ok_h &= (rep - outputs(x)[hi_idx]).abs() <= 1e-9 * scale
        g = torch.nn.functional.pad(g, (0, atoms.T))
        q = torch.nn.functional.pad(q, (0, a.numel() - q.shape[1]))
        zero_q = torch.zeros(k.numel(), a.numel(), dtype=torch.float64, device=dev)
        k, g, q, ok = torch.cat([k, kh]), torch.cat([g, gh]), torch.cat([q, qh]), torch.cat([ok, ok_h])
        errors = (torch.cat([torch.zeros_like(k[:quad_idx.numel()]), hi_err[0]]),
                  torch.cat([torch.zeros_like(g[:quad_idx.numel()]), hi_err[1]]),
                  torch.cat([zero_q, hi_err[2]]))
        out_idx = torch.cat([quad_idx, hi_idx])
    if a.numel() > max_pairs:
        print(f"[LiftedLP] {a.numel()} products exceed max_pairs={max_pairs}; disabled")
        return None
    bad_out = out_idx[~ok]
    if bad_out.numel():
        spec_ok &= ~uses[:, bad_out].any(1)
    Cq = Cm[:, out_idx]                                             # (S, m_q)
    ks, gs, qs = Cq @ k, Cq @ g, Cq @ q                             # spec-row representation
    Ka, Ga, Qa = Cq.abs() @ k.abs(), Cq.abs() @ g.abs(), Cq.abs() @ q.abs()
    obj, obj_mag = None, None
    if bool(spec_ok[0]):
        obj, obj_mag = (float(ks[0]), gs[0], qs[0]), (float(Ka[0]), Ga[0], Qa[0])
    if errors is not None:                                          # float conversion of coefficients
        Ke, Ge, Qe = (Cq.abs() @ t for t in errors)
    row_spec = spec_ok.nonzero().reshape(-1)
    row_spec = row_spec[row_spec >= 1]
    rows = (ks[row_spec], gs[row_spec], qs[row_spec]) if row_spec.numel() else None
    # Keep only the products some lifted row or the objective uses, and the products defining them.
    used = torch.zeros(a.numel(), dtype=torch.bool, device=dev)
    if rows is not None:
        used |= (rows[2] != 0).any(0)
    if obj is not None:
        used |= obj[2] != 0
    while True:
        factors = torch.cat([a[used], b[used]])
        closed = used.clone()
        closed[factors[factors >= nb] - nb] = True
        if torch.equal(closed, used):
            break
        used = closed
    keep = used.nonzero().reshape(-1)
    if keep.numel() == 0 and nb == n:
        return None     # nothing lifted: the node LP over CROWN planes already has it
    new_col = nb + torch.cumsum(used, 0) - 1
    a, b = a[keep], b[keep]
    a = torch.where(a >= nb, new_col[(a - nb).clamp(min=0)], a)
    b = torch.where(b >= nb, new_col[(b - nb).clamp(min=0)], b)
    if rows is not None:
        rows = (rows[0], rows[1], rows[2][:, keep])
    if obj is not None:
        obj = (obj[0], obj[1], obj[2][keep])
        obj_mag = (obj_mag[0], obj_mag[1], obj_mag[2][keep])
    row_mag = (Ka[row_spec], Ga[row_spec], Qa[row_spec][:, keep]) if rows is not None else None
    lp = LiftedLP(n, a, b, obj, obj_mag, rows, row_spec, row_mag, dev,
                  atoms if atoms is not None and atoms.T else None)
    if errors is not None:
        if rows is not None:
            lp.row_error = (Ke[row_spec], Ge[row_spec], Qe[row_spec][:, keep])
        if obj is not None:
            lp.obj_error = (float(Ke[0]), Ge[0], Qe[0][keep])
    print(f"[LiftedLP] {int(row_spec.numel())}/{Cm.shape[0] - 1} constraint rows and "
          f"{'a' if obj is not None else 'no'} {'polynomial' if errors is not None else 'quadratic'} "
          f"objective lifted: n={n}, {nb - n} atoms, "
          f"{lp.K} products ({lp.n_sq} squares), {lp.n_mc} McCormick rows")
    return lp
