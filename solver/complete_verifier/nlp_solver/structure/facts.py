"""Per-row structure facts from the exact forms of the NLP's rows."""
import math
import time
from dataclasses import dataclass, field
from fractions import Fraction as F

import numpy as np
import torch

from .forms import BudgetExceeded, DEFAULT_BUDGET, Form, FormAlgebra, exact
from .rowpool import RowPool
from .walk import build_forms, output_forms

INF = float('inf')


@dataclass
class RowFacts:
    cls: str = 'other'              # const affine quad poly laurent frac exp atoms other
    tags: set = field(default_factory=set)
    degree: float = INF
    nterms: int = 0
    support: object = None          # frozenset or None (unknown)
    inexact: bool = False
    reason: str = None
    gp: bool = False                # posynomial-shaped: nonconstant coefficients > 0, support strictly positive on the box
    sepuni: bool = False            # every monomial of degree > 2 is a power of a single variable
    atom_kinds: set = field(default_factory=set)
    atom_deg: float = None          # max degree of the atoms' polynomial arguments (inf if not polynomial)
    pole_free: bool = None          # True only when every negative power has a base whose box range excludes 0; None = unknown
    curvature: str = None           # affine / convex / concave / indefinite (quad only; None if undecided)
    lambda_min: float = None        # rigorous lower bound on the smallest Hessian eigenvalue (exact quad rows)
    rank: int = None
    square: bool = False            # f = lam*(v.x+d)^2 + c  (square-row strip)
    box: tuple = None               # interval of the form on the root box (floats, outward)
    duplicate_of: tuple = None      # (row, scale, offset), scale > 0: this row = scale * row[row] + offset exactly
    opposite_of: tuple = None       # same with scale < 0
    alias: tuple = None             # (alpha, beta): row = alpha*objective + beta
    circle: tuple = None            # (i, j): row = a*(x_i^2 + x_j^2) + k exactly, and the opposite sense is also a row (an equality band)
    redundant: bool = False         # implied by the root box at the relaxed rhs
    infeasible: bool = False        # violated everywhere on the root box (> relaxed rhs)


@dataclass
class StructureReport:
    rows: list                      # list[RowFacts]; rows[0] is the objective
    objective: RowFacts
    n: int
    atoms: dict
    poles: set
    spent: int
    seconds: float
    skipped: str = None
    forms: list = field(default_factory=list)   # the exact Form of every row (row 0 = objective)
    pool: object = None             # RowPool: the rows with their tau applied once, parallel ones merged (see build_structure)
    algebra: object = None          # the FormAlgebra of `forms` when the caller supplied one (atom ids live there)


def atom_facts(w, e, I, lo, hi):
    """tags of the factored form: separable sum of atoms, exp-substitutable, rational"""
    n = w.n
    nonconst = [(m, c) for m, c in e.poly.items() if m]
    atoms = {v for m, _ in nonconst for v, _ in m if v >= n}
    infos = [w.atom_info[v] for v in atoms]
    I.atom_kinds = {i[0] for i in infos}

    def argdeg(i):
        arg = dict(i[2])
        single = [m for m in arg if m]
        if len(single) == 1 and len(single[0]) == 1 and single[0][0][0] >= n and single[0][0][1] == 1:
            return argdeg(w.atom_info[single[0][0][0]])          # chain of unary functions: degree of the innermost polynomial
        if any((v >= n and w.atom_info[v][0] != 'exp') or F(ex).denominator != 1 or ex < 0 for m in arg for v, ex in m):
            return INF
        return max((sum(ex for _, ex in m) for m in arg), default=0)
    I.atom_deg = max(argdeg(i) for i in infos)
    sep = all(len(m) == 1 and m[0][1] == 1 and m[0][0] >= n for m, _ in nonconst if any(v >= n for v, _ in m))
    exp_ok = 'exp' in I.atom_kinds and I.atom_kinds <= {'exp', 'log'} and all(
        (i[0] == 'exp' and len(i[2]) == 1 and i[2][0][1] == 1 and len(i[2][0][0]) == 1 and i[2][0][0][0][1] == 1)
        or (i[0] == 'log' and argdeg(i) <= 1) for i in infos)
    I.tags.add('atoms')
    if sep:
        I.tags.add('sepatoms')
    if exp_ok:
        I.tags.add('exp')
    if I.atom_kinds == {'pow'} and all(i[1] < 0 and sign_definite(i[2], lo, hi) for i in infos):
        I.tags.add('rational')
    return sep, exp_ok


def sign_definite(items, lo, hi):
    """the polynomial given as (monomial, coefficient) items has a box range that excludes 0 (False when unknown)"""
    r = box_range(Form(dict(items)), lo, hi)
    return r is not None and (r[0] > 0 or r[1] < 0)


def pole_free(w, poly, lo, hi, seen=None):
    """every negative power in poly, including inside atom arguments, has a base proved nonzero on the box.
    A range that contains 0 proves nothing (interval arithmetic overestimates), so the answer is True or None."""
    seen = set() if seen is None else seen
    for m in poly:
        for v, ex in m:
            if v < w.n:
                if ex < 0 and not (lo[v] > 0 or hi[v] < 0):
                    return None
            elif v not in seen:
                seen.add(v)
                kind, param, arg, _ = w.atom_info[v]
                if ex < 0 and kind != 'exp':          # a negative power of log(.), sqrt(.), ...: its zeros are unknown
                    return None
                if kind == 'pow' and param < 0 and not sign_definite(arg, lo, hi):
                    return None
                if pole_free(w, dict(arg), lo, hi, seen) is None:
                    return None
    return True


def classify(w, e, lo, hi):
    """e: the row's factored Form.  Class/degree come from its expansion; atom tags from the factored form."""
    n = w.n
    I = RowFacts()
    if e.poly is None:
        I.tags = {'other'}
        I.support, I.reason = e.support, e.reason
        return I, e
    x = w.expand(e)
    if x.poly is None:
        I.tags = {'other'}
        I.support, I.reason = x.support, x.reason
        I.inexact = e.inexact           # e is returned and pooled as the row; its coefficients may be rounded
        return I, e
    I.inexact = x.inexact
    I.pole_free = pole_free(w, e.poly, lo, hi)
    I.support = w.support(x)
    I.nterms = len(x.poly)
    nonconst = [(m, c) for m, c in x.poly.items() if m]
    if not nonconst:
        I.cls, I.degree, I.tags = 'const', 0, {'const'}
        return I, x
    atoms = {v for m, _ in nonconst for v, _ in m if v >= n}
    if atoms or any(v >= n for m in e.poly for v, _ in m):
        sep, exp_ok = atom_facts(w, e if not atoms else x, I, lo, hi)
        I.cls = 'exp' if exp_ok else 'atoms'
        if not atoms:                       # factored squares expand to a polynomial: keep its exact class too
            I.tags.add('factored')
        else:
            return I, x
    deg, ints, neg, pos_coef, vs = 0, True, False, True, set()
    multi_nonquad = False           # a monomial that is not quadratic and involves >1 variable (breaks separability)
    for m, c in nonconst:
        d = 0
        if len(m) > 1 and not all(type(ex) is int and ex >= 0 for _, ex in m) or sum(ex for _, ex in m) > 2 and len(m) > 1:
            multi_nonquad = True
        for v, ex in m:
            vs.add(v)
            if type(ex) is not int and ex.denominator != 1:
                ints = False
            d += ex
            neg = neg or ex < 0
        deg = max(deg, d)
        pos_coef = pos_coef and c > 0
    nonneg = ints and not neg
    I.degree = deg if nonneg else INF
    I.gp = pos_coef and all(lo[v] > 0 for v in vs)
    keep = set(I.tags)
    if nonneg:
        I.cls = 'affine' if deg <= 1 else 'quad' if deg == 2 else 'poly'
        I.tags = {I.cls}
        if I.cls != 'affine':
            I.tags.add('poly')
    elif ints:
        I.cls = 'laurent'
        I.tags = {'laurent'}
    else:
        I.cls = 'frac'
        I.tags = {'frac'}
    if I.cls in ('poly', 'laurent', 'frac'):
        I.sepuni = not multi_nonquad          # separable univariate terms: quadratic part + c*x_i^k
    I.tags |= keep
    if I.gp:
        I.tags.add('posy')
    return I, x


def _float_toward(x, direction):
    """float of a Fraction, rounded toward direction (-INF or INF)"""
    f = float(x)
    if F(f) == x or (F(f) > x) == (direction > 0):
        return f
    return float(np.nextafter(f, direction))


def curvature(e, support):
    """Hessian facts of an exact quadratic form: (lambda_min, curvature, rank, is_square_strip).
    lambda_min is a rigorous lower bound on the smallest Hessian eigenvalue: the larger of the exact Gershgorin bound
    (Fractions, rounded down) and, for small blocks, eigvalsh minus a rounding margin. Curvature follows from the
    bounds: convex iff lambda_min >= 0, concave iff the analogous upper bound on the largest eigenvalue is <= 0,
    indefinite iff the Hessian has diagonal entries of both signs or the rounded spectrum has both signs; a PSD
    Hessian whose zero eigenvalue the bounds cannot certify stays None. is_square_strip: f = lam*(v.x + d)^2 + c,
    i.e. a rank-1 Hessian with the gradient inside span(v)."""
    H, g = {}, {}
    for m, c in e.poly.items():
        d = sum(ex for _, ex in m)
        if d == 1:
            g[m[0][0]] = g.get(m[0][0], 0) + c
        elif d == 2 and len(m) == 1:
            H[m[0][0], m[0][0]] = H.get((m[0][0], m[0][0]), 0) + 2 * c
        elif d == 2:
            a, b = m[0][0], m[1][0]
            H[a, b] = H.get((a, b), 0) + c
            H[b, a] = H.get((b, a), 0) + c
    if not H:
        return 0.0, 'affine', 0, False
    vs = sorted(support)
    ix = {v: i for i, v in enumerate(vs)}
    k = len(vs)
    diag = {v: H.get((v, v), 0) for v in vs}
    off = dict.fromkeys(vs, 0)
    for (a, b), c in H.items():
        if a != b:
            off[a] += abs(c)
    lo = min(diag[v] - off[v] for v in vs)          # Gershgorin: every eigenvalue lies in some [d_i - r_i, d_i + r_i]
    hi = max(diag[v] + off[v] for v in vs)
    lam_lo, lam_hi = _float_toward(F(lo), -INF), _float_toward(F(hi), INF)
    indefinite = min(diag.values()) < 0 < max(diag.values())      # Rayleigh quotients of two unit vectors
    separable = not any(off.values())
    rank, square = (sum(1 for c in diag.values() if c) if separable else None), False
    structural = lam_lo >= 0 or lam_hi <= 0
    if k <= 1500 and not (structural and k > 64):      # the dense eigensolve is for small blocks or undecided ones
        M = np.zeros((k, k))
        for (a, b), c in H.items():
            M[ix[a], ix[b]] = float(c)
        ev, V = np.linalg.eigh(M)
        # backward error of a symmetric eigensolver ~ n * eps * ||M||_F (+ one rounding of the final sum)
        margin = (k + 1) * 2.2e-16 * float(np.linalg.norm(M))
        lam_lo, lam_hi = max(lam_lo, ev[0] - margin), min(lam_hi, ev[-1] + margin)
        indefinite = indefinite or (ev[0] + margin < 0 < ev[-1] - margin)
        scale = max(abs(ev[0]), abs(ev[-1]), 1e-300)
        nz = np.abs(ev) > 1e-9 * scale
        rank = int(nz.sum())
        if rank == 1:
            v = V[:, np.flatnonzero(nz)[0]]
            gv = np.array([float(g.get(u, 0)) for u in vs])
            perp = gv - v * (v @ gv)
            square = bool(np.abs(perp).max(initial=0.0) <= 1e-9 * max(np.abs(gv).max(initial=0.0), 1e-300))
    kind = 'convex' if lam_lo >= 0 else 'concave' if lam_hi <= 0 else 'indefinite' if indefinite else None
    return lam_lo, kind, rank, square


def box_range(e, lo, hi):
    """interval of a nonatom form with integer exponents on the root box, in floats with a forward-error margin
    (4.4e-16 per operation on the term magnitudes); None if unsupported (atoms, fractional powers, pole in the box,
    an infinite bound, or a float overflow)"""
    if e.poly is None or len(e.poly) > 5000:
        return None
    tlo = thi = marg = 0.0
    try:
        for m, c in e.poly.items():
            a = b = float(c)
            mag = abs(a)
            for v, ex in m:
                if v >= len(lo) or type(ex) is not int and ex.denominator != 1:
                    return None
                ex = int(ex)
                l, u = float(lo[v]), float(hi[v])
                if not (math.isfinite(l) and math.isfinite(u)):
                    return None
                if ex < 0:
                    if l <= 0 <= u:
                        return None
                    l, u, ex = 1 / u, 1 / l, -ex
                vl, vu = sorted((l ** ex, u ** ex))
                if ex % 2 == 0 and l <= 0 <= u:
                    vl = 0.0
                cs = (a * vl, a * vu, b * vl, b * vu)
                a, b = min(cs), max(cs)
                mag = max(abs(a), abs(b), mag)
            tlo += a
            thi += b
            marg += 4.4e-16 * (len(m) + 2) * max(abs(a), abs(b), mag)
        marg += 4.4e-16 * len(e.poly) * max(abs(tlo), abs(thi))
    except OverflowError:
        return None
    box = tlo - marg, thi + marg
    return box if all(math.isfinite(v) for v in box) else None


def point_box(e, lo, hi):
    """value of an affine/constant form whose variables are all fixed (lo == hi), as an outward float interval
    (a single float when the exact value is representable and the form is exact); None if some variable is not
    fixed. An inexact form is widened by its coefficients' rounding, a few ulps of the term magnitudes."""
    val, mag = F(0), F(0)
    for m, c in e.poly.items():
        term = c if not m else c * F(float(lo[m[0][0]])) if lo[m[0][0]] == hi[m[0][0]] else None
        if term is None:
            return None
        val += term
        mag += abs(term)
    f = float(val)
    if F(f) == val and not e.inexact:
        return f, f
    slack = 4 * 2.2e-16 * float(mag) if e.inexact else 0.0
    return float(np.nextafter(f - slack, -INF)), float(np.nextafter(f + slack, INF))


def direction_key(e):
    """direction of an affine form normalised by its first coefficient's magnitude, in exact Fractions: equal keys
    <=> positive multiples, negated keys <=> negative multiples"""
    nz = sorted((m[0][0], c) for m, c in e.poly.items() if m)
    s = abs(nz[0][1])
    return tuple((v, F(c) / s) for v, c in nz)


def signed_rows(w, outs, C):
    """The rows C[0] @ outs as (lazy) Forms; a row that runs out of budget is an unknown form."""
    import scipy.sparse as sp
    Cs = sp.csr_matrix(C[0].detach().cpu().double().numpy())
    rows = []
    for i in range(Cs.shape[0]):
        lo_, hi_ = Cs.indptr[i], Cs.indptr[i + 1]
        pairs = [(outs[j], exact(float(c))) for j, c in zip(Cs.indices[lo_:hi_], Cs.data[lo_:hi_])]
        try:
            rows.append(w.lincomb(pairs))
        except BudgetExceeded:
            rows.append(Form(None, None, 'budget'))
    return rows


def build_structure(bounded_module, x0, C, rhs, lower, upper, budget=DEFAULT_BUDGET, pool=None, base_rows=None,
                    algebra=None):
    """Facts of every spec row C[0] @ f(x) (row 0 = objective) on the root hull (lower.amin(0), upper.amax(0))
    of all domains, so every box fact holds on each domain. Redundancy and infeasibility compare against the
    thresholds the verify predicate accepts (pool.thresholds(rhs)[0]); the report is skipped ('per-domain specs')
    unless C and rhs[:, 1:] are the same for all domains. Duplicate/opposite/alias/curvature claims need exact (not
    rounded) forms.
    The report's pool (RowPool) is `pool`, or a new one over rhs and base_rows (the problem's raw rows A x <= b); the
    rows' exact polynomials merge its parallel rows, so the pool is complete once per instance.
    algebra: a FormAlgebra class to build the forms with (e.g. certificates.SquareForms); the report keeps its instance."""
    t0 = time.monotonic()
    n = x0.numel()
    pool = RowPool(rhs, base_rows) if pool is None else pool
    if C.shape[0] > 1 and not (torch.equal(C, C[:1].expand_as(C))
                               and torch.equal(rhs[:, 1:], rhs[:1, 1:].expand_as(rhs[:, 1:]))):
        return StructureReport([], RowFacts(), n, {}, set(), 0, time.monotonic() - t0, 'per-domain specs', pool=pool)
    lo = lower.reshape(lower.shape[0], -1).amin(0).detach().cpu().double().numpy()
    hi = upper.reshape(upper.shape[0], -1).amax(0).detach().cpu().double().numpy()
    thr = pool.thresholds(rhs)[0].detach().cpu().double().numpy().reshape(-1)
    w = (algebra or FormAlgebra)(n, lo, budget)
    outs = output_forms(build_forms(bounded_module, x0, w))
    raw = signed_rows(w, outs, C)
    infos, rows = [], []
    for r in raw:
        try:
            I, e = classify(w, r, lo, hi)
        except BudgetExceeded:
            I, e = RowFacts(tags={'other'}, reason='budget'), Form(None, None, 'budget')
        infos.append(I)
        rows.append(e)
    for e, I in zip(rows, infos):
        if I.cls in ('quad', 'affine') and I.support is not None and not I.inexact:
            if I.cls == 'quad':
                try:
                    I.lambda_min, I.curvature, I.rank, I.square = curvature(e, I.support)
                except OverflowError:               # a Hessian entry beyond the float range: curvature stays unknown
                    pass
            else:
                I.lambda_min, I.curvature, I.rank = 0.0, 'affine', 0
    # affine duplicates / equality pairs: bucket by support first, exact Fractions only inside multi-row buckets
    buckets = {}
    for r, (e, I) in enumerate(zip(rows, infos)):
        if r > 0 and I.cls == 'affine' and not I.inexact:
            buckets.setdefault(tuple(sorted(m[0][0] for m in e.poly if m)), []).append(r)
    lin = lambda r: {m[0][0]: c for m, c in rows[r].poly.items() if m}
    for rs in buckets.values():
        if len(rs) < 2:
            continue
        seen = {}
        for r in rs:
            key = direction_key(rows[r])
            nk = tuple((v, -c) for v, c in key)
            j = seen.get(key) or seen.get(nk)
            if j is None:
                seen[key] = r
                continue
            v0 = next(iter(lin(r)))
            scale = F(lin(r)[v0]) / lin(j)[v0]
            offset = exact(rows[r].poly.get((), 0) - scale * rows[j].poly.get((), 0))
            if scale > 0:
                infos[r].duplicate_of = (j, exact(scale), offset)
            else:
                infos[r].opposite_of = (j, exact(scale), offset)
    obj = rows[0]
    if obj.poly is not None and not obj.inexact:
        okeys = {m for m in obj.poly if m}
        if okeys:
            m0 = min(okeys)
            for r, e in enumerate(rows[1:], 1):
                if e.poly is None or e.inexact:
                    continue
                if {m for m in e.poly if m} == okeys:
                    a = F(e.poly[m0]) / obj.poly[m0]
                    if a and all(e.poly[m] == a * obj.poly[m] for m in okeys):
                        infos[r].alias = (a, e.poly.get((), F(0)) - a * obj.poly.get((), F(0)))
    # circle equalities a*(x_i^2 + x_j^2) = k: rows a*S + k1 <= 0 and -a*S + k2 <= 0 with equal exact quadratic
    # parts and constants within 1e-3 (the model's equality band is ~2e-5)
    sides = {}
    for r, (e, I) in enumerate(zip(rows, infos)):
        sq = [m for m in e.poly if m] if r > 0 and I.cls == 'quad' and not I.inexact else []
        if len(sq) == 2 and all(len(m) == 1 and m[0][1] == 2 for m in sq) and e.poly[sq[0]] == e.poly[sq[1]]:
            a = e.poly[sq[0]]
            pair = tuple(sorted(m[0][0] for m in sq))
            sides.setdefault((pair, abs(a)), {True: [], False: []})[a > 0].append((r, e.poly.get((), 0)))
    for (pair, _), side in sides.items():
        for r, k1 in side[True]:
            for s_, k2 in side[False]:
                if abs(k1 + k2) <= 1e-3 * abs(k1):
                    infos[r].circle = infos[s_].circle = pair
    # root-box ranges. Affine rows: all-fixed rows exactly, else one sparse float pass with a forward-error
    # margin (|c|-sum * (nnz+2) * 1.2e-16); polynomial rows: box_range
    import scipy.sparse as sp
    ar, floats = [], {}
    for r in range(1, len(rows)):
        if infos[r].cls in ('affine', 'const'):
            try:
                pb = point_box(rows[r], lo, hi)
                if pb is None:
                    floats[r] = [(m, float(c)) for m, c in rows[r].poly.items()]
                    ar.append(r)
                else:
                    infos[r].box = pb
            except OverflowError:               # a value beyond the float range: no box
                pass
    if ar:
        ri, ci, vi, c0 = [], [], [], np.zeros(len(ar))
        for k, r in enumerate(ar):
            for m, c in floats[r]:
                if m:
                    ri.append(k)
                    ci.append(m[0][0])
                    vi.append(c)
                else:
                    c0[k] = c
        A = sp.csr_matrix((vi, (ri, ci)), shape=(len(ar), n))
        Ap, Am = A.maximum(0), A.minimum(0)
        mx = Ap @ hi + Am @ lo + c0
        mn = Ap @ lo + Am @ hi + c0
        M = np.maximum(np.abs(lo), np.abs(hi))
        marg = (np.diff(A.indptr) + 2) * 1.2e-16 * (abs(A) @ M + np.abs(c0))
        for k, r in enumerate(ar):
            box = float(mn[k] - marg[k]), float(mx[k] + marg[k])
            infos[r].box = None if any(math.isnan(v) for v in box) else box
    for r in range(1, len(rows)):
        if infos[r].cls in ('quad', 'poly', 'laurent'):
            infos[r].box = box_range(rows[r], lo, hi)
        if infos[r].box is not None:
            infos[r].redundant = infos[r].box[1] <= thr[r]
            infos[r].infeasible = infos[r].box[0] > thr[r]
    pool.merge([None if e.poly is None or I.inexact else e.poly for e, I in zip(rows, infos)],
               [I.redundant for I in infos])
    return StructureReport(infos, infos[0], n, w.atom_info, w.poles, w.spent, time.monotonic() - t0, None, rows, pool,
                           w if algebra is not None else None)
