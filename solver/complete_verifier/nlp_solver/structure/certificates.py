"""Root certificates of exact global structure: a floor on the objective over every feasible point of the root hull,
and a constructive candidate point.

They are compiled once, from the exact forms of the structure pass (facts.build_structure run with SquareForms as its
algebra), and enter the BaB as one scalar valid on every box: the root objective floor that bounding applies with the
constant-plane replacement. Each detector has its own flag, solver.nlp.struct_cert_<kind>:
  diameter    ordered polar polygon, adjacent signed area, complete squared-distance graph (polygon*)
  assignment  almost-binary assignment with integer cardinalities and identical squared-affine group costs
              (cute_grouping)
  eigenblock  normalized symmetric eigenpair rows with an affine eigenvalue objective (cute_eigminb; spectral.py)
A detector recognizes exact rational identities of the forms and RowPool rows, never names, and declines (NoPattern)
on anything else. Bounds are evaluated outward: 60-digit interval arithmetic, or outward FP64 tensor kernels
(cert_kernels.py). Numerical solvers only propose. A candidate becomes an incumbent only after the caller verifies it
on the original model.
"""
from dataclasses import dataclass, field
from fractions import Fraction as F
import functools
import math
import time

import numpy as np
import torch

from .cert_kernels import CostPattern, CoverPattern, combination_ids, lp_prices
from .forms import Form, FormAlgebra
from .rowpool import round_down, round_up


class Expired(Exception):
    pass


class NoPattern(Exception):
    pass


def require(ok, reason):
    if not ok:
        raise NoPattern(reason)


def key(p):
    return tuple(sorted(p.items()))


class SquareForms(FormAlgebra):
    """FormAlgebra that also remembers every exact polynomial it squared, keyed by its terms: the expanded forms lose
    the factorization that recognizes a squared-affine objective. The forms themselves are unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.squares = {}

    def _square(self, a):
        if a.poly is not None and not a.inexact:
            self.squares[key(a.poly)] = a

    def mul(self, a, b):
        if a.poly is not None and b.poly is not None and a.poly == b.poly and not b.inexact:
            self._square(a)
        return super().mul(a, b)

    def pow(self, a, k):
        if k == 2:
            self._square(a)
        return super().pow(a, k)


class Graph:
    """The root problem in exact form: the objective `obj` and the RowPool rows (tolerance included) as canonical
    polynomials, over the root hull [lo, hi], with a private 60-digit interval context and the run deadline."""

    def __init__(self, w, forms, pool, lo, hi, deadline=math.inf, device='cpu'):
        self.w, self.n, self.lo, self.hi = w, w.n, lo, hi
        self.deadline, self.device = deadline, torch.device(device)
        self.forms, self.base_pool = forms, pool
        from mpmath.ctx_iv import MPIntervalContext
        self.iv = MPIntervalContext()
        self.iv.dps = 60
        self.obj = self.normal(forms[0]) if forms and forms[0].poly is not None and not forms[0].inexact else None

    def check(self):
        if time.time() >= self.deadline:
            raise Expired

    @functools.cached_property
    def pool(self):
        """A private RowPool view whose spec rows are the canonical forms (trig signs normalized), so that rows compare
        by exact equality. The solver's pool is not replaced."""
        import copy
        rows = []
        for e in self.forms:
            self.check()
            rows.append(self.normal(e) if e.poly is not None and not e.inexact else None)
        pool = copy.copy(self.base_pool)
        pool.merge(rows, [False] * len(rows))
        return pool

    def num(self, v):
        v = F(v)
        return self.iv.mpf(v.numerator) / v.denominator

    def down(self, x):
        return float(np.nextafter(float(x.a), -math.inf))

    def up(self, x):
        return float(np.nextafter(float(x.b), math.inf))

    def normal(self, form):
        """Expand integer powers and write sin(-a), cos(-a) as -sin(a), cos(a), exactly. The terms are summed into one
        dict (linear in their number; adding each term to a fresh copy was quadratic) under the run deadline."""
        w = self.w
        p = w.expand(form)
        require(p.poly is not None and not p.inexact, 'inexact/unsupported expression')
        out = {}
        for m, c in p.poly.items():
            self.check()
            f = Form({(): c})
            for v, e in m:
                if v >= self.n and w.atom_info[v][0] in ('sin', 'cos'):
                    kind, _, arg, _ = w.atom_info[v]
                    arg = dict(arg)
                    s = -1 if arg and arg[min(arg)] < 0 else 1
                    if s < 0:
                        arg = {k: -a for k, a in arg.items()}
                    a = w.atom(kind, None, Form(arg))
                    if kind == 'sin' and s < 0:
                        a = w.neg(a)
                else:
                    a = Form({((v, 1),): 1})
                f = w.mul(f, w.pow(a, e))
            if f.poly is None:
                return None
            w._spend(len(f.poly))
            for t, v in f.poly.items():
                v += out.get(t, 0)
                if v:
                    out[t] = v
                else:
                    out.pop(t, None)
        return out

    def add(self, *ps):
        return self.normal(self.w.lincomb([(Form(p), 1) for p in ps]))

    def sq(self, p):
        return self.normal(self.w.mul(Form(p), Form(p)))

    def variable(self, v):
        return {((v, 1),): 1}

    def trig(self, kind, p):
        return self.normal(self.w.atom(kind, None, Form(p)))


@dataclass
class Certificate:
    kind: str
    lower_bound: float              # valid for every feasible point of the root hull
    details: dict = field(default_factory=dict)
    candidate: object = None        # a proposed point (n,), not yet verified


def scaled_row(row, scale):
    """A RowPool row divided by `scale`: (terms, lower, upper), exact, sides swapped for a negative scale."""
    p = {m: F(c) / scale for m, c in row.terms}
    lo, hi = row.ex_lo, row.ex_hi
    if scale < 0:
        lo, hi = hi, lo
    return p, None if lo is None else lo / scale, None if hi is None else hi / scale


def diameter(g):
    """Proof: K-K lies in the radius-d disk and has at most 2N sides; its perimeter is twice K's, so
    per(K) <= 2Nd sin(pi/(2N)). The polygonal isoperimetric inequality gives area <= per^2/(4N tan(pi/N)), i.e.
    U_N d^2 (insert collinear vertices if the hull has fewer than N). In a half-plane the positive adjacent polar
    triangles cover the hull at most once plus backward ray crossings, each costing at most R^2/2 times its angle, so
    signed area <= hull area + R^2/2 * sum of backward angles; the RowPool order rows bound that sum with their
    tolerance. The closing triangle vanishes at the verified zero vertex. theory/hardfam/hardfam.md."""
    require(g.obj is not None, 'unsupported area expression')
    mapping, area_edges = {}, {}
    for m, c in g.obj.items():
        g.check()
        require(len(m) == 3 and all(e == 1 for _, e in m), 'not adjacent polar area')
        radii = [v for v, _ in m if v < g.n]
        atoms = [v for v, _ in m if v >= g.n]
        require(len(radii) == 2 and len(atoms) == 1, 'not polar triangles')
        kind, _, arg, _ = g.w.atom_info[atoms[0]]
        p = dict(arg)
        require(kind == 'sin' and len(p) == 2 and sorted(p.values()) == [-1, 1], 'not sine angle difference')
        require(all(len(mm) == 1 and mm[0][1] == 1 and mm[0][0] < g.n for mm in p), 'nonlinear angle')
        plus = next(mm[0][0] for mm, v in p.items() if v == 1)
        minus = next(mm[0][0] for mm, v in p.items() if v == -1)
        require(abs(c) == F(1, 2), 'area coefficient')
        edge = (minus, plus) if c < 0 else (plus, minus)
        require(edge not in area_edges, 'duplicate area edge')
        area_edges[edge] = tuple(radii)
    angles = set(v for ab in area_edges for v in ab)
    successor, allowances = {}, {}
    for row in g.pool.rows:
        g.check()
        p = dict(row.terms)
        if len(p) == 2 and sorted(p.values()) == [-1, 1] and all(
                len(m) == 1 and m[0][1] == 1 and m[0][0] in angles for m in p) and row.ex_hi is not None:
            a = next(m[0][0] for m, c in p.items() if c == 1)
            b = next(m[0][0] for m, c in p.items() if c == -1)
            require(a not in successor, 'branched order graph')
            successor[a], allowances[a, b] = b, max(F(0), row.ex_hi)
    starts = angles - set(successor.values())
    require(len(starts) == 1, 'missing angle chain')
    chain = [starts.pop()]
    while chain[-1] in successor:
        require(len(chain) < len(angles), 'cyclic order')
        chain.append(successor[chain[-1]])
    require(set(chain) == angles and set(area_edges) == set(zip(chain, chain[1:])), 'area/order mismatch')
    # adjacent radius pairs share a radius at their common angle
    incident = {a: [] for a in angles}
    for (a, b), rs in area_edges.items():
        incident[a].append(set(rs))
        incident[b].append(set(rs))
    for a in angles:
        intersection = set.intersection(*incident[a])
        if len(intersection) == 1:
            mapping[a] = intersection.pop()
    for a, b in area_edges:
        rs = set(area_edges[a, b])
        if a in mapping and b not in mapping:
            require(mapping[a] in rs and len(rs) == 2, 'inconsistent radius pairing')
            mapping[b] = (rs - {mapping[a]}).pop()
        elif b in mapping and a not in mapping:
            require(mapping[b] in rs and len(rs) == 2, 'inconsistent radius pairing')
            mapping[a] = (rs - {mapping[b]}).pop()
    require(len(mapping) == len(angles) and len(set(mapping.values())) == len(angles), 'ambiguous radius pairing')
    require(all({mapping[a], mapping[b]} == set(rs) for (a, b), rs in area_edges.items()), 'radius/area mismatch')
    N = len(chain)
    require(N >= 3 and all(g.lo[a] >= 0 and g.num(float(g.hi[a])).b <= g.iv.pi.a for a in angles),
            'angles outside half-plane')
    require(all(g.lo[r] >= 0 for r in mapping.values()), 'negative radius')
    R = max(F(float(g.hi[r])) for r in mapping.values())
    require(g.lo[mapping[chain[-1]]] == g.hi[mapping[chain[-1]]] == 0, 'closing vertex not zero')
    cap, seen = F(0), set()
    for row in g.pool.rows:
        g.check()
        p = dict(row.terms)
        if not any(v >= g.n for m in p for v, _ in m):
            continue
        cos = [v for m in p for v, _ in m if v >= g.n and g.w.atom_info[v][0] == 'cos']
        if len(cos) != 1:
            continue
        arg = dict(g.w.atom_info[cos[0]][2])
        if len(arg) != 2 or sorted(arg.values()) != [-1, 1]:
            continue
        if not all(len(m) == 1 and m[0][1] == 1 and m[0][0] < g.n for m in arg):
            continue
        ab = tuple(sorted(m[0][0] for m in arg))
        if not set(ab) <= angles:
            continue
        r, s = mapping[ab[0]], mapping[ab[1]]
        want = g.add({((r, 2),): 1, ((s, 2),): 1},
                     g.normal(g.w.scale(g.w.mul(g.w.mul(Form(g.variable(r)), Form(g.variable(s))),
                                                Form(g.trig('cos', arg))), -2)))
        require(p == want and row.ex_hi is not None and row.ex_hi >= 0 and ab not in seen, 'invalid diameter edge')
        cap = max(cap, row.ex_hi)
        seen.add(ab)
    require(len(seen) == N * (N - 1) // 2, 'incomplete diameter graph')
    U = N / 2 * (g.iv.sin(g.iv.pi / N) - g.iv.tan(g.iv.pi / (2 * N)))
    bound = -U * g.num(cap) - g.num(R * R * sum(allowances.values(), F(0)) / 2)
    # Candidate: a regular odd polygon of diameter <= sqrt(cap) (one vertex at the origin, the rest in the upper
    # half-plane), the general formula for every admitted N. It never affects the bound.
    candidate = (g.lo + g.hi) / 2
    count = N if N % 2 else N - 1
    theta = math.pi * np.arange(1, count) / count
    d = min(float(R), math.sqrt(float(cap)))
    radial = np.minimum(d * np.sin(theta) / math.cos(math.pi / (2 * count)), float(R))
    if count != N:
        theta, radial = np.r_[theta[0], theta], np.r_[radial[0], radial]
    theta, radial = np.r_[theta, g.hi[chain[-1]]], np.r_[radial, 0.]
    for a, t, r in zip(chain, theta, radial):
        candidate[a], candidate[mapping[a]] = t, r
    return Certificate('diameter-area', g.down(bound), dict(vertices=N, edges=len(seen)), candidate=candidate)


def assignment(g):
    """Almost-binary cardinality assignment cover.

    |x(x-1)| <= tau on [0,1] puts x within delta of {0,1}. For an integer cardinality k on m variables,
    m*delta + tau < 1/2 proves the rounding satisfies it exactly, so every tolerance-feasible point rounds to a
    partition of the items into equal groups; it is never assumed binary. Identical group costs allow an unlabeled
    cover: D = min over partitions of the sum of pattern cost lower bounds (pattern_costs), computed by cover().
    theory/hardfam/hardfam.md (grouping)."""
    require(g.obj is not None and all(sum(e for _, e in m) <= 2 and all(v < g.n for v, _ in m) for m in g.obj),
            'objective is not a quadratic polynomial')
    binary, linear = {}, []
    for row in g.pool.rows:
        g.check()
        p, low, high = dict(row.terms), row.ex_lo, row.ex_hi
        if row.ex_lo is None or row.ex_hi is None:
            continue
        if len(p) == 2:
            vs = {v for m in p for v, _ in m}
            if len(vs) == 1:
                v = next(iter(vs))
                if ((v, 2),) in p:
                    p, low, high = scaled_row(row, p[((v, 2),)])
                if p == {((v, 2),): 1, ((v, 1),): -1} and v < g.n and g.lo[v] >= 0 and g.hi[v] <= 1:
                    tau = max(abs(low), abs(high))
                    if tau < F(1, 4):
                        binary[v] = tau
        if p and len(set(p.values())) == 1:
            p, low, high = scaled_row(row, next(iter(p.values())))
        if p and all(len(m) == 1 and m[0][1] == 1 and m[0][0] < g.n and c == 1 for m, c in p.items()):
            k = round(float((low + high) / 2))
            tau = max(abs(low - k), abs(high - k))
            linear.append((tuple(sorted(m[0][0] for m in p)), k, tau))
    require(binary, 'no almost-binary rows')
    items = [vs for vs, k, _ in linear if k == 1 and set(vs) <= binary.keys()]
    groups = [(vs, k, tau) for vs, k, tau in linear if k > 1 and set(vs) <= binary.keys()]
    require(items and groups, 'no assignment/cardinality block')
    n, ng = len(items), len(groups)
    k = groups[0][1]
    require(ng * k == n and all(len(vs) == n and kk == k for vs, kk, _ in groups), 'unequal cardinality groups')
    owner = {v: i for i, vs in enumerate(items) for v in vs}
    require(len(owner) == n * ng and all(len(vs) == ng for vs in items), 'overlapping assignment rows')
    matrix = []
    for vs, _, _ in groups:
        require(set(vs) <= owner.keys() and len({owner[v] for v in vs}) == n, 'not an assignment matrix')
        matrix.append(tuple(sorted(vs, key=lambda v: owner[v])))
    require(len({v for vs in matrix for v in vs}) == n * ng, 'overlapping groups')
    tau = max(binary[v] for v in owner)
    delta = F(g.up(2 * g.num(tau) / (1 + g.iv.sqrt(1 - 4 * g.num(tau)))))
    for vs, kk, t in linear:
        if set(vs) <= owner.keys():
            require(len(vs) * delta + t < F(1, 2), 'rounding does not preserve cardinality')
    # work admission from the combinatorial size, never a name
    require(math.comb(n, k) <= 65536 and n <= 64, 'cover pattern budget')
    by_support = {}
    for f in g.w.squares.values():
        g.check()
        try:
            p = g.normal(f)
        except NoPattern:           # an unsupported square cannot be a group cost
            continue
        if all(not m or len(m) == 1 and m[0][1] == 1 for m in p):
            by_support.setdefault(frozenset(m[0][0] for m in p if m), []).append(p)
    features = []
    for vs in matrix:
        ff = sorted((tuple(F(p.get(((v, 1),), 0)) for v in vs), F(p.get((), 0)))
                    for p in by_support.get(frozenset(vs), []))
        require(ff, 'no squared-affine group objective')
        features.append(ff)
    require(all(ff == features[0] for ff in features), 'nonidentical group costs')
    expected = g.add(*(g.sq({**{((v, 1),): a for v, a in zip(vs, weights)}, (): target})
                       for vs, ff in zip(matrix, features) for weights, target in ff))
    require(expected == g.obj, 'assignment objective has extra terms')
    g.check()
    patterns = combination_ids(n, k, g.device)
    costs = pattern_costs(patterns, features[0], delta, max(t for _, _, t in groups), g.device)
    require(bool(torch.isfinite(costs).all()), 'nonfinite pattern costs')
    return cover(g, patterns, costs, matrix, n)


def pattern_costs(patterns, features, delta, tau, device):
    """Proof: for x = 1_S + e with e in [-delta, 0] on S, [0, delta] elsewhere and |sum e| <= tau, every lambda gives
    a.e <= |lambda| tau + delta sum_S (lambda - a)_+ + delta sum_notS (a - lambda)_+. Minimize over 0 and the
    coefficient breakpoints, independently for +a and -a; the squared distance of each enclosed feature residual from
    zero bounds the group cost. Exact rational coefficients are rounded outward first. (P,) lower bounds."""
    def constant(values):
        return torch.tensor(values, dtype=torch.float64, device=device)
    al = [[round_down(v) for v in aa] for aa, _ in features]
    au = [[round_up(v) for v in aa] for aa, _ in features]
    slopes = [[(F(0), *(-v for v in aa)), (F(0), *aa)] for aa, _ in features]
    pattern = CostPattern(
        patterns, constant(al), constant(au),
        constant([round_down(t) for _, t in features]), constant([round_up(t) for _, t in features]),
        constant([[[round_down(v) for v in ss] for ss in ff] for ff in slopes]),
        constant([[[round_up(v) for v in ss] for ss in ff] for ff in slopes]),
        constant([[[round_up(abs(v)) for v in ss] for ss in ff] for ff in slopes]))
    return pattern.costs(constant([round_up(delta)]), constant([round_up(tau)]))[0]


def cover(g, patterns, costs, matrix, n):
    """The minimum over all partitions of the items into patterns of the pattern cost bounds (CoverPattern), with
    HiGHS item prices as the proposal. An incomplete search keeps only the repaired dual bound. The candidate assigns
    the best cover's patterns to the groups."""
    g.check()
    search = CoverPattern(patterns, n)
    value, path, complete, states = search.solve(costs, lp_prices(patterns, costs, n),
                                                 stop=lambda: time.time() >= g.deadline)
    candidate = (g.lo + g.hi) / 2
    columns = np.asarray(matrix)
    candidate[columns.ravel()] = 0.
    chosen = patterns[path].cpu().numpy()                            # (groups, k) item ids
    candidate[np.take_along_axis(columns, chosen, 1).ravel()] = 1.
    return Certificate('assignment-cover', value, dict(items=n, cardinality=patterns.shape[1],
                                                       patterns=len(patterns), states=states, complete=complete),
                       candidate=candidate)


def detectors():
    from .spectral import spectral
    return dict(diameter=diameter, assignment=assignment, eigenblock=spectral)


def root_certificates(report, kinds, lower, upper, deadline=math.inf):
    """The certificates of `kinds` (names of detectors()) on the root hull of lower/upper (B, ...), from a structure
    report built with SquareForms. Returns (certificates, skipped) and logs one [StructCert] line. A detector that
    declines, expires or fails is recorded in `skipped`; only completed results are returned."""
    start = time.time()
    found, skipped = [], {}
    lo = lower.reshape(lower.shape[0], -1).amin(0).detach().cpu().double().numpy()
    hi = upper.reshape(upper.shape[0], -1).amax(0).detach().cpu().double().numpy()
    w = getattr(report, 'algebra', None)
    if report is None or report.skipped or w is None or not report.forms:
        skipped['graph'] = (report and report.skipped) or 'no exact structure'
    elif not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        skipped['graph'] = 'nonfinite root hull'
    else:
        try:
            g = Graph(w, report.forms, report.pool, lo, hi, deadline, lower.device)
            available = detectors()
            for kind in kinds:
                try:
                    g.check()
                    c = available[kind](g)
                    if math.isfinite(c.lower_bound):
                        found.append(c)
                except (Expired, NoPattern) as e:
                    skipped[kind] = str(e) or type(e).__name__
        except (Expired, NoPattern) as e:
            skipped['graph'] = str(e) or type(e).__name__
    print(f'[StructCert] certificates={[(c.kind, c.lower_bound) for c in found]} '
          f'skip={dict(sorted(skipped.items()))} time={time.time() - start:.3f}s')
    return found, skipped
