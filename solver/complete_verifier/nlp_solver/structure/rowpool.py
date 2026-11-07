"""RowPool: the rows of one instance that a bound may use, once, with a stable id each and the verify predicate's
tolerance applied once.

A row is  lo <= q <= hi  with q the NON-constant part of an exact polynomial (an equality is one ranged row with both
sides finite, its constant folded into the bounds). The pool holds

  * the spec rows C @ out <= rhs + tau (source 'spec', id = the row's index; row 0 is the objective, not a row) and the
    problem's own linear rows A x <= b + tau (source 'base', ids after the spec rows);
  * parallel rows merged exactly: rows whose non-constant parts are positive or negative multiples of one direction
    (the same monomials with proportional exact coefficients) become ONE row. The representative is the member with
    the smallest id; every member's bound maps by its exact scale into the representative's units (hi = min,
    lo = max, in rationals) and is rounded outward once. A spec row that restates a base row merges with it;
  * a row implied by the root box is kept, with its id, but masked.

tau is applied ONCE, here. It comes from the verify predicate (`spec_tau`, `base_tau`: the same tolerance the primal and
the domain filter accept), and every bound consumer reads the relaxed thresholds from `thresholds`; none relaxes a
right-hand side itself. A bound built on the raw rhs cuts off points the verify predicate accepts (the ex7_2_1 class:
D rose above the optimum of the relaxed problem), and a bound built on an already relaxed rhs is relaxed twice.
`audit` proves that every original row got exactly its tau. `thresholds` is rhs + tau in the tensor's own dtype, one spec row
per column, whatever the merged rows are; the exact bounds `lo` / `hi` are the same numbers rounded outward once, and are
what a consumer that takes rows rather than thresholds reads. The standard-form copy of the base rows that the root pass and
the BaB read is relaxed where the rows are born, in the spec handler, before any model exists
(input_split.utils.relax_input_constraints); the pool records the same tolerance for its own base rows.

Ids are stable and never reused, so a dual state indexed by id stays valid when rows are added or masked. Rows that
other producers add later (symmetry rows, cuts) are not in the pool yet: each carries the tolerance of the row it was
derived from.
"""
from dataclasses import dataclass, field
from fractions import Fraction as F
from math import gcd, isfinite

import numpy as np
import torch

INF = float('inf')
_MAX = float(np.finfo(np.float64).max)


def round_up(x):
    """the float >= the Fraction x (+inf beyond the float range)"""
    try:
        f = float(x)
    except OverflowError:
        return INF if x > 0 else -_MAX
    return f if F(f) >= x else float(np.nextafter(f, INF))


def round_down(x):
    """the float <= the Fraction x (-inf beyond the float range)"""
    try:
        f = float(x)
    except OverflowError:
        return _MAX if x > 0 else -INF
    return f if F(f) <= x else float(np.nextafter(f, -INF))


def _direction(nz):
    """exact direction of the non-constant part: the coefficients as integers with no common factor and a positive
    first one. Equal keys <=> the two parts are (positive or negative) multiples of each other."""
    den = 1
    for _, c in nz:
        if type(c) is not int:
            d = c.denominator
            if den % d:
                den = den * d // gcd(den, d)
    ints = [c * den if type(c) is int else c.numerator * (den // c.denominator) for _, c in nz]
    g = 0
    for v in ints:
        g = gcd(g, v)
    if ints[0] < 0:
        g = -g
    return tuple((m, v // g) for (m, _), v in zip(nz, ints))


@dataclass
class Row:
    id: int
    source: str                 # 'spec' or 'base'
    terms: tuple                # ((monomial, exact coefficient), ...) of the representative's non-constant part
    ex_lo: object               # exact Fraction bounds of that part (tau included, constant folded in), None for no side
    ex_hi: object
    lo: float                   # the same, rounded outward once; -inf / +inf for no side
    hi: float
    tau: float                  # the representative's tolerance
    members: list = field(default_factory=list)   # (id, rho = s_rep / s_member, bound with tau), exact; None: opaque
    masked: bool = False        # implied by the root box
    opaque: bool = False        # no exact polynomial, or a threshold that is not a number: kept alone, no bounds


def spec_tau(m):
    """The verify predicate's tolerance of each of the m spec rows (row 0, the objective, has none): the one place that
    reads it for spec rows. No-op outside solving mode."""
    from input_split.utils import relax_constraint_rhs
    return relax_constraint_rhs(torch.zeros(1, m, dtype=torch.float64))[0].tolist()


def base_tau(rows):
    """the same for the problem's own linear rows A x <= b"""
    from input_split.utils import relax_input_constraint_rhs
    return relax_input_constraint_rhs(torch.zeros(rows, dtype=torch.float64)).tolist()


def _group_row(members):
    """the Row of one parallel group; members: (id, s, beta, non-constant items, redundant, source, tau), s the first
    coefficient and beta the bound with tau of the member's non-constant part"""
    members.sort(key=lambda t: t[0])
    rid, s_rep, _, nz, _, source, tau = members[0]
    hi = lo = None
    recs = []
    for mid, s, beta, *_ in members:
        rho = s_rep / s
        recs.append((mid, rho, beta))
        if rho > 0:
            hi = rho * beta if hi is None else min(hi, rho * beta)
        else:
            lo = rho * beta if lo is None else max(lo, rho * beta)
    return Row(rid, source, tuple(nz), lo, hi, -INF if lo is None else round_down(lo),
               INF if hi is None else round_up(hi), tau, recs, masked=all(t[4] for t in members))


class RowPool:
    def __init__(self, rhs, base_rows=None):
        """rhs: the raw thresholds of the spec rows, (m,) or (B, m) with the same rows in every domain (row 0, the
        objective's cutoff, is ignored); base_rows: the raw rows A x <= b as ((R, n), (R,)). Every row is a singleton
        until `merge` gives the spec rows their exact polynomials."""
        rhs = torch.as_tensor(rhs).detach().cpu().double()
        raw = rhs.reshape(-1, rhs.shape[-1])[0].tolist()
        self.m = len(raw)
        self.tau = spec_tau(self.m)
        self._rows = [(k, 'spec', raw[k], self.tau[k], None) for k in range(1, self.m)]
        if base_rows is not None and torch.as_tensor(base_rows[1]).numel():
            A = np.asarray(torch.as_tensor(base_rows[0]).detach().cpu().double())
            b = torch.as_tensor(base_rows[1]).detach().cpu().double().reshape(-1).tolist()
            A = A.reshape(len(b), -1)
            tau = base_tau(len(b))
            for i in range(len(b)):
                nz = np.flatnonzero(A[i])
                self._rows.append((self.m + i, 'base', b[i], tau[i], {((int(j), 1),): F(float(A[i, j])) for j in nz}))
        self._tau_like = {}
        self._build({}, {})

    @property
    def tolerance(self):
        """the verify predicate's tolerance as a scalar, for a consumer that relaxes rows it derives itself (the square-row
        cuts and the objective alias floor carry the tolerance of the spec row they come from)"""
        return max(self.tau, default=0.0)

    def merge(self, polys, redundant):
        """polys[k]: the exact polynomial {monomial: coefficient} of spec row k (None: unknown or inexact); redundant[k]:
        row k is implied by the root box. Groups the parallel rows, base rows included."""
        self._build({k: p for k, p in enumerate(polys) if k and p is not None}, {k: bool(r) for k, r in enumerate(redundant)})

    def _build(self, polys, redundant):
        """All rows from the original ones, replacing the pool's rows only once every group is built"""
        rows, raw_beta, groups = [], {}, {}
        for rid, source, raw, tau, poly in self._rows:
            poly = polys.get(rid, poly)
            nz = None if poly is None else sorted((m, c) for m, c in poly.items() if m)
            if not nz or not isfinite(raw):
                rows.append(Row(rid, source, (), None, None, -INF, INF, tau, [(rid, F(1), None)], opaque=True))
                continue
            c0 = F(poly.get((), 0))
            raw_beta[rid] = F(raw) - c0
            groups.setdefault(_direction(nz), []).append(
                (rid, F(nz[0][1]), F(raw) + F(tau) - c0, nz, redundant.get(rid, False), source, tau))
        rows += [_group_row(members) for members in groups.values()]
        rows.sort(key=lambda r: r.id)
        self.rows, self._raw_beta = rows, raw_beta
        self.row_of = {mid: i for i, row in enumerate(rows) for mid, *_ in row.members}

    # ------------------------------------------------------------------------------------------ views
    def thresholds(self, rhs):
        """The spec thresholds as every bound must see them: rhs (..., m) with the constraint rows (1:) relaxed by the
        verify predicate's tolerance, once. Row 0, the objective's cutoff, is the caller's and stays as given. The rows
        keep their order: a bound consumer works on the spec rows one by one, whatever the merged rows are."""
        if rhs is None or not torch.is_tensor(rhs) or rhs.ndim < 2 or rhs.shape[-1] < 2 or not any(self.tau):
            return rhs
        if rhs.shape[-1] != self.m:
            raise ValueError(f'{rhs.shape[-1]} thresholds for a pool of {self.m} spec rows')
        key = (rhs.device, rhs.dtype)
        if key not in self._tau_like:
            self._tau_like[key] = torch.tensor(self.tau[1:], dtype=torch.float64).to(rhs)
        out = rhs.clone()
        out[..., 1:] = rhs[..., 1:] + self._tau_like[key]
        return out

    def derived_cut_view(self, lower, upper, cutoff):
        """One applicable scoped input aggregate per domain, as A x <= b.

        These RHS values are already outward and carry source-row tolerances.
        They never enter _rows/merge/thresholds and never get another tau.
        A sibling or increased cutoff gets the inactive row 0 <= 0.
        """
        registry = getattr(self, 'derived_cut_registry', None)
        if registry is None:
            return None
        B, n = len(lower), lower.flatten(1).shape[1]
        A, b = lower.new_zeros(B, 1, n), lower.new_zeros(B, 1)
        selected = torch.zeros(B, dtype=torch.bool, device=lower.device)
        if not hasattr(self, '_derived_like'):
            self._derived_like = {}
        # Newer scopes are descendants or strengthened versions of their ancestors.
        for scope, mask in reversed(registry.references(lower, upper, cutoff)):
            if scope.input_cut is None:
                continue
            mask = mask & ~selected
            if not mask.any():
                continue
            key = (id(scope), lower.dtype, lower.device)
            if key not in self._derived_like:
                raw_A, raw_b = scope.input_cut
                aa = torch.as_tensor(raw_A).to(dtype=lower.dtype)
                # Account exactly for converting input coefficients to the consumer dtype.
                rhs = F(float(raw_b)) + sum(
                    max((F(float(new))-F(float(old)))*F(float(lo)),
                        (F(float(new))-F(float(old)))*F(float(hi)))
                    for old, new, lo, hi in zip(raw_A, aa, scope.scope_lower, scope.scope_upper))
                bb = torch.tensor(round_up(rhs), dtype=lower.dtype)
                if F(float(bb)) < rhs:
                    bb = torch.nextafter(bb, torch.full_like(bb, INF))
                self._derived_like[key] = aa.to(lower.device), bb.to(lower.device)
            aa, bb = self._derived_like[key]
            A[mask, 0], b[mask, 0] = aa, bb
            selected |= mask
        return (A, b) if selected.any() else None

    def stats(self):
        return dict(rows=len(self.rows), ranged=sum(r.ex_lo is not None and r.ex_hi is not None for r in self.rows),
                    merged=sum(len(r.members) - 1 for r in self.rows), masked=sum(r.masked for r in self.rows),
                    opaque=sum(r.opaque for r in self.rows))

    def audit(self, expected_tau):
        """tau applied exactly once: for every original row of every non-opaque row, beta - beta_raw is the verify
        predicate's tolerance of that row (expected_tau: id -> float, from an independent call), the row's exact bounds
        are the extremes of rho * beta over its members, and its float bounds are those rounded outward once. Returns the
        list of violations."""
        bad = []
        for row in self.rows:
            if row.opaque:
                continue
            hi = lo = None
            for mid, rho, beta in row.members:
                if beta - self._raw_beta[mid] != F(expected_tau[mid]):
                    bad.append(('tau', mid))
                if rho > 0:
                    hi = rho * beta if hi is None else min(hi, rho * beta)
                else:
                    lo = rho * beta if lo is None else max(lo, rho * beta)
            if (hi, lo) != (row.ex_hi, row.ex_lo):
                bad.append(('exact', row.id))
            want_hi = INF if hi is None else round_up(hi)
            want_lo = -INF if lo is None else round_down(lo)
            if (row.hi, row.lo) != (want_hi, want_lo):
                bad.append(('float', row.id))
        return bad
