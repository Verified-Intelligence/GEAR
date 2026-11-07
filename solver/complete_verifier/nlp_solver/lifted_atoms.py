"""Atom columns of the lifted LP: t = phi(arg) for exp, log, real powers (1/x included), sin and
cos of an affine argument arg = A z + a0 over earlier columns (inputs and other atoms).

phi is monotone with constant curvature on the root range of arg, which is finite (log, x^p need
arg > 0, 1/x one sign). On a box the column t lies between the values at the ends of arg's
range; kappa = +1 (convex) bounds it from below by tangents and from above by the secant,
kappa = -1 (concave) the other way round. Products of atoms with inputs or other atoms are
ordinary product columns of the lifted LP (McCormick on the boxes of their factors).

sin and cos (cos u = sin(u + pi/2): one code, the phase is the exponent) change curvature, so
each box gets lines on both sides, from the support function of sin over the argument's range
[a, b]. For a slope m in [-1, 1] and a side sigma = +1 (upper line) or -1 (lower line),
    sigma (t - m u) <= H(m) = max over [a, b] of sigma (sin u - m u),
which holds for every point of the curve because H is the maximum itself. g = sigma (sin u - m u)
has g'' = -sigma sin u, so its maximum over [a, b] is at an end or where cos u = m and
sigma sin u >= 0, that is u = sigma acos(m) + 2 pi k; there g = sigma (sigma sqrt(1 - m^2) - m u) is
linear in u, so only the first and the last such u in [a, b] compete. H is therefore four sine
evaluations, whatever the width of [a, b] (wider than 2 pi, straddling inflection points, tiny).
Slopes: the tangent slopes cos u at fractions of [a, b] (the line is the tangent where the
tangent is valid on [a, b] and the tightest line of its slope otherwise), and per side the
slopes of the lines through each end that are tight on the whole range (the envelope's tangent
from the end, or the secant when the curve stays on one side), found by bisection on the
predicate "the support line of slope m passes through the end": any slope gives a valid line, so
the bisection only decides how tight it is.

Every evaluation of phi is a float64 libm value (exp, log, pow are accurate to a few ulp)
widened outward by _WIDEN relative: enclosures of values and slopes, the argument's range
(widened by its own summation error and cut to the root range), and a tangent or secant
row only ever moves to the safe side by the resulting error. A column that cannot be
lifted raises NotLifted: the output is then simply not lifted."""
from fractions import Fraction

import numpy as np
import torch

EXP, LOG, POW, SIN = 0, 1, 2, 3
_KINDS = {'exp': (EXP, 0.0), 'log': (LOG, 0.0), 'sin': (SIN, 0.0), 'cos': (SIN, np.pi / 2)}
_WIDEN = 1e-14      # relative outward widening of a float evaluation of phi (a few ulp is 1e-15)
_TINY = 1e-300      # absolute floor, for values that underflow to 0
_ARG_EPS = 2.3e-16  # relative error of one operation of the argument's float sum
_CUSHION = 1e-12    # relative rounding cushion of a row, as in lifted_lp
_TANGENTS = (0.0, 0.25, 0.5, 0.75, 1.0)     # tangent points, as fractions of the argument's range
_TRIG_TANGENTS = (0.25, 0.5, 0.75)    # tangent slopes of sin and cos, as fractions of the argument's range
_BISECT = 24        # steps of the search for the slope of the line through an end of the range
_NARROW = 1e-9      # a range narrower than this (relative) gets a constant secant


class NotLifted(Exception):
    """An atom or monomial outside the liftable class (the output keeps today's path)."""


def _float(c):
    """The float of an exact Fraction/int, which must be that very number."""
    if Fraction(float(c)) != c:
        raise NotLifted('constant not representable')
    return float(c)


def _phi(code, p, x, xp):
    """phi and phi' at x (..., T, P) for atoms with kind `code` (T,) and exponent `p` (T,) (the
    phase of sin and cos)."""
    c, e = code[:, None], p[:, None]
    with np.errstate(all='ignore'):     # the unselected kinds give nan or inf
        ex, lg, pw = xp.exp(x), xp.log(x), x ** e
        val = xp.where(c == EXP, ex, xp.where(c == LOG, lg, xp.where(c == SIN, xp.sin(x + e), pw)))
        slope = xp.where(c == EXP, ex, xp.where(c == LOG, 1.0 / x,
                                                xp.where(c == SIN, xp.cos(x + e), e * pw / x)))
    return val, slope


def _arg_range(A, a0, eps, zl, zu, xp):
    """Outward range (lo, hi) (..., m) of A z + a0 for the column box zl <= z <= zu (..., W)."""
    absf = np.abs if xp is np else torch.abs
    Ap, Am = (A + absf(A)) / 2, (A - absf(A)) / 2
    mag = xp.maximum(absf(zl), absf(zu)) @ absf(A).T + absf(a0)
    lo = zl @ Ap.T + zu @ Am.T + a0 - eps * mag
    hi = zu @ Ap.T + zl @ Am.T + a0 + eps * mag
    return lo, hi


def _value_box(code, p, lo, hi, xp):
    """Box (tl, tu) of phi over [lo, hi] (..., T): phi is monotone, so its end values; sin also
    reaches +-1 where the range contains pi/2 or -pi/2 modulo 2 pi."""
    v = _phi(code, p, xp.stack([lo, hi], -1), xp)[0]
    absf = np.abs if xp is np else torch.abs
    w = _WIDEN * absf(v) + _TINY
    tl = xp.minimum(v[..., 0], v[..., 1]) - xp.maximum(w[..., 0], w[..., 1])
    tu = xp.maximum(v[..., 0], v[..., 1]) + xp.maximum(w[..., 0], w[..., 1])
    if not (code == SIN).any():
        return tl, tu
    a, b = lo + p, hi + p
    peak = xp.ceil((a - np.pi / 2) / (2 * np.pi)) * (2 * np.pi) + np.pi / 2 <= b + 1e-12 * (1 + absf(b))
    trough = xp.ceil((a + np.pi / 2) / (2 * np.pi)) * (2 * np.pi) - np.pi / 2 <= b + 1e-12 * (1 + absf(b))
    sin = code == SIN
    return (xp.where(sin & trough, -1.0, tl), xp.where(sin & peak, 1.0, tu))


def _support(m, sigma, a, b, xp):
    """H(m) = max over [a, b] of sigma (sin u - m u), for slopes m in [-1, 1] (module docstring):
    the values at both ends and at the first and the last u = sigma acos(m) + 2 pi k in [a, b]
    (a candidate outside [a, b] is moved to the nearest end, which is a candidate anyway)."""
    crit = sigma * xp.arccos(m.clip(-1.0, 1.0))
    first = crit + 2 * np.pi * xp.ceil((a - crit) / (2 * np.pi))
    last = crit + 2 * np.pi * xp.floor((b - crit) / (2 * np.pi))
    h = None
    for u in (a + 0 * m, b + 0 * m, first, last):
        u = xp.minimum(xp.maximum(u, a), b)
        v = sigma * (xp.sin(u) - m * u)
        h = v if h is None else xp.maximum(h, v)
    return h


def _end_slope(a, b, sigma, left, xp):
    """Slope of the line of side sigma through the end (a if left, else b) of [a, b] that is
    as tight as possible, by bisection on "the support line of slope m touches the end"; the
    result is only a slope, so the rows built on it are valid whatever the bisection ends at."""
    e = a if left else b
    d = sigma if left else -sigma       # larger d m: the line passes the end more easily
    lo_, hi_ = -xp.ones_like(a), xp.ones_like(a)
    for _ in range(_BISECT):
        mid = (lo_ + hi_) / 2
        m = (d * mid)[..., None]
        gap = _support(m, sigma, a[..., None], b[..., None], xp)[..., 0] - sigma * (xp.sin(e) - d * mid * e)
        ok = gap <= 1e-11 * (1 + xp.abs(e))
        lo_, hi_ = xp.where(ok, lo_, mid), xp.where(ok, mid, hi_)
    return d * hi_


class LiftedAtoms:
    """The atom columns n .. n + T of a LiftedLP, built (registered) from forms.py atoms and
    then used per box. After building, `A` (T, n + T) holds the argument rows (zero beyond the
    earlier columns) and `a0` the constants."""

    def __init__(self, n, lower, upper, atom_info):
        self.n, self.info = n, atom_info
        self.lower, self.upper = list(lower), list(upper)    # root box of every column so far
        self.of, self.key = {}, {}
        self.args, self.kind, self.expo, self.kappa, self.root = [], [], [], [], []
        self.points = []    # per atom: arguments of the root LP's points, extra tangent slopes of sin and cos
        self._np, self._torch, self._csr = None, {}, None

    @property
    def T(self):
        return len(self.kind)

    # ---- registration -----------------------------------------------------------------
    def columns(self, mono):
        """Base columns (with multiplicity) of a forms.py monomial ((var, exponent), ...)."""
        out = []
        for v, e in mono:
            if v >= self.n:
                if not (type(e) is int and e > 0):
                    raise NotLifted('atom exponent')
                out += [self._atom(v)] * e
            elif type(e) is int and e > 0:
                out += [v] * e
            else:
                out.append(self._add(POW, _float(Fraction(e)), {v: Fraction(1)}, Fraction(0)))
        return tuple(out)

    def _atom(self, v):
        if v not in self.of:
            kind, param, arg, _ = self.info[v]
            if kind not in ('exp', 'log', 'pow', 'sin', 'cos'):
                raise NotLifted(kind)
            coef, a0 = {}, Fraction(0)
            for mono, c in arg:
                if not mono:
                    a0 = Fraction(c)
                    continue
                cols = self.columns(mono) if len(mono) == 1 else ()
                if len(cols) != 1:
                    raise NotLifted('argument is not affine')
                coef[cols[0]] = coef.get(cols[0], Fraction(0)) + c
            code, p = _KINDS[kind] if kind != 'pow' else (POW, _float(Fraction(param)))
            self.of[v] = self._add(code, p, coef, a0)
        return self.of[v]

    def _add(self, code, p, coef, a0):
        cols = sorted(coef)
        coefs = [_float(coef[c]) for c in cols]
        key = (code, p, tuple(cols), tuple(coefs), _float(a0))
        if key in self.key:
            return self.key[key]
        W = self.n + self.T
        A = np.zeros((1, W))
        A[0, cols] = coefs
        lo, hi = (t[0] for t in _arg_range(A, np.array([key[4]]), np.array([_ARG_EPS * (len(cols) + 2)]),
                                          np.array(self.lower), np.array(self.upper), np))
        code_a, p_a = np.array([code]), np.array([p])
        if not (np.isfinite([lo, hi]).all() and lo <= hi):
            raise NotLifted('argument range')
        if code == POW and p > 0 and -1e-9 * max(1.0, hi) <= lo < 0:
            lo = 0.0    # the domain of a real power; lo < 0 only by the widening (ex8_3_14: -7e-14)
        if code == SIN:
            kappa = 0       # no fixed curvature: lines on both sides, see _trig_rows
        elif code == EXP:
            kappa = 1
        elif code == LOG and lo > 0:
            kappa = -1
        elif code == POW and (lo > 0 or lo == 0 and p > 0):
            kappa = -1 if 0 < p < 1 else 1
        elif code == POW and hi < 0 and p == -1:
            kappa = -1
        else:
            raise NotLifted(f'domain or curvature: kind {code}, exponent {p}, range [{lo}, {hi}]')
        tl, tu = (t[0] for t in _value_box(code_a, p_a, np.array([lo]), np.array([hi]), np))
        if not np.isfinite([tl, tu]).all():
            raise NotLifted('atom range')
        self.key[key] = W
        self.args.append((cols, coefs, key[4]))
        self.kind.append(code); self.expo.append(p); self.kappa.append(kappa)
        self.root.append((lo, hi))
        self.points.append([])
        self.lower.append(tl); self.upper.append(tu)
        self._np = None
        return W

    def restrict(self, j, lo, hi):
        """Cut the root range of atom j's argument to [lo, hi] (a certified range over the box and
        the relaxed linear rows), which every box's range is cut to as well: the atom's rows then
        hold for the points satisfying those rows, which is all a lower bound has to cover."""
        self.root[j] = (max(self.root[j][0], lo), min(self.root[j][1], hi))
        self._np = None

    def add_points(self, z):
        """Add a tangent slope to every sin and cos atom: the argument of the LP point z (nb,)
        (clipped into the atom's range per box). One point per round of the root refinement."""
        c = self.arrays()
        for j, v in zip(c['trig'], z[:self.n + self.T] @ c['A'][c['trig']].T + c['a0'][c['trig']]):
            self.points[j].append(float(v))
        self._np = None

    def rollback(self, T):
        """Forget the atoms registered after the first T (an output that turned out not liftable)."""
        for k in [k for k, c in self.key.items() if c >= self.n + T]:
            del self.key[k]
        for v in [v for v, c in self.of.items() if c >= self.n + T]:
            del self.of[v]
        for lst in (self.args, self.kind, self.expo, self.kappa, self.root, self.points):
            del lst[T:]
        del self.lower[self.n + T:], self.upper[self.n + T:]
        self._np = None

    # ---- arrays -----------------------------------------------------------------------
    def arrays(self, xp=np, like=None):
        """The constants as numpy arrays, or as tensors on like's device and dtype."""
        if self._np is None:
            T, nb = self.T, self.n + self.T
            A = np.zeros((T, nb))
            for j, (cols, coefs, _) in enumerate(self.args):
                A[j, cols] = coefs
            depth = np.zeros(nb, dtype=int)
            for j, (cols, _, _) in enumerate(self.args):
                depth[self.n + j] = 1 + max([depth[c] for c in cols if c >= self.n], default=0)
            support = np.zeros((T, self.n), dtype=bool)
            for j, (cols, _, _) in enumerate(self.args):
                for c in cols:
                    if c < self.n:
                        support[j, c] = True
                    else:
                        support[j] |= support[c - self.n]
            kappa = np.array(self.kappa, dtype=float)
            trig = np.array(self.kind) == SIN
            mono, trig = np.nonzero(~trig)[0], np.nonzero(trig)[0]
            # rows, monotone atoms first: sgn (-t + slope A z) <= rhs, the row's atom is `owner`:
            # P tangents (convex side) and the secant; for a trig atom the upper (sgn -1) and
            # then the lower lines (sgn +1): tangent slopes, then those through the two ends
            nt = len(_TRIG_TANGENTS) + 2 + (len(self.points[trig[0]]) if len(trig) else 0)
            owner = np.concatenate([np.repeat(mono, len(_TANGENTS) + 1), np.repeat(trig, 2 * nt)])
            sgn = np.concatenate([
                np.concatenate([np.repeat(kappa[mono, None], len(_TANGENTS), 1), -kappa[mono, None]], 1).reshape(-1),
                np.tile(np.repeat([-1.0, 1.0], nt), len(trig))])
            points = np.zeros((T, nt - len(_TRIG_TANGENTS) - 2))
            for j in trig:
                points[j] = self.points[j]
            self._torch, self._csr = {}, None
            self._np = dict(points=points, mono=mono, trig=trig, owner=owner, sgn=sgn,
                code=np.array(self.kind), expo=np.array(self.expo), kappa=kappa, A=A,
                a0=np.array([a[2] for a in self.args]),
                eps=_ARG_EPS * ((A != 0).sum(1) + 2),
                lo=np.array([r[0] for r in self.root]), hi=np.array([r[1] for r in self.root]),
                levels=[np.nonzero(depth[self.n:] == d)[0] for d in range(1, depth.max() + 1)],
                support=support)
        if xp is np:
            return self._np
        key = (like.device, like.dtype)
        if key not in self._torch:
            self._torch[key] = {
                k: ([torch.as_tensor(i, device=like.device) for i in v] if k == 'levels'
                    else torch.as_tensor(v, device=like.device,
                                         dtype=torch.bool if k == 'support' else
                                         torch.long if k in ('code', 'mono', 'trig', 'owner') else like.dtype))
                for k, v in self._np.items()}
        return self._torch[key]

    # ---- per box ----------------------------------------------------------------------
    def _range(self, c, ks, zl, zu, xp):
        """Range of the arguments of atoms ks, cut to the root range, ends in order."""
        lo, hi = _arg_range(c['A'][ks], c['a0'][ks], c['eps'][ks], zl, zu, xp)
        lo, hi = xp.maximum(lo, c['lo'][ks]), xp.minimum(hi, c['hi'][ks])
        return lo, xp.maximum(lo, hi)

    def fill_boxes(self, lz, uz, xp):
        """Boxes of the atom columns, in place, level by level (arguments before atoms)."""
        c, n = self.arrays(xp, lz), self.n
        nb = n + self.T
        for ks in c['levels']:
            lo, hi = self._range(c, ks, lz[..., :nb], uz[..., :nb], xp)
            lz[..., n + ks], uz[..., n + ks] = _value_box(c['code'][ks], c['expo'][ks], lo, hi, xp)

    def rows(self, lz, uz, xp):
        """Slopes and right-hand sides (..., R) of the atom rows sgn * (-t + slope * (A z)) <= rhs,
        row r belonging to atom `owner[r]` (see arrays): the rows of the monotone atoms, then
        those of sin and cos."""
        c, n = self.arrays(xp, lz), self.n
        nb = n + self.T
        absf = np.abs if xp is np else torch.abs
        lo, hi = self._range(c, slice(None), lz[..., :nb], uz[..., :nb], xp)
        # |A z| + |a0| and |t| over the box, for the rounding cushions
        M = xp.maximum(absf(lz), absf(uz))[..., :nb]
        marg, mt = M @ absf(c['A']).T + absf(c['a0']), M[..., n:]
        cat = np.concatenate if xp is np else torch.cat
        out = []
        for ks, rows in ((c['mono'], self._monotone_rows), (c['trig'], self._trig_rows)):
            slope, rhs, scale = rows(c, ks, lo[..., ks], hi[..., ks], marg[..., ks], mt[..., ks], xp)
            rhs = rhs + _CUSHION * (scale + absf(rhs))
            out.append((slope.reshape(*slope.shape[:-2], -1), rhs.reshape(*rhs.shape[:-2], -1)))
        return cat([o[0] for o in out], -1), cat([o[1] for o in out], -1)

    def _monotone_rows(self, c, ks, lo, hi, marg, mt, xp):
        """(slope, rhs, cushion scale) (..., T', P + 1) for atoms ks of a fixed curvature: P
        tangents (on the convex side) and the secant."""
        absf = np.abs if xp is np else torch.abs
        code, expo, kappa, a0 = c['code'][ks], c['expo'][ks], c['kappa'][ks], c['a0'][ks]
        frac = lo.new_tensor(_TANGENTS) if xp is not np else np.array(_TANGENTS)
        with np.errstate(all='ignore'):
            sign = xp.where(hi < 0, -1.0, 1.0)
            geometric = sign[..., None] * absf(lo)[..., None] * (absf(hi) / absf(lo))[..., None] ** frac
            p = xp.where(((lo > 0) | (hi < 0))[..., None], geometric,
                         lo[..., None] + frac * (hi - lo)[..., None])
        # a tangent at the end 0 of a power's range has an infinite slope (sqrt): start 1/64 in
        p = xp.maximum(p, xp.where(lo <= 0, lo + (hi - lo) / 64, lo)[..., None])
        p = xp.minimum(xp.maximum(p, lo[..., None]), hi[..., None])
        val, slope = _phi(code, expo, p, xp)
        bad = ~(xp.isfinite(val) & xp.isfinite(slope))      # an empty range at the pole: no tangent
        val, slope = xp.where(bad, 0.0, val), xp.where(bad, 0.0, slope)
        reach = xp.maximum(absf(lo[..., None] - p), absf(hi[..., None] - p))
        err = _WIDEN * absf(val) + _TINY + _WIDEN * absf(slope) * reach
        a0_, k_ = a0[:, None], kappa[:, None]
        rhs_t = xp.where(bad, float('inf'), k_ * (slope * (p - a0_) - val) + err)
        scale_t = 1 + absf(val) + absf(slope) * (absf(p) + marg[..., None]) + mt[..., None]
        # secant through the end values on the safe side (upper for convex, lower for concave)
        ends, _ = _phi(code, expo, xp.stack([lo, hi], -1), xp)
        side = ends + k_ * (_WIDEN * absf(ends) + _TINY)
        width = hi - lo
        narrow = width <= _NARROW * (xp.maximum(absf(lo), absf(hi)) + _TINY)
        extreme = xp.where(kappa > 0, xp.maximum(side[..., 0], side[..., 1]),
                           xp.minimum(side[..., 0], side[..., 1]))
        with np.errstate(all='ignore'):
            sigma = xp.where(narrow, 0.0, (side[..., 1] - side[..., 0]) / width)
        base = xp.where(narrow, extreme, side[..., 0] + sigma * (a0 - lo))
        rhs_s = kappa * base
        scale_s = (1 + absf(side[..., 0]) + absf(side[..., 1]) + mt
                   + absf(sigma) * (absf(lo) + marg))
        cat = np.concatenate if xp is np else torch.cat
        return (cat([slope, sigma[..., None]], -1), cat([rhs_t, rhs_s[..., None]], -1),
                cat([scale_t, scale_s[..., None]], -1))

    def _trig_rows(self, c, ks, lo, hi, marg, mt, xp):
        """(slope, rhs, cushion scale) (..., T', 2 (P + 2)) of the sin and cos atoms ks: for the
        upper side and then the lower side, the tangent slopes at P fractions of the argument's
        range and the slopes of the lines through its two ends, with the offsets H of the
        module docstring (sigma (t - m (A z)) <= H + sigma m (a0 + phase) plus a rounding error)."""
        absf = np.abs if xp is np else torch.abs
        phase, a0 = c['expo'][ks], c['a0'][ks]
        a, b = lo + phase, hi + phase
        frac = lo.new_tensor(_TRIG_TANGENTS) if xp is not np else np.array(_TRIG_TANGENTS)
        cat = np.concatenate if xp is np else torch.cat
        pts = c['points'][ks] + phase[:, None] + 0 * a[..., None]
        u = cat([a[..., None] + frac * (b - a)[..., None], pts], -1)
        tangent = xp.cos(xp.minimum(xp.maximum(u, a[..., None]), b[..., None]))
        reach = xp.maximum(absf(a), absf(b))
        slopes, rhs, scale = [], [], []
        for sigma in (1.0, -1.0):
            ends = xp.stack([_end_slope(a, b, sigma, True, xp), _end_slope(a, b, sigma, False, xp)], -1)
            m = cat([tangent, ends], -1)
            err = _WIDEN * (1 + absf(m) * reach[..., None])
            H = _support(m, sigma, a[..., None], b[..., None], xp)
            shift = sigma * m * (a0 + phase)[..., None]
            slopes.append(m)
            rhs.append(H + shift + err)
            scale.append(1 + absf(H) + absf(m) * (absf(a0 + phase)[..., None] + marg[..., None]) + mt[..., None])
        return cat(slopes, -1), cat(rhs, -1), cat(scale, -1)

    def dense(self, lz, uz, N):
        """(G (B, R, N), h (B, R)) of the atom rows for the column boxes lz, uz (B, nb + K)."""
        c, n = self.arrays(torch, lz), self.n
        slope, rhs = self.rows(lz, uz, torch)
        T, nb = self.T, n + self.T
        E = torch.zeros(T, nb, dtype=lz.dtype, device=lz.device)
        E[torch.arange(T, device=lz.device), n + torch.arange(T, device=lz.device)] = 1.0
        G = c['sgn'][None, :, None] * (slope[..., None] * c['A'][c['owner']][None] - E[c['owner']][None])
        return torch.nn.functional.pad(G, (0, N - nb)), rhs

    def sparse(self, lz, uz, N):
        """(G csr (R, N), h (R,)) of the atom rows for one box (numpy)."""
        import scipy.sparse as sp
        c, n = self.arrays(np), self.n
        slope, rhs = self.rows(lz, uz, np)
        nb, R = n + self.T, self.n_rows
        if self._csr is None:
            self._csr = (sp.csr_matrix(c['A'])[c['owner']],
                         sp.csr_matrix((np.ones(R), (np.arange(R), n + c['owner'])), shape=(R, nb)))
        G = sp.diags(c['sgn'] * slope) @ self._csr[0] - sp.diags(c['sgn']) @ self._csr[1]
        return sp.hstack([G, sp.csr_matrix((R, N - nb))]).tocsr(), rhs

    @property
    def n_rows(self):
        return len(self.arrays()['owner'])

    def residual(self, z, xp):
        """|t - phi(arg)| (..., T) at lifted points z (..., N): the violation of each atom."""
        c, n = self.arrays(xp, z), self.n
        nb = n + self.T
        arg = z[..., :nb] @ c['A'].T + c['a0']
        arg = xp.minimum(xp.maximum(arg, c['lo']), c['hi'])
        val, _ = _phi(c['code'], c['expo'], arg[..., None], xp)
        return (z[..., n:nb] - val[..., 0]).abs() if xp is torch else np.abs(z[..., n:nb] - val[..., 0])
