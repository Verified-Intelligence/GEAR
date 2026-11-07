"""Exact forms of tensor elements: sparse polynomials over input variables and interned
atoms (exp, log, non-polynomial powers, ...), with int/Fraction coefficients.

A Form with poly=None is outside the representable class; `support` and `reason` say
which inputs it depends on and why. `inexact` marks coefficients rounded from a
transcendental constant. Work is bounded by an operation budget, never by wall time,
so the same model always gives the same forms."""
import math
from auto_LiRPA.cap_log import cap_log
from fractions import Fraction

MAX_TERMS = 200000          # terms per form
MAX_PRODUCT = 4000000       # |a| * |b| per product
DEFAULT_BUDGET = 50_000_000  # term operations per FormAlgebra


class BudgetExceeded(Exception):
    """The operation budget of a FormAlgebra is used up."""


def exact(x):
    """Exact rational of a float/int/Fraction; a plain int when integral (int arithmetic
    is ~10x faster than Fraction). A float keeps its exact binary value."""
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x) if x.is_integer() else Fraction(x)
    return int(x) if x.denominator == 1 else x


class Form:
    """poly: {monomial: coefficient} with monomial = tuple of (var, exponent) sorted by var
    (var < n is an input coordinate, var >= n an interned atom), or None when the element
    is not representable. support: input indices a non-representable element depends on
    (None = unknown/all). reason: first reason the element left the representable class."""
    __slots__ = ('poly', 'support', 'reason', 'inexact')

    def __init__(self, poly, support=None, reason=None, inexact=False):
        self.poly, self.support, self.reason, self.inexact = poly, support, reason, inexact


ZERO = Form({})


def flagged(a, inexact):
    """a, marked inexact when inexact: a rounded zero operand still taints the result"""
    return a if a.inexact or not inexact else Form(a.poly, inexact=True)


def constant(value, inexact=False):
    if value == 0:
        return Form({}, inexact=True) if inexact else ZERO
    return Form({(): exact(value)}, inexact=inexact)


def constant_value(form):
    """The constant value of a form (Fraction or int), or None."""
    if form.poly is None:
        return None
    if not form.poly:
        return Fraction(0)
    if len(form.poly) == 1 and () in form.poly:
        return form.poly[()]
    return None


def _rational_power(c, k):
    """c**k for Fraction c > 0 and Fraction k: exact when possible, else a rounded float
    (flagged inexact). Returns (value, inexact)."""
    if k.denominator == 1:
        return exact(Fraction(c) ** int(k)), False
    q = k.denominator
    n, d = c.numerator, c.denominator
    rn, rd = round(n ** (1.0 / q)), round(d ** (1.0 / q))
    if rn ** q == n and rd ** q == d:
        return exact(Fraction(rn, rd) ** int(k.numerator)), False
    return exact(Fraction(float(c) ** float(k))), True


def _monomial_product(a, b):
    if not a:
        return b
    if not b:
        return a
    d = dict(a)
    for v, e in b:
        x = d.get(v, 0) + e
        if x:
            d[v] = x
        else:
            d.pop(v, None)
    return tuple(sorted(d.items()))


class FormAlgebra:
    """Ring operations on Forms over n input variables with lower box bounds
    (None = unknown). atom_info: atom id -> (kind, param, poly items, support);
    poles: variables raised to a negative power (-1: a pole of a sum)."""

    def __init__(self, n, lower, budget=DEFAULT_BUDGET):
        self.n, self.lower, self.budget = n, lower, budget
        self.atoms, self.atom_info = {}, {}
        self.poles = set()
        self.spent = 0
        self._expanded = {}

    def _spend(self, cost):
        self.spent += cost
        if self.spent > self.budget:
            cap_log('structure.forms', 'term_ops', self.budget, self.spent, 'form unknown (budget)')
            raise BudgetExceeded

    # ---- helpers -------------------------------------------------------------------------
    def support(self, form):
        if form.poly is None:
            return form.support
        out = set()
        for m in form.poly:
            for v, _ in m:
                if v < self.n:
                    out.add(v)
                else:
                    sp = self.atom_info[v][3]
                    if sp is None:
                        return None
                    out |= sp
        return frozenset(out)

    def unknown(self, reason, *forms):
        """A non-representable form depending on the union of the arguments' supports."""
        s = frozenset()
        for f in forms:
            si = self.support(f)
            if si is None:
                s = None
                break
            s = s | si
        why = next((f.reason for f in forms if f.poly is None and f.reason), reason)
        return Form(None, s, why, any(f.inexact for f in forms))

    # ---- ring ops ------------------------------------------------------------------------
    def add(self, a, b):
        if a.poly is None or b.poly is None:
            return self.unknown('add', a, b)
        if not b.poly:
            return flagged(a, b.inexact)
        if not a.poly:
            return flagged(b, a.inexact)
        if len(a.poly) < len(b.poly):
            a, b = b, a
        self._spend(len(b.poly))
        t = dict(a.poly)
        for m, c in b.poly.items():
            v = t.get(m, 0) + c
            if v:
                t[m] = v
            else:
                t.pop(m, None)
        if len(t) > MAX_TERMS:
            return self.unknown('terms', a, b)
        return Form(t, inexact=a.inexact or b.inexact)

    def scale(self, a, c):
        if a.poly is None:
            return a
        if c == 0:
            return flagged(ZERO, a.inexact)
        if c == 1:
            return a
        return Form({m: v * c for m, v in a.poly.items()}, inexact=a.inexact)

    def neg(self, a):
        return self.scale(a, -1)

    def mul(self, a, b):
        if a.poly is None or b.poly is None:
            return self.unknown('mul', a, b)
        ca, cb = constant_value(a), constant_value(b)
        if ca is not None:
            if ca == 1 and not a.inexact:
                return Form(b.poly, inexact=a.inexact or b.inexact)
            return self._scale_flagged(b, ca, a.inexact)
        if cb is not None:
            return self._scale_flagged(a, cb, b.inexact)
        if len(a.poly) * len(b.poly) > MAX_PRODUCT:
            return self.unknown('product-size', a, b)
        self._spend(len(a.poly) * len(b.poly))
        t = {}
        for ma, va in a.poly.items():
            for mb, vb in b.poly.items():
                m = _monomial_product(ma, mb)
                v = t.get(m, 0) + va * vb
                if v:
                    t[m] = v
                else:
                    t.pop(m, None)
        if len(t) > MAX_TERMS:
            return self.unknown('terms', a, b)
        return Form(t, inexact=a.inexact or b.inexact)

    def _scale_flagged(self, a, c, inexact):
        if c == 0 and not inexact:
            return ZERO
        r = self.scale(a, c)
        if inexact and r.poly is not None:
            return Form(r.poly, inexact=True)
        return r

    def positive(self, v):
        if v < self.n:
            return self.lower is not None and self.lower[v] > 0
        return self.atom_info[v][0] == 'exp'          # exp(.) > 0; other atoms: sign unknown

    def affine(self, a):
        return all(all(v < self.n for v, _ in m) and sum(ex for _, ex in m) <= 1 for m in a.poly)

    def _int_power(self, a, k):
        out = constant(1)
        for _ in range(k):
            out = self.mul(out, a)
            if out.poly is None:
                break
        return out

    def expand(self, form):
        """Substitute every pow(int k >= 0) atom by its polynomial; other atoms stay (a
        polynomial in atoms)."""
        if form.poly is None or not any(v >= self.n for m in form.poly for v, _ in m):
            return form
        memo = self._expanded

        def expanded_atom(v):
            if v not in memo:
                kind, param, arg, _ = self.atom_info[v]
                memo[v] = (self.expand(Form(dict(arg)))
                           if kind == 'pow' and isinstance(param, int) and param >= 0 else None)
                if memo[v] is not None and memo[v].poly is not None:
                    memo[v] = self._int_power(memo[v], param)
            return memo[v]

        def substitutable(v, ex):
            return v >= self.n and expanded_atom(v) is not None and type(ex) is int and ex > 0

        pairs = []
        for m, c in form.poly.items():
            f = Form({tuple((v, ex) for v, ex in m if not substitutable(v, ex)): c}, inexact=form.inexact)
            for v, ex in m:
                if substitutable(v, ex):
                    f = self.mul(f, self._int_power(expanded_atom(v), ex))
            pairs.append((f, 1))
        return self.lincomb(pairs)

    def monomial(self, a):
        return a.poly is not None and len(a.poly) == 1 and () not in a.poly

    def pow(self, a, k):
        k = Fraction(k)
        if a.poly is None:
            return a
        c = constant_value(a)
        if c is not None:
            if c == 0:
                return self.unknown('0**k', a) if k <= 0 else constant(0, a.inexact)
            if c < 0 and k.denominator != 1:
                return self.unknown('neg-base-root', a)
            r, inexact = _rational_power(abs(c), k)
            if c < 0 and k.numerator % 2:
                r = -r
            return constant(r, inexact or a.inexact)
        if self.monomial(a) and (k.denominator == 1 or all(self.positive(v) for v, _ in next(iter(a.poly)))):
            # (c x^a)^k = c^k x^(ak) is an identity for integer k; for fractional k only
            # when every x_v > 0 on the box
            (m, c), = a.poly.items()
            if c < 0 and k.denominator != 1:
                return self.unknown('neg-base-root', a)
            r, inexact = _rational_power(abs(c), k)
            if c < 0 and k.numerator % 2:
                r = -r
            e = int(k) if k.denominator == 1 else k
            m2 = tuple((v, ex * e) for v, ex in m)
            if k < 0:
                self.poles.update(v for v, _ in m2 if v < self.n)
            return Form({m2: r}, inexact=a.inexact or inexact)
        if k.denominator == 1 and 0 <= k <= 8:
            if k >= 2 and not self.affine(a):
                return self.atom('pow', int(k), a)       # keep the factored square/cube; expand() gives the polynomial
            return self._int_power(a, int(k))
        if k == 0:
            return constant(1)
        if k < 0:
            self.poles.add(-1)                 # a pole of a sum: not tied to one variable
        return self.atom('pow', k, a)

    def div(self, a, b):
        if a.poly is None or b.poly is None:
            return self.unknown('div', a, b)
        cb = constant_value(b)
        if cb is not None:
            return self._scale_flagged(a, exact(Fraction(1) / cb), b.inexact) if cb != 0 else self.unknown('div0', a, b)
        return self.mul(a, self.pow(b, -1))

    def lincomb(self, pairs):
        """sum_i c_i * form_i (c_i rational), accumulated into one dict: linear in the number of terms."""
        t, inexact = {}, False
        for f, c in pairs:
            if f.poly is None:
                return self.unknown('lincomb', *[x for x, _ in pairs if x.poly is None][:1])
            inexact = inexact or f.inexact
            if not f.poly or c == 0:
                continue
            self._spend(len(f.poly))
            for m, v in f.poly.items():
                x = t.get(m, 0) + v * c
                if x:
                    t[m] = x
                else:
                    t.pop(m, None)
            if len(t) > MAX_TERMS:
                return self.unknown('terms', f)
        return Form(t, inexact=inexact) if t or inexact else ZERO

    # ---- scalar functions of a polynomial are interned atoms ---------------------------------
    def atom(self, kind, param, a):
        if a.poly is None:
            return self.unknown(kind, a)
        key = (kind, param, tuple(sorted(a.poly.items())))
        if key not in self.atoms:
            self.atoms[key] = self.n + len(self.atoms)
            self.atom_info[self.atoms[key]] = (kind, param, key[2], self.support(a))
        return Form({((self.atoms[key], 1),): 1}, inexact=a.inexact)

    def exp(self, a):
        if a.poly is None:
            return self.unknown('exp', a)
        c = constant_value(a)
        if c is not None:
            return constant(Fraction(math.exp(float(c))), True) if c != 0 else constant(1, a.inexact)
        # exp(sum c_i log(p_i) + t) = exp(t) * prod p_i**c_i (constant exponent); rounded exponents taint the product
        out, t = constant(1, a.inexact), 0
        for m, c in a.poly.items():
            if m == ():
                t = c
            elif len(m) == 1 and m[0][1] == 1 and m[0][0] >= self.n and self.atom_info[m[0][0]][0] == 'log':
                out = self.mul(out, self.pow(Form(dict(self.atom_info[m[0][0]][2])), c))
            else:
                return self.atom('exp', None, a)
        if t != 0:
            out = self.mul(out, constant(Fraction(math.exp(float(t))), True))
        return out

    def fn(self, a, name, f):
        if a.poly is None:
            return self.unknown(name, a)
        c = constant_value(a)
        if c is not None:
            try:
                return constant(Fraction(f(float(c))), True)
            except (ValueError, OverflowError):
                return self.unknown(name + '-domain', a)
        return self.atom(name, None, a)

    def log(self, a):
        c = constant_value(a)
        if c is not None and c <= 0:
            return self.unknown('log<=0', a)
        return self.fn(a, 'log', math.log)
