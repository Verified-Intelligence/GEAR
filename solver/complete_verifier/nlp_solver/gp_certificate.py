"""Root GP certificate, proved on the bound graph rather than inferred from samples.

Exponents and rational coefficients retain the exact binary graph constants. Irrational
coefficients (powers, exp and log) use 50-digit outward intervals. Signed term lists
are retained through algebraic operations to interpret generated residuals and the
solver's signed rows. They are not classified as posynomials until all coefficients
are positive; powers, log, and denominators require positive forms. Unsupported
nodes remain unknown.

In log space a log-posynomial is log-sum-exp of affine forms. For any exact probability
vector q, sum q_i (log c_i + a_i.y - log q_i) is a global supporting plane (the entropy
formula). Softmax gradients propose q; rational normalization and interval evaluation
certify the plane, including coefficient conversion over the entire log box. Thus
rounding a gradient cannot invalidate a cut. HiGHS only proposes nonnegative dual
multipliers; the box Lagrangian is evaluated with outward intervals as well.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from .continuation import Atom, retained

import math
import time

import numpy as np
import torch
from mpmath.ctx_iv import MPIntervalContext

from nlp_solver.quadratic_tangent import _const, _is_const, _DATA_MOVEMENT
from nlp_solver.structure.cert_kernels import cert_crown_dual

_IV = MPIntervalContext()
_IV.dps = 50
_MAX_TERMS = 512


def _interval(value):
    if isinstance(value, Fraction):
        return _IV.mpf(value.numerator) / value.denominator
    return _IV.mpf(value)


def _down(value):
    return float(np.nextafter(float(_interval(value).a), -np.inf))


def _up(value):
    return float(np.nextafter(float(_interval(value).b), np.inf))


def _add(a, b):
    return a + b if isinstance(a, Fraction) and isinstance(b, Fraction) else _interval(a) + _interval(b)


def _mul(a, b):
    return a * b if isinstance(a, Fraction) and isinstance(b, Fraction) else _interval(a) * _interval(b)


def _power(a, b):
    if isinstance(a, Fraction) and b.denominator == 1:
        return a ** int(b)
    return _interval(a) ** _interval(b)


@dataclass
class Form:
    # poly: exponent tuple -> coefficient; affine: intercept and rational y slopes.
    kind: str
    terms: dict | None = None
    intercept: object = None
    slopes: tuple | None = None

    @staticmethod
    def constant(c, n):
        return Form('poly', {tuple([Fraction(0)] * n): c} if c != 0 else {})

    def constant_value(self):
        if self.kind != 'poly' or any(any(a) for a in self.terms):
            return None
        return next(iter(self.terms.values()), Fraction(0))

    def positive(self):
        return self.kind == 'poly' and bool(self.terms) and all(_down(c) > 0 for c in self.terms.values())

    def monomial(self):
        return self.positive() and len(self.terms) == 1


OTHER = Form('other')


def _scale(f, c):
    if c == 0:
        # This is only called with structurally known forms, never 0 * unknown.
        if f.kind == 'poly':
            return Form('poly', {})
    if f.kind == 'poly':
        return Form('poly', {a: _mul(v, c) for a, v in f.terms.items() if _mul(v, c) != 0})
    if f.kind == 'affine' and isinstance(c, Fraction):
        return Form('affine', intercept=_mul(f.intercept, c), slopes=tuple(c * a for a in f.slopes))
    return OTHER


def _sum(a, b):
    if a.kind == b.kind == 'poly':
        terms = a.terms.copy()
        for e, c in b.terms.items():
            terms[e] = _add(terms.get(e, Fraction(0)), c)
            if terms[e] == 0:
                del terms[e]
        return Form('poly', terms) if len(terms) <= _MAX_TERMS else OTHER
    if a.kind == 'affine' and b.constant_value() is not None:
        return Form('affine', intercept=_add(a.intercept, b.constant_value()), slopes=a.slopes)
    if b.kind == 'affine' and a.constant_value() is not None:
        return _sum(b, a)
    if a.kind == b.kind == 'affine':
        return Form('affine', intercept=_add(a.intercept, b.intercept),
                    slopes=tuple(x + y for x, y in zip(a.slopes, b.slopes)))
    return OTHER


def _product(a, b):
    ac, bc = a.constant_value(), b.constant_value()
    if ac is not None:
        return _scale(b, ac)
    if bc is not None:
        return _scale(a, bc)
    # Generated residual graphs place the row's minus sign inside a product or
    # quotient. Exact signed expansion preserves that sign until the spec is applied.
    if a.kind != 'poly' or b.kind != 'poly' or len(a.terms) * len(b.terms) > _MAX_TERMS:
        return OTHER
    terms = {}
    for x, c in a.terms.items():
        for y, d in b.terms.items():
            e = tuple(i + j for i, j in zip(x, y))
            terms[e] = _add(terms.get(e, Fraction(0)), _mul(c, d))
    return Form('poly', terms)


def _pow(f, k):
    if not isinstance(k, Fraction) or not f.positive():
        return OTHER
    if f.monomial():
        a, c = next(iter(f.terms.items()))
        return Form('poly', {tuple(k * e for e in a): _power(c, k)})
    if k.denominator != 1 or k <= 0 or k > _MAX_TERMS:
        return OTHER
    out = f
    for _ in range(int(k) - 1):
        out = _product(out, f)
        if out.kind == 'other':
            break
    return out


def _unary(f, name):
    if name == 'BoundNeg':
        return _scale(f, Fraction(-1))
    if name in ('BoundSqrt', 'BoundSqr', 'BoundReciprocal'):
        return _pow(f, {'BoundSqrt': Fraction(1, 2), 'BoundSqr': Fraction(2),
                        'BoundReciprocal': Fraction(-1)}[name])
    if name == 'BoundLog' and f.monomial():
        a, c = next(iter(f.terms.items()))
        return Form('affine', intercept=_IV.ln(_interval(c)), slopes=a)
    if name == 'BoundExp' and f.kind == 'affine':
        return Form('poly', {f.slopes: _IV.exp(_interval(f.intercept))})
    return OTHER


def _binary(a, b, name):
    if name == 'BoundAdd':
        return _sum(a, b)
    if name == 'BoundSub':
        return _sum(a, _scale(b, Fraction(-1)))
    if name == 'BoundMul':
        return _product(a, b)
    if name == 'BoundDiv':
        if not b.monomial():
            return OTHER
        return _product(a, _pow(b, Fraction(-1)))
    if name == 'BoundPow':
        return _pow(a, b.constant_value())
    return OTHER


def _object_array(values, shape):
    out = np.empty(len(values), dtype=object)
    out[:] = values
    return out.reshape(shape)


def _node_form(node, forms, lower, input_name):
    name, ins = type(node).__name__, node.inputs
    shape = tuple(node.forward_value.shape)
    n = lower.size
    if name == 'BoundInput':
        if node.name != input_name or math.prod(shape) != n:
            return np.full(shape, OTHER, dtype=object)
        vals = []
        for j in range(n):
            a = [Fraction(0)] * n
            a[j] = Fraction(1)
            vals.append(Form('poly', {tuple(a): Fraction(1)}) if lower[j] > 0 else OTHER)
        return _object_array(vals, shape)
    if _is_const(node):
        return _object_array([Form.constant(Fraction(v.item()), n)
                              for v in _const(node).reshape(-1)], shape)
    args = [forms[i.name] for i in ins]
    if name in ('BoundAdd', 'BoundSub', 'BoundMul', 'BoundDiv', 'BoundPow'):
        a, b = np.broadcast_arrays(args[0], args[1])
        return _object_array([_binary(x, y, name) for x, y in zip(a.flat, b.flat)], shape)
    if name in ('BoundNeg', 'BoundSqr', 'BoundSqrt', 'BoundReciprocal', 'BoundLog', 'BoundExp'):
        return _object_array([_unary(f, name) for f in args[0].flat], shape)
    if name == 'BoundLinear':
        if getattr(node, 'transA', 0) or not all(_is_const(i) for i in ins[1:]):
            return np.full(shape, OTHER, dtype=object)
        _, w, bias = node._preprocess(torch.zeros(args[0].shape, dtype=torch.float64),
                                      *[_const(i).double() for i in ins[1:]])
        out = []
        for row in args[0].reshape(-1, args[0].shape[-1]):
            for j, weights in enumerate(w):
                f = Form.constant(Fraction(float(bias[j])) if bias is not None else Fraction(0), n)
                for x, c in zip(row, weights):
                    if float(c) != 0:
                        f = _sum(f, _scale(x, Fraction(float(c))))
                out.append(f)
        return _object_array(out, shape)
    if name == 'BoundMatMul' and args[0].ndim <= 2 and args[1].ndim <= 2:
        # Scalar Form operations are deliberately explicit; numpy matmul on object
        # arrays could silently invoke unsupported arithmetic.
        a, b = args[0], args[1]
        av, bv = a.ndim == 1, b.ndim == 1
        a, b = a.reshape(1, -1) if av else a, b.reshape(-1, 1) if bv else b
        out = []
        for row in a:
            for col in b.T:
                f = Form.constant(Fraction(0), n)
                for x, y in zip(row, col):
                    f = _sum(f, _product(x, y))
                out.append(f)
        return _object_array(out, shape)
    if name == 'BoundReduceSum':
        axis = getattr(node, 'axis', None)
        axes = tuple(range(args[0].ndim)) if axis is None else tuple(
            (a % args[0].ndim) for a in (axis if isinstance(axis, (list, tuple)) else [axis]))
        out = np.full(shape, None, dtype=object)
        keep = bool(getattr(node, 'keepdim', False))
        for idx in np.ndindex(args[0].shape):
            dst = tuple(0 if i in axes else k for i, k in enumerate(idx)) if keep else tuple(
                k for i, k in enumerate(idx) if i not in axes)
            out[dst] = _sum(out[dst] or Form.constant(Fraction(0), n), args[0][idx])
        return out
    if name == 'BoundATenIndex_add' and all(_is_const(ins[k]) for k in (1, 2, 4)):
        axis = int(_const(ins[1])) % args[0].ndim
        idx = _const(ins[2]).long().reshape(-1).tolist()
        alpha = Fraction(float(_const(ins[4])))
        data, src = args[0], args[3]
        target = tuple(max(a, b) if k != axis else a for k, (a, b) in enumerate(zip(data.shape, src.shape)))
        data = np.broadcast_to(data, target).copy()
        src_shape = list(target)
        src_shape[axis] = src.shape[axis]
        src = np.broadcast_to(src, src_shape)
        for loc in np.ndindex(src.shape):
            dst = list(loc)
            dst[axis] = idx[loc[axis]]
            data[tuple(dst)] = _sum(data[tuple(dst)], _scale(src[loc], alpha))
        return data.reshape(shape)
    if name in _DATA_MOVEMENT:
        # Forward integer labels through the actual shape/index operator. Only
        # Concat has multiple data arguments; all remaining inputs are parameters.
        count = len(ins) if name == 'BoundConcat' else 1
        table, labels = [], []
        for i, arr in enumerate(args):
            if i >= count:
                if not _is_const(ins[i]):
                    return np.full(shape, OTHER, dtype=object)
                labels.append(_const(ins[i]))
            else:
                offset = len(table)
                table.extend(arr.flat)
                labels.append(torch.arange(offset, len(table), dtype=torch.int64).reshape(arr.shape))
        moved = node.forward(*labels).detach().cpu().reshape(-1)
        if moved.is_floating_point() and not (moved == moved.long()).all():
            return np.full(shape, OTHER, dtype=object)
        if name == 'BoundCast' and node.forward_value.dtype != ins[0].forward_value.dtype:
            return np.full(shape, OTHER, dtype=object)  # a numeric cast can round inputs
        return _object_array([table[int(v)] for v in moved], shape)
    return np.full(shape, OTHER, dtype=object)


@torch.no_grad()
def output_forms(bounded_module, x, lower):
    """Structural forms of all output elements. Unknown operators fail closed."""
    bounded_module(x)
    order, seen = [], set()

    def visit(node):
        if node.name not in seen:
            seen.add(node.name)
            for inp in node.inputs:
                visit(inp)
            order.append(node)

    final = bounded_module[bounded_module.final_name]
    visit(final)
    forms = {}
    try:
        for node in order:
            try:
                forms[node.name] = _node_form(node, forms, lower, bounded_module.input_name[0])
            except (ValueError, TypeError, IndexError, AttributeError, OverflowError, ZeroDivisionError):
                forms[node.name] = np.full(tuple(node.forward_value.shape), OTHER, dtype=object)
        return list(forms[final.name].flat)
    finally:
        bounded_module(x)  # shape-operator forwards must not alter cached values


def _split_offset(f):
    if f.kind != 'poly':
        return None
    nonconstant = {a: c for a, c in f.terms.items() if any(a)}
    offset = next((c for a, c in f.terms.items() if not any(a)), Fraction(0))
    p = Form('poly', nonconstant)
    return p, offset


def _log_data(f):
    return [(a, _IV.ln(_interval(c))) for a, c in f.terms.items()]


def _support(f, point, lo, hi):
    """A rounded global supporting plane on [lo,hi], returned as slope, intercept."""
    data = _log_data(f)
    # Float softmax proposes an exact rational probability vector. The entropy
    # inequality is valid for ANY such vector, including underflowed zero weights.
    scores = np.array([float(logc.mid) + np.dot([float(a) for a in exps], point)
                       for exps, logc in data])
    weights = np.exp(scores - scores.max())
    q = [Fraction(float(w)) for w in weights]
    total = sum(q)
    q = [w / total for w in q]
    slope = [sum(w * a[j] for w, (a, _) in zip(q, data)) for j in range(len(point))]
    intercept = _IV.mpf(0)
    for w, (_, logc) in zip(q, data):
        if w:
            intercept += _interval(w) * (logc - _IV.ln(_interval(w)))
    rounded = np.array([float(s) for s in slope])
    # Correct slope conversion over the entire log box.
    for j, exact in enumerate(slope):
        intercept += (_interval(exact) - _interval(Fraction(float(rounded[j])))) * _IV.mpf([lo[j], hi[j]])
    return rounded, _down(intercept)


def _lp_dual(c, G, h, lo, hi, multipliers):
    """Exact box Lagrangian with outward arithmetic; solver residuals are irrelevant."""
    value = _IV.mpf(0)
    residual = [_interval(Fraction(float(v))) for v in c]
    for row, rhs, multiplier in zip(G, h, multipliers):
        if not math.isfinite(float(multiplier)):
            return -np.inf
        w = _interval(Fraction(max(0., float(multiplier))))
        value -= w * _interval(Fraction(float(rhs)))
        for j, coef in enumerate(row):
            residual[j] += w * _interval(Fraction(float(coef)))
    for r, l, u in zip(residual, lo, hi):
        value += r * _IV.mpf([l, u])
    return _down(value)


def _cutoff_box(lower, upper, target, tolerance, offset, log):
    """cert_crown_dual: the epigraph column t is capped by the cutoff P - offset (its log in the entropy variant) and the
    kernel stopped at P - tolerance. The master is then the relaxation of {f <= P}: its bound is valid as min(D, P)
    (L2 semantics) on the capped box, which the weak-duality evaluation uses too. An uncapped column of width ~1e20
    would void a first-order dual (a residual 1e-10 times the width)."""
    upper, stop = upper.copy(), None
    if target is None or not math.isfinite(target):
        return upper, None
    scale = (lambda v: math.log(v) if v > 0 else None) if log else (lambda v: v)
    cap, stop = scale(float(target) - float(offset)), scale(float(target) - float(offset) - tolerance)
    if cap is not None:
        upper[-1] = max(min(upper[-1], cap + 1e-9 * max(1., abs(cap))), lower[-1])
    if stop is not None:
        stop -= 1e-9 * max(1., abs(stop))
    return upper, stop


def _master(c, matrix, rhs, lower, upper, stop, deadline):
    """cert_crown_dual: (x, y) of the master LP from the Halpern kernel; None if the lane proves it empty."""
    from nlp_solver.structure.cert_kernels import halpern_lp
    return halpern_lp(c, matrix, rhs, lower, upper, stop, deadline)


@dataclass
class GPCertificate:
    objective: Form
    offset: object
    rows: list  # (positive posynomial, log upper bound), including equality slabs
    lower: np.ndarray
    upper: np.ndarray

    @retained
    def solve(self, incumbent_x=None, time_budget=20., target=None, tolerance=0., sliced=False):
        try:
            solve = self._solve_crown if getattr(self, 'crown', False) else self._solve
            return (yield from solve(incumbent_x, time_budget, target, tolerance, sliced))
        except (ValueError, OverflowError, FloatingPointError, ZeroDivisionError):
            # A structurally valid GP can exceed the LP's numeric range. It then
            # supplies no certificate, just as an unproved solver status does.
            return -np.inf, 0

    def _solve(self, incumbent_x, time_budget, target, tolerance, sliced=False):
        from scipy.optimize import linprog
        start = time.monotonic()
        lo = np.array([_down(_IV.ln(_interval(Fraction(float(v))))) for v in self.lower])
        hi = np.array([_up(_IV.ln(_interval(Fraction(float(v))))) for v in self.upper])
        n = len(lo)
        point = (lo + hi) / 2
        points = [point]
        if incumbent_x is not None:
            inc = np.asarray(torch.as_tensor(incumbent_x).detach().cpu()).reshape(-1)
            if inc.size == n and np.isfinite(inc).all() and (inc > 0).all():
                points.append(np.clip(np.log(inc), lo, hi))
        # Finite bounds on the epigraph permit weak-duality correction even when
        # HiGHS's stationarity residual is nonzero.
        data = _log_data(self.objective)
        ranges = [logc + sum((_interval(a[j]) * _IV.mpf([lo[j], hi[j]]) for j in range(n)), _IV.mpf(0))
                  for a, logc in data]
        tlo = max(_down(z) for z in ranges)
        thi = _up(_IV.ln(sum((_IV.exp(z) for z in ranges), _IV.mpf(0))))
        lower, upper = np.r_[lo, tlo], np.r_[hi, thi]
        c = np.r_[np.zeros(n), 1.]
        fo = cert_crown_dual()
        capped, stop = _cutoff_box(lower, upper, target, tolerance, self.offset, True) if fo else (upper, None)
        A, b = [], []
        best, previous = -np.inf, -np.inf
        rounds = 0
        while time.monotonic() - start < time_budget:
            for point in points:
                slope, intercept = _support(self.objective, point, lo, hi)
                A.append(np.r_[slope, -1.])
                b.append(_up(-_interval(Fraction(intercept))))
                for f, log_rhs in self.rows:
                    slope, intercept = _support(f, point, lo, hi)
                    A.append(np.r_[slope, 0.])
                    b.append(_up(log_rhs - _interval(Fraction(intercept))))
            matrix, rhs = np.array(A), np.array(b)
            if fo:
                result = _master(c, matrix, rhs, lower, capped, stop, time.time() + max(.001, time_budget - (time.monotonic() - start)))
                rounds += 1
                if result is None:
                    break
                point, multipliers = result[0], result[1]
            else:
                result = linprog(c, A_ub=matrix, b_ub=rhs, bounds=np.stack([lower, upper], 1),
                                 method='highs', options={'time_limit': max(.001, time_budget - (time.monotonic() - start))})
                rounds += 1
                if result.status != 0 or result.ineqlin.marginals is None:
                    break  # infeasibility is never taken on solver trust
                point, multipliers = result.x, np.maximum(-result.ineqlin.marginals, 0.)
            lb = _lp_dual(c, matrix, rhs, lower, capped, multipliers)
            dual = _down(_IV.exp(_interval(Fraction(lb))) + _interval(self.offset))
            best = max(best, min(dual, target) if fo and target is not None else dual)
            if sliced:
                yield Atom("gp_certificate", bound=best)
            if target is not None and 0 <= target - best <= tolerance:
                break
            if rounds > 5 and best - previous <= 1e-10 * max(1., abs(best)):
                break
            previous = best
            points = [point[:n]]
        return best, rounds

    def _solve_crown(self, incumbent_x, time_budget, target, tolerance, sliced=False):
        """The same cutting-plane loop with native CROWN planes of the posynomials in log coordinates
        (nlp_solver/crown_rewrite.py; solver.nlp.gp_crown), in the epigraph of the objective posynomial itself.
        The planes' intercepts and the LP dual are evaluated with outward intervals, as in _solve."""
        from scipy.optimize import linprog
        from nlp_solver.crown_rewrite import PosynomialPlanes
        start = time.monotonic()
        lo = np.array([_down(_IV.ln(_interval(Fraction(float(v))))) for v in self.lower])
        hi = np.array([_up(_IV.ln(_interval(Fraction(float(v))))) for v in self.upper])
        n = len(lo)
        planes = PosynomialPlanes([self.objective] + [f for f, _ in self.rows], lo, hi)
        row_rhs = [_up(_IV.exp(log_rhs)) for _, log_rhs in self.rows]
        points = [(lo + hi) / 2]
        if incumbent_x is not None:
            inc = np.asarray(torch.as_tensor(incumbent_x).detach().cpu()).reshape(-1)
            if inc.size == n and np.isfinite(inc).all() and (inc > 0).all():
                points.append(np.clip(np.log(inc), lo, hi))
        # Finite bounds on the epigraph keep the weak-duality evaluation finite: a posynomial is positive,
        # and at most the sum of its terms' maxima on the box.
        data = _log_data(self.objective)
        ranges = [logc + sum((_interval(a[j]) * _IV.mpf([lo[j], hi[j]]) for j in range(n)), _IV.mpf(0))
                  for a, logc in data]
        thi = _up(sum((_IV.exp(z) for z in ranges), _IV.mpf(0)))
        if not math.isfinite(thi):
            return -np.inf, 0
        lower, upper = np.r_[lo, 0.], np.r_[hi, thi]
        c = np.r_[np.zeros(n), 1.]
        fo = cert_crown_dual()
        capped, stop = _cutoff_box(lower, upper, target, tolerance, self.offset, False) if fo else (upper, None)
        A, b = [], []
        best, previous = -np.inf, -np.inf
        rounds = 0
        while time.monotonic() - start < time_budget:
            for point in points:
                found = planes.planes(point)
                slope, intercept = found[0]
                A.append(np.r_[slope, -1.])
                b.append(_up(-_interval(Fraction(intercept))))
                for (slope, intercept), rhs in zip(found[1:], row_rhs):
                    A.append(np.r_[slope, 0.])
                    b.append(_up(_interval(Fraction(rhs)) - _interval(Fraction(intercept))))
            matrix, rhs = np.array(A), np.array(b)
            if fo:
                result = _master(c, matrix, rhs, lower, capped, stop, time.time() + max(.001, time_budget - (time.monotonic() - start)))
                rounds += 1
                if result is None:
                    break
                point, multipliers = result[0], result[1]
            else:
                result = linprog(c, A_ub=matrix, b_ub=rhs, bounds=np.stack([lower, upper], 1),
                                 method='highs', options={'time_limit': max(.001, time_budget - (time.monotonic() - start))})
                rounds += 1
                if result.status != 0 or result.ineqlin.marginals is None:
                    break  # infeasibility is never taken on solver trust
                point, multipliers = result.x, np.maximum(-result.ineqlin.marginals, 0.)
            lb = _lp_dual(c, matrix, rhs, lower, capped, multipliers)
            dual = _down(_interval(Fraction(lb)) + _interval(self.offset))
            best = max(best, min(dual, target) if fo and target is not None else dual)
            if sliced:
                yield Atom("gp_certificate", bound=best)
            if target is not None and 0 <= target - best <= tolerance:
                break
            if rounds > 5 and best - previous <= 1e-10 * max(1., abs(best)):
                break
            previous = best
            points = [point[:n]]
        return best, rounds


def build_gp_certificate(bounded_module, model_ori, x, C, rhs, lower, upper, diagnostics=None, *, rowpool):
    """Rows that are not GP-usable are left out of the relaxation (sound: a superset of
    the feasible set only lowers its minimum), e.g. hs101-103's "objective >= 100".

    Monomial equality pairs use two linear log inequalities, with their independently
    relaxed thresholds (`rowpool.thresholds`); replacing a tolerance slab with exact
    equality is unsound.
    """
    diagnostics = diagnostics if diagnostics is not None else {}

    def reject(reason):
        diagnostics['reason'] = reason
        return None

    lo = lower.detach().cpu().double().numpy().reshape(-1)
    hi = upper.detach().cpu().double().numpy().reshape(-1)
    if not (np.isfinite(lo).all() and np.isfinite(hi).all() and (lo > 0).all() and (lo <= hi).all()):
        return reject('root box is not finite and strictly positive')
    if C.ndim != 3 or C.shape[0] != 1 or rhs.shape[0] != 1:
        return reject('requires a single conjunctive specification')
    outputs = output_forms(bounded_module, x, lo)
    if C.shape[-1] != len(outputs) or C.shape[1] != rhs.numel() or not torch.isfinite(C).all():
        return reject('invalid specification dimensions or coefficients')
    signed = []
    for row in C[0].detach().cpu().double().numpy():
        f = Form.constant(Fraction(0), len(lo))
        for coefficient, output in zip(row, outputs):
            if coefficient != 0:
                f = _sum(f, _scale(output, Fraction(float(coefficient))))
        signed.append(f)
    split = _split_offset(signed[0])
    if split is None or not split[0].positive():
        # A positive constant objective is also a posynomial.
        if not signed[0].positive():
            return reject('objective is not a posynomial with a constant offset')
        objective, offset = signed[0], Fraction(0)
    else:
        objective, offset = split
    # Sanity check of the symbolic forms against the model, after the cheap structural
    # rejections. Deterministic local RNG, so enabling the proof cannot change primal/BaB seeds.
    rng = np.random.default_rng(1729)
    try:
        for _ in range(3):
            point = np.exp(rng.uniform(np.log(lo), np.log(hi)))
            # Logs once per point; exponent vectors are sparse (cute_gouldqp2: 699 outputs x 699
            # variables of 50-digit interval arithmetic per point took over 80 s).
            logs = [_IV.ln(_interval(Fraction(float(p)))) for p in point]
            z = torch.as_tensor(point, dtype=x.dtype, device=x.device).reshape(x.shape)
            with torch.no_grad():
                actual = model_ori(z).detach().cpu().double().numpy().reshape(-1)
            if actual.size != len(outputs):
                return reject('symbolic/model sanity mismatch')
            for f, v in zip(outputs, actual):
                if f.kind == 'poly':
                    predicted = sum((_interval(c) * _IV.exp(sum((_interval(aj) * logs[j] for j, aj in enumerate(a) if aj),
                                                                _IV.mpf(0))) for a, c in f.terms.items()), _IV.mpf(0))
                elif f.kind == 'affine':
                    predicted = _interval(f.intercept) + sum((_interval(a) * logs[j] for j, a in enumerate(f.slopes) if a),
                                                             _IV.mpf(0))
                else:
                    continue
                if not math.isfinite(v) or abs(float(predicted.mid) - v) > 1e-8 * max(1., abs(v)):
                    return reject('symbolic/model sanity mismatch')
    finally:
        bounded_module(x)
    relaxed = rowpool.thresholds(rhs).detach().cpu().double().numpy().reshape(-1)
    original_rhs = rhs.detach().cpu().double().numpy().reshape(-1)
    rows, dropped = [], []
    for r, f in enumerate(signed[1:], 1):
        if np.isposinf(relaxed[r]):
            continue
        if not math.isfinite(relaxed[r]):
            dropped.append(r)
            continue
        split = _split_offset(f)
        if split is None:
            dropped.append(r)
            continue
        p, off = split
        bound = _add(Fraction(float(relaxed[r])), -off)
        if p.positive() and _down(bound) > 0:
            rows.append((p, _IV.ln(_interval(bound))))
            continue
        # A negative monomial is allowed only as the reverse half of a structural
        # equality pair in the ORIGINAL signed rows (not by approximate matching).
        reverse = _scale(p, Fraction(-1))
        pair = any(math.isfinite(original_rhs[s])
                   and Fraction(float(original_rhs[s])) == -Fraction(float(original_rhs[r]))
                   and other.kind == 'poly' and _sum(f, other).kind == 'poly'
                   and not _sum(f, other).terms for s, other in enumerate(signed[1:], 1))
        if reverse.monomial() and pair and _down(-bound) > 0:
            rows.append((_pow(reverse, Fraction(-1)), -_IV.ln(_interval(-bound))))
            continue
        dropped.append(r)
        continue
    diagnostics.update(reason='proved GP', rows=len(rows), dropped_rows=dropped,
                       terms=len(objective.terms))
    return GPCertificate(objective, offset, rows, lo, hi)
