"""Root certificate for objectives convex after y_j = exp(x_j).

Exact rational algebra on the bound graph recognizes relative entropy, never
inferring a symbolic identity from samples. Each entropy gets its own epigraph.
Outward-rounded tangents and affine rows enlarge the relaxation; HiGHS only
proposes multipliers, whose box Lagrangian is re-evaluated with intervals.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from .continuation import Atom, retained

import math
import time
import warnings

import numpy as np
import torch

from nlp_solver.gp_certificate import _IV, _interval, _down, _up, _lp_dual, _object_array
from nlp_solver.quadratic_tangent import _const, _is_const, _DATA_MOVEMENT
from nlp_solver.structure.cert_kernels import cert_crown_dual

_MAX_TERMS = 512
# A polynomial's atoms are ('x', j), ('y', j), or ('log', affine-y-tuple).
# None denotes an unproved expression. Constants have the empty monomial ().


def _constant(c):
    return {(): c} if c else {}


def _scale(f, c):
    return None if f is None else {a: v * c for a, v in f.items() if v * c}


def _sum(a, b):
    if a is None or b is None:
        return None
    out = a.copy()
    for key, value in b.items():
        out[key] = out.get(key, Fraction(0)) + value
        if not out[key]:
            del out[key]
    return out if len(out) <= _MAX_TERMS else None


def _product(a, b):
    if a is None or b is None or len(a) * len(b) > _MAX_TERMS:
        return None
    out = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            key = tuple(sorted(ka + kb, key=repr))
            out[key] = out.get(key, Fraction(0)) + va * vb
            if not out[key]:
                del out[key]
    return out


def _constant_value(f):
    return f.get((), Fraction(0)) if f is not None and all(not a for a in f) else None


def _affine(f, n, exp_variables):
    if f is None:
        return None
    data = [f.get((), Fraction(0))] + [Fraction(0)] * n
    for atoms, coefficient in f.items():
        if not atoms:
            continue
        if len(atoms) != 1:
            return None
        kind, j = atoms[0]
        if kind not in ('x', 'y') or (kind == 'y') != (j in exp_variables):
            return None
        data[j + 1] += coefficient
    return tuple(data)


def _binary(a, b, name):
    if name == 'BoundAdd':
        return _sum(a, b)
    if name == 'BoundSub':
        return _sum(a, _scale(b, Fraction(-1)))
    if name == 'BoundMul':
        return _product(a, b)
    if name == 'BoundDiv':
        c = _constant_value(b)
        return _scale(a, 1 / c) if c else None
    return None


def _node_form(node, forms, n, input_name, exp_variables):
    name, ins = type(node).__name__, node.inputs
    shape = tuple(node.forward_value.shape)
    unknown = lambda: np.full(shape, None, dtype=object)
    if name == 'BoundInput':
        if node.name != input_name or math.prod(shape) != n:
            return unknown()
        return _object_array([{(('x', j),): Fraction(1)} for j in range(n)], shape)
    if _is_const(node):
        return _object_array([_constant(Fraction(v.item())) for v in _const(node).reshape(-1)], shape)
    args = [forms[i.name] for i in ins]
    if name in ('BoundAdd', 'BoundSub', 'BoundMul', 'BoundDiv'):
        a, b = np.broadcast_arrays(args[0], args[1])
        return _object_array([_binary(x, y, name) for x, y in zip(a.flat, b.flat)], shape)
    if name == 'BoundNeg':
        return _object_array([_scale(f, Fraction(-1)) for f in args[0].flat], shape)
    if name in ('BoundExp', 'BoundLog'):
        out = []
        for f in args[0].flat:
            if name == 'BoundExp':
                atom = next(iter(f)) if f is not None and len(f) == 1 else ()
                if len(atom) == 1 and atom[0][0] == 'x' and f[atom] == 1:
                    j = atom[0][1]
                    exp_variables.add(j)
                    out.append({(('y', j),): Fraction(1)})
                else:
                    out.append(None)
            else:
                # Logs can only depend on exp variables, never raw x variables.
                affine = _affine(f, n, set(range(n)))
                out.append({(('log', affine),): Fraction(1)} if affine is not None else None)
        return _object_array(out, shape)
    if name == 'BoundLinear':
        if getattr(node, 'transA', 0) or not all(_is_const(i) for i in ins[1:]):
            return unknown()
        _, w, bias = node._preprocess(torch.zeros(args[0].shape, dtype=torch.float64),
                                     *[_const(i).double() for i in ins[1:]])
        out = []
        for row in args[0].reshape(-1, args[0].shape[-1]):
            for j, weights in enumerate(w):
                f = _constant(Fraction(float(bias[j])) if bias is not None else Fraction(0))
                for value, c in zip(row, weights):
                    if float(c):
                        f = _sum(f, _scale(value, Fraction(float(c))))
                out.append(f)
        return _object_array(out, shape)
    if name == 'BoundMatMul' and args[0].ndim <= 2 and args[1].ndim <= 2:
        a, b = args[:2]
        a = a.reshape(1, -1) if a.ndim == 1 else a
        b = b.reshape(-1, 1) if b.ndim == 1 else b
        return _object_array([_matmul_sum(row, col) for row in a for col in b.T], shape)
    if name == 'BoundReduceSum':
        axis = getattr(node, 'axis', None)
        axes = tuple(range(args[0].ndim)) if axis is None else tuple(
            a % args[0].ndim for a in (axis if isinstance(axis, (list, tuple)) else [axis]))
        out = np.empty(shape, dtype=object)
        for dst in np.ndindex(shape):
            out[dst] = {}
        keep = bool(getattr(node, 'keepdim', False))
        for idx in np.ndindex(args[0].shape):
            dst = tuple(0 if i in axes else k for i, k in enumerate(idx)) if keep else tuple(
                k for i, k in enumerate(idx) if i not in axes)
            out[dst] = _sum(out[dst], args[0][idx])
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
        count = len(ins) if name == 'BoundConcat' else 1
        table, labels = [], []
        for i, arr in enumerate(args):
            if i >= count:
                if not _is_const(ins[i]):
                    return unknown()
                labels.append(_const(ins[i]))
            else:
                offset = len(table)
                table.extend(arr.flat)
                labels.append(torch.arange(offset, len(table), dtype=torch.int64).reshape(arr.shape))
        moved = node.forward(*labels).detach().cpu().reshape(-1)
        if moved.is_floating_point() and not (moved == moved.long()).all():
            return unknown()
        if name == 'BoundCast' and node.forward_value.dtype != ins[0].forward_value.dtype:
            return unknown()
        return _object_array([table[int(v)] for v in moved], shape)
    return unknown()


def _matmul_sum(row, col):
    f = {}
    for x, y in zip(row, col):
        f = _sum(f, _product(x, y))
    return f


@torch.no_grad()
def output_forms(bounded_module, x):
    """Return exact output polynomials and the structurally identified exp inputs."""
    bounded_module(x)
    order, seen, exp_variables = [], set(), set()

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
                forms[node.name] = _node_form(node, forms, x.numel(), bounded_module.input_name[0], exp_variables)
            except (ValueError, TypeError, IndexError, AttributeError, OverflowError, ZeroDivisionError):
                forms[node.name] = np.full(tuple(node.forward_value.shape), None, dtype=object)
        return list(forms[final.name].flat), exp_variables
    finally:
        bounded_module(x)


def _affine_value(data, point):
    return _interval(data[0]) + sum((_interval(c) * p for c, p in zip(data[1:], point)), _IV.mpf(0))


def _range(data, lo, hi):
    return _affine_value(data, [_IV.mpf([l, u]) for l, u in zip(lo, hi)])


def _evaluate(f, point, y):
    value = _IV.mpf(0)
    for atoms, c in f.items():
        term = _interval(c)
        for kind, data in atoms:
            term *= (_interval(Fraction(float(point[data]))) if kind == 'x' else
                     y[data] if kind == 'y' else _IV.ln(_affine_value(data, y)))
        value += term
    return float(value.mid)


@dataclass
class EntropyTerm:
    weight: Fraction
    numerator: tuple  # affine, either y_j or the constant 1
    denominator: tuple

    def support(self, point, lo, hi):
        p = [_interval(Fraction(float(v))) for v in point]
        a, b = _affine_value(self.numerator, p), _affine_value(self.denominator, p)
        ga, gb = _IV.ln(a / b) + 1, -a / b
        slopes = [_interval(self.weight) * (ga * _interval(c) + gb * _interval(d))
                  for c, d in zip(self.numerator[1:], self.denominator[1:])]
        intercept = _interval(self.weight) * a * _IV.ln(a / b) - sum(
            (s * v for s, v in zip(slopes, p)), _IV.mpf(0))
        rounded = np.array([float(v.mid) for v in slopes])
        for s, r, l, u in zip(slopes, rounded, lo, hi):
            intercept += (s - _interval(Fraction(float(r)))) * _IV.mpf([l, u])
        return rounded, _down(intercept)


def _objective(f, n, exp_variables):
    if f is None:
        return None
    affine, entropies, logy = {}, [], {}
    one = (Fraction(1),) + (Fraction(0),) * n
    for atoms, c in f.items():
        if len(atoms) == 2 and set(a[0] for a in atoms) == {'x', 'y'}:
            ax, ay = sorted(atoms)
            if ax[1] != ay[1] or ax[1] not in exp_variables or c < 0:
                return None
            logy[ax[1]] = c
        elif len(atoms) == 2 and set(a[0] for a in atoms) == {'y', 'log'}:
            log, y = sorted(atoms)
            if c > 0:
                return None
            numerator = [Fraction(0)] * (n + 1)
            numerator[y[1] + 1] = Fraction(1)
            entropies.append(EntropyTerm(-c, tuple(numerator), log[1]))
        elif len(atoms) == 1 and atoms[0][0] == 'log':
            if c > 0:
                return None
            entropies.append(EntropyTerm(-c, one, atoms[0][1]))
        else:
            affine[atoms] = c
    for term in entropies:
        if term.numerator != one:
            j = term.numerator[1:].index(Fraction(1))
            logy[j] = logy.get(j, Fraction(0)) - term.weight
    for j, c in logy.items():
        if c < 0:
            return None
        if c:
            numerator = [Fraction(0)] * (n + 1)
            numerator[j + 1] = Fraction(1)
            entropies.append(EntropyTerm(c, tuple(numerator), one))
    data = _affine(affine, n, exp_variables)
    return (data, entropies) if data is not None and entropies else None


def _rounded_row(data, rhs, lo, hi):
    slopes = np.array([float(c) for c in data[1:]])
    bound = _interval(rhs) - _interval(data[0])
    # rounded_slope.z <= exact_rhs + max((rounded-exact).z)
    for c, r, l, u in zip(data[1:], slopes, lo, hi):
        bound += (_interval(Fraction(float(r))) - _interval(c)) * _IV.mpf([l, u])
    return slopes, _up(bound)


@dataclass
class ExpCertificate:
    objective: tuple
    terms: list[EntropyTerm]
    rows: list
    lower: np.ndarray
    upper: np.ndarray
    exp_variables: set

    @retained
    def solve(self, incumbent_x=None, time_budget=20., target=None, tolerance=0., sliced=False):
        try:
            return (yield from self._solve(incumbent_x, time_budget, target, tolerance, sliced))
        except (ValueError, OverflowError, FloatingPointError, ZeroDivisionError):
            return -np.inf, 0

    def _solve(self, incumbent_x, time_budget, target, tolerance, sliced=False):
        from scipy.optimize import linprog, OptimizeWarning
        start = time.monotonic()
        lo, hi = self.lower, self.upper
        n, m = len(lo), len(self.terms)
        lower, upper = list(lo), list(hi)
        for term in self.terms:
            a, b = _range(term.numerator, lo, hi), _range(term.denominator, lo, hi)
            lower.append(_down(-_interval(term.weight) * b / _IV.exp(1)))
            upper.append(_up(_interval(term.weight) * a * _IV.ln(a / b)))
        lower, upper = np.array(lower), np.array(upper)
        c = np.r_[[float(v) for v in self.objective[1:]], np.ones(m)]
        offset = _interval(self.objective[0])
        for v, r, l, u in zip(self.objective[1:], c[:n], lo, hi):
            offset += (_interval(v) - _interval(Fraction(float(r)))) * _IV.mpf([l, u])
        A, rhs = [], []
        for data, bound in self.rows:
            slope, rounded_bound = _rounded_row(data, bound, lo, hi)
            A.append(np.r_[slope, np.zeros(m)])
            rhs.append(rounded_bound)
        points = [(lo + hi) / 2]
        if incumbent_x is not None:
            inc = np.asarray(torch.as_tensor(incumbent_x).detach().cpu()).reshape(-1).copy()
            if inc.size == n and np.isfinite(inc).all():
                for j in self.exp_variables:
                    inc[j] = np.exp(inc[j])
                points.append(np.clip(inc, lo, hi))
        best, rounds = -np.inf, 0
        fo = cert_crown_dual()
        # cert_crown_dual: the kernel stops once the bound reaches P - tolerance (the certificate is re-evaluated anyway)
        stop = None
        if fo and target is not None and math.isfinite(target):
            stop = float(target) - tolerance - float(offset.mid) - 1e-9 * max(1., abs(float(target)))
        while time.monotonic() - start < time_budget:
            for point in points:
                # Cut points need not be feasible. Keep exp numerators away from
                # numerical zero even if HiGHS ignores a very small lower bound.
                point = np.clip(point, lo, hi)
                for j in self.exp_variables:
                    point[j] = max(point[j], min(hi[j], max(lo[j], 1e-10 * hi[j])))
                for k, term in enumerate(self.terms):
                    slope, intercept = term.support(point, lo, hi)
                    row = np.r_[slope, np.zeros(m)]
                    row[n + k] = -1.
                    A.append(row)
                    rhs.append(_up(-_interval(Fraction(intercept))))
            matrix, bounds = np.array(A), np.array(rhs)
            if fo:
                from nlp_solver.gp_certificate import _master
                out = _master(c, matrix, bounds, lower, upper, stop, time.time() + max(.001, time_budget - (time.monotonic() - start)))
                rounds += 1
                if out is None:
                    break
                point, multipliers = out
            else:
                # Run serial simplex without changing HiGHS's process-wide thread
                # count: a preceding root LP may already have initialized its pool.
                # SciPy forwards the native option with an otherwise harmless warning.
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message="Unrecognized options detected:.*parallel",
                                            category=OptimizeWarning)
                    result = linprog(c, A_ub=matrix, b_ub=bounds, bounds=np.stack([lower, upper], 1),
                                     method='highs-ds', options={'parallel': False,
                                         'time_limit': max(.001, time_budget - (time.monotonic() - start))})
                rounds += 1
                if result.status != 0 or result.ineqlin.marginals is None:
                    break  # never trust a floating-point infeasibility claim
                point, multipliers = result.x, -result.ineqlin.marginals
            dual = _lp_dual(c, matrix, bounds, lower, upper, multipliers)
            value = _down(_interval(Fraction(dual)) + offset)
            best = max(best, min(value, target) if fo and target is not None else value)
            if sliced:
                yield Atom("exp_certificate", bound=best)
            if target is not None and 0 <= target - best <= tolerance:
                break
            points = [point[:n]]
        return best, rounds


def build_exp_certificate(bounded_module, model_ori, x, C, rhs, lower, upper, diagnostics=None, *, rowpool):
    diagnostics = diagnostics if diagnostics is not None else {}

    def reject(reason):
        diagnostics['reason'] = reason
        return None

    lo = lower.detach().cpu().double().numpy().reshape(-1).copy()
    hi = upper.detach().cpu().double().numpy().reshape(-1).copy()
    if not (np.isfinite(lo).all() and np.isfinite(hi).all() and (lo <= hi).all()):
        return reject('requires a finite root box')
    if C.ndim != 3 or C.shape[0] != 1 or rhs.shape[0] != 1:
        return reject('requires a single conjunctive specification')
    if not any(type(m).__name__ == 'BoundExp' for m in bounded_module.modules()):
        # output_forms finds exp variables only at BoundExp nodes; without one the exact walk of the
        # whole graph (up to 103 s on large pooling models, past the deadline) can only reject.
        return reject('no exp substitution')
    outputs, exp_variables = output_forms(bounded_module, x)
    if C.shape[-1] != len(outputs) or C.shape[1] != rhs.numel() or not torch.isfinite(C).all():
        return reject('invalid specification dimensions or coefficients')
    if not exp_variables:
        return reject('no exp substitution')
    # A local RNG preserves all primal and BaB random seeds. This is only a sanity
    # check; exact graph algebra above is the proof of the symbolic identities.
    rng = np.random.default_rng(1729)
    try:
        for _ in range(3):
            point = rng.uniform(lo, hi)
            z = torch.as_tensor(point, dtype=x.dtype, device=x.device).reshape(x.shape)
            with torch.no_grad():
                actual = model_ori(z).detach().cpu().double().numpy().reshape(-1)
            if actual.size != len(outputs):
                return reject('symbolic/model sanity mismatch')
            y = [_IV.exp(_interval(Fraction(float(p)))) for p in point]  # once per point, not per output
            for f, value in zip(outputs, actual):
                if f is not None:
                    predicted = _evaluate(f, point, y)
                    if (not math.isfinite(value) or not math.isfinite(predicted)
                            or abs(predicted - value) > 1e-8 * max(1., abs(value))):
                        return reject('symbolic/model sanity mismatch')
    except (ValueError, OverflowError, ZeroDivisionError):
        return reject('symbolic/model sanity mismatch')
    finally:
        bounded_module(x)
    signed = []
    for row in C[0].detach().cpu().double().numpy():
        f = {}
        for c, output in zip(row, outputs):
            if c:
                f = _sum(f, _scale(output, Fraction(float(c))))
        signed.append(f)
    parsed = _objective(signed[0], len(lo), exp_variables)
    if parsed is None:
        return reject('objective is not a convex relative-entropy sum')
    objective, terms = parsed
    for j in exp_variables:
        lo[j] = max(0., _down(_IV.exp(_interval(Fraction(float(lo[j]))))))
        hi[j] = _up(_IV.exp(_interval(Fraction(float(hi[j])))))
    relaxed = rowpool.thresholds(rhs).detach().cpu().double().numpy().reshape(-1)
    rows, dropped = [], []
    for r, f in enumerate(signed[1:], 1):
        if np.isposinf(relaxed[r]):
            continue
        data = _affine(f, len(lo), exp_variables)
        if data is None or not math.isfinite(relaxed[r]):
            dropped.append(r)
            continue
        bound = Fraction(float(relaxed[r]))
        rows.append((data, bound))
        # Nonnegative exp rows bound each y_j. Affine raw variables are allowed:
        # minimize their contribution over their original boxes as well.
        if all(data[j + 1] >= 0 for j in exp_variables):
            for j in exp_variables:
                if data[j + 1] > 0:
                    rest = list(data)
                    rest[j + 1] = Fraction(0)
                    cap = _up((_interval(bound) - _range(rest, lo, hi)) / _interval(data[j + 1]))
                    hi[j] = min(hi[j], cap)
    if not (np.isfinite(lo).all() and np.isfinite(hi).all() and (lo <= hi).all()):
        return reject('empty or numerically unbounded transformed box')
    if any(_down(_range(t.numerator, lo, hi)) <= 0 or
           _down(_range(t.denominator, lo, hi)) <= 0 for t in terms):
        return reject('entropy argument is not strictly positive on the transformed box')
    diagnostics.update(reason='proved exp-substitution convexity', rows=len(rows),
                       dropped_rows=dropped, terms=len(terms), exp_variables=sorted(exp_variables),
                       upper=hi.tolist())
    return ExpCertificate(objective, terms, rows, lo, hi, exp_variables)
