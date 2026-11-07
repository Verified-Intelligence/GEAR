"""Elementwise DCP certificates on the root box and global convex tangents.

Curvature is structural; IBP is used only for domains and monotonicity. Unsupported
operators, poles, nonfinite intervals and ambiguous signs give no certificate.
No live LiRPA interval/bound caches are used or changed by the interval walk.
"""
from __future__ import annotations

import copy
import math
from fractions import Fraction

import torch

from .interval_hessian import (IntervalHessian, Jet, outward, add, neg, mul, power,
                               jet_add, jet_mul, jet_unary)
from .quadratic_tangent import QuadraticTangent, output_degrees, _const, _is_const, _DATA_MOVEMENT

# Bit flags: affine satisfies both curvature properties; unknown satisfies neither.
UNKNOWN, CONVEX, CONCAVE, AFFINE = 0, 1, 2, 3

# Univariate curvature: f'' is enclosed on fixed slices of one variable's root interval,
# 64 uniform ones plus dyadic ones shrinking toward both ends (a term whose f'' touches 0
# at a box end, like x*exp(-c*x^3) at 0, has an enclosure sign only on a tiny end slice:
# 8e-10 over [0, .16], 1e-30 over [0, 6e-7]). The jets of all nodes together are capped.
_EDGES = torch.tensor(sorted({k / 64 for k in range(65)} | {f(2. ** -k) for k in range(7, 25) for f in (float, lambda t: 1 - t)}), dtype=torch.float64)
SLICES, MAX_JET_SCALARS = len(_EDGES) - 1, 1 << 22
CONST, MULTI = -1, -2  # support ids besides a variable index
_FAILURES = (NotImplementedError, ValueError, RuntimeError, TypeError, IndexError, AttributeError)
# Operators affine in their inputs: a jet-proved term with a cut error may feed these only.
_AFFINE_OPS = _DATA_MOVEMENT | {'BoundGatherElements', 'BoundNeg', 'BoundAdd', 'BoundSub', 'BoundLinear',
                                'BoundMatMul', 'BoundReduceSum', 'BoundReduceMean', 'BoundATenIndex_add'}
_UNARY = ('BoundSqr', 'BoundPow', 'BoundReciprocal', 'BoundExp', 'BoundLog', 'BoundSqrt',
          'BoundSin', 'BoundCos', 'BoundTanh')


def _negative(c):
    return ((c & CONVEX) << 1) | ((c & CONCAVE) >> 1)


def _scale(c, weight):
    return torch.where(weight > 0, c, torch.where(weight < 0, _negative(c), AFFINE))


def _up(t, terms=1):
    """t rounded up; an exact 0 stays 0 (rounding it to the smallest subnormal marked every
    exactly proved flag as inexact, and cute_batch lost its convex objective)."""
    return torch.where(t == 0, t, outward(t, t, t, terms)[1])


def _strict(c, error):
    """Flags whose curvature is proved exactly (no cut error), the only ones nonlinear rules may use."""
    return c & ~((error[0] > 0).to(c.dtype) * CONVEX | (error[1] > 0).to(c.dtype) * CONCAVE)


def _scale_error(error, weight):
    """Cut errors of weight*f: |weight|*error, a negative weight swaps the convex and concave ones."""
    swap = weight < 0
    return (_up(torch.where(swap, error[1], error[0]) * weight.abs()),
            _up(torch.where(swap, error[0], error[1]) * weight.abs()))


def signed_curvature(curvature, coefficients):
    """Curvature of each signed spec row, ignoring exactly zero coefficients."""
    c = _scale(curvature.cpu(), coefficients.detach().cpu())
    return ((c & CONVEX).ne(0).all(-1).to(torch.uint8)
            | ((c & CONCAVE).ne(0).all(-1).to(torch.uint8) << 1))


class ConvexityProof:
    """One root IBP/DCP walk. Values have original node shapes, on the CPU."""

    @torch.no_grad()
    def __init__(self, bounded_module, x, lower, upper):
        self.module = bounded_module
        self.x = x
        self.order, self.shapes, self.curvature, self.bounds = [], {}, {}, {}
        # errors[node] = (convex, concave): how far below its tangents the element may lie
        # on the root box when it carries that flag (0 = proved exactly; see _univariate).
        self.errors = {}
        self.movements, self.forms = {}, {}
        seen = set()
        bounded_module(x)
        final = bounded_module[bounded_module.final_name]

        def visit(node):
            if node.name in seen:
                return
            seen.add(node.name)
            for parent in node.inputs:
                visit(parent)
            self.order.append(node)

        visit(final)
        lo, hi = lower.detach().cpu().double().reshape(x.shape), upper.detach().cpu().double().reshape(x.shape)
        if not (torch.isfinite(lo).all() and torch.isfinite(hi).all() and (lo <= hi).all()):
            raise ValueError('Convexity proof requires a finite ordered root box')
        try:
            self._walk(lo, hi, univariate=False)
            if (self.curvature[final.name] == UNKNOWN).any():
                self._walk(lo, hi, univariate=True)
        finally:
            bounded_module(x)  # Restore forward caches after movement probes.
        self.output = self.curvature[final.name].reshape(-1)
        self.output_errors = tuple(t.reshape(-1) for t in self.errors[final.name])

    def cut_error(self, coefficients):
        """Rounded-up amount a tangent of sum(coefficients*outputs) may exceed it on the root box."""
        c = coefficients.detach().cpu().double()
        e = torch.where(c > 0, self.output_errors[0], self.output_errors[1]) * c.abs()
        total = e.sum()  # exact proofs stay exactly 0, with no rounding margin
        return 0. if total == 0 else float(outward(total, total, total, e.numel() + 1)[1])

    def _walk(self, lo, hi, univariate):
        self.jets, self.jet_budget = {}, MAX_JET_SCALARS
        for node in self.order:
            self.shapes[node.name] = tuple(node.forward_value.shape)
            self.errors[node.name] = (torch.zeros(self.shapes[node.name]),) * 2
            # Movement forward methods may update indexing/shape attributes.
            state = node.__dict__.copy()
            try:
                try:
                    c, interval = self._node(node, lo, hi)
                except _FAILURES:
                    c = torch.zeros(self.shapes[node.name], dtype=torch.uint8)
                    interval = (torch.full(self.shapes[node.name], -torch.inf),
                                torch.full(self.shapes[node.name], torch.inf))
                if univariate:
                    try:
                        c, interval = self._univariate(node, c, interval, lo, hi)
                    except _FAILURES:
                        pass  # an unsupported jet never discards the DCP result
                valid = torch.isfinite(interval[0]) & torch.isfinite(interval[1]) & (interval[0] <= interval[1])
                self.curvature[node.name] = torch.where(valid, c, UNKNOWN).to(torch.uint8)
                self.bounds[node.name] = interval
            except _FAILURES:  # an interval that cannot even be formed
                self.curvature[node.name] = torch.full(self.shapes[node.name], UNKNOWN, dtype=torch.uint8)
                self.bounds[node.name] = (torch.full(self.shapes[node.name], -torch.inf),
                                          torch.full(self.shapes[node.name], torch.inf))
            finally:
                node.__dict__.clear()
                node.__dict__.update(state)

    def _univariate(self, node, c, interval, lo, hi):
        """Add the curvature of elements that depend on ONE variable, proved by f''.

        Each element carries a 1-D jet (f, f', f'') in its own variable over SLICES
        slices of that variable's root interval; elements of several variables are
        MULTI and never use a jet. A sign of f'' on every slice proves it on the box.
        """
        self.jets[node.name] = None
        shape, name, ins = self.shapes[node.name], type(node).__name__, node.inputs
        size = math.prod(shape)
        if size * SLICES > self.jet_budget:
            return c, interval
        parents = [self.jets.get(i.name) for i in ins]
        zero = torch.zeros(SLICES, size, dtype=torch.float64)
        if name == 'BoundInput':
            edges = lo.reshape(1, -1) + (hi - lo).reshape(1, -1) * _EDGES[:, None]
            bottom = torch.maximum(torch.nextafter(edges[:-1], torch.full_like(edges[:-1], -torch.inf)), lo.reshape(1, -1))
            top = torch.minimum(torch.nextafter(edges[1:], torch.full_like(edges[1:], torch.inf)), hi.reshape(1, -1))
            support = torch.arange(size).reshape(shape)
            jet = Jet((bottom, top), (torch.ones(SLICES, size, 1),) * 2, (zero[..., None, None],) * 2,
                      torch.ones(SLICES, size, dtype=torch.bool))
        elif _is_const(node):
            v = _const(node).double().reshape(1, -1).expand(SLICES, -1)
            support = torch.full(shape, CONST)
            jet = Jet((v, v), (zero[..., None],) * 2, (zero[..., None, None],) * 2, torch.isfinite(v))
        elif any(p is None for p in parents) or name not in (
                _DATA_MOVEMENT | {'BoundGatherElements', 'BoundNeg', 'BoundAdd', 'BoundSub', 'BoundMul', 'BoundDiv', *_UNARY}):
            return c, interval
        elif name in _DATA_MOVEMENT or name == 'BoundGatherElements':
            slots, indices = self._movement(node)
            support = torch.cat([parents[i][0].reshape(-1) for i in slots])[indices].reshape(shape)
            fields = [tuple(torch.cat([getattr(parents[i][1], attr)[k] for i in slots], 1) for k in (0, 1))
                      for attr in ('value', 'gradient', 'hessian')]
            jet = Jet(*fields, torch.cat([parents[i][1].valid for i in slots], 1)).select(indices)
        elif name in ('BoundNeg', *_UNARY):
            support, a = parents[0][0], parents[0][1]
            exponent = float(_const(ins[1])) if name == 'BoundPow' else None
            jet = (Jet(*(neg(p) for p in (a.value, a.gradient, a.hessian)), a.valid) if name == 'BoundNeg'
                   else jet_unary(a, name, exponent))
        else:
            ids, jets = [], []
            for (support, a), parent in zip(parents, ins):
                index = torch.arange(math.prod(self.shapes[parent.name])).reshape(self.shapes[parent.name]).expand(shape).reshape(-1)
                ids.append(support.reshape(-1)[index])
                jets.append(a.select(index))
            a, b = jets
            if name == 'BoundSub':
                b = Jet(*(neg(p) for p in (b.value, b.gradient, b.hessian)), b.valid)
            elif name == 'BoundDiv':
                b = jet_unary(b, 'BoundReciprocal').finish()
            jet = jet_add(a, b) if name in ('BoundAdd', 'BoundSub') else jet_mul(a, b)
            x, y = ids
            same = (x == y) | (x == CONST) | (y == CONST)
            support = torch.where((x == MULTI) | (y == MULTI) | ~same, MULTI, torch.maximum(x, y)).reshape(shape)
        jet.valid = jet.valid & (support != MULTI).reshape(1, -1)
        self.jets[node.name] = support, jet.finish()
        self.jet_budget -= size * SLICES
        second = self.jets[node.name][1].hessian
        proved = self.jets[node.name][1].valid.all(0).reshape(shape) & (support >= 0)
        low, high = second[0][..., 0, 0], second[1][..., 0, 0]
        # Deficit d = how far f'' may go below 0 (convex) / above 0 (concave) on any slice.
        # f'' >= -d on the box gives f(x) >= tangent - d/2 (x-x0)^2, so a tangent cut is off by
        # at most d*w^2/2 (w = the variable's width); d = 0 when every slice proves the sign.
        # Only a tiny d is accepted (f'' touching 0 at a box end, x*exp(-c*x^3) at 0, is
        # enclosed with rounding noise), and such an element may only feed affine operators.
        width = (hi - lo).reshape(-1)[support.clamp(min=0).reshape(-1)].reshape(shape)
        scale = torch.maximum(low.abs(), high.abs()).amax(0).reshape(shape)
        errors = []
        for k, deficit in enumerate(((-low).clamp(min=0), high.clamp(min=0))):  # convex, concave
            d = deficit.amax(0).reshape(shape)
            error = outward(d * width * width / 2, d * width * width / 2, d * width * width / 2, 3)[1]
            current = torch.where((c & (1 << k)) != 0, self.errors[node.name][k], torch.inf)
            best = torch.minimum(current, torch.where(proved & (d <= 1e-9 * scale), error, torch.inf))
            c = c | torch.where(torch.isfinite(best), 1 << k, 0)
            errors.append(torch.where(torch.isfinite(best), best, 0.))
        self.errors[node.name] = tuple(errors)
        # The slices' hull also encloses the value, which lets operators the DCP walk lacks (sin) through.
        value = self.jets[node.name][1].value
        hull = tuple(t.reshape(SLICES, *shape) for t in (value[0], value[1]))
        interval = (torch.where(proved, torch.maximum(interval[0], hull[0].amin(0)), interval[0]),
                    torch.where(proved, torch.minimum(interval[1], hull[1].amax(0)), interval[1]))
        return c, interval

    def _movement(self, node):
        if node.name not in self.movements:
            # Reuse index compilation; never compile dense affine maps for a scan.
            compiler = object.__new__(IntervalHessian)
            compiler.shapes, compiler.n = self.shapes, self.x.numel()
            rule = compiler._compile(node)
            if rule[0] != 'movement':
                raise NotImplementedError(type(node).__name__)
            self.movements[node.name] = rule[1:]
        return self.movements[node.name]

    def _linear(self, node, c, intervals, fn, terms, errors, bias=0.):
        """fn has nonnegative weights; negatives are supplied as separate terms."""
        lower, upper = fn([v[0] for v in intervals]), fn([v[1] for v in intervals])
        magnitude = fn([torch.maximum(v[0].abs(), v[1].abs()) for v in intervals])
        bias = torch.as_tensor(bias, dtype=lower.dtype)
        interval = add(outward(lower, upper, magnitude, terms), (bias, bias))
        self.errors[node.name] = tuple(_up(t, terms) for t in (
            fn([e[0] for e in errors]), fn([e[1] for e in errors])))
        bad_convex = fn([((v & CONVEX) == 0).double() for v in c]) > 0
        bad_concave = fn([((v & CONCAVE) == 0).double() for v in c]) > 0
        return (~bad_convex).to(torch.uint8) | ((~bad_concave).to(torch.uint8) << 1), interval

    def _node(self, node, lower, upper):
        name, ins, shape = type(node).__name__, node.inputs, self.shapes[node.name]
        if name == 'BoundInput':
            if math.prod(shape) != self.x.numel():
                raise NotImplementedError('multiple perturbed inputs')
            return torch.full(shape, AFFINE, dtype=torch.uint8), (lower, upper)
        if _is_const(node):
            v = _const(node).double()
            return torch.full(shape, AFFINE, dtype=torch.uint8), (v, v)
        c, v = [self.curvature[i.name] for i in ins], [self.bounds[i.name] for i in ins]
        e = [self.errors[i.name] for i in ins]
        if not (name in _AFFINE_OPS or (name == 'BoundMul' and any(_is_const(i) for i in ins))
                or (name == 'BoundDiv' and _is_const(ins[1]))):
            c = [_strict(a, b) for a, b in zip(c, e)]
        if name in _DATA_MOVEMENT or name == 'BoundGatherElements':
            slots, indices = self._movement(node)
            curvature = torch.cat([c[i].reshape(-1) for i in slots])[indices].reshape(shape)
            bounds = tuple(torch.cat([v[i][k].reshape(-1) for i in slots])[indices].reshape(shape) for k in (0, 1))
            self.errors[node.name] = tuple(torch.cat([e[i][k].reshape(-1) for i in slots])[indices].reshape(shape) for k in (0, 1))
            return curvature, bounds
        if name == 'BoundNeg':
            self.errors[node.name] = e[0][1], e[0][0]
            return _negative(c[0]), neg(v[0])
        if name in ('BoundAdd', 'BoundSub'):
            other = e[1] if name == 'BoundAdd' else e[1][::-1]
            self.errors[node.name] = tuple(_up(a + b) for a, b in zip(e[0], other))
            return (c[0] & (c[1] if name == 'BoundAdd' else _negative(c[1])),
                    add(v[0], v[1] if name == 'BoundAdd' else neg(v[1])))
        if name in ('BoundMul', 'BoundDiv'):
            if name == 'BoundDiv':
                if type(ins[1]).__name__ == 'BoundSqrt' and ins[0].inputs:
                    raise NotImplementedError('normalization division')
                denom = v[1]
                valid = (((denom[0] > 0) | (denom[1] < 0))
                         & torch.isfinite(v[0][0]) & torch.isfinite(v[0][1]))
                interval = mul(v[0], power(denom, -1))
                interval = (torch.where(valid, interval[0], -torch.inf),
                            torch.where(valid, interval[1], torch.inf))
                if _is_const(ins[1]):
                    self.errors[node.name] = _scale_error(e[0], 1. / _const(ins[1]))
                    return _scale(c[0], 1. / _const(ins[1])), interval
                if _is_const(ins[0]):
                    reciprocal = self._unary(c[1], denom, 'BoundReciprocal', -1)[0]
                    return _scale(reciprocal, _const(ins[0])), interval
                curvature = self._fraction(ins[0], ins[1], shape)
                return torch.where(valid, curvature, UNKNOWN), interval
            interval = mul(v[0], v[1])
            defined = (torch.isfinite(v[0][0]) & torch.isfinite(v[0][1])
                       & torch.isfinite(v[1][0]) & torch.isfinite(v[1][1]))
            interval = (torch.where(defined, interval[0], -torch.inf),
                        torch.where(defined, interval[1], torch.inf))
            for k in (0, 1):
                if _is_const(ins[k]):
                    self.errors[node.name] = _scale_error(e[1-k], _const(ins[k]))
                    return torch.where(defined, _scale(c[1-k], _const(ins[k])), UNKNOWN), interval
                if type(ins[k]).__name__ == 'BoundReciprocal':
                    return self._fraction(ins[1-k], ins[k].inputs[0], shape), interval
            return torch.full(shape, UNKNOWN, dtype=torch.uint8), interval
        if name in ('BoundExp', 'BoundLog', 'BoundSqrt', 'BoundReciprocal', 'BoundSqr', 'BoundPow'):
            k = {'BoundSqr': 2., 'BoundReciprocal': -1., 'BoundSqrt': .5}.get(name)
            if name == 'BoundPow':
                if not _is_const(ins[1]) or _const(ins[1]).numel() != 1:
                    raise NotImplementedError('variable exponent')
                k = float(_const(ins[1]))
            return self._unary(c[0], v[0], name, k)
        if name in ('BoundMatMul', 'BoundLinear'):
            if name == 'BoundLinear':
                if getattr(node, 'transA', 0) or not all(_is_const(i) for i in ins[1:]):
                    raise NotImplementedError('nonconstant linear weights')
                _, weight, bias = node._preprocess(torch.zeros_like(v[0][0]), *[_const(i).double() for i in ins[1:]])
                # GEMM alpha scales the data, not the weights returned by
                # _preprocess. Keep its sign exact for the curvature proof.
                alpha = torch.as_tensor(getattr(node, 'alpha_linear', 1.), dtype=torch.float64)
                defined = torch.isfinite(v[0][0]) & torch.isfinite(v[0][1])
                c[0], scaled = _scale(c[0], alpha), mul(v[0], (alpha, alpha))
                e[0] = _scale_error(e[0], alpha)
                v[0] = (torch.where(defined, scaled[0], -torch.inf),
                        torch.where(defined, scaled[1], torch.inf))
                weight = weight.T
                fn = lambda a: a[0] @ weight.clamp(min=0) + a[1] @ (-weight).clamp(min=0)
                terms = weight.shape[-2]
            elif _is_const(ins[1]):
                weight, bias = _const(ins[1]).double(), 0.
                fn = lambda a: a[0] @ weight.clamp(min=0) + a[1] @ (-weight).clamp(min=0)
                terms = v[0][0].shape[-1]
            elif _is_const(ins[0]):
                weight, bias = _const(ins[0]).double(), 0.
                fn = lambda a: weight.clamp(min=0) @ a[0] + (-weight).clamp(min=0) @ a[1]
                c, v, e = [c[1]], [v[1]], [e[1]]
                terms = weight.shape[-1]
            else:
                raise NotImplementedError('two perturbed matmul factors')
            if bias is None:
                bias = 0.
            if not torch.isfinite(weight).all():
                raise NotImplementedError('nonfinite weights')
            return self._linear(node, [c[0], _negative(c[0])], [v[0], neg(v[0])], fn, terms, [e[0], e[0][::-1]], bias)
        if name in ('BoundReduceSum', 'BoundReduceMean'):
            if not all(_is_const(i) for i in ins[1:]):
                raise NotImplementedError('variable reduction axes')
            return self._linear(node, [c[0]], [v[0]], lambda a: node.forward(a[0]), v[0][0].numel(), [e[0]])
        if name == 'BoundATenIndex_add':
            if not all(_is_const(ins[i]) for i in (1, 2, 4)):
                raise NotImplementedError('variable scatter controls')
            alpha = float(_const(ins[4]))
            def fn(a):
                return node.forward(a[0], _const(ins[1]), _const(ins[2]), a[1], abs(alpha))
            source_c, source_v = (_negative(c[3]), neg(v[3])) if alpha < 0 else (c[3], v[3])
            source_e = e[3][::-1] if alpha < 0 else e[3]
            return self._linear(node, [c[0], source_c], [v[0], source_v], fn, v[3][0].numel() + 1, [e[0], source_e])
        if name == 'BoundWhere' and _is_const(ins[0]):
            condition = _const(ins[0]).bool()
            return torch.where(condition, c[1], c[2]), tuple(torch.where(condition, v[1][k], v[2][k]) for k in (0, 1))
        raise NotImplementedError(name)

    def _unary(self, c, v, name, k):
        lo, hi = v
        convex, concave = (c & CONVEX) != 0, (c & CONCAVE) != 0
        affine = c == AFFINE
        result = torch.zeros_like(c)
        if name == 'BoundExp':
            return torch.where(convex, CONVEX, UNKNOWN), outward(lo.exp(), hi.exp())
        if name == 'BoundLog':
            return torch.where(concave & (lo > 0), CONCAVE, UNKNOWN), outward(lo.log(), hi.log())
        if k is None or not math.isfinite(k):
            raise NotImplementedError('nonfinite power')
        interval = power(v, k)
        domain = (torch.isfinite(lo) & torch.isfinite(hi) & (lo <= hi))
        if k != int(k):
            domain &= lo >= 0 if k > 0 else lo > 0
        elif k < 0:
            domain &= (lo > 0) | (hi < 0)
        interval = (torch.where(domain, interval[0], -torch.inf),
                    torch.where(domain, interval[1], torch.inf))
        if k == 0:
            result = torch.full_like(c, AFFINE)
        elif k == 1:
            result = c
        elif k == -1:
            result = torch.where(concave & (lo > 0), CONVEX, result)
            result = torch.where(convex & (hi < 0), CONCAVE, result)
        elif k > 1:
            even = k == int(k) and int(k) % 2 == 0
            result = torch.where(convex & (lo >= 0), CONVEX, result)
            if even:
                result = torch.where(affine | (concave & (hi <= 0)), CONVEX, result)
        elif 0 < k < 1:
            result = torch.where(concave & (lo >= 0), CONCAVE, result)
        return torch.where(domain, result, UNKNOWN), interval

    def _form(self, node, index):
        """Exact scalar affine a*z+b in a shared atom, only for fraction rewrites.

        Fractions avoid numerical proportionality guesses. Unhandled affine
        expressions become distinct atoms, so the rewrite is conservative.
        """
        key = node.name, index
        if key in self.forms:
            return self.forms[key]
        name, ins = type(node).__name__, node.inputs
        result = (key, Fraction(1), Fraction(0))
        if _is_const(node):
            result = (None, Fraction(0), Fraction(float(_const(node).reshape(-1)[index])))
        elif name in _DATA_MOVEMENT or name == 'BoundGatherElements':
            slots, indices = self._movement(node)
            position = int(indices[index])
            for slot in slots:
                size = math.prod(self.shapes[ins[slot].name])
                if position < size:
                    result = self._form(ins[slot], position)
                    break
                position -= size
        elif name in ('BoundAdd', 'BoundSub', 'BoundMul', 'BoundNeg'):
            args = []
            for parent in ins:
                indices = torch.arange(math.prod(self.shapes[parent.name])).reshape(self.shapes[parent.name])
                position = int(indices.expand(self.shapes[node.name]).reshape(-1)[index])
                args.append(self._form(parent, position))
            if name == 'BoundNeg':
                atom, a, b = args[0]
                result = atom, -a, -b
            elif name == 'BoundMul':
                for i in (0, 1):
                    if args[i][1] == 0:
                        atom, a, b = args[1-i]
                        result = atom, a * args[i][2], b * args[i][2]
                        break
            else:
                atom, a, b = args[0]
                other, d, e = args[1]
                if name == 'BoundSub':
                    d, e = -d, -e
                if atom == other or a == 0 or d == 0:
                    result = (other if a == 0 else atom), a + d, b + e
        self.forms[key] = result
        return result

    def _fraction(self, numerator, denominator, shape):
        result = torch.zeros(shape, dtype=torch.uint8)
        nc, dc = self.curvature[numerator.name], self.curvature[denominator.name]
        dv = self.bounds[denominator.name]
        ni = torch.arange(nc.numel()).reshape(nc.shape).expand(shape).reshape(-1)
        di = torch.arange(dc.numel()).reshape(dc.shape).expand(shape).reshape(-1)
        for j, (a_index, b_index) in enumerate(zip(ni.tolist(), di.tolist())):
            if nc.reshape(-1)[a_index] != AFFINE or dc.reshape(-1)[b_index] != AFFINE:
                continue
            positive = dv[0].reshape(-1)[b_index] > 0
            negative = dv[1].reshape(-1)[b_index] < 0
            if not (positive or negative):
                continue
            atom, a, b = self._form(numerator, a_index)
            other, d, e = self._form(denominator, b_index)
            if d == 0 or (atom != other and a != 0):
                continue
            # (a*z+b)/(d*z+e) = a/d + (b-a*e/d)/(d*z+e).
            beta = b - a * e / d
            result.reshape(-1)[j] = (AFFINE if beta == 0 else
                                     CONVEX if (beta > 0) == bool(positive) else CONCAVE)
        return result


class ConvexTangent(QuadraticTangent):
    """QTB interface, zero curvature correction, certified point AD rounding."""

    def __init__(self, objective_fn, certificate, x_hat, lower, upper, coefficients, cut_error=0.):
        self.objective_fn, self.certificate = objective_fn, certificate
        self.cut_error = cut_error  # proved amount the function may lie below its tangents (_univariate)
        self.coefficients = coefficients.detach().cpu().double()
        self.n = lower.numel()
        self.root_lower, self.root_upper = lower.reshape(-1).double(), upper.reshape(-1).double()
        self.fixed_mask = self.root_lower == self.root_upper
        self.fixed_values = torch.where(self.fixed_mask, self.root_lower, 0.)
        self.lam_lb = 0.
        self.polished = False
        self.linear_constraint_outputs = False
        self.update(x_hat)

    def update(self, x_hat):
        xh = x_hat.detach().reshape(-1).to(self.root_lower)
        # A certificate on the root box does not license an anchor outside it.
        if xh.numel() != self.n or not torch.isfinite(xh).all() or (xh < self.root_lower).any() or (xh > self.root_upper).any():
            return False
        xh = xh.clone().requires_grad_(True)
        f = self.objective_fn(xh)
        g, = torch.autograd.grad(f, xh)
        # Signed rows at the same incumbent share their point interval AD.
        certificate = self.certificate
        if not hasattr(certificate, 'anchor_x') or not torch.equal(certificate.anchor_x, xh.detach()):
            certificate.anchor_x = xh.detach().clone()
            certificate.anchor_jet = certificate.evaluate(xh.detach()[None], xh.detach()[None])
        jet = certificate.anchor_jet
        used = self.coefficients != 0
        if not (torch.isfinite(f) and torch.isfinite(g).all() and jet.valid[:, used].all()):
            return False
        weights = self.coefficients[used].to(xh)
        value = mul(tuple(t[:, used] for t in jet.value), (weights, weights))
        gradient = mul(tuple(t[:, used] for t in jet.gradient), (weights[:, None], weights[:, None]))
        value = outward(value[0].sum(1), value[1].sum(1), torch.maximum(value[0].abs(), value[1].abs()).sum(1), weights.numel())
        gradient = outward(gradient[0].sum(1), gradient[1].sum(1), torch.maximum(gradient[0].abs(), gradient[1].abs()).sum(1), weights.numel())
        f_error = max(abs(float(f.detach()) - float(t[0])) for t in value)
        g_error = torch.maximum(*(abs(g.detach() - t[0]) for t in gradient))
        if not (math.isfinite(f_error) and torch.isfinite(g_error).all()):
            return False
        self.x_hat, self.f, self.g = xh.detach(), float(f.detach()), g.detach()
        self.f_error, self.g_error = f_error, g_error
        return True

    def plane(self, l, u):
        lo, hi = l.double(), u.double()
        if (lo < self.root_lower.to(lo)).any() or (hi > self.root_upper.to(hi)).any():
            raise ValueError('Convex tangent used outside its certified root box')
        a, b = super().plane(lo, hi)
        displacement = add((lo, hi), (-self.x_hat.to(lo), -self.x_hat.to(lo)))
        distance = torch.maximum(displacement[0].abs(), displacement[1].abs())
        terms = mul((self.g_error.to(lo),) * 2, (distance, distance))[1]
        total = terms.sum(-1) + self.f_error + self.cut_error
        error = outward(total, total, total, self.n)[1]
        converted = a.to(l)
        cast_error = ((converted.double() - a).abs() * torch.maximum(lo.abs(), hi.abs())).sum(-1)
        b = outward(b - error - cast_error, b - error - cast_error, b.abs() + error + cast_error, self.n)[0]
        return converted, torch.nextafter(b.to(l), torch.full_like(l[:, 0], -torch.inf))


def _certificate(bounded_module, x, max_affine=2_000_000):
    certificate = IntervalHessian(bounded_module, x, 0, first_order=True, max_affine=max_affine)
    # All signed outputs share one first-order evaluation; its usual scalar
    # selection can also take a vector of indices.
    certificate.obj_index = list(range(math.prod(certificate.shapes[certificate.final_name])))
    return certificate


def build_convex_tangent(bounded_module, model_ori, x_input, obj_index, x_hat,
                         root_lower, root_upper, *, enabled=False, max_vars=2000):
    if not enabled or x_hat is None or x_input.numel() > max_vars:
        return None
    # QTB retains priority, including when its option is off.
    degrees = output_degrees(bounded_module, x_input)
    if obj_index >= degrees.numel() or degrees[obj_index] <= 2:
        return None
    lower = root_lower.flatten(1).amin(0, keepdim=True).detach()
    upper = root_upper.flatten(1).amax(0, keepdim=True).detach()
    if not (torch.isfinite(lower).all() and torch.isfinite(upper).all() and (lower <= upper).all()):
        return None
    certificate = ConvexityProof(bounded_module, x_input, lower, upper)
    proof = certificate.output
    if not (int(proof[obj_index]) & CONVEX):
        return None
    model = copy.deepcopy(model_ori).double()
    def objective(z):
        return model(z.reshape(x_input.shape)).reshape(-1)[obj_index]
    coefficients = torch.zeros(proof.numel(), dtype=torch.float64)
    coefficients[obj_index] = 1.
    cut_error = certificate.cut_error(coefficients)
    if not math.isfinite(cut_error):
        return None
    tangent = ConvexTangent(objective, _certificate(bounded_module, x_input), x_hat, lower, upper, coefficients, cut_error)
    if not hasattr(tangent, 'x_hat'):
        return None
    others = torch.cat((proof[:obj_index], proof[obj_index+1:]))
    tangent.linear_constraint_outputs = bool((others == AFFINE).all())
    return tangent


def build_convex_tangent_rows(bounded_module, model_ori, x_input, c, rhs,
                              root_lower, root_upper, x_hat, *, enabled=False,
                              quadratic_rows=False, max_vars=2000, max_rows=50, rowpool):
    from .convex_row_cuts import ConvexRowCuts, build_convex_row_cuts
    if not enabled or x_hat is None or x_input[0].numel() > max_vars or c.shape[1] < 2:
        return None
    if not (torch.equal(c, c[:1].expand_as(c)) and torch.equal(rhs[:, 1:], rhs[:1, 1:].expand_as(rhs[:, 1:]))):
        return None
    lower = root_lower.flatten(1).amin(0, keepdim=True).detach()
    upper = root_upper.flatten(1).amax(0, keepdim=True).detach()
    if not (torch.isfinite(lower).all() and torch.isfinite(upper).all() and (lower <= upper).all()):
        return None
    certificate = ConvexityProof(bounded_module, x_input[:1], lower, upper)
    proof = certificate.output
    if proof.numel() != c.shape[-1]:
        return None
    relaxed_rhs = rowpool.thresholds(rhs)[0]
    curvatures = signed_curvature(proof, c[0])
    model = copy.deepcopy(model_ori).double()
    jets = _certificate(bounded_module, x_input[:1])
    candidates = []
    for r in range(1, c.shape[1]):
        coeff = c[0, r].detach().double()
        if not (int(curvatures[r]) & CONVEX) or int(curvatures[r]) == AFFINE:
            continue  # Existing affine rows already supply their exact planes.
        cut_error = certificate.cut_error(coeff)
        if not torch.isfinite(coeff).all() or not torch.isfinite(relaxed_rhs[r]) or not math.isfinite(cut_error):
            continue
        candidates.append((r, coeff, cut_error))
    if len(candidates) > max_rows:
        print(f"[ConvexRowCuts] skipped: {len(candidates)} convex rows exceed the cap {max_rows}")
        return None  # same cap as build_convex_row_cuts
    tangents, row_rhs, admitted_rows = [], [], []
    for r, coeff, cut_error in candidates:
        def row_fn(z, coeff=coeff):
            outputs = model(z.reshape(x_input[:1].shape)).reshape(-1)
            used = coeff != 0
            return (outputs[used] * coeff[used].to(outputs)).sum()
        tangent = ConvexTangent(row_fn, jets, x_hat, lower, upper, coeff, cut_error)
        if hasattr(tangent, 'x_hat'):
            tangents.append(tangent)
            row_rhs.append(relaxed_rhs[r].detach().double())
            admitted_rows.append(r)
    # Preserve the old Hessian proof for quadratic rows not covered by DCP.
    if quadratic_rows:
        old = build_convex_row_cuts(bounded_module, model_ori, x_input, c, rhs, root_lower, root_upper,
                                    x_hat, enabled=True, skip_rows=admitted_rows,
                                    max_rows=max_rows - len(tangents), rowpool=rowpool)
        if old is not None:
            tangents.extend(old.tangents)
            row_rhs.extend(old.row_rhs)
    return ConvexRowCuts(tangents, row_rhs, lower, upper, x_input.dtype) if tangents else None
