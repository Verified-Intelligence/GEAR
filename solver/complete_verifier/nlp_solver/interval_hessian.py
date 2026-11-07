"""Second-order interval AD on the bound graph, without modifying its bound caches.

All arithmetic is float64 with outward margins (including absolute term sums for
reductions). Invalid elements carry a separate validity mask: an unrelated output
with a pole must not poison the objective. Unsupported dependent operators give
no certificate. Tensor movement is compiled into element indices, so neither the
domain axis nor the derivative axes can accidentally be reshaped or reduced.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .quadratic_tangent import _const, _is_const, _DATA_MOVEMENT


def outward(lo, hi, magnitude=None, terms=1):
    if magnitude is None:
        magnitude = torch.maximum(lo.abs(), hi.abs())
    info = torch.finfo(lo.dtype)
    # Dot products can underflow each product before adding them. A relative
    # margin alone (or one nextafter after the sum) cannot enclose that error.
    error = (magnitude * ((terms + 16) * info.eps * 4)
             + (terms + 16) * info.tiny * info.eps)
    return (torch.nextafter(lo - error, torch.full_like(lo, -torch.inf)),
            torch.nextafter(hi + error, torch.full_like(hi, torch.inf)))


def add(a, b):
    return outward(a[0] + b[0], a[1] + b[1],
                   torch.maximum(a[0].abs(), a[1].abs())
                   + torch.maximum(b[0].abs(), b[1].abs()))


def neg(a):
    return -a[1], -a[0]  # Negation is exact.


def mul(a, b):
    products = torch.stack(torch.broadcast_tensors(
        a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]))
    lo, hi = outward(products.amin(0), products.amax(0))
    # Structural zeros remain exact, even if the other factor is infinite.
    zero = ((a[0] == 0) & (a[1] == 0)) | ((b[0] == 0) & (b[1] == 0))
    return torch.where(zero, 0., lo), torch.where(zero, 0., hi)


def power(a, exponent):
    """Real power on a C2 domain; caller checks that domain separately."""
    if exponent == 0:
        return torch.ones_like(a[0]), torch.ones_like(a[1])
    if exponent == 1:
        return a
    lo, hi = a[0].pow(exponent), a[1].pow(exponent)
    lower, upper = torch.minimum(lo, hi), torch.maximum(lo, hi)
    if exponent > 0 and exponent == int(exponent) and int(exponent) % 2 == 0:
        lower = torch.where((a[0] <= 0) & (a[1] >= 0), 0., lower)
    return outward(lower, upper)


def trig(a, cosine=False):
    """Enclose endpoints and every possible interior extremum.

    Phase comparisons are widened too; huge arguments use [-1,1] rather than
    relying on floating-point period reduction to locate their extrema.
    """
    fn = torch.cos if cosine else torch.sin
    lower = torch.minimum(fn(a[0]), fn(a[1]))
    upper = torch.maximum(fn(a[0]), fn(a[1]))
    phase = 0. if cosine else math.pi / 2
    eps = 1e-12 * (1. + torch.maximum(a[0].abs(), a[1].abs()))
    def contains(offset):
        return (torch.ceil((a[0] - offset - eps) / (2 * math.pi))
                <= torch.floor((a[1] - offset + eps) / (2 * math.pi)))
    wide = (a[1] - a[0] >= 2 * math.pi) | (eps > 1e-3)
    lower = torch.where(wide | contains(phase + math.pi), -1., lower)
    upper = torch.where(wide | contains(phase), 1., upper)
    return outward(lower, upper, torch.ones_like(lower))


def unary(a, name, exponent=None):
    """Intervals of phi, phi', phi'', and the C2-domain mask."""
    valid = torch.ones_like(a[0], dtype=torch.bool)
    one = (torch.ones_like(a[0]), torch.ones_like(a[0]))
    def scale(interval, c):
        t = torch.full_like(a[0], c)
        return mul(interval, (t, t))
    if name in ('BoundSqr', 'BoundPow', 'BoundReciprocal', 'BoundSqrt'):
        k = {'BoundSqr': 2., 'BoundReciprocal': -1., 'BoundSqrt': .5}.get(name, exponent)
        if k is None or not math.isfinite(k):
            raise NotImplementedError('nonconstant exponent')
        if k != int(k):
            valid &= a[0] > 0  # Also ensures finite second derivatives for sqrt.
        elif k < 0:
            valid &= (a[0] > 0) | (a[1] < 0)
        safe = tuple(torch.where(valid, t, 1.) for t in a)
        value = power(safe, k)
        if k != int(k):
            # Avoid rounding a fractional derivative exponent k-1 or k-2.
            # Subtraction error in that exponent can be amplified by log(x).
            first = scale(mul(value, power(safe, -1)), k)
            km1 = torch.full_like(a[0], k - 1.)
            coefficient = outward(km1, km1, torch.full_like(km1, abs(k) + 1.))
            second = scale(mul(coefficient, mul(value, power(safe, -2))), k)
        else:
            first = scale(power(safe, k - 1), k) if k else scale(one, 0.)
            second = scale(scale(power(safe, k - 2), k - 1), k) if k not in (0, 1) else scale(one, 0.)
        return value, first, second, valid
    if name == 'BoundExp':
        value = outward(a[0].exp(), a[1].exp())
        return value, value, value, valid
    if name == 'BoundLog':
        valid &= a[0] > 0
        safe = tuple(torch.where(valid, t, 1.) for t in a)
        return outward(safe[0].log(), safe[1].log()), power(safe, -1), neg(power(safe, -2)), valid
    if name in ('BoundSin', 'BoundCos'):
        sine, cosine = trig(a), trig(a, True)
        return (sine, cosine, neg(sine), valid) if name == 'BoundSin' else (cosine, neg(sine), neg(cosine), valid)
    if name == 'BoundTanh':
        value = outward(a[0].tanh(), a[1].tanh())
        first = add(one, neg(power(value, 2)))
        return value, first, scale(mul(value, first), -2.), valid
    raise NotImplementedError(name)


@dataclass
class Jet:
    value: tuple
    gradient: tuple
    hessian: tuple
    valid: torch.Tensor

    def select(self, indices):
        return Jet(*(None if pair is None else tuple(t.index_select(1, indices) for t in pair)
                     for pair in (self.value, self.gradient, self.hessian)),
                   self.valid.index_select(1, indices))

    def finish(self):
        for pair in (self.value, self.gradient, self.hessian):
            if pair is None:
                continue
            for t in pair:
                if t.numel() == 0 or self.valid.numel() == 0:
                    # an empty element set (e.g. no output selected, jit1) has nothing to
                    # validate, and reshape(..., -1) of 0 elements is ambiguous
                    continue
                self.valid = self.valid & torch.isfinite(t).reshape(*self.valid.shape, -1).all(-1)
        # Invalid elements never participate numerically in other outputs.
        pairs = [None if pair is None else tuple(torch.where(self.valid.reshape(*self.valid.shape, *([1] * (t.ndim - 2))), t, 0.)
                       for t in pair) for pair in (self.value, self.gradient, self.hessian)]
        return Jet(*pairs, self.valid)


def jet_add(a, b):
    return Jet(*(None if x is None else add(x, y) for x, y in zip(
        (a.value, a.gradient, a.hessian), (b.value, b.gradient, b.hessian))), a.valid & b.valid)


def jet_mul(a, b):
    av = tuple(t[..., None] for t in a.value)
    bv = tuple(t[..., None] for t in b.value)
    value, gradient = mul(a.value, b.value), add(mul(av, b.gradient), mul(bv, a.gradient))
    if a.hessian is None:
        return Jet(value, gradient, None, a.valid & b.valid)
    ah = tuple(t[..., None, None] for t in a.value)
    bh = tuple(t[..., None, None] for t in b.value)
    cross = mul(tuple(t[..., :, None] for t in a.gradient), tuple(t[..., None, :] for t in b.gradient))
    return Jet(value, gradient,
               add(add(mul(ah, b.hessian), mul(bh, a.hessian)),
                   add(cross, tuple(t.transpose(-1, -2) for t in cross))), a.valid & b.valid)


def jet_unary(a, name, exponent=None):
    v, d, dd, valid = unary(a.value, name, exponent)
    if a.hessian is None:
        # C1 at zero is enough for a convex tangent, even if phi'' is infinite.
        k = 2. if name == 'BoundSqr' else exponent if name == 'BoundPow' else None
        if k is not None and k > 1 and k != int(k):
            zero = (a.value[0] == 0) & (a.value[1] == 0)
            valid |= zero
            v = tuple(torch.where(zero, 0., t) for t in v)
            d = tuple(torch.where(zero, 0., t) for t in d)
        return Jet(v, mul(tuple(t[..., None] for t in d), a.gradient), None, a.valid & valid)
    outer = mul(tuple(t[..., :, None] for t in a.gradient), tuple(t[..., None, :] for t in a.gradient))
    return Jet(v, mul(tuple(t[..., None] for t in d), a.gradient),
               add(mul(tuple(t[..., None, None] for t in d), a.hessian),
                   mul(tuple(t[..., None, None] for t in dd), outer)), a.valid & valid)


class IntervalHessian:
    """Compile a bound graph once; evaluate jets for (B,n) boxes.

    first_order omits Hessian storage for convex tangent rounding certificates.
    obj_index may also be a sequence to select several signed-row outputs.
    """

    def __init__(self, bounded_module, x, obj_index, *, first_order=False, max_affine=2_000_000):
        self.n = x.numel()
        self.max_affine = max_affine   # largest dense matrix (outputs x inputs) compiled for an affine node
        self.first_order = first_order
        self.obj_index = obj_index
        self.order = []
        self.shapes = {}
        self.constants = {}
        self.rules = {}
        seen = set()
        final = bounded_module[bounded_module.final_name]
        self.final_name = final.name
        def visit(node):
            if node.name in seen:
                return
            seen.add(node.name)
            for parent in node.inputs:
                visit(parent)
            self.order.append(node)
        visit(final)
        for node in self.order:
            value = _const(node)
            self.shapes[node.name] = tuple(value.shape)
            if _is_const(node):
                self.constants[node.name] = value.clone()
            else:
                # Some Bound.forward methods update shape/index attributes. Restore
                # those even on failure, and never run the live BoundedModule here.
                state = node.__dict__.copy()
                try:
                    self.rules[node.name] = self._compile(node)
                except Exception:  # noqa: BLE001 - unknown shapes/operators give no certificate
                    self.rules[node.name] = ('unsupported',)
                finally:
                    node.__dict__.clear()
                    node.__dict__.update(state)

    def _compile(self, node):
        name, ins = type(node).__name__, node.inputs
        shape = self.shapes[node.name]
        if name == 'BoundInput':
            if math.prod(shape) != self.n:
                raise NotImplementedError('multiple perturbed inputs')
            return ('input',)
        if name in _DATA_MOVEMENT or name == 'BoundGatherElements':
            if name == 'BoundCast' and node.type != ins[0].forward_value.dtype:
                raise NotImplementedError('nonidentity floating point cast')
            slots = list(range(len(ins))) if name == 'BoundConcat' else [0]
            if not all(_is_const(parent) for i, parent in enumerate(ins) if i not in slots):
                raise NotImplementedError('perturbed movement controls')
            offset, args = 0, []
            for i, parent in enumerate(ins):
                if i in slots:
                    size = math.prod(self.shapes[parent.name])
                    args.append(torch.arange(offset, offset + size).reshape(self.shapes[parent.name]))
                    offset += size
                else:
                    args.append(_const(parent).clone())
            indices = torch.as_tensor(node.forward(*args)).long().reshape(-1)
            return ('movement', slots, indices)
        if name in ('BoundAdd', 'BoundSub', 'BoundMul', 'BoundDiv'):
            if (name == 'BoundDiv' and type(ins[1]).__name__ == 'BoundSqrt'
                    and ins[0].inputs):
                # BoundDiv.forward has a special normalization path in this case.
                raise NotImplementedError('normalization division')
            indices = [torch.arange(math.prod(self.shapes[p.name])).reshape(self.shapes[p.name])
                       .expand(shape).reshape(-1) for p in ins]
            return ('binary', name, indices)
        if name in ('BoundNeg', 'BoundSqr', 'BoundPow', 'BoundReciprocal', 'BoundExp',
                    'BoundLog', 'BoundSqrt', 'BoundSin', 'BoundCos', 'BoundTanh'):
            exponent = None
            if name == 'BoundPow':
                if not _is_const(ins[1]) or _const(ins[1]).numel() != 1:
                    raise NotImplementedError('nonconstant exponent')
                exponent = float(_const(ins[1]))
                if not math.isfinite(exponent):
                    raise NotImplementedError('nonfinite exponent')
            return ('unary', name, exponent)
        if name in ('BoundLinear', 'BoundMatMul', 'BoundReduceSum', 'BoundReduceMean', 'BoundATenIndex_add'):
            if name == 'BoundLinear':
                if not all(_is_const(p) for p in ins[1:]):
                    raise NotImplementedError('perturbed linear weights')
                slots = [0]
            elif name == 'BoundMatMul':
                slots = [i for i, p in enumerate(ins) if not _is_const(p)]
                if len(slots) != 1:
                    raise NotImplementedError('perturbed matmul factors')
            elif name == 'BoundATenIndex_add':
                slots = [0, 3]
                if not all(_is_const(ins[i]) for i in (1, 2, 4)):
                    raise NotImplementedError('perturbed index_add controls')
            else:
                slots = [0]
                if not all(_is_const(p) for p in ins[1:]):
                    raise NotImplementedError('perturbed reduction axes')
            args = [_const(p).double() if i in slots or _const(p).is_floating_point() else _const(p)
                    for i, p in enumerate(ins)]
            for i in slots:
                args[i] = torch.zeros(self.shapes[ins[i].name], dtype=torch.float64)
            bias = torch.as_tensor(node.forward(*args)).double().reshape(-1).clone()
            if name == 'BoundLinear' and len(ins) > 2:
                args[2] = torch.zeros_like(args[2])
            matrices = []
            for i in slots:
                size = args[i].numel()
                if size * math.prod(shape) > self.max_affine:
                    raise NotImplementedError('affine map too large')
                columns = []
                for j in range(size):
                    args[i].reshape(-1)[j] = 1.
                    columns.append(torch.as_tensor(node.forward(*args)).reshape(-1).clone())
                    args[i].reshape(-1)[j] = 0.
                matrices.append(torch.stack(columns, -1))
            return ('linear', slots, matrices, bias)
        raise NotImplementedError(name)

    def _zero(self, batch, size, device, valid=True, value=None):
        v = torch.zeros(batch, size, dtype=torch.float64, device=device) if value is None else value
        zero = v.new_zeros(())
        g = zero.expand(batch, size, self.n)
        h = None if self.first_order else (zero.expand(batch, size, self.n, self.n),) * 2
        return Jet((v, v), (g, g), h, torch.full((batch, size), valid, device=device, dtype=torch.bool))

    def _linear(self, jets, slots, matrices, bias, batch, device):
        out = self._zero(batch, bias.numel(), device)
        for slot, weights in zip(slots, matrices):
            w = weights.to(device)
            positive, negative = w.clamp(min=0), w.clamp(max=0)
            wlo, whi = outward(w, w, terms=2)
            coefficient_error = torch.maximum(w - wlo, whi - w)
            a = jets[slot]
            pairs = []
            for pair in (a.value, a.gradient, a.hessian):
                if pair is None:
                    pairs.append(None)
                    continue
                lower, upper = pair
                tail = lower.shape[2:]
                lo, hi = lower.reshape(batch, w.shape[1], -1), upper.reshape(batch, w.shape[1], -1)
                max_abs = torch.maximum(lo.abs(), hi.abs())
                magnitude = w.abs() @ max_abs
                # Compiled coefficients can include rounded alpha scaling, or
                # underflow on the one-hot probe. Enclose that error explicitly.
                uncertainty = coefficient_error @ max_abs
                uncertainty = outward(uncertainty, uncertainty, uncertainty, w.shape[1])[1]
                pair = outward(positive @ lo + negative @ hi - uncertainty,
                               positive @ hi + negative @ lo + uncertainty,
                               magnitude + uncertainty, w.shape[1] + 2)
                pairs.append(tuple(t.reshape(batch, w.shape[0], *tail) for t in pair))
            valid = ((~a.valid).double() @ ((w != 0) | (coefficient_error > 0)).double().T) == 0
            out = jet_add(out, Jet(*pairs, valid))
        bias = bias.to(device).expand(batch, -1)
        b = self._zero(batch, bias.shape[1], device, value=bias)
        b.value = outward(bias, bias)
        return jet_add(out, b)

    @torch.no_grad()
    def evaluate(self, lower, upper):
        """Return a scalar objective Jet; invalid boxes have valid=False."""
        lower, upper = lower.double(), upper.double()
        batch, n = lower.shape
        if upper.shape != lower.shape or n != self.n:
            raise ValueError('Expected matching (B,n) boxes')
        values = {}
        for node in self.order:
            size = math.prod(self.shapes[node.name])
            if node.name in self.constants:
                v = self.constants[node.name].to(lower).reshape(1, -1).expand(batch, -1)
                out = self._zero(batch, size, lower.device, value=v)
                out.valid = torch.isfinite(v)
                out.value = tuple(torch.where(out.valid, t, 0.) for t in out.value)
                values[node.name] = out
                continue
            else:
                rule = self.rules[node.name]
                args = [values[p.name] for p in node.inputs]
                if rule[0] == 'input':
                    out = self._zero(batch, n, lower.device)
                    out.value = lower, upper
                    eye = torch.eye(n, device=lower.device, dtype=lower.dtype).expand(batch, -1, -1)
                    out.gradient = eye, eye
                    out.valid = (lower <= upper) & torch.isfinite(lower) & torch.isfinite(upper)
                elif rule[0] == 'movement':
                    slots, indices = rule[1:]
                    pairs = [None if attr == 'hessian' and self.first_order else
                             tuple(torch.cat([getattr(args[i], attr)[k] for i in slots], 1) for k in (0, 1))
                             for attr in ('value', 'gradient', 'hessian')]
                    out = Jet(*pairs, torch.cat([args[i].valid for i in slots], 1)).select(indices.to(lower.device))
                elif rule[0] == 'binary':
                    name, indices = rule[1:]
                    a, b = [v.select(idx.to(lower.device)) for v, idx in zip(args, indices)]
                    if name == 'BoundSub':
                        b = Jet(*(None if p is None else neg(p) for p in (b.value, b.gradient, b.hessian)), b.valid)
                    if name == 'BoundDiv':
                        b = jet_unary(b, 'BoundReciprocal').finish()
                    out = jet_add(a, b) if name in ('BoundAdd', 'BoundSub') else jet_mul(a, b)
                elif rule[0] == 'unary':
                    name, exponent = rule[1:]
                    a = args[0]
                    out = (Jet(*(None if p is None else neg(p) for p in (a.value, a.gradient, a.hessian)), a.valid)
                           if name == 'BoundNeg' else jet_unary(a, name, exponent))
                elif rule[0] == 'linear':
                    out = self._linear(args, *rule[1:], batch, lower.device)
                else:
                    out = self._zero(batch, size, lower.device, valid=False)
            values[node.name] = out.finish()
        return values[self.final_name].select(torch.as_tensor(self.obj_index, device=lower.device).reshape(-1))

    def lower_eigenvalue(self, lower, upper, fixed_mask):
        jet = self.evaluate(lower, upper)
        free = ~fixed_mask.to(lower.device)
        lo, hi = [t[:, 0][:, free][:, :, free] for t in jet.hessian]
        valid = jet.valid[:, 0]
        if lo.shape[-1] == 0:
            return torch.where(valid, 0., -torch.inf)
        diagonal = lo.diagonal(dim1=-2, dim2=-1)
        off = torch.maximum(lo.abs(), hi.abs())
        off.diagonal(dim1=-2, dim2=-1).zero_()
        radius = outward(off.sum(-1), off.sum(-1), off.sum(-1), off.shape[-1])[1]
        ordinary = add((diagonal, diagonal), neg((radius, radius)))[0].amin(-1)
        # Any positive scaling gives a proven bound. An approximate eigenvector
        # only chooses the scaling; it is never used as an eigenvalue certificate.
        comparison = torch.diag_embed(diagonal) - off
        try:
            _, vectors = torch.linalg.eigh(comparison)
            weights = vectors[..., 0].abs().clamp(min=1e-12)
            ratio = mul((weights[:, None, :], weights[:, None, :]),
                        power((weights[:, :, None], weights[:, :, None]), -1))
            scaled_off = mul((off, off), ratio)[1]
            radius_scaled = outward(scaled_off.sum(-1), scaled_off.sum(-1),
                                     scaled_off.abs().sum(-1), off.shape[-1])[1]
            scaled = add((diagonal, diagonal), neg((radius_scaled, radius_scaled)))[0].amin(-1)
            bound = torch.maximum(ordinary, scaled)
        except RuntimeError:
            bound = ordinary
        return torch.where(valid & torch.isfinite(bound), bound, -torch.inf)
