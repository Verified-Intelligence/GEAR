"""Laurent-polynomial rows proved from the bound graph, for denominator clearing.

Coefficients are exact rationals of the graph's float constants. Unsupported operators
and division by a sum produce an unknown form, propagated only to dependent elements.
Numerical checks are an additional guard, never the proof of the representation.
"""
from fractions import Fraction
import math

import numpy as np
import torch

from nlp_solver.quadratic_tangent import _const, _is_const, _DATA_MOVEMENT

_MAX_TERMS = 2048


class RationalForm:
    """Sum of coefficient * prod x_i**k_i; keys are sorted (i, k_i) tuples."""

    def __init__(self, terms=None):
        self.terms = terms

    @staticmethod
    def constant(value):
        value = Fraction(float(value))
        return RationalForm({(): value} if value else {})

    def __add__(self, other):
        if not isinstance(other, RationalForm):
            other = self.constant(other)
        if self.terms is None or other.terms is None:
            return RationalForm()
        terms = self.terms.copy()
        for key, value in other.terms.items():
            terms[key] = terms.get(key, Fraction(0)) + value
            if not terms[key]:
                del terms[key]
        return RationalForm(terms) if len(terms) <= _MAX_TERMS else RationalForm()

    __radd__ = __add__

    def __neg__(self):
        return RationalForm(None if self.terms is None else
                            {key: -value for key, value in self.terms.items()})

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, RationalForm):
            other = self.constant(other)
        if self.terms is None or other.terms is None:
            return RationalForm()
        if len(self.terms) * len(other.terms) > _MAX_TERMS:
            return RationalForm()
        terms = {}
        for ka, va in self.terms.items():
            for kb, vb in other.terms.items():
                powers = dict(ka)
                for i, k in kb:
                    powers[i] = powers.get(i, 0) + k
                key = tuple(sorted((i, k) for i, k in powers.items() if k))
                value = terms.get(key, Fraction(0)) + va * vb
                if value:
                    terms[key] = value
                else:
                    terms.pop(key, None)
        return RationalForm(terms)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self * other.power(-1)

    def power(self, exponent):
        if self.terms is None or not math.isfinite(exponent) or int(exponent) != exponent:
            return RationalForm()
        exponent = int(exponent)
        if abs(exponent) > 32:
            return RationalForm()
        if exponent < 0:
            if len(self.terms) != 1:
                return RationalForm()
            key, value = next(iter(self.terms.items()))
            return RationalForm({tuple((i, k * exponent) for i, k in key): value ** exponent})
        out = self.constant(1)
        for _ in range(exponent):
            out = out * self
            if out.terms is None:
                break
        return out

    def evaluate(self, points):
        if self.terms is None:
            return None
        out = torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)
        scale = torch.ones_like(out)
        for key, value in self.terms.items():
            term = torch.full_like(out, float(value))
            for i, k in key:
                term = term * points[:, i] ** k
            out += term
            scale += term.abs()
        return out, scale


def _symbolic_node(node, forms, input_name, n):
    name, ins = type(node).__name__, node.inputs
    shape = tuple(node.forward_value.shape)
    if name == 'BoundInput':
        if node.name != input_name or node.forward_value.numel() != n:
            raise ValueError('multiple symbolic inputs')
        return np.array([RationalForm({((i, 1),): Fraction(1)}) for i in range(n)],
                        dtype=object).reshape(shape)
    if _is_const(node):
        return np.frompyfunc(RationalForm.constant, 1, 1)(_const(node).numpy())
    args = [forms[i.name] for i in ins]
    if name == 'BoundAdd':
        return args[0] + args[1]
    if name == 'BoundSub':
        return args[0] - args[1]
    if name == 'BoundMul':
        return args[0] * args[1]
    if name == 'BoundNeg':
        return -args[0]
    if name == 'BoundDiv':
        return args[0] / args[1]
    if name == 'BoundReciprocal':
        return np.frompyfunc(lambda a: a.power(-1), 1, 1)(args[0])
    if name == 'BoundSqr':
        return args[0] * args[0]
    if name == 'BoundPow' and _is_const(ins[1]):
        return np.frompyfunc(lambda a, k: a.power(float(k)), 2, 1)(args[0],
                                                                  _const(ins[1]).numpy())
    if name == 'BoundMatMul':
        return np.matmul(*args)
    if name == 'BoundLinear' and not getattr(node, 'transA', 0) and all(
            _is_const(i) for i in ins[1:]):
        w = args[1].T if getattr(node, 'transB', 0) else args[1]
        out = (args[0] * float(node.alpha_linear)) @ w
        return out if len(args) < 3 else out + args[2] * float(node.beta_linear)
    if name == 'BoundReduceSum':
        axis = getattr(node, 'axis', None)
        axis = tuple(axis) if isinstance(axis, (list, tuple)) else axis
        return args[0].sum(axis=axis, keepdims=bool(getattr(node, 'keepdim', False)))
    if name == 'BoundATenIndex_add' and all(_is_const(ins[k]) for k in (1, 2, 4)):
        axis = int(_const(ins[1])) % args[0].ndim
        data, src = args[0], args[3]
        target = [max(a, b) for a, b in zip(data.shape, src.shape)]
        data_shape, src_shape = target.copy(), target.copy()
        data_shape[axis], src_shape[axis] = data.shape[axis], src.shape[axis]
        out = np.broadcast_to(data, data_shape).copy()
        src = np.broadcast_to(src, src_shape) * float(_const(ins[4]))
        for j, idx in enumerate(_const(ins[2]).reshape(-1).tolist()):
            dst_sl, src_sl = [slice(None)] * out.ndim, [slice(None)] * out.ndim
            dst_sl[axis], src_sl[axis] = idx, j
            out[tuple(dst_sl)] = out[tuple(dst_sl)] + src[tuple(src_sl)]
        return out
    if name == 'BoundWhere' and _is_const(ins[0]):
        return np.where(_const(ins[0]).numpy(), args[1], args[2])
    if name == 'BoundCast':
        source, target = ins[0].forward_value, node.forward_value
        if source.dtype == target.dtype or (source.is_floating_point() and
                target.is_floating_point() and
                torch.finfo(source.dtype).bits <= torch.finfo(target.dtype).bits):
            return args[0]
        raise ValueError('cast changes symbolic values')
    if name in _DATA_MOVEMENT and name != 'BoundCast':
        if name != 'BoundConcat' and not all(_is_const(i) for i in ins[1:]):
            raise ValueError('symbolic indexing or shape control')
        # Move integer IDs, never symbolic coefficients, through the operator's exact
        # indexing/shape semantics. Arithmetic operators are handled above.
        table = []
        move_args = []
        for i, arg in zip(ins, args):
            # Concat's operands are all data; other operators' constant inputs are indices
            # or shapes (dea, hs055: a constant Concat operand used to fail the whole walk).
            if _is_const(i) and name != 'BoundConcat':
                move_args.append(_const(i))
            else:
                ids = torch.arange(len(table), len(table) + arg.size).reshape(arg.shape)
                table.extend(arg.reshape(-1))
                move_args.append(ids)
        ids = node.forward(*move_args).detach().cpu().numpy()
        return np.array(table, dtype=object)[ids]
    raise ValueError('unsupported symbolic operator: ' + name)


@torch.no_grad()
def extract_rational_rows(bounded_module, model_ori, x_center, C, points):
    """One RationalForm per signed spec row C[r] @ model(x); unknown rows have terms=None."""
    bounded_module(x_center)
    final = bounded_module[bounded_module.final_name]
    forms = {}
    n = x_center.numel()

    def visit(node):
        if node.name in forms:
            return
        for i in node.inputs:
            visit(i)
        try:
            out = _symbolic_node(node, forms, bounded_module.input_name[0], n)
            forms[node.name] = np.asarray(out, dtype=object).reshape(node.forward_value.shape)
        except (ValueError, TypeError, RuntimeError, IndexError, OverflowError,
                AttributeError, NotImplementedError):
            forms[node.name] = np.full(node.forward_value.shape, RationalForm(), dtype=object)

    try:
        visit(final)
        outputs = forms[final.name].reshape(-1)
        rows = []
        for c in C[0].detach().cpu().double().numpy():
            row = RationalForm.constant(0)
            for j in np.flatnonzero(c):
                row = row + outputs[j] * float(c[j])
            rows.append(row)
        # Restore graph caches touched by the data-movement forwards.
        bounded_module(x_center)
        try:
            values = torch.stack([model_ori(p.reshape(x_center.shape).to(x_center)).reshape(-1)
                                  for p in points]).double() @ C[0].to(points).T
        except (ValueError, RuntimeError, OverflowError):
            return [RationalForm() for _ in rows]
        for r, row in enumerate(rows):
            try:
                checked = row.evaluate(points)
                if checked is None:
                    continue
                rep, scale = checked
                if not bool((torch.isfinite(rep) & torch.isfinite(values[:, r]) &
                             ((rep - values[:, r]).abs() <= 1e-9 * scale)).all()):
                    rows[r] = RationalForm()
            except (ValueError, OverflowError, RuntimeError):
                rows[r] = RationalForm()
        return rows
    finally:
        bounded_module(x_center)


def clear_positive_denominators(row, relaxed_rhs, lower):
    """Prove m>0 from root lower endpoints and return m*(row-relaxed_rhs), degree <=2.

    A positive *variable* denominator is required; sums, sign-unknown denominators,
    and cleared polynomials of higher degree are declined.
    """
    if row.terms is None or not math.isfinite(relaxed_rhs):
        return None
    powers = {}
    for key in row.terms:
        for i, k in key:
            if k < 0:
                if not float(lower[i]) > 0:
                    return None
                powers[i] = max(powers.get(i, 0), -k)
    if not powers:
        return None
    denominator = RationalForm({tuple(sorted(powers.items())): Fraction(1)})
    cleared = (row - RationalForm.constant(relaxed_rhs)) * denominator
    if cleared.terms is None or any(sum(k for _, k in key) > 2 or any(k < 0 for _, k in key)
                                    for key in cleared.terms):
        return None
    return cleared


def rounded_coefficients(row):
    """Nearest float coefficients and outward-rounded absolute conversion errors.

    Exact Fraction arithmetic covers every intermediate product/sum of float constants.
    Each returned error encloses |float(coefficient) - exact coefficient|, including
    underflow; the LP adds their box-weighted sum to the row's RHS.
    """
    out = {}
    for key, exact in row.terms.items():
        value = float(exact)
        if not math.isfinite(value):
            raise ValueError('non-finite coefficient')
        delta = abs(Fraction(value) - exact)
        error = np.nextafter(float(delta), np.inf) if delta else 0.0
        if not math.isfinite(error):
            raise ValueError('non-finite coefficient error')
        out[key] = (value, error)
    return out
