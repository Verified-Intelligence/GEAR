"""Structural convex-of-quadratic objective epigraphs for the lifted LP.

The polynomial walker uses exact rational coefficients. Each nonlinear atom is a
scalar unary chain in one proved quadratic, with a finite root domain. Positive
power chains are reduced only for their curvature proof (exact rational exponent
products); interval evaluation retains the original chain, including (c/q)**p.
Unknown operators, mixed nonlinear products and unproved domains are declined.
"""
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np
import torch
from mpmath.ctx_iv import MPIntervalContext

from .quadratic_tangent import _const, _is_const
from .rational_rows import RationalForm, _symbolic_node, rounded_coefficients


def _quadratic(poly):
    return poly.terms is not None and all(
        all(k >= 0 for _, k in key) and sum(k for _, k in key) <= 2
        for key in poly.terms)


def polynomial_range(poly, lower, upper):
    """Exact interval arithmetic on rational coefficients and float box endpoints."""
    lo = hi = Fraction(0)
    for key, coefficient in poly.terms.items():
        a = b = coefficient
        for i, k in key:
            l, u = Fraction(float(lower[i])), Fraction(float(upper[i]))
            vl, vu = min(l**k, u**k), max(l**k, u**k)
            if k % 2 == 0 and l <= 0 <= u:
                vl = Fraction(0)
            candidates = a * vl, a * vu, b * vl, b * vu
            a, b = min(candidates), max(candidates)
        lo, hi = lo + a, hi + b
    return np.nextafter(float(lo), -np.inf), np.nextafter(float(hi), np.inf)


@dataclass(frozen=True)
class Atom:
    polynomial: tuple
    chain: tuple

    @property
    def poly(self):
        return RationalForm(dict(self.polynomial))

    @lru_cache(maxsize=4096)
    def intervals(self, lower, upper):
        """Directed interval enclosures of phi, phi', phi'' (CPU scalar arithmetic)."""
        ctx = MPIntervalContext()
        ctx.dps = 30
        value, first, second = ctx.mpf([lower, upper]), ctx.mpf(1), ctx.mpf(0)
        for name, argument in self.chain:
            k = ctx.mpf(argument.numerator) / ctx.mpf(argument.denominator)
            if name == 'scale':
                value, first, second = k * value, k * first, k * second
                continue
            if name == 'pow':
                # Positive inputs cover real powers and reciprocal chains. Integer
                # positive powers also work on the whole real line.
                if argument.denominator != 1 or argument < 0:
                    if not float(value.a) > 0:
                        raise ValueError('unproved positive power domain')
                result = value ** k
                if argument == 0:
                    d = dd = ctx.mpf(0)
                elif argument == 1:
                    d, dd = ctx.mpf(1), ctx.mpf(0)
                elif float(value.a) > 0:
                    d, dd = k * result / value, k * (k - 1) * result / value**2
                else:
                    d = k * value ** (int(argument) - 1)
                    dd = k * (k - 1) * value ** (int(argument) - 2)
            elif name == 'exp':
                result = d = dd = ctx.exp(value)
            elif name == 'log':
                if not float(value.a) > 0:
                    raise ValueError('unproved logarithm domain')
                result, d, dd = ctx.ln(value), 1 / value, -1 / value**2
            else:
                raise ValueError(name)
            value, first, second = result, d * first, dd * first**2 + d * second
        def endpoints(v):
            a, b = float(v.a), float(v.b)
            if not (math.isfinite(a) and math.isfinite(b)):
                raise ValueError('nonfinite unary chain')
            return np.nextafter(a, -np.inf), np.nextafter(b, np.inf)
        return endpoints(value), endpoints(first), endpoints(second)

    def convex(self, lower, upper):
        _, _, second = self.intervals(lower, upper)  # Also proves all domains.
        if second[0] >= 0:
            return True
        # Interval AD alone loses the sign in (c/z)**p through dependency.
        # On z>0, positive scales and powers give C*z**r with C>0.
        exponent, sign = Fraction(1), 1
        for name, argument in self.chain:
            if name == 'scale':
                if argument <= 0:
                    sign *= -1 if argument < 0 else 0
            elif name == 'pow' and sign > 0:
                exponent *= argument
            else:
                return False
        return lower > 0 and sign > 0 and (exponent <= 0 or exponent >= 1)


class ObjectiveForm:
    """A quadratic polynomial plus a signed linear combination of unary atoms."""
    def __init__(self, poly=None, atoms=None):
        self.poly = RationalForm() if poly is None else poly
        self.atoms = {} if atoms is None else atoms

    @staticmethod
    def constant(value):
        return ObjectiveForm(RationalForm.constant(value))

    def constant_value(self):
        if not self.atoms and self.poly.terms is not None and not any(self.poly.terms):
            return self.poly.terms.get((), Fraction(0))
        return None

    def __add__(self, other):
        if not isinstance(other, ObjectiveForm):
            other = self.constant(other)
        poly = self.poly + other.poly
        if poly.terms is None:
            return ObjectiveForm()
        atoms = self.atoms.copy()
        for atom, weight in other.atoms.items():
            atoms[atom] = atoms.get(atom, Fraction(0)) + weight
            if not atoms[atom]:
                del atoms[atom]
        return ObjectiveForm(poly, atoms)

    __radd__ = __add__

    def __neg__(self):
        return ObjectiveForm(-self.poly, {a: -c for a, c in self.atoms.items()})

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, ObjectiveForm):
            other = self.constant(other)
        for a, b in ((self, other), (other, self)):
            c = a.constant_value()
            if c is not None:
                return ObjectiveForm(b.poly * RationalForm({(): c}),
                                     {a: w * c for a, w in b.atoms.items() if w * c})
        if not self.atoms and not other.atoms:
            poly = self.poly * other.poly
            return ObjectiveForm(poly) if _quadratic(poly) else ObjectiveForm()
        return ObjectiveForm()

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, ObjectiveForm):
            other = self.constant(other)
        c = other.constant_value()
        if c is not None:
            return self * ObjectiveForm(RationalForm({(): 1 / c})) if c else ObjectiveForm()
        return self * other.unary('pow', Fraction(-1))

    def power(self, exponent):
        exponent = Fraction(float(exponent))
        if not self.atoms and exponent.denominator == 1 and 0 <= exponent <= 2:
            poly = self.poly.power(int(exponent))
            if _quadratic(poly):
                return ObjectiveForm(poly)
        return self.unary('pow', exponent)

    def unary(self, name, argument=Fraction(0)):
        if _quadratic(self.poly) and not self.atoms:
            atom = Atom(tuple(sorted(self.poly.terms.items())), ((name, argument),))
        elif self.poly.terms == {} and len(self.atoms) == 1:
            a, c = next(iter(self.atoms.items()))
            atom = Atom(a.polynomial, a.chain + (('scale', c), (name, argument)))
        else:
            return ObjectiveForm()
        return ObjectiveForm(RationalForm.constant(0), {atom: Fraction(1)})


@torch.no_grad()
def recognize_convex_objective(bounded_module, x, C, lower, upper):
    """Return the exact spec forms and convex objective atoms, or None.

    The shared polynomial walk proves degree <=2; convexity and domains are proved
    in the scalar atoms, independently of curvature in the original variables.
    """
    bounded_module(x)
    forms = {}
    final = bounded_module[bounded_module.final_name]
    def visit(node):
        if node.name in forms:
            return
        for parent in node.inputs:
            visit(parent)
        state = node.__dict__.copy()
        try:
            name = type(node).__name__
            if name == 'BoundInput':
                if node.name != bounded_module.input_name[0] or node.forward_value.numel() != x.numel():
                    raise ValueError('multiple inputs')
                out = np.array([ObjectiveForm(RationalForm({((i, 1),): Fraction(1)}))
                                for i in range(x.numel())], dtype=object).reshape(x.shape)
            elif _is_const(node):
                out = np.frompyfunc(ObjectiveForm.constant, 1, 1)(_const(node).numpy())
            elif name == 'BoundSqr':
                out = np.frompyfunc(lambda a: a.power(2), 1, 1)(forms[node.inputs[0].name])
            elif (name == 'BoundDiv' and type(node.inputs[1]).__name__ == 'BoundSqrt'
                  and node.inputs[0].inputs):
                raise ValueError('normalization division')
            elif name in ('BoundExp', 'BoundLog', 'BoundSqrt'):
                op = {'BoundExp': 'exp', 'BoundLog': 'log', 'BoundSqrt': 'pow'}[name]
                out = np.frompyfunc(lambda a: a.unary(op, Fraction(1, 2)), 1, 1)(forms[node.inputs[0].name])
            elif name == 'BoundReduceMean':
                axis = node.axis
                axis = tuple(axis) if isinstance(axis, (list, tuple)) else axis
                out = forms[node.inputs[0].name].mean(axis=axis, keepdims=bool(node.keepdim))
            else:
                out = _symbolic_node(node, forms, bounded_module.input_name[0], x.numel())
            forms[node.name] = np.asarray(out, dtype=object).reshape(node.forward_value.shape)
        except (ValueError, TypeError, RuntimeError, IndexError, OverflowError,
                AttributeError, NotImplementedError, ZeroDivisionError):
            forms[node.name] = np.full(node.forward_value.shape, ObjectiveForm(), dtype=object)
        finally:
            node.__dict__.clear()
            node.__dict__.update(state)
    try:
        visit(final)
        outputs = forms[final.name].reshape(-1)
        specs = []
        for coefficients in C[0].detach().cpu().double().numpy():
            row = ObjectiveForm.constant(0)
            for j in np.flatnonzero(coefficients):
                row = row + outputs[j] * float(coefficients[j])
            specs.append(row)
        objective = specs[0]
        if not _quadratic(objective.poly) or not objective.atoms:
            return None
        atoms = []
        lo, hi = lower.reshape(-1).cpu().numpy(), upper.reshape(-1).cpu().numpy()
        for atom, weight in objective.atoms.items():
            if weight < 0:
                atom = Atom(atom.polynomial, atom.chain + (('scale', Fraction(-1)),))
                weight = -weight
            domain = polynomial_range(atom.poly, lo, hi)
            if not atom.convex(*domain):
                return None
            atoms.append((atom, weight, domain))
        return specs, atoms
    except (ValueError, OverflowError, TypeError, ZeroDivisionError):
        return None
    finally:
        bounded_module(x)


@dataclass
class EpigraphTerm:
    atom: Atom
    weight: float
    weight_error: float
    domain: tuple
    coefficients: np.ndarray
    errors: np.ndarray
    constant: float
    constant_error: float
    adaptive_points: list = field(default_factory=list)

    def bounds(self):
        return self.atom.intervals(*self.domain)[0]

    def points(self):
        lo, hi = self.domain
        grid = np.geomspace(lo, hi, 12) if lo > 0 else np.linspace(lo, hi, 12)
        return np.concatenate([grid, self.adaptive_points])

    def remember(self, point):
        """All tangents are valid on the root domain; retain 32 recent adaptive ones."""
        if any(abs(point-p) <= 1e-10 * (1+abs(point)) for p in self.adaptive_points):
            return
        self.adaptive_points.append(point)
        if len(self.adaptive_points) > 32:
            self.adaptive_points.pop(0)

    def cut(self, point, lower, upper, column):
        """s*q_hat-t <= rhs, widened for value, slope and polynomial error.

        Choosing an enclosed slope's midpoint costs delta*max|q-point|. Thus the
        rounded affine cut stays below the exact tangent over the proved domain.
        """
        value, derivative, _ = self.atom.intervals(point, point)
        slope = 0.5 * derivative[0] + 0.5 * derivative[1]
        delta = np.nextafter(max(abs(slope - derivative[0]), abs(slope - derivative[1])), np.inf)
        N = len(lower)
        coef = np.zeros(N)
        coef[:len(self.coefficients)] = slope * self.coefficients
        coef[column] = -1.
        M = np.maximum(abs(lower), abs(upper))
        error = self.constant_error + self.errors @ M[:len(self.errors)]
        rhs = slope * (point - self.constant) - value[0]
        scale = 1 + abs(rhs) + abs(slope * point) + abs(slope * self.constant) + abs(coef) @ M
        rhs += delta * max(abs(self.domain[0] - point), abs(self.domain[1] - point)) + abs(slope) * error
        return coef, np.nextafter(rhs + 1e-12 * scale, np.inf)


def build_convex_lifted_lp(bounded_module, model_ori, x, C, lower, upper,
                          max_pairs, clear_denominators, relaxed_rhs_rows):
    recognized = recognize_convex_objective(bounded_module, x, C, lower, upper)
    if recognized is None:
        return None
    from .lifted_lp import LiftedLP
    from .rational_rows import clear_positive_denominators, extract_rational_rows
    specs, atoms = recognized
    n, dev = x.numel(), x.device
    gen = torch.Generator().manual_seed(0)
    points = lower.cpu().double().reshape(-1) + (upper-lower).cpu().double().reshape(-1) * torch.rand(4, n, generator=gen, dtype=torch.float64)
    selected, cleared = {0: specs[0].poly}, {}
    for r, form in enumerate(specs[1:], 1):
        if not form.atoms and _quadratic(form.poly):
            selected[r] = form.poly
    if clear_denominators and relaxed_rhs_rows is not None and len(selected) < len(specs):
        # Keep the existing monomial-denominator cuts when the objective uses
        # epigraphs. Rational constraint forms need their own shared graph walk.
        rational = extract_rational_rows(bounded_module, model_ori, x, C, points.to(dev))
        for r in range(1, len(specs)):
            if r in selected:
                continue
            poly = clear_positive_denominators(rational[r], float(relaxed_rhs_rows.reshape(-1)[r-1]),
                                               lower.reshape(-1))
            if poly is not None:
                selected[r], cleared[r] = poly, float(relaxed_rhs_rows.reshape(-1)[r-1])
    polynomials = list(selected.values()) + [a.poly for a, _, _ in atoms]
    pairs = set()
    for poly in polynomials:
        for key in poly.terms:
            variables = [i for i, power in key for _ in range(power)]
            if len(variables) == 2:
                pairs.add(tuple(variables))
    if len(pairs) > max_pairs:
        return None
    pairs = sorted(pairs)
    pair_col = {p: n+j for j, p in enumerate(pairs)}
    def coefficients(poly):
        values, errors = np.zeros(n+len(pairs)+1), np.zeros(n+len(pairs)+1)
        for key, (value, error) in rounded_coefficients(poly).items():
            variables = [i for i, power in key for _ in range(power)]
            col = -1 if not variables else variables[0] if len(variables) == 1 else pair_col[tuple(variables)]
            values[col], errors[col] = value, error
        return values, errors
    representations = {r: coefficients(p) for r, p in selected.items()}
    def rep(r, error=False):
        v = representations[r][int(error)]
        return torch.tensor(v[-1], device=dev), torch.tensor(v[:n], device=dev), torch.tensor(v[n:-1], device=dev)
    indices = sorted(r for r in selected if r > 0)
    def rows(error=False):
        return tuple(torch.stack([rep(r, error)[j] for r in indices]) for j in range(3)) if indices else None
    obj, rs = rep(0), rows()
    lp = LiftedLP(n, torch.tensor([a for a, b in pairs], device=dev, dtype=torch.long),
                  torch.tensor([b for a, b in pairs], device=dev, dtype=torch.long), obj,
                  tuple(t.abs() for t in obj), rs, torch.tensor(indices, device=dev, dtype=torch.long),
                  None if rs is None else tuple(t.abs() for t in rs), dev)
    lp.obj_error, lp.row_error = rep(0, True), rows(True)
    lp.cleared_rhs = torch.tensor([cleared.get(r, float('nan')) for r in indices],
                                  device=dev, dtype=torch.float64)
    for atom, weight, domain in atoms:
        v, e = coefficients(atom.poly)
        w = float(weight)
        if not math.isfinite(w):
            return None
        we = np.nextafter(float(abs(Fraction(w) - weight)), np.inf)
        lp.epigraph.append(EpigraphTerm(atom, w, we, domain, v[:-1], e[:-1], v[-1], e[-1]))
    lp.N += len(lp.epigraph)
    # Numerical parity is an additional guard on the graph/model correspondence.
    for point in points:
        value = float(C[0, 0].to(x) @ model_ori(point.reshape(x.shape).to(x)).reshape(-1))
        z = np.concatenate([point.numpy(), point[lp.a.cpu()].numpy() * point[lp.b.cpu()].numpy()])
        estimate = float(obj[0]) + obj[1].cpu().numpy() @ z[:n] + obj[2].cpu().numpy() @ z[n:]
        for term in lp.epigraph:
            q = term.constant + term.coefficients @ z
            interval = term.atom.intervals(q, q)[0]
            estimate += term.weight * (0.5 * interval[0] + 0.5 * interval[1])
        if not math.isfinite(value) or abs(value-estimate) > 1e-9 * (1+abs(value)):
            return None
    print(f'[LiftedLP] convex objective: {len(atoms)} epigraphs, {len(indices)}/{len(specs)-1} lifted rows, {len(pairs)} products')
    return lp
