"""Opt-in presolve identities, proved over the exact encoded float constants.

The proof treats unsupported nodes as independent atoms. No sampled values or
approximate equalities establish an identity. Rational coefficients are emitted
as sums of floats; an unrepresentable coefficient causes refusal.
"""
from fractions import Fraction
import math

import numpy as np
import torch

from .bound_ops import (BoundAdd, BoundATenIndex_add, BoundConcat, BoundConstant,
                        BoundConstantOfShape, BoundGather,
                        BoundInput, BoundLinear, BoundMatMul, BoundMul, BoundReduceSum,
                        BoundPow, BoundReshape, BoundSqr, BoundSub, BoundTranspose)
from .op_budget import OpBudgetExhausted, commit_whole, stage_budget
from .optimize_graph import (_HiddenSquarePolynomial, _HiddenSquareProof, _HiddenSquareUnsupported,
                             _copy_node_properties)
from .bounded_tensor import BoundedTensor
from .perturbations import PerturbationLpNorm


def directed_float(q, like, upward=False):
    """Round a rational to the tensor dtype in the requested direction."""
    v = torch.tensor(float(q), dtype=like.dtype, device=like.device)
    if not torch.isfinite(v):
        raise _HiddenSquareUnsupported
    f = Fraction(float(v))
    if (upward and f < q) or (not upward and f > q):
        v = torch.nextafter(v, v.new_tensor(math.inf if upward else -math.inf))
    return v


def nearest_float(q, like):
    """Round once to the model dtype, avoiding float64-to-float32 double rounding."""
    lower = directed_float(q, like)
    upper = directed_float(q, like, upward=True)
    dl, du = q-Fraction(float(lower)), Fraction(float(upper))-q
    if dl != du:
        return lower if dl < du else upper
    integer_dtype = {torch.float64: torch.int64, torch.float32: torch.int32,
                     torch.float16: torch.int16, torch.bfloat16: torch.int16}[lower.dtype]
    bits = lower.view(integer_dtype)
    return lower if int(bits) % 2 == 0 else upper


def split_exact(q, like):
    """Finite dyadic expansion in the model dtype, or refuse (e.g. underflow)."""
    parts = []
    for _ in range(16):
        if not q:
            return parts
        v = torch.tensor(float(q), dtype=like.dtype, device=like.device)
        if not torch.isfinite(v) or not v:
            break
        parts.append(float(v))
        q -= Fraction(float(v))
    raise _HiddenSquareUnsupported


def outward_sqrt(q, like):
    """Scale before sqrt so a finite radius survives q overflow/underflow."""
    if q < 0:
        raise _HiddenSquareUnsupported
    if not q:
        return like.new_tensor(0.0)
    exponent = q.numerator.bit_length() - q.denominator.bit_length()
    exponent -= exponent % 2
    scaled = q / (Fraction(2) ** exponent)
    radius = like.new_tensor(math.ldexp(math.sqrt(float(scaled)), exponent//2))
    if not torch.isfinite(radius):
        raise _HiddenSquareUnsupported
    while Fraction(float(radius)) ** 2 < q:
        radius = torch.nextafter(radius, radius.new_tensor(math.inf))
    return radius


class _PresolveProof(_HiddenSquareProof):
    def atom(self, node, value):
        # Unsupported constant computations (e.g. exp(buffer)) are still
        # exact shared nodes, not their rounded cached forward values.
        result = np.empty(tuple(value.shape), dtype=object)
        for i in range(value.numel()):
            result.flat[i] = _HiddenSquarePolynomial(
                {((node.name, i),): Fraction(1)}, self.budget)
        return result

    def _build(self, node):
        if type(node) is BoundConstantOfShape:
            return self.constant(self.model.get_forward_value(node))
        if type(node) is BoundTranspose:
            return self.build(node.inputs[0]).transpose(node.perm)
        if type(node) is BoundATenIndex_add:
            if any(node.inputs[i].perturbed for i in (1, 2, 4)):
                return self.atom(node, self.model.get_forward_value(node))
            value = self.model.get_forward_value(node)
            data = np.broadcast_to(self.build(node.inputs[0]), value.shape).copy()
            src = self.build(node.inputs[3])
            shape = list(value.shape)
            shape[node.axis] = src.shape[node.axis]
            src = np.broadcast_to(src, shape)
            indices = self.model.get_forward_value(node.inputs[2]).tolist()
            alpha = self.build(node.inputs[4]).item()
            for i, index in enumerate(indices):
                dest = [slice(None)] * data.ndim
                source = dest.copy()
                dest[node.axis], source[node.axis] = index, i
                data[tuple(dest)] = data[tuple(dest)] + src[tuple(source)] * alpha
            return data
        return super()._build(node)


class _SquareAtomsProof(_PresolveProof):
    def _build(self, node):
        # Keeping squares opaque also avoids expanding the outer square of a
        # high-degree expression. The same square node is the same exact atom.
        if type(node) is BoundSqr or '/presolve_product_residual' in node.name:
            return self.atom(node, self.model.get_forward_value(node))
        return super()._build(node)


class _AffineProof(_PresolveProof):
    def __init__(self, model, budget):
        super().__init__(model, budget)
        self.products = {}

    @staticmethod
    def normalized(p):
        if not p.terms:
            return (), Fraction(0)
        anchor = p.terms[min(p.terms)]
        return tuple(sorted((k, c/anchor) for k, c in p.terms.items())), anchor

    def _build(self, node):
        if type(node) is BoundPow:
            return self.atom(node, self.model.get_forward_value(node))
        if type(node) is BoundSqr or (
                type(node) is BoundMul and all(p.perturbed for p in node.inputs)):
            # Vectorized graphs often repeat one scalar product at different
            # channel indices, possibly with a sign absorbed into a factor.
            # Intern its exact affine operand structures, up to exact scales.
            # Nonlinear products remain opaque; this is not a polynomial guess.
            a = self.build(node.inputs[0])
            b = a if type(node) is BoundSqr else self.build(node.inputs[1])
            a, b = np.broadcast_arrays(a, b)
            result = np.empty(a.shape, dtype=object)
            for i, (pa, pb) in enumerate(zip(a.flat, b.flat)):
                ka, ca = self.normalized(pa)
                kb, cb = self.normalized(pb)
                scale = ca*cb
                terms = {}
                if scale:
                    key = tuple(sorted((ka, kb)))
                    if key not in self.products:
                        self.products[key] = ((node.name, i), scale)
                    atom, original_scale = self.products[key]
                    terms = {(atom,): scale/original_scale}
                result.flat[i] = _HiddenSquarePolynomial(terms, self.budget)
            return result
        return super()._build(node)


class _Emitter:
    def __init__(self, model, ref):
        self.model, self.ref = model, ref
        self.value = model.get_forward_value(ref)
        self.nodes = []
        self.atoms = {}

    def node(self, kind, inputs=(), attr=None, ref=None):
        attr = dict(attr or {}, device=self.value.device)
        n = kind(attr, list(inputs), 0, self.model.bound_opts)
        n.name = f'{self.ref.name}/presolve_{len(self.nodes)}'
        _copy_node_properties(n, ref or self.ref)
        n.perturbed = any(p.perturbed for p in inputs)
        self.nodes.append(n)
        return n

    def constant(self, value):
        if not torch.is_tensor(value):
            value = self.value.new_tensor(value)
        n = self.node(BoundConstant, attr={'value': value})
        n.output_shape = value.shape
        n.batch_dim = -1
        return n

    def add(self, a, b):
        return self.node(BoundAdd, (a, b))

    def scale(self, a, q):
        parts = split_exact(q, self.value)
        if not parts:
            return self.constant(0.0)
        terms = [a if p == 1 else self.node(BoundMul, (a, self.constant(p)))
                 for p in parts]
        result = terms[0]
        for term in terms[1:]:
            result = self.add(result, term)
        return result

    def offset(self, a, q):
        # Add each part to a perturbed expression. Folding a sum of constant
        # nodes into one rounded forward value would discard the low parts.
        for p in split_exact(q, self.value):
            a = self.add(a, self.constant(p))
        return a

    def atom(self, key):
        if key in self.atoms:
            return self.atoms[key]
        name, index = key
        source = self.model[name]
        value = self.model.get_forward_value(source)
        if source.batch_dim != 0 or value.shape[0] != 1:
            raise _HiddenSquareUnsupported
        width = value.numel()
        if len(value.shape) != 2:
            source = self.node(BoundReshape, (source, self.constant(
                torch.tensor([-1, width], device=value.device))))
        n = self.node(BoundGather, (source, self.constant(
            torch.tensor([index], device=value.device))), {'axis': 1})
        n.output_shape = torch.Size((1, 1))
        self.atoms[key] = n
        return n

    def polynomial(self, terms, anchor):
        result = None
        for monomial, q in terms.items():
            if not monomial:
                continue
            term = self.atom(monomial[0])
            for key in monomial[1:]:
                other = self.atom(key)
                term = self.node(BoundSqr, (term,)) if term is other else self.node(
                    BoundMul, (term, other))
            term = self.scale(term, q)
            result = term if result is None else self.add(result, term)
        if result is None:
            # A perturbed zero keeps exact constant expansions from folding.
            result = self.scale(anchor, Fraction(0))
            result = self.node(BoundMul, (anchor, result))
        return self.offset(result, terms.get((), Fraction(0)))

    def vector(self, elements):
        if len(elements) == 1:
            return elements[0]
        n = self.node(BoundConcat, elements, {'axis': 1})
        n.output_shape = torch.Size((1, len(elements)))
        return n

    def finish(self, node, suffix):
        node.name = f'{self.ref.name}/{suffix}'
        _copy_node_properties(node, self.ref)
        self.model.add_nodes(self.nodes)
        return node


def _near_products(model, budget):
    """u*v = c*v^2 + v*(u-c*v), with the residual expanded exactly.

    Matching polynomial support proposes a scale, never proves proportionality.
    Every coefficient of u-c*v is retained, including nonzero rounding residuals.
    """
    proof = _PresolveProof(model, budget)
    count = 0
    for node in list(model.nodes()):
        if type(node) is not BoundMul or not all(p.perturbed for p in node.inputs):
            continue
        value = model.get_forward_value(node)
        if value.ndim != 2 or value.shape[0] != 1 or any(
                model.get_forward_value(p).shape != value.shape for p in node.inputs):
            continue
        try:
            u_node, v_node = node.inputs
            def direct_scale(a, b):
                return type(a) is BoundMul and any(
                    a.inputs[j] is b and not a.inputs[1-j].perturbed for j in (0, 1))
            if direct_scale(v_node, u_node):
                u_node, v_node = v_node, u_node
            u, v = [proof.build(p) for p in (u_node, v_node)]
            plans = []
            for pu, pv in zip(u.flat, v.flat):
                if not pu.terms or pu.terms.keys() != pv.terms.keys():
                    raise _HiddenSquareUnsupported
                key = next(iter(pv.terms))
                c = Fraction(float(nearest_float(pu.terms[key] / pv.terms[key], value)))
                if not c:
                    raise _HiddenSquareUnsupported
                residual = (pu - pv * proof.constant(float(c)).item()).terms
                # A one-term product needs no polynomial expansion: only
                # normalize a directly scaled copy of the same node. This
                # avoids expanding every unrelated scaled product in a model.
                if len(pu.terms) == 1 and (residual or not direct_scale(u_node, v_node)):
                    raise _HiddenSquareUnsupported
                # Restrict expansion to rounding-size residuals. This is only
                # a cost/quality guard: the identity retains the full residual.
                eps = Fraction(float(torch.finfo(value.dtype).eps)) * 64
                if any(abs(r) > eps * max(abs(pu.terms[k]), abs(c * pv.terms[k]))
                       for k, r in residual.items()):
                    raise _HiddenSquareUnsupported
                plans.append((c, residual))
            emitter = _Emitter(model, node)
            parts = []
            for i, (c, residual) in enumerate(plans):
                z = emitter.atom((v_node.name, i))
                sq = emitter.node(BoundSqr, (z,))
                main = emitter.scale(sq, c)
                if residual:
                    rem = emitter.polynomial(residual, z)
                    product = emitter.node(BoundMul, (z, rem))
                    # Give this explicit residual product its own opaque atom
                    # in the completion proof, avoiding a degree-six expansion.
                    product.name = f'{node.name}/presolve_product_residual_{i}'
                    main = emitter.add(main, product)
                parts.append(main)
            result = emitter.finish(emitter.vector(parts), 'presolve_residual_square')
        except TimeoutError:
            budget.log_cap()
            break
        except (_HiddenSquareUnsupported, OverflowError, ValueError, RecursionError):
            continue
        model.replace_node(node, result)
        count += len(plans)
    return count


def rewrite_complete_squares(model):
    """Complete c*z^2+k*z exactly, for input or shared inner nodes z.

    s is rounded to the model dtype, r=k-2*c*s and q=-c*s*s are
    computed as rationals. The full remainder (including r*z) is emitted
    exactly; no inexact coefficient is silently rounded away.
    """
    if not model.bound_opts.get('complete_square_rewrite', False):
        return 0
    model._mark_perturbed_nodes(model.global_input)
    budget = stage_budget(model, 'complete_squares')
    near_count = _near_products(model, budget)
    model._mark_perturbed_nodes(model.global_input)
    proof = _SquareAtomsProof(model, budget)
    count = 0
    # Work from large sums to small ones, and do not complete the same square
    # twice through different ancestors of the same expression.
    used = set()
    for node in reversed(list(model.nodes())):
        if type(node) not in (BoundAdd, BoundSub, BoundATenIndex_add, BoundLinear,
                              BoundMatMul, BoundReduceSum) or '/presolve_' in node.name:
            continue
        value = model.get_forward_value(node)
        if value.ndim != 2 or value.shape[0] != 1:
            continue
        try:
            polys = proof.build(node)
            plans = []
            for i, p in enumerate(polys.flat):
                plan = []
                remainder = dict(p.terms)
                for monomial, c in p.terms.items():
                    c = remainder.get(monomial, Fraction(0))
                    if len(monomial) != 1:
                        continue
                    name, index = monomial[0]
                    square = model[name]
                    if type(square) is not BoundSqr or (name, index) in used:
                        continue
                    zpoly = proof.build(square.inputs[0]).flat[index]
                    keys = [k for k in zpoly.terms if k]
                    if not keys or any(k not in remainder for k in keys):
                        continue
                    k = remainder[keys[0]] / zpoly.terms[keys[0]]
                    if not k or not c:
                        continue
                    s = Fraction(float(nearest_float(k / (2*c), value)))
                    if not s:
                        continue
                    del remainder[monomial]
                    for key, coeff in zpoly.terms.items():
                        remainder[key] = remainder.get(key, Fraction(0)) - 2*c*s*coeff
                        if not remainder[key]:
                            del remainder[key]
                    # c*(z+s)^2 contributes c*s^2 as well as 2*c*s*z.
                    remainder[()] = remainder.get((), Fraction(0)) - c*s*s
                    plan.append((square, index, c, s))
                plans.append((plan, remainder))
            if not any(p[0] for p in plans):
                continue
            emitter = _Emitter(model, node)
            parts = []
            original = None
            for i, (plan, remainder) in enumerate(plans):
                if not plan:
                    # Clone the original linear node for untouched elements;
                    # referring to node itself would create a replacement cycle.
                    if original is None:
                        original = emitter.node(type(node), node.inputs, node.attr)
                    selected = emitter.node(BoundGather, (original, emitter.constant(
                        torch.tensor([i], device=value.device))), {'axis': 1})
                    parts.append(selected)
                    continue
                main = None
                for square, index, c, s in plan:
                    z = emitter.atom((square.inputs[0].name, index))
                    shifted = emitter.offset(z, s)
                    term = emitter.scale(emitter.node(BoundSqr, (shifted,)), c)
                    main = term if main is None else emitter.add(main, term)
                parts.append(emitter.add(main, emitter.polynomial(remainder, z)))
            result = emitter.finish(emitter.vector(parts), 'completed_square')
        except TimeoutError:
            budget.log_cap()
            break
        except (_HiddenSquareUnsupported, OverflowError, ValueError, RecursionError):
            continue
        model.replace_node(node, result)
        used.update((p[0].name, p[1]) for plan, _ in plans for p in plan)
        count += sum(len(plan) for plan, _ in plans)
        # Replacements invalidate cached ancestor proofs.
        proof = _SquareAtomsProof(model, budget)
    budget.finish()
    if budget.logged and commit_whole(model):
        raise OpBudgetExhausted('complete_squares')
    model.presolve_rewrite_counts = {'complete_square_rewrite': count,
                                    'product_residual': near_count}
    print(f'[Presolve] complete_square_rewrite={count} product_residual={near_count}')
    return count


def _row_polynomial(proof, outputs, coefficients):
    result = proof.constant(0.0).item()
    for p, c in zip(outputs.flat, coefficients.tolist()):
        if c:
            result = result + p * proof.constant(c).item()
    return result


def prove_rows(model, C):
    """Prove signed row relations by exact affine graph structure (cached)."""
    cache = getattr(model, '_presolve_row_cache', None)
    if cache is None:
        model._presolve_row_cache = cache = {}
    signature = (C.dtype, tuple(C.shape), C.detach().cpu().numpy().tobytes())
    if signature in cache:
        return cache[signature]
    budget = stage_budget(model, 'prove_rows')
    proof = _AffineProof(model, budget)
    squares, aliases = [], []
    try:
        outputs = proof.build(model.final_node())
        rows = []
        for row in C.detach().cpu():
            budget.spend()
            rows.append(_row_polynomial(proof, outputs, row).terms)
        obj = rows[0]
        objkeys = [k for k in obj if k]
        for i, row in enumerate(rows[1:], 1):
            keys = [k for k in row if k]
            if len(keys) == 1 and len(keys[0]) == 1:
                name, index = keys[0][0]
                n = model[name]
                if type(n) is BoundSqr and row[keys[0]] > 0:
                    h = proof.build(n.inputs[0]).flat[index]
                    affine = all(not k or (len(k) == 1 and
                                 type(model[k[0][0]]) is BoundInput) for k in h.terms)
                    squares.append((i, n.inputs[0].name, index,
                                    row[keys[0]], row.get((), Fraction(0)), affine))
            if objkeys and set(keys) == set(objkeys):
                alpha = row[objkeys[0]] / obj[objkeys[0]]
                if alpha and all(row[k] == alpha * obj[k] for k in objkeys):
                    beta = row.get((), Fraction(0)) - alpha * obj.get((), Fraction(0))
                    aliases.append((i, alpha, beta))
    except TimeoutError:
        budget.log_cap()
        if commit_whole(model):
            squares, aliases = [], []
    except (_HiddenSquareUnsupported, OverflowError, ValueError, RecursionError):
        pass
    budget.finish()
    cache[signature] = squares, aliases
    return squares, aliases


def objective_alias_floor(model, C, rhs, tol):
    if not model.bound_opts.get('objective_row_alias', False):
        return None
    floors = []
    count = 0
    if C.shape[0] == 1:
        C = C.expand(rhs.shape[0], -1, -1)
    for c, t in zip(C, rhs):
        _, aliases = prove_rows(model, c)
        bounds = []
        for i, alpha, beta in aliases:
            if alpha < 0 and torch.isfinite(t[i]):
                q = (Fraction(float(t[i])) + Fraction(float(tol)) - beta) / alpha
                try:
                    bounds.append(directed_float(q, t))
                except (_HiddenSquareUnsupported, OverflowError):
                    continue
        floors.append(torch.stack(bounds).max() if bounds else t.new_tensor(-math.inf))
        count = max(count, len(bounds))
    model.presolve_rewrite_counts = dict(getattr(model, 'presolve_rewrite_counts', {}),
                                        objective_row_alias=count)
    return torch.stack(floors)


def square_row_cuts(model, C, rhs, x, tol, nonlinear_only=False):
    """Additional A*x <= b planes implied by the ORIGINAL relaxed square rows.

    CROWN supplies lower planes for h and -h, so this also handles non-affine h.
    Returned b already includes tolerance; callers must not add it a second time.
    Original model outputs/specs remain the sole primal feasibility test.
    """
    if not model.bound_opts.get('square_row_to_abs', False):
        return None
    # The solver normally repeats one specification across all domains. Refuse
    # mixed specs here instead of accidentally applying one branch's cuts to another.
    if not torch.equal(C, C[:1].expand_as(C)) or not torch.equal(
            rhs[:, 1:], rhs[:1, 1:].expand_as(rhs[:, 1:])):
        return None
    squares, _ = prove_rows(model, C[0])
    # input_name also includes traced buffers/parameters. Only the actual
    # model input is a variable in the solver's A*x <= b rows.
    inputs = [name for name in model.input_name if type(model[name]) is BoundInput]
    if len(inputs) != 1 or any(n.perturbed and type(n) is not BoundInput for n in model.roots()):
        return None
    if x.ptb.x_L is None or x.ptb.x_U is None:
        return None
    # These are global base cuts. Prove them over the full box, independently
    # of any caller's possibly unrelaxed input constraints or previous cuts.
    box = BoundedTensor(x.detach(), PerturbationLpNorm(x_L=x.ptb.x_L, x_U=x.ptb.x_U))
    rows, limits = [], []
    for i, name, index, a, beta, affine in squares:
        if nonlinear_only and affine:
            continue
        if not torch.isfinite(rhs[0, i]):
            continue
        q = (Fraction(float(rhs[0, i])) + Fraction(float(tol)) - beta) / a
        if q < 0:
            continue  # Original row's bounds handle infeasibility.
        try:
            radius = outward_sqrt(q, rhs)
        except (_HiddenSquareUnsupported, OverflowError):
            continue
        node = model[name]
        width = math.prod(node.output_shape[1:])
        spec = rhs.new_zeros((x.shape[0], 2, width))
        spec[:, 0, index], spec[:, 1, index] = 1, -1
        _, _, A = model.compute_bounds(x=(box,), C=spec, method='CROWN',
                                      final_node_name=name, bound_upper=False,
                                      return_A=True,
                                      needed_A_dict={name: inputs})
        plane = A[name][inputs[0]]
        coeff, bias = plane['lA'], plane['lbias']
        if coeff is None or bias is None or not torch.isfinite(coeff).all() or not torch.isfinite(bias).all():
            continue
        coeff = coeff.flatten(2)
        # Subtraction is rounded outward; the plane itself carries CROWN's
        # arithmetic enclosure. No extra tolerance is needed on these rows.
        b = torch.nextafter(radius - bias, torch.full_like(bias, math.inf))
        rows.append(coeff)
        limits.append(b)
    count = sum(r.shape[1] for r in rows)
    if nonlinear_only:
        count = max(count, getattr(model, 'presolve_rewrite_counts', {}).get('square_row_to_abs', 0))
    model.presolve_rewrite_counts = dict(getattr(model, 'presolve_rewrite_counts', {}),
                                        square_row_to_abs=count)
    return (torch.cat(rows, 1), torch.cat(limits, 1)) if rows else None
