#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Optimize the graph to merge nodes and remove unnecessary ones.

Initial and experimental code only.
"""

from auto_LiRPA.bound_ops import *
from auto_LiRPA.utils import logger
from auto_LiRPA.op_budget import OpBudgetExhausted, commit_whole, stage_budget
from fractions import Fraction
import numpy as np
import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule


def _optimize_graph(self: 'BoundedModule'):
    """Optimize the graph to remove some unnecessary nodes."""
    merge_identical_gather(self)
    merge_common_subexpressions(self)
    merge_identical_act(self)
    convert_sqr(self)
    rewrite_hidden_squares(self)
    from .presolve import rewrite_complete_squares
    rewrite_complete_squares(self)
    pow_variable_exponent_to_exp_log(self)
    merge_common_subexpressions(self)
    div_to_mul(self)
    merge_sec(self)
    minmax_to_relu(self)
    optimize_relu_relation(self)

    if self.bound_opts['optimize_graph']['optimizer'] is not None:
        # Use the custom graph optimizer
        self.bound_opts['optimize_graph']['optimizer'](self)

    for node in list(self.nodes()):
        if (not node.output_name
                and node.name != self.final_name
                and node.name not in self.root_names):
            self.delete_node(node)
    share_node_options(self)


# Options that operators read through get_bound_opt and that must also reach the
# nodes the rewrites above create without the model's option dict (BoundReciprocal
# from div_to_mul, BoundSqr from convert_sqr, ...).
SHARED_NODE_OPTIONS = ('undefined_relaxation_rows',)


def share_node_options(model: 'BoundedModule'):
    for key in SHARED_NODE_OPTIONS:
        if key not in model.bound_opts:
            continue
        for node in model.nodes():
            opts = node.bound_opts
            if isinstance(opts, dict) and opts is not model.bound_opts:
                opts[key] = model.bound_opts[key]


def _copy_node_properties(new, ref):
    new.output_shape = ref.output_shape
    new.device = ref.device
    new.attr['device'] = ref.attr['device']
    new.batch_dim = ref.batch_dim
    new.from_complex_node = ref.from_complex_node


def merge_sec(model: 'BoundedModule'):
    nodes = list(model.nodes())
    for node in nodes:
        if type(node) == BoundReciprocal and type(node.inputs[0]) == BoundCos:
            node_new = BoundSec(inputs=[node.inputs[0].inputs[0]])
            node_new.name = f'{node.inputs[0].name}/sec'
            _copy_node_properties(node_new, node)
            if node_new.name in model._modules:
                node_existing = model._modules[node_new.name]
                assert isinstance(node_existing, BoundSec)
                assert node_existing.inputs[0] == node.inputs[0].inputs[0]
                model.replace_node(node, node_existing)
            else:
                model.add_nodes([node_new])
                model.replace_node(node, node_new)


def div_to_mul(model: 'BoundedModule'):
    nodes = list(model.nodes())
    for node in nodes:
        if type(node) == BoundDiv:
            logger.debug('Replacing BoundDiv node: %s', node)
            node_reciprocal = BoundReciprocal(inputs=[node.inputs[1]])
            node_reciprocal.name = f'{node.name}/reciprocal'
            # Properties of the reciprocal node only depend on inputs[1], i.e.
            # the node of denominator. They can be different from those of
            # the original BoundDiv node, due to possible broadcasting and
            # perturbed/unperturbed switching in multiplication.
            _copy_node_properties(node_reciprocal, node.inputs[1])
            model.add_nodes([node_reciprocal])
            node_mul = BoundMul(inputs=[node.inputs[0], node_reciprocal],
                                options=model.bound_opts)
            node_mul.name = f'{node.name}/mul'
            _copy_node_properties(node_mul, node)
            model.add_nodes([node_mul])
            model.replace_node(node, node_mul)


def convert_sqr(model: 'BoundedModule'):
    """Replace BoundMul or Bound Pow with BoundSqr if applicable.

    1. If the two inputs nodes of a BoundMul node are the same, use BoundSqr.
    2. Pow(x, 2) can be replaced with BoundSqr.
    """
    nodes = list(model.nodes())
    for node in nodes:
        replace = False
        if type(node) == BoundMul and node.inputs[0] == node.inputs[1]:
            replace = True
        elif type(node) == BoundPow:
            exponent = node.inputs[1]
            if (isinstance(exponent, (BoundBuffers, BoundConstant, BoundCast))
                    and model.get_forward_value(exponent) == 2):
                replace = True
        if replace:
            node_new = BoundSqr(inputs=[node.inputs[0]])
            node_new.name = f'{node.name}/sqr'
            _copy_node_properties(node_new, node)
            model.add_nodes([node_new])
            logger.debug('Replaceing %s with %s', node, node_new)
            model.replace_node(node, node_new)


class _HiddenSquareUnsupported(ValueError):
    """The exact proof cannot handle this expression within its size limits."""


class _HiddenSquarePolynomial:
    """Sparse rational polynomial; NumPy object arrays supply tensor operations.

    Monomials are sorted tuples of (input or opaque node name, flat element index).
    Arithmetic never uses floating point, including matrix/reduction sums.
    """

    def __init__(self, terms, budget):
        self.terms = terms
        self.budget = budget
        self.check()

    def check(self):
        self.budget.spend()
        if len(self.terms) > 2000 or any(len(k) > 4 for k in self.terms):
            raise _HiddenSquareUnsupported

    def __add__(self, other):
        # Object-array sums/matmul can start their accumulation with integer 0.
        if not isinstance(other, _HiddenSquarePolynomial):
            if other == 0:
                return self
            raise _HiddenSquareUnsupported
        terms = self.terms.copy()
        for key, value in other.terms.items():
            terms[key] = terms.get(key, Fraction(0)) + value
            if not terms[key]:
                del terms[key]
        return _HiddenSquarePolynomial(terms, self.budget)

    __radd__ = __add__

    def __neg__(self):
        return _HiddenSquarePolynomial(
            {k: -v for k, v in self.terms.items()}, self.budget)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        terms = {}
        for ka, va in self.terms.items():
            self.check()
            self.budget.spend(len(other.terms))
            for kb, vb in other.terms.items():
                key = tuple(sorted(ka + kb))
                if len(key) > 4:
                    raise _HiddenSquareUnsupported
                terms[key] = terms.get(key, Fraction(0)) + va * vb
                if not terms[key]:
                    del terms[key]
                if len(terms) > 2000:
                    raise _HiddenSquareUnsupported
        return _HiddenSquarePolynomial(terms, self.budget)

    def __truediv__(self, other):
        if other.terms.keys() != {()} or not other.terms[()]:
            raise _HiddenSquareUnsupported
        return _HiddenSquarePolynomial(
            {k: v / other.terms[()] for k, v in self.terms.items()}, self.budget)

    def __pow__(self, exponent):
        terms = _HiddenSquarePolynomial({(): Fraction(1)}, self.budget)
        if not self.terms:
            return terms if exponent == 0 else self
        if max(map(len, self.terms)) * exponent > 4:
            raise _HiddenSquareUnsupported
        base = self
        while exponent:
            if exponent & 1:
                terms = terms * base
            exponent //= 2
            if exponent:
                base = base * base
        return terms


class _HiddenSquareProof:
    def __init__(self, model, budget):
        self.model = model
        self.budget = budget
        self.cache = {}

    def constant(self, value):
        values = (value if torch.is_tensor(value) else
                  torch.as_tensor(value, dtype=torch.float64)).detach().cpu()
        result = np.empty(tuple(values.shape), dtype=object)
        for i, scalar in enumerate(values.reshape(-1).tolist()):
            # Fraction(float) preserves the exact binary value of constants.
            coeff = Fraction(float(scalar)) if isinstance(scalar, float) else Fraction(scalar)
            result.flat[i] = _HiddenSquarePolynomial(
                {(): coeff} if coeff else {}, self.budget)
        return result

    def atom(self, node, value):
        if not node.perturbed:
            return self.constant(value)
        result = np.empty(tuple(value.shape), dtype=object)
        for i in range(value.numel()):
            result.flat[i] = _HiddenSquarePolynomial(
                {((node.name, i),): Fraction(1)}, self.budget)
        return result

    def build(self, node):
        self.budget.spend()
        if node in self.cache:
            result = self.cache[node]
            if result is None:
                raise _HiddenSquareUnsupported
            return result
        try:
            # NumPy's arithmetic on zero-dimensional object arrays can return
            # a bare polynomial. Keep the cache uniformly shaped, even scalars.
            result = np.asarray(self._build(node), dtype=object)
        except (_HiddenSquareUnsupported, RecursionError):
            self.cache[node] = None
            raise _HiddenSquareUnsupported from None
        self.cache[node] = result
        return result

    def _build(self, node):
        kind = type(node)
        value = self.model.get_forward_value(node)
        if kind is BoundInput:
            return self.atom(node, value)
        if kind is BoundConstant or (
                kind in (BoundBuffers, BoundParams) and not node.perturbed):
            return self.constant(value)

        if kind in (BoundGather, BoundSlice, BoundReshape, BoundUnsqueeze,
                    BoundSqueeze, BoundExpand):
            if any(p.perturbed for p in node.inputs[1:]):
                return self.atom(node, value)
            data = self.build(node.inputs[0])
            # These operators only select/rearrange elements. Running their
            # forward on integer indices gives an exact map, without arithmetic
            # on polynomial coefficients or a rounded autograd transpose.
            indices = torch.arange(data.size, device=value.device).reshape(data.shape)
            args = [self.model.get_forward_value(p) for p in node.inputs[1:]]
            selected = node.forward(indices, *args).detach().cpu().numpy()
            return data.reshape(-1)[selected]
        if kind is BoundConcat:
            return np.concatenate([self.build(p) for p in node.inputs], axis=node.axis)
        if kind is BoundReduceSum:
            if any(p.perturbed for p in node.inputs[1:]):
                return self.atom(node, value)
            axis = node.axis
            if axis is not None and not isinstance(axis, int):
                axis = tuple(axis)
            return np.asarray(self.build(node.inputs[0]).sum(
                axis=axis, keepdims=node.keepdim if axis is not None else False),
                dtype=object)
        if kind in (BoundMatMul, BoundLinear):
            if (kind is BoundMatMul and all(p.perturbed for p in node.inputs)) or (
                    kind is BoundLinear and any(p.perturbed for p in node.inputs[1:])):
                return self.atom(node, value)
            a, b = [self.build(p) for p in node.inputs[:2]]
            if kind is BoundLinear:
                if node.transA:
                    a = a.swapaxes(-1, -2)
                if node.transB:
                    b = b.swapaxes(-1, -2)
                a = a * self.constant(node.alpha_linear)
            result = np.asarray(np.matmul(a, b), dtype=object)
            if len(node.inputs) == 3:
                result = result + self.build(node.inputs[2]) * self.constant(node.beta_linear)
            return result
        if kind is BoundPow:
            if node.inputs[1].perturbed:
                return self.atom(node, value)
            exponents = self.build(node.inputs[1])
            for e in exponents.flat:
                if any(k for k in e.terms):
                    return self.atom(node, value)
                exponent = e.terms.get((), Fraction(0))
                if exponent < 0 or exponent.denominator != 1:
                    return self.atom(node, value)
            a, exponents = np.broadcast_arrays(self.build(node.inputs[0]), exponents)
            result = np.empty(a.shape, dtype=object)
            for i, (p, e) in enumerate(zip(a.flat, exponents.flat)):
                exponent = e.terms.get((), Fraction(0))
                result.flat[i] = p ** int(exponent)
            return result
        if kind in (BoundNeg, BoundSqr):
            a = self.build(node.inputs[0])
            return -a if kind is BoundNeg else a * a
        if kind in (BoundAdd, BoundSub, BoundMul, BoundDiv):
            if kind is BoundDiv and node.inputs[1].perturbed:
                return self.atom(node, value)
            a, b = [self.build(p) for p in node.inputs]
            if kind is BoundAdd:
                return a + b
            if kind is BoundSub:
                return a - b
            return a * b if kind is BoundMul else a / b
        # An unsupported subgraph is an independent variable per element. Its
        # inputs are not expanded: only the same node can supply matching atoms.
        return self.atom(node, value)

    def scales(self, node):
        u, v = [self.build(p) for p in node.inputs]
        scales = []
        for pu, pv in zip(u.flat, v.flat):
            self.budget.spend()
            pu, pv = pu.terms, pv.terms
            if not pu or pu.keys() != pv.keys():
                return None
            anchor = next(iter(pu))
            c = pv[anchor] / pu[anchor]
            if not c or any(pv[k] != c * pu[k] for k in pu):
                return None
            if float(c) != c:
                return None
            scales.append(c)
        value = self.model.get_forward_value(node)
        result = torch.tensor([float(c) for c in scales],
                              dtype=value.dtype, device=value.device).reshape(value.shape)
        # Also reject rounding to the model's dtype (e.g. float32).
        if any(Fraction(float(a)) != c for a, c in zip(result.flatten().tolist(), scales)):
            return None
        return result


def rewrite_hidden_squares(model: 'BoundedModule'):
    """Opt-in exact proof of elementwise v = c*u, replacing u*v by c*Sqr(u).

    The replacement computes the identical polynomial in inputs and opaque
    atoms, proved with rational coefficients. Unsupported perturbed nodes are
    atoms keyed by node name and element index; unperturbed ones are constants.
    Degree > 4, > 2000 terms, and inexact scales are refused. An operation
    budget (bound_opts['op_budget']['hidden_squares']) stops proof work on the remaining Mul
    nodes; with commit mode 'whole' exhaustion raises OpBudgetExhausted instead.
    Returns the rewrite count before subsequent CSE/orphan cleanup.
    """
    if not model.bound_opts.get('hidden_square_rewrite', False):
        return 0
    budget = stage_budget(model, 'hidden_squares')
    # convert_sqr has just inserted nodes whose perturbation flags are not set.
    model._mark_perturbed_nodes(model.global_input)
    proof = _HiddenSquareProof(model, budget)
    rewritten = 0
    for node in list(model.nodes()):
        try:
            budget.spend()
        except OpBudgetExhausted:
            budget.log_cap()
            break
        if type(node) is not BoundMul or not all(p.perturbed for p in node.inputs):
            continue
        value = model.get_forward_value(node)
        if any(model.get_forward_value(p).shape != value.shape for p in node.inputs):
            continue
        try:
            scales = proof.scales(node)
        except TimeoutError:
            budget.log_cap()
            break
        except Exception:
            # A proof that fails for any reason (an operator attribute this walk
            # does not know, a shape it cannot map) just means no rewrite.
            continue
        if scales is None:
            continue
        square = BoundSqr(inputs=[node.inputs[0]], options=model.bound_opts)
        square.name = f'{node.name}/hidden_square'
        _copy_node_properties(square, node)
        square.perturbed = True
        constant = BoundConstant(attr={'value': scales, 'device': value.device},
                                 options=model.bound_opts)
        constant.name = f'{node.name}/hidden_scale'
        constant.output_shape = scales.shape
        product = BoundMul(inputs=[square, constant], options=model.bound_opts)
        product.name = f'{node.name}/hidden_scaled_square'
        _copy_node_properties(product, node)
        product.perturbed = True
        model.add_nodes([square, constant, product])
        model.replace_node(node, product)
        rewritten += 1
        logger.debug('Hidden square: replacing %s with %s', node, product)
    budget.finish()
    if budget.logged and commit_whole(model):
        raise OpBudgetExhausted('hidden_squares')
    return rewritten


def pow_variable_exponent_to_exp_log(model: 'BoundedModule'):
    """Rewrite x ** y with a non-constant exponent y as exp(y * log(x)).

    Opt-in (bound option `pow_variable_exponent_exp_log`). BoundPow bounds only a
    constant exponent; CROWN and IBP raise otherwise (Gulf: (c - x2) ** x3;
    ex8_4_4: 0.157 ** x15). A power with a real, variable exponent is defined for
    a positive base only (GAMS/AMPL `**` with a real exponent), where
    x ** y = exp(y * log(x)) holds exactly; outside it BoundLog treats a base
    <= 0 as it treats any log argument (domain_aware_ops). A constant base
    becomes a constant log node.
    """
    if not model.bound_opts.get('pow_variable_exponent_exp_log', False):
        return
    constant_types = (BoundBuffers, BoundConstant, BoundCast)
    for node in list(model.nodes()):
        if type(node) != BoundPow or isinstance(node.inputs[1], constant_types):
            continue
        base, exponent = node.inputs
        node_log = BoundLog(inputs=[base], options=model.bound_opts)
        node_log.name = f'{node.name}/log'
        _copy_node_properties(node_log, base)
        node_mul = BoundMul(inputs=[exponent, node_log], options=model.bound_opts)
        node_mul.name = f'{node.name}/mul'
        _copy_node_properties(node_mul, node)
        node_exp = BoundExp(inputs=[node_mul], options=model.bound_opts)
        node_exp.name = f'{node.name}/exp'
        _copy_node_properties(node_exp, node)
        model.add_nodes([node_log, node_mul, node_exp])
        logger.debug('Replacing %s with exp(y * log(x))', node)
        model.replace_node(node, node_exp)


def merge_identical_gather(model: 'BoundedModule'):
    """Merge BoundGather nodes that select the same constant index from the same node.

    Generated models write x_i * x_i as the product of two separate gathers of x_i. As
    distinct nodes the product is relaxed as a bilinear term of independent operands
    (on [-100, 100] its lower bound is -1e4 instead of 0) and convert_sqr, which needs
    one node on both sides, never fires. The merged nodes compute the same values, so
    this is exact.
    """
    seen = {}
    for node in list(model.nodes()):
        if type(node) is not BoundGather or len(node.inputs) != 2:
            continue
        x, index = node.inputs
        # A computed index may change from one forward to the next; only leaves are fixed.
        if index.perturbed or index.inputs or x.output_shape is None:
            continue
        idx = model.get_forward_value(index)
        if not isinstance(idx, torch.Tensor):
            continue
        axis = node.axis if node.axis >= 0 else node.axis + len(x.output_shape)
        key = (x.name, axis, idx.dtype, tuple(idx.shape), tuple(idx.reshape(-1).tolist()))
        if key in seen:
            logger.debug('Merging node %s to %s', node, seen[key])
            model.replace_node(node, seen[key])
        else:
            seen[key] = node


def _cse_key(model, node):
    """Hashable identity of what `node` computes, or None if it must not be merged."""
    if not node.inputs:
        if node.perturbed or type(node).__name__ == 'BoundInput':
            return None
        try:
            v = model.get_forward_value(node)
        except Exception:  # noqa: BLE001
            return None
        # Large tensors (weight matrices) are rarely duplicated; hashing their bytes
        # would copy them.
        if not isinstance(v, torch.Tensor) or v.numel() > 100_000:
            return None
        v = v.detach().cpu()
        return ('const', type(node).__name__, str(v.dtype), tuple(v.shape),
                v.contiguous().view(-1).numpy().tobytes())
    attr = getattr(node, 'attr', None) or {}
    try:
        attr_key = tuple(sorted((k, repr(v)) for k, v in attr.items() if k != 'device'))
    except Exception:  # noqa: BLE001
        return None
    # Not every op keeps its parameters in `attr` (a Cast's target dtype does not), so
    # the output dtype and shape are part of the key too.
    fv = getattr(node, 'forward_value', None)
    if not isinstance(fv, torch.Tensor):
        return None
    return (type(node).__name__, attr_key, tuple(i.name for i in node.inputs),
            getattr(node, 'output_index', 0), str(fv.dtype), tuple(fv.shape))


def merge_common_subexpressions(model: 'BoundedModule'):
    """Merge nodes that compute the same thing: leaf constants with equal values, and
    nodes of the same type and attributes on the same inputs (in topological order, so
    merged inputs expose merged consumers).

    Generated models give every sub-expression its own constant buffers, which the
    ONNX export cannot merge, so a repeated expression such as (x1 - x0^2) appears as
    distinct nodes and its product with itself is relaxed as a bilinear term of
    independent operands (-2800 instead of 0 for 100 (x1 - x0^2)^2 on [-2, 2]^2).
    Merged nodes compute identical values, so this is exact.
    """
    order, seen_names = [], set()

    def visit(node):
        stack = [(node, False)]
        while stack:
            n, done = stack.pop()
            if done:
                order.append(n)
                continue
            if n.name in seen_names:
                continue
            seen_names.add(n.name)
            stack.append((n, True))
            for i in n.inputs:
                if i.name not in seen_names:
                    stack.append((i, False))

    visit(model[model.final_name])
    canon = {}
    for node in order:
        if node.name not in model._modules:
            continue
        key = _cse_key(model, node)
        if key is None:
            continue
        other = canon.get(key)
        if other is None or other is node:
            canon[key] = node
            continue
        logger.debug('CSE: merging %s into %s', node, other)
        model.replace_node(node, other)


def merge_identical_act(model: 'BoundedModule'):
    """Merge identical BoundActivation"""
    nodes = list(model.nodes())
    merged = [False] * len(nodes)
    for i in range(len(nodes)):
        if (not merged[i]
                and isinstance(nodes[i], BoundActivation)
                and len(nodes[i].inputs) == 1):
            for j in range(i + 1, len(nodes)):
                if (not merged[j]
                        and type(nodes[j]) == type(nodes[i])
                        and len(nodes[i].inputs) == 1):
                    if nodes[i].inputs[0] == nodes[j].inputs[0]:
                        logger.debug('Merging node %s to %s', nodes[j], nodes[i])
                        model.replace_node(nodes[j], nodes[i])
                        merged[j] = True


def minmax_to_relu(model: 'BoundedModule'):
    """Replace BoundMinMax with BoundRelu if one of its inputs is constant"""
    nodes = list(model.nodes())
    for node in nodes:
        if type(node) == BoundMax:
            for i, input_node in enumerate(node.inputs):
                if not input_node.perturbed:
                    logger.debug('Replacing BoundMax node %s', node)
                    # max(x, c) = ReLU(x - c) + c
                    node_sub = BoundSub(inputs=[node.inputs[1-i], input_node],
                                        options=model.bound_opts)
                    node_sub.name = f'{node.name}/sub'
                    _copy_node_properties(node_sub, node)
                    node_relu = BoundRelu(inputs=[node_sub],
                                          options=model.bound_opts)
                    node_relu.name = f'{node.name}/relu'
                    _copy_node_properties(node_relu, node)
                    node_add = BoundAdd(inputs=[node_relu, input_node],
                                        options=model.bound_opts)
                    node_add.name = f'{node.name}/add'
                    _copy_node_properties(node_add, node)
                    model.add_nodes([node_sub, node_relu, node_add])
                    model.replace_node(node, node_add)
                    break
        elif type(node) == BoundMin:
            for i, input_node in enumerate(node.inputs):
                if not input_node.perturbed:
                    logger.debug('Replacing BoundMin node %s', node)
                    # min(x, c) = -ReLU(c - x) + c
                    node_sub_1 = BoundSub(inputs=[input_node, node.inputs[1-i]],
                                          options=model.bound_opts)
                    node_sub_1.name = f'{node.name}/sub/1'
                    _copy_node_properties(node_sub_1, node)
                    node_relu = BoundRelu(inputs=[node_sub_1],
                                          options=model.bound_opts)
                    node_relu.name = f'{node.name}/relu'
                    _copy_node_properties(node_relu, node)
                    node_sub_2 = BoundSub(inputs=[input_node, node_relu],
                                          options=model.bound_opts)
                    node_sub_2.name = f'{node.name}/sub/2'
                    _copy_node_properties(node_sub_2, node)
                    model.add_nodes([node_sub_1, node_relu, node_sub_2])
                    model.replace_node(node, node_sub_2)
                    break

def _check_merge(W_merge, skip, pairs, bias=None):
    """
    Check that the merge layer C has the exact structure required by the current fusion logic.

    W_merge: Tensor of shape (n_out, n_mid)
             or (n_out, n_mid, 1, 1)        
    skip: set[int] of skipped src indices
    pairs: dict[int,int] mapping src=j -> merge row r (from _pair_row), for adj pair (j,j+1)
    bias: optional Tensor of shape (n_out,) for merge bias. If provided, we enforce bias ~ 0
          for one-hot rows (since ReLU(h + b) == h only if b == 0 in general).

    Returns: (ok: bool)
    """
    if W_merge.dim() == 4:
        W = W_merge[..., 0, 0].detach()
    elif W_merge.dim() == 2:
        W = W_merge.detach()
    else:
        return False

    n_out, n_mid = W.shape
    dst2src = [s for s in range(n_mid) if s not in skip]
    if len(dst2src) != n_out:
        return False

    b = bias.detach() if bias is not None else None

    for r, src in enumerate(dst2src):
        nz = torch.nonzero(W[r].abs() > 1e-8, as_tuple=False).flatten()
        nnz = len(nz)

        # In this case, the rows should contain a +1 at the correct location.
        # And the same position in b should be 0.
        if nnz == 1:
            c = int(nz[0].item())
            if (c != src) or ((W[r, c] - 1.0).abs().item() > 1e-8) or \
               (b is not None and b[r].abs().item() > 1e-8):
                return False

        # In this case, the row should contain a pair of +1, -1 at the correct location.
        elif nnz == 2:
            c0 = int(nz[0].item())
            c1 = int(nz[1].item())
            if not (c0 == src and c1 == src + 1) or ((W[r, c0] - 1.0).abs().item() > 1e-8) \
                or ((W[r, c1] + 1.0).abs().item() > 1e-8):
                return False
            # Must correspond to a detected pair, and the mapping must hit this row.
            if pairs.get(src, None) != r:
                return False

    return True

def _pair_row(Ws, bs, Wm, j, atol=1e-8):
    """
    Checks the relation ReLU(x) - ReLU(-x) = x. Return
    the index at the merge weight if the relation exists,
    otherwise return None.
    """
    # Check whether this fits the pattern in docstring.
    if not (torch.allclose(Ws[j+1], -Ws[j], atol=atol)
            and abs(float(bs[j] + bs[j+1])) < atol):
        return None

    # Make merge weight 4D so Gemm and Conv share same indexing
    if Wm.dim() == 2:                 # Gemm path
        Wm4 = Wm.unsqueeze(-1).unsqueeze(-1)    
    else:                             # Conv path 
        Wm4 = Wm

    # Find corresponding columns of the merge weight
    # We check 1) The two nonzero element are in the same row
    #          2) The two entries are +1 and -1
    # If the check pass, we return the row index, otherwise it 
    # is not a valid pattern match and we return None.
    rows = torch.nonzero(Wm4[:, [j, j+1], 0, 0], as_tuple=False)
    if rows.size(0) != 2 or rows[0, 0] != rows[1, 0]:
        return None
    r = int(rows[0, 0])

    ok = (abs(float(Wm4[r, j, 0, 0] - 1)) < atol and
          abs(float(Wm4[r, j+1, 0, 0] + 1)) < atol and
          torch.count_nonzero(Wm4[r]) == 2)
    return r if ok else None
                
def optimize_relu_relation(model: 'BoundedModule'):
    """
    This graph optimization detects the optimizable path with
    the identity
        ReLU(ReLU(x + b) - ReLu(-x - b)) = ReLU(x + b)
    for both linear layer and convolution layer. Replace the 
    sequence of nodes with pattern
        Gemm -> ReLU -> Gemm -> ReLU or
        Conv -> ReLU -> Conv -> ReLU
    to one single Gemm -> ReLU or Conv -> ReLU.
    """
    nodes = list(model.nodes())
    i = 0
    while i + 3 < len(nodes):
        A, B, C, D = nodes[i:i+4]
        
        # In Conv layers, we detect whether the optimization can be done
        # for pairs of channels. If so, the optimization eliminates one
        # Conv layer and recover the original results with the identity 
        # in docstring.
        if (isinstance(A, BoundConv) and isinstance(B, BoundRelu) and
            isinstance(C, BoundConv) and isinstance(D, BoundRelu) and tuple(C.attr['kernel_shape'])==(1,1)):
            
            Ws = model.get_forward_value(C.inputs[1])
            Wc = model.get_forward_value(A.inputs[1])
            
            # We only care about 2D conv
            if Ws.ndim != 4 or Wc.ndim != 4:
                i += 1
                continue
            
            bs = (model.get_forward_value(C.inputs[2]) if C.has_bias
                  else torch.zeros_like(Ws[:, 0, 0, 0]))
            bc = (model.get_forward_value(A.inputs[2]) if A.has_bias
                  else torch.zeros_like(Wc[:, 0, 0, 0]))
            
            # Detect whether and where the identity presents in the weight matrix.
            pairs, skip = {}, set()
            for j in range(0, Wc.size(0) - 1):
                r = _pair_row(Wc, bc, Ws, j)
                if r is not None:
                    pairs[j] = r
                    skip.add(j + 1)

            ok = _check_merge(Ws, skip, pairs, bias=bs)
            if not ok:
                i += 1
                continue
            
            if pairs:
                Cout, Cin, kH, kW = Ws.size(0), Wc.size(1), *Wc.shape[2:]
                W_new = torch.empty((Cout, Cin, kH, kW), dtype=Wc.dtype, device=Wc.device)
                b_new = torch.empty((Cout,), dtype=bc.dtype, device=bc.device)

                
                # Build fused weight and bias
                dst = 0
                for src in range(Wc.size(0)):
                    if src in skip:
                        continue
                    b_new[dst] = bs[pairs[src]] + bc[src] if src in pairs else bc[src]
                    W_new[dst] = Wc[src]
                    dst += 1
                
                # Modify the graph using the newly built weights and bias
                weight_node = BoundParams('fused_weight', torch.nn.Parameter(W_new))
                bias_node = BoundParams('fused_bias', torch.nn.Parameter(b_new))
                weight_node.name = f'{A.name}/optimized/weight' 
                bias_node.name = f'{A.name}/optimized/bias'
                
                fused = BoundConv(
                    attr=A.attr.copy(),
                    inputs=[A.inputs[0], weight_node, bias_node],
                    output_index=A.output_index,
                    options=model.bound_opts
                )
                fused.name = f'{A.name}/optimized'
                _copy_node_properties(fused, A)
                relu = BoundRelu(inputs=[fused], options=model.bound_opts)
                relu.name = f'{A.name}/optimized/relu'
                _copy_node_properties(relu, D)
                
                model.add_nodes([weight_node, bias_node, fused, relu])
                model.replace_node(D, relu)
                model.replace_node(A, fused)
                model.delete_node(B)
                model.delete_node(C) 
                
                # Skip the full sequence once the pattern is detected
                i += 4
                continue
        
        # In Linear layer, we detect whether the optimization can be 
        # done for pair of rows. The code structure is similar the 
        # one at Conv branch. 
        elif (isinstance(A, BoundLinear) and isinstance(B, BoundRelu) and
            isinstance(C, BoundLinear) and isinstance(D, BoundRelu)):
            
            Ws = model.get_forward_value(A.inputs[1])
            Wm = model.get_forward_value(C.inputs[1])
            bs = (model.get_forward_value(A.inputs[2])
                  if len(A.inputs) == 3 else torch.zeros_like(Ws[:, 0]))
            bm = (model.get_forward_value(C.inputs[2])
                  if len(C.inputs) == 3 else torch.zeros_like(Wm[:, 0]))
            
            pairs, skip = {}, set()
            for j in range(0, Ws.size(0) - 1):
                r = _pair_row(Ws, bs, Wm, j)
                if r is not None:
                    pairs[j] = r
                    skip.add(j + 1)

            ok = _check_merge(Wm, skip, pairs, bias=bm)
            if not ok:
                i += 1
                continue
                 
            if pairs:
                n_out = Wm.shape[0]
                W_new = torch.empty((n_out, Ws.shape[1]), dtype=Ws.dtype, device=A.attr['device'])
                b_new = torch.empty((n_out,), dtype=bs.dtype, device=A.attr['device'])

                dst = 0
                for src in range(Ws.size(0)):
                    if src in skip:
                        continue
                    b_new[dst] = bm[pairs[src]] + bs[src] if src in pairs else bs[src]
                    W_new[dst] = Ws[src]
                    dst += 1
                
                weight_node = BoundParams('fused_weight', torch.nn.Parameter(W_new), attr=dict(device=A.attr['device']))
                bias_node = BoundParams('fused_bias', torch.nn.Parameter(b_new), attr=dict(device=A.attr['device']))
                weight_node.name = f'{A.name}/optimized/weight'
                bias_node.name = f'{A.name}/optimized/bias'
                
                fused = BoundLinear(
                    attr=A.attr.copy(),
                    inputs=[A.inputs[0], weight_node, bias_node],
                    output_index=A.output_index,
                    options=model.bound_opts
                )
                fused.name = f'{A.name}/optimized'
                _copy_node_properties(fused, A)
                relu = BoundRelu(inputs=[fused], options=model.bound_opts)
                relu.name = f'{A.name}/optimized/relu'
                _copy_node_properties(relu, D)
                
                model.add_nodes([weight_node, bias_node, fused, relu])
                model.replace_node(D, relu)
                model.delete_node(A)
                model.delete_node(B)
                model.delete_node(C)
                
                i += 4
                continue
        i += 1
