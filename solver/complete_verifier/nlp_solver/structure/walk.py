"""One pass over a BoundedModule computing the exact Form of every element of every node.

Constant (input-independent) nodes stay tensors (LazyConstant) until an operation needs
them as Forms; matrix products with a constant matrix touch only its nonzeros."""
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import torch

from .forms import BudgetExceeded, Form, ZERO, constant, exact

_MOVE = {"BoundGather", "BoundTranspose", "BoundReshape", "BoundConcat", "BoundUnsqueeze", "BoundSqueeze",
         "BoundSlice", "BoundExpand", "BoundFlatten", "BoundIdentity", "BoundSplit", "BoundGatherElements"}


def _obj(shape, fill=None):
    a = np.empty(shape, dtype=object)
    a.fill(fill) if fill is not None else None
    return a


def _arr(values, shape):
    a = np.empty(len(values), dtype=object)
    a[:] = values
    return a.reshape(shape)


class LazyConstant:
    """a constant node: kept as a tensor until some op needs it as Forms (big weight matrices never are)"""
    def __init__(self, v):
        self.v = v
        self.shape = tuple(v.shape)
        self.ndim = len(self.shape)

    def materialize(self):
        return constant_array(self.v)


def constant_array(v):
    v = torch.as_tensor(v).detach().cpu()
    out = np.empty(v.numel(), dtype=object)
    flat = v.reshape(-1).tolist()
    for i, x in enumerate(flat):
        if isinstance(x, bool):
            x = int(x)
        out[i] = ZERO if x == 0 else constant(exact(x))
    return out.reshape(tuple(v.shape))


def _frac_cache():
    cache = {}

    def f(x):
        v = cache.get(x)
        if v is None:
            v = cache[x] = exact(x)
        return v
    return f


def matmul_constant(w, a, W, side):
    """a (object array, <=2-D) times a constant float matrix W using only W's nonzeros.
    side='right': out = a @ W (W: k x m);  side='left': out = W @ a (W: m x k)."""
    import scipy.sparse as sp
    exact_of = _frac_cache()
    a2 = a.reshape(1, -1) if a.ndim == 1 else a
    if side == 'right':
        csc = sp.csc_matrix(W)                              # columns = outputs
        cols = [(csc.indices[csc.indptr[j]:csc.indptr[j + 1]], csc.data[csc.indptr[j]:csc.indptr[j + 1]]) for j in range(W.shape[1])]
        out = np.empty((a2.shape[0], W.shape[1]), dtype=object)
        for i in range(a2.shape[0]):
            for j, (ks, cs) in enumerate(cols):
                out[i, j] = w.lincomb([(a2[i, k], exact_of(float(c))) for k, c in zip(ks, cs)])
        return out
    csr = sp.csr_matrix(W)
    out = np.empty((W.shape[0], a2.shape[1]), dtype=object)
    for i in range(W.shape[0]):
        ks, cs = csr.indices[csr.indptr[i]:csr.indptr[i + 1]], csr.data[csr.indptr[i]:csr.indptr[i + 1]]
        for j in range(a2.shape[1]):
            out[i, j] = w.lincomb([(a2[k, j], exact_of(float(c))) for k, c in zip(ks, cs)])
    return out


def matmul(w, a, b):
    """object-array matmul (both operands symbolic), skipping zero entries"""
    av, bv = a.ndim == 1, b.ndim == 1
    a2 = a.reshape(1, -1) if av else a
    b2 = b.reshape(-1, 1) if bv else b
    out = np.empty((a2.shape[0], b2.shape[1]), dtype=object)
    for i in range(a2.shape[0]):
        # a rounded zero (poly {} but inexact) is kept: it taints the sum
        nzk = [k for k in range(a2.shape[1]) if a2[i, k].poly != {} or a2[i, k].inexact]
        for j in range(b2.shape[1]):
            out[i, j] = w.lincomb([(w.mul(a2[i, k], b2[k, j]), 1) for k in nzk])
    return out


def node_forms(w, node, forms, dep, input_name):
    name, ins = type(node).__name__, node.inputs
    shape = tuple(node.forward_value.shape)
    if name == 'BoundInput':
        if node.name != input_name or math.prod(shape) != w.n:
            return _obj(shape, Form(None, None, 'multi-input'))
        return _arr([Form({((j, 1),): 1}) for j in range(w.n)], shape)
    if not dep[node.name]:
        return LazyConstant(node.forward_value)
    lazy_ok = name in ('BoundMatMul', 'BoundLinear')
    args = []
    for i in ins:
        f = forms[i.name]
        if isinstance(f, LazyConstant) and not lazy_ok:
            f = forms[i.name] = f.materialize()
        args.append(f)
    ew2 = lambda f: np.frompyfunc(f, 2, 1)(*np.broadcast_arrays(args[0], args[1])).reshape(shape)
    ew1 = lambda f: np.frompyfunc(f, 1, 1)(args[0]).reshape(shape)
    if name == 'BoundAdd':
        return ew2(w.add)
    if name == 'BoundSub':
        return ew2(lambda x, y: w.add(x, w.neg(y)))
    if name == 'BoundMul':
        return ew2(w.mul)
    if name == 'BoundDiv':
        return ew2(w.div)
    if name == 'BoundNeg':
        return ew1(w.neg)
    if name == 'BoundSqr':
        return ew1(lambda x: w.pow(x, 2))
    if name == 'BoundReciprocal':
        return ew1(lambda x: w.pow(x, -1))
    if name == 'BoundSqrt':
        return ew1(lambda x: w.pow(x, Fraction(1, 2)))
    if name == 'BoundPow':
        if dep[ins[1].name]:
            return _obj(shape, w.unknown('var-exponent', *args[0].flat))
        ks = np.broadcast_to(np.asarray(ins[1].forward_value.detach().cpu()), shape)
        a = np.broadcast_to(args[0], shape)
        return _arr([w.pow(x, Fraction(float(k))) for x, k in zip(a.flat, ks.flat)], shape)
    if name == 'BoundExp':
        return ew1(w.exp)
    if name == 'BoundLog':
        return ew1(w.log)
    if name in ('BoundSin', 'BoundCos', 'BoundTan', 'BoundTanh', 'BoundAbs'):
        f = {'BoundSin': math.sin, 'BoundCos': math.cos, 'BoundTan': math.tan, 'BoundTanh': math.tanh, 'BoundAbs': abs}[name]
        return ew1(lambda x: w.fn(x, name[5:].lower(), f))
    if name == 'BoundMatMul' and args[0].ndim <= 2 and args[1].ndim <= 2:
        if not dep[ins[1].name]:
            W = ins[1].forward_value.detach().cpu().double().numpy()
            return matmul_constant(w, args[0], W.reshape(-1, 1) if W.ndim == 1 else W, 'right').reshape(shape)
        if not dep[ins[0].name]:
            W = ins[0].forward_value.detach().cpu().double().numpy()
            return matmul_constant(w, args[1], W.reshape(1, -1) if W.ndim == 1 else W, 'left').reshape(shape)
        return matmul(w, args[0], args[1]).reshape(shape)
    if name == 'BoundLinear':
        if getattr(node, 'transA', 0) or any(dep[i.name] for i in ins[1:]):
            return _obj(shape, w.unknown('linear-nonconst-weight', *args[0].flat))
        _, W, bias = node._preprocess(torch.zeros(1), *[i.forward_value.double() for i in ins[1:]])
        out = matmul_constant(w, args[0], W.t().double().cpu().numpy(), 'right')
        if float(node.alpha_linear) != 1.0:      # (alpha x) W^T = alpha (x W^T) in exact rationals; fl(alpha W) is not exact
            c = exact(float(node.alpha_linear))
            out = np.frompyfunc(lambda f: w.scale(f, c), 1, 1)(out).reshape(out.shape)
        if bias is not None:
            out = np.frompyfunc(w.add, 2, 1)(*np.broadcast_arrays(out.reshape(-1, out.shape[-1]), constant_array(bias)))
        return out.reshape(shape)
    if name in ('BoundReduceSum', 'BoundReduceMean'):
        axis = getattr(node, 'axis', None)
        a0 = args[0]
        axes = tuple(range(a0.ndim)) if axis is None else tuple(
            k % a0.ndim for k in (axis if isinstance(axis, (list, tuple)) else [axis]))
        keep = bool(getattr(node, 'keepdim', False))
        groups = {}
        for idx in np.ndindex(a0.shape):
            dst = tuple(0 if i in axes else k for i, k in enumerate(idx)) if keep else tuple(
                k for i, k in enumerate(idx) if i not in axes)
            groups.setdefault(dst, []).append((a0[idx], 1))
        out = np.full(shape, ZERO, dtype=object)
        for dst, pairs in groups.items():
            out[dst] = w.lincomb(pairs)
        if name == 'BoundReduceMean':
            cnt = math.prod(a0.shape[i] for i in axes)
            out = np.frompyfunc(lambda x: w.scale(x, exact(Fraction(1, cnt))), 1, 1)(out)
        return out
    if name == 'BoundATenIndex_add' and not any(dep[ins[k].name] for k in (1, 2, 4)):
        axis = int(ins[1].forward_value) % args[0].ndim
        idx = ins[2].forward_value.long().reshape(-1).tolist()
        alpha = exact(float(ins[4].forward_value))
        data, src = args[0], args[3]
        target = tuple(max(a, b) if k != axis else a for k, (a, b) in enumerate(zip(data.shape, src.shape)))
        data = np.broadcast_to(data, target).copy()
        ss = list(target)
        ss[axis] = src.shape[axis]
        src = np.broadcast_to(src, ss)
        groups = {}
        for loc in np.ndindex(src.shape):
            dst = list(loc)
            dst[axis] = idx[loc[axis]]
            groups.setdefault(tuple(dst), []).append((src[loc], alpha))
        for dst, pairs in groups.items():
            data[dst] = w.lincomb([(data[dst], 1)] + pairs)
        return data.reshape(shape)
    if name == 'BoundWhere' and not dep[ins[0].name]:
        return np.where(ins[0].forward_value.cpu().numpy().astype(bool), args[1], args[2]).reshape(shape)
    if name == 'BoundCast':
        s, t = ins[0].forward_value, node.forward_value
        if s.dtype == t.dtype or (s.is_floating_point() and t.is_floating_point()
                                   and torch.finfo(s.dtype).bits <= torch.finfo(t.dtype).bits):
            return args[0]
        return _obj(shape, w.unknown('cast', *args[0].flat))
    if name in _MOVE:
        count = len(ins) if name == 'BoundConcat' else 1
        table, labels = [], []
        for i, arr in enumerate(args):
            if i >= count:
                if dep[ins[i].name]:
                    return _obj(shape, w.unknown('symbolic-index', *table))
                labels.append(ins[i].forward_value)
            else:
                off = len(table)
                table.extend(arr.flat)
                labels.append(torch.arange(off, off + arr.size, dtype=torch.int64).reshape(arr.shape))
        state = node.__dict__.copy()
        try:
            moved = node.forward(*labels).detach().cpu().reshape(-1)
        finally:
            node.__dict__.clear(); node.__dict__.update(state)
        return _arr([table[int(v)] for v in moved], shape)
    return _obj(shape, Form(None, None, 'op:' + name))



@dataclass
class Forms:
    forms: dict      # node name -> object array of Form, or LazyConstant
    final: str
    nodes: int


def build_forms(bounded_module, x0, algebra):
    """Forms of every node of bounded_module at the input x0 (shape-defining); the cached
    forward values are those of x0 before and after the walk. An exception in a node makes
    that node's elements unknown; an exhausted budget makes this and all later nodes unknown."""
    bounded_module(x0)
    final = bounded_module[bounded_module.final_name]
    order, seen = [], set()
    stack = [(final, 0)]
    # iterative post-order (graphs can be deep)
    while stack:
        node, state = stack.pop()
        if state == 0:
            if node.name in seen:
                continue
            seen.add(node.name)
            stack.append((node, 1))
            for i in node.inputs:
                stack.append((i, 0))
        else:
            order.append(node)
    dep, forms = {}, {}
    input_name = bounded_module.input_name[0]
    out_of_budget = False
    for node in order:
        tn = type(node).__name__
        dep[node.name] = tn == 'BoundInput' or (tn != 'BoundShape' and any(dep[i.name] for i in node.inputs))
        shape = tuple(node.forward_value.shape)
        if out_of_budget:
            forms[node.name] = _obj(shape, Form(None, None, 'budget'))
            continue
        try:
            f = node_forms(algebra, node, forms, dep, input_name)
            forms[node.name] = f if isinstance(f, LazyConstant) else np.asarray(f, dtype=object).reshape(shape)
        except BudgetExceeded:
            out_of_budget = True
            forms[node.name] = _obj(shape, Form(None, None, 'budget'))
        except (ValueError, TypeError, IndexError, AttributeError, OverflowError, ZeroDivisionError, RuntimeError, KeyError) as e:
            forms[node.name] = _obj(shape, Form(None, None, 'exc:' + type(e).__name__))
    bounded_module(x0)
    return Forms(forms, final.name, len(order))


def output_forms(forms):
    """Flat object array of the final node's elements."""
    out = forms.forms[forms.final]
    if isinstance(out, LazyConstant):
        out = out.materialize()
    return out.reshape(-1)
