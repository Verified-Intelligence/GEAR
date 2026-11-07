"""Legacy node translator (G5 WP6, decision D4).

Translates the module tree of a legacy GEAR artifact (the classes of the two ``modules.py`` files plus the
``api.py`` output wrappers) straight into the node tuples that ``auto_LiRPA.parse_graph.parse_module``
produces today, without tracing the module with ``torch.jit``.  ``torch.export`` cannot be used for these
artifacts (their forward methods call ``.item()`` at trace time), and the JIT tracer is being retired.

How the tuples are reproduced.  ``parse_module`` is ``torch.jit`` tracing, followed by a fixed pipeline of
JIT passes and the ONNX symbolic functions.  The translator keeps the same stages, on a tiny IR of its own:

1. *Emission* (``_Tr`` and the ``_emit_*`` functions).  Each class is replayed by hand-written code that
   mirrors its ``forward`` and records the ATen-level nodes the tracer would record, constants included, in
   the same order.  The replay runs real tensors alongside (batch size 1), so shapes, dtypes and the
   data-dependent branches (``.item()``) are evaluated by torch itself.
2. *JIT passes* (``_jit_passes``): dead code elimination, constant propagation of constant lists, common
   subexpression elimination (with the alias guard for tensors that are later mutated), the ``Int(NumToTensor)``
   peephole, and removal of the in-place ``index_add_`` / ``add_``.
3. *ONNX conversion* (``_Onnx``): the symbolic functions of ``torch.onnx`` (opset 17) for the ATen ops used,
   the list-to-``Concat`` peephole, dead code elimination.
4. *Naming*.  Values are renamed ``/<k>`` in order (inputs first).  The tracer additionally names a value after
   a Python local variable when an op runs inside a helper function (``v``, ``input``); the collision rules of
   ``Value::setDebugName`` and of the ONNX pass are reproduced in ``_NameTable`` / ``_onnx_names``.

Correctness is proven by exact tuple equality with ``parse_module`` (see ``tests/test_legacy_translate.py``).  Anything outside the vocabulary
raises ``LegacyTranslateError``; there is no silent fallback.
"""
from __future__ import annotations

import re
from collections import OrderedDict

import torch

from auto_LiRPA.parse_graph import Node

INT_MAX = 9223372036854775807


class LegacyTranslateError(NotImplementedError):
    """The module is outside the translator's vocabulary."""


# ---------------------------------------------------------------------------------------------------------
# Debug-name table: a Python model of torch::jit::Value::setDebugName.
# ---------------------------------------------------------------------------------------------------------
class _NameTable:
    """Names of the values of one graph.  A name is owned by one value; asking for a taken name renames the
    previous owner to ``base.<k>`` with a per-base counter that only grows (``Graph::name_base_suffix``)."""

    def __init__(self):
        self.owner = {}
        self.suffix = {}
        self.own = []

    def new(self):
        self.own.append('')
        return len(self.own) - 1

    def get(self, i):
        return self.own[i]

    def set(self, i, name):
        cur = self.own[i]
        if cur:
            del self.owner[cur]
            self.own[i] = ''
        if not name:
            return
        if name in self.owner:
            old = self.owner[name]
            suffix, base = 1, name
            m = re.fullmatch(r'(.*)\.(\d+)', name)
            if m:
                base, suffix = m.group(1), int(m.group(2))
            if base in self.suffix:
                suffix = max(suffix, self.suffix[base] + 1)
            while True:
                rep = f'{base}.{suffix}'
                suffix += 1
                if rep not in self.owner:
                    break
            self.suffix[base] = suffix
            self.set(old, rep)
        self.owner[name] = i
        self.own[i] = name


def _base(name):
    m = re.fullmatch(r'(.*)\.(\d+)', name)
    return m.group(1) if m else name


# ---------------------------------------------------------------------------------------------------------
# ATen-level IR (what the tracer records)
# ---------------------------------------------------------------------------------------------------------
class _Val:
    __slots__ = ('idx', 'ty', 'node', 'inp')

    def __init__(self, idx, ty, node=None, inp=None):
        self.idx, self.ty, self.node, self.inp = idx, ty, node, inp  # inp: position among the graph inputs


class _Nd:
    __slots__ = ('kind', 'ins', 'outs', 'value')

    def __init__(self, kind, ins, outs, value=None):
        self.kind, self.ins, self.outs, self.value = kind, ins, outs, value


class _T:
    """A traced tensor: the IR value plus the real tensor computed alongside (batch size 1)."""
    __slots__ = ('v', 't')

    def __init__(self, v, t):
        self.v, self.t = v, t


class _Sz:
    """An element of ``x.shape``: a traced integer (``NumToTensor(size(x, d))``)."""
    __slots__ = ('n2t', 'val')

    def __init__(self, n2t, val):
        self.n2t, self.val = n2t, val


def _ctype(x):
    if x is None:
        return 'none'
    if isinstance(x, bool):
        return 'bool'
    if isinstance(x, int):
        return 'int'
    if isinstance(x, float):
        return 'float'
    if isinstance(x, str):
        return 'device'
    if isinstance(x, (list, tuple)):
        return 'ilist'
    raise LegacyTranslateError(f'constant of type {type(x)}')


class _Tr:
    """Records the ATen graph of one module call.  Mirrors the order in which the JIT tracer creates nodes:
    non-tensor arguments become ``prim::Constant`` nodes just before the op, in argument order."""

    def __init__(self):
        self.nodes = []
        self.vals = []
        self.names = _NameTable()
        self.inputs = []

    # -- plumbing ---------------------------------------------------------------------------------------
    def val(self, ty, node=None, inp=None):
        v = _Val(self.names.new(), ty, node, inp)
        self.vals.append(v)
        return v

    def add(self, kind, ins, out_tys, value=None):
        n = _Nd(kind, list(ins), [], value)
        for ty in out_tys:
            n.outs.append(self.val(ty, node=n))
        self.nodes.append(n)
        return n

    def const(self, x):
        return self.add('prim::Constant', [], [_ctype(x)], value=x).outs[0]

    def input(self, tensor):
        v = self.val('T', inp=len(self.inputs))
        self.inputs.append(v)
        return _T(v, tensor)

    def touch(self, t, fr):
        """``TracingState::getValue``: an unnamed value used in a helper function takes the name of the caller's
        local variable that holds it."""
        if fr is not None and not self.names.get(t.v.idx):
            for k, o in fr.items():
                if o is t and k != 'self':
                    self.names.set(t.v.idx, k)
                    return

    def tens(self, t, fr):
        self.touch(t, fr)
        return t.v

    def ilist(self, items):
        """Integer-list argument: traced sizes first (``aten::Int``), then constants, then ListConstruct."""
        ints = []
        for it in items:
            if isinstance(it, _Sz):
                ints.append(self.add('aten::Int', [it.n2t], ['int']).outs[0])
            else:
                ints.append(None)
        vs = []
        for it, iv in zip(items, ints):
            vs.append(iv if iv is not None else self.const(int(it)))
        return self.add('prim::ListConstruct', vs, ['ilist']).outs[0]

    def shape(self, t, fr=None):
        """``t.shape``: size nodes for every dim (dead ones are removed later)."""
        self.touch(t, fr)
        out = []
        for d in range(t.t.dim()):
            c = self.const(d)
            s = self.add('aten::size', [t.v, c], ['int']).outs[0]
            n = self.add('prim::NumToTensor', [s], ['NT']).outs[0]
            out.append(_Sz(n, int(t.t.shape[d])))
        return out

    def _res(self, kind, ins, real, out_ty='T'):
        n = self.add(kind, ins, [out_ty])
        return _T(n.outs[0], real)

    # -- ops (real tensors alongside) -------------------------------------------------------------------
    def index_select(self, x, dim, idx, fr=None):
        a = self.tens(x, fr)
        c = self.const(dim)
        i = self.tens(idx, fr)
        return self._res('aten::index_select', [a, c, i], torch.index_select(x.t, dim, idx.t))

    def binary(self, kind, a, b, real, fr=None, alpha=False):
        av, bv = self.tens(a, fr), self.tens(b, fr)
        ins = [av, bv]
        if alpha:
            ins.append(self.const(1))
        return self._res(kind, ins, real)

    def add_t(self, a, b, fr=None):
        return self.binary('aten::add', a, b, a.t + b.t, fr, alpha=True)

    def sub(self, a, b, fr=None):
        return self.binary('aten::sub', a, b, a.t - b.t, fr, alpha=True)

    def mul(self, a, b, fr=None):
        return self.binary('aten::mul', a, b, a.t * b.t, fr)

    def div(self, a, b, fr=None):
        return self.binary('aten::div', a, b, a.t / b.t, fr)

    def pow(self, a, b, fr=None):
        return self.binary('aten::pow', a, b, torch.pow(a.t, b.t), fr)

    def mul_scalar(self, a, s, fr=None):
        av = self.tens(a, fr)
        return self._res('aten::mul', [av, self.const(float(s))], a.t * float(s))

    def unary(self, name, a, fr=None):
        fn = {'log': torch.log, 'sin': torch.sin, 'cos': torch.cos, 'tan': torch.tan, 'exp': torch.exp,
              'sqrt': torch.sqrt, 'tanh': torch.tanh, 'abs': torch.abs, 'neg': torch.neg, 'atan': torch.atan}[name]
        return self._res('aten::' + name, [self.tens(a, fr)], fn(a.t))

    def t(self, a, fr=None):
        return self._res('aten::t', [self.tens(a, fr)], a.t.t())

    def matmul(self, a, b, fr=None):
        return self._res('aten::matmul', [self.tens(a, fr), self.tens(b, fr)], a.t @ b.t)

    def view(self, a, items, kind='aten::view', fr=None):
        av = self.tens(a, fr)
        lst = self.ilist(items)
        sizes = [it.val if isinstance(it, _Sz) else int(it) for it in items]
        return self._res(kind, [av, lst], a.t.reshape(*sizes))

    def reshape(self, a, items, fr=None):
        return self.view(a, items, 'aten::reshape', fr)

    def unsqueeze(self, a, dim, fr=None):
        av = self.tens(a, fr)
        return self._res('aten::unsqueeze', [av, self.const(dim)], a.t.unsqueeze(dim))

    def slice(self, a, dim, start, end, step=1, fr=None):
        av = self.tens(a, fr)
        ins = [av, self.const(dim), self.const(start), self.const(end), self.const(step)]
        e = None if end == INT_MAX else end
        idx = [slice(None)] * a.t.dim()
        idx[dim] = slice(start, e, step)
        return self._res('aten::slice', ins, a.t[tuple(idx)])

    def cat(self, ts, dim, fr=None):
        vs = [self.tens(t, fr) for t in ts]
        lst = self.add('prim::ListConstruct', vs, ['tlist']).outs[0]
        return self._res('aten::cat', [lst, self.const(dim)], torch.cat([t.t for t in ts], dim))

    def sum(self, a, dims, keepdim, fr=None):
        av = self.tens(a, fr)
        lst = self.add('prim::ListConstruct', [self.const(d) for d in dims], ['ilist']).outs[0]
        ins = [av, lst, self.const(bool(keepdim)), self.const(None)]
        return self._res('aten::sum', ins, a.t.sum(dim=tuple(dims), keepdim=keepdim))

    def reduce_dim(self, kind, a, dim, keepdim, fr=None):
        """``torch.min/max(a, dim=dim, keepdim=keepdim).values``"""
        av = self.tens(a, fr)
        n = self.add('aten::' + kind, [av, self.const(dim), self.const(bool(keepdim))], ['T', 'T'])
        fn = torch.min if kind == 'min' else torch.max
        return _T(n.outs[0], fn(a.t, dim=dim, keepdim=keepdim).values)

    def mean(self, a, dims, keepdim, fr=None):
        av = self.tens(a, fr)
        lst = self.add('prim::ListConstruct', [self.const(d) for d in dims], ['ilist']).outs[0]
        ins = [av, lst, self.const(bool(keepdim)), self.const(None)]
        return self._res('aten::mean', ins, a.t.mean(dim=tuple(dims), keepdim=keepdim))

    def stack(self, ts, dim, fr=None):
        vs = [self.tens(t, fr) for t in ts]
        lst = self.add('prim::ListConstruct', vs, ['tlist']).outs[0]
        return self._res('aten::stack', [lst, self.const(dim)], torch.stack([t.t for t in ts], dim))

    def zeros(self, items, dtype, fr=None):
        lst = self.ilist(items)
        ins = [lst, self.const(_DTYPE_CODE[dtype]), self.const(None), self.const('cpu'), self.const(False)]
        sizes = [it.val if isinstance(it, _Sz) else int(it) for it in items]
        return self._res('aten::zeros', ins, torch.zeros(*sizes, dtype=dtype))

    def zeros_like(self, a, fr=None):
        av = self.tens(a, fr)
        ins = [av, self.const(None), self.const(None), self.const(None), self.const(False), self.const(None)]
        return self._res('aten::zeros_like', ins, torch.zeros_like(a.t))

    def to(self, a, dtype, fr=None):
        av = self.tens(a, fr)
        ins = [av, self.const('cpu'), self.const(_DTYPE_CODE[dtype]), self.const(False), self.const(False),
               self.const(None)]
        return self._res('aten::to', ins, a.t.to(dtype))

    def expand(self, a, items, fr=None):
        av = self.tens(a, fr)
        lst = self.ilist(items)
        sizes = [it.val if isinstance(it, _Sz) else int(it) for it in items]
        return self._res('aten::expand', [av, lst, self.const(False)], a.t.expand(*sizes))

    def index_add(self, s, dim, idx, src, inplace=False, fr=None):
        sv = self.tens(s, fr)
        d = self.const(dim)
        iv = self.tens(idx, fr)
        srcv = self.tens(src, fr)
        al = self.const(1)
        kind = 'aten::index_add_' if inplace else 'aten::index_add'
        real = s.t.index_add(dim, idx.t, src.t.to(s.t.dtype))
        if inplace:
            n = self.add(kind, [sv, d, iv, srcv, al], ['T'])
            s.v, s.t = n.outs[0], real
            return s
        return self._res(kind, [sv, d, iv, srcv, al], real)

    def add_inplace(self, a, b, fr=None):
        """``a += b``: the tensor object keeps its identity, its IR value moves to the result."""
        av, bv = self.tens(a, fr), self.tens(b, fr)
        n = self.add('aten::add_', [av, bv, self.const(1)], ['T'])
        a.v, a.t = n.outs[0], a.t + b.t
        return a

    def name_input(self, t, name):
        """The caller-local lookup of a helper that is not in modules.py (``torch.unique`` names ``input``)."""
        if not self.names.get(t.v.idx):
            self.names.set(t.v.idx, name)

    def linear(self, x, w, b, fr=None):
        return self._res('aten::linear', [self.tens(x, fr), self.tens(w, fr), self.tens(b, fr)],
                         torch.nn.functional.linear(x.t, w.t, b.t))


_DTYPE_CODE = {torch.float64: 7, torch.float32: 6, torch.int64: 4, torch.bool: 11}
_CODE_DTYPE = {v: k for k, v in _DTYPE_CODE.items()}


# ---------------------------------------------------------------------------------------------------------
# JIT passes on the ATen graph
# ---------------------------------------------------------------------------------------------------------
_INPLACE = ('aten::index_add_', 'aten::add_')


def _replace_uses(nodes, old, new):
    for n in nodes:
        n.ins = [new if v is old else v for v in n.ins]


def _dce(nodes, outputs):
    live = {v.idx for v in outputs}
    keep = []
    for n in reversed(nodes):
        if any(o.idx in live for o in n.outs):
            keep.append(n)
            live.update(v.idx for v in n.ins)
    keep.reverse()
    return keep


def _const_key(n):
    v = n.value
    return ('c', n.outs[0].ty, tuple(v) if isinstance(v, list) else v)


def _jit_passes(tr, outputs):
    nodes = _dce(tr.nodes, outputs)
    # constant propagation: a list of int constants is itself a constant
    out = []
    for n in nodes:
        if (n.kind == 'prim::ListConstruct' and n.outs[0].ty == 'ilist' and n.ins
                and all(v.node is not None and v.node.kind == 'prim::Constant' and v.ty == 'int' for v in n.ins)):
            c = _Nd('prim::Constant', [], [n.outs[0]], [v.node.value for v in n.ins])
            n.outs[0].node = c
            out.append(c)
        else:
            out.append(n)
    nodes = _dce(out, outputs)
    # tensors that are mutated in place must not be merged
    mutated = {n.ins[0].idx for n in nodes if n.kind in _INPLACE}
    # common subexpression elimination
    seen, out = {}, []
    for n in nodes:
        if n.kind in _INPLACE or any(o.idx in mutated for o in n.outs):
            out.append(n)
            continue
        key = _const_key(n) if n.kind == 'prim::Constant' else (n.kind, tuple(v.idx for v in n.ins))
        if key in seen:
            first = seen[key]
            for o, f in zip(n.outs, first.outs):
                _replace_uses(nodes, o, f)
                outputs[:] = [f if v is o else v for v in outputs]
            continue
        seen[key] = n
        out.append(n)
    nodes = out
    # peephole: Int(NumToTensor(s)) -> s
    for n in nodes:
        if n.kind == 'aten::Int' and n.ins[0].node is not None and n.ins[0].node.kind == 'prim::NumToTensor':
            s = n.ins[0].node.ins[0]
            _replace_uses(nodes, n.outs[0], s)
    nodes = _dce(nodes, outputs)
    # remove in-place ops: the op becomes out-of-place; for index_add_ the producer of ``self`` moves right
    # before it (measured on the JIT pass)
    for n in [n for n in nodes if n.kind in _INPLACE]:
        if n.kind == 'aten::index_add_':
            n.kind = 'aten::index_add'
            p = n.ins[0].node
            if p is not None and p in nodes and nodes.index(p) < nodes.index(n):
                nodes.remove(p)
                nodes.insert(nodes.index(n), p)
        else:
            n.kind = 'aten::add'
    return nodes


# ---------------------------------------------------------------------------------------------------------
# ONNX conversion (symbolic functions of torch.onnx, opset 17) on a tiny graph IR
# ---------------------------------------------------------------------------------------------------------
class _OV:
    """An ONNX-level value.  ``os``: ONNX shape (None entries are dynamic); ``cv``: python value of a scalar
    constant; ``elems``: members of a packed int list."""
    __slots__ = ('node', 'dt', 'os', 'cv', 'elems', 'inp', 'req', 'pos')

    def __init__(self, node, dt, os, cv=None, elems=None, inp=None):
        self.node, self.dt, self.os, self.cv, self.elems, self.inp = node, dt, os, cv, elems, inp
        self.req = None  # base name requested by the tracer (ONNX-pass naming)
        self.pos = None


class _ON:
    __slots__ = ('op', 'ins', 'attrs', 'out')

    def __init__(self, op, ins, attrs, out):
        self.op, self.ins, self.attrs, self.out = op, list(ins), attrs, out


_ONNX_DTYPE = {torch.float64: 11, torch.float32: 1, torch.int64: 7, torch.bool: 9}


def _bcast(a, b):
    n = max(len(a), len(b))
    a = (1,) * (n - len(a)) + tuple(a)
    b = (1,) * (n - len(b)) + tuple(b)
    out = []
    for x, y in zip(a, b):
        if x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        elif x is None:
            out.append(y)
        else:
            out.append(x)
    return tuple(out)


class _Onnx:
    def __init__(self):
        self.nodes = []

    def op(self, op, ins, attrs=None, dt=None, os=None, cv=None, elems=None):
        out = _OV(None, dt, os, cv, elems)
        n = _ON(op, ins, attrs or {}, out)
        out.node = n
        self.nodes.append(n)
        return out

    def const(self, tensor):
        cv = tensor.item() if tensor.dim() == 0 else None
        return self.op('Constant', [], {'value': tensor}, tensor.dtype, tuple(tensor.shape), cv=cv)

    def insert_before(self, node, new_nodes):
        i = self.nodes.index(node)
        self.nodes[i:i] = new_nodes


def _pyconst(v):
    if v.node is None or v.node.kind != 'prim::Constant':
        raise LegacyTranslateError('non-constant argument')
    return v.node.value


_UNARY_ONNX = {'exp': 'Exp', 'log': 'Log', 'sqrt': 'Sqrt', 'sin': 'Sin', 'cos': 'Cos', 'tan': 'Tan', 'tanh': 'Tanh',
               'abs': 'Abs', 'neg': 'Neg', 'atan': 'Atan'}
_BINARY_ONNX = {'add': 'Add', 'sub': 'Sub', 'mul': 'Mul', 'div': 'Div', 'pow': 'Pow'}


def _convert(tr, nodes, outputs, in_ov):
    g = _Onnx()
    m = {}
    named = []  # onnx values that carry a tracer name, in graph order
    for v, ov in zip(tr.inputs, in_ov):
        m[v.idx] = ov

    def T(v):
        r = m[v.idx]
        if not isinstance(r, _OV):
            raise LegacyTranslateError('expected a tensor value')
        return r

    def i64(vals):
        return torch.tensor(vals, dtype=torch.long)

    def unsq(x, axes):
        c = g.const(i64(axes))
        os = list(x.os)
        for a in axes:
            os.insert(a if a >= 0 else len(os) + a + 1, 1)
        return g.op('Unsqueeze', [x, c], dt=x.dt, os=tuple(os))

    def shape_of_list(ov):
        return tuple(e.cv for e in ov.elems)

    for n in nodes:
        k = n.kind
        o = n.outs[0]
        if k == 'prim::Constant':
            val = n.value
            if o.ty in ('int', 'bool', 'float'):
                dt = {'int': torch.long, 'bool': torch.bool, 'float': torch.get_default_dtype()}[o.ty]
                m[o.idx] = g.const(torch.tensor(val, dtype=dt))
            elif o.ty == 'ilist':
                m[o.idx] = ('py', val, g.const(torch.tensor(list(val), dtype=torch.long)))
            else:
                m[o.idx] = ('py', val)
            continue
        if k == 'prim::ListConstruct':
            if o.ty == 'tlist':
                m[o.idx] = [T(v) for v in n.ins]
            else:
                elems = [T(v) for v in n.ins]
                m[o.idx] = g.op('prim::ListConstruct', elems, dt=torch.long, os=(len(elems),), elems=elems)
            continue
        if k == 'aten::size':
            x, dim = T(n.ins[0]), _pyconst(n.ins[1])
            if dim < 0:
                dim += len(x.os)
            if x.os[dim] is not None:
                r = g.const(torch.tensor(x.os[dim], dtype=torch.long))
            else:
                sh = g.op('Shape', [x], dt=torch.long, os=(len(x.os),))
                idx = g.const(torch.tensor(dim, dtype=torch.long))
                r = g.op('Gather', [sh, idx], {'axis': 0}, dt=torch.long, os=())
            m[o.idx] = r
            continue
        if k == 'aten::index_select':
            x, dim, idx = T(n.ins[0]), _pyconst(n.ins[1]), T(n.ins[2])
            d = dim if dim >= 0 else dim + len(x.os)
            r = g.op('Gather', [x, idx], {'axis': dim}, dt=x.dt, os=x.os[:d] + idx.os + x.os[d + 1:])
        elif k in ('aten::add', 'aten::sub', 'aten::mul', 'aten::div', 'aten::pow'):
            a, b = T(n.ins[0]), T(n.ins[1])
            # a python float operand is re-created as a tensor of the other operand's dtype, next to its consumer
            if n.ins[1].ty == 'float':
                b = g.const(torch.tensor(b.cv, dtype=a.dt))
            elif n.ins[0].ty == 'float':
                a = g.const(torch.tensor(a.cv, dtype=b.dt))
            if a.dt != b.dt:
                raise LegacyTranslateError(f'implicit dtype promotion in {k}: {a.dt} vs {b.dt}')
            if len(n.ins) == 3 and _pyconst(n.ins[2]) != 1:
                raise LegacyTranslateError('alpha != 1')
            r = g.op(_BINARY_ONNX[k[6:]], [a, b], dt=a.dt, os=_bcast(a.os, b.os))
        elif k[6:] in _UNARY_ONNX and k.startswith('aten::'):
            a = T(n.ins[0])
            r = g.op(_UNARY_ONNX[k[6:]], [a], dt=a.dt, os=a.os)
        elif k == 'aten::t':
            a = T(n.ins[0])
            if len(a.os) == 2:
                r = g.op('Transpose', [a], {'perm': [1, 0]}, dt=a.dt, os=a.os[::-1])
            else:
                r = a
        elif k == 'aten::matmul':
            a, b = T(n.ins[0]), T(n.ins[1])
            if len(b.os) != 2 or len(a.os) < 2:
                raise LegacyTranslateError('matmul ranks')
            r = g.op('MatMul', [a, b], dt=a.dt, os=a.os[:-1] + b.os[-1:])
        elif k in ('aten::view', 'aten::reshape'):
            a = T(n.ins[0])
            sz = m[n.ins[1].idx]
            if isinstance(sz, tuple):
                lst = list(sz[1])
                c = g.const(i64(lst))
                os = list(lst)
                if -1 in os:
                    if all(d is not None for d in a.os):
                        tot = 1
                        for d in a.os:
                            tot *= d
                        rest = 1
                        for d in os:
                            if d != -1:
                                rest *= d
                        os[os.index(-1)] = tot // rest
                    else:
                        os[os.index(-1)] = None
                r = g.op('Reshape', [a, c], {'allowzero': 0}, dt=a.dt, os=tuple(os))
            else:
                r = g.op('Reshape', [a, sz], {'allowzero': 0}, dt=a.dt, os=shape_of_list(sz))
        elif k == 'aten::unsqueeze':
            r = unsq(T(n.ins[0]), [_pyconst(n.ins[1])])
        elif k == 'aten::slice':
            x = T(n.ins[0])
            dim, st, en, step = (T(v) for v in n.ins[1:])
            if st.cv == 0 and en.cv == INT_MAX and step.cv == 1:
                r = x
            else:
                parts = [unsq(v, [0]) for v in (dim, st, en, step)]
                axes, starts, ends, steps = parts
                d = dim.cv if dim.cv >= 0 else dim.cv + len(x.os)
                os = list(x.os)
                if x.os[d] is not None:
                    os[d] = len(range(*slice(st.cv, None if en.cv == INT_MAX else en.cv, step.cv).indices(x.os[d])))
                r = g.op('Slice', [x, starts, ends, axes, steps], dt=x.dt, os=tuple(os))
        elif k == 'aten::cat':
            items, dim = m[n.ins[0].idx], _pyconst(n.ins[1])
            d = dim if dim >= 0 else dim + len(items[0].os)
            os = list(items[0].os)
            os[d] = None if any(t.os[d] is None for t in items) else sum(t.os[d] for t in items)
            r = g.op('Concat', items, {'axis': dim}, dt=items[0].dt, os=tuple(os))
        elif k == 'aten::sum':
            a = T(n.ins[0])
            dl, keep = m[n.ins[1].idx], _pyconst(n.ins[2])
            if not isinstance(dl, tuple):
                raise LegacyTranslateError('sum with traced dims')
            dims, c = list(dl[1]), dl[2]
            if _pyconst(n.ins[3]) is not None:
                raise LegacyTranslateError('sum dtype')
            os = list(a.os)
            for d in sorted((d if d >= 0 else d + len(os) for d in dims), reverse=True):
                if keep:
                    os[d] = 1
                else:
                    del os[d]
            r = g.op('ReduceSum', [a, c], {'keepdims': int(keep)}, dt=a.dt, os=tuple(os))
        elif k in ('aten::min', 'aten::max'):
            a, dimc = T(n.ins[0]), T(n.ins[1])
            os = list(a.os)
            d = dimc.cv if dimc.cv >= 0 else dimc.cv + len(os)
            if _pyconst(n.ins[2]):
                os[d] = 1
            else:
                del os[d]
            axes = g.const(i64([dimc.cv]))
            r = g.op('Reduce' + k[6:].capitalize(), [a, axes], {'keepdims': int(_pyconst(n.ins[2]))}, dt=a.dt,
                     os=tuple(os))
        elif k == 'aten::mean':
            a = T(n.ins[0])
            dl, keep = m[n.ins[1].idx], _pyconst(n.ins[2])
            if not isinstance(dl, tuple) or _pyconst(n.ins[3]) is not None:
                raise LegacyTranslateError('mean arguments')
            os = list(a.os)
            for d in sorted((d if d >= 0 else d + len(os) for d in dl[1]), reverse=True):
                if keep:
                    os[d] = 1
                else:
                    del os[d]
            r = g.op('ReduceMean', [a, dl[2]], {'keepdims': int(keep)}, dt=a.dt, os=tuple(os))
        elif k == 'aten::stack':
            items, dim = m[n.ins[0].idx], _pyconst(n.ins[1])
            us = [unsq(t, [dim]) for t in items]
            os = list(us[0].os)
            os[dim if dim >= 0 else dim + len(os)] = len(us)
            r = g.op('Concat', us, {'axis': dim}, dt=items[0].dt, os=tuple(os))
        elif k == 'aten::zeros':
            sz = m[n.ins[0].idx]
            if not isinstance(sz, _OV):
                raise LegacyTranslateError('zeros with constant sizes')
            dt = _CODE_DTYPE[_pyconst(n.ins[1])]
            r = g.op('ConstantOfShape', [sz], {'value': torch.tensor([0], dtype=dt)}, dt=dt, os=shape_of_list(sz))
        elif k == 'aten::zeros_like':
            a = T(n.ins[0])
            sh = g.op('Shape', [a], dt=torch.long, os=(len(a.os),))
            r = g.op('ConstantOfShape', [sh], {'value': torch.tensor([0], dtype=a.dt)}, dt=a.dt, os=a.os)
        elif k == 'aten::to':
            a = T(n.ins[0])
            dt = _CODE_DTYPE[_pyconst(n.ins[2])]
            r = g.op('Cast', [a], {'to': _ONNX_DTYPE[dt]}, dt=dt, os=a.os)
        elif k == 'aten::expand':
            a = T(n.ins[0])
            sz = m[n.ins[1].idx]
            if isinstance(sz, tuple):
                size = g.const(i64(list(sz[1])))
                dims = tuple(sz[1])
            else:
                dims = shape_of_list(sz)
                us = [unsq(e, [0]) for e in sz.elems]
                cc = g.op('Concat', us, {'axis': 0}, dt=torch.long, os=(len(us),))
                g.const(i64([-1]))
                size = g.op('Reshape', [cc, g.const(i64([-1]))], {'allowzero': 0}, dt=torch.long, os=(len(us),))
            shp = g.op('Shape', [size], dt=torch.long, os=(1,))
            ones = g.op('ConstantOfShape', [shp], {'value': torch.tensor([1], dtype=torch.long)}, dt=torch.long,
                        os=size.os)
            negc = g.const(torch.tensor(-1, dtype=torch.long))
            neg = g.op('Mul', [ones, negc], dt=torch.long, os=size.os)
            eq = g.op('Equal', [size, neg], dt=torch.bool, os=size.os)
            wh = g.op('Where', [eq, ones, size], dt=torch.long, os=size.os)
            r = g.op('Expand', [a, wh], dt=a.dt, os=_bcast(a.os, dims))
        elif k == 'aten::index_add':
            s, dm, idx, src, al = (T(v) for v in n.ins)
            r = g.op('aten::ATen', [s, dm, idx, src, al], {'operator': 'index_add', 'overload_name': ''},
                     dt=s.dt, os=s.os)
        elif k == 'aten::linear':
            x, w, b = (T(v) for v in n.ins)
            if len(x.os) != 2:
                raise LegacyTranslateError('linear on non-2D input')
            r = g.op('Gemm', [x, w, b], {'alpha': 1.0, 'beta': 1.0, 'transB': 1}, dt=x.dt, os=(x.os[0], w.os[0]))
        else:
            raise LegacyTranslateError(f'no ONNX rule for {k}')
        m[o.idx] = r
        if tr.names.get(o.idx) and isinstance(r, _OV) and r.node is not None and r.node.out is r:
            named.append((r, _base(tr.names.get(o.idx))))
    return g, [T(v) for v in outputs], named


def _erase_list_constructs(g):
    """``_jit_pass_onnx_peephole`` eraseListConstruct: a packed list of scalars feeding an op becomes
    Unsqueeze/Concat nodes inserted right before that op."""
    for n in list(g.nodes):
        if n.op == 'prim::ListConstruct':
            continue
        for i, v in enumerate(n.ins):
            if v.node is not None and v.node.op == 'prim::ListConstruct':
                lc = v.node
                new = []
                parts = []
                for e in v.node.ins:
                    c = _OV(None, torch.long, (1,))
                    cn = _ON('Constant', [], {'value': torch.tensor([0], dtype=torch.long)}, c)
                    c.node = cn
                    u = _OV(None, torch.long, (1,))
                    un = _ON('Unsqueeze', [e, c], {}, u)
                    u.node = un
                    new += [cn, un]
                    parts.append(u)
                cc = _OV(None, torch.long, (len(parts),), elems=v.elems)
                ccn = _ON('Concat', parts, {'axis': 0}, cc)
                cc.node = ccn
                new.append(ccn)
                g.insert_before(lc, new)
                n.ins[i] = cc


def _onnx_dce(g, outs):
    live = {id(o) for o in outs}
    keep = []
    for n in reversed(g.nodes):
        if id(n.out) in live:
            keep.append(n)
            live.update(id(v) for v in n.ins)
    keep.reverse()
    g.nodes = keep


# ---------------------------------------------------------------------------------------------------------
# Class emitters: hand-written replays of the forward methods (modules.py of gams/minlplib/minlp_dev, the
# minlplib_linear residual wrappers, and the output wrappers of api.py)
# ---------------------------------------------------------------------------------------------------------
class _Ctx:
    def __init__(self, tr, bufs):
        self.tr, self.bufs = tr, bufs

    def B(self, tensor):
        try:
            return self.bufs[id(tensor)]
        except KeyError:
            raise LegacyTranslateError('tensor that is not in the state dict') from None


def _call(m, x, c):
    name = type(m).__name__
    fn = _EMIT.get(name)
    if fn is None:
        if hasattr(m, '_root') and name.endswith('Model'):
            return _call(m._root, x, c)
        raise LegacyTranslateError(f'no emitter for {name}')
    return fn(m, x, c)


def _e_gather(m, x, c):
    return c.tr.index_select(x, -1, c.B(m.indices))


def _e_constvec(m, x, c):
    return c.B(m.values)


def _e_scalar(m, x, c):
    return c.B(m.value)


def _e_unary(m, x, c):
    v = _call(m.operand_module, x, c)
    if m.operation not in _UNARY_ONNX:
        raise LegacyTranslateError(f'unary op {m.operation}')
    return c.tr.unary(m.operation, v)


def _e_binary(m, x, c):
    tr = c.tr
    l = _call(m.left_module, x, c)
    r = _call(m.right_module, x, c)
    op = m.operation
    const_left = type(m.left_module).__name__ == 'ConstantVectorModule'
    if op == 'add':
        return tr.add_t(r, l) if const_left else tr.add_t(l, r)
    if op == 'sub':
        return tr.sub(l, r)
    if op == 'mul':
        return tr.mul(r, l) if const_left else tr.mul(l, r)
    if op == 'div':
        return tr.div(l, r)
    if op == 'pow':
        return tr.pow(l, r)
    raise LegacyTranslateError(f'binary op {op}')


def _e_scale(m, x, c):
    tr = c.tr
    v = _call(m.vector_module, x, c)
    s = c.B(m.scale)
    if v.t.dim() >= 1:
        for _ in range(v.t.dim() - 1):
            s = tr.unsqueeze(s, 0)
        return tr.mul(v, s)
    return tr.mul(v, s)


def _e_reduce(m, x, c):
    v = _call(m.vector_module, x, c)
    if m.op == 'sum':
        return c.tr.sum(v, [-1], True)
    if m.op in ('min', 'max'):
        return c.tr.reduce_dim(m.op, v, -1, True)
    if m.op == 'mean':
        return c.tr.mean(v, [-1], True)
    raise LegacyTranslateError(f'reduce op {m.op}')


def _e_min_module(m, x, c):
    tr = c.tr
    results = [_call(mod, x, c) for mod in m.arg_modules]
    return tr.reduce_dim('min', tr.stack(results, -1), -1, False)


def _e_concat_modules(m, x, c):
    tr = c.tr
    outs = []
    for mod in m.modules_list:
        y = _call(mod, x, c)
        if y.t.dim() == 0:
            y = tr.view(y, [1])
        if y.t.dim() == 1:
            y = tr.unsqueeze(y, -1)
        outs.append(y)
    return tr.cat(outs, -1)


def _e_vsum(m, x, c):
    tr = c.tr
    s = cc = None
    for mod in m.modules_list:
        y = _call(mod, x, c)
        if s is None:
            s = y
            cc = tr.zeros_like(y)
            continue
        y_k = tr.sub(y, cc)
        t = tr.add_t(s, y_k)
        cc = tr.sub(tr.sub(t, s), y_k)
        s = t
    return s


def _e_slice(m, x, c):
    return c.tr.slice(x, x.t.dim() - 1, m.start, m.end)


def _e_shared_linear(m, x, c):
    tr = c.tr
    v = x if m.straight_forward else tr.index_select(x, -1, c.B(m.indices))
    coef = c.B(m.coef_matrix)
    if v.t.dtype != coef.t.dtype:
        v = tr.to(v, coef.t.dtype)
    y = tr.matmul(v, tr.t(coef))
    bias = c.B(m.bias)
    if bool(bias.t.numel() > 0 and (bias.t != 0).any().item()):
        return tr.add_t(y, bias)
    return y


def _e_single_core(m, x, c):
    lin = _call(m.shared_linear, x, c)
    return _call(m.expr_over_lin, lin, c)


def _e_augment(m, x, c):
    lin = _call(m.shared_linear, x, c)
    aug = c.tr.cat([x, lin], -1)
    return _call(m.inner, aug, c)


def _e_batched_affine(m, x, c):
    tr = c.tr
    batch = tr.shape(x, None)[:-1]
    gathered = tr.index_select(x, -1, c.B(m.idx_flat))
    wts = c.B(m.weights)
    if m.w == 1:
        gathered = tr.view(gathered, [*batch, m.n])
        w = tr.view(wts, [1] * len(batch) + [m.n])
        out = tr.mul(gathered, w)
    else:
        gathered = tr.view(gathered, [*batch, m.n, m.w])
        w = tr.view(wts, [1] * len(batch) + [m.n, m.w])
        out = tr.sum(tr.mul(gathered, w), [-1], False)
    bias = c.B(m.bias)
    if bool(bias.t.numel() > 0 and (bias.t != 0).any().item()):
        out = tr.add_t(out, bias)
    return out


def _e_structural_wrapper(m, x, c):
    tr = c.tr
    base = _call(m.base_model, x, c)
    if base.t.dim() != 2:
        raise LegacyTranslateError('structural wrapper expects 2-D output')
    sh = tr.shape(x, None)
    x_flat = tr.reshape(x, [sh[0], -1])
    A, b = c.B(m.constraint_A), c.B(m.constraint_b)
    nb = tr.unary('neg', b)
    st = tr.linear(x_flat, A, nb)
    a = tr.slice(tr.slice(base, 0, 0, INT_MAX), 1, 0, 1)
    z = tr.slice(tr.slice(base, 0, 0, INT_MAX), 1, 1, INT_MAX)
    return tr.cat([a, st, z], 1)


def _e_objective_wrapper(m, x, c):
    tr = c.tr
    base = _call(m.base_model, x, c)
    if base.t.dim() != 2:
        raise LegacyTranslateError('objective wrapper expects 2-D output')
    row = tr.unsqueeze(c.B(m.objective_row), 0)
    obj = tr.sum(tr.mul(base, row), [1], True)
    return tr.cat([obj, base], 1)


def _as_last_dim_vector(tr, v, batch_shape, fr):
    """minlplib_linear ``_as_last_dim_vector``: a helper function, so its traced ops name the caller's local."""
    if v.t.dim() == 0:
        v = tr.reshape(v, [1], fr)
    if v.t.dim() == 1 and len(batch_shape) > 0:
        v = tr.expand(v, [*batch_shape, tr.shape(v, fr)[-1]], fr)
    return v


def _broadcast_scalar_last_dim(tr, v, size, fr):
    tr.touch(v, fr)  # ``int(v.shape[-1])``
    if int(v.t.shape[-1]) == 1 and int(size) != 1:
        return tr.expand(v, [*tr.shape(v, fr)[:-1], int(size)], fr)
    return v


def _e_coef_scale(m, x, c):
    tr = c.tr
    fr = OrderedDict(x=x)
    v = fr['v'] = _call(m.bucket_module, x, c)
    v = fr['v'] = _as_last_dim_vector(tr, v, tr.shape(x)[:-1], fr)
    v = fr['v'] = _broadcast_scalar_last_dim(tr, v, m.n, fr)
    w = tr.view(c.B(m.coef), [1] * (v.t.dim() - 1) + [m.n])
    return tr.mul(v, w)


def _e_lift_compact(m, x, c):
    if hasattr(m, 'lift_matrix'):
        raise LegacyTranslateError('minlp_dev LiftCompactToFullModule is outside the 505 vocabulary')
    tr = c.tr
    fr = OrderedDict(x=x)
    v = fr['v'] = _call(m.child_module, x, c)
    v = fr['v'] = _as_last_dim_vector(tr, v, tr.shape(x)[:-1], fr)
    v = fr['v'] = _broadcast_scalar_last_dim(tr, v, int(c.B(m.indices).t.numel()), fr)
    out = tr.zeros([*tr.shape(x)[:-1], m.total_outputs], v.t.dtype)
    return tr.index_add(out, out.t.dim() - 1, c.B(m.indices), v)


def _e_general_residual(m, x, c):
    if not hasattr(m, '_has_indices'):
        raise LegacyTranslateError('minlp_dev GeneralResidualModule is outside the 505 vocabulary')
    tr = c.tr
    fr = OrderedDict(x=x)
    y = fr['y'] = tr.zeros([*tr.shape(x)[:-1], m.total_outputs], m.dtype)
    for i, module in enumerate(m.input_modules):
        fr['i'], fr['module'] = i, module
        v = fr['v'] = _call(module, x, c)
        v = _as_last_dim_vector(tr, v, tr.shape(x)[:-1], fr)
        v = fr['v'] = tr.to(v, y.t.dtype)
        if m._has_indices[i]:
            idx = fr['idx'] = c.B(getattr(m, f'_idx_{i}'))
            v = fr['v'] = _broadcast_scalar_last_dim(tr, v, int(idx.t.numel()), fr)
            y = fr['y'] = tr.index_add(y, y.t.dim() - 1, idx, v)
        else:
            y = fr['y'] = tr.add_t(y, v)
    return y


def _e_residual_bucket(m, x, c):
    tr = c.tr
    v = _call(m.bucket_module, x, c)
    batch = tr.shape(x)[:-1]
    if v.t.dim() == 0:
        v = tr.reshape(v, [1])
    if v.t.dim() == 1 and len(batch) > 0:
        v = tr.expand(v, [*batch, tr.shape(v)[-1]])
    if int(v.t.shape[-1]) == 1 and m.n != 1:
        v = tr.expand(v, [*tr.shape(v)[:-1], m.n])
    if int(v.t.shape[-1]) != m.n:
        raise LegacyTranslateError('ResidualBucketScatterAddModule width mismatch')
    w = c.B(m.coef)
    all_ones = bool((w.t == 1).all().item()) if m.n > 0 else True
    scalar = False
    coef_scalar = 1.0
    if (not all_ones) and m.n > 0:
        tr.name_input(w, 'input')  # torch.unique(w): its helper frame holds the tensor as ``input``
        uniq = torch.unique(w.t)
        if int(uniq.numel()) == 1:
            scalar = True
            coef_scalar = float(uniq[0].item())
    if all_ones:
        return v
    if scalar:
        return tr.mul_scalar(v, coef_scalar)
    return tr.mul(v, tr.view(w, [1] * (v.t.dim() - 1) + [m.n]))


def _e_cross_formula(m, x, c):
    tr = c.tr
    if m.use_scatter:
        if m.shared_linear is None:
            raise LegacyTranslateError('shared linear should not be none')
        y = _call(m.shared_linear, x, c)
        contribs = [_call(mod, x, c) for mod in m.residual_module]
        contribs = tr.cat(contribs, 1)
        tr.index_add(y, 1, c.B(m.idxs), contribs, inplace=True)
        return y
    if m.shared_linear is not None:
        y = _call(m.shared_linear, x, c)
    else:
        y = tr.zeros([*tr.shape(x)[:-1], m.total_outputs], m.dtype)
    for mod in m.residual_module:
        tr.add_inplace(y, _call(mod, x, c))
    return y


def _e_combined_ast(m, x, c):
    tr = c.tr
    parts = []
    for mid in m.needed:
        y = _call(m.ast_modules[mid], x, c)
        if y.t.dim() == 0:
            y = tr.view(y, [1])
        if y.t.dim() == 1:
            y = tr.unsqueeze(y, -1)
        k = int(m.mod_dim[mid])
        if y.t.shape[-1] > k:
            y = tr.slice(y, y.t.dim() - 1, 0, k)
        elif y.t.shape[-1] < k:
            pad = tr.zeros([*tr.shape(y)[:-1], k - int(y.t.shape[-1])], y.t.dtype)
            y = tr.cat([y, pad], -1)
        parts.append(y)
    if len(parts) == 0:
        packed = tr.zeros([*tr.shape(x)[:-1], 0], x.t.dtype)
    elif len(parts) == 1:
        packed = parts[0]
    else:
        packed = tr.cat(parts, -1)
    if m.has_missing:
        z = tr.zeros([*tr.shape(x)[:-1], 1], x.t.dtype)
        packed = tr.cat([packed, z], -1)
    if m.identity_select:
        return packed
    return tr.index_select(packed, -1, c.B(m.select_indices))


_EMIT = {
    'VariableVectorGatherModule': _e_gather,
    'ConstantVectorModule': _e_constvec,
    'ScalarConstantModule': _e_scalar,
    'VectorUnaryOpModule': _e_unary,
    'VectorBinaryOpModule': _e_binary,
    'VectorScaleModule': _e_scale,
    'VectorReduceModule': _e_reduce,
    'VectorSumModule': _e_vsum,
    'MinModule': _e_min_module,
    'ConcatModules': _e_concat_modules,
    'LastDimSliceModule': _e_slice,
    'SharedLinearFormModule': _e_shared_linear,
    'SharedLinearSingleCoreModule': _e_single_core,
    'SharedLinearAugmentModule': _e_augment,
    'BatchedAffineGatherModule': _e_batched_affine,
    'ResidualCoefScaleModule': _e_coef_scale,
    'LiftCompactToFullModule': _e_lift_compact,
    'GeneralResidualModule': _e_general_residual,
    'ResidualBucketScatterAddModule': _e_residual_bucket,
    'CrossFormulaLinearResidualVectorModule': _e_cross_formula,
    'CombinedASTModule': _e_combined_ast,
    '_StructuralConstraintOutputWrapper': _e_structural_wrapper,
    '_ObjectiveOutputWrapper': _e_objective_wrapper,
}


# ---------------------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------------------
def _unique_state(module, param_exclude, param_include):
    sd = module.state_dict(keep_vars=True)
    seen, out = set(), OrderedDict()
    for k, v in sd.items():
        if id(v) in seen:
            continue
        seen.add(id(v))
        out[k] = v
    ex = re.compile(param_exclude) if param_exclude is not None else None
    inc = re.compile(param_include) if param_include is not None else None
    res = OrderedDict()
    for k, v in out.items():
        if ex is not None and ex.match(k) is not None:
            continue
        if inc is not None and inc.match(k) is None:
            continue
        res[k] = v
    return res


def translate_module(module, inputs, param_exclude='.*AuxLogits.*', param_include=None):
    """Same return value as ``auto_LiRPA.parse_graph.parse_module``: ``(nodesOP, nodesIn, nodesOut, template)``."""
    x_in = inputs[0] if isinstance(inputs, tuple) else inputs
    if isinstance(inputs, tuple) and len(inputs) != 1:
        raise LegacyTranslateError('one input tensor expected')
    if x_in.dim() < 2:
        raise LegacyTranslateError('the input needs a batch dimension')
    params = _unique_state(module, param_exclude, param_include)

    tr = _Tr()
    xt = tr.input(torch.zeros(x_in.shape, dtype=x_in.dtype))
    bufs = {}
    for k, p in params.items():
        bufs[id(p)] = tr.input(p.detach())
    in_ov = [_OV(None, x_in.dtype, (None,) + tuple(x_in.shape[1:]))]
    for k, p in params.items():
        in_ov.append(_OV(None, p.dtype, tuple(p.shape)))

    try:
        y = _call(module, xt, _Ctx(tr, bufs))
    except (RuntimeError, TypeError) as e:  # e.g. dtype mismatches between the input and the buffers
        raise LegacyTranslateError(f'{type(e).__name__}: {e}') from e
    outputs = [y.v]
    nodes = _jit_passes(tr, outputs)
    g, outs, named = _convert(tr, nodes, outputs, in_ov)
    _erase_list_constructs(g)
    _onnx_dce(g, outs)
    return _assemble(tr, g, outs, named, in_ov, x_in, params)


def _assemble(tr, g, outs, named, in_ov, x_in, params):
    from auto_LiRPA.bounded_tensor import BoundedTensor, BoundedParameter
    n_in = len(in_ov)
    names = _NameTable()
    for _ in range(n_in):
        names.new()
    for i, v in enumerate(tr.inputs):
        if tr.names.get(v.idx):
            names.set(i, tr.names.get(v.idx))
    idx_of = {id(ov): i for i, ov in enumerate(in_ov)}
    for n in g.nodes:
        idx_of[id(n.out)] = names.new()
    # the ONNX pass renames the tracer's named values; the first one of a base keeps the plain name
    first = {}
    for ov, b in named:
        if id(ov) not in idx_of:
            continue
        i = idx_of[id(ov)]
        names.set(i, b)
        if b in first:
            names.set(first[b], b)
        else:
            first[b] = i

    def nm(ov):
        i = idx_of[id(ov)]
        return ('/' + (names.get(i) or str(i))).replace('.', '-')

    used = set()
    for n in g.nodes:
        used.update(id(v) for v in n.ins)
    used.update(id(o) for o in outs)
    plist = list(params.items())
    nodes_in = []
    in_by_name = {}
    for i, ov in enumerate(in_ov):
        if id(ov) not in used:
            continue
        if i == 0:
            ptb = x_in.ptb if isinstance(x_in, BoundedTensor) else None
            node = Node(name=nm(ov), ori_name='input_0', op='Parameter', inputs=[], attr='Tensor', param=None,
                        model_input_index=0, perturbation=ptb)
        else:
            key, p = plist[i - 1]
            ptb = p.ptb if isinstance(p, BoundedParameter) else None
            node = Node(name=nm(ov), ori_name=key, op='Parameter', inputs=[], attr='Tensor', param=p,
                        model_input_index=None, perturbation=ptb)
        nodes_in.append(node)
        in_by_name[node.name] = node
    nodes_op = []
    for n in g.nodes:
        ins = [nm(v) for v in n.ins]
        param = OrderedDict()
        for nme in ins:
            if nme in in_by_name:
                param[nme] = in_by_name[nme].param
        op = n.op if '::' in n.op else 'onnx::' + n.op
        nodes_op.append(Node(name=nm(n.out), op=op, inputs=ins, attr=n.attrs, output_index=0, param=param))
    return nodes_op, nodes_in, [nm(o) for o in outs], None


def use_translator(model):
    """Opt in: ``BoundedModule(model, ...)`` takes the node tuples from this translator instead of tracing
    (``auto_LiRPA/bound_general.py`` ``_convert_nodes`` seam).  Default off; returns ``model``."""
    model.__gear_program__ = translate_module
    return model
