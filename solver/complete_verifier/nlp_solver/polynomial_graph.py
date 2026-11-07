"""Polynomial graph: the CPU lifted LP's rows inside CROWN's own computation graph (opt-in).

PolynomialGraph(net, C, thr, x_L, x_U, E, h, rational_rows) extends a copy of GEAR's BoundedModule `net` so that CROWN
on it (auto_LiRPA's, face_dual's, compiled_crown's) sees the rows the CPU lifted LP adds (lifted_lp.py; the step3 G3
table): the extension only rewires exact duplicates and adds nodes and output rows, so every bound on the extended
graph is a bound of the original problem. Nothing outside this module calls it; the original graph is never touched.

Columns. The inputs x and every element of every face node (compiled_crown.face_nodes: BoundMul with two perturbed
operands, the unary operators). One exact pass over the graph (_Fold, rational arithmetic on the graph's float constants,
no rounding) writes every element of every node as an affine form over the columns.

Shared products (exact CSE). A product element z = a(v) b(v) whose operands are, as exact affine forms over the columns,
s_a A(v) and s_b B(v) with A, B normalised (leading coefficient 1) has the key {A, B} (squares: BoundSqr(s_a A), key A).
Two products with one key are one product: z' = (s'_a s'_b) / (s_a s_b) z exactly (kappa); the later one is rewired to
kappa times the first (when kappa is a float, else it stays). Keys are taken in topological order over the forms of the
rewired graph, so the sharing is transitive. CROWN sees one node: the coefficients of every row reaching it add up
before it is relaxed. Same rule as the step3 G3 table's exact share map (exact operand forms instead of folded floats).

Monomials. x_i x_j is the product of key {x_i, x_j} ({x_i, x_i}, else a BoundSqr of x_i, for a square): a graph
product when one exists (at its scale), else a new node (one BoundMul of two gathers of x, one BoundSqr), McCormick /
the square's secant and tangent pool by the operators' own code, as every other product.

Derived rows, each an affine row F(v) <= 0 over the columns, valid at every point (x, v(x)) of the graph on the box
x in [x_L, x_U] that satisfies the original rows; proofs:
  link    a product z = a(x) b(x) of two input-affine operands (a = a0 + sum_i a_i x_i, b likewise), not itself a
          monomial: z - sum_ij a_i b_j x_i x_j - a0 sum_j b_j x_j - b0 sum_i a_i x_i - a0 b0 = 0 is the expansion, an
          identity of exact rationals (both signs: two rows). lifted_lp writes these rows' polynomials directly.
          A link is added only while the graph's columns stay within max_columns (below).
  rlt     reduced RLT, lifted_lp.add_rlt's rule: rows p, q with exactly opposite input-linear parts, g x <= t_hi and
          -g x <= -t_lo (spec rows at their relaxed thresholds, base rows), make the band t_lo <= g x <= t_hi; it is
          multiplied by x_i iff x_i x_j is a product of the graph (or a link monomial) for some j in supp(g).
          P = sum_j g_j x_i x_j = x_i (g x) exactly, and on the box (x_i in [l_i, u_i]):
            l_i >= 0:   x_i (t_hi - g x) >= 0, x_i (g x - t_lo) >= 0:   P - t_hi x_i <= 0,  t_lo x_i - P <= 0
            u_i <= 0:   the factors' signs swap:                          P - t_lo x_i <= 0,  t_hi x_i - P <= 0
            otherwise:  x_i (g x - b) in [l_i, u_i] x [t_lo - b, t_hi - b] (b the midpoint): P - b x_i <= its interval
                        max, b x_i - P <= -(its interval min)
          (lifted_lp._rlt_block, box by box). A multiplier of fixed sign on the root box keeps it on every sub-box:
          two rows. One that straddles 0 on the root box gets all six; on each box the rows of its class carry the
          rule above, the others a threshold no lower than their left side's maximum over the column boxes (the
          inputs' box, the monomials' corner products), which no column box point violates: the LP is the CPU's.
          Caps (lifted_lp.add_rlt's, all or nothing): 2 x (multiplier, equality) pairs <= max_rows and n + the
          graph's product columns (with the new monomials) <= max_columns, so the graph never holds more RLT than the
          CPU lifted LP.
  cleared rational_rows' sign-proven denominator clearing: a spec row r(x) <= thr_r (relaxed threshold) whose exact
          Laurent form has the denominator monomial m(x) = prod_i x_i^k_i with every l_i > 0 on the root box (so on
          every sub-box): m(x) > 0, hence m(x) (r(x) - thr_r) <= 0, a polynomial of degree <= 2 in exact rationals
          (rational_rows.clear_positive_denominators); its monomials map to columns (each row within max_columns).
Every coefficient is the exact rational of the row; the graph holds its nearest float W_c and the bias W_0. At any point
of the box, sum_c W_c v_c + W_0 = F(v) - sum_c (q_c - W_c) v_c - (q_0 - W_0) <= sum_c |q_c - W_c| M_c + |q_0 - W_0|,
M_c >= |v_c| on the box (inputs: max(|l|, |u|); a monomial at scale s: |s| M_i M_j); thresholds() returns that bound
(and the RLT interval terms) rounded outward, per box: rows 'F(v) <= 0' with threshold 0 up to the coefficients'
rounding.

The extended graph: the derived rows' columns gathered into one vector V, D = V W + W_0 (one MatMul), and the new final
node concat(original outputs, D); the spec C_ext = blockdiag(C, I): rows S + k are the derived rows. The rows were
derived from `thr` and the base rows given here: they stay valid while those thresholds do (GEAR's relaxed thresholds
are fixed; the objective's cutoff, row 0, is not used).
"""
import copy
import math
import types
from fractions import Fraction

import numpy as np
import torch

from auto_LiRPA.bound_ops import (BoundAdd, BoundConcat, BoundConstant, BoundGather, BoundMatMul, BoundMul,
                                  BoundReshape, BoundSqr)
from auto_LiRPA.cap_log import cap_log
from auto_LiRPA.optimize_graph import _copy_node_properties

from .structure.forms import exact


def face_nodes(net):
    """the perturbed nonlinear nodes of `net` in net.nodes() order, except the linear family (they have no faces)"""
    from auto_LiRPA.operators import BoundActivation, BoundLinear, BoundMul

    def nonlinear(node):
        if isinstance(node, BoundMul):
            return not node.is_linear_op
        if isinstance(node, BoundLinear):
            return sum(i.perturbed for i in node.inputs[:2]) > 1
        return isinstance(node, BoundActivation)
    return [node.name for node in net.nodes() if node.perturbed and nonlinear(node) and not isinstance(node, BoundLinear)]

_MOVE = {'BoundGather', 'BoundTranspose', 'BoundReshape', 'BoundConcat', 'BoundUnsqueeze', 'BoundSqueeze',
         'BoundSlice', 'BoundExpand', 'BoundFlatten', 'BoundIdentity', 'BoundSplit', 'BoundGatherElements'}
U64 = 2.0 ** -53
# lifted_lp.add_rlt's caps: rows of the RLT block, and columns of the lifted LP (inputs + products)
MAX_ROWS = 1200
MAX_COLUMNS = 7000


def _ex(x):
    """the exact rational of a number (a float keeps its binary value; forms.exact)"""
    if isinstance(x, (bool, np.bool_, np.integer)):
        x = int(x)
    return exact(x if isinstance(x, (int, Fraction)) else float(x))


class Aff:
    """sum_c t[c] v_c + k over the columns, exact (int / Fraction); t None: not affine over the columns (unknown)"""
    __slots__ = ('t', 'k')

    def __init__(self, t, k=0):
        self.t, self.k = t, k

    def scaled(self, c):
        if self.t is None:
            return self
        if c == 0:
            return Aff({}, 0)
        return Aff({j: v * c for j, v in self.t.items()}, self.k * c)

    def __add__(self, o):
        if not isinstance(o, Aff):
            return self if self.t is None else Aff(self.t, self.k + _ex(o))
        if self.t is None or o.t is None:
            return UNKNOWN
        a, b = (self.t, o.t) if len(self.t) >= len(o.t) else (o.t, self.t)
        t = dict(a)
        for j, v in b.items():
            s = t.get(j, 0) + v
            if s:
                t[j] = s
            else:
                t.pop(j, None)
        return Aff(t, self.k + o.k)

    __radd__ = __add__

    def __neg__(self):
        return self.scaled(-1)

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, Aff):
            if o.t == {}:
                return self.scaled(o.k)
            if self.t == {}:
                return o.scaled(self.k)
            return UNKNOWN
        return self.scaled(_ex(o))

    __rmul__ = __mul__

    def __truediv__(self, o):
        c = o.k if isinstance(o, Aff) and o.t == {} else (None if isinstance(o, Aff) else _ex(o))
        return UNKNOWN if c is None or c == 0 else self.scaled(Fraction(1) / c)


UNKNOWN = Aff(None)


def _canon(f):
    """(key, s): f = s * (the form of key), key = (columns, coefficients / s, constant / s); None for an unknown or
    constant form"""
    if f.t is None or not f.t:
        return None, None
    items = sorted(f.t.items())
    s = items[0][1]
    return (tuple(j for j, _ in items), tuple(Fraction(v) / s for _, v in items), Fraction(f.k) / s), s


def _const_array(v):
    v = torch.as_tensor(v).detach().cpu()
    out = np.empty(v.numel(), dtype=object)
    for i, x in enumerate(v.reshape(-1).tolist()):
        out[i] = Aff({}, _ex(x))
    return out.reshape(tuple(v.shape))


def _matmul_const(a, W):
    """a (..., k) object array of Aff times the constant matrix W (k, m), over W's nonzeros"""
    W = torch.as_tensor(W).detach().cpu().double().numpy()
    a2 = a.reshape(-1, a.shape[-1])
    out = np.empty((a2.shape[0], W.shape[1]), dtype=object)
    cols = [[(k, _ex(W[k, j])) for k in np.flatnonzero(W[:, j])] for j in range(W.shape[1])]
    for r in range(a2.shape[0]):
        for j, terms in enumerate(cols):
            out[r, j] = sum((a2[r, k].scaled(w) for k, w in terms), Aff({}, 0))
    return out.reshape(*a.shape[:-1], W.shape[1])


def _postorder(roots):
    """the nodes reachable from roots through their inputs, every node after its inputs (iterative)"""
    order, seen = [], set()
    for root in roots:
        stack = [(root, 0)]
        while stack:
            node, state = stack.pop()
            if state == 0:
                if node.name in seen:
                    continue
                seen.add(node.name)
                stack.append((node, 1))
                stack.extend((i, 0) for i in reversed(node.inputs))
            else:
                order.append(node)
    return order


class _Fold:
    """Exact affine forms over the columns of every node element of `net`, the products interned by key (see the module
    docstring): forms {name: object ndarray of Aff or a constant tensor}, cols [(node name, flat element) or
    ('input', j)], alias {(node, element): (column, kappa)}, products {key: (column, scale)}, operands {column: (a, b)}
    (a product's exact operand forms)."""

    def __init__(self, net):
        self.net = net
        self.input = net[net.input_name[0]]
        self.n = int(self.input.forward_value.numel())
        self.cols = [('input', j) for j in range(self.n)]
        self.faces = set(face_nodes(net))
        self.forms, self.alias, self.products, self.operands, self.kind, self.col_index = {}, {}, {}, {}, {}, {}
        self.order = _postorder([net[net.final_name]])
        for node in self.order:
            self.forms[node.name] = self._node(node)

    def array(self, node):
        f = self.forms[node.name]
        return _const_array(f) if torch.is_tensor(f) else f

    def _new_col(self, name, e):
        self.cols.append((name, e))
        self.col_index[(name, e)] = len(self.cols) - 1
        return len(self.cols) - 1

    def key(self, kind, fa, fb=None):
        """(key, scale) of a product of the forms fa, fb (BoundSqr: fa twice), or (None, None)"""
        ka, sa = _canon(fa)
        if ka is None:
            return None, None
        if kind == 'sqr':
            return ('sqr', ka), sa * sa
        kb, sb = _canon(fb)
        if kb is None:
            return None, None
        return ('mul',) + tuple(sorted([ka, kb])), sa * sb

    def _face(self, node):
        shape = tuple(node.forward_value.shape)
        E = int(np.prod(shape, dtype=np.int64))
        out = np.empty(E, dtype=object)
        kind = 'mul' if isinstance(node, BoundMul) else 'sqr' if isinstance(node, BoundSqr) else None
        ops = [np.broadcast_to(self.array(i), shape).reshape(-1) for i in node.inputs[:2 if kind == 'mul' else 1]]
        for e in range(E):
            key, scale = (None, None) if kind is None else self.key(kind, ops[0][e], ops[-1][e])
            if key is not None and key in self.products:
                col, s = self.products[key]
                kappa = Fraction(scale) / s
                if Fraction(float(kappa)) == kappa:
                    out[e] = Aff({col: kappa})
                    self.alias[(node.name, e)] = (col, kappa)
                    continue
            col = self._new_col(node.name, e)
            self.kind[col] = kind
            if key is not None and key not in self.products:
                self.products[key] = (col, scale)
            if kind is not None:
                self.operands[col] = (ops[0][e], ops[-1][e])
            out[e] = Aff({col: 1})
        return out.reshape(shape)

    def _node(self, node):
        tn = type(node).__name__
        fv = node.forward_value
        # Shape depends on fixed tensor dimensions, never on input values.
        if tn == "BoundShape":
            return fv.detach().cpu()
        if node is self.input:
            out = np.empty(self.n, dtype=object)
            for j in range(self.n):
                out[j] = Aff({j: 1})
            return out.reshape(tuple(fv.shape))
        if not any(not torch.is_tensor(self.forms[i.name]) for i in node.inputs):
            if not torch.is_tensor(fv):
                fv = torch.as_tensor(fv)
            return fv.detach().cpu()                                    # a constant (input-independent) node
        shape = tuple(fv.shape)
        if node.name in self.faces:
            return self._face(node)
        try:
            out = self._affine(node, tn)
        except (ValueError, TypeError, IndexError, AttributeError, RuntimeError, ZeroDivisionError, KeyError):
            out = None
        if out is None:
            out = np.full(int(np.prod(shape, dtype=np.int64)), UNKNOWN, dtype=object)
        return np.asarray(out, dtype=object).reshape(shape)

    def _affine(self, node, tn):
        """the forms of an affine node, or None (unsupported: its elements are unknown)"""
        ins = node.inputs
        arr = lambda i: self.array(ins[i])
        dep = [not torch.is_tensor(self.forms[i.name]) for i in ins]
        if tn in ('BoundAdd', 'BoundSub'):
            return arr(0) + arr(1) if tn == 'BoundAdd' else arr(0) - arr(1)
        if tn == 'BoundNeg':
            return -arr(0)
        if tn in ('BoundMul', 'BoundDiv') and sum(dep[:2]) == 1:
            k = 0 if dep[0] else 1
            c = np.vectorize(_ex, otypes=[object])(self.forms[ins[1 - k].name].double().numpy())
            if tn == 'BoundDiv':
                if k != 0 or any(v == 0 for v in c.reshape(-1)):
                    return None
                c = np.vectorize(lambda v: Fraction(1) / v, otypes=[object])(c)
            return arr(k) * c
        if tn == 'BoundCast':
            src = ins[0].forward_value
            if torch.is_tensor(src) and src.is_floating_point() and node.forward_value.is_floating_point() and \
                    torch.finfo(src.dtype).bits <= torch.finfo(node.forward_value.dtype).bits:
                return arr(0)
            return None
        if tn == 'BoundMatMul' and sum(dep[:2]) == 1:
            if dep[0]:
                W = self.forms[ins[1].name]
                return _matmul_const(arr(0), W) if W.ndim == 2 else None
            W = self.forms[ins[0].name]
            if W.ndim != 2:
                return None
            a = arr(1)                                                  # W @ a = (a^T W^T)^T
            return np.swapaxes(_matmul_const(np.swapaxes(a, -1, -2), W.t()), -1, -2)
        if tn == 'BoundLinear' and dep[0] and not any(dep[1:]):
            if getattr(node, 'transA', 0):
                return None
            w = self.forms[ins[1].name]
            w = w if node.transB else w.t()                             # forward: x @ w^T, w (out, in)
            out = _matmul_const(arr(0) * _ex(node.alpha_linear), w.t())
            if len(ins) > 2:
                out = out + np.vectorize(_ex, otypes=[object])(self.forms[ins[2].name].double().numpy()) \
                    * _ex(node.beta_linear)
            return out
        if tn in ('BoundReduceSum', 'BoundReduceMean') and not any(dep[1:]):
            a = arr(0)
            axis = getattr(node, 'axis', None)
            axes = tuple(range(a.ndim)) if axis is None else tuple(
                k % a.ndim for k in (axis if isinstance(axis, (list, tuple)) else [axis]))
            out = a.sum(axis=axes, keepdims=bool(getattr(node, 'keepdim', False)))
            if tn == 'BoundReduceMean':
                out = out * Fraction(1, int(np.prod([a.shape[k] for k in axes])))
            return np.asarray(out, dtype=object)
        if tn == 'BoundATenIndex_add' and not any(dep[k] for k in (1, 2, 4)):
            data, src = arr(0), arr(3)
            axis = int(self.forms[ins[1].name]) % data.ndim
            target = [max(a, b) for a, b in zip(data.shape, src.shape)]
            dshape, sshape = list(target), list(target)
            dshape[axis], sshape[axis] = data.shape[axis], src.shape[axis]
            out = np.broadcast_to(data, dshape).copy()
            src = np.broadcast_to(src, sshape) * _ex(float(self.forms[ins[4].name]))
            for j, idx in enumerate(self.forms[ins[2].name].reshape(-1).tolist()):
                dst, sl = [slice(None)] * out.ndim, [slice(None)] * out.ndim
                dst[axis], sl[axis] = idx, j
                out[tuple(dst)] = out[tuple(dst)] + src[tuple(sl)]
            return out
        if tn == 'BoundWhere' and not dep[0]:
            return np.where(self.forms[ins[0].name].numpy().astype(bool), arr(1), arr(2))
        if tn in _MOVE:
            # data movement: run the operator's own forward on integer ids, never on the forms
            if tn != 'BoundConcat' and any(dep[1:]):
                return None
            table, args = [], []
            for i, inp in enumerate(ins):
                if tn != 'BoundConcat' and i > 0:
                    args.append(self.forms[inp.name])
                    continue
                a = self.array(inp)
                args.append(torch.arange(len(table), len(table) + a.size, dtype=torch.int64).reshape(a.shape))
                table.extend(a.reshape(-1))
            state = node.__dict__.copy()
            try:
                ids = node.forward(*args).detach().cpu().numpy()
            finally:
                node.__dict__.clear()
                node.__dict__.update(state)
            return np.array(table + [None], dtype=object)[:-1][ids]
        return None


def _copy_module(net):
    """copy.deepcopy of a BoundedModule whose caches hold tensor subclasses (BoundedTensor, BoundedParameter) that
    deepcopy cannot copy: those are copied as plain tensors (the copy's forward recomputes its values)"""
    memo, seen = {}, set()
    stack = [net]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, torch.Tensor):
            if isinstance(o, torch.nn.Parameter) and type(o) is not torch.nn.Parameter:
                memo[id(o)] = torch.nn.Parameter(o.detach().as_subclass(torch.Tensor).clone(), o.requires_grad)
            elif not isinstance(o, torch.nn.Parameter) and type(o) is not torch.Tensor:
                memo[id(o)] = o.detach().as_subclass(torch.Tensor).clone()
            continue
        if isinstance(o, dict):
            stack.extend(o.values())
        elif isinstance(o, (list, tuple, set)):
            stack.extend(o)
        elif hasattr(o, '__dict__') and not isinstance(o, (type, types.FunctionType, types.BuiltinFunctionType,
                                                            types.ModuleType)):
            stack.extend(vars(o).values())
    return copy.deepcopy(net, memo)


def _ell_columns(M):
    """the nonzeros of every column of M (U, K): row indices idx (K, L) and values val (K, L), L the longest column,
    padded with (0, 0.0)"""
    nz = M != 0
    L = max(1, int(nz.sum(0).max())) if M.numel() else 1
    idx = torch.zeros(M.shape[1], L, dtype=torch.long)
    val = torch.zeros(M.shape[1], L, dtype=M.dtype)
    for k in range(M.shape[1]):
        r = torch.nonzero(nz[:, k]).reshape(-1)
        idx[k, :len(r)], val[k, :len(r)] = r, M[r, k]
    return idx, val


class _Row:
    """a derived row F(v) = sum_c q[c] v_c + q0 <= threshold, exact; tag; the RLT rule (multiplier i, its class
    'pos' / 'neg' / 'mid', the row's sign side, t_lo, t_hi) or None"""
    __slots__ = ('q', 'q0', 'tag', 'rlt')

    def __init__(self, q, q0, tag, rlt=None):
        self.q, self.q0, self.tag, self.rlt = q, q0, tag, rlt


class PolynomialGraph:
    """The extended graph of one GEAR problem (see the module docstring).

    net: GEAR's BoundedModule (copied, never changed); C (S, n_out) the spec (row 0 the objective); thr (S,) the
    relaxed thresholds (row 0 unused); x_L, x_U (1, n) the root box (sign classes, positivity proofs and caps; it must
    be finite); E (R, n), h (R,) the base rows E x <= h; rational_rows: rational_rows.extract_rational_rows's forms of
    the spec rows (None: no cleared rows). links / rlt / cleared: the three row families.

    Attributes: net (the extended BoundedModule), C (S + K, n_out + K), rows (the derived rows), stats; methods
    thresholds(x_L, x_U) -> (B, S + K), interm_bounds(IB, x_L, x_U) (frozen intermediate bounds of the original graph
    carried to the extended one)."""

    def __init__(self, net, C, thr, x_L, x_U, E=None, h=None, rational_rows=None, links=True, rlt=True, cleared=True,
                 max_rows=MAX_ROWS, max_columns=MAX_COLUMNS, copy_net=True, squares=False, mixed_squares=False,
                 curvature_squares=False, curvature_leaf_names=()):
        self.net = _copy_module(net) if copy_net else net
        net = self.net
        self.squares_rewritten = 0
        ref = net[net.input_name[0]].forward_value
        self.dtype, self.device = ref.dtype, net.device
        lo, hi = x_L.reshape(-1).double().cpu(), x_U.reshape(-1).double().cpu()
        if not bool(torch.isfinite(lo).all() and torch.isfinite(hi).all()):
            raise ValueError('the root box must be finite')
        C = C.reshape(-1, C.shape[-1]).double().cpu()
        thr = thr.reshape(-1).double().cpu()
        self.thr_spec = thr
        self.S, self.n_out = C.shape
        net(*net.global_input)                  # every node's forward value (the caller's last pass may have cleared some)
        if squares or mixed_squares:
            # a product of two exactly equal operand forms is a square: BoundSqr (its tangent pool) in place of
            # BoundMul's McCormick faces (hs119's 27 squares), the operand the product's first
            self.squares_rewritten = self._rewrite_squares(mixed_squares)
            if self.squares_rewritten:
                net(*net.global_input)
        fold = _Fold(net)
        self.fold = fold
        n = self.n = fold.n
        self.mono_new = []                       # [(column, i, j)] new monomial nodes
        self.mono_of = {}                        # column -> (i, j, scale): x_i x_j = v_column / scale
        for key, (col, s) in fold.products.items():
            ij = self._single_pair(key)
            if ij is not None:
                self.mono_of[col] = (*ij, s)
        out = fold.array(net[net.final_name]).reshape(-1)
        self.rows = []
        self.stats = {'shared': len(fold.alias), 'products': sum(k is not None for k in fold.kind.values()),
                      'columns': len(fold.cols),
                      'mul_squares': sum(1 for key in fold.products if key[0] == 'mul' and key[1] == key[2])}
        spec = [sum((out[j] * _ex(float(C[r, j])) for j in torch.nonzero(C[r]).reshape(-1).tolist()), Aff({}, 0))
                for r in range(self.S)]
        self.curvature_leaves = set()
        self.curvature_leaves.update(fold.col_index[key] for key in curvature_leaf_names if key in fold.col_index)
        if curvature_squares and spec[0].t is not None:
            used = set().union(*(set(f.t) for i, f in enumerate(spec[1:], 1)
                                 if f.t is not None and math.isfinite(float(thr[i]))))
            for fa, fb in fold.operands.values():
                used.update(() if fa.t is None else fa.t)
                used.update(() if fb.t is None else fb.t)
            self.curvature_leaves.update(col for col, (fa, fb) in fold.operands.items()
                if col not in used and spec[0].t.get(col, 0) > 0
                and fa.t is not None and fa.t == fb.t and fa.k == fb.k)
        self.max_columns, self._device_data = max_columns, {}
        if links:
            self._links()
        if rlt:
            self._rlt(spec, thr, E, h, lo, hi, max_rows)
        if cleared and rational_rows is not None:
            self._cleared(rational_rows, thr, lo)
        self.stats.update(rows=len(self.rows), new_monomials=len(self.mono_new),
                          tags={t: sum(r.tag == t for r in self.rows) for t in sorted({r.tag for r in self.rows})})
        self._emit(C)

    # ------------------------------------------------------------------------------------------------ monomials
    def _single_pair(self, key):
        """(i, j) when key is the product of two single inputs x_i, x_j (coefficient 1 after normalisation)"""
        single = lambda k: k[0][0] if len(k[0]) == 1 and k[0][0] < self.n and k[2] == 0 else None
        if key[0] == 'sqr':
            i = single(key[1])
            return None if i is None else (i, i)
        i, j = single(key[1]), single(key[2])
        return None if i is None or j is None else (min(i, j), max(i, j))

    def _mono_key(self, i, j):
        x = lambda k: ((k,), (Fraction(1),), Fraction(0))
        return ('mul',) + tuple(sorted([x(i), x(j)]))

    def _mono(self, i, j, create=True):
        """(column, scale) with x_i x_j = v_column / scale; a new monomial column when none exists (create)"""
        P = self.fold.products
        for key in (self._mono_key(i, j),) + ((('sqr', ((i,), (Fraction(1),), Fraction(0))),) if i == j else ()):
            if key in P:
                return P[key]
        if not create:
            return None
        col = len(self.fold.cols)
        self.fold.cols.append(('mono', len(self.mono_new)))
        key = self._mono_key(i, j) if i != j else ('sqr', ((i,), (Fraction(1),), Fraction(0)))
        P[key] = (col, 1)
        self.mono_new.append((col, min(i, j), max(i, j)))
        self.mono_of[col] = (min(i, j), max(i, j), 1)
        return col, 1

    def _columns_needed(self, pairs):
        new = {(min(i, j), max(i, j)) for i, j in pairs if self._mono(i, j, create=False) is None}
        products = sum(k is not None for k in self.fold.kind.values()) + len(self.mono_new) + len(new)
        return self.n + products

    def _room(self, pairs):
        """whether the monomials x_i x_j of pairs fit: n + product columns + new monomials <= max_columns"""
        return self._columns_needed(pairs) <= self.max_columns

    def _add(self, q, col, v):
        s = q.get(col, 0) + v
        if s:
            q[col] = s
        else:
            q.pop(col, None)

    # ------------------------------------------------------------------------------------------------ rows
    def _links(self):
        """z = a(x) b(x) of two input-affine operands, z not a monomial: z - expansion = 0 (two rows)"""
        n = self.n
        for col, (fa, fb) in sorted(self.fold.operands.items()):
            if col in self.curvature_leaves:
                # Its exact square relation is retained by the QP adapter.
                # Do not create an optional duplicate monomial/link that
                # would turn an objective-only leaf into a coupled column.
                continue
            if col in self.mono_of or fa.t is None or fb.t is None:
                continue
            if any(j >= n for j in fa.t) or any(j >= n for j in fb.t):
                continue
            if not self._room([(i, j) for i in fa.t for j in fb.t]):
                self.stats['links_skipped'] = self.stats.get('links_skipped', 0) + 1
                continue
            q, q0 = {col: 1}, -Fraction(fa.k) * fb.k
            for i, ai in fa.t.items():
                for j, bj in fb.t.items():
                    c, s = self._mono(min(i, j), max(i, j))
                    self._add(q, c, -Fraction(ai) * bj / s)
            for j, bj in fb.t.items():
                self._add(q, j, -Fraction(fa.k) * bj)
            for i, ai in fa.t.items():
                self._add(q, i, -Fraction(fb.k) * ai)
            self.rows.append(_Row(q, q0, 'link'))
            self.rows.append(_Row({c: -v for c, v in q.items()}, -q0, 'link'))

    def _rlt(self, spec, thr, E, h, lo, hi, max_rows):
        n = self.n
        src = []                                       # (g {input: coef}, rhs): g x <= rhs, exact
        for r in range(1, self.S):
            f = spec[r]
            if f.t is None or not math.isfinite(float(thr[r])) or any(j >= n for j in f.t) or not f.t:
                continue
            src.append((f.t, _ex(float(thr[r])) - f.k))
        if E is not None:
            E2, h2 = E.reshape(-1, n).double().cpu(), h.reshape(-1).double().cpu()
            for r in range(E2.shape[0]):
                g = {j: _ex(float(E2[r, j])) for j in torch.nonzero(E2[r]).reshape(-1).tolist()}
                if g and math.isfinite(float(h2[r])):
                    src.append((g, _ex(float(h2[r]))))
        # opposite pairs (lifted_lp.add_rlt): g x <= t_hi and -g x <= -t_lo
        seen, matched, eqs = {}, set(), []
        for g, rhs in src:
            key = tuple(sorted(g.items()))
            opp = tuple(sorted((j, -v) for j, v in g.items()))
            if key in matched:
                continue
            if opp in seen:
                g0, r0 = seen.pop(opp)
                eqs.append((g0, -Fraction(rhs), Fraction(r0)))  # t_lo = -rhs(-g), t_hi = rhs(g)
                matched.update([key, opp])
            elif key not in seen:
                seen[key] = (g, rhs)
        partners = {}
        for key in list(self.fold.products):
            ij = self._single_pair(key)
            if ij is not None:
                partners.setdefault(ij[0], set()).add(ij[1])
                partners.setdefault(ij[1], set()).add(ij[0])
        sel = sorted({(i, e) for e, (g, _, _) in enumerate(eqs) for j in g for i in partners.get(j, ())})
        self.stats.update(rlt_equalities=len(eqs), rlt_selected=len(sel))
        if not sel:
            return
        if 2 * len(sel) > max_rows:
            cap_log('rlt', 'rows', max_rows, 2 * len(sel), 'RLT skipped (all or nothing)')
            return
        pairs = [(i, j) for i, e in sel for j in eqs[e][0]]
        if not self._room(pairs):
            self.stats['rlt_skipped_columns'] = True
            cap_log('rlt', 'columns', self.max_columns, self._columns_needed(pairs), 'RLT skipped (all or nothing)')
            return
        for i, e in sel:
            g, t_lo, t_hi = eqs[e]
            P = {}
            for j, gj in g.items():
                c, s = self._mono(min(i, j), max(i, j))
                self._add(P, c, Fraction(gj) / s)
            cls = 'pos' if lo[i] >= 0 else 'neg' if hi[i] <= 0 else 'mid'
            b = (t_lo + t_hi) / 2
            forms = {'pos': [(1, t_hi), (-1, t_lo)], 'neg': [(1, t_lo), (-1, t_hi)], 'mid': [(1, b), (-1, b)]}
            for c in (('pos', 'neg', 'mid') if cls == 'mid' else (cls,)):
                for side, coef in forms[c]:
                    q = {k: side * v for k, v in P.items()}
                    self._add(q, i, -side * coef)
                    self.rows.append(_Row(q, Fraction(0), 'rlt', (i, c, side, t_lo, t_hi)))

    def _cleared(self, rational_rows, thr, lo):
        from .rational_rows import clear_positive_denominators
        n, count = self.n, 0
        for r in range(1, self.S):
            form = rational_rows[r] if r < len(rational_rows) else None
            if form is None or form.terms is None or not math.isfinite(float(thr[r])):
                continue
            if not any(k < 0 for key in form.terms for _, k in key):
                continue
            poly = clear_positive_denominators(form, float(thr[r]), lo)
            if poly is None:
                continue
            pairs = [(key[0][0], key[-1][0]) for key in poly.terms if key and not (len(key) == 1 and key[0][1] == 1)]
            if not self._room(pairs):
                continue
            q, q0 = {}, Fraction(0)
            for key, v in poly.terms.items():
                if not key:
                    q0 += v
                elif len(key) == 1 and key[0][1] == 1:
                    self._add(q, key[0][0], v)
                else:
                    i, j = (key[0][0], key[0][0]) if len(key) == 1 else (key[0][0], key[1][0])
                    c, s = self._mono(min(i, j), max(i, j))
                    self._add(q, c, Fraction(v) / s)
            self.rows.append(_Row(q, q0, 'cleared'))
            count += 1
        self.stats['cleared'] = count
    # ------------------------------------------------------------------------------------------------ the graph
    def _emit(self, C):
        """rewire the shared products, add the monomial nodes, the derived output D and the new final node"""
        net, fold = self.net, self.fold
        self.new, self.carry, self.shapes, self._n = [], {}, {}, 0
        self.ref = net[net.input_name[0]]
        loc = {j: ('input', j) for j in range(self.n)}      # column -> (node name, flat element) in the new graph
        xf = self._flat(self.ref)
        # ---- shared products: a node with aliased elements becomes (its kept elements | kappa x their reps)[perm]
        for node in [nd for nd in fold.order if nd.name in fold.faces]:
            shape = self._shape(node)
            E = int(np.prod(shape, dtype=np.int64))
            keep = [e for e in range(E) if (node.name, e) not in fold.alias]
            if len(keep) == E:
                for e in range(E):
                    loc[fold.col_index[(node.name, e)]] = (node.name, e)
                continue
            parts, pick = [], {}
            if keep:
                ops = []
                for src in node.inputs[:2 if isinstance(node, BoundMul) else 1]:
                    ss = self._shape(src)
                    idx = np.broadcast_to(np.arange(int(np.prod(ss, dtype=np.int64))).reshape(ss), shape).reshape(-1)
                    idx = idx[keep]
                    ops.append(self._gather(self._flat(src), idx))
                    self.carry[ops[-1].name] = (src.name, idx)
                red = self._node(type(node), ops, dict(node.attr), shape=(1, len(keep)))
                self.carry[red.name] = (node.name, np.asarray(keep))
                parts.append(red)
                for k, e in enumerate(keep):
                    loc[fold.col_index[(node.name, e)]] = (red.name, k)
                    pick[e] = k
            dup = [e for e in range(E) if (node.name, e) in fold.alias]
            srcs = [loc[fold.alias[(node.name, e)][0]] for e in dup]
            kap = torch.tensor([[float(fold.alias[(node.name, e)][1]) for e in dup]], dtype=self.dtype)
            parts.append(self._node(BoundMul, (self._gather_cols(srcs, xf), self._const(kap)), shape=(1, len(dup))))
            for k, e in enumerate(dup):
                pick[e] = len(keep) + k
            cat = self._node(BoundConcat, parts, {'axis': 1}, shape=(1, E)) if len(parts) > 1 else parts[0]
            new = self._gather(cat, np.array([pick[e] for e in range(E)]))
            if len(shape) != 2:
                new = self._node(BoundReshape, (new, self._const(torch.tensor([-1, *shape[1:]]))), shape=shape)
            self.shapes[new.name] = shape
            self.carry[new.name] = (node.name, np.arange(E))
            self._commit()
            net.replace_node(node, new)
        # ---- new monomials: one BoundMul of two gathers of x, one BoundSqr of a gather
        for sq in (False, True):
            group = [(c, i, j) for c, i, j in self.mono_new if (i == j) == sq]
            if not group:
                continue
            ops = []
            for idx in ([np.array([i for _, i, _ in group])] if sq else
                        [np.array([i for _, i, _ in group]), np.array([j for _, _, j in group])]):
                ops.append(self._gather(xf, idx))
                self.carry[ops[-1].name] = ('input', idx)
            m = self._node(BoundSqr if sq else BoundMul, ops, shape=(1, len(group)))
            for k, (c, _, _) in enumerate(group):
                loc[c] = (m.name, k)
        # ---- the derived rows: D = V W + W0 over the referenced columns V; the final node concat(outputs, D)
        K = len(self.rows)
        self.used = sorted({c for r in self.rows for c in r.q})
        pos = {c: k for k, c in enumerate(self.used)}
        W = torch.zeros(len(self.used), K, dtype=torch.float64)
        W0 = torch.zeros(K, dtype=torch.float64)
        self.dq = torch.zeros(len(self.used), K, dtype=torch.float64)
        self.dq0 = torch.zeros(K, dtype=torch.float64)
        for k, r in enumerate(self.rows):
            for c, v in r.q.items():
                W[pos[c], k], self.dq[pos[c], k] = self._round(v)
            W0[k], self.dq0[k] = self._round(r.q0)
        self.W, self.W0 = W, W0
        if K:
            final = net[net.final_name]
            V = self._gather_cols([loc[c] for c in self.used], xf)
            D = self._node(BoundAdd, (self._node(BoundMatMul, (V, self._const(W.to(self.dtype))), shape=(1, K)),
                                      self._const(W0.to(self.dtype).reshape(1, K))), shape=(1, K))
            fin = self._node(BoundConcat, (self._flat(final), D), {'axis': 1}, shape=(1, self.n_out + K))
            self._commit()
            final.is_final_node = False
            fin.is_final_node = True
            net.final_name, net.output_name, net.alpha_start_nodes = fin.name, [fin.name], [fin.name]
        self._commit()
        # the operands of rewired products that nothing else reads (as _optimize_graph's cleanup)
        orphan = [nd for nd in net.nodes() if not nd.output_name and nd.name != net.final_name
                  and nd.name not in net.root_names]
        while orphan:
            for nd in orphan:
                net.delete_node(nd)
            orphan = [nd for nd in net.nodes() if not nd.output_name and nd.name != net.final_name
                      and nd.name not in net.root_names]
        self._reorder()
        net.last_bound_target_name = None
        net.get_optimizable_activations()
        out = net(*net.global_input)
        net.final_shape = out.shape
        net.bound_opts['final_shape'] = out.shape
        C2 = torch.zeros(self.S + K, self.n_out + K, dtype=self.dtype)
        C2[:self.S, :self.n_out] = C.to(self.dtype)
        C2[self.S:, self.n_out:] = torch.eye(K, dtype=self.dtype)
        self.C = C2.to(self.device)
        # the boxes of the referenced columns: an input, a monomial x_i x_j at scale s (v = s x_i x_j), or none (its
        # rows round no coefficient of it and no RLT row uses it)
        rule = {'in': ([], []), 'mono': ([], [], [], [])}
        in_rlt = {c for r in self.rows if r.rlt is not None for c in r.q}
        for k, c in enumerate(self.used):
            if c < self.n:
                rule['in'][0].append(k)
                rule['in'][1].append(c)
            elif c in self.mono_of:
                i, j, sc = self.mono_of[c]
                if Fraction(float(sc)) != Fraction(sc):
                    raise ValueError('a monomial scale that is not a float')
                for lst, v in zip(rule['mono'], (k, i, j, float(sc))):
                    lst.append(v)
            elif bool((self.dq[pos[c]] != 0).any()) or c in in_rlt:
                raise ValueError(f'column {c}: a rounded coefficient without a magnitude bound')
        L = lambda v, dt=torch.long: torch.tensor(v, dtype=dt)
        self.rule_in = tuple(L(v) for v in rule['in'])
        self.rule_mono = (*(L(v) for v in rule['mono'][:3]), L(rule['mono'][3], torch.float64))
        rl = [k for k, r in enumerate(self.rows) if r.rlt is not None]
        code = {'pos': 0, 'neg': 1, 'mid': 2}
        el, eu, ee = [], [], []
        for k in rl:
            i, cls, side, t_lo, t_hi = self.rows[k].rlt
            b = (t_lo + t_hi) / 2
            el.append(float(t_lo - b))
            eu.append(float(t_hi - b))
            ee.append(float(abs(Fraction(el[-1]) - (t_lo - b)) + abs(Fraction(eu[-1]) - (t_hi - b))))
        self.rule_rlt = (L(rl), L([self.rows[k].rlt[0] for k in rl]), L([code[self.rows[k].rlt[1]] for k in rl]),
                         L([self.rows[k].rlt[2] for k in rl], torch.float64), L(el, torch.float64),
                         L(eu, torch.float64), L(ee, torch.float64))

    def _rewrite_squares(self, mixed=False):
        """every BoundMul whose two operands are the same node, or have the same exact affine forms element by element
        with the product's own shape, replaced by a BoundSqr of the first operand; returns how many"""
        net = self.net
        fold = _Fold(net)
        self.new, self.carry, self.shapes, self._n = [], {}, {}, 0
        self.ref = net[net.input_name[0]]
        eq = lambda a, b: a.t is not None and b.t is not None and a.t == b.t and a.k == b.k
        count = 0
        for node in [nd for nd in fold.order if isinstance(nd, BoundMul) and nd.name in fold.faces]:
            a, b = node.inputs[0], node.inputs[1]
            shape = tuple(node.forward_value.shape)
            if a is not b:
                if tuple(a.forward_value.shape) != shape or tuple(b.forward_value.shape) != shape:
                    continue
                fa, fb = fold.array(a).reshape(-1), fold.array(b).reshape(-1)
                mask = np.array([eq(x, y) for x, y in zip(fa, fb)])
                if not mask.all():
                    if not mixed or not mask.any():
                        continue
                    # Split mixed vector products by exact affine identity, then
                    # restore the original element order. No numerical test.
                    sq, other = np.flatnonzero(mask), np.flatnonzero(~mask)
                    aa, bb = self._flat(a), self._flat(b)
                    left = self._node(BoundSqr, (self._gather(aa, sq),), shape=(1, len(sq)))
                    right = self._node(BoundMul, (self._gather(aa, other), self._gather(bb, other)),
                                       shape=(1, len(other)))
                    cat = self._node(BoundConcat, (left, right), {'axis': 1}, shape=(1, len(mask)))
                    new = self._gather(cat, np.argsort(np.concatenate([sq, other])))
                    if len(shape) != 2:
                        new = self._node(BoundReshape, (new, self._const(torch.tensor([-1, *shape[1:]]))), shape=shape)
                    for created in self.new:
                        created.name = created.name.replace('/s3poly/', '/s3sqr/mixed/')
                    self._commit()
                    net.replace_node(node, new)
                    count += 1
                    continue
            elif tuple(a.forward_value.shape) != shape:
                continue
            new = self._node(BoundSqr, (a,), {}, shape=shape)
            new.name = f'/s3sqr/{count + 1}'           # (_emit's nodes count from 1 again)
            self._commit()
            net.replace_node(node, new)
            count += 1
        if count:
            self._reorder()
        self.new, self.carry, self.shapes, self._n = [], {}, {}, 0
        return count

    def _shape(self, node):
        if node.name in self.shapes:
            return self.shapes[node.name]
        return tuple(node.forward_value.shape)

    def _round(self, v):
        """the nearest float of an exact rational and an upper bound of its error"""
        f = float(v)
        err = abs(Fraction(f) - Fraction(v))
        return f, (float(np.nextafter(float(err), np.inf)) if err else 0.0)

    def _node(self, kind, inputs=(), attr=None, shape=None):
        attr = dict(attr or {}, device=self.device)
        node = kind(attr, list(inputs), 0, self.net.bound_opts)
        self._n += 1
        node.name = f'/s3poly/{self._n}'
        _copy_node_properties(node, self.ref)
        node.perturbed = any(p.perturbed for p in inputs)
        if shape is not None:
            self.shapes[node.name] = tuple(shape)
        self.new.append(node)
        return node

    def _const(self, value):
        node = self._node(BoundConstant, attr={'value': value.to(self.device)})
        node.output_shape = value.shape
        node.batch_dim = -1
        return node

    def _commit(self):
        if self.new:
            self.net.add_nodes(self.new)
            self.new = []

    def _flat(self, node):
        """node as (batch, elements)"""
        shape = self._shape(node)
        if len(shape) == 2:
            return node
        m = int(np.prod(shape[1:], dtype=np.int64))
        return self._node(BoundReshape, (node, self._const(torch.tensor([-1, m]))), shape=(1, m))

    def _gather(self, node, idx):
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        return self._node(BoundGather, (node, self._const(torch.as_tensor(idx))), {'axis': 1}, shape=(1, len(idx)))

    def _lookup(self, name):
        if name in self.net._modules:
            return self.net[name]
        return next(nd for nd in self.new if nd.name == name)

    def _gather_cols(self, locs, xf):
        """one vector (1, len(locs)) of the elements locs [(node name or 'input', flat element)]: one concat, one gather"""
        names = list(dict.fromkeys(name for name, _ in locs))
        nodes = [xf if name == 'input' else self._flat(self._lookup(name)) for name in names]
        width = [int(np.prod(self._shape(nd)[1:], dtype=np.int64)) for nd in nodes]
        start = dict(zip(names, np.cumsum([0] + width[:-1]).tolist()))
        cat = self._node(BoundConcat, nodes, {'axis': 1}, shape=(1, sum(width))) if len(nodes) > 1 else nodes[0]
        return self._gather(cat, [start[name] + e for name, e in locs])

    def _reorder(self):
        """the modules in topological order (every node after its inputs)"""
        net = self.net
        order = _postorder([net[net.final_name]] + list(net.nodes()))
        items = dict(net._modules)
        net._modules.clear()
        for node in order:
            net._modules[node.name] = items.pop(node.name)
        net._modules.update(items)

    # ------------------------------------------------------------------------------------------------ per box
    def thresholds(self, x_L, x_U, thr=None):
        """(B, S + K): the spec's thresholds thr ((S,) or (B, S); default the relaxed ones given) and every derived
        row's on the boxes x_L, x_U (B, n) (sub-boxes of the root box), rounded outward: its coefficients' rounding
        sum_c |q_c - W_c| M_c + |q_0 - W_0|, plus for an RLT row the interval term of its multiplier's class on the box,
        or, when the box is not of the row's class, the maximum of its left side over the column boxes. Fixed-shape
        tensor algebra on the boxes' device (no host synchronisation)."""
        B = x_L.shape[0]
        l, u = x_L.reshape(B, -1).double(), x_U.reshape(B, -1).double()
        dev = l.device
        T = self._on(dev)
        base = (T.thr if thr is None else thr.double()).reshape(-1, self.S).expand(B, self.S)
        K = len(self.rows)
        if not K:
            return base.to(self.dtype)
        g = lambda k: k * U64 / (1 - k * U64)
        up = lambda t: torch.nextafter(t, torch.full_like(t, math.inf))
        Mx = torch.maximum(l.abs(), u.abs())
        nU = len(self.used)
        # the referenced columns' magnitudes M and boxes [lo, hi] (inputs: the box; monomials: s times the corner
        # products, padded by their rounding)
        M = torch.zeros(B, nU, dtype=torch.float64, device=dev)
        lo, hi = torch.zeros_like(M), torch.zeros_like(M)
        k_in, c_in = T.rule_in
        M[:, k_in], lo[:, k_in], hi[:, k_in] = Mx[:, c_in], l[:, c_in], u[:, c_in]
        k_m, i, j, sc = T.rule_mono
        if k_m.numel():
            li, ui, lj, uj = l[:, i], u[:, i], l[:, j], u[:, j]
            cand = torch.stack([li * lj, li * uj, ui * lj, ui * uj])
            a, b = cand.min(0).values, cand.max(0).values
            sq = i == j
            a = torch.where(sq, torch.where(li > 0, li * li, torch.where(ui < 0, ui * ui, 0.0)), a)
            a, b = torch.where(sc > 0, sc * a, sc * b), torch.where(sc > 0, sc * b, sc * a)
            mag = sc.abs() * Mx[:, i] * Mx[:, j] * (1 + g(3))
            M[:, k_m], lo[:, k_m], hi[:, k_m] = up(mag), a - g(3) * mag, b + g(3) * mag
        # every row's few nonzeros as a fixed-width gather (_ell_columns), not a dense product
        ic, vc = T.dq_ell
        cushion = up(((M[:, ic] * vc).sum(-1) + T.dq0) * (1 + g(ic.shape[1] + 2)))
        out = cushion
        rl, ri, rc, side, el, eu, ee = T.rule_rlt
        if rl.numel():
            iw, vw = T.W_rlt_ell
            hw, lw = hi[:, iw], lo[:, iw]
            mx = torch.where(vw > 0, hw * vw, lw * vw).sum(-1) + T.W0_rlt
            mag = ((hw.abs() + lw.abs()) * vw.abs()).sum(-1) + T.W0_rlt.abs()
            redundant = up(mx + g(iw.shape[1] + 2) * mag)
            li, ui = l[:, ri], u[:, ri]
            now = torch.where(li >= 0, 0, torch.where(ui <= 0, 1, 2))
            # a straddling multiplier's rows: the max (side +1) or -min (side -1) of x_i tau over x_i in [l_i, u_i],
            # tau in [t_lo - b, t_hi - b], from the corner products of the float band ends (each within u |p|), those
            # ends within e of the exact ones (e |x_i| <= e M_i)
            p = torch.stack([li * el, li * eu, ui * el, ui * eu])
            ext = torch.where(side > 0, p.max(0).values, -p.min(0).values)
            mid = ext + 2 * U64 * p.abs().max(0).values + 2 * ee * Mx[:, ri]
            act = up(torch.where(rc == 2, mid, 0.0) + cushion[:, rl])
            out = cushion.clone()
            out[:, rl] = torch.where(now == rc, act, redundant)
        return torch.cat([base, out], 1).to(self.dtype)

    def _on(self, dev):
        """the thresholds' data on a device (copied once)"""
        cache = self._device_data
        if dev not in cache:
            to = lambda t: t.to(dev)
            rl = self.rule_rlt[0]
            cache[dev] = types.SimpleNamespace(
                thr=to(self.thr_spec), dq_ell=tuple(map(to, _ell_columns(self.dq))), dq0=to(self.dq0),
                rule_in=tuple(map(to, self.rule_in)), rule_mono=tuple(map(to, self.rule_mono)),
                rule_rlt=tuple(map(to, self.rule_rlt)), W_rlt_ell=tuple(map(to, _ell_columns(self.W[:, rl]))),
                W0_rlt=to(self.W0[rl]))
        return cache[dev]

    def interm_bounds(self, IB, x_L, x_U):
        """the frozen intermediate bounds {name: (lower, upper)} of the original graph, with those of the new nodes that
        need them (the operands of the new face nodes, the rewired products' own bounds) carried over: exact moves"""
        B = x_L.shape[0]
        out = dict(IB)
        for name, (src, idx) in self.carry.items():
            if src == 'input':
                l, u = x_L.reshape(B, -1), x_U.reshape(B, -1)
            elif src in out:
                l, u = out[src]
                l, u = l.reshape(l.shape[0], -1), u.reshape(u.shape[0], -1)
            else:
                continue
            t = torch.as_tensor(idx, dtype=torch.long, device=l.device)
            shape = self.shapes.get(name)
            lo, hi = l[:, t], u[:, t]
            if shape is not None and len(shape) != 2:
                lo, hi = lo.reshape(-1, *shape[1:]), hi.reshape(-1, *shape[1:])
            out[name] = (lo, hi)
        return out
