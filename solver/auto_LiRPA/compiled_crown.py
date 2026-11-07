"""Compiled CROWN: CROWN's computation on one graph and one set of operator faces, as fixed-shape tensor algebra
batched over boxes x right-hand sides and capturable as one CUDA graph (crown_absorb/compiled, Q1).

Compilation (box independent, once per graph). Columns v: the inputs and every element of every face node (the
nonlinear nodes of face_dual.install); every affine node is folded into the rows: each operand of each face node, and
the output, is an affine form M v + c over the columns, read off by one forward-mode probe of the graph (each face node
held at its columns, one tangent per column, as pivot_crown and test_pivot_crown.dense_lp). The face nodes are graded
by level: 1 + the highest level of a face node their operands depend on (the inputs: level 0). The compiled column
order is level-major, so the operands of a level depend on a prefix of the columns. Rows A v <= b, as pivot_crown:
    faces      lower  sum_p a_p (M_p v + c_p) + b - z <= 0,  upper  z - sum_p a'_p (M_p v + c_p) - b' <= 0
    outputs    C[k] (M_out v + c_out) <= thr[k]   (k in `rows`)
    base rows  E x <= h
and the column boxes lo <= v <= hi (the input box, the node intervals); objective C[0] (M_out v + c_out). The boxes
share the pattern; face coefficients, offsets, column boxes, thresholds and h are per box (Data). Faces are padded to
the largest face count of any node (a mask marks the real ones), operands to two (a unary node's second is zero).

Primitives (B boxes x Q right-hand sides, fixed shapes, no host synchronisation):
    F_A(d, v)        b - A v of every row: one product with the stacked rows, then elementwise
    R_A(d, y)        r = c + A^T y and D(y) = c0 - b^T y + sum_j min(r_j lo_j, r_j hi_j): one product
    densify, F_Ad, R_Ad   the rows as one per-box matrix (B, nrow, ncol + 1): F_A / R_A as one batched GEMM each
    F_T(d, s, v)     the forward through the selected faces s (`select`), level by level (pivot_crown.Graph.forward)
    R_T(d, s, ...)   the backward through the selected faces, level by level (pivot_crown.Graph.backward)
    crown(...)       the sign backward through each side's default relaxation (CROWN, face_dual mode 'sign')
    bound(x_L, x_U)  CROWN with its intermediate bounds on the compiled form: per level the bounds of its operands
                     (CROWN through the levels below), its faces rebuilt from them by the operators' own code, its node
                     intervals (IBP of the operator on the operand bounds, intersected with the node's own CROWN bound
                     where auto_LiRPA computes one: the node is an operand of a face node); then the output rows
    certify(d, y)    D(y) in fp64 with a forward-error enclosure (gamma_k, Higham Lemma 3.1/3.3) of every reduction and
                     outward rounding of the box pricing and of the final sums (theory3/xrev_ipm/audit_core.py), the
                     rows relaxed by the fold's own rounding: a lower bound of the exact D(y) of the graph's exact LP.
Every product with a shared matrix (the operand forms, the stacked rows, C M_out, E) is a dense GEMM or, sparse=True, a
sliced ELL over the matrix's nonzeros (Mat, _Reduce). sparse=True never forms a dense matrix: the probe runs over chunks
of the columns and keeps every form by its nonzeros (indices and values), C M_out is a sparse product, every Mat is
built from the nonzeros, and bound()'s sign backward runs slot by slot (Plan): every right-hand side lives on the
columns it can reach, level by level, so its cost is that of its nonzeros, not B Q ncol. Everything above captures as
one CUDA graph (Captured): the
operators' own face code runs with its validity checks deferred (operators.base.deferred_checks: a box on which a
relaxation is undefined is flagged, Data.undefined, and gets -inf, where the operator would raise), a unary node's on a
private copy of the node (BoundActivation.faces restore=False); bound() and the primitives also trace whole under
dynamo (fuse: inductor fuses them; the fused kernels may round differently, e.g. with FMA, which the certificate's
error bounds cover and BoundMul's choice, by the exact corner-product rule, does not depend on).
The compiled LP is the folded graph LP: its rows hold M as the probe computed it in fp64, not the exact composition of
the affine nodes' weights (a cancelling composition such as (1e16 x - x) - 1e16 x folds to 0). Compilation also bounds
that rounding (_fold_error: dM, dc of every operand form and of C's rows, box independent), and certify relaxes every
row by it on the box (_fold_slack): the object certify is rigorous for is the graph's exact LP (the affine nodes
composed exactly, their fp64 constants and the stored faces, thresholds and boxes as exact numbers).

The intermediate bounds are CROWN's (auto_LiRPA's compute_bounds(method='CROWN') with
compare_crown_with_ibp(_intermediate) off, no reference bounds, no constraints on the input box). margins (A68; the
default with sparse=True) makes the pass's own numbers rigorous for the exact graph: every operand bound, own bound and
output bound is a rigorous lower bound of the exact D(y*) at the multipliers it selects (_plan_bound: the backward's
rounding by gamma_k of the magnitudes it carries in a second channel, the fold's error bounds, underflow), the node
intervals are the operators' IBP rounded outward (BoundMul: one step, one rounded product per end; others: _REL, the
libm accuracy assumed) and floored at a width of 2^-(p/2) of their magnitude (_widen: never a near-degenerate column,
A71), and BoundMul's faces are moved to hold for the exact product on the operand box (_mul_valid).
So, on a box, every point of the exact graph satisfies the stored BoundMul faces and lies in every node box: certify's
LP is a relaxation of the exact graph, and certify bounds the exact graph's minimum. Not enclosed: a unary operator's
own faces (its tangent and secant lines, computed by its code from rigorous bounds; their rounding and libm's).

Layouts. Compiled: the columns [inputs | level-1 elements | level-2 ...]; faces (B, F, [2 operands,] Et) per side.
Canonical: pivot_crown.Graph's (inputs, then the face nodes in the order of the cache / net.nodes(); rows per node its
lower faces, then its upper faces, face-major, then the output rows, then the base rows); `canon_cols`/`compiled_cols`,
`canon_rows`/`compiled_rows` map vectors between them.
"""
from .deadline import check_deadline
from .cuda_memory import capture

import copy
import math

import torch
import torch.autograd.forward_ad as fwAD

from .operators.base import Interval, deferred_checks

INF = float('inf')
# per floating-point dtype: the significand bits p (unit roundoff 2^-p) and a bound of the absolute error of an
# underflowing product: the dtype's smallest normal (fp64 2^-1022 >= 2^-1075, float32 2^-126 >= 2^-150), not its
# smallest subnormal, so that no allowance is subnormal: a GPU's arithmetic on subnormal fp64 operands is slow (a sum
# of per-row subnormal allowances took 8 ms in place of 6 us, wastewater13m1's extended graph), and Triton types a
# float literal outside float32's normal range as fp64 (which breaks a float32 loop-carried reduction); every rounding
# constant below is a Python float exact in its dtype
_P = {torch.float64: 53, torch.float32: 24}
_ETA = {torch.float64: 2.0 ** -1022, torch.float32: 2.0 ** -126}
# the relative accuracy assumed of the unary operators' interval code (libm), fp_enclosure's rounding_error(|v|, 1)
_REL = {dt: 68 * 2.0 ** (1 - p) for dt, p in _P.items()}
PROBE_ELEMS = 1 << 24            # the probe's tangents: at most this many elements of the node values per chunk


def face_nodes(net):
    """the face nodes of `net` in net.nodes() order: the perturbed nonlinear nodes (pivot_crown._nonlinear) whose
    face-dual step runs (face_dual.install's predicate less the linear family, which has no faces)"""
    from .operators import BoundLinear
    from .pivot_crown import _nonlinear
    return [node.name for node in net.nodes() if node.perturbed and _nonlinear(node) and not isinstance(node, BoundLinear)]


class Node:
    """A face node: its compiled element range [s, s + E), canonical first column c0, operand count P (1 or 2; `same`
    when both are one node), operand node names, face counts Fl, Fu, level, `own`: auto_LiRPA computes its own
    intermediate bound (it is an operand of a face node)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class Data:
    """Per-box tensors (B leading): lo, hi (B, ncol) the column boxes; lw (B, Fl, 2, Et), lb (B, Fl, Et) the lower faces
    in operand terms, uw, ub the upper ones; dl (B, Fl, Et), du the default relaxation as face weights; thr (B, S);
    h (B, R). `derive`: lbc, ubc (B, F, Et) the faces' constants in column terms (lb + sum_p lw_p c_p).
    CompiledGraph._data adds undefined (B,) and slack (the fold's row slack on these boxes, CompiledGraph._fold_slack)."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def derive(self, cop):
        self.lbc = self.lb + _fsum(self.lw * cop, 2)
        self.ubc = self.ub + _fsum(self.uw * cop, 2)
        return self

    def rhs(self, rows, Ccr):
        """b of every row in the flat structured layout (B, nflat): -lbc, ubc, thr - C c_out, h"""
        B = self.lo.shape[0]
        self.bflat = torch.cat([-self.lbc.reshape(B, -1), self.ubc.reshape(B, -1), self.thr[:, rows] - Ccr, self.h], 1)
        return self

    def to(self, dtype):
        return Data(**{key: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
                       for key, v in vars(self).items()})


def _fsum(t, dim):
    """the sum over a short dim (the faces, the operands) unrolled, left to right: pointwise for inductor, which
    fuses it into its consumer (fuse turns the unrolling of small reductions off, a Triton workaround, so a .sum over a
    dim of 2 would be a kernel of its own)"""
    if t.shape[dim] == 0:
        return t.sum(dim)
    parts = t.unbind(dim)
    out = parts[0]
    for u in parts[1:]:
        out = out + u
    return out


def _price(r, lo, hi):
    """sum over the last dim of min(r lo, r hi) (face_dual._pay): an infinite end counts only where r selects it (a
    product 0 x inf is NaN, which fmin passes over; both ends NaN only at r = 0 with both ends infinite: 0)"""
    return torch.fmin(r * lo, r * hi).nan_to_num_(nan=0.0).sum(-1)


def _price_shared(g, lo, hi):
    """_price of a coefficient g (Q, m) shared by the boxes lo, hi (B, m): (B, Q), by GEMMs"""
    fl, fu = torch.isfinite(lo), torch.isfinite(hi)
    p = torch.where(fl, lo, torch.zeros_like(lo)) @ g.clamp(min=0).t() + \
        torch.where(fu, hi, torch.zeros_like(hi)) @ g.clamp(max=0).t()
    bad = (~fl).to(g.dtype) @ (g > 0).to(g.dtype).t() + (~fu).to(g.dtype) @ (g < 0).to(g.dtype).t()
    return torch.where(bad > 0, torch.full_like(p, -INF), p)


def _round(x, dt, up):
    """the Python float x (0 or normal in dt) rounded to dt's precision, upward (up) or downward: a number of dt"""
    if x == 0.0 or not math.isfinite(x):
        return x
    m, e = math.frexp(x)
    s = math.ldexp(m, _P[dt])
    return math.ldexp(math.ceil(s) if up else math.floor(s), e - _P[dt])


def _gamma(k, dt=torch.float64):
    """gamma_k = k u / (1 - k u) (Higham; u = 2^-p the unit roundoff of dt) as a number of dt: in fp64 the expression
    as it rounds (every caller counts at least one rounding more than its terms undergo, which covers the last bit:
    gamma_{k+1} - gamma_k ~ u >> ulp(gamma_k)), in a lower precision rounded up (its conversion would round)"""
    u = 2.0 ** -_P[dt]
    g = k * u / (1 - k * u)
    return g if dt == torch.float64 else _round(math.nextafter(g, INF), dt, True)


def _below1(g, dt):
    """a number of dt not above 1 - g (a divisor that does not round the quotient down; fp64: the next below)"""
    return _round(math.nextafter(1.0 - g, 0.0), dt, False)


def _gq(g, dt):
    """a number of dt not below g / (1 - g) (a multiplier in place of a division, which is slow in fp64 on a GPU)"""
    return _round(math.nextafter(math.nextafter(g / (1.0 - g), INF), INF), dt, True)


def _one_minus(g, dt):
    """1 - g as certify divides by it: in fp64 as it rounds (its count's slack covers it, as _gamma's), else _below1"""
    return 1 - g if dt == torch.float64 else _below1(g, dt)


def _down(x):
    return torch.nextafter(x, torch.full_like(x, -INF))


def _up(x):
    return torch.nextafter(x, torch.full_like(x, INF))


def _upper(s, m, P):
    """an upper bound of the exact value of a sum of nonnegative terms that fp64 computed as s, each term rounded at
    most m times and P products per entry (Higham Lemma 3.1: s >= (1 - gamma_m) exact - P ETA, gamma one step wider
    for its own rounding)"""
    dt = s.dtype
    return _up(_up(s + P * _ETA[dt]) / _below1(_gamma(m + 1, dt), dt))


def _to(t, dt, up):
    """t in dtype dt, rounded outward: never below t (up) / never above it (a no-op for dt = t's dtype)"""
    if t.dtype == dt:
        return t
    r = t.to(dt)
    w = r.to(t.dtype)
    return torch.where(w < t, _up(r), r) if up else torch.where(w > t, _down(r), r)


def _widen(l, u):
    """node intervals narrower than tau (1 + their end magnitude) (tau = 2^-(p/2) of the dtype: fp64 1.5e-8, float32
    2.4e-4) widened to that width about their middle, never inside the original (a relaxation). margins turn an exactly
    degenerate interval (a product of fixed operands) into one a few ulps wide, which makes the scaled LP's pivots on
    that column tiny and the sign-CROWN start basis singular (A71: cond up to 1e16 on boxes with fixed inputs); the
    floor keeps them at least tau relative"""
    tau = 2.0 ** -(_P[l.dtype] // 2)
    s = tau * (1 + torch.maximum(l.abs(), u.abs()))
    narrow = (u - l < s) & torch.isfinite(s)
    mid = (l + u) / 2
    return (torch.where(narrow, torch.minimum(l, _down(mid - s / 2)), l),
            torch.where(narrow, torch.maximum(u, _up(mid + s / 2)), u))


def column_scale(lo, hi):
    """a pivot's column scale for boxes lo, hi: max((hi - lo) / 2, tau (1 + |centre|)), tau as _widen's, so that an
    exactly fixed column (an LP solver's usual scale 1) and a near-fixed one scale alike: with it the sign-CROWN start
    bases on boxes with fixed inputs have cond <= 25 (ex5_2_5) and 16 (wastewater13m1), against up to 6e9 with the
    width as the scale and 1 for the fixed (A71; compiled/sparse REPORT)"""
    tau = 2.0 ** -(_P[lo.dtype] // 2)
    return torch.maximum((hi - lo) / 2, tau * (1 + ((lo + hi) / 2).abs()))


def _cast(v, dv, dt):
    """v rounded to the nearest number of dtype dt, and its error bound dv (|exact - v| <= dv) widened by that
    rounding: (fl(v), dv + |v - fl(v)| rounded up in dt)"""
    w = v.to(dt)
    return w, _to(_up((v - w.to(v.dtype)).abs() + dv), dt, True)


def _cast_coo(a, da, dt, ncol):
    """_cast of a matrix given by its nonzeros a and its error bounds' da (on the union of their patterns): the values'
    nonzeros and the error bounds' in dtype dt"""
    dev = a[2].device
    r, c, v, dv = _merge(a, da, ncol)
    w, e = _cast(v, dv, dt)
    put = lambda k, x: (r[k].to(dev), c[k].to(dev), x[k].to(dev))
    return put(w != 0, w), put(e > 0, e)


def _ends(r, e, lo, hi):
    """the most min(r* lo, r* hi) can fall below min(r lo, r hi) for any r* within e of r: e times the magnitude of the
    box end r* selects (r > e: lo, r < -e: hi, else the larger: min(. lo, . hi) is max(|lo|, |hi|)-Lipschitz); 0
    where e is 0"""
    v = torch.where(r > e, lo.abs(), torch.where(r < -e, hi.abs(), torch.maximum(lo.abs(), hi.abs())))
    return torch.where(e > 0, e * v, 0.0)


def _mul_valid(w, b, lo, up, lower):
    """BoundMul faces z >= w_0 x + w_1 y + b (lower) / z <= ... (upper), w (F, B, 2, E), b (F, B, E), made valid for
    the exact product z = x y on the operand box [lo, up] (B, 2, E): their offsets moved by a rigorous upper bound of
    the largest violation. The violation w_0 x + w_1 y + b - x y (lower; its negative for an upper face) is bilinear, so
    its maximum over the box is at one of the 4 corners; each corner's value is computed with its forward error
    enclosed (4 terms, each rounded at most 4 times: gamma_5 of their magnitudes (their computed sum m, rounded at most
    5 times: the exact one at most m / (1 - gamma_5)), 3 products' underflow), rounded up.
    The moved offset is rounded outward. A face with a non-finite corner value becomes vacuous (offset -inf / +inf)."""
    dt = b.dtype
    g, eta = _gamma(5, dt), _ETA[dt]
    nu = None
    for x in (lo[:, 0], up[:, 0]):
        for y in (lo[:, 1], up[:, 1]):
            t = (w[:, :, 0] * x, w[:, :, 1] * y, b, -(x * y))
            if not lower:
                t = tuple(-u for u in t)
            s = t[0] + t[1] + t[2] + t[3]
            m = t[0].abs() + t[1].abs() + t[2].abs() + t[3].abs()
            v = _up(s + _up(m * _gq(g, dt) + 4 * eta))
            nu = v if nu is None else torch.maximum(nu, v)
    nu = torch.where(torch.isfinite(nu), nu.clamp(min=0.0), INF)
    return torch.where(nu > 0, _down(b - nu) if lower else _up(b + nu), b)


# The affine operators the fold composes (every perturbed node between the face nodes): per class the roundings any
# term of one output entry undergoes in its forward (Higham's k), the arguments that are values (else: indices, shapes,
# axes; value arguments enter the magnitude pass by their magnitudes) and the magnitude map where the forward is not
# one (Sub: a + b, Neg: a).
_FOLD_EXACT = ('BoundGather', 'BoundSlice', 'BoundConcat', 'BoundReshape', 'BoundUnsqueeze', 'BoundSqueeze',
               'BoundTranspose', 'BoundFlatten', 'BoundExpand', 'BoundNeg', 'BoundCast')


def _fold_rule(node, args):
    """(magnitude map, k, value argument indices) of the affine node `node` on its forward's arguments `args`"""
    cls = type(node).__name__
    allargs = tuple(range(len(args)))
    if cls == 'BoundSub':
        return (lambda a, b: a + b), 1, allargs
    if cls == 'BoundNeg':
        return (lambda a: a), 0, (0,)
    if cls == 'BoundCast' and args[0].dtype != node.type:
        raise NotImplementedError(f'compiled_crown: a cast to {node.type} rounds ({node.name})')
    if cls in _FOLD_EXACT:
        return node.forward, 0, (0,)
    if cls in ('BoundAdd', 'BoundMul') or cls == 'BoundDiv' and not node.inputs[1].perturbed:
        return node.forward, 1, allargs
    if cls == 'BoundMatMul':
        return node.forward, args[0].shape[-1], allargs
    if cls == 'BoundLinear' and node.alpha_linear >= 0 and node.beta_linear >= 0:   # (a negative scaling: none)
        K = args[0].shape[-2] if node.transA else args[0].shape[-1]
        bias = len(args) > 2 and args[2] is not None
        return node.forward, K + bias + (node.alpha_linear != 1.0) + (bias and node.beta_linear != 1.0), allargs
    if cls == 'BoundATenIndex_add':
        return node.forward, int(torch.bincount(args[2].reshape(-1)).max()) + 1 + (float(args[4]) != 1.0), (0, 3, 4)
    if cls == 'BoundReduceSum':
        return node.forward, max(1, args[0][0].numel() // node.forward(*args)[0].numel()), (0,)
    raise NotImplementedError(f'compiled_crown: no rounding rule for the affine operator {cls} ({node.name})')


def _coo(M):
    """the nonzeros of a dense 2-d M: (rows, cols, vals), row-major"""
    r, c = torch.nonzero(M, as_tuple=True)
    return r, c, M[r, c]


def _layout(q, Q):
    """the positions of the entries q (sorted by their row, Q rows) within their rows, and the most entries of a row
    (at least 1)"""
    cnt = torch.bincount(q, minlength=Q)
    return torch.arange(len(q), device=q.device) - (torch.cumsum(cnt, 0) - cnt)[q], \
        max(1, int(cnt.max())) if len(q) else 1


def _rows(coo, rows):
    """the nonzeros of the rows `rows` (in that order, renumbered 0, 1, ...) of a matrix given by its row-major
    nonzeros"""
    r, c, v = coo
    rows = rows.to(r.device)
    S = 1 + max(int(r.max()) if len(r) else 0, int(rows.max()) if len(rows) else 0)
    cnt = torch.bincount(r, minlength=S)
    ptr = torch.cumsum(cnt, 0) - cnt
    k = cnt[rows]
    t = torch.repeat_interleave(torch.arange(len(rows), device=r.device), k)
    j = ptr[rows][t] + torch.arange(len(t), device=r.device) - (torch.cumsum(k, 0) - k)[t]
    return t, c[j], v[j]


def _merge(a, b, ncol):
    """the union of the patterns of two matrices given by their nonzeros (unique entries): (rows, cols, a's values,
    b's values) on the CPU, 0 where one has none, sorted by (row, col)"""
    ka, kb = a[0].cpu() * ncol + a[1].cpu(), b[0].cpu() * ncol + b[1].cpu()
    uk = torch.unique(torch.cat([ka, kb]))
    put = lambda k, v: torch.zeros(len(uk), dtype=v.dtype).index_put_((torch.searchsorted(uk, k),), v.cpu())
    return uk // ncol, uk % ncol, put(ka, a[2]), put(kb, b[2])


def _spmm(a, b, ncol):
    """the entries (rows, cols, vals) of A B for A, B given by their nonzeros: every structurally reached entry, its
    terms summed on the CPU in A's entry order (deterministic), sorted by (row, col)"""
    ar, ac, av = (t.cpu() for t in a)
    br, bc, bv = (t.cpu() for t in b)
    o = torch.argsort(br, stable=True)
    br, bc, bv = br[o], bc[o], bv[o]
    nb = 1 + max(int(ac.max()) if len(ac) else 0, int(br.max()) if len(br) else 0)
    cnt = torch.bincount(br, minlength=nb)
    ptr = torch.cumsum(cnt, 0) - cnt
    k = cnt[ac]
    t = torch.repeat_interleave(torch.arange(len(ac)), k)
    j = ptr[ac][t] + torch.arange(len(t)) - (torch.cumsum(k, 0) - k)[t]
    uk, inv = torch.unique(ar[t] * ncol + bc[j], return_inverse=True)
    return uk // ncol, uk % ncol, torch.zeros(len(uk), dtype=av.dtype).index_add_(0, inv, av[t] * bv[j])


class _Reduce:
    """y[..., j] = sum over output j's entries (o, i, v) of f(v, x[..., i]) (default v x) for the entries of a matrix
    (v (..., nnz): leading channels allowed), every output's in increasing i, as a sliced ELL: the outputs grouped by
    their entry count into buckets of power-of-two widths, adjacent buckets merged while the padded work stays within 4x
    the entries (one bucket, the plain ELL over the outputs up to the last nonzero one, whenever that holds); a gather, a
    product and a fixed-length sum per bucket, the buckets' results put in place by one gather (deterministic, fixed
    shapes; padding entries are index 0, value 0). K: the most entries of an output (the terms its sum rounds)."""

    def __init__(self, o, i, v, n_out):
        dev = v.device
        o, i, v = o.cpu(), i.cpu(), v.cpu()
        order = torch.argsort(o * (int(i.max()) + 1 if i.numel() else 1) + i)
        o, i, v = o[order], i[order], v[..., order]
        cnt = torch.bincount(o, minlength=n_out)
        nnz, self.n_out = len(o), n_out
        self.K = max(1, int(cnt.max())) if n_out else 1
        start = torch.cumsum(cnt, 0) - cnt
        pos = torch.arange(nnz) - start[o]
        last = int(torch.nonzero(cnt).max()) + 1 if nnz else 0
        if last * self.K <= 4 * nnz or nnz == 0:
            groups, self.inv = [(self.K, torch.arange(last))], None
        else:
            nz = torch.nonzero(cnt).reshape(-1)
            width = 2 ** torch.ceil(torch.log2(cnt[nz].double())).long()
            groups = [[int(w), nz[width == w]] for w in torch.unique(width).tolist()]
            work = lambda g: g[0] * len(g[1])
            while len(groups) > 1:
                inc = [groups[k + 1][0] * len(groups[k][1]) - work(groups[k]) for k in range(len(groups) - 1)]
                k = min(range(len(inc)), key=inc.__getitem__)
                if sum(map(work, groups)) + inc[k] > 4 * nnz:
                    break
                groups[k:k + 2] = [[groups[k + 1][0], torch.sort(torch.cat([groups[k][1], groups[k + 1][1]])).values]]
            slot = torch.full((n_out,), sum(len(g[1]) for g in groups), dtype=torch.long)
            off = 0
            for w, outs in groups:
                slot[outs] = off + torch.arange(len(outs))
                off += len(outs)
            self.inv = slot.to(dev)
        self.buckets = []
        for w, outs in groups:
            row = torch.full((n_out,), -1, dtype=torch.long)
            row[outs] = torch.arange(len(outs))
            sel = row[o] >= 0
            idx = torch.zeros(len(outs), w, dtype=torch.long)
            val = torch.zeros(*v.shape[:-1], len(outs), w, dtype=v.dtype)
            idx[row[o[sel]], pos[sel]] = i[sel]
            val[..., row[o[sel]], pos[sel]] = v[..., sel]
            self.buckets.append((idx.to(dev), val.to(dev)))

    def __call__(self, x, f=None):
        outs = [(x[..., idx] * val if f is None else f(val, x[..., idx])).sum(-1) for idx, val in self.buckets]
        if self.inv is None:
            return torch.nn.functional.pad(outs[0], (0, self.n_out - outs[0].shape[-1]))
        return torch.cat(outs + [outs[0].new_zeros(*outs[0].shape[:-1], 1)], -1)[..., self.inv]

    def to(self, dtype):
        out = _Reduce.__new__(_Reduce)
        out.__dict__.update(vars(self))
        out.buckets = [(idx, val.to(dtype)) for idx, val in self.buckets]
        return out


def _private(node):
    """a shallow copy of `node` with its own options dict, for its face code to run on without restoring its state
    (BoundActivation.faces restore=False: the copy holds the relaxation state; the node is left untouched)"""
    c = copy.copy(node)
    if isinstance(getattr(node, 'options', None), dict):
        c.options = dict(node.options)
    return c


def _form(e, shape):
    """_fold_error's (et (m, ...), ec (...)) of a node broadcast to `shape` (a face node's): (dM (E, m), dc (E,))"""
    et, ec = e
    m = et.shape[0]
    return et.broadcast_to(m, *shape).reshape(m, -1).t(), ec.broadcast_to(1, *shape).reshape(-1)


class Mat:
    """A shared matrix M (R, C) of the compiled form, given dense (M) or by its nonzeros (coo = rows, cols, vals and its
    shape), for x @ M (`left`, over its rows) and x @ M^T (`right`, over its columns): dense GEMMs, or (sparse) by its
    nonzeros as sliced ELL (_Reduce), never densified. K_left, K_right: the number of terms summed per output entry
    (the certificate's rounding count)."""

    def __init__(self, M=None, sparse=False, coo=None, shape=None):
        self.shape = tuple(M.shape) if M is not None else tuple(shape)
        self.sparse = sparse and self.shape[0] * self.shape[1] > 0        # an empty product stays a (zero) GEMM
        self.K_left, self.K_right = self.shape
        if self.sparse:
            r, c, v = _coo(M) if coo is None else coo
            self.cl, self.cr = _Reduce(c, r, v, self.shape[1]), _Reduce(r, c, v, self.shape[0])
            self.K_left, self.K_right = self.cl.K, self.cr.K
        else:
            if M is None:
                r, c, v = coo
                M = torch.zeros(self.shape, dtype=v.dtype, device=v.device).index_put_((r, c), v)
            self.M, self.Ma = M, M.abs()

    def left(self, x):
        return self.cl(x) if self.sparse else x @ self.M

    def right(self, x):
        return self.cr(x) if self.sparse else x @ self.M.t()

    def left_abs(self, x, xa):
        """(x @ M, xa @ |M|) (sparse: one gather of both)"""
        if not self.sparse:
            return x @ self.M, xa @ self.Ma
        out = self.cl(torch.stack([x, xa]), lambda v, g: g * torch.stack([v, v.abs()]).reshape(
            2, *[1] * (x.ndim - 1), *v.shape))
        return out[0], out[1]

    def price_rows(self, lo, hi):
        """sum_j min(M_ij lo_j, M_ij hi_j) of every row i on the boxes lo, hi (B, C): (B, R)"""
        if not self.sparse:
            return _price_shared(self.M, lo, hi)
        return self.cr(torch.stack([lo, hi]), lambda v, g: torch.fmin(v * g[0], v * g[1]).nan_to_num_(nan=0.0))

    def to(self, dtype):
        out = Mat.__new__(Mat)
        out.__dict__.update({k: v.to(dtype) if isinstance(v, _Reduce) or torch.is_tensor(v) and v.is_floating_point()
                             else v for k, v in vars(self).items()})
        return out


class _Stage:
    """A level crossed by a Plan: lev; la (Q, KA) the level's slots among the entering ones (flat indices into the
    (Q, K) slots, padding 0), le (Q, KA) their elements within the level, lm (Q, KA) real; red the slots after the
    level (Q K1) from [the entering slots (Q K) | the slopes (Q, KA, 2 operands)]: the slots below the level carried
    (value 1), the slopes through the operand forms' values"""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def to(self, dtype):
        return _Stage(**{k: v.to(dtype) if isinstance(v, _Reduce) else v for k, v in vars(self).items()})


class Plan:
    """The sign backward (crown) of Q rows over the compiled columns shared by the boxes, slot by slot
    (CompiledGraph._plan builds it, _plan_bound runs it). A row's coefficient lives on its slots: the columns it reaches
    (structurally, box independent), K of them per row (padded). Entering level `top` they are the row's own nonzeros
    (g0 (Q K,)); every stage (_Stage, levels top, ..., 0) passes the slots on its level through their elements' default
    relaxations (the slopes into the operand forms' columns, the constants and residuals priced on the node boxes) and
    carries the others; the remaining slots are inputs (cols (Q, K), cm real) priced on the input box. const (Q,): the
    rows' constants. The work is that of the rows' nonzeros and their reach, not of B Q ncol."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def to(self, dtype):
        cast = lambda v: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
        return Plan(**{k: [s.to(dtype) for s in v] if k == 'stages' else cast(v) for k, v in vars(self).items()})


# a level's view (B, NF, E): the default relaxation of every element in column terms, side-major (lower, upper):
# slopes dw on its 2 operands (0..3), constants dc (4, 5), summed weights ds (6, 7), and its node box lo, hi (8, 9);
# margins also the magnitudes: sum_f |d_f| |w_f| (10..13), sum_f |d_f| (|b_f| + sum_p |w_fp c_p|) (14, 15), sum_f |d_f|
# (16, 17), sum_p (sum_f |d_f| |w_fp|) dc_p with dc the operand constants' error bounds (18, 19)
NF = 10


def _unpack(V):
    """a level's view as crown's (dw (B, 1, 2, 2, E), dc (B, 1, 2, E), ds (B, 1, 2, E), lo (B, 1, E), hi)"""
    B, E = V.shape[0], V.shape[-1]
    return (V[:, 0:4].reshape(B, 1, 2, 2, E), V[:, 4:6].reshape(B, 1, 2, E), V[:, 6:8].reshape(B, 1, 2, E),
            V[:, 8:9], V[:, 9:10])


class CompiledGraph:
    """The compiled graph of `net` for B boxes [x_L, x_U] (B, *input shape), the spec C (S, n_out), thresholds thr (S,)
    or (B, S), output rows `rows` (indices >= 1 of C; default: those with a finite thr), base rows E (R, n), h (R,) or
    (B, R). The first Data (self.data) comes from `cache` (face_dual.per_output's {name: Faces}: its faces and node
    intervals; the canonical layout is then pivot_crown.Graph's), else from `interm_bounds` (frozen intermediate
    bounds, the faces by the operators' own code, every face kept), else from the compiled CROWN pass (`bound`).
    K, free_tangent: the face pool of the tangent-point operators (face_dual's). sparse: every product with a shared
    matrix (the operand forms, the stacked rows, C M_out, E) over its nonzeros (Mat, sliced ELL) instead of a dense
    GEMM, no dense matrix formed, and bound()'s backward slot by slot (Plan). margins (default: sparse; needs sparse):
    the compiled pass's own intermediate bounds, node intervals and BoundMul faces made rigorous (_plan_bound, _node_faces,
    _mul_valid): its bounds and faces hold for the exact graph, so certify's LP is a relaxation of the exact graph on the
    box (A68). The graph's dtype is the boxes' (x_L): the probe runs in the network's own precision (its parameters') and
    a lower one is reached as by to(dtype)."""

    def __init__(self, net, x_L, x_U, C, thr=None, E=None, h=None, rows=None, cache=None, interm_bounds=None, K=5,
                 free_tangent=False, sparse=False, margins=None, trig_tangents=0, memory=None):
        self.memory = memory
        self.net, self.K, self.free_tangent, self.sparse = net, K, free_tangent, sparse
        self.trig_tangents = int(trig_tangents)
        self.operand_caps = None
        self.margins = sparse if margins is None else bool(margins)
        if self.margins and not sparse:
            raise ValueError('compiled_crown: margins need sparse=True (the rigorous backward runs on the Plans)')
        self.nf = 2 * NF if self.margins else NF
        want, dev = x_L.dtype, x_L.device
        # (the probe and the fold in the network's precision; the graph lowered to the boxes' dtype before any data)
        dt = next((q.dtype for q in net.parameters() if q.is_floating_point()), want)
        dt = dt if _P.get(dt, 0) > _P.get(want, 0) else want
        x_L, x_U = x_L.to(dt), x_U.to(dt)
        C, thr = C.to(dt), (None if thr is None else thr.to(dt))
        E, h = (None if E is None else E.to(dt)), (None if h is None else h.to(dt))
        self.in_shape = tuple(x_L.shape[1:])
        self.n = n = math.prod(self.in_shape)
        names = list(cache) if cache is not None else face_nodes(net)
        from .pivot_crown import _nonlinear
        missing = [nd.name for nd in net.nodes() if nd.perturbed and _nonlinear(nd) and nd.name not in names]
        if missing:
            raise ValueError(f'nonlinear nodes without faces: {missing}')
        self.S = S = C.shape[0]
        if rows is None:
            rows = [] if thr is None else [k for k in range(1, S) if bool(torch.isfinite(thr.reshape(-1, S)[:, k]).all())]
        self.rows = torch.as_tensor(rows, dtype=torch.long, device=dev).reshape(-1)
        self.R = 0 if E is None else E.shape[0]
        # ---- the operand forms of every face node and the output's (canonical columns) by their nonzeros, with the
        # fold's error bounds, by the probe
        shapes = {k: tuple(net[k].output_shape[1:]) for k in names}
        Es = [math.prod(shapes[k]) for k in names]
        self.Et = sum(Es)
        self.ncol = ncol = n + self.Et
        forms, out_form = self._probe(names, shapes, ncol, dt, dev)
        nodes, c0 = [], n
        for k, Ek in zip(names, Es):
            node, P = net[k], len(forms[k])
            nodes.append(Node(name=k, E=Ek, shape=shapes[k], c0=c0, P=P, same=P == 2 and node.inputs[0] is node.inputs[1],
                              operands=[i.name for i in node.inputs[:P]]))
            c0 += Ek
        owner = torch.zeros(ncol, dtype=torch.long)
        for i, nd in enumerate(nodes):
            owner[nd.c0:nd.c0 + nd.E] = i + 1
        level = {}

        def lev(i):
            # (the forms' pattern and their error bounds': a term that cancels exactly still depends on its column)
            if i not in level:
                used = torch.cat([f[key][1] for f in forms[nodes[i].name] for key in ('M', 'dM')]).cpu()
                deps = set(owner[used].tolist()) - {0}
                level[i] = 1 + max([lev(j - 1) for j in deps], default=0)
            return level[i]
        operand_of = {o for nd in nodes for o in nd.operands}
        for i, nd in enumerate(nodes):
            nd.level, nd.own = lev(i), nd.name in operand_of
        # ---- compiled order: level-major, stable
        self.nodes = sorted(nodes, key=lambda nd: nd.level)
        s, levels = 0, []
        for nd in self.nodes:
            nd.s = s
            s += nd.E
            if levels and levels[-1][2] == nd.level:
                levels[-1][1] = s
            else:
                levels.append([nd.s, s, nd.level])
        self.levels = [(a, b) for a, b, _ in levels]
        # per level, its BoundMul nodes of the same options (their faces and IBP elementwise, one call for all of them:
        # fewer kernels) and every other node alone: (members, the group as one node (its elements side by side), their
        # elements' positions in the level)
        from .operators import BoundMul
        key = lambda nd: (nd.same, net[nd.name]._is_softmax(), *(getattr(net[nd.name], k, None) for k in (
            'coeff_aware', 'adaptive', 'middle', 'output_dtype', 'coeff_aware_exact')))
        self.groups = []
        for a, b in self.levels:
            grp = {}
            for nd in self.nodes:
                if a <= nd.s < b:
                    grp.setdefault(key(nd) if isinstance(net[nd.name], BoundMul) else nd.name, []).append(nd)
            self.groups.append([])
            for m in grp.values():
                idx = torch.cat([torch.arange(x.s - a, x.s - a + x.E) for x in m]).to(dev)
                one = Node(name=m[0].name, E=len(idx), shape=(len(idx),), P=2, same=m[0].same) if len(m) > 1 else m[0]
                self.groups[-1].append((m, one, idx))
        perm = torch.cat([torch.arange(n)] + [torch.arange(nd.c0, nd.c0 + nd.E) for nd in self.nodes]).to(dev)
        self.k2c, self.c2k = perm, torch.argsort(perm)     # compiled column j = canonical k2c[j]; inverse c2k
        # the operand forms (Mop, 2 x Et rows p Et + e, compiled columns) and their error bounds (dMop) by their
        # nonzeros; dense (sparse=False) they are stored so
        self.cop = torch.zeros(2, self.Et, dtype=dt, device=dev)
        self.dcop = torch.zeros_like(self.cop)
        ops, dops = [], []
        for nd in self.nodes:
            for p, f in enumerate(forms[nd.name]):
                for coo, (r, k, v) in ((ops, f['M']), (dops, f['dM'])):
                    coo.append((p * self.Et + nd.s + r, self.c2k[k], v))
                self.cop[p, nd.s:nd.s + nd.E] = f['c']
                self.dcop[p, nd.s:nd.s + nd.E] = f['dc']
        cat = lambda coo: tuple(torch.cat([t[j] for t in coo]) if coo else
                                torch.zeros(0, dtype=torch.long if j < 2 else dt, device=dev) for j in range(3))
        self._Mop = self._dMop = self._CM = self._dCM = None
        self.op, self.dop = cat(ops), cat(dops)
        if not sparse:
            self.Mop, self.dMop = self._dense(self.op), self._dense(self.dop)
        # C (M_out, c_out); it rounds once more: |C M* - fl(C M)| <= |C| dM + gamma_K |C| |M| (K = n_out terms)
        n_out = math.prod(net[net.output_name[0]].output_shape[1:])
        K, Ca = C.shape[1], C.abs()
        g = _gamma(K, dt)
        self.Cc = C @ out_form['c']                                                            # (S,)
        Z = out_form['dc'] + g * out_form['c'].abs()
        self.dCc = torch.where((Ca != 0).to(dt) @ (Z != 0).to(dt) > 0, _upper(Ca @ Z, K + 2, 3 * K), 0.0)
        Mo, dMo = ((r, self.c2k[k], v) for r, k, v in (out_form['M'], out_form['dM']))
        if sparse:
            # by their nonzeros: C M_out, and |C| (dM + gamma |M|) (every entry of it a sum of positive terms)
            r, k, v = _spmm(_coo(C), Mo, ncol)
            nz = v != 0
            self._cmc = (r[nz].to(dev), k[nz].to(dev), v[nz].to(dev))
            r, k, vm, vd = _merge((Mo[0], Mo[1], g * Mo[2].abs()), dMo, ncol)
            r, k, v = _spmm(_coo(Ca), (r, k, vd + vm), ncol)
            self._dcmc = (r.to(dev), k.to(dev), _upper(v, K + 2, 3 * K).to(dev))
        else:
            dense = lambda coo: torch.zeros(n_out, ncol, dtype=dt, device=dev).index_put_(coo[:2], coo[2])
            Mout, dMout = dense(Mo), dense(dMo)
            self.CM = C @ Mout                                                                  # (S, ncol)
            Z = dMout + g * Mout.abs()
            self.dCM = torch.where((Ca != 0).to(dt) @ (Z != 0).to(dt) > 0, _upper(Ca @ Z, K + 2, 3 * K), 0.0)
            del Mout, dMout, Z
        del forms, out_form
        self.E = torch.zeros(0, n, dtype=dt, device=dev) if E is None else E.to(dt)
        self.dE = None                                   # the base rows' error bounds (a copy in a lower precision: to())
        if want != dt:
            self._lower(want)
        self._mats()
        self.fill = {'nnz_folded': len(self._op()[0]) + len(self._cm()[0]), 'dense_folded': 2 * self.Et * ncol + S * ncol}
        self._privates()
        # ---- the first data
        if cache is not None:
            self.data = self._from_cache(cache, x_L, x_U, thr, h)
        elif interm_bounds is not None:
            self.data = self.frozen(x_L, x_U, interm_bounds, thr, h)
        else:
            self.data, self.out = self.bound(x_L, x_U, thr, h)
        self._row_maps()

    def with_base_rows(self, E, h):
        """Reuse the compiled expression, rebuilding only changed base-row maps.

        The caller establishes identical symbolic outputs and derived rows. Old
        CUDA graphs keep their own graph object and are never rebound in place.
        """
        import copy
        cg = copy.copy(self)
        cg.E = self.cop.new_zeros(0, self.n) if E is None else E.to(self.cop).clone()
        cg.R, cg.dE = len(cg.E), None
        cg._mats()
        cg._row_maps()
        return cg

    # ------------------------------------------------------------------------------------------------ compilation
    # Mop, dMop (2, Et, ncol): the operand forms and their error bounds; CM, dCM (S, ncol): C M_out and its error
    # bounds. Dense (sparse=False) they are stored so; else by their nonzeros (op, dop: rows p Et + e, columns,
    # values; _cmc, _dcmc) and densified only when read (densify, tests). Assigning a dense one replaces them.
    @property
    def Mop(self):
        return self._Mop if self._Mop is not None else self._dense(self.op)

    @Mop.setter
    def Mop(self, M):
        self._Mop, self.op = M, None

    @property
    def dMop(self):
        return self._dMop if self._dMop is not None else self._dense(self.dop)

    @dMop.setter
    def dMop(self, M):
        self._dMop, self.dop = M, None

    @property
    def CM(self):
        return self._CM if self._CM is not None else self._dense(self._cmc, (self.S, self.ncol))

    @CM.setter
    def CM(self, M):
        self._CM, self._cmc = M, None

    @property
    def dCM(self):
        return self._dCM if self._dCM is not None else self._dense(self._dcmc, (self.S, self.ncol))

    @dCM.setter
    def dCM(self, M):
        self._dCM, self._dcmc = M, None

    def _dense(self, coo, shape=None):
        r, c, v = coo
        shape = (2, self.Et, self.ncol) if shape is None else shape
        return torch.zeros(math.prod(shape[:-1]), shape[-1], dtype=v.dtype, device=v.device).index_put_(
            (r, c), v).reshape(shape)

    def _op(self, d=False):
        """the nonzeros (rows p Et + e, columns, values) of Mop (d: dMop)"""
        coo, M = (self.dop, self._dMop) if d else (self.op, self._Mop)
        return coo if M is None else _coo(M.reshape(2 * self.Et, self.ncol))

    def _cm(self, d=False):
        """the nonzeros of CM (d: dCM)"""
        coo, M = (self._dcmc, self._dCM) if d else (self._cmc, self._CM)
        return coo if M is None else _coo(M)

    def _mats(self):
        """the shared matrices derived from Mop, CM, Cc, E (rebuilt when they change)"""
        if self.sparse:
            return self._mats_sparse()
        n, Et, ncol = self.n, self.Et, self.ncol
        dt, dev = self.cop.dtype, self.cop.device
        self.MopF = self.Mop.reshape(2 * Et, ncol)
        self.CMr, self.Ccr, self.CM0 = self.CM[self.rows], self.Cc[self.rows], self.CM[0]
        # every row's linear part over the columns, stacked: [operand forms (2 Et) | the face nodes' own columns (Et) |
        # output rows | base rows (inputs only)], one product for F_A and R_A
        unit = torch.zeros(Et, ncol, dtype=dt, device=dev)
        unit[torch.arange(Et), n + torch.arange(Et)] = 1.0
        Epad = torch.cat([self.E, torch.zeros(self.R, Et, dtype=dt, device=dev)], 1)
        self.Mstack = torch.cat([self.MopF, unit, self.CMr, Epad])
        sp = self.sparse
        self.Lstack, self.Lcm, self.Lop, self.Lcmr, self.LE = (Mat(M, sp) for M in (self.Mstack, self.CM, self.MopF,
                                                                                      self.CMr, self.E))
        self.Llev = [Mat(self.Mop[:, a:b, :n + a].reshape(2 * (b - a), n + a).contiguous(), sp) for a, b in self.levels]
        # level 1's operand bounds: the operand forms (both signs) priced on the input box
        a, b = self.levels[0] if self.levels else (0, 0)
        M0 = self.Mop[:, a:b, :n]
        self.Lspec0 = Mat(torch.cat([M0, -M0]).reshape(4 * (b - a), n), sp)
        # the magnitudes the certificate's error bounds use
        self.Mcol = self.MopF.abs().sum(0)
        # the fold's error bounds of the operand forms and the rows of C, stacked (_fold_slack)
        self.dG = torch.cat([self.dMop.reshape(2 * Et, ncol), self.dCM])
        self.dg = torch.cat([self.dcop.reshape(-1), self.dCc])

    def _mats_sparse(self):
        """_mats from the nonzeros (sparse=True): every shared matrix as a sliced ELL built from Mop's, CM's, E's
        nonzeros (none densified), and bound()'s Plans (_plans)"""
        n, Et, ncol, nr, R, S = self.n, self.Et, self.ncol, len(self.rows), self.R, self.S
        dt, dev = self.cop.dtype, self.cop.device
        Sp = lambda coo, shape: Mat(coo=coo, shape=shape, sparse=True)
        r, c, v = self._op()
        cm = self._cm()
        cmr = _rows(cm, self.rows)
        self.Ccr = self.Cc[self.rows]
        self.CM0 = torch.zeros(ncol, dtype=dt, device=dev).index_put_((cm[1][cm[0] == 0],), cm[2][cm[0] == 0])
        ee, er = _coo(self.E), torch.arange(Et, device=dev)
        self.Lstack = Sp((torch.cat([r, 2 * Et + er, 3 * Et + cmr[0], 3 * Et + nr + ee[0]]),
                          torch.cat([c, n + er, cmr[1], ee[1]]),
                          torch.cat([v, torch.ones(Et, dtype=dt, device=dev), cmr[2], ee[2]])), (3 * Et + nr + R, ncol))
        self.Lcm, self.Lcmr, self.LE = Sp(cm, (S, ncol)), Sp(cmr, (nr, ncol)), Mat(self.E, True)
        self.Lop = Sp((r, c, v), (2 * Et, ncol))
        p, e = r // Et, r % Et
        self.Llev = []
        # (a level's operand forms reach the columns below it only: the dense path's slices)
        for a, b in self.levels:
            m = (e >= a) & (e < b) & (c < n + a)
            self.Llev.append(Sp((p[m] * (b - a) + e[m] - a, c[m], v[m]), (2 * (b - a), n + a)))
        self.Mcol = torch.zeros(ncol, dtype=dt).index_add_(0, c.cpu(), v.abs().cpu()).to(dev)
        dr, dc_, dv = self._op(d=True)
        dcm = self._cm(d=True)
        dG = (torch.cat([dr, 2 * Et + dcm[0]]), torch.cat([dc_, dcm[1]]), torch.cat([dv, dcm[2]]))
        self.LdG, self.LdGi = Sp(dG, (2 * Et + S, ncol)), Sp((*dG[:2], torch.ones_like(dG[2])), (2 * Et + S, ncol))
        self.dg = torch.cat([self.dcop.reshape(-1), self.dCc])
        self._plans()

    def _plans(self):
        """bound()'s Plans: per level its operand forms, both signs (plan_lev; level 0's: on the inputs, no level
        crossed), per level its own nodes' unit rows, both signs (plan_own, None without), the rows of C (plan_out),
        the objective (plan_obj)"""
        n, Et, ncol, dt, dev = self.n, self.Et, self.ncol, self.cop.dtype, self.cop.device
        r, c, v = self._op()
        dr, dc_, dv = self._op(d=True)
        # Mop by row (and margins: on the union of its pattern and its error bounds', the error bound a 2nd value)
        if self.margins:
            rr, cc, vm, vd = _merge((r, c, v), (dr, dc_, dv), ncol)
        else:
            o = torch.argsort(r.cpu(), stable=True)
            rr, cc, vm = r.cpu()[o], c.cpu()[o], v.cpu()[o]
            vd = torch.zeros_like(vm)
        cnt = torch.bincount(rr, minlength=2 * Et)
        self._opcsr = (torch.cat([cnt.new_zeros(1), torch.cumsum(cnt, 0)]), cc, vm, vd)
        p, e = r // Et, r % Et
        dp, de = dr // Et, dr % Et
        self.plan_lev, self.plan_own = [], []
        for lev, (a, b) in enumerate(self.levels):
            E = b - a
            m = (e >= a) & (e < b) & (c < n + a)
            q = p[m] * E + e[m] - a
            const = torch.cat([self.cop[:, a:b], -self.cop[:, a:b]]).reshape(4 * E)
            md = (de >= a) & (de < b) & (dc_ < n + a)
            qd = dp[md] * E + de[md] - a
            dq = (torch.cat([qd, 2 * E + qd]), torch.cat([dc_[md], dc_[md]]), torch.cat([dv[md], dv[md]]))
            self.plan_lev.append(self._plan(torch.cat([q, 2 * E + q]), torch.cat([c[m], c[m]]),
                                            torch.cat([v[m], -v[m]]), 4 * E, lev - 1, const, dq,
                                            torch.cat([self.dcop[:, a:b]] * 2).reshape(4 * E)))
            cols = [torch.arange(n + nd.s, n + nd.s + nd.E) for nd in self.nodes if nd.own and a <= nd.s < b]
            if cols:
                k = torch.cat(cols)
                i, one = torch.arange(len(k)), torch.ones(len(k), dtype=dt)
                self.plan_own.append(self._plan(torch.cat([i, len(k) + i]), torch.cat([k, k]), torch.cat([one, -one]),
                                                2 * len(k), lev, torch.zeros(2 * len(k), dtype=dt, device=dev)))
            else:
                self.plan_own.append(None)
        cm, dcm = self._cm(), self._cm(d=True)
        top, zero = len(self.levels) - 1, torch.zeros(1, dtype=torch.long)
        self.plan_out = self._plan(*cm, self.S, top, self.Cc, dcm, self.dCc)
        self.plan_obj = self._plan(*_rows(cm, zero), 1, top, self.Cc[:1], _rows(dcm, zero), self.dCc[:1])

    def _plan(self, q, c, v, Q, top, const, dq=None, dconst=None):
        """the Plan of the Q rows with nonzeros (q, c, v) (compiled columns, all below the end of level `top`; top -1:
        on the inputs) and constants const (Q,), crossing levels top, ..., 0 (CPU, once per graph): per stage the
        slots after it are the carried ones and the operand forms' nonzeros of its slots' elements (Mop by row,
        _opcsr). margins: dq the rows' coefficients' error bounds by their nonzeros, dconst (Q,) their constants'; the
        slots cover both patterns, and every stage's reduction carries the error bounds as a second channel"""
        n, Et, ncol, dt, dev, mg = self.n, self.Et, self.ncol, self.cop.dtype, self.cop.device, self.margins
        ptr, ocol, oval, oerr = self._opcsr
        # (the dense path's slices: a row's columns up to the end of level top, a level's forms' below it)
        end = n + (self.levels[top][1] if top >= 0 else 0)
        cut = lambda q, c, v: (lambda k: (q.cpu()[k], c.cpu()[k], v.cpu().to(dt)[k]))(c.cpu() < end)
        q, c, v = cut(q, c, v)
        dq = cut(*dq) if mg and dq is not None else (q[:0], c[:0], v[:0])
        keys = torch.unique(torch.cat([q * ncol + c, dq[0] * ncol + dq[1]]))
        uq, uc = keys // ncol, keys % ncol
        pos, K = _layout(uq, Q)
        at0 = lambda qq, cc_: (lambda i: uq[i] * K + pos[i])(torch.searchsorted(keys, qq * ncol + cc_))
        g0 = torch.zeros(Q * K, dtype=dt).index_add_(0, at0(q, c), v)
        e0 = torch.zeros(Q * K, dtype=dt).index_add_(0, at0(dq[0], dq[1]), dq[2])
        stages, kred, nterms, nprod = [], 0, 0, 0
        for s in range(top, -1, -1):
            a, b = self.levels[s]
            lvl = uc >= n + a
            lq, le = uq[lvl], uc[lvl] - n - a
            kpos, KA = _layout(lq, Q)
            la, lem, lm = (torch.zeros(Q, KA, dtype=t) for t in (torch.long, torch.long, torch.bool))
            la[lq, kpos], lem[lq, kpos], lm[lq, kpos] = lq * K + pos[lvl], le, True
            # the targets: both operand forms' nonzeros of every level slot's element, from the slope (row, slot, p)
            t, pp = torch.arange(len(lq)).repeat(2), torch.arange(2).repeat_interleave(len(lq))
            orow = pp * Et + a + le[t]
            k = ptr[orow + 1] - ptr[orow]
            tt = torch.repeat_interleave(torch.arange(len(orow)), k)
            j = ptr[orow][tt] + torch.arange(len(tt)) - (torch.cumsum(k, 0) - k)[tt]
            below = ocol[j] < n + a
            tt, j = tt[below], j[below]
            tq, tc = lq[t][tt], ocol[j]
            tin = Q * K + (tq * KA + kpos[t][tt]) * 2 + pp[tt]
            cq, cc, cin = uq[~lvl], uc[~lvl], uq[~lvl] * K + pos[~lvl]
            nk = torch.unique(torch.cat([cq * ncol + cc, tq * ncol + tc]))
            nuq, nuc = nk // ncol, nk % ncol
            npos, K1 = _layout(nuq, Q)
            if self.memory is not None:
                # The query rows share a padded slot width. A single dense
                # query can therefore make an otherwise sparse plan enormous.
                # Charge views, coefficient/error channels and endpoint work
                # before transferring those plans or running the first bound.
                self.memory.require(torch.empty((), dtype=dt).element_size()
                                    * (self.nf + 20) * Q * max(K, KA, K1))
            at = lambda qq, cc_: (lambda i: nuq[i] * K1 + npos[i])(torch.searchsorted(nk, qq * ncol + cc_))
            one = torch.ones(len(cq), dtype=dt)
            vals = torch.cat([one, oval[j].to(dt)])
            if mg:      # (the carried error bounds: weight 1 + gamma, _plan_bound)
                vals = torch.stack([vals, torch.cat([one, oerr[j].to(dt)])])
            red = _Reduce(torch.cat([at(cq, cc), at(tq, tc)]), torch.cat([cin, tin]), vals.to(dev), Q * K1)
            stages.append(_Stage(lev=s, la=la.to(dev), le=lem.to(dev), lm=lm.to(dev), red=red, K=K, KA=KA, K1=K1))
            kred, nterms, nprod = kred + red.K, nterms + 2 * KA + 1, nprod + 4 * KA
            uq, uc, pos, K = nuq, nuc, npos, K1
        assert bool((uc < n).all()), 'compiled_crown: a plan reaches a column above its levels'
        cols, cm = torch.zeros(Q, K, dtype=torch.long), torch.zeros(Q, K, dtype=torch.bool)
        cols[uq, pos], cm[uq, pos] = uc, True
        return Plan(Q=Q, top=top, g0=g0.to(dev), stages=stages, cols=cols.to(dev), cm=cm.to(dev), const=const.to(dev),
                    kred=kred, nterms=nterms + K + 2, nprod=nprod + 2 * K + 4, e0=e0.to(dev) if mg else None,
                    dconst=(torch.zeros_like(const) if dconst is None else dconst).to(dev) if mg else None)

    def _probe(self, names, shapes, ncol, dt, dev):
        """the operand forms of every face node (broadcast to its shape) and the output's, in canonical columns, by
        their nonzeros, with _fold_error's bounds of their rounding: {face node: [{'M': (rows, cols, vals), 'c': (E,),
        'dM': (rows, cols, vals), 'dc': (E,)} per operand]} and the output's {...}. The probe (_probe_chunk) runs over
        structurally orthogonal colour groups, each chunk at most PROBE_ELEMS elements of all node values: no dense
        (E, ncol) form is held at once. Decompression retains the original column chunk order. Every chunk's primal
        is the same (the constants: the first chunk's)."""
        from .probe_pattern import ProbePattern
        net = self.net
        pattern = ProbePattern(self, names, shapes, ncol)
        colors = torch.as_tensor(pattern.colors, device=dev)
        size = lambda nd: math.prod(nd.output_shape[1:]) if getattr(nd, 'output_shape', None) is not None else 1
        per = self.n + sum(size(nd) for nd in net.nodes() if nd.perturbed)
        width = max(1, min(ncol, PROBE_ELEMS // max(1, per)))
        m = min(width, pattern.ncolors)
        recovery = pattern.sparse_recovery(dt, dev) if pattern.ncolors > width else None
        out = net[net.output_name[0]]
        oshape = tuple(out.output_shape[1:])
        maps = {}
        def nz(X, name, shape, j0):
            key = (name, shape)
            if key not in maps:
                maps[key] = tuple(torch.as_tensor(v, dtype=torch.long, device=dev)
                                  for v in pattern.form(name, shape))
            r, k = maps[key]
            col = colors[k] - j0
            take = (col >= 0) & (col < X.shape[1])
            r, k, col = r[take], k[take], col[take]
            v = X[r, col]
            take = v != 0
            return r[take], k[take], v[take]
        def coefficients(X, name, shape, j0, error=False):
            if recovery is None:
                return nz(X, name, shape, j0)
            return pattern.recovered_form(recovered, name, shape, error)

        acc = {k: None for k in names}
        oacc = None
        for j0 in range(0, m if recovery is not None else pattern.ncolors, m):
            check_deadline()
            lanes = min(m, pattern.ncolors - j0)
            small = recovery is not None and dt == torch.float64 and lanes > 1
            fc, oc, vals = self._probe_chunk(names, shapes, j0, 1 if small else lanes, dt, dev, colors)
            err = self._fold_error(names, vals)
            if small and not pattern.singleton_constants(vals, err):
                del fc, oc, vals, err
                fc, oc, vals = self._probe_chunk(names, shapes, j0, lanes, dt, dev, colors)
                err = self._fold_error(names, vals)
            recovered = pattern.recover(recovery, vals, dt, dev) if recovery is not None else None
            del vals
            for k in names:
                parts = []
                for (M, c), name in zip(fc[k], (i.name for i in net[k].inputs)):
                    dM, dc = _form(err[name], shapes[k])
                    parts.append({'M': [coefficients(M, name, shapes[k], j0)], 'c': c,
                                  'dM': [coefficients(dM, name, shapes[k], j0, True)], 'dc': dc})
                if acc[k] is None:
                    acc[k] = parts
                else:
                    for a, b in zip(acc[k], parts):
                        a['M'] += b['M']
                        a['dM'] += b['dM']
            dM, dc = _form(err[out.name], oshape)
            part = {'M': [coefficients(oc[0], out.name, oshape, j0)], 'c': oc[1],
                    'dM': [coefficients(dM, out.name, oshape, j0, True)], 'dc': dc}
            if oacc is None:
                oacc = part
            else:
                oacc['M'] += part['M']
                oacc['dM'] += part['dM']
            del fc, oc, err
        def join(f):
            for key in ('M', 'dM'):
                r, k, v = (torch.cat([t[j] for t in f[key]]) for j in range(3))
                # Preserve the original dense probe's chunk-major, then row /
                # column order, including when the colour chunks differ.
                order = torch.argsort(((k // width) * len(f['c']) + r) * ncol + k)
                f[key] = r[order], k[order], v[order]
            return f
        return {k: [join(f) for f in acc[k]] for k in names}, join(oacc)

    def _probe_chunk(self, names, shapes, j0, m, dt, dev, colors=None):
        """{face node: [(M (E, m), c (E,)) per operand, broadcast to the node's shape]} and the output's (M, c) on the
        canonical columns j0, ..., j0 + m - 1: one forward-mode pass at batch m (x = 0 and every face node 0, the
        tangents the colour indicators when colors is supplied, else the unit vectors of those columns); and every
        evaluated perturbed node's {name: (tangent (m, ...),
        primal (m, ...), its inputs' values (None where perturbed))} (_fold_error). The nodes' forward state is restored
        afterwards."""
        from .operators import BoundMul
        net, n = self.net, self.n
        saved = {nd.name: (nd.forward_value, vars(nd).get('input_shape')) for nd in net.nodes()}
        pert = {nd.name for nd in net.nodes() if nd.perturbed} | {net.input_name[0]}
        binding = [(net[k], net[k].value, net[k].perturbation) for k in net.input_name]   # (the probe clears it)

        def unit(off, E):
            # rows j0 .. j0 + m - 1 of the identity's columns off .. off + E - 1
            t = torch.zeros(m, E, dtype=dt, device=dev)
            if colors is not None:
                col = colors[off:off + E] - j0
                k = torch.nonzero((col >= 0) & (col < m)).flatten()
                t[col[k], k] = 1.0
                return t
            lo = min(m, max(0, off - j0))
            k = torch.arange(lo, max(lo, min(m, off + E - j0)), device=dev)
            t[k, j0 + k - off] = 1.0
            return t
        forms, vals = {}, {}
        try:
            with torch.no_grad(), fwAD.dual_level():
                x = fwAD.make_dual(torch.zeros(m, n, dtype=dt, device=dev), unit(0, n))
                off = n
                for k in names:
                    E = math.prod(shapes[k])
                    z = fwAD.make_dual(torch.zeros(m, *shapes[k], dtype=dt, device=dev),
                                       unit(off, E).reshape(m, *shapes[k]))
                    vars(net[k])['forward'] = (lambda *a, _z=z: _z)
                    off += E
                net.set_input(x.reshape(m, *self.in_shape), clear_forward_only=True, reset_perturbed_nodes=False)

                def form(t, shape):
                    d = fwAD.unpack_dual(t)
                    tan = torch.zeros_like(d.primal) if d.tangent is None else d.tangent
                    return (tan.broadcast_to(m, *shape).reshape(m, -1).t().contiguous(),
                            d.primal.broadcast_to(m, *shape).reshape(m, -1)[0].clone())
                for k in names:
                    node = net[k]
                    P = 2 if isinstance(node, BoundMul) else 1
                    forms[k] = [form(net.get_forward_value(i), shapes[k]) for i in node.inputs[:P]]
                out = net[net.output_name[0]]
                out_form = form(net.get_forward_value(out), tuple(out.output_shape[1:]))
                for nd in net.nodes():
                    d = fwAD.unpack_dual(nd.forward_value) if nd.name in pert and torch.is_tensor(nd.forward_value) \
                        else None
                    if d is not None and d.tangent is not None:
                        vals[nd.name] = (d.tangent.clone(), d.primal.clone(),
                                         [None if i.name in pert else i.forward_value for i in nd.inputs])
        finally:
            for k in names:
                vars(net[k]).pop('forward', None)
            for nd in net.nodes():
                nd.forward_value, shape = saved[nd.name]
                if shape is not None:
                    nd.input_shape = shape
            for nd, value, ptb in binding:
                nd.value, nd.perturbation = value, ptb
        return forms, out_form, vals

    def _fold_error(self, names, vals):
        """{node: (et (ncol, ...), ec (...))} for every evaluated perturbed node: bounds of |exact - computed| of the
        probe's tangent (its form's coefficients) and primal (its constant), the exact form being the composition of
        the affine nodes in exact arithmetic, every constant (weights, biases, the non-perturbed nodes' values) taken as
        the exact number fp64 holds. The input and the face nodes are exact. An affine node y = L x + b (_fold_rule:
        its forward rounds every term at most k times) computed fl(L t) and fl(L c + b) from the operands' computed
        t, c: |L t* - fl(L t)| <= |L| (et + gamma_k |t|), |L c* + b - fl(L c + b)| <= |L| (ec + gamma_k |c|) +
        gamma_k |b| (Higham Lemma 3.1, any order, FMA or not; plus underflow). |L| (.) and |b| are the node's forward
        on the operands' magnitudes (as forward-mode tangents of a zero primal) with its value constants' magnitudes,
        one pass for the tangents and one for the constants (the probe's own batch and constants); rounded up by
        _upper."""
        net = self.net
        err = {}
        state = {nd.name: dict(vars(nd)) for nd in net.nodes() if nd.name in vals}
        try:
            with torch.no_grad(), fwAD.dual_level():
                def ev(nd):
                    if nd.name in err:
                        return err[nd.name]
                    t, c, consts = vals[nd.name]
                    if nd.name == net.input_name[0] or nd.name in names:
                        err[nd.name] = (torch.zeros_like(t), torch.zeros_like(c[0]))
                        return err[nd.name]
                    raw = [vals[i.name][0] if v is None else v for i, v in zip(nd.inputs, consts)]
                    fn, k, value = _fold_rule(nd, raw)
                    g = _gamma(k, t.dtype)

                    def run(Z, mag):
                        args = [v if v is not None else fwAD.make_dual(torch.zeros_like(z), z)
                                for v, z in zip(consts, Z)]
                        args = [mag(a) if p in value and torch.is_tensor(a) and a.is_floating_point() and
                                consts[p] is not None else a for p, a in enumerate(args)]
                        d = fwAD.unpack_dual(fn(*args))
                        return d.primal, (torch.zeros_like(d.primal) if d.tangent is None else d.tangent)
                    Zt, Zc = [], []
                    for i, v in zip(nd.inputs, consts):
                        if v is not None:
                            Zt.append(None)
                            Zc.append(None)
                            continue
                        ti, ci, _ = vals[i.name]
                        et, ec = ev(i)
                        Zt.append(et + g * ti.abs())
                        Zc.append((ec + g * ci[0].abs()).expand_as(ci))
                    b, Tt = run(Zt, torch.abs)
                    Tc = run(Zc, torch.abs)[1][0]
                    # the same map on the supports: an entry without a nonzero term is computed exactly (no
                    # rounding, no underflow): its bound stays 0
                    one = lambda t: (t != 0).to(t.dtype)
                    sb, st = run([None if z is None else one(z) for z in Zt], one)
                    sc = run([None if z is None else one(z) for z in Zc], one)[1][0] + sb[0]
                    # m: Z's product and sum, the node's k, the constant row's gamma |b|; P: the products per entry of
                    # this magnitude pass and of the node's own computation (their underflows)
                    m, P = k + 3, 4 * (k + 2)
                    err[nd.name] = (torch.where(st > 0, _upper(Tt, m, P), 0.0),
                                    torch.where(sc > 0, _upper(Tc + g * b[0], m, P), 0.0))
                    return err[nd.name]
                for name in vals:
                    ev(net[name])
        finally:
            for name, st in state.items():
                nd = net[name]
                vars(nd).clear()
                vars(nd).update(st)
        return err

    def _set_counts(self, counts):
        """the face counts {name: (Fl, Fu)}: padding and the masks vl (Fl, Et), vu (Fu, Et) of the real faces"""
        for nd in self.nodes:
            nd.Fl, nd.Fu = counts[nd.name]
        self.Fl, self.Fu = max([nd.Fl for nd in self.nodes], default=0), max([nd.Fu for nd in self.nodes], default=0)
        dev = self.cop.device
        self.vl = torch.zeros(self.Fl, self.Et, dtype=torch.bool, device=dev)
        self.vu = torch.zeros(self.Fu, self.Et, dtype=torch.bool, device=dev)
        for nd in self.nodes:
            self.vl[:nd.Fl, nd.s:nd.s + nd.E] = True
            self.vu[:nd.Fu, nd.s:nd.s + nd.E] = True

    def _row_maps(self):
        """the canonical rows as indices into the flat structured layout [lower (Fl, Et) | upper (Fu, Et) | output rows
        | base rows]"""
        Fl, Fu, Et = self.Fl, self.Fu, self.Et
        idx = []
        for nd in sorted(self.nodes, key=lambda nd: nd.c0):
            e = torch.arange(nd.s, nd.s + nd.E)
            idx += [j * Et + e for j in range(nd.Fl)] + [(Fl + j) * Et + e for j in range(nd.Fu)]
        base = (Fl + Fu) * Et
        idx += [base + torch.arange(len(self.rows)), base + len(self.rows) + torch.arange(self.R)]
        self.ridx = torch.cat(idx).to(self.cop.device)
        self.nrow, self.nflat = len(self.ridx), base + len(self.rows) + self.R

    # ------------------------------------------------------------------------------------------------ data
    def _data(self, x_L, x_U, faces, boxes, thr, h, undefined=None):
        """Data from per-node raw faces {name: (lw (Fl, B, P, E), lb (Fl, B, E), uw, ub, dl, du)} and node intervals
        {name: (l, u) (B, E)} (faces beyond a node's count: zero slopes, offsets and weights); undefined (B,) bool
        the boxes on which an operator's relaxation is undefined (_node_faces; default none)"""
        B, dt, dev = x_L.shape[0], self.cop.dtype, x_L.device
        z = lambda *s: torch.zeros(*s, dtype=dt, device=dev)
        Fl, Fu, Et = self.Fl, self.Fu, self.Et
        # (inputs of another dtype rounded outward: the boxes, thresholds and h only ever relax)
        x_L, x_U = _to(x_L, dt, False), _to(x_U, dt, True)
        lw, lb, dl, uw, ub, du = z(B, Fl, 2, Et), z(B, Fl, Et), z(B, Fl, Et), z(B, Fu, 2, Et), z(B, Fu, Et), z(B, Fu, Et)
        for nd in self.nodes:
            a, b = nd.s, nd.s + nd.E
            f = faces[nd.name]
            lw[:, :nd.Fl, :nd.P, a:b] = f[0].transpose(0, 1)
            lb[:, :nd.Fl, a:b] = f[1].transpose(0, 1)
            uw[:, :nd.Fu, :nd.P, a:b] = f[2].transpose(0, 1)
            ub[:, :nd.Fu, a:b] = f[3].transpose(0, 1)
            dl[:, :nd.Fl, a:b] = f[4].transpose(0, 1)
            du[:, :nd.Fu, a:b] = f[5].transpose(0, 1)
        lo = torch.cat([x_L.reshape(B, -1)] + [boxes[nd.name][0] for nd in self.nodes], 1)
        hi = torch.cat([x_U.reshape(B, -1)] + [boxes[nd.name][1] for nd in self.nodes], 1)
        thr = torch.full((B, self.S), INF, dtype=dt, device=dev) if thr is None else _to(thr, dt, True).reshape(
            -1, self.S).expand(B, self.S)
        h = torch.zeros(B, self.R, dtype=dt, device=dev) if h is None or not self.R else _to(h, dt, True).reshape(
            -1, self.R).expand(B, self.R)
        undefined = torch.zeros(B, dtype=torch.bool, device=dev) if undefined is None else undefined
        d = Data(lo=lo, hi=hi, lw=lw, lb=lb, uw=uw, ub=ub, dl=dl, du=du, thr=thr, h=h, undefined=undefined).derive(
            self.cop).rhs(self.rows, self.Ccr)
        d.slack = self._fold_slack(d)
        self._cert_b(d)
        return d

    def _from_cache(self, cache, x_L, x_U, thr, h):
        """Data from face_dual's Faces (F, B, *shape) per node: its (deduplicated) faces and node intervals"""
        B = x_L.shape[0]
        faces, boxes = {}, {}
        for nd in self.nodes:
            f = cache[nd.name]
            flat = lambda t: t.reshape(t.shape[0], B, nd.E)
            faces[nd.name] = (torch.stack([flat(w) for w in f.lw], 2), flat(f.lb),
                              torch.stack([flat(w) for w in f.uw], 2), flat(f.ub), flat(f.dl), flat(f.du))
            boxes[nd.name] = (f.l.reshape(B, nd.E), f.u.reshape(B, nd.E))
        self._set_counts({k: (f[1].shape[0], f[3].shape[0]) for k, f in faces.items()})
        return self._data(x_L, x_U, faces, boxes, thr, h)

    def _node_faces(self, nd, lo, up):
        """node nd's faces on its operand bounds lo, up (B, P, E) by the operator's own code (BoundMul: its choice by
        the exact corner-product rule; a unary node: on its private copy), every face kept, and its IBP interval;
        the operators' validity checks deferred (operators.base.deferred_checks: no host synchronisation): undefined
        (B,) the boxes on which one fails (the operator would raise). Returns (lw (Fl, B, P, E), lb, uw, ub, dl, du),
        (l, u) (B, E), undefined, in the graph's dtype. margins: the IBP interval rounded outward (BoundMul: every end
        one correctly rounded product, so one step out encloses the exact one; another operator: by _REL of its
        magnitude, the libm accuracy assumed) and, BoundMul, the faces made valid for the exact product on the
        operand box (_mul_valid, after the faces are rounded to the graph's dtype)."""
        from .operators import BoundMul
        node = self.net[nd.name]
        own = self._private.get(nd.name, node)
        B = lo.shape[0]
        shp = (B, *nd.shape)
        ops = [Interval(lo[:, p].reshape(shp), up[:, p].reshape(shp)) for p in range(nd.P)]
        if nd.same:
            ops[1] = ops[0]
        with deferred_checks() as bad:
            if isinstance(node, BoundMul):
                lower, upper, dl, du = own.faces(ops[0], ops[1], exact_ties=True)
            else:
                lower, upper, dl, du = own.faces(ops[0], self.K, self.free_tangent, restore=False)
            ivs = ops + [Interval(v, v) for v in (self.net.get_forward_value(i) for i in node.inputs[nd.P:])]
            with torch.no_grad():
                ibp = own.interval_propagate(*ivs)
        undefined = torch.zeros(B, dtype=torch.bool, device=lo.device)
        for m in bad:
            undefined = undefined | (m.reshape(B, -1).any(1) if m.ndim and m.shape[0] == B else m.any().expand(B))
        dt = lo.dtype
        flat = lambda t: torch.broadcast_to(t, shp).reshape(B, nd.E).to(dt)
        stack = lambda side: (torch.stack([torch.stack([flat(w[p]) for p in range(nd.P)], 1) for w, _ in side]),
                              torch.stack([flat(b) for _, b in side]))
        wt = lambda d: torch.broadcast_to(d, (d.shape[0], *shp)).reshape(d.shape[0], B, nd.E).to(dt)
        lw, lb, uw, ub = *stack(lower), *stack(upper)
        l, u = (torch.broadcast_to(t, shp).reshape(B, nd.E) for t in (ibp.lower, ibp.upper))
        if self.margins:
            if isinstance(node, BoundMul):
                box_lo, box_up = (torch.stack([t[:, 0], t[:, 0 if nd.same else 1]], 1) for t in (lo, up))
                lb, ub = _mul_valid(lw, lb, box_lo, box_up, True), _mul_valid(uw, ub, box_lo, box_up, False)
                l, u = _down(_to(l, dt, False)), _up(_to(u, dt, True))
            else:
                rel, eta = _REL[l.dtype], 4 * _ETA[l.dtype]
                l = torch.where(torch.isfinite(l), _down(l - _up(l.abs() * rel + eta)), l)
                u = torch.where(torch.isfinite(u), _up(u + _up(u.abs() * rel + eta)), u)
                l, u = _to(l, dt, False), _to(u, dt, True)
        return (lw, lb, uw, ub, wt(dl), wt(du)), (l.to(dt), u.to(dt)), undefined

    def frozen(self, x_L, x_U, interm_bounds, thr=None, h=None):
        """Data on frozen intermediate bounds {name: (lower, upper)}: each face node's faces on its operands' bounds by
        the operator's own code, every face kept; its interval IBP on them, intersected with its own bound when given
        (face_dual._interval)"""
        B = x_L.shape[0]
        faces, boxes, undefined = {}, {}, None
        for nd in self.nodes:
            lo, up = [], []
            for name in nd.operands:
                l, u = (x_L, x_U) if name == self.net.input_name[0] else interm_bounds[name]
                lo.append(_to(torch.broadcast_to(l, (B, *nd.shape)).reshape(B, nd.E), self.cop.dtype, False))
                up.append(_to(torch.broadcast_to(u, (B, *nd.shape)).reshape(B, nd.E), self.cop.dtype, True))
            faces[nd.name], (l, u), bad = self._node_faces(nd, torch.stack(lo, 1), torch.stack(up, 1))
            undefined = bad if undefined is None else undefined | bad
            own = interm_bounds.get(nd.name)
            if own is not None:
                l = torch.maximum(l, own[0].reshape(-1, nd.E).expand_as(l))
                u = torch.minimum(u, own[1].reshape(-1, nd.E).expand_as(u))
            boxes[nd.name] = _widen(l, u) if self.margins else (l, u)
        if not hasattr(self, 'Fl'):
            self._set_counts({k: (f[1].shape[0], f[3].shape[0]) for k, f in faces.items()})
        return self._data(x_L, x_U, faces, boxes, thr, h, undefined)

    # ------------------------------------------------------------------------------------------------ CROWN
    def _view(self, d, lev):
        """level lev's view (B, NF, E) from Data d: its elements' default relaxation and node boxes"""
        a, b = self.levels[lev]
        n = self.n
        sl = lambda t: t[..., a:b].transpose(0, 1)                                         # (F, B, [2,] E)
        return self._stack_view([(sl(w), sl(o), sl(wt)) for w, o, wt in ((d.lw, d.lb, d.dl), (d.uw, d.ub, d.du))],
                                self.cop[:, a:b], self.dcop[:, a:b], d.lo[:, n + a:n + b], d.hi[:, n + a:n + b])

    def _stack_view(self, sides, c, dc, lo, hi):
        """a view (B, nf, E) from both sides' faces (w (F, B, P, E), offsets o (F, B, E), weights (F, B, E)), the
        operand constants c (P, E) and their error bounds dc, the node boxes lo, hi (B, E): one stack (the sums over
        faces and operands unrolled, _fsum: all pointwise)"""
        f, extra = [None] * 10, [None] * 10
        zero = lo.new_zeros(lo.shape)
        for side, (w, o, wt) in enumerate(sides):
            P = w.shape[2]
            dw = _fsum(wt.unsqueeze(2) * w, 0)                                            # (B, P, E)
            f[2 * side:2 * side + 2] = [dw[:, p] if p < P else zero for p in range(2)]
            f[4 + side] = _fsum(wt * (o + _fsum(w * c[:P], 2)), 0)
            f[6 + side] = _fsum(wt, 0)
            if self.margins:
                da, wa = wt.abs(), w.abs()
                dwa = _fsum(da.unsqueeze(2) * wa, 0)
                extra[2 * side:2 * side + 2] = [dwa[:, p] if p < P else zero for p in range(2)]
                extra[4 + side] = _fsum(da * (o.abs() + _fsum(wa * c[:P].abs(), 2)), 0)
                extra[6 + side] = _fsum(da, 0)
                extra[8 + side] = _fsum(dwa * dc[:P], 1)
        f[8], f[9] = lo, hi
        return torch.stack(f + (extra if self.margins else []), 1)

    def _level_view(self, faces, boxes, lev):
        """level lev's view (B, NF, E) from per-node raw faces and node boxes"""
        a, b = self.levels[lev]
        nds = [nd for nd in self.nodes if a <= nd.s < b]
        f0 = faces[nds[0].name]
        if len(nds) > 1 and all(faces[nd.name][0].shape[:3] == f0[0].shape[:3] and faces[nd.name][2].shape[:3] ==
                                f0[2].shape[:3] for nd in nds):
            # (every node with the same face counts and operands: the level's faces side by side, one stack)
            cat = lambda k: torch.cat([faces[nd.name][k] for nd in nds], -1)
            box = lambda k: torch.cat([boxes[nd.name][k] for nd in nds], -1)
            P = nds[0].P
            return self._stack_view([(cat(0), cat(1), cat(4)), (cat(2), cat(3), cat(5))], self.cop[:P, a:b],
                                    self.dcop[:P, a:b], box(0), box(1))
        parts = []
        for nd in nds:
            lw, lb, uw, ub, dl, du = faces[nd.name]
            s = slice(nd.s, nd.s + nd.E)
            parts.append(self._stack_view([(lw, lb, dl), (uw, ub, du)], self.cop[:nd.P, s], self.dcop[:nd.P, s],
                                          *boxes[nd.name]))
        return torch.cat(parts, -1) if len(parts) > 1 else parts[0]

    def crown(self, views, g, top, keep=None):
        """The sign backward (CROWN; face_dual mode 'sign') of the coefficient g on the columns up to the end of level
        `top`: (Q, n + e_top) shared by the boxes, or (B, Q, n + e_top), dense over the columns. Through the levels
        top, ..., 0 (views: per level, `_view`), each element's coefficient A passes through its default lower
        relaxation (A >= 0) or its default upper one (A < 0). Returns the input coefficient (B|1, Q, n) and the bias
        (B|1, Q) (the node residuals, zero up to rounding, priced on the node boxes, included). keep: a list that
        receives each level's arriving coefficient A (B|1, Q, E), from level 0 up. (sparse=True: _plan_bound.)"""
        n, bias = self.n, 0.0
        for lev in range(top, -1, -1):
            a, b = self.levels[lev]
            dw, dc, ds, lo, hi = _unpack(views[lev])
            A = g[..., n + a:n + b]
            A = A.unsqueeze(0) if A.ndim == 2 else A
            if keep is not None:
                keep.insert(0, A)
            Ap, An = A.clamp(min=0), (-A).clamp(min=0)
            W = Ap.unsqueeze(2) * dw[:, :, 0] - An.unsqueeze(2) * dw[:, :, 1]                # (B, Q, 2, E)
            bias = bias + (Ap * dc[:, :, 0] - An * dc[:, :, 1]).sum(-1) + _price(
                A - Ap * ds[:, :, 0] + An * ds[:, :, 1], lo, hi)
            g = g[..., :n + a] + self.Llev[lev].left(W.flatten(-2))
        return g, bias

    def _plan_bound(self, views, P, xl, xu, F=0, keep=None, plane=False):
        """CROWN's lower bounds (B, Q) of the rows of Plan P (their constants included) on the input boxes xl, xu
        (B, n): crown's sign backward slot by slot. Per stage, the slots on its level gather their elements' view
        (B, NF, Q, KA): A >= 0 passes through the default lower relaxation, A < 0 the upper one; the constants and the
        residual A - A+ ds_l + A- ds_u priced on the node boxes go to the bias; the slopes and the carried slots make
        the next slots (_Stage.red: one gather over [slots | slopes]). The input slots are priced on xl, xu. keep: as
        crown's (each level's arriving coefficient, dense over the level). plane: also the input slots' coefficient
        (B|1, Q, K) and the bias with the constants (B, Q).

        margins (P.e0): the bound is rigorous, a lower bound of the exact D(y*) of the graph's exact LP (the affine
        nodes composed exactly, the stored faces and boxes as exact numbers; F the most faces of a side) at y* = the
        multipliers the computed coefficients select (A+ d_l, A- d_u: real numbers, y* >= 0), so of the exact row's
        minimum over every point of that LP (weak duality). Proof. D(y*) = c* - b*.y* + sum_j min(r*_j lo_j, r*_j hi_j)
        with r* = q* + A*^T y*. Every entry the backward computes is a sum of products of stored numbers and computed
        multipliers; an elementary term is rounded at most k = F + kred + 8 times on its way (dw: F products and sums,
        x A, x M, the reductions' sums, kred the stages' most entries per output, carried sums included, the residual's
        2), so (Higham, Lemma 3.1, any order or FMA) a slot's coefficient differs from its exact counterpart by at most
        e = gamma_k sum |terms| + the exact forms' error bounds dM (_fold_error) times the multipliers + underflow: e
        starts at gamma_k |q| + dq and every stage adds (gamma_k |M| + dM) Wa (Wa = A+ sum|d||w| + A- ..., with W's
        underflow folded in), carrying e by 1 + gamma_k. A level slot's residual r* is within er = e + gamma_k (|A| +
        A+ sum|d_l| + A- sum|d_u|) + 2 eta of the computed r (the exact sum_f d_f against the computed ds, the
        residual's own rounding), so its price min(r* lo, r* hi) is within _ends(r, er) of the computed one; an input
        slot's likewise with e. The constants' terms A+ d_f (b_f + sum_p w_fp c_p*) are within gamma_k (A+ dca) +
        A+ dcf (the operand constants' fold error); the spec's constant within dconst. The computed total is a sum of
        nterms terms (gamma_{nterms+2} of their magnitudes, products and sums). Every margin term is a sum of
        nonnegative terms computed in the same dtype: _upper divides out its own rounding. The result is the computed
        total less that margin, rounded down. The same holds in any dtype dt (u = 2^-p, eta its smallest subnormal)."""
        B, Q = xl.shape[0], P.Q
        if self.memory is not None:
            width = max([P.cols.shape[1]] + [max(s.K, s.KA, s.K1) for s in P.stages])
            self.memory.require(xl.element_size() * B * (self.nf + 20) * Q * width)
        g, bias, mg = P.g0, 0.0, P.e0 is not None
        if mg:
            dt = xl.dtype
            k = F + P.kred + 8
            gam, eta = _gamma(k, dt), _ETA[dt]
            e = gam * P.g0.abs() + P.e0
            marg = mag = 0.0
        for st in P.stages:
            V = views[st.lev][..., st.le]                                                  # (B, NF, Q, KA)
            A = torch.where(st.lm, g[..., st.la], 0.0)                                     # (B|-, Q, KA)
            Ap, An = A.clamp(min=0), (-A).clamp(min=0)
            W = torch.stack([Ap * V[:, 0] - An * V[:, 2], Ap * V[:, 1] - An * V[:, 3]], -1)  # (B, Q, KA, 2)
            r = A - Ap * V[:, 6] + An * V[:, 7]
            lo, hi = V[:, 8], V[:, 9]
            p = torch.where(st.lm, torch.fmin(r * lo, r * hi).nan_to_num(nan=0.0), 0.0)
            cst = torch.where(st.lm, Ap * V[:, 4] - An * V[:, 5], 0.0)
            bias = bias + (cst + p).sum(-1)
            if keep is not None:
                Ak = A.expand(B, Q, -1)
                keep.insert(0, Ak.new_zeros(B, Q, views[st.lev].shape[-1]).scatter_add(-1, st.le.expand(B, Q, -1),
                                                                                         Ak))
            X = torch.cat([g.expand(B, -1), W.flatten(-3)], -1)
            if not mg:
                g = st.red(X)
                continue
            As = Ap + An
            # W's error bounds' weights, W's own underflow ((F + 2) products: gamma |M| (F + 2) eta / gamma)
            Wa = torch.stack([Ap * V[:, 10] + An * V[:, 12], Ap * V[:, 11] + An * V[:, 13]], -1) + \
                (As * ((F + 2) * eta / gam)).unsqueeze(-1)
            er = torch.where(st.lm, e[..., st.la] + gam * (A.abs() + Ap * V[:, 16] + An * V[:, 17]) + 2 * eta, 0.0)
            marg = marg + _ends(r, er, lo, hi).sum(-1) + torch.where(
                st.lm, gam * (Ap * V[:, 14] + An * V[:, 15]) + Ap * V[:, 18] + An * V[:, 19] + As * ((3 * F + 3) * eta),
                0.0).sum(-1)
            mag = mag + (cst.abs() + p.abs()).sum(-1)
            Xe = torch.cat([e.expand(B, -1), Wa.flatten(-3)], -1)
            ge = st.red(torch.stack([X, Xe]), lambda v, x: x * torch.stack([v[0], gam * v[0].abs() + v[1]]).unsqueeze(1))
            g, e = ge[0], ge[1] + st.red.K * eta
        Kf = P.cols.shape[1]
        gf = g.reshape(-1, Q, Kf)
        lo, hi = xl[:, P.cols], xu[:, P.cols]                                              # (B, Q, K)
        p = torch.where(P.cm, torch.fmin(gf * lo, gf * hi).nan_to_num(nan=0.0), 0.0)
        out = p.sum(-1) + bias + P.const
        if mg:
            marg = marg + torch.where(P.cm, _ends(gf, e.reshape(-1, Q, Kf), lo, hi), 0.0).sum(-1)
            mag = mag + p.abs().sum(-1) + P.const.abs()
            marg = marg + _gamma(P.nterms + 2, dt) * mag + P.dconst + P.nprod * eta
            out = _down(out - _upper(marg, k + P.nterms + 4, P.nprod))
        return (out, gf, bias + P.const) if plane else out

    def _crown_bound(self, views, spec, const, top, xl, xu):
        """CROWN's lower bound (B, Q) of spec . v + const (spec (Q, n + e_top) shared) on the input boxes xl, xu (B, n);
        spec None: level 1's operand forms (Lspec0) (sparse=False)"""
        if top < 0:
            return _price_shared(self.Lspec0.M if spec is None else spec, xl, xu) + const
        g, bias = self.crown(views, spec, top)
        return _price(g, xl[:, None], xu[:, None]) + bias + const

    def _operand_ibp(self, lev, lo, hi):
        """an enclosure (B, 2, E) of level lev's operands M v + c (E its elements) on the column boxes lo, hi
        (B, n + a) of the columns below it (the inputs and the lower levels' node intervals): sum_j min / max of
        M_ij v_j, its rounding (gamma_k of the magnitudes, k the row's terms + 3) and the fold's error bounds of the
        forms (dM . max|v| + dc) outward (bound()'s ibp_operands)"""
        a, b = self.levels[lev]
        E, n = b - a, self.n
        dt = lo.dtype
        if not hasattr(self, '_Llev_abs'):
            r, c, v = self._op()
            p, e = r // self.Et, r % self.Et
            self._Llev_abs = []
            for a_, b_ in self.levels:
                m = (e >= a_) & (e < b_) & (c < n + a_)
                self._Llev_abs.append(Mat(coo=(p[m] * (b_ - a_) + e[m] - a_, c[m], v[m].abs()),
                                          shape=(2 * (b_ - a_), n + a_), sparse=self.sparse))
        L, La = self.Llev[lev], self._Llev_abs[lev]
        S_lo = L.price_rows(lo, hi)                                                        # (B, 2E)
        S_hi = -L.price_rows(-hi, -lo)
        vmax = torch.maximum(lo.abs(), hi.abs())
        fin = torch.isfinite(vmax)
        mag = La.right(torch.where(fin, vmax, 0.0))
        k = La.K_right + 3
        g = _gamma(k, dt)
        cst = torch.cat([self.cop[0, a:b], self.cop[1, a:b]])
        # the fold's error bounds of these forms: dMop over the known columns (their forms reach no others)
        full = torch.zeros(lo.shape[0], self.ncol, dtype=dt, device=lo.device)
        full[:, :n + a] = torch.where(fin, vmax, 0.0)
        if self.sparse:
            t = self.LdG.right(full)
        else:
            t = full @ self.dG.t()
        t = t + self.dg
        rows = torch.cat([torch.arange(a, b, device=lo.device), self.Et + torch.arange(a, b, device=lo.device)])
        fold = _upper(t[:, rows], self.ncol + 1, self.ncol)
        err = _up(_up(g * (mag + cst.abs()) / _one_minus(g, dt)) + fold + (k + 2) * _ETA[dt])
        bad = ~fin.all(1, keepdim=True)
        il = torch.where(bad, -INF, _down(_down(S_lo + cst) - err))
        iu = torch.where(bad, INF, _up(_up(S_hi + cst) + err))
        return il.reshape(-1, 2, E), iu.reshape(-1, 2, E)

    def bound(self, x_L, x_U, thr=None, h=None, caps=None):
        """CROWN on the compiled form: per level the bounds of its operands (CROWN through the levels below), its faces
        by the operators' own code, its node intervals; then the lower bounds (B, S) of the rows of C. Returns
        (Data, bound); the Data also holds the intermediate bounds: ol, ou (B, 2, Et) every element's operand bounds,
        nl, nu (B, Et) the own bounds of the `own` nodes (-inf, inf elsewhere), undefined (B,) the boxes on which an
        operator's relaxation is undefined (their bound -inf). The first call (from __init__) reads off the face
        counts. sparse=True: every backward slot by slot (the Plans)."""
        B, n = x_L.shape[0], self.n
        xl, xu = x_L.reshape(B, n), x_U.reshape(B, n)
        dt = self.cop.dtype
        xl, xu = _to(xl, dt, False), _to(xu, dt, True)
        faces, boxes, views, obs, owns, F = {}, {}, [], [], {}, 0
        undefined = torch.zeros(B, dtype=torch.bool, device=xl.device)
        diagnostic = getattr(self, 'diagnostics', False)
        bad_node = torch.full((B,), -1, dtype=torch.long, device=xl.device) if diagnostic else None
        for lev, (a, b) in enumerate(self.levels):
            E = b - a
            if self.sparse:
                v = self._plan_bound(views, self.plan_lev[lev], xl, xu, F)
            else:
                M = self.Mop[:, a:b, :n + a]
                const = torch.cat([self.cop[:, a:b], -self.cop[:, a:b]]).reshape(4 * E)
                v = self._crown_bound(views, None if lev == 0 else torch.cat([M, -M]).reshape(4 * E, n + a), const,
                                      lev - 1, xl, xu)
            v = v.reshape(B, 2, 2, E)
            ol, ou = v[:, 0], -v[:, 1]                                                       # (B, 2, E)
            if self.operand_caps is not None:
                ol = torch.maximum(ol, _to(self.operand_caps[0][..., a:b], dt, False))
                ou = torch.minimum(ou, _to(self.operand_caps[1][..., a:b], dt, True))
            if getattr(self, 'ibp_operands', False) and lev > 0:
                # the operands' CROWN bounds intersected with their interval over the (capped) lower levels'
                # node boxes, so that every level's faces are built from the tighter of the two (a reciprocal's
                # operand away from 0 where CROWN alone straddles it)
                lo_c = torch.cat([xl] + [boxes[nd.name][0] for nd in self.nodes if nd.s < a], 1)
                hi_c = torch.cat([xu] + [boxes[nd.name][1] for nd in self.nodes if nd.s < a], 1)
                il, iu = self._operand_ibp(lev, lo_c, hi_c)
                ol, ou = torch.maximum(ol, il), torch.minimum(ou, iu)
            obs.append((ol, ou))
            nds = [nd for nd in self.nodes if a <= nd.s < b]
            for grp, one, idx in self.groups[lev]:
                nd = grp[0]
                if len(grp) == 1:
                    i, j = nd.s - a, nd.s - a + nd.E
                    faces[nd.name], boxes[nd.name], bad = self._node_faces(nd, ol[:, :nd.P, i:j], ou[:, :nd.P, i:j])
                else:
                    f, bx, bad = self._node_faces(one, ol[..., idx], ou[..., idx])
                    k = 0
                    for m in grp:
                        faces[m.name] = tuple(t[..., k:k + m.E] for t in f)
                        boxes[m.name] = tuple(t[..., k:k + m.E] for t in bx)
                        k += m.E
                if diagnostic:
                    bad_node = torch.where((bad_node < 0) & bad, self.nodes.index(nd), bad_node)
                undefined = undefined | bad
                F = max(F, faces[nd.name][1].shape[0], faces[nd.name][3].shape[0])
            views.append(self._level_view(faces, boxes, lev))
            own = [nd for nd in nds if nd.own]
            if own:
                # a node's own CROWN bound: from its columns through its level and those below
                if self.sparse:
                    v = self._plan_bound(views, self.plan_own[lev], xl, xu, F)
                else:
                    eye = torch.eye(n + b, dtype=xl.dtype, device=xl.device)
                    unit = torch.cat([eye[n + nd.s:n + nd.s + nd.E] for nd in own])
                    zero = torch.zeros(2 * unit.shape[0], dtype=xl.dtype, device=xl.device)
                    v = self._crown_bound(views, torch.cat([unit, -unit]), zero, lev, xl, xu)
                v = v.reshape(B, 2, -1)
                k = 0
                for nd in own:
                    l, u = boxes[nd.name]
                    owns[nd.name] = (v[:, 0, k:k + nd.E], -v[:, 1, k:k + nd.E])
                    boxes[nd.name] = (torch.maximum(l, owns[nd.name][0]), torch.minimum(u, owns[nd.name][1]))
                    k += nd.E
            if caps is not None:
                # given caps of the node intervals (B, Et) (FBBT's, valid where the caller's points live)
                for nd in nds:
                    l, u = boxes[nd.name]
                    boxes[nd.name] = (torch.maximum(l, caps[0][:, nd.s:nd.s + nd.E]),
                                      torch.minimum(u, caps[1][:, nd.s:nd.s + nd.E]))
            if self.margins:
                for nd in nds:
                    boxes[nd.name] = _widen(*boxes[nd.name])
            if own or self.margins:
                views[-1] = self._level_view(faces, boxes, lev)
        if not hasattr(self, 'Fl'):
            self._set_counts({k: (f[1].shape[0], f[3].shape[0]) for k, f in faces.items()})
        d = self._data(x_L, x_U, faces, boxes, thr, h, undefined)
        if diagnostic:
            d.bad_node = bad_node
        d.ol, d.ou = torch.cat([o[0] for o in obs], -1), torch.cat([o[1] for o in obs], -1)
        inf = torch.full((B, 1), INF, dtype=xl.dtype, device=xl.device)
        d.nl = torch.cat([owns[nd.name][0] if nd.own else -inf.expand(B, nd.E) for nd in self.nodes], 1)
        d.nu = torch.cat([owns[nd.name][1] if nd.own else inf.expand(B, nd.E) for nd in self.nodes], 1)
        if self.sparse:
            out = self._plan_bound(views, self.plan_out, xl, xu, F)
        else:
            out = self._crown_bound(views, self.CM, self.Cc, len(self.levels) - 1, xl, xu)
        return d, torch.where(undefined[:, None], -INF, out)

    def interm_bounds(self, d):
        """the pass's intermediate bounds in auto_LiRPA's format {node: (lower, upper) (B, *shape)}: the operands of the
        face nodes (the input excluded) and the `own` nodes (an operand broadcast to its face node is read back from
        any copy)"""
        out, B = {}, d.lo.shape[0]
        for nd in self.nodes:
            for p, name in enumerate(nd.operands):
                if name == self.net.input_name[0] or name in out:
                    continue
                shape = tuple(self.net[name].output_shape[1:])
                m = math.prod(shape)
                idx = torch.arange(m, device=d.lo.device).reshape(1, *shape).broadcast_to(1, *nd.shape).reshape(1, -1)
                idx = idx.expand(B, -1)
                out[name] = tuple(d.lo.new_zeros(B, m).scatter(1, idx, t[:, p, nd.s:nd.s + nd.E]).reshape(B, *shape)
                                  for t in (d.ol, d.ou))
            if nd.own:
                out[nd.name] = tuple(t[:, nd.s:nd.s + nd.E].reshape(B, *nd.shape) for t in (d.nl, d.nu))
        return out

    def output_bound(self, d, spec=None, const=None):
        """CROWN's lower bound (B, Q) of spec . v + const (spec (Q, ncol) compiled columns, default the rows of C) on
        Data d; with the input plane (B, Q, n) and its bias (B, Q)"""
        views = [self._view(d, k) for k in range(len(self.levels))]
        xl, xu = d.lo[:, :self.n], d.hi[:, :self.n]
        if spec is None and self.sparse:
            P = self.plan_out
            b, gf, bias = self._plan_bound(views, P, xl, xu, max(self.Fl, self.Fu), plane=True)
            Bg = gf.shape[0]
            g = gf.new_zeros(Bg, P.Q, self.n).scatter_add(-1, P.cols.expand(Bg, -1, -1), torch.where(P.cm, gf, 0.0))
            return b, g, bias
        spec, const = (self.CM, self.Cc) if spec is None else (spec, const)
        g, bias = self.crown(views, spec, len(self.levels) - 1)
        bias = bias + const
        return _price(g, xl[:, None], xu[:, None]) + bias, g, bias

    def sign_y(self, d):
        """CROWN's multipliers of the objective (row 0 of C) on Data d: every face's mu = A^+ dl, nu = A^- du (face_dual
        mode 'sign'), no output or base row: certify(d, sign_y(d)) is the rigorous form of CROWN's bound of row 0"""
        keep = []
        views = [self._view(d, k) for k in range(len(self.levels))]
        if self.sparse:
            self._plan_bound(views, self.plan_obj, d.lo[:, :self.n], d.hi[:, :self.n], max(self.Fl, self.Fu), keep)
        else:
            self.crown(views, self.CM[:1], len(self.levels) - 1, keep)
        B, dt, dev = d.lo.shape[0], d.lo.dtype, d.lo.device
        A = torch.cat([a.expand(B, 1, a.shape[-1]) for a in keep], -1) if keep else d.lo.new_zeros(B, 1, 0)
        return (A.clamp(min=0).unsqueeze(2) * d.dl.unsqueeze(1), (-A).clamp(min=0).unsqueeze(2) * d.du.unsqueeze(1),
                torch.zeros(B, 1, len(self.rows), dtype=dt, device=dev), torch.zeros(B, 1, self.R, dtype=dt, device=dev))

    # ------------------------------------------------------------------------------------------------ primitives
    def F_A(self, d, v):
        """b - A v of every row at the columns v (B, Q, ncol): the slacks of the lower faces (B, Q, Fl, Et), of the
        upper faces (B, Q, Fu, Et) (padding entries, vl/vu false, are no rows), of the output rows (B, Q, len(rows)),
        of the base rows (B, Q, R). One GEMM over the stacked rows, then elementwise."""
        B, Q, Et = v.shape[0], v.shape[1], self.Et
        Z = self.Lstack.right(v)
        U = Z[..., :2 * Et].unflatten(-1, (2, 1, Et))                                      # operand values less c
        z = Z[..., None, 2 * Et:3 * Et]
        lw, uw = d.lw.unsqueeze(1), d.uw.unsqueeze(1)
        sl = (z - d.lbc.unsqueeze(1)) - torch.addcmul(lw[:, :, :, 0] * U[:, :, 0], lw[:, :, :, 1], U[:, :, 1])
        su = torch.addcmul(uw[:, :, :, 0] * U[:, :, 0], uw[:, :, :, 1], U[:, :, 1]) - (z - d.ubc.unsqueeze(1))
        nr = len(self.rows)
        so = d.bflat[:, None, (self.Fl + self.Fu) * Et:(self.Fl + self.Fu) * Et + nr] - Z[..., 3 * Et:3 * Et + nr]
        sb = d.h[:, None] - Z[..., 3 * Et + nr:]
        return sl, su, so, sb

    def R_A(self, d, y, objective=1.0):
        """r = c + A^T y (B, Q, ncol) and D(y) (B, Q) = c0 - b^T y + sum_j min(r_j lo_j, r_j hi_j), objective weight
        `objective` (0: the Farkas value), y in F_A's layout (any sign: no clipping here). One GEMM over the stacked
        rows."""
        yl, yu, yo, yb = y
        W = (yl.unsqueeze(3) * d.lw.unsqueeze(1)).sum(2) - (yu.unsqueeze(3) * d.uw.unsqueeze(1)).sum(2)
        r = self.Lstack.left(torch.cat([W.flatten(-2), yu.sum(2) - yl.sum(2), yo, yb], -1)) + objective * self.CM0
        yf = torch.cat([yl.flatten(2), yu.flatten(2), yo, yb], -1)
        const = objective * self.Cc[0] - (yf @ d.bflat.unsqueeze(-1)).squeeze(-1)
        return r, const + _price(r, d.lo[:, None], d.hi[:, None])

    def densify(self, d):
        """the rows as one per-box matrix: d.Ab (B, nrow, ncol + 1), canonical rows, compiled columns, the last
        column -b. F_Ad / R_Ad then take one batched GEMM each (a search copy: its entries are the rounded products
        of the faces and the operand forms; the certificate keeps the factored data)"""
        B, Et, Fl, Fu, n = d.lo.shape[0], self.Et, self.Fl, self.Fu, self.n
        Mop, dt, dev = self.Mop, d.lo.dtype, d.lo.device
        er = torch.arange(Et, device=dev)
        unit = torch.zeros(Et, self.ncol, dtype=dt, device=dev).index_put_((er, n + er), torch.ones_like(er, dtype=dt))
        rest = torch.cat([self.CM[self.rows], torch.cat([self.E, torch.zeros(self.R, Et, dtype=dt, device=dev)], 1)])
        Al = torch.einsum('bfpe,pec->bfec', d.lw, Mop) - unit
        Au = unit - torch.einsum('bfpe,pec->bfec', d.uw, Mop)
        A = torch.cat([Al.reshape(B, Fl * Et, -1), Au.reshape(B, Fu * Et, -1), rest.expand(B, -1, -1)], 1)
        d.Ab = torch.cat([A, -d.bflat.unsqueeze(-1)], -1)[:, self.ridx]
        return d

    def F_Ad(self, d, v):
        """b - A v (B, Q, nrow) at the columns v (B, Q, ncol), canonical rows, on densify's matrix"""
        return torch.baddbmm(-d.Ab[:, None, :, -1], v, d.Ab[:, :, :-1].transpose(1, 2), alpha=-1.0)

    def R_Ad(self, d, y, objective=1.0):
        """r = c + A^T y (B, Q, ncol) and D(y) (B, Q) of canonical-row multipliers y (B, Q, nrow), on densify's
        matrix (its last column gives c0 - b^T y)"""
        c = torch.cat([objective * self.CM0, (objective * self.Cc[0]).reshape(1)])
        rc = torch.baddbmm(c.expand(y.shape[0], 1, -1), y, d.Ab)
        r = rc[..., :-1]
        return r, rc[..., -1] + _price(r, d.lo[:, None], d.hi[:, None])

    def select(self, d, sl, su):
        """the selected faces sl (B, Q|1, Fl, Et), su (B, Q|1, Fu, Et) (one-hot over the faces of a side, at most one
        face per element over both sides): their slopes (B, Q|1, 2, Et), offsets in operand terms, `has` (B, Q|1, Et)"""
        ws = (sl.unsqueeze(3) * d.lw.unsqueeze(1)).sum(2) + (su.unsqueeze(3) * d.uw.unsqueeze(1)).sum(2)
        bs = (sl * d.lb.unsqueeze(1)).sum(2) + (su * d.ub.unsqueeze(1)).sum(2)
        return ws, bs, sl.sum(2) + su.sum(2), sl, su

    def F_T(self, d, sel, v, linear=False):
        """the forward through the selected faces (pivot_crown.Graph.forward): level by level, every element with a
        selected face is the face at its operands' values plus v (its column), every other element is v; the inputs
        are v[..., :n]. v (B, Q, ncol). linear: the map's linear part (no constants: the tangents of a k-point batch).
        Returns the columns (B, Q, ncol)."""
        ws, bs, has = sel[:3]
        n, out = self.n, v.clone()
        for lev, (a, b) in enumerate(self.levels):
            U = self.Llev[lev].right(out[..., :n + a]).unflatten(-1, (2, b - a))
            if not linear:
                U = U + self.cop[:, a:b]
            z = (ws[..., a:b] * U).sum(-2) + (0.0 if linear else bs[..., a:b])
            out[..., n + a:n + b] = has[..., a:b] * z + v[..., n + a:n + b]
        return out

    def R_T(self, d, sel, out=None, base=None, face=None, col=None):
        """the backward through the selected faces (pivot_crown.Graph.backward, face_dual mode 'basis') of Q
        right-hand sides per box: out (B, Q, S) weights on the rows of C (row 0 the objective), base (B, Q, R) on the
        base rows, face = (yl (B, Q, Fl, Et), yu) multipliers on the faces (acting as given), col (B, Q, ncol)
        coefficients injected at the columns. Every element with a selected face passes the rest of its coefficient,
        of either sign, through it. Returns y = (yl, yu, yo (B, Q, len(rows)), yb) and r (B, Q, ncol) = c + A^T y."""
        ws, bs, has, sl, su = sel
        n = self.n
        ref = next(t for t in (out, base, col) if t is not None) if face is None else face[0]
        B, Q, dt, dev = ref.shape[0], ref.shape[1], ref.dtype, ref.device
        g = torch.zeros(B, Q, self.ncol, dtype=dt, device=dev) if out is None else self.Lcm.left(out)
        if col is not None:
            g = g + col
        if face is None:
            Wg, mg = torch.zeros(B, Q, 2, self.Et, dtype=dt, device=dev), torch.zeros(B, Q, self.Et, dtype=dt, device=dev)
            yl0 = yu0 = None
        else:
            yl0, yu0 = face
            Wg = (yl0.unsqueeze(3) * d.lw.unsqueeze(1)).sum(2) - (yu0.unsqueeze(3) * d.uw.unsqueeze(1)).sum(2)
            mg = yl0.sum(2) - yu0.sum(2)
        rest = []
        for lev in range(len(self.levels) - 1, -1, -1):
            a, b = self.levels[lev]
            rho = g[..., n + a:n + b] - mg[..., a:b]
            W = Wg[..., a:b] + rho.unsqueeze(-2) * ws[..., a:b]
            g = g[..., :n + a] + self.Llev[lev].left(W.flatten(-2))
            rest.insert(0, rho)
        rest = torch.cat(rest, -1)                                                          # (B, Q, Et)
        yl = rest.unsqueeze(2) * sl + (0.0 if yl0 is None else yl0)
        yu = (0.0 if yu0 is None else yu0) - rest.unsqueeze(2) * su
        yb = base if base is not None else torch.zeros(B, Q, self.R, dtype=dt, device=dev)
        rx = g + self.LE.left(yb)
        yo = out[..., self.rows] if out is not None else torch.zeros(B, Q, len(self.rows), dtype=dt, device=dev)
        return (yl, yu, yo, yb), torch.cat([rx, rest * (1 - has)], -1)

    # ------------------------------------------------------------------------------------------------ certificate
    def _fold_slack(self, d):
        """the fold's rounding (_fold_error) as slack of Data d's rows, rounded up: every face row's sum_p |a_p| (dM_p .
        vmax + dc_p) (B, Fl, Et), (B, Fu, Et), the output rows' and the objective's dCM . vmax + dCc (B, len(rows)),
        (B,); vmax (B, ncol) the larger end magnitude of each column's box (0 x inf = 0: a column without error does not
        count). For every v in the box, |(exact form - compiled form)(v)| is at most the slack. Box dependent, y
        independent: _data stores it (Data.slack, its boxes' and faces'); certify recomputes it for a Data without.
        The base rows' (B, R): dE . vmax (dE the rounding of E in a copy of a lower precision; else 0)."""
        dt, Et = d.lo.dtype, self.Et
        vmax = torch.maximum(d.lo.abs(), d.hi.abs())
        fin = torch.isfinite(vmax)
        if self.sparse:
            t = self.LdG.right(torch.where(fin, vmax, 0.0)) + self.dg                       # (B, 2 Et + S)
            bad = self.LdGi.right((~fin).to(dt))
        else:
            t = torch.where(fin, vmax, 0.0) @ self.dG.t() + self.dg
            bad = (~fin).to(dt) @ (self.dG > 0).to(dt).t()
        t = torch.where(bad > 0, INF, _upper(t, self.ncol + 1, self.ncol))
        top = t[:, None, :2 * Et].unflatten(-1, (2, Et))                                    # (B, 1, 2, Et)
        face = lambda w: _upper(torch.nan_to_num(w.abs() * top, nan=0.0, posinf=INF).sum(2), 2, 2)
        if self.dE is None:
            tb = d.lo.new_zeros(d.lo.shape[0], self.R)
        else:
            x = vmax[:, :self.n]
            tb = torch.where((~torch.isfinite(x)).to(dt) @ (self.dE > 0).to(dt).t() > 0, INF,
                             _upper(torch.where(torch.isfinite(x), x, 0.0) @ self.dE.t(), self.n + 1, self.n))
        return face(d.lw), face(d.uw), t[:, 2 * Et:][:, self.rows], t[:, 2 * Et], tb

    def _cert_b(self, d, slack=None):
        """Data d's face rows' b relaxed by the fold's slack and their magnitudes, per box (y independent): lb +
        sum_p lw_p c_p - slack, ub + sum_p uw_p c_p + slack, and |.| as certify sums them (B, F, Et); stored on d
        (cbl, cbu, cbla, cbua) with d.slack (slack None), so certify loads them instead of recomputing them per lane
        inside its reduction (whose fp64 body would spill: 8 ms on wastewater13m1's extended graph)"""
        tl, tu = (d.slack if slack is None else slack)[:2]
        cw = self.cop                                                                       # (2, Et)
        b = (d.lb + _fsum(d.lw * cw, 2) - tl, d.ub + _fsum(d.uw * cw, 2) + tu,
             d.lb.abs() + _fsum(d.lw.abs() * cw.abs(), 2) + tl, d.ub.abs() + _fsum(d.uw.abs() * cw.abs(), 2) + tu)
        if slack is None:
            d.cbl, d.cbu, d.cbla, d.cbua = b
        return b

    def certify(self, d, y, objective=1.0, target=None):
        """The rigorous D(y) (B, Q) (fp64): a lower bound of the exact c0 - b^T y + sum_j min(r_j lo_j, r_j hi_j),
        r = c + A^T y, of the graph's exact LP at y clipped at 0 (padding faces 0), objective weight `objective` (0: the
        Farkas value, whose positivity proves the LP infeasible). The graph's exact LP: its rows and objective with the
        exact affine forms of the operands and the output (the composition of the affine nodes in exact arithmetic,
        _fold_error), the stored faces, thresholds, h and boxes as exact numbers; the compiled LP's rows relaxed by the
        fold's slack (_fold_slack) contain it, so a lower bound of the relaxed LP's D(y) (b + slack, the objective less
        its slack) is one of the exact LP's. Every reduction is enclosed by the forward-error
        bound |fl(s) - s| <= gamma_k sum |terms| + (underflow), k the most roundings any of its terms undergoes in any
        summation order, FMA or not (Higham, Lemma 3.1 and eq. 3.4), sum |terms| from the same reductions on the
        magnitudes (its own rounding divided out); underflow: every product contributes at most ETA. The exact r* is
        in [r - err, r + err]; each column is priced by min(r lo, r hi) less err times the magnitude of the end r*
        selects, the sums rounded outward. Non-finite y or a non-finite result give -inf. Returns (rigorous, plain)
        with plain the round-to-nearest value of R_A at the clipped y."""
        dt = d.lo.dtype
        Eta = _ETA[dt]
        raw = y
        yl, yu, yo, yb = (torch.nan_to_num(t, 0.0, 0.0, 0.0).clamp(min=0) for t in y)
        yl, yu = torch.where(self.vl, yl, 0.0), torch.where(self.vu, yu, 0.0)
        y = (yl, yu, yo, yb)
        F, Et, nr, R, ncol = max(self.Fl, self.Fu), self.Et, len(self.rows), self.R, self.ncol
        # r = c + A^T y (as R_A) and its magnitude rho = |c| + |A|^T y by one gather over the stacked rows (Lstack: the
        # operand forms, the face columns' units, the output rows, the base rows). A face term y lw M: 1 product, F - 1
        # face sums, 1 side difference, 1 product and K - 1 sums (K the widest column's entries), the objective's add
        W = _fsum(yl.unsqueeze(3) * d.lw.unsqueeze(1), 2) - _fsum(yu.unsqueeze(3) * d.uw.unsqueeze(1), 2)
        Wa = _fsum(yl.unsqueeze(3) * d.lw.abs().unsqueeze(1), 2) + _fsum(yu.unsqueeze(3) * d.uw.abs().unsqueeze(1), 2)
        sl, su = _fsum(yl, 2), _fsum(yu, 2)
        r, rho = self.Lstack.left_abs(torch.cat([W.flatten(-2), su - sl, yo, yb], -1),
                                      torch.cat([Wa.flatten(-2), su + sl, yo, yb], -1))
        r = r + objective * self.CM0
        rho = rho + abs(objective) * self.CM0.abs()
        if target is not None:
            # Opt-in column objectives: exact stored coefficients, no graph
            # folding error. The existing residual enclosure also covers this
            # extra addition (the count above already includes objective add).
            # Use objective=0 so exactly one objective addition is nonzero.
            if objective != 0:
                raise ValueError('column target requires objective=0')
            r = r + target[0]
            rho = rho + target[0].abs()
        g = _gamma(F + self.Lstack.K_left + 4 + 2, dt)                              # + 2: its own
        eta = 2 * Eta * (F * self.Mcol + 2 * Et + nr + R + 1)  # an underflow in W, carried by |M|; the other products
        err = _up(g * (rho + eta) / _one_minus(g, dt) + eta)                         # |r* - r| <= err (> 0)
        # every column's exact min(r* lo, r* hi) >= p - err v: p = min(r lo, r hi) (0 x inf: no contribution), v the
        # magnitude of the end r* selects (r > err: lo; r < -err: hi; else either: the larger, its min being
        # max(|lo|, |hi|)-Lipschitz in r)
        lo, hi = d.lo[:, None], d.hi[:, None]
        p = torch.fmin(r * lo, r * hi).nan_to_num(nan=0.0, posinf=INF, neginf=-INF)
        v = torch.where(r > err, lo.abs(), torch.where(r < -err, hi.abs(), torch.maximum(lo.abs(), hi.abs())))
        # the constant part c0 - b^T y, row by row, every row's b relaxed by the fold's slack: a face row's b = lb +
        # sum_p lw_p c_p - slack (2 products, 3 sums), times y (1); a row with y = 0 contributes 0 (also where its b or
        # slack is infinite); a non-finite y poisons the sum (0 y: NaN), which gives -inf
        if hasattr(d, 'slack'):              # (the per-box b stored with the slack it came from, _cert_b)
            tl, tu, to, tobj, tb = d.slack
            bl, bu, bl_a, bu_a = (d.cbl, d.cbu, d.cbla, d.cbua) if hasattr(d, 'cbl') else self._cert_b(d)
        else:
            slack = self._fold_slack(d)
            tl, tu, to, tobj, tb = slack
            bl, bu, bl_a, bu_a = self._cert_b(d, slack)
        tb = tb[:, None]
        thr = d.thr[:, None, self.rows]
        to = to[:, None]
        at = lambda yy, v: torch.where(yy > 0, yy * v, 0.0)
        tob = (abs(objective) * tobj if objective else torch.zeros_like(tobj))[:, None].expand(-1, yl.shape[1])
        # every row block summed on its own (one reduction each, each lean: a reduction over the blocks side by side
        # makes every element evaluate every block's expression, which costs milliseconds in fp64), then the blocks'
        # sums added: a term is rounded at most M + 5 times all the same; a non-finite y poisons the sum (0 y: NaN)
        s = lambda t: t.flatten(2).sum(-1)
        poison = sum(s(t * 0.0) for t in raw)
        sc = s(at(yl, bl.unsqueeze(1))) - s(at(yu, bu.unsqueeze(1))) + (-at(yo, thr + to) + yo * self.Ccr).sum(-1) - \
            at(yb, d.h[:, None] + tb).sum(-1) - tob + poison
        ac = s(at(yl, bl_a.unsqueeze(1))) + s(at(yu, bu_a.unsqueeze(1))) + \
            at(yo, thr.abs() + to + self.Ccr.abs()).sum(-1) + at(yb, d.h[:, None].abs() + tb).sum(-1) + tob
        # underflow: per row its products with y (up to 2) and a product lw c_p carried by y; per column its product
        M = (self.Fl + self.Fu) * Et + nr + R + 1
        su = 2 * Eta * (2 * (s(yl) + s(yu)) + 2 * M) + 2 * Eta * ncol
        sp, ap, se = p.sum(-1), p.abs().sum(-1), (err * v).sum(-1)
        c0 = objective * self.Cc[0] if target is None else target[1]
        gs, g0 = _gamma(ncol + 1, dt), _gamma(M + 5 + 2, dt)
        price = _down(_down(sp - _up(gs * ap / _one_minus(gs, dt))) - _upper(se, ncol + 1, ncol))
        e0 = _up(g0 * (ac + abs(c0)) / _one_minus(g0, dt) + su + 2 * Eta)
        D = _down(_down(sc + c0 - e0) + price)
        D = torch.where(torch.isfinite(D) | (D == -INF), D, torch.full_like(D, -INF))
        if getattr(d, 'undefined', None) is not None:                     # (no valid relaxation on these boxes)
            D = torch.where(d.undefined[:, None], -INF, D)
        plain = self.R_A(d, y, objective)[1] if target is None else sc + c0 + sp
        return D, plain

    def to(self, dtype):
        """a copy in `dtype` (e.g. float32) as rigorous in it as the original: every stored number of the compiled LP
        (the operand forms Mop, cop, C M_out, C c_out, E) rounded to the nearest number of dtype, its error bound
        widened by that rounding (_cast: |exact - fl(v)| <= dv + |v - fl(v)|, rounded up; E's rounding becomes the base
        rows' error bound dE, _fold_slack); the Mats and Plans rebuilt in dtype (so certify and the margins' gamma_k,
        eta are dtype's). The data: from bound() it is recomputed by the copy's own bound() (faces, node intervals and
        bounds computed and made rigorous in dtype; the boxes, thresholds and h rounded outward); from a cache or
        frozen bounds it is rounded (a search copy: its faces are not re-validated), its boxes, thresholds and h
        outward."""
        out = CompiledGraph.__new__(CompiledGraph)
        out.__dict__.update(vars(self))
        dt, n = dtype, self.n
        out._lower(dt)
        out._mats()
        out._privates()
        d = self.data
        if hasattr(self, 'out'):
            out.data, out.out = out.bound(d.lo[:, :n], d.hi[:, :n], d.thr, d.h)
        else:
            out.data = d.to(dt)
            out.data.lo, out.data.hi = _to(d.lo, dt, False), _to(d.hi, dt, True)
            out.data.thr, out.data.h = _to(d.thr, dt, True), _to(d.h, dt, True)
            out.data.derive(out.cop).rhs(out.rows, out.Ccr)
            out.data.slack = out._fold_slack(out.data)
            out._cert_b(out.data)
        return out

    def _privates(self):
        """every unary face node's private copy, on which its own face code runs (it leaves its relaxation state
        there); in a graph below the network's precision also every BoundMul's, its output_dtype the graph's (its
        faces are built in that dtype instead of being built in the network's and rounded)"""
        from .operators import BoundMul, BoundSin
        dt, net = self.cop.dtype, self.net
        self._private = {}
        for nd in self.nodes:
            node = net[nd.name]
            if not isinstance(node, BoundMul):
                self._private[nd.name] = _private(node)
                if isinstance(node, BoundSin):
                    self._private[nd.name].trig_tangents = self.trig_tangents
            elif getattr(node, 'output_dtype', dt) != dt:
                self._private[nd.name] = c = _private(node)
                c.output_dtype = dt

    def _lower(self, dt):
        """the stored numbers of the compiled LP in dtype dt, their error bounds widened by the rounding (to)"""
        self.cop, self.dcop = _cast(self.cop, self.dcop, dt)
        self.Cc, self.dCc = _cast(self.Cc, self.dCc, dt)
        self.E, dE = _cast(self.E, self.E.new_zeros(self.E.shape) if self.dE is None else self.dE, dt)
        self.dE = dE if bool((dE > 0).any()) else None
        if self.sparse:
            self.op, self.dop = _cast_coo(self._op(), self._op(d=True), dt, self.ncol)
            self._cmc, self._dcmc = _cast_coo(self._cm(), self._cm(d=True), dt, self.ncol)
            self._Mop = self._dMop = self._CM = self._dCM = None
        else:
            M, dM, CM, dCM = self.Mop, self.dMop, self.CM, self.dCM
            self._Mop, self._dMop = _cast(M, dM, dt)
            self._CM, self._dCM = _cast(CM, dCM, dt)
            self.op = self.dop = self._cmc = self._dcmc = None

    # ------------------------------------------------------------------------------------------------ layouts
    def compiled_cols(self, v):
        """canonical columns (..., ncol) -> compiled"""
        return v[..., self.k2c]

    def canon_cols(self, v):
        return v[..., self.c2k]

    def compiled_rows(self, y):
        """canonical rows (B, Q, nrow) -> (yl, yu, yo, yb)"""
        flat = y.new_zeros(*y.shape[:-1], self.nflat).index_copy(-1, self.ridx, y)
        Fl, Fu, Et, nr = self.Fl, self.Fu, self.Et, len(self.rows)
        yl = flat[..., :Fl * Et].unflatten(-1, (Fl, Et))
        yu = flat[..., Fl * Et:(Fl + Fu) * Et].unflatten(-1, (Fu, Et))
        return yl, yu, flat[..., (Fl + Fu) * Et:(Fl + Fu) * Et + nr], flat[..., (Fl + Fu) * Et + nr:]

    def canon_rows(self, y):
        return torch.cat([t.flatten(2) for t in y], -1)[..., self.ridx]

    def selection(self, sel):
        """pivot_crown.Graph.selection's {name: (lower (Fl, B, *shape), upper)} -> (sl, su) (B, 1, F, Et)"""
        ref = next(iter(sel.values()))[0]
        B = ref.shape[1]
        sl = ref.new_zeros(B, 1, self.Fl, self.Et)
        su = ref.new_zeros(B, 1, self.Fu, self.Et)
        for nd in self.nodes:
            lo, up = sel[nd.name]
            sl[:, 0, :nd.Fl, nd.s:nd.s + nd.E] = lo.reshape(nd.Fl, B, nd.E).transpose(0, 1)
            su[:, 0, :nd.Fu, nd.s:nd.s + nd.E] = up.reshape(nd.Fu, B, nd.E).transpose(0, 1)
        return sl, su


def certifier(net, prob, cache, sparse=False):
    """A rigorous drop-in for pivot_crown.Graph.certify on (net, prob, cache): y (B, nrow) in its row layout ->
    D (B,) (objective 0: the Farkas value). pivot_crown.solve(..., certify=certifier(net, prob, cache))."""
    cg = CompiledGraph(net, prob.x_L, prob.x_U, prob.C, prob.thr, prob.E, prob.h, rows=prob.rows.tolist(), cache=cache,
                       sparse=sparse)

    def certify(y, objective=1.0):
        return cg.certify(cg.data, cg.compiled_rows(y.unsqueeze(1)), objective)[0][:, 0]
    certify.graph = cg
    return certify


def fuse(fn):
    """fn traced whole by dynamo and fused by inductor (fixed shapes; every primitive and bound() trace whole). Inductor
    (torch 2.8) unrolls reductions of <= 8 elements into loads at constant offsets that its Triton code broadcasts as
    Python ints, which Triton rejects (broadcast_to of a constexpr): unrolling off."""
    return torch.compile(fn, fullgraph=True, dynamic=False, options={'unroll_reductions_threshold': 1})


class Captured:
    """fn() on static tensors captured as one CUDA graph (side-stream warm-up first); replay() reruns it on their
    current contents and returns fn's outputs (static tensors)"""

    def __init__(self, fn, warmup=3):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        self.out = capture(self.graph, fn)

    def replay(self):
        self.graph.replay()
        return self.out
