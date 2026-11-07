"""Pivoting CROWN: the graph LP of a box solved exactly by a bounded-variable dual simplex whose basis solves are
CROWN's own propagation (derivation sec. 2 and 3).

The LP is face_dual.py's problem on a frozen face cache (the graph LP G1). Columns v = (x, z): the inputs and every
element of every face node (the nonlinear nodes of face_dual.install); affine nodes are not columns, their own forward
and backward fold them. Rows A v <= b:
    faces       lower a_j.z_in + b_j - z <= 0, upper z - a'_j.z_in - b'_j <= 0 (every face of the cache)
    outputs     C[k] f(v) <= thr[k] (prob.rows)
    base rows   E x <= h
and the column boxes [x_L, x_U] and [l_k, u_k] (the cache's node intervals); objective min C[0] f(v). Its dual is
DC-CROWN's bound with a multiplier per row (face_dual mode 'param').

A basis: every column basic or at a bound (L, U), every row tight (slack nonbasic) or not, as many tight rows as basic
columns. Per element:
    F    basic with a tight face, its primary face (a further tight face of it is a coupling row: type M)
    L/U  at a bound; a tight face of it is a coupling row
    C    basic without a tight face
The primary faces are triangular in the DAG order (Thm 1: CROWN's sign backward is such a basis). The k coupling
rows (tight output and base rows, tight faces that are not primary) replace the k anchors J (C elements and basic
inputs), derivation (2.2):  B = T + E_J V^T,  W = T^-1 E_J,  K = I + V^T W = A_C W  (W is the identity on the anchors).
The solves, derivation (2.3):
    F  T^-1: the graph's forward, every F element its primary face at its parents' values, the L/U ones at the bound,
       the anchors injected. W: the forward-mode tangents of one k-point batch (the linear part, exact). B^-1: the
       anchors from K du = the coupling rows' slacks at a base point, then one forward.
    R  T^-T: CROWN's backward (face_dual mode 'basis'), every F element passing its coefficient, of either sign,
       through its primary face, every other column keeping it as its residual. B^-T q: the coupling multipliers
       rho_C = -K^-T W^T q enter as spec rows (outputs), base-row multipliers and given face multipliers.
No matrix of the LP is formed; the only dense algebra is the k x k system K.

The dual simplex (Koberstein 2005; sec. 3.2): the leaving variable is the most violated row or basic column of
the primal basic solution; its pivot row is one R (the same backward as the fresh dual, in the spec dimension); the
ratio test passes the boxed columns' breakpoints while the slope stays positive (bound flipping). Finite under any
degeneracy by a lexicographic perturbation (2.4)-(2.7): every dual variable is a polynomial in eps whose
constant term is the true dual, the eps part identity on the starting basis. No breakpoint left: the pivot row is a
Farkas ray. Every reported bound is an independent face_dual 'param' backward of the clipped multipliers with the
residuals priced on the boxes (derivation (2.8), `Graph.certify`); a ray likewise (2.9).

All tensors carry the box dimension B; `solve` runs one box (B = 1).
"""
import math
import time

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD

from . import face_dual
from .bounded_tensor import BoundedTensor
from .perturbations import PerturbationLpNorm

INF = float('inf')
TOL = 1e-9               # floating-point equality only (ties, zero rates, feasibility); every bound is certified
BASIC, LOWER, UPPER = 0, 1, 2


class Basis:
    """stat (B, ncol): BASIC / LOWER / UPPER; tight (B, nrow): the row's slack is nonbasic; prim (B, ncol): the primary
    face (a row index) of a basic element, -1 for none (and for every input)."""

    def __init__(self, stat, tight, prim):
        self.stat, self.tight, self.prim = stat, tight, prim

    def clone(self):
        return Basis(self.stat.clone(), self.tight.clone(), self.prim.clone())


def _split(t):
    """(primal, tangent) of a forward-mode dual tensor; a zero tangent when there is none"""
    p, d = fwAD.unpack_dual(t)
    return p, (torch.zeros_like(p) if d is None else d)


class Graph:
    """The graph LP of `prob` on the face cache `cache` (face_dual.per_output's), in a flat layout: columns are the
    inputs, then the elements of every face node; rows are the faces of every element (per node: its lower faces,
    then its upper faces, face-major), the output rows prob.rows, the base rows. lo, hi (B, ncol): the column boxes."""

    def __init__(self, net, prob, cache):
        self.net, self.prob, self.cache = net, prob, cache
        self.B = prob.x_L.shape[0]
        self.n = prob.x_L[0].numel()
        self.node = {}           # name -> (first column, first row, elements E, lower faces Fl, upper faces Fu, shape)
        col, row = self.n, 0
        for name, f in cache.items():
            shape = tuple(f.lb.shape[2:])
            E, Fl, Fu = math.prod(shape), f.lb.shape[0], f.ub.shape[0]
            self.node[name] = (col, row, E, Fl, Fu, shape)
            col, row = col + E, row + (Fl + Fu) * E
        self.ncol, self.nface = col, row
        self.out = prob.rows
        self.nbase = 0 if prob.E is None else prob.E.shape[0]
        self.nrow = self.nface + len(self.out) + self.nbase
        B = self.B
        self.lo = torch.cat([prob.x_L.reshape(B, -1)] + [f.l.reshape(B, -1) for f in cache.values()], 1)
        self.hi = torch.cat([prob.x_U.reshape(B, -1)] + [f.u.reshape(B, -1) for f in cache.values()], 1)
        perturbed = [n.name for n in net.nodes() if n.perturbed and _nonlinear(n) and n.name not in cache]
        if perturbed:
            raise ValueError(f'nonlinear nodes without faces: {perturbed}')

    # ------------------------------------------------------------------------------------------- the basis
    def crown_start(self):
        """CROWN's sign backward of the objective (lambda = 0) with each side's default face (its largest default
        weight, the first on a tie): every element basic with the face of its arriving coefficient's sign (zero:
        lower, with zero mass), the inputs at the corner of the plane's signs, no output or base row tight."""
        net, prob, B = self.net, self.prob, self.B
        onehot = {}
        for name, f in self.cache.items():
            jl = f.dl.reshape(f.dl.shape[0], B, -1).argmax(0)
            ju = f.du.reshape(f.du.shape[0], B, -1).argmax(0)
            oh = lambda F, j: torch.zeros(F, *j.shape, dtype=f.lb.dtype).scatter_(0, j[None], 1.0).reshape(
                F, *f.lb.shape[1:])
            onehot[name] = (jl, ju, f._replace(dl=oh(f.dl.shape[0], jl), du=oh(f.du.shape[0], ju)))
        face_dual.install(net, net.final_name, 'sign', cache={k: v[2] for k, v in onehot.items()})
        try:
            spec = prob.C[:1].reshape(1, 1, -1).expand(B, 1, -1)
            with torch.no_grad():
                lA = self._compute(spec)
            stat = torch.full((B, self.ncol), BASIC, dtype=torch.long)
            stat[:, :self.n] = torch.where(lA[:, 0] >= 0, LOWER, UPPER)
            prim = torch.full((B, self.ncol), -1, dtype=torch.long)
            for name, (c0, r0, E, Fl, Fu, _) in self.node.items():
                A = vars(net[name])['face_dual'].used[0][0].reshape(B, E)
                jl, ju, _ = onehot[name]
                prim[:, c0:c0 + E] = r0 + torch.where(A >= 0, jl, Fl + ju) * E + torch.arange(E)
        finally:
            face_dual.uninstall(net)
        tight = torch.zeros(B, self.nrow, dtype=torch.bool)
        tight.scatter_(1, prim[:, self.n:], True)
        return Basis(stat, tight, prim)

    def structure(self, basis):
        """The triangular part and the coupling rows of `basis` (renewing basis.prim: a basic element keeps its
        primary face while it is tight, else takes its first tight face). Per box: the anchors J (columns of the C
        elements and the basic inputs) and the coupling rows (the tight rows that are no primary face), as index
        lists of equal length k."""
        stat, tight, prim, B = basis.stat, basis.tight, basis.prim, self.B
        for name, (c0, r0, E, Fl, Fu, _) in self.node.items():
            tf = tight[:, r0:r0 + (Fl + Fu) * E].reshape(B, Fl + Fu, E)
            old = prim[:, c0:c0 + E]
            j_old = torch.where(old >= 0, (old - r0) // E, torch.zeros_like(old))
            still = (old >= 0) & tf.gather(1, j_old[:, None]).squeeze(1)
            j = torch.where(still, j_old, tf.long().argmax(1))
            has = (stat[:, c0:c0 + E] == BASIC) & tf.any(1)
            prim[:, c0:c0 + E] = torch.where(has, r0 + j * E + torch.arange(E), torch.full_like(j, -1))
        primary = torch.zeros_like(tight)
        for b in range(B):
            p = prim[b][prim[b] >= 0]
            primary[b, p] = True
        J = [torch.nonzero((stat[b] == BASIC) & (prim[b] < 0)).reshape(-1) for b in range(B)]
        C = [torch.nonzero(tight[b] & ~primary[b]).reshape(-1) for b in range(B)]
        for b in range(B):
            if len(J[b]) != len(C[b]):
                raise RuntimeError(f'not a basis: {len(J[b])} anchors, {len(C[b])} coupling rows')
        return J, C

    def selection(self, prim):
        """{name: (lower (Fl, B, *shape), upper (Fu, B, *shape))}: the primary face of every element, one-hot"""
        out = {}
        dt = self.prob.x_L.dtype
        for name, (c0, r0, E, Fl, Fu, shape) in self.node.items():
            p = prim[:, c0:c0 + E]
            j = torch.where(p >= 0, (p - r0) // E, torch.zeros_like(p))
            oh = torch.zeros(Fl + Fu, self.B, E, dtype=dt).scatter_(0, j[None], (p >= 0)[None].to(dt))
            out[name] = (oh[:Fl].reshape(Fl, self.B, *shape), oh[Fl:].reshape(Fu, self.B, *shape))
        return out

    # ------------------------------------------------------------------------------------------- the solves
    def forward(self, sel, v, dv=None):
        """F: the graph's forward at P points per box. Every element with a selected face is that face at its parents'
        values plus v (its column), every other element is v; the inputs are v[..., :n]. v (B, P, ncol); dv the same
        shape: forward-mode tangents (the linear part of the map, exact). Returns the columns (B, P, ncol), the
        objective C[0] f (B, P), the slacks b - A v of every row (B, P, nrow), a scale of each row's terms
        (B, P, nrow) and, with dv, the tangents of the first three."""
        net, prob, B, P = self.net, self.prob, v.shape[0], v.shape[1]
        rep = lambda t: t.repeat_interleave(P, dim=0)              # (B, ...) -> (B * P, ...)
        rep1 = lambda t: t.repeat_interleave(P, dim=1)             # (F, B, ...) -> (F, B * P, ...)
        tan = dv is not None
        saved = []
        with torch.no_grad(), fwAD.dual_level():
            def point(c0, E, shape):
                t = v[:, :, c0:c0 + E].reshape(B * P, *shape).contiguous()
                return fwAD.make_dual(t, dv[:, :, c0:c0 + E].reshape(B * P, *shape).contiguous()) if tan else t
            x = point(0, self.n, prob.x_L.shape[1:])
            try:
                for name, (c0, r0, E, Fl, Fu, shape) in self.node.items():
                    f, (sl, su) = self.cache[name], sel[name]
                    on = lambda w, s: torch.where(s > 0, w, torch.zeros_like(w))
                    ws = [rep((on(a, sl)).sum(0) + (on(b, su)).sum(0)) for a, b in zip(f.lw, f.uw)]
                    bs = rep(on(f.lb, sl).sum(0) + on(f.ub, su).sum(0))
                    has = rep(sl.sum(0) + su.sum(0))
                    z0 = point(c0, E, shape)

                    def fwd(*inputs, _ws=ws, _bs=bs, _has=has, _z0=z0):
                        return _has * (sum(w * z for w, z in zip(_ws, inputs)) + _bs) + _z0
                    node = net[name]
                    saved.append(node)
                    vars(node)['forward'] = fwd
                net.set_input(x, clear_forward_only=True, reset_perturbed_nodes=False)
                out = net.get_forward_value(net[net.output_name[0]]).reshape(B * P, -1)
                cols, slack, scale = [x.reshape(B * P, -1)], [], []
                for name, (c0, r0, E, Fl, Fu, shape) in self.node.items():
                    f, node = self.cache[name], net[name]
                    zin = [net.get_forward_value(i) for i in node.inputs[:len(f.lw)]]
                    z = net.get_forward_value(node)
                    cols.append(z.reshape(B * P, E))
                    terms_l = [rep1(a) * zi for a, zi in zip(f.lw, zin)]
                    terms_u = [rep1(a) * zi for a, zi in zip(f.uw, zin)]
                    sl_ = z - sum(terms_l) - rep1(f.lb)
                    su_ = sum(terms_u) + rep1(f.ub) - z
                    for s_, terms, b_ in ((sl_, terms_l, f.lb), (su_, terms_u, f.ub)):
                        F = s_.shape[0]
                        slack.append(s_.reshape(F, B * P, E).transpose(0, 1).reshape(B * P, F * E))
                        mag = 1 + _p(z).abs() + sum(_p(t).abs() for t in terms) + rep1(b_).abs()
                        scale.append(mag.reshape(F, B * P, E).transpose(0, 1).reshape(B * P, F * E))
                rows = prob.C[self.out]
                fo = out @ rows.t()
                slack.append(prob.thr[self.out] - fo)
                scale.append(1 + prob.thr[self.out].abs() + _p(fo).abs())
                if self.nbase:
                    ex = x.reshape(B * P, -1) @ prob.E.t()
                    slack.append(prob.h - ex)
                    scale.append(1 + prob.h.abs() + _p(ex).abs())
                cols, slack = torch.cat(cols, 1), torch.cat(slack, 1)
                scale = torch.cat(scale, 1)
                obj = out @ prob.C[0]
                res = [_split(t) for t in (cols, obj, slack)]
            finally:
                for node in saved:
                    vars(node).pop('forward', None)
        shape = lambda t, *s: t.reshape(B, P, *s)
        (c, dc), (o, do), (s, ds) = res
        ret = (shape(c, -1), shape(o), shape(s, -1), shape(scale, -1))
        return ret + ((shape(dc, -1), shape(do), shape(ds, -1)) if tan else ())

    def _compute(self, spec, box=None):
        """one lower-bound backward of spec (B, S, n_out) from the input box (or `box`); the input coefficient
        (B, S, n), and (with `box`) the bound (B, S)"""
        net, prob = self.net, self.prob
        x = BoundedTensor(prob.x_L, box or PerturbationLpNorm(x_L=prob.x_L, x_U=prob.x_U))
        final, inp = net.final_name, net.input_name[0]
        ret = net.compute_bounds(x=(x,), C=spec, method='backward', bound_upper=False, return_A=True,
                                 needed_A_dict={final: [inp]}, interm_bounds=prob.interm_bounds)
        d = ret[-1].get(final, {}).get(inp)
        lA = (torch.zeros(spec.shape[0], spec.shape[1], self.n, dtype=spec.dtype) if d is None or d['lA'] is None
              else d['lA'].reshape(spec.shape[0], spec.shape[1], -1))
        return (lA, ret[0].reshape(spec.shape[0], spec.shape[1])) if box is not None else lA

    def backward(self, sel, out=None, base=None, face=None, col=None):
        """R: one CROWN backward (face_dual mode 'basis') of S right-hand sides per box, q = the weights out
        (B, S, len(C)) on the rows of C (row 0 the objective), base (B, S, nbase) on the base rows, face (B, S, nface)
        on the faces, and col (B, S, ncol) coefficients injected at the columns. Every element with a selected face
        passes its coefficient through it. Returns y (B, S, nrow), the multiplier of every row (the given ones and
        the selected faces'), and r (B, S, ncol), the residual coefficient of every column (c + A^T y)."""
        net, prob, B = self.net, self.prob, self.B
        S = next(t.shape[1] for t in (out, base, face, col) if t is not None)
        dt = prob.x_L.dtype
        out = torch.zeros(B, S, prob.C.shape[0], dtype=dt) if out is None else out
        spec = out @ prob.C
        face_dual.install(net, net.final_name, 'basis', cache=self.cache)
        try:
            for name, (c0, r0, E, Fl, Fu, shape) in self.node.items():
                st = vars(net[name])['face_dual']
                st.sel = sel[name]
                if face is not None:
                    m = face[:, :, r0:r0 + (Fl + Fu) * E].reshape(B, S, Fl + Fu, E).permute(2, 1, 0, 3)
                    m = m.reshape(Fl + Fu, S, B, *shape)
                    st.mu, st.nu = m[:Fl], m[Fl:]
                if col is not None:
                    st.inject = col[:, :, c0:c0 + E].transpose(0, 1).reshape(S, B, *shape)
            with torch.no_grad():
                rx = self._compute(spec)
            ys, rs = [], [rx]
            for name, (c0, r0, E, Fl, Fu, shape) in self.node.items():
                st = vars(net[name])['face_dual']
                _, mu, nu = st.used
                ys.append(torch.cat([mu, nu]).reshape(Fl + Fu, S, B, E).permute(2, 1, 0, 3).reshape(B, S, -1))
                rs.append(st.r.reshape(S, B, E).transpose(0, 1))
        finally:
            face_dual.uninstall(net)
        ys.append(out[:, :, self.out])
        if self.nbase:
            base = torch.zeros(B, S, self.nbase, dtype=dt) if base is None else base
            ys.append(base)
            rs[0] = rs[0] + base @ prob.E
        r = torch.cat(rs, -1)
        if col is not None:
            r[:, :, :self.n] += col[:, :, :self.n]
        return torch.cat(ys, -1), r

    def certify(self, y, objective=1.0):
        """D(y) (B,): DC-CROWN's bound (face_dual mode 'param') with the multipliers y (B, nrow) clipped at 0, every
        residual coefficient priced on its box (derivation (2.8)). With objective 0: the Farkas value of the ray y, whose
        positivity proves the LP infeasible (derivation (2.9))."""
        net, prob, B = self.net, self.prob, self.B
        y = y.clamp(min=0)
        face_dual.install(net, net.final_name, 'param', cache=self.cache)
        try:
            for name, (c0, r0, E, Fl, Fu, shape) in self.node.items():
                st = vars(net[name])['face_dual']
                m = y[:, r0:r0 + (Fl + Fu) * E].reshape(B, Fl + Fu, E).transpose(0, 1).reshape(Fl + Fu, 1, B, *shape)
                st.mu, st.nu = m[:Fl], m[Fl:]
            lam = y[:, self.nface:self.nface + len(self.out)]
            box = face_dual.LagrangianBox(prob.x_L, prob.x_U, prob.E, prob.h)
            if self.nbase:
                box.eta = y[:, self.nface + len(self.out):].reshape(B, 1, -1)
            spec = (objective * prob.C[0] + lam @ prob.C[self.out]).reshape(B, 1, -1)
            with torch.no_grad():
                _, bound = self._compute(spec, box)
        finally:
            face_dual.uninstall(net)
        return bound[:, 0] - lam @ prob.thr[self.out]


def _p(t):
    return fwAD.unpack_dual(t).primal


def _nonlinear(node):
    """a node whose forward is not affine in its perturbed inputs (it must be a face node of the LP)"""
    from .operators import BoundActivation, BoundMul, BoundLinear
    if isinstance(node, BoundMul):
        return not node.is_linear_op
    if isinstance(node, BoundLinear):
        return sum(i.perturbed for i in node.inputs[:2]) > 1
    return isinstance(node, BoundActivation)


def _lex_order(T):
    """the rows of T (n, m) in lexicographic order, entries within TOL (relative) taken as equal"""
    T = T.cpu().numpy()

    def order(idx, c):
        if len(idx) <= 1 or c == T.shape[1]:
            return list(idx)
        idx = idx[np.argsort(T[idx, c], kind='stable')]
        col, out, a = T[idx, c], [], 0
        for b in range(1, len(idx) + 1):
            if b == len(idx) or col[b] - col[a] > TOL * (1 + abs(col[a])):
                out += order(idx[a:b], c + 1)
                a = b
        return out
    return order(np.arange(T.shape[0]), 0)


class Factor:
    """The basis solves of one basis (B = 1). Its structure (G.structure: the anchors J, the coupling rows C, k of
    each), W and K from one k-point forward batch at a base point (the anchors at `prev`, the previous primal solution,
    or the box midpoint; the nonbasic columns at their bounds), and the primal basic solution B^-1 b: the anchors from
    K du = the coupling rows' slacks at the base point, then one forward: the columns v, the objective, every row's
    slack s and its scale sc (b - A v; the coupling rows' and the primary faces' are zero up to rounding)."""

    def __init__(self, G, bs, prev=None):
        self.G, self.bs = G, bs
        J, C = G.structure(bs)
        self.J, self.C, self.k = J[0], C[0], len(J[0])
        self.sel = G.selection(bs.prim)
        dt, ncol, k = G.prob.x_L.dtype, G.ncol, self.k
        st = bs.stat[0]
        base = torch.where(st == LOWER, G.lo[0], torch.where(st == UPPER, G.hi[0], torch.zeros(ncol, dtype=dt)))
        anchor = torch.zeros(ncol, dtype=torch.bool)
        anchor[self.J] = True
        base = torch.where(anchor, (G.lo[0] + G.hi[0]) / 2 if prev is None else prev, base)
        self.forwards = 1
        if k:
            dv = torch.zeros(1, k, ncol, dtype=dt)
            dv[0, torch.arange(k), self.J] = 1.0
            _, _, s0, _, Wv, Wobj, Ws = G.forward(self.sel, base.expand(1, k, ncol), dv)
            self.Wv, self.Wobj, self.Ws = Wv[0], Wobj[0], Ws[0]     # (k, ncol), (k,), (k, nrow): W, c.W, -A W
            self.K = -self.Ws[:, self.C].t()                        # K[r, r'] = a_{C_r} . W_{r'}
            base = base.clone()
            base[self.J] += torch.linalg.solve(self.K, s0[0, 0, self.C])
            self.forwards += 1
        v, obj, s, sc = G.forward(self.sel, base.reshape(1, 1, ncol))
        self.v, self.obj, self.s, self.sc = v[0, 0], float(obj[0, 0]), s[0, 0], sc[0, 0]

    def transpose(self, obj, row, col):
        """B^-T q for S right-hand sides q = obj (S,) x the objective + row (S, nrow) weights on the rows + col
        (S, ncol) coefficients on the columns: the coupling multipliers rho_C = -K^-T W^T q, then one backward.
        Returns y (S, nrow), the multipliers (the given ones, rho_C and the primary faces'), and r (S, ncol), the
        residual coefficients (zero at the basic columns up to rounding)."""
        G = self.G
        row = row.clone()
        if self.k:
            WTq = obj[:, None] * self.Wobj[None] - row @ self.Ws.t() + col @ self.Wv.t()     # (S, k)
            row[:, self.C] += torch.linalg.solve(self.K.t(), -WTq.t()).t()
        no = len(G.out)
        out = torch.zeros(row.shape[0], G.prob.C.shape[0], dtype=row.dtype)
        out[:, 0] = obj
        out[:, G.out] += row[:, G.nface:G.nface + no]
        y, r = G.backward(self.sel, out=out[None], base=row[None, :, G.nface + no:] if G.nbase else None,
                          face=row[None, :, :G.nface], col=col[None])
        return y[0], r[0]


def solve(net, prob, cache, max_pivots=None, target=None, basis=None, log=None, diag=False, allow=None, certify=None):
    """The graph LP of one box (B = 1) by Pivoting CROWN from `basis` (default: CROWN's sign basis, crown_start).
    Stops at optimality (the primal basic solution feasible: the certified bound is G1), a certified Farkas ray,
    max_pivots (default 3 x the rows), or, with `target`, the first certified bound >= target. allow (nrow,) bool:
    only these rows may become tight (the LP without the others, a relaxation: its bounds and rays hold for G1;
    'optimal' then means optimal for it). Per iteration: one
    k-point forward batch (W), one forward (the primal basic solution and every row's slack), one backward of two
    right-hand sides (the fresh dual and the pivot row), one certifying backward. diag: also cond(K) per iteration.
    certify: a function (y (B, nrow), objective=1.0) -> (B,) in place of Graph.certify (compiled_crown.certifier: the
    rigorous one). Returns {'status', 'bound' (the best certified), 'farkas', 'pivots', 'trace', 'basis', 'v', 'y', ...}."""
    G = Graph(net, prob, cache)
    certify = G.certify if certify is None else certify
    if G.B != 1:
        raise NotImplementedError('one box per solve')
    bs = basis.clone() if basis is not None else G.crown_start()
    max_pivots = 3 * G.nrow if max_pivots is None else max_pivots
    dt = prob.x_L.dtype
    nrow, ncol = G.nrow, G.ncol
    # the dual variables (tight rows, nonbasic columns) as polynomials in eps: column 0 the value, the eps part the
    # identity on the starting basis (derivation (2.4)); natural signs (row multipliers; reduced costs)
    dual_rows = torch.nonzero(bs.tight[0]).reshape(-1)
    dual_cols = torch.nonzero(bs.stat[0] != BASIC).reshape(-1)
    N1 = len(dual_rows) + len(dual_cols)
    Pr = torch.zeros(nrow, 1 + N1, dtype=dt)
    Pc = torch.zeros(ncol, 1 + N1, dtype=dt)
    Pr[dual_rows, 1 + torch.arange(len(dual_rows))] = 1.0
    Pc[dual_cols, 1 + len(dual_rows) + torch.arange(len(dual_cols))] = torch.where(
        bs.stat[0, dual_cols] == LOWER, 1.0, -1.0).to(dt)
    rng = (G.hi - G.lo)[0]
    lo, hi = G.lo[0], G.hi[0]
    sweeps = {'F_batch': 0, 'F_batch_points': 0, 'F': 0, 'R': 0, 'R_specs': 0, 'R_cert': 0}
    trace, best, status, farkas, v_prev = [], -INF, 'budget', None, None
    pivots = degenerate = flips_total = 0
    t_start = time.time()
    for it in range(max_pivots + 1):
        t0 = time.time()
        # ---- the primal basic solution
        f = Factor(G, bs, v_prev)
        sweeps['F_batch'] += int(f.k > 0)
        sweeps['F_batch_points'] += f.k
        sweeps['F'] += 1
        v, s, sc, k, Crow = f.v, f.s, f.sc, f.k, f.C
        v_prev = v
        coupling_residual = float((s[Crow].abs() / sc[Crow]).max()) if k else 0.0
        # ---- leaving variable: the most violated row or basic column (scaled infeasibility)
        st, tight = bs.stat[0], bs.tight[0]
        basic = st == BASIC
        vr = (~tight) & (s < -TOL * sc) & (True if allow is None else allow)
        vl = basic & (v < lo - TOL * (1 + lo.abs()))
        vh = basic & (v > hi + TOL * (1 + hi.abs()))
        score = torch.cat([torch.where(vr, -s / (1 + s.abs()), torch.zeros_like(s)),
                           torch.where(vl, (lo - v) / (1 + v.abs()), torch.where(vh, (v - hi) / (1 + v.abs()),
                                                                                 torch.zeros_like(v)))])
        optimal = not bool((score > 0).any())
        # ---- one backward of [the objective, the pivot row's right-hand side]
        S = 1 if optimal else 2
        obj = torch.zeros(S, dtype=dt)
        obj[0] = 1.0
        rowq = torch.zeros(S, nrow, dtype=dt)
        colq = torch.zeros(S, ncol, dtype=dt)
        if not optimal:
            p = int(score.argmax())
            if p < nrow:
                kind, delta = 'row', float(-s[p])
                rowq[1, p] = 1.0
            else:
                p -= nrow
                kind = 'col_lo' if bool(vl[p]) else 'col_hi'
                delta = float(lo[p] - v[p]) if kind == 'col_lo' else float(v[p] - hi[p])
                colq[1, p] = -1.0 if kind == 'col_lo' else 1.0
        y, r = f.transpose(obj, rowq, colq)
        sweeps['R'] += 1
        sweeps['R_specs'] += S
        y0 = torch.where(tight, y[0], torch.zeros_like(y[0]))
        r0 = torch.where(basic, torch.zeros_like(r[0]), r[0])
        stationarity = float(r[0][basic].abs().max()) if bool(basic.any()) else 0.0
        sig_c = torch.where(st == LOWER, r0, -r0)
        dual_infeasibility = max([0.0] + [float(-t.min()) for t in (y0[tight], sig_c[~basic]) if t.numel()])
        D = float(certify(y0[None])[0])
        sweeps['R_cert'] += 1
        best = max(best, D)
        rec = {'it': it, 'D_cert': D, 'primal': f.obj, 'k': k, 'coupling_residual': coupling_residual,
               'stationarity': stationarity, 'dual_infeasibility': dual_infeasibility}
        if diag and k:
            rec['cond_K'] = float(torch.linalg.cond(f.K))
        stop = 'optimal' if optimal else ('target' if target is not None and D >= target else
                                          ('budget' if it == max_pivots else None))
        if stop is not None:
            status = stop
            rec['seconds'] = time.time() - t0
            trace.append(rec)
            break
        # ---- the bounded ratio test with bound flipping, lexicographic (derivation (2.6))
        rho_r = torch.where(tight, y[1], torch.zeros_like(y[1]))
        alpha = torch.where(basic, torch.zeros_like(r[1]), r[1])
        sgn = torch.where(st == LOWER, 1.0, -1.0).to(dt)
        if it:
            rec['drift'] = float(torch.cat([Pr[:, 0] - y0, Pc[:, 0] - r0]).abs().max()) / (1 + float(y0.abs().max()))
        Pr[:, 0], Pc[:, 0] = y0, r0
        rate = torch.cat([rho_r, sgn * alpha])
        polys = torch.cat([Pr, sgn[:, None] * Pc])
        # a dual value is >= 0 in exact arithmetic: a negative one is rounding, read as 0 (no backward step); among
        # equal values the eps part, divided by |rate|, puts a small pivot element last
        if diag:
            rec['sigma'], rec['rate'] = polys[:, 0].tolist(), rate.tolist()
        polys[:, 0] = polys[:, 0].clamp(min=0)
        cand = torch.nonzero(rate < -TOL * float(rate.abs().max())).reshape(-1)
        T = polys[cand] / (-rate[cand])[:, None]
        slope, flips, q = delta, [], None
        for i in _lex_order(T):
            j = int(cand[i])
            if j >= nrow:
                dec = float(alpha[j - nrow].abs() * rng[j - nrow])
                if math.isfinite(dec) and slope - dec > 0:
                    flips.append(j - nrow)
                    slope -= dec
                    continue
            q, Theta = j, T[i]
            break
        rec.update(leave=kind, delta=delta, flips=len(flips), candidates=len(cand),
                   pivot_rate=float(rate[q].abs()) / float(rate.abs().max()) if q is not None else None)
        if q is None:
            # no breakpoint left: the pivot row is a ray of the dual (Farkas), certified with the objective 0
            w = rho_r.clone()
            if kind == 'row':
                w[p] = 1.0
            phi = float(certify(w[None], objective=0.0)[0])
            sweeps['R_cert'] += 1
            farkas = {'value': phi, 'ray_min': float(rho_r.min()), 'slope_left': slope, 'w': w}
            status = 'infeasible' if phi > 0 else 'farkas_uncertified'
            rec.update(farkas=phi, seconds=time.time() - t0)
            trace.append(rec)
            break
        # ---- the update: the polynomials (y += Theta rho, d += Theta alpha), the flips, the exchange
        Pr[tight] += rho_r[tight, None] * Theta[None]
        Pc[~basic] += alpha[~basic, None] * Theta[None]
        for j in flips:
            st[j] = UPPER if int(st[j]) == LOWER else LOWER
        if q < nrow:
            bs.tight[0, q] = False
            Pr[q] = 0
            rec['enter'] = 'row'
        else:
            st[q - nrow] = BASIC
            Pc[q - nrow] = 0
            rec['enter'] = 'col'
        if kind == 'row':
            bs.tight[0, p] = True
            Pr[p] = Theta
        else:
            st[p] = LOWER if kind == 'col_lo' else UPPER
            Pc[p] = Theta if kind == 'col_lo' else -Theta
        pivots += 1
        flips_total += len(flips)
        deg = float(Theta[0]) * delta <= TOL * (1 + abs(D))
        degenerate += int(deg)
        rec.update(theta=float(Theta[0]), degenerate=deg, seconds=time.time() - t0)
        trace.append(rec)
        if log is not None and it % log == 0:
            print(f'  it {it} D {D:.12g} k {k} leave {kind} enter {rec["enter"]} theta {float(Theta[0]):.3g} '
                  f'flips {len(flips)}', flush=True)
    return {'status': status, 'bound': best, 'final': trace[-1]['D_cert'], 'farkas': farkas, 'pivots': pivots,
            'degenerate': degenerate, 'flips': flips_total, 'trace': trace, 'basis': bs, 'v': v_prev, 'y': y0,
            'm': nrow, 'ncol': ncol, 'sweeps': sweeps, 'seconds': time.time() - t_start}
