"""Exact sparse derivatives of an NLP from the structure walk's forms.

The structure walk (structure.walk) writes every model output as an exact Form: a sparse polynomial over the inputs x
and interned atoms a = phi(p_a), where phi is exp, log, sin, cos, tan, tanh, abs or a power and p_a is a polynomial over
x and earlier atoms. `FormTape` compiles a list of such polynomials (its owners) once into flat index arrays. At a point
it returns the owners' values, their Jacobian on a fixed structural pattern, and the Hessian of a weighted sum of them
(a Lagrangian) on a fixed structural lower-triangular pattern:

- a term c * prod_j v_j^e_j gives its partials as products of the other factors (no division, so zeros are safe);
- atoms enter by the chain rule. With node values v = (x, atoms) and Z = dv/dx, the Hessian of sum_o w_o p_o(v(x)) is
  Z^T K Z, where K = sum_o w_o d2p_o + sum_a (w_a d2p_a + omega_a phi_a'' dp_a dp_a^T), omega_a is the adjoint of
  atom a and w_a = omega_a phi_a'(p_a);
- without atoms (affine, quadratic and polynomial rows), Z = I and both patterns are filled directly.

`SparseNLP` is the IPOPT view of one problem on the tape: the objective plus the problem's rows, merged by exact
direction the way the RowPool merges them, so an equality becomes ONE ranged row. `sparse_problem(problem)` builds it
for an NNProblemIPOPT from the walk that the IPOPT path already runs, keeps it per model, and refuses a tape that does
not reproduce the model's own arithmetic. The judge stays the original model: IPOPT's callbacks use the tape, and every
returned point is evaluated by the torch model.
"""
import copy
import hashlib
import threading
import time
import weakref
from fractions import Fraction

import numpy as np
import scipy.sparse as sp
import torch

from nlp_solver.structure.rowpool import _direction

# kinds whose scalar function is twice differentiable on its domain; abs is not (at 0)
_SMOOTH = {'exp', 'log', 'sin', 'cos', 'tan', 'tanh', 'pow'}


def _phi(kind, param, u):
    """(phi, phi', phi'') of an atom's scalar function at u"""
    if kind == 'exp':
        e = np.exp(u)
        return e, e, e
    if kind == 'log':
        r = 1.0 / u
        return np.log(u), r, -r * r
    if kind == 'sin':
        s, c = np.sin(u), np.cos(u)
        return s, c, -s
    if kind == 'cos':
        s, c = np.sin(u), np.cos(u)
        return c, -s, -c
    if kind == 'tan':
        t = np.tan(u)
        s = 1.0 + t * t
        return t, s, 2.0 * t * s
    if kind == 'tanh':
        t = np.tanh(u)
        s = 1.0 - t * t
        return t, s, -2.0 * t * s
    if kind == 'abs':
        return np.abs(u), np.sign(u), np.zeros_like(u)
    if kind == 'pow':
        k = float(param)
        return np.power(u, k), k * np.power(u, k - 1), k * (k - 1) * np.power(u, k - 2)
    raise ValueError(f'atom kind {kind}')


def _prod_except(pw, skip):
    """product over the columns of pw not in skip (ones when none is left)"""
    keep = [j for j in range(pw.shape[1]) if j not in skip]
    return pw[:, keep].prod(1) if keep else np.ones(pw.shape[0])


class _Bucket:
    """the terms with d factors: owner, coefficient, (T, d) node indices and exponents"""
    def __init__(self, own, coef, node, ex):
        self.own, self.coef, self.node, self.ex = own, coef, node, ex
        self.d = node.shape[1]
        self.linear = bool((ex == 1).all())
        self.diag = [ex[:, j] != 1 for j in range(self.d)]       # a second partial in one factor exists
        self.pairs = [(j, k) for j in range(self.d) for k in range(j + 1, self.d)]


class _Poly:
    """Owner o is const[o] + the sum of its terms c * prod_j v[node_j] ** e_j over the node values v."""

    def __init__(self, polys, node_of):
        self.size = len(polys)
        self.const = np.zeros(self.size)
        groups = {}
        for o, poly in enumerate(polys):
            for mono, c in poly.items():
                if not mono:
                    self.const[o] += float(c)
                    continue
                g = groups.setdefault(len(mono), ([], [], [], []))
                g[0].append(o)
                g[1].append(float(c))
                g[2].extend(node_of[v] for v, _ in mono)
                g[3].extend(float(e) for _, e in mono)
        self.buckets = [_Bucket(np.array(own, dtype=np.int64), np.array(coef),
                                np.array(node, dtype=np.int64).reshape(-1, d), np.array(ex).reshape(-1, d))
                        for d, (own, coef, node, ex) in sorted(groups.items())]

    def first_pattern(self):
        """(owner, node) of every first partial, in the order of `evaluate`'s first partials"""
        rows = [np.repeat(b.own, b.d) for b in self.buckets]
        cols = [b.node.ravel() for b in self.buckets]
        return _cat(rows, np.int64), _cat(cols, np.int64)

    def second_pattern(self):
        """(owner, node_a, node_b), node_a >= node_b, of every second partial, in the order of `evaluate`"""
        own, ra, rb = [], [], []
        for b in self.buckets:
            for j in range(b.d):
                m = b.diag[j]
                own.append(b.own[m])
                ra.append(b.node[m, j])
                rb.append(b.node[m, j])
            for j, k in b.pairs:
                own.append(b.own)
                ra.append(np.maximum(b.node[:, j], b.node[:, k]))
                rb.append(np.minimum(b.node[:, j], b.node[:, k]))
        return _cat(own, np.int64), _cat(ra, np.int64), _cat(rb, np.int64)

    def evaluate(self, v, order=0):
        """(values, first partials, second partials) at the node values v; partials up to `order`"""
        val = self.const.copy()
        first, second = [], []
        for b in self.buckets:
            x = v[b.node]
            pw = x if b.linear else np.power(x, b.ex)
            val += np.bincount(b.own, b.coef * pw.prod(1), minlength=self.size)
            if order < 1:
                continue
            d1 = None if b.linear else b.ex * np.power(x, b.ex - 1)
            excl = [_prod_except(pw, (j,)) for j in range(b.d)]
            for j in range(b.d):
                f = b.coef * excl[j]
                first.append(f if b.linear else f * d1[:, j])
            if order < 2:
                continue
            if not b.linear:
                d2 = b.ex * (b.ex - 1) * np.power(x, b.ex - 2)
                for j in range(b.d):
                    m = b.diag[j]
                    second.append(b.coef[m] * d2[m, j] * excl[j][m])
            else:
                second.extend(np.zeros(0) for _ in range(b.d))
            for j, k in b.pairs:
                s = b.coef * _prod_except(pw, (j, k))
                second.append(s if b.linear else s * d1[:, j] * d1[:, k])
        return val, _reorder_first(first, self.buckets), _cat(second, np.float64)

    def affine_in(self, v, free):
        """Every owner as const[o] + sum_j a[o, j] v_j over the free nodes, the other nodes held at their values in v:
        (rows, cols, values, const). Raises ValueError when a term is not affine in the free nodes (two free factors,
        or a free factor with an exponent other than 1)."""
        const = self.const.copy()
        rows, cols, vals = [], [], []
        for b in self.buckets:
            fr = free[b.node]
            nf = fr.sum(1)
            if np.any(nf > 1) or np.any(fr & (b.ex != 1)):
                raise ValueError('a term is not affine in the free nodes')
            x = v[b.node]
            pw = np.where(fr, 1.0, x if b.linear else np.power(x, b.ex))
            coef = b.coef * pw.prod(1)
            lin = nf == 1
            const += np.bincount(b.own[~lin], coef[~lin], minlength=self.size)
            rows.append(b.own[lin])
            cols.append(b.node[lin][np.arange(int(lin.sum())), np.argmax(fr[lin], 1)])
            vals.append(coef[lin])
        return _cat(rows, np.int64), _cat(cols, np.int64), _cat(vals, np.float64), const


def _reorder_first(first, buckets):
    """first partials appended factor by factor per bucket -> the (term, factor) order of first_pattern"""
    out, i = [], 0
    for b in buckets:
        out.append(np.stack(first[i:i + b.d], 1).ravel())
        i += b.d
    return _cat(out, np.float64)


def _cat(parts, dtype):
    return np.concatenate(parts).astype(dtype, copy=False) if parts else np.zeros(0, dtype)


def _csr(rows, cols, vals, shape):
    return sp.csr_matrix((vals, (rows, cols)), shape=shape)


def _sample(M, rows, cols):
    """the entries M[rows[i], cols[i]] of a sparse matrix (0 where M has none)"""
    M = sp.csr_matrix(M)
    M.sum_duplicates()
    return np.asarray(M[rows, cols]).ravel() if len(rows) else np.zeros(0)


class FormTape:
    """Owners' exact polynomials over the inputs and atoms, with their values, Jacobian and weighted Hessian at a point.

    polys: {monomial: coefficient} per owner, monomials over input indices (< n) and atom ids (>= n) of the walk's
    algebra; atom_info: that algebra's atom id -> (kind, param, argument items, support). Raises ValueError for an atom
    kind without a scalar function here."""

    def __init__(self, polys, atom_info, n):
        self.n, self.size = n, len(polys)
        level = {}
        stack = [v for p in polys for m in p for v, _ in m if v >= n]
        while stack:
            a = stack.pop()
            if a in level:
                continue
            level[a] = None
            stack.extend(v for m, _ in atom_info[a][2] for v, _ in m if v >= n)
        for a in sorted(level):                     # atom ids are created in topological order
            level[a] = 1 + max((level[v] for m, _ in atom_info[a][2] for v, _ in m if v >= n), default=0)
        order = sorted(level, key=lambda a: (level[a], a))
        node_of = {j: j for j in range(n)}
        node_of.update({a: n + i for i, a in enumerate(order)})
        self.N = n + len(order)
        self.levels = []
        start = 0
        for lv in sorted(set(level.values())):
            atoms = [a for a in order if level[a] == lv]
            arg = _Poly([dict(atom_info[a][2]) for a in atoms], node_of)
            groups = {}
            for i, a in enumerate(atoms):
                kind, param = atom_info[a][0], atom_info[a][1]
                _phi(kind, param, np.ones(1))       # unsupported kinds fail here, at compile time
                groups.setdefault((kind, param), []).append(i)
            groups = [(k, p, np.array(ix, dtype=np.int64)) for (k, p), ix in groups.items()]
            self.levels.append(_Level(arg, groups, start, n))
            start += len(atoms)
        self.n_atoms = start
        self.hessian_available = all(k in _SMOOTH for lv in self.levels for k, _, _ in lv.groups)
        self.out = _Poly(polys, node_of)
        self._out_rows, self._out_cols = self.out.first_pattern()
        self._out_second = self.out.second_pattern()
        self._structure()
        self._local = threading.local()        # the last point's forward pass, per thread (IPOPT-in-BaB runs threads)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['_local']                     # per-thread caches stay in their process
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()

    # ------------------------------------------------------------------------------------------------ structure
    def _structure(self):
        """Fixed patterns: the Jacobian (owner, x) entries and the Hessian's lower triangle (structural, so a superset
        of every numeric nonzero), and the direct scatter positions when there are no atoms."""
        n, N = self.n, self.N
        Ja = sp.csr_matrix((0, n))
        for lv in self.levels:
            ones = np.ones(len(lv.rows))
            Jl = _csr(lv.rows[lv.mx], lv.cols[lv.mx], ones[lv.mx], (lv.size, n))
            if lv.start:
                Jl = Jl + _csr(lv.rows[~lv.mx], lv.cols[~lv.mx] - n, ones[~lv.mx], (lv.size, lv.start)) @ Ja
            Ja = sp.vstack([Ja, Jl]).tocsr()
        J = self._compose(self._out_rows, self._out_cols, np.ones(len(self._out_rows)), Ja)
        J = J.tocoo()
        keep = J.row >= 1
        order = np.lexsort((J.col[keep], J.row[keep]))
        self.jac_rows, self.jac_cols = J.row[keep][order] - 1, J.col[keep][order]   # constraint owners only
        own, ra, rb = self._out_second
        K = [_csr(ra, rb, np.ones(len(ra)), (N, N))]
        for lv in self.levels:
            _, la, lb = lv.second
            K.append(_csr(la, lb, np.ones(len(la)), (N, N)))
            P = _csr(lv.rows, lv.cols, np.ones(len(lv.rows)), (lv.size, N))
            K.append(P.T @ P)
        K = sum(K[1:], K[0])
        K = K + K.T
        Z = sp.vstack([sp.identity(n, format='csr'), Ja]).tocsr()
        H = sp.tril(Z.T @ K @ Z).tocoo()
        order = np.lexsort((H.col, H.row))
        self.hess_rows, self.hess_cols = H.row[order], H.col[order]
        if not self.levels:     # direct scatter positions of every partial into the fixed patterns
            m = len(self.jac_rows)
            jkey = self.jac_rows * n + self.jac_cols
            r, c = self._out_rows, self._out_cols
            pos = np.searchsorted(jkey, (r - 1) * n + c)
            self._jpos = np.where(r == 0, m + c, pos)         # the objective's partials go to the gradient
            self._jlen = m + n
            hkey = self.hess_rows * n + self.hess_cols
            self._hpos = np.searchsorted(hkey, ra * n + rb)

    def _compose(self, rows, cols, vals, Ja):
        """owners x inputs Jacobian from the owners' first partials over the nodes and the atoms' Jacobian Ja"""
        mx = cols < self.n
        J = _csr(rows[mx], cols[mx], vals[mx], (self.size, self.n))
        if self.n_atoms:
            J = J + _csr(rows[~mx], cols[~mx] - self.n, vals[~mx], (self.size, self.n_atoms)) @ Ja
        return J

    # ------------------------------------------------------------------------------------------------ evaluation
    def _state(self, x):
        """forward pass at x (cached for the last point): node values, the atoms' scalar derivatives and Jacobian"""
        x = np.asarray(x, dtype=np.float64)
        last = getattr(self._local, 'last', None)
        if last is not None and np.array_equal(x, last[0]):
            return last[1]
        v = np.empty(self.N)
        v[:self.n] = x
        st = {'levels': []}
        Ja = sp.csr_matrix((0, self.n))
        for lv in self.levels:
            u, first, second = lv.arg.evaluate(v, order=2)
            phi, d1, d2 = np.empty(lv.size), np.empty(lv.size), np.empty(lv.size)
            for kind, param, ix in lv.groups:
                phi[ix], d1[ix], d2[ix] = _phi(kind, param, u[ix])
            v[self.n + lv.start:self.n + lv.start + lv.size] = phi
            Jl = _csr(lv.rows[lv.mx], lv.cols[lv.mx], first[lv.mx], (lv.size, self.n))
            Pa = _csr(lv.rows[~lv.mx], lv.cols[~lv.mx] - self.n, first[~lv.mx], (lv.size, lv.start))
            if lv.start:
                Jl = Jl + Pa @ Ja
            Ja = sp.vstack([Ja, sp.diags(d1) @ Jl]).tocsr()
            st['levels'].append((first, second, d1, d2, Pa))
        st['Ja'] = Ja
        val, first, second = self.out.evaluate(v, order=2)
        st.update(val=val, first=first, second=second)
        self._local.last = (x.copy(), st)
        return st

    def values(self, x):
        return self._state(x)['val']

    def jacobian(self, x):
        """(values of the constraint owners' Jacobian on (jac_rows, jac_cols), the objective owner's gradient)"""
        st = self._state(x)
        if not self.levels:
            out = np.bincount(self._jpos, st['first'], minlength=self._jlen)
            return out[:len(self.jac_rows)], out[len(self.jac_rows):]
        if 'jac' not in st:
            J = self._compose(self._out_rows, self._out_cols, st['first'], st['Ja'])
            st['jac'] = _sample(J, self.jac_rows + 1, self.jac_cols), np.asarray(J[0].todense()).ravel()
        return st['jac']

    def hessian(self, x, w):
        """values on (hess_rows, hess_cols) of the Hessian of sum_o w[o] * owner_o"""
        st = self._state(x)
        own, ra, rb = self._out_second
        sw = st['second'] * w[own]
        if not self.levels:
            return np.bincount(self._hpos, sw, minlength=len(self.hess_rows))
        n, N = self.n, self.N
        L = _csr(ra, rb, sw, (N, N))            # second partials, lower triangle
        R = sp.csr_matrix((N, N))               # the atoms' rank-one terms, both triangles
        mx = self._out_cols < n
        omega = _csr(self._out_rows[~mx], self._out_cols[~mx] - n, st['first'][~mx],
                     (self.size, self.n_atoms)).T @ w
        for lv, (first, second, d1, d2, Pa) in zip(reversed(self.levels), reversed(st['levels'])):
            om = omega[lv.start:lv.start + lv.size]
            wl = om * d1
            if lv.start:
                omega[:lv.start] += Pa.T @ wl
            lown, la, lb = lv.second
            L = L + _csr(la, lb, second * wl[lown], (N, N))
            P = _csr(lv.rows, lv.cols, first, (lv.size, N))
            R = R + P.T @ sp.diags(om * d2) @ P
        K = L + sp.tril(L, -1).T + R
        Z = sp.vstack([sp.identity(n, format='csr'), st['Ja']]).tocsr()
        return _sample(Z.T @ K @ Z, self.hess_rows, self.hess_cols)


class _Level:
    """the atoms of one nesting level: their arguments' polynomials, scalar-function groups and partial patterns"""
    def __init__(self, arg, groups, start, n):
        self.arg, self.groups, self.start, self.size = arg, groups, start, arg.size
        self.rows, self.cols = arg.first_pattern()
        self.mx = self.cols < n
        self.second = arg.second_pattern()


# ---------------------------------------------------------------------------------------------------- the NLP view
def _combine(pairs):
    """sum_i c_i * poly_i of exact polynomials (c_i exact), zero terms dropped"""
    out = {}
    for poly, c in pairs:
        for m, v in poly.items():
            out[m] = out.get(m, 0) + v * c
    return {m: v for m, v in out.items() if v != 0}


def _exact(c):
    c = float(c)
    return int(c) if c.is_integer() else Fraction(c)


class SparseNLP:
    """An objective and raw rows P_i(x) (the problem's g_i = P_i - rhs_i <= 0, constants included) on one FormTape.

    Rows whose non-constant parts are positive or negative multiples of one exact direction (rowpool._direction, the
    RowPool's merge rule) are ONE tape row q_r, the non-constant part of the group's first member, and every raw row is
    P_i = rho_i * q_r + c_i; an equality is therefore one ranged row. A raw row without a non-constant part has no tape
    row (nothing can move it)."""

    def __init__(self, objective, rows, atom_info, n):
        reps, lead, group = [], [], {}
        gr, rho, const = [], [], []
        for p in rows:
            nz = sorted((m, c) for m, c in p.items() if m)
            const.append(float(p.get((), 0)))
            if not nz:
                gr.append(-1)
                rho.append(0.0)
                continue
            key = _direction(nz)
            r = group.get(key)
            if r is None:
                r = group[key] = len(reps)
                reps.append(dict(nz))
                lead.append(Fraction(nz[0][1]))
            gr.append(r)
            rho.append(float(Fraction(nz[0][1]) / lead[r]))
        self.row_group, self.row_rho, self.row_const = np.array(gr, dtype=np.int64), np.array(rho), np.array(const)
        self.m = len(reps)
        self.polys = [objective] + reps         # the tape's owners, exact
        self.tape = FormTape(self.polys, atom_info, n)
        self.n = n

    def problem(self, f0, rhs):
        """the IPOPT callbacks for objective - f0 and raw rows P_i - rhs_i"""
        return SparseProblem(self, float(f0), np.asarray(rhs, dtype=np.float64).reshape(-1))


class SparseProblem:
    """IPOPT callbacks of one problem: min f(x) s.t. cl <= q(x) <= cu on a SparseNLP (its rhs and objective shift)."""

    def __init__(self, nlp, f0, rhs):
        self.nlp, self.f0, self.rhs = nlp, f0, rhs
        self.m, self.hessian_available = nlp.m, nlp.tape.hessian_available

    def bounds(self, slack):
        """(cl, cu) of the tape rows for the raw rows g_i <= slack"""
        nlp = self.nlp
        beta = (self.rhs + slack - nlp.row_const) / np.where(nlp.row_rho != 0, nlp.row_rho, 1.0)
        cl, cu = np.full(nlp.m, -np.inf), np.full(nlp.m, np.inf)
        pos, neg = (nlp.row_group >= 0) & (nlp.row_rho > 0), (nlp.row_group >= 0) & (nlp.row_rho < 0)
        np.minimum.at(cu, nlp.row_group[pos], beta[pos])
        np.maximum.at(cl, nlp.row_group[neg], beta[neg])
        return cl, cu

    def raw_rows(self, x):
        """g_i(x) = P_i(x) - rhs_i of every raw row, from the tape (for a check on the tape's own arithmetic)"""
        nlp = self.nlp
        q = np.concatenate([nlp.tape.values(x)[1:], [0.0]])
        return nlp.row_rho * q[nlp.row_group] + nlp.row_const - self.rhs

    def objective(self, x):
        return float(self.nlp.tape.values(x)[0]) - self.f0

    def gradient(self, x):
        return self.nlp.tape.jacobian(x)[1]

    def constraints(self, x):
        return self.nlp.tape.values(x)[1:]

    def jacobianstructure(self):
        return self.nlp.tape.jac_rows, self.nlp.tape.jac_cols

    def jacobian(self, x):
        return self.nlp.tape.jacobian(x)[0]

    def hessianstructure(self):
        return self.nlp.tape.hess_rows, self.nlp.tape.hess_cols

    def hessian(self, x, lam, obj_factor):
        return self.nlp.tape.hessian(x, np.concatenate([[obj_factor], np.asarray(lam, dtype=np.float64)]))


    def unmerged(self):
        """The same tape with the original inequality rows, for an IPOPT recovery solve.

        Merging opposite inequalities changes IPOPT's barrier problem. In particular,
        zero slack turns dependent pairs into more equalities than free variables.
        Keep the sparse arithmetic and Hessian while restoring the original row layout.
        """
        return UnmergedSparseProblem(self)


class UnmergedSparseProblem:
    """Raw inequality callbacks backed by a merged tape (no second structure walk)."""
    def __init__(self, sparse):
        self.sparse, self.nlp = sparse, sparse.nlp
        self.m = len(self.nlp.row_group)
        self.hessian_available = sparse.hessian_available
        jr, jc = sparse.jacobianstructure()
        rows, cols, positions = [], [], []
        order = np.argsort(jr, kind='stable')
        starts = np.searchsorted(jr[order], np.arange(sparse.m + 1))
        for row, group in enumerate(self.nlp.row_group):
            if group < 0:
                continue
            pos = order[starts[group]:starts[group + 1]]
            rows.extend([row] * len(pos))
            cols.extend(jc[pos])
            positions.extend(pos)
        self.rows, self.cols, self.positions = (
            np.asarray(v, dtype=np.int64) for v in (rows, cols, positions))

    def bounds(self, slack):
        return np.full(self.m, -np.inf), np.full(self.m, slack)

    def objective(self, x):
        return self.sparse.objective(x)

    def gradient(self, x):
        return self.sparse.gradient(x)

    def constraints(self, x):
        return self.sparse.raw_rows(x)

    def jacobianstructure(self):
        return self.rows, self.cols

    def jacobian(self, x):
        return self.sparse.jacobian(x)[self.positions] * self.nlp.row_rho[self.rows]

    def hessianstructure(self):
        return self.sparse.hessianstructure()

    def hessian(self, x, lam, obj_factor):
        weights = np.zeros(self.sparse.m)
        keep = self.nlp.row_group >= 0
        np.add.at(weights, self.nlp.row_group[keep],
                  np.asarray(lam)[keep] * self.nlp.row_rho[keep])
        return self.sparse.hessian(x, weights, obj_factor)


class TapeProblem:
    """The exact problem on a tape in the interface IPOPTSolver reads from an NNProblemIPOPT (box, raw rows g <= 0,
    objective), for a process that has the compiled problem but not the model. IPOPTSolver's acceptance then uses the
    tape's raw rows; GEAR's verifier on the original model stays with the caller."""

    def __init__(self, sparse, lb, ub):
        self.sparse, self.lb, self.ub = sparse, np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)
        self.n_vars, self.n_constraints = len(self.lb), len(sparse.nlp.row_group)

    def objective(self, x):
        return self.sparse.objective(x)

    def constraints(self, x):
        return self.sparse.raw_rows(x)


def output_walk(model, x0, lower, input_shape):
    """(output forms, algebra) of `model` from the structure walk at the shape-defining point x0, with lower box bounds
    `lower` deciding the walk's positivity rules. Raises when the walk cannot be built."""
    from auto_LiRPA import BoundedModule
    from nlp_solver.structure.forms import FormAlgebra
    from nlp_solver.structure.walk import build_forms, output_forms
    from nlp_solver.utils import _infer_device_dtype
    # A private CPU copy: BoundedModule moves the module it is given, and the root ALM is
    # evaluating this very model on the GPU in the other thread (alkylation and 73 more runs
    # crashed with "index is on cpu" when the shared model was handed over).
    private = copy.deepcopy(model).cpu()
    dtype = _infer_device_dtype(private, torch.device('cpu'), torch.float64)[1]
    x = torch.as_tensor(x0, dtype=dtype).reshape(1, *input_shape)
    bounded = BoundedModule(private, x, bound_opts={'pow_negative_exponent': True}, device='cpu')
    algebra = FormAlgebra(x.numel(), lower)
    return list(output_forms(build_forms(bounded, x, algebra))), algebra


def _triples(M):
    """(rows, cols, values) of the nonzeros of a 2-D tensor, as numpy"""
    M = M.reshape(-1, M.shape[-1]) if M.ndim else M.reshape(0, 0)
    idx = (M != 0).nonzero()
    vals = M[idx[:, 0], idx[:, 1]] if idx.numel() else M.new_zeros(0)
    return (idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy(), vals.detach().cpu().double().numpy())


_BUILD = threading.Lock()      # the root IPOPT and the root primal lane may ask for the same model at once
_WALKS = weakref.WeakKeyDictionary()     # model -> (forms, algebra, lower bounds, {spec key: SparseNLP}) or None


def sparse_problem(problem):
    """The SparseProblem of an NNProblemIPOPT, or None: when the problem has several OR clauses, when the walk gives no
    exact form for an output a row uses, when the tape does not reproduce the model's arithmetic, or when the box reaches
    below the box the walk's forms were built on. The walk and the compiled NLP (per spec matrix) are kept per source
    model, outside it (the model is deep-copied elsewhere): the in-BaB IPOPT builds a new problem per call."""
    if isinstance(problem, TapeProblem):
        return problem.sparse
    if problem.batch_size != 1:
        return None
    with _BUILD:
        return _sparse_problem(problem)


def _sparse_problem(problem):
    model = problem._source_model
    walk = _WALKS.get(model, False)
    if walk is False:
        try:
            forms, algebra = output_walk(model, np.clip(0.5 * (problem.lb + problem.ub), problem.lb, problem.ub),
                                         problem.lb, problem.input_shape)
            walk = (forms, algebra, problem.lb.copy(), {})
        except Exception:  # noqa: BLE001 - an op the walk cannot follow at all
            walk = None
        _WALKS[model] = walk
    if walk is None:
        return None
    forms, algebra, lower, compiled = walk
    if np.any(problem.lb < lower):
        return None
    parts = [_triples(problem.C_obj[0])]
    if problem.n_output_constraints:
        parts.append(_triples(problem.C_constr[0]))
    if problem.n_input_constraints:
        parts.append(_triples(problem.input_constr_A[0]))
    key = hashlib.sha1(b''.join(a.tobytes() for t in parts for a in t)
                       + repr([len(t[0]) for t in parts] + [problem.n_output_constraints,
                                                            problem.n_input_constraints]).encode()).hexdigest()
    rhs = [problem.rhs_constr.reshape(-1)] if problem.n_output_constraints else []
    if problem.n_input_constraints:
        rhs.append(problem.input_constr_b.reshape(-1))
    rhs = torch.cat(rhs).detach().cpu().double().numpy() if rhs else np.zeros(0)
    if key not in compiled:
        t0 = time.time()
        nlp = _compile(problem, parts, forms, algebra)
        same = nlp is not None and _reproduces(problem, nlp.problem(float(problem.rhs_obj.reshape(-1)[0]), rhs))
        compiled[key] = nlp if same else None
        if nlp is None:
            print('[SparseDerivatives] no exact form for every row: torch callbacks')
        else:
            print(f"[SparseDerivatives] rows={len(nlp.row_group)} merged={nlp.m} atoms={nlp.tape.n_atoms} "
                  f"nnzJ={len(nlp.tape.jac_rows)} nnzH={len(nlp.tape.hess_rows)} "
                  f"hessian={'exact' if nlp.tape.hessian_available else 'quasi-Newton'} "
                  f"reproduces={same} build={time.time() - t0:.2f}s")
    nlp = compiled[key]
    return None if nlp is None else nlp.problem(float(problem.rhs_obj.reshape(-1)[0]), rhs)


def _reproduces(problem, sparse, points=3):
    """Whether the tape reproduces the model's own arithmetic: objective and every row agree with the torch model to
    sqrt(eps) relative at the box centre and two seeded box points (where both are finite). The forms are exact, but an
    expanded product of large factors can cancel in floating point where the model's factored graph does not (hs087's
    regularised step functions differ by 5e-4); IPOPT on such a tape optimises a different function."""
    lo = np.where(np.isfinite(problem.lb), problem.lb, np.minimum(problem.ub, 0.0) - 1.0)
    hi = np.where(np.isfinite(problem.ub), problem.ub, np.maximum(lo, 0.0) + 1.0)
    rng = np.random.default_rng(0)
    eta = np.sqrt(np.finfo(np.float64).eps)
    for k in range(points):
        x = 0.5 * (lo + hi) if k == 0 else lo + (hi - lo) * rng.random(len(lo))
        with torch.no_grad():
            ref = np.concatenate([[problem.objective(x)], problem.constraints(x)])
        own = np.concatenate([[sparse.objective(x)], sparse.raw_rows(x)])
        both = np.isfinite(ref) & np.isfinite(own)
        if np.any(np.abs(own[both] - ref[both]) > eta * (1.0 + np.abs(ref[both]))):
            return False
    return True


def _compile(problem, parts, forms, algebra):
    """the SparseNLP of the problem's objective and rows, or None when one of them has no exact form"""
    def rows_of(triple, count, poly_of):
        r, c, v = triple
        pairs = [[] for _ in range(count)]
        for i, j, a in zip(r.tolist(), c.tolist(), v.tolist()):
            pairs[i].append((j, _exact(a)))
        out = []
        for pr in pairs:
            polys = [(poly_of(j), a) for j, a in pr]
            if any(p is None for p, _ in polys):
                return None
            out.append(_combine(polys))
        return out
    out_poly = lambda k: forms[k].poly
    obj = rows_of(parts[0], 1, out_poly)
    rows, k = [], 1
    if problem.n_output_constraints:
        rows += rows_of(parts[k], problem.n_output_constraints, out_poly) or [None]
        k += 1
    if problem.n_input_constraints:
        rows += rows_of(parts[k], problem.n_input_constraints, lambda j: {((j, 1),): 1}) or [None]
    if obj is None or any(r is None for r in rows):
        return None
    try:
        return SparseNLP(obj[0], rows, algebra.atom_info, problem.n_vars)
    except (ValueError, OverflowError):     # an atom kind without a scalar function here; a float overflow
        return None
