"""Structural forward sparsity and CPR column colouring for CompiledGraph's probe.

Rows are tensor elements, columns are the canonical inputs / face elements. All
intermediate rows participate in colouring: cancellation at a later node must
not hide a conflict (or the fold's nonzero rounding allowance). No numerical AD
values are used to infer zeros. The sparse pass and incidence colouring avoid
both an element-by-column dense tensor and the column conflict graph's cliques.
"""
import copy
import math

import numpy as np
import scipy.sparse as sp
import torch

from .deadline import check_deadline


class ProbePattern:
    def __init__(self, cg, names, shapes, ncol):
        self.net, self.ncol = cg.net, ncol
        self.patterns, self.shapes, self.routes = {}, {}, {}
        self.seeds = {self.net.input_name[0]: (0, (1, *cg.in_shape))}
        off = cg.n
        for k in names:
            self.seeds[k] = off, (1, *shapes[k])
            off += math.prod(shapes[k])
        # Operands of a face are evaluated even though its output is a fresh seed.
        from .operators import BoundMul
        saved = {nd.name: dict(vars(nd)) for nd in self.net.nodes()}
        try:
            for k in names:
                for nd in self.net[k].inputs[:2 if isinstance(self.net[k], BoundMul) else 1]:
                    self._eval(nd)
            self._eval(self.net[self.net.output_name[0]])
        finally:
            for nd in self.net.nodes():
                state = saved[nd.name]
                vars(nd).clear()
                vars(nd).update(state)
        self.colors = self._color()
        self.ncolors = int(self.colors.max()) + 1

    def _empty(self, shape):
        return sp.csr_matrix((math.prod(shape), self.ncol), dtype=bool)

    def _value(self, nd):
        v = self.net.get_forward_value(nd)
        return v.detach().cpu() if torch.is_tensor(v) else v

    def _broadcast(self, nd, shape):
        M = self._eval(nd)
        if not M.nnz:
            return self._empty(shape)
        ids = np.arange(M.shape[0]).reshape(self.shapes[nd.name])
        return M[np.broadcast_to(ids, shape).reshape(-1)]

    def _eval(self, nd):
        if nd.name in self.patterns:
            return self.patterns[nd.name]
        check_deadline()
        node = copy.copy(nd)           # forward may set axes, input_shape, indices, etc.
        cls = type(nd).__name__
        if nd.name in self.seeds:
            off, shape = self.seeds[nd.name]
            E = math.prod(shape)
            M = sp.csr_matrix((np.ones(E, dtype=bool), (np.arange(E), off + np.arange(E))),
                              shape=(E, self.ncol))
        elif not nd.perturbed:
            v = self._value(nd)
            shape = tuple(v.shape) if torch.is_tensor(v) else ()
            # Constants have no column support. Do not allocate a CSR row for
            # every entry of a dense weight matrix (that would be quadratic).
            M = self._empty((1,))
        else:
            args = [self._value(i) if not i.perturbed else None for i in nd.inputs]
            for i in nd.inputs:
                self._eval(i)
            shape = (1, *nd.output_shape[1:])
            M = self._affine(node, cls, args, shape)
        M = M.astype(bool).tocsr()
        M.eliminate_zeros()
        M.sort_indices()
        self.patterns[nd.name], self.shapes[nd.name] = M, shape
        return M

    def _affine(self, node, cls, args, shape):
        inputs = node.inputs
        def route(p, rows, cols, weights=None, mode='mul', alpha=1):
            self.routes.setdefault(node.name, []).append((p, np.asarray(rows).reshape(-1),
                np.asarray(cols).reshape(-1), weights, mode, alpha))
        def broadcast(p):
            ids = np.arange(math.prod(self.shapes[inputs[p].name])).reshape(self.shapes[inputs[p].name])
            return np.broadcast_to(ids, shape).reshape(-1)
        if cls in ('BoundNeg', 'BoundCast'):
            route(0, np.arange(math.prod(shape)), broadcast(0))
            return self._broadcast(inputs[0], shape)
        if cls in ('BoundGather', 'BoundSlice', 'BoundConcat', 'BoundReshape', 'BoundUnsqueeze',
                   'BoundSqueeze', 'BoundTranspose', 'BoundFlatten', 'BoundExpand'):
            # Run the actual index operator on element ids, including constant
            # concat operands (whose pattern rows are empty).
            ids, matrices, off = [], [], 0
            for p, nd in enumerate(inputs):
                if p == 0 or cls == 'BoundConcat':
                    M = self._broadcast(nd, self.shapes[nd.name])
                    ids.append(torch.arange(off, off + M.shape[0]).reshape(self.shapes[nd.name]))
                    matrices.append(M)
                    off += M.shape[0]
                else:
                    ids.append(args[p])
            idx = node.forward(*ids).reshape(-1).numpy()
            start = 0
            for p, M in enumerate(matrices):
                take = (idx >= start) & (idx < start + M.shape[0])
                route(p, np.flatnonzero(take), idx[take] - start)
                start += M.shape[0]
            return sp.vstack(matrices, format='csr')[idx]
        if cls in ('BoundAdd', 'BoundSub', 'BoundMul', 'BoundDiv'):
            M = self._empty(shape)
            for p, nd in enumerate(inputs):
                if args[p] is not None:
                    continue
                part = self._broadcast(nd, shape)
                weights = None
                if cls in ('BoundMul', 'BoundDiv'):
                    weights = np.broadcast_to(args[1 - p].numpy(), shape).reshape(-1)
                route(p, np.arange(math.prod(shape)), broadcast(p), weights,
                      'div' if cls == 'BoundDiv' else 'mul')
                if cls == 'BoundMul':
                    mask = np.broadcast_to((args[1 - p] != 0).numpy(), shape).reshape(-1)
                    part = part.multiply(mask[:, None])
                M = M.maximum(part)
            return M
        if cls in ('BoundMatMul', 'BoundLinear'):
            # The local Jacobian is sparse too. Compose it with the incoming
            # patterns using boolean products, so exact cancellation never
            # deletes a dependency.
            a, b = inputs[:2]
            ash, bsh = self.shapes[a.name], self.shapes[b.name]
            av, bv = args[:2]
            # Element ids are needed only for the perturbed operand. A dense
            # constant's shape is enough; its values supply the sparse mask.
            ids = lambda sh, v: (torch.arange(math.prod(sh)).reshape(sh) if v is None else
                                 torch.zeros((), dtype=torch.long).expand(sh))
            ai, bi = ids(ash, av), ids(bsh, bv)
            if cls == 'BoundLinear':
                if node.transA:
                    ai = ai.transpose(-2, -1)
                    if av is not None:
                        av = av.transpose(-2, -1)
                if node.transB:
                    bi = bi.transpose(-2, -1)
                    if bv is not None:
                        bv = bv.transpose(-2, -1)
            # Promote vector operands following torch.matmul's convention.
            avec, bvec = ai.ndim == 1, bi.ndim == 1
            if avec:
                ai = ai.unsqueeze(0)
                if av is not None:
                    av = av.unsqueeze(0)
            if bvec:
                bi = bi.unsqueeze(-1)
                if bv is not None:
                    bv = bv.unsqueeze(-1)
            batch = torch.broadcast_shapes(ai.shape[:-2], bi.shape[:-2])
            m, k, n = ai.shape[-2], ai.shape[-1], bi.shape[-1]
            ai = ai.expand(*batch, m, k).reshape(-1, m, k).numpy()
            bi = bi.expand(*batch, k, n).reshape(-1, k, n).numpy()
            out = np.arange(math.prod(shape)).reshape(-1, m, n)
            M = self._empty(shape)
            alpha = node.alpha_linear if cls == 'BoundLinear' else 1
            if alpha != 0:
                if bv is not None:
                    q, j, t = (bv != 0).expand(*batch, k, n).reshape(-1, k, n).nonzero(as_tuple=True)
                    q, j, t = q.numpy(), j.numpy(), t.numpy()
                    r = out[q, :, t].reshape(-1)
                    c = ai[q, :, j].reshape(-1)
                    w = bv.expand(*batch, k, n).reshape(-1, k, n).numpy()[q, j, t]
                    route(0, r, c, np.repeat(w, m), alpha=alpha)
                    J = sp.csr_matrix((np.ones(len(r), dtype=bool), (r, c)),
                                      shape=(math.prod(shape), self.patterns[a.name].shape[0]))
                    M = M.maximum(J @ self.patterns[a.name])
                if av is not None:
                    q, i, j = (av != 0).expand(*batch, m, k).reshape(-1, m, k).nonzero(as_tuple=True)
                    q, i, j = q.numpy(), i.numpy(), j.numpy()
                    r = out[q, i, :].reshape(-1)
                    c = bi[q, j, :].reshape(-1)
                    w = av.expand(*batch, m, k).reshape(-1, m, k).numpy()[q, i, j]
                    route(1, r, c, np.repeat(w * alpha, n))
                    J = sp.csr_matrix((np.ones(len(r), dtype=bool), (r, c)),
                                      shape=(math.prod(shape), self.patterns[b.name].shape[0]))
                    M = M.maximum(J @ self.patterns[b.name])
            if len(inputs) > 2 and node.beta_linear != 0:
                route(2, np.arange(math.prod(shape)), broadcast(2),
                      np.full(math.prod(shape), node.beta_linear))
                M = M.maximum(self._broadcast(inputs[2], shape))
            return M
        if cls == 'BoundReduceSum':
            ish = self.shapes[inputs[0].name]
            node._parse_input_and_axis(torch.empty(ish), *args[1:])
            axes = tuple(range(len(ish))) if node.axis is None else tuple(node.axis)
            expanded = list(shape)
            if not node.keepdim:
                for a in sorted(axes):
                    expanded.insert(a, 1)
            rows = np.broadcast_to(np.arange(math.prod(shape)).reshape(expanded), ish).reshape(-1)
            route(0, rows, np.arange(len(rows)))
            return self._scatter(self.patterns[inputs[0].name], rows, shape)
        if cls == 'BoundATenIndex_add':
            route(0, np.arange(math.prod(shape)), broadcast(0))
            M = self._broadcast(inputs[0], shape)
            if float(args[4]) != 0:
                ids = torch.arange(math.prod(shape)).reshape(shape)
                rows = ids.index_select(int(args[1]), args[2]).reshape(-1).numpy()
                source = self._broadcast(inputs[3], tuple(ids.index_select(int(args[1]), args[2]).shape))
                ssh = tuple(ids.index_select(int(args[1]), args[2]).shape)
                src = np.arange(math.prod(self.shapes[inputs[3].name])).reshape(self.shapes[inputs[3].name])
                route(3, rows, np.broadcast_to(src, ssh).reshape(-1), np.full(len(rows), float(args[4])))
                M = M.maximum(self._scatter(source, rows, shape))
            return M
        raise NotImplementedError(f'compiled_crown: no probe pattern for {cls} ({node.name})')

    def _scatter(self, M, rows, shape):
        r, c = M.nonzero()
        return sp.csr_matrix((np.ones(len(r), dtype=bool), (rows[r], c)),
                             shape=(math.prod(shape), self.ncol))

    def form(self, name, shape):
        nd = self.net[name]
        M = self._broadcast(nd, (1, *shape))
        return M.nonzero()

    def _color(self):
        M = sp.vstack([M for M in self.patterns.values() if M.nnz], format='csr')
        columns = M.tocsc()
        # An upper bound on degree supplies a deterministic largest-first order,
        # without constructing a potentially quadratic column conflict graph.
        degree = np.asarray(M.T @ np.diff(M.indptr).astype(np.int64)).reshape(-1)
        order = np.argsort(-degree, kind='stable')
        occupied = [0] * M.shape[0]     # bit set of colours already used in each row
        colors = np.empty(self.ncol, dtype=np.int64)
        for j in order:
            rows = columns.indices[columns.indptr[j]:columns.indptr[j + 1]]
            used = 0
            for r in rows:
                used |= occupied[r]
            bit = (used + 1) & ~used   # first unused nonnegative colour
            colors[j] = bit.bit_length() - 1
            for r in rows:
                occupied[r] |= bit
        return colors

    def sparse_recovery(self, dt, dev):
        """Routes for a sparse fold with single-term reductions, or None.

        Add/Sub are evaluated as the same two operands. For sums, matrix
        products and index_add, require at most one contributing term per
        (element, column), including structural cancellations upstream. Then
        deleting zero terms cannot change a reduction's value or rounding.
        Other graphs retain the colour probe, with no arithmetic substitution.
        """
        routes = {}
        for name, parts in self.routes.items():
            out = self.patterns[name]
            rr, cc = out.nonzero()
            keys = rr.astype(np.int64) * self.ncol + cc
            entries, allkeys = [], []
            for p, rows, cols, weights, mode, alpha in parts:
                src = self.patterns[self.net[name].inputs[p].name]
                if not src.nnz:
                    continue
                count = np.diff(src.indptr)[cols]
                t = np.repeat(np.arange(len(rows)), count)
                if not len(t):
                    continue
                j = np.repeat(src.indptr[cols], count) + np.arange(len(t)) - np.repeat(np.cumsum(count) - count, count)
                if weights is not None and mode != 'div':
                    take = weights[t] != 0
                    t, j = t[take], j[take]
                k = rows[t].astype(np.int64) * self.ncol + src.indices[j]
                if len(np.unique(k)) != len(k):
                    return None
                allkeys.append(k)
                integer = lambda a: torch.as_tensor(a, dtype=torch.long, device=dev)
                w = None if weights is None else torch.as_tensor(weights[t].copy(), dtype=dt, device=dev)
                entries.append((p, integer(j), integer(np.searchsorted(keys, k)), w, mode, alpha))
            if type(self.net[name]).__name__ not in ('BoundAdd', 'BoundSub') and allkeys:
                k = np.concatenate(allkeys)
                if len(np.unique(k)) != len(k):
                    return None
            routes[name] = entries
        return routes

    def recovered_form(self, recovered, name, shape, error=False):
        v = recovered[name][int(error)]
        r, c = self.form(name, shape)
        M = self.patterns[name]
        # Broadcasting can repeat a source element's entire row.
        src = np.broadcast_to(np.arange(M.shape[0]).reshape(self.shapes[name]), (1, *shape)).reshape(-1)
        mr, mc = M.nonzero()
        pos = np.searchsorted(mr.astype(np.int64) * self.ncol + mc, src[r].astype(np.int64) * self.ncol + c)
        r, c, pos = (torch.as_tensor(a, dtype=torch.long, device=v.device) for a in (r, c, pos))
        v = v[pos]
        take = v != 0
        return r[take], c[take], v[take]

    def recover(self, routes, vals, dt, dev):
        """Sparse tangents and exactly the original _fold_error tangent formula.

        Constants and their enclosures still come from the ordinary probe, at
        its original first-chunk batch size. Only single-term reductions use
        these routes; all products, divisions, binary additions and _upper
        calls keep their original order and dtype/device.
        """
        from .compiled_crown import _fold_rule, _gamma, _upper
        result = {}
        def evaluate(name):
            if name in result:
                return result[name]
            check_deadline()
            M = self.patterns[name]
            if name in self.seeds:
                v = torch.ones(M.nnz, dtype=dt, device=dev)
                result[name] = v, torch.zeros_like(v)
                return result[name]
            nd = self.net[name]
            if not nd.perturbed:
                z = torch.zeros(0, dtype=dt, device=dev)
                result[name] = z, z
                return result[name]
            t, c, consts = vals[name]
            raw = [vals[i.name][0] if v is None else v for i, v in zip(nd.inputs, consts)]
            _, k, _ = _fold_rule(copy.copy(nd), raw)
            g = _gamma(k, dt)
            incoming = [evaluate(i.name) for i in nd.inputs]
            cls = type(nd).__name__
            def run(channel):
                pieces = {}
                for p, src, dest, w, mode, alpha in routes.get(name, []):
                    v, e = incoming[p]
                    z = v if channel == 'value' else e + g * v.abs()
                    if channel == 'support':
                        z = (z != 0).to(dt)
                    z = z[src]
                    if alpha != 1:
                        z = z * alpha
                    if w is not None:
                        w = w if channel == 'value' else (w != 0).to(dt) if channel == 'support' else w.abs()
                        z = z / w if mode == 'div' else z * w
                    part = pieces.setdefault(p, torch.zeros(M.nnz, dtype=dt, device=dev))
                    part[dest] = z
                zero = torch.zeros(M.nnz, dtype=dt, device=dev)
                if cls in ('BoundAdd', 'BoundSub'):
                    a, b = pieces.get(0, zero), pieces.get(1, zero)
                    return a - b if cls == 'BoundSub' and channel == 'value' else a + b
                if cls == 'BoundNeg' and channel == 'value':
                    return -pieces.get(0, zero)
                # Different concat operands occupy disjoint entries. All
                # other accepted reductions have exactly one term per entry.
                for part in pieces.values():
                    zero = zero + part
                return zero
            v, magnitude, support = run('value'), run('error'), run('support')
            e = torch.where(support > 0, _upper(magnitude, k + 3, 4 * (k + 2)), 0.0)
            result[name] = v, e
            return result[name]
        for name in vals:
            evaluate(name)
        evaluate = None  # Break the recursive closure; do not retain dense probe values.
        return result

    def singleton_constants(self, vals, errors):
        """Whether a one-lane constant pass has the original batch's arithmetic.

        GEMV/GEMM and sum kernels may choose different reduction trees with a
        different batch size. Use one lane only if each of their constant and
        constant-error reductions has at most one nonzero term. The check is
        conservative (it includes |c| even when gamma is zero).
        """
        for name, parts in self.routes.items():
            cls = type(self.net[name]).__name__
            if cls not in ('BoundMatMul', 'BoundLinear', 'BoundReduceSum', 'BoundATenIndex_add'):
                continue
            counts = np.zeros(math.prod(self.shapes[name]), dtype=np.int64)
            for p, rows, cols, weights, mode, alpha in parts:
                if cls == 'BoundLinear' and (p == 2 or self.net[name].transA):
                    if self.net[name].transA:
                        return False
                    continue
                nd = self.net[name].inputs[p]
                if nd.name in vals:
                    c = vals[nd.name][1][0].reshape(-1).detach().cpu().numpy()
                    ec = errors[nd.name][1].reshape(-1).detach().cpu().numpy()
                    active = (c[cols] != 0) | (ec[cols] != 0)
                else:
                    c = self._value(nd).reshape(-1).numpy()
                    active = c[cols] != 0
                if not np.all(np.isfinite(c)):
                    return False
                if weights is not None:
                    if not np.all(np.isfinite(weights)):
                        return False
                    active &= weights != 0
                if alpha == 0:
                    continue
                counts += np.bincount(rows[active], minlength=len(counts))
            if np.any(counts > 1):
                return False
        return True
