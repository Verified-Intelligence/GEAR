"""Exact sparse recipes for M3 (input_split/m3_reduce.py), compiled once on the CPU from the exact forms of the root
structure pass; no model-name heuristics.

Accepted rows: affine rows; one product plus affine terms; and the signed-square identity k - sum (x_i - x_j)^2 with
disjoint endpoints. Coefficients must be exactly representable FP64 rationals. Unsupported rows are omitted (a valid
relaxation), never approximated, and counted in a census. theory/kall_m3/kall_m3.md.
"""
from collections import Counter
from fractions import Fraction
import math

import torch


def _parts(form, n):
    """(constant, linear {j: c}, quadratic {(i, j): c}) of an exact polynomial of degree <= 2 in the inputs, else None"""
    if form.poly is None or form.inexact:
        return None
    k, g, q = 0, {}, {}
    for mon, c in form.poly.items():
        try:
            if not math.isfinite(float(c)) or Fraction(float(c)) != c:
                return None
        except OverflowError:
            return None
        if any(j < 0 or j >= n or not isinstance(e, int) or e <= 0 for j, e in mon):
            return None
        degree = sum(e for _, e in mon)
        if degree == 0:
            k = c
        elif degree == 1:
            g[mon[0][0]] = c
        elif degree == 2:
            q[tuple(j for j, e in mon for _ in range(e))] = c
        else:
            return None
    return k, g, q


def _differences(q):
    """the pairs (i, j) when the whole quadratic dictionary equals -sum (x_i - x_j)^2 over disjoint pairs, else None"""
    edges, used = [], set()
    for (i, j), c in q.items():
        if i == j:
            continue
        if c != 2 or i in used or j in used or q.get((i, i)) != -1 or q.get((j, j)) != -1:
            return None
        edges.append((i, j))
        used.update((i, j))
    return edges if edges and len(q) == 3 * len(edges) else None


def _sum_forest(groups, leaves):
    """A balanced pairwise sum forest over the leaf slots of each group: levels of (a, b, dst) slot triples and the
    root slot of every group; O(nnz) metadata, built only for unfinished branches."""
    roots = [g[0] for g in groups]
    active = [(r, g) for r, g in enumerate(groups) if len(g) > 1]
    levels, nodes = [], leaves
    while active:
        a, b, dest, following = [], [], [], []
        for r, group in active:
            ng = []
            for t in range(0, len(group) - 1, 2):
                a.append(group[t])
                b.append(group[t + 1])
                dest.append(nodes)
                ng.append(nodes)
                nodes += 1
            if len(group) % 2:
                ng.append(group[-1])
            if len(ng) == 1:
                roots[r] = ng[0]
            else:
                following.append((r, ng))
        levels.append((a, b, dest))
        active = following
    return levels, roots, nodes


class M3Pattern:
    """The device pattern: affine/product leaves (col, coef, owner row, product factors pi/pj), the row constants k
    and their spec index, base-row flags, square factors (fi, fj, owning row fo) with accepted order bounds
    [sl, su] on x_fi - x_fj, and the two sum forests. `complete`: an affine objective and every row supported, the
    precondition of the cheap covering continuation."""

    def __init__(self, forms, n, base, device):
        self.n, self.device = n, torch.device(device)
        self.census = Counter()
        objective = _parts(forms[0], n) if forms else None
        self.affine_objective = objective is not None and not objective[2]
        if not self.affine_objective:
            self.census['non-affine objective'] += 1
            objective = (0, {}, {})
        rows, squares = [(0, *objective)], []
        for s, form in enumerate(forms[1:], 1):
            parts = _parts(form, n)
            if parts is None:
                self.census['inexact/unsupported'] += 1
                continue
            k, g, q = parts
            if not q or len(q) == 1:
                rows.append((s, k, g, q))
                self.census['product' if q else 'affine'] += 1
            else:
                edges = _differences(q) if not g else None
                if edges is None:
                    self.census['unknown quadratic identity'] += 1
                else:
                    squares.append((s, k, edges))
                    self.census['signed-square'] += 1
        # base rows A x + b <= 0 already carry their accepted tolerance
        signs = {}
        if base is not None:
            A, b = base
            A, b = A.detach().cpu().double(), b.detach().cpu().double()
            A = A.reshape(A.shape[0] if A.ndim >= 3 else 1, -1, n)
            nr = A.shape[1]
            b = b.reshape(-1, nr) if nr else b.reshape(1, 0)
            if not (torch.equal(A, A[:1].expand_as(A)) and torch.equal(b, b[:1].expand_as(b))):
                # a shared recipe may not intersect distinct OR-domain row sets; omitting them is a valid relaxation
                self.census['per-domain base'] += 1
                A, b = A.new_empty((0, n)), b.new_empty(0)
            else:
                A, b = A[0], b[0]
            for a, bias in zip(A.reshape(-1, n), b.reshape(-1)):
                g = {j: float(a[j]) for j in a.nonzero().flatten().tolist()}
                k = float(bias)
                if not math.isfinite(k) or not all(math.isfinite(v) for v in g.values()):
                    self.census['nonfinite base'] += 1
                    continue
                rows.append((-1, k, g, {}))
                if len(g) == 2 and sorted(g.values()) == [-1., 1.]:
                    i = next(j for j in g if g[j] == 1)
                    j = next(j for j in g if g[j] == -1)
                    signs[i, j] = min(signs.get((i, j), math.inf), -k)
        col, coef, owner, prod_i, prod_j, isprod, linear = [], [], [], [], [], [], []
        groups, product_slots, inverse_slots = [], [], []
        for r, (_, _, g, q) in enumerate(rows):
            group = []
            for j, c in g.items():
                group.append(len(col))
                col.append(j); coef.append(float(c)); owner.append(r)
                prod_i.append(0); prod_j.append(0); isprod.append(False); linear.append(True)
            if q:
                (i, j), c = next(iter(q.items()))
                slot = len(col)
                group.append(slot)
                product_slots.append(slot)
                if i != j and i not in g and j not in g:
                    inverse_slots.append(slot)
                col.append(0); coef.append(float(c)); owner.append(r)
                prod_i.append(i); prod_j.append(j); isprod.append(True); linear.append(False)
            if not group:
                group.append(len(col))
                col.append(0); coef.append(0.); owner.append(r)
                prod_i.append(0); prod_j.append(0); isprod.append(False); linear.append(False)
            groups.append(group)
        self.E, self.R = len(col), len(rows)
        levels, roots, self.nodes = _sum_forest(groups, self.E)
        self.F = len(squares)
        fi, fj, fo, sl, su, square_groups = [], [], [], [], [], []
        for r, (_, _, edges) in enumerate(squares):
            ids = []
            for i, j in edges:
                ids.append(len(fi))
                fi.append(i); fj.append(j); fo.append(r)
                sl.append(-signs.get((j, i), math.inf)); su.append(signs.get((i, j), math.inf))
            square_groups.append(ids)
        square_levels, froots, self.sqnodes = _sum_forest(square_groups, len(fi))
        self.H = len(fi)
        self.eligible = self.affine_objective or self.H > 0 or len(rows) > 1
        data = dict(col=col, coef=coef, owner=owner, pi=prod_i, pj=prod_j, product=isprod, linear=linear,
                    products=product_slots, inverse=inverse_slots,
                    objective=[t for t, r in enumerate(owner) if r == 0], roots=roots,
                    spec=[max(0, r[0]) for r in rows], base=[r[0] < 0 for r in rows], k=[float(r[1]) for r in rows],
                    fi=fi, fj=fj, fo=fo, sl=sl, su=su, fs=[r[0] for r in squares], fk=[float(r[1]) for r in squares],
                    froots=froots)
        floats, bools = {'coef', 'k', 'sl', 'su', 'fk'}, {'base', 'product', 'linear'}
        for name, values in data.items():
            dtype = torch.float64 if name in floats else torch.bool if name in bools else torch.long
            setattr(self, name, torch.tensor(values, dtype=dtype, device=self.device))
        self.levels = [tuple(torch.tensor(v, dtype=torch.long, device=self.device) for v in level) for level in levels]
        self.square_levels = [tuple(torch.tensor(v, dtype=torch.long, device=self.device) for v in level)
                              for level in square_levels]
        self.complete = self.affine_objective and sum(self.census[k] for k in (
            'inexact/unsupported', 'unknown quadratic identity', 'nonfinite base', 'per-domain base')) == 0
