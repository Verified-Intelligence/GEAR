"""Branching space of a polynomial problem (arch2 spec section 3): which coordinates make every nonconvex term exact.

From the exact polynomials of the objective and the rows:
  * the product graph: an edge between two variables of one monomial;
  * `fixed`: variables that occur nonlinearly by themselves (a square or higher power, a fractional or negative
    exponent, an atom's argument). They belong to every exact set and are never free in a block LP;
  * a colouring of the product graph on the other variables into independent sets (2-colouring per bipartite
    component, DSATUR on a component with an odd cycle). With every colour but one held fixed, every row and the
    objective are affine in the free colour and the product-free variables: the blocks of the LP alternation;
  * the exact set S: `fixed` plus a vertex cover of the remaining edges (Koenig's minimum cover when the graph is
    bipartite; otherwise the complement of a colour class or the ends of a maximal matching), the candidate with the
    smallest basis;
  * the branching basis B within S and the affine map T, c: on the linear equalities A_eq x = b_eq every coordinate
    of S \\ B is determined by x_B, x_{S \\ B} = c + T x_B, and |B| = rank([A_eq; E_S]) - rank(A_eq).

Nothing here decides soundness: S, B and T only steer search (spec 3.4). Ranks are numerical (pivoted QR on
row-scaled data); the dense linear algebra is skipped above a structural size, and then B = S.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_bipartite_matching

MAX_DENSE = 25_000_000      # entries of A_eq (rows x n) the dense rank computation is admitted for


@dataclass
class BranchingSpace:
    n: int
    edges: np.ndarray       # (E, 2) product pairs between non-fixed variables, i < j
    fixed: np.ndarray       # (n,) bool
    colour: np.ndarray      # (n,) colour class of a non-fixed product variable, -1 otherwise
    bipartite: bool
    cover: np.ndarray       # the exact set S (sorted indices)
    basis: np.ndarray       # B within S
    dependent: np.ndarray   # S \ B
    T: np.ndarray           # (|S \ B|, |B|)
    c: np.ndarray           # (|S \ B|,)

    @property
    def colours(self):
        return int(self.colour.max()) + 1 if self.colour.size and self.colour.max() >= 0 else 0

    def block(self, k):
        """the variables free in block k: colour k and the product-free variables, never a fixed one"""
        return ((self.colour == k) | (self.colour < 0)) & ~self.fixed


def product_structure(polys, n, atom_support=None):
    """(edges between non-fixed variables, fixed mask) of exact polynomials over inputs (< n) and atoms (>= n);
    atom_support: atom id -> its input support (all of it becomes fixed)."""
    fixed = np.zeros(n, bool)
    pairs = set()
    for poly in polys:
        for mono in poly:
            xs = [v for v, _ in mono if v < n]
            for v, e in mono:
                if v >= n:
                    fixed[list(atom_support[v])] = True
                elif e != 1:
                    fixed[v] = True
            for a in range(len(xs)):
                for b in range(a + 1, len(xs)):
                    pairs.add((min(xs[a], xs[b]), max(xs[a], xs[b])))
    edges = np.array(sorted((i, j) for i, j in pairs if not (fixed[i] or fixed[j])), dtype=np.int64).reshape(-1, 2)
    return edges, fixed


def colouring(edges, n):
    """(colour per variable, -1 for none; every component bipartite): BFS 2-colouring per component, DSATUR on a
    component with an odd cycle"""
    adj = [[] for _ in range(n)]
    for i, j in edges.tolist():
        adj[i].append(j)
        adj[j].append(i)
    col = np.full(n, -1, dtype=np.int64)
    bipartite = True
    for s in range(n):
        if col[s] >= 0 or not adj[s]:
            continue
        comp, queue, ok = [s], deque([s]), True
        col[s] = 0
        while queue:
            v = queue.popleft()
            for w in adj[v]:
                if col[w] < 0:
                    col[w] = 1 - col[v]
                    comp.append(w)
                    queue.append(w)
                elif col[w] == col[v]:
                    ok = False
        if ok:
            continue
        bipartite = False
        col[comp] = -1
        sat = {v: set() for v in comp}
        left = set(comp)
        while left:
            v = max(left, key=lambda z: (len(sat[z]), len(adj[z]), -z))
            k = 0
            while k in sat[v]:
                k += 1
            col[v] = k
            left.discard(v)
            for w in adj[v]:
                if w in left:
                    sat[w].add(k)
    return col, bipartite


def koenig_cover(edges, col):
    """minimum vertex cover of a bipartite graph (colour 0 = left) from a maximum matching (Koenig's theorem)"""
    prod = np.unique(edges)
    left, right = prod[col[prod] == 0], prod[col[prod] == 1]
    li = {v: i for i, v in enumerate(left.tolist())}
    ri = {v: i for i, v in enumerate(right.tolist())}
    r, c = [], []
    for a, b in edges.tolist():
        if col[a] == 1:
            a, b = b, a
        r.append(li[a])
        c.append(ri[b])
    G = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(left), len(right)))
    match = maximum_bipartite_matching(G, perm_type='column')
    rmatch = np.full(len(right), -1)
    rmatch[match[match >= 0]] = np.flatnonzero(match >= 0)
    # Z: reachable from unmatched left vertices by alternating paths; cover = (L \ Z) + (R n Z)
    zl, zr = set(np.flatnonzero(match < 0).tolist()), set()
    stack = list(zl)
    while stack:
        i = stack.pop()
        for j in G.indices[G.indptr[i]:G.indptr[i + 1]].tolist():
            if j not in zr and match[i] != j:
                zr.add(j)
                k = int(rmatch[j])
                if k >= 0 and k not in zl:
                    zl.add(k)
                    stack.append(k)
    return np.array(sorted([int(left[i]) for i in range(len(left)) if i not in zl]
                           + [int(right[j]) for j in zr]), dtype=np.int64)


def basis_of(S, A_eq, b_eq, n):
    """(B, D = S \\ B, T, c) with x_D = c + T x_B on {A_eq x = b_eq}; B = S when there are no equalities or they are
    beyond MAX_DENSE"""
    S = np.asarray(S, dtype=np.int64)
    none = (S, np.zeros(0, np.int64), np.zeros((0, len(S))), np.zeros(0))
    if A_eq is None or A_eq.shape[0] == 0 or len(S) == 0 or A_eq.shape[0] * n > MAX_DENSE:
        return none
    A = np.asarray(A_eq.todense() if sp.issparse(A_eq) else A_eq, dtype=np.float64)
    b = np.asarray(b_eq, dtype=np.float64).reshape(-1)
    scale = np.abs(A).max(1)
    keep = scale > 0
    A, b = A[keep] / scale[keep, None], b[keep] / scale[keep]
    if A.shape[0] == 0:
        return none
    inS = np.zeros(n, bool)
    inS[S] = True
    AF, AS = A[:, ~inS], A[:, S]
    # numerical rank threshold on the scale of A (rows scaled to max |a| = 1), for both factorizations: what is left of
    # A_S after projecting out range(A_F) is rounding noise when A_F already spans A's rows
    tiny = max(A.shape) * np.finfo(np.float64).eps * np.linalg.norm(A)
    # project out the range of the non-S columns: the rows' constraints on x_S alone
    if AF.shape[1]:
        Q, R, _ = scipy.linalg.qr(AF, mode='economic', pivoting=True)
        U = Q[:, :int(np.sum(np.abs(np.diag(R)) > tiny))]
        M, rhs = AS - U @ (U.T @ AS), b - U @ (U.T @ b)
    else:
        M, rhs = AS, b
    Q2, R2, piv = scipy.linalg.qr(M, mode='economic', pivoting=True)
    k = int(np.sum(np.abs(np.diag(R2)) > tiny))
    if k == 0:
        return none
    R11, R12 = R2[:k, :k], R2[:k, k:]
    T = -scipy.linalg.solve_triangular(R11, R12)
    c = scipy.linalg.solve_triangular(R11, Q2[:, :k].T @ rhs)
    return S[piv[k:]], S[piv[:k]], T, c


def branching_space(polys, n, atom_support=None, A_eq=None, b_eq=None):
    """The BranchingSpace of exact polynomials (objective and rows) with linear equalities A_eq x = b_eq."""
    edges, fixed = product_structure(polys, n, atom_support)
    col, bipartite = colouring(edges, n)
    cands = []
    if len(edges):
        if bipartite:
            cands.append(koenig_cover(edges, col))
        prod = np.unique(edges)
        for k in range(int(col.max()) + 1):
            cands.append(prod[col[prod] != k])       # the complement of an independent set covers every edge
        if not bipartite:
            used = np.zeros(n, bool)
            for i, j in edges.tolist():
                if not (used[i] or used[j]):
                    used[i] = used[j] = True
            cands.append(np.flatnonzero(used))
    else:
        cands.append(np.zeros(0, np.int64))
    best = None
    for cover in cands:
        S = np.union1d(cover, np.flatnonzero(fixed))
        B, D, T, c = basis_of(S, A_eq, b_eq, n)
        if best is None or len(B) < len(best[1]):
            best = (S, B, D, T, c)
    S, B, D, T, c = best
    return BranchingSpace(n, edges, fixed, col, bipartite, S, B, D, T, c)
