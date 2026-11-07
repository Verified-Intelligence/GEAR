"""Outward-rounded tensor kernels of the root structure certificates (structure/certificates.py).

Every kernel is a fixed pattern of gathers, elementwise IEEE operations rounded outward with nextafter, and fixed
pairwise reduction trees, batched over its leading dimension. Only reduction levels and cover depths loop on the host.
"""
import math

import numpy as np
import torch

DT = torch.float64
SCALE = 1 << 40            # exact dyadic unit of the cover search
LIMIT = 1 << 60            # int64 sentinel; the overflow guards below keep every sum under it


def down(x):
    return torch.nextafter(x, torch.full_like(x, -math.inf))


def up(x):
    return torch.nextafter(x, torch.full_like(x, math.inf))


def tree(values, direction):
    """Pairwise sum over the last dimension, rounding each addition by `direction` (odd tails included)."""
    while values.shape[-1] > 1:
        width = values.shape[-1]
        summed = direction(values[..., :width - width % 2:2] + values[..., 1:width:2])
        values = torch.cat((summed, values[..., -1:]), -1) if width % 2 else summed
    return values[..., 0]


def combination_ids(n, k, device):
    """All k-subsets of range(n) in lexicographic order, (C(n, k), k), by unranking on the device: O(C(n, k) n k) work in
    k fixed stages (torch.combinations would allocate an n**k mesh)."""
    P = math.comb(n, k)
    table = torch.empty(k, n, dtype=torch.long, device=device)
    column = torch.ones(n, dtype=torch.long, device=device)
    table[k - 1] = column.flip(0)
    for order in range(1, k):
        column = torch.cat((column.new_zeros(1), column[:-1])).cumsum(0)
        table[k - 1 - order] = column.flip(0)
    rank = torch.arange(P, device=device)
    previous = torch.full_like(rank, -1)
    vertices = torch.arange(n, device=device)
    out = torch.empty(P, k, dtype=torch.long, device=device)
    for j in range(k):
        counts = torch.where(vertices[None] > previous[:, None], table[j], 0)
        ends = counts.cumsum(1)
        choice = (ends > rank[:, None]).long().argmax(1)
        rank -= (ends - counts).gather(1, choice[:, None])[:, 0]
        previous = choice
        out[:, j] = choice
    return out


class CostPattern:
    """Interval group costs, vectorized over (batch, patterns, features, signs, slopes).

    ids (P, k) are the selected items of each pattern; al/au (F, n) the feature coefficients and target_l/target_u (F,)
    the targets, rounded outward; cl/cu/abs_c (F, 2, n+1) the signed breakpoint slopes. The unselected sum is enclosed
    as (all - selected), so no P x n x n mask is needed."""

    def __init__(self, ids, al, au, target_l, target_u, cl, cu, abs_c):
        self.ids, self.al, self.au = ids, al, au
        self.target_l, self.target_u = target_l, target_u
        self.cl, self.cu, self.abs_c = cl, cu, abs_c

    def costs(self, delta, tau):
        """delta, tau (B,), already rounded upward: (B, P) lower bounds of the group cost of each pattern."""
        sl = torch.stack((-self.au, self.al), 1)
        su = torch.stack((-self.al, self.au), 1)
        pos = up(self.cu[..., None] - sl[:, :, None, :]).clamp_min(0)
        neg = up(su[:, :, None, :] - self.cl[..., None]).clamp_min(0)
        # (P, F, sign, slope, k), gathered on the shared integer pattern
        selected_pos = pos[..., self.ids].permute(3, 0, 1, 2, 4)
        selected_neg = neg[..., self.ids].permute(3, 0, 1, 2, 4)
        rest = up(tree(neg, up)[None] - tree(selected_neg, down))
        deviations = up(rest + tree(selected_pos, up))
        val = up(up(self.abs_c[None] * tau[:, None, None, None])[:, None]
                 + up(delta[:, None, None, None, None] * deviations[None]))
        error = val.amin(-1)                                       # (B, P, F, sign)
        rl = down(self.target_l[None] + tree(self.al[:, self.ids].permute(1, 0, 2), down))
        ru = up(self.target_u[None] + tree(self.au[:, self.ids].permute(1, 0, 2), up))
        l, u = down(rl[None] - error[..., 0]), up(ru[None] + error[..., 1])
        dist = torch.maximum(torch.maximum(l, -u), torch.zeros_like(l))
        return tree(down(dist.square()), down).clamp_min(0)


def cert_crown_dual():
    """solver.nlp.cert_crown_dual: the root certificates' LP duals come from the restarted Halpern kernel, not HiGHS."""
    try:
        import arguments
        return bool(arguments.Config['solver']['nlp'].get('cert_crown_dual', False))
    except (ImportError, KeyError, AttributeError, TypeError):
        return False


def halpern_lp(c, A, b, lower, upper, stop=None, deadline=None, budget=100000):
    """min c.z s.t. A z <= b, lower <= z <= upper by the restarted Halpern kernel (auto_LiRPA.restarted_qp.solve,
    implied box). Returns (z, y >= 0) or None when the lane proves the box empty. Only a proposal: the callers
    re-evaluate the box Lagrangian of y with outward intervals. stop: the objective value at which the dual bound is
    already enough (checked by the kernel at each checkpoint); deadline: wall-clock time.time()."""
    from auto_LiRPA.restarted_qp import solve
    from nlp_solver.rigorous_dual import Rows
    f64 = torch.float64
    G = torch.as_tensor(np.asarray(A, dtype=np.float64))
    m, n = G.shape
    vec = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float64))[None]
    res = solve(Rows.from_dense(G[None]), vec(c), torch.zeros(1, dtype=f64), vec(b), vec(lower), vec(upper),
                torch.zeros(1, n, dtype=f64), budget=budget, checkpoint=25, capture=False,
                stop=None if stop is None else torch.tensor([stop], dtype=f64), deadline=deadline, warm=None,
                iteration_scheme='halpern', implied_box=True, convergence=1e-9)
    if bool(res.empty[0]):
        return None
    return res.z[0].numpy(), res.best_y[0].clamp(min=0).numpy()


def lp_prices(ids, costs, n):
    """A proposal of item prices: the row duals of min sum_S c_S z_S s.t. sum_{S contains i} z_S = 1, z >= 0 (HiGHS;
    with solver.nlp.cert_crown_dual the Halpern kernel on the same LP, equalities as two rows).
    Only a proposal: cover() repairs any prices into a valid bound. Zeros when the solve fails."""
    P, k = ids.shape
    if cert_crown_dual():
        A = np.zeros((n, P))
        A[ids.detach().cpu().numpy().ravel(), np.repeat(np.arange(P), k)] = 1.
        out = halpern_lp(costs.detach().cpu().double().numpy(), np.vstack([A, -A]), np.r_[np.ones(n), -np.ones(n)],
                         np.zeros(P), np.ones(P))
        # sum_S contains i z_S = 1 <-> max 1.pi s.t. A^T pi <= c; pi = -(y+ - y-)
        pi = np.zeros(n) if out is None else -(out[1][:n] - out[1][n:])
        return torch.as_tensor(pi if np.isfinite(pi).all() else np.zeros(n), dtype=DT, device=ids.device)
    import highspy
    import scipy.sparse as sp
    cols = ids.detach().cpu().numpy()
    A = sp.csr_matrix((np.ones(P * k), (cols.ravel(), np.repeat(np.arange(P), k))), shape=(n, P))
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.setOptionValue('threads', 1)
    h.addVars(P, np.zeros(P), np.full(P, highspy.kHighsInf))
    h.changeColsCost(P, np.arange(P, dtype=np.int32), costs.detach().cpu().double().numpy())
    h.addRows(n, np.ones(n), np.ones(n), A.nnz, A.indptr.astype(np.int32), A.indices.astype(np.int32), A.data)
    h.run()
    dual = np.asarray(h.getSolution().row_dual, dtype=np.float64)
    ok = h.getModelStatus() == highspy.HighsModelStatus.kOptimal and dual.shape == (n,) and np.isfinite(dual).all()
    return torch.as_tensor(dual if ok else np.zeros(n), dtype=DT, device=ids.device)


class CoverPattern:
    """Exact dyadic minimum-cost partition of n items into patterns of k items, with a fixed-width device frontier.

    Costs are rounded down to multiples of 2^-40 and prices repaired in exact int64 so that every reduced cost is
    nonnegative; for any partition, cost = sum(prices) + sum(reduced costs). The frontier pivots on the smallest
    uncovered item, so it enumerates every partition; the search is complete when no depth overflows the width, and
    an incomplete or expired search returns sum(prices), never the best cover found."""

    def __init__(self, ids, n, width=512):
        self.ids, self.n, self.k = ids, n, ids.shape[1]
        self.P, self.groups, self.width = len(ids), n // self.k, width
        self.masks = (torch.ones_like(ids) << ids).sum(1)

    def solve(self, costs, prices, stop=lambda: False):
        """costs (P,) >= 0, prices (n,) a proposal: (value, best path (groups,), complete, states)."""
        dev, W, G = costs.device, self.width, self.groups
        safe = bool(torch.isfinite(costs).all() & (costs >= 0).all()
                    & (costs.amax() * SCALE < LIMIT // (16 * self.n * G)))
        if not safe:                     # the universally valid zero
            return 0., torch.zeros(G, dtype=torch.long, device=dev), False, 0
        integer = torch.floor(costs * SCALE).long()
        price_limit = LIMIT // (16 * self.n * self.n)
        prices = torch.floor((torch.where(torch.isfinite(prices), prices, 0.) * SCALE)
                             .clamp(-price_limit, price_limit)).long()
        excess = (prices[self.ids].sum(-1) - integer).amax().clamp_min(0)
        prices = prices - (excess + self.k - 1) // self.k
        reduced = integer - prices[self.ids].sum(-1)
        dual = prices.sum()
        full = (1 << self.n) - 1 if self.n < 64 else -1
        # greedy completion: a feasible cover, the first incumbent of the search
        rem = torch.tensor(full, dtype=torch.long, device=dev)
        bestpath = torch.zeros(G, dtype=torch.long, device=dev)
        best = torch.zeros((), dtype=torch.long, device=dev)
        for depth in range(G):
            pivot = rem & -rem
            feasible = ((rem & self.masks) == self.masks) & ((pivot & self.masks) != 0)
            pick = torch.where(feasible, integer, LIMIT).argmin()
            bestpath[depth] = pick
            best = best + reduced[pick]
            rem = rem ^ self.masks[pick]
        expired, states, complete = False, 0, False
        # The first pass finds a strong upper bound; the second restarts with it, so an early frontier overflow does
        # not prevent an exact proof once the admissible tree is small.
        for _ in range(2):
            rem = torch.full((1,), full, dtype=torch.long, device=dev)
            spent = torch.zeros_like(rem)
            path = torch.zeros(1, G, dtype=torch.long, device=dev)
            complete = True
            for depth in range(G):
                if stop():
                    expired, complete = True, False
                    break
                pivot = rem & -rem
                trial = spent[:, None] + reduced[None]
                valid = (((rem[:, None] & self.masks) == self.masks) & ((pivot[:, None] & self.masks) != 0)
                         & (trial <= best))
                count = int(valid.sum())
                states += count
                if depth == G - 1:
                    val = torch.where(valid, trial, LIMIT).amin()
                    tied = valid & (trial == val)
                    # canonical lexicographic path among equal covers (never topk's device-specific ties)
                    for column in range(G):
                        digits = path[:, column, None] if column < depth else torch.arange(self.P, device=dev)[None]
                        tied &= digits == torch.where(tied, digits, self.P).amin()
                    pick = int(tied.flatten().long().argmax())
                    chosen = path[pick // self.P].clone()
                    chosen[depth] = pick % self.P
                    differs = chosen != bestpath
                    first = int(differs.long().argmax())
                    lex_less = bool(differs.any()) and bool(chosen[first] < bestpath[first])
                    if bool(val < best) or (bool(val == best) and lex_less):
                        bestpath = chosen
                    best = torch.minimum(best, val)
                    continue
                complete = complete and count <= W
                size = min(W, trial.numel())
                vals, picks = torch.topk(torch.where(valid, trial, LIMIT).flatten(), size, largest=False, sorted=True)
                parent, edge = picks // self.P, picks % self.P
                path = path[parent]
                path[:, depth] = edge
                rem = rem[parent] ^ self.masks[edge]
                spent = vals
        # a feasible cover with zero reduced cost already attains the dual bound
        complete = (complete or int(best) == 0) and not expired
        value_int = int(best + dual) if complete else int(dual)
        # the integer sums are exact; the conversion to double rounds downward
        value = float(np.nextafter(value_int / SCALE, -np.inf)) if abs(value_int) > 1 << 53 else value_int / SCALE
        return value, bestpath, complete, states
