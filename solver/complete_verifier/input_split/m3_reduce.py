"""M3: device-native exact-pattern propagation and one-candidate covering splits (flag solver.nlp.m3_reduce).

It applies when the compiled pattern (nlp_solver/structure/m3.py) is complete (an affine objective, every row
supported) and has signed squared differences; elsewhere it declines and the BaB is unchanged. Two node operations:
- propagate: three outward Jacobi sweeps of affine rows, one-product rows and signed squared differences over a
  batch of boxes, under the objective cutoff; returns contracted boxes, an outward interval bound of the affine
  objective, and the boxes proved empty;
- the covering continuation (continue_cover), which replaces bounding and splitting of a picked batch: split on the
  square-uncertainty score, propagate both midpoint children in one [2B, n] batch, commit the children with their
  boxes and certificates.
All numeric loops are the fixed sweeps or reduction-tree levels; nothing loops over rows or domains on the host.
Proofs: theory/kall_m3/kall_m3.md ("M3: the sufficient mechanism and its proof").
"""
import math
import time
from types import SimpleNamespace

import torch

import arguments
from input_split.tree_slice import checkpoint, region


def down(x):
    return torch.nextafter(x, torch.full_like(x, -math.inf))


def up(x):
    return torch.nextafter(x, torch.full_like(x, math.inf))


def lower_like(value, reference):
    """A lower bound cast to the reference's dtype, rounded downward when that dtype is narrower."""
    result = value.to(reference)
    if torch.finfo(result.dtype).eps > torch.finfo(value.dtype).eps:
        result = down(result)
    return result


def _forest(values, levels, roots, direction):
    for a, b, dst in levels:
        checkpoint()
        values[:, dst] = direction(values[:, a] + values[:, b])
    return values[:, roots]


def _clean(x, lower):
    return torch.where(torch.isnan(x), -math.inf if lower else math.inf, x)


def _intersect(lo, hi, idx, lower, upper):
    lo.scatter_reduce_(1, idx[None].expand(len(lo), -1), _clean(lower, True), reduce='amax', include_self=True)
    hi.scatter_reduce_(1, idx[None].expand(len(hi), -1), _clean(upper, False), reduce='amin', include_self=True)


def _cutoff_box(p, l, u, cutoff):
    """The box intersected with the affine objective cutoff f <= cutoff, before a coordinate is chosen. An impossible
    cutoff box stays a box with lo > hi, which propagate certifies as empty; it is never clamped."""
    if not p.affine_objective:
        return l, u
    terms = down(p.coef * torch.where(p.coef >= 0, l[:, p.col], u[:, p.col]))
    vals = l.new_zeros((len(l), p.nodes))
    vals[:, :p.E] = terms
    total = _forest(vals, p.levels, p.roots, down)[:, :1]
    ids = p.objective
    room = up(up(cutoff[:, None] - p.k[0]) - down(total - terms[:, ids]))
    coef = p.coef[ids]
    div = room / torch.where(coef != 0, coef, 1.)
    nl, nu = l.clone(), u.clone()
    _intersect(nl, nu, p.col[ids], torch.where(coef < 0, down(div), -math.inf),
               torch.where(coef > 0, up(div), math.inf))
    return nl, nu


@torch.no_grad()
def propagate(pattern, lower, upper, rhs, floor=None, cutoff=None, *, deadline=math.inf):
    """Three outward Jacobi sweeps; rhs already includes the accepted tolerances. With a cutoff, the returned boxes
    contain every feasible point with f <= cutoff and an empty box proves only f >= cutoff; without one they contain
    every feasible point. NaN operations never tighten. Returns lo, hi, lb (the objective interval bound, at least
    `floor`; the cutoff on empty boxes), live, and the sign/hole event counts."""
    p, B, dev = pattern, len(lower), lower.device
    lo, hi = lower.flatten(1).double().clone(), upper.flatten(1).double().clone()
    # finite-only dispatch: 0*inf and unbounded inverses are no-ops; contradictory boxes stay provable
    finite = torch.isfinite(lo).all(1) & torch.isfinite(hi).all(1)
    live = ~(lo > hi).any(1)
    l, u = torch.where(finite[:, None], lo, 0.), torch.where(finite[:, None], hi, 0.)
    cut = lo.new_full((B,), math.inf) if cutoff is None else torch.as_tensor(
        cutoff, device=dev, dtype=torch.float64).expand(B)
    cut = torch.where(torch.isnan(cut), math.inf, cut)
    accepted = rhs.double().clone()
    accepted[:, 0] = cut
    h = up(accepted[:, p.spec] - p.k)
    h = torch.where(p.base, -p.k, h)
    if not p.affine_objective:
        h[:, 0] = math.inf
    forced = torch.zeros(B, device=dev, dtype=torch.long)
    holes = torch.zeros(B, device=dev, dtype=torch.long)
    completed = 0
    for _ in range(3):
        checkpoint()
        if time.time() >= deadline:
            break
        nl, nu = l.clone(), u.clone()
        vals = l.new_zeros((B, p.nodes))
        term = down(p.coef * torch.where(p.coef >= 0, l[:, p.col], u[:, p.col]))
        ps = p.products
        ends = torch.stack((l[:, p.pi[ps]] * l[:, p.pj[ps]], l[:, p.pi[ps]] * u[:, p.pj[ps]],
                            u[:, p.pi[ps]] * l[:, p.pj[ps]], u[:, p.pi[ps]] * u[:, p.pj[ps]]), -1)
        pl, pu = down(ends.amin(-1)), up(ends.amax(-1))
        term[:, ps] = down(p.coef[ps] * torch.where(p.coef[ps] > 0, pl, pu))
        vals[:, :p.E] = term
        total = _forest(vals, p.levels, p.roots, down)
        bad = (total > h).any(1)
        # total is a downward sum of the stored lower terms: subtracting one *stored* term downward lower-bounds the
        # other terms, even under cancellation; no rounded scatter-add certifies anything
        rest = down(total[:, p.owner] - term)
        room = up(h[:, p.owner] - rest)
        div = room / torch.where(p.coef != 0, p.coef, 1.)
        _intersect(nl, nu, p.col, torch.where(p.linear & (p.coef < 0), down(div), -math.inf),
                   torch.where(p.linear & (p.coef > 0), up(div), math.inf))
        iv = p.inverse
        room = up(h[:, p.owner[iv]] - down(total[:, p.owner[iv]] - term[:, iv]))
        limit = torch.where(p.coef[iv] > 0, up(room / p.coef[iv]), down(room / p.coef[iv]))
        # both inverse directions at once; only a strictly positive denominator is eligible
        a, b = torch.cat((p.pi[iv], p.pj[iv])), torch.cat((p.pj[iv], p.pi[iv]))
        lim, c = limit.repeat(1, 2), p.coef[iv].repeat(2)
        positive = l[:, b] > 0
        q1 = lim / torch.where(positive, l[:, b], 1.)
        q2 = lim / torch.where(positive, u[:, b], 1.)
        _intersect(nl, nu, a, torch.where(positive & (c < 0), down(torch.minimum(q1, q2)), -math.inf),
                   torch.where(positive & (c > 0), up(torch.maximum(q1, q2)), math.inf))
        if p.H:
            aa = torch.maximum(down(l[:, p.fi] - u[:, p.fj]), p.sl)
            bb = torch.minimum(up(u[:, p.fi] - l[:, p.fj]), p.su)
            ms = up(torch.maximum(aa.square(), bb.square()))
            buf = l.new_zeros((B, p.sqnodes))
            buf[:, :p.H] = ms
            sums = _forest(buf, p.square_levels, p.froots, up)
            need = down(p.fk - accepted[:, p.fs])
            bad |= (aa > bb).any(1) | (sums < need).any(1)
            others = up(sums[:, p.fo] - ms)
            rad = down(torch.sqrt(down(need[:, p.fo] - others).clamp_min(0))).clamp_min(0)
            pos, neg = (rad > 0) & (aa > -rad), (rad > 0) & (bb < rad)
            forced += ((pos | neg) & live[:, None] & finite[:, None]).sum(1)
            holes += ((rad > 0) & ~pos & ~neg & live[:, None] & finite[:, None]).sum(1)
            aa, bb = torch.where(pos, torch.maximum(aa, rad), aa), torch.where(neg, torch.minimum(bb, -rad), bb)
            bad |= (aa > bb).any(1)
            _intersect(nl, nu, p.fi, down(aa + l[:, p.fj]), up(bb + u[:, p.fj]))
            _intersect(nl, nu, p.fj, down(l[:, p.fi] - bb), up(u[:, p.fi] - aa))
        bad |= (nl > nu).any(1)
        live &= ~(bad & finite)
        l, u = torch.where(live[:, None], nl, l), torch.where(live[:, None], nu, u)
        completed += 1
    # the affine objective on the last completed box, without the cutoff on its value; inherited floors kept
    vals = l.new_zeros((B, p.nodes))
    vals[:, :p.E] = down(p.coef * torch.where(p.coef >= 0, l[:, p.col], u[:, p.col]))
    obj = down(_forest(vals, p.levels, p.roots, down)[:, 0] + p.k[0])
    if not p.affine_objective:
        obj.fill_(-math.inf)
    bound = torch.where(finite, _clean(obj, True), -math.inf)
    bound = torch.where(live, torch.minimum(bound, cut), cut)
    if floor is not None:
        bound = torch.maximum(bound, _clean(floor.double(), True))
    lo = torch.where(finite[:, None], l, lo)
    hi = torch.where(finite[:, None], u, hi)
    return SimpleNamespace(lo=lo, hi=hi, lb=bound, live=live, forced=forced, holes=holes, sweeps=completed,
                           visits=completed * B * (p.E + 2 * p.H))


@torch.no_grad()
def children(pattern, lower, upper, rhs, floor, cutoff, *, deadline=math.inf, counter=None, history=None):
    """One uncertainty proposal, two covering children (left batch then right batch), one batched propagation. The
    minimum over both sides certifies the parent and is inherited by both. A midpoint that rounds to an endpoint
    still covers the parent and cannot invent progress."""
    p, B, dev = pattern, len(lower), lower.device
    l, u = lower.flatten(1).double(), upper.flatten(1).double()
    cut = torch.as_tensor(cutoff, device=dev, dtype=torch.float64).expand(B)
    l, u = _cutoff_box(p, l, u, cut)
    width = u - l
    score = torch.zeros_like(l)
    edge = width[:, p.fi] + width[:, p.fj]
    score.scatter_add_(1, p.fi[None].expand(B, -1), width[:, p.fi] * edge)
    score.scatter_add_(1, p.fj[None].expand(B, -1), width[:, p.fj] * edge)
    score = torch.where(width > 0, score, -math.inf)
    j = score.argmax(1)                  # first index wins ties; scores have no pruning authority
    if counter is not None:
        from input_split.branching_heuristics import _exhaustive_subdivision_guard
        choice, counter = _exhaustive_subdivision_guard(j[:, None], counter, l, u, history)
        j = choice[:, 0]
    ar = torch.arange(B, device=dev)
    # the committed boundary is representable in the queue's dtype: outward endpoint casts may overlap, not cross
    mid = (l[ar, j] + .5 * (u[ar, j] - l[ar, j])).to(lower.dtype).double()
    interior = torch.isfinite(mid) & (l[ar, j] < mid) & (mid < u[ar, j])
    cl, cu = l.repeat(2, 1), u.repeat(2, 1)
    cu[ar, j] = torch.where(interior, mid, u[ar, j])
    cl[B + ar, j] = torch.where(interior, mid, l[ar, j])
    out = propagate(p, cl, cu, rhs.repeat(2, 1), floor.repeat(2), cut.repeat(2), deadline=deadline)
    parent = out.lb.reshape(2, B).amin(0)
    out.lb = torch.maximum(out.lb, parent.repeat(2))
    out.index, out.split_j, out.split_mid, out.split_ok = ar.repeat(2), j, mid, interior
    out.counter = None if counter is None else (counter + 1).repeat(2, *([1] * (counter.ndim - 1)))
    return out


def setup(history, structure, n, base, device, deadline):
    """Compile the pattern from the root structure report and the root base rows; history.m3_pattern is set only
    when the continuation applies (complete pattern with signed squares) and the input-bound state is plain."""
    from nlp_solver.structure.m3 import M3Pattern
    history.m3_pattern, history.m3_deadline = None, deadline
    if structure is None or structure.skipped or not structure.forms:
        print('[M3] declined: exact structure unavailable')
        return
    if base is not None and not all(torch.is_tensor(v) for v in base):
        print('[M3] declined: unsupported base-row storage')
        return
    pattern = M3Pattern(structure.forms, n, base, device)
    reason = ('' if pattern.complete and pattern.H else 'needs complete rows and signed squares') or (
        'alpha/beta state uses the standard splitter' if history.use_alpha or history.use_beta else '')
    print(f'[M3] compiler complete={pattern.complete} squares={pattern.H} census={dict(pattern.census)}'
          + (f' declined: {reason}' if reason else ''))
    if not reason:
        history.m3_pattern = pattern


def continue_cover(domains, ret, primal, stats, cfg):
    """One continuation step on the picked batch `ret` (pick_out_batch's tuple): children propagated and committed
    to `domains` directly (no second evaluation, no CROWN), ancestor rows and state selected by ancestry. Children
    certified at the target T = P - gap allowance (rounded up) are retired with T as their discarded threshold.
    Returns the global bound min(T, P, open minimum, discarded thresholds) as a (1, 1) tensor."""
    from auto_LiRPA.utils import select_rows
    from input_split.utils import bound_thresholds
    p = domains.history.m3_pattern
    lb, lo, hi, cs, thresholds, rows, base, specs, split_idx, attacks, splits = ret
    B = len(lo)
    P = lo.new_tensor(math.inf, dtype=torch.float64) if primal is None else torch.as_tensor(
        primal, device=lo.device, dtype=torch.float64).reshape(-1)[0]
    P = torch.where(torch.isnan(P), math.inf, P)
    allowance = torch.maximum(P.new_tensor(float(cfg.get('bab_primal_dual_abs_tol', 1e-5))),
                              float(cfg.get('bab_primal_dual_rel_tol', 1e-4)) * P.abs().clamp_min(1))
    target = torch.where(torch.isfinite(P), up(P - allowance), P)
    stored = target.to(lb)
    target = torch.where(stored.double() < target, up(stored), stored).double()
    with region('m3_children', B):
        out = children(p, lo, hi, bound_thresholds(domains.history.rowpool, thresholds), lb[:, 0], target,
                       deadline=domains.history.m3_deadline, counter=splits, history=domains.history)
    sides = torch.arange(2 * B, device=lo.device) >= B
    keep = (out.lb < target) & (out.split_ok.repeat(2) | ~sides)   # a failed midpoint keeps one covering parent
    index = out.index[keep]
    childlb = lb[index].clone()
    childlb[:, 0] = torch.maximum(childlb[:, 0], lower_like(out.lb[keep], childlb[:, 0]))

    def take(value):
        if torch.is_tensor(value) and value.shape[0] > 1 and value.stride(0) == 0:
            return value[:1].expand(len(index), *value.shape[1:])
        return None if value is None else select_rows(value, index)
    # LP split hints belong to the last bounded batch, which M3 bypasses; M3 supplies its own covering split
    domains.history.lifted_branch_score = domains.history.lifted_split_point = None
    domains.history.filter_opt_variables(index, len(index))
    childlo = torch.maximum(lo[index].flatten(1), lower_like(out.lo[keep], lo))
    childhi = torch.minimum(hi[index].flatten(1), -lower_like(-out.hi[keep], hi))
    side = sides[keep]
    row = torch.arange(len(index), device=lo.device)
    j, mid, valid = out.split_j[index], out.split_mid[index].to(lo), out.split_ok[index]
    childlo[row, j] = torch.where(side & valid, torch.maximum(childlo[row, j], mid), childlo[row, j])
    childhi[row, j] = torch.where(~side & valid, torch.minimum(childhi[row, j], mid), childhi[row, j])
    with region('m3_add', len(index)):
        domains.add(childlb, childlo.reshape(-1, *lo.shape[1:]), childhi.reshape(-1, *hi.shape[1:]), take(cs),
                    take(thresholds), constraints=None if rows is None else tuple(take(v) for v in rows),
                    base_constraints=None if base is None else tuple(take(v) for v in base), spec_sizes=take(specs),
                    split_idx=take(split_idx), attack_counter=take(attacks),
                    split_counter=None if out.counter is None else out.counter[keep], check_dm_lbs=True,
                    check_input_boxes=True)
    # a cutoff contradiction certifies the target, neither P nor unrestricted infeasibility
    cap = arguments.Globals.get('discarded_threshold_min')
    arguments.Globals['discarded_threshold_min'] = min(float(target), math.inf if cap is None else cap)
    D = torch.minimum(target, P)
    if len(domains):
        key = domains._objective_key(0, len(domains), 'lb_only', 0)
        if key is None:
            key = domains.lb._storage[:len(domains), 0]
        D = torch.minimum(D, key.min().to(D))
    D = torch.minimum(D, D.new_tensor(arguments.Globals['discarded_threshold_min']))
    stats.visited += 2 * B
    print(f'[M3] continuation batch={B} children={2 * B} retained={len(index)} visits={out.visits} '
          f'empty={int((~out.live).sum())} forced={int(out.forced.sum())} holes={int(out.holes.sum())} '
          f'open={len(domains)} D={float(D):.12g}')
    stored = D.to(lb)
    stored = torch.where(stored.double() > D, down(stored), stored)
    return stored.reshape(1, 1)
