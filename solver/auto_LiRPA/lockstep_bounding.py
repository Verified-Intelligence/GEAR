"""GEAR's bounding call on the lockstep block dual: one call per BaB iteration on a batch of domains, every domain
with a resumable state (crown_absorb/lockstep; ARCH_ONE_TREE.md).

    lb = LockstepBounding(m, n, levels, kind, ext, certifier, R=8, buckets=(64, 256, 1024), dict_K=64)
    out = lb.bound(tables, state, T)      # state: DomainState of the parents (inherited by the children) or None

`tables`: the domains' face tables in the compiled CROWN form (dense rows R (D, m, n), h (D, m), column boxes l, u
(D, n), default faces dlow, dup (D, n)) and the certificate's face data (lw, lb, uw, ub, as CompiledCertifier.load);
`certifier(B)` makes a CompiledCertifier of B lanes on the graph. Per call and per lane bucket: setup (sign-CROWN bound
and ray, the parent's multipliers re-certified, the inherited labels repaired), the dictionary's certificates (the
last dict_K bound multipliers and Farkas rays found by earlier calls, evaluated in the search's dtype, each lane's best
bound entry and best ray certified rigorously), R lockstep rounds of the survivors, finish;
captured CUDA graphs, fp32 search by default with a per-lane fp64 redo of the lanes it flags (want64). Returns the
certified lower bounds, the prune kind (0 open, 1 bound >= T, 2 Farkas), the new states (the children's parents), the
basic points (branching), and the optimal / numerical / fp64 flags. Nothing outside this module calls it yet."""
import collections
from contextlib import nullcontext
import math
import time

import torch

from .compiled_crown import Captured, fuse
from .lockstep_dual import LockstepDual, UpdatedRound, Q
from .cuda_memory import CudaMemoryBudget, capture, memory_error, cuda_launch_error, release_cuda
from .utils import select_rows

INF = float('inf')
DomainState = collections.namedtuple('DomainState', 'ids best y')       # (D, n) long (-1: none), (D,), (D, m) fp64
BoundResult = collections.namedtuple('BoundResult', 'lower kind state witness optimal numerical fp64 yfin',
                                     defaults=(None,))


class _PlaneValues:
    """Search arithmetic only; callers certify the returned nonnegative masses."""
    def prepare(self):
        pass

    def __call__(self, y, objective):
        e = self.engine
        # The certifier interface carries fp64 masses even for an fp32 search.
        # These values schedule proposals only; strict replay is separate.
        if getattr(self, 'cast_search', False):
            y, objective = y.to(e.dt), objective.to(e.dt)
        r = objective[None, :, None] * e.c[:, None] + y @ e.R
        return -(y * e.h[:, None]).sum(-1) + torch.minimum(r * e.l[:, None], r * e.u[:, None]).sum(-1)


class _PlaneFrameDual(LockstepDual):
    """Use the last nonzero of each plane as a triangular frame row.

    Every other coefficient precedes that column. Thus any retained row per
    tail forms a unit lower triangular CROWN frame; duplicate tails remain
    Schur corrections. No coefficient or constraint is discarded. The cold
    sign basis still uses box faces, and original-plane beta is certified.
    """
    def _scaled_lp(self):
        super()._scaled_lp()
        # A lane can lose the union's tail when a coefficient becomes zero.
        # New edges must also respect the frozen CROWN levels. Otherwise keep
        # the row in Schur, so changing numeric sparsity never invalidates T.
        self.loc[:, :self.m].copy_((self.tail[:, :self.m] == self._frame_tail)
            & (self.tail[:, :self.m] >= 0)
            & ~((self.A[:, :self.m] != 0) & ~self._frame_mask).any(2))

    def _F_T(self, T, q):
        value = q.clone()
        for s, e in self._frame_levels[1:]:
            product = T[:, s:e, :s] @ (value[:, :s, None] if q.ndim == 2 else value[:, :s])
            value[:, s:e] = q[:, s:e] - (product[..., 0] if q.ndim == 2 else product)
        return value

    def _R_T(self, T, q):
        value = q.clone()
        for s, e in reversed(self._frame_levels[1:]):
            product = T[:, s:e, :s].transpose(1, 2) @ (value[:, s:e, None] if q.ndim == 2 else value[:, s:e])
            value[:, :s] -= product[..., 0] if q.ndim == 2 else product
        return value


class PlaneCROWN:
    """Block beta-CROWN on input planes, with an identity CROWN backbone.

    The only nonlocal rows are the constraint planes, hence the Schur order
    is at most min(m, n, cap). The legacy path runs exactly `rounds` rounds;
    fast mode uses it as a ceiling and checks progress every block.
    The packed state [basis labels | beta] can be filtered/repeated with boxes.
    No scalar search bound is inherited: even parent beta is re-certified on
    the fresh planes by `certify`. Search values and statuses are never proofs.
    """
    def __init__(self, certify, cap=None, max_elems=None, capture=True, fast=False, memory=None,
                 memory_limit=float('inf'), memory_reserve=0, rebase_memory=False, adaptive=False, cast_search=False):
        self.certify, self.cap = certify, cap
        self.max_elems = (32_000_000 if fast else 4_000_000) if max_elems is None else max_elems
        self.capture = capture
        self.cast_search = cast_search
        self.fast = fast
        self.adaptive = adaptive
        self.adaptive_lanes = adaptive and max_elems is None
        self.buckets = {}
        self.plane_pools = {}
        self.plane_layouts = {}
        self.work = collections.Counter()
        self.lane_limits = {}
        self.memory = memory
        self.owns_memory = memory is None
        self.memory_limit, self.memory_reserve = memory_limit, memory_reserve
        self.rebase_memory = rebase_memory

    @torch.no_grad()
    def bound(self, c, c0, G, h, lower, upper, rounds, state=None, cutoff=None, deadline=None,
              max_domains=None):
        B, m, n = G.shape
        if deadline is not None and time.time() >= deadline:
            return self._weak(c, c0, h, lower, upper, state)
        if self.adaptive and min(m, n, n if self.cap is None else self.cap) <= 64:
            # These Schur systems already fit the reference's single-CTA
            # inverse. Triangular BLAS and adaptive capture add overhead;
            # retain the reference arithmetic and residency for small tables.
            adaptive, lanes = self.adaptive, self.adaptive_lanes
            self.adaptive = self.adaptive_lanes = False
            try:
                return self.bound(c,c0,G,h,lower,upper,rounds,state,cutoff,deadline,max_domains)
            finally:
                self.adaptive, self.adaptive_lanes = adaptive, lanes
        if c.is_cuda:
            if self.adaptive and self.owns_memory:
                # The caller's native CROWN batch can grow since the root.
                # It is not newly allocated plane workspace. Rebase admission
                # on that live caller state, while charging our resident
                # buckets against the configured plane cap. Driver headroom
                # and the allocator guard still apply to every search.
                resident = sum(e.B*self._scratch(e.c,e.m,e.n) for e in self.buckets.values())
                self.memory = CudaMemoryBudget(c.device, max(0, self.memory_limit-resident), self.memory_reserve)
            elif self.memory is None:
                self.memory = CudaMemoryBudget(c.device, self.memory_limit, self.memory_reserve)
        cutoff = (torch.as_tensor(cutoff, device=c.device, dtype=c.dtype).expand(B)
                  if self.fast and cutoff is not None else None)
        if max_domains is not None and B > max(0, int(max_domains)):
            result = self._weak(c, c0, h, lower, upper, state)
            count = max(0, int(max_domains))
            if count:
                # A proposal ranking only: these rounded box values never
                # supply a certificate. Every unselected domain stays present.
                price = c0 + torch.minimum(c * lower, c * upper).sum(-1)
                eligible = torch.isfinite(price)
                if cutoff is not None:
                    eligible &= price < cutoff
                gap = price if cutoff is None else price - cutoff
                count = min(count, int(eligible.sum()))
                if count:
                    ids = torch.nonzero(eligible).flatten()
                    ids = ids[torch.topk(gap[ids], count).indices]
                    chosen = self.bound(c[ids], c0[ids], select_rows(G, ids), h[ids],
                        lower[ids], upper[ids], rounds, None if state is None else state[ids],
                        None if cutoff is None else cutoff[ids], deadline)
                    for target, value in zip(result, chosen):
                        target.index_copy_(0, ids, value.to(target))
            return result
        # Dense exchange work includes n*n basis rows and eight trial RHSs.
        ceiling = 32
        if self.fast and c.is_cuda:
            if c.device not in self.lane_limits:
                sms = torch.cuda.get_device_properties(c.device).multi_processor_count
                # Small GJ matrices use one block per lane. Several waves fill
                # the SMs; large tables remain bounded by the matrix budget.
                self.lane_limits[c.device] = 4 * (1 << (sms-1).bit_length())
            ceiling = self.lane_limits[c.device]
        lanes = max(1, min(ceiling, self.max_elems // max((m + 3*n)*n, 1)))
        if self.memory is not None:
            scratch = self._scratch(c, m, n)
            if self.adaptive_lanes:
                # The triangular frame no longer needs a full-width Schur
                # LU per lane. Let measured allocator headroom admit the
                # wave; explicit coefficient limits still take precedence.
                lanes = ceiling
            available = self.memory.available()
            if deadline is not None and available < scratch:
                # Reclaim once for the whole call. Retrying every domain as a
                # single lane under a full budget only repeats GC/capture work.
                self.buckets.clear()
                if self.rebase_memory and isinstance(self.memory, CudaMemoryBudget):
                    # Native CROWN/tree allocations grew since the first plane
                    # call. After releasing every owned bucket, account that
                    # live baseline separately from this optional workspace.
                    self.memory = CudaMemoryBudget(c.device, self.memory_limit, self.memory_reserve)
                else:
                    release_cuda(c.device)
                available = self.memory.available()
                if available < scratch:
                    print(f'[crown-memory] plane batch skipped B={B}: one lane exceeds admission', flush=True)
                    return self._weak(c, c0, h, lower, upper, state)
            lanes = max(1, int(min(lanes, available // scratch)))
        lanes = 1 << (lanes.bit_length()-1)
        parts = []
        for begin in range(0, B, lanes):
            if deadline is not None and time.time() >= deadline:
                parts.append(self._weak(c[begin:], c0[begin:], h[begin:], lower[begin:], upper[begin:],
                                        None if state is None else state[begin:]))
                break
            s = slice(begin, min(begin + lanes, B))
            parts.append(self._run(c[s], c0[s], G[s], h[s], lower[s], upper[s], rounds,
                                     None if state is None else state[s],
                                     None if cutoff is None else cutoff[s],
                                     **({} if deadline is None else {'deadline': deadline})))
        return tuple(torch.cat([p[i] for p in parts]) for i in range(4))

    def _scratch(self, c, m, n):
        width = max(Q, m)
        return c.element_size() * max(1, 3*width*n + 12*n*n + (0 if self.fast else width*width))

    def _run(self, c, c0, G, h, lower, upper, rounds, state, cutoff=None, deadline=None):
        B, m, n = G.shape
        timed = {} if deadline is None else {'deadline': deadline}
        if deadline is not None and time.time() >= deadline:
            return self._weak(c, c0, h, lower, upper, state)
        try:
            with self.memory.allocations() if self.memory is not None else nullcontext():
                if self.memory is not None:
                    self.memory.require(B * self._scratch(c, m, n))
                result = self._bound(c, c0, G, h, lower, upper, rounds, state, cutoff, **timed)
                if self.memory is not None:
                    self.memory.require(0)
                return result
        except RuntimeError as error:
            if not memory_error(error) and not cuda_launch_error(error):
                raise
            print(f'[crown-memory] plane fallback B={B}: {error}', flush=True)
        # Release failed captures and their cyclic engine/certifier references
        # outside the exception scope before retrying with fewer eager lanes.
        result = None
        self.buckets.clear()
        captured, self.capture = self.capture, False
        release_cuda(c.device)
        if B > 1:
            chunk = max(1, B//2)
            parts = [self._run(c[s:s+chunk], c0[s:s+chunk], G[s:s+chunk], h[s:s+chunk],
                               lower[s:s+chunk], upper[s:s+chunk], rounds,
                               None if state is None else state[s:s+chunk],
                               None if cutoff is None else cutoff[s:s+chunk], **timed)
                     for s in range(0, B, chunk)]
            return tuple(torch.cat([p[k] for p in parts]) for k in range(4))
        if captured:
            return self._run(c, c0, G, h, lower, upper, rounds, state, cutoff, **timed)
        # No search result is a proof. Keep the caller's inherited proposal;
        # a vacuous bound lets native/inherited certificates remain in force.
        return self._weak(c, c0, h, lower, upper, state)

    @staticmethod
    def _weak(c, c0, h, lower, upper, state):
        B, n, m = len(c), c.shape[1], h.shape[1]
        packed = (state.clone() if state is not None and state.shape == (B, n+m) else
                  torch.cat((c.new_full((B, n), -1), torch.zeros_like(h)), 1))
        return c0.new_full((B,), -INF), torch.zeros_like(h), (lower+upper)/2, packed

    def _bound(self, c, c0, G, h, lower, upper, rounds, state, cutoff=None, deadline=None):
        B, m, n = G.shape
        width = max(Q, m)
        lanes = 1 << (B-1).bit_length()
        cap = max(1, min(m, n, n if self.cap is None else self.cap))
        active = torch.isfinite(h) & torch.isfinite(G).all(-1)
        matrix, rhs = c.new_zeros(B, width, n), c.new_ones(B, width)
        matrix[:, :m] = torch.where(active[..., None], G, 0.)
        rhs[:, :m] = torch.where(active, h, 1.)
        original = order = inverse = None
        if self.adaptive:
            layout_key = width, n, c.device
            layout = self.plane_layouts.get(layout_key)
        if self.adaptive and layout is None:
            # Normalize each row's vote: a global sum must not make its
            # private outputs appear more shared than their local operands.
            nonzero = matrix != 0
            degree = (nonzero / nonzero.sum(2, keepdim=True).clamp(min=1)).sum((0, 1))
            order = torch.argsort(degree, descending=True, stable=True)
            # Build the union DAG once per wave. Rows end at their last
            # nonzero, so all predecessors already have a level. Grouping
            # independent columns allows ordinary CROWN level propagation.
            pattern = nonzero[:, :, order].any(0).cpu().numpy()
            depth, tails = [0]*n, []
            for row in pattern:
                columns = row.nonzero()[0]
                tails.append(int(columns[-1]) if len(columns) else -1)
            for j in range(n):
                predecessors = pattern[[i for i, tail in enumerate(tails) if tail == j], :j].any(0)
                depth[j] = max((depth[k]+1 for k in predecessors.nonzero()[0]), default=0)
            groups = sorted(range(n), key=depth.__getitem__)
            remap = [0]*n
            for j, old in enumerate(groups):
                remap[old] = j
            levels = tuple((sum(d < level for d in depth), sum(d <= level for d in depth))
                           for level in sorted(set(depth)))
            frame_tail = torch.tensor([remap[t] if t >= 0 else -1 for t in tails], device=c.device)
            order = order[torch.tensor(groups, device=c.device)]
            inverse = torch.argsort(order)
            column_depth = torch.tensor(sorted(depth), device=c.device)
            mask = ((column_depth[None] < column_depth[frame_tail.clamp(min=0), None])
                    | (torch.arange(n, device=c.device)[None] == frame_tail[:, None]))
            # Numeric zeros must not create a new captured topology for every
            # child wave. Freeze the ordering/levels and check new edges when
            # scaling each table; incompatible rows remain exact Schur rows.
            layout = order, inverse, levels, frame_tail, mask
            self.plane_layouts[layout_key] = layout
        if self.adaptive:
            order, inverse, levels, frame_tail, mask = layout
            original = c, matrix, lower, upper
            c, matrix = c[:, order], matrix[:, :, order]
            lower, upper = lower[:, order], upper[:, order]
        key = lanes, width, n, cap, c.device, c.dtype
        if self.adaptive:
            key += (levels,)
        if key not in self.buckets:
            cert = _PlaneValues()
            cert.cast_search = self.cast_search
            dual = _PlaneFrameDual if self.adaptive else LockstepDual
            e = dual(width, n, [(0, n)], torch.zeros(width, dtype=torch.long), cert, lanes,
                             cap=cap, device=c.device, dtype=c.dtype, kernel='lapack' if cap > 64 else None,
                             bland='fable', trial='exchange')
            cert.engine = e
            e.finite_factors = True
            e.identity_backbone = self.fast and not self.adaptive
            e._plane_key = key
            # On an identity backbone the sign-start row masses are all zero.
            # Avoid constructing its m*m zero bank; shortlist the single-row
            # Farkas candidate directly below and rigorously check it there.
            e.bind(torch.arange(0 if self.fast else width, device=c.device))
            self.buckets[key] = e
        e = self.buckets[key]
        if self.adaptive:
            e._frame_levels, e._frame_tail = levels, frame_tail
            e._frame_mask = mask
        ids = beta = None
        if state is not None and state.shape == (B, n+m):
            ids = state[:, :n].long()
            if self.adaptive:
                box = (ids >= width) & (ids < width+2*n)
                ids = torch.where(box, width+(ids-width).div(n, rounding_mode='floor')*n
                                  + inverse[(ids-width) % n], ids)
            beta = torch.nn.functional.pad(state[:, n:], (0, width-m))
        def load():
            e.load(0, matrix, rhs, c, torch.zeros_like(c0), lower, upper,
                   torch.full((B, n), -1, device=c.device, dtype=torch.long),
                   torch.full((B, n), -1, device=c.device, dtype=torch.long),
                   c0.new_full((B,), INF) if cutoff is None else cutoff-c0,
                   pids=ids, pbest=c0.new_full((B,), -INF), py=beta)
            e.deactivate(B)
        load()
        if self.adaptive:
            # Setup/finish each run once. Size their factors from the current
            # labels instead of capturing full-capacity padded LU workspaces
            # for every batch bucket (which can evict the useful round graphs).
            e._scaled_lp()
            box_ids = width+torch.arange(n, device=c.device).expand(e.B, -1)
            parent = torch.where(e.has_pids[:, None], e.pids, box_ids)
            self._frame_width(e, parent)
            e.setup()
            if e.graphs is None:
                e.graphs = {'round': _Eager(e.round)}
            self._search(e, e.round, rounds, B, deadline)
            self._frame_width(e, e.ids)
            e.final_masses()
            e.finish()
            e.set_kap(e.cap)
        elif c.is_cuda and self.capture:
            if e.graphs is None:
                pool = torch.cuda.graph_pool_handle()
                e.graphs = {'setup': _Graph(e.setup, pool),
                            'round': _Eager(e.round) if self.adaptive else _Graph(e.round, pool),
                            'finish': _Graph(lambda: (e.final_masses(), e.finish()), pool)}
                load()  # capture warmups mutate the search, never the caller state
            e.graphs['setup'].replay()
            if self.fast:
                self._search(e, e.graphs['round'].replay, rounds, B, deadline)
            else:
                for iteration in range(rounds):
                    if deadline is not None and iteration % Q == 0:
                        torch.cuda.synchronize(c.device)
                        if time.time() >= deadline:
                            break
                    e.graphs['round'].replay()
            e.graphs['finish'].replay()
        else:
            e.setup()
            if self.fast:
                self._search(e, e.round, rounds, B, deadline)
            else:
                for iteration in range(rounds):
                    if deadline is not None and iteration % Q == 0:
                        if c.is_cuda:
                            torch.cuda.synchronize(c.device)
                        if time.time() >= deadline:
                            break
                    e.round()
            e.final_masses()
            e.finish()
        if self.adaptive:
            c, matrix, lower, upper = original
        G, h = matrix[:, :m], rhs[:, :m]
        best_y = torch.zeros_like(h)
        value = self.certify(c, c0, G, h, lower, upper, best_y)
        candidates = [e.besty[:B, :m], e.yfin[:B, :m]]
        if beta is not None:
            candidates.append(beta[:, :m])
        for y in candidates:
            y = torch.where(torch.isfinite(y), y, 0.).clamp(min=0)
            if self.fast:
                # Converged bases often return the same beta as besty, yfin
                # and the parent. Reuse only a certificate from this table;
                # never inherit its scalar from a different box or objective.
                fresh = (y != best_y).any(1)
                v = value.new_full(value.shape, -INF)
                if bool(fresh.any()):
                    v[fresh] = self.certify(c[fresh], c0[fresh], select_rows(G, fresh), h[fresh],
                                           lower[fresh], upper[fresh], y[fresh])
            else:
                v = self.certify(c, c0, G, h, lower, upper, y)
            best_y = torch.where((v > value)[:, None], y, best_y)
            value = torch.maximum(value, v)
        ray = torch.where(torch.isfinite(e.rayy[:B, :m]), e.rayy[:B, :m], 0.).clamp(min=0)
        if self.fast:
            phi = c0.new_zeros(B)
            live_ray = (ray != 0).any(1)
            if bool(live_ray.any()):
                phi[live_ray] = self.certify(torch.zeros_like(c[live_ray]), torch.zeros_like(c0[live_ray]),
                    select_rows(G, live_ray), h[live_ray], lower[live_ray], upper[live_ray], ray[live_ray])
        else:
            phi = self.certify(torch.zeros_like(c), torch.zeros_like(c0), G, h, lower, upper, ray)
        if self.fast and m:
            row_value = torch.minimum(G*lower[:, None], G*upper[:, None]).sum(2)-h
            shortlist = row_value.amax(1) > 0
            if bool(shortlist.any()):
                candidate = torch.nn.functional.one_hot(row_value[shortlist].argmax(1), m).to(h)
                single = self.certify(torch.zeros_like(c[shortlist]), torch.zeros_like(c0[shortlist]),
                    select_rows(G, shortlist), h[shortlist], lower[shortlist], upper[shortlist], candidate)
                phi[shortlist] = torch.maximum(phi[shortlist], single)
        value = torch.where(phi > 0, INF, value)
        point = e.xw[:B] if not self.adaptive else e.xw[:B, inverse]
        point = torch.where(torch.isfinite(point), point, (lower+upper)/2)
        point = torch.minimum(torch.maximum(point, lower), upper)
        ids = e.out_ids[:B]
        if self.adaptive:
            box = (ids >= width) & (ids < width+2*n)
            ids = torch.where(box, width+(ids-width).div(n, rounding_mode='floor')*n
                              + order[(ids-width) % n], ids)
        packed = torch.cat((ids.to(best_y), best_y), 1)
        return value, best_y, point, packed

    @staticmethod
    def _frame_width(e, ids):
        k = int(torch.where(e.active, e.kcount(ids), 0).max())
        e.set_kap(min(e.cap, 1 << max(0, (max(1, k)-1).bit_length())))

    def _search(self, e, step, rounds, count, deadline=None):
        """Stop completed lanes in blocks; every returned mass is certified below.

        The inherited basis is repaired in setup on the current table. Most
        children therefore need only a few exchanges, rather than 128 repeats
        of an already optimal basis. The status here only schedules proposals.
        """
        previous = e.best.clone()
        exchanges = e.stats['exchanges'].clone()
        stagnant = torch.zeros_like(e.best, dtype=torch.long)
        for begin in range(0, rounds, Q):
            if deadline is not None and time.time() >= deadline:
                break
            block = min(Q, rounds-begin)
            # A single difficult lane need not keep refactoring 31 completed
            # bases. Reuse a smaller captured bucket, preserving the complete
            # phase-one/basis state rather than restarting its LP epoch.
            ids = torch.nonzero(e.active & ~e.closed).flatten()
            if not len(ids):
                break
            size = 1 << max(0, (len(ids)-1).bit_length())
            smaller = self.buckets.get((size, *e._plane_key[1:])) if size < e.B else None
            if smaller is not None and (not e.c.is_cuda or not self.capture or smaller.graphs is not None):
                self._copy_lanes(e, smaller, ids)
                run = smaller.graphs['round'].replay if e.c.is_cuda and self.capture else smaller.round
            else:
                smaller, run = None, step
            target = smaller if smaller is not None else e
            if self.adaptive:
                # At most Q new nonlocal rows enter per round. Size this
                # block's Schur workspace from the actual inherited bases,
                # retaining room for every exchange in the block.
                needed = int(target.kcount(target.ids).max()) + Q*block
                kap = min(target.cap, 1 << max(0, (max(1, needed)-1).bit_length()))
                target.set_kap(kap)
                if target.c.is_cuda and self.capture:
                    key = ('adaptive_round', kap)
                    if key not in target.graphs:
                        tensors = [v for v in vars(target).values() if torch.is_tensor(v)]
                        tensors += list(target.stats.values())
                        saved = [v.clone() for v in tensors]
                        try:
                            if target.dev not in self.plane_pools:
                                self.plane_pools[target.dev] = torch.cuda.graph_pool_handle()
                            elif not any(bucket.graphs and any(isinstance(name, tuple)
                                    and name[0] == 'adaptive_round' for name in bucket.graphs)
                                    for bucket in self.buckets.values() if bucket.dev == target.dev):
                                # A pool handle outlives its allocator only
                                # while a captured graph still owns the pool.
                                # Memory admission can evict every bucket.
                                self.plane_pools[target.dev] = torch.cuda.graph_pool_handle()
                            # Rounds never return private-pool tensors and
                            # replay sequentially. Share their scratch across
                            # widths AND lane buckets instead of retaining a
                            # new allocator pool for every captured shape.
                            target.graphs[key] = _Graph(target.round, self.plane_pools[target.dev])
                        finally:
                            for tensor, old in zip(tensors, saved):
                                tensor.copy_(old)
                    run = target.graphs[key].replay
                else:
                    run = target.round
            for _ in range(block):
                run()
            if self.adaptive:
                target.set_kap(target.cap)
            if smaller is not None:
                self._copy_lanes(e, smaller, ids, restore=True)
            self.work['rounds'] += block
            # Degenerate exchanges can make real progress through a flat dual
            # objective. Retain their budget; only a stationary basis stalls.
            improved = (e.best > previous) | (e.stats['exchanges'] > exchanges)
            stagnant = torch.where(improved | e.aux, 0, stagnant+block)
            previous = e.best.clone()
            exchanges = e.stats['exchanges'].clone()
            # One failed repair may recover at the sign basis. Repeating the
            # same failed restart twice cannot justify a proof and need not
            # consume every lane's remaining proposal budget.
            done = e.optimal | e.closed | (e.stats['fallbacks'] >= 2) | (stagnant >= 2*Q)
            e.closed.logical_or_(done)
            if bool(done[:count].all()):
                break

    @staticmethod
    def _copy_lanes(full, small, ids, restore=False):
        """Transfer all per-lane buffers; constants and graph pointers stay put."""
        for name, tensor in vars(full).items():
            if (name.startswith('_') or name in ('local', 'kind') or not torch.is_tensor(tensor)
                    or not tensor.ndim or tensor.shape[0] != full.B):
                continue
            other = getattr(small, name)
            if restore:
                tensor.index_copy_(0, ids, other[:len(ids)])
            else:
                other[:len(ids)].copy_(select_rows(tensor, ids))
                other[len(ids):].copy_(other[:1])
        for name, tensor in full.stats.items():
            if restore:
                tensor.index_copy_(0, ids, small.stats[name][:len(ids)])
            else:
                small.stats[name][:len(ids)].copy_(tensor[ids])
        if not restore:
            small.active[len(ids):] = False
            small.closed[len(ids):] = True


class _Eager:
    def __init__(self, fn):
        self.fn = fn

    def replay(self):
        return self.fn()


class _Graph:
    """compiled_crown.Captured with a memory pool shared by a bucket's graphs: they replay one after another and
    keep their results in the engine's own buffers, so one pool holds the temporaries of all of them (separate pools
    held 23.5 GB for wastewater13m1's 256 lanes)"""

    def __init__(self, fn, pool, warmup=3):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        self.out = capture(self.graph, fn, pool=pool)

    def replay(self):
        self.graph.replay()
        return self.out


class _Bucket:
    """one engine of B lanes and its captured graphs (setup, dictionary, round, finish)"""

    def __init__(self, owner, B, dtype):
        self.B, self.dtype, self.eng, self.graphs = B, dtype, None, None
        if owner.memory is None:
            self.materialize(owner)

    def materialize(self, owner):
        o = owner
        B, dtype = self.B, self.dtype
        kw = dict(cap=o.cap, device=o.dev, dtype=dtype, trial=o.trial, bland=o.bland, perturb=o.perturb,
                  kernel='lapack' if (o.cap or 0) > 64 else None,
                  k_update=o.k_update, k_rebuild=o.k_rebuild, structure=o.structure, scc=o.scc,
                  subopt=o.subopt, strict_precision=o.strict_precision)    # (the GJ kernel: k <= 64)
        if o.sparse is not None:                    # (the sparse engine: the compiled form's nonzeros)
            from .lockstep_sparse import SparseLockstepDual
            cg, face_flat = o.sparse
            self.eng = e = SparseLockstepDual(cg, face_flat, o.kind, o.certifier(B), B, **kw)
        else:
            self.eng = e = LockstepDual(o.m, o.n, o.levels, o.kind, o.certifier(B), B, **kw)
        e.bind(o.ext)
        self.graphs = None

    def wrap(self, owner, fn):
        if not owner.capture_enabled:
            return _Eager(fn)
        if owner.memory is not None and owner.memory.available() < 2 * self.B * owner.scratch_bytes:
            return _Eager(fn)
        try:
            graph = _Graph(fn, self.pool)
        except RuntimeError as error:
            if memory_error(error) or 'previous error during capture' in str(error):
                from .cuda_memory import MemoryBudgetExceeded
                raise MemoryBudgetExceeded(f'capture unavailable: {error}') from None
            raise
        if owner.memory is not None:
            owner.memory.require(0)
        return graph

    def dict_check(self, owner):
        """the dictionary's certificates (capturable): every entry's D / Phi in the search's dtype (one batched
        product with the dense rows), then the rigorous certificate of each lane's best bound entry and best ray"""
        e, K = self.eng, owner.K
        if not K:
            return
        Y, w = owner.dictm.to(e.dt), owner.dictw.to(e.dt)
        D = e.dict_values(Y, w)
        v = owner.dictv[None]
        iy = torch.where(v & (owner.dictw[None] > 0), D, -INF).argmax(1)
        ir = torch.where(v & (owner.dictw[None] == 0), D, -INF).argmax(1)
        vals = e.cert(torch.stack([owner.dictm[iy], owner.dictm[ir]], 1), owner.cw2)
        ok = torch.stack([owner.dictv[iy], owner.dictv[ir]], 1) & torch.isfinite(vals)
        dbest = torch.where(ok[:, 0], vals[:, 0], -INF)
        dfar = ok[:, 1] & (vals[:, 1] > 0)
        hit = e.active & ~e.closed & ((dbest >= e.T) | dfar)
        if e.strict_precision:
            up = e.active & (dbest > e.best)
            e.besty.copy_(torch.where(up[:, None], owner.dictm[iy], e.besty))
        e.best.copy_(torch.where(e.active, torch.maximum(e.best, dbest), e.best))
        e.closed.copy_(e.closed | hit)
        # a Farkas entry's closing is a ray (kind 2: the box holds no point), not a bound: reported as a bound it
        # left the domain open at its weak bound and never searched again (cute_optprloc after a root Farkas)
        far = e.active & ~e.ray & dfar
        e.ray.copy_(e.ray | far)
        e.rayy.copy_(torch.where(far[:, None], owner.dictm[ir].to(e.rayy.dtype), e.rayy))

    def capture(self, owner):
        if owner.budget is not None:
            return self.capture_budgeted(owner)
        wrap = fuse if owner.fused else (lambda f: f)
        e = self.eng
        self.pool = torch.cuda.graph_pool_handle() if owner.capture_enabled else None
        # Cheap bases keep the exact legacy round, including adaptive widths.
        # Capture expensive structured rounds only after their cost gate opens.
        self.optional = (e.structure, e.subopt)
        if owner.structure or owner.subopt:
            e.structure = e.subopt = False
        round_graph = lambda: (UpdatedRound(e, lambda fn: self.wrap(owner, wrap(fn))) if e.k_update else
                               self.wrap(owner, wrap(e.round)))
        self.graphs = {'setup': self.wrap(owner, wrap(e.setup)),
                       'dict': self.wrap(owner, wrap(lambda: self.dict_check(owner))),
                       'round': round_graph(), 'finish': self.wrap(owner, wrap(e.finish))}
        # adaptive K: a round per K width (the Schur kernel's LU / inverse costs its width cubed: padded to the
        # capacity, foulds4pq's lanes ran 16x slower at cap 256 than at 64)
        self.kaps = []
        if owner.kadapt:
            full = e.kap
            w = 16
            while w < full:
                e.set_kap(w)
                self.graphs[('round', w)] = round_graph()
                self.kaps.append(w)
                w *= 2
            e.set_kap(full)
        self.kaps.append(e.kap)

    def capture_budgeted(self, owner):
        """Resume cold capture by graph/shape; every warmup is charged too."""
        e, budget = self.eng, owner.budget
        wrap = fuse if owner.fused else (lambda f: f)
        if self.graphs is None:
            self.pool = torch.cuda.graph_pool_handle() if owner.capture_enabled else None
            self.graphs = {}
            self.optional = (e.structure, e.subopt)
            if owner.structure or owner.subopt:
                e.structure = e.subopt = False
            full = e.kap
            self.kaps = []
            w = 16
            while owner.kadapt and w < full:
                self.kaps.append(w)
                w *= 2
            self.kaps.append(full)
        specs = [('setup', e.setup), ('dict', lambda: self.dict_check(owner)), ('finish', e.finish)]
        specs += [(('round', w) if w != self.kaps[-1] else 'round', e.round) for w in self.kaps]
        for name, fn in specs:
            if name in self.graphs:
                continue
            key = (self.B, e.dt, name)
            if not budget.allows('capture', key):
                e.set_kap(self.kaps[-1])
                return False
            if name == 'round' or isinstance(name, tuple):
                e.set_kap(name[1] if isinstance(name, tuple) else self.kaps[-1])
            with budget.measure('capture', key):
                self.graphs[name] = (UpdatedRound(e, lambda f: self.wrap(owner, wrap(f)))
                                     if (name == 'round' or isinstance(name, tuple)) and e.k_update
                                     else self.wrap(owner, wrap(fn)))
        e.set_kap(self.kaps[-1])
        return True

    def optional_graph(self, owner, name):
        key = ('optional', name)
        if key not in self.graphs:
            e = self.eng
            # Graph warmups execute real exchanges. Preserve the complete lane
            # state before a lazy capture in the middle of a search epoch.
            tensors = [v for v in vars(e).values() if torch.is_tensor(v)] + list(e.stats.values())
            saved = [v.clone() for v in tensors]
            e.structure, e.subopt = self.optional
            kap = e.kap
            e.set_kap(self.kaps[-1])
            try:
                wrap = fuse if owner.fused else (lambda f: f)
                fn = e.final_masses if name == 'yfin' else getattr(e, name)
                self.graphs[key] = (UpdatedRound(e, lambda f: self.wrap(owner, wrap(f)))
                                    if name == 'round' and e.k_update else self.wrap(owner, wrap(fn)))
            finally:
                e.structure = e.subopt = False
                e.set_kap(kap)
                for v, old in zip(tensors, saved):
                    v.copy_(old)
        return self.graphs[key]

    def yfin_graph(self, owner):
        if 'yfin' not in self.graphs:
            self.graphs['yfin'] = self.wrap(owner, (fuse if owner.fused else (lambda f: f))(self.eng.final_masses))
        return self.graphs['yfin']


class LockstepBounding:
    def __init__(self, m, n, levels, kind, ext, certifier, R=8, buckets=(64, 256, 1024), dtype=torch.float32,
                 dict_K=64, trial='exchange', bland='fable', fused=True, device='cuda', buckets64=None, R_max=None,
                 cap=None, perturb=0.0, sparse=None, kadapt=False, near_cap=None, k_update=False, k_rebuild=32,
                 structure=False, scc=False, subopt=False, strict_precision=False, memory=None,
                 scratch_bytes=0, cache_buckets=False):
        self.memory, self.scratch_bytes = memory, scratch_bytes
        self.cache_buckets = cache_buckets
        self.memory_blocked = False
        self.capture_enabled = torch.device(device).type == 'cuda'
        self.m, self.n, self.levels, self.kind, self.ext = m, n, levels, kind, ext
        self.certifier, self.R, self.trial, self.bland, self.fused = certifier, R, trial, bland, fused
        # R_max > R: the rounds go on in blocks of R while any lane is still searching (open, not optimal, not sent
        # to fp64), up to R_max (one host synchronisation per block); default R_max = R (exactly R rounds)
        self.R_max = R if R_max is None else max(R, R_max)
        self.cap = cap                   # the Schur capacity (LockstepDual's; None: its default min(64, n - 1))
        self.k_update, self.k_rebuild = bool(k_update), int(k_rebuild)
        self.structure, self.scc, self.subopt = bool(structure), bool(scc), bool(subopt)
        self.strict_precision = bool(strict_precision)
        self.work_stats = collections.Counter()
        self.last_reasons = None
        self.k_stats = dict(updates=0, rebuilds=0, residual=0, lu=0) if self.k_update else None
        self.want_yfin = False           # also the last bases' masses of the lanes' own objectives (OBBT lanes)
        self.kadapt = kadapt
        self.optional_threshold = None
        if self.structure or self.subopt:
            # Compare the two n*k^2 Schur assembly products with the bounded
            # sparse peel/direction work (16 layers, 32 RHS). The guard switches
            # before an ordinary Q-row exchange can fill the legacy kernel.
            entries = n*n
            if sparse is not None:
                cg = sparse[0]
                degree = int(torch.bincount(cg._op()[0] % cg.Et, minlength=cg.Et).max()) if cg.Et else 0
                entries = n*(degree+2) + len(cg._cm()[0]) + int((cg.E != 0).sum())
            self.optional_threshold = max(1, min((self.cap or min(64, n-1))-Q,
                                                  math.ceil(math.sqrt(16*32*entries/(2*n)))))
        self.adaptive64 = False          # fp64 first once most fp32 lanes go to fp64 (measured: foulds4pq 511
                                         # lanes in 80 s against 3327 in 61 s, no lane optimal either way: off)
        self.n32 = self.n64 = self.n_direct = 0
        self.near_cap = near_cap         # fp32 lanes with this many Schur slots go to fp64 (None: never)
        self.deadline = float('inf')     # no further block of rounds after this time (time.time())
        self.perturb = perturb           # the search's cost perturbation (LockstepDual's; 0: off)
        self.sparse = sparse             # (CompiledGraph, face_flat): SparseLockstepDual lanes (no dense tables)
        self.rounds_run = []
        self.dev = torch.device(device)
        self.K = dict_K
        K2 = max(1, 2 * dict_K)
        self.dictm = torch.zeros(K2, m, dtype=torch.float64, device=self.dev)       # [K bound y | K Farkas rays]
        self.dictw = torch.cat([torch.ones(max(1, dict_K)), torch.zeros(max(1, dict_K))])[:K2].double().to(self.dev)
        self.dictv = torch.zeros(K2, dtype=torch.bool, device=self.dev)
        self.cw2 = torch.tensor([1.0, 0.0], dtype=torch.float64, device=self.dev)
        self.buckets = [_Bucket(self, b, dtype) for b in sorted(buckets)]
        self.fp32 = dtype == torch.float32
        b64 = buckets if buckets64 is None else buckets64
        self.buckets64 = [_Bucket(self, b, torch.float64) for b in sorted(b64)] if self.fp32 else []
        self.captured = False
        self.budget = None
        self.phase = 'rounds'
        self.fp64_retries = 0
        self.last_processed = None

    def _load(self, bk, tables, state, sel, T):
        e, b = bk.eng, len(sel)
        c = tables['c'][sel] if 'c' in tables else self._c.expand(b, -1)                # (per-lane objectives)
        c0 = tables['c0'][sel] if 'c0' in tables else torch.full((b,), self._c0)
        e.load(0, select_rows(tables['R'], sel) if 'R' in tables else None, tables['h'][sel] if 'h' in tables else None, c, c0,
               tables['l'][sel], tables['u'][sel], tables['dlow'][sel], tables['dup'][sel],
               T[sel] if torch.is_tensor(T) else torch.full((b,), T),
               pids=None if state is None else state.ids[sel], pbest=None if state is None else state.best[sel],
               py=None if state is None or state.y is None else state.y[sel])
        thr = tables['thr'][sel] if 'thr' in tables else self._thr.expand(b, -1)
        opt = {a: tables[k][sel] for k, a in (('hb', 'h'), ('undefined', 'undefined'), ('rowmask', 'rowmask'))
               if k in tables}
        e.cert.load(0, *(select_rows(tables[k], sel) for k in ('lw', 'lb', 'uw', 'ub', 'l', 'u')), thr, **opt)
        e.deactivate(b)

    def _admit_bucket(self, bk):
        try:
            self.memory.require(bk.B * self.scratch_bytes)
            return
        except RuntimeError as error:
            if not memory_error(error):
                raise
        # Residency and graph private pools are charged by the same budget as
        # workspaces. Evict only when that admission actually needs the space.
        for old in self.buckets + self.buckets64:
            if old is not bk:
                old.graphs, old.eng = None, None
        release_cuda(self.dev)
        self.memory.require(bk.B * self.scratch_bytes)

    def _run(self, bk, tables, state, sel, T):
        if self.memory is None:
            return self._run_bucket(bk, tables, state, sel, T)
        failed = False
        try:
            if bk.eng is None:
                if self.cache_buckets:
                    self._admit_bucket(bk)
                else:
                    # Legacy A76 policy: one precision/shape resident.
                    for old in self.buckets + self.buckets64:
                        if old is not bk:
                            old.graphs, old.eng = None, None
                    release_cuda(self.dev)
                    self.memory.require(bk.B * self.scratch_bytes)
                bk.materialize(self)
            if self.cache_buckets:
                self._admit_bucket(bk)
            else:
                self.memory.require(bk.B * self.scratch_bytes)
            return self._run_bucket(bk, tables, state, sel, T)
        except Exception as error:
            if not memory_error(error) and not cuda_launch_error(error):
                raise
            print(f'[crown-memory] lane/capture fallback B={bk.B} dtype={bk.dtype}: {error}', flush=True)
            failed = True
        # Leave the exception scope before freeing: tracebacks retain tensors.
        if failed:
            bk.graphs, bk.eng = None, None
            self.capture_enabled = False
            release_cuda(self.dev)
            if bk.B > 1:
                width = max(1, bk.B // 2)
                smaller = _Bucket(self, width, bk.dtype)
                choices = self.buckets64 if bk.dtype == torch.float64 and self.fp32 else self.buckets
                choices.insert(0, smaller)
                choices.sort(key=lambda b: b.B)
                parts = [self._run(smaller, tables, state, sel[i:i+width], T) for i in range(0, len(sel), width)]
                return {k: torch.cat([part[k] for part in parts]) for k in parts[0]}
            self.memory_blocked = True
            return self._skipped(tables, state, sel)

    def _run_bucket(self, bk, tables, state, sel, T):
        """one chunk (domains sel) on bucket bk; returns the lane results"""
        e, b = bk.eng, len(sel)
        budget = self.budget
        if budget is not None and not budget.allows(self.phase, (bk.B, e.dt)):
            return self._skipped(tables, state, sel)
        self._load(bk, tables, state, sel, T)
        if budget is not None:
            if not bk.capture(self):
                return self._skipped(tables, state, sel)
            self._load(bk, tables, state, sel, T)
        elif bk.graphs is None:            # captured at first use (the warm-up runs change the state: load again)
            bk.capture(self)
            self._load(bk, tables, state, sel, T)
        g = bk.graphs
        optional = self.optional_threshold is not None
        warm_large = optional and bool((e.has_pids & (e.kcount(e.pids) >= self.optional_threshold)).any())
        with budget.measure('setup', (bk.B, e.dt)) if budget is not None else nullcontext():
            (bk.optional_graph(self, 'setup') if warm_large else g['setup']).replay()
        if budget is None or budget.allows('dict', (bk.B, e.dt)):
            with budget.measure('dict', (bk.B, e.dt)) if budget is not None else nullcontext():
                g['dict'].replay()
        closed_free = e.closed[:b].clone()
        rounds = 0
        while budget is None or budget.allows(self.phase, (bk.B, e.dt)):
            # A graph replay is indivisible. Admission uses the last measured block.
            tick = time.perf_counter()
            # blocks of R rounds until no lane is still searching, or R_max rounds
            for _ in range(self.R):
                if optional:
                    live = e.active & ~e.closed & ~e.want64 & ~e.stalled
                    kmax = int(torch.where(live, e.kcount(e.ids), 0).max())
                    if kmax >= self.optional_threshold:
                        bk.optional_graph(self, 'round').replay()
                        continue
                if len(bk.kaps) > 1:     # the narrowest K that holds every searching lane's slots
                    live = e.active & ~e.closed
                    if optional and self.strict_precision:
                        live = live & ~e.want64 & ~e.stalled
                    kmax = int(torch.where(live, e.kcount(e.ids), 0).max())
                    if self.near_cap is not None and e.dt == torch.float32:
                        # lanes about to fill the Schur capacity: to fp64 now, not after their failures
                        risky = live & (e.kcount(e.ids) >= self.near_cap) & ~e.want64
                        if bool(risky.any()):
                            e.want64.copy_(e.want64 | risky)
                            e.active.copy_(e.active & ~risky)
                    kb = next(w for w in bk.kaps if w >= kmax)
                    (g[('round', kb)] if kb != bk.kaps[-1] else g['round']).replay()
                else:
                    g['round'].replay()
            if budget is not None:
                budget.sync()
                elapsed = time.perf_counter() - tick
                budget.spent[self.phase] += elapsed
                key = (self.phase, (bk.B, e.dt))
                budget.estimates[key] = max(elapsed, budget.estimates.get(key, 0.))
            rounds += self.R
            if rounds >= self.R_max or time.time() > self.deadline or \
                    not bool((e.active & ~e.closed & ~e.optimal & ~e.want64 &
                              (~e.stalled if self.strict_precision else True)).any()):
                break
        self.rounds_run.append(rounds)
        if self.k_update:
            # UpdatedRound selects its width eagerly. Auxiliary graphs always
            # factor at full capacity, including the first lazy yfin capture.
            e.set_kap(bk.kaps[-1])
        finish_large = optional and bool((e.active & (e.kcount(e.ids) >= self.optional_threshold)).any())
        if self.want_yfin:
            if budget is None:
                (bk.optional_graph(self, 'yfin') if finish_large else bk.yfin_graph(self)).replay()
            else:
                key = (bk.B, e.dt, 'yfin')
                if 'yfin' not in g and budget.allows('capture', key):
                    with budget.measure('capture', key):
                        bk.yfin_graph(self)
                if 'yfin' in g:
                    with budget.measure('finish', (bk.B, e.dt, 'yfin')):
                        g['yfin'].replay()
                else:
                    e.yfin.zero_()
        with budget.measure('finish', (bk.B, e.dt)) if budget is not None else nullcontext():
            (bk.optional_graph(self, 'finish') if finish_large else g['finish']).replay()
        if self.structure or self.subopt or self.strict_precision:
            for name, values in e.stats.items():
                self.work_stats[name] += int(values[:b].sum())
        if self.k_update:
            for name in self.k_stats:
                self.k_stats[name] += int(e.stats['k_' + name].sum())
        return dict(lower=e.best[:b].clone(), kind=e.kind_out[:b].clone(), ids=e.out_ids[:b].clone(),
                    y=e.besty[:b].clone(), ray=e.rayy[:b].clone(), witness=e.xw[:b].double(),
                    optimal=e.optimal[:b].clone(), numerical=e.numerical[:b].clone(), want64=e.want64[:b].clone(),
                    free=closed_free, **(dict(processed=torch.ones(b, dtype=torch.bool, device=self.dev))
                                         if budget is not None or self.memory is not None else {}), **(dict(stalled=e.stalled[:b].clone(), resource=e.resource[:b].clone())
                                         if self.strict_precision else {}),
                    **({'yfin': e.yfin[:b].clone()} if self.want_yfin else {}))

    def _skipped(self, tables, state, sel):
        b = len(sel)
        z = torch.zeros(b, dtype=torch.bool, device=self.dev)
        best = tables['l'].new_full((b,), -INF) if state is None else state.best[sel].clone()
        ids = torch.full((b, self.n), -1, dtype=torch.long, device=self.dev) if state is None else state.ids[sel].clone()
        y = tables['l'].new_zeros(b, self.m) if state is None or state.y is None else state.y[sel].clone()
        return dict(lower=best, kind=torch.zeros(b, dtype=torch.long, device=self.dev), ids=ids, y=y,
                    ray=torch.zeros_like(y), witness=tables['l'].new_full((b, self.n), float('nan')),
                    optimal=z.clone(), numerical=z.clone(), want64=z.clone(), free=z.clone(), processed=z.clone(),
                    **(dict(stalled=z.clone(), resource=z.clone()) if self.strict_precision else {}),
                    **(dict(yfin=torch.zeros_like(y)) if self.want_yfin else {}))

    def bound(self, tables, state, T, c, c0, thr=None):
        """the bounding call (see the module doc). c (n,), c0: the objective; thr (S,): the certificate's
        thresholds (CompiledCertifier's rows), or per domain tables['thr'] (D, S); optional per domain tables['hb']
        (D, R) the base rows' right-hand sides, ['undefined'] (D,), ['rowmask'] (D, m) (CompiledCertifier.load); T a
        float or (D,)"""
        self._c, self._c0, self._thr = c, float(c0), thr
        D = tables['l'].shape[0]
        top = self.buckets[-1].B
        parts = []
        # fp64 first: when most fp32 lanes of the last calls went to fp64 anyway (their redo doubled the cost),
        # the lanes start in fp64 (the measured share, not a per-lane failure)
        direct64 = self.fp32 and self.adaptive64 and self.n32 >= 256 and self.n64 > self.n32 / 2
        for c_ in range(0, D, top):
            sel = torch.arange(c_, min(D, c_ + top), device=self.dev)
            if direct64:
                r = None
                top64 = self.buckets64[-1].B
                for c2 in range(0, len(sel), top64):
                    rs = sel[c2:c2 + top64]
                    r64 = self._run(next(x for x in self.buckets64 if x.B >= len(rs)), tables, state, rs, T)
                    r64['want64'] = torch.ones_like(r64['want64'])
                    r = r64 if r is None else {k: torch.cat([r[k], r64[k]]) for k in r}
                self.n_direct += len(sel)
                parts.append(r)
                continue
            bk = next(x for x in self.buckets if x.B >= len(sel))
            r = self._run(bk, tables, state, sel, T)
            self.n32 += len(sel)
            self.n64 += int(r['want64'].sum())
            if self.n32 > 4096:          # (a running share: the last few thousand lanes)
                self.n32, self.n64 = self.n32 // 2, self.n64 // 2
            if self.fp32 and bool(r['want64'].any()):
                where64 = torch.nonzero(r['want64']).flatten()     # (before the redo overwrites want64)
                redo = sel[where64]
                top64 = self.buckets64[-1].B
                for c2 in range(0, len(redo), top64):
                    rs = redo[c2:c2 + top64]
                    b64 = next(x for x in self.buckets64 if x.B >= len(rs))
                    w = where64[c2:c2 + top64]
                    if self.budget is not None and not self.budget.allows(self.phase, (b64.B, b64.dtype)):
                        continue
                    if self.strict_precision:
                        # Continue the failed search from its current labels
                        # on ORIGINAL fp64 data, rather than repeat the parent
                        # (often the cold sign basis) and lose its whole prefix.
                        current = DomainState(r['ids'][w], r['lower'][w], r['y'][w])
                        sub = {key: value[rs] for key, value in tables.items()}
                        threshold = T[rs] if torch.is_tensor(T) else T
                        r64 = self._run(b64, sub, current, torch.arange(len(rs), device=self.dev), threshold)
                    else:
                        r64 = self._run(b64, tables, state, rs, T)
                    self.fp64_retries += int(r64['processed'].sum()) if self.budget is not None else len(rs)
                    if self.strict_precision:
                        # Recovery must not erase a better independently
                        # certified fp32 proposal (or an existing ray).
                        better = r['lower'][w] > r64['lower']
                        r64['lower'] = torch.maximum(r64['lower'], r['lower'][w])
                        r64['y'] = torch.where(better[:, None], r['y'][w], r64['y'])
                        ray32 = r['kind'][w] == 2
                        r64['ray'] = torch.where(ray32[:, None], r['ray'][w], r64['ray'])
                        r64['kind'] = torch.where(ray32, 2, torch.where(r['kind'][w] == 1, 1, r64['kind']))
                        r64['want64'] = torch.ones_like(r64['want64'])
                    for k in r:
                        if k != 'free':
                            if self.budget is not None:
                                done = r64['processed']
                                r[k][w[done]] = r64[k][done].to(r[k].dtype)
                            else:
                                r[k][w] = r64[k].to(r[k].dtype)
            parts.append(r)
        cat = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        self.last_processed = cat.get('processed')
        if self.strict_precision:
            # 0 BUDGET, 1 CERTIFIED_THRESHOLD, 2 FARKAS, 3 STALLED,
            # 4 NUMERICAL, 5 RESOURCE. Public kind remains proof-only.
            self.last_reasons = torch.where(cat['resource'], 5, torch.where(cat['numerical'], 4,
                torch.where(cat['stalled'], 3, 0)))
            self.last_reasons = torch.where(cat['kind'] > 0, cat['kind'], self.last_reasons)
        if self.K:                       # new certificates of lanes closed by pivots enter the dictionary
            piv = ~cat['free'] & (cat['kind'] > 0)
            for i, (mask, src) in enumerate(((piv & (cat['kind'] == 1), cat['y']), (piv & (cat['kind'] == 2), cat['ray']))):
                new = src[mask][-self.K:]
                if i == 1:
                    new = new / new.abs().amax(1, keepdim=True).clamp(min=1e-300)
                q = new.shape[0]
                if q:
                    blk, vb = self.dictm[i * self.K:(i + 1) * self.K], self.dictv[i * self.K:(i + 1) * self.K]
                    blk.copy_(torch.cat([blk[q:], new.double()]))
                    vb.copy_(torch.cat([vb[q:], torch.ones(q, dtype=torch.bool, device=self.dev)]))
        return BoundResult(lower=cat['lower'], kind=cat['kind'], state=DomainState(cat['ids'], cat['lower'], cat['y']),
                           witness=cat['witness'], optimal=cat['optimal'], numerical=cat['numerical'], fp64=cat['want64'],
                           yfin=cat.get('yfin'))
