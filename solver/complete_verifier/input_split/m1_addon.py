"""Measured native/sparse roles; the legacy add-on remains the default."""
import time
import math
import json
from types import SimpleNamespace
import torch


def eligible(net, C, thresholds):
    """Compatibility entry point: structure does not decide node admission."""
    return True


class Addon:
    def __init__(self, engine, enabled, *, sigma_main=False):
        self.engine, self.enabled = engine, enabled
        self.sigma_main = sigma_main
        self.main = 'D'
        self.native_peak_bytes = 0
        self.native_measured = False
        self.sigma_bytes = getattr(engine, 'scratch_bytes', 0)
        self.native_rate = self.sigma_rate = 0.
        self.native_lanes = 1
        self.native_oom = False
        self.batches = self.raised = self.decisive = self.boxes = 0
        self.seconds = self.gain = 0.
        self.capacity = max(1, getattr(engine, 'top', 1))
        memory=getattr(engine,'memory',None)
        if (memory is not None and hasattr(memory,'available')
                and (not sigma_main or math.isfinite(memory.available()))):
            # engine.top also sizes the existing primal/root machinery. Node
            # lanes have their own width, derived from available owned memory.
            self.capacity=max(1,int(memory.available()//engine.scratch_bytes))
        self.lanes = 1 if enabled else 0
        # lanes is the lane budget L handed to NodePool.bound. A first or
        # re-enabled trial starts at the pool's measured width W*.
        self.fresh = bool(enabled)
        # Ski rental in seconds. An unpaid trial halves L and puts M1 to sleep until the native
        # bounding time accrued since it reaches that trial's own M1 seconds x 2^(f-1), f counting
        # consecutive unpaid trials: M1 never spends more than native has earned. A paying trial
        # resets f and doubles L. `enabled` is False exactly while M1 sleeps.
        # crown_engine_m1_trial_rental is this default (kept as an accepted no-op flag).
        self.unpaid = 0
        self.accrued = 0.
        self.need = 0. if enabled else math.inf  # constructed off: never wakes
        # Consecutive trials that did not run (memory pause, deadline) and the M3 turn counter.
        self.stalled = self.wait = 0
        self.clock = time.perf_counter
        # Termination progress (n10). Best-first BaB terminates when the frontier minimum D reaches the
        # cutoff. The clean native rate r_off is D's gap fraction per native second on batches whose
        # parents M1 did not touch (M1's lagged effects, shrinks and split points, are not native's).
        # A trial pays if the M1 batch cycle moves D faster per second than r_off. While native's
        # clean rate is zero it cannot terminate on its own, and the frontier decides instead.
        self.off_f = self.off_t = 0.
        self.r_off = None
        self.prev_ran = False
        # The first BaB batch is measured against the root's bound, which another procedure produced
        # (root CROWN, RootHybrid, alias floors): it is not a native step and never enters the window.
        self.first = True
        self.take_ref = False  # the next batch's native part is the reference, even after an M1 batch
        # Rate-crossing wake: the last unpaid trial's progress rate and seconds.
        self.last_r_on = 0.
        self.last_s = math.inf
        self.floor_lanes = 1
        # Coverage wake: boxes one call holds at the floor width, armed by an unpaid trial that
        # raised a bound while covering every open box.
        self.cover = 1
        self.cover_armed = False
        self.cover_chain = False  # armed by a trial that a coverage wake started: its wake waits for the rental
        self.question_used = False  # '?' is one-shot per run
        self.last_min = None

    @property
    def batch_hint(self):
        memory = getattr(self.engine, 'memory', None)
        room = memory.available() if memory is not None else math.inf
        pool = getattr(self.engine, 'node_pool', None)
        bank = getattr(getattr(pool, 'parents', None), 'data', None)
        # Match NodePool's current admission, including its ancestry workspace.
        per_box = self.sigma_bytes + (len(bank[0])*(3*self.engine.n+9) if bank is not None else 0)
        width = (int(room // per_box) if math.isfinite(room) else self.capacity)
        return 1 << max(0, max(1, width).bit_length()-1)

    def admitted(self, lower):
        """Memory-admitted lane width, in lanes: the pool's, else (no pool) the capacity."""
        width = getattr(getattr(self.engine, 'node_pool', None), 'admitted_lanes', None)
        if callable(width): width = width(lower)
        return max(1, int(width)) if width else self.capacity

    def m3_yield(self):
        """M3 propagates children without bounding them, so M1 never sees them (A161): it yields
        while M1 can use the batch. While M1 sleeps, or keeps not running (pause), a batch counts as
        native and M3 runs on all but every (2^(stalled-1)+1)-th one, so neither side starves."""
        if self.enabled and not self.stalled: return True
        self.wait += 1
        if self.wait > 1 << max(0, self.stalled-1):
            self.wait = 0
            return True
        return False

    def select_main(self):
        if self.sigma_main:
            memory = getattr(self.engine, 'memory', None)
            available = memory.available() if memory is not None else math.inf
            self.main = ('sigma' if self.native_oom or self.native_peak_bytes > available
                         or self.sigma_rate > self.native_rate else 'D')
        return self.main

    def measure_native(self, call, device):
        """One root domain, excluding the allocations already live on entry."""
        cuda = torch.device(device).type == 'cuda'
        if cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            before = torch.cuda.memory_allocated(device)
        started = time.perf_counter()
        result = call()
        if cuda:
            torch.cuda.synchronize(device)
            self.native_peak_bytes = torch.cuda.max_memory_allocated(device)-before
            self.native_measured = True
        return result, time.perf_counter()-started

    def report_role(self):
        print(f'[sigma-role] main={self.main} native_peak_bytes={self.native_peak_bytes} '
              f'native_measured={self.native_measured} sigma_per_box_bytes={self.sigma_bytes} '
              f'native_rate={self.native_rate:.9g} sigma_rate={self.sigma_rate:.9g}', flush=True)

    def measure_root(self, root, deadline):
        from auto_LiRPA.cuda_memory import memory_error
        from input_constraints import standard_form_to_input_constraints
        inherited = root.global_lb[:1].clone()
        thr = root.rowpool.thresholds(root.rhs[:1].double()).to(inherited)
        # The retained root plane is a lower bound on D's per-domain peak.
        if root.lA is not None:
            self.native_peak_bytes = root.lA[:1].numel()*root.lA.element_size()
        if self.select_main() == 'sigma':
            self.report_role()
            return
        try:
            result, seconds = self.measure_native(lambda: root.net.get_lower_bound_naive(
                dm_lb=inherited, dm_l=root.dm_l[:1], dm_u=root.dm_u[:1], history=root.history,
                bounding_method=root.bounding_method, C=root.c[:1], thresholds=root.rhs[:1],
                stop_criterion=root.stop_criterion, return_A=True, return_b=True,
                base_input_linear_constraints=standard_form_to_input_constraints(root.base_input_constraints), stats=root.stats),
                root.dm_l.device)
            self.native_rate = self.reward(inherited, result[0], thr)/seconds
        except Exception as error:
            if not memory_error(error): raise
            self.native_oom = True
        if self.native_oom:
            from auto_LiRPA.cuda_memory import release_cuda
            release_cuda(root.dm_l.device)
        else:
            self.bound_main(root.dm_l[:1], root.dm_u[:1], inherited, thr, deadline, root.history)
        self.select_main()
        self.report_role()

    @torch.no_grad()
    def bound_main(self, lower, upper, inherited, thresholds, deadline, history):
        """Sparse CROWN rows plus the current-table lane certificate; no dense A."""
        from nlp_solver.rigorous_dual import round_down
        from auto_LiRPA.cuda_memory import memory_error
        from contextlib import nullcontext
        e = self.engine
        old = (thresholds.new_full(thresholds.shape, -torch.inf) if inherited is None
               else inherited.clone())
        history.engine_witness = None
        history.engine_valid = torch.zeros(len(lower), dtype=torch.bool, device=lower.device)
        history.engine_box = None
        history.lifted_branch_score = history.lifted_split_point = None
        if deadline is not None and time.time() >= deadline:
            return old
        started = time.perf_counter()
        lo, hi = (t.flatten(1).to(e.dev, torch.float64).clone() for t in (lower, upper))
        result = old.clone()
        width = min(len(lower), self.batch_hint)
        try:
            with e.memory.allocations() if e.memory is not None else nullcontext():
                if e.memory is not None: e.memory.require(len(lo)*self.sigma_bytes)
                _, out, _, _ = e._bound_fbbt(lo, hi, slice(None), thresholds[:, 0].to(lo))
                rows = e.grow >= 0
                values = round_down(out[:, e.grow[rows]], result.dtype).to(result.device)
                result[:, rows] = torch.fmax(result[:, rows], values)
                # FBBT used the incumbent cutoff. It can certify that cutoff,
                # never a stronger objective value on the discarded region.
                result[:, 0] = torch.maximum(old[:, 0], torch.minimum(result[:, 0], thresholds[:, 0]))
                history.engine_box = (lo.reshape_as(lower), hi.reshape_as(upper))
                result = e.node_pool.bound(lo.reshape_as(lower), hi.reshape_as(upper), result,
                    thresholds, deadline, width, history)
        except Exception as error:
            if not (memory_error(error) or isinstance(error, ValueError) and str(error).startswith('strict ')):
                raise
            # Unsupported strict domains or temporary pressure add no proof.
            # The inherited cover is still exhaustive and may be split again.
            print(f'[sigma-main] inherited fallback: {error}', flush=True)
        elapsed = time.perf_counter()-started
        reward = self.reward(old, result, thresholds)
        self.sigma_rate = reward/elapsed if elapsed > 0 else 0.
        self.select_main()
        print(f'[sigma-main] boxes={len(lower)} reward_per_second={self.sigma_rate:.9g} '
              f'native_per_second={self.native_rate:.9g} lanes={width} next={self.main}', flush=True)
        return result

    def native_addon(self, call, sparse, inherited, thresholds):
        """A measured native prefix can earn the main role back; OOM disables it."""
        from auto_LiRPA.cuda_memory import memory_error, release_cuda
        memory = getattr(self.engine, 'memory', None)
        room = memory.available() if memory is not None else math.inf
        count = min(len(sparse), self.native_lanes)
        if self.native_peak_bytes:
            count = min(count, int(room//self.native_peak_bytes)) if math.isfinite(room) else count
        if self.native_oom or count < 1:
            return sparse
        started = time.perf_counter()
        try:
            native = call(count)[0]
            elapsed = time.perf_counter()-started
            old = inherited[:count] if inherited is not None else torch.full_like(native, -torch.inf)
            self.native_rate = self.reward(old, native, thresholds[:count])/elapsed
            reward = self.reward(sparse[:count], torch.fmax(sparse[:count], native), thresholds[:count])
            self.native_lanes = (min(self.capacity, 2*self.native_lanes)
                if reward > 0 and self.native_rate >= self.sigma_rate else self.native_lanes//2)
            sparse = sparse.clone()
            sparse[:count] = torch.fmax(sparse[:count], native)
        except Exception as error:
            if not memory_error(error): raise
            self.native_oom = True
        if self.native_oom: release_cuda(self.engine.dev)
        self.select_main()
        return sparse

    @staticmethod
    def reward(old, new, thresholds):
        # A gap fraction counts one box of progress; a newly pruned box counts
        # one decision. Both are dimensionless, with no objective-unit scale.
        finite = torch.isfinite(old[:, 0]) & torch.isfinite(thresholds[:, 0])
        decided = (old > thresholds).any(1) | (old[:, 0] >= thresholds[:, 0])
        gap = thresholds[:, 0]-old[:, 0]
        gain = (torch.minimum(new[:, 0], thresholds[:, 0])-old[:, 0]).clamp_min(0)
        use = finite & ~decided & (gap > 0)
        fraction = torch.where(use, gain/gap.clamp_min(torch.finfo(gap.dtype).tiny), 0.)
        # Without an incumbent use relative endpoint movement. Its L1
        # denominator is scale invariant and needs no objective-unit constant.
        no_cutoff=~decided & torch.isposinf(thresholds[:,0]) & torch.isfinite(old[:,0]) & torch.isfinite(new[:,0])
        relative=(new[:,0]-old[:,0]).clamp_min(0)/(old[:,0].abs()+new[:,0].abs()).clamp_min(torch.finfo(gap.dtype).tiny)
        fraction=torch.where(no_cutoff,relative,fraction)
        fraction=torch.where(~decided & torch.isneginf(old[:,0]) & torch.isfinite(new[:,0]),1.,fraction)
        pruned = ~decided & ((new>thresholds).any(1) | (new[:,0]>=thresholds[:,0]))
        return float(fraction.sum()+pruned.sum())

    def record_planes(self, planes, lane=None):
        if self.sigma_main and planes is not None:
            print('[sigma-ab] '+json.dumps(dict(plane=planes[0].tolist(),
                plane_exact=planes[1].tolist(), lane=([None]*len(planes[0]) if lane is None
                                                   else lane.tolist()))), flush=True)

    def _frontier_min(self, values, thresholds, floor):
        """(D, cutoff, scale) of one batch: D is the minimum objective bound over the boxes still open
        (the cutoff if none is), scale the cutoff gap above the parents' minimum `floor` (relative
        movement without an incumbent). Python floats; scheduling only."""
        decided = torch.isfinite(values) & torch.isfinite(thresholds) & (values > thresholds)
        decided[:, 0] |= (values[:, 0] >= thresholds[:, 0]) & torch.isfinite(thresholds[:, 0])
        open_ = ~decided.any(1)
        finite = thresholds[:, 0][torch.isfinite(thresholds[:, 0])]
        cut = float(finite.min()) if len(finite) else math.inf
        low = float(values[open_, 0].min()) if bool(open_.any()) else cut
        low = min(low, cut)
        if floor is None or not math.isfinite(floor):
            floor = low
        scale = cut-floor if math.isfinite(cut) and cut > floor else max(1., abs(floor))
        tau = max(getattr(self.engine, 'abs_tol', 0.), getattr(self.engine, 'rel_tol', 0.)*max(1., abs(cut) if math.isfinite(cut) else abs(floor)))
        return low, cut, floor, scale, tau, open_, decided

    def bound(self, lower, upper, gear, thresholds, deadline, inherited=None,
              native_seconds=None, history=None, planes=None):
        saved = dict(enabled=self.enabled, lanes=self.lanes, unpaid=self.unpaid,
                     accrued=self.accrued, need=self.need, fresh=self.fresh, cover_armed=self.cover_armed)
        # The parents' minimum (best-first pickout: the frontier minimum before this batch).
        floor = getattr(history, 'm1_floor', None)
        floor = float(floor) if floor is not None else self.last_min
        D_N, cut, floor, scale, tau, open_n, _ = self._frontier_min(gear, thresholds, floor)
        f_N = (D_N-floor)/scale if D_N-floor > tau else 0.
        if (not self.prev_ran or self.take_ref) and not self.first and native_seconds:
            self.off_f += f_N; self.off_t += native_seconds
        self.first = self.take_ref = False
        self.prev_ran = False
        self.last_min = D_N
        n_open = int(open_n.sum())
        covered = False  # this trial was started by a coverage wake
        if not self.enabled:
            self.accrued += native_seconds or 0.
            # The coverage look also needs a clean native batch since the last trial. A chained look (armed by
            # a coverage-woken trial) also waits for one trial's native seconds, like the rate crossing: each
            # look past the first is paid for by native time, so the rental bound holds (opus_review_n10, B2).
            woke = (self.cover_armed and n_open > self.cover and self.off_t > 0
                    and (not self.cover_chain or self.accrued >= self.last_s))
            # Native out-progressed M1 at the last trial; once native has had that trial's seconds and its
            # clean rate since has fallen below M1's measured rate, the reason for the sleep is gone.
            crossed = (self.off_t > 0 and self.accrued >= self.last_s
                       and self.off_f/self.off_t < self.last_r_on)
            if self.accrued < self.need and not woke and not crossed:
                self.record_planes(planes)
                return gear
            if woke:
                # The coverage wake is one look at the whole frontier: L holds every open box.
                self.cover_armed = False
                covered = True
                per_box = getattr(getattr(self.engine, 'node_pool', None), 'lanes_per_box', None) or 1
                self.lanes = max(self.lanes, min(self.admitted(lower), int(math.ceil(n_open*per_box))))
            self.enabled = True
        start = self.clock()
        setup = 0.
        state = SimpleNamespace(engine_state=None)
        # Local history: no states, witnesses or branch proposals enter GEAR.
        pool = getattr(self.engine, 'node_pool', None)
        if pool is None:
            candidate, _, _ = self.engine.bound(lower, upper, gear, thresholds, state, deadline)
        else:
            baseline = inherited if self.sigma_main and inherited is not None else gear
            if self.fresh:
                start_width = getattr(pool, 'lane_start', None)
                width = int(start_width(lower)) if start_width else self.capacity
                if width > 0:
                    # L never drops below W*: with t(W)=a+bW a call at L<=W* costs at most 2a, so halving
                    # below it saves no time and only loses coverage (opus_review_n9 C(a)).
                    self.lanes = self.floor_lanes = max(1, min(self.admitted(lower), width))
                    self.fresh = False
                setup = self.clock()-start  # one-time W* timing: charged to the rental, not to the rate
            candidate = baseline if self.fresh else pool.bound(
                lower, upper, baseline, thresholds, deadline, self.lanes, history)
        if pool is not None and (self.fresh or not getattr(pool, 'last_ran', True)):
            # No M1 work happened (memory pause, deadline before start): the trial is no evidence for
            # or against M1, so the controller keeps its state. A pause during the build or capture
            # of this configuration halves L, so that the same configuration is not retried forever.
            self.__dict__.update(saved)
            self.stalled += 1
            if getattr(pool, 'last_paused', False):
                self.lanes = max(1, self.lanes//2)
                # A call that ran and then paused spent real seconds: M1 sleeps until native seconds cover
                # them. A pause carries no evidence about M1, so it does not raise the unpaid exponent that
                # real trials double from (opus_review_n10b, 3). The call still returns the native bound.
                spent = self.clock()-start
                if spent > 0:
                    self.need = spent
                    self.accrued = 0.
                    self.last_r_on, self.last_s = 0., spent
                    self.cover_armed = False
                    self.enabled = False
            self.record_planes(planes)
            return gear
        proposal = torch.where(candidate[:,0].isnan(), -torch.inf, candidate[:,0])
        wins = proposal > gear[:,0]
        result = gear.clone(); result[:,0] = torch.maximum(gear[:,0],proposal)
        # GEAR discards a box if any row reaches its threshold. Credit M1
        # only for a newly decisive objective certificate on an otherwise open box.
        decided = torch.isfinite(gear) & torch.isfinite(thresholds) & (gear > thresholds)
        decided[:,0] |= (gear[:,0] >= thresholds[:,0]) & torch.isfinite(thresholds[:,0])
        open_gear = ~decided.any(1)
        decisive = int((open_gear & torch.isfinite(thresholds[:,0]) & (result[:,0] >= thresholds[:,0])).sum())
        still_open = open_gear & ~(torch.isfinite(thresholds[:,0]) & (result[:,0] >= thresholds[:,0]))
        # Certified probe bits on boxes that stay open, as box-equivalents of subtree removed: a box
        # shrunk by b bits keeps 2^-b of its volume (the dyadic subtree under it), a decided box none.
        bits = getattr(pool, 'last_bits', None)
        bits = (bits.to(still_open.device)[still_open] if bits is not None and len(bits) == len(lower)
                else torch.zeros(0, dtype=torch.float64))
        box_bits = float((1-torch.exp2(-bits.double())).sum()) if len(bits) else 0.
        bits = float(bits.sum()) if len(bits) else 0.
        credit = decisive + box_bits
        raised = int(wins.sum()); elapsed = self.clock()-start
        self.batches += 1; self.boxes += len(lower); self.raised += raised; self.decisive += decisive; self.seconds += elapsed
        finite = open_gear & torch.isfinite(candidate[:,0]) & torch.isfinite(gear[:,0])
        gain = float(torch.where(finite, (candidate[:,0]-gear[:,0]).clamp_min(0), 0.).sum())
        self.gain += gain
        reward = self.reward(gear, result, thresholds)
        native_decided = int(decided.any(1).sum())
        reward_rate = reward/elapsed if elapsed > 0 else 0.
        if self.sigma_main:
            self.sigma_rate = (self.reward(inherited, candidate, thresholds)/elapsed
                               if inherited is not None and elapsed > 0 else reward_rate)
            if native_seconds and inherited is not None:
                self.native_rate = self.reward(inherited, gear, thresholds)/native_seconds
            self.select_main()
        # Termination progress. D_M is the batch minimum after M1; M1's own move must clear the
        # solver's tolerance tau and lift D above what native and the parents already certified.
        D_M = self._frontier_min(result, thresholds, floor)[0]
        t_M = max(elapsed-setup, 0.)
        t_N = native_seconds or 0.
        move = D_M-max(D_N, floor)
        f_on = (D_M-floor)/scale if D_M-floor > tau else 0.
        if self.off_t > 0:
            self.r_off = self.off_f/self.off_t
            self.off_f = self.off_t = 0.
        ref = self.r_off is not None
        r_off = self.r_off or 0.
        r_on = f_on/(t_N+t_M) if t_N+t_M > 0 else 0.
        B = len(lower)
        verdict = '-'
        if ref and move > tau and r_on > r_off:
            verdict = 'G'
        elif r_off == 0.:
            # Native moves no frontier minimum: judge the generation. In the stationary branching model
            # a frontier that grows (decided < B/2 of the B children) has unbounded remaining work, so M1
            # pays iff its decisions and shrinks make this generation non-growing; if native already
            # shrinks it, compare net open-box drift per second.
            # M1 holds the lowest-bound (hardest) open boxes first; its decided-and-shrunk fraction q on
            # the boxes it held is extrapolated to every open box (a paid trial doubles L towards that).
            held = max(1, min(n_open, int((getattr(pool, 'metrics', None) or {}).get('nodes', n_open))))
            q = min(1., credit/held)
            if 2*native_decided < B <= 2*(native_decided+q*(B-native_decided)):
                verdict = 'W'
            elif 2*native_decided > B and t_M > 0 and t_N > 0 and credit/t_M >= (native_decided-B/2)/t_N:
                verdict = 'O'
        paid = verdict != '-'
        self.stalled = 0
        self.accrued = 0.
        # Without a reference, the next batch's native part becomes it when M1 stays on (paid: no batch
        # free of M1 follows). After '?' M1 sleeps, so the reference waits for the first batch whose parents
        # M1 did not touch: the batch right after M1 carries its shrinks, which are not native's progress.
        if not ref and paid: self.take_ref = True
        if not paid and not ref and raised > 0 and not self.question_used:
            # No native reference yet (only the root batch, whose parent bound another procedure made) and
            # M1 did lift bounds. The G comparison is deferred, not lost: M1 sleeps on the rental of this
            # trial, and wakes as soon as native's clean rate, measured from the next batch on, falls below
            # this trial's rate r_on, i.e. as soon as G would have paid it (no wait for the rental here:
            # there is at most one such trial per run).
            verdict = '?'
            self.question_used = True
            self.unpaid = 1
            self.need = elapsed
            self.last_r_on, self.last_s = r_on, 0.
            per_box = getattr(pool, 'lanes_per_box', None) or 1
            self.cover = max(1, int(self.floor_lanes//per_box))
            self.cover_armed = n_open <= self.cover
        elif paid:
            self.unpaid = 0
            self.lanes = min(self.admitted(lower), max(1, 2*self.lanes))
            self.need = 0.
        else:
            self.unpaid += 1
            self.lanes = max(self.floor_lanes, self.lanes//2)
            # Ski rental on all M1 seconds of the trial, setup included.
            self.need = elapsed*(1 << min(self.unpaid-1, 30))
            self.last_r_on, self.last_s = r_on, elapsed
            per_box = getattr(pool, 'lanes_per_box', None) or 1
            self.cover = max(1, int(self.floor_lanes//per_box))
            self.cover_armed = raised > 0 and n_open <= self.cover
            self.cover_chain = covered
        self.enabled = paid
        self.prev_ran = True
        self.last_min = D_M
        print(f'[m1-addon] batches={self.batches} boxes={len(lower)} raised={raised} '
              f'decisive={decisive} bits={bits:.6g} total_decisive={self.decisive} '
              f'native_decided={native_decided} open={n_open} credit={credit:.6g} '
              f'D0={floor:.10g} DN={D_N:.10g} DM={D_M:.10g} cut={cut:.10g} tau={tau:.3g} '
              f'r_on={r_on:.6g} r_off={r_off:.6g} tM={t_M:.4g} tN={t_N:.4g} setup={setup:.4g} '
              f'verdict={verdict} cover={self.cover} armed={self.cover_armed} need={self.need:.4g} '
              f'lanes={self.lanes} active={self.enabled}', flush=True)
        self.record_planes(planes, proposal)
        return result


def root_addon(root, deadline):
    cfg = root.nlp_config
    if cfg.get('bound_engine','gear') != 'gear':
        raise ValueError('M1/M2 add-ons require bound_engine=gear')
    from input_split.s12_structural import root_structural_addon
    if root_structural_addon(root, deadline): return
    thr = root.rowpool.thresholds(root.rhs[:1].double()).reshape(-1)
    # Native CROWN screens; every supported nonlinear table can earn node lanes.
    sigma = cfg.get('crown_engine_sigma_main',False)
    m1 = cfg.get('crown_engine_m1',False) or sigma
    if not (m1 or cfg.get('crown_engine_m2',False)): return
    from input_split.crown_engine import build_engine
    started = time.perf_counter()
    private = dict(cfg, crown_engine_m1=True, crown_engine_lanes=256, crown_engine_fbbt=2)
    engine = build_engine(root, cfg=private)
    if engine is None: return
    from input_split.lane_primal import LaneLocal
    root.net.lane_local = LaneLocal(root, engine)
    if m1:
        from input_split.node_pool import NodePool
        engine.node_pool = NodePool(engine)
        addon = Addon(engine, True, sigma_main=sigma)
        if sigma:
            addon.measure_root(root, deadline)
        root.net.m1_addon = addon
    if cfg.get('crown_engine_m2',False) and len(root.dm_l)==1:
        from input_split.m2_interleave import M2Interleave
        session = M2Interleave(engine,root.dm_l,root.dm_u)
        session.local = root.net.lane_local
        if engine.dev.type == 'cuda': torch.cuda.synchronize(engine.dev)
        session.m2_seconds = time.perf_counter()-started
        root.net.m2_interleave = session
    # M1 is measured on node batches after GEAR has produced their bounds.
    # M2 runs at committed tree boundaries with an equal measured time budget.
