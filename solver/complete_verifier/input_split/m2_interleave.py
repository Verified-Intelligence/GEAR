"""Root M2 and the native GEAR tree share time by measured marginal value (at most equal service time)."""
from contextlib import nullcontext
import json
import math
import time
import torch


class M2Interleave:
    def __init__(self, engine, lower, upper):
        self.engine = engine
        self.lower, self.upper = lower.clone(), upper.clone()
        self.unapplied = False  # M2's box is ahead of the frontier after a failed intersect
        self.tree_seconds = self.m2_seconds = 0.
        self.since = None
        self.tree_clock = None
        self.accruing = True
        self.disabled = False
        self.checked_storage = False
        self.expired = False
        # Stall backoff. One round is the unit of the restarted solver's cycle
        # and may only refresh its state, so a stall needs two in a row.
        self.stall_rounds = 2
        self.stalled = 0
        self.skip = 1          # tree boundaries to skip after the next stall; doubles
        self.skip_left = 0
        self.value = 0.        # M2 decisions: boxes dropped + certified root-bound steps
        self.frontier = None   # open boxes at the previous boundary
        self.tree_value = 0.   # net boxes the tree removed from the frontier

    def share_ok(self):
        """M2 time <= tree time * (M2 value rate / tree value rate), at most 1:1.

        A decision is a box dropped. The one-decision prior keeps a round with no
        measured value from blocking M2 forever; its weight fades with M2 time."""
        if self.tree_value <= 0 or self.m2_seconds <= 0: return True
        return self.m2_seconds*self.tree_value <= self.tree_seconds*(self.value+1.)

    def clock(self):
        if self.engine.dev.type == 'cuda': torch.cuda.synchronize(self.engine.dev)
        return time.perf_counter()

    def stopped(self):
        state = getattr(self.engine,'m2_state',None)
        return self.disabled or (state is not None and state['stop'] in
                                ('closed','fbbt_empty','undefined','global_deadline'))

    def check_deadline(self, deadline):
        self.expired = deadline is not None and time.time() >= deadline
        return self.expired

    def start(self):
        if not self.stopped():
            self.since = self.clock() if self.tree_clock is None else self.tree_clock()

    def ready(self, cutoff, tau):
        if self.disabled: return False
        state = getattr(self.engine,'m2_state',None)
        if getattr(self.engine,'m2_memory_need',None) is not None: return False
        if state is None: return True
        if state['stop'] in ('retired',):
            from input_split.m1_qp import m2_refresh
            return m2_refresh(state,cutoff,tau)
        return state['stop'] == 'yield'

    def release(self):
        """Evict optional workspaces/graphs; retain the certified box and caps."""
        from input_split.m2_async import release
        return release(self.engine)

    def pause_memory(self):
        from input_split.m1_qp import memory_pause
        e = self.engine
        lanes = getattr(e,'m2_lanes',256)
        self.disabled = memory_pause(e,'m2',lanes)
        if lanes > 1:
            e.m2_lanes = max(1,lanes//2)
            # The support kernel is lane-separable; admit a smaller wave even
            # when the failed full wave can never fit the configured cap.
            e.m2_memory_need = math.ceil(e.m2_memory_need*e.m2_lanes/lanes)

    def root_floor(self):
        """min(D_M2, T): the root value M2 has certified, capped at its cutoff T (-inf if none). Reporting only.
        T is also capped by the live table's cutoff: root_obbt lowers it before it writes state['cutoff'], and a
        caught memory error in between leaves the state cutoff stale, above the cutoff that certified batch.bound."""
        state = getattr(self.engine, 'm2_state', None)
        batch = None if state is None else state.get('batch')
        if batch is None:
            return -math.inf
        return min(float(batch.bound), float(state['cutoff']), float(batch.cutoff))

    def service(self, domains, primal, found, deadline):
        """At a committed tree boundary, run one round while tree cost >= M2 cost.

        All domain intersection and synchronization costs belong to M2.
        Return a cutoff certificate when M2 closes or the frontier empties.
        """
        if self.check_deadline(deadline):
            self.disabled = True
            self.engine.stats['m2_stop'] = 'global_deadline'
            self.release()
            return None
        if self.stopped(): return None
        from input_split.crown_engine import gap_cutoff
        from input_split.m1_qp import root_obbt, memory_ready
        from auto_LiRPA.cuda_memory import memory_error
        from tensor_storage import StackTensorStorage
        import arguments
        now = self.clock()
        if self.frontier is not None: self.tree_value += max(0,self.frontier-len(domains))
        tree_now = now if self.tree_clock is None else self.tree_clock()
        if self.since is not None and self.accruing: self.tree_seconds += max(0.,tree_now-self.since)
        round_seconds = 0.
        e = self.engine
        if not self.checked_storage:
            self.checked_storage = True
            self.disabled = (not callable(getattr(domains,'_per_domain_storages',None))
                             or domains.use_alpha or domains.use_beta or
                             not all(isinstance(st,StackTensorStorage) for st in domains._per_domain_storages()))
            if self.disabled: e.stats['m2_stop'] = 'unsupported_history'
        P = float(primal.reshape(-1)[0]) if found else math.inf
        certificate = None
        if math.isfinite(P):
            memory_ready(e,'m2')
            cutoff = gap_cutoff(P,e.abs_tol,e.rel_tol)
            tau = max(e.abs_tol,e.rel_tol*max(1.,abs(P)))
            state = getattr(e,'m2_state',None)
            if state is not None: cutoff = min(cutoff,state['cutoff'])
            backing_off = self.skip_left > 0
            if backing_off: self.skip_left -= 1
            while (not backing_off and self.skip_left == 0 and self.ready(cutoff,tau) and len(domains) and self.share_ok()
                   and self.tree_seconds >= self.m2_seconds
                   + (self.local.charge() if hasattr(self,'local') else 0.)
                   and not getattr(e,'local_waiting',False)
                   and not self.check_deadline(deadline)):
                tick = self.clock()
                floor_before = self.root_floor()
                width_before = (self.upper-self.lower).flatten()
                dropped = 0
                failed = undefined = False
                try:
                    memory = getattr(e,'memory',None)
                    with memory.allocations() if memory is not None else nullcontext():
                        if memory is not None:
                            memory.require(getattr(e,'m2_lanes',1)*getattr(e,'scratch_bytes',0))
                        lo,hi = root_obbt(e,self.lower,self.upper,cutoff,deadline,
                                         round_limit=1,refresh_tol=tau)
                    changed = self.unapplied or bool((lo > self.lower).any() or (hi < self.upper).any())
                    if changed and not getattr(e,'m2_closed',False):
                        cap = arguments.Globals.get('discarded_threshold_min')
                        arguments.Globals['discarded_threshold_min'] = cutoff if cap is None else min(cap,cutoff)
                        # M2 resumes from its own box (sub-box rule); a failed intersect is retried next service.
                        self.lower,self.upper,self.unapplied = lo,hi,True
                        dropped = domains.intersect_root(lo,hi)
                        self.unapplied = False
                        from nlp_solver.rigorous_dual import round_down
                        history = domains.history
                        box = getattr(history,'lifted_root_box',None)
                        hl,hu = lo.detach(),hi.detach()
                        if box is not None:
                            hl = torch.maximum(box[0],round_down(hl.to(box[0].device),box[0].dtype))
                            hu = torch.minimum(box[1],-round_down(-hu.to(box[1].device),box[1].dtype))
                        history.lifted_root_box = (hl.clone(),hu.clone())
                        width = (hu-hl).flatten()
                        old_width = getattr(history,'root_width',None)
                        history.root_width = (width if old_width is None else
                                              torch.minimum(old_width,width.to(old_width)))
                except Exception as error:
                    undefined = isinstance(error,ValueError) and str(error).startswith(
                        ('strict OBBT unary domain','strict QP affine-fold',
                         # capmem refusal of M2's own (second) table: M2 off, the run continues ([cap] line printed)
                         'graph QP exceeds fixed table capacity','epigraph exceeds fixed table capacity'))
                    if not undefined and not memory_error(error): raise
                    failed = True
                if failed:
                    if undefined:
                        self.disabled = True
                        e.stats['m2_stop'] = 'undefined'
                    else:
                        self.pause_memory()
                elapsed = self.clock()-tick
                # Progress: the certified root bound rose by more than tau, or some
                # root width shrank by more than the relative tolerance.
                raised = self.root_floor() > floor_before + tau
                shrunk = bool(((width_before-(self.upper-self.lower).flatten())
                               > e.rel_tol*width_before).any())
                self.value += dropped + float(raised)
                self.stalled = 0 if (raised or shrunk) else self.stalled+1
                if self.stalled >= self.stall_rounds:
                    self.skip_left, self.skip, self.stalled = self.skip, 2*self.skip, 0
                elif raised or shrunk:
                    self.skip = 1
                self.m2_seconds += elapsed
                round_seconds += elapsed
                event = dict(round=len(getattr(e,'m2_trace',())),cutoff=cutoff,tree_seconds=self.tree_seconds,
                             m2_seconds=self.m2_seconds,round_seconds=elapsed,
                             dropped=dropped,open=len(domains),stop=e.stats['m2_stop'],
                             memory_pauses=e.stats.get('m2_memory_pauses',0),
                             memory_resumes=e.stats.get('m2_memory_resumes',0))
                print('[m2-interleave] '+json.dumps(event),flush=True)
                e.stats.update(m2_seconds=self.m2_seconds,m2_tree_seconds=self.tree_seconds)
                if (getattr(e,'m2_closed',False) or
                        (len(domains) == 0 and not getattr(domains,'_domains_lost',False))):
                    certificate = cutoff
                    break
                if getattr(e,'local_waiting',False): break
        self.frontier = None if self.disabled else len(domains)
        self.accruing = math.isfinite(P) and self.ready(cutoff,tau)
        if not self.accruing:
            self.release()
            self.tree_seconds = min(self.tree_seconds,self.m2_seconds)
        finished = self.clock()
        self.m2_seconds += max(0.,finished-now-round_seconds)
        self.since = finished if self.tree_clock is None else self.tree_clock()
        e.stats.update(m2_seconds=self.m2_seconds,m2_tree_seconds=self.tree_seconds)
        self.check_deadline(deadline)
        if self.expired and certificate is not None:
            e.stats['m2_late_certificate'] = True
        return None if self.expired else certificate
