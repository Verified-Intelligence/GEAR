"""Bounded primal proposals from PDHG lanes; GEAR alone accepts incumbents."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import time
import weakref
import torch


class LanePoints:
    def __init__(self, n, capacity):
        self.n, self.capacity = n, capacity
        self.points = self.bounds = None
        self.previous = None

    @torch.no_grad()
    def offer(self, z, bounds):
        x = z[:, :self.n].detach().double()
        b = bounds.detach().to(x).reshape(-1).expand(len(x))
        good = torch.isfinite(x).all(1) & ~b.isnan() & (b < torch.inf)
        x, b = x[good], b[good]
        if self.previous is not None:
            xold = self.previous.to(x)
            fresh = ~(x[:, None] == xold[None]).all(2).any(1)
            x, b = x[fresh], b[fresh]
        if self.points is not None:
            x = torch.cat((self.points.to(x), x))
            b = torch.cat((self.bounds.to(b), b))
        # Sort before deduplication so a repeated seed retains its lowest bound.
        order = b.argsort(stable=True)
        x, b = x[order], b[order]
        _, inverse = torch.unique(x, dim=0, return_inverse=True)
        first = torch.full((len(x),), len(x), dtype=torch.long, device=x.device)
        first.scatter_reduce_(0, inverse, torch.arange(len(x), device=x.device), reduce='amin')
        keep = first[first < len(x)].sort().values[:self.capacity]
        self.points, self.bounds = x[keep].clone(), b[keep].clone()

    def take(self):
        result = self.points
        self.previous = result.clone()
        self.points, self.bounds = result[:0], self.bounds[:0]
        return result


def cadence(cost, tree_seconds, waves):
    return max(1, math.ceil(cost*waves/tree_seconds)) if tree_seconds > 0 else 1


def local_batch(data, config, seeds, deadline, publish=None):
    """Existing CPU local solver, with one bounded batch and no nested workers/LPs."""
    from input_split.lifted_bound import make_local_solver
    tick = time.perf_counter()
    found = []
    for i, seed in enumerate(seeds):
        left = deadline-time.time()
        if left <= 0: break
        cfg = dict(config, controller=False, lifted_lp_leaf_ipopt_time=left/(len(seeds)-i))
        solve = make_local_solver(*data[:-1], cfg, data[-1])
        if solve is None: break
        result = solve(seed.numpy())
        if result is not None:
            found.append(result)
            if publish is not None: publish(*result)
    return found, time.perf_counter()-tick


class LaneLocal:
    """One CPU batch in flight and one owner-thread batched ALM continuation.

    Each batch reserves its deadline budget in the M2/tree ledger. C is the
    measured previous local cost divided by mean tree-wave cost. No backlog
    grows with the tree, and no background task changes solver state.
    """
    def __init__(self, root, engine):
        self.root, self.engine = root, engine
        self.capacity = min(engine.top, max(1, int(root.nlp_config.get('ipopt_bab_topk', 2))))
        self.queue = engine.lane_points = LanePoints(engine.n, self.capacity)
        self.future = self.pool = self.data = self.verify = None
        self.alm = None
        self.seconds = self.last_cost = self.reserved = 0.
        self.jobs = self.waves = self.last_wave = 0
        self.batch_cost = 0.
        self.tree_seconds = 0.
        self.accepted = []
        self.started = None
        self.context = None
        # The engine's memory walker must not follow this callback into the
        # caller's RootState and native tree storage.
        engine.lane_local = weakref.ref(self)

    def discovered(self, x, obj):
        # make_local_solver has checked the original CPU model, bounds and
        # raw rows already. Publication is a mailbox; only service() commits.
        print('[lane-candidate] '+json.dumps(dict(objective=obj,
              gear_seconds=time.time()-self.root.start)), flush=True)

    def submit(self, credit, primal, deadline):
        if self.future is not None or self.alm is not None or credit <= 0: return
        if self.queue.points is None or not len(self.queue.points): return
        C = cadence(self.last_cost, self.tree_seconds, self.waves)
        if self.waves-self.last_wave < C: return
        tick = self.clock()
        if self.data is None: self.setup()
        seeds = self.queue.take()
        cfg = self.root.nlp_config
        budget = min(credit, float(cfg.get('lifted_lp_leaf_ipopt_time', 5.)), deadline-time.time())
        if budget > 0:
            use_alm = self.engine.dev.type == 'cuda' and cfg.get('enable_alm', False)
            each = budget/(2 if use_alm else 1)
            self.reserved = budget
            self.batch_cost = 0.
            self.started = time.perf_counter(); self.last_wave = self.waves
            self.future = self.pool.submit(local_batch, self.data, dict(cfg), seeds.cpu(),
                                          min(deadline,time.time()+each), self.discovered)
            if use_alm: self.start_alm(seeds, each, primal)
            print('[lane-local] '+json.dumps(dict(batch=len(seeds), C=C, budget=budget,
                tree_seconds=self.tree_seconds, local_seconds=self.seconds)), flush=True)
        self.seconds += self.clock()-tick

    def proposal_ready(self):
        # Calibrate once at the first completed PDHG wave using the existing
        # local-solver cap. All later batches must earn measured tree credit.
        if self.context is not None and self.jobs == 0 and self.started is None:
            primal, deadline = self.context
            cost = float(self.root.nlp_config.get('lifted_lp_leaf_ipopt_time', 5.))
            self.submit(cost, primal, deadline)
            if self.future is not None:
                # Give the native tree a turn while calibration runs. A long
                # support wave must not delay collecting an early incumbent.
                self.engine.local_yield = self.engine.local_waiting = True

    def clock(self):
        if self.engine.dev.type == 'cuda': torch.cuda.synchronize(self.engine.dev)
        return time.perf_counter()

    def charge(self):
        """Completed work plus the entire reservation of the outstanding batch."""
        return self.seconds+self.reserved

    def setup(self):
        from input_split.lifted_bound import make_verifier
        r = self.root
        # Incumbents use GEAR's strict 1e-6 feasibility, as root_controller does.
        config = dict(r.nlp_config, constr_viol_tol=1e-6)
        self.verify = make_verifier(r.net.model_ori, r.x.ptb.x_L, r.x.ptb.x_U, r.c, r.rhs,
                                    r.or_spec_size, r.base_input_linear_constraints, config)
        model = copy.deepcopy(r.net.model_ori).cpu()
        cpu = lambda x: x.detach().cpu().clone() if x is not None else None
        lo, hi, c, rhs, spec = [cpu(t) for t in (r.x.ptb.x_L, r.x.ptb.x_U, r.c, r.rhs, r.or_spec_size)]
        rows = None if r.base_input_linear_constraints is None else tuple(cpu(t) for t in r.base_input_linear_constraints)
        verify = make_verifier(model, lo, hi, c, rhs, spec, rows, config)
        self.data = model, lo, hi, c, rhs, spec, rows, verify
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='lane-local')

    def collect(self):
        if self.future is None or not self.future.done(): return
        points, elapsed = self.future.result()
        self.future = None
        self.engine.local_waiting = False
        self.seconds += elapsed
        self.batch_cost += elapsed
        self.jobs += 1
        for x, _ in points:
            ok, obj, used = self.verify(x)
            if ok and math.isfinite(obj): self.accepted.append((used, obj))
        if self.alm is None:
            self.reserved = 0.
            self.last_cost = self.batch_cost

    def start_alm(self, seeds, budget, primal):
        """GEAR's batched ALM/polish, resumed only on the owner thread."""
        from input_split.attack_in_input_split import alm_attack_on_batch
        from nlp_solver.continuation import Continuation
        r = self.root; B = len(seeds)
        repeat = lambda x: x[:1].expand(B, *x.shape[1:]).clone()
        rows = r.base_input_linear_constraints
        rows = None if rows is None else tuple(repeat(t) for t in rows)
        rhs = repeat(r.rhs); rhs[:, 0] = primal.reshape(-1)[0]
        # One restart per offered point: no random-restart pool is multiplied
        # by the number of lane seeds.
        cfg = dict(r.nlp_config, controller=True, bab_timeout=budget,
                   bab_num_restarts=1, bab_br_budget=B)
        iterator = alm_attack_on_batch(r.net.model_ori, repeat(r.x.ptb.x_L), repeat(r.x.ptb.x_U),
            repeat(r.c), rhs, repeat(r.or_spec_size), nlp_config=cfg,
            x_warm=seeds.to(r.x.ptb.x_L).reshape(B, *r.x.ptb.x_L.shape[1:]),
            base_input_linear_constraints=rows, incumbent=primal, _iterate=True)
        # Proposals arrive from the no_grad PDHG boundary. ALM needs its own
        # gradient context, restored by Continuation between owner turns.
        with torch.enable_grad():
            self.alm = Continuation(iterator, r.net.model_ori)

    def poll_alm(self):
        if self.alm is None: return
        tick = self.clock()
        atom = self.alm.step()
        elapsed = self.clock()-tick
        self.seconds += elapsed
        self.batch_cost += elapsed
        if atom is not None and atom.point is not None:
            x = torch.as_tensor(atom.point).detach().cpu().numpy().reshape(-1)
            ok, obj, used = self.verify(x)
            if ok and math.isfinite(obj): self.accepted.append((used, obj))
        if self.alm.done:
            result = self.alm.result
            if result is not None and result[2] is not None:
                x = result[2].detach().cpu().numpy().reshape(-1)
                ok, obj, used = self.verify(x)
                if ok and math.isfinite(obj): self.accepted.append((used, obj))
            self.alm = None
            if self.future is None:
                self.reserved = 0.
                self.last_cost = self.batch_cost

    def service(self, domains, primal, found, alm_state, deadline, tree_seconds, m2_seconds):
        tick = self.clock()
        self.tree_seconds = max(self.tree_seconds, tree_seconds)
        self.context = (primal, deadline) if found else None
        self.waves += 1
        self.collect()
        # An atom is bounded by the existing ALM step/batch caps. The
        # reservation pays for it before M2 can spend that same credit.
        alm_tick = self.clock()
        if time.time() < deadline: self.poll_alm()
        alm_elapsed = self.clock()-alm_tick
        for x, obj in self.accepted:
            if not found or obj < float(primal.reshape(-1)[0]):
                primal = primal.new_tensor([obj]); found = True
                point = torch.as_tensor(x, device=primal.device, dtype=self.root.x.ptb.x_L.dtype).reshape_as(self.root.x.ptb.x_L)
                domains.history.alm_best_primal_x = point.detach().cpu()
                domains.sort_heuristic = 'lb_only'; domains.sort_index = 0
                if alm_state is not None:
                    alm_state.best_primal_x = alm_state.warmstart_x = point
                if self.root.controller:
                    self.root.primal_solution, self.root.best_primal_x = primal, point
                    self.root.primal_solution_found = True
                print('[lane-incumbent] '+json.dumps(dict(objective=obj, gear_seconds=time.time()-self.root.start,
                      jobs=self.jobs)), flush=True)
        self.accepted.clear()
        credit = self.tree_seconds-m2_seconds-self.charge()
        if (self.future is None and self.alm is None and self.queue.points is not None
                and len(self.queue.points) and found and credit > 0
                and time.time() < deadline):
            before = self.seconds
            self.submit(credit, primal, deadline)
            # submit() is also called from PDHG's proposal boundary.
            alm_elapsed += self.seconds-before
        self.seconds += max(0., self.clock()-tick-alm_elapsed)
        self.engine.stats.update(local_seconds=self.seconds, local_reserved=self.reserved, local_jobs=self.jobs)
        return primal, found

    def close(self):
        if self.alm is not None:
            self.alm.close(); self.alm = None
        if self.pool is not None:
            # The job already has a deadline reservation; never extend it.
            self.pool.shutdown(wait=True)
            self.collect()
        self.reserved = 0.
        self.engine.stats.update(local_seconds=self.seconds, local_reserved=0., local_jobs=self.jobs)


def offer(engine, z, bound):
    queue = getattr(engine, 'lane_points', None)
    if queue is not None:
        queue.offer(engine.qp.expand(z), bound)
        ref = getattr(engine, 'lane_local', None)
        lane = None if ref is None else ref()
        if lane is not None: lane.proposal_ready()
