"""S4 owner-thread allocator: retained primal/LP atoms and complete CROWN batches.

Work is measured exclusive host service seconds. Exploration tickets follow a
doubling work calendar, independently of dispatch counts. Censored primal
episodes retain an unknown claim until an observation can settle them.
"""
import copy
import math
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from nlp_solver.continuation import Atom, Continuation
from input_split.tree_progress import ProgressForecast
from input_split.root_controller import RootClosed, RootDeadline, offer, poll, finish_timeout


@dataclass
class Job:
    name: str
    kind: str
    continuation: Continuation
    ready: object = lambda: True
    cap: int = 1
    work: float = 0.0
    pending: float = 0.0
    rate: float = 0.0
    previous_work: float = 0.0
    last: int = -1
    quote: float = 0.0
    gain: float = 0.0
    checking: float = 0.0
    wait_for: object = None
    observed: bool = False
    transaction_gain: float = 0.0
    settlement_gain: float = 0.0
    context: object = None
    initial_work: float = 0.0
    ticket_work: float = 0.0
    ticket_epoch: int = -1
    progress: ProgressForecast = field(default_factory=ProgressForecast)
    snapshot: object = None
    pricing_score: float = 0.0

    @property
    def score(self):
        return self.rate * self.previous_work / max(self.previous_work + self.pending, 1e-12)


class Allocator:
    """Work-paid dyadic probes, pending forecasts and proportional class service.

    The first bounded slices calibrate a host-service quantum, once. This is a
    measured-work mode, not the design's frozen cross-host reference profile.
    Tickets buy that much work, possibly many short continuation boundaries.
    Probe work cannot advance the exploitation clock or mint more tickets.
    """
    def __init__(self, jobs):
        self.jobs = jobs
        self.seq = self.epoch = 0
        self.exploit = self.probe_work = 0.0
        self.quantum = self.horizon = None
        self.tolerance = 1e-4
        self.credit = {'primal': 0.0, 'dual': 0.0}
        self.shares = dict(self.credit)

    def _calendar(self):
        if self.quantum is None:
            costs = [j.initial_work for j in self.jobs if j.initial_work > 0]
            if not costs:
                return
            self.quantum = sum(costs) / len(costs)
            self.horizon = self.quantum * len(self.jobs)
        while self.exploit >= self.horizon:
            self.epoch += 1
            self.horizon *= 2

    def choose(self, remaining=math.inf, P=0.0, D=0.0):
        self.tolerance = max(1e-5, 1e-4 * max([1.0] + [abs(v) for v in (P, D) if math.isfinite(v)]))
        ready = [j for j in self.jobs if not j.continuation.done and j.ready() and j.quote <= remaining
                 and (j.wait_for is None or j.wait_for.done())]
        if not ready:
            return None, 'idle'
        fresh = [j for j in ready if j.last < 0]
        if fresh:
            return fresh[0], 'initial'
        self._calendar()
        tickets = [j for j in ready if j.ticket_epoch != self.epoch or j.ticket_work < self.quantum]
        if tickets:
            return min(tickets, key=lambda j: (j.last, j.name)), 'explore'
        if not math.isfinite(P):
            primal = [j for j in ready if j.kind == 'primal']
            if primal:
                self.shares = {'primal': 1.0, 'dual': 0.0}
                return min(primal, key=lambda j: (j.work, j.name)), 'exploit'
        if math.isfinite(P) and not math.isfinite(D):
            dual = [j for j in ready if j.kind == 'dual']
            if dual:
                self.shares = {'primal': 0.0, 'dual': 1.0}
                return min(dual, key=lambda j: (j.name != 'root_lp', j.work, j.name)), 'exploit'
        tau = max(1e-5, 1e-4 * max(1.0, abs(P), abs(D)))
        H = max(0.0, P-D-tau)
        self.gap = H
        C = self.quantum * len(self.jobs)
        def measured(j):
            # Rebase the mass forecast before the next tree snapshot. Old
            # certified frontier rates remain compatible with a tighter cutoff.
            if j.progress.cutoff != P:
                j.progress.invalidate()
            forecast = j.progress.score(H, P)
            # The current own-frontier layer supersedes an older global-gain
            # observation. Keeping their maximum can indefinitely replay a
            # cheap initial bound rise while its present layer is stalled.
            if j.progress.bound is not None and math.isfinite(j.progress.bound):
                j.pricing_score = forecast
            else:
                j.pricing_score = max(j.score, forecast)
            return j.pricing_score
        def claim(j):
            # A zero on one completed box is not proof that later boxes cannot
            # close the gap. Keep a finite opportunity ceiling beside the
            # measured forecast, symmetrically for primal and dual work. All
            # family work stays in its denominator across episode/seed changes;
            # an unsuccessful completion cannot renew this claim for free.
            return max(measured(j), H / (C + j.work))
        champions = {k: max((j for j in ready if j.kind == k),
                           key=lambda j: (claim(j), j.name in ('gpu_tree', 'lp_tree'), -j.work), default=None)
                     for k in self.credit}
        dual = champions['dual']
        # Unknown primal opportunity is active while it dominates the measured
        # dual forecast. Otherwise its earned probes remain payable. No optimum,
        # instance label, residual reward or fitted success constant is used.
        def value(kind, j):
            if j is None:
                return 0.0
            rate = claim(j)
            if kind == 'primal' and dual is not None and rate < measured(dual):
                rate = j.score
            return min(H, self.horizon * rate)
        values = {k: value(k, j) for k, j in champions.items()}
        total = sum(values.values())
        self.shares = {k: v / total if total > 0 and math.isfinite(total) else 0.0
                       for k, v in values.items()}
        for k, f in self.shares.items():
            if f <= 0:
                self.credit[k] = 0.0
        if total <= 0:
            tree = next((j for j in ready if j.name == 'gpu_tree'), None)
            return tree or min(ready, key=lambda j: (j.work, j.name)), 'exploit'
        kind = max((k for k in self.credit if self.shares[k] > 0),
                   key=lambda k: (self.credit[k], self.shares[k], k == 'dual'))
        return champions[kind], 'exploit'

    def account(self, job, mode, seconds, gain, terminal):
        self.seq += 1
        job.last = self.seq
        job.quote = max(job.quote, seconds)
        job.work += seconds
        job.pending += seconds
        job.progress.account(seconds, job.snapshot, self.tolerance)
        if job.snapshot is not None:
            print(f'[TreeProgress] family={job.name} cutoff={job.progress.cutoff} '
                  f'mass={job.progress.mass} anchor={job.progress.anchor} bound={job.progress.bound} '
                  f'velocity={job.progress.velocity:.12g} bound_velocity={job.progress.frontier.velocity:.12g} '
                  f'pending_work={job.progress.pending:.12g} work={job.work:.12g}', flush=True)
        job.snapshot = None
        if mode == 'initial':
            job.initial_work = seconds
        job.transaction_gain += gain
        if job.settlement_gain == 0:
            job.settlement_gain = self.tolerance
        if (terminal or (job.kind == 'primal' and job.transaction_gain > 0)
                or (job.kind == 'dual' and job.transaction_gain >= job.settlement_gain)):
            # A censored prefix, including a tiny partial improvement, is not
            # a completed comparable primal episode.
            job.observed = True
            job.previous_work = job.pending
            job.rate = job.transaction_gain / max(job.pending, 1e-12)
            job.pending = job.transaction_gain = job.settlement_gain = 0.0
        if mode in ('initial', 'explore'):
            if job.ticket_epoch != self.epoch:
                job.ticket_epoch, job.ticket_work = self.epoch, 0.0
            job.ticket_work += seconds
            self.probe_work += seconds
        elif mode == 'exploit':
            self.exploit += seconds
            for kind, share in self.shares.items():
                self.credit[kind] += share * seconds
            self.credit[job.kind] -= seconds

    def report(self):
        total = sum(j.work for j in self.jobs)
        for j in self.jobs:
            print(f'[WorkShare] family={j.name} work={j.work:.9g} '
                  f'share={j.work/total if total else 0:.9g} pending={j.pending:.9g} '
                  f'observed={int(j.observed)} mass={j.progress.mass} mass_rate={j.progress.velocity:.9g} '
                  f'W={self.exploit:.9g} '
                  f'probe_work={self.probe_work:.9g} epoch={self.epoch}', flush=True)


def receipt_observer(allocator, backlog=()):
    """Credit the producer named by the commit, including asynchronous returns.

    A native incumbent arriving during a CROWN/LP slice must never become that
    slice's measured reward. Owner verification is charged to its producer too.
    """
    jobs = {j.name: j for j in allocator.jobs}
    def observe(name, seconds, gain):
        if name in ('ipopt_publish', 'ipopt_seed_publish'):
            native = jobs.get('ipopt_seed')
            if native is not None:
                allocator.account(native, 'publication', seconds, gain, True)
            return
        if name.startswith(('restore_', 'sqp_', 'polish_', 'alm_init_')):
            name = 'alm_' + name
        job = next((j for key, j in jobs.items() if name.startswith(key + '_')), None)
        if job is not None:
            job.gain += gain
            if name.endswith('_publish'):
                job.checking += seconds
    for receipt in backlog:
        observe(*receipt)
    return observe


def _check(root):
    terminal = getattr(root, 'controller_terminal', None)
    if terminal is not None:
        raise terminal
    if root.time_left() <= 0:
        raise RootDeadline()
    poll(root, service=False)
    if root.closes(root.values()[1]):
        raise RootClosed('commit')


def _alm(root, view):
    from nlp_solver.alm import ALMConfig, ALMPGDSolver
    from nlp_solver.problem import NNProblemSingleDomain
    from nlp_solver.falsification import _ends_with_structural_rows
    from nlp_solver.utils import _split_or_clauses
    from input_split.nlp_in_bab import apply_root_alm_overrides
    cfg = apply_root_alm_overrides(root.nlp_config)
    C_obj, rhs_obj, C_constr, rhs_constr = list(_split_or_clauses(
        view.c.squeeze(0), view.rhs.squeeze(0), view.or_spec_size))[0]
    rows = view.base_input_linear_constraints
    alm_rows = rows
    if (cfg.get('expand_input_constraints_as_outputs', False) and
            cfg.get('alm_deduplicate_expanded_input_constraints', False) and
            _ends_with_structural_rows(C_constr, rhs_constr, rows)):
        alm_rows = None
    problem = NNProblemSingleDomain(
        view.net.model_ori, view.dm_l.squeeze(0), view.dm_u.squeeze(0),
        C_obj, rhs_obj, C_constr, rhs_constr,
        input_constr_A=None if alm_rows is None else alm_rows[0],
        input_constr_b=None if alm_rows is None else alm_rows[1],
        initialization_constr_A=None if rows is None else rows[0],
        initialization_constr_b=None if rows is None else rows[1], device=None, dtype=None)
    alm_cfg = ALMConfig.from_dict(cfg)
    alm_cfg.obj_patience, alm_cfg.inner_plateau_check = -1, 0
    solver = ALMPGDSolver(alm_cfg)
    mu = getattr(view, 'root_dual_mu', None)
    if mu is not None and mu.shape != (problem.n_constraints,):
        mu = None
    result = yield from solver.solve_with_restarts(
        problem, timeout=max(0.0, root.time_left()), mu_init=mu, _iterate=True)
    if result.x_solution is not None:
        yield Atom('alm', result.x_solution)
    return result


def _bab_alm(root, view):
    """Resume primal work on an immutable oldest offered tree batch.

    Tree progress can offer one waiting batch while this episode is live. It
    cannot replace its optimizer/restoration state or create fairness tickets.
    """
    from input_split.attack_in_input_split import alm_attack_on_batch
    from input_split.batch_branch_and_bound import _primal_input_rows
    while root.time_left() > 0:
        batch = root.history.controller_primal_batch
        root.history.controller_primal_batch = None
        if batch is None:
            yield Atom('bab_alm')
            continue
        root.history.controller_primal_live = True
        _, lo, hi, cs, rhs, _, base, specs, *_ = batch
        import hashlib
        scope = hashlib.sha256(lo.numpy().tobytes() + hi.numpy().tobytes()).hexdigest()
        yield Atom('bab_alm', context=scope)
        device = view.dm_l.device
        lo, hi, cs, rhs, specs = (v.to(device) for v in (lo, hi, cs, rhs, specs))
        rhs = rhs.clone()
        rhs[:, 0] = view.rhs.reshape(-1)[0]
        rows = _primal_input_rows(base, getattr(root.history, 'input_rows_raw', None))
        if rows is not None:
            rows = tuple(v.to(device) for v in rows)
        cfg = dict(root.nlp_config, bab_timeout=max(0.0, root.time_left()))
        result = yield from alm_attack_on_batch(
            view.net.model_ori, lo, hi, cs, rhs, specs, history=root.history,
            nlp_config=cfg, x_warm=root.best_primal_x, base_input_linear_constraints=rows,
            incumbent=root.primal_solution, _iterate=True)
        yield Atom('bab_alm', result[2], settled=True)
        root.history.controller_primal_live = False


def _root_lp(root, view):
    from input_constraints import standard_form_to_input_constraints
    from input_split.lifted_bound import RootLifted
    setup = RootLifted.build(view.net, root.history, view.dm_l, view.dm_u, view.c, view.rhs,
                            standard_form_to_input_constraints(view.base_input_constraints),
                            root.nlp_config, allow_obbt=view.use_reordered_bab)
    root.lifted_setup = setup
    if setup is None:
        return
    # Initial finite bound has its own ticket, even before a feasible point exists.
    yield Atom('root_lp', bound=setup.bound(root.primal_solution, root.time_left()))
    setup.add_rlt()
    root.controller_rlt_ready = True
    root.controller_lp_cutoff = root.values()[0]
    yield Atom('root_lp', bound=setup.bound(root.primal_solution, root.time_left()))
    while root.time_left() > 0:
        root.controller_lp_cutoff = root.values()[0]
        yield Atom('root_lp', bound=setup.bound(root.primal_solution, root.time_left()))


def _obbt(root):
    setup = root.lifted_setup
    if setup is None or not setup.use_obbt:
        return
    setup.cutoff = root.values()[0]
    # Count-defined full sweeps replace both the five-second warm-up and the
    # residual OBBT allowance. The cursor and solver state live in the generator.
    def cutoff():
        setup.cutoff = min(setup.cutoff, root.values()[0])
        return setup.cutoff
    iterator = setup.lp.obbt(
        setup.lo, setup.hi, setup.rhs_rows, setup.eg, setup.eh,
        cutoff=setup.cutoff, rounds=setup.rounds, sliced=True, deadline=root.deadline, cutoff_fn=cutoff,
        _iterate=True)
    coordinates = 0
    try:
        while True:
            try:
                atom = next(iterator)
            except StopIteration as done:
                setup.lo, setup.hi = done.value
                break
            if atom.box is not None:
                setup.lo, setup.hi = atom.box
                # Tightening and its checked objective re-bound are one priced
                # dependency bundle. Width shrink itself earns no reward.
                coordinates += 1
                if coordinates % 4 == 0:
                    atom.bound = setup.bound(root.primal_solution, root.time_left())
            yield atom
    finally:
        iterator.close()
    root.history.lifted_box = (setup.lo.reshape(setup.dm_l.shape).clone(),
                               setup.hi.reshape(setup.dm_l.shape).clone())
    root.history.lifted_cutoff = setup.cutoff
    yield Atom('obbt', bound=setup.bound(root.primal_solution, root.time_left()))


def _lp_tree(root):
    from input_split.lifted_bound import _relaxes_problem
    setup = root.lifted_setup
    if setup is None or setup.lp.obj is None or not _relaxes_problem(root.history, setup.lp, setup.rhs_rows):
        return
    result = yield from setup.lp.branch_and_bound(
        setup.lo, setup.hi, setup.rhs_rows, setup.eg, setup.eh,
        cutoff=root.values()[0], floor=root.values()[1],
        tol_abs=float(root.nlp_config.get('bab_primal_dual_abs_tol', 1e-5)),
        tol_rel=float(root.nlp_config.get('bab_primal_dual_rel_tol', 1e-4)),
        max_nodes=int(root.nlp_config.get('lifted_lp_bab_max_nodes', 200000)),
        node_cutoff=bool(root.nlp_config.get('lifted_lp_node_cutoff', True)),
        node_obbt_gain=bool(root.nlp_config.get('lifted_lp_node_obbt_gain', False)),
        sliced=True, deadline=root.deadline, cutoff_fn=lambda: root.values()[0],
        _iterate=True)
    yield Atom('lp_tree', bound=result[0])


def _alternation(root, view):
    from nlp_solver import lp_alternation as alt
    setup = alt._setup(view)
    if setup is None:
        return
    def publish(x, obj, source):
        offer(root, 'lp_alternation', x, 0.0)
    def seed(point):
        if root.controller_seed is None:
            root.controller_seed = point.copy()
    config = dict(root.nlp_config, _controller_native_submit=seed)
    yield from alt._search(setup[2], config, alt._seeds(view), root.deadline,
                          publish, _iterate=True)


def _wait_native(root):
    """Owner-native calls wait for the currently admitted isolated seed."""
    while root.controller_native is not None and not root.controller_native.done():
        yield Atom('native_wait', wait_for=root.controller_native)


def _probes(root, view, names=("tangent", "gp_certificate", "exp_certificate", "convex_cuts")):
    from input_split import batch_branch_and_bound as bab
    for name in names:
        op = next(o for o in bab._ROOT_OPERATIONS if o.name == name)
        if not op.eligible(view):
            continue
        view.primal_solution, view.primal_solution_found = root.primal_solution, root.primal_solution_found
        view.best_primal_x = root.best_primal_x
        if name in ('gp_certificate', 'exp_certificate'):
            import importlib
            build = getattr(importlib.import_module('nlp_solver.' + name), 'build_' + name)
            diagnostics = {}
            certificate = build(view.net.net, view.net.model_ori,
                                ((view.x.ptb.x_L + view.x.ptb.x_U)/2).detach(),
                                view.c, view.rhs, view.x.ptb.x_L, view.x.ptb.x_U,
                                diagnostics, rowpool=view.rowpool)
            print(f'[{name}] {diagnostics}')
            yield Atom(name)
            if certificate is not None:
                yield from certificate.solve(root.best_primal_x, root.time_left(),
                                              sliced=True, _iterate=True)
        else:
            op.run(view, root.deadline)
            if view.best_primal_x is not None:
                offer(root, name, view.best_primal_x, 0.0)
            yield Atom(name)


def _seed_solver(root, view, executor, stop):
    from nlp_solver.ipopt_solver import IPOPTSolver, NNProblemIPOPT
    from nlp_solver.utils import _split_or_clauses
    clause = list(_split_or_clauses(view.c.squeeze(0), view.rhs.squeeze(0), view.or_spec_size))[0]
    rows = view.base_input_linear_constraints
    problem = NNProblemIPOPT(view.net.model_ori, view.dm_l.squeeze(0), view.dm_u.squeeze(0), *clause,
                             input_constr_A=None if rows is None else rows[0],
                             input_constr_b=None if rows is None else rows[1], device=None, dtype=None)
    def solve(seed):
        t0 = time.monotonic()
        cfg = dict(root.nlp_config, _controller_stop=stop,
                   _controller_monotonic_deadline=root.controller_monotonic_deadline)
        result = IPOPTSolver(cfg).solve_single(problem, x_init=seed, timeout=max(0.0, root.time_left()))
        offer(root, 'ipopt_seed', result.x_solution, time.monotonic()-t0)
        return result
    while root.time_left() > 0:
        first = root.controller_seed_initial
        root.controller_seed_initial = False
        seed = root.controller_seed
        root.controller_seed = None
        if root.controller_native is not None and not first:
            root.controller_native.result()  # ready() admits only a completed native atom
        from input_split.native_lane import NativeExecutor, solve_seed
        if isinstance(executor, NativeExecutor):
            cfg = dict(root.nlp_config, ipopt_timeout=max(0.0, root.time_left()),
                       _controller_monotonic_deadline=root.controller_monotonic_deadline)
            root.controller_native = executor.submit(
                solve_seed, view.net.model_ori, view.dm_l, view.dm_u, *clause, cfg, rows, seed)
        else:
            root.controller_native = executor.submit(solve, seed)
        yield Atom('ipopt_seed')


def _start_ipopt(root, view, stop, executor):
    from nlp_solver.falsification import _run_local_solver
    from nlp_solver.utils import _split_or_clauses
    cfg = dict(root.nlp_config, ipopt_timeout=max(0.0, root.time_left()),
               _controller_stop=stop, _controller_deadline=root.deadline,
               _controller_monotonic_deadline=root.controller_monotonic_deadline)
    cfg['_controller_publish'] = lambda family, point, seconds: offer(root, family, point, seconds)
    clause = list(_split_or_clauses(view.c.squeeze(0), view.rhs.squeeze(0), view.or_spec_size))[0]
    return executor.submit(_run_local_solver, view.net.model_ori, view.dm_l.squeeze(0),
                           view.dm_u.squeeze(0), *clause, cfg, bool(cfg.get('verbose', False)),
                           1e-6, view.base_input_linear_constraints)


def run(root, vnnlib, return_domains, index):
    from input_split import batch_branch_and_bound as bab
    jobs, future, executor = [], None, None
    # A fast native result can arrive during mandatory initial-domain setup,
    # before the allocator registry exists. Retain that paid receipt as well.
    early_receipts = []
    root.ledger.observer = lambda *receipt: early_receipts.append(receipt)
    seed_future = None
    allocator = None
    stop = threading.Event()
    training = [m.training for m in root.net.model_ori.modules()]
    root.net.model_ori.eval()
    def job(name, kind, iterator, ready=lambda: True, cap=1):
        j = Job(name, kind, Continuation(iterator, root.net.model_ori), ready, cap)
        jobs.append(j)
        return j
    from input_split.tree_slice import OwnerInterrupts, ServiceSizer
    primal_sizes = {'alm': ServiceSizer(), 'bab_alm': ServiceSizer()}
    interrupts = OwnerInterrupts(root)
    try:
        # Coverage prerequisites only. Optional root producers no longer sit in
        # front of the initial domains or join the root ALM stack.
        for name in ('reference_bounds', 'rows', 'struct_forms', 'struct_cert', 'hybrid', 'initial_verify',
                     'search_state'):
            op = next(o for o in bab._ROOT_OPERATIONS if o.name == name)
            if root.time_left() <= 0:
                return finish_timeout(root)
            if op.eligible(root):
                t0 = time.monotonic()
                outcome = op.run(root, root.deadline)
                root.ledger.commit_current(name, time.monotonic()-t0, root.values)
                _check(root)
                if outcome is not None:
                    return outcome
        root.root_nlp = SimpleNamespace(nlp_best_x=None, ipopt_x=None)
        root.primal_solution = root.primal_solution if root.primal_solution is not None else root.rhs.new_full((1,), math.inf)
        root.warmstart_x = root.best_primal_x
        root.same_or_spec_size = (root.or_spec_size == root.or_spec_size[0]).all()
        primal_rhs = root.rhs
        bab._root_consistency(root, root.deadline)
        bab._root_domain_list(root, root.deadline)
        bab._root_symmetry(root, root.deadline)
        view = copy.copy(root)  # root producers retain the unsplit box, rows and C
        view.rhs = primal_rhs
        if root.nlp_config.get('enabled', False) and not root.initial_verified:
            t0 = time.monotonic()
            bab._prepare_root_nlp(view)
            root.ledger.commit_current('primal_setup', time.monotonic()-t0, root.values)
            local_enabled = (root.nlp_config.get('enable_local_solver', False) or
                             (root.nlp_config.get('enable_alm', False) and
                              root.nlp_config.get('root_local_solver_warmstart', True)) or
                             root.nlp_config.get('ipopt_enabled', False))
            if local_enabled:
                from input_split.native_lane import NativeExecutor
                executor = NativeExecutor(root, stop)
                future = _start_ipopt(root, view, stop, executor)
                seed = getattr(view, 'root_lp_x', None)
                if seed is not None:
                    from input_split.native_lane import solve_seed
                    from nlp_solver.utils import _split_or_clauses
                    clause = list(_split_or_clauses(view.c.squeeze(0), view.rhs.squeeze(0), view.or_spec_size))[0]
                    cfg = dict(root.nlp_config, ipopt_timeout=max(0.0, root.time_left()),
                               _controller_monotonic_deadline=root.controller_monotonic_deadline)
                    seed_future = executor.submit(solve_seed, view.net.model_ori, view.dm_l, view.dm_u,
                                                  *clause, cfg, view.base_input_linear_constraints,
                                                  seed.detach().reshape(-1).cpu().double().numpy())
        root.controller_native = seed_future if seed_future is not None else future
        root.controller_seed_initial = seed_future is None
        root.controller_seed = (None if getattr(view, "root_lp_x", None) is None else
                                view.root_lp_x.detach().reshape(-1).cpu().double().numpy())
        t0 = time.monotonic()
        bab._root_initial_domains(root, root.deadline)
        root.ledger.commit_current('initial_domains', time.monotonic()-t0, root.values)
        _check(root)
        tree = job('gpu_tree', 'dual', bab._input_bab_loop(root, vnnlib, return_domains, index, _iterate=True))
        job('root_lp', 'dual', _root_lp(root, view),
            lambda: not getattr(root, 'controller_rlt_ready', False) or
                    root.values()[0] < getattr(root, 'controller_lp_cutoff', math.inf), cap=2)
        if executor is not None:
            job('ipopt_seed', 'primal', _seed_solver(root, view, executor, stop),
                lambda: root.controller_seed is not None and
                        (root.controller_seed_initial or root.controller_native is None or root.controller_native.done()))
        if root.nlp_config.get('lp_alternation', False):
            job('lp_alternation', 'primal', _alternation(root, view))
        if root.nlp_config.get('enabled', False) and root.nlp_config.get('enable_alm', False):
            job('alm', 'primal', _alm(root, view))
            if root.nlp_config.get('in_bab', False):
                root.history.controller_primal_batch = None
                root.history.controller_primal_ready = True
                job('bab_alm', 'primal', _bab_alm(root, view),
                    lambda: root.history.controller_primal_batch is not None or
                            getattr(root.history, 'controller_primal_live', False))
        for name in ('tangent', 'gp_certificate', 'exp_certificate', 'convex_cuts'):
            native_ready = (lambda: root.controller_native is None or root.controller_native.done())
            job(name, 'dual', _probes(root, view, (name,)),
                native_ready if name == 'tangent' else lambda: True)
        ready = lambda: bool(getattr(root, 'controller_rlt_ready', False))
        job('obbt', 'dual', _obbt(root),
            lambda: ready() and math.isfinite(root.values()[0]), cap=8)
        job('lp_tree', 'dual', _lp_tree(root), ready, cap=4)
        allocator = Allocator(jobs)
        root.ledger.observer = receipt_observer(allocator, early_receipts)
        while root.time_left() > 0:
            _check(root)
            if root.controller_native is not None and root.controller_native.done():
                root.controller_native.result()
            selected, mode = allocator.choose(root.time_left(), *root.values())
            if selected is None:
                break
            root.controller_tree_atom_quantum = allocator.quantum
            root.controller_tree_quantum = (None if allocator.quantum is None else
                                            allocator.quantum * len(jobs))
            sizer = primal_sizes.get(selected.name)
            if sizer is not None:
                selected.cap = sizer.size(math.inf, min(allocator.quantum, root.time_left())
                                          if allocator.quantum is not None else None)
            t0 = time.monotonic()
            nested = getattr(root, 'controller_nested_s', 0.0)
            if selected is tree:
                root.controller_tree_dispatch = t0
            print(f'[Schedule] family={selected.name} mode={mode} cap={selected.cap} '
                  f'rate={selected.pricing_score:.9g} global_rate={selected.score:.9g} unknown={int(not selected.observed)} pending={selected.pending:.6g} '
                  f'W={allocator.exploit:.9g} epoch={allocator.epoch} quantum={allocator.quantum} '
                  f'context={selected.context} mass={selected.progress.mass} '
                  f'mass_rate={selected.progress.velocity:.9g} '
                  f'forecast={selected.progress.score(getattr(allocator, "gap", 0.)):.9g}', flush=True)
            settled = False
            atoms = 0
            try:
                for _ in range(selected.cap):
                    atom_start = time.monotonic()
                    try:
                        atom = selected.continuation.step()
                        atoms += 1
                    except (RootClosed, RootDeadline):
                        root.ledger.commit_current(selected.name + '_interrupted',
                                           time.monotonic()-atom_start, root.values)
                        raise
                    with root.ledger.publication():
                        if atom is not None:
                            settled = settled or atom.settled
                            if atom.progress is not None:
                                selected.snapshot = atom.progress
                            if atom.context is not None and atom.context != selected.context:
                                selected.context = atom.context
                                # Ordinary new boxes/seeds retain the last comparable
                                # observation, paid cost, and existing tickets.
                            selected.wait_for = atom.wait_for
                            if atom.bound is not None and math.isfinite(atom.bound):
                                root.certified_dual = max(root.certified_dual, float(atom.bound))
                            if atom.family == 'lp_tree' and atom.point is not None:
                                root.controller_seed = atom.point.copy()
                            if atom.point is not None:
                                offer(root, selected.name, atom.point, 0.0)
                        root.ledger.commit_current(selected.name + ('_done' if selected.continuation.done else
                                                             '_episode' if settled else '_slice'),
                                           time.monotonic()-atom_start, root.values)
                    _check(root)
                    if (selected.continuation.done or selected.wait_for is not None or root.time_left() <= 0
                            or (sizer is not None and (settled or
                                (allocator.quantum is not None and time.monotonic()-t0 >= allocator.quantum)))):
                        break
            finally:
                seconds = max(0.0, time.monotonic()-t0 -
                              (getattr(root, 'controller_nested_s', 0.0) - nested)) + selected.checking
                allocator.account(selected, mode, seconds, selected.gain, selected.continuation.done or settled)
                if sizer is not None:
                    sizer.account(atoms, seconds)
                    print(f'[SliceBatch] family={selected.name} atoms={atoms} s={seconds:.9g}', flush=True)
                if selected is tree:
                    root.controller_tree_paid = getattr(root, 'controller_tree_paid', 0.) + seconds
                    root.controller_tree_dispatch = None
                selected.gain = selected.checking = 0.0
            if tree.continuation.done:
                return tree.continuation.result
        return finish_timeout(root)
    except RootDeadline:
        interrupts.finish()
        return finish_timeout(root)
    except RootClosed as closed:
        interrupts.finish()
        return root.finish(closed.source, root.values()[1])
    finally:
        interrupts.finish()
        root.ledger.observer = None
        stop.set()
        for j in jobs:
            pending = not j.continuation.done
            j.continuation.close()
            if pending:
                root.ledger.commit_current(j.name + "_cancelled", 0.0, root.values)
        if executor is not None:
            executor.shutdown(wait=True)
        for module, value in zip(root.net.model_ori.modules(), training):
            module.training = value
        # Final queued work is charged, but closure/deadline cannot be replaced
        # by a late result. Native IPOPT observes stop in its existing callback.
        while not root.controller_inbox.empty():
            family, _, seconds = root.controller_inbox.get_nowait()
            root.ledger.commit_current(family + '_cancelled', seconds, root.values)
            if allocator is not None:
                native = next((j for j in jobs if j.name == 'ipopt_seed'), None)
                if native is not None:
                    allocator.account(native, 'publication', seconds, 0.0, True)
        if allocator is not None:
            allocator.report()
        interrupts.close()
