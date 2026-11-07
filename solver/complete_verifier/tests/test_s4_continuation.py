"""Interrupted equal-work state equivalence and scheduling/cancellation contracts."""
from dataclasses import replace
import math

import pytest
import torch

from nlp_solver.alm import ALMConfig, ALMPGDSolver
from nlp_solver.problem import NNProblemSingleDomain
from nlp_solver.continuation import Atom, Continuation
from input_split.slice_controller import Allocator, Job


class Equality(torch.nn.Module):
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        h = x[:, 0] * x[:, 1] - 1
        return torch.stack(((x[:, 0]-2)**2 + (x[:, 1]-2)**2, h, -h), 1)


def problem(model):
    d = torch.float64
    return NNProblemSingleDomain(
        model, torch.full((2,), .1, dtype=d), torch.full((2,), 2., dtype=d),
        torch.tensor([[1., 0., 0.]], dtype=d), torch.zeros(1, dtype=d),
        torch.tensor([[0., 1., 0.], [0., 0., 1.]], dtype=d), torch.zeros(2, dtype=d),
        device=torch.device('cpu'), dtype=d)


@pytest.mark.parametrize('endgame', [False, True, 'raw'])
def test_interrupted_alm_equal_work_state_and_incumbent(endgame):
    cfg = ALMConfig(outer_max_iter=4, inner_steps=5, num_restarts=4,
                    timeout=600., obj_patience=-1, first_feas_patience=-1,
                    raw_polish_steps=4 if endgame else 0,
                    raw_polish_restarts=4, raw_polish_jitter=.01,
                    batched_gn_steps=3 if endgame is True else 0,
                    batched_gn_seeds=4,
                    opt_polish_candidates=4 if endgame else 0,
                    opt_polish_iters=3, opt_polish_seeds=2)
    def solve(sliced, interrupt):
        torch.manual_seed(123)
        model = Equality()
        gen = ALMPGDSolver(replace(cfg, controller=sliced)).solve_with_restarts(problem(model), _iterate=True)
        state = Continuation(gen, model)
        events, snapshots = [], []
        while not state.done:
            atom = state.step()
            if atom is not None:
                events.append(atom.family)
                # Inspect retained optimizer/multipliers and all loop-local state
                # while alternating unrelated random/autograd work on this thread.
                frame = gen.gi_frame
                snapshots.append(frame.f_locals['mu_constr'].detach().clone())
            if interrupt:
                assert torch.is_grad_enabled()
                torch.rand(7)
                with torch.enable_grad():
                    t = torch.ones(1, requires_grad=True)
                    (t*t).sum().backward()
        return state.result, state.rng, events, snapshots
    direct, rng_direct, _, _ = solve(False, False)
    a, rng_a, events_a, states_a = solve(True, False)
    b, rng_b, events_b, states_b = solve(True, True)
    for left, right in [(direct, a), (a, b)]:
        assert torch.equal(left.x_solution, right.x_solution)
        for name in ('objective_value', 'constraint_violation', 'outer_iterations',
                     'total_inner_steps', 'final_rho', 'success'):
            assert getattr(left, name) == getattr(right, name)
        assert torch.equal(left.final_mu, right.final_mu)
    assert torch.equal(rng_direct, rng_a) and torch.equal(rng_a, rng_b)
    assert events_a == events_b and 'alm' in events_a
    assert len(states_a) > 4
    assert all(torch.equal(a, b) for a, b in zip(states_a, states_b))
    if endgame:
        assert 'sqp' in events_a
        assert ('polish' if endgame == 'raw' else 'restore') in events_a


def test_continuation_close_restores_context_and_runs_cleanup():
    cleanup = []
    def work():
        try:
            with torch.no_grad():
                yield Atom('work')
                raise AssertionError('cancelled work resumed')
        finally:
            cleanup.append(True)
    state = Continuation(work())
    state.step()
    assert torch.is_grad_enabled()
    state.close()
    assert state.done and cleanup == [True] and torch.is_grad_enabled()


def test_floor_serves_pending_zero_gain_families_without_resetting_work():
    def work():
        while True:
            yield Atom('work')
    jobs = [Job(str(i), 'dual' if i else 'primal', Continuation(work())) for i in range(4)]
    allocator = Allocator(jobs)
    counts = [0]*4
    for i in range(200):
        job, mode = allocator.choose()
        counts[int(job.name)] += 1
        job.continuation.step()
        allocator.account(job, mode, .01, 1.0 if job is jobs[1] else 0.0, False)
    assert min(counts) >= 5
    assert jobs[0].pending == pytest.approx(.01*counts[0])
    assert math.isfinite(jobs[1].score)
    for job in jobs:
        job.continuation.close()


def test_lp_atoms_keep_equal_work_frontier_and_obbt_cursor():
    import time
    import numpy as np
    from tests.test_lifted_lp import _setup, build_lifted_lp, TOL
    def solve(interrupt):
        m, bm, x, c, rhs, lo, hi = _setup()
        lp = build_lifted_lp(bm, m, x, c, lo, hi)
        state = Continuation(lp.obbt(lo, hi, rhs[:, 1:]+TOL, cutoff=0., rounds=2,
                                     verbose=False, sliced=True, deadline=time.time()+600,
                                     _iterate=True))
        n = 0
        while not state.done:
            state.step()
            n += 1
            if interrupt:
                torch.rand(3)
        box = state.result
        tree = Continuation(lp.branch_and_bound(*box, rhs[:, 1:]+TOL, cutoff=0.,
                                                 max_nodes=12, return_leaves=True,
                                                 sliced=True, deadline=time.time()+600,
                                                 _iterate=True))
        bounds = []
        while not tree.done:
            atom = tree.step()
            if atom is not None:
                bounds.append(atom.bound)
            if interrupt:
                torch.rand(3)
        return box, n, tree.result, bounds
    a, b = solve(False), solve(True)
    assert a[1] == b[1] and a[1] > 4
    assert all(torch.equal(x, y) for x, y in zip(a[0], b[0]))
    assert a[2][:3] == b[2][:3] and a[3] == b[3]
    for x, y in zip(a[2][3][:4], b[2][3][:4]):
        assert np.array_equal(x, y, equal_nan=True)


def test_controller_rejects_nonfinite_objective():
    from unittest.mock import patch
    from tests.test_root_controller import root
    from input_split import root_controller as ctl
    r = root()
    ctl.offer(r, 'alm', [0.5], .01)
    with patch('input_split.lifted_bound.make_verifier', return_value=lambda x: (True, -math.inf, x)):
        ctl.poll(r, service=False)
    assert not r.primal_solution_found and math.isinf(r.values()[0])


def test_owner_interleaves_tree_and_primal_and_drops_live_state(monkeypatch):
    from types import SimpleNamespace
    from input_split import slice_controller as ctl, batch_branch_and_bound as bab
    from tests.test_root_controller import root as make_root
    r = make_root(P=5., D=-10.)
    r.nlp_config.update(enabled=True, enable_alm=True)
    r.controller_monotonic_deadline = math.inf
    r.history = SimpleNamespace()
    r.initial_verified = True  # avoid the native lane in this owner-thread unit
    r.rowpool = None
    r.or_spec_size = torch.ones(1, dtype=torch.long)
    r.domains = None
    r.warmstart_x = None
    def finish(source, D):
        # Result/export observes a stable incumbent after native notifications
        # have been disabled, including synchronous owner-thread closure.
        assert r.controller_notify is None
        return D, 1, 'safe'
    r.finish = finish
    events, closed = [], []
    for name in ('_root_consistency', '_root_domain_list', '_root_symmetry', '_root_initial_domains'):
        monkeypatch.setattr(bab, name, lambda *args: None)
    monkeypatch.setattr(bab, '_ROOT_OPERATIONS', tuple(
        bab.RootOp(name, lambda *args: None) for name in
        ('reference_bounds', 'rows', 'struct_forms', 'struct_cert', 'hybrid', 'initial_verify', 'search_state')))
    def empty(*args):
        if False:
            yield
    monkeypatch.setattr(ctl, '_root_lp', empty)
    monkeypatch.setattr(ctl, '_probes', empty)
    def primal(*args):
        try:
            for _ in range(1000):
                events.append('primal')
                yield Atom('alm')
        finally:
            closed.append('primal')
    def tree(*args, **kwargs):
        for _ in range(4):
            events.append('tree')
            yield Atom('gpu_tree', bound=-10.)
        yield Atom('gpu_tree', bound=5.)
    monkeypatch.setattr(ctl, '_alm', primal)
    monkeypatch.setattr(bab, '_input_bab_loop', tree)
    assert ctl.run(r, None, False, None)[2] == 'safe'
    assert events[0] == 'tree' and 'primal' in events
    assert events.count('primal') < 1000 and closed == ['primal']


def test_deadline_admission_retains_unaffordable_job():
    def work():
        yield Atom('work')
    j = Job('work', 'primal', Continuation(work()), quote=2.)
    allocator = Allocator([j])
    assert allocator.choose(1.) == (None, 'idle')
    assert not j.continuation.done and j.work == 0
    assert allocator.choose(3.)[0] is j
    j.continuation.close()


def test_seed_projection_can_yield_inside_initialization():
    class AffineEquality(torch.nn.Module):
        def forward(self, x):
            h = x[:, 0] + x[:, 1] - 1
            return torch.stack((x.square().sum(1), h, -h), 1)
    p = problem(AffineEquality())
    cfg = ALMConfig(controller=True, output_affine_project_init=True,
                    output_affine_project_sweeps=7, output_affine_project_fraction=.5)
    x = torch.tensor([[.2, .4], [.8, .9], [1., 1.], [1.4, 1.6]], dtype=torch.float64)
    plain = ALMPGDSolver(replace(cfg, controller=False))._project_output_affine_equalities(x.clone(), p, 1, 4)
    solver = ALMPGDSolver(cfg)
    state = Continuation(solver._project_output_affine_equalities(x.clone(), p, 1, 4, _iterate=True), p.model)
    count = 0
    while not state.done:
        atom = state.step()
        if atom is not None and atom.family == 'alm_init':
            count += 1
    assert count == 7 and torch.equal(plain, state.result)


def test_async_incumbent_is_credited_to_its_producer():
    from input_split.slice_controller import receipt_observer
    from input_split.root_controller import ReceiptLedger
    def work():
        yield Atom('work')
    tree = Job('gpu_tree', 'dual', Continuation(work()))
    native = Job('ipopt_seed', 'primal', Continuation(work()))
    alm = Job('alm', 'primal', Continuation(work()))
    allocator = Allocator([tree, native, alm])
    ledger = ReceiptLedger()
    ledger.P, ledger.D = 10., 0.
    ledger.observer = receipt_observer(allocator)
    ledger.commit('gpu_tree_slice', .5, 10., 1.)
    ledger.commit('ipopt_publish', .2, 6., 1.)
    ledger.commit('restore_publish', .1, 5., 1.)
    assert tree.gain == 1. and tree.checking == 0.
    assert native.rate == pytest.approx(4./.2)
    assert alm.gain == 1. and alm.checking == .1
    assert tree.rate == 0.
    for j in allocator.jobs:
        j.continuation.close()


def test_early_polynomial_lp_does_not_repeat_the_convex_objective_walk():
    from types import SimpleNamespace
    from unittest.mock import patch
    from tests.test_lifted_lp import _setup
    from input_split.lifted_bound import RootLifted
    m, bm, x, C, rhs, lo, hi = _setup()
    net = SimpleNamespace(net=bm, model_ori=m)
    history = SimpleNamespace(qtb=None, rowpool=None)
    cfg = dict(controller=True, lifted_lp_bound=True, lifted_lp_convex_objective=True)
    with patch('nlp_solver.convex_objective.build_convex_lifted_lp',
               side_effect=AssertionError('polynomial objective entered the atom walk')):
        setup = RootLifted.build(net, history, lo, hi, C, rhs, None, cfg)
    assert setup is not None and setup.lp.obj is not None


def test_native_wait_yields_to_tree_service_and_resumes_after_completion():
    from concurrent.futures import Future
    from types import SimpleNamespace
    from input_split.slice_controller import _wait_native
    future = Future()
    root = SimpleNamespace(controller_native=future)
    state = Continuation(_wait_native(root))
    atom = state.step()
    assert atom.wait_for is future
    waiter = Job('lp_alternation', 'primal', state, wait_for=atom.wait_for)
    def work():
        yield Atom('gpu_tree')
    tree = Job('gpu_tree', 'dual', Continuation(work()))
    allocator = Allocator([waiter, tree])
    assert allocator.choose()[0] is tree
    future.set_result(None)
    assert allocator.choose()[0] is waiter
    state.step()
    assert state.done
    tree.continuation.close()


def test_initial_ipopt_atom_leaves_the_lp_restart_for_owner_admission():
    from types import SimpleNamespace
    from tests.test_root_controller import root
    from input_split.slice_controller import _start_ipopt
    r = root()
    r.controller_monotonic_deadline = r.deadline
    r.or_spec_size = torch.ones(1, dtype=torch.long)
    r.root_lp_x = torch.ones(1, 1)
    captured = []
    class Executor:
        def submit(self, fn, *args):
            captured.append(args)
            return 'future'
    import threading
    assert _start_ipopt(r, r, threading.Event(), Executor()) == 'future'
    # _run_local_solver's extra_x0 is omitted: the first native solve must return
    # before the allocator may admit the separately offered LP-point start.
    assert len(captured[0]) == 11


def _pending_jobs():
    def work():
        while True:
            yield Atom('work')
    return [Job('gpu_tree', 'dual', Continuation(work())),
            Job('alm', 'primal', Continuation(work()))]


def test_calendar_buys_work_and_probe_work_cannot_issue_tickets():
    jobs = _pending_jobs()
    a = Allocator(jobs)
    for j, cost in zip(jobs, (10., .01)):
        a.account(j, 'initial', cost, 1. if j.kind == 'dual' else 0., False)
    j, mode = a.choose(P=10., D=0.)
    assert j is jobs[1] and mode == 'explore'
    epoch = a.epoch
    for _ in range(100):
        j, mode = a.choose(P=10., D=0.)
        assert j is jobs[1] and mode == 'explore'
        a.account(j, mode, .01, 0., False)
    assert a.exploit == 0 and a.epoch == epoch
    assert jobs[1].ticket_work > 1.
    a.account(jobs[0], 'exploit', 1000., 0., False)
    j, mode = a.choose(P=10., D=0.)
    assert mode == 'explore' and a.epoch > epoch
    a.account(j, mode, a.quantum, 0., False)
    assert j.ticket_epoch == a.epoch
    for j in jobs:
        j.continuation.close()


def test_unknown_primal_keeps_work_claim_beside_positive_dual_rate():
    jobs = _pending_jobs()
    a = Allocator(jobs)
    totals = {'primal': 0., 'dual': 0.}
    for j in jobs:
        a.account(j, 'initial', .01, 1. if j.kind == 'dual' else 0., False)
    for _ in range(300):
        j, mode = a.choose(P=10., D=0.)
        cost = .01 if j.kind == 'primal' else .3
        a.account(j, mode, cost, 0. if j.kind == 'primal' else 1., False)
        if mode == 'exploit':
            totals[j.kind] += cost
    assert totals['primal'] > 1. and totals['dual'] > 0
    # The unknown claim is finite and work-diluted; class shares can change
    # when its opportunity rate falls below the measured dual rate.
    assert not jobs[1].observed and jobs[1].pending == pytest.approx(jobs[1].work)
    a.account(jobs[1], 'exploit', .1, 2., False)
    assert jobs[1].observed and jobs[1].pending == jobs[1].transaction_gain == 0.
    assert jobs[1].rate == pytest.approx(2. / jobs[1].work)
    before = jobs[1].score
    a.account(jobs[1], 'exploit', 1., 0., False)
    assert 0 < jobs[1].score < before
    a.account(jobs[1], 'exploit', .1, 0., True)
    assert jobs[1].observed and jobs[1].score == 0
    for j in jobs:
        j.continuation.close()


def test_tree_batch_enlargement_is_work_bounded_and_deadline_checked():
    from types import SimpleNamespace
    from input_split.tree_slice import BatchCap, active, checkpoint
    from input_split.root_controller import RootDeadline
    cap = BatchCap(1, 65536)
    assert cap.batch(262144, None, 180.) == 1
    cap.account(1, .02)
    assert cap.batch(262144, .2, 180.) == 2
    assert cap.batch(262144, .2, .04) == 2
    for count in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
        cap.account(count, .02)
    assert cap.batch(262144, 1e6, 1e6) == 32768
    with pytest.raises(RootDeadline):
        with active(SimpleNamespace(time_left=lambda: 0.)):
            pass
    checkpoint()


def test_native_lane_cancels_an_atom_at_the_shared_deadline():
    import queue
    import threading
    import time
    from types import SimpleNamespace
    from input_split.native_lane import NativeExecutor
    deadline = time.monotonic() + .2
    root = SimpleNamespace(time_left=lambda: deadline-time.monotonic(), controller_inbox=queue.Queue())
    lane = NativeExecutor(root, threading.Event())
    try:
        # Spawn/import alone takes longer than this test's deadline. The owner
        # must cancel the child independently of a Python/native safe point.
        future = lane.submit(_slow_native_test_atom, Equality())
        assert future.result(timeout=3.) is None
        family, point, seconds = root.controller_inbox.get_nowait()
        assert family == 'ipopt' and point is None and 0 < seconds < 2.
    finally:
        lane.stop.set()
        lane.shutdown()


def _slow_native_test_atom(model):
    import time
    time.sleep(10.)
    raise AssertionError('the native atom should have been cancelled')


def _native_point_test_atom(model):
    from types import SimpleNamespace
    return SimpleNamespace(x=torch.ones(1, dtype=torch.float64))


def test_native_lane_delivers_completed_restarts_without_an_exit_race():
    import queue
    import threading
    from types import SimpleNamespace
    from input_split.native_lane import NativeExecutor
    root = SimpleNamespace(time_left=lambda: 30., controller_inbox=queue.Queue())
    lane = NativeExecutor(root, threading.Event())
    try:
        for _ in range(8):
            # A cold fresh interpreter imports Torch; subsequent restarts reuse
            # that interpreter, rather than inheriting the owner's native pools.
            result = lane.submit(_native_point_test_atom, Equality()).result(timeout=10.)
            assert result.x.item() == 1.
            assert root.controller_inbox.get_nowait()[1].item() == 1.
    finally:
        lane.stop.set()
        lane.shutdown()


def test_missing_endpoint_service_does_not_form_infinite_rates():
    jobs = _pending_jobs()
    a = Allocator(jobs)
    for j in jobs:
        a.account(j, 'initial', .01, 0., False)
    assert a.choose(P=math.inf, D=0.)[0] is jobs[1]
    assert a.choose(P=1., D=-math.inf)[0] is jobs[0]
    assert all(math.isfinite(j.score) for j in jobs)
    for j in jobs:
        j.continuation.close()


def test_owner_deadline_wakes_inside_an_atom_and_restores_signal_handlers():
    import signal
    import time
    from types import SimpleNamespace
    from input_split.tree_slice import OwnerInterrupts
    from input_split.root_controller import RootDeadline
    deadline = time.monotonic() + .05
    root = SimpleNamespace(time_left=lambda: deadline-time.monotonic(), ledger=SimpleNamespace())
    old_alarm, old_notify = signal.getsignal(signal.SIGALRM), signal.getsignal(signal.SIGUSR1)
    interrupts = OwnerInterrupts(root)
    try:
        with pytest.raises(RootDeadline):
            time.sleep(.2)
            # Signals latch termination; only a safe boundary unwinds work.
            from input_split.tree_slice import checkpoint
            checkpoint()
    finally:
        interrupts.finish()
        interrupts.close()
    assert signal.getsignal(signal.SIGALRM) is old_alarm
    assert signal.getsignal(signal.SIGUSR1) is old_notify


def test_bab_primal_atoms_preserve_equal_work_results():
    from input_split.attack_in_input_split import alm_attack_on_batch
    model = Equality()
    lo, hi = torch.full((2, 2), .1, dtype=torch.float64), torch.full((2, 2), 2., dtype=torch.float64)
    C = torch.eye(3, dtype=torch.float64)[None].expand(2, -1, -1)
    rhs = torch.zeros(2, 3, dtype=torch.float64)
    specs = torch.full((2,), 3, dtype=torch.long)
    cfg = dict(bab_num_restarts=4, bab_br_budget=8, bab_timeout=120., bab_outer_max_iter=2,
               bab_inner_steps=3, alm_obj_patience=-1, alm_first_feas_patience=-1,
               alm_batched_gn_steps=2, alm_batched_gn_seeds=4, alm_opt_polish_candidates=2,
               alm_opt_polish_iters=2)
    torch.manual_seed(123)
    direct = alm_attack_on_batch(model, lo, hi, C, rhs, specs, nlp_config=cfg)
    torch.manual_seed(123)
    state = Continuation(alm_attack_on_batch(model, lo, hi, C, rhs, specs,
                                           nlp_config=dict(cfg, controller=True), _iterate=True), model)
    atoms = 0
    while not state.done:
        if state.step() is not None:
            atoms += 1
        torch.rand(3)
    assert atoms > 2
    for a, b in zip(direct, state.result):
        assert (a is None and b is None) or torch.equal(a, b)


def test_sub_tolerance_dual_gain_keeps_the_frozen_layer_pending():
    jobs = _pending_jobs()
    a = Allocator(jobs)
    tree = jobs[0]
    a.account(tree, 'initial', .1, 1e-12, False)
    assert not tree.observed and tree.pending == .1 and tree.transaction_gain == 1e-12
    frozen = tree.settlement_gain
    a.choose(P=1e8, D=0.)  # changing the snapshot cannot erase/renew old layer cost
    assert tree.settlement_gain == frozen
    a.account(tree, 'exploit', .2, frozen, False)
    assert tree.observed and tree.rate == pytest.approx((frozen+1e-12)/.3)
    for j in jobs:
        j.continuation.close()


def test_lp_alternation_offers_native_seeds_and_settles_without_waiting(monkeypatch):
    import time
    from tests.test_lp_alternation import Haverly, LO, HI, state_of
    from nlp_solver import lp_alternation as la, ipopt_solver
    state = state_of(Haverly(), LO, HI)
    alt = la._setup(state)[2]
    seeds = []
    def forbidden(*args, **kwargs):
        raise AssertionError('LP alternation waited for an owner-native solve')
    monkeypatch.setattr(ipopt_solver.IPOPTSolver, 'solve_single', forbidden)
    iterator = la._search(alt, dict(state.nlp_config, controller=True,
                                   _controller_native_submit=seeds.append), [], time.time()+10.,
                          lambda *args: None, _iterate=True)
    try:
        for atom in iterator:
            if atom.settled:
                break
        assert seeds and atom.settled
    finally:
        iterator.close()


def test_obbt_refreshes_cutoff_at_coordinate_boundaries(monkeypatch):
    import time
    from tests.test_lifted_lp import _setup, build_lifted_lp, TOL
    m, bm, x, C, rhs, lo, hi = _setup()
    lp = build_lifted_lp(bm, m, x, C, lo, hi)
    original, cutoffs = lp.sparse_lp, []
    def sparse(*args):
        cutoffs.append(args[-1])
        return original(*args)
    monkeypatch.setattr(lp, 'sparse_lp', sparse)
    current = [0.]
    iterator = lp.obbt(lo, hi, rhs[:, 1:]+TOL, cutoff=0., rounds=1, verbose=False,
                       sliced=True, deadline=time.time()+10., cutoff_fn=lambda: current[0], _iterate=True)
    try:
        next(iterator)  # lower LP of the first retained coordinate
        current[0] = -.1
        list(iterator)
    finally:
        iterator.close()
    assert cutoffs[0] == 0. and -.1 in cutoffs[1:]


def test_continuation_keeps_private_contextvars_between_deep_yields():
    from contextvars import ContextVar
    active = ContextVar('test_retained_scope', default='owner')
    seen = []
    def work():
        token = active.set('pending bound')
        try:
            yield Atom('gpu_tree')
            seen.append(active.get())
        finally:
            active.reset(token)
    state = Continuation(work())
    state.step()
    assert active.get() == 'owner'
    state.step()
    assert seen == ['pending bound'] and active.get() == 'owner'


def test_ipopt_publishes_a_completed_feasible_restart_before_optional_polishing(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from nlp_solver import ipopt_solver as ip
    events = []
    def solve(self, problem, x, timeout, **kwargs):
        events.append('solve')
        return ip.IPOPTResult(True, np.array([.5]), -.5, 0., .01, 'ok', 1)
    monkeypatch.setattr(ip, 'CYIPOPT_AVAILABLE', True)
    monkeypatch.setattr(ip.IPOPTSolver, '_solve_single_cyipopt', solve)
    config = dict(controller=True, ipopt_constraint_slack_frac=.99,
                  _controller_point=lambda point: events.append('candidate'))
    result = ip.IPOPTSolver(config).solve_single(SimpleNamespace(lb=np.zeros(1), ub=np.ones(1)), timeout=30.)
    assert events == ['solve', 'candidate', 'solve', 'candidate']
    assert result.x_solution[0] == .5


def test_native_work_before_registry_creation_is_carried_into_family_accounting():
    from input_split.slice_controller import receipt_observer
    j = Job('ipopt_seed', 'primal', Continuation(atom for atom in ()))
    allocator = Allocator([j])
    observer = receipt_observer(allocator, [('ipopt_publish', .7, 0.)])
    assert j.work == j.previous_work == .7 and j.observed and j.rate == 0.
    observer('ipopt_seed_publish', .3, .6)
    assert j.work == 1. and j.rate == pytest.approx(2.)
    j.continuation.close()


def test_completed_zero_episodes_do_not_starve_later_primal_boxes():
    jobs = _pending_jobs()
    allocator = Allocator(jobs)
    for job in jobs:
        allocator.account(job, 'initial', .01, 0., True)
    exploited = {'primal': 0., 'dual': 0.}
    for _ in range(1000):
        job, mode = allocator.choose(P=10., D=0.)
        cost = .003 if job.kind == 'primal' else .3
        allocator.account(job, mode, cost, 0., True)
        if mode == 'exploit':
            exploited[job.kind] += cost
    assert exploited['primal'] > 0. and exploited['dual'] > 0.
    assert all(j.observed and j.score == 0. for j in jobs)
    # Paid history survives each unsuccessful completion, rather than minting
    # a fresh opportunity denominator or resetting exploration tickets.
    assert all(j.work > j.previous_work for j in jobs)
    for job in jobs:
        job.continuation.close()


def test_root_objective_rebound_can_skip_discarded_node_obbt(monkeypatch):
    from tests.test_lifted_lp import _setup, build_lifted_lp, TOL
    import numpy as np
    m, bm, x, C, rhs, lo, hi = _setup()
    lp = build_lifted_lp(bm, m, x, C, lo, hi)
    def forbidden(*args):
        raise AssertionError('a root objective re-bound launched optional node probes')
    monkeypatch.setattr(lp, '_variable_violation', forbidden)
    bound, point, _, _, _ = lp.solve_bab_node(
        lo[0].numpy(), hi[0].numpy(), (rhs[:, 1:]+TOL).reshape(-1).numpy(),
        None, None, np.inf, probe=False)
    assert np.isfinite(bound) and point is not None
