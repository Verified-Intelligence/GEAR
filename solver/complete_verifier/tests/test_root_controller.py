"""S4 stage 1-2: publication, cancellation and certified reward boundaries."""
import math
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from input_split import root_controller as ctl
from input_split.batch_branch_and_bound import RootState


def root(P=None, D=-10.0):
    r = object.__new__(RootState)
    r.controller = r.unified = True
    r.use_reordered_bab = True
    r.deadline = time.time() + 10
    r.nlp_config = {'controller': True}
    r.primal_solution = None if P is None else torch.tensor([P])
    r.primal_solution_found = P is not None
    r.best_primal_x = None
    r.report_only_primal = None
    r.global_lb = torch.tensor([[D]])
    r.certified_dual = -math.inf
    r.controller_inbox = queue.Queue()
    r.ledger = ctl.ReceiptLedger()
    r.net = SimpleNamespace(model_ori=torch.nn.Identity())
    r.dm_l, r.dm_u = torch.zeros(1, 1), torch.ones(1, 1)
    r.x = SimpleNamespace(ptb=SimpleNamespace(x_L=r.dm_l, x_U=r.dm_u))
    r.c, r.rhs = torch.ones(1, 1, 1), torch.zeros(1, 1)
    r.or_spec_size, r.base_input_linear_constraints = None, None
    r.base_input_constraints = None
    return r


def test_receipts_first_endpoints_are_events_and_gains_telescope(capsys):
    ledger = ctl.ReceiptLedger()
    ledger.commit('initial', 0.2, 8., 0.)
    ledger.commit('unverified_or_contracted', 0.3, 8., 0.)
    ledger.commit('checked', 0.4, 6., 1.)
    lines = capsys.readouterr().out.splitlines()
    assert 'dD=0 dP=0' in lines[0]
    assert 'first_feasible=1 first_bound=1' in lines[0]
    assert 'dD=0 dP=0' in lines[1]
    assert 'dD=1 dP=2' in lines[2]
    assert 'work=0.4 unit=host_seconds job=checked:3' in lines[2]
    assert 'P=6 D=1' in lines[2]


def test_lifted_local_gain_and_contraction_do_not_earn_global_gain(capsys):
    ledger = ctl.ReceiptLedger()
    ledger.P, ledger.D = 8., 0.
    ledger.note('lifted_bab', 0.4, dD=100, dP=100)
    ledger.iteration(SimpleNamespace(time_sum={'bounding': .2}), 2, 8., 0.)
    assert 'dD=0 dP=0' in capsys.readouterr().out


@patch('input_split.lifted_bound.make_verifier')
def test_only_verified_points_enter_root_state(verify):
    r = root()
    verify.return_value = lambda point: (False, -100., point)
    ctl.offer(r, 'ipopt', torch.zeros(1), .1)
    ctl.poll(r)
    assert r.values()[0] == math.inf
    verify.return_value = lambda point: (True, 3., point)
    ctl.offer(r, 'ipopt', torch.zeros(1), .1)
    ctl.poll(r)
    assert r.values()[0] == 3. and r.primal_solution_found


@patch('input_split.lifted_bound.make_verifier')
def test_mid_batch_publication_exports_the_new_incumbents_coordinates(verify, monkeypatch):
    from input_split.batch_branch_and_bound import export_best_primal
    r = root(P=4., D=0.)
    r.domains = SimpleNamespace(history=SimpleNamespace(alm_best_primal_x=torch.ones(1)))
    verify.return_value = lambda point: (True, 3., point)
    ctl.offer(r, 'ipopt', torch.zeros(1), .1)
    ctl.poll(r)
    monkeypatch.delenv('ABCROWN_BEST_PRIMAL_PATH', raising=False)
    point = export_best_primal(r.net, r.domains, r.best_primal_x,
                               r.primal_solution, r.primal_solution_found)
    assert torch.equal(point, torch.zeros(1)) and r.values()[0] == 3.


@patch('input_split.root_controller.early_lp')
@patch('input_split.lifted_bound.make_verifier')
def test_ipopt_publish_unwinds_inside_first_alm_inner_forward(verify, early):
    r = root(D=3.)
    verify.return_value = lambda point: (True, 3., point)
    config_seen = {}
    def runner(**kwargs):
        config_seen.update(kwargs['nlp_config'])
        kwargs['nlp_config']['_controller_publish']('ipopt', torch.zeros(1), .1)
        r.net.model_ori(torch.ones(1))  # first inner iteration, no outer-loop join
        pytest.fail('ALM kept running after closure')
    with pytest.raises(ctl.RootClosed, match='ipopt_publish'):
        ctl.run_nlp(r, runner, nlp_config=r.nlp_config)
    assert config_seen['_controller_stop'].is_set()
    assert not r.net.model_ori._forward_pre_hooks
    assert r.values() == (3., 3.)


@patch('input_split.root_controller.early_lp')
@patch('input_split.lifted_bound.make_verifier')
def test_nonclosing_candidate_does_not_stop_alm_or_skip_its_restoration(verify, early):
    r = root(D=0.)
    verify.return_value = lambda point: (True, 3., point)
    phases = []
    def runner(**kwargs):
        kwargs['nlp_config']['_controller_publish']('ipopt', torch.zeros(1), .1)
        for phase in ('inner', 'outer', 'restoration'):
            r.net.model_ori(torch.ones(1))
            phases.append(phase)
        return 'done'
    assert ctl.run_nlp(r, runner, nlp_config=r.nlp_config) == 'done'
    assert phases == ['inner', 'outer', 'restoration']


@patch('input_split.lifted_bound.RootLifted.build')
def test_early_bound_needs_no_primal_and_commits_nonclosing_certificate(build):
    r = root()
    r.rowpool = None
    r.deadline = time.time() + 10
    setup = SimpleNamespace(bound=lambda P, seconds: -2. if P is None else pytest.fail('unexpected P'))
    build.return_value = setup
    ctl.early_lp(r)
    assert r.values() == (math.inf, -2.)
    assert float(r.global_lb[0, 0]) == -10.  # original CROWN plane is untouched


@patch('input_split.root_controller.early_lp')
@patch('input_split.lifted_bound.make_verifier')
def test_worker_thread_only_enqueues_and_owner_thread_commits(verify, early):
    r = root()
    verify.return_value = lambda point: (True, 3., point)
    def runner(**kwargs):
        worker = threading.Thread(target=lambda: kwargs['nlp_config']['_controller_publish'](
            'ipopt', torch.zeros(1), .1))
        worker.start()
        worker.join()
        assert r.values()[0] == math.inf
        r.net.model_ori(torch.ones(1))
        assert r.values()[0] == 3.
    ctl.run_nlp(r, runner, nlp_config=r.nlp_config)


@patch('input_split.lifted_bound._relaxes_problem', return_value=True)
def test_independent_strengthening_closes_before_alm_resumes(coverage):
    r = root(P=3., D=0.)
    r.deadline = time.time() + 10
    r.controller_nested_s = 0.
    phases = []
    history = SimpleNamespace(lifted_box=None)
    setup = SimpleNamespace(history=history, lp=object(), rhs_rows=torch.zeros(1),
                            lo=torch.zeros(1, 1), hi=torch.ones(1, 1))
    setup.warm_up = lambda seconds: phases.append('warm_up')
    setup.add_rlt = lambda: phases.append('rlt')
    def obbt(point, seconds):
        phases.append('obbt')
        history.lifted_box = (torch.tensor([[.5]]), torch.ones(1, 1))
    setup.obbt = obbt
    setup.bound = lambda P, seconds: 3. if float(setup.lo[0, 0]) == .5 else 2.
    with pytest.raises(ctl.RootClosed):
        ctl.advance_lifted(r, setup, 3.)
    assert phases == ['warm_up', 'rlt', 'obbt']
    assert r.values() == (3., 3.)


@patch('input_split.lifted_bound.RootLifted.build', return_value=None)
def test_early_lp_uses_relaxed_dual_input_rows(build):
    r = root()
    r.rowpool = None
    r.deadline = time.time() + 10
    A, rhs = torch.ones(1, 1, 1), torch.tensor([[1.000001]])
    r.base_input_constraints = (A, -rhs)
    r.base_input_linear_constraints = (A, torch.ones(1, 1))
    ctl.early_lp(r)
    assert build.call_args.kwargs["allow_obbt"] is True
    dual_rows = build.call_args.args[6]
    assert torch.equal(dual_rows[1], rhs)
    assert not torch.equal(dual_rows[1], r.base_input_linear_constraints[1])


@pytest.mark.parametrize('D,closed', [(1., False), (3., True)])
@patch('input_split.lifted_bound.make_verifier')
@patch('input_split.lifted_bound.run_lifted_bab')
def test_independent_batch_retains_frontier_and_commits_checked_bound(batch, verifier, D, closed):
    r = root(P=3., D=0.)
    r.nlp_config['lifted_lp_bab_time'] = 30.
    r.controller_nested_s = 0.
    r.early_strengthened = True
    history = SimpleNamespace(lifted_bab_open='last committed frontier')
    r.early_lifted_setup = SimpleNamespace(history=history, lo=r.dm_l, hi=r.dm_u)
    def solve(*args, **kwargs):
        history.lifted_bab_D = D
        return D if closed else None
    batch.side_effect = solve
    if closed:
        with pytest.raises(ctl.RootClosed):
            ctl.independent_bab(r)
    else:
        ctl.independent_bab(r)
    assert r.values() == (3., D)
    assert history.lifted_bab_open == 'last committed frontier'
    assert batch.call_args.kwargs['time_share'] == 1.


def test_zero_gain_dependency_work_is_carried_to_the_certified_rebound(capsys):
    ledger = ctl.ReceiptLedger()
    ledger.P, ledger.D = 8., 0.
    ledger.commit('early_lifted_obbt', .3, 8., 0.)
    ledger.commit('early_lifted_bound', .2, 8., 1.)
    lines = capsys.readouterr().out.splitlines()
    assert 'pending_work=0.3 settled_work=0' in lines[0]
    assert 'pending_work=0 settled_work=0.5' in lines[1]
    assert 'transaction=root_lifted:0' in lines[0] and 'transaction=root_lifted:0' in lines[1]


@patch('input_split.root_controller.early_lp')
def test_deadline_cancels_inside_alm_and_keeps_last_committed_incumbent(early):
    r = root(P=3., D=1.)
    config_seen = {}
    def runner(**kwargs):
        config_seen.update(kwargs['nlp_config'])
        r.deadline = time.time() - 1.
        r.net.model_ori(torch.ones(1))
        pytest.fail('ALM continued past deadline')
    with pytest.raises(ctl.RootDeadline):
        ctl.run_nlp(r, runner, nlp_config=r.nlp_config)
    assert r.values() == (3., 1.)
    assert config_seen['_controller_stop'].is_set()
    assert not r.net.model_ori._forward_pre_hooks


def test_deadline_rejects_even_untimed_root_setup():
    from input_split import batch_branch_and_bound as bbb
    r = root(P=3., D=1.)
    r.deadline = time.time() - 1.
    ran = []
    with patch.object(bbb, '_ROOT_OPERATIONS', (bbb.RootOp('initial_domains', lambda *args: ran.append(1)),)), \
         patch.object(ctl, 'finish_timeout', return_value=('timeout',)):
        assert bbb._run_root(r) == ('timeout',)
    assert not ran


def test_controller_deadline_includes_process_startup():
    from input_split import batch_branch_and_bound as bbb
    conf = __import__('arguments').ConfigHandler()
    conf.construct_config_dict(conf.default_args)
    conf['solving']['solving_mode'] = True
    conf['solver']['nlp']['controller'] = True
    conf['bab']['override_timeout'] = 180.
    x = SimpleNamespace(ptb=SimpleNamespace(x_L=torch.zeros(1, 2), x_U=torch.ones(1, 2)))
    net = SimpleNamespace(net=SimpleNamespace())
    now = time.time()
    with patch.object(bbb.arguments, 'Config', conf), patch.object(bbb, '_process_start_time', return_value=now - 10):
        r = RootState(net, x, torch.ones(1, 2, 2), torch.zeros(1, 2), torch.tensor([2]), None, 179., None)
    assert r.deadline == pytest.approx(now + 170.)
    assert 169. < r.timeout <= 170.
    assert r.stats.timer.deadline == r.deadline


def test_no_verified_primal_leaves_an_infinite_unrewarded_incumbent():
    from input_split import batch_branch_and_bound as bbb
    r = root()
    r.or_spec_size = torch.tensor([1])
    r.root_nlp = SimpleNamespace(nlp_best_x=None, ipopt_x=None)
    bbb._root_incumbent(r, r.deadline)
    assert torch.isinf(r.primal_solution).all()
    assert not r.primal_solution_found and r.best_primal_x is None


@patch('input_split.root_controller.early_lp')
@patch('input_split.lifted_bound.make_verifier')
@patch('input_split.lifted_bound.run_lifted_bab')
def test_candidate_closes_inside_native_batch_and_keeps_previous_frontier(batch, verifier, early, capsys):
    r = root(P=3., D=1.)
    r.ledger.P, r.ledger.D = 3., 1.
    r.nlp_config['lifted_lp_bab_time'] = 30.
    r.early_strengthened, r.early_lifted_cutoff = True, 3.
    history = SimpleNamespace(lifted_bab_open='previous committed frontier')
    r.early_lifted_setup = SimpleNamespace(history=history, lo=r.dm_l, hi=r.dm_u)
    verifier.return_value = lambda point: (True, 1., point)
    def native(*args, **kwargs):
        ctl.offer(r, 'ipopt', torch.zeros(1), .01)
        kwargs['checkpoint']()
        pytest.fail('native batch continued after closure')
    batch.side_effect = native
    with pytest.raises(ctl.RootClosed):
        ctl.run_nlp(r, lambda **kwargs: r.net.model_ori(torch.ones(1)), nlp_config=r.nlp_config)
    assert r.values() == (1., 1.)
    assert history.lifted_bab_open == 'previous committed frontier'
    assert 'early_lifted_bab_cancelled' in capsys.readouterr().out
    assert batch.call_count == 1


def test_obbt_checkpoint_cancels_before_a_coordinate_lp_and_keeps_old_box():
    from nlp_solver.lifted_lp import LiftedLP
    lp = object.__new__(LiftedLP)
    lp.qvars = torch.tensor([0])
    lo, hi = torch.zeros(1, 1), torch.ones(1, 1)
    def cancel():
        raise ctl.RootClosed('ipopt_publish')
    with pytest.raises(ctl.RootClosed):
        lp.obbt(lo, hi, torch.zeros(1, 0), checkpoint=cancel)
    assert torch.equal(lo, torch.zeros_like(lo)) and torch.equal(hi, torch.ones_like(hi))


def test_native_bab_checkpoint_runs_between_completed_nodes():
    import numpy as np
    from nlp_solver.lifted_lp import LiftedLP
    lp = object.__new__(LiftedLP)
    lp.obj, lp.n = object(), 1
    calls = []
    def solve(lo, hi, *args, **kwargs):
        calls.append('node')
        return -1., np.array([.5]), lo, hi, None
    lp.solve_bab_node = solve
    checkpoints = []
    def checkpoint():
        checkpoints.append(1)
        if len(checkpoints) == 2:
            raise ctl.RootClosed('ipopt_publish')
    with pytest.raises(ctl.RootClosed):
        lp.branch_and_bound(torch.zeros(1, 1), torch.ones(1, 1), torch.zeros(1, 0),
                             cutoff=3., checkpoint=checkpoint)
    assert calls == ['node'] and len(checkpoints) == 2


@patch('input_split.root_controller.early_lp')
@patch('input_split.lifted_bound.make_verifier')
def test_late_publication_cannot_turn_a_timeout_into_a_solve(verifier, early):
    r = root(P=3., D=1.)
    def runner(**kwargs):
        ctl.offer(r, 'ipopt', torch.zeros(1), .01)
        r.deadline = time.time() - 1.
        return 'finished late'
    with pytest.raises(ctl.RootDeadline):
        ctl.run_nlp(r, runner, nlp_config=r.nlp_config)
    assert r.values() == (3., 1.)
    assert not r.closes(3.)
    verifier.assert_not_called()


@patch('nlp_solver.compare_solvers')
def test_independent_alm_deadline_does_not_enlarge_legacy_ipopt_budget(compare):
    from input_split.nlp_in_bab import run_root_nlp_and_ipopt
    compare.return_value = SimpleNamespace(
        best_method='none', best_obj=None, best_x=None, local_method='IPOPT',
        local_verified=False, local_obj=None, local_time=0., alm_verified=False,
        alm_obj=None, alm_time=0.)
    r = root()
    config = {'enabled': True, 'enable_alm': True, 'enable_local_solver': True,
              'controller': True, 'timeout': 60., 'ipopt_timeout': 30.}
    run_root_nlp_and_ipopt(r.net, r.dm_l, r.dm_u, r.c, r.rhs, torch.tensor([1]),
                           reference_dict={}, nlp_config=config, initial_verified=False,
                           root_wall_budget=170.)
    passed = compare.call_args.kwargs['config']
    assert passed['timeout'] == 170.
    assert passed['ipopt_timeout'] == 30.
    assert passed['_controller_parallel_ipopt_budget'] == 60.


@pytest.mark.parametrize('publisher', ['ipopt_seed', 'lp_tree', 'alm'])
def test_native_publication_stale_slice_replay_credits_each_gain_once(publisher, capsys):
    """impl3 diffcircles: new native P, captured old P, repeated native P."""
    from input_split.slice_controller import Job, Allocator, receipt_observer
    from nlp_solver.continuation import Continuation
    old, new, dual = 21.3375031369, 20.9903106792, 3.28494220948
    ledger = ctl.ReceiptLedger()
    jobs = [Job(n, k, Continuation(iter(()))) for n, k in
            [('ipopt_seed', 'primal'), ('lp_tree', 'dual'), ('alm', 'primal')]]
    allocator = Allocator(jobs)
    ledger.commit('initial', 0., old, dual)
    observer = receipt_observer(allocator)
    receipts = []
    def observe(name, seconds, gain):
        receipts.append((name, seconds, gain, ledger.P, ledger.D))
        observer(name, seconds, gain)
    ledger.observer = observe
    captured = old, dual
    ledger.commit(publisher + '_publish', .1, new, dual + 1.)
    ledger.commit('lp_tree_slice', .2, *captured)
    ledger.commit(publisher + '_publish', .3, new, dual + 1.)
    assert [r[3:] for r in receipts] == [(new, dual + 1.)] * 3
    assert [r[2] for r in receipts] == pytest.approx([old - new + 1., 0., 0.])
    assert sum(r[1] for r in receipts) == pytest.approx(.6)
    # A real later improvement is still credited to its producer.
    ledger.commit('alm_publish', .4, new - .25, dual + 1.)
    assert sum(r[2] for r in receipts) == pytest.approx(old - new + 1.25)
    if publisher == 'ipopt_seed':
        assert jobs[0].work == pytest.approx(.4)
        assert jobs[0].rate == 0.  # repeat settled its own zero-gain episode
        assert jobs[1].gain == 0. and jobs[2].gain == .25
    lines = capsys.readouterr().out.splitlines()
    assert all('first_feasible=0 first_bound=0' in line for line in lines[1:])


@patch('input_split.lifted_bound.make_verifier')
def test_message_cannot_take_another_producers_pending_dual(verify):
    """Only the producer of the checked bound can commit its improvement."""
    r = root(P=8., D=0.)
    r.ledger.commit('initial', 0., *r.values())
    receipts = []
    r.ledger.observer = lambda n, s, g: receipts.append((n, g))
    r.certified_dual = 2.  # interrupted LP has checked evidence awaiting receipt
    verify.return_value = lambda point: (True, 6., point)
    ctl.offer(r, 'ipopt_seed', torch.zeros(1), .1)
    ctl.poll(r, service=False)
    assert receipts == [('ipopt_seed_publish', 2.)]
    r.ledger.commit_current('lp_tree_slice', .2, r.values)
    assert receipts == [('ipopt_seed_publish', 2.), ('lp_tree_slice', 2.)]
    assert (r.ledger.P, r.ledger.D) == (6., 2.)


@patch('input_split.lifted_bound.make_verifier')
def test_signal_between_capture_and_commit_is_deferred_to_producer_receipt(verify):
    """Use the real signal handler at the former *root.values() race window."""
    import os
    import signal
    from input_split.tree_slice import OwnerInterrupts
    r = root(P=8., D=0.)
    r.ledger.commit('initial', 0., *r.values())
    verify.return_value = lambda point: (True, 6., point)
    receipts = []
    r.ledger.observer = lambda n, s, g: receipts.append((n, g, r.ledger.P, r.ledger.D))
    interrupts = OwnerInterrupts(r)
    def captured_values():
        assert r.ledger.busy
        captured = r.values()
        ctl.offer(r, 'ipopt_seed', torch.zeros(1), .1)
        os.kill(os.getpid(), signal.SIGUSR1)
        assert r.values() == captured  # signal cannot reenter publication
        return captured
    try:
        with r.ledger.publication():
            r.certified_dual = 1.
            r.ledger.commit_current('lp_tree_slice', .2, captured_values)
            assert r.ledger.busy  # nested commits preserve the outer guard
        assert not r.ledger.busy
        ctl.poll(r, service=False)  # deferred inbox evidence remains available
        ctl.offer(r, 'alm', torch.zeros(1), .3)  # duplicate P earns nothing
        ctl.poll(r, service=False)
        assert receipts == [('lp_tree_slice', 1., 8., 1.),
                            ('ipopt_seed_publish', 2., 6., 1.),
                            ('alm_publish', 0., 6., 1.)]
    finally:
        interrupts.finish()
        interrupts.close()


def test_publication_guard_is_restored_when_endpoint_capture_fails():
    ledger = ctl.ReceiptLedger()
    def fail():
        raise ValueError('capture failed')
    with pytest.raises(ValueError, match='capture failed'):
        ledger.commit_current('lp_tree_slice', .1, fail)
    assert not ledger.busy and ledger.seq == 0
