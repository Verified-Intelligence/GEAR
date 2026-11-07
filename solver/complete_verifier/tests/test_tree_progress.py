"""Potential is invariant to splitting and independent of the reward ledger."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from input_split.tree_progress import GapMass, ProgressForecast, domain_snapshot
from input_split.slice_controller import Allocator, Job
from nlp_solver.continuation import Continuation


def test_split_cannot_mint_mass_reduction_and_checked_bounds_can():
    m = GapMass([0., 2.], [1., 2.], 10.)
    m.add([[0., 2.]], [[1., 2.]], [0.])
    assert m.snapshot() == (10., 10.)
    m.add([[0., 2.]], [[1., 2.]], [0.], -1)
    m.add([[0., 2.], [.25, 2.]], [[.25, 2.], [1., 2.]], [0., 0.])
    assert m.snapshot() == (10., 10.)
    m.add([[0., 2.]], [[.25, 2.]], [0.], -1)
    m.add([[0., 2.]], [[.25, 2.]], [8.])
    assert m.snapshot() == (10., 8.)


def test_unweighted_leaf_sum_is_not_split_invariant():
    assert sum([10., 10.]) > sum([10.])
    m = GapMass([0.], [1.], 10.)
    m.add([[0.], [.5]], [[.5], [1.]], [0., 0.])
    assert m.mass == 10.


@pytest.mark.parametrize('bound', [math.inf, -math.inf, math.nan])
def test_nonfinite_certificate_is_unknown(bound):
    m = GapMass([0.], [1.], 10.)
    m.add([[0.]], [[1.]], [bound])
    assert m.snapshot() is None


def test_underflow_is_unknown_and_zero_mass_does_not_certify_closure():
    m = GapMass(np.zeros(2000), np.ones(2000), 1.)
    m.add(np.zeros((1, 2000)), np.full((1, 2000), .5), [0.])
    assert m.snapshot() is None
    f = ProgressForecast()
    f.account(1., (10., 5.))
    f.account(1., (10., 0.))
    assert f.mass == 0.  # numerical zero cannot be used as a closure certificate


def test_forecast_carries_zero_prefix_pending_cost_and_rebases_external_P():
    f = ProgressForecast()
    f.account(2., (10., 10.))
    f.account(3., (10., 10.))
    f.account(5., (10., 5.))
    assert f.rate == .5  # all 10 paid units, including initial/flat work
    assert f.score(4.) == .2
    f.account(10., (10., 5.))
    assert f.score(4.) == .1
    f.account(1., (9., 4.))  # incumbent gain belongs to its primal producer
    assert f.score(4.) == 0.
    f.account(9., (9., 2.))
    assert f.previous_work == 20.
    assert f.rate == .1


def test_measured_mass_forecast_can_select_a_tree_with_zero_global_gain():
    def work():
        while True:
            yield
    lp = Job('lp_tree', 'dual', Continuation(work()))
    gpu = Job('gpu_tree', 'dual', Continuation(work()))
    a = Allocator([lp, gpu])
    for j in (lp, gpu):
        j.snapshot = (10., 10.)
        a.account(j, 'initial', 1., 0., False)
    a._calendar()
    for j in (lp, gpu):
        j.ticket_work = 100.
    lp.snapshot = (10., 1.)
    a.account(lp, 'exploit', 1., 0., False)
    assert a.choose(P=10., D=0.)[0] is lp
    assert lp.transaction_gain == lp.rate == 0.
    assert lp.progress.score(10.) > 0.
    for j in (lp, gpu):
        j.continuation.close()


def test_incremental_domain_mass_matches_rescan_and_cutoff_rebase():
    from input_split.branching_domains import UnsortedInputDomainList
    d = UnsortedInputDomainList(output_device='cpu', storage_depth=1, use_split_idx=False)
    def add(lo, hi, lb):
        d.add(torch.tensor(lb)[:, None], torch.tensor(lo)[:, None], torch.tensor(hi)[:, None],
              torch.ones(len(lb), 1, 1), torch.full((len(lb), 1), 10.))
    add([0., .5], [.5, 1.], [0., 2.])
    r = SimpleNamespace(values=lambda: (10., 0.), dm_l=torch.tensor([[0.]]), dm_u=torch.tensor([[1.]]))
    r.dm_l, r.dm_u = r.dm_l.repeat(2, 1), r.dm_u.repeat(2, 1)
    assert domain_snapshot(d, r)[:2] == (10., 9.)
    batch = d.pick_out_batch(1, device='cpu')
    lo, hi, lb = batch[2].item(), batch[3].item(), batch[1].item()
    midpoint = (lo+hi)/2
    add([lo, midpoint], [midpoint, hi], [lb+1, lb+1])
    m = d.controller_measure.mass
    d.controller_measure = None
    assert domain_snapshot(d, r)[1] == pytest.approx(m)
    r.values = lambda: (9., 0.)
    assert domain_snapshot(d, r)[0] == 9.


def test_incumbent_change_invalidates_rate_before_next_tree_sample():
    def work():
        while True:
            yield
    j = Job('lp_tree', 'dual', Continuation(work()))
    a = Allocator([j])
    j.snapshot = (10., 10.)
    a.account(j, 'initial', 1., 0., False)
    j.snapshot = (10., 5.)
    a.account(j, 'exploit', 1., 0., False)
    a._calendar()
    j.ticket_epoch, j.ticket_work = a.epoch, 100.
    assert j.progress.velocity > 0.
    a.choose(P=9., D=0.)
    assert j.progress.velocity == 0.
    assert j.work == 2.
    j.continuation.close()


def test_own_frontier_progress_is_visible_below_global_D():
    f = ProgressForecast()
    f.account(1., (20., 10., 4.))
    f.account(1., (20., 10., 8.))
    assert f.frontier.velocity == 2.
    assert f.score(5.) == pytest.approx(5./6.)
    # A flat interval retains the observation and pays its pending denominator.
    f.account(2., (20., 10., 8.))
    assert f.score(5.) == pytest.approx(5./12.)


def test_tiny_remaining_volume_cannot_amplify_forecast():
    f = ProgressForecast()
    f.account(1., (10., 1.))
    f.account(1., (10., 1e-20))
    f.account(1., (10., 1e-21))
    assert f.score(10.) < 1e-18


def test_tiny_frontier_changes_keep_the_layer_and_its_cost_pending():
    f = ProgressForecast()
    f.account(1., (20., 10., 4.), tolerance=.01)
    f.account(1., (20., 10., 8.), tolerance=.01)
    old = f.frontier.rate
    f.account(1., (20., 10., 8.+1e-12), tolerance=.01)
    assert f.frontier.rate == old
    assert f.frontier.pending == 1.
    f.account(1., (20., 10., 8.02), tolerance=.01)
    assert f.frontier.previous_work == 2.
    assert f.frontier.rate == pytest.approx(.01)


def test_unknown_mass_does_not_hide_certified_frontier_progress():
    f = ProgressForecast()
    f.account(1., (20., None, 4.))
    f.account(1., (20., None, 8.))
    assert f.mass is None
    assert f.score(5.) == pytest.approx(5./6.)


def test_external_cutoff_rebases_without_renewing_opportunity_or_bound_reward():
    f = ProgressForecast()
    f.account(1., (20., 100., 4.))
    f.account(1., (20., 50., 8.))
    old_rate = f.frontier.rate
    f.account(1., (19., 20., 9.))
    assert f.anchor == 100.
    assert f.velocity == 0.
    assert f.frontier.rate == old_rate
    assert f.frontier.pending == 1.
    assert f.frontier.gain == 0.


def test_current_tree_layer_supersedes_stale_global_gain_forecast():
    def work():
        while True:
            yield
    main = Job('gpu_tree', 'dual', Continuation(work()))
    lp = Job('lp_tree', 'dual', Continuation(work()))
    a = Allocator([main, lp])
    main.snapshot = (10., 4., 6.)
    a.account(main, 'initial', 1., 1000., True)
    lp.snapshot = (10., 10., 0.)
    a.account(lp, 'initial', 1., 0., False)
    a._calendar()
    for j in (main, lp):
        j.ticket_work = 100.
    lp.snapshot = (10., 5., 5.)
    a.account(lp, 'exploit', 1., 0., False)
    # The independent global D=6 hides the LP's own certified 0 -> 5 rise.
    # The main tree's latest layer is flat, despite its old cheap global rise.
    assert main.score == 1000.
    assert a.choose(P=10., D=6.)[0] is lp
    assert main.score == 1000.  # observation history and rewards stay intact
    for j in (main, lp):
        j.continuation.close()
