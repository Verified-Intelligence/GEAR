"""M3 (input_split/m3_reduce.py, nlp_solver/structure/m3.py): exact pattern detection, outward propagation that
keeps every feasible point, cutoff semantics, covering children, and the queue commit."""
from fractions import Fraction as F
from types import SimpleNamespace
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # test_batch_branch_and_bound, whatever the cwd
import arguments
from nlp_solver.structure.forms import Form
from nlp_solver.structure.m3 import M3Pattern
from input_split.m3_reduce import propagate, children, continue_cover, setup

DT = torch.float64


def affine(g, k=0):
    return Form({(): k, **{((j, 1),): v for j, v in g.items()}})


def square(k=1, edges=((0, 1), (2, 3))):
    """k - sum (x_i - x_j)^2 <= rhs"""
    q = {(): k}
    for i, j in edges:
        q[((i, 2),)] = -1
        q[((j, 2),)] = -1
        q[((i, 1), (j, 1))] = 2
    return Form(q)


def pattern(forms=None, base=None):
    return M3Pattern(forms or [affine({4: 1}), square()], 5, base, 'cpu')


def run(p, l, u, rhs=None, **kwargs):
    l, u = torch.as_tensor(l, dtype=DT).reshape(-1, 5), torch.as_tensor(u, dtype=DT).reshape(-1, 5)
    rhs = torch.zeros(len(l), 2, dtype=DT) if rhs is None else torch.tensor(rhs, dtype=DT)
    return propagate(p, l, u, rhs, **kwargs)


def test_detector_proves_the_identity_and_rejects_inexact_or_unknown_rows():
    bad = square()
    bad.poly[((0, 1), (1, 1))] = F(7, 4)
    inexact = affine({0: 1})
    inexact.inexact = True
    p = pattern([affine({4: 1}), square(), bad, inexact])
    assert p.H == 2 and p.F == 1 and not p.complete
    assert p.census['unknown quadratic identity'] == 1 and p.census['inexact/unsupported'] == 1
    assert not pattern([square()]).affine_objective and not pattern([square()]).complete
    assert pattern().complete


def test_sign_ambiguity_never_chooses_a_branch():
    out = run(pattern(), [-1, 0, 0, 0, 0], [1, 0, 0, 0, 2])
    assert out.live.all() and out.lo[0, 0] == -1 and out.hi[0, 0] == 1
    assert out.holes.sum() > 0


@pytest.mark.parametrize('sign', [-1, 1])
def test_radius_equality_and_zero_width_are_retained(sign):
    x = [sign, 0, 0, 0, 0]
    out = run(pattern(), x, x)
    assert out.live.all() and (out.lo <= torch.tensor(x)).all() and (out.hi >= torch.tensor(x)).all()


def test_a_tolerated_order_row_is_not_an_exact_sort():
    A = torch.tensor([[[1., -1., 0., 0., 0.]]], dtype=DT)
    p = pattern([affine({4: 1}), square(k=F(1, 2 ** 44))], (A, torch.tensor([[-1e-6]], dtype=DT)))
    x = torch.tensor([[5e-7, 0, 0, 0, 1]], dtype=DT)
    out = propagate(p, x, x, torch.zeros(1, 2, dtype=DT))
    assert out.live.all() and (out.lo <= x).all() and (out.hi >= x).all()


def test_distinct_domain_base_rows_are_not_intersected():
    A = torch.zeros(2, 1, 5, dtype=DT)
    A[0, 0, 0], A[1, 0, 0] = 1, -1
    p = pattern([affine({4: 1})], (A, torch.zeros(2, 1, dtype=DT)))
    assert p.census['per-domain base'] == 1 and not p.complete and p.R == 1
    x = torch.tensor([[-1., 0, 0, 0, 1], [1., 0, 0, 0, 1]], dtype=DT)
    out = propagate(p, x, x, torch.zeros(2, 1, dtype=DT))
    assert out.live.all() and torch.equal(out.lo, x) and torch.equal(out.hi, x)


def test_product_zero_crossing_and_positive_inverse():
    p = pattern([affine({4: 1}), Form({((0, 1), (1, 1)): -1, (): 1})])
    out = run(p, [-2, -1, 0, 0, 0], [2, 1, 0, 0, 2])
    assert out.lo[0, 0] == -2 and out.hi[0, 0] == 2
    out = run(p, [0, 1, 0, 0, 0], [2, 2, 0, 0, 2])
    assert 0 < out.lo[0, 0] <= .5 and out.live.all()


def test_contradiction_and_cutoff_are_distinguished():
    p = pattern([affine({4: 1}), affine({0: 1}, 1)])
    out = run(p, [0] * 5, [1] * 5, cutoff=torch.tensor([.75], dtype=DT))
    assert not out.live.any() and out.lb.item() == .75          # proves only f >= cutoff
    out = run(p, [0] * 5, [1] * 5)
    assert not out.live.any() and out.lb.item() == math.inf
    p = pattern([affine({4: 1}), affine({0: 1}, -1)])
    out = run(p, [0] * 5, [1] * 5, cutoff=torch.tensor([.25], dtype=DT))
    assert out.live.all() and .25 <= out.hi[0, 4] < .251


def test_nonfinite_inputs_and_arithmetic_do_not_prune():
    p = pattern()
    out = run(p, [-math.inf, 0, 0, 0, 0], [math.inf, 0, 0, 0, 1])
    assert out.live.all() and out.lb.item() == -math.inf
    out = run(p, [-1e308, 0, -1e308, 0, 0], [1e308, 0, 1e308, 0, 1])
    assert out.live.all() and not torch.isnan(out.lo).any()
    out = run(p, [-1, 0, 0, 0, 0], [1, 0, 0, 0, 1],
              floor=torch.tensor([math.nan], dtype=DT), cutoff=torch.tensor([math.nan], dtype=DT))
    assert out.live.all() and not torch.isnan(out.lb).any()


def test_expired_deadline_publishes_only_completed_sweeps():
    out = run(pattern(), [0] * 5, [1] * 5, deadline=0.)
    assert out.sweeps == 0 and out.live.all() and torch.equal(out.lo, torch.zeros(1, 5, dtype=DT))


def test_affine_contractions_contain_exact_feasible_points():
    # cancellation, mixed signs, dyadic rows: every accepted point evaluated in rational arithmetic
    rng = torch.Generator().manual_seed(17)
    points = (torch.randint(-16, 17, (96, 5), generator=rng) / 8).double()
    coeff = [F(3, 8), F(-7, 4), F(1, 2), F(-1, 8), F(0)]
    p = pattern([affine({4: 1}), affine(dict(enumerate(coeff)))])
    h = [sum(c * F(float(x)) for c, x in zip(coeff, point)) for point in points]
    rhs = torch.tensor([[0, float(v) + 1e-6] for v in h], dtype=DT)
    out = propagate(p, points - .5, points + .5, rhs)
    assert out.live.all() and (out.lo <= points).all() and (out.hi >= points).all()


@pytest.mark.parametrize('product', [False, True])
def test_rational_feasible_points_remain_in_sign_and_product_boxes(product):
    rng = torch.Generator().manual_seed(912)
    x = (torch.randint(-16, 17, (128, 5), generator=rng) / 8).double()
    form = Form({((0, 1), (1, 1)): -1, ((4, 1),): 1}) if product else square(k=1)
    p = pattern([affine({4: 1}), form])

    def evaluate(point):
        point = [F(float(v)) for v in point]
        value = F(0)
        for mon, c in form.poly.items():
            term = F(c)
            for j, e in mon:
                term *= point[j] ** e
            value += term
        return float(value) + 1e-6
    rhs = torch.tensor([[0., evaluate(point)] for point in x], dtype=DT)
    out = propagate(p, x - .25, x + .25, rhs)
    assert out.live.all() and (out.lo <= x).all() and (out.hi >= x).all()


def test_random_boxes_keep_every_sampled_feasible_point_under_the_cutoff():
    """outward replay: points sampled in random boxes that satisfy the rows and f <= cutoff stay in the result"""
    rng = torch.Generator().manual_seed(5)
    p = pattern([affine({4: 1}), square(k=1), affine({0: 1, 2: -1}, F(-1, 4)),
                 Form({((1, 1), (3, 1)): 1, ((4, 1),): -1})])
    lo = torch.rand(64, 5, generator=rng, dtype=DT) * 2 - 1
    hi = lo + torch.rand(64, 5, generator=rng, dtype=DT) * 2
    rhs = torch.zeros(64, 4, dtype=DT)
    cutoff = torch.full((64,), .8, dtype=DT)
    out = propagate(p, lo, hi, rhs, cutoff=cutoff)
    pts = lo[:, None] + (hi - lo)[:, None] * torch.rand(64, 4000, 5, generator=rng, dtype=DT)
    x = pts
    rows = torch.stack([1 - (x[..., 0] - x[..., 1]) ** 2 - (x[..., 2] - x[..., 3]) ** 2,
                        x[..., 0] - x[..., 2] - .25, x[..., 1] * x[..., 3] - x[..., 4]], -1)
    feasible = (rows <= 0).all(-1) & (x[..., 4] <= .8)
    assert feasible.sum() > 100
    inside = ((pts >= out.lo[:, None]) & (pts <= out.hi[:, None])).all(-1)
    assert inside[feasible].all() and out.live[feasible.any(1)].all()
    assert (out.lb[:, None].expand_as(feasible)[feasible] <= x[..., 4][feasible]).all()


def test_full_batch_children_cover_and_inherit_the_floor():
    p = pattern()
    lo, hi = torch.zeros(64, 5, dtype=DT), torch.ones(64, 5, dtype=DT) * 2
    rhs = torch.tensor([[0., 2.]], dtype=DT).expand(64, -1)       # inactive square row: literal overlap and ancestry
    out = children(p, lo, hi, rhs, torch.ones(64, dtype=DT) * .1, torch.tensor(1., dtype=DT))
    assert len(out.lo) == 128 and torch.equal(out.index, torch.arange(64).repeat(2))
    assert (out.lb >= .1).all() and torch.equal(out.lo[:64], lo)
    assert (out.hi[64:] >= hi.minimum(torch.tensor([2, 2, 2, 2, 1.], dtype=DT)) - 1e-14).all()
    assert (out.hi[:64] >= out.lo[64:]).all()


def test_failed_midpoint_keeps_both_covering_sides():
    p = pattern()
    l = torch.tensor([[-math.inf, 0., 0., 0., 0.]], dtype=DT)
    u = torch.tensor([[math.inf, 0., 0., 0., 1.]], dtype=DT)
    out = children(p, l, u, torch.tensor([[0., 2.]], dtype=DT), torch.tensor([-.5], dtype=DT),
                   torch.tensor(math.inf, dtype=DT))
    assert out.live.all() and not torch.isnan(out.lo).any() and not torch.isnan(out.hi).any()
    assert (out.lo <= l.repeat(2, 1)).all() and (out.hi >= u.repeat(2, 1)).all() and (out.lb >= -.5).all()


def test_fairness_guard_transports_counters():
    l, u = torch.zeros(2, 5, dtype=DT), torch.ones(2, 5, dtype=DT)
    u[:, 4] = 10                      # the objective is widest but absent from the square score
    out = children(pattern(), l, u, torch.tensor([[0., 2.]], dtype=DT).repeat(2, 1), torch.zeros(2, dtype=DT),
                   torch.tensor(math.inf, dtype=DT), counter=torch.tensor([100, 100]), history=SimpleNamespace())
    assert (out.hi[:2, 4] == 5).all() and (out.lo[2:, 4] == 5).all()
    assert torch.equal(out.counter, torch.zeros(4, dtype=torch.long))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_queue_commit_retains_untreated_boxes_rows_and_the_discarded_floor(monkeypatch, dtype):
    from history import SplitHistory
    from nlp_solver.structure import RowPool
    from test_batch_branch_and_bound import _setup_mock_config
    config = _setup_mock_config()
    config['solving']['solving_mode'] = True
    monkeypatch.setattr(arguments, 'Config', config)
    monkeypatch.setitem(arguments.Globals, 'discarded_threshold_min', .075)
    lo, hi = torch.zeros(2, 5, dtype=dtype), torch.ones(2, 5, dtype=dtype) * 2
    rhs = torch.tensor([[1., 2.]], dtype=dtype).repeat(2, 1)
    rows = (torch.arange(10, dtype=DT).reshape(2, 1, 5), torch.tensor([[1.], [2.]], dtype=DT))

    class Queue:
        def __init__(self):
            self.history = SplitHistory()
            self.history.m3_pattern, self.history.m3_deadline = pattern(), math.inf
            self.history.rowpool = RowPool(rhs)
            self.history.lifted_branch_score = torch.ones(1, 5, dtype=DT)
            self.history.lifted_split_point = torch.zeros(1, 5, dtype=DT)
            self.keys = torch.tensor([.05], dtype=DT)             # an untreated box
        def add(self, lb, lower, upper, cs, threshold, **kw):
            self.added = (lb, lower, upper, kw)
            self.keys = torch.cat((self.keys, lb[:, 0].double()))
        def __len__(self):
            return len(self.keys)
        def _objective_key(self, *args):
            return self.keys
    q = Queue()
    parent_lb = torch.zeros(2, 2, dtype=dtype)
    parent_lb[:, 0] = .1
    ret = (parent_lb, lo, hi, torch.eye(2, dtype=DT)[None].repeat(2, 1, 1), rhs, rows, None, torch.tensor([2, 2]),
           None, None, None)
    result = continue_cover(q, ret, torch.tensor([1.], dtype=DT), SimpleNamespace(visited=0), {})
    assert .0499 < result.item() <= .05                            # no untreated region is lost
    assert arguments.Globals['discarded_threshold_min'] == .075
    lb, lower, upper, kw = q.added
    assert len(lb) == 4 and (lower >= lo.repeat(2, 1)).all() and (upper <= hi.repeat(2, 1)).all()
    assert torch.equal(kw['constraints'][0], rows[0].repeat(2, 1, 1))
    assert (upper[:2, 0] <= 1).all() and (lower[2:, 0] >= 1).all()       # the cast keeps the split scope
    assert q.history.lifted_branch_score is None and q.history.lifted_split_point is None


def test_float32_empty_cutoff_cover_passes_the_literal_gap(monkeypatch):
    from nlp_solver.structure import RowPool
    from test_batch_branch_and_bound import _setup_mock_config
    config = _setup_mock_config()
    config['solving']['solving_mode'] = True
    monkeypatch.setattr(arguments, 'Config', config)
    monkeypatch.setitem(arguments.Globals, 'discarded_threshold_min', None)
    rhs = torch.tensor([[1., 2.]], dtype=torch.float32)

    class Empty:
        history = SimpleNamespace(m3_pattern=pattern(), m3_deadline=math.inf, rowpool=RowPool(rhs),
                                  filter_opt_variables=lambda index, count: None)
        def __len__(self):
            return 0
        def add(self, lb, *args, **kwargs):
            assert len(lb) == 0
    x = torch.tensor([[0., 0., 0., 0., 1.]], dtype=torch.float32)
    ret = (torch.zeros(1, 2, dtype=torch.float32), x, x, torch.eye(2, dtype=torch.float32)[None], rhs, None, None,
           torch.tensor([2]), None, None, None)
    D = continue_cover(Empty(), ret, torch.tensor([1.], dtype=torch.float32), SimpleNamespace(visited=0), {})
    assert D.dtype == torch.float32 and 0 <= 1 - float(D) <= 1e-4


def test_setup_declines_incomplete_patterns_and_alpha_state():
    history = SimpleNamespace(use_alpha=False, use_beta=False)
    report = SimpleNamespace(skipped=None, forms=[affine({4: 1}), square()])
    setup(history, report, 5, None, 'cpu', math.inf)
    assert history.m3_pattern is not None
    for forms in ([square(), square()], [affine({4: 1}), affine({0: 1})]):     # non-affine objective; no squares
        setup(history, SimpleNamespace(skipped=None, forms=forms), 5, None, 'cpu', math.inf)
        assert history.m3_pattern is None
    history.use_alpha = True
    setup(history, report, 5, None, 'cpu', math.inf)
    assert history.m3_pattern is None


def test_flag_is_registered_default_false():
    config = arguments.ConfigHandler()
    config.parse_config([])
    assert config['solver']['nlp']['m3_reduce'] is False
    config.parse_config(['--nlp_m3_reduce'])
    assert config['solver']['nlp']['m3_reduce'] is True


def test_ledger_gives_the_cover_step_the_bound_gain(capsys):
    from input_split.batch_branch_and_bound import OpLedger
    ledger, timer = OpLedger(), SimpleNamespace(time_sum={})
    ledger.start_bab(timer)
    ledger.P, ledger.D = 2.0, 1.0
    timer.time_sum = {'pickout': .1, 'm3': .2}            # an M3 iteration: no split, no bounding
    ledger.iteration(timer, 4, 2.0, 1.5)
    lines = {line.split()[1]: line for line in capsys.readouterr().out.splitlines() if line.startswith('[Op] ')}
    assert set(lines) == {'pickout', 'm3'}
    assert ' dD=0.5 ' in lines['m3'] and ' dD=0 ' in lines['pickout']
