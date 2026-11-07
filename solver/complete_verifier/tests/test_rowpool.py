"""RowPool: ids, exact merge of parallel rows, tau applied once, and the consumers that read it."""
import os
import re
import sys
from fractions import Fraction as F

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import arguments
from auto_LiRPA import BoundedModule
from nlp_solver.structure import RowPool, build_structure
from nlp_solver.structure.rowpool import base_tau, round_down, round_up, spec_tau

TOL = 1e-4


@pytest.fixture(autouse=True)
def solving_tolerance():
    """the verify predicate's tolerance as the solver runs: spec rows <= rhs + constr_viol_tol"""
    arguments.Config.parse_config([], verbose=False)
    arguments.Config['solving']['solving_mode'] = True
    arguments.Config['solver']['nlp']['constr_viol_tol'] = TOL
    yield
    arguments.Config.parse_config([], verbose=False)


class Pairs(nn.Module):
    # 0 objective; 1: a = x0 x1 - x2; 2: -a (opposite); 3: 2a + 1 (same direction as 1); 4: a different row;
    # 5: 0.1 * x0 x1 (not a float coefficient); 6: x0 - 9 (implied by the box)
    def forward(self, x):
        a = x[:, :1] * x[:, 1:2] - x[:, 2:3]
        return torch.cat([x[:, :1], a, -a, 2 * a + 1, x[:, :1] * x[:, 2:3], 0.1 * x[:, :1] * x[:, 1:2], x[:, :1] - 9], 1)


def report(module=Pairs(), rhs=None, m=7, n=3, lo=0.0, hi=1.0, **kwargs):
    x = torch.full((1, n), 0.5, dtype=torch.float64)
    bm = BoundedModule(module, x)
    rhs = torch.zeros(1, m, dtype=torch.float64) if rhs is None else rhs
    return build_structure(bm, x, torch.eye(m, dtype=torch.float64).unsqueeze(0), rhs,
                           torch.full((1, n), lo, dtype=torch.float64), torch.full((1, n), hi, dtype=torch.float64), **kwargs)


def test_parallel_rows_merge_into_one_ranged_row_with_the_smallest_id():
    pool = report().pool
    assert [r.id for r in pool.rows] == [1, 4, 5, 6]                    # rows 2 and 3 joined row 1
    r1 = pool.rows[0]
    assert sorted(m[0] for m in r1.members) == [1, 2, 3]
    assert r1.ex_lo is not None and r1.ex_hi is not None                # an opposite pair is one ranged row
    assert pool.row_of[2] == pool.row_of[3] == pool.row_of[1] == 0
    assert r1.terms == ((((0, 1), (1, 1)), 1), (((2, 1),), -1))          # the representative's x0 x1 - x2, without its constant


def test_merged_bounds_are_exact_and_tau_enters_once():
    tau = spec_tau(7)
    pool = report().pool
    r1 = pool.rows[0]
    # rows (rep 1: q - x2 <= tau; row 2: -(q - x2) <= tau; row 3: 2 (q - x2) + 1 <= tau)
    assert r1.ex_hi == min(F(tau[1]), (F(tau[3]) - 1) / 2)
    assert r1.ex_lo == -F(tau[2])
    assert pool.audit(dict(enumerate(tau))) == []


def test_a_merged_row_is_exactly_the_intersection_of_its_members():
    """Rows c_k q + d_k <= rhs_k + tau of one direction q, with scales of both signs: a point satisfies every member iff it is
    inside the merged ranged row (exact rationals)."""
    from math import prod
    rng = np.random.default_rng(0)
    q_poly = {((0, 1), (1, 1)): 3, ((2, 1),): -2}
    tau = [F(t) for t in spec_tau(6)]
    for _ in range(300):
        k = int(rng.integers(2, 6))
        scales = [F(int(rng.choice([-5, -3, -2, -1, 1, 2, 3, 5])), int(rng.integers(1, 4))) for _ in range(k)]
        consts = [F(int(rng.integers(-5, 6)), int(rng.integers(1, 4))) for _ in range(k)]
        rhs = [0.0] + [float(rng.integers(-8, 9)) / 4 for _ in range(k)]
        polys = [None] + [{**{m: s * c for m, c in q_poly.items()}, (): d} for s, d in zip(scales, consts)]
        pool = RowPool(torch.tensor([rhs], dtype=torch.float64))
        pool.merge(polys, [False] * (k + 1))
        assert len(pool.rows) == 1 and pool.audit(dict(enumerate(tau[:k + 1]))) == []
        row = pool.rows[0]
        for _ in range(30):
            point = [F(int(v), int(rng.integers(1, 8))) for v in rng.integers(-12, 13, 3)]
            value = lambda terms: sum(c * prod(point[v] ** e for v, e in m) for m, c in terms)
            satisfied = all(value(polys[i].items()) <= F(rhs[i]) + tau[i] for i in range(1, k + 1))
            q = value(row.terms)
            assert satisfied == ((row.ex_lo is None or row.ex_lo <= q) and (row.ex_hi is None or q <= row.ex_hi))


def test_audit_catches_a_second_tau():
    tau = spec_tau(7)
    assert report().pool.audit({i: 2 * t for i, t in enumerate(tau)}) != []


def test_audit_catches_a_merged_bound_moved_inward():
    tau = dict(enumerate(spec_tau(7)))
    pool = report().pool
    pool.rows[0].ex_hi -= F(1, 2 ** 60)
    assert ('exact', 1) in pool.audit(tau)
    pool = report().pool
    pool.rows[0].hi = float(np.nextafter(pool.rows[0].hi, -np.inf))
    assert ('float', 1) in pool.audit(tau)


def test_float_bounds_round_outward_once():
    pool = report(rhs=torch.full((1, 7), 0.1, dtype=torch.float64)).pool
    row = pool.rows[pool.row_of[5]]                                     # 0.1 x0 x1 <= 0.1 + tau, in exact rationals
    assert F(row.hi) >= row.ex_hi and F(row.lo) <= row.ex_lo if row.ex_lo is not None else True
    assert F(row.hi) - row.ex_hi < F(2) ** -50
    assert round_up(F(1, 3)) > F(1, 3) > round_down(F(1, 3))
    assert round_up(F(1, 2)) == round_down(F(1, 2)) == 0.5
    assert round_up(F(10) ** 400) == float('inf') and round_down(-F(10) ** 400) == float('-inf')


def test_box_redundant_row_is_kept_and_masked():
    pool = report().pool
    assert [r.masked for r in pool.rows] == [False, False, False, True]    # x0 - 9 <= tau on [0, 1]
    assert pool.rows[pool.row_of[6]].id == 6


def test_opaque_rows_stay_singletons():
    class M(nn.Module):             # a ReLU of the data: no exact form
        def forward(self, x):
            s = torch.relu(x[:, 1:2] - 0.7)
            return torch.cat([x[:, :1], s, s + 1, x[:, :1] * x[:, 1:2]], 1)
    pool = report(M(), m=4, lo=0.5, hi=2.0).pool
    assert [row.opaque for row in pool.rows] == [True, True, False]       # one singleton each, the product row is a row


@pytest.mark.parametrize('threshold', [float('inf'), -float('inf'), float('nan')])
def test_a_row_with_a_threshold_that_is_not_a_number_is_opaque_and_the_rest_stands(threshold):
    """+-inf and nan have no exact value (Fraction(inf) raises): the row stands as an opaque row (it is no constraint, or
    an infeasible one), its id stays, and no other row is lost"""
    rhs = torch.zeros(1, 7, dtype=torch.float64)
    rhs[0, 2] = threshold                              # -a <= threshold: the opposite twin of row 1
    rhs[0, 4] = threshold                              # x0 x2 <= threshold
    rep = report(rhs=rhs)
    assert len(rep.rows) == 7
    assert {r.id for r in rep.pool.rows if r.opaque} == {2, 4}
    twin = rep.pool.rows[rep.pool.row_of[1]]
    assert sorted(m[0] for m in twin.members) == [1, 3] and twin.ex_lo is None      # row 2 no longer bounds it below
    assert rep.pool.audit(dict(enumerate(spec_tau(7)))) == []


def test_laurent_rows_are_not_opaque():
    class M(nn.Module):
        def forward(self, x):
            return torch.cat([x[:, :1], 1 / x[:, 1:2], 1 / x[:, 1:2] + 1, torch.sqrt(x[:, :1]) - x[:, 2:3]], 1)
    pool = report(M(), m=4, lo=0.5, hi=2.0).pool
    assert not any(row.opaque for row in pool.rows)
    assert sorted(len(row.members) for row in pool.rows) == [1, 2]       # 1/x1 and 1/x1 + 1 merge


# ------------------------------------------------------------------------------------------ base rows, one tau
class Base(nn.Module):
    # objective x0 + x1 + 0.1 x0 x1 ; spec row 1: x0 x1 <= 5 ; spec row 2 restates the base row x0 + x1 >= 2
    def forward(self, x):
        a, b = x[:, :1], x[:, 1:2]
        return torch.cat([a + b + 0.1 * a * b, a * b - 5, 2 - a - b], 1)


def test_a_base_row_and_the_spec_row_that_restates_it_are_one_row_with_each_tau_once():
    A = torch.tensor([[-1.0, -1.0], [1.0, 0.0]], dtype=torch.float64)       # -x0 - x1 <= -2 (the twin of spec row 2), x0 <= 1.9
    b = torch.tensor([-2.0, 1.9], dtype=torch.float64)
    rep = report(Base(), m=3, n=2, lo=0.0, hi=2.0, base_rows=(A, b))
    pool, tau = rep.pool, spec_tau(3)
    assert pool.row_of[2] == pool.row_of[3] and sorted(m[0] for m in pool.rows[pool.row_of[2]].members) == [2, 3]
    assert pool.rows[pool.row_of[4]].source == 'base'                   # the new base row: its id follows the spec rows
    assert pool.rows[pool.row_of[2]].ex_hi == F(tau[2]) - 2 and F(tau[2]) == F(base_tau(2)[0])
    expected = {0: 0.0, 1: tau[1], 2: tau[2], 3: base_tau(2)[0], 4: base_tau(2)[1]}
    assert pool.audit({i: t for i, t in expected.items() if i}) == []
    assert report(Base(), m=3, n=2, lo=0.0, hi=2.0).pool.audit(dict(enumerate(tau))) == []
    assert len(pool.rows) == len(report(Base(), m=3, n=2, lo=0.0, hi=2.0).pool.rows) + 1
    assert pool.audit({i: 2 * t for i, t in expected.items() if i}) != []   # a second tau on any row is caught


def test_a_failed_merge_leaves_the_pool_as_it_was(monkeypatch):
    import nlp_solver.structure.rowpool as module
    pool = RowPool(torch.zeros(1, 7, dtype=torch.float64))
    before = (pool.rows, dict(pool.row_of))

    def boom(nz):
        raise RuntimeError
    monkeypatch.setattr(module, '_direction', boom)
    with pytest.raises(RuntimeError):
        pool.merge([None] + [{((0, 1),): 1}] * 6, [False] * 7)
    assert pool.rows is before[0] and pool.row_of == before[1]


def test_no_base_rows_is_no_rows():
    pool = RowPool(torch.zeros(1, 3, dtype=torch.float64), (torch.zeros(0, 2), torch.zeros(0)))
    assert [r.id for r in pool.rows] == [1, 2]


def test_the_pool_can_be_built_before_the_report_and_the_report_completes_it():
    pool = RowPool(torch.zeros(1, 7, dtype=torch.float64))
    assert all(r.opaque for r in pool.rows) and len(pool.rows) == 6      # every spec row alone, until the forms are known
    rep = report(pool=pool)
    assert rep.pool is pool and len(pool.rows) == 4 and pool.stats()['merged'] == 2


# ------------------------------------------------------------------------------------------ what consumers read
def old_relax(rhs):
    """relax_constraint_rhs as it was before the pool owned it"""
    out = rhs.clone()
    out[..., 1:] = out[..., 1:] + TOL
    return out


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_thresholds_are_the_relaxed_rhs_bit_for_bit(dtype):
    g = torch.Generator().manual_seed(0)
    rhs = (torch.randn(5, 6, generator=g, dtype=torch.float64) * 10 ** torch.randint(-3, 4, (5, 6), generator=g)).to(dtype)
    pool = RowPool(rhs[:1])
    before = rhs.clone()
    got = pool.thresholds(rhs)
    assert torch.equal(got, old_relax(rhs)) and got.dtype == dtype
    assert torch.equal(got[:, 0], rhs[:, 0])                             # the objective's cutoff is the caller's
    assert torch.equal(rhs, before)                                      # the caller's tensor is not touched


def test_thresholds_leave_inactive_rows_and_shapes_alone():
    pool = RowPool(torch.zeros(1, 3, dtype=torch.float64))
    rhs = torch.tensor([[float('inf'), float('inf'), 1.0]], dtype=torch.float64)
    assert torch.isposinf(pool.thresholds(rhs)[0, :2]).all() and pool.thresholds(rhs)[0, 2] == 1.0 + TOL
    assert pool.thresholds(None) is None
    one_row = torch.zeros(1, 1, dtype=torch.float64)
    assert RowPool(one_row).thresholds(one_row) is one_row               # no constraint rows: nothing to relax
    with pytest.raises(ValueError):
        pool.thresholds(torch.zeros(1, 4, dtype=torch.float64))


def test_thresholds_outside_solving_mode_are_the_rhs():
    arguments.Config['solving']['solving_mode'] = False
    rhs = torch.zeros(1, 3, dtype=torch.float64)
    assert RowPool(rhs).thresholds(rhs) is rhs and RowPool(rhs).tolerance == 0.0


def test_clip_reads_the_pool_and_relaxes_the_constraint_row_once():
    from input_split.clip import clip_domains
    # row 0: x <= -inf+ (no cutoff), row 1: x - 0.5 <= rhs. With rhs = 0 the box [0, 1] is clipped to [0, 0.5 + tau].
    rhs = torch.tensor([[float('inf'), 0.0]], dtype=torch.float64)
    lA = torch.tensor([[[0.0], [1.0]]], dtype=torch.float64)
    lbias = torch.tensor([[0.0, -0.5]], dtype=torch.float64)
    lo, hi = torch.zeros(1, 1, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64)
    pool = RowPool(rhs)
    _, up = clip_domains(lo.clone(), hi.clone(), rhs, lA, lbias, rowpool=pool)
    assert float(up) == pytest.approx(0.5 + TOL, abs=1e-12)           # the raw rhs would clip to 0.5
    with pytest.raises(ValueError, match='RowPool'):
        clip_domains(lo.clone(), hi.clone(), rhs, lA, lbias)           # solving mode without a pool


def test_filter_reads_the_pool_and_discards_a_domain_only_beyond_the_relaxed_threshold():
    from input_split.branching_domains import UnsortedInputDomainList
    rhs = torch.tensor([[float('inf'), 1.0]], dtype=torch.float64)
    pool = RowPool(rhs)
    lb = torch.tensor([[0.0, 1.0 + 0.5 * TOL], [0.0, 1.0 + 1.5 * TOL]], dtype=torch.float64)    # row 1 lower bounds
    thr = rhs.expand(2, -1)
    box = torch.zeros(2, 1, dtype=torch.float64), torch.ones(2, 1, dtype=torch.float64)
    kept = UnsortedInputDomainList.get_remaining_index(2, lb, thr, *box, check_dm_lbs=True, check_input_boxes=False,
                                                       rowpool=pool)
    assert kept.tolist() == [0]                                          # inside the tolerance band: kept, beyond: discarded
    with pytest.raises(ValueError, match='RowPool'):                     # both would be discarded on the raw rhs
        UnsortedInputDomainList.get_remaining_index(2, lb, thr, *box, check_dm_lbs=True, check_input_boxes=False)


def test_the_pool_owns_the_tolerance():
    """The verify predicate's tolerance is read in ONE place for bounds (rowpool.spec_tau / base_tau); the spec handler is the
    source of the standard-form base rows and relaxes them there, before any model exists. A new caller of the relaxation
    functions would apply the tolerance a second time or read the raw rhs: this table pins the callers."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    pat = re.compile(r'\b(relax_constraint_rhs|relax_input_constraint_rhs|relax_input_constraints)\(')
    found = {}
    for dp, dn, fn in os.walk(root):
        rel = os.path.relpath(dp, root)
        if rel.split(os.sep)[0] in ('tests', 'analysis_outputs') or any(c.startswith('.') and c != '.' for c in rel.split(os.sep)):
            continue
        for f in fn:
            if not f.endswith('.py'):
                continue
            with open(os.path.join(dp, f)) as fh:
                n = sum(1 for line in fh if pat.search(line) and not line.lstrip().startswith(('def ', '#')))
            if n:
                found[os.path.normpath(os.path.join(rel, f))] = n
    assert found == {'nlp_solver/structure/rowpool.py': 2,      # spec_tau, base_tau
                     'input_split/utils.py': 1,                 # relax_input_constraints -> relax_input_constraint_rhs
                     'specifications.py': 1}                    # the base rows' source
