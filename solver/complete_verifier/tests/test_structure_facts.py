import math
import os
import sys
from fractions import Fraction

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from auto_LiRPA import BoundedModule
from nlp_solver.structure import build_structure
from nlp_solver.structure import forms
from nlp_solver.structure.facts import direction_key
from nlp_solver.structure.forms import Form


class Rows(nn.Module):
    # outputs: objective x0^2 + x1^2, affine a, a again, -a, 2*objective + 3, x0 - 10,
    # a fixed-coordinate term x2
    def forward(self, x):
        a = x[:, :1] + 2 * x[:, 1:2]
        obj = x[:, :1] ** 2 + x[:, 1:2] ** 2
        return torch.cat([obj, a, a, -a, 2 * obj + 3, x[:, :1] - 10, x[:, 2:3]], 1)


def report(lower=(-1., -1., 0.5), upper=(1., 1., 0.5)):
    x = torch.tensor([[0., 0., 0.5]], dtype=torch.float64)
    bm = BoundedModule(Rows(), x)
    C = torch.eye(7, dtype=torch.float64).unsqueeze(0)
    rhs = torch.zeros(1, 7, dtype=torch.float64)
    lo = torch.tensor([lower], dtype=torch.float64)
    hi = torch.tensor([upper], dtype=torch.float64)
    return build_structure(bm, x, C, rhs, lo, hi)


def test_classes_curvature_duplicates_opposites_alias_redundancy():
    r = report()
    assert r.objective.cls == 'quad' and r.objective.curvature == 'convex'
    assert r.objective is r.rows[0]
    assert r.rows[1].cls == 'affine'
    assert r.rows[2].duplicate_of == (1, 1, 0)
    assert r.rows[3].opposite_of == (1, -1, 0)
    assert r.rows[4].alias == (2, 3)
    assert r.rows[5].redundant          # x0 - 10 <= 1e-4 holds on the whole box


def test_fixed_coordinate_rows_are_exact_points():
    r = report()
    assert r.rows[6].cls in ('affine', 'const')
    assert r.rows[6].box == (0.5, 0.5)


def test_per_domain_specs_are_skipped_not_guessed():
    x = torch.zeros(2, 3, dtype=torch.float64)
    bm = BoundedModule(Rows(), x[:1])
    C = torch.eye(7, dtype=torch.float64).unsqueeze(0).repeat(2, 1, 1)
    C[1, 1, 1] = 2.0
    r = build_structure(bm, x[:1], C, torch.zeros(2, 7, dtype=torch.float64),
                        -torch.ones(2, 3, dtype=torch.float64), torch.ones(2, 3, dtype=torch.float64))
    assert r.skipped == 'per-domain specs' and not r.rows


class Rows2(nn.Module):
    def forward(self, x):
        a = x[:, :1] + 2 * x[:, 1:2]
        obj = x[:, :1] ** 2 + x[:, 1:2] ** 2
        k = torch.exp(0.3 + 0 * x[:, :1])
        return torch.cat([obj, k * a, k * a, k * obj, x[:, :1] - 0.5], 1)


def _multi(model, m, lower, upper, rhs):
    x = torch.zeros(1, 2, dtype=torch.float64)
    bm = BoundedModule(model, x)
    C = torch.eye(m, dtype=torch.float64).unsqueeze(0).repeat(len(lower), 1, 1)
    f = lambda v: torch.tensor(v, dtype=torch.float64)
    return build_structure(bm, x, C, f(rhs), f(lower), f(upper))


def test_box_facts_use_the_hull_of_all_domains():
    r = _multi(Rows2(), 5, [[-1., -1.], [-1., -1.]], [[0.4, 1.], [1., 1.]], [[0.] * 5] * 2)
    assert not r.rows[4].redundant      # x0 - 0.5 <= 0 holds on domain 0 only
    assert _multi(Rows2(), 5, [[-1., -1.]] * 2, [[0.4, 1.]] * 2, [[0.] * 5] * 2).rows[4].redundant


def test_different_thresholds_are_per_domain_specs():
    rhs = [[0.] * 5, [0., 1., 0., 0., 0.]]
    r = _multi(Rows2(), 5, [[-1., -1.]] * 2, [[1., 1.]] * 2, rhs)
    assert r.skipped == 'per-domain specs'


def test_inexact_rows_make_no_duplicate_alias_or_curvature_claims():
    r = _multi(Rows2(), 5, [[-1., -1.]], [[1., 1.]], [[0.] * 5])
    assert r.rows[1].inexact and r.rows[2].duplicate_of is None
    assert r.rows[3].alias is None and r.rows[3].curvature is None


class InexactPair(nn.Module):
    # r and 2 r with r = sqrt(2) x0 + (x0 x1 + x1 x2 + ... + 1)^2: an inexact coefficient, and a pow atom of a
    # non-affine argument, kept by the walk and expanded by classify
    def forward(self, x):
        s = (x[:, :1] * 0 + 2.0) ** 0.5
        big = ((x[:, :-1] * x[:, 1:]).sum(1, keepdim=True) + 1.0) ** 2
        r = s * x[:, :1] + big
        return torch.cat([x[:, :1], r, 2 * r], 1)


@pytest.mark.parametrize('n, max_terms', [(10, 20), (3, forms.MAX_TERMS)], ids=['expansion_fails', 'expansion_succeeds'])
def test_an_inexact_row_is_not_a_mergeable_pool_row(monkeypatch, n, max_terms):
    monkeypatch.setattr(forms, 'MAX_TERMS', max_terms)     # n = 10: the 55 terms of the pow atom's expansion exceed it
    x = torch.full((1, n), 0.1, dtype=torch.float64)
    bm = BoundedModule(InexactPair(), x)
    r = build_structure(bm, x, torch.eye(3, dtype=torch.float64).unsqueeze(0), torch.zeros(1, 3, dtype=torch.float64),
                        torch.zeros(1, n, dtype=torch.float64), torch.ones(1, n, dtype=torch.float64))
    assert all(f.inexact and f.poly is not None for f in r.forms[1:])        # the repro's premise
    assert r.rows[1].inexact and r.rows[2].inexact
    assert r.pool.stats()['merged'] == 0 and all(row.opaque for row in r.pool.rows)     # exact merges need exact rows


class RoundedAndFloat(nn.Module):
    # sqrt(2) x0 + exp(x1) and 1.4142135623730951 x0 + exp(x1): different functions whose rounded forms coincide
    def forward(self, x):
        s = (x[:, :1] * 0 + 2.0) ** 0.5
        e = torch.exp(x[:, 1:2])
        return torch.cat([x[:, :1], s * x[:, :1] + e, 1.4142135623730951 * x[:, :1] + e], 1)


def test_a_rounded_row_is_not_merged_with_the_row_of_its_float():
    x = torch.full((1, 2), 0.1, dtype=torch.float64)
    r = build_structure(BoundedModule(RoundedAndFloat(), x), x, torch.eye(3, dtype=torch.float64).unsqueeze(0),
                        torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64),
                        torch.ones(1, 2, dtype=torch.float64))
    assert r.rows[1].inexact and not r.rows[2].inexact
    assert r.pool.stats()['merged'] == 0 and sum(row.opaque for row in r.pool.rows) == 1


class Poles(nn.Module):
    # x0, 1/x0, 1/(x0+x1), 1/log(x0), 1/(x0+3)
    def forward(self, x):
        a, b = x[:, :1], x[:, 1:2]
        return torch.cat([a, 1 / a, 1 / (a + b), 1 / torch.log(a), 1 / (a + 3)], 1)


def _poles(lower, upper):
    x = torch.tensor([[1., 0.]], dtype=torch.float64)
    bm = BoundedModule(Poles(), x, bound_opts={'pow_negative_exponent': True})
    C = torch.eye(5, dtype=torch.float64).unsqueeze(0)
    f = lambda v: torch.tensor([v], dtype=torch.float64)
    return build_structure(bm, x, C, torch.zeros(1, 5, dtype=torch.float64), f(lower), f(upper))


def test_pole_free_is_only_claimed_when_proved():
    r = _poles([0.5, -1.], [2., 1.])
    assert r.rows[1].pole_free is True               # 1/x0 on [0.5, 2]
    assert r.rows[2].pole_free is not True           # x0 + x1 ranges over [-0.5, 3]
    assert 'rational' not in r.rows[2].tags
    assert r.rows[3].pole_free is not True           # log(x0) vanishes at 1
    assert r.rows[4].pole_free is True and 'rational' in r.rows[4].tags   # x0 + 3 > 0


def test_pole_free_is_unknown_when_the_box_contains_zero():
    r = _poles([-1., -1.], [2., 1.])
    assert r.rows[1].pole_free is not True


class Shifted(nn.Module):
    def forward(self, x):
        a = x[:, :1] + x[:, 1:2]
        return torch.cat([x[:, :1] ** 2, a, a + 5, 3 * a - 2, 2 - a, 1e-300 * a], 1)


def test_duplicates_and_opposites_carry_scale_and_offset():
    x = torch.zeros(1, 2, dtype=torch.float64)
    bm = BoundedModule(Shifted(), x)
    f = lambda v: torch.tensor([v], dtype=torch.float64)
    r = build_structure(bm, x, torch.eye(6, dtype=torch.float64).unsqueeze(0), torch.zeros(1, 6, dtype=torch.float64),
                        f([-1., -1.]), f([1., 1.]))
    assert r.rows[2].duplicate_of == (1, 1, 5)       # x0 + x1 + 5 = 1 * (x0 + x1) + 5
    assert r.rows[3].duplicate_of == (1, 3, -2)
    assert r.rows[4].opposite_of == (1, -1, 2)
    assert r.rows[1].duplicate_of is None
    assert r.rows[5].duplicate_of == (1, Fraction(1e-300), 0)       # tiny, exact


def test_direction_key_survives_coefficients_below_the_float_range():
    tiny = Form({((0, 1),): Fraction(1, 10 ** 400), ((1, 1),): Fraction(2, 10 ** 400)})
    assert direction_key(tiny) == ((0, 1), (1, 2))


class Quads(nn.Module):
    def forward(self, x):
        a, b, c = x[:, :1], x[:, 1:2], x[:, 2:3]
        return torch.cat([a, a ** 2 + b ** 2, a * b, (a + b) ** 2, (a + 2 * b + 3 * c) ** 2,
                          -a ** 2 - b ** 2, a + b], 1)


def test_lambda_min_is_a_rigorous_lower_bound_and_decides_the_curvature():
    x = torch.zeros(1, 3, dtype=torch.float64)
    bm = BoundedModule(Quads(), x)
    ones = torch.ones(1, 3, dtype=torch.float64)
    r = build_structure(bm, x, torch.eye(7, dtype=torch.float64).unsqueeze(0), torch.zeros(1, 7, dtype=torch.float64),
                        -ones, ones)
    convex, bilinear, dominant, rank_one, concave, affine = r.rows[1:7]
    assert convex.curvature == 'convex' and convex.lambda_min == 2          # Hessian diag(2, 2)
    assert bilinear.curvature == 'indefinite' and -1.0001 < bilinear.lambda_min <= -1   # eigenvalues -1, 1
    assert dominant.curvature == 'convex' and dominant.lambda_min == 0      # [[2, 2], [2, 2]]: diagonally dominant
    # PSD and singular, but neither Gershgorin nor a rounded eigenvalue can prove it: undecided, never 'convex'
    assert rank_one.curvature is None and -1e-12 < rank_one.lambda_min <= 0 and rank_one.rank == 1
    assert concave.curvature == 'concave'
    assert affine.curvature == 'affine' and affine.lambda_min == 0
    assert not hasattr(convex, 'curvature_exact')


def test_curvature_of_a_quadratic_is_never_proved_by_float_sums():
    from nlp_solver.structure.facts import curvature
    # H = [[2e-16 * 3, ...]]: the exact Hessian is diagonally dominant by a margin below float resolution
    third = Fraction(1, 3)
    e = Form({((0, 2),): third, ((1, 2),): third, ((0, 1), (1, 1)): Fraction(2, 3) + Fraction(1, 10 ** 30)})
    lambda_min, kind, _, _ = curvature(e, frozenset({0, 1}))
    assert kind != 'convex' and lambda_min < 0       # H = [[2/3, 2/3 + d], [2/3 + d, 2/3]] has lambda_min = -d


def test_float_toward_rounds_in_the_requested_direction():
    from nlp_solver.structure.facts import _float_toward
    for x in (Fraction(1, 3), Fraction(-1, 3), Fraction(2, 3), Fraction(-2, 3)):
        assert Fraction(_float_toward(x, -float('inf'))) <= x <= Fraction(_float_toward(x, float('inf')))
    assert _float_toward(Fraction(1, 2), float('inf')) == 0.5


def test_point_box_of_an_inexact_row_is_widened():
    from nlp_solver.structure.facts import point_box
    lo = hi = np.array([0.5])
    exact_row = Form({((0, 1),): 1, (): 0})
    assert point_box(exact_row, lo, hi) == (0.5, 0.5)
    inexact_row = Form({((0, 1),): 1}, inexact=True)
    low, high = point_box(inexact_row, lo, hi)
    assert low < 0.5 < high


class Unbounded(nn.Module):
    def forward(self, x):
        a, b = x[:, :1], x[:, 1:2]
        return torch.cat([a, a * b - 1e9, a ** 8 * b ** 8, a + b], 1)


def test_infinite_or_overflowing_bounds_give_no_box_instead_of_nan_or_an_exception():
    x = torch.ones(1, 2, dtype=torch.float64)
    bm = BoundedModule(Unbounded(), x)
    f = lambda v: torch.tensor([v], dtype=torch.float64)
    C, rhs = torch.eye(4, dtype=torch.float64).unsqueeze(0), torch.zeros(1, 4, dtype=torch.float64)
    inf = float('inf')
    r = build_structure(bm, x, C, rhs, f([0., -inf]), f([inf, inf]))
    assert r.rows[1].box is None
    assert r.rows[3].box is None or not any(math.isnan(v) for v in r.rows[3].box)
    r = build_structure(bm, x, C, rhs, f([1e50, 1e50]), f([1e60, 1e60]))      # (1e60)^8 overflows a float
    assert r.rows[2].box is None


class Feasibility(nn.Module):
    def forward(self, x):
        return torch.cat([x[:, :1], x[:, :1] + 10, x[:, :1] ** 2 + 5, x[:, :1] ** 2 - 5], 1)


def test_infeasible_means_violated_everywhere_on_the_box():
    x = torch.zeros(1, 2, dtype=torch.float64)
    bm = BoundedModule(Feasibility(), x)
    ones = torch.ones(1, 2, dtype=torch.float64)
    r = build_structure(bm, x, torch.eye(4, dtype=torch.float64).unsqueeze(0), torch.zeros(1, 4, dtype=torch.float64),
                        -ones, ones)
    assert r.rows[1].infeasible and not r.rows[1].redundant      # x0 + 10 > 0 on [-1, 1]
    assert r.rows[2].infeasible                                  # x0^2 + 5 > 0
    assert r.rows[3].redundant and not r.rows[3].infeasible      # x0^2 - 5 <= 0
