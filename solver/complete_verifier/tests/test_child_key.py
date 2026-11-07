"""The key of a child domain (input_split/batch_branch_and_bound.py: _plane_minimum_down, _raise_child_keys, WP15-pre).

The key is max(parent's bound, minimum of the parent's objective plane over the clipped child box); the plane minimum is rounded
toward -inf, so that it is never above the exact minimum, which is what makes the key (and the reported dual) a bound."""
import inspect
import math
import os
import random
import sys
import textwrap
from fractions import Fraction

import pytest
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import input_split.batch_branch_and_bound as bab  # noqa: E402

CV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUNDING = 'return torch.nextafter(plane_min - margin, torch.full_like(plane_min, -math.inf))'


def _mutant(replacement):
    source = textwrap.dedent(inspect.getsource(bab._plane_minimum_down))
    assert ROUNDING in source
    namespace = dict(vars(bab))
    exec(source.replace(ROUNDING, replacement), namespace)  # noqa: S102
    return namespace['_plane_minimum_down']


def _exact(a, b, lo, hi):
    """(exact minimum of a x + b over the box, exact sum |a| (|c| + r) + |b|) of every row, as rationals"""
    minima, scales = [], []
    for i in range(a.shape[0]):
        low, scale = Fraction(float(b[i])), abs(Fraction(float(b[i])))
        for j in range(a.shape[1]):
            aj, l, u = Fraction(float(a[i, j])), Fraction(float(lo[i, j])), Fraction(float(hi[i, j]))
            low += aj * (l if aj > 0 else u)
            scale += abs(aj) * (abs(l + u) / 2 + (u - l) / 2)
        minima.append(low)
        scales.append(scale)
    return minima, scales


def _cases(dtype, seed, trials=250, rows=20):
    """random planes and boxes of every scale; half of the planes carry the bias that cancels their minimum, which leaves a
    plane minimum near 0 against a rounding error that scales with the terms"""
    gen = random.Random(seed)
    torch.manual_seed(seed)
    for trial in range(trials):
        n = gen.choice([1, 2, 3, 5, 10, 30])
        a = torch.randn(rows, n, dtype=dtype) * 10 ** gen.uniform(-6, 6) * 10 ** gen.uniform(-3, 3)
        lo = torch.randn(rows, n, dtype=dtype) * 10 ** gen.uniform(-3, 3) + gen.choice([0, 1e4, -1e4])
        hi = lo + torch.rand(rows, n, dtype=dtype) * 10 ** gen.uniform(-6, 2)
        b = torch.randn(rows, dtype=dtype) * 10 ** gen.uniform(-6, 6) * gen.choice([1, 1e3, 1e-3])
        if trial % 2:
            b = -torch.where(a > 0, a * lo, a * hi).sum(1)
        yield a, b, lo, hi


def _worst(plane_minimum_down, dtype, seed):
    """(cases where the plane minimum is above the exact one, the largest (exact - plane minimum) / (u sum |a| (|c| + r) + |b|), cases)"""
    u = torch.finfo(dtype).eps / 2
    violations, loosest, cases = 0, 0.0, 0
    for a, b, lo, hi in _cases(dtype, seed):
        down = plane_minimum_down(a[:, None], b[:, None], lo, hi)
        minima, scales = _exact(a, b, lo, hi)
        for i in range(a.shape[0]):
            cases += 1
            gap = minima[i] - Fraction(float(down[i]))
            violations += gap < 0
            loosest = max(loosest, float(gap / (scales[i] * Fraction(u))) / (a.shape[1] + 4))
    return violations, loosest, cases


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_plane_minimum_is_never_above_the_exact_minimum(dtype):
    violations, loosest, cases = _worst(bab._plane_minimum_down, dtype, seed=0)
    assert cases == 5000 and violations == 0
    # and it is a bound worth having: at most 2 (n + 4) u of the terms below the exact minimum, never -inf
    assert loosest <= 2.5


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
@pytest.mark.parametrize('name,replacement', [
    ('nearest rounding', 'return plane_min'),
    ('one step and no margin', 'return torch.nextafter(plane_min, torch.full_like(plane_min, -math.inf))'),
])
def test_the_fuzz_catches_a_key_that_is_not_rounded_down(dtype, name, replacement):
    violations, _, _ = _worst(_mutant(replacement), dtype, seed=0)
    assert violations > 0, name


def test_only_the_objective_row_and_nonempty_boxes_with_a_finite_plane_are_raised():
    lA = torch.tensor([[[1., 0.], [5., 5.]], [[1., 0.], [5., 5.]], [[1., 0.], [5., 5.]], [[1., 0.], [5., 5.]]], dtype=torch.float64)
    lbias = torch.tensor([[0., 7.]] * 4, dtype=torch.float64)
    lo = torch.tensor([[2., 0.], [2., 0.], [-9., 0.], [2., 0.]], dtype=torch.float64)
    hi = torch.tensor([[3., 1.], [3., 1.], [3., 1.], [1., 1.]], dtype=torch.float64)    # box 3 is empty
    dm_lb = torch.tensor([[1., -1.], [5., -1.], [-4., -1.], [0., -1.]], dtype=torch.float64)
    lA[1, 0, 0] = float('nan')                                                          # box 1: no plane
    out = bab._raise_child_keys(dm_lb, lo, hi, lA, lbias)
    assert out[0, 0] <= 2. and out[0, 0] > 1.99999999        # min of x_0 over [2, 3] is 2, above the parent's bound 1
    assert out[1, 0] == 5.                                    # NaN plane: the parent's bound stays
    assert out[2, 0] == -4.                                   # min over [-9, 3] is -9 < the parent's bound
    assert out[3, 0] == 0.                                    # empty box
    assert torch.equal(out[:, 1], dm_lb[:, 1])                # constraint rows are not touched
    assert out is not dm_lb and out[0, 0] != dm_lb[0, 0] and dm_lb[0, 0] == 1.   # the parent's tensor is not written


def test_the_production_profile_sorts_every_iteration():
    with open(os.path.join(CV, 'exp_configs', 'NLP_solver', 'gear.yaml')) as f:
        assert yaml.safe_load(f)['bab']['sort_domain_interval'] == 1
