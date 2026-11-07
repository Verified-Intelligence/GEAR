"""The BaB loop at an injected incumbent (WP15-pre: sort every iteration, child key max(parent's bound, plane minimum)), through
tests/bab_fixture.py: CPU, production profile gear.yaml, every primal channel off, so counts and bounds depend on the code alone.

Slow: a few seconds to three minutes per instance.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bab_fixture  # noqa: E402

needs_benchmarks = pytest.mark.skipif(not os.path.isdir(bab_fixture.BENCHMARKS), reason='benchmark directory is absent')

# DECLARED CHANGE (WP15-pre). Iterations to close at the optimal incumbent v, B = 2. Old: the base e7113bd9e (sort every 5th
# iteration, the parent's bound as key). New: sort every iteration, the child key and the exhaustive subdivision guard, which is
# E1's rec loop: the new counts are E1's exactly (E1 ran 773bb0a19 with hooks; this runs the production profile). The first
# WP15-pre commit had no guard (E1a's recng arm: 48 / 216 / 52 / 177). The change moves the branching order and so every count;
# no golden of the parity harness moves, since its stages do not run the loop.
#   instance     old   no guard   new
#   Shekel7       66      48       49
#   Hartman3     270     216      212
#   schitt_s377   53      52       48     (the guard never fires; 52->48 benign, A155)
#   MultiGauss   185     177       59     (the guard carries the whole gain)
CLOSES = [
    ('gams_world', 'Shekel7', -10.402940566818662, 66, 49),
    ('gams_world', 'Hartman3', -3.862782147820756, 270, 212),
    ('gams_world', 'schitt_s377', -795.0093265620962, 53, 40),
    ('gams_world', 'MultiGauss', -1.2969540459537794, 185, 59),
]


@pytest.mark.slow
@needs_benchmarks
@pytest.mark.parametrize('bench,name,v,old,new', CLOSES, ids=[c[1] for c in CLOSES])
def test_iterations_to_close_at_the_optimal_incumbent(bench, name, v, old, new):
    status, records = bab_fixture.run(bench, name, v)
    assert status == 'verified'
    assert [it for it, _, _ in records] == list(range(1, new + 1))
    assert new <= old
    assert all(abs(p - v) <= 1e-7 * max(1.0, abs(v)) for _, p, _ in records), 'the incumbent moved'


# Past-T_R audit (spec section 6). An incumbent 10 eps above the optimum v, run to 2 T_R (T_R = iterations to close at P = v:
# 265 for the old loop, 260 for the new one; 540 iterations cover both), must never lift the dual above v, and nothing may be
# verified: P - v = 10 eps is more than the closing tolerance eps. The audit sees a dual that is safe only while the incumbent
# is optimal. Without 0c7f51b00 (the base input rows relaxed by tau at the source) the code reports D = v + 9.16 eps and
# "verified" at iteration 330 (1.25 T_R). For the child key it sees gross errors only: a key taken from the plane's maximum
# instead of its minimum is "verified" at iteration 1 with D = v + 10 eps, but the plane at the box centre passes (max D - v =
# -0.002 eps over 540 iterations, as the correct key). Rounding and small errors are the fuzz's (test_child_key.py).
ADHYA = ('minlplib_linear', 'pooling_adhya3tp', -561.0566058319218)
MAX_ITERATIONS = 540


@pytest.mark.slow
@needs_benchmarks
def test_dual_stays_below_the_optimum_past_t_r():
    bench, name, v = ADHYA
    eps = max(1e-5, 1e-4 * max(1.0, abs(v)))
    status, records = bab_fixture.run(bench, name, v + 10 * eps, max_iterations=MAX_ITERATIONS)
    assert len(records) == MAX_ITERATIONS and status != 'verified'
    assert max(d for _, _, d in records) <= v + 1e-9 * max(1.0, abs(v))
    assert records[-1][2] >= v - 5 * eps, 'the audit is vacuous: the dual never came close to v'


# The same audit where the child key does the work. At pooling_adhya3tp a key taken at the box centre (which is not a bound) stays
# below v as well (max D - v = -0.002 eps over 540 iterations, as the correct key), so that audit sees gross errors only. At
# ex6_2_14 with P = v + 6.95 eps and 600 iterations (T_R = 259, so 2.3 T_R) the correct key raises 429 children and the dual stays
# below v (max D - v = -0.011 eps, the last iteration's), while the centre key lifts it to v + 0.85 eps from iteration 151 and a
# key from the plane's maximum is "verified" at iteration 26.
EX6_2_14 = ('minlplib_linear', 'ex6_2_14', -0.6954461429770405)


@pytest.mark.slow
@needs_benchmarks
def test_dual_stays_below_the_optimum_past_t_r_with_the_key_active():
    bench, name, v = EX6_2_14
    eps = max(1e-5, 1e-4 * max(1.0, abs(v)))
    status, records, raised = bab_fixture.run_full(bench, name, v + 6.95 * eps, max_iterations=600)
    assert len(records) == 600 and status != 'verified'
    assert raised > 0, 'the key never moved: the audit does not test it'
    assert max(d for _, _, d in records) <= v + 1e-9 * max(1.0, abs(v))
    assert records[-1][2] >= v - eps, 'the audit is vacuous: the dual never came close to v'
