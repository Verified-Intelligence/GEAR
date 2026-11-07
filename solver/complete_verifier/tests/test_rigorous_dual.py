"""Unit tests of nlp_solver/rigorous_dual.py against exact rational arithmetic (fractions.Fraction).

The claim under test: dual_lower_bound(...) <= D(y) EXACTLY (rational value of the same float data), for every input,
and certify_infeasible only where the exact Farkas value is > 0. Run: python test_rigorous_dual.py [n_random]
"""
import math
import os
import sys
from fractions import Fraction as F

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from nlp_solver import rigorous_dual as rd  # noqa: E402

INF = math.inf


def exact_D(c, c0, G, h, l, u, y):
    """exact D(y) with Fractions; -inf (float) when a column term is -infinite; rows with y == 0 skipped."""
    m, n = G.shape
    tot = F(float(c0))
    for i in range(m):
        if y[i] != 0:
            tot -= F(float(h[i])) * F(float(y[i]))
    for j in range(n):
        r = F(float(c[j]))
        for i in range(m):
            if y[i] != 0 and G[i, j] != 0:
                r += F(float(G[i, j])) * F(float(y[i]))
        if r == 0:
            continue
        lo, hi = float(l[j]), float(u[j])
        if r > 0:
            if lo == -INF:
                return -INF
            tot += r * F(lo)                         # min(r lo, r hi) = r lo for r > 0 (lo <= hi)
        else:
            if hi == INF:
                return -INF
            tot += r * F(hi)
    return tot


def run(rows, c, c0, h, l, u, y, tol):
    T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64))[None]
    lb, E, dd = rd.dual_lower_bound(rows, T(c), T([c0]).reshape(1), T(h), T(l), T(u), T(y), tol)
    return float(lb[0]), float(E[0]), bool(dd[0])


def check(name, G, c, c0, h, l, u, y, tol, stats):
    rows = rd.Rows.from_dense(torch.as_tensor(G)[None])
    ex = exact_D(c, c0, G, h, l, u, y)
    lb, E, dd = run(rows, c, c0, h, l, u, y, tol)
    if ex == -INF:
        assert lb == -INF, (name, 'exact is -inf but lb', lb)
    elif lb != -INF:
        excess = F(lb) - ex
        stats['n'] += 1
        stats['dd'] += dd
        if excess > 0:
            stats['viol'].append((name, float(excess)))
        stats['loss'] = max(stats['loss'], float(ex - F(lb)))
    return lb, ex


def random_lp(rng, m, n, scale_exp, density=0.5, cancel=False):
    G = rng.normal(size=(m, n)) * 10.0 ** rng.integers(-scale_exp, scale_exp + 1, size=(m, n))
    G[rng.random((m, n)) > density] = 0.0
    y = rng.exponential(size=m) * (rng.random(m) < 0.7) * 10.0 ** rng.uniform(-3, 6)
    c = rng.normal(size=n) * 10.0 ** rng.uniform(-3, 3)
    if cancel:
        c = -(G.T @ y) + rng.normal(size=n) * 10.0 ** rng.uniform(-12, -3)
    h = rng.normal(size=m) * 10.0 ** rng.uniform(-4, 4)
    l = rng.normal(size=n) * 10.0 ** rng.uniform(-2, 3)
    u = l + np.abs(rng.normal(size=n)) * 10.0 ** rng.uniform(-3, 3)
    return G, c, float(rng.normal() * 10.0 ** rng.uniform(-2, 4)), h, l, u, y


def test_random(n_random=300, seed=0):
    rng = np.random.default_rng(seed)
    stats = dict(n=0, dd=0, viol=[], loss=0.0)
    for t in range(n_random):
        m, n = int(rng.integers(1, 30)), int(rng.integers(1, 20))
        G, c, c0, h, l, u, y = random_lp(rng, m, n, int(rng.integers(0, 9)), cancel=bool(t % 2))
        if t % 5 == 0:                                   # force the double-double path
            check(f'rand{t}_dd', G, c, c0, h, l, u, y, 0.0, stats)
        check(f'rand{t}', G, c, c0, h, l, u, y, 1e-9, stats)
    assert not stats['viol'], stats['viol'][:5]
    return stats


def test_infinite_and_special():
    stats = dict(n=0, dd=0, viol=[], loss=0.0)
    INFT = np.inf
    # column 0: l = -inf, r > 0 -> -inf ; column 1 free with r exactly 0 -> 0 ; column 2 free with r < 0, u = +inf -> -inf
    G = np.array([[1.0, 0.0, 1.0]]); y = np.array([2.0]); h = np.array([0.5])
    l = np.array([-INFT, -INFT, -INFT]); u = np.array([INFT, INFT, INFT])
    check('free_pos', G, np.array([0.0, 0.0, 0.0]), 0.0, h, l, u, y, 1e-9, stats)
    # sign-certified one-sided: r = c + G^T y = 1 + 2 > 0, l finite, u = +inf -> r l
    check('onesided', G, np.array([1.0, 0.0, 1.0]), 0.0, h, np.array([-1.0, -INFT, 0.0]), np.array([INFT, INFT, INFT]), y, 1e-9, stats)
    lb, ex = check('onesided2', G, np.array([1.0, 0.0, 1.0]), 0.0, h, np.array([-1.0, -INFT, 0.0]), np.array([INFT, INFT, INFT]), y, 1e-9, stats)
    assert math.isfinite(lb) and abs(lb - float(ex)) < 1e-12, (lb, ex)       # column 1: r = 0 exactly, no loss
    # r exactly 0 through cancellation with an infinite side must NOT be treated as zero when it is not (float nonzero rounding)
    G2 = np.array([[1.0], [1.0]]); y2 = np.array([0.1, 0.2]); c2 = np.array([-0.30000000000000004])
    check('cancel_inf', G2, c2, 0.0, np.array([0.0, 0.0]), np.array([-INFT]), np.array([INFT]), y2, 1e-9, stats)
    # y = 0 rows with h = +inf and G = 0 are inactive (the production masks them like this)
    G3 = np.array([[0.0, 0.0], [1.0, 1.0]]); y3 = np.array([0.0, 1.0]); h3 = np.array([INFT, 1.0])
    check('inactive', G3, np.array([-1.0, -1.0]), 0.0, h3, np.array([0.0, 0.0]), np.array([1.0, 1.0]), y3, 1e-9, stats)
    # non-finite -> -inf
    rows = rd.Rows.from_dense(torch.as_tensor(np.array([[1.0, 1.0]]))[None])
    T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64))[None]
    for bad_c, bad_y, bad_h in ((np.nan, 1.0, 1.0), (0.0, np.nan, 1.0), (0.0, 1.0, np.nan), (0.0, np.inf, 1.0), (np.inf, 1.0, 1.0)):
        lb, _, _ = rd.dual_lower_bound(rows, T([bad_c, 0.0]), T([0.0]).reshape(1), T([bad_h]), T([0.0, 0.0]), T([1.0, 1.0]), T([bad_y]), 1e-9)
        assert float(lb[0]) == -INF, (bad_c, bad_y, bad_h, float(lb[0]))
    # negative y is clamped to 0: the value equals the value at y = 0
    lb_neg, _, _ = rd.dual_lower_bound(rows, T([1.0, 1.0]), T([0.0]).reshape(1), T([1.0]), T([0.0, 0.0]), T([1.0, 1.0]), T([-5.0]), 1e-9)
    lb_zero, _, _ = rd.dual_lower_bound(rows, T([1.0, 1.0]), T([0.0]).reshape(1), T([1.0]), T([0.0, 0.0]), T([1.0, 1.0]), T([0.0]), 1e-9)
    assert float(lb_neg[0]) == float(lb_zero[0])
    assert not stats['viol'], stats['viol']
    return stats


def test_farkas(n_random=200, seed=1):
    rng = np.random.default_rng(seed)
    n_cert = n_feas_wrong = 0
    for t in range(n_random):
        m, n = int(rng.integers(1, 12)), int(rng.integers(1, 8))
        G, c, c0, h, l, u, y = random_lp(rng, m, n, 3)
        rows = rd.Rows.from_dense(torch.as_tensor(G)[None])
        T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64))[None]
        cert = bool(rd.certify_infeasible(rows, T(h), T(l), T(u), T(y))[0])
        ex = exact_D(np.zeros(n), 0.0, G, h, l, u, y)
        if cert:
            n_cert += 1
            assert ex != -INF and ex > 0, ('wrong certificate', t, ex)
    # a real infeasible system: x <= -1 and x >= 1 on [-5, 5]; y = (1, 1): D0 = -h.y + min(r x) = 0 + 0 ... use margin
    G = np.array([[1.0], [-1.0]]); h = np.array([-1.0, -1.0]); l = np.array([-5.0]); u = np.array([5.0]); y = np.array([1.0, 1.0])
    rows = rd.Rows.from_dense(torch.as_tensor(G)[None])
    T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64))[None]
    assert bool(rd.certify_infeasible(rows, T(h), T(l), T(u), T(y))[0])             # D0 = 2 > 0
    assert not bool(rd.certify_infeasible(rows, T(np.array([1.0, 1.0])), T(l), T(u), T(y))[0])   # feasible system
    assert not bool(rd.certify_infeasible(rows, T(np.array([0.0, 0.0])), T(l), T(u), T(y))[0])   # boundary: D0 = 0 not > 0
    return n_cert


def test_double_double_effect():
    """The critic's cancellation setup: the float value is off by more than tol/10, the double-double step recovers it."""
    rng = np.random.default_rng(3)
    n, m = 120, 240
    worst = 0.0
    for trial in range(3):
        ymag = 10.0 ** rng.uniform(2, 6)
        G = rng.normal(size=(m, n)) * 10.0 ** rng.uniform(0, 3)
        y = rng.exponential(size=m) * ymag / m
        c = -(G.T @ y) + rng.normal(size=n) * 1e-9
        u = rng.uniform(100, 1000, size=n); l = -u
        h = rng.uniform(0, 1e-4, size=m)
        rows = rd.Rows.from_dense(torch.as_tensor(G)[None])
        T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64))[None]
        lb1, E1, dd1 = rd.dual_lower_bound(rows, T(c), T([0.0]).reshape(1), T(h), T(l), T(u), T(y), INF)   # tol = inf: no dd
        lb2, E2, dd2 = rd.dual_lower_bound(rows, T(c), T([0.0]).reshape(1), T(h), T(l), T(u), T(y), 1e-5)
        ex = exact_D(c, 0.0, G, h, l, u, y)
        assert F(float(lb1[0])) <= ex and F(float(lb2[0])) <= ex, (float(lb1[0]), float(lb2[0]), float(ex))
        assert bool(dd2[0]) and not bool(dd1[0]) and float(E2[0]) < 1e-3 * float(E1[0]), (float(E1[0]), float(E2[0]))
        worst = max(worst, float(ex - F(float(lb2[0]))))
    return worst


def test_batched_shared_pattern(seed=2):
    """B domains with one sparsity pattern and per-domain values: every domain's bound is <= its exact value, and the
    double-double step runs only on the flagged domains (the others keep their first-stage result)."""
    rng = np.random.default_rng(seed)
    B, m, n = 6, 25, 12
    mask = rng.random((m, n)) < 0.4
    Gs, cs, hs, ls, us, ys, c0s = [], [], [], [], [], [], []
    for b in range(B):
        G = rng.normal(size=(m, n)) * 10.0 ** rng.integers(-2, 3, size=(m, n)) * mask
        y = rng.exponential(size=m) * (rng.random(m) < 0.5) * 10.0 ** rng.uniform(0, 5)
        c = -(G.T @ y) + rng.normal(size=n) * 1e-9 if b % 2 else rng.normal(size=n)       # odd domains cancel: they get flagged
        l = rng.normal(size=n) * 100; u = l + np.abs(rng.normal(size=n)) * 100
        Gs.append(G); cs.append(c); hs.append(rng.normal(size=m)); ls.append(l); us.append(u); ys.append(y); c0s.append(float(rng.normal()))
    row, col = torch.as_tensor(mask).nonzero(as_tuple=True)
    val = torch.stack([torch.as_tensor(G)[row, col] for G in Gs])
    rows = rd.Rows(row, col, val, m, n)
    T = lambda a: torch.as_tensor(np.stack(a))
    lb, E, dd = rd.dual_lower_bound(rows, T(cs), torch.as_tensor(c0s, dtype=torch.float64), T(hs), T(ls), T(us), T(ys), 1e-5)
    for b in range(B):
        ex = exact_D(cs[b], c0s[b], Gs[b], hs[b], ls[b], us[b], ys[b])
        assert F(float(lb[b])) <= ex, (b, float(lb[b]), float(ex))
    _, E1, dd1 = rd.dual_lower_bound(rows, T(cs), torch.as_tensor(c0s, dtype=torch.float64), T(hs), T(ls), T(us), T(ys), float('inf'))
    assert not bool(dd1.any()) and bool((dd == (E1 > 1e-6)).all()), (dd, E1)      # flagged exactly where E exceeds tol/10
    assert 0 < int(dd.sum()) < B                                                   # and the test mixes flagged and unflagged domains
    return int(dd.sum())


def test_subnormal_data(seed=7):
    """A product that underflows to 0 still errs by up to eta/2, and every one of them counts: the 1000 columns below
    (c_j l_j = -0.3 eta each, rounded to 0) make the exact value -300 eta, which the bound has to stay under."""
    n = 1000
    stats = dict(n=0, dd=0, viol=[], loss=0.0)
    lb, ex = check('underflow', np.zeros((1, n)), np.full(n, 0.3), 0.0, np.ones(1), np.full(n, -rd.ETA),
                   np.full(n, rd.ETA), np.zeros(1), 0.0, stats)
    assert lb < 0 and not stats['viol'], (lb, stats['viol'])
    rng = np.random.default_rng(seed)
    for t in range(100):                                     # data around eta: every product may underflow
        m, n = int(rng.integers(1, 6)), int(rng.integers(1, 30))
        tiny = lambda *s: rng.integers(-8, 9, size=s) * rd.ETA
        G = np.where(rng.random((m, n)) < 0.5, rng.normal(size=(m, n)), tiny(m, n))
        y = rng.exponential(size=m) * (rng.random(m) < 0.7)
        c = np.where(rng.random(n) < 0.5, rng.normal(size=n), tiny(n))
        l = tiny(n) - rd.ETA
        check(f'subnormal{t}', G, c, float(tiny(1)[0]), np.where(rng.random(m) < 0.5, tiny(m), 1.0), l,
              l + np.abs(tiny(n)), y, 0.0 if t % 2 else None, stats)
    assert not stats['viol'], stats['viol'][:5]


def test_default_tolerance_and_caller_errors(seed=4):
    """tol=None is max(1e-5, 1e-4 |value|); c_err, c0_err widen the bound to every c, c0 within them."""
    rng = np.random.default_rng(seed)
    T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64))[None]
    above = 0
    for t in range(60):
        m, n = int(rng.integers(1, 12)), int(rng.integers(1, 8))
        G, c, c0, h, l, u, y = random_lp(rng, m, n, 3)
        rows = rd.Rows.from_dense(torch.as_tensor(G)[None])
        lb0, _, _ = rd.dual_lower_bound(rows, T(c), T([c0]).reshape(1), T(h), T(l), T(u), T(y))
        tol = max(1e-5, 1e-4 * abs(float(lb0[0])))
        same, _, _ = rd.dual_lower_bound(rows, T(c), T([c0]).reshape(1), T(h), T(l), T(u), T(y), tol)
        assert float(lb0[0]) == float(same[0]) or abs(float(lb0[0]) - float(same[0])) <= tol / 10
        c_err, c0_err = np.abs(c) * 1e-3, abs(c0) * 1e-3
        lb, _, _ = rd.dual_lower_bound(rows, T(c), T([c0]).reshape(1), T(h), T(l), T(u), T(y), c_err=T(c_err),
                                       c0_err=T([c0_err]).reshape(1))
        for _ in range(3):
            c1, c01 = c + rng.uniform(-1, 1, size=n) * c_err, c0 + rng.uniform(-1, 1) * c0_err
            ex = exact_D(c1, c01, G, h, l, u, y)
            assert ex == -INF or F(float(lb[0])) <= ex, (t, float(lb[0]), float(ex))
            above += ex != -INF and F(float(lb0[0])) > ex             # the same perturbations without the errors
    assert above > 0                                    # the check can fail: without c_err the bound is not valid for them
    # an error on a column whose box side is infinite and whose sign is not certified gives -inf, even at c_j = 0
    rows = rd.Rows.from_dense(torch.zeros(1, 1, 1, dtype=torch.float64))
    lb, _, _ = rd.dual_lower_bound(rows, T([0.0]), T([0.0]).reshape(1), T([1.0]), T([0.0]), T([INF]), T([0.0]),
                                   c_err=T([1e-30]))
    assert float(lb[0]) == -INF


def test_empty_dimensions(seed=5):
    """No row, no column or no entry: r = c exactly, the double-double step has nothing to recompute (it used to raise
    on the empty reduction when a domain was flagged)."""
    rng = np.random.default_rng(seed)
    stats = dict(n=0, dd=0, viol=[], loss=0.0)
    for m, n in ((0, 3), (3, 0), (0, 0), (3, 4)):               # (3, 4): rows without an entry
        G, h, y = np.zeros((m, n)), rng.normal(size=m), rng.random(m)
        for scale in (1.0, 1e300):
            for l, u in ((-np.ones(n), np.ones(n)), (np.full(n, -INF), np.full(n, INF))):
                for tol in (None, 0.0):
                    lb, ex = check(f'empty{m}x{n}', G, rng.normal(size=n) * scale, 1.0, h, l, u, y, tol, stats)
                    assert (ex == -INF) == (lb == -INF), (m, n, scale, lb, ex)       # finite exactly where the exact value is
    assert not stats['viol'], stats['viol']
    T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64))[None]
    box = (T(-np.ones(4)), T(np.ones(4)))
    empty = rd.Rows.from_dense(torch.zeros(1, 3, 4, dtype=torch.float64))
    assert bool(rd.certify_infeasible(empty, T([-1.0, 1.0, 1.0]), *box, T([1.0, 0.0, 0.0]))[0])     # 0 <= -1 is empty
    no_rows = rd.Rows.from_dense(torch.zeros(1, 0, 4, dtype=torch.float64))
    assert not bool(rd.certify_infeasible(no_rows, T(np.zeros(0)), *box, T(np.zeros(0)))[0])


def test_float32_callers_get_a_bound_rounded_toward_minus_inf(seed=6):
    """_lp_dual_value on float32 data casts the float64 bound back rounded toward -inf, so it stays <= the exact value
    of the float32 data (a cast to nearest exceeded it on about half of these LPs)."""
    from nlp_solver.dual_ascent import _lp_dual_value
    rng = np.random.default_rng(seed)
    T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))[None]
    nearest_above = 0
    for t in range(200):
        m, n = int(rng.integers(1, 6)), int(rng.integers(1, 5))
        G, c, c0, h, l, u, y = (np.asarray(a, dtype=np.float32) for a in random_lp(rng, m, n, 1))
        lb = _lp_dual_value(T(c), T([c0]).reshape(1), T(G), T(h), T(l), T(u), T(y))
        assert lb.dtype == torch.float32
        ex = exact_D(c, c0, G, h, l, u, y)
        assert F(float(lb[0])) <= ex, (t, float(lb[0]), float(ex))
        lb64, _, _ = rd.dual_lower_bound(rd.Rows.from_dense(torch.as_tensor(G.astype(np.float64))[None]), T(c).double(),
                                         T([c0]).reshape(1).double(), T(h).double(), T(l).double(), T(u).double(),
                                         T(y).double())
        nearest_above += F(float(lb64.float()[0])) > ex                   # what the plain cast would have returned
    assert nearest_above > 0                                              # the check can fail
    x = torch.tensor([0.1], dtype=torch.float64)
    assert F(float(x.float()[0])) > F(0.1) >= F(float(rd.round_down(x, torch.float32)[0]))     # nearest is above, down is not
    big = rd.round_down(torch.tensor([1e39, -1e39], dtype=torch.float64), torch.float32)       # overflow stays a lower bound
    assert float(big[0]) == torch.finfo(torch.float32).max and float(big[1]) == -INF


def test_zero_ray_skips_double_double(seed=0):
    """A ray that clamps to 0 has exact Farkas value 0, so it never enters the double-double step (A197);
    a cancellation-heavy ray that only the double-double step proves still gets it."""
    rng = np.random.default_rng(seed)
    m, n = 6, 3
    y = rng.exponential(size=m) * 1e3
    G0 = rng.normal(size=(m, n)) * 1e3
    G = G0 - np.outer(y, y @ G0) / (y @ y)                                # G^T y = 0 up to rounding
    u = np.full(n, 1e9); l = -u
    h = -np.full(m, 1.0 / y.sum())                                        # exact Farkas value about 1 - |G^T y| u
    Y = np.stack([np.zeros_like(y), -y - 1.0, y])                         # zero, clamped to zero, the ray
    rows = rd.Rows.from_dense(torch.as_tensor(G)[None])
    T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64))[None].expand(len(Y), -1)
    rows._single_pass = True
    assert not bool(rd.certify_infeasible(rows, T(h), T(l), T(u), torch.as_tensor(Y))[2])   # float alone cannot
    rows._single_pass = False
    lanes, orig = [], rd._reduced_dd
    def counted(rows_, val, c, y_):
        lanes.append(int(y_.shape[0]))
        return orig(rows_, val, c, y_)
    rd._reduced_dd = counted
    try:
        cert = rd.certify_infeasible(rows, T(h), T(l), T(u), torch.as_tensor(Y))
    finally:
        rd._reduced_dd = orig
    assert not bool(cert[0]) and not bool(cert[1])
    assert lanes == [1], lanes                                            # only the ray enters the step
    assert bool(cert[2]) and exact_D(np.zeros(n), 0.0, G, h, l, u, y) > 0


if __name__ == '__main__':
    torch.set_num_threads(1)
    n_random = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    st = test_random(n_random)
    print(f'random: {st["n"]} finite evaluations, {st["dd"]} with double-double, violations {len(st["viol"])}, max loss {st["loss"]:.3e}')
    st = test_infinite_and_special()
    print(f'special: {st["n"]} finite evaluations, violations {len(st["viol"])}')
    print('farkas: certificates issued on random data', test_farkas(), '(all exact-checked)')
    print('double-double effect: max loss vs exact on the critic cancellation setup', test_double_double_effect())
    print('batched shared pattern: flagged domains', test_batched_shared_pattern())
    test_default_tolerance_and_caller_errors()
    test_empty_dimensions()
    test_subnormal_data()
    test_float32_callers_get_a_bound_rounded_toward_minus_inf()
    test_zero_ray_skips_double_double()
    print('ALL OK')
