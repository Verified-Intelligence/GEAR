"""Root structure certificates (nlp_solver/structure/certificates.py, spectral.py): soundness on sampled feasible
points and exact rationals, declining near misses, the outward kernels, and the root wiring with its flags."""
import itertools
import math
import os
import sys
import time
from fractions import Fraction as F
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import arguments
from auto_LiRPA import BoundedModule
from nlp_solver.structure import RowPool, build_structure
from nlp_solver.structure.certificates import (SquareForms, Graph, Form, NoPattern, Expired, root_certificates,
                                              pattern_costs, cover)
from nlp_solver.structure.cert_kernels import CoverPattern, combination_ids, lp_prices
from nlp_solver.structure.spectral import detect_eigenblock, verified_shift, spectral, gram_error

KINDS = ['diameter', 'assignment', 'eigenblock']


@pytest.fixture(autouse=True)
def tolerance():
    arguments.Config.parse_config([], verbose=False)
    arguments.Config['solving']['solving_mode'] = True
    arguments.Config['solver']['nlp']['constr_viol_tol'] = 1e-6
    yield
    arguments.Config.parse_config([], verbose=False)


def detect(model, lo, hi, kinds=KINDS, deadline=None):
    lo = torch.tensor([lo], dtype=torch.float64)
    hi = torch.tensor([hi], dtype=torch.float64)
    x = (lo + hi) / 2
    bm = BoundedModule(model, x, bound_opts=dict(pow_negative_exponent=True))
    m = model(x).shape[1]
    C, rhs = torch.eye(m, dtype=torch.float64)[None], torch.zeros(1, m, dtype=torch.float64)
    report = build_structure(bm, x, C, rhs, lo, hi, pool=RowPool(rhs), algebra=SquareForms)
    found, skipped = root_certificates(report, kinds, lo, hi, time.time() + 30 if deadline is None else deadline)
    return {c.kind: c for c in found}, skipped


def violation(model, x):
    """max over the model's rows y[1:] <= 0 (all spec rows here) of the violation, and the objective"""
    with torch.no_grad():
        y = model(torch.as_tensor(x, dtype=torch.float64).reshape(1, -1))[0]
    return float(y[1:].max()), float(y[0])


# ---------------------------------------------------------------------------------------------- diameter graph
class Polygon(nn.Module):
    def __init__(self, n=5, missing=False, unordered=False, shifted=False):
        super().__init__()
        self.n, self.missing, self.unordered, self.shifted = n, missing, unordered, shifted

    def forward(self, x):
        n = self.n
        r, t = x[:, :n], x[:, n:]
        angle = t[:, 1:] - t[:, :-1] if not self.shifted else t[:, 1:] - 1
        obj = -.5 * (r[:, 1:] * r[:, :-1] * torch.sin(angle)).sum(1, keepdim=True)
        distances = [r[:, i:i + 1] ** 2 + r[:, j:j + 1] ** 2
                     - 2 * r[:, i:i + 1] * r[:, j:j + 1] * torch.cos(t[:, i:i + 1] - t[:, j:j + 1]) - 1
                     for i, j in itertools.combinations(range(n), 2) if not (self.missing and (i, j) == (0, n - 1))]
        order = [] if self.unordered else [t[:, :-1] - t[:, 1:]]
        return torch.cat([obj, *distances, *order], 1)


def polygon_box(n):
    return [0.] * (2 * n), [1.] * (n - 1) + [0.] + [float(np.nextafter(np.pi, -np.inf))] * n


@pytest.mark.parametrize('n', [4, 5, 6, 9])
def test_diameter_floor_holds_on_sampled_tolerance_feasible_polygons(n):
    lo, hi = polygon_box(n)
    certs, _ = detect(Polygon(n), lo, hi)
    c = certs['diameter-area']
    tau = 1e-6
    from mpmath.ctx_iv import MPIntervalContext
    iv = MPIntervalContext()
    iv.dps = 60
    U = n * iv.mpf(.5) * (iv.sin(iv.pi / n) - iv.tan(iv.pi / (2 * n)))
    assert c.lower_bound <= -float((U * (1 + iv.mpf(tau)) + (n - 1) * iv.mpf(tau) / 2).b)  # order allowance kept
    model, rng = Polygon(n), np.random.default_rng(n)
    feasible = 0
    for _ in range(4000):
        t = np.sort(rng.uniform(0, np.pi, n))
        if rng.random() < .3:            # tolerated backward steps of the order rows
            t[1:] -= rng.uniform(0, tau, n - 1)
        r = rng.uniform(0, 1, n)
        r[-1] = 0.
        r *= rng.uniform(.2, 1.)
        x = np.r_[r, np.clip(t, 0, hi[-1])]
        viol, obj = violation(model, x)
        if viol <= tau:
            feasible += 1
            assert obj >= c.lower_bound
    assert feasible > 100
    viol, obj = violation(model, c.candidate)
    assert viol <= tau and obj >= c.lower_bound
    if n % 2:                            # U_N is attained by the regular N-gon only for odd N
        assert obj - c.lower_bound <= 1e-4


@pytest.mark.parametrize('kw', [dict(missing=True), dict(unordered=True), dict(shifted=True)])
def test_diameter_declines_missing_edge_order_or_shifted_angle(kw):
    lo, hi = polygon_box(5)
    certs, skipped = detect(Polygon(**kw), lo, hi)
    assert 'diameter-area' not in certs and 'diameter' in skipped


def test_diameter_declines_outside_the_half_plane():
    lo, hi = polygon_box(5)
    hi = hi[:5] + [3.5] * 5
    certs, skipped = detect(Polygon(), lo, hi)
    assert not certs and 'half-plane' in skipped['diameter']


# ---------------------------------------------------------------------------------------------- assignment cover
class Assignment(nn.Module):
    """items x groups one-hot matrix with cardinality rows and one squared-affine cost per (group, feature)"""

    def __init__(self, items=4, groups=2, weights=((1., 2., 3., 4.),), targets=(5.,), extra=False):
        super().__init__()
        self.items, self.groups, self.extra = items, groups, extra
        self.register_buffer('w', torch.tensor(weights, dtype=torch.float64))
        self.register_buffer('t', torch.tensor(targets, dtype=torch.float64))

    def forward(self, x):
        p = x.reshape(-1, self.items, self.groups)
        a = (p[:, None] * self.w[None, :, :, None]).sum(2) - self.t[None, :, None]       # (batch, feature, group)
        obj = (a * a).sum((1, 2))[:, None] + (x[:, :1] if self.extra else 0)
        binary = x * (x - 1)
        item, group = p.sum(2) - 1, p.sum(1) - self.items // self.groups
        return torch.cat([obj, binary, -binary, item, -item, group, -group], 1)


def brute_force(model):
    """the exact minimum over all binary assignments"""
    n, g, k = model.items, model.groups, model.items // model.groups
    best = math.inf
    for labels in itertools.product(range(g), repeat=n):
        if all(labels.count(j) == k for j in range(g)):
            x = np.zeros((n, g))
            x[np.arange(n), labels] = 1
            best = min(best, violation(model, x.ravel())[1])
    return best


@pytest.mark.parametrize('args', [dict(), dict(items=6, groups=3, weights=((1., -2., 3., 0.5, 2., -1.),
                                                                          (2., 1., .5, 1., -3., 2.)),
                                             targets=(1., 0.5))])
def test_assignment_floor_is_below_every_assignment_and_the_cover_point_verifies(args):
    model = Assignment(**args)
    n = model.items * model.groups
    certs, _ = detect(model, [0.] * n, [1.] * n)
    c = certs['assignment-cover']
    assert c.details['complete']
    assert c.lower_bound <= brute_force(model)
    viol, obj = violation(model, c.candidate)
    assert viol <= 1e-6 and obj - c.lower_bound <= max(1e-5, 1e-4 * max(1, abs(obj)))     # the GEAR gap rule


def test_assignment_declines_an_objective_with_extra_terms():
    certs, skipped = detect(Assignment(extra=True), [0.] * 8, [1.] * 8)
    assert 'assignment-cover' not in certs and 'assignment' in skipped


def test_pattern_cost_interval_contains_every_near_binary_group():
    features = [(tuple(map(F, [1, -2, 3, -4])), F(3))]
    patterns = torch.tensor([[0, 2]])
    costs = pattern_costs(patterns, features, F(1, 1000), F(1, 10000), 'cpu')
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(4000):
        e = rng.uniform(0, .001, 4)
        e[[0, 2]] *= -1                      # selected items sit below 1, the others above 0
        if abs(e.sum()) <= .0001:
            checked += 1
            assert costs[0] <= (np.dot([1, -2, 3, -4], np.array([1, 0, 1, 0]) + e) + 3) ** 2
    assert checked > 50


def test_cover_is_exact_on_random_costs_and_any_prices_are_repaired():
    rng = np.random.default_rng(3)
    patterns = combination_ids(6, 2, 'cpu')
    assert [tuple(p) for p in patterns.tolist()] == list(itertools.combinations(range(6), 2))
    for _ in range(20):
        costs = torch.as_tensor(rng.uniform(0, 5, len(patterns)))
        exact = min(sum(float(costs[patterns.tolist().index(list(p))]) for p in cover_)
                    for cover_ in itertools.combinations(itertools.combinations(range(6), 2), 3)
                    if len({v for p in cover_ for v in p}) == 6)
        for prices in (lp_prices(patterns, costs, 6), torch.as_tensor(rng.normal(0, 10, 6))):
            value, _, complete, _ = CoverPattern(patterns, 6).solve(costs, prices)
            assert complete and value <= exact and exact - value <= 1e-9


def test_unfinished_cover_returns_only_the_dual_bound():
    patterns = combination_ids(6, 2, 'cpu')
    costs = torch.tensor([float((i + 1) * (j + 1)) for i, j in patterns.tolist()])
    calls = [0]

    def stop():
        calls[0] += 1
        return calls[0] > 1
    value, _, complete, _ = CoverPattern(patterns, 6).solve(costs, lp_prices(patterns, costs, 6), stop=stop)
    full, _, _, _ = CoverPattern(patterns, 6).solve(costs, lp_prices(patterns, costs, 6))
    assert not complete and value <= full


# ---------------------------------------------------------------------------------------------- eigenblock
def eigen_graph(A, scales=None, normalization=True, wrong=False):
    """the exact rows of: min 3 lambda + 1/8 s.t. sum (s_i q_i)^2 = 1, c_i (lambda q_i - sum_j A_ij s_j/s_i q_j) = 0"""
    n = len(A)
    scales = [F(1)] * n if scales is None else list(map(F, scales))
    polys = [Form({((0, 1),): 3, (): F(1, 8)})]
    if normalization:
        p = {((i + 1, 2),): s * s for i, s in enumerate(scales)} | {(): -1}
        polys += [Form(p), Form({m: -c for m, c in p.items()})]
    for i in range(n):
        c = F((-1) ** i * (i + 1))           # independent signed row scales scale the band exactly once
        p = {((0, 1), (i + 1, 1)): c}
        for j in range(n):
            if A[i, j]:
                p[((j + 1, 1),)] = -F(float(A[i, j])) * scales[j] / scales[i] * c
        if wrong and i == 0:
            p[((i + 1, 2),)] = 1
        polys += [Form(p), Form({m: -v for m, v in p.items()})]
    lo, hi = np.full(n + 1, -10.), np.full(n + 1, 10.)
    return Graph(SquareForms(n + 1, lo), polys, RowPool(torch.zeros(1, len(polys), dtype=torch.float64)), lo, hi)


@pytest.mark.parametrize('n', [1, 2, 7, 16, 64])
@pytest.mark.parametrize('offset', [-3., 0., 4.])
def test_random_sparse_symmetric_shift_and_band_sampling(n, offset):
    rng = np.random.default_rng(110 + n)
    A = rng.integers(-8, 9, (n, n)) / 8
    A *= rng.random((n, n)) < min(1., 3 / n)
    A = (A + A.T) / 2 + np.eye(n) * offset
    g = eigen_graph(A)
    mu, estimate, rho, _ = verified_shift(detect_eigenblock(g), 'cpu')
    exact, vectors = np.linalg.eigh(A)
    assert mu <= exact[0] and estimate - mu >= rho >= 0
    c = spectral(g)
    assert c.lower_bound <= .125 + 3 * exact[0]
    # points across the normalization and eigen-row bands, lambda below the minimum and norm below one included
    q = vectors[:, 0] + rng.uniform(-1e-9 / n, 1e-9 / n, (4000, n))
    q *= np.sqrt(1 + rng.uniform(-1e-6, 1e-6, (4000, 1))) / np.linalg.norm(q, axis=1)[:, None]
    lam = exact[0] + rng.uniform(-1e-6 / n, 1e-6 / n, 4000)
    r = lam[:, None] * q - q @ A.T
    feasible = (np.abs(r) * np.arange(1, n + 1) <= 1e-6).all(1)
    assert feasible.any()
    assert (c.lower_bound <= .125 + 3 * lam[feasible]).all()


def test_eigenblock_brute_force_on_a_wide_band():
    arguments.Config['solver']['nlp']['constr_viol_tol'] = .1
    c = spectral(eigen_graph(np.array([[1., -.5], [-.5, 1.]])))
    a, b, lam = np.meshgrid(np.linspace(-1.1, 1.1, 60), np.linspace(-1.1, 1.1, 60), np.linspace(.2, 2., 40),
                            indexing='ij')
    valid = (abs(a * a + b * b - 1) <= .1) & (abs(lam * a - a + .5 * b) <= .1) & (2 * abs(lam * b - b + .5 * a) <= .1)
    assert valid.sum() > 100
    assert (.125 + 3 * lam[valid] >= c.lower_bound).all()


@pytest.mark.parametrize('scales', [[1, 2, F(1, 2)], [2, 4, 1]])
def test_rational_coordinate_scaling(scales):
    A = np.array([[2., -1., 0.], [-1., 2., -1.], [0., -1., 2.]])
    block = detect_eigenblock(eigen_graph(A, scales))
    np.testing.assert_array_equal(
        np.array([[float(block.entries.get((i, j), 0)) for j in range(3)] for i in range(3)]), A)
    assert spectral(eigen_graph(A, scales)).lower_bound > .125 + 3 * (np.linalg.eigvalsh(A)[0] - 1e-5)


@pytest.mark.parametrize('kw', [dict(normalization=False), dict(wrong=True)])
def test_eigenblock_declines_missing_normalization_and_wrong_rows(kw):
    with pytest.raises(NoPattern, match='normalization'):
        detect_eigenblock(eigen_graph(np.eye(3), **kw))


def test_eigenblock_declines_one_bit_asymmetry_and_a_negative_objective():
    with pytest.raises(NoPattern, match='asymmetric'):
        detect_eigenblock(eigen_graph(np.array([[2., 1.], [np.nextafter(1., 2.), 2.]])))
    g = eigen_graph(np.eye(2))
    g.obj[((0, 1),)] = -1
    with pytest.raises(NoPattern, match='positive affine'):
        detect_eigenblock(g)


def test_gram_residual_catches_bad_factors_and_rational_conversion():
    block = detect_eigenblock(eigen_graph(np.eye(2)))
    block.entries[0, 0] = F(1, 3)
    assert F(verified_shift(block, 'cpu')[0]) <= F(1, 3)
    eye = torch.eye(2, dtype=torch.float64)
    with patch('nlp_solver.structure.spectral.torch.linalg.cholesky_ex',
               return_value=(torch.zeros_like(eye), torch.tensor(0))):
        mu = verified_shift(detect_eigenblock(eigen_graph(np.eye(2))), 'cpu')[0]
    assert mu <= 1                       # an arbitrary finite factor cannot overstate mu
    assert gram_error(eye, eye, eye.new_tensor(0.), 100 * eye) >= 9999


def test_eigenblock_deadline_and_work_limit_fail_closed():
    block = detect_eigenblock(eigen_graph(np.eye(2)))

    def expired():
        raise Expired
    with pytest.raises(Expired):
        verified_shift(block, 'cpu', expired)
    block.coordinates = tuple(range(513))
    with pytest.raises(NoPattern, match='work limit'):
        verified_shift(block, 'cpu')


class EigenModel(nn.Module):
    def forward(self, x):
        lam, q = x[:, :1], x[:, 1:]
        norm = (q * q).sum(1, keepdim=True) - 1
        residual = lam * q - q
        return torch.cat((lam, norm, -norm, residual, -residual), 1)


def test_eigenblock_on_a_model_and_its_candidate_verifies():
    certs, _ = detect(EigenModel(), [-2.] * 4, [2.] * 4)
    c = certs['normalized-eigenblock']
    assert .99999 < c.lower_bound <= 1
    viol, obj = violation(EigenModel(), c.candidate)
    assert viol <= 1e-6 and obj - c.lower_bound <= 1e-5


# ---------------------------------------------------------------------------------------------- compile and wiring
def test_square_forms_leave_the_structure_report_unchanged():
    model, (lo, hi) = Assignment(), ([0.] * 8, [1.] * 8)
    lo, hi = torch.tensor([lo], dtype=torch.float64), torch.tensor([hi], dtype=torch.float64)
    x = (lo + hi) / 2
    bm = BoundedModule(model, x)
    m = model(x).shape[1]
    C, rhs = torch.eye(m, dtype=torch.float64)[None], torch.zeros(1, m, dtype=torch.float64)
    plain = build_structure(bm, x, C, rhs, lo, hi, pool=RowPool(rhs))
    square = build_structure(bm, x, C, rhs, lo, hi, pool=RowPool(rhs), algebra=SquareForms)
    assert plain.algebra is None and isinstance(square.algebra, SquareForms) and square.algebra.squares
    assert [f.poly for f in plain.forms] == [f.poly for f in square.forms]
    assert [(r.cls, r.tags, r.alias, r.opposite_of) for r in plain.rows] == \
        [(r.cls, r.tags, r.alias, r.opposite_of) for r in square.rows]
    assert plain.pool.stats() == square.pool.stats()


def test_expired_detection_returns_nothing():
    lo, hi = polygon_box(5)
    certs, skipped = detect(Polygon(), lo, hi, deadline=0)
    assert not certs and skipped == {'graph': 'Expired'}       # the exact objective is built under the deadline


def test_flags_default_off():
    for kind in KINDS:
        assert arguments.Config['solver']['nlp'][f'struct_cert_{kind}'] is False


def test_root_operations_run_before_the_root_nlp_after_the_rows():
    from input_split import batch_branch_and_bound as bbb
    names = [op.name for op in bbb._ROOT_OPERATIONS]
    assert names.index('rows') < names.index('struct_forms') < names.index('struct_cert') < names.index('nlp')
    root = SimpleNamespace(solving_mode=True, nlp_config={'struct_cert_diameter': True},
                           structure=SimpleNamespace(skipped=None, forms=[object()]))
    assert bbb._struct_cert_due(root)
    for change in (dict(nlp_config={}), dict(solving_mode=False), dict(structure=None),
                   dict(structure=SimpleNamespace(skipped='per-domain specs', forms=[object()])),
                   dict(structure=SimpleNamespace(skipped=None, forms=[]))):
        assert not bbb._struct_cert_due(SimpleNamespace(**{**vars(root), **change}))


@pytest.mark.parametrize('enabled,unified', [(False, False), (True, False), (False, True), (True, True)])
def test_root_wiring_closes_before_the_root_nlp_and_off_is_untouched(enabled, unified, capsys):
    from test_batch_branch_and_bound import _setup_mock_config
    from input_split import batch_branch_and_bound as bbb
    config = _setup_mock_config()
    config['solving']['solving_mode'] = True
    config['solver']['nlp']['constr_viol_tol'] = 1e-6
    config['solver']['nlp']['unified_root'] = unified
    for kind in KINDS:
        config['solver']['nlp'][f'struct_cert_{kind}'] = enabled and kind == 'eigenblock'
    lo, hi = torch.full((1, 4), -2., dtype=torch.float64), torch.full((1, 4), 2., dtype=torch.float64)
    model = EigenModel()
    bm = BoundedModule(model, (lo + hi) / 2)
    m = model(lo).shape[1]
    C, rhs = torch.eye(m, dtype=torch.float64)[None], torch.zeros(1, m, dtype=torch.float64)
    lb, A = torch.full((1, m), -2., dtype=torch.float64), torch.ones(1, m, 4, dtype=torch.float64)
    ref = {'global_lb': lb.clone(), 'lA': {bm.input_name[0]: A.clone()}}
    x = SimpleNamespace(ptb=SimpleNamespace(x_L=lo, x_U=hi), shape=lo.shape)
    net = SimpleNamespace(net=bm, model_ori=model, final_name=bm.final_name)
    with patch.object(arguments, 'Config', config), \
            patch('nlp_solver.structure.certificates.root_certificates', wraps=root_certificates) as build, \
            patch.object(bbb, 'pre_incomplete_to_root_result', side_effect=StopIteration) as primal, \
            patch.object(bbb, '_finish_with_lifted_dual', return_value='closed') as finish:
        if enabled:
            assert bbb.input_bab_parallel(net, x, C, rhs, torch.tensor([m]), reference_dict=ref) == 'closed'
            primal.assert_not_called()
            _, domains, point, P, found, D, _ = finish.call_args.args
            assert domains is None and found and 0 <= float(P[0]) - D <= 1e-5 and D > .99999
            assert violation(model, point)[0] <= 1e-6
            assert bm.objective_alias_floor_value is not None and bm.objective_alias_floor_value <= D
            assert ('[UnifiedRoot] closed after struct_cert' in capsys.readouterr().out) == unified
        else:
            with pytest.raises(StopIteration):
                bbb.input_bab_parallel(net, x, C, rhs, torch.tensor([m]), reference_dict=ref)
            finish.assert_not_called()
            assert getattr(bm, 'objective_alias_floor_value', None) is None
        assert build.call_count == int(enabled)
    assert torch.equal(ref['global_lb'], lb) and torch.equal(ref['lA'][bm.input_name[0]], A)


@pytest.mark.parametrize('nlp_obj,expect', [(5., 'earlier'), (3., 'nlp'), (4., 'nlp')])
def test_root_incumbent_keeps_a_lower_verified_root_point(nlp_obj, expect):
    from input_split import batch_branch_and_bound as bbb
    x_earlier, x_nlp = torch.zeros(1, 2), torch.ones(1, 2)
    root = SimpleNamespace(
        or_spec_size=torch.tensor([2]), solving_mode=True, global_lb=torch.zeros(1, 2), rhs=torch.zeros(1, 2),
        primal_solution=torch.tensor([4.]), primal_solution_found=True, best_primal_x=x_earlier,
        nlp_obj=torch.tensor([nlp_obj]), ipopt_obj=torch.tensor([math.inf]), ipopt_found=False,
        root_nlp=SimpleNamespace(nlp_best_x=x_nlp, ipopt_x=None), report_only_primal=None)
    with patch.object(arguments, 'Config', {'solving': {'enable_solver_res': False}}):
        bbb._root_incumbent(root, math.inf)
    P, x = (4., x_earlier) if expect == 'earlier' else (nlp_obj, x_nlp)
    assert root.primal_solution_found and float(root.primal_solution[0]) == P
    assert root.best_primal_x is x and root.warmstart_x is x


def test_cert_crown_dual_prices_come_from_the_kernel_and_cover_value_is_unchanged(monkeypatch):
    import scipy.optimize
    import arguments
    from nlp_solver.structure import cert_kernels
    patterns = combination_ids(6, 2, 'cpu')
    costs = torch.as_tensor(np.random.default_rng(5).uniform(0, 5, len(patterns)))
    highs = lp_prices(patterns, costs, 6)
    conf = arguments.ConfigHandler()
    conf.construct_config_dict(conf.default_args)
    conf['solver']['nlp']['cert_crown_dual'] = True
    monkeypatch.setattr(arguments, 'Config', conf)
    monkeypatch.setattr(scipy.optimize, 'linprog', lambda *a, **k: (_ for _ in ()).throw(AssertionError('linprog')))
    monkeypatch.setitem(sys.modules, 'highspy', None)         # an import of HiGHS would fail
    prices = lp_prices(patterns, costs, 6)
    assert prices.shape == (6,) and torch.isfinite(prices).all()
    assert not torch.equal(prices, highs)                       # not HiGHS's; cover() repairs them either way
    full = CoverPattern(patterns, 6).solve(costs, highs)[0]
    assert CoverPattern(patterns, 6).solve(costs, prices)[0] == full
