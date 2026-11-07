"""Sparse exact derivatives from the structure walk's forms (nlp_solver.sparse_derivatives): values, Jacobians and
Lagrangian Hessians against torch autograd and central differences, pattern containment, the row merge, the
reproduction guard, and the IPOPT path behind solver.nlp.sparse_derivatives (off: the v406 path)."""
import os
import sys
from unittest import mock

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nlp_solver import ipopt_solver  # noqa: E402
from nlp_solver import sparse_derivatives as sd  # noqa: E402
from nlp_solver.ipopt_solver import NNProblemIPOPT, IPOPTSolver  # noqa: E402

BENCH = str(__import__("pathlib").Path(__file__).resolve().parents[3] / "benchmarks")


class Ops(nn.Module):
    """every atom kind of the walk, nested twice, and polynomial terms of degree 1-3"""
    def forward(self, x):
        a, b, c, d = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        obj = a * b * c + a ** 3 - 2.0 * b ** 2 + torch.exp(a * b) + torch.log(1.5 + c * c) \
            + torch.sin(a - d) * torch.cos(b + c)
        r1 = torch.tan(0.3 * a) + torch.tanh(b * d) - c / (2.0 + a * a)
        r2 = torch.sqrt(1.0 + b * b) * d - a * b + c
        r3 = a + b - c
        r4 = -(a + b - c) - 0.5                     # the opposite of r3: one ranged row
        r5 = torch.exp(torch.sin(a * c)) * d
        r6 = 0.0 * a + 3.0                          # no non-constant part
        return torch.cat([obj, r1, r2, r3, r4, r5, r6], 1)


class WithAbs(nn.Module):
    def forward(self, x):
        return torch.cat([x[:, :1] * x[:, 1:2] + torch.abs(x[:, :1] - 0.2), x[:, :1] + x[:, 1:2] - 1.0], 1)


class Cancels(nn.Module):
    """(x - 1e6)^2: the walk expands it, and near x = 1e6 the expanded form cancels in floating point"""
    def forward(self, x):
        return torch.cat([(x[:, :1] - 1e6) ** 2, x[:, :1] - 1e6 - 0.03], 1)


class Bilinear(nn.Module):
    """min -x0 x1 s.t. x0 + x1 = 1 (two opposite rows)"""
    def forward(self, x):
        s = x[:, :1] + x[:, 1:2] - 1.0
        return torch.cat([-x[:, :1] * x[:, 1:2], s, -s], 1)


def problem_of(model, lo, hi):
    lo = torch.tensor(lo, dtype=torch.float64)
    hi = torch.tensor(hi, dtype=torch.float64)
    with torch.no_grad():
        K = model(lo[None]).shape[1]
    C = torch.eye(K, dtype=torch.float64)
    return NNProblemIPOPT(model.double(), lo, hi, C[:1], torch.zeros(1, dtype=torch.float64), C[1:],
                          torch.zeros(K - 1, dtype=torch.float64), device=torch.device('cpu'), dtype=torch.float64)


def dense(rows, cols, vals, shape):
    out = np.zeros(shape)
    np.add.at(out, (rows, cols), vals)
    return out


def tape_jacobian(spp, x):
    """the raw rows' Jacobian from the tape rows: raw row i = rho_i q_{g_i} + c_i"""
    nlp = spp.nlp
    Jq = np.vstack([dense(nlp.tape.jac_rows, nlp.tape.jac_cols, spp.jacobian(x), (spp.m, nlp.n)), np.zeros((1, nlp.n))])
    return nlp.row_rho[:, None] * Jq[nlp.row_group]


def tape_hessian(spp, x, lam, sigma):
    t = spp.nlp.tape
    H = dense(t.hess_rows, t.hess_cols, spp.hessian(x, lam, sigma), (spp.nlp.n, spp.nlp.n))
    return H + np.tril(H, -1).T


def torch_lagrangian(model, spp, lam, sigma):
    """sigma f + sum_r lam_r q_r with q_r recovered from the group's first raw row"""
    nlp = spp.nlp
    first = {}
    for i, g in enumerate(nlp.row_group.tolist()):
        if g >= 0:
            first.setdefault(g, i)

    def f(z):
        y = model(z[None])[0]
        q = [(y[1 + first[r]] - nlp.row_const[first[r]]) / nlp.row_rho[first[r]] for r in range(spp.m)]
        return sigma * y[0] + sum(float(lam[r]) * q[r] for r in range(spp.m))
    return f


def check_against_torch(model, spp, points, hessian=True):
    """worst relative errors of values, gradient, Jacobian and Hessian; pattern misses (finite torch nonzeros outside
    the tape's patterns)"""
    nlp, n = spp.nlp, spp.nlp.n
    err = dict(value=0.0, grad=0.0, jac=0.0, hess=0.0, fd=0.0, jpat=0, hpat=0)
    rel = lambda a, b: float(np.max(np.abs(a - b) / (1.0 + np.abs(b)), initial=0.0))
    rng = np.random.default_rng(1)
    for x in points:
        xt = torch.tensor(x, requires_grad=True)
        y = model(xt[None])[0]
        err['value'] = max(err['value'], rel(np.r_[spp.objective(x), spp.raw_rows(x)], y.detach().numpy()))
        g = torch.autograd.grad(y[0], xt)[0].numpy()
        err['grad'] = max(err['grad'], rel(spp.gradient(x), g))
        if y.numel() > 1:
            Jm = torch.autograd.functional.jacobian(lambda z: model(z[None])[0][1:], torch.tensor(x)).numpy()
            err['jac'] = max(err['jac'], rel(tape_jacobian(spp, x), Jm))
            pat = np.zeros((spp.m + 1, n), bool)
            pat[nlp.tape.jac_rows, nlp.tape.jac_cols] = True
            err['jpat'] += int((np.isfinite(Jm) & (Jm != 0) & ~pat[nlp.row_group]).sum())
        if hessian:
            lam, sigma = rng.standard_normal(spp.m), 0.7
            Hm = torch.autograd.functional.hessian(torch_lagrangian(model, spp, lam, sigma), torch.tensor(x)).numpy()
            Ht = tape_hessian(spp, x, lam, sigma)
            err['hess'] = max(err['hess'], rel(Ht, Hm))
            patH = np.zeros((n, n), bool)
            patH[nlp.tape.hess_rows, nlp.tape.hess_cols] = True
            err['hpat'] += int((np.isfinite(Hm) & (Hm != 0) & ~(patH | patH.T)).sum())

            def lag_grad(z):        # the tape's own Lagrangian gradient
                Jq = dense(nlp.tape.jac_rows, nlp.tape.jac_cols, spp.jacobian(z), (spp.m, n))
                return sigma * spp.gradient(z) + Jq.T @ lam
            for j in range(n):      # central differences, column by column
                h = 1e-6 * (1.0 + abs(x[j]))
                e = np.zeros(n)
                e[j] = h
                col = (lag_grad(x + e) - lag_grad(x - e)) / (2 * h)
                err['fd'] = max(err['fd'], float(np.max(np.abs(col - Ht[:, j]) / (1.0 + np.abs(Ht[:, j])))))
    return err


def test_every_atom_kind_matches_autograd_and_central_differences():
    lo, hi = [-0.8, 0.1, -0.5, 0.2], [0.9, 1.3, 0.7, 1.1]
    model = Ops().double()
    spp = sd.sparse_problem(problem_of(model, lo, hi))
    assert spp is not None and spp.hessian_available
    assert len(spp.nlp.tape.levels) == 2                    # exp(sin(a c)): nested atoms
    rng = np.random.default_rng(0)
    pts = [np.array(lo) + (np.array(hi) - np.array(lo)) * rng.random(4) for _ in range(5)]
    err = check_against_torch(model, spp, pts)
    assert err['value'] < 1e-13 and err['grad'] < 1e-13 and err['jac'] < 1e-13 and err['hess'] < 1e-12
    assert err['fd'] < 1e-6
    assert err['jpat'] == 0 and err['hpat'] == 0


def test_opposite_rows_merge_into_one_ranged_row_and_constant_rows_drop():
    lo, hi = [-0.8, 0.1, -0.5, 0.2], [0.9, 1.3, 0.7, 1.1]
    spp = sd.sparse_problem(problem_of(Ops().double(), lo, hi))
    nlp = spp.nlp
    assert len(nlp.row_group) == 6 and spp.m == 4           # r3/r4 share a row, r6 has none
    assert nlp.row_group[2] == nlp.row_group[3] and nlp.row_rho[2] == 1.0 and nlp.row_rho[3] == -1.0
    assert nlp.row_group[5] == -1
    cl, cu = spp.bounds(1e-6)
    r = nlp.row_group[2]
    assert cu[r] == pytest.approx(1e-6) and cl[r] == pytest.approx(-0.5 - 1e-6)
    x = np.array([0.1, 0.5, 0.2, 0.6])
    with torch.no_grad():
        y = Ops().double()(torch.tensor(x)[None])[0].numpy()
    assert np.allclose(spp.raw_rows(x), y[1:], rtol=0, atol=1e-14)


def test_jacobian_pattern_lies_inside_the_walk_supports():
    model = Ops().double()
    prob = problem_of(model, [-0.8, 0.1, -0.5, 0.2], [0.9, 1.3, 0.7, 1.1])
    spp = sd.sparse_problem(prob)
    forms, algebra, _, _ = sd._WALKS[model]
    nlp = spp.nlp
    first = {}
    for i, g in enumerate(nlp.row_group.tolist()):
        if g >= 0:
            first.setdefault(g, i)
    for r, c in zip(nlp.tape.jac_rows.tolist(), nlp.tape.jac_cols.tolist()):
        assert c in algebra.support(forms[1 + first[r]])


def test_an_atom_without_a_second_derivative_falls_back_to_quasi_newton():
    spp = sd.sparse_problem(problem_of(WithAbs(), [-1.0, -1.0], [1.0, 1.0]))
    assert spp is not None and not spp.hessian_available
    options = capture_ipopt_options(problem_of(WithAbs(), [-1.0, -1.0], [1.0, 1.0]), True)
    assert options['hessian_approximation'] == 'limited-memory'
    options = capture_ipopt_options(problem_of(Bilinear(), [0.0, 0.0], [1.0, 1.0]), True)
    assert options['hessian_approximation'] == 'exact'


def test_a_tape_that_does_not_reproduce_the_model_is_refused():
    assert sd.sparse_problem(problem_of(Cancels(), [1e6], [1e6 + 0.02])) is None
    assert sd.sparse_problem(problem_of(Cancels(), [-1.0], [1.0])) is not None     # no cancellation there


def capture_ipopt_options(problem, sparse):
    if not ipopt_solver.CYIPOPT_AVAILABLE:
        pytest.skip('cyipopt not available')
    options = {}

    class FakeProblem:
        def __init__(self, *args, **kwargs):
            options['m'] = kwargs['m']

        def add_option(self, key, value):
            options[key] = value

        def solve(self, x0):
            return np.asarray(x0, dtype=np.float64), {'status': 0, 'status_msg': b'ok'}

    solver = IPOPTSolver({'sparse_derivatives': sparse, 'ipopt_constr_viol_tol': 1e-6})
    with mock.patch.object(ipopt_solver.cyipopt, 'Problem', FakeProblem):
        solver._solve_single_cyipopt(problem, 0.5 * (problem.lb + problem.ub), 5.0, slack=0.0)
    return options


def test_ipopt_with_sparse_derivatives_reaches_the_verified_optimum():
    if not ipopt_solver.CYIPOPT_AVAILABLE:
        pytest.skip('cyipopt not available')
    prob = problem_of(Bilinear(), [0.0, 0.0], [1.0, 1.0])
    res = IPOPTSolver({'sparse_derivatives': True, 'ipopt_constr_viol_tol': 1e-6,
                       'ipopt_constraint_slack_frac': 0.99}).solve_single(prob, timeout=10.0)
    assert res.x_solution is not None
    assert res.objective_value == pytest.approx(-0.25, abs=1e-6)
    assert res.constraint_violation <= 1e-6
    assert capture_ipopt_options(problem_of(Bilinear(), [0.0, 0.0], [1.0, 1.0]), True)['m'] == 1   # one ranged row


def test_flag_off_never_builds_a_tape_and_keeps_the_torch_rows():
    prob = problem_of(Bilinear(), [0.0, 0.0], [1.0, 1.0])
    with mock.patch.object(ipopt_solver, 'sparse_problem', side_effect=AssertionError('tape built')):
        options = capture_ipopt_options(prob, False)
    assert options['hessian_approximation'] == 'limited-memory' and options['m'] == 2


def test_unmerged_callbacks_preserve_raw_rows_jacobian_and_lagrangian():
    model = Ops().double()
    prob = problem_of(model, [-0.8, 0.1, -0.5, 0.2], [0.9, 1.3, 0.7, 1.1])
    spp = sd.sparse_problem(prob)
    raw = spp.unmerged()
    x = np.array([0.1, 0.5, 0.2, 0.6])
    assert raw.m == prob.n_constraints
    assert np.allclose(raw.constraints(x), prob.constraints(x))
    r, c = raw.jacobianstructure()
    assert np.allclose(dense(r, c, raw.jacobian(x), (raw.m, prob.n_vars)), prob.jacobian(x))
    lam = np.random.default_rng(0).normal(size=raw.m)
    H = tape_hessian(raw, x, lam, 0.7)
    ref = torch.autograd.functional.hessian(
        lambda z: (0.7 * model(z[None])[0, 0]
                   + (torch.tensor(lam) * model(z[None])[0, 1:]).sum()), torch.tensor(x)).numpy()
    assert np.allclose(H, ref, rtol=1e-12, atol=1e-12)
    lo, hi = raw.bounds(1e-6)
    assert np.isneginf(lo).all() and (hi == 1e-6).all()


class DependentEqualities(nn.Module):
    def forward(self, x):
        a, b = x[:, :1], x[:, 1:2]
        rows = (a + b - 1, a - b, 2 * a + b - 1.5)
        return torch.cat([-a * b, *[s for r in rows for s in (r, -r)]], 1)


def test_sparse_ipopt_recovers_from_overdetermined_merged_equalities():
    # Three dependent equalities in two variables. IPOPT rejects the merged
    # formulation before iteration 1; the original inequalities remain solvable.
    if not ipopt_solver.CYIPOPT_AVAILABLE:
        pytest.skip('cyipopt not available')
    prob = problem_of(DependentEqualities(), [0., 0.], [1., 1.])
    cfg = {'sparse_derivatives': True, 'ipopt_constr_viol_tol': 1e-6}
    solver = IPOPTSolver(cfg)
    failed = solver._solve_single_cyipopt(prob, np.array([.2, .3]), 5., slack=0.)
    assert failed.exit_code == -10 and failed.x_solution is None
    recovered = solver.solve_single(prob, x_init=np.array([.2, .3]), timeout=5.)
    assert recovered.x_solution is not None
    assert recovered.objective_value == pytest.approx(-.25, abs=1e-6)
    assert recovered.constraint_violation <= 1e-6


@pytest.mark.skipif(not os.path.isdir(BENCH), reason='eqexact benchmarks not available')
def test_pooling_bental5stp_sparse_ipopt_keeps_v406_convergence():
    if not ipopt_solver.CYIPOPT_AVAILABLE:
        pytest.skip('cyipopt not available')
    import structure_instances as si
    for name in [m for m in sys.modules if m.split('.')[0] in ('modules', 'mod_files_gams', 'py_minlplib_linear')]:
        del sys.modules[name]
    data = si.load('minlplib_linear', 'pooling_bental5stp', {}, root=BENCH)
    model = data['model']
    base = problem_of(model, data['L'].tolist(), data['U'].tolist())
    # The production root has both proof-time expanded output rows and the
    # original input rows. Preserve that layout, including its dependencies.
    prob = NNProblemIPOPT(model, data['L'], data['U'], base.C_obj, base.rhs_obj,
                          base.C_constr, base.rhs_constr,
                          input_constr_A=model.constraint_A, input_constr_b=model.constraint_b)
    result = IPOPTSolver({'sparse_derivatives': True, 'ipopt_constr_viol_tol': 1e-6,
                          'ipopt_constraint_slack_frac': .99}).solve_single(prob, timeout=30.)
    assert result.x_solution is not None
    assert result.constraint_violation <= 1e-6
    assert result.objective_value == pytest.approx(-3500., abs=1e-3)


def _v406_output_supports(model, x0, lower, input_shape):
    """gear/v406-fixes' _output_supports, verbatim except for the cache (the reference of the refactor)"""
    import copy
    import scipy.sparse as sp
    from auto_LiRPA import BoundedModule
    from nlp_solver.structure.forms import FormAlgebra
    from nlp_solver.structure.walk import build_forms, output_forms
    from nlp_solver.utils import _infer_device_dtype
    private = copy.deepcopy(model).cpu()
    dtype = _infer_device_dtype(private, torch.device('cpu'), torch.float64)[1]
    x = torch.as_tensor(x0, dtype=dtype).reshape(1, *input_shape)
    bounded = BoundedModule(private, x, bound_opts={'pow_negative_exponent': True}, device='cpu')
    algebra = FormAlgebra(x.numel(), lower)
    found = [algebra.support(f) for f in output_forms(build_forms(bounded, x, algebra))]
    sizes = [len(s) for s in found]
    return sp.csr_matrix(
        (np.ones(sum(sizes), dtype=bool),
         np.fromiter((v for s in found for v in sorted(s)), dtype=np.int64, count=sum(sizes)),
         np.concatenate([[0], np.cumsum(sizes)])),
        shape=(len(found), x.numel()))


def test_output_supports_are_those_of_v406():
    lo, hi = np.array([-0.8, 0.1, -0.5, 0.2]), np.array([0.9, 1.3, 0.7, 1.1])
    new = ipopt_solver._output_supports(Ops().double(), 0.5 * (lo + hi), lo, (4,))
    old = _v406_output_supports(Ops().double(), 0.5 * (lo + hi), lo, (4,))
    assert new.shape == old.shape and (new != old).nnz == 0


SAMPLE = [('minlplib_linear', n) for n in ('pooling_haverly1pq', 'waterund01', 'wastewater02m1', 'kall_circles_c6a',
                                           'pooling_foulds3tp', 'polygon25', 'ex6_2_13', 'kriging_peaks-full050')] \
    + [('gams_world', n) for n in ('ex8_3_14', 'cute_hs110', 'fekete3', 'cute_avion2', 'Paviani', 'MieleCantrell',
                                   'schitt_s376', 'elena_chi')]


@pytest.mark.skipif(not os.path.isdir(BENCH), reason='eqexact benchmarks not available')
@pytest.mark.parametrize('bench,name', SAMPLE)
def test_benchmark_models_match_autograd(bench, name):
    import structure_instances as si
    for m in [m for m in sys.modules if m.split('.')[0] in ('modules', 'mod_files_gams', 'py_minlplib_linear')]:
        del sys.modules[m]
    d = si.load(bench, name, {}, root=BENCH)
    model, L, U = d['model'], d['L'].numpy(), d['U'].numpy()
    spp = sd.sparse_problem(problem_of(model, L, U))
    assert spp is not None
    lo = np.where(np.isfinite(L), L, -1.0)
    hi = np.where(np.isfinite(U), U, 1.0)
    rng = np.random.default_rng(0)
    pts = [lo + (hi - lo) * rng.random(len(lo)) for _ in range(2)]
    err = check_against_torch(model, spp, pts, hessian=len(lo) <= 60 and spp.hessian_available)
    assert err['value'] < 1e-10 and err['grad'] < 1e-10 and err['jac'] < 1e-10 and err['hess'] < 1e-9
    assert err['jpat'] == 0 and err['hpat'] == 0
