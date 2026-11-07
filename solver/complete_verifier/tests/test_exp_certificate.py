"""CPU recognition, outward entropy cuts, row semantics and root fallback."""
from fractions import Fraction
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from auto_LiRPA import BoundedModule

import arguments
from nlp_solver.exp_certificate import EntropyTerm, _rounded_row, build_exp_certificate
from nlp_solver.structure import RowPool


@pytest.fixture(autouse=True)
def config(monkeypatch):
    conf = arguments.ConfigHandler()
    conf.construct_config_dict(conf.default_args)
    conf['solving']['solving_mode'] = True
    conf['solver']['nlp']['constr_viol_tol'] = 1e-4
    monkeypatch.setattr(arguments, 'Config', conf)
    torch.set_default_dtype(torch.float64)
    return conf


def setup(model, n=3, lower=None, upper=None, C=None, rhs=None):
    lo = torch.tensor([lower or [-3.] * n])
    hi = torch.tensor([upper or [2.] * n])
    x = (lo + hi) / 2
    model = model.double().eval()
    bm = BoundedModule(model, x, device='cpu')
    m = model(x).numel()
    C = torch.eye(m).unsqueeze(0) if C is None else C
    rhs = torch.tensor([[float('inf')] + [0.] * (m-1)]) if rhs is None else rhs
    diag = {}
    cert = build_exp_certificate(bm, model, x, C, rhs, lo, hi, diag, rowpool=RowPool(rhs))
    return cert, diag, bm, model, x, lo, hi, C, rhs


class Gibbs(torch.nn.Module):
    def __init__(self, eps=0., seed=0):
        super().__init__()
        self.eps = eps
        rng = np.random.default_rng(seed)
        self.register_buffer('c', torch.tensor(rng.uniform(-3., 1., 3)))
        self.register_buffer('a', torch.tensor(rng.uniform(.5, 2., 3)))

    def forward(self, x):
        y = torch.exp(x)
        b = (y * self.a).sum(1, keepdim=True) + self.eps
        obj = (y * (x + self.c - torch.log(b))).sum(1, keepdim=True) + 7
        row = y.sum(1, keepdim=True) - 2
        return torch.cat([obj, row - 1e-5, -row - 1e-5], dim=1)


@pytest.mark.parametrize('eps', [0., 1e-4])
def test_accepts_gibbs_and_caps_large_exp_box(eps):
    cert, diag, *_ = setup(Gibbs(eps), lower=[-100.] * 3, upper=[100.] * 3)
    assert cert is not None, diag
    assert len(cert.terms) == 3
    assert np.all(cert.upper > 2 + 1.1e-4)
    assert np.all(cert.upper < 2.001)
    assert cert.solve(time_budget=.5)[0] > -100


class Bad(torch.nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant

    def forward(self, x):
        y = torch.exp(x)
        b = y.sum(1, keepdim=True)
        if self.variant == 'nonaffine':
            b = b * b
        if self.variant == 'nonpositive':
            b = b - 100
        obj = (y * (x - torch.log(b))).sum(1, keepdim=True)
        if self.variant == 'negative':
            obj = -obj
        if self.variant == 'raw_exp_input':
            obj = obj + x[:, :1]
        if self.variant == 'positive_log':
            obj = torch.log(b)
        if self.variant == 'extra_negative_logy':
            obj = obj - y[:, :1] * x[:, :1]
        return obj


@pytest.mark.parametrize('variant', ['negative', 'nonaffine', 'nonpositive', 'raw_exp_input',
                                     'positive_log', 'extra_negative_logy'])
def test_rejects_unproved_or_nonconvex_variants(variant):
    cert, diag, *_ = setup(Bad(variant))
    assert cert is None, diag


@pytest.mark.parametrize('seed', range(5))
def test_random_gibbs_bound_below_sampled_relaxed_feasible_minimum(seed):
    model = Gibbs(1e-4 if seed % 2 else 0., seed)
    cert, diag, *_ = setup(model)
    assert cert is not None, diag
    bound, rounds = cert.solve(time_budget=1.)
    points = torch.tensor(np.random.default_rng(seed + 99).uniform(-3., 2., (20000, 3)))
    y = torch.exp(points)
    points = torch.log(2 * y / y.sum(1, keepdim=True))
    feasible = (points >= -3.).all(1) & (points <= 2.).all(1)
    values = model(points[feasible])
    assert rounds > 0 and np.isfinite(bound)
    assert (values[:, 1:] <= 1e-4).all()
    assert bound <= float(values[:, 0].min())


class OneVariable(torch.nn.Module):
    def __init__(self, row_sign=1):
        super().__init__()
        self.row_sign = row_sign

    def forward(self, x):
        y = torch.exp(x)
        return torch.cat([y * (x - 2), self.row_sign * (y - 2 - 1e-5)], dim=1)


def test_row_tolerance_and_signed_specification(config):
    cert, diag, *_ = setup(OneVariable(), n=1)
    assert cert is not None, diag
    bound, _ = cert.solve(time_budget=.5)
    expected = (2 + 1.1e-4) * (np.log(2 + 1.1e-4) - 2)
    assert expected - 1e-6 <= bound <= expected
    config['solver']['nlp']['constr_viol_tol'] = 0.
    tight, *_ = setup(OneVariable(), n=1)
    tight_bound, _ = tight.solve(time_budget=.5)
    assert bound < tight_bound
    signed, diag, *_ = setup(OneVariable(-1), n=1, C=torch.tensor([[[1., 0.], [0., -1.]]]))
    assert signed is not None, diag
    assert signed.solve(time_budget=.5)[0] == tight_bound


class Mixed(torch.nn.Module):
    def forward(self, x):
        a, z = x[:, :1], x[:, 1:]
        y = torch.exp(a)
        obj = y * a - torch.log(1 + y) + 2 * z
        return torch.cat([obj, y + z - 2, y * z - 1], dim=1)


def test_affine_raw_variable_and_dropped_nonlinear_row():
    cert, diag, *_ = setup(Mixed(), n=2, lower=[-3., 0.], upper=[2., 1.])
    assert cert is not None, diag
    assert diag['exp_variables'] == [0]
    assert diag['dropped_rows'] == [2]
    assert len(cert.terms) == 2
    assert np.isfinite(cert.solve(time_budget=.5)[0])


def test_interval_rounded_entropy_support_is_global():
    term = EntropyTerm(Fraction(7, 3), (Fraction(0), Fraction(1), Fraction(0)),
                       (Fraction(1, 7), Fraction(2, 3), Fraction(5, 4)))
    lo, hi = np.array([1e-12, .001]), np.array([3., 5.])
    slope, offset = term.support(np.array([.12, 2.3]), lo, hi)
    points = np.random.default_rng(4).uniform(lo, hi, (10000, 2))
    a, b = points[:, 0], 1/7 + 2/3 * points[:, 0] + 5/4 * points[:, 1]
    assert np.all(points @ slope + offset <= 7/3 * a * np.log(a / b))


def test_interval_rounded_rational_row_contains_exact_feasible_points():
    data = (Fraction(2, 7), Fraction(1, 3), Fraction(-7, 5))
    lo, hi = np.array([1e-20, -4.]), np.array([3., 5.])
    bound = Fraction(3, 11)
    slope, rhs = _rounded_row(data, bound, lo, hi)
    # Generate points on the exact rational boundary, evaluate feasibility with
    # Fractions, and check the rounded LP cannot discard any of them.
    for value in np.linspace(lo[0], hi[0], 100):
        a = Fraction(float(value))
        b = (bound - data[0] - data[1] * a) / data[2]
        point = [a, b]
        lhs = sum(Fraction(float(c)) * p for c, p in zip(slope, point))
        assert lhs <= Fraction(rhs)


def test_nonexp_variable_must_be_affine_in_objective():
    class NonlinearMixed(Mixed):
        def forward(self, x):
            output = super().forward(x)
            return torch.cat([output[:, :1] + x[:, 1:] * x[:, 1:], output[:, 1:]], dim=1)
    cert, diag, *_ = setup(NonlinearMixed(), n=2)
    assert cert is None, diag


def test_sanity_mismatch_and_random_state_preservation():
    cert, diag, bm, model, x, lo, hi, C, rhs = setup(Gibbs())
    assert cert is not None, diag
    class Wrong(torch.nn.Module):
        def forward(self, z):
            return model(z) + 1
    numpy_state, torch_state = np.random.get_state(), torch.random.get_rng_state()
    assert build_exp_certificate(bm, Wrong(), x, C, rhs, lo, hi, diag, rowpool=RowPool(rhs)) is None
    assert diag['reason'] == 'symbolic/model sanity mismatch'
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert torch.equal(torch_state, torch.random.get_rng_state())


def test_untrusted_lp_status_and_numeric_failure():
    cert, *_ = setup(Gibbs())
    with patch('scipy.optimize.linprog', return_value=SimpleNamespace(status=2)):
        assert cert.solve(time_budget=1.)[0] == -np.inf
    with patch('scipy.optimize.linprog', side_effect=ValueError('numeric range')):
        assert cert.solve(time_budget=1.) == (-np.inf, 0)


def test_defaults_and_zero_time(config):
    assert config['solver']['nlp']['exp_certificate'] is False
    assert config['solver']['nlp']['exp_certificate_time'] == 20.
    cert, *_ = setup(Gibbs())
    assert cert.solve(time_budget=0.) == (-np.inf, 0)


def test_highs_pool_initialized_by_prior_root_lp():
    # A fresh process is necessary because HiGHS retains its global thread pool.
    # The production root pipeline can solve an LP before trying this certificate.
    code = '''
import arguments
import torch
from scipy.optimize import linprog
from tests.test_exp_certificate import OneVariable, setup
torch.set_default_dtype(torch.float64)
conf = arguments.ConfigHandler()
conf.construct_config_dict(conf.default_args)
conf['solving']['solving_mode'] = True
arguments.Config = conf
assert linprog([1.], bounds=[(0., 1.)], method='highs').status == 0
cert, diag, *_ = setup(OneVariable(), n=1)
assert cert is not None, diag
assert cert.solve(time_budget=1.)[0] > -3
'''
    subprocess.run([sys.executable, '-c', code], check=True, timeout=30,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                            OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'))


@pytest.mark.parametrize('enabled,proposed', [(False, 99.995), (True, 99.995),
                                            (True, 99.), (True, 100.1)])
def test_root_finish_and_fallback(enabled, proposed):
    from input_split.batch_branch_and_bound import input_bab_parallel
    from tests.test_batch_branch_and_bound import _setup_mock_config
    config = _setup_mock_config()
    config['solving']['solving_mode'] = True
    config['solver']['nlp']['exp_certificate'] = enabled
    lo, hi = torch.full((1, 2), -3.), torch.ones(1, 2)
    x = SimpleNamespace(ptb=SimpleNamespace(x_L=lo, x_U=hi), shape=lo.shape)
    ref = {'global_lb': torch.tensor([[10., -1.]]), 'lA': {'input': torch.ones(1, 2, 2)},
           'pre_incomplete_primal': {'verified': True, 'nlp_found': False, 'objective': 100.,
                                    'x': torch.full((2,), .5), 'peak_mem_bytes': None, 'total_mem_bytes': None}}
    net = SimpleNamespace(net=MagicMock(), model_ori=MagicMock(), final_name='final')
    net.net.input_name = ['input']
    net.net.get_splittable_activations.return_value = []
    net.net.bound_opts = {}
    cert = MagicMock()
    cert.solve.return_value = (proposed, 2)
    with patch.object(arguments, 'Config', config), \
            patch('input_split.batch_branch_and_bound.run_root_diagnostics_and_budget'), \
            patch('input_split.batch_branch_and_bound.UnsortedInputDomainList',
                  side_effect=lambda *args, **kw: MagicMock(history=kw['history'])), \
            patch('nlp_solver.exp_certificate.build_exp_certificate', return_value=cert) as build, \
            patch('input_split.lifted_bound.RootLifted.build', side_effect=StopIteration), \
            patch('input_split.batch_branch_and_bound._finish_with_lifted_dual', return_value='certified') as finish:
        if enabled and proposed == 99.995:
            assert input_bab_parallel(net, x, torch.eye(2).unsqueeze(0), torch.zeros(1, 2),
                                      torch.tensor([2]), reference_dict=ref) == 'certified'
            assert finish.call_args.args[-2] <= proposed
        else:
            with pytest.raises(StopIteration):
                input_bab_parallel(net, x, torch.eye(2).unsqueeze(0), torch.zeros(1, 2),
                                   torch.tensor([2]), reference_dict=ref)
            finish.assert_not_called()
        assert build.call_count == int(enabled)
    assert torch.equal(ref['global_lb'], torch.tensor([[10., -1.]]))


def test_cert_crown_dual_replaces_linprog_and_matches_highs(config):
    from nlp_solver.structure import cert_kernels
    cert, diag, *_ = setup(Gibbs(1e-4, 1))
    assert cert is not None, diag
    ref = cert.solve(time_budget=5.)[0]
    config['solver']['nlp']['cert_crown_dual'] = True
    with patch('scipy.optimize.linprog', side_effect=AssertionError('HiGHS called')), \
            patch.object(cert_kernels, 'halpern_lp', wraps=cert_kernels.halpern_lp) as kernel:
        bound = cert.solve(time_budget=20.)[0]
    assert kernel.call_count > 0
    assert np.isfinite(bound) and bound <= ref + 1e-6 * max(1., abs(ref)) and ref - bound <= 1e-4 * max(1., abs(ref))
    with patch.object(cert_kernels, 'halpern_lp', side_effect=AssertionError('kernel called')):
        config['solver']['nlp']['cert_crown_dual'] = False
        assert cert.solve(time_budget=2.)[0] == pytest.approx(ref, abs=1e-6)
