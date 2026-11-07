"""CPU structural GP proof, row semantics and outward dual certificate tests."""
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pytest
import torch
from auto_LiRPA import BoundedModule

import arguments
from nlp_solver.gp_certificate import (
    Form, _lp_dual, _pow, _support, build_gp_certificate, output_forms,
)
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


def setup(model, n=2, rhs=None, lower=None, upper=None, C=None):
    lower = torch.full((1, n), .2) if lower is None else torch.tensor([lower])
    upper = torch.full((1, n), 3.) if upper is None else torch.tensor([upper])
    x = (lower + upper) / 2
    model = model.double().eval()
    bm = BoundedModule(model, x, device='cpu', bound_opts={'pow_negative_exponent': True})
    m = model(x).numel()
    C = torch.eye(m).unsqueeze(0) if C is None else C
    rhs = torch.tensor([[float('inf')] + [0.] * (m - 1)]) if rhs is None else rhs
    diag = {}
    gp = build_gp_certificate(bm, model, x, C, rhs, lower, upper, diag, rowpool=RowPool(rhs))
    return gp, diag, bm, x, lower, upper


class Supported(torch.nn.Module):
    def forward(self, x):
        a, b = x[:, :1], x[:, 1:]
        mono = torch.exp(1.3 * torch.log(2 * a) - .7 * torch.log(b) + .2)
        p = (a + 2 * b) ** 2
        obj = mono * p + torch.sqrt(a) + a / b + 1 / b + a ** -1.2 + 7
        return torch.cat([obj, a + b - 3], dim=1)


def test_supported_operations_and_offset():
    gp, diag, bm, x, lower, upper = setup(Supported())
    assert gp is not None, diag
    assert gp.offset == 7
    assert gp.objective.positive()
    assert gp.solve(time_budget=2.)[0] > 7
    assert bm(x).shape == (1, 2)


class Bad(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x):
        a, b = x[:, :1], x[:, 1:]
        if self.op == 'negative':
            return a - b
        if self.op == 'exp_nonmonomial_log':
            return torch.exp(torch.log(a + b) + torch.log(a))
        if self.op == 'exp_input':
            return torch.exp(a + torch.log(b))
        if self.op == 'fractional_posynomial':
            return (a + b) ** 1.5
        if self.op == 'div_posynomial':
            return a / (a + b)
        if self.op == 'variable_power':
            return a ** b
        if self.op == 'negative_base':
            return torch.exp(torch.log(-a))
        return torch.sin(a)


@pytest.mark.parametrize('op', ['negative', 'exp_nonmonomial_log', 'exp_input',
                                'fractional_posynomial', 'div_posynomial',
                                'variable_power', 'sin'])
def test_rejects_unsupported_or_negative_forms(op):
    gp, diag, *_ = setup(Bad(op))
    assert gp is None, diag


def test_nonpositive_root_box_is_rejected():
    assert setup(Supported(), lower=[0., .2])[0] is None


class Rows(torch.nn.Module):
    def __init__(self, reverse=False):
        super().__init__()
        self.reverse = reverse

    def forward(self, x):
        a = x[:, :1]
        if self.reverse:
            return torch.cat([1 / a, 2 - a], dim=1)
        return torch.cat([1 / a + 5, a - 2], dim=1)


def test_constraint_tolerance_and_signed_C(config):
    gp, diag, *_ = setup(Rows(), n=1)
    assert gp is not None, diag
    dual, _ = gp.solve(time_budget=2.)
    exact = 5 + 1 / (2 + 1e-4)
    assert exact - 1e-8 <= dual <= exact
    config['solver']['nlp']['constr_viol_tol'] = 0.
    tight, *_ = setup(Rows(), n=1)
    tight_dual, _ = tight.solve(time_budget=2.)
    assert dual < tight_dual <= 5.5
    # Output signs are interpreted only after applying the actual specification.
    C = torch.tensor([[[1., 0.], [0., -1.]]])
    signed, diag, *_ = setup(Rows(reverse=True), n=1, C=C)
    assert signed is not None, diag


def test_finite_wrong_sense_row_is_dropped_from_the_relaxation():
    # Leaving a row out only enlarges the relaxed feasible set, so its minimum stays a
    # valid lower bound (hs101-103: "objective >= 100").
    signed, diag, *_ = setup(Rows(reverse=True), n=1)
    assert signed is not None and diag['dropped_rows'] == [1], diag
    rhs = torch.tensor([[float('inf'), float('inf')]])
    signed, diag, *_ = setup(Rows(reverse=True), n=1, rhs=rhs)
    assert signed is not None and diag['dropped_rows'] == [], diag


class Equality(torch.nn.Module):
    def forward(self, x):
        a, b = x[:, :1], x[:, 1:]
        mono = 2 * a / b
        return torch.cat([a, mono - 1, 1 - mono], dim=1)


def test_monomial_equality_keeps_relaxed_slab(config):
    gp, diag, *_ = setup(Equality(), lower=[.1, 1.], upper=[2., 2.])
    assert gp is not None, diag
    dual, _ = gp.solve(time_budget=2.)
    assert .5 * (1 - 1e-4) - 1e-8 <= dual <= .5 * (1 - 1e-4)
    # The lower half is inactive: the remaining upper half is still GP-usable.
    rhs = torch.tensor([[float('inf'), 0., float('inf')]])
    relaxed, *_ = setup(Equality(), lower=[.1, 1.], upper=[2., 2.], rhs=rhs)
    assert relaxed is not None


class RandomGP(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.register_buffer('coefficients', torch.tensor(rng.uniform(.3, 2., (4,))))
        self.register_buffer('powers', torch.tensor(rng.uniform(-2., 2., (4, 3))))

    def forward(self, x):
        terms = self.coefficients * torch.exp(torch.log(x) @ self.powers.T)
        return torch.cat([terms.sum(1, keepdim=True) - 2,
                          x.sum(1, keepdim=True) - 4], dim=1)


@pytest.mark.parametrize('seed', range(5))
def test_random_gp_bound_below_sampled_feasible_minimum(seed):
    model = RandomGP(seed)
    gp, diag, *_ = setup(model, n=3)
    assert gp is not None, diag
    bound, _ = gp.solve(time_budget=2.)
    rng = np.random.default_rng(seed + 12)
    points = torch.tensor(rng.uniform(.2, 3., (20000, 3)))
    values = model(points)
    feasible = values[:, 1] <= 1e-4
    assert feasible.any()
    assert bound <= float(values[feasible, 0].min())


def test_rounded_support_plane_is_below_log_posynomial():
    form = Form('poly', {(Fraction(1, 3), Fraction(-7, 5)): Fraction(11, 3),
                         (Fraction(-8, 9), Fraction(5, 3)): Fraction(2, 7)})
    lo, hi = np.array([-9., -4.]), np.array([7., 10.])
    slope, offset = _support(form, np.array([1.2, -.7]), lo, hi)
    points = np.random.default_rng(3).uniform(lo, hi, (10000, 2))
    scores = np.array([np.log(float(c)) + points @ np.array([float(e) for e in a])
                       for a, c in form.terms.items()])
    from scipy.special import logsumexp
    assert np.all(points @ slope + offset <= logsumexp(scores, axis=0))


def test_dual_is_valid_with_nonstationary_or_wrong_sign_multipliers():
    c = np.array([1., 0.])
    G, rhs = np.array([[-1., -1.]]), np.array([-1.])
    lo, hi = np.zeros(2), np.ones(2)
    # True min is zero. Arbitrary multipliers cannot manufacture a positive bound.
    for multiplier in [-1., 0., .1, 1., 100000., 1. + 1e-15]:
        assert _lp_dual(c, G, rhs, lo, hi, np.array([multiplier])) <= 0


def test_sanity_mismatch_disables_structural_certificate():
    gp, diag, bm, x, lo, hi = setup(Rows(), n=1)
    class WrongModel(torch.nn.Module):
        def forward(self, z):
            return Rows()(z) + 1
    rhs = torch.tensor([[float('inf'), 0.]])
    assert build_gp_certificate(bm, WrongModel(), x, torch.eye(2).unsqueeze(0), rhs, lo, hi, diag,
                                rowpool=RowPool(rhs)) is None
    assert diag['reason'] == 'symbolic/model sanity mismatch'


def test_untrusted_lp_status_never_certifies_infeasibility():
    from types import SimpleNamespace
    gp, *_ = setup(Rows(), n=1)
    with patch('scipy.optimize.linprog', return_value=SimpleNamespace(status=2)):
        assert gp.solve(time_budget=1.)[0] == -np.inf


def test_numeric_lp_failure_falls_back():
    gp, *_ = setup(Rows(), n=1)
    with patch('scipy.optimize.linprog', side_effect=ValueError('invalid numeric range')):
        assert gp.solve(time_budget=1.) == (-np.inf, 0)


def test_time_cap_and_defaults(config):
    assert config['solver']['nlp']['gp_certificate'] is False
    assert config['solver']['nlp']['gp_certificate_time'] == 20.
    gp, *_ = setup(Rows(), n=1)
    assert gp.solve(time_budget=0.) == (-np.inf, 0)


def test_expansion_cap_is_conservative():
    form = Form('poly', {(Fraction(j),): Fraction(1) for j in range(30)})
    assert _pow(form, Fraction(2)).kind == 'other'


class Movement(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weights', torch.tensor([[2., 3., 4.]]))

    def forward(self, x):
        moved = x.unsqueeze(2).transpose(1, 2).reshape(-1, 3)
        moved = torch.cat([moved[:, :1], moved[:, 1:]], dim=1)
        return torch.nn.functional.linear(moved, self.weights, torch.tensor([5.]))


def test_data_movement_and_linear_constant_bias():
    gp, diag, bm, x, lo, _ = setup(Movement(), n=3)
    assert gp is not None, diag
    assert gp.offset == 5
    assert len(output_forms(bm, x, lo.numpy().reshape(-1))[0].terms) == 4


class GeneratedResidual(torch.nn.Module):
    def forward(self, x):
        a, b = x[:, :1], x[:, 1:]
        # The generator distributes its >= row's sign before assembling outputs.
        negative_product = (-.5 * a) * b
        row = -(negative_product + (-.2 * a) / b) - 1
        return torch.cat([1 / a + b, row], dim=1)


def test_generated_signed_residual_becomes_positive_only_at_row_boundary():
    gp, diag, *_ = setup(GeneratedResidual())
    assert gp is not None, diag


@pytest.mark.parametrize('enabled,proposed', [(False, 99.995), (True, 99.995),
                                            (True, 99.), (True, 100.1)])
def test_root_integration_certificate_and_fallback(config, enabled, proposed):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from input_split.batch_branch_and_bound import input_bab_parallel
    from tests.test_batch_branch_and_bound import _setup_mock_config

    # Use the same root pipeline fixture as existing lifted-LP handoff tests, and
    # stop at the next stage to prove that an inconclusive GP leaves BaB unchanged.
    config = _setup_mock_config()
    config['solving']['solving_mode'] = True
    config['solver']['nlp']['gp_certificate'] = enabled
    lo, hi = torch.full((1, 2), .1), torch.ones(1, 2)
    x = SimpleNamespace(ptb=SimpleNamespace(x_L=lo, x_U=hi), shape=lo.shape)
    ref = {
        'global_lb': torch.tensor([[10., -1.]]),
        'lA': {'input': torch.ones(1, 2, 2)},
        'pre_incomplete_primal': {'verified': True, 'nlp_found': False,
                                 'objective': 100., 'x': torch.full((2,), .5),
                                 'peak_mem_bytes': None, 'total_mem_bytes': None},
    }
    net = SimpleNamespace(net=MagicMock(), model_ori=MagicMock(), final_name='final')
    net.net.input_name = ['input']
    net.net.get_splittable_activations.return_value = []
    net.net.bound_opts = {}
    gp = MagicMock()
    gp.solve.return_value = (proposed, 2)
    with patch.object(arguments, 'Config', config), \
            patch('input_split.batch_branch_and_bound.run_root_diagnostics_and_budget'), \
            patch('input_split.batch_branch_and_bound.UnsortedInputDomainList',
                  side_effect=lambda *args, **kw: MagicMock(history=kw['history'])), \
            patch('nlp_solver.gp_certificate.build_gp_certificate', return_value=gp) as build, \
            patch('input_split.lifted_bound.RootLifted.build', side_effect=StopIteration), \
            patch('input_split.batch_branch_and_bound._finish_with_lifted_dual',
                  return_value='certified') as finish:
        if enabled and proposed == 99.995:
            result = input_bab_parallel(net, x, torch.eye(2).unsqueeze(0),
                                        torch.zeros(1, 2), torch.tensor([2]), reference_dict=ref)
            assert result == 'certified'
            assert finish.call_args.args[-2] <= proposed
        else:
            with pytest.raises(StopIteration):
                input_bab_parallel(net, x, torch.eye(2).unsqueeze(0),
                                    torch.zeros(1, 2), torch.tensor([2]), reference_dict=ref)
            finish.assert_not_called()
        assert build.call_count == int(enabled)
    assert torch.equal(ref['global_lb'], torch.tensor([[10., -1.]]))



def test_cert_crown_dual_flag_off_still_uses_linprog_only():
    from nlp_solver.structure import cert_kernels
    gp, *_ = setup(Supported())
    with patch.object(cert_kernels, 'halpern_lp', side_effect=AssertionError('kernel called')):
        assert gp.solve(time_budget=2.)[0] > 7

