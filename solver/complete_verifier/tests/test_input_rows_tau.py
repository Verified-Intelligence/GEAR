"""The base input rows A x <= b are relaxed by constr_viol_tol exactly once, at the source.

The primal and the verifier accept A x <= b + constr_viol_tol, so every dual bound has to bound the
relaxed set. vnnlibHandler builds the standard-form rows every bound consumer reads (root CROWN pass,
BaB: clipping, constrained concretization, the domain filter, the dual-ascent LP, the lifted OBBT) from
the problem's A x <= b through relax_input_constraints; A x <= b itself stays raw for the primal.
The consumers use the rows they are given as they are. ex7_2_1 (the reproducer of the soundness alarm):
its tau-optimum sits at A2 x - b2 = +1e-4, so a raw row cuts it off.
"""
import os
import re
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arguments  # noqa: E402
from input_constraints import (  # noqa: E402
    affine_constraints_domain_lower_bound,
    input_constraints_to_standard_form,
    standard_form_to_input_constraints,
)
from nlp_solver.structure import RowPool  # noqa: E402

TOL = 1e-4
F64 = torch.float64
BENCHMARKS = str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'benchmarks' / 'minlplib_linear')

# A synthetic problem: the box [0, 1]^2 and one input row x0 + x1 <= 1.
A0 = torch.tensor([[[1.0, 1.0]]], dtype=F64)
B0 = torch.tensor([[1.0]], dtype=F64)
LO, HI = [0.0, 0.0], [1.0, 1.0]


@pytest.fixture(autouse=True)
def solving_config(monkeypatch):
    config = arguments.ConfigHandler()
    config.construct_config_dict(config.default_args)
    config.file = None
    config['solving']['solving_mode'] = True
    config['solver']['nlp']['constr_viol_tol'] = TOL
    config['specification']['norm'] = float('inf')
    config['general']['device'] = 'cpu'
    monkeypatch.setattr(arguments, 'Config', config)
    default = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield config
    torch.set_default_dtype(default)


def make_handler(A=A0, b=B0, lo=LO, hi=HI):
    from specifications import vnnlibHandler
    spec = [(np.array([[1.0, 0.0]]), np.array([0.0]))]
    return vnnlibHandler([([[float(l), float(h)] for l, h in zip(lo, hi)], spec)], [-1, len(lo)],
                         input_linear_constraints=(A, b))


def pooled_history(thresholds):
    """The SplitHistory of a solving-mode instance: it carries the RowPool every bound reads."""
    from history import SplitHistory
    history = SplitHistory()
    history.rowpool = RowPool(thresholds)
    return history


def box_above(s, width=1.0):
    """The box [s/2, 1]^2: its minimum of x0 + x1 is s."""
    return torch.tensor([[s / 2, s / 2]], dtype=F64), torch.full((1, 2), width, dtype=F64)


# --- the source ------------------------------------------------------------------------------------

def test_the_source_relaxes_the_standard_form_rows_and_keeps_the_raw_rows():
    handler = make_handler()
    A, c = handler.base_input_constraints
    assert torch.equal(A, A0) and torch.equal(c, -(B0 + TOL))
    assert torch.equal(handler.base_input_linear_constraints[0], A0)
    assert torch.equal(handler.base_input_linear_constraints[1], B0)       # the primal's rows


def test_the_source_is_a_no_op_outside_solving_mode(solving_config):
    solving_config['solving']['solving_mode'] = False
    assert torch.equal(make_handler().base_input_constraints[1], -B0)


def test_the_api_reference_carries_the_relaxed_rows_and_the_raw_rows():
    from api import ABCrownSolver
    reference = {}
    ABCrownSolver._attach_input_constraint_reference(SimpleNamespace(vnnlib_handler=make_handler()), reference)
    assert torch.equal(reference['base_input_constraints'][1], -(B0 + TOL))
    assert torch.equal(reference['base_input_linear_constraints'][1], B0)


# --- the root CROWN pass ---------------------------------------------------------------------------

def test_root_crown_pass_sees_b_plus_tol_once():
    """min -(x0 + x1) over the box and x0 + x1 <= 1 + tol is -(1 + tol): not -1 (raw), not -(1 + 2 tol)."""
    from auto_LiRPA import BoundedModule
    from incomplete_verifier_func import SpecHandler
    x = SpecHandler(make_handler()).x
    assert torch.equal(x.ptb.base_input_constraints[1], -(B0 + TOL))
    model = torch.nn.Linear(2, 1, bias=False).double()
    model.weight.data = torch.tensor([[-1.0, -1.0]], dtype=F64)
    lb, _ = BoundedModule(model, torch.zeros(1, 2, dtype=F64)).compute_bounds(x=(x,), method='CROWN')
    assert abs(float(lb) + 1.0 + TOL) < 1e-12


# --- the consumers read the rows as they are -------------------------------------------------------

def test_filter_discards_a_box_only_beyond_b_plus_tol():
    """The tolerance band: a box whose rows reach b + 0.5 tol is kept, b + 1.5 tol is discarded."""
    from input_split.branching_domains import UnsortedInputDomainList
    std = make_handler().base_input_constraints
    lo, hi = zip(*[box_above(1.0 + k * TOL) for k in (0.5, 1.5)])
    lo, hi = torch.cat(lo), torch.cat(hi)
    kept = UnsortedInputDomainList.get_remaining_index(
        2, torch.zeros(2, 2), torch.zeros(2, 2), lo, hi, check_dm_lbs=False, check_input_boxes=False,
        base_constraints=tuple(t.expand(2, *t.shape[1:]) for t in std))
    assert kept.tolist() == [0]
    raw = input_constraints_to_standard_form((A0, B0))                    # what the filter used to see
    lb = affine_constraints_domain_lower_bound(tuple(t.expand(2, *t.shape[1:]) for t in raw), lo, hi)
    assert (lb > 0).all()                                                 # both boxes were discarded


def test_clip_keeps_a_box_only_up_to_b_plus_tol():
    from input_split.clip import clip_domains
    std = make_handler().base_input_constraints
    lo, hi = zip(*[box_above(1.0 + k * TOL) for k in (0.5, 1.5)])
    lo, hi = torch.cat(lo), torch.cat(hi)
    lA, lbias = torch.zeros(2, 1, 2, dtype=F64), torch.full((2, 1), -1e9, dtype=F64)
    two = tuple(t.expand(2, *t.shape[1:]).contiguous() for t in std)
    x_L, x_U = clip_domains(lo.clone(), hi.clone(), torch.zeros(2, 1, dtype=F64), lA, lbias, num_iters=1,
                            base_input_linear_constraints=standard_form_to_input_constraints(two),
                            base_input_constraints=two, domain_constraints=None,
                            rowpool=RowPool(torch.zeros(1, 1, dtype=F64)))
    empty = (x_L > x_U).any(1)
    assert empty.tolist() == [False, True]


def test_constrained_concretization_in_bab_sees_b_plus_tol_once():
    """get_lower_bound_naive concretizes over the constraints it is given (the rows, merged with cuts)."""
    from auto_LiRPA import BoundedModule
    from input_split.bounding import get_lower_bound_naive
    from utils import Stats
    std = make_handler().base_input_constraints
    model = torch.nn.Linear(2, 1, bias=False).double()
    model.weight.data = torch.tensor([[-1.0, -1.0]], dtype=F64)
    bm = BoundedModule(model, torch.zeros(1, 2, dtype=F64))
    net = SimpleNamespace(net=bm, model_ori=model)
    config = arguments.Config
    config['bab']['clip_n_verify']['clip_input_domain']['enabled'] = True
    config['bab']['clip_n_verify']['clip_input_domain']['clip_type'] = 'complete'
    lb, _, _, _ = get_lower_bound_naive(
        net, dm_lb=None, dm_l=torch.zeros(1, 2, dtype=F64), dm_u=torch.ones(1, 2, dtype=F64),
        history=pooled_history(torch.zeros(1, 1, dtype=F64)), bounding_method='crown', C=torch.ones(1, 1, 1, dtype=F64),
        stop_criterion=lambda t: lambda x: x > t, thresholds=torch.zeros(1, 1, dtype=F64),
        constraints=std, stats=Stats())
    assert abs(float(lb) + 1.0 + TOL) < 1e-12


def _dual_ascent_inputs(rows, thresholds=None, history=None):
    """Run get_lower_bound_naive's dual-ascent block on a mocked net; what dual_ascent_tighten got."""
    import nlp_solver.dual_ascent as dual_ascent
    from input_split.bounding import get_lower_bound_naive
    from history import ensure_split_history
    from utils import Stats
    arguments.Config['solver']['nlp'].update({'dual_ascent_enabled': True, 'dual_lp_pdhg_iters': 10})
    net = MagicMock()
    net.net.output_name, net.net.input_name = ['output'], ['input']
    net.net.infeasible_bounds_constraints = None
    net.net.bound_opts = {}
    net.net.objective_alias_floor_value = None
    thresholds = torch.tensor([[10.0]], dtype=F64) if thresholds is None else thresholds
    m = thresholds.shape[1]
    net.net.compute_bounds.return_value = (
        torch.full((1, m), -100.0, dtype=F64),
        {'output': {'input': {'lA': torch.ones(1, m, 2, dtype=F64),
                              'lbias': torch.full((1, m), -100.0, dtype=F64)}}})
    seen = {}
    real = dual_ascent.dual_ascent_tighten

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    with patch.object(dual_ascent, 'dual_ascent_tighten', spy):
        get_lower_bound_naive(
            net, dm_lb=None, dm_l=torch.zeros(1, 2, dtype=F64), dm_u=torch.ones(1, 2, dtype=F64),
            history=ensure_split_history(history), bounding_method='alpha-crown',
            C=torch.ones(1, m, 1, dtype=F64), thresholds=thresholds,
            stop_criterion=lambda t: lambda x: x > t, return_A=True, return_b=True,
            base_input_linear_constraints=rows, stats=Stats())
    return seen


def test_dual_ascent_lp_uses_the_rows_it_is_given_as_they_are():
    rows = standard_form_to_input_constraints(make_handler().base_input_constraints)
    seen = _dual_ascent_inputs(rows, history=pooled_history(torch.tensor([[10.0]], dtype=F64)))
    assert torch.equal(seen['input_constr_A'], A0)
    assert torch.equal(seen['input_constr_b'], B0 + TOL)        # once: not b, not b + 2 tol


def test_dual_ascent_rhs_reads_the_pool_once():
    thresholds = torch.tensor([[10.0, 0.3]], dtype=F64)
    seen = _dual_ascent_inputs(None, thresholds, pooled_history(thresholds))
    assert torch.equal(seen['rhs'], torch.tensor([[10.0, 0.3 + TOL]], dtype=F64))      # the constraint row once, the cutoff as given
    with pytest.warns(UserWarning, match='RowPool'):         # no pool in solving mode: the optional bound is skipped, not run on the raw rhs
        assert _dual_ascent_inputs(None, thresholds) == {}


def test_a_bound_without_a_pool_in_solving_mode_raises_rather_than_reading_the_raw_rhs(solving_config):
    from input_split.utils import bound_thresholds
    thresholds = torch.tensor([[10.0, 0.3]], dtype=F64)
    with pytest.raises(ValueError, match='RowPool'):
        bound_thresholds(None, thresholds)
    solving_config['solving']['solving_mode'] = False
    assert bound_thresholds(None, thresholds) is thresholds                         # NN verification has no pool: as given


class _Stop(Exception):
    pass


def test_output_beta_reads_the_pool():
    from input_split.bounding import get_lower_bound_naive
    from utils import Stats
    thresholds = torch.tensor([[10.0, 0.3]], dtype=F64)
    history = pooled_history(thresholds)
    history.set_usage_flags(False, True)
    net = MagicMock()
    net.net.bound_opts = {}
    net.net.set_output_beta.side_effect = _Stop        # the bias is all this test reads
    with pytest.raises(_Stop):
        get_lower_bound_naive(
            net, dm_lb=None, dm_l=torch.zeros(1, 2, dtype=F64), dm_u=torch.ones(1, 2, dtype=F64), history=history,
            bounding_method='crown', C=torch.ones(1, 2, 1, dtype=F64), thresholds=thresholds,
            stop_criterion=lambda t: lambda x: x > t, stats=Stats())
    assert torch.equal(net.net.set_output_beta.call_args.kwargs['bias'],
                       torch.tensor([[10.0, 0.3 + TOL]], dtype=F64))                # the constraint row once, the cutoff as given


def test_lifted_obbt_uses_the_rows_it_is_given_as_they_are(monkeypatch):
    import nlp_solver.lifted_lp as module
    from input_split.lifted_bound import setup_lifted_lp
    lp = SimpleNamespace(atoms=None, rows=None, cleared_rhs=None, obj=None)
    monkeypatch.setattr(module, 'build_lifted_lp', lambda *a, **k: lp)
    rows = standard_form_to_input_constraints(make_handler().base_input_constraints)
    history = SimpleNamespace(qtb=None, rowpool=RowPool(torch.zeros(1, 2, dtype=F64)))
    setup_lifted_lp(SimpleNamespace(net=None, model_ori=None), history, torch.zeros(1, 2, dtype=F64),
                    torch.ones(1, 2, dtype=F64), torch.zeros(1, 2, 2, dtype=F64),
                    torch.zeros(1, 2, dtype=F64), rows,
                    {'lifted_lp_bound': True, 'lifted_lp_obbt': False, 'lifted_lp_rlt': False})
    _, eg, eh = history.lifted_obbt_rows
    assert torch.equal(eg, A0) and torch.equal(eh, B0 + TOL)


# --- the root verdict -----------------------------------------------------------------------------

@pytest.mark.parametrize('method', ['crown', 'alpha-crown'])
@pytest.mark.parametrize('gap, status', [(TOL / 2, 'unknown'), (2 * TOL, 'safe-incomplete')])
def test_the_root_verdict_calls_a_row_infeasible_only_beyond_rhs_plus_tol(solving_config, method, gap, status):
    """y1 = x1 + 1 >= 1 on [0, 1]^2 and the constraint y1 <= 1 - gap. The root lower bound of y1 is 1: for gap = tol / 2
    it lies in (rhs, rhs + tol], which the verify predicate accepts, so no infeasibility verdict; for 2 tol it is beyond."""
    from incomplete_verifier_func import incomplete_verifier
    from specifications import vnnlibHandler
    solving_config['solver']['bound_prop_method'] = method
    model = torch.nn.Linear(2, 2).double()
    model.weight.data = torch.eye(2, dtype=F64)
    model.bias.data = torch.tensor([0.0, 1.0], dtype=F64)
    handler = vnnlibHandler([([[0.0, 1.0], [0.0, 1.0]], [(np.array([[0.0, 1.0]]), np.array([1.0 - gap]))])], [-1, 2])
    verdict, _ = incomplete_verifier(SimpleNamespace(vnnlib_handler=handler), model)
    assert verdict == status


# --- the primal keeps the raw rows -----------------------------------------------------------------

def test_primal_rows_are_the_raw_rows_followed_by_the_appended_rows():
    from input_constraints import merge_standard_constraints
    from input_split.batch_branch_and_bound import _primal_input_rows
    handler = make_handler()
    cut = (torch.tensor([[[0.0, 1.0]]], dtype=F64), torch.tensor([[-0.9]], dtype=F64))
    std = merge_standard_constraints(handler.base_input_constraints, cut)
    A, b = _primal_input_rows(std, handler.base_input_linear_constraints)
    assert torch.equal(b[:, :1], B0)                          # raw, as the user gave it
    assert torch.equal(b[:, 1:], -cut[1]) and torch.equal(A[:, 1:], cut[0])
    # the dual's view of the same rows
    assert torch.equal(standard_form_to_input_constraints(std)[1][:, :1], B0 + TOL)
    assert _primal_input_rows(None, handler.base_input_linear_constraints) is None
    assert torch.equal(_primal_input_rows(std, None)[1][:, :1], B0 + TOL)      # no record: the rows as given


# --- BaB: root assembly, all three paths -----------------------------------------------------------

def _bab_config():
    config = arguments.Config
    input_split = config['bab']['branching']['input_split']
    input_split.update({'split_hint': None, 'update_rhs_with_attack': False, 'compare_with_old_bounds': False,
                        'skip_getting_worst_domain': False, 'show_progress': False, 'reorder_bab': True,
                        'adv_check': -1, 'split_partitions': 2, 'presplit_domains': None,
                        'sort_index': None, 'sort_descending': True, 'bf_iters': 0, 'bf_batch_size': 1})
    clip = config['bab']['clip_n_verify']['clip_input_domain']
    clip.update({'enabled': True, 'clip_iterations': 1, 'clip_calculate_volume_metrics': False,
                 'clip_type': 'complete'})
    config['bab'].update({'timeout': 100, 'max_iterations': 10, 'sort_domain_interval': 0})
    config['bab']['branching']['method'] = 'sb'
    config['solver'].update({'batch_size': 64, 'bound_prop_method': 'crown', 'init_bound_prop_method': 'same',
                             'min_batch_size_ratio': 0.1, 'auto_enlarge_batch_size': False})
    config['attack']['pgd_order'] = 'skip'
    config['attack']['input_split_check_adv']['enabled'] = 'false'
    config['model']['with_jacobian'] = False
    config['solver']['nlp'].update({'quadratic_tangent_bound': False, 'lifted_lp_bound': True,
                                    'lifted_lp_bab_time': 0.0})


def _run_root(path, cuts=None):
    """input_bab_parallel up to the first split, with the rows reaching it through `path`
    ('reference', 'build' or 'ptb'); (RootLifted.build args, root get_bounds args, make_verifier args, the history
    the bounds read, which carries the instance's RowPool)."""
    from input_split.batch_branch_and_bound import input_bab_parallel
    _bab_config()
    handler = make_handler(lo=[0.0, 0.0, 0.0], hi=[1.0, 1.0, 1.0],
                           A=torch.tensor([[[1.0, 0.0, 0.0]]], dtype=F64), b=torch.tensor([[0.5]], dtype=F64))
    std, lin = handler.base_input_constraints, handler.base_input_linear_constraints
    x_L, x_U = torch.zeros(1, 3, dtype=F64), torch.ones(1, 3, dtype=F64)
    ptb = SimpleNamespace(x_L=x_L, x_U=x_U)
    ret = {'global_lb': torch.tensor([[10.0, -1.0]], dtype=F64),
           'lA': {'input': torch.ones(1, 2, 3, dtype=F64)},
           'pre_incomplete_primal': {'verified': True, 'nlp_found': False, 'objective': 100.0,
                                     'x': torch.full((3,), 0.1), 'peak_mem_bytes': None,
                                     'total_mem_bytes': None}}
    reference = built = None
    if path == 'reference':             # reference_dict from incomplete verification
        ret.update(base_input_linear_constraints=lin, base_input_constraints=std)
        reference = ret
    elif path == 'build':               # net.build fallback
        built = dict(ret, base_input_linear_constraints=lin, base_input_constraints=std)
    else:                               # x.ptb
        ptb.base_input_linear_constraints, ptb.base_input_constraints = lin, std
        built = ret
    x = SimpleNamespace(ptb=ptb, shape=x_L.shape)
    net = SimpleNamespace(net=MagicMock(), model_ori=MagicMock(), final_name='final')
    net.net.input_name = ['input']
    net.net.get_splittable_activations.return_value = []
    net.net.bound_opts = {}
    net.build = MagicMock(return_value=(ret['global_lb'], built))
    qtb = SimpleNamespace(x_hat=torch.full((3,), 0.1), lam_lb=0.0, f=100.0,
                          fixed_mask=x_L[0] == x_U[0], fixed_values=x_L[0])
    manager = None if cuts is None else SimpleNamespace(rounds=1, offer=MagicMock(return_value=cuts))
    arguments.Config['solver']['nlp'].update(
        {'quadratic_tangent_bound': True, 'dual_lp_pdhg_iters': 500, 'lifted_lp_bab_time': 20.0,
         'lifted_lp_bab_handoff': True, 'convex_row_cuts': cuts is not None})

    def lifted_bab(history, *args, **kwargs):
        history.lifted_bab_leaves = (torch.zeros(3, 3), torch.full((3, 3), 0.5), torch.tensor([5.0, 20.0, 35.0]))
        return None

    with patch('input_split.batch_branch_and_bound.run_root_diagnostics_and_budget'), \
         patch('input_split.batch_branch_and_bound.UnsortedInputDomainList',
               side_effect=lambda *args, **kwargs: MagicMock(history=kwargs['history'])), \
         patch('nlp_solver.quadratic_tangent.build_quadratic_tangent', return_value=qtb), \
         patch('nlp_solver.convex_row_cuts.build_convex_row_cuts', return_value=manager), \
         patch('input_split.nlp_in_bab.compute_qtb_tangent_point', return_value=(None, None)), \
         patch('input_split.lifted_bound.RootLifted.build', return_value=None) as lifted, \
         patch('input_split.lifted_bound.make_verifier', return_value=None) as verifier, \
         patch('input_split.lifted_bound.run_lifted_bab', side_effect=lifted_bab), \
         patch('input_split.batch_branch_and_bound.get_bounds',
               return_value=(torch.tensor([[25.0, 99.0]], dtype=F64), None, None, None)) as bounds, \
         patch('input_split.batch_branch_and_bound.input_split_branching', side_effect=StopIteration):
        with pytest.raises(StopIteration):
            input_bab_parallel(net, x, torch.eye(2, dtype=F64).unsqueeze(0), torch.zeros(1, 2, dtype=F64),
                               torch.tensor([2]), reference_dict=reference)
    return lifted.call_args.args[6], bounds.call_args.args[8], verifier.call_args.args[6], lifted.call_args.args[1]


@pytest.mark.parametrize('path', ['reference', 'build', 'ptb'])
def test_bab_feeds_the_bound_consumers_b_plus_tol_and_the_primal_the_raw_rhs(path):
    lifted, bounds, primal, _ = _run_root(path)
    for dual in (lifted, bounds):
        assert torch.equal(dual[1], torch.tensor([[0.5 + TOL]], dtype=F64))
    assert torch.equal(primal[1], torch.tensor([[0.5]], dtype=F64))


def test_bab_builds_one_pool_over_the_raw_rows():
    """The pool the bounds read holds the problem's raw rows with the tolerance once, and the tolerance of the spec rows."""
    from fractions import Fraction
    *_, history = _run_root('reference')
    pool = history.rowpool
    assert pool.m == 2 and pool.tolerance == TOL
    base = pool.rows[pool.row_of[2]]                                  # ids: objective 0, spec row 1, then the base rows
    assert base.source == 'base' and base.ex_hi == Fraction(0.5) + Fraction(TOL)
    assert pool.audit({1: TOL, 2: TOL}) == []


def test_the_pool_records_the_tolerance_the_source_applied():
    """The spec handler relaxes the standard-form base rows before any model exists; the pool built later from the raw rows
    must hold the same bound (to the one rounding the pool makes outward)."""
    from nlp_solver.structure import RowPool
    handler = make_handler()
    _, c = handler.base_input_constraints
    A_raw, b_raw = handler.base_input_linear_constraints
    pool = RowPool(torch.zeros(1, 1, dtype=F64), (A_raw[0], b_raw[0]))
    row = pool.rows[pool.row_of[1]]
    assert row.hi >= float(-c[0, 0]) and row.hi <= float(np.nextafter(-c[0, 0].numpy(), np.inf))


def test_bab_cuts_are_shared_by_both_views_and_not_relaxed_again():
    cut_A, cut_b = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=F64), torch.tensor([[-0.9]], dtype=F64)
    lifted, bounds, primal, _ = _run_root('reference', cuts=(cut_A, cut_b))
    for dual in (lifted, bounds):
        assert torch.equal(dual[1], torch.tensor([[0.5 + TOL, 0.9]], dtype=F64))
    assert torch.equal(primal[1], torch.tensor([[0.5, 0.9]], dtype=F64))


def test_bab_loop_gives_the_alm_the_raw_rows_and_the_bounds_the_relaxed_rows():
    from input_split.batch_branch_and_bound import batch_verification_input_split
    from utils import Stats
    _bab_config()
    handler = make_handler()
    std, lin = handler.base_input_constraints, handler.base_input_linear_constraints
    lb, thr = torch.tensor([[1.0]], dtype=F64), torch.zeros(1, 1, dtype=F64)
    x_L, x_U = torch.zeros(1, 2, dtype=F64), torch.ones(1, 2, dtype=F64)
    cs, sizes, split_idx = torch.ones(1, 1, 1, dtype=F64), torch.ones(1, dtype=torch.int32), torch.zeros(1, 10, dtype=torch.long)
    domains = MagicMock()
    domains.history = SimpleNamespace(input_rows_raw=lin)
    domains.pick_out_batch.return_value = (lb, x_L, x_U, cs, thr, None, std, sizes, split_idx, None, None)
    domains.__len__.return_value = 0
    net = MagicMock()
    net.device = 'cpu'
    stats = Stats()
    stats.storage_depth = 10
    stop = StopIteration
    with patch('input_split.batch_branch_and_bound.run_bab_alm_step',
               return_value=(torch.tensor([0.0], dtype=F64), True)) as alm, \
         patch('input_split.batch_branch_and_bound.get_bounds', side_effect=stop) as bounds:
        with pytest.raises(stop):
            batch_verification_input_split(
                domains, net, batch=1, num_iter=1, decision_thresh=thr, shape=(1, 2), use_reordered_bab=True,
                stats=stats, primal_solution=torch.tensor([1.0], dtype=F64), alm_state=SimpleNamespace())
    picked = alm.call_args.args[2]
    assert torch.equal(picked.base_input_linear_constraints[1], B0)                  # raw
    assert torch.equal(bounds.call_args.args[8][1], B0 + TOL)                        # relaxed
    assert torch.equal(picked.base_input_constraints[1], -(B0 + TOL))


# --- the reproducer --------------------------------------------------------------------------------

def test_box_around_a_point_on_the_band_edge_is_kept():
    """A point with A x - b = +tol (the verifier accepts it): the filter used to discard a box around it."""
    x = torch.tensor([[0.5 + TOL / 2, 0.5 + TOL / 2]], dtype=F64)           # x0 + x1 = 1 + tol
    lo, hi = x - 1e-5, x + 1e-5
    raw = input_constraints_to_standard_form((A0, B0))
    lb_raw, _ = affine_constraints_domain_lower_bound(raw, lo, hi, return_infeasible=True)
    assert float(lb_raw) > 0                                                   # discarded: the bug
    lb, bad = affine_constraints_domain_lower_bound(make_handler().base_input_constraints, lo, hi,
                                                    return_infeasible=True)
    assert float(lb) <= 0 and not bool(bad.any())


@pytest.mark.skipif(not os.path.isdir(BENCHMARKS), reason=f'benchmark directory {BENCHMARKS} is absent')
def test_ex7_2_1_box_around_the_tau_optimum_is_kept():
    """tests/repro_input_rows.py of the soundness alarm: x* is the optimum with every row <= 1e-4
    (objective 1222.50869); its second input row sits at A2 x* - b2 = +1e-4."""
    state = torch.load(f'{BENCHMARKS}/py_minlplib_linear/ex7_2_1_model_state.pt', map_location='cpu',
                       weights_only=False)['linear_constr']
    A, b = state['A'].double().reshape(1, -1, 7), state['b'].double().reshape(1, -1)
    text = open(f'{BENCHMARKS}/vnnlib_minlplib_linear/ex7_2_1.vnnlib').read()
    bounds = {}
    for op, i, v in re.findall(r'\(assert \((>=|<=) X_(\d+) ([-\d.eE+]+)\)\)', text):
        bounds[(op, int(i))] = float(v)
    lo = torch.tensor([bounds[('>=', i)] for i in range(7)], dtype=F64)
    hi = torch.tensor([bounds[('<=', i)] for i in range(7)], dtype=F64)
    x = torch.tensor([1697.9516704005152, 53.3191789200095, 3031.271861011236, 90.10493784730929, 95.0,
                      10.485534436176254, 153.5641424243337], dtype=F64)
    viol = A[0] @ x - b[0]
    assert float(viol.max()) <= TOL * (1 + 1e-6) and abs(float(viol[1]) - TOL) < 2e-6
    delta = 2e-5 / float(A[0, 1].abs().sum())
    l, u = torch.maximum(x - delta, lo)[None], torch.minimum(x + delta, hi)[None]
    raw = input_constraints_to_standard_form((A, b))
    assert float(affine_constraints_domain_lower_bound(raw, l, u)[0, 1]) > 0     # the raw filter discards
    handler = make_handler(A=A, b=b, lo=lo.tolist(), hi=hi.tolist())
    lb, bad = affine_constraints_domain_lower_bound(handler.base_input_constraints, l, u, return_infeasible=True)
    assert float(lb.max()) <= 0 and not bool(bad.any())                          # kept with b + tol
