"""The unified root (solver.nlp.unified_root, arch2 S1): one ordered list of root operations under the run's deadline,
the gap rule after each, the lifted root bound before OBBT, and the root ALM's progress against verified IPOPT."""
import math
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import arguments
from input_split import batch_branch_and_bound as bbb
from input_split.batch_branch_and_bound import RootOp, RootState, _run_root

DT = torch.float64


# ----------------------------------------------------------------------------- the operation list

def test_operation_list_runs_the_lifted_root_bound_before_obbt_and_the_bab():
    names = [op.name for op in bbb._ROOT_OPERATIONS]
    assert names[0] == "reference_bounds" and names[-1] == "initial_domains"
    lifted = names[names.index("lifted_lp"):names.index("initial_domains")]
    assert lifted == ["lifted_lp", "lifted_bound", "lifted_warm_up", "lifted_rlt", "lifted_bound", "lifted_obbt",
                      "lifted_bab"]
    assert names.index("nlp") < names.index("incumbent") < names.index("tangent") < names.index("lifted_lp")
    timed = {op.name for op in bbb._ROOT_OPERATIONS if op.timed}
    assert timed == {"lp_alternation", "tangent", "gp_certificate", "exp_certificate", "lifted_bound", "lifted_warm_up",
                     "lifted_obbt", "lifted_bab"}


def test_lifted_bound_is_due_only_in_the_unified_root_with_an_incumbent_and_a_stronger_relaxation():
    setup = SimpleNamespace(version=2)
    root = SimpleNamespace(unified=True, primal_solution_found=True, lifted_setup=setup, lifted_bound_version=1)
    assert bbb._lifted_bound_due(root)
    for change in (dict(unified=False), dict(primal_solution_found=False), dict(lifted_setup=None),
                   dict(lifted_bound_version=2)):
        assert not bbb._lifted_bound_due(SimpleNamespace(**{**vars(root), **change}))


# ----------------------------------------------------------------------------- the driver

def _root(unified, deadline=math.inf, P=None, D=-10.0):
    """A RootState with only what _run_root and the gap rule read."""
    root = object.__new__(RootState)
    root.unified, root.deadline, root.start = unified, deadline, time.time()
    root.nlp_config = {"bab_primal_dual_abs_tol": 1e-5, "bab_primal_dual_rel_tol": 1e-4}
    root.primal_solution = None if P is None else torch.tensor([P], dtype=DT)
    root.primal_solution_found = P is not None
    root.report_only_primal = None
    root.global_lb = torch.tensor([[D, -1.0]], dtype=DT)
    root.best_primal_x, root.net, root.stats = torch.zeros(2), None, SimpleNamespace(visited=0)
    root.ledger = bbb.OpLedger()
    return root


def _ops(ran):
    def incumbent(root, deadline):
        ran.append("incumbent")
        root.primal_solution, root.primal_solution_found = torch.tensor([5.0], dtype=DT), True

    def bound(root, deadline):
        ran.append("bound")
        root.global_lb = torch.tensor([[5.0 - 1e-7, -1.0]], dtype=DT)

    def later(root, deadline):
        ran.append("later")

    return (RootOp("incumbent", incumbent), RootOp("bound", bound), RootOp("later", later))


@patch.object(bbb, "_finish_with_lifted_dual", side_effect=lambda net, dom, x, P, found, dual, stats: ("closed", dual))
def test_unified_root_ends_after_the_operation_that_closes_the_gap(finish, capsys):
    ran = []
    with patch.object(bbb, "_ROOT_OPERATIONS", _ops(ran)):
        outcome = _run_root(_root(unified=True))
    assert ran == ["incumbent", "bound"] and outcome == ("closed", pytest.approx(5.0 - 1e-7))
    assert "[UnifiedRoot] closed after bound" in capsys.readouterr().out


@patch.object(bbb, "_finish_with_lifted_dual")
def test_without_the_unified_root_every_operation_runs(finish):
    ran = []
    with patch.object(bbb, "_ROOT_OPERATIONS", _ops(ran)):
        assert _run_root(_root(unified=False)) is None
    assert ran == ["incumbent", "bound", "later"] and not finish.called


def test_gap_rule_closes_only_a_consistent_gap_within_tolerance():
    # P < D - tol is the optimality guard's case: the gap rule refuses it and the root goes on.
    root = _root(unified=True, P=4.0)
    assert not root.closes(5.0) and root.closes(4.0 - 1e-6) and not root.closes(3.0)
    assert not _root(unified=True).closes(-10.0)            # no incumbent yet


def test_unified_root_admits_timed_operations_only_before_the_deadline():
    for unified, expect in ((True, ["untimed"]), (False, ["timed", "untimed"])):
        ran = []
        ops = (RootOp("timed", lambda root, deadline: ran.append("timed"), timed=True),
               RootOp("untimed", lambda root, deadline: ran.append("untimed")))
        with patch.object(bbb, "_ROOT_OPERATIONS", ops):
            assert _run_root(_root(unified, deadline=time.time() - 1.0)) is None
        assert ran == expect


def test_time_left_is_the_deadline_only_in_the_unified_root():
    assert _root(unified=False, deadline=time.time() + 5).time_left() == math.inf
    assert 4.0 < _root(unified=True, deadline=time.time() + 5).time_left() <= 5.0


@patch.object(bbb, "_finish_with_lifted_dual", side_effect=lambda net, dom, x, P, found, dual, stats: ("closed", dual))
def test_lifted_root_bound_closes_or_is_left_to_the_bab(finish):
    root = _root(unified=True, P=5.0)
    root.lifted_bound_version = 0
    root.lifted_setup = SimpleNamespace(version=1, bound=lambda P, t: 5.0 - 1e-6)
    assert bbb._root_lifted_bound(root, time.time() + 10) == ("closed", 5.0 - 1e-6)
    root.lifted_setup = SimpleNamespace(version=2, bound=lambda P, t: 4.0)
    assert bbb._root_lifted_bound(root, time.time() + 10) is None
    assert root.lifted_bound_version == 2
    assert float(root.global_lb[0, 0]) == -10.0     # a bound that does not close commits nothing


def test_root_lifted_bound_is_the_lifted_bab_root_bound():
    from tests.test_lifted_lp import _setup, TOL
    from nlp_solver.lifted_lp import build_lifted_lp
    from input_split.lifted_bound import RootLifted
    m, bm, x, C, rhs, l, u = _setup()
    lp = build_lifted_lp(bm, m, x, C, l, u)
    rows = (rhs[:, 1:] + TOL).clone()
    lifted = object.__new__(RootLifted)
    lifted.history, lifted.lp, lifted.lo, lifted.hi = SimpleNamespace(qtb=None), lp, l, u
    lifted.rhs_rows, lifted.eg, lifted.eh = rows, None, None
    lifted.nlp_config = {}
    assert lifted.bound(torch.tensor([0.0]), 10.0) is None      # a finite row (tanh) is not lifted
    rows[0, 3] = float("inf")
    for P in (5.0, -0.5):
        D, nodes, closed = lp.branch_and_bound(l, u, rows, cutoff=P, time_budget=10.0, max_nodes=0)
        assert lifted.bound(torch.tensor([P]), 10.0) == D <= P


# ----------------------------------------------------------------------------- the root ALM's stop rule

class _Identity(torch.nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


def _alm(rival, outer=6, seed=0, **cfg_overrides):
    """min x0 over [0, 1]^2 (the optimum 0), no rows; patience rules off so only the rival can stop it."""
    from nlp_solver import ALMConfig, ALMPGDSolver, NNProblemSingleDomain
    problem = NNProblemSingleDomain(
        _Identity(), torch.zeros(2, dtype=DT), torch.ones(2, dtype=DT),
        torch.tensor([[1.0, 0.0]], dtype=DT), torch.tensor([0.0], dtype=DT),
        torch.zeros(0, 2, dtype=DT), torch.zeros(0, dtype=DT), device=torch.device("cpu"), dtype=DT)
    cfg = ALMConfig(outer_max_iter=outer, inner_steps=5, num_restarts=4, timeout=60.0, feas_tol=1e-6,
                    obj_patience=-1, first_feas_patience=-1, **cfg_overrides)
    torch.manual_seed(seed)
    return ALMPGDSolver(cfg).solve_with_restarts(problem, rival=rival)


def test_alm_measures_progress_before_stopping_and_preserves_the_endgame():
    from nlp_solver import ALMPGDSolver
    with patch.object(ALMPGDSolver, "_batched_gn_restore", autospec=True,
                      side_effect=lambda self, p, x, obj, viol, feas, *a, **kw:
                      (x, obj, viol, feas, 0)) as restore:
        result = _alm(rival=lambda: -1.0)       # below the optimum
    assert result.outer_iterations >= 1
    assert restore.called


def test_rival_progress_credits_infeasible_progress_and_is_invariant_to_service_units():
    from nlp_solver.alm import _RivalProgress
    for scale in (0.1, 1., 100.):
        progress = _RivalProgress()
        # No feasible points in any of these observations. A reduction in
        # target residual buys work, including a temporary regression.
        assert not progress.stop(7., 1e5, 1*scale, 100*scale)
        assert not progress.stop(7., 1e3, 2*scale, 99*scale)
        assert not progress.stop(7., 1e2, 4*scale, 97*scale)
        assert not progress.stop(7., 2e2, 5*scale, 96*scale)
        assert progress.stop(7., 2e2, 8*scale, 93*scale)
        # A new rival changes the target; stale observations cannot stop it.
        assert not progress.stop(6., 2e2, 9*scale, 92*scale)


def test_rival_progress_keeps_an_improving_basin():
    from nlp_solver.alm import _RivalProgress
    progress = _RivalProgress()
    assert not progress.stop(7., 100., 1., 20.)
    assert not progress.stop(7., 1., 2., 19.)
    assert not progress.stop(7., 2., 19., 2.)  # a retained improving point still counts


def test_rival_can_use_measured_initial_candidates_without_a_fixed_iteration_delay():
    from nlp_solver.alm import _RivalProgress
    flat = _RivalProgress()
    assert flat.stop(7., 100., 1., 20., initial_residual=100.)
    moving = _RivalProgress()
    assert not moving.stop(7., 10., 1., 20., initial_residual=100.)


def test_nonfinite_residuals_cannot_establish_a_rival_progress_rate():
    from nlp_solver.alm import _RivalProgress
    progress = _RivalProgress()
    assert not progress.stop(7., float('nan'), 1., 20.)
    assert not progress.stop(7., 10., 2., 19., initial_residual=float('inf'))
    assert not progress.stop(7., 1., 3., 18.)


def test_restoration_trial_keeps_its_reserve_without_a_second_endgame():
    from nlp_solver import ALMPGDSolver
    seen = []

    def restore(self, p, x, obj, viol, feas, *args, deadline=None, **kwargs):
        seen.append((deadline, time.time()))
        return x, obj, viol, feas, 0

    with patch('nlp_solver.alm._RivalProgress.stop', return_value=True), \
            patch.object(ALMPGDSolver, '_batched_gn_restore', autospec=True, side_effect=restore):
        _alm(rival=lambda: -1., endgame_time_frac=.25)
    assert len(seen) == 1
    deadline, entered = seen[0]
    assert entered < deadline <= entered + .25 * 60.


def test_rival_progress_stops_when_measured_contraction_cannot_pay_for_feasibility():
    from nlp_solver.alm import _RivalProgress
    progress = _RivalProgress()
    assert not progress.stop(7., 1e6, 1., 20.)
    assert progress.stop(7., 9e5, 2., 19.)


def test_infeasible_outer_zero_retains_a_better_basin_for_restoration():
    from nlp_solver import ALMConfig, ALMPGDSolver
    from tests.test_sparse_derivatives import Bilinear, problem_of
    # An asymmetric box makes the midpoint infeasible as well as the seeded
    # random restarts. There is no archived feasible point to mask a lost basin.
    problem = problem_of(Bilinear(), [0., 0.], [1., 2.])
    cfg = ALMConfig(outer_max_iter=6, inner_steps=5, inner_lr=.01,
                    num_restarts=4, timeout=5., feas_tol=1e-6,
                    obj_patience=-1, first_feas_patience=-1,
                    batched_gn_steps=10, batched_gn_seeds=4)
    torch.manual_seed(0)
    result = ALMPGDSolver(cfg).solve_with_restarts(problem, rival=lambda: -.1)
    assert result.x_solution is not None and result.constraint_violation <= 1e-6
    assert result.objective_value < -.1
    out = problem.model(result.x_solution.reshape(1, -1))[0]
    assert float(out[1:].max()) <= cfg.feas_tol
    assert float(out[0]) < -.1


def test_alm_goes_on_while_it_beats_the_rival_and_runs_its_budget_without_one():
    beaten = _alm(rival=lambda: 10.0)
    assert beaten.success and beaten.objective_value < 10.0
    none_yet = _alm(rival=lambda: None)                      # IPOPT still running, failed, or unverified
    plain = _alm(rival=None)
    assert none_yet.outer_iterations == plain.outer_iterations == 6
    assert torch.equal(none_yet.x_solution, plain.x_solution)
    assert none_yet.objective_value == plain.objective_value


def _compare(config, local):
    """compare_solvers with a stubbed local solver result and a recording ALM; the rival the ALM got."""
    from nlp_solver.falsification import compare_solvers, LocalSolverRunResult
    got = {}

    def solve(problem, num_restarts=None, timeout=None, x_init=None, mu_init=None, rival=None):
        got["rival"] = rival
        return SimpleNamespace(success=True, x_solution=torch.full((1,), 0.5), objective_value=0.5,
                               constraint_violation=0.0, outer_iterations=1, final_rho=1.0)

    result = LocalSolverRunResult(success=local is not None, objective=local, violation=0.0, time_sec=0.0,
                                  verified=local is not None, status="stub",
                                  x=None if local is None else torch.zeros(1))
    with patch("nlp_solver.falsification.IPOPT_AVAILABLE", True), \
            patch("nlp_solver.falsification._run_local_solver", return_value=result), \
            patch("nlp_solver.falsification.verify_counterexample", return_value=(True, 0.5, 0.0)), \
            patch("nlp_solver.falsification.ALMPGDSolver") as cls:
        cls.return_value.solve_with_restarts.side_effect = solve
        compare_solvers(None, _Identity(), torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([[[1.0]]]),
                        torch.tensor([[0.0]]), torch.tensor([1], dtype=torch.int32), config=config)
    return got["rival"]


@pytest.mark.parametrize("parallel", [True, False])
def test_compare_solvers_hands_alm_only_a_verified_ipopt_point(parallel):
    config = {"enable_local_solver": True, "root_parallel": parallel, "verbose": False}
    assert _compare(dict(config), local=-2.0) is None                      # flag off: no rival
    rival = _compare(dict(config, unified_root=True), local=-2.0)
    assert rival is not None and rival() == -2.0
    failed = _compare(dict(config, unified_root=True), local=None)        # IPOPT found no verified point
    assert failed is None or failed() is None


# ----------------------------------------------------------------------------- flag-off parity, end to end

def _input_bab(unified):
    """The root pipeline of test_gp_certificate (pre-incomplete incumbent 100, root bound 99.9995) up to the lifted
    LP, which stops the run when reached: (the outcome or 'reached lifted LP', the finish mock)."""
    from tests.test_batch_branch_and_bound import _setup_mock_config
    config = _setup_mock_config()
    config['solving']['solving_mode'] = True
    config['solver']['nlp']['unified_root'] = unified
    lo, hi = torch.full((1, 2), .1), torch.ones(1, 2)
    x = SimpleNamespace(ptb=SimpleNamespace(x_L=lo, x_U=hi), shape=lo.shape)
    ref = {'global_lb': torch.tensor([[99.9995, -1.]]), 'lA': {'input': torch.ones(1, 2, 2)},
           'pre_incomplete_primal': {'verified': True, 'nlp_found': False, 'objective': 100.,
                                     'x': torch.full((2,), .5), 'peak_mem_bytes': None, 'total_mem_bytes': None}}
    net = SimpleNamespace(net=MagicMock(), model_ori=MagicMock(), final_name='final')
    net.net.input_name = ['input']
    net.net.get_splittable_activations.return_value = []
    net.net.bound_opts = {}
    with patch.object(arguments, 'Config', config), \
            patch('input_split.batch_branch_and_bound.run_root_diagnostics_and_budget'), \
            patch('input_split.batch_branch_and_bound.UnsortedInputDomainList',
                  side_effect=lambda *args, **kw: MagicMock(history=kw['history'])), \
            patch('input_split.lifted_bound.RootLifted.build', side_effect=StopIteration), \
            patch('input_split.batch_branch_and_bound._finish_with_lifted_dual', return_value='certified') as finish:
        try:
            outcome = bbb.input_bab_parallel(net, x, torch.eye(2).unsqueeze(0), torch.zeros(1, 2),
                                             torch.tensor([2]), reference_dict=ref)
        except StopIteration:
            outcome = 'reached lifted LP'
    return outcome, finish


def test_flag_off_runs_the_v406_root_and_flag_on_closes_at_the_incumbent():
    outcome, finish = _input_bab(unified=False)
    assert outcome == 'reached lifted LP' and not finish.called
    outcome, finish = _input_bab(unified=True)
    assert outcome == 'certified'
    assert finish.call_args.args[-2] == pytest.approx(99.9995)


def test_certificate_residual_discards_invalid_restarts():
    from nlp_solver.alm import _certificate_residual
    phi = torch.tensor([float('nan'), 5., -10.], dtype=DT)
    viol = torch.tensor([0., 0., float('nan')], dtype=DT)
    r = _certificate_residual(phi, viol, 5., 1e-5, 1e-4, 1e-6)
    assert torch.isinf(r[[0, 2]]).all()
    assert float(r.min()) == pytest.approx(2.)


def test_early_restoration_stops_a_certificate_equivalent_basin_once():
    from nlp_solver import ALMPGDSolver
    calls = []
    def restore(self, p, x, obj, viol, feas, *args, **kwargs):
        calls.append(1)
        # Keep a real feasible improvement below the certificate resolution.
        x[:] = 0.
        obj[:] = 0.
        viol[:] = 0.
        feas[:] = True
        return x, obj, viol, feas, 1
    with patch.object(ALMPGDSolver, '_batched_gn_restore', autospec=True, side_effect=restore):
        result = _alm(rival=lambda: 1e-8)
    assert result.success and result.objective_value == 0.
    assert result.outer_iterations == 1 and calls == [1]


def test_root_progress_quotes_inner_work_and_not_elapsed_wall_seconds():
    from nlp_solver import ALMPGDSolver
    quotes = []
    def stop(self, rival, residual, work, remaining, **kwargs):
        quotes.append((work, remaining))
        return True
    with patch('nlp_solver.alm._RivalProgress.stop', stop):
        result = _alm(rival=lambda: -1.)
    assert quotes and quotes[0][0] == result.total_inner_steps
    assert quotes[0][1] == 5 * 5  # configured remaining steps


def test_material_restoration_gain_buys_a_fresh_global_measurement():
    from nlp_solver import ALMPGDSolver
    calls = []
    def restore(self, p, x, obj, viol, feas, *args, **kwargs):
        calls.append(1)
        x[:] = .1
        obj[:] = .1
        viol[:] = 0.
        feas[:] = True
        return x, obj, viol, feas, 1
    with patch.object(ALMPGDSolver, '_batched_gn_restore', autospec=True, side_effect=restore):
        result = _alm(rival=lambda: 1.)
    assert result.success and result.objective_value == .1
    assert result.outer_iterations == 2  # productive trial, then a measured stall


def test_trial_does_not_shadow_restoration_of_the_later_global_population():
    from nlp_solver import ALMPGDSolver
    seen = []
    def restore(self, p, x, obj, viol, feas, *args, **kwargs):
        seen.append(float(obj.min()))
        value = .1 if len(seen) == 1 else .05
        x[:] = value
        obj[:] = value
        viol[:] = 0.
        feas[:] = True
        return x, obj, viol, feas, 1
    with patch.object(ALMPGDSolver, '_batched_gn_restore', autospec=True, side_effect=restore):
        result = _alm(rival=lambda: 1.)
    assert len(seen) == 2 and seen[1] > .1  # PGD's own retained population
    assert result.outer_iterations == 2 and result.objective_value == .05
