"""Cascade transitions, certificate scope and independent validation; CPU only."""
from dataclasses import replace
import importlib
import json
import math
from types import SimpleNamespace

import pytest
import torch

import gear as gr
from gear.dynamics.problem import reference_samples, sample_box
from gear.dynamics.training import Normalization

flow = importlib.import_module('gear.dynamics.solve')


class Initial(torch.nn.Module):
    def forward(self, z):
        return z[:, :1] * 0


def simulate(z):
    return torch.stack((0 * z, z, 2 * z), 1)


def outputs(t, z):
    return torch.cat((-t[:, -1], t[:, -1] - .8), 1)


def residual(t, z):
    return t[:, 1:] - t[:, :-1] - z[:, None]


def problem(**kw):
    return gr.DynamicsProblem(**dict(original='original', box=([0.0], [1.0]), simulate=simulate,
                                    initial_state=Initial(), outputs=outputs, residual=residual, **kw))


def options(**kw):
    return gr.DynamicsOptions(original=gr.Options(device='cpu', time_limit=0),
        surrogate=gr.Options(device='cpu', time_limit=1),
        training=gr.VAPINNOptions(train_size=32, validation_size=16, test_size=32,
            steps=2, refinement_steps=2, width=4, depth=2, ridge_every=1, evaluate_every=1),
        **{'max_refinements': 0, **kw})


def result(status=gr.Status.NO_SOLUTION, p=math.inf, d=-math.inf, z=None, sense=gr.MIN,
           termination=gr.Termination.TIME_LIMIT):
    replay = None if z is None else gr.Replay(True, 1e-6, 0, None, 0, 0, p)
    stats = gr.Stats({}, 0, {}, 0, 0, 0)
    provenance = gr.Provenance('test', {}, '', '', (), 'test', '', False, '', 'solve', {}, 'partial',
                              'schema_default', None, False, {}, 'test', 'cpu')
    return gr.Result(status, termination, sense, p, d, abs(p - d), abs(p - d),
                     None if z is None else {'X_0': z}, replay, None, stats, provenance, ())


class Perfect(torch.nn.Module):
    def forward(self, z):
        return outputs(simulate(z), z)

    def trajectory(self, z):
        return simulate(z)


class Trainer:
    def __init__(self, models=None):
        self.calls = []
        self.models = models or [Perfect()]

    def fit(self, p, data, validation, norm, initial, k, tau, device):
        self.calls.append((data, validation, initial))
        return self.models[min(k, len(self.models) - 1)]


def harness(monkeypatch, l1, l2, *, domains=False):
    state = SimpleNamespace(solves=[], compiled=[])
    def compile_(module, **kw):
        state.compiled.append(module)
        return SimpleNamespace(hash=f'model-{len(state.compiled)}', module=module)
    class Solver:
        def __init__(self, options):
            self.incumbent = self.domains = None
        def on_incumbent(self, fn):
            self.incumbent = fn
        def on_domains(self, fn, limit=32):
            self.domains = fn
        def solve(self, source):
            state.solves.append(source)
            if source == 'original':
                if isinstance(l1, Exception):
                    raise l1
                return l1
            r = l2[len(state.solves) - 2] if isinstance(l2, list) else l2
            if r.x is not None:
                self.incumbent(r.primal_bound, r.x)
            if domains:
                self.domains(torch.tensor([[.1]], dtype=torch.float64), torch.tensor([[.2]], dtype=torch.float64))
            return r
    monkeypatch.setattr(flow, 'compile', compile_)
    monkeypatch.setattr(flow, 'Solver', Solver)
    return state


def test_original_optimum_replays_one_point_then_stops_before_training(monkeypatch):
    p = problem()
    calls = []
    def simulate_point(z):
        calls.append(len(z)); return simulate(z)
    p = replace(p, simulate=simulate_point)
    r0 = result(gr.Status.OPTIMAL, -.8, -.8, .4, termination=gr.Termination.GAP_CLOSED)
    state = harness(monkeypatch, r0, None)
    r = gr.solve_dynamics(p, options(), trainer=Trainer())
    assert r.accepted and r.level == 1 and r.certificate_scope == 'original'
    assert r.original is r0 and not r.rounds and len(state.solves) == 1
    assert calls == [1] and r.original_validation['passed']


def test_l1_point_failing_reference_model_falls_back(monkeypatch):
    r0 = result(gr.Status.OPTIMAL, -1.6, -1.6, .8, termination=gr.Termination.GAP_CLOSED)
    harness(monkeypatch, r0, result(gr.Status.OPTIMAL, -.8, -.8, .4))
    r = gr.solve_dynamics(problem(), options(), trainer=Trainer())
    assert not r.original_validation['passed']
    assert r.accepted and r.level == 2 and r.certificate_scope == 'surrogate'


def test_l1_source_sense_must_match_the_reference_model(monkeypatch):
    harness(monkeypatch, result(gr.Status.OPTIMAL, .8, .8, .4, sense=gr.MAX), None)
    with pytest.raises(gr.ModelError, match='objective senses'):
        gr.solve_dynamics(problem(), options(), trainer=Trainer())


def test_original_infeasible_stops_without_learning(monkeypatch):
    r0 = result(gr.Status.INFEASIBLE, termination=gr.Termination.PROVED_INFEASIBLE)
    harness(monkeypatch, r0, None)
    trainer = Trainer()
    r = gr.solve_dynamics(problem(), options(), trainer=trainer)
    assert r.accepted and r.level == 1 and r.x is None and not trainer.calls


def test_fallback_accepts_only_surrogate_scope_and_retains_l1(monkeypatch):
    r0 = result()
    r2 = result(gr.Status.OPTIMAL, -.8, -.80000001, .4)
    harness(monkeypatch, r0, r2)
    trainer = Trainer()
    r = gr.solve_dynamics(problem(), options(), trainer=trainer)
    assert r.accepted and r.level == 2 and r.certificate_scope == 'surrogate'
    assert r.original is r0 and not r.original.solved and r.surrogate is r2
    assert all(m['passed'] for m in r.rounds[0].validation)
    doc = r.to_json()
    assert doc['validation_kind'] == 'empirical' and doc['original']['status'] == 'NO_SOLUTION'
    json.dumps(doc, allow_nan=False)
    fit, val, _ = trainer.calls[0]
    test1 = sample_box(problem().box, 32, torch.Generator().manual_seed(1000000))
    assert not torch.isin(test1.flatten(), fit.z.flatten()).any()
    assert not torch.isin(test1.flatten(), val.z.flatten()).any()


def test_unsupported_l1_graph_falls_back_but_model_errors_propagate(monkeypatch):
    harness(monkeypatch, gr.UnsupportedModelError('implicit solve'), result(gr.Status.OPTIMAL, -.8, -.8, .4))
    r = gr.solve_dynamics(problem(), options(), trainer=Trainer())
    assert r.accepted and r.original is None and 'implicit solve' in r.original_error
    harness(monkeypatch, gr.ModelError('malformed box'), None)
    with pytest.raises(gr.ModelError, match='malformed'):
        gr.solve_dynamics(problem(), options(), trainer=Trainer())


def test_interruption_does_not_start_training(monkeypatch):
    harness(monkeypatch, result(termination=gr.Termination.INTERRUPTED), None)
    trainer = Trainer()
    r = gr.solve_dynamics(problem(), options(), trainer=trainer)
    assert not r.accepted and not trainer.calls


def test_solver_error_is_retained_when_falling_back(monkeypatch):
    harness(monkeypatch, result(gr.Status.ERROR, termination=gr.Termination.INTERNAL_ERROR),
            result(gr.Status.OPTIMAL, -.8, -.8, .4))
    r = gr.solve_dynamics(problem(), options(), trainer=Trainer())
    assert r.accepted and r.original.status is gr.Status.ERROR


def test_surrogate_infeasible_never_certifies_original_infeasible(monkeypatch):
    harness(monkeypatch, result(), result(gr.Status.INFEASIBLE, termination=gr.Termination.PROVED_INFEASIBLE))
    r = gr.solve_dynamics(problem(), options(), trainer=Trainer())
    assert not r.accepted and r.level is None and r.certificate_scope is None
    assert r.original.status is gr.Status.NO_SOLUTION


def test_reference_infeasibility_rejects_a_surrogate_incumbent(monkeypatch):
    harness(monkeypatch, result(), result(gr.Status.OPTIMAL, -1.6, -1.6, .8))
    r = gr.solve_dynamics(problem(), options(), trainer=Trainer())
    assert not r.accepted and r.rounds[0].optimization['reference_max_violation'] > 0


def test_simulated_reference_point_refutes_a_false_optimality_claim(monkeypatch):
    harness(monkeypatch, result(), result(gr.Status.OPTIMAL, -.4, -.4, .2))
    r = gr.solve_dynamics(problem(), options(), trainer=Trainer())
    assert not r.accepted and r.rounds[0].optimization['refuted_by_reference_point']
    assert all(v['passed'] for v in r.rounds[0].validation)


def test_objective_error_rejects_even_a_closed_surrogate_gap(monkeypatch):
    class Biased(Perfect):
        def forward(self, z):
            y = super().forward(z)
            return y + y.new_tensor([.1, 0])
    harness(monkeypatch, result(), result(gr.Status.OPTIMAL, -.7, -.7, .4))
    r = gr.solve_dynamics(problem(), options(), trainer=Trainer([Biased()]))
    assert not r.accepted and r.rounds[0].optimization['objective_error'] > r.rounds[0].optimization['tau_f']


def test_trajectory_accuracy_is_required_in_addition_to_point_checks(monkeypatch):
    class WrongTrajectory(Perfect):
        def trajectory(self, z):
            return super().trajectory(z) + 1
    harness(monkeypatch, result(), result(gr.Status.OPTIMAL, -.8, -.8, .4))
    r = gr.solve_dynamics(problem(), options(), trainer=Trainer([WrongTrajectory()]))
    assert not r.accepted and r.rounds[0].optimization['passed']
    assert all(v['trajectory_nrmse_median'] > .01 for v in r.rounds[0].validation)


def test_second_independent_test_set_must_also_pass(monkeypatch):
    audit = sample_box(problem().box, 32, torch.Generator().manual_seed(2000000))
    bad = audit[audit[:, 0] > .4][0, 0]
    class AuditFailure(Perfect):
        def forward(self, z):
            y = super().forward(z)
            return torch.stack((y[:, 0], torch.where(z[:, 0] == bad, -1., y[:, 1])), 1)
    harness(monkeypatch, result(), result(gr.Status.OPTIMAL, -.8, -.8, .4))
    r = gr.solve_dynamics(problem(), options(), trainer=Trainer([AuditFailure()]))
    a, b = r.rounds[0].validation
    assert a['passed'] and not b['passed'] and b['false_feasible'] == 1 and not r.accepted


def test_refinement_uses_open_boxes_holds_out_new_data_and_restarts_from_round0(monkeypatch):
    harness(monkeypatch, result(), [result(gr.Status.FEASIBLE, -.8, -1., .4), result(gr.Status.OPTIMAL, -.8, -.8, .4)], domains=True)
    trainer = Trainer()
    r = gr.solve_dynamics(problem(), options(max_refinements=1, additional_fraction=1), trainer=trainer)
    assert r.accepted and len(trainer.calls) == 2
    d0, v0, _ = trainer.calls[0]
    d1, v1, initial = trainer.calls[1]
    assert initial is not None and len(d1) > len(d0) and len(v1) > len(v0)
    new = d1.z[len(d0):]
    assert ((new[:, 0] >= .1) & (new[:, 0] <= .2)).any()
    assert not torch.isin(new.flatten(), v1.z.flatten()).any()


def test_bounds_from_different_rounds_are_not_merged(monkeypatch):
    class Biased(Perfect):
        def forward(self, z):
            y = super().forward(z)
            return y + y.new_tensor([.1, 0])
    harness(monkeypatch, result(), [result(gr.Status.OPTIMAL, -.7, -.7, .4), result(gr.Status.FEASIBLE, -.8, -1., .4)])
    r = gr.solve_dynamics(problem(), options(max_refinements=1), trainer=Trainer([Biased(), Perfect()]))
    assert not r.accepted and len(r.rounds) == 2
    assert r.rounds[0].compiled_hash != r.rounds[1].compiled_hash
    assert r.rounds[1].optimization['gap'] == pytest.approx(.2)
    assert r.rounds[1].result.dual_bound == -1.


def test_maximization_keeps_the_users_bound_direction(monkeypatch):
    class MaxModel(Perfect):
        def forward(self, z):
            return super().forward(z) * z.new_tensor([-1, 1])
    p = replace(problem(), sense=gr.MAX, outputs=lambda t, z: outputs(t, z) * z.new_tensor([-1, 1]))
    harness(monkeypatch, result(sense=gr.MAX), result(gr.Status.OPTIMAL, .8, .80000001, .4, sense=gr.MAX))
    r = gr.solve_dynamics(p, options(), trainer=Trainer([MaxModel()]))
    assert r.accepted and r.rounds[0].optimization['gap'] == pytest.approx(1e-8)


def test_nonexistent_reference_trajectory_cannot_be_accepted(monkeypatch):
    p = replace(problem(), exists=lambda z, t: torch.zeros(len(z), dtype=torch.bool))
    harness(monkeypatch, result(), result(gr.Status.OPTIMAL, -.8, -.8, .4))
    r = gr.solve_dynamics(p, options(), trainer=Trainer())
    assert not r.accepted and not r.rounds[0].optimization['reference_exists']


def test_reference_simulator_must_satisfy_discrete_equations():
    p = replace(problem(), residual=lambda t, z: torch.ones_like(t[:, 1:]))
    with pytest.raises(gr.ModelError, match='discrete residual'):
        reference_samples(p, torch.tensor([[.3]], dtype=torch.float64))


def test_existing_case_protocol_adapter_preserves_the_formulation():
    def group_values(z, t):
        return t[:, -1] - .8
    case = SimpleNamespace(lower=torch.tensor([0.], dtype=torch.float64), upper=torch.tensor([1.], dtype=torch.float64),
        simulate=simulate, residual=residual, init_M=torch.zeros(1, 1, dtype=torch.float64),
        init_c=torch.zeros(1, dtype=torch.float64), obj=SimpleNamespace(fn=lambda t: -t[:, -1]),
        obj_z=lambda z: .1 * z.square(), groups=[SimpleNamespace(values=group_values)], traj_exists=None)
    adapted = gr.DynamicsProblem.from_case(case, original=None)
    z = torch.tensor([[.1], [.4]], dtype=torch.float64)
    data = reference_samples(adapted, z)
    expected = outputs(simulate(z), z) + z.new_tensor([[.001, 0], [.016, 0]])
    torch.testing.assert_close(data.values, expected)
    assert torch.equal(adapted.initial_state(z), simulate(z)[:, 0])


def test_native_trainer_interval_bounds_training_and_direct_export():
    p, o = problem(), options().training
    rng = torch.Generator().manual_seed(71)
    data = reference_samples(p, sample_box(p.box, 32, rng))
    val = reference_samples(p, sample_box(p.box, 16, rng))
    norm = Normalization(data)
    model = gr.VAPINNTrainer(o).fit(p, data, val, norm, None, 0, norm.sigma_f * .001, 'cpu')
    points = torch.linspace(.1, .9, 17, dtype=torch.float64)[:, None]
    lo, hi = model.interval(points.new_tensor([[.1]]), points.new_tensor([[.9]]))
    assert ((model(points) >= lo - 1e-12) & (model(points) <= hi + 1e-12)).all()
    assert torch.equal(model.trajectory(points)[:, 0], p.initial_state(points))
    c = gr.compile(model, box=p.box, objective=0, le=[1], options=gr.Options(device='cpu'))
    assert c.path == 'direct:torch'


def test_domain_observer_is_read_only_and_optional():
    from gear.solver import _Sink
    from gear import report
    class Domains:
        lower = torch.zeros(3, 1)
        upper = torch.ones(3, 1)
        def __len__(self): return 3
        def get_topk_indices(self, k, largest): return torch.arange(k)
        def __getitem__(self, indices):
            return (torch.zeros(len(indices), 1), self.lower[:len(indices)], self.upper[:len(indices)])
    received = []
    sink = _Sink(1, [], [], [(lambda l, u: received.append((l, u)), 2)])
    report.install(sink)
    try: report.record_open_domains(Domains())
    finally: report.uninstall()
    assert received[0][0].dtype == torch.float64 and received[0][0].shape == (2, 1)
    received[0][0].fill_(7)
    assert Domains.lower.eq(0).all()
    report.record_open_domains(None)  # default path does not inspect a domain list


@pytest.mark.parametrize('kw', [{'max_refinements': -1}, {'tolerance': 'loose'}, {'feasibility_tolerance': -1},
                               {'additional_fraction': math.nan}])
def test_invalid_cascade_options(kw):
    with pytest.raises(gr.OptionError):
        options(**kw)


def test_zero_total_budget_does_not_compile_or_train(monkeypatch):
    state = harness(monkeypatch, None, None)
    trainer = Trainer()
    r = gr.solve_dynamics(problem(), options(total_time_limit=0), trainer=trainer)
    assert r.budget_exhausted and not r.accepted and r.time_limit == 0
    assert not state.solves and not trainer.calls and r.phase == 'original_compile'
    json.dumps(r.to_json(), allow_nan=False)


def test_total_budget_includes_reference_data_and_preserves_l1(monkeypatch):
    deadline = importlib.import_module('gear.dynamics.deadline')
    clock = [100.0]
    monkeypatch.setattr(deadline, '_now', lambda: clock[0])
    l1 = result()
    state = harness(monkeypatch, l1, None)
    trainer = Trainer()
    def costly_reference(z):
        clock[0] += 11
        return simulate(z)
    phases = []
    r = gr.solve_dynamics(replace(problem(), simulate=costly_reference),
                         options(total_time_limit=10), trainer=trainer,
                         on_progress=lambda snapshot: phases.append(snapshot['phase']))
    assert r.budget_exhausted and not r.accepted and r.original is l1
    assert r.phase == 'reference_data' and r.elapsed_seconds == 11
    assert len(state.solves) == 1 and not trainer.calls
    assert phases == ['original_compile', 'original_solve', 'original_validation', 'reference_data']


def test_all_solves_are_capped_by_remaining_shared_budget(monkeypatch):
    deadline = importlib.import_module('gear.dynamics.deadline')
    clock = [100.0]
    monkeypatch.setattr(deadline, '_now', lambda: clock[0])
    budget = deadline.Budget(10)
    from gear.config import resolve
    from gear.solver import _time_limit
    opts = gr.Options(device='cpu', time_limit=100)
    clock[0] += 7
    assert _time_limit(resolve(budget.solve_options(opts)).config) == 3
    clock[0] += 3
    with pytest.raises(deadline.BudgetExpired):
        budget.solve_options(opts)


def test_training_deadline_returns_fitted_model_before_first_adam_update(monkeypatch):
    training = importlib.import_module('gear.dynamics.training')
    p = problem()
    data = reference_samples(p, sample_box(p.box, 32, torch.Generator().manual_seed(0)))
    norm = Normalization(data)
    monkeypatch.setattr(training, '_now', lambda: 20.)
    model = training.VAPINNTrainer(options().training).fit(p, data, data, norm, None, 0, .001, 'cpu', deadline=10.)
    assert model.training_stopped_by_budget and model.training_updates == 0
    assert torch.isfinite(model(data.z)).all()


def test_timeout_during_l2_validation_retains_unvalidated_result(monkeypatch):
    deadline = importlib.import_module('gear.dynamics.deadline')
    clock = [100.]
    monkeypatch.setattr(deadline, '_now', lambda: clock[0])
    l1, l2 = result(), result(gr.Status.OPTIMAL, -.8, -.8, .4)
    harness(monkeypatch, l1, l2)
    def progress(snapshot):
        if snapshot['phase'] == 'surrogate_validation':
            clock[0] += 11
    r = gr.solve_dynamics(problem(), options(total_time_limit=10), trainer=Trainer(), on_progress=progress)
    assert r.budget_exhausted and not r.accepted and not r.rounds
    assert r.original is l1 and r.pending_surrogate is l2 and r.surrogate is l2
    assert r.to_json()['pending_surrogate']['status'] == 'OPTIMAL'
