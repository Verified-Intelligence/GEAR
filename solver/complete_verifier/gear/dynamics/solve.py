"""Original graph first, then independently validated VA-PINN refinement."""
from __future__ import annotations

import copy
import math
import time
from dataclasses import replace

import torch
from torch import nn

from ..compile import Compiled, compile
from ..config import resolve
from ..errors import ModelError, UnsupportedModelError, UnsupportedOperatorError, UnsupportedProblemError
from ..result import Status, Termination
from ..solver import Solver
from .problem import (DynamicsOptions, DynamicsResult, DynamicsRound, concatenate,
                      reference_samples, sample_box)
from .training import Normalization, VAPINNTrainer, trajectory_errors
from .deadline import Budget, BudgetExpired


class _Tightened(nn.Module):
    def __init__(self, model, margins):
        super().__init__()
        self.model = copy.deepcopy(model).cpu().double().eval()
        self.register_buffer("shift", torch.cat((torch.zeros(1, dtype=torch.float64), margins)))

    def forward(self, z):
        return self.model(z) + self.shift


def _point(x, d):
    if x is None or len(x) != d or any(f"X_{i}" not in x for i in range(d)):
        return None
    z = torch.tensor([[x[f"X_{i}"] for i in range(d)]], dtype=torch.float64)
    return z if torch.isfinite(z).all() else None


def _max_row(values):
    return values[:, 1:].amax(1) if values.shape[1] > 1 else values.new_full((len(values),), -math.inf)


def _original_validation(problem, result, options):
    if result.status == Status.INFEASIBLE:
        return dict(passed=True, reason='original graph proved infeasible')
    z = problem.project_original(result.x) if problem.project_original else _point(result.x, len(problem.box[0]))
    if z is None:
        raise ModelError('a full-space original incumbent requires project_original to recover its controls')
    z = torch.as_tensor(z, dtype=torch.float64).reshape(1, -1)
    lo, hi = problem.box
    if z.shape != (1, len(lo)) or not torch.isfinite(z).all() or ((z < lo) | (z > hi)).any():
        return dict(passed=False, reason='original controls outside the declared box')
    sample = reference_samples(problem, z)
    f = float(sample.values[0, 0])
    violation = float(_max_row(sample.values)[0])
    cfg = resolve(options.original).config['solver']['nlp']
    gap = int(problem.sense) * (f - result.dual_bound)
    tolerance = max(float(cfg['bab_primal_dual_abs_tol']),
                    float(cfg['bab_primal_dual_rel_tol']) * max(1, abs(f), abs(result.dual_bound)))
    roundoff = 1e-12 * max(1, abs(f), abs(result.dual_bound))
    return dict(passed=bool(sample.exists[0]) and violation <= options.feasibility_tolerance and
                math.isfinite(gap) and -roundoff <= gap <= tolerance,
                reference_exists=bool(sample.exists[0]), reference_objective=f,
                reference_max_violation=violation, gap=gap, gap_tolerance=tolerance)


def _metrics(problem, model, margins, data, norm, tau, feas_tol):
    with torch.no_grad():
        predicted = model.trajectory(data.z)
        yh = model(data.z)
        if predicted.shape != data.trajectory.shape or yh.shape != data.values.shape:
            raise ModelError("surrogate shapes disagree with reference outputs/trajectories")
        error = (yh[:, 0] - data.values[:, 0]).abs()
        nrmse = trajectory_errors(problem, predicted, data, norm)
        shifted = torch.cat((yh[:, :1], yh[:, 1:] + margins), 1)
        predicted_feasible = _max_row(shifted) <= feas_tol
        feasible = data.exists & (_max_row(data.values) <= feas_tol)
        median, p95, state = float(torch.quantile(error, .5)), float(torch.quantile(error, .95)), float(torch.quantile(nrmse, .5))
        ff = int((predicted_feasible & ~feasible).sum())
        finite = bool(torch.isfinite(yh).all() and torch.isfinite(predicted).all() and torch.isfinite(nrmse).all())
        return dict(n=len(data), objective_error_median=median, objective_error_p95=p95,
                    trajectory_nrmse_median=state, false_feasible=ff, reference_feasible=int(feasible.sum()),
                    passed=finite and median <= tau and p95 <= 5 * tau and state <= .01 and ff == 0)


def _optimization(problem, model, margins, result, tau, gap_tol, seen, feas_tol):
    z = _point(result.x, len(problem.box[0]))
    report = dict(passed=False, tau_f=tau, gap_tolerance=gap_tol)
    if z is None or result.replay is None or not result.replay.ok or result.status not in (Status.OPTIMAL, Status.FEASIBLE):
        report['reason'] = 'no replayed surrogate incumbent'
        return report
    lo, hi = problem.box
    if ((z < lo) | (z > hi)).any():
        report['reason'] = 'surrogate incumbent outside the control box'
        return report
    ref = reference_samples(problem, z)
    with torch.no_grad():
        yh = model(z)[0]
    sign = int(problem.sense)
    gap = sign * (float(yh[0]) - result.dual_bound)
    error = abs(float(yh[0]) - float(ref.values[0, 0]))
    feasible = bool(ref.exists[0]) and float(_max_row(ref.values)[0]) <= feas_tol
    predicted_feasible = bool((yh[1:] + margins <= feas_tol).all())
    feasible_seen = seen.exists & (_max_row(seen.values) <= feas_tol)
    best_seen = float((sign * seen.values[feasible_seen, 0]).min()) if feasible_seen.any() else math.inf
    refuted = best_seen < sign * result.dual_bound - tau
    roundoff = 1e-12 * max(1, abs(float(yh[0])), abs(result.dual_bound))
    finite = math.isfinite(gap) and math.isfinite(error) and bool(torch.isfinite(yh).all())
    replay_matches = abs(float(yh[0]) - result.primal_bound) <= roundoff
    report.update(gap=gap, objective_error=error, reference_objective=float(ref.values[0, 0]),
                  reference_max_violation=float(_max_row(ref.values)[0]), reference_exists=bool(ref.exists[0]),
                  refuted_by_reference_point=refuted, replay_matches=replay_matches,
                  passed=finite and replay_matches and -roundoff <= gap <= gap_tol and
                  feasible and predicted_feasible and error <= tau and not refuted)
    return report


def _refinement_points(problem, n, regions, candidates, generator):
    lo, hi = problem.box
    d = len(lo)
    half = n // 2
    if regions:
        lower, upper = regions[-1]
        lower, upper = lower.reshape(-1, d), upper.reshape(-1, d)
        pick = torch.arange(half) % len(lower)
        region = lower[pick] + (upper[pick] - lower[pick]) * torch.rand(half, d, generator=generator, dtype=torch.float64)
    else:
        region = sample_box(problem.box, half, generator)
    if candidates:
        centres = torch.cat(candidates)
        idx = torch.arange(n - half)
        radii = torch.tensor([.01, .03, .1], dtype=torch.float64)[(idx // len(centres)) % 3, None]
        unit = 2 * torch.rand(n - half, d, generator=generator, dtype=torch.float64) - 1
        local = centres[idx % len(centres)] + radii * (hi - lo) * unit / 2
        local = torch.minimum(torch.maximum(local, lo), hi)
    else:
        local = sample_box(problem.box, n - half, generator)
    return torch.cat((region, local))


def _snapshot(state, budget):
    return DynamicsResult(state.get('original'), state.get('original_error'), tuple(state.get('rounds', ())),
        None, False, None, None, state.get('sigma_f'), budget.elapsed(),
        original_validation=state.get('original_validation'), surrogate_model=state.get('model'),
        time_limit=budget.limit, phase=budget.phase, pending_surrogate=state.get('last_surrogate'))


def solve_dynamics(problem, options=None, *, trainer=None, on_progress=None):
    """Run Algorithm 1's L1 -> L2 cascade, preserving every certificate's scope.

    The existing solve() is unchanged. Each L2 round solves a new fixed model;
    bounds of different models are never merged. With total_time_limit, every
    phase shares one deadline. A process watchdog is needed to preempt a single
    long-running native kernel or an uncooperative user callback.
    """
    options = DynamicsOptions() if options is None else options
    budget, state = Budget(options.total_time_limit), {}
    def phase(name):
        budget.check(name)
        if on_progress is not None:
            on_progress(_snapshot(state, budget).to_json())
    try:
        result = _run(problem, options, trainer, budget, state, phase)
        budget.check()
    except BudgetExpired:
        result = replace(_snapshot(state, budget), budget_exhausted=True)
    return replace(result, time_limit=budget.limit, elapsed_seconds=budget.elapsed())


def _reference(problem, z, budget):
    parts = []
    for first in range(0, len(z), 128):
        budget.check()
        parts.append(reference_samples(problem, z[first:first + 128]))
        budget.check()
    return concatenate(*parts)


def _run(problem, options, trainer, budget, state, phase):
    start = budget.start
    phase('original_compile')
    original, original_error, original_validation = None, None, None
    try:
        if problem.original is None:
            original_error = 'original graph not supplied'
        else:
            original_options = budget.solve_options(options.original)
            source = problem.original(original_options) if callable(problem.original) and not isinstance(problem.original, Compiled) else problem.original
            if isinstance(source, Compiled) and source.sense != problem.sense:
                raise ModelError('original graph and dynamics outputs have different objective senses')
            phase('original_solve')
            original = Solver(budget.solve_options(options.original)).solve(source)
    except (UnsupportedModelError, UnsupportedOperatorError, UnsupportedProblemError) as error:
        original_error = f'{type(error).__name__}: {error}'
    state.update(original=original, original_error=original_error)
    if original is not None and (original.status == Status.ERROR or original.termination == Termination.INTERNAL_ERROR):
        phase('original_error')
        return _snapshot(state, budget)
    phase('original_validation')
    if original is not None and original.sense != problem.sense:
        raise ModelError('original graph and dynamics outputs have different objective senses')
    if original is not None and original.solved:
        original_validation = _original_validation(problem, original, options)
        state['original_validation'] = original_validation
        budget.check()
        if original_validation['passed']:
            return DynamicsResult(original, original_error, (), 1, True, 'original', original.x,
                                  elapsed_seconds=time.monotonic() - start, original_validation=original_validation)
    if original is not None and (original.status == Status.UNBOUNDED or original.termination == Termination.INTERRUPTED):
        return DynamicsResult(original, original_error, (), None, False, None, original.x,
                              elapsed_seconds=time.monotonic() - start)

    o = options.training
    n = o.train_size or 4096 + 256 * len(problem.box[0])
    nv = o.validation_size or max(2, n // 4)
    rng = torch.Generator().manual_seed(o.seed)
    phase('reference_data')
    t = time.monotonic()
    data = _reference(problem, sample_box(problem.box, n, rng), budget)
    validation = _reference(problem, sample_box(problem.box, nv, rng), budget)
    # Independent generators; neither test set is given to the trainer.
    tests = tuple(_reference(problem, sample_box(problem.box, o.test_size,
                  torch.Generator().manual_seed(offset + 17 * o.seed)), budget) for offset in (1000000, 2000000))
    reference_seconds = time.monotonic() - t
    norm = Normalization(data)
    state['sigma_f'] = norm.sigma_f
    tau = norm.sigma_f * (.001 if options.tolerance == 'strict' else .01)
    gap_tol = 1e-4 if options.tolerance == 'strict' else tau
    trainer = VAPINNTrainer(o) if trainer is None else trainer
    device = resolve(options.surrogate).config['general']['device']
    feas_tol = options.feasibility_tolerance
    seen = concatenate(data, validation, *tests)
    if original is not None and original.x is not None:
        z = problem.project_original(original.x) if problem.project_original else _point(original.x, len(problem.box[0]))
        if z is not None:
            z = torch.as_tensor(z, dtype=torch.float64).reshape(1, -1)
            if z.shape[1] == len(problem.box[0]) and ((z >= problem.box[0]) & (z <= problem.box[1])).all():
                seen = concatenate(seen, reference_samples(problem, z))
    initial, rounds = None, []
    state['rounds'] = rounds
    for k in range(options.max_refinements + 1):
        phase('training')
        t = time.monotonic()
        extra = {}
        if budget.end is not None and isinstance(trainer, VAPINNTrainer):
            extra['deadline'] = time.monotonic() + options.training_time_fraction * budget.remaining()
        model = trainer.fit(problem, data, validation, norm, initial, k, tau, device, **extra).cpu().double().eval()
        state['model'] = model
        budget.check()
        training_seconds = time.monotonic() - t
        if initial is None:
            initial = copy.deepcopy(model)
        phase('calibration')
        with torch.no_grad():
            errors = validation.values[:, 1:] - model(validation.z)[:, 1:]
            margins = torch.quantile(errors, o.quantile, dim=0).clamp_min(0) if errors.shape[1] else torch.zeros(0, dtype=torch.float64)
        if not torch.isfinite(margins).all():
            raise ModelError('nonfinite calibrated margins')
        tightened = _Tightened(model, margins)
        phase('surrogate_compile')
        compiled = compile(tightened, box=problem.box, objective=0, le=tuple(range(1, data.values.shape[1])),
                           sense=problem.sense, options=options.surrogate)
        candidates, regions = [], []
        phase('surrogate_solve')
        solver = Solver(budget.solve_options(options.surrogate))
        def incumbent(value, x):
            z = _point(x, len(problem.box[0]))
            if z is not None:
                candidates.append(z)
                if len(candidates) > 16:
                    del candidates[0]
        solver.on_incumbent(incumbent)
        solver.on_domains(lambda lower, upper: regions.append((lower, upper)), limit=32)
        result = solver.solve(compiled)
        state['last_surrogate'] = result
        if result.status == Status.ERROR or result.termination == Termination.INTERNAL_ERROR:
            phase('surrogate_error')
            return _snapshot(state, budget)
        phase('surrogate_validation')
        t = time.monotonic()
        point = _point(result.x, len(problem.box[0]))
        if point is not None:
            candidates.append(point)
        if candidates:
            seen = concatenate(seen, _reference(problem, torch.cat(candidates), budget))
        metrics = tuple(_metrics(problem, model, margins, test, norm, tau, feas_tol) for test in tests)
        opt = _optimization(problem, model, margins, result, tau, gap_tol, seen, feas_tol)
        budget.check()
        accepted = opt['passed'] and all(m['passed'] for m in metrics)
        rounds.append(DynamicsRound(k, result, tuple(margins.tolist()), opt, metrics, accepted,
                                    training_seconds, reference_seconds + time.monotonic() - t, compiled.hash,
                                    getattr(model, 'training_updates', None), getattr(model, 'training_stopped_by_budget', False)))
        state.pop('last_surrogate', None)
        phase('round_complete')
        if accepted:
            return DynamicsResult(original, original_error, tuple(rounds), 2, True, 'surrogate', result.x,
                                  norm.sigma_f, time.monotonic() - start,
                                  original_validation=original_validation, surrogate_model=model)
        if k == options.max_refinements:
            break
        phase('refinement_data')
        t = time.monotonic()
        acquired = _reference(problem, _refinement_points(problem, max(5, math.ceil(n * options.additional_fraction)),
                                                          regions, candidates, rng), budget)
        holdout = max(1, len(acquired) // 5)
        validation = concatenate(validation, acquired.select(slice(0, holdout)))
        data = concatenate(data, acquired.select(slice(holdout, None)))
        seen = concatenate(seen, acquired)
        reference_seconds = time.monotonic() - t
    return DynamicsResult(original, original_error, tuple(rounds), None, False, None, None,
                          norm.sigma_f, time.monotonic() - start,
                          original_validation=original_validation, surrogate_model=model)
