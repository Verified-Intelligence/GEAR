"""Discrete dynamics and the separate original-model / surrogate results."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Optional

import torch

from ..config import Options
from ..errors import ModelError, OptionError
from ..result import MIN, Result, Sense, _plain


@dataclass(frozen=True)
class DynamicsProblem:
    """Callbacks operate on batches, in the user's objective sense.

    ``original`` is a Compiled/Legacy source, or ``factory(options)``. It may be
    None when an original graph cannot be supplied. ``simulate(z)`` returns
    [B,T+1,S]; ``outputs(trajectory,z)`` returns [B,1+R], objective first and
    constraint rows <= 0. ``residual(trajectory,z)`` returns [B,T,S].
    ``initial_state(z)`` returns [B,S] and is imposed exactly by the surrogate.
    Optional ``exists`` marks reference simulations that have no trajectory;
    ``relevant_nodes`` returns the valid/decision-relevant future prefix length.
    ``project_original`` maps an original full-space incumbent to controls.
    """
    original: Any
    box: tuple
    simulate: Callable
    initial_state: Callable
    outputs: Callable
    residual: Callable
    sense: Sense = MIN
    exists: Optional[Callable] = None
    relevant_nodes: Optional[Callable] = None
    project_original: Optional[Callable] = None
    reference_tolerance: float = 1e-6

    @classmethod
    def from_case(cls, case, *, original, **kwargs):
        """Adapt the paper's existing Case protocol without importing its package."""
        from .adapter import from_case
        return from_case(case, original=original, **kwargs)

    def __post_init__(self):
        lo, hi = (torch.as_tensor(v, dtype=torch.float64).detach().cpu().flatten().clone() for v in self.box)
        if lo.numel() == 0 or lo.shape != hi.shape or not torch.isfinite(torch.cat((lo, hi))).all() or (lo > hi).any():
            raise ModelError("dynamics box must have finite, nonempty, ordered endpoints")
        if self.sense not in (Sense.MIN, Sense.MAX):
            raise ModelError("dynamics sense must be MIN or MAX")
        if not math.isfinite(self.reference_tolerance) or self.reference_tolerance < 0:
            raise ModelError("reference_tolerance must be finite and nonnegative")
        object.__setattr__(self, "box", (lo, hi))


@dataclass(frozen=True)
class VAPINNOptions:
    train_size: Optional[int] = None          # default 4096 + 256 d
    validation_size: Optional[int] = None    # default train_size / 4
    test_size: int = 512                    # each of two independent sets
    steps: int = 150000
    refinement_steps: int = 20000
    width: int = 64
    depth: int = 3
    batch_size: int = 1024
    learning_rate: float = 2e-4
    refinement_rate: float = 3e-4
    ridge_every: int = 250
    ridge: float = 1e-8
    evaluate_every: int = 500
    relaxation_weight: float = 1e-4
    bound_weight: float = 1e-4
    bound_boxes: int = 16
    bound_points: int = 8
    quantile: float = .99
    seed: int = 0
    device: Optional[str] = None             # inherits surrogate solve device

    def __post_init__(self):
        for key in ("test_size", "steps", "refinement_steps", "width", "depth", "batch_size",
                    "ridge_every", "evaluate_every", "bound_boxes", "bound_points"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise OptionError(f"{key} must be a positive integer")
        for key in ("train_size", "validation_size"):
            if getattr(self, key) is not None and (type(getattr(self, key)) is not int or getattr(self, key) < 2):
                raise OptionError(f"{key} must be at least two")
        for key in ("learning_rate", "refinement_rate", "ridge", "relaxation_weight", "bound_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise OptionError(f"{key} must be finite and nonnegative")
        if not 0 < self.quantile <= 1:
            raise OptionError("quantile must be in (0,1]")
        if self.ridge <= 0:
            raise OptionError("ridge must be positive")
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 31:
            raise OptionError("seed must be a nonnegative integer within the generator range")


@dataclass(frozen=True)
class DynamicsOptions:
    original: Options = field(default_factory=lambda: Options(time_limit=600))
    surrogate: Options = field(default_factory=lambda: Options(time_limit=600))
    training: VAPINNOptions = field(default_factory=VAPINNOptions)
    max_refinements: int = 6
    tolerance: str = "strict"               # standard: .01 sigma_F; strict: .001 sigma_F
    feasibility_tolerance: float = 1e-6
    additional_fraction: float = .1
    total_time_limit: Optional[float] = None   # includes simulation, training, compile, solve, validation
    training_time_fraction: float = .7        # of remaining global time, reserving time for solve/checks

    def __post_init__(self):
        if type(self.max_refinements) is not int or self.max_refinements < 0:
            raise OptionError("max_refinements must be a nonnegative integer")
        if self.tolerance not in ("standard", "strict"):
            raise OptionError("tolerance must be 'standard' or 'strict'")
        if not math.isfinite(self.feasibility_tolerance) or self.feasibility_tolerance < 0:
            raise OptionError("feasibility_tolerance must be finite and nonnegative")
        if not math.isfinite(self.additional_fraction) or self.additional_fraction <= 0:
            raise OptionError("additional_fraction must be positive and finite")
        if self.total_time_limit is not None and (not math.isfinite(self.total_time_limit) or self.total_time_limit < 0):
            raise OptionError('total_time_limit must be finite and nonnegative')
        if not 0 < self.training_time_fraction < 1:
            raise OptionError('training_time_fraction must lie in (0,1)')


@dataclass(frozen=True)
class Samples:
    z: torch.Tensor
    trajectory: torch.Tensor
    values: torch.Tensor
    exists: torch.Tensor

    def __len__(self):
        return len(self.z)

    def select(self, indices):
        return Samples(*(getattr(self, k)[indices] for k in ("z", "trajectory", "values", "exists")))


def concatenate(*parts):
    return Samples(*(torch.cat([getattr(p, k) for p in parts]) for k in ("z", "trajectory", "values", "exists")))


def sample_box(box, n, generator):
    lo, hi = box
    return lo + (hi - lo) * torch.rand(n, len(lo), generator=generator, dtype=torch.float64)


def reference_samples(problem, z):
    """Independent reference evaluation, on CPU in float64; never a learned path."""
    z = z.detach().cpu().double()
    with torch.no_grad():
        trajectory = problem.simulate(z).detach().cpu().double()
        values = problem.outputs(trajectory, z).detach().cpu().double()
        initial = problem.initial_state(z).detach().cpu().double()
        exists = torch.ones(len(z), dtype=torch.bool) if problem.exists is None else problem.exists(z, trajectory).cpu().bool()
    if trajectory.ndim != 3 or trajectory.shape[0] != len(z) or trajectory.shape[1] < 2 or trajectory.shape[2] < 1:
        raise ModelError("simulate must return [batch,T+1,S], with T >= 1")
    if values.ndim != 2 or values.shape[0] != len(z) or values.shape[1] < 1 or exists.shape != (len(z),):
        raise ModelError("outputs must return [batch,1+R] and exists must return [batch]")
    if not torch.isfinite(trajectory).all() or not torch.isfinite(values).all():
        raise ModelError("reference simulation/output is nonfinite; supply finite placeholders and exists=False for collapse")
    if initial.shape != trajectory[:, 0].shape or not torch.allclose(initial, trajectory[:, 0], atol=1e-9, rtol=1e-9):
        raise ModelError("initial_state disagrees with the reference simulation")
    with torch.no_grad():
        residual = problem.residual(trajectory, z)
    if residual.shape != trajectory[:, 1:].shape or not torch.isfinite(residual).all():
        raise ModelError("reference residual must be finite with shape [batch,T,S]")
    if exists.any() and float(residual[exists].abs().max()) > problem.reference_tolerance:
        raise ModelError("reference simulator does not satisfy the declared discrete residual")
    return Samples(z, trajectory, values, exists)


@dataclass(frozen=True)
class DynamicsRound:
    index: int
    result: Result
    margins: tuple
    optimization: dict
    validation: tuple
    accepted: bool
    training_seconds: float
    reference_seconds: float
    compiled_hash: str
    training_updates: Optional[int] = None
    training_stopped_by_budget: bool = False


@dataclass(frozen=True)
class DynamicsResult:
    """An L2 acceptance is empirical; its bound belongs to the tightened surrogate.

    The original Result is always retained and never replaced by an L2 bound.
    Different round models' bounds are never combined.
    """
    original: Optional[Result]
    original_error: Optional[str]
    rounds: tuple[DynamicsRound, ...]
    level: Optional[int]
    accepted: bool
    certificate_scope: Optional[str]
    x: Optional[dict]
    sigma_f: Optional[float] = None
    elapsed_seconds: float = 0.0
    original_validation: Optional[dict] = None
    surrogate_model: Any = field(default=None, repr=False, compare=False)
    time_limit: Optional[float] = None
    budget_exhausted: bool = False
    phase: str = 'complete'
    pending_surrogate: Optional[Result] = None

    @property
    def surrogate(self):
        return self.pending_surrogate if self.pending_surrogate is not None else self.rounds[-1].result if self.rounds else None

    def to_json(self):
        rounds = [dict(index=r.index, result=r.result.to_json(), margins=list(r.margins),
                       optimization=_plain(r.optimization), validation=_plain(r.validation), accepted=r.accepted,
                       training_seconds=r.training_seconds, reference_seconds=r.reference_seconds,
                       compiled_hash=r.compiled_hash, training_updates=r.training_updates,
                       training_stopped_by_budget=r.training_stopped_by_budget) for r in self.rounds]
        return dict(schema=1, level=self.level, accepted=self.accepted, certificate_scope=self.certificate_scope,
                    validation_kind="empirical" if rounds else None,
                    original=None if self.original is None else self.original.to_json(),
                    original_validation=_plain(self.original_validation),
                    original_error=self.original_error, rounds=rounds, x=self.x, sigma_f=self.sigma_f,
                    elapsed_seconds=self.elapsed_seconds, time_limit=self.time_limit,
                    budget_exhausted=self.budget_exhausted, phase=self.phase,
                    pending_surrogate=None if self.pending_surrogate is None else self.pending_surrogate.to_json())
