"""Result, Replay, Stats and Provenance of a solve (G5 DESIGN section 6).

``Status`` and ``Termination`` are the enums of the WP1 sink (``gear/report.py``); the status mapping itself is
``report.map_result``. Values are in the user's sense. ``Result.to_json`` keeps every key of the WP5 record
(``report.ResultRecord.to_json``), so the JSON consumers of the CLI read it unchanged, and adds ``stats`` and
``provenance``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal, Mapping, Optional

from .report import Status, Termination  # noqa: F401  (re-exported)

SCHEMA = 2


class Sense(IntEnum):
    MIN = 1
    MAX = -1


MIN, MAX = Sense.MIN, Sense.MAX


@dataclass(frozen=True)
class Replay:
    """The incumbent re-evaluated on ``Compiled.original`` (section 6 'Replay predicates')."""
    ok: bool
    feas_tol: float
    max_row_violation: float
    worst_row: Optional[str]
    max_bound_violation: float
    max_integrality_violation: float
    objective: float          # recomputed, user sense; nan without an incumbent
    method: Literal["rational", "interval", "float64+margin"] = "float64+margin"


@dataclass(frozen=True)
class Stats:
    phases: Mapping[str, float]                       # monotonic seconds per phase
    deadline_overshoot: Optional[float]               # end - T under timing="solve"; None under "legacy"
    op_budgets: Mapping[str, tuple]                   # stage -> (max used, cap, exhausted), from the stages' logs
    nodes: int                                        # BaB domains visited (the solver's bab_ret)
    peak_gpu_mb: float                                # torch.cuda.max_memory_allocated of the process; 0 on CPU
    rejected_incumbents: int                          # incumbents the final replay rejected (0 or 1, see REPORT)
    reuse: Literal["none", "template", "fresh_fallback"] = "none"
    op_ledger: tuple = ()                             # the [Op] ledger lines


@dataclass(frozen=True)
class Provenance:
    path: str
    source: Mapping[str, str]
    compiled_hash: str
    program_hash: str
    pipeline: tuple
    preset: str
    preset_sha256: str
    historical_preset: bool
    options_hash: str
    timing: Literal["solve", "legacy"]
    op_budgets: Mapping[str, int]
    op_budget_commit: Literal["partial", "whole"]
    op_budget_commit_source: Literal["schema_default", "explicit", "forced_whole"]
    integer_receipt: Any
    open_route: bool
    cpu_solvers: Mapping[str, int]
    commit: str
    device: str
    extensions: tuple = ()


def _num(v):
    """JSON has no inf/nan literals (the WP5 record's convention)."""
    return v if isinstance(v, float) and math.isfinite(v) else (repr(float(v)) if isinstance(v, float) else v)


def _plain(v):
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, IntEnum):
        return int(v)
    if isinstance(v, float):
        return _num(v)
    if hasattr(v, "__dataclass_fields__"):
        return {k: _plain(getattr(v, k)) for k in v.__dataclass_fields__}
    return v


@dataclass(frozen=True)
class Result:
    status: Status
    termination: Termination
    sense: Sense
    primal_bound: float
    dual_bound: float
    gap_abs: float
    gap_rel: float
    x: Optional[dict]
    replay: Optional[Replay]
    certificate: Any
    stats: Stats
    provenance: Provenance
    anomalies: tuple
    internal_status: str = ""          # the frozen solver's normalized status (verified | unknown | ...)

    @property
    def solved(self) -> bool:
        return self.status in (Status.OPTIMAL, Status.INFEASIBLE)

    def to_json(self) -> dict:
        r = self.replay
        return {"schema": SCHEMA, "status": self.status.value, "termination": self.termination.value,
                "sense": "max" if self.sense == Sense.MAX else "min",
                "primal_bound": _num(self.primal_bound), "dual_bound": _num(self.dual_bound),
                "gap_abs": _num(self.gap_abs), "gap_rel": _num(self.gap_rel), "x": self.x,
                "internal_status": self.internal_status,
                "replay": None if r is None else {
                    "ok": r.ok, "feas_tol": r.feas_tol, "max_row_violation": _num(r.max_row_violation),
                    "max_bound_violation": _num(r.max_bound_violation), "worst_row": r.worst_row,
                    "max_integrality_violation": _num(r.max_integrality_violation),
                    "objective": _num(r.objective), "method": r.method},
                "anomalies": list(self.anomalies),
                "stats": _plain(self.stats), "provenance": _plain(self.provenance)}
