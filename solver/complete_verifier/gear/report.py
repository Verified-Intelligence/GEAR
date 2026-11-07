"""Result sink of the frozen solver (G5 WP1, DESIGN section 6).

The solver's report sites call ``record_pd``, ``record_best_primal`` and ``record_event`` after the lines they
print today. With no sink installed these are no-ops, so logs and verdicts do not change. A CLI or facade installs a
``ResultSink``, runs the solve, and turns what the sink saw plus the solver's internal status into a ``ResultRecord``
with ``map_result`` (the status mapping of section 6). Values reach the sink in the user's sense (a maximization
problem reports its own P and D, as the printed ``Primal Solution`` line does).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

SCHEMA = 1


class Status(Enum):
    OPTIMAL = "OPTIMAL"
    INFEASIBLE = "INFEASIBLE"
    FEASIBLE = "FEASIBLE"
    NO_SOLUTION = "NO_SOLUTION"
    ERROR = "ERROR"
    UNBOUNDED = "UNBOUNDED"
    UNBOUNDED_TAU = "UNBOUNDED_TAU"


class Termination(Enum):
    GAP_CLOSED = "GAP_CLOSED"
    PROVED_INFEASIBLE = "PROVED_INFEASIBLE"
    RAY_CERTIFIED = "RAY_CERTIFIED"
    TIME_LIMIT = "TIME_LIMIT"
    MEMORY_LIMIT = "MEMORY_LIMIT"
    DOMAIN_ERROR = "DOMAIN_ERROR"
    INTERRUPTED = "INTERRUPTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# Sink events and the termination each one means for a solve that did not close.
EVENT_TERMINATION = {
    "time_limit": Termination.TIME_LIMIT,
    "memory_limit": Termination.MEMORY_LIMIT,
    "domain_error": Termination.DOMAIN_ERROR,
    "interrupted": Termination.INTERRUPTED,
    "internal_error": Termination.INTERNAL_ERROR,
}


class ResultSink:
    def __init__(self):
        self.primal: Optional[float] = None      # last printed P, user sense
        self.dual: Optional[float] = None        # last printed D, user sense
        self.objective: Optional[float] = None   # objective of the last exported incumbent
        self.x: Optional[list] = None            # that incumbent
        self.events: list[str] = []
        self.ops: list[str] = []                 # the [Op] ledger lines


_sink: Optional[ResultSink] = None


def install(sink: Optional[ResultSink] = None) -> ResultSink:
    global _sink
    _sink = sink if sink is not None else ResultSink()
    return _sink


def uninstall() -> None:
    global _sink
    _sink = None


def _scalar(v) -> float:
    return float(v.reshape(-1)[0]) if hasattr(v, "reshape") else float(v)


def record_pd(primal, dual) -> None:
    """The ``Primal Solution P, Dual Solution D`` line, values in the user's sense (tensors or floats)."""
    if _sink is not None:
        _sink.primal, _sink.dual = _scalar(primal), _scalar(dual)


def record_best_primal(objective, x) -> None:
    """The ``[best-primal]`` line: the incumbent's objective and coordinates."""
    if _sink is not None:
        _sink.objective = float(objective)
        _sink.x = [float(v) for v in x.tolist()] if hasattr(x, "tolist") else [float(v) for v in x]


def record_op(line: str) -> None:
    """An ``[Op]`` ledger line, as printed."""
    if _sink is not None:
        _sink.ops.append(line)


def record_event(name: str) -> None:
    if _sink is not None:
        if name not in EVENT_TERMINATION:
            raise ValueError(f"unknown sink event {name!r}")
        _sink.events.append(name)


@dataclass(frozen=True)
class ReplayRecord:
    ok: bool
    feas_tol: float
    max_row_violation: float
    max_bound_violation: float
    worst_row: Optional[str] = None


def replay_float64(y, A, b, x, lo, hi, feas_tol: float) -> ReplayRecord:
    """Replay predicate 'float64 + margin' (section 6) for a legacy artifact, as ``analyze_f6.feasible_at``.

    y: model output at x (row 0 is the objective, rows i >= 1 must be <= 0); A, b: dense linear rows A x <= b or None;
    lo, hi: box (``inf`` entries are skipped). Every value is reduced by 1e-12 * max(1, |scale|); ok iff all are finite
    and <= feas_tol.
    """
    import numpy as np
    y, x = np.asarray(y, dtype=np.float64).reshape(-1), np.asarray(x, dtype=np.float64).reshape(-1)
    rows, worst = [], None
    if y.size > 1:
        r = y[1:] - 1e-12 * np.maximum(np.abs(y[1:]), 1.0)
        rows.append(r)
    if A is not None:
        A = np.asarray(A, dtype=np.float64)
        rows.append(A @ x - np.asarray(b, dtype=np.float64).reshape(-1)
                    - 1e-12 * np.maximum(np.abs(A) @ np.abs(x), 1.0))
    row_viol = np.concatenate(rows) if rows else np.zeros(0)
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    bound_viol = np.concatenate([(lo - x)[np.isfinite(lo)], (x - hi)[np.isfinite(hi)]])
    max_row = float(row_viol.max()) if row_viol.size else -math.inf
    max_bound = float(bound_viol.max()) if bound_viol.size else -math.inf
    finite = bool(np.isfinite(y).all() and np.isfinite(row_viol).all() and np.isfinite(x).all())
    if row_viol.size and finite:
        worst = f"row {int(row_viol.argmax())}"
    ok = finite and max_row <= feas_tol and max_bound <= feas_tol
    return ReplayRecord(ok, feas_tol, max_row, max_bound, worst)


@dataclass(frozen=True)
class ResultRecord:
    status: Status
    termination: Termination
    sense: str                       # "min" | "max"
    primal_bound: float              # user sense; +inf (min) / -inf (max) with no incumbent
    dual_bound: float                # user sense; -inf (min) / +inf (max) if none
    gap_abs: float
    gap_rel: float
    x: Optional[dict]                # {"X_i": value}; None without an incumbent
    internal_status: str
    replay: Optional[ReplayRecord]
    anomalies: tuple = ()

    def to_json(self) -> dict:
        def num(v):  # JSON has no inf/nan literals
            return v if math.isfinite(v) else repr(float(v))
        return {"schema": SCHEMA, "status": self.status.value, "termination": self.termination.value,
                "sense": self.sense, "primal_bound": num(self.primal_bound), "dual_bound": num(self.dual_bound),
                "gap_abs": num(self.gap_abs), "gap_rel": num(self.gap_rel), "x": self.x,
                "internal_status": self.internal_status,
                "replay": None if self.replay is None else {
                    "ok": self.replay.ok, "feas_tol": self.replay.feas_tol,
                    "max_row_violation": num(self.replay.max_row_violation),
                    "max_bound_violation": num(self.replay.max_bound_violation), "worst_row": self.replay.worst_row},
                "anomalies": list(self.anomalies)}

    @staticmethod
    def from_json(d: dict) -> "ResultRecord":
        if d.get("schema") != SCHEMA:
            raise ValueError(f"unsupported result schema {d.get('schema')!r}")
        rp = d["replay"]
        return ResultRecord(
            Status(d["status"]), Termination(d["termination"]), d["sense"], float(d["primal_bound"]),
            float(d["dual_bound"]), float(d["gap_abs"]), float(d["gap_rel"]), d["x"], d["internal_status"],
            None if rp is None else ReplayRecord(rp["ok"], rp["feas_tol"], float(rp["max_row_violation"]),
                                                  float(rp["max_bound_violation"]), rp["worst_row"]),
            tuple(d["anomalies"]))

    def dumps(self) -> str:
        return json.dumps(self.to_json(), sort_keys=True, indent=1) + "\n"


def _gap(p: float, d: float) -> tuple[float, float]:
    if math.isinf(p) or math.isinf(d):
        return (0.0, 0.0) if p == d else (math.inf, math.inf)
    return abs(p - d), abs(p - d) / max(1.0, abs(p), abs(d))


def map_result(internal_status: str, primal: Optional[float], dual: Optional[float], *, sense: str,
               replay: Optional[ReplayRecord], x: Optional[dict], gap_abs: float, gap_rel: float,
               events: tuple = (), incumbent_ever: bool = True) -> ResultRecord:
    """The status table of section 6. primal and dual are in the user's sense (None: never printed).

    internal_status: ``verified`` | ``unknown`` | ``unbounded`` | ``unbounded_tau`` | ``error``.
    replay: the incumbent's replay, None when there is no incumbent. incumbent_ever: False iff no incumbent was ever
    found (needed for INFEASIBLE). gap_abs / gap_rel: the resolved gap rule (closed iff P-D <= abs or rel).
    """
    if sense not in ("min", "max"):
        raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")
    worst_p = math.inf if sense == "min" else -math.inf        # P with no incumbent
    worst_d = -math.inf if sense == "min" else math.inf        # D with no bound
    sgn = 1.0 if sense == "min" else -1.0                       # internal minimization value = sgn * user value
    p = worst_p if primal is None else float(primal)
    d = worst_d if dual is None else float(dual)
    anomalies: list[str] = []
    term_event = next((EVENT_TERMINATION[e] for e in events if e in EVENT_TERMINATION), None)

    def out(status, term, p_=None, d_=None, x_=x, replay_=replay):
        p_, d_ = p if p_ is None else p_, d if d_ is None else d_
        ga, gr = _gap(p_, d_)
        return ResultRecord(status, term, sense, p_, d_, ga, gr, x_, internal_status, replay_, tuple(anomalies))

    if internal_status in ("unbounded", "unbounded_tau"):
        ray = -math.inf if sense == "min" else math.inf       # P = D = -inf (min) / +inf (max)
        status = Status.UNBOUNDED if internal_status == "unbounded" else Status.UNBOUNDED_TAU
        return out(status, Termination.RAY_CERTIFIED, ray, ray, None)
    if internal_status == "error":
        keep_x = x if (replay is not None and replay.ok) else None
        return out(Status.ERROR, Termination.INTERNAL_ERROR, p if keep_x is not None else worst_p, d, keep_x,
                   replay if keep_x is not None else None)
    has_incumbent = math.isfinite(p) and replay is not None and replay.ok
    if internal_status == "verified":
        if math.isfinite(p):
            if replay is None or not replay.ok:
                anomalies.append("replay_failed_incumbent_dropped")
                return out(Status.NO_SOLUTION, term_event or Termination.INTERNAL_ERROR, worst_p, worst_d, None)
            ip, id_ = sgn * p, sgn * d                          # internal minimization values
            gap = ip - id_
            closed = gap <= gap_abs or gap / max(1.0, abs(ip), abs(id_)) <= gap_rel
            if id_ > ip + gap_abs:
                anomalies.append("dual_above_primal")
                return out(Status.FEASIBLE, term_event or Termination.INTERNAL_ERROR)
            if closed:
                return out(Status.OPTIMAL, Termination.GAP_CLOSED)
            anomalies.append("verified_gap_open")
            return out(Status.FEASIBLE, term_event or Termination.INTERNAL_ERROR)
        if incumbent_ever:
            anomalies.append("infeasible_claim_with_incumbent")
            return out(Status.FEASIBLE if has_incumbent else Status.NO_SOLUTION, term_event or Termination.INTERNAL_ERROR)
        return out(Status.INFEASIBLE, Termination.PROVED_INFEASIBLE, worst_p, worst_p)
    if internal_status != "unknown":
        raise ValueError(f"unknown internal status {internal_status!r}")
    if term_event is None:
        anomalies.append("unknown_without_reason")
    term = term_event or Termination.INTERNAL_ERROR
    if has_incumbent:
        return out(Status.FEASIBLE, term)
    return out(Status.NO_SOLUTION, term, worst_p, d, None, replay if replay is not None else None)
