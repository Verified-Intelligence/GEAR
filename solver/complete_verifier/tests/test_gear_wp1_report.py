"""Result sink and status mapping (G5 WP1, DESIGN section 6). CPU only."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gear import report
from gear.report import ReplayRecord, ResultRecord, Status, Termination, map_result

OK = ReplayRecord(True, 1e-6, -1.0, -1.0)
BAD = ReplayRecord(False, 1e-6, 5.0, -1.0)
RULE = dict(gap_abs=1e-5, gap_rel=1e-4)


def m(internal, p, d, **kw):
    kw.setdefault("sense", "min")
    kw.setdefault("replay", OK if p is not None and math.isfinite(p) else None)
    kw.setdefault("x", {"X_0": 1.0} if kw["replay"] is not None else None)
    return map_result(internal, p, d, **RULE, **{k: v for k, v in kw.items()})


def test_optimal_gap_closed():
    r = m("verified", 1.0, 1.0 - 5e-6)
    assert (r.status, r.termination) == (Status.OPTIMAL, Termination.GAP_CLOSED) and not r.anomalies


def test_relative_gap_rule():
    r = m("verified", 1e6, 1e6 - 50)          # 50 <= 1e-4 * 1e6
    assert r.status is Status.OPTIMAL


def test_dual_above_primal_is_feasible_with_anomaly():
    r = m("verified", 1.0, 2.0)
    assert r.status is Status.FEASIBLE and r.anomalies == ("dual_above_primal",)


def test_verified_but_gap_open_is_feasible():
    r = m("verified", 10.0, 1.0)
    assert r.status is Status.FEASIBLE and "verified_gap_open" in r.anomalies


@pytest.mark.parametrize("sense,inf", [("min", math.inf), ("max", -math.inf)])
def test_replay_failure_drops_incumbent_and_dual(sense, inf):
    r = m("verified", 1.0, 1.0, sense=sense, replay=BAD, x={"X_0": 1.0})
    assert r.status is Status.NO_SOLUTION and r.x is None
    assert r.primal_bound == inf and r.dual_bound == -inf
    assert "replay_failed_incumbent_dropped" in r.anomalies


def test_max_sign():
    r = m("verified", 5.0, 5.000001, sense="max")     # internal -5 vs -5.000001: closed
    assert r.status is Status.OPTIMAL
    r = m("verified", 5.0, 4.0, sense="max")          # D below P in user sense is D > P internally
    assert r.status is Status.FEASIBLE and r.anomalies == ("dual_above_primal",)
    r = m("unknown", None, None, sense="max", events=("time_limit",))
    assert r.primal_bound == -math.inf and r.dual_bound == math.inf


def test_infeasible():
    r = m("verified", None, None, incumbent_ever=False)
    assert (r.status, r.termination) == (Status.INFEASIBLE, Termination.PROVED_INFEASIBLE)
    assert r.primal_bound == math.inf and r.x is None


@pytest.mark.parametrize("internal,status", [("unbounded", Status.UNBOUNDED), ("unbounded_tau", Status.UNBOUNDED_TAU)])
@pytest.mark.parametrize("sense,inf", [("min", -math.inf), ("max", math.inf)])
def test_unbounded_rows(internal, status, sense, inf):
    r = m(internal, None, None, sense=sense)
    assert (r.status, r.termination) == (status, Termination.RAY_CERTIFIED)
    assert r.primal_bound == inf == r.dual_bound and r.gap_abs == 0.0


@pytest.mark.parametrize("event,term", [("time_limit", Termination.TIME_LIMIT), ("memory_limit", Termination.MEMORY_LIMIT),
                                        ("domain_error", Termination.DOMAIN_ERROR),
                                        ("interrupted", Termination.INTERRUPTED)])
def test_unknown_rows(event, term):
    r = m("unknown", 3.0, 1.0, events=(event,))
    assert (r.status, r.termination) == (Status.FEASIBLE, term)
    r = m("unknown", None, 1.0, events=(event,))
    assert (r.status, r.termination) == (Status.NO_SOLUTION, term) and r.x is None


def test_unknown_with_failed_replay_is_no_solution():
    r = m("unknown", 3.0, 1.0, replay=BAD, events=("time_limit",))
    assert r.status is Status.NO_SOLUTION


def test_unknown_without_event_is_flagged():
    assert "unknown_without_reason" in m("unknown", None, None).anomalies


def test_error_keeps_dual_and_replayed_incumbent():
    r = m("error", 2.0, 1.0, events=("internal_error",))
    assert (r.status, r.termination) == (Status.ERROR, Termination.INTERNAL_ERROR)
    assert r.dual_bound == 1.0 and r.x == {"X_0": 1.0}
    r = m("error", 2.0, 1.0, replay=BAD)
    assert r.x is None and r.dual_bound == 1.0


def test_json_round_trip_with_infinities():
    for rec in (m("verified", 1.0, 1.0), m("unknown", None, None, events=("time_limit",)),
                m("unbounded", None, None, sense="max")):
        text = rec.dumps()
        back = ResultRecord.from_json(json.loads(text))
        assert back == rec
        assert json.loads(text)["schema"] == report.SCHEMA


def test_json_schema_version_checked():
    d = m("verified", 1.0, 1.0).to_json()
    d["schema"] = 99
    with pytest.raises(ValueError):
        ResultRecord.from_json(d)


def test_hooks_are_noops_without_a_sink():
    report.uninstall()
    report.record_pd(torch.tensor([1.0]), 2.0)
    report.record_event("time_limit")
    report.record_best_primal(1.0, torch.zeros(2))


def test_sink_records_and_rejects_unknown_event():
    sink = report.install()
    try:
        report.record_pd(torch.tensor([[1.5]]), torch.tensor(0.5))
        report.record_best_primal(1.5, torch.tensor([1.0, 2.0]))
        report.record_event("time_limit")
        assert (sink.primal, sink.dual, sink.objective, sink.x, sink.events) == (1.5, 0.5, 1.5, [1.0, 2.0], ["time_limit"])
        with pytest.raises(ValueError):
            report.record_event("bogus")
    finally:
        report.uninstall()


def test_solver_export_site_feeds_the_sink(capsys):
    from input_split.batch_branch_and_bound import export_best_primal
    sink = report.install()
    try:
        net = SimpleNamespace()
        export_best_primal(net, None, torch.tensor([1.0, 2.0]), torch.tensor([3.0]), True)
    finally:
        report.uninstall()
    assert sink.objective == 3.0 and sink.x == [1.0, 2.0]
    assert "[best-primal] objective=3.0 x=[1.0, 2.0]" in capsys.readouterr().out


def test_replay_float64_matches_the_analyze_f6_predicate():
    y = [0.0, -1.0, 1e-7]                 # rows y[1:] <= 0 within tol
    r = report.replay_float64(y, np.array([[1.0, 1.0]]), np.array([2.0]), [1.0, 1.0], [0.0, 0.0], [2.0, math.inf], 1e-6)
    assert r.ok and r.max_row_violation <= 1e-6
    r = report.replay_float64(y, None, None, [3.0, 1.0], [0.0, 0.0], [2.0, 2.0], 1e-6)
    assert not r.ok and r.max_bound_violation == 1.0
    r = report.replay_float64([0.0, 1e-3], None, None, [1.0], [0.0], [2.0], 1e-6)
    assert not r.ok
    r = report.replay_float64([0.0, math.nan], None, None, [1.0], [0.0], [2.0], 1e-6)
    assert not r.ok
