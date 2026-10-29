#!/usr/bin/env python3
"""Per-run evaluation: two criteria, soundness filter, local matching.

Criteria (per solver run, per instance)
  (a) strict:  AMPL-audited violation <= 1e-8 and a native global certificate
      with normalized gap <= 1e-6.  BARON/Couenne/LindoGlobal/SCIP use the paper
      analyzer's certificate parser unchanged; Gurobi/Xpress need status
      'solved' and closed solver bounds (gap <= 1e-6) consistent with the AMPL
      objective.
  (b) uniform standard: AMPL-audited violation <= 1e-6 and the solver's own final
      primal P_s and dual D_s with relative gap |P_s - D_s| / max(1,|P_s|,|D_s|) <= 1e-4 (tol_b's 1e-5 floor
      never binds, since 1e-4*max(1,...) >= 1e-4),
      and the AMPL objective within the same tolerance of P_s (the returned point
      is the one the solver priced).  AMPL status 'solved?' (the solver warns that
      the result may be wrong, e.g. BARON "Globality is therefore not guaranteed")
      is not a certificate.
A certificate is *sound* if no point from any run, audited at the same feasibility
level, beats its objective by more than the criterion's gap tolerance.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parent                                   # <repo>/baselines
RUNS = Path(os.environ.get("GEAR_BASELINE_RUNS", ROOT / "runs"))
BASE = HARNESS / "ampl"
sys.path.insert(0, str(BASE))
import analyze_solver505 as az  # noqa: E402
import run_solver505 as rs  # noqa: E402

NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
GLOBAL = ["baron", "couenne", "scip", "lindoglobal", "gurobi", "xpress"]
LOCAL = ["lgo", "knitro", "ipopt", "snopt", "minos", "conopt"]
INF = 1e19
# Final run of each solver. SCIP's final configuration (s2_obbt_pow1: s2_obbt with powers of sums kept unexpanded,
# expr/pow/expandmaxexponent = 1) was found on the full benchmark after tuning; it certifies 8 more instances and loses
# none. The replaced final run stays in the pool of points that can contradict a certificate.
FINAL_RUN = {"scip": "final_scip_pow1"}
REPLACED = [(f"final_{s}", s) for s in FINAL_RUN]


def final_run(solver: str) -> str:
    return FINAL_RUN.get(solver, f"final_{solver}")


def tol_b(p: float, d: float) -> float:
    return max(1e-5, 1e-4 * max(1.0, abs(p), abs(d)))


def tol_a(p: float, d: float) -> float:
    return 1e-6 * max(1.0, abs(p), abs(d))


def last(text, pat):
    m = re.findall(pat, text, re.M | re.I)
    return m[-1] if m else None


def solver_bounds(solver: str, text: str):
    """(P, D) as printed by the solver's own final log (None if absent)."""
    p = d = None
    if solver == "scip":
        p = last(text, rf"^\s*Primal Bound\s*:\s*({NUM})")
        d = last(text, rf"^\s*Dual Bound\s*:\s*({NUM})")
    elif solver == "couenne":
        p = last(text, rf"^\s*Upper bound:\s*({NUM})")
        d = last(text, rf"^\s*Lower bound:\s*({NUM})")
    elif solver == "baron":
        rows = re.findall(rf"^\s*\*?\s*\d+\+?\s+{NUM}\s+\S+\s+({NUM})\s+({NUM})", text, re.M)
        if rows:
            d, p = rows[-1]
        lb = last(text, rf"^\s*Lower bound is\s+({NUM})")
        ub = last(text, rf"^\s*Upper bound is\s+({NUM})")
        pre = last(text, rf"Preprocessing found feasible solution with value\s+({NUM})")
        if lb is not None:
            d = lb
        if ub is not None:
            p = ub
        elif p is None and pre is not None and "solved during preprocessing" in text:
            p = pre
        # AUDIT PATCH (baron): "Problem solved during preprocessing" prints only "Lower bound is"; the
        # "Preprocessing found feasible solution" value can be a stale incumbent (st_bpv1: 160 vs optimum 10)
        # and may be absent (schitt_s234, prob09).  The driver line "Objective X" is the returned solution value.
        if ub is None and lb is not None and "solved during preprocessing" in text \
                and re.search(r"optimal within tolerances", text):
            o = last(text, rf"^Objective\s+({NUM})\s*$")
            if o is not None:
                p = o
    elif solver == "lindoglobal":
        p = last(text, rf"^\s*Objective value\s*:\s*({NUM})")
        d = last(text, rf"^\s*Best Bound\s*:\s*({NUM})")
    elif solver == "gurobi":
        m = re.findall(rf"Best objective\s+({NUM}), best bound\s+({NUM})", text)
        if m:
            p, d = m[-1]
        elif "Solving non-convex" not in text and re.search(r"^Gurobi [\d.]+: optimal solution", text, re.M):
            # AUDIT PATCH (gurobi): presolve-solved or convex LP/QP/QCP finished by simplex/barrier, no B&B line.
            # "Optimal objective X" with no nonconvex spatial B&B is a proof of global optimality (P = D = X).
            o = last(text, rf"^Optimal objective\s+({NUM})")
            if o is not None:
                p = d = o
    elif solver == "xpress":
        p = last(text, rf"Final MI\w*P objective[^:]*:\s*({NUM})")
        d = last(text, rf"Final MINLP bound\s*:\s*({NUM})")
        # AUDIT PATCH (xpress): (i) "Final MIQP objective (global optimum)" with no bound line (st_bpv2);
        # (ii) convex LP/QP solved by simplex/barrier: no "Final MI..." block, final status "optimal solution"
        # (not "locally optimal"), "Final objective : X" => P = D = X.
        if p is not None and d is None and re.search(r"Final MI\w*P objective \(global optimum\)", text):
            d = p
        elif p is None and d is None and not re.search(r"Final MI\w*P ", text) \
                and re.search(r"^XPRESS [\d.]+ \([\d.]+\): optimal solution", text, re.M):
            o = last(text, rf"^Final objective\s*:\s*({NUM})")
            if o is not None:
                p = d = o
    f = lambda x: float(x) if x is not None else None  # noqa: E731
    p, d = f(p), f(d)
    if p is not None and abs(p) >= INF:
        p = None
    if d is not None and abs(d) >= INF:
        d = None
    return p, d


RES_RE = re.compile(r"^RESOURCE_USAGE (.*)$", re.M)


def load_manifest():
    return rs.load_instances()


def eval_run(run: str, solver: str, subset: set[str] | None = None) -> list[dict]:
    logs = RUNS / run / "logs" / solver
    rows = []
    for inst in load_manifest():
        k = f"{inst.family}/{inst.name}"
        if subset is not None and k not in subset:
            continue
        path = logs / inst.family / f"{inst.name}.log"
        r = az.parse_log(path, solver if solver in az.SOLVERS else "none")
        text = path.read_text(errors="replace").rsplit("=== ATTEMPT", 1)[-1] if path.exists() else ""
        res = RES_RE.findall(text)
        cpu = None
        if res:
            kv = dict(x.split("=", 1) for x in res[-1].split())
            try:
                cpu = float(kv["user"]) + float(kv["sys"])
            except (KeyError, ValueError):
                pass
        ps, ds = solver_bounds(solver, text) if solver in GLOBAL else (None, None)
        pa = r["objective"]
        feas8, feas6 = bool(r["feasible_1e8"]), bool(r["feasible_1e6"])
        sr = r.get("solve_result")
        # criterion (a)
        cert_a = False
        if solver in ("baron", "couenne", "scip", "lindoglobal"):
            cert_a = bool(r["native_global_certificate"])
        elif solver in ("gurobi", "xpress") and feas8 and sr == "solved" \
                and ps is not None and ds is not None and pa is not None:
            cert_a = (abs(ps - ds) <= tol_a(ps, ds) and abs(pa - ps) <= tol_a(ps, ds))
        # criterion (b)
        cert_b = False
        if solver in GLOBAL and feas6 and ps is not None and ds is not None \
                and pa is not None and sr not in ("infeasible", "unbounded", "solved?"):
            t = tol_b(ps, ds)
            cert_b = abs(ps - ds) <= t and abs(pa - ps) <= t
        rows.append({
            "run": run, "solver": solver, "family": inst.family,
            "instance": inst.name, "sense": inst.sense, "nvars": inst.nvars,
            "ncons": inst.ncons, "runner_status": r["runner_status"],
            "solve_result": sr, "wall_time": r.get("wall_time"),
            "P_ampl": pa, "max_violation": r["max_violation"],
            "feas8": feas8, "feas6": feas6, "P_solver": ps, "D_solver": ds,
            "cert_a_raw": cert_a, "cert_b_raw": cert_b, "cpu_s": cpu,
        })
    return rows


# --------------------------------------------------------------- aggregation
def sign(sense: str) -> float:
    return 1.0 if sense.startswith("min") else -1.0


def best_known(rows_all: list[dict]) -> dict[str, dict[str, float]]:
    """Best objective over all runs, per feasibility level, per instance."""
    best = {"a": {}, "b": {}}
    for r in rows_all:
        if r["P_ampl"] is None:
            continue
        k = f"{r['family']}/{r['instance']}"
        s = sign(r["sense"])
        for lvl, ok in (("a", r["feas8"]), ("b", r["feas6"])):
            if ok:
                cur = best[lvl].get(k)
                if cur is None or s * r["P_ampl"] < s * cur:
                    best[lvl][k] = r["P_ampl"]
    return best


def mark_sound(rows: list[dict], best: dict) -> None:
    for r in rows:
        k = f"{r['family']}/{r['instance']}"
        s = sign(r["sense"])
        for lvl, tolf in (("a", tol_a), ("b", tol_b)):
            raw = r[f"cert_{lvl}_raw"]
            ok = raw
            if raw and r["P_ampl"] is not None:
                b = best[lvl].get(k)
                if b is not None and s * b < s * r["P_ampl"] - tolf(r["P_ampl"], b):
                    ok = False
            r[f"cert_{lvl}"] = ok
            r[f"contra_{lvl}"] = raw and not ok


def credit_infeasible(final: dict, pool: list[dict], min_solvers: int = 2, limit: float = 180.0) -> set:
    """Rule (b): an instance is confirmed infeasible when no run of any solver (global or local, final or
    tuning) returned a point feasible at 1e-6 and at least `min_solvers` global solvers report AMPL status
    'infeasible' within the time limit.  Those global solvers' verdicts count as solved (cert_b);
    local solvers get no credit.  Returns the set of (family, instance) keys."""
    feasible = {(r["family"], r["instance"]) for r in pool if r["feas6"]}
    said = {}
    for s in GLOBAL:
        for r in final.get(s, []):
            if r["solve_result"] == "infeasible" and r["wall_time"] is not None and float(r["wall_time"]) <= limit:
                said.setdefault((r["family"], r["instance"]), []).append(r)
    keys = {k for k, v in said.items() if k not in feasible and len(v) >= min_solvers}
    for k in keys:
        for r in said[k]:
            r["cert_b"] = r["cert_b_raw"] = True
            r["infeasible_cert"] = True
    return keys


def summarize(rows: list[dict]) -> dict:
    t = [float(r["wall_time"]) for r in rows if r["cert_a"] and r["wall_time"]]
    return {
        "n": len(rows),
        "a_raw": sum(r["cert_a_raw"] for r in rows),
        "a_sound": sum(r["cert_a"] for r in rows),
        "b_raw": sum(r["cert_b_raw"] for r in rows),
        "b_sound": sum(r["cert_b"] for r in rows),
        "contra_a": sum(r["contra_a"] for r in rows),
        "contra_b": sum(r["contra_b"] for r in rows),
        "feas8": sum(r["feas8"] for r in rows),
        "feas6": sum(r["feas6"] for r in rows),
        "median_wall_a": sorted(t)[len(t) // 2] if t else None,
        "total_wall": sum(float(r["wall_time"] or 0) for r in rows),
    }


def score_run(run: str, solver: str, subset: set[str] | None, extra_rows=()):
    rows = eval_run(run, solver, subset)
    best = best_known(list(extra_rows) + rows)
    mark_sound(rows, best)
    return rows, summarize(rows)


if __name__ == "__main__":
    # usage: bl_eval.py score <solver> <run> [subset-file]
    cmd, solver, run = sys.argv[1:4]
    subset = None
    if len(sys.argv) > 4:
        subset = {x.strip() for x in Path(sys.argv[4]).read_text().splitlines() if x.strip()}
    rows, s = score_run(run, solver, subset)
    print(json.dumps(s))
