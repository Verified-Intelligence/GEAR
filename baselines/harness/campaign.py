#!/usr/bin/env python3
"""Unattended, resumable campaign: tune each global solver on the 40-instance
subset, pick one configuration, then run all 505 instances at 180 s.

Every solve runs inside gpu_queue.py waves (see bl.py).  Re-running this script
resumes: finished solves are skipped by the harness hash, finished tuning runs
are re-scored (cheap), and a chosen configuration is frozen in results/chosen.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bl_eval as ev  # noqa: E402

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parent                                   # <repo>/baselines
PY = sys.executable
CFGS = ROOT / "configs"
SUBSET = CFGS / "subset40.txt"
RES = ROOT / "results"
CHOSEN = CFGS / "chosen"
LOG = RES / "tuning_log.csv"
TUNE_T = {"baron": 180, "couenne": 180, "lindoglobal": 180,
          "scip": 60, "gurobi": 60, "xpress": 60}
LAYOUT_SOLVERS = {"scip", "gurobi", "xpress"}           # native-parallel
GLOBAL_ORDER = ["baron", "couenne", "lindoglobal", "gurobi", "xpress", "scip"]
LOCAL_ORDER = ["ipopt", "snopt", "minos", "conopt", "lgo", "knitro"]
FINAL_T = 180


def say(*a):
    print(dt.datetime.now().strftime("%m-%d %H:%M:%S"), *a, flush=True)


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def drive(cfg_path: Path, run: str, T: float, subset: Path | None):
    cmd = [PY, str(HARNESS / "bl.py"), "drive", "--config", str(cfg_path),
           "--run", run, "--time-limit", str(T)]
    if subset:
        cmd += ["--subset", str(subset)]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t0


def subset_keys():
    return {x.strip() for x in SUBSET.read_text().splitlines()
            if x.strip() and not x.startswith("#")}


def run_and_score(cfg: dict, cfg_path: Path, run: str, T: float, tune: bool):
    sub = SUBSET if tune else None
    elapsed = drive(cfg_path, run, T, sub)
    rows, s = ev.score_run(run, cfg["solver"], subset_keys() if tune else None)
    s.update(solver=cfg["solver"], config=cfg["name"], run=run,
             layout=f"{cfg['jobs']}x{cfg['threads']}", T=T, tune=int(tune),
             elapsed_s=round(elapsed), stamp=dt.datetime.now().isoformat(timespec="seconds"))
    new = not LOG.exists()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", newline="") as fh:
        w = csv.DictWriter(fh, list(s))
        if new:
            w.writeheader()
        w.writerow(s)
    say("scored", json.dumps(s))
    return rows, s


def key_global(s: dict):
    return (s["b_sound"], s["a_sound"], -s["total_wall"])


def local_match_count(rows, best) -> int:
    n = 0
    for r in rows:
        k = f"{r['family']}/{r['instance']}"
        b = best["b"].get(k)
        if r["feas6"] and r["P_ampl"] is not None and b is not None \
                and abs(r["P_ampl"] - b) <= ev.tol_b(r["P_ampl"], b):
            n += 1
    return n


def derive_layouts(base: dict, d: Path) -> list[Path]:
    out = []
    for jobs, thr in ((2, 8), (4, 4)):
        c = dict(base, name=f"{base['name']}_L{jobs}x{thr}", jobs=jobs, threads=thr,
                 stage="layout",
                 env={k: str(thr) for k in base["env"]},
                 notes=base["notes"] + f" | layout {jobs} jobs x {thr} threads")
        p = d / f"{c['name']}.json"
        p.write_text(json.dumps(c, indent=2) + "\n")
        out.append(p)
    return out


def tune_global(solver: str) -> dict:
    CHOSEN.mkdir(parents=True, exist_ok=True)
    frozen = CHOSEN / f"{solver}.json"
    if frozen.exists():
        return load(frozen)
    T = TUNE_T[solver]
    cdir = CFGS / "candidates" / solver
    cfgs = [(p, load(p)) for p in sorted(cdir.glob("*.json"))]
    stage1 = [(p, c) for p, c in cfgs if c.get("stage") == "settings" and c.get("select", True)]
    results = []
    for p, c in stage1:
        _, s = run_and_score(c, p, f"tune_{c['name']}", T, True)
        results.append((key_global(s), p, c, s))
    best = max(results, key=lambda x: x[0])
    say(f"[{solver}] stage-1 winner {best[2]['name']} {best[3]['b_sound']} (strict extra {best[3]['a_sound']})")
    if solver in LAYOUT_SOLVERS:
        lay = []
        for p in derive_layouts(best[2], cdir):
            c = load(p)
            _, s = run_and_score(c, p, f"tune_{c['name']}", T, True)
            lay.append((key_global(s), p, c, s))
        results += lay
        best = max(results, key=lambda x: x[0])
    for p, c in cfgs:                                   # evidence-only configs
        if c.get("stage") == "recheck":
            run_and_score(c, p, f"tune_{c['name']}", T, True)
    chosen = dict(best[2], tuned={"subset": "subset40", "T": T,
                                  "score": best[3],
                                  "all": [{"config": r[2]["name"], **{k: r[3][k] for k in (
                                      "a_sound", "b_sound", "a_raw", "b_raw", "contra_a",
                                      "contra_b", "total_wall")}} for r in results]})
    frozen.write_text(json.dumps(chosen, indent=2) + "\n")
    say(f"[{solver}] CHOSEN {chosen['name']} layout={chosen['jobs']}x{chosen['threads']}")
    return chosen


def final_run(solver: str, cfg: dict):
    CHOSEN.mkdir(parents=True, exist_ok=True)
    p = CHOSEN / f"{solver}_cfg.json"
    p.write_text(json.dumps({k: v for k, v in cfg.items() if k != "tuned"}, indent=2) + "\n")
    run_and_score(load(p), p, f"final_{solver}", FINAL_T, False)


def pick_knitro() -> dict:
    frozen = CHOSEN / "knitro.json"
    if frozen.exists():
        return load(frozen)
    cdir = CFGS / "candidates" / "knitro"
    best = None
    for n in ("knitro_final", "knitro_ms16"):
        p = cdir / f"{n}.json"
        c = load(p)
        rows, s = run_and_score(c, p, f"tune_{n}", 60, True)
        bk = ev.best_known(rows)
        m = local_match_count(rows, bk)
        say(f"[knitro] {n} matches best-known (1e-6 feas, gap rule): {m}")
        key = (m, -s["total_wall"] if n == "knitro_ms16" else 1e9)
        # serial 16-job layout wins ties (equal quality, shorter total wall)
        if best is None or m > best[0] or (m == best[0] and n == "knitro_final"):
            best = (m, c, s)
    chosen = dict(best[1], tuned={"subset": "subset40", "T": 60, "matches": best[0]})
    frozen.write_text(json.dumps(chosen, indent=2) + "\n")
    return chosen


def main():
    only = sys.argv[1:]
    for solver in LOCAL_ORDER:
        if only and solver not in only:
            continue
        if solver == "knitro":
            cfg = pick_knitro()
        else:
            cfg = load(next((CFGS / "candidates" / solver).glob("*.json")))
        final_run(solver, cfg)
    for solver in GLOBAL_ORDER:
        if only and solver not in only:
            continue
        cfg = tune_global(solver)
        final_run(solver, cfg)
    say("CAMPAIGN COMPLETE")


if __name__ == "__main__":
    main()
