#!/usr/bin/env python3
"""Baseline campaign harness (16 logical CPUs, 180 s, waves, optionally under an external lock).

Reuses harness/ampl/run_solver505.py unchanged for the
AMPL program, audit, watchdog, retry policy, hashing and append-only logs.  A
configuration (configs/<solver>/<name>.json) only changes solver options, a
native settings file, thread/job layout and process environment.

Sub-commands
  wave   run one lock-held wave (optionally called through the GEAR_BASELINE_LOCK wrapper; <= ~200 s)
  drive  loop waves until the (config, instance set, time limit) is complete
  smoke  <= 10 s single-process option check (outside the lock, by rule)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parent                                   # <repo>/baselines
RUNS = Path(os.environ.get("GEAR_BASELINE_RUNS", ROOT / "runs"))
BASE = HARNESS / "ampl"
# optional wave lock: path to a wrapper taking `--priority N --tag T -- cmd...`
GPU_QUEUE = os.environ.get("GEAR_BASELINE_LOCK")
PY = sys.executable
sys.path.insert(0, str(BASE))
import run_solver505 as rs  # noqa: E402

TIME_FMT = ("RESOURCE_USAGE wall=%e user=%U sys=%S maxrss_kb=%M "
            "cpu_pct=%P exit=%x")

# --- solvers that the paper harness does not know -------------------------
for _n in ("gurobi", "xpress", "conopt"):
    rs.SOLVERS[_n] = rs.SolverSpec(
        name=_n, version=f"{_n} (AMPL module)", executable=_n, option_name=_n,
        options_template="", default_jobs=16, native_parallel=False)

_cfg_env: dict[str, str] = {}


class _TimedSubprocess:
    """Stand-in for the runner's `subprocess`: config env + /usr/bin/time."""

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def Popen(self, args, **kw):  # noqa: N802
        env = kw.get("env")
        if env is not None and _cfg_env:
            kw["env"] = dict(env, **_cfg_env)
        return subprocess.Popen(["/usr/bin/time", "-f", TIME_FMT] + list(args),
                                **kw)


rs.subprocess = _TimedSubprocess()

# the full AMPL-for-Academics licenses lift the BARON/Knitro demo size limits
for _n in ("baron", "knitro"):
    rs.SOLVERS[_n] = dataclasses.replace(rs.SOLVERS[_n], demo_size_limit=False)


# --- configuration ----------------------------------------------------------
def load_config(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    cfg.setdefault("name", Path(path).stem)
    cfg.setdefault("threads", 1)
    cfg.setdefault("jobs", 16)
    cfg.setdefault("native", "")
    cfg.setdefault("pfx", {})
    cfg.setdefault("env", {})
    cfg.setdefault("couenne_rule", "paper")
    cfg.setdefault("notes", "")
    if cfg["threads"] * cfg["jobs"] > 16:
        raise SystemExit(f"{cfg['name']}: jobs*threads > 16")
    return cfg


def install(cfg: dict, run_dir: Path):
    solver = cfg["solver"]
    cdir = run_dir / "config"
    cdir.mkdir(parents=True, exist_ok=True)
    native = cfg["native"]
    if cfg["pfx"]:
        pdir = cdir / "pfx"
        pdir.mkdir(exist_ok=True)
        for fname, text in cfg["pfx"].items():
            (pdir / fname).write_text(text)
        native += f'concurrent/paramsetprefix = "{pdir}/"\n'
    (cdir / "native.txt").write_text(native)
    (cdir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    opts = cfg["options"]
    if native.strip():
        if solver == "scip":
            opts += f" tech:param:read={cdir / 'native.txt'}"
        elif solver == "baron":
            opts += f" optfile={cdir / 'native.txt'}"
        elif solver != "couenne":
            raise SystemExit(f"{solver}: no native settings channel")
    base = rs.SOLVERS[solver]
    spec = dataclasses.replace(
        base, version=f"{base.version}; cfg={cfg['name']}:{digest}",
        options_template=opts, native_parallel=cfg["threads"] > 1,
        default_jobs=cfg["jobs"])
    if solver == "couenne":
        paper = rs.couenne_option_text

        def couenne_text(inst, tl, _p=paper, _c=cfg):
            text = _p(inst, tl) if _c["couenne_rule"] == "paper" \
                else f"time_limit {tl:g}\n"
            return text + _c["native"]
        rs.couenne_option_text = couenne_text
    return spec, digest


# --- instances and scheduling ------------------------------------------------
def key(i) -> str:
    return f"{i.family}/{i.name}"


def read_subset(path) -> set[str] | None:
    if not path:
        return None
    return {x.strip() for x in Path(path).read_text().splitlines()
            if x.strip() and not x.startswith("#")}


def pending(spec, insts, run_dir, tl, threads, jobs):
    out = []
    for inst in insts:
        digest = rs.config_hash(spec, inst, tl, threads, jobs)
        lp = run_dir / "logs" / spec.name / inst.family / f"{inst.name}.log"
        if not rs.terminal_for_hash(lp, digest):
            out.append(inst)
    return out


def select(args) -> list:
    wanted = read_subset(args.subset)
    return [i for i in rs.load_instances() if wanted is None or key(i) in wanted]


def cmd_wave(args) -> None:
    global _cfg_env
    cfg = load_config(args.config)
    run_dir = RUNS / args.run
    spec, _ = install(cfg, run_dir)
    _cfg_env = {k: str(v) for k, v in cfg["env"].items()}
    ampl, _, env = rs.get_module_environment(rs.DEFAULT_PREFIX)
    todo = pending(spec, select(args), run_dir, args.time_limit,
                   cfg["threads"], cfg["jobs"])
    window = max(0.0, args.budget - args.time_limit - 5.0)
    lock, t0, done = threading.Lock(), time.time(), []

    def worker():
        first = True
        while True:
            with lock:
                if not todo or (not first and time.time() - t0 > window):
                    return
                inst = todo.pop(0)
            first = False
            rs.run_one(inst, spec, ampl, env, run_dir, args.time_limit,
                       cfg["threads"], cfg["jobs"], False)
            done.append(key(inst))
            # scratch is only needed while the solver runs
            subprocess.run(["rm", "-rf", str(run_dir / "work" / spec.name
                                              / inst.family / inst.name)])

    with concurrent.futures.ThreadPoolExecutor(cfg["jobs"]) as pool:
        for f in [pool.submit(worker) for _ in range(cfg["jobs"])]:
            f.result()
    print(f"[wave {args.run}] done={len(done)} left={len(todo)} "
          f"{time.time() - t0:.0f}s", flush=True)


def cmd_drive(args) -> None:
    cfg = load_config(args.config)
    run_dir = RUNS / args.run
    spec, _ = install(cfg, run_dir)
    insts = select(args)
    zero = 0
    while True:
        left = pending(spec, insts, run_dir, args.time_limit,
                       cfg["threads"], cfg["jobs"])
        if not left:
            break
        print(f"[drive {args.run}] pending={len(left)}", flush=True)
        lock = ([PY, GPU_QUEUE, "--priority", "1", "--tag", f"base_{cfg['solver']}",
                 "--"] if GPU_QUEUE else [])
        cmd = lock + [PY, str(Path(__file__).resolve()), "wave",
               "--config", str(args.config), "--run", args.run,
               "--time-limit", str(args.time_limit), "--budget", str(args.budget)]
        if args.subset:
            cmd += ["--subset", str(args.subset)]
        subprocess.run(cmd)
        now = pending(spec, insts, run_dir, args.time_limit,
                      cfg["threads"], cfg["jobs"])
        zero = zero + 1 if len(now) >= len(left) else 0
        if zero >= 3:
            raise SystemExit(f"{args.run}: no progress in 3 waves")


def cmd_smoke(args) -> None:
    """<= 10 s, one process, one instance; for option validation only."""
    global _cfg_env
    cfg = load_config(args.config)
    cfg["jobs"] = 1
    run_dir = RUNS / "_smoke" / args.run
    spec, _ = install(cfg, run_dir)
    _cfg_env = {k: str(v) for k, v in cfg["env"].items()}
    ampl, _, env = rs.get_module_environment(rs.DEFAULT_PREFIX)
    inst = [i for i in rs.load_instances() if i.name == args.instance][0]
    res = rs.run_one(inst, spec, ampl, env, run_dir, min(args.time_limit, 5.0),
                     cfg["threads"], 1, True)
    print(res)
    lp = run_dir / "logs" / spec.name / inst.family / f"{inst.name}.log"
    print(lp.read_text()[-3500:])


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("wave", "drive", "smoke"):
        s = sub.add_parser(name)
        s.add_argument("--config", type=Path, required=True)
        s.add_argument("--run", required=True)
        s.add_argument("--time-limit", type=float, default=180.0)
        s.add_argument("--budget", type=float, default=200.0)
        s.add_argument("--subset", type=Path)
        if name == "smoke":
            s.add_argument("--instance", required=True)
    args = p.parse_args()
    {"wave": cmd_wave, "drive": cmd_drive, "smoke": cmd_smoke}[args.cmd](args)


if __name__ == "__main__":
    main()
