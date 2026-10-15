#!/usr/bin/env python3
"""Resume-safe AMPL solver runner for the frozen 505-instance benchmark.

Each solver receives its strongest validated installed configuration.  Solver
stdout/stderr, AMPL-side objective/feasibility re-evaluation, and runner status
are kept in independent append-only per-instance logs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


EXPERIMENT_DIR = Path(__file__).resolve().parent
BENCH_DIR = EXPERIMENT_DIR.parents[1]          # <repo>/baselines
WORKSPACE = BENCH_DIR.parent                   # <repo>
SOURCE_MANIFEST = BENCH_DIR / "benchmark/manifest.tsv"
GAMS_DIR = BENCH_DIR / "benchmark/gams_world"
MINL_DIR = BENCH_DIR / "benchmark/minlplib_linear"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "runs"
DEFAULT_PREFIX = Path(os.environ.get("GEAR_AMPL_PREFIX", sys.prefix))

VAR_RE = re.compile(r"^var\s+", re.MULTILINE)
CON_RE = re.compile(r"^e\d+\s*:", re.MULTILINE)
DISCRETE_RE = re.compile(r"^var\s+.*\b(?:binary|integer)\b", re.MULTILINE)
SENSE_RE = re.compile(r"^\s*(minimize|maximize)\s+\w+\s*:", re.MULTILINE)
FUNCTION_RE = re.compile(
    r"(?<![A-Za-z_])(sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|"
    r"log10|log|exp|sqrt|abs)\s*\("
)
COUENNE_EXPENSIVE_BT_MAX_VARS = 100


@dataclass(frozen=True)
class SolverSpec:
    name: str
    version: str
    executable: str
    option_name: str
    options_template: str
    default_jobs: int
    native_parallel: bool
    demo_size_limit: bool = False
    external_limit_only: bool = False
    note: str = ""


SOLVERS: dict[str, SolverSpec] = {
    "baron": SolverSpec(
        name="baron",
        version="BARON 25.12.10; AMPL driver 20260624",
        executable="baron",
        option_name="baron",
        options_template="maxtime={time} threads=1 outlev=1",
        default_jobs=16,
        native_parallel=False,
        demo_size_limit=True,
        note="All 505 models are continuous, so BARON's MIP-subsolver threads do not apply.",
    ),
    "couenne": SolverSpec(
        name="couenne",
        version="Couenne 0.5.8; ASL 20221115",
        executable="couenne",
        option_name="couenne",
        options_template="bonmin.time_limit={time} outlev=1",
        default_jobs=16,
        native_parallel=False,
    ),
    "ipopt": SolverSpec(
        name="ipopt",
        version="Ipopt 3.14.19; ASL 20231111",
        executable=str(DEFAULT_PREFIX / "bin/ipopt"),
        option_name="ipopt",
        options_template=(
            "max_wall_time={time} print_level=5 timing_statistics=yes "
            "honor_original_bounds=yes"
        ),
        default_jobs=4,
        native_parallel=True,
        note="Four concurrent solves with four OpenBLAS/OpenMP threads each by default.",
    ),
    "knitro": SolverSpec(
        name="knitro",
        version="Artelys Knitro 15.1.0; AMPL driver 20260108",
        executable="knitro",
        option_name="knitro",
        options_template=(
            "maxtime={time} numthreads={threads} nlp_algorithm=5 "
            "ms_enable=1 ms_numthreads={threads} ms_deterministic=0 "
            "ms_maxsolves=0 ms_terminate=4 concurrent_evals=1 "
            "soltype=1 outlev=2 timing=1"
        ),
        default_jobs=1,
        native_parallel=True,
        demo_size_limit=True,
        note="Parallel multi-algorithm multistart; available license is size-limited.",
    ),
    "lgo": SolverSpec(
        name="lgo",
        version="LGO 2015-01-17; AMPL driver 20190908",
        executable="lgo",
        option_name="lgo",
        options_template="timelim={time} outlev=1 timing=1",
        default_jobs=16,
        native_parallel=False,
    ),
    "lindoglobal": SolverSpec(
        name="lindoglobal",
        version="LINDO Global 16.0.7099.148; AMPL driver 20260615",
        executable="lindoglobal",
        option_name="lindoglobal",
        options_template="maxtime={time} threads=1 outlev=1",
        default_jobs=16,
        native_parallel=False,
        note=(
            "Forced to one thread because all tested 16-thread modes returned "
            "a point inconsistent with the solver-reported objective."
        ),
    ),
    "minos": SolverSpec(
        name="minos",
        version="AMPL/MINOS 5.51; ASL2 20251121",
        executable="minos",
        option_name="minos",
        options_template="timelim={time} outlev=1 timing=1",
        default_jobs=16,
        native_parallel=False,
    ),
    "scip": SolverSpec(
        name="scip",
        version="SCIP 10.0.2; AMPL/MP driver 20260624",
        executable="scip",
        option_name="scip",
        options_template=(
            "lim:time={time} alg:concurrent=1 "
            "par:minnthreads={threads} par:maxnthreads={threads} "
            "par:mode=0 lp:threads=1 tech:outlev=1 tech:timing=2"
        ),
        default_jobs=1,
        native_parallel=True,
        note="Sixteen-way opportunistic concurrent SCIP by default.",
    ),
    "snopt": SolverSpec(
        name="snopt",
        version="SNOPT 7.5-1.2; AMPL driver 20211109",
        executable="snopt",
        option_name="snopt",
        options_template="outlev=1 timing=1",
        default_jobs=16,
        native_parallel=False,
        external_limit_only=True,
        note="The AMPL driver exposes no solve time limit; runner sends SIGINT.",
    ),
}


@dataclass(frozen=True)
class Instance:
    family: str
    name: str
    model_path: Path
    relative_path: str
    sha256: str
    nvars: int
    ncons: int
    discrete_vars: int
    sense: str
    functions: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_instances() -> list[Instance]:
    records: list[Instance] = []
    with SOURCE_MANIFEST.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip() or raw.startswith(("#", "family\t")):
                continue
            family, name, *_ = raw.split()
            if family == "gams_world":
                path = GAMS_DIR / f"{name}.mod"
            elif family == "minlplib_linear":
                path = MINL_DIR / f"{name}.mod"
            else:
                raise ValueError(f"Unexpected benchmark family: {family}")
            if not path.is_file():
                raise FileNotFoundError(path)
            text = path.read_text(encoding="utf-8", errors="replace")
            sense_match = SENSE_RE.search(text)
            if not sense_match:
                raise ValueError(f"No objective sense found in {path}")
            functions = tuple(sorted(set(FUNCTION_RE.findall(text))))
            records.append(
                Instance(
                    family=family,
                    name=name,
                    model_path=path,
                    relative_path=str(path.relative_to(WORKSPACE)),
                    sha256=sha256_file(path),
                    nvars=len(VAR_RE.findall(text)),
                    ncons=len(CON_RE.findall(text)),
                    discrete_vars=len(DISCRETE_RE.findall(text)),
                    sense=sense_match.group(1),
                    functions=functions,
                )
            )
    if len(records) != 505:
        raise ValueError(f"Expected 505 instances, found {len(records)}")
    if sum(i.family == "gams_world" for i in records) != 265:
        raise ValueError("Expected 265 gams_world instances")
    if sum(i.family == "minlplib_linear" for i in records) != 240:
        raise ValueError("Expected 240 minlplib_linear instances")
    if len({(i.family, i.name) for i in records}) != 505:
        raise ValueError("Duplicate family/name records in the frozen manifest")
    return records


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def prepare_output(output_root: Path, instances: list[Instance], module_path: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    header = [
        "family",
        "instance",
        "relative_model_path",
        "sha256",
        "nvars",
        "ncons",
        "discrete_vars",
        "sense",
        "functions",
    ]
    lines = ["\t".join(header)]
    for inst in instances:
        lines.append(
            "\t".join(
                [
                    inst.family,
                    inst.name,
                    inst.relative_path,
                    inst.sha256,
                    str(inst.nvars),
                    str(inst.ncons),
                    str(inst.discrete_vars),
                    inst.sense,
                    ",".join(inst.functions),
                ]
            )
        )
    atomic_write(output_root / "manifest.tsv", "\n".join(lines) + "\n")

    metadata = {
        "protocol": {
            "instances": 505,
            "gams_world": 265,
            "minlplib_linear": 240,
            "default_time_limit_sec": 180,
            "strict_feasibility_audit": 1e-8,
            "loose_feasibility_audit": 1e-6,
            "objective_source": "AMPL re-evaluation of returned point",
            "source_manifest": str(SOURCE_MANIFEST.relative_to(WORKSPACE)),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "ampl_module_path": module_path,
        "solvers": {name: asdict(spec) for name, spec in SOLVERS.items()},
    }
    atomic_write(
        output_root / "metadata.json",
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )


def get_module_environment(prefix: Path) -> tuple[Path, str, dict[str, str]]:
    module_python = prefix / "bin/python"
    if not module_python.is_file():
        raise FileNotFoundError(module_python)
    module_path = subprocess.check_output(
        [str(module_python), "-m", "amplpy.modules", "path"],
        text=True,
    ).strip()
    ampl = subprocess.check_output(
        [str(module_python), "-m", "amplpy.modules", "find", "ampl"],
        text=True,
    ).strip()
    if not Path(ampl).is_file():
        raise FileNotFoundError(ampl)
    env = os.environ.copy()
    env["PATH"] = module_path + os.pathsep + env.get("PATH", "")
    return Path(ampl), module_path, env


def config_hash(
    spec: SolverSpec,
    inst: Instance,
    time_limit: float,
    threads: int,
    jobs: int,
) -> str:
    payload = {
        "runner_schema": 4,
        "solver": asdict(spec),
        "model_sha256": inst.sha256,
        "time_limit": time_limit,
        "threads": threads,
        "jobs": jobs,
        "evaluation_error_retry": "bounded_midpoint_for_every_solver",
        "couenne_bound_tightening_policy": (
            "defaults_through_100_vars_else_disable_optimality_and_aggressive_fbbt"
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def ampl_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def couenne_option_text(inst: Instance, time_limit: float) -> str:
    """Return the native Couenne option file used from the solver cwd.

    Couenne's documented OBBT performs 2*n LP solves and is recommended only
    for small models.  Its initialization does not reliably observe the native
    time limit, so retain defaults through 100 variables and disable the two
    expensive root tightening methods above that size.
    """
    lines = [f"time_limit {time_limit:g}"]
    if inst.nvars > COUENNE_EXPENSIVE_BT_MAX_VARS:
        lines.extend(["optimality_bt no", "aggressive_fbbt no"])
    return "\n".join(lines) + "\n"


def ampl_program(
    inst: Instance,
    spec: SolverSpec,
    options: str,
    initialization: str,
) -> str:
    # The feasibility audit uses original AMPL entities after loading the
    # returned solution.  A directly feasible original start is still a valid
    # primal if a solver terminates without moving it.
    lines = [f'model "{ampl_quote(str(inst.model_path))}";']
    if initialization == "bounded_midpoint":
        lines.append(
            "let {i in 1.._nvars} _var[i] := "
            "if _var[i].lb > -Infinity and _var[i].ub < Infinity then "
            "0.5 * (_var[i].lb + _var[i].ub) else "
            "if _var[i].lb > -Infinity then max(0, _var[i].lb + 1) else "
            "if _var[i].ub < Infinity then min(0, _var[i].ub - 1) else 0.125;"
        )
    elif initialization != "model_default":
        raise ValueError(f"Unknown initialization strategy: {initialization}")
    lines.extend(
        [
            f'option solver "{ampl_quote(spec.executable)}";',
            f'option {spec.option_name}_options "{ampl_quote(options)}";',
            "solve;",
            "param harness_con_violation default 0;",
            "param harness_var_violation default 0;",
            (
                "let harness_con_violation := if _ncons = 0 then 0 else "
                "max {i in 1.._ncons} "
                "(if _con[i].body < _con[i].lb then "
                "_con[i].lb - _con[i].body else "
                "if _con[i].body > _con[i].ub then "
                "_con[i].body - _con[i].ub else 0);"
            ),
            (
                "let harness_var_violation := if _nvars = 0 then 0 else "
                "max {i in 1.._nvars} "
                "(if _var[i] < _var[i].lb then _var[i].lb - _var[i] else "
                "if _var[i] > _var[i].ub then _var[i] - _var[i].ub else 0);"
            ),
            (
                'printf "HARNESS_RESULT '
                f'solver={spec.name} family={inst.family} instance={inst.name} '
                'solve_result=%s solve_result_num=%d '
                'objective=%.17g constraint_violation=%.17g '
                'variable_violation=%.17g nvars=%d ncons=%d elapsed=%.9g\\n", '
                "solve_result, solve_result_num, _obj[1], "
                "harness_con_violation, harness_var_violation, "
                "_nvars, _ncons, _solve_elapsed_time;"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def terminal_for_hash(log_path: Path, digest: str) -> bool:
    if not log_path.is_file():
        return False
    with log_path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - (1 << 17)))
        tail = handle.read().decode("utf-8", errors="replace")
    return bool(
        re.search(
            rf"^RUNNER_RESULT .*config_hash={re.escape(digest)}(?:\s|$)",
            tail,
            re.MULTILINE,
        )
    )


def classify_log(text: str, returncode: int, timed_out: bool) -> str:
    lower = text.lower()
    if "demo license is limited to 10 variables" in lower:
        return "license_limit"
    if "does not support operation" in lower or "operation not supported" in lower:
        return "unsupported_operation"
    if "error evaluating" in lower or "can't evaluate" in lower:
        return "evaluation_error"
    if "too few degrees of freedom" in lower or "too_few_dof" in lower:
        return "too_few_dof"
    if "harness_result " in lower:
        return "completed_after_interrupt" if timed_out else "completed"
    if timed_out:
        return "watchdog_timeout"
    if returncode != 0:
        return "runner_error"
    return "missing_harness_result"


def read_tail(path: Path, bytes_to_read: int = 1 << 18) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - bytes_to_read))
        return handle.read().decode("utf-8", errors="replace")


def append_preclassified(
    log_path: Path,
    inst: Instance,
    spec: SolverSpec,
    digest: str,
    status: str,
    detail: str,
) -> dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"\nRUN_METADATA solver={spec.name} family={inst.family} "
            f"instance={inst.name} model_sha256={inst.sha256} "
            f"config_hash={digest}\n"
        )
        log.write(detail.rstrip() + "\n")
        log.write(
            f"RUNNER_RESULT status={status} exit_code=0 timed_out=0 "
            f"wall_time=0 config_hash={digest}\n"
        )
    return {
        "family": inst.family,
        "instance": inst.name,
        "status": status,
        "exit_code": 0,
        "wall_time": 0.0,
        "config_hash": digest,
    }


def run_one(
    inst: Instance,
    spec: SolverSpec,
    ampl: Path,
    base_env: dict[str, str],
    output_root: Path,
    time_limit: float,
    threads: int,
    jobs: int,
    force: bool,
) -> dict[str, object]:
    digest = config_hash(spec, inst, time_limit, threads, jobs)
    log_path = output_root / "logs" / spec.name / inst.family / f"{inst.name}.log"
    if not force and terminal_for_hash(log_path, digest):
        return {
            "family": inst.family,
            "instance": inst.name,
            "status": "skipped_terminal",
            "exit_code": 0,
            "wall_time": 0.0,
            "config_hash": digest,
        }

    if spec.demo_size_limit and (inst.nvars > 10 or inst.ncons + 1 > 10):
        return append_preclassified(
            log_path,
            inst,
            spec,
            digest,
            "license_limit",
            (
                f"PRECHECK demo license limit: nvars={inst.nvars}, "
                f"ncons_plus_objective={inst.ncons + 1}; maximum is 10/10."
            ),
        )

    solver_threads = threads if spec.native_parallel else 1
    options = spec.options_template.format(
        time=f"{time_limit:g}",
        threads=solver_threads,
    )
    scratch = output_root / "work" / spec.name / inst.family / inst.name
    scratch.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.name == "couenne":
        atomic_write(scratch / "couenne.opt", couenne_option_text(inst, time_limit))

    env = base_env.copy()
    if spec.name == "ipopt":
        per_process_threads = max(1, threads // max(1, jobs))
    elif spec.native_parallel:
        per_process_threads = max(1, threads)
    else:
        per_process_threads = 1
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = str(per_process_threads)
    env["OMP_DYNAMIC"] = "FALSE"

    strategies = ["model_default", "bounded_midpoint"]
    total_start = time.monotonic()
    final_status = "runner_error"
    final_returncode = -999
    final_timed_out = False
    for strategy_index, initialization in enumerate(strategies):
        attempt = read_tail(log_path, 1 << 20).count("RUN_METADATA ") + 1
        attempt_start = time.monotonic()
        timed_out = False
        program = ampl_program(inst, spec, options, initialization)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\n=== ATTEMPT {attempt} ===\n"
                f"RUN_METADATA solver={spec.name} version={spec.version!r} "
                f"family={inst.family} instance={inst.name} "
                f"model={inst.relative_path} model_sha256={inst.sha256} "
                f"time_limit={time_limit:g} solver_threads={solver_threads} "
                f"process_threads={per_process_threads} jobs={jobs} "
                f"initialization={initialization} "
                f"options={shlex.quote(options)} config_hash={digest}\n"
            )
            log.flush()
            proc = subprocess.Popen(
                [str(ampl)],
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=scratch,
                env=env,
                start_new_session=True,
            )
            watchdog = time_limit + (2.0 if spec.external_limit_only else 30.0)
            try:
                proc.communicate(input=program, timeout=watchdog)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.communicate()
            returncode = int(proc.returncode)
            attempt_wall = time.monotonic() - attempt_start
            log.flush()
            tail = read_tail(log_path)
            current_attempt = tail.rsplit(f"=== ATTEMPT {attempt} ===", 1)[-1]
            status = classify_log(current_attempt, returncode, timed_out)
            log.write(
                f"ATTEMPT_RESULT status={status} exit_code={returncode} "
                f"timed_out={int(timed_out)} wall_time={attempt_wall:.9g} "
                f"initialization={initialization} config_hash={digest}\n"
            )
        final_status = status
        final_returncode = returncode
        final_timed_out = timed_out
        if not (
            status == "evaluation_error"
            and strategy_index + 1 < len(strategies)
        ):
            break

    wall = time.monotonic() - total_start
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"RUNNER_RESULT status={final_status} "
            f"exit_code={final_returncode} "
            f"timed_out={int(final_timed_out)} wall_time={wall:.9g} "
            f"config_hash={digest}\n"
        )
    return {
        "family": inst.family,
        "instance": inst.name,
        "status": final_status,
        "exit_code": final_returncode,
        "wall_time": wall,
        "config_hash": digest,
    }


def last_runner_record(log_path: Path) -> str:
    records = [
        line
        for line in read_tail(log_path, 1 << 20).splitlines()
        if line.startswith("RUNNER_RESULT ")
    ]
    return records[-1] if records else ""


def write_solver_summary(
    output_root: Path,
    solver: str,
    instances: Iterable[Instance],
) -> None:
    lines = ["family\tinstance\trunner_result"]
    for inst in instances:
        log_path = output_root / "logs" / solver / inst.family / f"{inst.name}.log"
        lines.append(
            f"{inst.family}\t{inst.name}\t{last_runner_record(log_path)}"
        )
    atomic_write(
        output_root / "summaries" / f"{solver}.tsv",
        "\n".join(lines) + "\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=sorted(SOLVERS))
    parser.add_argument("--all", action="store_true", help="run all solver arms sequentially")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--time-limit", type=float, default=180.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument(
        "--instances",
        help="comma-separated instance names; family/name is also accepted",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def selected_instances(
    all_instances: list[Instance],
    selector: str | None,
    maximum: int,
) -> list[Instance]:
    chosen = all_instances
    if selector:
        wanted = {item.strip() for item in selector.split(",") if item.strip()}
        chosen = [
            inst
            for inst in chosen
            if inst.name in wanted or f"{inst.family}/{inst.name}" in wanted
        ]
        matched = {inst.name for inst in chosen} | {
            f"{inst.family}/{inst.name}" for inst in chosen
        }
        missing = sorted(wanted - matched)
        if missing:
            raise ValueError(f"Unknown instance selectors: {missing}")
    if maximum > 0:
        chosen = chosen[:maximum]
    return chosen


def run_solver_arm(
    solver: str,
    instances: list[Instance],
    args: argparse.Namespace,
    ampl: Path,
    env: dict[str, str],
) -> None:
    spec = SOLVERS[solver]
    jobs = args.jobs if args.jobs is not None and not args.all else spec.default_jobs
    jobs = max(1, jobs)
    print(
        f"[{solver}] instances={len(instances)} jobs={jobs} "
        f"threads={args.threads} time_limit={args.time_limit:g}s",
        flush=True,
    )
    counts: dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(
                run_one,
                inst,
                spec,
                ampl,
                env,
                args.output_root,
                args.time_limit,
                args.threads,
                jobs,
                args.force,
            ): inst
            for inst in instances
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            status = str(result["status"])
            counts[status] = counts.get(status, 0) + 1
            completed += 1
            if completed <= 5 or completed % 25 == 0 or completed == len(instances):
                print(
                    f"[{solver}] {completed}/{len(instances)} "
                    f"last={result['family']}/{result['instance']} "
                    f"status={status} counts={dict(sorted(counts.items()))}",
                    flush=True,
                )
    write_solver_summary(args.output_root, solver, load_instances())
    print(f"[{solver}] complete counts={dict(sorted(counts.items()))}", flush=True)


def main() -> int:
    args = parse_args()
    if not args.prepare_only and not args.all and not args.solver:
        raise SystemExit("Specify --solver, --all, or --prepare-only")
    if args.time_limit <= 0:
        raise SystemExit("--time-limit must be positive")
    if args.threads <= 0:
        raise SystemExit("--threads must be positive")

    instances = load_instances()
    ampl, module_path, env = get_module_environment(args.prefix)
    prepare_output(args.output_root, instances, module_path)
    if args.prepare_only:
        print(f"Prepared {args.output_root} for {len(instances)} instances")
        return 0

    chosen = selected_instances(instances, args.instances, args.max_instances)
    solvers = list(SOLVERS) if args.all else [args.solver]
    for solver in solvers:
        assert solver is not None
        run_solver_arm(solver, chosen, args, ampl, env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
