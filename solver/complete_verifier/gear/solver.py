"""``gear.solve`` and ``Solver`` (G5 DESIGN sections 1 Timing, 2 Handoff and Reuse, 6).

A solve resolves its ``Options``, compiles the source if needed, checks the ``Compiled`` (unchanged since compile,
compiled under the same COMPILE_KEYS), re-reads the source, and hands a fresh model to the frozen solver
(``api.ABCrownSolver``) exactly as ``nlp_example.run_problem`` does. Two hooks of the frozen code receive the facade's
objects, and do nothing without them:

- ``ABCrownSolver.gear_handoff`` runs after ``_maybe_augment_model_with_structural_constraints``: it checks the wrapper
  decision and the rows against the ``Compiled`` and attaches ``__gear_program__`` (the compiled node program, its
  tensors bound to the live model's after a bitwise check) and ``__gear_build__`` (the BoundedModule build: CPU
  template reuse and the op-budget record);
- ``bab.solve_deadline`` (timing="solve") clamps the two roots of the frozen timers, ``_verify_impl`` and
  ``RootState``.

After the solve, the incumbent is replayed on a fresh model of ``Compiled.original`` and the sink's values are mapped
to a ``Result`` (``report.map_result``).

Deadline contract (timing="solve"): one monotonic deadline T = entry + time limit. A compile that ends after T gives
NO_SOLUTION / TIME_LIMIT. Inside the solver every root operation with a time budget, the root NLP included (A293),
is admitted only before T and capped by the time left; BaB batches are admitted only if predicted to fit. Work that
started before T is not interrupted, and the root's budget-free operations (rows, consistency, domain list, symmetry,
convex cuts, the lifted-LP setup, the initial domains) and the report's Replay still run, so the solve can end after
T by the last admitted unit plus those operations. ``Stats.deadline_overshoot`` records it. Measured (CPU, cute_portfl6,
T = 3 s): 0.3-0.7 s under host load; 0.9 s with T forced inside the root (0.5 s of it injected); before A293's fix,
a root NLP reached after T ran on its own budget and overran by up to 27 s.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
import os
import random
import subprocess
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from . import report
from .compile import (MAX, MIN, Compiled, Sense, build_model, cfg_get, check_unchanged, compile as _compile,
                      compile_keys, read_artifact, _float64_default)
from .config import Options, resolve, thaw
from .errors import GearError, InternalError, LegacyArtifactError, OptionError
from .result import Provenance, Replay, Result, Stats, Status, Termination

_monotonic = time.monotonic   # the facade's clock; tests replace these two
_wall = time.time
REPO = Path(__file__).resolve().parents[2]
CPU_SOLVER_KINDS = ("linprog", "highs", "ipopt", "lanes", "native")


# ---- process setup (as nlp_example.run_problem) ------------------------------------------------------------------
def set_runtime_environment(config) -> None:
    """Set global runtime knobs from a config dict: seeds, TF32 off, determinism, default dtype, JIT precompile."""
    general_cfg = config["general"]
    seed = general_cfg["seed"]
    device = general_cfg["device"]

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_printoptions(precision=8)

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Always disable TF32 (precision is too low for verification).
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            torch.backends.cudnn.allow_tf32 = False

    if general_cfg["deterministic"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False

    if general_cfg["double_fp"]:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    if general_cfg["precompile_jit"]:
        from jit_precompile import precompile_jit_kernels
        precompile_jit_kernels()

    if general_cfg["reset_seed_after_precompile"]:
        # Reset the seed after precompilation to ensure reproducibility.
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _torch_device(device: str) -> str:
    return "cuda:0" if device == "cuda" else device if device.startswith("cuda") else "cpu"


# ---- CPU-solver provenance (G2 P0d audit, non-strict) ------------------------------------------------------------
def _cpu_counts() -> dict:
    import cpu_solver_audit as audit
    if not audit._installed:
        with contextlib.redirect_stdout(io.StringIO()):   # its "enabled" line is not part of the solve transcript
            audit.install(strict=False)
        import atexit
        atexit.unregister(audit._report)                  # the counts go to Provenance, not to the log
    with audit._lock:
        return dict(audit.COUNTS)


# ---- handoff (section 2) -----------------------------------------------------------------------------------------
from .lower.seam import ProgramSource as _ProgramSource, same_bits as _same_bits  # noqa: E402,F401  (moved, A291)


def _storages(module: torch.nn.Module) -> set:
    ptrs = set()
    for m in module.modules():
        for v in list(vars(m).values()) + list(m._parameters.values()) + list(m._buffers.values()):
            if isinstance(v, torch.Tensor) and v.numel():
                ptrs.add(v.untyped_storage().data_ptr())
    return ptrs


class _Build:
    """``__gear_build__``: builds the BoundedModule of each LiRPANet of a solve and records its op-budget log. With a
    template store (CPU, ``solve_many``), the first build of a key keeps a deep copy taken right after the build, and
    later builds of that key use a fresh deep copy of it, unless a tensor of the copy shares a storage with the
    template (then a fresh build)."""

    def __init__(self, store: Optional[dict], key):
        self.store, self.key, self.calls, self.reuse, self.op_budgets, self.anomalies = store, key, 0, "none", {}, []

    def __deepcopy__(self, memo):
        return self

    @staticmethod
    def _fresh(build, bound_opts):
        if bound_opts.get("op_budget", {}).get("commit") == "whole":
            # section 4.1: a stage that exhausts is rebuilt with its flag off (WP12a's gear.lower.setup)
            from .lower.setup import build_bounded_module
            return build_bounded_module(build, bound_opts)
        return build()

    def __call__(self, build, bound_opts):
        slot = (self.key, self.calls)
        self.calls += 1
        template = None if self.store is None else self.store.get(slot)
        if template is None:
            net = self._fresh(build, bound_opts)
            if self.store is not None:
                self.store[slot] = copy.deepcopy(net)
        else:
            net = copy.deepcopy(template)
            if _storages(net) & _storages(template):
                net, self.reuse = self._fresh(build, bound_opts), "fresh_fallback"
            elif self.reuse == "none":
                self.reuse = "template"
        # a template copy keeps the template's gear_anomalies, so every solve of a key reports the rebuild
        self.anomalies += [a for a in getattr(net, "gear_anomalies", ()) if a not in self.anomalies]
        for stage, used, cap, exhausted in getattr(net, "op_budget_log", []):
            u, c, e = self.op_budgets.get(stage, (0, cap, False))
            self.op_budgets[stage] = (max(u, used), cap, e or exhausted)
        return net


def _expanded_rows(c: Compiled):
    """C and rhs of the runtime spec, as ``_expand_input_constraints_as_outputs`` builds them for the variant
    (float64); None when the property has no rows."""
    rs = c.rowspec
    if rs.nrows == 0:
        return None
    C, rhs = rs.to_dense(), np.asarray(rs.rhs, dtype=np.float64)
    s = int(c.linear[0].shape[0]) if c.structural_outputs else 0
    if not s:
        return C, rhs
    k = C.shape[1]
    out = np.zeros((rs.nrows + s, k + s))
    out[:rs.nrows, :1] = C[:, :1]
    if k > 1:
        out[:rs.nrows, 1 + s:] = C[:, 1:]
    out[rs.nrows + np.arange(s), 1 + np.arange(s)] = 1.0
    return out, np.concatenate([rhs, np.zeros(s)])


def _bits(a: np.ndarray, b: torch.Tensor) -> bool:
    b = b.detach().cpu()
    return b.dtype == torch.float64 and tuple(b.shape) == a.shape and \
        np.array_equal(np.ascontiguousarray(a).view(np.int64), b.contiguous().numpy().view(np.int64))


def _make_handoff(c: Compiled, build: _Build):
    def handoff(model, runtime_spec):
        from api import _StructuralConstraintOutputWrapper
        wrapped = isinstance(model, _StructuralConstraintOutputWrapper)
        if wrapped != c.structural_outputs:
            raise InternalError(f"handoff: the solver {'wrapped' if wrapped else 'did not wrap'} the model, the "
                                f"Compiled has structural_outputs={c.structural_outputs}")
        rows = [pair for clause in runtime_spec.clauses for pair in clause]
        want = _expanded_rows(c)
        if (want is None) != (not rows) or (want is not None and not (
                len(rows) == 1 and _bits(want[0], rows[0][0]) and _bits(want[1], rows[0][1]))):
            raise InternalError("handoff: the runtime rows differ from the compiled rows")
        model.__gear_program__ = _ProgramSource(c.program)
        model.__gear_build__ = build
    return handoff


# ---- callbacks ---------------------------------------------------------------------------------------------------
class _Interrupt(BaseException):
    """A callback raised: the solve ends (Termination.INTERRUPTED). A BaseException, so the solver's own
    ``except Exception`` handlers do not swallow it."""


class _Sink(report.ResultSink):
    """The WP1 sink, also calling the Solver's callbacks when a bound or an incumbent is reported."""

    def __init__(self, sign: float, on_bound, on_incumbent):
        self._sign, self._on_bound, self._on_incumbent = sign, on_bound, on_incumbent
        self._dual = self._x = None
        super().__init__()

    def _call(self, fns, *args):
        for fn in fns:
            try:
                fn(*args)
            except Exception as error:
                raise _Interrupt(error) from error

    @property
    def dual(self):
        return self._dual

    @dual.setter
    def dual(self, value):
        self._dual = value
        if value is not None:
            self._call(self._on_bound, float(value))

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        if value is not None and self.objective is not None:
            self._call(self._on_incumbent, self._sign * self.objective, {f"X_{i}": v for i, v in enumerate(value)})


# ---- replay (section 6, float64 + margin) ------------------------------------------------------------------------
def _replay_legacy(c: Compiled, spec, x, tol: float, sign: float) -> Replay:
    art, source = read_artifact(c.original)
    if dict(source) != dict(c.source):
        raise LegacyArtifactError(f"{c.original.name}: the files changed since compile")
    with _float64_default():
        model = build_model(art).to("cpu").double().eval()
        with torch.no_grad():
            y = model(torch.tensor([x], dtype=torch.float64))[0].numpy()
    A = b = None
    if art.input_linear_constraints is not None:
        A, b = art.input_linear_constraints
        A = (A.to_dense() if A.is_sparse else A).double().numpy()
        b = b.double().flatten().numpy()
    lo = spec.input_spec.lower.detach().cpu().double().flatten().numpy()
    hi = spec.input_spec.upper.detach().cpu().double().flatten().numpy()
    r = report.replay_float64(y, A, b, x, lo, hi, tol)
    return Replay(r.ok, r.feas_tol, r.max_row_violation, r.worst_row, r.max_bound_violation, 0.0,
                  float(sign * y[0]) if y.size else math.nan)


def _replay_torch(c: Compiled, x, tol: float) -> Replay:
    from .io.torch_direct import evaluate
    xa = np.asarray(x, dtype=np.float64)
    y = np.asarray(evaluate(c.torch_compiled, xa), dtype=np.float64).reshape(-1)
    a = c.assignment
    parts, names = [], []
    if a.le:
        v = y[list(a.le)]
        parts.append(v - 1e-12 * np.maximum(np.abs(v), 1.0)); names += [f"le {i}" for i in a.le]
    if a.eq:
        v = np.abs(y[list(a.eq)])
        parts.append(v - 1e-12 * np.maximum(v, 1.0)); names += [f"eq {i}" for i in a.eq]
    if c.linear is not None:
        A, b = c.linear
        parts.append(A @ xa - b - 1e-12 * np.maximum(np.abs(A) @ np.abs(xa), 1.0))
        names += [f"linear {r}" for r in range(len(b))]
    rows = np.concatenate(parts) if parts else np.zeros(0)
    lo, hi = c.box
    bound = np.concatenate([(lo - xa)[np.isfinite(lo)], (xa - hi)[np.isfinite(hi)]])
    max_row = float(rows.max()) if rows.size else -math.inf
    max_bound = float(bound.max()) if bound.size else -math.inf
    finite = bool(np.isfinite(y).all() and np.isfinite(rows).all() and np.isfinite(xa).all())
    worst = names[int(rows.argmax())] if rows.size and finite else None
    obj = float(y[a.objective]) if a.objective is not None else 0.0
    return Replay(finite and max_row <= tol and max_bound <= tol, tol, max_row, worst, max_bound, 0.0, obj)


# ---- setup per path ----------------------------------------------------------------------------------------------
def _setup_legacy(c: Compiled, config):
    from .io.legacy import _build_spec
    art, source = read_artifact(c.original)
    if dict(source) != dict(c.source):
        raise LegacyArtifactError(f"{c.original.name}: the files changed since compile")
    spec = _build_spec(art)
    model = build_model(art, device=_torch_device(config["general"]["device"]))
    if art.input_linear_constraints is not None:
        A = art.input_linear_constraints[0]
        n_constraints = A.shape[-2] if A.ndim >= 2 else A.shape[0]
        print(f"[nlp_example] Loaded {n_constraints} input linear constraints from {art.pt_path.name}")  # CLI transcript
    is_max = bool(model.is_maximization) if hasattr(model, "is_maximization") else None
    return spec, model, is_max


def _setup_torch(c: Compiled, config):
    from .io.legacy import _build_spec
    from .io.torch_direct import computing_graph
    lin = None if c.linear is None else tuple(torch.from_numpy(np.array(t)) for t in c.linear)
    spec = _build_spec(SimpleNamespace(rowspec=c.rowspec, lower=np.array(c.box[0]), upper=np.array(c.box[1]),
                                       input_linear_constraints=lin))
    model = computing_graph(c.torch_compiled, device=_torch_device(config["general"]["device"]))
    return spec, model, c.sense == MAX


# ---- the solve ---------------------------------------------------------------------------------------------------
_commit_cache: list = []


def _code_commit() -> str:
    if not _commit_cache:
        try:
            out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                                 timeout=10)
            _commit_cache.append(out.stdout.strip() if out.returncode == 0 else "unknown")
        except (OSError, subprocess.SubprocessError):
            _commit_cache.append("unknown")
    return _commit_cache[0]


def _sha256_json(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, default=repr).encode()).hexdigest()


# timing="solve" routes the deadline through _verify_impl and RootState (WP13a timer audit, timer_audit.tsv). The
# configurations below are the ones whose timers the audit found routed: input BaB in solving mode with the unified
# root, which admits every root operation against RootState.deadline. Anything else keeps clocks the deadline does
# not reach (activation-split BaB, MIP refinement, a root without admission) and is refused.
SOLVE_TIMING_NEEDS = (("solver.nlp.unified_root", True), ("solving.solving_mode", True),
                      ("bab.branching.input_split.enable", True), ("general.complete_verifier", "bab"))


def _check_solve_timing(config) -> None:
    bad = [f"{k}={cfg_get(config, k)!r}" for k, want in SOLVE_TIMING_NEEDS if cfg_get(config, k) != want]
    if bad:
        raise OptionError(f'timing="solve" routes its one deadline only through input BaB in solving mode with the '
                          f'unified root; this config has {", ".join(bad)} (use timing="legacy")')


def _time_limit(config) -> float:
    bab = config["bab"]
    if bab.get("override_timeout") is not None:
        return float(bab["override_timeout"])
    return float(bab["timeout"]) * float(bab.get("timeout_scale", 1))


class Solver:
    """Solves sources or ``Compiled`` problems under one ``Options``. ``solve_many`` is sequential (v1); on the CPU it
    reuses the BoundedModule template of an equal key (section 2 'Reuse')."""

    def __init__(self, options: Optional[Options] = None):
        self.options = options if options is not None else Options()
        self._on_incumbent, self._on_bound, self._templates = [], [], {}

    def on_incumbent(self, fn) -> None:
        """``fn(value, x)``: a new incumbent, user sense, ``x`` by variable name. An exception ends the solve."""
        self._on_incumbent.append(fn)

    def on_bound(self, fn) -> None:
        """``fn(value)``: a new certified bound, user sense. An exception ends the solve."""
        self._on_bound.append(fn)

    def solve(self, src) -> Result:
        return self._solve(src, None)

    def solve_many(self, srcs) -> list:
        return [self._solve(src, self._templates) for src in srcs]

    # -- one solve --
    def _solve(self, src, store) -> Result:
        o = self.options
        t_entry = _monotonic()
        resolved = resolve(o)
        config = thaw(resolved.config)
        if o.certificate:
            raise OptionError("certificates are not available")
        if o.timing == "solve":
            _check_solve_timing(resolved.config)
        T = t_entry + _time_limit(config) if o.timing == "solve" else None
        phases, compiled = {}, src if isinstance(src, Compiled) else None
        if compiled is None:
            if T is not None and _monotonic() >= T:
                return self._time_up(resolved, phases, T, None)
            t = _monotonic()
            compiled = _compile(src, options=o)
            phases["compile"] = _monotonic() - t
            if T is not None and _monotonic() >= T:
                return self._time_up(resolved, phases, T, None)   # a Compiled made after T is not published
        check_unchanged(compiled)
        keys = compile_keys(resolved.config)
        want = dict(compiled.compile_options["keys"])
        if keys != want:
            diff = {k: (want.get(k), keys.get(k)) for k in sorted(set(keys) | set(want)) if keys.get(k) != want.get(k)}
            raise OptionError(f"compiled under {({k: v[0] for k, v in diff.items()})}, solve requests "
                              f"{({k: v[1] for k, v in diff.items()})}; recompile")
        commit, commit_source = self._commit_mode(compiled, config)
        if config["specification"].get("shrink_eps") is not None:
            raise OptionError("specification.shrink_eps rewrites the rows; the direct paths solve the compiled rows")
        log = open(o.log, "a") if o.log else None
        with contextlib.ExitStack() as stack:
            if log is not None:
                stack.enter_context(log)
                stack.enter_context(contextlib.redirect_stdout(log))
                stack.enter_context(contextlib.redirect_stderr(log))
            return self._run(compiled, config, resolved, T, phases, store, commit, commit_source)

    def _commit_mode(self, c: Compiled, config):
        """Section 4.1 commit modes: ``partial`` only on direct:legacy; ``whole`` forced on every other path."""
        explicit = "model.op_budget.commit" in self.options.advanced
        if c.path == "direct:legacy":
            return config["model"]["op_budget"]["commit"], "explicit" if explicit else "schema_default"
        if explicit and self.options.advanced["model.op_budget.commit"] == "partial":
            raise OptionError(f'model.op_budget.commit="partial" is allowed only on direct:legacy, not {c.path}')
        config["model"]["op_budget"]["commit"] = "whole"
        return "whole", "explicit" if explicit else "forced_whole"

    def _run(self, c, config, resolved, T, phases, store, commit, commit_source) -> Result:
        t = _monotonic()
        import arguments
        from api import ABCrownSolver
        options_hash = _sha256_json(thaw(resolved.config))
        set_runtime_environment(config)
        nlp = config["solver"]["nlp"]
        if nlp.get("controller", False) and nlp.get("enabled", False):
            from input_split.native_lane import prewarm
            prewarm()
        arguments.Config.parse_config([])
        arguments.Config["general"]["device"] = config["general"]["device"]
        device = config["general"]["device"]
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise OptionError("CUDA was requested but is not available.")
        phases["init"] = _monotonic() - t          # imports, runtime environment, global config
        t = _monotonic()
        spec, model, is_max = (_setup_legacy if c.path == "direct:legacy" else _setup_torch)(c, config)
        phases["setup"] = _monotonic() - t
        if device.startswith("cuda"):
            torch.set_num_threads(1)   # as the CLI: local solvers' one-point evaluations, see nlp_example
            torch.backends.cuda.preferred_linalg_library("cusolver")
        if is_max is not None:
            config.setdefault("solving", {})["is_maximization"] = is_max
        sense = Sense.MAX if config.get("solving", {}).get("is_maximization") else Sense.MIN
        sign = -1.0 if sense == MAX else 1.0
        if T is not None:
            now = _monotonic()
            if now >= T:
                return self._time_up(resolved, phases, T, c)
            T_wall = _wall() + (T - now)
            config["bab"]["solve_deadline"] = T_wall
            config["bab"]["override_timeout"] = max(0.0, T_wall - _wall())
        key = (c.hash, options_hash, device, "float64") if (store is not None and device == "cpu") else None
        build = _Build(store if key is not None else None, key)
        solver = ABCrownSolver(spec, model, config=config)
        solver.gear_handoff = _make_handoff(c, build)
        sink = _Sink(sign, self._on_bound, self._on_incumbent)
        cpu_before = _cpu_counts()
        report.install(sink)
        t, out, error, internal = _monotonic(), None, None, None
        try:
            out = solver.solve()
            internal = str(out.status)
        except GearError:
            raise
        except (_Interrupt, KeyboardInterrupt) as e:
            internal, error = "interrupted", e
        except Exception as e:      # after search started a failure becomes a Result (section 6)
            traceback.print_exc()
            internal, error = "error", e
        finally:
            report.uninstall()
            phases["solve"] = _monotonic() - t
        cpu_after = _cpu_counts()
        late = T is not None and _monotonic() >= T
        t = _monotonic()
        tol = float(config["solver"]["nlp"]["constr_viol_tol"])
        replay = None
        if sink.x is not None:
            replay = (_replay_legacy(c, spec, sink.x, tol, sign) if c.path == "direct:legacy"
                      else _replay_torch(c, sink.x, tol))
        primal = sink.primal if sink.primal is not None else (None if sink.objective is None else sign * sink.objective)
        x = None if sink.x is None else {f"X_{i}": v for i, v in enumerate(sink.x)}
        anomalies = list(resolved.anomalies)
        events = tuple(sink.events)
        mapped = internal
        if internal == "interrupted":
            mapped, events = "unknown", ("interrupted",) + events
        elif internal not in ("verified", "unknown", "error"):
            mapped = "error"
            anomalies.append(f"internal_status:{internal}")
        if internal == "interrupted":
            anomalies.append(f"interrupted:{type(error.__cause__ or error).__name__}")
        elif error is not None:
            anomalies.append(f"exception:{type(error).__name__}")
        rr = None if replay is None else report.ReplayRecord(replay.ok, replay.feas_tol, replay.max_row_violation,
                                                             replay.max_bound_violation, replay.worst_row)
        rec = report.map_result(mapped, primal, sink.dual, sense="max" if sense == MAX else "min", replay=rr, x=x,
                                gap_abs=float(nlp["bab_primal_dual_abs_tol"]),
                                gap_rel=float(nlp["bab_primal_dual_rel_tol"]), events=events,
                                incumbent_ever=sink.x is not None)
        termination = rec.termination
        if late and rec.status not in (Status.OPTIMAL, Status.INFEASIBLE, Status.ERROR) \
                and termination != Termination.INTERRUPTED:
            termination = Termination.TIME_LIMIT
        anomalies += list(rec.anomalies)
        for stage, (used, cap, exhausted) in sorted(build.op_budgets.items()):
            if exhausted and f"op_budget_exhausted:{stage}:{cap}" not in build.anomalies:
                anomalies.append(f"op_budget_exhausted:{stage}:{cap}")
        anomalies += build.anomalies
        cpu = {k: cpu_after.get(k, 0) - cpu_before.get(k, 0) for k in CPU_SOLVER_KINDS}
        if resolved.cpu_solver_free:
            anomalies += [f"cpu_solver_used:{k}={n}" for k, n in cpu.items() if n]
        phases["report"] = _monotonic() - t
        bab = (out.stats or {}).get("bab") if out is not None else None
        stats = Stats(phases=dict(phases), deadline_overshoot=None if T is None else _monotonic() - T,
                      op_budgets=dict(build.op_budgets), nodes=int(sum(r[2] for r in bab or [])),
                      peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else 0.0,
                      rejected_incumbents=int(replay is not None and not replay.ok), reuse=build.reuse,
                      op_ledger=tuple(sink.ops))
        prov = self._provenance(c, resolved, options_hash, commit, commit_source, cpu, device)
        return Result(rec.status, termination, sense, rec.primal_bound, rec.dual_bound, rec.gap_abs, rec.gap_rel,
                      rec.x, replay, None, stats, prov, tuple(anomalies), internal)

    def _provenance(self, c, resolved, options_hash, commit, commit_source, cpu, device) -> Provenance:
        ob = resolved.config["model"]["op_budget"]
        return Provenance(
            path="" if c is None else c.path, source={} if c is None else dict(c.source),
            compiled_hash="" if c is None else c.hash, program_hash="" if c is None else c.program_hash,
            pipeline=() if c is None else c.pipeline, preset=resolved.preset, preset_sha256=resolved.preset_sha256,
            historical_preset=resolved.historical, options_hash=options_hash, timing=self.options.timing,
            op_budgets={k: int(ob[k]) for k in ("hidden_squares", "complete_squares", "prove_rows")},
            op_budget_commit=commit, op_budget_commit_source=commit_source,
            integer_receipt=None if c is None else c.integer_receipt, open_route=False, cpu_solvers=cpu,
            commit=_code_commit(), device=device, extensions=())

    def _time_up(self, resolved, phases, T, c) -> Result:
        """T passed before search: NO_SOLUTION / TIME_LIMIT with the phase times (section 1 Timing)."""
        sense = MIN if c is None else c.sense
        p, d = (math.inf, -math.inf) if sense == MIN else (-math.inf, math.inf)
        stats = Stats(phases=dict(phases), deadline_overshoot=_monotonic() - T, op_budgets={}, nodes=0,
                      peak_gpu_mb=0.0, rejected_incumbents=0)
        commit = resolved.config["model"]["op_budget"]["commit"]
        prov = self._provenance(c, resolved, _sha256_json(thaw(resolved.config)), commit, "schema_default",
                                {k: 0 for k in CPU_SOLVER_KINDS}, cfg_get(resolved.config, "general.device"))
        return Result(Status.NO_SOLUTION, Termination.TIME_LIMIT, sense, p, d, math.inf, math.inf, None, None, None,
                      stats, prov, tuple(resolved.anomalies), "unknown")


def solve(src, options: Optional[Options] = None, **option_fields) -> Result:
    """Solve one source or ``Compiled`` (section 1). ``option_fields`` replace fields of ``options``."""
    options = options if options is not None else Options()
    if option_fields:
        options = options.replace(**option_fields)
    return Solver(options).solve(src)
