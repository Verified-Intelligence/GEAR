#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################

# An example of using the ABCrown API to verify an NLP benchmark instance.
# It can still be used as a single-file example, and now also accepts CLI
# arguments so batch scripts can call it directly.

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

# Overlap clean native-worker imports with the owner's Torch/model imports.
# Importing this example as a library has no process side effects.
if __name__ == '__main__' and '--config' in sys.argv:
    import yaml
    _config_arg = Path(sys.argv[sys.argv.index('--config') + 1])
    if not _config_arg.is_absolute():
        _config_arg = Path(__file__).resolve().parent / _config_arg
    with _config_arg.open() as _config_source:
        _early_nlp = (yaml.safe_load(_config_source) or {}).get('solver', {}).get('nlp', {})
    if _early_nlp.get('controller', False) and _early_nlp.get('enabled', False):
        from input_split.native_lane import prewarm
        prewarm()

import torch

import sys
import arguments
from api import ABCrownSolver, ConfigBuilder, VerificationSpec


DEFAULT_PROBLEM_NAME = "hs059"  # e.g. eq6_1, polygon25
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VNNLIB_BASE = SCRIPT_DIR / "../../nlp_benchmarks/vnnlib"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "exp_configs/NLP_solver/nlplib_new.yaml"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "mod_files"
_CONTROLLER_CLI = False


def _str_to_bool(value: str | bool | None) -> bool | None:
    """Parse a CLI bool while allowing None to mean "leave config unchanged"."""
    if value is None or isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _stem_to_class_name(stem: str) -> str:
    """Convert a problem stem to the generated Model class name."""
    s = stem.replace("-", "_").replace(".", "_")
    parts = s.split("_")
    return "".join(p.capitalize() for p in parts if p) + "Model"


def _resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to this file so callers can run from any cwd."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = SCRIPT_DIR / resolved
    return resolved.resolve()


def _prepend_sys_path(path: str | Path) -> None:
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


def _load_models_module(model_dir: str | Path):
    """Import generated benchmark models from ``model_dir``."""
    md = _resolve_path(model_dir)
    pkg_name = md.name
    module_search_dirs = [md]
    if not (md / "modules.py").is_file():
        fallback_modules_dir = md.parent.parent / "minlplib" / "mod_files"
        if (fallback_modules_dir / "modules.py").is_file():
            module_search_dirs.append(fallback_modules_dir)

    _prepend_sys_path(md.parent)
    for module_dir in reversed(module_search_dirs):
        _prepend_sys_path(module_dir)
    return importlib.import_module(pkg_name)


def _resolve_existing_path(stem: str, base_dir: Path, suffix: str, alternates: tuple[str, ...]) -> Path:
    """Resolve an artifact path, allowing a small set of stem variants."""
    candidates = [stem, *alternates]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        path = base_dir / f"{candidate}{suffix}"
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Could not find artifact for '{stem}' under {base_dir} with suffix '{suffix}'.")


def _resolve_problem_paths(problem_name: str,
                           *,
                           model_dir: str | Path,
                           vnnlib_base: str | Path,
                           benchmark: str = None,
                           benchmark_root: str | Path = None) -> tuple[Path, Path, Path]:
    """Resolve the model checkpoint, vnnlib paths, and module dir for a problem."""
    
    if benchmark and benchmark_root:
        br = Path(benchmark_root).resolve()
        if benchmark in ("minlplib", "minnlp"):
            model_dir_path = br / "minlplib" / "mod_files"
            vnnlib_base_path = br / "minlplib" / "vnnlib"
        elif benchmark == "minlplib_linear":
            model_dir_path = br / "minlplib_linear" / "py_minlplib_linear"
            vnnlib_base_path = br / "minlplib_linear" / "vnnlib_minlplib_linear"
        elif benchmark == "gams_world":
            model_dir_path = br / "gams_world" / "mod_files_gams"
            vnnlib_base_path = br / "gams_world" / "vnnlib_gams"
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
    else:
        model_dir_path = _resolve_path(model_dir)
        vnnlib_base_path = _resolve_path(vnnlib_base)
        
    model_variants = (problem_name.replace("-", "_"),)
    vnnlib_variants = (problem_name.replace("_", "-"),)

    pt_path = _resolve_existing_path(problem_name, model_dir_path, "_model_state.pt", model_variants)
    vnnlib_path = _resolve_existing_path(problem_name, vnnlib_base_path, ".vnnlib", vnnlib_variants)
    return pt_path, vnnlib_path, model_dir_path


def _load_input_linear_constraints(pt_path: str | Path):
    """Load optional input constraints from the model state."""
    state = torch.load(pt_path, map_location="cpu", weights_only=False)
    linear_constrs = state.get("linear_constr")
    if linear_constrs is None:
        return None

    A = linear_constrs.get("A")
    b = linear_constrs.get("b")
    if A is None and b is None:
        return None
    if A is None or b is None:
        raise ValueError("linear_constr must provide both A and b, or neither.")
    return A, b


def _build_config(args: argparse.Namespace) -> ConfigBuilder:
    """Load the YAML config and apply CLI overrides that mirror run_nlp.sh."""
    config = ConfigBuilder.from_yaml(str(_resolve_path(args.config)))
    overrides: dict[str, object] = {
        "general__device": args.device,
    }

    for flag in ('halpern', 'implied_box'):
        value = getattr(args, 'nlp_crown_engine_'+flag, None)
        if value is not None:
            overrides['solver__nlp__crown_engine_'+flag] = value

    if args.batch_size is not None:
        overrides["solver__batch_size"] = args.batch_size
    if args.override_timeout is not None:
        overrides["bab__override_timeout"] = args.override_timeout
    if args.clip_type is not None:
        overrides["bab__clip_n_verify__clip_input_domain__enabled"] = True
        overrides["bab__clip_n_verify__clip_input_domain__clip_type"] = args.clip_type
    if args.branching_method is not None:
        overrides["bab__branching__method"] = args.branching_method
    if args.sb_coeff_thresh is not None:
        overrides["bab__branching__input_split__sb_coeff_thresh"] = args.sb_coeff_thresh
    if args.min_batch_size_ratio is not None:
        overrides["solver__min_batch_size_ratio"] = args.min_batch_size_ratio
    if args.nlp_bab_br_budget is not None:
        overrides["solver__nlp__bab_br_budget"] = args.nlp_bab_br_budget

    if args.use_beta is not None:
        overrides["bab__branching__input_split__beta"] = args.use_beta
        if args.use_beta:
            overrides["bab__branching__input_split__ibp_enhancement"] = False
        # --bound_prop_method has the highest priority; otherwise keep the
        # legacy implicit override (with a warning, since it silently stomps
        # any bound_prop_method configured in the YAML).
        if args.bound_prop_method is None:
            implicit = "alpha-crown" if args.use_beta else "crown"
            print(f"[nlp_example] --use_beta implicitly sets bound_prop_method={implicit} "
                  f"(overrides the YAML setting; pass --bound_prop_method to control it explicitly)")
            overrides["solver__bound_prop_method"] = implicit

    if args.bound_prop_method is not None:
        overrides["solver__bound_prop_method"] = args.bound_prop_method

    if args.use_nlp is not None:
        overrides["solver__nlp__enabled"] = args.use_nlp
        if args.use_nlp:
            overrides.update({
                "solver__nlp__enable_alm": True,
                "solver__nlp__verbose": True,
                "solver__nlp__in_bab": True,
                "solver__nlp__enable_ipopt_in_bab": args.enable_ipopt_in_bab if args.enable_ipopt_in_bab is not None else True,
                "solver__nlp__ipopt_bab_topk": 2,
                "solver__nlp__ipopt_bab_timeout": 10.0,
                "solver__nlp__ipopt_timeout": 30.0,
                "solver__nlp__root_parallel": True,
                "solver__nlp__alm_outer_max_iter": 20,
                "solver__nlp__alm_inner_steps": 200,
                "solver__nlp__dual_ascent_enabled": True,
                "attack__pgd_order": "skip",
                "solver__min_batch_size_ratio": 0.0,
            })

    # Dual-only evaluation: inject external primal objective (e.g. from SCIP log) and
    # skip root NLP / Ipopt and BaB-loop ALM — measure CROWN / dual bound progress only.
    if args.dual_eval_disable_primal_solvers:
        overrides["solver__nlp__enabled"] = False
        overrides["solver__nlp__in_bab"] = False
        overrides["solver__nlp__enable_alm"] = False
        overrides["solver__nlp__ipopt_enabled"] = False
        overrides["solver__nlp__root_local_solver_warmstart"] = False

    if args.inject_primal_pt:
        pt_path = str(_resolve_path(args.inject_primal_pt))
        overrides["solving__enable_solver_res"] = True
        overrides["solving__solver_res"] = pt_path

    if args.branching_method and args.branching_method.startswith("sb"):
        overrides["bab__branching__input_split__sb_sum"] = True

    # Env-var ablation toggles for ALM improvements (1.1a / 1.3 / 3.2).
    import os as _os
    def _env_bool(name: str) -> bool | None:
        v = _os.environ.get(name)
        if v is None:
            return None
        return v.strip().lower() in ("1", "true", "yes", "on")

    for env_name, cfg_path in (
        ("ALM_ROOT_MU_WARMSTART", "solver__nlp__alm_root_mu_warmstart"),
        ("ALM_IPOPT_IN_BAB", "solver__nlp__enable_ipopt_in_bab"),
        ("NLP_ROOT_PARALLEL", "solver__nlp__root_parallel"),
        ("NLP_ROOT_LOCAL_SOLVER_WARMSTART", "solver__nlp__root_local_solver_warmstart"),
    ):
        val = _env_bool(env_name)
        if val is not None:
            overrides[cfg_path] = val

    return config.set(**overrides)


from gear.solver import set_runtime_environment as _set_runtime_environment  # noqa: E402  (moved to gear)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one NLP benchmark instance through the ABCrown API.")
    parser.add_argument("--problem_name", default=DEFAULT_PROBLEM_NAME,
                        help="Benchmark instance name, e.g. ex2_1_1 or polygon25.")
    parser.add_argument('--nlp_crown_engine_halpern', action='store_true', default=None,
                        help='Enable reflected restarted Halpern PDHG.')
    parser.add_argument('--nlp_crown_engine_implied_box', action='store_true', default=None,
                        help='Enable certified current-table FBBT before each solve.')
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="YAML config path.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                        help="Verifier device.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override solver.batch_size.")
    parser.add_argument("--override_timeout", type=float, default=None,
                        help="Override bab timeout in seconds.")
    parser.add_argument("--clip_type", choices=["relaxed", "complete"], default=None,
                        help="Override input clip type.")
    parser.add_argument("--branching_method", default=None,
                        help="Override bab branching method.")
    parser.add_argument("--use_beta", type=_str_to_bool, default=None,
                        help="If set, override beta branching on input splits.")
    parser.add_argument("--bound_prop_method", default=None,
                        choices=["crown", "alpha-crown", "ibp", "crown-ibp", "forward", "alpha-forward"],
                        help="Override solver.bound_prop_method (highest priority; "
                             "prevents the implicit override from --use_beta).")
    parser.add_argument("--use_nlp", type=_str_to_bool, default=None,
                        help="If set, override solver.nlp.enabled.")
    parser.add_argument("--enable_ipopt_in_bab", type=_str_to_bool, default=None,
                        help="Enable/disable IPOPT-in-BaB parallel heuristic.")
    parser.add_argument("--sb_coeff_thresh", type=float, default=None,
                        help="Override SB coefficient clamp threshold.")
    parser.add_argument("--min_batch_size_ratio", type=float, default=None,
                        help="Override solver.min_batch_size_ratio.")
    parser.add_argument("--nlp_bab_br_budget", type=int, default=None,
                        help="Override solver.nlp.bab_br_budget.")
    parser.add_argument("--vnnlib_base", default=str(DEFAULT_VNNLIB_BASE),
                        help="Directory containing *.vnnlib benchmark files.")
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR),
                        help="Directory containing generated *_model_state.pt files.")
    parser.add_argument("--benchmark", default=None,
                        choices=["minlplib", "minnlp", "minlplib_linear", "gams_world"],
                        help="If set with --benchmark_root, set vnnlib_base and model_dir like run_nlp_scip_logged.sh.")
    parser.add_argument("--benchmark_root", default=None,
                        help="Root directory of benchmarks (minlplib/ or minnlp alias, minlplib_linear/, gams_world/).")
    parser.add_argument("--inject_primal_pt", default=None,
                        help="Path to a .pt file: torch.save(scalar_float_or_0dim_tensor) = incumbent objective "
                             "in the verifier's convention (same as SCIP primal for minimization).")
    parser.add_argument("--dual_eval_disable_primal_solvers", action="store_true",
                        help="Disable root NLP/ALM/Ipopt and BaB-loop ALM; use with --inject_primal_pt for dual-only eval.")
    parser.add_argument("--result_json", default=None,
                        help="Write the solve outcome (status, termination, P, D, incumbent, replay) to this JSON file.")
    parser.add_argument("--loader", choices=["vnnlib", "legacy_fast"], default="vnnlib",
                        help="vnnlib: today's read_vnnlib path. legacy_fast: the strict reader of the frozen "
                             "artifacts (gear.io.legacy); it needs no .vnnlib.compiled cache.")
    parser.add_argument("--scip_cuts_file", default=None,
                        help="Path to a SCIP cuts JSON file produced by scip_cut_extractor.py. "
                             "Cuts are merged into input_linear_constraints for complete clipping.")
    return parser


def _apply_benchmark_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Mutate args.model_dir / args.vnnlib_base when --benchmark and --benchmark_root are set."""
    if args.benchmark is None or not args.benchmark_root:
        return args
    root = _resolve_path(args.benchmark_root)
    if args.benchmark in ("minlplib", "minnlp"):
        args.model_dir = str(root / "minlplib" / "mod_files")
        args.vnnlib_base = str(root / "minlplib" / "vnnlib")
    elif args.benchmark == "minlplib_linear":
        args.model_dir = str(root / "minlplib_linear" / "py_minlplib_linear")
        args.vnnlib_base = str(root / "minlplib_linear" / "vnnlib_minlplib_linear")
    elif args.benchmark == "gams_world":
        args.model_dir = str(root / "gams_world" / "mod_files_gams")
        args.vnnlib_base = str(root / "gams_world" / "vnnlib_gams")
    return args


def run_problem(args: argparse.Namespace):
    """Run one problem and return the ABCrown API result object."""
    args = _apply_benchmark_paths(args)
    if args.dual_eval_disable_primal_solvers and not args.inject_primal_pt:
        raise ValueError("--dual_eval_disable_primal_solvers requires --inject_primal_pt (incumbent objective .pt).")
    config = _build_config(args)
    config_dict = config.to_dict()
    if getattr(args, "loader", "vnnlib") == "legacy_fast" and not getattr(args, "scip_cuts_file", None):
        return _run_gear(args, config_dict)
    _set_runtime_environment(config_dict)
    _nlp = config_dict.get('solver', {}).get('nlp', {})
    global _CONTROLLER_CLI
    _CONTROLLER_CLI = bool(_nlp.get('controller', False) and _nlp.get('enabled', False))
    if _nlp.get('controller', False) and _nlp.get('enabled', False):
        from input_split.native_lane import prewarm
        prewarm()
    arguments.Config.parse_config([])
    arguments.Config["general"]["device"] = config_dict["general"]["device"]

    if getattr(args, "loader", "vnnlib") == "legacy_fast":
        print("[nlp_example] --loader legacy_fast does not support --scip_cuts_file; using the vnnlib loader")
    pt_path, vnnlib_path, actual_model_dir = _resolve_problem_paths(
        args.problem_name,
        model_dir=args.model_dir,
        vnnlib_base=args.vnnlib_base,
        benchmark=args.benchmark,
        benchmark_root=args.benchmark_root,
    )
    input_linear_constraints = _load_input_linear_constraints(pt_path)
    if input_linear_constraints is not None:
        constraint_A, _ = input_linear_constraints
        n_constraints = constraint_A.shape[-2] if constraint_A.ndim >= 2 else constraint_A.shape[0]
        print(f"[nlp_example] Loaded {n_constraints} input linear constraints from {pt_path.name}")

    # ---- Merge SCIP cutting planes (if provided) ----
    if getattr(args, 'scip_cuts_file', None):
        from auto_LiRPA.not_in_release import not_in_release
        not_in_release('--scip_cuts_file (SCIP cut import)')


    spec = VerificationSpec.build_spec(
        vnnlib_path=str(vnnlib_path),
        input_linear_constraints=input_linear_constraints,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.device == "cuda":
        # With the bounds on the GPU, torch's CPU work is the local solvers' one-point
        # evaluations of small models, where extra intra-op threads cost more than they
        # compute: at 8 threads a 133-variable Jacobian took 73 ms instead of 4 ms and
        # the root Ipopt of wastewater05m2 ran out of time; at 1 it converged in 5.9 s.
        torch.set_num_threads(1)
        # MAGMA's batched factorizations allocate their pointer arrays with a raw
        # cudaMalloc and abort the process (assert) when the caching allocator holds the
        # memory: cute_gouldqp2/3 (699 vars) died after the root ALM. cuSOLVER takes its
        # workspace from the caching allocator, so a shortage is a catchable OOM.
        torch.backends.cuda.preferred_linalg_library("cusolver")

    model_class_name = _stem_to_class_name(args.problem_name)
    models_mod = _load_models_module(args.model_dir)
    model_cls = getattr(models_mod, model_class_name)
    model = model_cls(pt_path=str(pt_path)).to("cuda:0" if args.device == "cuda" else "cpu")

    # Propagate optimization direction from the model into the config system,
    # so downstream code (e.g. BaB solving-mode display) reads from config
    # instead of from model_ori (which may be wrapped).
    if hasattr(model, 'is_maximization'):
        config_dict.setdefault("solving", {})["is_maximization"] = bool(model.is_maximization)

    solver = ABCrownSolver(spec, model, config=config_dict)
    if not getattr(args, "result_json", None):
        return solver.solve()
    from gear import report
    sink = report.install()
    try:
        result = solver.solve()
    finally:
        report.uninstall()
    _write_result_json(args.result_json, result, sink, model_cls, pt_path, input_linear_constraints, spec, config_dict)
    return result


def _leaves(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict) and v:
            yield from _leaves(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def _gear_options(config_dict):
    """The gear Options whose resolved config equals this CLI config (values and types): the preset with the same A67
    setting (gear-f0, or the historical ref505 when the config leaves A67 off) plus every differing leaf as an
    advanced key, under the CLI's clock (timing="legacy")."""
    import ast
    from gear.config import Options, resolve, thaw
    from gear.errors import OptionError
    opts = ast.literal_eval(config_dict["model"]["bound_opts"] or "{}")
    a67 = isinstance(opts.get("fp_enclosure"), dict) and opts["fp_enclosure"].get("affine_backward") is True
    preset = "gear-f0" if a67 else "ref505"
    base = dict(_leaves(thaw(resolve(Options(preset=preset, timing="legacy")).config)))
    advanced = {k: v for k, v in _leaves(config_dict)
                if k not in base or type(base[k]) is not type(v) or base[k] != v}
    options = Options(preset=preset, timing="legacy", advanced=advanced)
    if thaw(resolve(options).config) != config_dict:
        raise OptionError("the gear options do not reproduce the CLI config")
    return options


def _run_gear(args, config_dict):
    """``--loader legacy_fast``: the frozen artifact through ``gear.solve`` (direct:legacy, timing="legacy")."""
    import types
    import gear
    global _CONTROLLER_CLI
    nlp = config_dict.get('solver', {}).get('nlp', {})
    _CONTROLLER_CLI = bool(nlp.get('controller', False) and nlp.get('enabled', False))
    if args.benchmark and args.benchmark_root:
        src = gear.read_legacy(args.problem_name, root=_resolve_path(args.benchmark_root), family=args.benchmark)
    else:
        from gear.io.legacy import resolve_paths
        pt, vnnlib, md, stem = resolve_paths(args.problem_name, model_dir=args.model_dir, vnnlib_base=args.vnnlib_base)
        src = gear.Legacy(md / f"{stem}_model.py", pt, vnnlib, args.problem_name)
    result = gear.solve(src, _gear_options(config_dict))
    if getattr(args, "result_json", None):
        import json
        Path(args.result_json).write_text(json.dumps(result.to_json(), sort_keys=True, indent=1) + "\n")
    return types.SimpleNamespace(status=result.internal_status, success=result.internal_status != "unknown",
                                 gear=result)


def _write_result_json(path, result, sink, model_cls, pt_path, input_linear_constraints, spec, config_dict):
    """Map what the sink saw and the solver's status to a ResultRecord, replay the incumbent on a fresh float64
    CPU copy of the model (the 'float64 + margin' predicate), and write the JSON. Prints nothing."""
    import json
    from gear import report
    nlp = config_dict["solver"]["nlp"]
    sense = "max" if config_dict.get("solving", {}).get("is_maximization") else "min"
    sign = -1.0 if sense == "max" else 1.0
    primal = sink.primal if sink.primal is not None else (None if sink.objective is None else sign * sink.objective)
    x = None if sink.x is None else {f"X_{i}": v for i, v in enumerate(sink.x)}
    tol = float(nlp["constr_viol_tol"])
    replay = None
    if sink.x is not None:
        mdl = model_cls(pt_path=str(pt_path)).to("cpu").double().eval()
        xt = torch.tensor([sink.x], dtype=torch.float64)
        with torch.no_grad():
            y = mdl(xt)[0].numpy()
        A = b = None
        if input_linear_constraints is not None:
            A, b = input_linear_constraints
            A = (A.to_dense() if A.is_sparse else A).double().numpy()
            b = b.double().flatten().numpy()
        lo = spec.input_spec.lower.detach().cpu().double().flatten().numpy()
        hi = spec.input_spec.upper.detach().cpu().double().flatten().numpy()
        replay = report.replay_float64(y, A, b, sink.x, lo, hi, tol)
    internal = str(result.status)
    extra = ()
    if internal not in ("verified", "unknown"):
        internal, extra = "error", (f"internal_status:{result.status}",)
    record = report.map_result(
        internal, primal, sink.dual, sense=sense, replay=replay, x=x,
        gap_abs=float(nlp["bab_primal_dual_abs_tol"]), gap_rel=float(nlp["bab_primal_dual_rel_tol"]),
        events=tuple(sink.events), incumbent_ever=sink.x is not None)
    if extra:
        from dataclasses import replace
        record = replace(record, anomalies=record.anomalies + extra)
    Path(path).write_text(record.dumps())


def main() -> int:
    args = _build_arg_parser().parse_args()
    result = run_problem(args)
    print(f"problem_name={args.problem_name} status={result.status} success={result.success}")
    return 0


def _finish_cli(status):
    if _CONTROLLER_CLI:
        # Solver continuations, native tasks and exported files are already
        # closed before main returns. Reap an unused early warm pool too, then
        # let the OS reclaim process-global Torch/C++ caches. CPython module
        # teardown is not a solver atom and can otherwise outlive its deadline.
        from input_split.native_lane import _close_unused
        _close_unused()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)
    raise SystemExit(status)


if __name__ == "__main__":
    _finish_cli(main())
