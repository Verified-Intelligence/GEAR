"""Public facade v1 without the builder (G5 WP13a, DESIGN sections 1, 2, 6). CPU only.

Unit tests run in-process; the real-path checks (handoff, wrapper variant, timing, reuse, callbacks, CPU-solver
provenance) run ``tests/gear_wp13a_probe.py`` in a fresh interpreter, as the WP5 CLI test does."""
import dataclasses
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import gear as gr
from gear import config as gconfig, errors, spec as gspec
from gear.compile import compiled_hash

CV = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("GEAR_BENCH", str(__import__("pathlib").Path(__file__).resolve().parents[3] / "benchmarks")))
have_bench = pytest.mark.skipif(not (ROOT / "gams_world").exists(), reason="benchmark artifacts missing")
CPU = dict(device="cpu", timing="legacy", time_limit=40.0)


def probe(mode, timeout=900, **env):
    proc = subprocess.run([sys.executable, "-W", "ignore", str(CV / "tests" / "gear_wp13a_probe.py"), mode], cwd=CV,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", **env),
                          capture_output=True, text=True, timeout=timeout)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("PROBE ")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-2000:]
    return json.loads(lines[-1][len("PROBE "):])


# ---- errors ------------------------------------------------------------------------------------------------------
def test_error_hierarchy():
    names = ["ParseError", "UnsupportedModelError", "ModelError", "UnsupportedProblemError", "UnsupportedOperatorError",
             "OptionError", "LegacyArtifactError", "InternalError"]
    for n in names:
        assert issubclass(getattr(errors, n), errors.GearError) and getattr(gr, n) is getattr(errors, n)
    assert gconfig.OptionError is errors.OptionError and issubclass(errors.OptionError, ValueError)
    assert gspec.LegacyArtifactError is errors.LegacyArtifactError
    e = errors.ParseError("bad token", path="m.mod", line=3, col=7)
    assert (e.path, e.line, e.col, str(e)) == ("m.mod", 3, 7, "bad token")
    with pytest.raises(errors.GearError):
        gconfig.resolve(gr.Options(advanced={"no.such.key": 1}))


# ---- options -----------------------------------------------------------------------------------------------------
def test_public_options_are_the_resolver_types_and_unset_keeps_the_preset():
    from gear import options
    assert options.Options is gconfig.Options and gr.UNSET is gconfig.UNSET and gr.DEFAULT_PRESET == "gear-f0"
    base = gconfig.thaw(gr.Options().resolved())
    assert gconfig.thaw(gr.Options(time_limit=gr.UNSET, seed=gr.UNSET, gap=gr.Gap()).resolved()) == base
    c = gconfig.thaw(gr.Options(time_limit=12.0, feas_tol=1e-7).resolved())
    assert c["bab"]["override_timeout"] == 12.0
    assert {c["solver"]["nlp"][k] for k in ("constr_viol_tol", "bab_feas_tol", "ipopt_constr_viol_tol")} == {1e-7}
    with pytest.raises(gr.OptionError):
        gr.Options(time_limit=5.0, advanced={"bab.override_timeout": 6.0}).resolved()


def test_advanced_is_deep_frozen():
    o = gr.Options(advanced={"solver.nlp.ipopt_bab_topk": 3, "x": {"y": [1, 2]}})
    with pytest.raises(TypeError):
        o.advanced["solver.nlp.ipopt_bab_topk"] = 4
    assert isinstance(o.advanced["x"], MappingProxyType) and o.advanced["x"]["y"] == (1, 2)
    with pytest.raises(TypeError):
        o.advanced["x"]["y"] = ()
    r = gr.Options().resolved()
    with pytest.raises(TypeError):
        r["bab"]["override_timeout"] = 1.0
    with pytest.raises(gr.OptionError):
        gr.Options(advanced={"k": object()})
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.time_limit = 3.0


def test_solve_option_fields_replace_options(monkeypatch):
    seen = []
    monkeypatch.setattr(gr.Solver, "solve", lambda self, src: seen.append(self.options) or "r")
    assert gr.solve("x", gr.Options(seed=1), time_limit=7.0) == "r"
    assert seen[0].seed == 1 and seen[0].time_limit == 7.0


# ---- results -----------------------------------------------------------------------------------------------------
def _result(**kw):
    stats = gr.Stats(phases={"solve": 1.0}, deadline_overshoot=None, op_budgets={"hidden_squares": (3, 8, False)},
                     nodes=2, peak_gpu_mb=0.0, rejected_incumbents=0)
    prov = gr.Provenance(path="direct:legacy", source={"f": "ab"}, compiled_hash="c", program_hash="p", pipeline=(),
                         preset="gear-f0", preset_sha256="s", historical_preset=False, options_hash="o",
                         timing="legacy", op_budgets={"prove_rows": 8}, op_budget_commit="partial",
                         op_budget_commit_source="schema_default", integer_receipt=None, open_route=False,
                         cpu_solvers={"ipopt": 2}, commit="abc", device="cpu")
    base = dict(status=gr.Status.OPTIMAL, termination=gr.Termination.GAP_CLOSED, sense=gr.MAX, primal_bound=3.0,
                dual_bound=3.0, gap_abs=0.0, gap_rel=0.0, x={"X_0": 1.0},
                replay=gr.Replay(True, 1e-6, -1.0, "row 0", 0.0, 0.0, 3.0), certificate=None, stats=stats,
                provenance=prov, anomalies=(), internal_status="verified")
    base.update(kw)
    return gr.Result(**base)


def test_result_json_keeps_the_wp5_keys_and_adds_stats_and_provenance():
    from gear import report
    d = _result().to_json()
    wp5 = report.map_result("verified", 3.0, 3.0, sense="max", replay=report.ReplayRecord(True, 1e-6, -1.0, 0.0),
                            x={"X_0": 1.0}, gap_abs=1e-5, gap_rel=1e-4).to_json()
    assert set(wp5) - {"schema"} <= set(d) and set(wp5["replay"]) <= set(d["replay"])
    assert d["sense"] == "max" and d["stats"]["op_budgets"]["hidden_squares"] == [3, 8, False]
    assert d["provenance"]["cpu_solvers"] == {"ipopt": 2} and json.loads(json.dumps(d)) == d
    d = _result(primal_bound=-math.inf, dual_bound=math.inf, x=None, replay=None,
                status=gr.Status.NO_SOLUTION, termination=gr.Termination.TIME_LIMIT).to_json()
    assert d["primal_bound"] == "-inf" and d["dual_bound"] == "inf" and d["replay"] is None
    assert _result().solved and not _result(status=gr.Status.FEASIBLE).solved


def test_certificate_and_unified_root_are_refused_before_search():
    with pytest.raises(gr.OptionError, match="certificates are not available"):
        gr.Solver(gr.Options(certificate=True)).solve("never read")
    for key, value in (("solver.nlp.unified_root", False), ("bab.branching.input_split.enable", False),
                       ("general.complete_verifier", "mip"), ("solving.solving_mode", False)):
        with pytest.raises(gr.OptionError, match=key):
            gr.Solver(gr.Options(timing="solve", advanced={key: value})).solve("never read")
    with pytest.raises(gr.OptionError, match="certificates"):   # legacy timing accepts the same configs
        gr.Solver(gr.Options(timing="legacy", certificate=True, advanced={key: value})).solve("never read")


def test_clamp_to_solve_deadline_sets_every_root_timer():
    import input_split.batch_branch_and_bound as bbb
    now = time.time()
    root = SimpleNamespace(deadline=now + 100.0, start=now, timeout=100.0, stats=SimpleNamespace(
        timer=SimpleNamespace(deadline=now + 100.0)), controller=True, controller_monotonic_deadline=None,
        ledger=SimpleNamespace(deadline=None))
    bbb.clamp_to_solve_deadline(root, now + 30.0)
    assert root.deadline == now + 30.0 and root.stats.timer.deadline == now + 30.0
    assert abs(root.timeout - 30.0) < 1e-9 and root.ledger.deadline == root.controller_monotonic_deadline
    assert abs(root.controller_monotonic_deadline - (time.monotonic() + 30.0 - (time.time() - now))) < 0.5
    bbb.clamp_to_solve_deadline(root, now + 1000.0)       # a later deadline never extends the root's own
    assert root.deadline == now + 30.0


# ---- compile -----------------------------------------------------------------------------------------------------
@have_bench
def test_legacy_compile_is_deterministic_and_checks_its_rules(tmp_path):
    src = gr.read_legacy("hs021", root=ROOT, family="gams_world")
    a, b = gr.compile(src), gr.compile(src)
    assert a.path == "direct:legacy" and a.hash == b.hash and a.program_hash == b.program_hash
    assert a.problem is None and a.pipeline == () and len(a.source) == 4 and a.sense == gr.MIN  # triple + modules.py
    with pytest.raises(ValueError):
        a.box[0][0] = 1.0                                  # private read-only arrays
    with pytest.raises(gr.UnsupportedModelError):
        gr.compile(src, optimize=True)
    with pytest.raises(gr.UnsupportedModelError):
        gr.compile(src, passes=("canonical",))
    with pytest.raises(gr.OptionError):
        gr.compile(src, passes=("canonical",), optimize=False)
    with pytest.raises(gr.OptionError):
        gr.compile(src, constants="decimal")
    with pytest.raises(gr.ModelError, match="box"):
        gr.compile(src, box=([0.0], [1.0]))
    with pytest.raises(gr.ModelError, match="sense"):
        gr.compile(src, sense=gr.MAX)
    with pytest.raises(gr.UnsupportedModelError):
        gr.compile(str(tmp_path / "m.mod"))
    with pytest.raises(gr.UnsupportedModelError):
        gr.compile(42)
    # a triple outside the manifest (L1): the same bytes under a name the manifest does not list (a byte-identical
    # copy under the listed name is the same artifact under any root: test_gear_v1b_portable.py)
    md = tmp_path / "gams_world" / "mod_files_gams"; vb = tmp_path / "gams_world" / "vnnlib_gams"
    md.mkdir(parents=True); vb.mkdir(parents=True)
    for p in (src.model_py, src.state_pt):
        (md / p.name.replace("hs021", "hs021x")).write_bytes(p.read_bytes())
    (md / "modules.py").write_bytes((src.model_py.parent / "modules.py").read_bytes())
    (vb / "hs021x.vnnlib").write_bytes(src.vnnlib.read_bytes())
    with pytest.raises(gr.LegacyArtifactError, match="not a frozen artifact"):
        gr.compile(gr.read_legacy("hs021x", root=tmp_path, family="gams_world"))


@have_bench
def test_compiled_hash_covers_box_vtype_sense_assignment_feas_tol_and_budgets():
    src = gr.read_legacy("hs021", root=ROOT, family="gams_world")
    c = gr.compile(src)
    lo, hi = c.box
    variants = [dict(box=(lo - 1.0, hi)), dict(vtype=np.zeros(len(lo), dtype=np.int8)), dict(sense=gr.MAX),
                dict(assignment=("objective", 0))]
    for kw in variants:
        assert compiled_hash(dataclasses.replace(c, hash="", **kw)) != c.hash, kw
    for adv in ({"solver.nlp.constr_viol_tol": 1e-5, "solver.nlp.bab_feas_tol": 1e-5,
                 "solver.nlp.ipopt_constr_viol_tol": 1e-5}, {"model.op_budget.hidden_squares": 1 << 19}):
        other = gr.compile(src, options=gr.Options(advanced=adv))
        assert other.program_hash == c.program_hash and other.hash != c.hash
    with pytest.raises(gr.OptionError, match="recompile"):
        gr.solve(c, gr.Options(feas_tol=1e-5))


@have_bench
def test_a_mutated_compiled_is_refused_before_any_model_is_built():
    c = gr.compile(gr.read_legacy("hs021", root=ROOT, family="gams_world"))
    tensor = next(n.param for n in c.program[1] if n.param is not None)
    tensor.view(-1)[0] += 1
    with pytest.raises(gr.ModelError, match="mutated"):
        gr.solve(c, gr.Options(**CPU))
    c = gr.compile(gr.read_legacy("hs021", root=ROOT, family="gams_world"))
    object.__setattr__(c, "box", (c.box[0] - 1.0, c.box[1]))
    with pytest.raises(gr.ModelError, match="mutated"):
        gr.solve(c, gr.Options(**CPU))


def test_torch_sources_go_to_wp12a_compile_torch(monkeypatch):
    """The dispatch hands nn.Module / callable sources to gear.io.torch_direct.compile_torch (stubbed here) and
    copies its TorchCompiled into a Compiled."""
    import types
    import torch
    from gear.spec import RowSpec
    calls = []
    lo, hi = np.zeros(2), np.ones(2)
    lo.setflags(write=False); hi.setflags(write=False)
    tc = SimpleNamespace(path="direct:torch", original=SimpleNamespace(export_sha256="e" * 64), program=((), (), ["/y"], None),
                         program_hash="p" * 64, rowspec=RowSpec(2, np.array([0, 1]), np.array([1]), np.array([1.0]),
                                                                np.array([0.0])),
                         linear=None, structural_outputs=False, box=(lo, hi), vtype=None, sense=1,
                         assignment=("o", 0), reasons=())
    stub = types.ModuleType("gear.io.torch_direct")
    stub.compile_torch = lambda src, **kw: calls.append(kw) or tc
    monkeypatch.setitem(sys.modules, "gear.io.torch_direct", stub)
    c = gr.compile(torch.nn.Linear(2, 2), box=(lo, hi), objective=0, le=(1,))
    assert c.path == "direct:torch" and c.torch_compiled is tc and c.source["export"] == "e" * 64
    kw = calls[0]
    assert kw["objective"] == 0 and kw["le"] == (1,) and kw["sense"] == 1
    assert kw["compile_keys"]["solver.nlp.expand_input_constraints_as_outputs"] is True
    c2 = gr.compile(lambda x: x, box=(lo, hi), objective=0, le=(1,))
    assert c2.hash == c.hash


# ---- real path (fresh interpreter) -------------------------------------------------------------------------------
@have_bench
def test_wrapper_variant_compiles_two_programs_and_refuses_the_other_value():
    d = probe("variant")
    assert d["structural"] == [True, False] and d["program_hash_differ"] and d["hash_differ"]
    assert d["cross"] == ["OptionError:recompile"] * 2
    for own, cols in zip(d["own"], d["expanded_cols"]):
        assert own["program"] and own["build"] and own["width"] == cols


@have_bench
def test_forced_disagreement_at_the_hook_raises_before_lirpanet():
    d = probe("disagree")
    assert d["raised"] == "InternalError" and d["lirpanet"] == 0


@have_bench
def test_option_change_without_linear_rows_also_raises():
    d = probe("nolinear")
    assert d == {"raised": "OptionError", "recompile": True, "structural": False}


@have_bench
def test_legacy_timing_hands_the_cli_config_to_the_solver():
    d = probe("legacy_config")
    assert d["equal"] and d["solve_deadline"] is None and d["override_timeout"] == 180.0
    assert d["status"] == "OPTIMAL" and d["overshoot"] is None and d["timing"] == "legacy"


@have_bench
def test_solve_timing_clamps_every_frozen_timer_to_one_deadline():
    d = probe("timing")
    T = d["T_wall"]
    assert d["root"]["deadline"] == T and d["root"]["timer"] == T
    assert abs(d["root"]["timeout"] - (T - d["root"]["start"])) < 1e-6
    assert len(d["logger"]) == 1 and 0 < d["logger"][0] <= 60.0
    assert d["status"] == "OPTIMAL" and d["overshoot"] < 0 and d["timing"] == "solve"
    assert {"compile", "init", "setup", "solve", "report"} <= set(d["phases"])


@have_bench
def test_a_deadline_inside_the_solver_stops_it_and_records_the_overrun():
    """cute_portfl6 needs about 6 s of solve on this host; with 3 s the deadline falls inside the root or BaB (the
    measurement in REPORT: TIME_LIMIT with an overrun of 0.44 s). Host-dependent by nature."""
    d = probe("tight", PROBE_NAME="cute_portfl6", PROBE_LIMIT="3")
    assert d["termination"] == "TIME_LIMIT" and d["status"] in ("FEASIBLE", "NO_SOLUTION")
    assert d["overshoot"] is not None and -1.0 < d["overshoot"] < 3.0
    assert {"compile", "init", "setup", "solve", "report"} <= set(d["phases"])


@have_bench
def test_a_compile_that_ends_after_the_deadline_gives_no_compiled(monkeypatch):
    import gear.solver as gs
    clock = [1000.0]
    monkeypatch.setattr(gs, "_monotonic", lambda: clock[0])

    def slow_compile(src, options=None):
        clock[0] += 61.0
        return gr.compile(src, options=options)
    monkeypatch.setattr(gs, "_compile", slow_compile)
    r = gs.Solver(gr.Options(device="cpu", timing="solve", time_limit=60.0)).solve(
        gr.read_legacy("hs021", root=ROOT, family="gams_world"))
    assert (r.status, r.termination) == (gr.Status.NO_SOLUTION, gr.Termination.TIME_LIMIT)
    assert r.provenance.compiled_hash == "" and r.x is None and r.stats.deadline_overshoot == 1.0
    assert r.primal_bound == math.inf and r.dual_bound == -math.inf and "compile" in r.stats.phases


@have_bench
def test_solve_many_reuses_the_cpu_template_and_falls_back_on_shared_storage():
    d = probe("many")
    assert d["keys"][0] == d["keys"][1] == d["one"] and d["keys"][0][0] == "OPTIMAL"
    assert d["reuse"] == ["none", "template"] and d["fallback"] == ["none", "fresh_fallback"]
    assert d["fallback_keys"][1] == d["one"]


@have_bench
def test_callbacks_and_a_raising_callback_interrupts():
    d = probe("callbacks")
    assert d["status"] == "OPTIMAL" and d["n_bounds"] >= 1 and d["last_bound"] == d["D"]
    assert d["incumbent"][0] == d["P"]
    assert d["interrupted"][1] == "INTERRUPTED" and "interrupted:RuntimeError" in d["anomalies"]


@have_bench
def test_cpu_solver_provenance_and_the_cpu_solver_free_anomaly(tmp_path):
    d = probe("cpu_free", PROBE_TMP=str(tmp_path))
    assert set(d["cpu"]) == {"linprog", "highs", "ipopt", "lanes", "native"} and sum(d["cpu"].values()) > 0
    assert d["anomalies0"] == []
    assert sorted(d["anomalies1"]) == sorted(f"cpu_solver_used:{k}={n}" for k, n in d["cpu"].items() if n)
    assert d["status"] == ["OPTIMAL", "OPTIMAL"] and {"stats", "provenance"} <= set(d["json_keys"])


# ---- CLI ---------------------------------------------------------------------------------------------------------
def test_cli_parses_options():
    from gear import cli
    a = cli.build_parser().parse_args(["solve", "--legacy", "hs021", "--family", "gams_world", "--root", "/r",
                                       "--time_limit", "5", "--set", "solver.nlp.ipopt_bab_topk=3",
                                       "--set", "general.device='cpu'"])
    o = cli.options_from_args(a)
    assert o.timing == "legacy" and o.time_limit == 5.0 and o.preset == gr.DEFAULT_PRESET
    assert dict(o.advanced) == {"solver.nlp.ipopt_bab_topk": 3, "general.device": "cpu"}
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["solve", "--family", "gams_world", "--root", "/r"])


@have_bench
def test_cli_and_nlp_example_shim_solve_a_frozen_artifact(tmp_path):
    out = tmp_path / "r.json"
    proc = subprocess.run([sys.executable, "-W", "ignore", "-m", "gear.cli", "solve", "--legacy", "hs021", "--family",
                           "gams_world", "--root", str(ROOT), "--device", "cpu", "--time_limit", "40",
                           "--result_json", str(out)], cwd=CV, capture_output=True, text=True, timeout=600,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"))
    assert proc.returncode == 0, proc.stderr[-1000:]
    assert proc.stdout.splitlines()[-1] == "problem_name=hs021 status=verified success=True"
    d = json.loads(out.read_text())
    assert d["status"] == "OPTIMAL" and d["provenance"]["path"] == "direct:legacy" and d["replay"]["ok"]
    f3 = __import__("pathlib").Path(__file__).resolve().parents[1] / "exp_configs" / "NLP_solver" / "f3.yaml"
    if f3.exists():
        out2 = tmp_path / "s.json"
        proc = subprocess.run([sys.executable, "-W", "ignore", "nlp_example.py", "--problem_name", "hs021", "--benchmark",
                               "gams_world", "--benchmark_root", str(ROOT), "--config", str(f3), "--device", "cpu",
                               "--use_beta", "False", "--override_timeout", "40", "--batch_size", "1", "--loader",
                               "legacy_fast", "--result_json", str(out2)], cwd=CV, capture_output=True, text=True,
                              timeout=600, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"))
        assert proc.returncode == 0, proc.stderr[-1000:]
        d2 = json.loads(out2.read_text())
        assert (d2["status"], d2["primal_bound"], d2["dual_bound"], d2["x"]) == \
            (d["status"], d["primal_bound"], d["dual_bound"], d["x"])
        assert d2["provenance"]["options_hash"] == d["provenance"]["options_hash"]
        assert d2["provenance"]["compiled_hash"] == d["provenance"]["compiled_hash"]
