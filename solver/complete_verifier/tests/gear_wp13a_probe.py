"""Real-path probes of the WP13a facade, one per mode, run in a fresh interpreter by test_gear_wp13a_facade.py (the
benchmark model packages are imported by name, and the solver sets process-global state). Prints one JSON line.
CPU only: run with CUDA_VISIBLE_DEVICES=''."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]
import torch  # noqa: E402

import gear as gr  # noqa: E402

ROOT = Path(os.environ.get("GEAR_BENCH", str(__import__("pathlib").Path(__file__).resolve().parents[3] / "benchmarks")))
CPU = dict(device="cpu", timing="legacy", time_limit=40.0)
EXPAND = "solver.nlp.expand_input_constraints_as_outputs"


class Stop(BaseException):
    """Ends a probe solve at a sentinel (not an Exception, so the solver's handlers and the facade let it through)."""


def key(r):
    return [r.status.value, r.termination.value, repr(r.primal_bound), repr(r.dual_bound), r.x]


def sentinel_lirpanet(check=None):
    """LiRPANet.__init__ replaced: record what the handoff attached, then stop the solve."""
    import beta_CROWN_solver as bcs
    seen = []

    def init(self, model_ori, in_size, c=None, device=None):
        seen.append(dict(program=hasattr(model_ori, "__gear_program__"), build=hasattr(model_ori, "__gear_build__"),
                         width=int(model_ori(torch.zeros(1, in_size[1], dtype=torch.float64)).shape[1])))
        raise Stop()
    bcs.LiRPANet.__init__ = init
    return seen


def forbid_bounded_module():
    import auto_LiRPA.bound_general as bg
    import beta_CROWN_solver as bcs

    def boom(*a, **k):
        raise AssertionError("a LiRPANet or BoundedModule was constructed")
    bcs.LiRPANet.__init__ = boom
    bg.BoundedModule.__init__ = boom


def mode_variant():
    """Wrapper variant (XREAD r2 H8) on alkyl (linear rows)."""
    src = gr.read_legacy("alkyl", root=ROOT, family="minlplib_linear")
    out = {}
    comp = {v: gr.compile(src, options=gr.Options(**CPU, advanced={EXPAND: v})) for v in (True, False)}
    out["structural"] = [comp[True].structural_outputs, comp[False].structural_outputs]
    out["program_hash_differ"] = comp[True].program_hash != comp[False].program_hash
    out["hash_differ"] = comp[True].hash != comp[False].hash
    from gear.solver import _expanded_rows
    rows = {v: _expanded_rows(comp[v]) for v in (True, False)}
    out["expanded_cols"] = [rows[True][0].shape[1], rows[False][0].shape[1]]
    out["own"] = []
    for v in (True, False):
        seen = sentinel_lirpanet()
        try:
            gr.solve(comp[v], gr.Options(**CPU, advanced={EXPAND: v}))
        except Stop:
            pass
        out["own"].append(seen[0] if seen else None)
    forbid_bounded_module()
    out["cross"] = []
    for v in (True, False):
        try:
            gr.solve(comp[v], gr.Options(**CPU, advanced={EXPAND: not v}))
            out["cross"].append("solved")
        except gr.OptionError as e:
            out["cross"].append("OptionError:" + ("recompile" in str(e)) * "recompile")
    return out


def mode_disagree():
    """A wrapper decision forced to disagree at the hook raises InternalError before LiRPANet."""
    import api
    src = gr.read_legacy("alkyl", root=ROOT, family="minlplib_linear")
    c = gr.compile(src, options=gr.Options(**CPU))
    api.ABCrownSolver._should_expand_input_constraints_as_outputs = lambda self, spec: False
    seen = sentinel_lirpanet()
    try:
        gr.solve(c, gr.Options(**CPU))
        return {"raised": None, "lirpanet": len(seen)}
    except gr.InternalError as e:
        return {"raised": "InternalError", "msg": str(e)[:80], "lirpanet": len(seen)}


def mode_nolinear():
    src = gr.read_legacy("hs021", root=ROOT, family="gams_world")
    c = gr.compile(src, options=gr.Options(**CPU))
    forbid_bounded_module()
    try:
        gr.solve(c, gr.Options(**CPU, advanced={EXPAND: False}))
        return {"raised": None}
    except gr.OptionError as e:
        return {"raised": "OptionError", "recompile": "recompile" in str(e), "structural": c.structural_outputs}


def mode_legacy_config():
    """timing="legacy": the config handed to the frozen solver equals the CLI's (f3 + F0 flags, --device cpu)."""
    import api
    import nlp_example
    seen = []
    orig = api.ABCrownSolver.__init__

    def init(self, *a, config=None, **k):
        seen.append(json.loads(json.dumps(config, default=repr)))
        orig(self, *a, config=config, **k)
    api.ABCrownSolver.__init__ = init
    r = gr.solve(gr.read_legacy("hs021", root=ROOT, family="gams_world"), gr.Options(device="cpu", timing="legacy"))
    argv = ["--problem_name", "hs021", "--config", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "exp_configs" / "NLP_solver" / "f3.yaml"),
            "--device", "cpu", "--use_beta", "False", "--override_timeout", "180", "--batch_size", "1"]
    cli = nlp_example._build_config(nlp_example._build_arg_parser().parse_args(argv)).to_dict()
    cli.setdefault("solving", {})["is_maximization"] = False
    cli = json.loads(json.dumps(cli, default=repr))
    return {"equal": seen[0] == cli, "solve_deadline": seen[0]["bab"]["solve_deadline"],
            "override_timeout": seen[0]["bab"]["override_timeout"], "status": r.status.value,
            "overshoot": r.stats.deadline_overshoot, "timing": r.provenance.timing}


def mode_timing():
    """timing="solve": T_wall clamps _verify_impl's threshold and RootState's fields; the result is in time."""
    import api
    import input_split.batch_branch_and_bound as bbb
    rec = {}
    orig_root = bbb.RootState.__init__

    def root_init(self, *a, **k):
        orig_root(self, *a, **k)
        rec["root"] = dict(deadline=self.deadline, timeout=self.timeout, start=self.start,
                           timer=self.stats.timer.deadline)
    bbb.RootState.__init__ = root_init
    orig_update = api._ApiLogger.update_timeout

    def update(self, t):
        rec.setdefault("logger", []).append(t)
        orig_update(self, t)
    api._ApiLogger.update_timeout = update
    orig_init = api.ABCrownSolver.__init__

    def init(self, *a, config=None, **k):
        rec["T_wall"] = config["bab"]["solve_deadline"]
        orig_init(self, *a, config=config, **k)
    api.ABCrownSolver.__init__ = init
    r = gr.solve(gr.read_legacy("hs021", root=ROOT, family="gams_world"),
                 gr.Options(device="cpu", timing="solve", time_limit=float(os.environ.get("PROBE_LIMIT", "60"))))
    return dict(rec, status=r.status.value, termination=r.termination.value, overshoot=r.stats.deadline_overshoot,
                phases=sorted(r.stats.phases), timing=r.provenance.timing)


def mode_tight():
    """A deadline inside the solve: the run stops after at most one unit and records its overrun."""
    lim = float(os.environ.get("PROBE_LIMIT", "4"))
    r = gr.solve(gr.read_legacy(os.environ.get("PROBE_NAME", "cute_hs071"), root=ROOT,
                                family=os.environ.get("PROBE_FAMILY", "gams_world")),
                 gr.Options(device="cpu", timing="solve", time_limit=lim))
    return dict(status=r.status.value, termination=r.termination.value, overshoot=r.stats.deadline_overshoot,
                phases=r.stats.phases, limit=lim)


def mode_many():
    """solve_many([c, c]) equals two solve(c); the second reuses the template; a forced shared storage falls back."""
    c = gr.compile(gr.read_legacy("hs021", root=ROOT, family="gams_world"), options=gr.Options(**CPU))
    s = gr.Solver(gr.Options(**CPU))
    many = s.solve_many([c, c])
    one = gr.Solver(gr.Options(**CPU)).solve(c)
    import gear.solver as gs
    gs._storages = lambda module: {1}
    fb = gr.Solver(gr.Options(**CPU)).solve_many([c, c])
    return dict(keys=[key(r) for r in many], one=key(one), reuse=[r.stats.reuse for r in many],
                fallback=[r.stats.reuse for r in fb], fallback_keys=[key(r) for r in fb])


def mode_callbacks():
    c = gr.compile(gr.read_legacy("hs021", root=ROOT, family="gams_world"), options=gr.Options(**CPU))
    s = gr.Solver(gr.Options(**CPU))
    bounds, incs = [], []
    s.on_bound(bounds.append)
    s.on_incumbent(lambda v, x: incs.append((v, x)))
    r = s.solve(c)
    s2 = gr.Solver(gr.Options(**CPU))

    def boom(v):
        raise RuntimeError("stop")
    s2.on_bound(boom)
    r2 = s2.solve(c)
    return dict(status=r.status.value, n_bounds=len(bounds), last_bound=bounds[-1] if bounds else None,
                incumbent=incs[-1] if incs else None, P=r.primal_bound, D=r.dual_bound,
                interrupted=[r2.status.value, r2.termination.value], anomalies=list(r2.anomalies))


def mode_cpu_free():
    """Provenance.cpu_solvers and the cpu_solver_used anomaly under a preset that claims cpu_solver_free."""
    import yaml
    from gear.config import PRESET_DIR
    doc = yaml.safe_load((PRESET_DIR / "gear-f0.yaml").read_text())
    doc["preset"]["cpu_solver_free"] = True
    doc["preset"]["name"] = "probe-cpu-free"
    path = Path(os.environ.get("PROBE_TMP") or tempfile.mkdtemp()) / "probe-cpu-free.yaml"
    path.write_text(yaml.safe_dump(doc))
    src = gr.read_legacy("hs021", root=ROOT, family="gams_world")
    r0 = gr.solve(src, gr.Options(**CPU))
    r1 = gr.solve(src, gr.Options(**CPU, preset=str(path)))
    return dict(cpu=dict(r0.provenance.cpu_solvers), anomalies0=list(r0.anomalies), anomalies1=list(r1.anomalies),
                status=[r0.status.value, r1.status.value], json_keys=sorted(r0.to_json()),
                prov_preset=r1.provenance.preset)


if __name__ == "__main__":
    out = globals()["mode_" + sys.argv[1]]()
    sys.stdout.flush()
    print("PROBE " + json.dumps(out, default=repr))
