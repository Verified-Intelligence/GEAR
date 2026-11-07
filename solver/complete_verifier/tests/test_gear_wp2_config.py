"""Config resolver and presets (G5 WP2, DESIGN section 1). CPU only."""
import os
import pickle
from pathlib import Path

import pytest

import arguments
from gear.config import (Batch, Gap, OptionError, Options, UNSET, load_preset, resolve, thaw)

G52 = Path(os.environ.get("GEAR_G52", "g52"))
F3 = __import__("pathlib").Path(__file__).resolve().parents[1] / "exp_configs" / "NLP_solver" / "f3.yaml"
INT3P = Path(os.environ.get("GEAR_INT3P", str(Path(__file__).resolve().parents[1] / "exp_configs" / "NLP_solver" / "int3p_final.yaml")))


def same(a, b, path=""):
    """Differences between two nested configs, comparing types as well as values."""
    if type(a) is not type(b):
        return [(path, type(a).__name__, type(b).__name__)]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}/{k}", "only in " + ("a" if k in a else "b")))
            else:
                out += same(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, (list, tuple)):
        return [(path, "len")] if len(a) != len(b) else [d for i, (x, y) in enumerate(zip(a, b)) for d in same(x, y, f"{path}[{i}]")]
    return [] if (a == b or a != a and b != b) else [(path, a, b)]


def today(config_yaml, device):
    import nlp_example
    argv = ["--problem_name", "x", "--config", str(config_yaml), "--device", device, "--use_beta", "False",
            "--override_timeout", "180", "--batch_size", "1"]
    return nlp_example._build_config(nlp_example._build_arg_parser().parse_args(argv)).to_dict()


# ---- precedence, validation, freezing -------------------------------------------------------------------------
def test_unset_never_overwrites_a_preset_value():
    base = thaw(resolve(Options()).config)
    assert thaw(resolve(Options(time_limit=UNSET, gap=Gap(), batch=Batch())).config) == base
    assert base["bab"]["override_timeout"] == 180.0 and base["solver"]["batch_size"] == 1


def test_a_set_field_replaces_the_preset_value():
    c = thaw(resolve(Options(time_limit=30.0, seed=7, gap=Gap(abs=1e-3), device="cpu")).config)
    assert (c["bab"]["override_timeout"], c["general"]["seed"], c["general"]["device"]) == (30.0, 7, "cpu")
    assert c["solver"]["nlp"]["bab_primal_dual_abs_tol"] == 1e-3 and c["solver"]["nlp"]["bab_primal_dual_rel_tol"] == 1e-4


def test_batch_domains_int_pins_the_batch():
    c = thaw(resolve(Options(batch=Batch(domains=4))).config)
    assert c["solver"]["batch_size"] == 4 and c["solver"]["auto_enlarge_batch_size"] is False
    auto = thaw(resolve(Options(batch=Batch(domains="auto"))).config)
    assert auto == thaw(resolve(Options()).config)


def test_advanced_key_replaces_and_field_plus_advanced_on_one_key_raises():
    assert thaw(resolve(Options(advanced={"general.seed": 5})).config)["general"]["seed"] == 5
    with pytest.raises(OptionError, match="general.seed"):
        resolve(Options(seed=1, advanced={"general.seed": 2}))
    with pytest.raises(OptionError, match="bab.override_timeout"):
        resolve(Options(time_limit=5.0, advanced={"bab.override_timeout": 6.0}))


@pytest.mark.parametrize("advanced", [{"solver.nlp.no_such_key": 1}, {"solver.nlp": 1}, {"nonsense.key": 1},
                                      {"solver.nlp.bab_primal_dual_abs_tol": "1e-6"},
                                      {"solver.nlp.crown_engine_tiered": 1},
                                      {"solver.nlp.ipopt_bab_topk": 2.0}])
def test_unknown_key_or_wrong_type_raises(advanced):
    with pytest.raises(OptionError):
        resolve(Options(advanced=advanced))


def test_int_is_accepted_for_a_float_key_and_becomes_float():
    c = thaw(resolve(Options(advanced={"solver.nlp.ipopt_bab_timeout": 3})).config)
    assert c["solver"]["nlp"]["ipopt_bab_timeout"] == 3.0 and type(c["solver"]["nlp"]["ipopt_bab_timeout"]) is float


def test_dtype_only_float64():
    resolve(Options(dtype="float64"))
    with pytest.raises(OptionError):
        resolve(Options(dtype="float32"))


def test_deep_freeze():
    o = Options(advanced={"model.bound_opts": "{}", "solver.nlp.crown_engine_tiered": False})
    with pytest.raises(TypeError):
        o.advanced["x"] = 1
    r = resolve(Options()).config
    with pytest.raises(TypeError):
        r["general"]["seed"] = 1
    with pytest.raises(TypeError):
        r["general"] = {}
    # a list becomes a tuple
    lists = [k for k, v in r["solver"]["nlp"].items() if isinstance(v, tuple)]
    assert all(isinstance(r["solver"]["nlp"][k], tuple) for k in lists)
    with pytest.raises(OptionError):
        Options(advanced={"a.b": object()})
    with pytest.raises(OptionError):
        Options(advanced={"a.b": {1: 2}})
    assert Options(advanced={"a.b": [1, [2]]}).advanced["a.b"] == (1, (2,))


def test_the_three_feas_tol_keys_agree():
    c = thaw(resolve(Options(feas_tol=1e-6)).config)["solver"]["nlp"]
    assert c["constr_viol_tol"] == c["bab_feas_tol"] == c["ipopt_constr_viol_tol"] == 1e-6
    with pytest.raises(OptionError, match="feas_tol"):
        resolve(Options(advanced={"solver.nlp.bab_feas_tol": 1e-3}))


def test_a67_is_required_unless_historical():
    resolve(Options())
    bad = "{'domain_aware_ops': True}"
    with pytest.raises(OptionError, match="A67"):
        resolve(Options(advanced={"model.bound_opts": bad}))
    with pytest.raises(OptionError, match="A67"):
        resolve(Options(advanced={"model.bound_opts": "{'fp_enclosure': {'affine_backward': False}}"}))
    r = resolve(Options(preset="ref505", advanced={"model.bound_opts": bad}))
    assert r.historical and r.anomalies == ("historical_preset:ref505:a67_off",)


def test_ref505_is_never_implicit_and_gear_f0_has_no_anomaly():
    assert resolve(Options()).anomalies == () and resolve(Options()).preset == "gear-f0"
    assert not resolve(Options()).historical and not resolve(Options()).cpu_solver_free


def test_unknown_preset_and_timing():
    with pytest.raises(OptionError):
        resolve(Options(preset="nope"))
    with pytest.raises(OptionError):
        resolve(Options(timing="wall"))


def test_preset_provenance_is_recorded():
    _, meta, digest = load_preset("gear-f0")
    assert meta["provenance"]["source_sha256"].startswith("1c4339a2") and meta["provenance"]["base_commit"] == "62094fd6e"
    assert len(digest) == 64
    assert load_preset("ref505")[1]["historical"] is True


# ---- parity with today's effective config ------------------------------------------------------------------------
@pytest.mark.skipif(not F3.exists(), reason="f3.yaml copy missing")
@pytest.mark.parametrize("preset,yaml_path", [("gear-f0", F3), ("ref505", INT3P)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resolver_equals_build_config_values_and_types(preset, yaml_path, device):
    got = thaw(resolve(Options(preset=preset, device=device)).config)
    assert same(got, today(yaml_path, device)) == []


@pytest.mark.skipif(not (G52 / "cfg_old_f3_cuda.pkl").exists(), reason="62094fd6e dumps missing")
@pytest.mark.parametrize("preset,tag", [("gear-f0", "f3"), ("ref505", "int3p")])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resolver_equals_the_config_of_62094fd6e_modulo_new_default_keys(preset, tag, device):
    old = pickle.load(open(G52 / f"cfg_old_{tag}_{device}.pkl", "rb"))
    got = thaw(resolve(Options(preset=preset, device=device)).config)
    diffs = same(got, old)
    # the keys that f3.yaml / int3p_final.yaml already set keep their value; the others appear at the code's default
    new_keys = {f"/solver/nlp/{k}" for k in arguments.ConfigHandler.NLP_GAP_DEFAULTS if k not in old["solver"]["nlp"]}
    assert {d[0] for d in diffs} == new_keys | {"/model/op_budget", "/bab/solve_deadline"} and \
        all(d[1].startswith("only in a") for d in diffs)
    assert got["bab"]["solve_deadline"] is None    # WP13a: the facade's timing="solve" deadline, unset
    for k in new_keys:
        v = arguments.ConfigHandler.NLP_GAP_DEFAULTS[k.rsplit("/", 1)[1]]
        assert got["solver"]["nlp"][k.rsplit("/", 1)[1]] == v
    assert got["model"]["op_budget"] == arguments.ConfigHandler.OP_BUDGET_DEFAULTS


def test_resolver_equals_the_post_parse_config():
    import api
    got = thaw(resolve(Options(device="cpu")).config)
    with api._config_context(got):
        assert same(got, arguments.Config.all_args) == []


def test_configurations_dump_omits_the_new_default_keys():
    h = arguments.ConfigHandler()
    h.parse_config([], verbose=False)
    text = h.dump_config(h.all_args)
    for k in arguments.ConfigHandler.NLP_GAP_DEFAULTS:
        assert k not in text
    assert "op_budget" not in text
    h = arguments.ConfigHandler()
    h.parse_config(["--nlp_ipopt_bab_topk", "5"], verbose=False)
    assert "ipopt_bab_topk: 5" in h.dump_config(h.all_args)
