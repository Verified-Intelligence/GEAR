"""G5 WP3: strict vnnlib subset parser and the fast legacy loader (CPU only)."""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gear.io import legacy  # noqa: E402
from gear.spec import LegacyArtifactError, RowSpec, parse_vnnlib_subset  # noqa: E402

ROOT = Path(str(__import__("pathlib").Path(__file__).resolve().parents[3] / "benchmarks"))
needs_root = pytest.mark.skipif(not ROOT.is_dir(), reason="frozen benchmark root not available")

GOOD = """; comment
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 -1.5))
(assert (<= X_0 0.1))
(assert (>= X_1 0.0))
(assert (<= X_1 1e3))
(assert (<= Y_1 0.0))
(assert (>= Y_1 0.0)) ; trailing comment
(assert (>= Y_0 2.5))
"""


def _header(extra=""):
    return ("(declare-const X_0 Real)\n(declare-const Y_0 Real)\n"
            "(assert (>= X_0 0.0))\n(assert (<= X_0 1.0))\n" + extra)


@pytest.fixture(params=[torch.float64, torch.float32], ids=["fp64", "fp32"])
def double_default(request):
    old = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    import arguments
    arguments.Config.parse_config([])
    yield
    torch.set_default_dtype(old)


# ---- parser -----------------------------------------------------------------------------------------
def test_parser_accepts_subset_and_keeps_rows_sparse():
    lo, hi, rs = parse_vnnlib_subset(GOOD)
    assert isinstance(rs, RowSpec) and rs.nrows == 3 and rs.nnz == 3 and rs.num_outputs == 2
    assert rs.indices.tolist() == [1, 1, 0] and rs.data.tolist() == [1.0, -1.0, -1.0]
    assert rs.rhs.tolist() == [0.0, 0.0, -2.5]
    assert np.signbit(rs.rhs[1])  # -1 * 0.0 is -0.0, as in read_vnnlib
    dense = rs.to_dense()
    assert dense.shape == (3, 2) and dense.tolist() == [[0, 1], [0, -1], [-1, 0]]


def test_parser_matches_read_vnnlib_bitwise(tmp_path, double_default):
    from read_vnnlib import read_vnnlib

    text = GOOD.replace("-1.5", "-0.7").replace("0.1", "0.3")
    f = tmp_path / "p.vnnlib"
    f.write_text(text)
    (box, specs), = read_vnnlib(str(f))
    (mat, rhs), = specs
    lo, hi, rs = parse_vnnlib_subset(text)
    assert np.array(box).tobytes() == np.stack([lo, hi], 1).tobytes()
    assert mat.tobytes() == rs.to_dense().tobytes() and rhs.tobytes() == rs.rhs.tobytes()


BAD = {
    "disjunction": _header("(assert (or (and (<= Y_0 1.0))(and (>= Y_0 2.0))))\n"),
    "y_vs_y": "(declare-const Y_1 Real)\n" + _header("(assert (<= Y_0 Y_1))\n"),
    "strict_op": _header("(assert (< Y_0 1.0))\n"),
    "x_vs_y": _header("(assert (<= X_0 Y_0))\n"),
    "multiline": _header("(assert\n (<= Y_0 1.0))\n"),
    "double_space": _header("(assert  (<= Y_0 1.0))\n"),
    "undeclared_y": _header("(assert (<= Y_3 1.0))\n"),
    "undeclared_x": _header("(assert (<= X_2 1.0))\n"),
    "index_gap": _header("(declare-const Y_2 Real)\n"),
    "duplicate_decl": _header("(declare-const Y_0 Real)\n"),
    "leading_zero_index": _header("(declare-const Y_01 Real)\n"),
    "inf_constant": _header("(assert (<= Y_0 inf))\n"),
    "nan_constant": _header("(assert (<= Y_0 nan))\n"),
    "overflow_constant": _header("(assert (<= Y_0 1e999))\n"),
    "underscore_literal": _header("(assert (<= Y_0 1_0))\n"),
    "symbolic": _header("(assert (<= Y_0 (+ 1 2)))\n"),
    "unbounded_x": "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n(assert (>= X_0 0.0))\n",
    "empty_range": "(declare-const X_0 Real)\n(assert (>= X_0 2.0))\n(assert (<= X_0 1.0))\n",
    "no_inputs": "(declare-const Y_0 Real)\n",
    "other_command": _header("(check-sat)\n"),
}


@pytest.mark.parametrize("name", sorted(BAD))
def test_parser_refuses_non_subset(name):
    with pytest.raises(LegacyArtifactError):
        parse_vnnlib_subset(BAD[name])


# ---- loader -----------------------------------------------------------------------------------------
def _copy_artifact(tmp_path, bench, name):
    """A private copy of one artifact laid out like the benchmark (so nothing under ROOT can be written)."""
    sub = {"gams_world": ("mod_files_gams", "vnnlib_gams"),
           "minlplib_linear": ("py_minlplib_linear", "vnnlib_minlplib_linear")}[bench]
    md, vb = tmp_path / bench / sub[0], tmp_path / bench / sub[1]
    md.mkdir(parents=True)
    vb.mkdir()
    if bench == "minlplib_linear":  # its modules.py loads this file by relative path
        (tmp_path / "minlplib" / "mod_files").mkdir(parents=True)
        shutil.copy(ROOT / "minlplib" / "mod_files" / "modules.py", tmp_path / "minlplib" / "mod_files")
    src = ROOT / bench / sub[0]
    for f in (f"{name}_model.py", f"{name}_model_state.pt", "modules.py"):
        shutil.copy(src / f, md / f)
    shutil.copy(ROOT / bench / sub[1] / f"{name}.vnnlib", vb / f"{name}.vnnlib")
    return md, vb


def _reference(md, vb, name):
    """Today's path (nlp_example helpers + read_vnnlib + package-style model import)."""
    import nlp_example as ne
    from api import VerificationSpec

    pt, vn, _ = ne._resolve_problem_paths(name, model_dir=md, vnnlib_base=vb)
    ilc = ne._load_input_linear_constraints(pt)
    spec = VerificationSpec.build_spec(vnnlib_path=str(vn), input_linear_constraints=ilc)
    return pt, vn, ilc, spec


def _same(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    if a.numel():
        w = {8: torch.int64, 4: torch.int32}[a.element_size()]
        assert torch.equal(a.contiguous().view(w), b.contiguous().view(w))


@needs_root
@pytest.mark.parametrize("bench,name", [("gams_world", "Hartman3"), ("gams_world", "cute_gouldqp2"),
                                        ("minlplib_linear", "alkyl"), ("minlplib_linear", "ex8_4_3")])
def test_loader_tensors_identical_and_no_cache_written(tmp_path, double_default, bench, name):
    (tmp_path / "a").mkdir()
    md, vb = _copy_artifact(tmp_path / "a", bench, name)
    got = legacy.load_legacy_fast(name, model_dir=md, vnnlib_base=vb, device="cpu")
    assert not list(tmp_path.rglob("*.compiled")), "the fast loader must not write a cache"
    ref_dir = tmp_path / "b"
    ref_dir.mkdir()
    md2, vb2 = _copy_artifact(ref_dir, bench, name)
    pt, vn, ilc, spec = _reference(md2, vb2, name)
    assert list(ref_dir.rglob("*.compiled")), "sanity: today's loader does write one"

    gi, ri = got.spec.input_spec, spec.input_spec
    _same(gi.lower, ri.lower)
    _same(gi.upper, ri.upper)
    assert (gi.input_linear_constraints is None) == (ri.input_linear_constraints is None)
    if ri.input_linear_constraints is not None:
        for x, y in zip(gi.input_linear_constraints, ri.input_linear_constraints):
            _same(x, y)
    assert got.spec.output_spec.num_outputs == spec.output_spec.num_outputs
    assert len(got.spec.output_spec.clauses) == len(spec.output_spec.clauses)
    for cg, cr in zip(got.spec.output_spec.clauses, spec.output_spec.clauses):
        assert len(cg) == len(cr)
        for (c1, r1), (c2, r2) in zip(cg, cr):
            _same(c1, c2)
            _same(r1, r2)
    if ilc is not None:
        for x, y in zip(got.input_linear_constraints, ilc):
            _same(x, y)

    # the model: same state_dict, same forward
    import importlib
    sys.modules.pop(md2.name, None)
    sys.modules.pop("modules", None)
    sys.path.insert(0, str(md2.parent))
    sys.path.insert(0, str(md2))
    mod = importlib.import_module(f"{md2.name}.{name}_model")
    ref_model = getattr(mod, legacy._stem_to_class_name(name))(pt_path=str(pt))
    s1, s2 = got.model.state_dict(), ref_model.state_dict()
    assert s1.keys() == s2.keys()
    for k in s1:
        _same(s1[k], s2[k])
    x = ((gi.lower + gi.upper) / 2).reshape(1, -1).to(torch.float64)
    _same(got.model(x), ref_model(x))


@needs_root
def test_state_is_read_once(tmp_path, double_default, monkeypatch):
    md, vb = _copy_artifact(tmp_path, "gams_world", "Hartman3")
    calls = []
    real = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: (calls.append(a[0]), real(*a, **k))[1])
    legacy.load_legacy_fast("Hartman3", model_dir=md, vnnlib_base=vb)
    assert len(calls) == 1


@needs_root
def test_hash_mismatch_is_an_error(tmp_path, monkeypatch):
    md, vb = _copy_artifact(tmp_path, "gams_world", "Hartman3")
    def rel(p):
        return str(p.relative_to(tmp_path))

    manifest = {rel(vb / "Hartman3.vnnlib"): "0" * 64}
    monkeypatch.setattr(legacy, "_manifest_cache", (tmp_path, manifest))
    with pytest.raises(LegacyArtifactError, match="sha256"):
        legacy.read_legacy("Hartman3", model_dir=md, vnnlib_base=vb)
    legacy.read_legacy("Hartman3", model_dir=md, vnnlib_base=vb, verify=False)


@needs_root
def test_manifest_matches_frozen_root():
    root, hashes = legacy.load_manifest()
    assert len(hashes) == 505 * 3 + 3
    art = legacy.read_legacy("Hartman3", benchmark="gams_world", benchmark_root=ROOT)  # verifies L1
    assert set(art.sha256) >= {"Hartman3.vnnlib", "Hartman3_model_state.pt", "Hartman3_model.py", "modules.py"}
    lin = legacy.read_legacy("alkyl", benchmark="minlplib_linear", benchmark_root=ROOT)
    assert "minlplib/mod_files/modules.py" in lin.sha256
