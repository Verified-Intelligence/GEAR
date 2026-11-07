"""Frozen-artifact identity is portable (G5 v1b, kit run 1 failure 1): rule L1 keys every file by its path relative
to the benchmark root it sits in (the directory two levels above the file's directory) plus its SHA-256, not by an
absolute path of the host that wrote the manifest. CPU only; solves run in a fresh interpreter (the frozen solver
keeps a global config, and one process imports one model package per directory name)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import gear as gr

CV = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("GEAR_BENCH", str(__import__("pathlib").Path(__file__).resolve().parents[3] / "benchmarks")))
have_bench = pytest.mark.skipif(not (ROOT / "gams_world").exists(), reason="benchmark artifacts missing")
CASES = {"hs021": "gams_world", "alkyl": "minlplib_linear"}


def _copy(dst: Path, name: str, family: str, new_name: str = None) -> gr.Legacy:
    """Copy one artifact (its triple and the modules.py files it loads) to ``dst`` with the same relative layout."""
    src = gr.read_legacy(name, root=ROOT, family=family)
    files = [src.model_py, src.state_pt, src.vnnlib, src.model_py.parent / "modules.py"]
    if family == "minlplib_linear":
        files.append(ROOT / "minlplib" / "mod_files" / "modules.py")
    for p in files:
        out = dst / p.relative_to(ROOT)
        if new_name and p.name.startswith(src.stem):
            out = out.with_name(new_name + p.name[len(src.stem):])
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, out)
    return gr.read_legacy(new_name or name, root=dst, family=family)


_SOLVE = r'''
import json, sys
import gear as gr
name, family, root = sys.argv[1:4]
o = gr.Options(preset="gear-f0", device="cpu", time_limit=60.0, timing="legacy")
c = gr.compile(gr.read_legacy(name, root=root, family=family), options=o)
r = gr.solve(c, o)
print("PORT " + json.dumps(dict(hash=c.hash, source=dict(c.source), status=r.status.value, P=r.primal_bound,
                                D=r.dual_bound, replay=None if r.replay is None else r.replay.ok)))
'''


def _solve(name, family, root):
    proc = subprocess.run([sys.executable, "-W", "ignore", "-c", _SOLVE, name, family, str(root)], cwd=CV,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"),
                          capture_output=True, text=True, timeout=900)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("PORT ")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-2000:]
    return json.loads(lines[-1][len("PORT "):])


@have_bench
@pytest.mark.parametrize("name", sorted(CASES))
def test_a_copy_under_another_root_compiles_to_the_same_problem_and_solves(tmp_path, name):
    family = CASES[name]
    other = tmp_path / "elsewhere" / "bench"           # a different absolute path from the manifest's host root
    _copy(other, name, family)
    here, there = _solve(name, family, ROOT), _solve(name, family, other)
    assert there["source"] == here["source"] and there["hash"] == here["hash"]   # identity is root-relative
    assert there["status"] == here["status"] == "OPTIMAL" and there["replay"] is True
    assert (there["P"], there["D"]) == (here["P"], here["D"])


@have_bench
@pytest.mark.parametrize("which", ["vnnlib", "state_pt", "model_py", "modules.py", "base modules.py"])
def test_a_modified_byte_is_still_refused(tmp_path, which):
    family = "minlplib_linear" if which == "base modules.py" else "gams_world"
    name = "alkyl" if family == "minlplib_linear" else "hs021"
    src = _copy(tmp_path, name, family)
    path = {"vnnlib": src.vnnlib, "state_pt": src.state_pt, "model_py": src.model_py,
            "modules.py": src.model_py.parent / "modules.py",
            "base modules.py": tmp_path / "minlplib" / "mod_files" / "modules.py"}[which]
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(gr.LegacyArtifactError, match="sha256"):
        gr.compile(src)


@have_bench
def test_an_unlisted_triple_is_still_refused(tmp_path):
    src = _copy(tmp_path, "hs021", "gams_world", new_name="hs021copy")   # same bytes, a name the manifest lacks
    with pytest.raises(gr.LegacyArtifactError, match="not a frozen artifact"):
        gr.compile(src)


@have_bench
def test_a_triple_split_across_two_roots_is_refused(tmp_path):
    src = _copy(tmp_path / "a", "hs021", "gams_world")
    _copy(tmp_path / "b", "hs021", "gams_world")
    mixed = gr.Legacy(src.model_py, src.state_pt, tmp_path / "b" / "gams_world" / "vnnlib_gams" / "hs021.vnnlib")
    with pytest.raises(gr.LegacyArtifactError, match="benchmark root"):
        gr.compile(mixed)


@have_bench
def test_one_process_compiles_the_same_instance_from_two_roots(tmp_path):
    """The model package is registered per directory name: a package from another root is dropped, not reused."""
    a = gr.compile(gr.read_legacy("hs021", root=ROOT, family="gams_world"))
    b = gr.compile(_copy(tmp_path, "hs021", "gams_world"))
    c = gr.compile(gr.read_legacy("hs021", root=ROOT, family="gams_world"))
    assert a.hash == b.hash == c.hash and a.program_hash == b.program_hash
    assert b.original.model_py.is_relative_to(tmp_path) and len(b.source) == 4
