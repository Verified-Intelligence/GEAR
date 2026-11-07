"""CLI wiring of the result sink and the fast loader (G5 WP5). CPU only."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import nlp_example

ROOT = Path(os.environ.get("GEAR_BENCH", str(__import__("pathlib").Path(__file__).resolve().parents[3] / "benchmarks")))
F3 = __import__("pathlib").Path(__file__).resolve().parents[1] / "exp_configs" / "NLP_solver" / "f3.yaml"


def test_flags_parse_and_default_does_not_flip():
    a = nlp_example._build_arg_parser().parse_args(["--problem_name", "x"])
    assert a.loader == "vnnlib" and a.result_json is None
    a = nlp_example._build_arg_parser().parse_args(["--loader", "legacy_fast", "--result_json", "r.json"])
    assert a.loader == "legacy_fast" and a.result_json == "r.json"
    with pytest.raises(SystemExit):
        nlp_example._build_arg_parser().parse_args(["--loader", "other"])


@pytest.mark.skipif(not (ROOT / "gams_world").exists() or not F3.exists(), reason="benchmark artifacts missing")
@pytest.mark.parametrize("loader", ["vnnlib", "legacy_fast"])
def test_result_json_of_a_small_solve(tmp_path, loader):
    """Runs the real CLI in a subprocess (the benchmark model packages are imported by name, which other tests pollute)."""
    out = tmp_path / "r.json"
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", "nlp_example.py", "--problem_name", "hs021", "--benchmark", "gams_world",
         "--benchmark_root", str(ROOT), "--config", str(F3), "--device", "cpu", "--use_beta", "False",
         "--override_timeout", "40", "--batch_size", "1", "--loader", loader, "--result_json", str(out)],
        cwd=Path(__file__).resolve().parents[1], env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"),
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-500:]
    d = json.loads(out.read_text())
    # legacy_fast goes through gear.solve (WP13a): its Result JSON (schema 2) keeps every key of the WP5 record
    assert d["schema"] == (2 if loader == "legacy_fast" else 1) and f"status={d['internal_status']}" in proc.stdout
    assert d["status"] == "OPTIMAL" and d["termination"] == "GAP_CLOSED" and d["replay"]["ok"]
    assert len(d["x"]) > 0 and float(d["dual_bound"]) <= float(d["primal_bound"]) + 1e-5
