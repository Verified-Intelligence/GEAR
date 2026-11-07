"""A293 (G5 v1c): under timing="solve" the root NLP is admitted only before the one deadline. Before the fix a root
NLP reached after T ran on its configured budgets (ALM 60 s, IPOPT 30 s; the cap applied only to a positive time
left), and cute_portfl6 overran a 3 s deadline by up to 27 s under host load (stack samples: falsification._run_alm ->
ALMPGDSolver.solve_with_restarts -> _solve_subproblem_adam). The load is replaced here by a forced sleep: the first
root operation sleeps until 0.5 s past T, so the NLP is reached after it. CPU, fresh interpreter."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

CV = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("GEAR_BENCH", str(__import__("pathlib").Path(__file__).resolve().parents[3] / "benchmarks")))
have_bench = pytest.mark.skipif(not (ROOT / "gams_world").exists(), reason="benchmark artifacts missing")

_PROBE = r'''
import json, sys, time
import gear as gr
import input_split.batch_branch_and_bound as bbb
def wrap(run):
    def slow(root, deadline):
        time.sleep(max(0.0, root.deadline - time.time()) + 0.5)    # T passes inside the first root operation
        return run(root, deadline)
    return slow
bbb._ROOT_OPERATIONS = tuple(op._replace(run=wrap(op.run)) if op.name == "reference_bounds" else op
                             for op in bbb._ROOT_OPERATIONS)
r = gr.solve(gr.read_legacy("cute_portfl6", root=sys.argv[1], family="gams_world"),
             gr.Options(device="cpu", timing=sys.argv[2], time_limit=3.0))
print("DEADLINE " + json.dumps(dict(status=r.status.value, termination=r.termination.value,
                                    overshoot=r.stats.deadline_overshoot)))
'''


def _run(timing):
    proc = subprocess.run([sys.executable, "-W", "ignore", "-c", _PROBE, str(ROOT), timing], cwd=CV,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"),
                          capture_output=True, text=True, timeout=900)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("DEADLINE ")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-2000:]
    return json.loads(lines[-1][len("DEADLINE "):]), proc.stdout


@have_bench
def test_a_root_nlp_reached_after_the_deadline_is_not_admitted():
    d, log = _run("solve")
    assert "[Root] NLP not admitted: the solve deadline has passed" in log
    assert d["termination"] == "TIME_LIMIT" and d["status"] in ("FEASIBLE", "NO_SOLUTION")
    assert d["overshoot"] is not None and 0.5 <= d["overshoot"] < 3.0     # 0.5 s is the injected sleep; was ~17 s
    assert "ALM-PGD:" not in log.split("NLP not admitted")[1]                 # no ALM after T


@have_bench
def test_timing_legacy_keeps_the_root_nlp():
    d, log = _run("legacy")
    assert "NLP not admitted" not in log and "ALM-PGD:" in log and d["overshoot"] is None
