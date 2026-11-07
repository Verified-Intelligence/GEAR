"""G5 public v1 integration (WP12a direct torch path through WP13a's facade). CPU only.

1. The whole-commit rebuild reached through the facade's ``__gear_build__`` (``gear.solver._Build``): a fresh build,
   a template reuse and a shared-storage fallback all give the stage-off graph and report the rebuild.
2. End to end: ``gr.compile(nn.Module, box=..., objective=..., le/eq/linear=...)`` -> ``gr.solve`` on small NLPs
   that also exist as frozen artifacts; the torch path and the legacy path agree. Each solve runs in a fresh
   interpreter (the frozen solver keeps a global config), as in test_gear_wp13a_facade.py.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from gear import solver as gs
from tests.test_gear_wp12a_setup import build, opts, signature

CV = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("GEAR_BENCH", str(__import__("pathlib").Path(__file__).resolve().parents[3] / "benchmarks")))
have_bench = pytest.mark.skipif(not (ROOT / "gams_world").exists(), reason="benchmark artifacts missing")
EXHAUSTED = "op_budget_exhausted:hidden_squares:1"


def _calls(store, n):
    """n LiRPANet builds of one solve each (a new _Build per solve, one shared template store)."""
    out = []
    for _ in range(n):
        b = gs._Build(store, "key")
        net = b(lambda o=opts(hidden_squares=1, commit="whole"): build(o), opts(hidden_squares=1, commit="whole"))
        out.append((net, b))
    return out


def test_whole_mode_rebuild_through_the_facade_hook():
    off = signature(build(dict(opts(), hidden_square_rewrite=False)))
    (net, b), = _calls(None, 1)
    assert signature(net) == off and b.anomalies == [EXHAUSTED] and b.reuse == "none"


def test_whole_mode_rebuild_on_template_reuse_and_on_the_shared_storage_fallback(monkeypatch):
    off = signature(build(dict(opts(), hidden_square_rewrite=False)))
    store = {}
    (n1, b1), (n2, b2) = _calls(store, 2)
    assert (b1.reuse, b2.reuse) == ("none", "template")
    assert signature(n1) == signature(n2) == off and b1.anomalies == b2.anomalies == [EXHAUSTED]
    monkeypatch.setattr(gs, "_storages", lambda module: {1})     # every copy "shares" storage: fresh fallback
    (n3, b3), = _calls(store, 1)
    assert b3.reuse == "fresh_fallback" and signature(n3) == off and b3.anomalies == [EXHAUSTED]


# ---- end to end ---------------------------------------------------------------------------------------------------
_E2E = r'''
import json, sys
import numpy as np, torch
import gear as gr
case, path, root = sys.argv[1:4]

class Hs021(torch.nn.Module):     # min 0.01 x0^2 + x1^2 - 100  s.t.  10 x0 - x1 >= 10
    def forward(self, x):
        return 0.01 * x[:, 0:1] ** 2 + x[:, 1:2] ** 2 - 100.0

class Hs071(torch.nn.Module):     # min x0 x3 (x0+x1+x2) + x2  s.t.  x0 x1 x2 x3 >= 25,  sum x^2 = 40
    def forward(self, x):
        x0, x1, x2, x3 = (x[:, i:i + 1] for i in range(4))
        return torch.cat([x0 * x3 * (x0 + x1 + x2) + x2, -(x0 * x1 * x2 * x3) + 25.0,
                          (x ** 2).sum(1, keepdim=True) - 40.0], 1)

f64 = dict(dtype=torch.float64)
cases = {"hs021": (Hs021, dict(box=(np.array([2.0, -50.0]), np.array([50.0, 50.0])), objective=0,
                               linear=(torch.tensor([[-10.0, 1.0]], **f64), torch.tensor([-10.0], **f64)))),
         "hs071": (Hs071, dict(box=(np.ones(4), np.full(4, 5.0)), objective=0, le=[1], eq=[2]))}
o = gr.Options(preset="gear-f0", device="cpu", time_limit=40.0)
if path == "legacy":
    c = gr.compile(gr.read_legacy(case, root=root, family="gams_world"), options=o)
else:
    model, kw = cases[case]
    c = gr.compile(model(), options=o, **kw)
r = gr.solve(c, o)
print("E2E " + json.dumps(dict(status=r.status.value, P=r.primal_bound, D=r.dual_bound, gap=r.gap_abs,
                               replay=None if r.replay is None else r.replay.ok, path=r.provenance.path,
                               structural=c.structural_outputs)))
'''


def _solve(case, path):
    proc = subprocess.run([sys.executable, "-W", "ignore", "-c", _E2E, case, path, str(ROOT)], cwd=CV,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"),
                          capture_output=True, text=True, timeout=900)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("E2E ")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-2000:]
    return json.loads(lines[-1][len("E2E "):])


@have_bench
@pytest.mark.parametrize("case", ["hs021", "hs071"])
def test_torch_module_solves_as_the_legacy_artifact(case):
    t, l = _solve(case, "torch"), _solve(case, "legacy")
    assert (t["path"], l["path"]) == ("direct:torch", "direct:legacy")
    assert t["status"] == l["status"] == "OPTIMAL" and t["replay"] is True
    assert t["structural"] is (case == "hs021")           # linear=(A, b) takes the structural-output variant
    assert abs(t["P"] - l["P"]) <= max(t["gap"], l["gap"]) + 1e-9 * max(1.0, abs(l["P"]))
