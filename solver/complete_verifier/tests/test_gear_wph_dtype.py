"""G5 WP-H (A228): inside a float64 model, no graph node may compute in a lower float precision.

torch's ONNX scalar-type pass types a rank-0 float tensor as a Python number of the default dtype, so under a
float32 default a product of two 0-d float64 tensors ran in float32 (cute_himmelp1: graph = model + 1.3e-6).
``parse_module`` now exports a trace whose floating tensors are all float64 under a float64 default, and
``BoundedModule`` refuses a float64 model whose graph still computes a value in a lower precision.
"""
import math
import os
import re
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from auto_LiRPA import BoundedModule
import auto_LiRPA.parse_graph as pg

f64 = torch.float64


@pytest.fixture
def default_float32():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(old)


class ZeroDProduct(nn.Module):
    """A constant objective term as the legacy artifacts build it: 0-d buffer times 0-d buffer."""

    def __init__(self, dtype=f64):
        super().__init__()
        self.register_buffer('v', torch.tensor(75.1963666677, dtype=dtype))
        self.register_buffer('s', torch.tensor(-1.0, dtype=dtype))

    def forward(self, x):
        return x.sum(-1, keepdim=True) + self.v * self.s


class ZeroDTimesNumber(nn.Module):
    """0-d buffer times a Python float: the pass re-creates the number as a constant of its inferred type."""

    def __init__(self):
        super().__init__()
        self.register_buffer('v', torch.tensor(3.0, dtype=f64))

    def forward(self, x):
        return x * (self.v * 0.1)


class ExplicitFloat32(nn.Module):
    def forward(self, x):
        return (x.float() * 0.1).double() + x


def _points(n, k=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(1, n, generator=g, dtype=f64) * 4 - 2 for _ in range(k)]


def _low_nodes(bm):
    return [n.name for n in bm.nodes()
            if isinstance(n.forward_value, torch.Tensor) and n.forward_value.is_floating_point()
            and n.forward_value.numel() > 0 and n.forward_value.dtype != f64]


@pytest.mark.parametrize('cls', [ZeroDProduct, ZeroDTimesNumber])
def test_zero_d_ops_bitwise_under_float32_default(default_float32, cls):
    model = cls()
    bm = BoundedModule(model, torch.zeros(1, 2, dtype=f64), device='cpu')
    assert torch.get_default_dtype() == torch.float32
    assert _low_nodes(bm) == []
    assert not any(type(n).__name__ == 'BoundCast' for n in bm.nodes())
    with torch.no_grad():
        for x in _points(2):
            assert torch.equal(bm(x), model(x))


def test_float64_default_unchanged():
    old = torch.get_default_dtype()
    torch.set_default_dtype(f64)
    try:
        model = ZeroDProduct()
        bm = BoundedModule(model, torch.zeros(1, 2, dtype=f64), device='cpu')
        with torch.no_grad():
            assert torch.equal(bm(torch.ones(1, 2, dtype=f64)), model(torch.ones(1, 2, dtype=f64)))
    finally:
        torch.set_default_dtype(old)


def test_float32_model_keeps_the_ambient_export(default_float32):
    model = ZeroDProduct(dtype=torch.float32)
    trace, _ = torch.jit._get_trace_graph(model, (torch.zeros(1, 2),))
    assert not pg._trace_is_float64(trace)
    bm = BoundedModule(model, torch.zeros(1, 2), device='cpu')
    x = torch.ones(1, 2)
    with torch.no_grad():
        assert torch.equal(bm(x), model(x))


def test_explicit_float32_cast_is_refused(default_float32):
    with pytest.raises(TypeError, match='float64.*lower precision'):
        BoundedModule(ExplicitFloat32(), torch.zeros(1, 2, dtype=f64), device='cpu')


def test_unrepaired_zero_d_product_is_refused(default_float32, monkeypatch):
    """Without the float64 export, the A228 graph computes in float32; the guard refuses it."""
    monkeypatch.setattr(pg, '_trace_is_float64', lambda graph: False)
    with pytest.raises(TypeError, match='lower precision'):
        BoundedModule(ZeroDProduct(), torch.zeros(1, 2, dtype=f64), device='cpu')


BENCH = Path(os.environ.get('GEAR_BENCH', str(__import__("pathlib").Path(__file__).resolve().parents[3] / "benchmarks")))
HIMMEL = BENCH / 'gams_world' / 'mod_files_gams' / 'cute_himmelp1_model_state.pt'


def _himmelp1():
    import importlib
    md = BENCH / 'gams_world' / 'mod_files_gams'
    for p in (md.parent, md):
        if str(p) in sys.path:
            sys.path.remove(str(p))
        sys.path.insert(0, str(p))
    for k in [k for k in sys.modules if k == 'modules' or k == md.name or k.startswith(md.name + '.')]:
        del sys.modules[k]
    model = importlib.import_module(md.name).CuteHimmelp1Model(pt_path=str(HIMMEL)).double()
    lo, hi = {}, {}
    for line in (BENCH / 'gams_world' / 'vnnlib_gams' / 'cute_himmelp1.vnnlib').read_text().splitlines():
        m = re.match(r'\(assert \((>=|<=) X_(\d+) (\S+)\)\)', line.strip())
        if m:
            (lo if m.group(1) == '>=' else hi)[int(m.group(2))] = float(m.group(3))
    n = max(max(lo), max(hi)) + 1
    L = torch.tensor([lo[i] for i in range(n)], dtype=f64)
    H = torch.tensor([hi[i] for i in range(n)], dtype=f64)
    assert all(math.isfinite(v) for v in (*L.tolist(), *H.tolist()))
    return model, L, H


@pytest.mark.skipif(not HIMMEL.exists(), reason='legacy artifacts not available')
@pytest.mark.parametrize('default', [torch.float32, f64])
def test_cute_himmelp1_graph_equals_model_bitwise(default):
    from gear.io.program_manifest import load_program_manifest, program_hash
    old = torch.get_default_dtype()
    torch.set_default_dtype(default)
    try:
        model, L, H = _himmelp1()
        x0 = ((L + H) / 2).reshape(1, -1)
        # the tuples equal the L4 manifest entry, which was traced under a float64 default
        assert program_hash(*pg.parse_module(model, (x0,))) == load_program_manifest()[1][('cute_himmelp1', False)]
        bm = BoundedModule(model, x0, device='cpu')
        assert _low_nodes(bm) == []
        g = torch.Generator().manual_seed(0)
        points = [x0] + [(L + (H - L) * torch.rand(L.shape, generator=g, dtype=f64)).reshape(1, -1)
                         for _ in range(8)]
        with torch.no_grad():
            for x in points:
                assert torch.equal(bm(x), model(x))
    finally:
        torch.set_default_dtype(old)
