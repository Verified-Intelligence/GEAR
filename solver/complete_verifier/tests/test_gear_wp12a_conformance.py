"""DIRECT_TABLE conformance (G5 WP12a, DESIGN section 2 'Conformance'). CPU, float64.

Each accepted form has a one-op test: E_prog (the BoundedModule built from the emitted program, optional rewrites off)
vs E_ref (the exported program as a module), bitwise, on a grid of edge values (0, -0, +-1, poles, +-1e+-300,
subnormals). A comparison is made only where E_ref is finite: a NaN-vs-NaN "match" is never asserted, and every test
checks that it compared at least one finite value. Each form is also compared with today's ``parse_module`` tuples,
names excluded. Each predicate has a refusal test; the two removed forms have a test that shows why.
"""
import itertools

import numpy as np
import pytest
import torch
from torch import nn

from gear.errors import UnsupportedModelError
from gear.io import torch_direct as td
from gear.lower import emit
from tests.gear_wp12a_helpers import EDGE, bits, evaluators, f64, isomorphic, parse_program


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


class Fn(nn.Module):
    def __init__(self, fn, **buffers):
        super().__init__()
        self.fn = fn
        for name, value in buffers.items():
            self.register_buffer(name, value)

    def forward(self, x):
        return self.fn(self, x)


def _check(module, rows, *, elementwise=True, iso=True, min_compared=1):
    """Bitwise E_prog == E_ref wherever E_ref is finite (per element if elementwise, else per row)."""
    rows = [f64(r).reshape(1, -1) for r in rows]
    built = {}
    compared = 0
    for x in rows:
        if x.shape[1] not in built:
            x0 = torch.full_like(x, 0.25)
            built[x.shape[1]] = (*evaluators(module, x0), x0)
        e_prog, e_ref, program, x0 = built[x.shape[1]]
        with torch.no_grad():
            ref, got = e_ref(x), e_prog(x)
        assert ref.shape == got.shape
        finite = torch.isfinite(ref)
        if not elementwise:
            if not bool(finite.all()):
                continue
            finite = torch.ones_like(finite)
        mismatch = finite & (bits(ref) != bits(got))
        assert not bool(mismatch.any()), (
            f"E_prog != E_ref at {x[0, :8].tolist()}: ref {ref[mismatch][:4].tolist()} got {got[mismatch][:4].tolist()}")
        compared += int(finite.sum())
    assert compared >= min_compared, "the conformance test compared nothing (all E_ref values non-finite)"
    if iso:
        for _, _, program, x0 in built.values():
            assert isomorphic(program, parse_program(module, x0)), "not isomorphic to parse_module's tuples"
    return program


_rng = np.random.default_rng(12)
RANDOM = (_rng.standard_normal(256) * 10.0 ** _rng.integers(-6, 6, 256)).tolist()  # fixed seed; beyond the edges
EDGE_ROW = [EDGE, RANDOM]
P = len(EDGE)
PAIRS = [[a for a, _ in itertools.product(EDGE, EDGE)] + [b for _, b in itertools.product(EDGE, EDGE)],
         (RANDOM * 4)[:P * P] + (RANDOM[::-1] * 4)[:P * P]]

UNARY = {
    "exp": torch.exp, "log": torch.log, "sqrt": torch.sqrt, "sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
    "atan": torch.atan, "tanh": torch.tanh, "abs": torch.abs, "relu": torch.relu, "sigmoid": torch.sigmoid,
    "neg": torch.neg, "reciprocal": torch.reciprocal,
}


@pytest.mark.parametrize("name", sorted(UNARY))
def test_unary(name):
    fn = UNARY[name]
    _check(Fn(lambda s, x: fn(x)), EDGE_ROW)


@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_binary_tensor_tensor(op):
    fn = {"add": torch.add, "sub": torch.sub, "mul": torch.mul}[op]
    _check(Fn(lambda s, x: fn(x[:, :P * P], x[:, P * P:])), PAIRS)


@pytest.mark.parametrize("op", ["add", "sub", "mul"])
@pytest.mark.parametrize("form", ["scalar_float", "scalar_int", "buffer_right", "buffer_left"])
def test_binary_constant_operand(op, form):
    fn = {"add": torch.add, "sub": torch.sub, "mul": torch.mul}[op]
    body = {"scalar_float": lambda s, x: fn(x, 2.5), "scalar_int": lambda s, x: fn(x, 3),
            "buffer_right": lambda s, x: fn(x, s.c), "buffer_left": lambda s, x: fn(s.c, x)}[form]
    for row in EDGE_ROW:
        _check(Fn(body, c=f64(row[::-1])), [row])


@pytest.mark.parametrize("exponent", [2, 4, 5, -3, 1.5, 2.5, 0, 1, 2.0, 4.0, 1.0 / 3])
def test_pow_constant_exponent(exponent):
    _check(Fn(lambda s, x: x ** exponent), EDGE_ROW)


@pytest.mark.parametrize("exponent", sorted(emit.POW_NOT_BITWISE))
def test_pow_removed_exponents(exponent, monkeypatch):
    # torch's pow(Tensor, Scalar) kernel special-cases these exponents (x*x*x, 1/x, 1/(x*x), sqrt, rsqrt); BoundPow
    # computes pow(x, 0-d tensor) with std::pow, so the two differ in the last bit. Refused by the predicate:
    with pytest.raises(UnsupportedModelError, match="exponent"):
        _compile(_one_output(lambda s, x: x ** exponent))
    monkeypatch.setattr(emit, "POW_NOT_BITWISE", frozenset())
    assert _any_mismatch(Fn(lambda s, x: x ** exponent), [RANDOM])


@pytest.mark.parametrize("op", ["minimum", "maximum"])
def test_minmax_both_input_dependent(op):
    fn = getattr(torch, op)
    _check(Fn(lambda s, x: fn(x[:, :P * P], x[:, P * P:])), PAIRS)


def test_clamp_constant_bounds():
    _check(Fn(lambda s, x: torch.clamp(x, -1.0, 2.0)), EDGE_ROW)


W = torch.linspace(-1.5, 2.0, 3 * 6, dtype=torch.float64).reshape(3, 6) / 7
B = f64([0.25, -1.0, 3.5])
MODERATE = [[0.5, -1.25, 3.0, 0.0, -0.0, 7.5], [1e-300, -2.0, 1e3, -1e3, 0.1, 5e-324], [1e150, 1e150, 0, 0, 0, 0]]


@pytest.mark.parametrize("form", ["linear_bias", "linear_nobias", "nn_linear", "mm_t", "mm_plain", "matmul_t",
                                  "matmul_plain", "addmm_t"])
def test_dense_forms(form):
    body = {
        "linear_bias": lambda s, x: nn.functional.linear(x, s.W, s.b),
        "linear_nobias": lambda s, x: nn.functional.linear(x, s.W),
        "nn_linear": None,
        "mm_t": lambda s, x: torch.mm(x, s.W.t()),
        "mm_plain": lambda s, x: torch.mm(x, s.Wt),
        "matmul_t": lambda s, x: torch.matmul(x, s.W.t()),
        "matmul_plain": lambda s, x: x @ s.Wt,
        "addmm_t": lambda s, x: torch.addmm(s.b, x, s.W.t()),
    }[form]
    if body is None:
        module = nn.Linear(6, 3)
        with torch.no_grad():
            module.weight.copy_(W)
            module.bias.copy_(B)
    else:
        module = Fn(body, W=W.clone(), b=B.clone(), Wt=W.t().contiguous())
    _check(module, MODERATE, elementwise=False)


def test_linear_on_3d_input_with_bias():
    _check(Fn(lambda s, x: nn.functional.linear(x.view(1, 2, 3), s.W[:, :3], s.b).reshape(1, 6),
              W=W.clone(), b=B.clone()), MODERATE, elementwise=False)


IDX = torch.tensor([3, 0, 2, 2, 5], dtype=torch.int64)


@pytest.mark.parametrize("form", ["index_select", "index_tensor", "select_int", "select_negative", "index_add"])
def test_index_forms(form):
    body = {
        "index_select": lambda s, x: torch.index_select(x, 1, s.idx),
        "index_tensor": lambda s, x: x[:, s.idx],
        "select_int": lambda s, x: x[:, 2].unsqueeze(1),
        "select_negative": lambda s, x: x[:, -1].unsqueeze(1),
        "index_add": lambda s, x: s.base.index_add(1, s.idx, x[:, :5]),
    }[form]
    _check(Fn(body, idx=IDX.clone(), base=f64([[0.5, -0.0, 0.0, 1e300, -2.0, 3.0]])), MODERATE + [EDGE[:6]])


SHAPES = {
    "cat": lambda s, x: torch.cat([x, x[:, :2] * 2.0], dim=1),
    "cat_negative_dim": lambda s, x: torch.cat([x[:, 3:], x[:, :3]], dim=-1),
    "view_reshape": lambda s, x: x.view(1, 2, 3).reshape(1, 6),
    "unsqueeze_squeeze": lambda s, x: x.unsqueeze(0).squeeze(0) * 1.0,
    "unsqueeze_mid": lambda s, x: (x.unsqueeze(1) * s.c6.view(1, 1, 6)).squeeze(1),
    "slice": lambda s, x: x[:, 1:4],
    "slice_open_end": lambda s, x: x[:, 2:],
    "expand": lambda s, x: x[:, :1].expand(1, 6) + x,
    "expand_minus_one": lambda s, x: x.view(1, 6, 1).expand(-1, -1, 2).reshape(1, 12),
    "permute": lambda s, x: x.view(1, 2, 3).permute(0, 2, 1).reshape(1, 6),
    "transpose": lambda s, x: x.view(2, 3).transpose(0, 1).reshape(1, 6),
    "t": lambda s, x: x.view(2, 3).t().reshape(1, 6),
    "sum_keepdim": lambda s, x: x.sum(dim=1, keepdim=True),
    "sum_nokeep": lambda s, x: x.view(1, 2, 3).sum(dim=2),
    "to_copy": lambda s, x: torch.ops.aten._to_copy.default(x, dtype=torch.float64),
    "flatten": lambda s, x: torch.flatten(x.view(1, 2, 3), 1),
}
NO_ISO = {"to_copy"}  # parse_module traces _to_copy as a no-op Cast chain with another name; values checked below


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_shape_forms(name):
    elementwise = name not in ("sum_keepdim", "sum_nokeep")
    _check(Fn(SHAPES[name], c6=f64([1.0, -1.0, 0.5, 2.0, -0.0, 3.0])), MODERATE + [EDGE[:6], EDGE[6:12]],
           elementwise=elementwise, iso=name not in NO_ISO)


def test_nn_flatten_on_flat_input():
    _check(nn.Sequential(nn.Flatten(), nn.Linear(6, 3)), MODERATE, elementwise=False)


def test_every_table_entry_has_a_conformance_test():
    covered = {"add.Tensor", "sub.Tensor", "mul.Tensor", "neg", "reciprocal", "pow.Tensor_Scalar", "minimum",
               "maximum", "clamp", "mm", "addmm", "linear", "matmul", "t", "index_select", "index.Tensor",
               "select.int", "index_add", "cat", "view", "reshape", "unsqueeze", "squeeze.dim", "slice.Tensor",
               "expand", "permute", "transpose.int", "sum.dim_IntList", "_to_copy", "flatten.using_ints",
               *UNARY}
    # _unsafe_view only appears after decompositions; it shares the view/reshape handler.
    assert set(emit.DIRECT_TABLE) - covered == {"_unsafe_view"}


# ---- refusals (T3 predicates) -----------------------------------------------------------------------------------
def _compile(module, n=6, **kw):
    lo, hi = np.full(n, -1.0), np.full(n, 2.0)
    kw.setdefault("objective", 0)
    return td.compile_torch(module, box=(lo, hi), **kw)


def _one_output(fn):
    return Fn(lambda s, x: fn(s, x).sum(dim=1, keepdim=True))


@pytest.mark.parametrize("name, body, match", [
    ("non-constant exponent", lambda s, x: x[:, :3] ** (x[:, 3:] * 0.0 + 2.0), "pow.Tensor_Tensor"),
    ("input-dependent weight", lambda s, x: x @ x.t(), "weight must not depend on the input"),
    ("float32 buffer", lambda s, x: x + s.f32, "float32"),
    ("float32 cast", lambda s, x: torch.ops.aten._to_copy.default(x, dtype=torch.float32).double(), "_to_copy"),
    ("max with a constant", lambda s, x: torch.maximum(x, s.c6), "does not depend on the input"),
    ("min with a constant", lambda s, x: torch.minimum(s.c6, x), "does not depend on the input"),
    ("clamp with tensor bounds", lambda s, x: torch.clamp(x, s.c6, None), "clamp.Tensor"),
    ("dropout", lambda s, x: nn.functional.dropout(x, 0.5, False), "dropout"),
    ("step slice", lambda s, x: x[:, ::2], "step must be 1"),
    ("sum over all", lambda s, x: x.sum().reshape(1, 1), "sum"),
])
def test_refusals(name, body, match):
    module = _one_output(body)
    module.register_buffer("f32", torch.ones(6, dtype=torch.float32))
    module.register_buffer("c6", f64([1.0, -1.0, 0.5, 2.0, -0.0, 3.0]))
    with pytest.raises(UnsupportedModelError, match=match):
        _compile(module)


def test_division_is_rebuild_only():
    with pytest.raises(UnsupportedModelError, match="rebuild:torch not available"):
        _compile(_one_output(lambda s, x: x / 3.0))
    with pytest.raises(UnsupportedModelError, match="optimize=False forbids"):
        _compile(_one_output(lambda s, x: x / 3.0), optimize=False)


def test_symbolic_shape_refused():
    module = Fn(lambda s, x: x * 2.0)
    batch = torch.export.Dim("batch", min=2)
    ep = torch.export.export(module, (torch.ones(2, 6),), dynamic_shapes=({0: batch},))
    with pytest.raises(UnsupportedModelError, match="T2"):
        td.check_signature(ep, 6)
    with pytest.raises(emit.EmitRefused, match="static"):
        emit.emit_program(ep)


# ---- why the removed forms are removed ---------------------------------------------------------------------------
def _trial(monkeypatch, name, handler):
    monkeypatch.setitem(emit._HANDLERS, name, handler)


def _any_mismatch(module, rows):
    rows = [f64(r).reshape(1, -1) for r in rows]
    for x in rows:
        e_prog, e_ref, _ = evaluators(module, torch.full_like(x, 0.25))
        with torch.no_grad():
            ref, got = e_ref(x), e_prog(x)
        finite = torch.isfinite(ref) & torch.isfinite(got)
        if bool((finite & (bits(ref) != bits(got))).any()):
            return True
    return False


def test_removed_division_is_not_bitwise(monkeypatch):
    # The always-on div_to_mul rewrite computes x * (1/y); 1/3 is inexact, so x/3 differs in the last bit.
    _trial(monkeypatch, "div.Tensor", emit._binary("onnx::Div"))
    assert _any_mismatch(Fn(lambda s, x: x / 3.0), [[0.1, 0.7, 1.0, 5.0, 10.0, 7.25]])


def test_removed_max_with_constant_is_not_bitwise(monkeypatch):
    # The always-on minmax_to_relu rewrite computes relu(x - c) + c, which rounds.
    _trial(monkeypatch, "maximum", lambda e, node, op: e.emit("onnx::Max", [e.tensor(a, op) for a in node.args]))
    wide = (np.random.default_rng(13).standard_normal(20000) * 10.0).tolist()
    assert _any_mismatch(Fn(lambda s, x: torch.maximum(x, s.c), c=f64([0.1] * 20000)), [wide])
