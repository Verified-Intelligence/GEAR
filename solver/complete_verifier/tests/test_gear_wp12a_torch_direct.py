"""Direct torch path (G5 WP12a): rules O1-O3, T1-T4, the value check, ownership and the solver adapter. CPU only."""
import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from gear.config import UNSET
from gear.errors import InternalError, ModelError, OptionError, UnsupportedModelError, UnsupportedProblemError
from gear.io import torch_direct as td
from gear.lower import emit
from tests.gear_wp12a_helpers import bits, f64


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


class MLP(nn.Module):
    def __init__(self, n=3, k=2):
        super().__init__()
        torch.manual_seed(0)
        self.l1, self.l2 = nn.Linear(n, 8), nn.Linear(8, k)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))


class Poly(nn.Module):
    """f = x0^2 + x0*x1 - 3 x1, g1 = x0 + x1 - 1 (<= 0), h = x0^2 - x1 (== 0)."""

    def forward(self, x):
        x0, x1 = x[:, 0:1], x[:, 1:2]
        return torch.cat([x0 ** 2 + x0 * x1 - 3.0 * x1, x0 + x1 - 1.0, x0 ** 2 - x1], dim=1)


class Lifted(nn.Module):
    """A lifted tensor constant (a plain tensor attribute) next to a parameter."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(f64([1.5, -0.5]))
        self.k = f64([0.25, 2.0])

    def forward(self, x):
        return (x * self.w + self.k).sum(dim=1, keepdim=True)


class TrainDep(nn.Module):
    """Deterministic and training-dependent: x * (2 if self.training else 3)."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)

    def forward(self, x):
        return self.lin(x) * (2.0 if self.training else 3.0)


BOX2 = (np.array([-1.0, -2.0]), np.array([2.0, 1.0]))
BOX3 = (np.full(3, -1.0), np.full(3, 1.0))


def compile_(src, box=BOX3, **kw):
    kw.setdefault("objective", 0)
    return td.compile_torch(src, box=box, **kw)


# ---- the direct path ---------------------------------------------------------------------------------------------
def test_mlp_and_polynomial_take_direct_torch():
    tc = compile_(MLP(), le=[1])
    assert tc.path == "direct:torch" and tc.num_inputs == 3 and tc.num_outputs == 2
    assert [n.op for n in tc.program.nodes_op] == ["onnx::Gemm", "onnx::Tanh", "onnx::Gemm"]
    assert tc.value_points.shape == (td.N_POINTS, 3)
    tc = compile_(Poly(), box=BOX2, le=[1], eq=[2])
    assert tc.assignment == td.OutputAssignment(0, (1,), (2,))
    assert tc.rowspec.to_dense().tolist() == [[0, 1, 0], [0, 0, 1], [0, 0, -1]]
    assert [bits(torch.tensor(v)).item() for v in tc.rowspec.rhs] == [0, 0, bits(torch.tensor(-0.0)).item()]


def test_callable_source():
    tc = compile_(lambda x: (x ** 2).sum(dim=1, keepdim=True))
    assert tc.path == "direct:torch"
    y = td.evaluate(tc, [0.5, -1.0, 0.25])
    assert y.tolist() == [0.25 + 1.0 + 0.0625]


def test_compile_is_deterministic():
    a, b = compile_(MLP(), le=[1]), compile_(MLP(), le=[1])
    assert a.original.export_sha256 == b.original.export_sha256 and a.program_hash == b.program_hash


def test_many_strict_exports_in_one_process():
    # torch 2.8 leaves a Dynamo cache entry per strict export; without the release the 9th one fails.
    for _ in range(12):
        compile_(nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)), le=[1])


def test_data_dependent_item_is_refused_without_a_jit_fallback():
    class Item(nn.Module):
        def forward(self, x):
            return x * 2.0 if x[0, 0].item() > 0 else x
    with patch("torch.jit.trace", side_effect=AssertionError("jit")), \
            patch("torch.jit._get_trace_graph", side_effect=AssertionError("jit")):
        with pytest.raises(UnsupportedModelError, match="torch.export failed"):
            compile_(Item(), objective=0, le=[1, 2])
        compile_(MLP(), le=[1])  # the direct path never touches torch.jit


def test_rebuild_is_not_available_until_wp12b():
    with pytest.raises(UnsupportedModelError, match="rebuild:torch not available"):
        compile_(MLP(), le=[1], optimize=True)
    with pytest.raises(UnsupportedModelError, match="rebuild:torch not available"):
        compile_(MLP(), le=[1], passes=("canonical",))
    with pytest.raises(OptionError):
        compile_(MLP(), le=[1], passes=("canonical",), optimize=False)
    with pytest.raises(OptionError):
        compile_(MLP(), le=[1], constants="decimal")
    with pytest.raises(UnsupportedModelError, match="rebuild:torch not available"):
        compile_(lambda x: (x / 3.0).sum(dim=1, keepdim=True))  # T3 failure (div is IMPORT_TABLE only)


# ---- rule O1 -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw, match", [
    (dict(objective=UNSET, le=[1]), "objective is required"),
    (dict(objective=0), r"unassigned \[1\]"),
    (dict(objective=0, le=[1], eq=[1]), r"doubly assigned \[1\]"),
    (dict(objective=0, le=[0, 1]), r"doubly assigned \[0\]"),
    (dict(objective=0, le=[1, 1]), r"doubly assigned \[1\]"),
    (dict(objective=2, le=[1]), "outside the 2 outputs"),
    (dict(objective=0, le=[-3]), "outside the 2 outputs"),
    (dict(objective=True, le=[1]), "one int or None"),
    (dict(objective=0, le=[1.0]), "not an int"),
])
def test_o1_output_assignment(kw, match):
    with pytest.raises(ModelError, match=match):
        compile_(MLP(), **kw)


def test_o1_index_likes_and_feasibility():
    tc = compile_(MLP(k=4), objective=-4, le=slice(1, 3), eq=range(3, 4))
    assert tc.assignment == td.OutputAssignment(0, (1, 2), (3,))
    tc = compile_(MLP(k=2), objective=None, le=[0, 1])
    assert tc.assignment.objective is None and tc.objective_row.tolist() == [0.0, 0.0]


# ---- rule O2 -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("box, match", [
    (None, "box is required"),
    ((np.full(3, -1.0, dtype=np.float32), np.full(3, 1.0)), "float64 required"),
    ((torch.full((3,), -1.0, dtype=torch.float32), torch.ones(3)), "float64 required"),
    ((np.array([-1.0, np.nan, 0.0]), np.ones(3)), "NaN"),
    ((np.array([-1.0, 2.0, 0.0]), np.ones(3)), "lo > hi"),
    ((np.full(2, -1.0), np.ones(3)), "shape"),
    (([-1, -1, 2 ** 60 + 1], [1, 1, 2 ** 61]), "not exactly a float64"),
    (([-1, -1, True], [1, 1, 1]), "Python floats or ints"),
])
def test_o2_box_errors(box, match):
    with pytest.raises(ModelError, match=match):
        compile_(MLP(), box=box, le=[1])


def test_o2_infinite_box_has_no_route_in_v1():
    with pytest.raises(UnsupportedProblemError, match="infinite"):
        compile_(MLP(), box=(np.array([-np.inf, 0.0, 0.0]), np.ones(3)), le=[1])
    with pytest.raises(UnsupportedProblemError, match="in-memory"):
        compile_(MLP(), box=(np.zeros(3), np.array([1.0, np.inf, 1.0])), le=[1],
                 compile_keys={td.OPEN_KEY: True})


def test_o2_accepts_tensors_arrays_and_numbers_without_rounding():
    for box in [(torch.full((3,), -1.0), torch.ones(3)), ([-1, -1.0, -1], [1.0, 1, 1]), BOX3]:
        tc = compile_(MLP(), box=box, le=[1])
        assert tc.box[0].tolist() == [-1.0] * 3 and not tc.box[0].flags.writeable


# ---- rule O3 -----------------------------------------------------------------------------------------------------
def test_o3_vtype():
    with pytest.raises(UnsupportedProblemError, match="no route"):
        compile_(MLP(), box=(np.zeros(3), np.full(3, 3.0)), le=[1], vtype=np.array([0, 1, 0], dtype=np.int8))
    with pytest.raises(UnsupportedProblemError, match="no route"):
        compile_(MLP(), box=(np.zeros(3), np.full(3, 3.0)), le=[1], vtype=np.array([2, 0, 0], dtype=np.int8),
                 compile_keys={td.MINLP_KEY: True})
    with pytest.raises(UnsupportedProblemError, match="integral endpoints"):
        compile_(MLP(), le=[1], box=(np.full(3, -0.5), np.ones(3)), vtype=np.array([1, 0, 0], dtype=np.int8))
    with pytest.raises(ModelError, match="int8"):
        compile_(MLP(), le=[1], vtype=np.array([0, 1, 0]))
    with pytest.raises(ModelError, match="VarType"):
        compile_(MLP(), le=[1], vtype=np.array([0, 3, 0], dtype=np.int8))
    assert compile_(MLP(), le=[1], vtype=np.zeros(3, dtype=np.int8)).vtype is None


# ---- T2 ----------------------------------------------------------------------------------------------------------
def test_t2_signature():
    with pytest.raises(UnsupportedModelError, match="T2"):
        compile_(lambda x: x.sum(dim=1))  # output [1], not [1, k]
    with pytest.raises(UnsupportedModelError, match="T2"):
        compile_(lambda x: (x.float() * 2).sum(dim=1, keepdim=True))

    class Mutates(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("count", torch.zeros(1))

        def forward(self, x):
            self.count.add_(1.0)
            return x[:, :1] + self.count
    with pytest.raises(UnsupportedModelError, match="T2.*mutat"):
        compile_(Mutates())

    class Random(nn.Module):
        def forward(self, x):
            return x[:, :1] + torch.rand(1, 1, dtype=torch.float64) * 0.0
    with pytest.raises(UnsupportedModelError, match="random|export changed|T3"):
        compile_(Random())


# ---- T4: the value check -----------------------------------------------------------------------------------------
def test_candidates_midpoint_and_off_diagonal():
    lo, hi = np.array([-1.0, 0.0, 10.0, -5.0]), np.array([1.0, 4.0, 20.0, 5.0])
    c = td.candidates(lo, hi)
    assert c.shape == (td.N_CANDIDATES, 4) and np.array_equal(c[0], (lo + hi) / 2)
    assert ((c >= lo) & (c <= hi)).all()
    u = td.halton_u(4)
    assert not any(np.all(row == row[0]) for row in u)  # no candidate i >= 1 on the box diagonal
    assert np.array_equal(u, td.halton_u(4))             # fixed seed
    assert td._radical_inverse(6, 2) == 0.375 and td._radical_inverse(5, 3) == 7 / 9


def test_log_exports_from_the_first_valid_halton_point():
    tc = compile_(lambda x: torch.log(x[:, :1]) + x[:, 1:2], box=(np.array([-1.0, -1.0]), np.array([1.0, 1.0])))
    assert tc.value_points[0][0] > 0           # the midpoint (0, 0) is not domain-valid
    c = td.candidates(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    first = next(i for i in range(td.N_CANDIDATES) if c[i][0] > 0)
    assert np.array_equal(tc.value_points[0], c[first]) and first > 0
    assert f"first valid candidate {first}" in tc.reasons[0]


def test_non_finite_everywhere_and_too_few_valid_points():
    with pytest.raises(UnsupportedModelError, match="none of the 64"):
        compile_(lambda x: torch.log(x[:, :1] - 5.0))
    # valid only where x0 > 0.9 on [-1, 1]: a few of 64 candidates
    with pytest.raises(UnsupportedModelError, match="m < 11"):
        compile_(lambda x: torch.log(x[:, :1] - 0.9), box=(np.array([-1.0, -1.0]), np.array([1.0, 1.0])))


def test_nan_never_counts_as_a_match():
    nan_mid = lambda x: torch.sqrt(x[:, :1]) - torch.sqrt(-x[:, :1])  # finite only at x0 == 0 (+-0)
    assert td.domain_valid(nan_mid, np.array([[0.5, 0.0], [0.0, 0.0]])) == [1]
    # Force every candidate "valid": E_ref == caller is NaN == NaN bitwise, which must not pass.
    with patch.object(td, "domain_valid", lambda caller, cands: list(range(len(cands)))):
        with pytest.raises((UnsupportedModelError, InternalError), match="export changed|E_prog"):
            compile_(lambda x: torch.log(x[:, :1]), box=(np.array([-1.0, -1.0]), np.array([1.0, 1.0])))


def test_a_coordinate_permutation_bug_is_caught(monkeypatch):
    real = emit.emit_program

    def permuted(ep, state_order=None):
        program = real(ep, state_order)
        if state_order is None:  # only the solver-facing program (the one E_prog uses)
            return program
        ops = [n._replace(attr={"value": n.attr["value"].flip(0)}) if n.op == "onnx::Constant"
               and n.attr["value"].dtype == torch.int64 and n.attr["value"].numel() == 2 else n
               for n in program.nodes_op]
        return program._replace(nodes_op=ops)

    class Gather(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("idx", torch.tensor([0, 1]), persistent=False)  # an onnx::Constant node
            self.register_buffer("scale", f64([1.0, 10.0]))

        def forward(self, x):
            return torch.index_select(x, 1, self.idx) * self.scale
    compile_(Gather(), box=BOX2, objective=0, le=[1])  # the real emitter passes
    monkeypatch.setattr(td, "emit_program", permuted)
    with pytest.raises(InternalError, match="E_prog != E_ref"):
        compile_(Gather(), box=BOX2, objective=0, le=[1])


def test_export_changing_values_is_refused():
    class Counter(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x[:, :1] * float(self.calls)
    with pytest.raises(UnsupportedModelError, match="export changed the source's values"):
        compile_(Counter())


# ---- ownership ---------------------------------------------------------------------------------------------------
def test_editing_the_caller_after_compile_changes_nothing():
    class Owned(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 1)
            self.scale = 2.0

        def forward(self, x):
            return self.lin(x) * (self.scale if self.training else -1.0)
    src = Owned().train()
    tc = compile_(src, box=BOX2)
    x = np.array([0.5, -0.25])
    before, phash, bytes_ = td.evaluate(tc, x), tc.program_hash, tc.original.export_bytes
    with torch.no_grad():
        src.lin.weight.add_(1.0)
    src.scale = 7.0
    src.eval()
    assert td.evaluate(tc, x).tolist() == before.tolist()
    assert tc.program_hash == phash and tc.original.export_bytes == bytes_
    td.check_integrity(tc)

    captured = {"c": 3.0}
    tc2 = compile_(lambda z: z[:, :1] * captured["c"], box=BOX2)
    captured["c"] = 100.0
    assert td.evaluate(tc2, x).tolist() == [1.5]


def test_mutating_a_compiled_array_or_program_tensor_raises():
    tc = compile_(Poly(), box=BOX2, le=[1], eq=[2], linear=(f64([[1.0, 1.0]]), f64([1.0])))
    td.check_integrity(tc)
    with pytest.raises(ValueError):
        tc.box[0][0] = 5.0  # read-only
    t2 = copy.deepcopy(tc)
    t2.box[0].setflags(write=True)
    t2.box[0][0] = -1.5
    with pytest.raises(ModelError, match="box.lo"):
        td.check_integrity(t2)
    t3 = compile_(MLP(), le=[1])
    with torch.no_grad():
        next(n for n in t3.program.nodes_in if n.param is not None).param.add_(1.0)
    with pytest.raises(ModelError, match="program"):
        td.check_integrity(t3)


def test_gear_program_supplies_fresh_copies():
    tc = compile_(MLP(), le=[1])
    supply = td.gear_program(tc)
    a, b = supply(None, (torch.zeros(1, 3),)), supply(None, (torch.zeros(1, 3),))
    pa = next(n for n in a[1] if n.param is not None).param
    pb = next(n for n in b[1] if n.param is not None).param
    assert pa is not pb and pa.data_ptr() != pb.data_ptr()
    assert all(n.param is None or n.param.data_ptr() != pa.data_ptr() for n in tc.program.nodes_in)


# ---- the solver adapter (XREAD r2 H7) ----------------------------------------------------------------------------
def test_raw_exported_module_refuses_eval_and_the_adapter_does_not():
    tc = compile_(TrainDep(), box=BOX2, le=[1])
    raw = tc.original.exported().module()
    with pytest.raises(NotImplementedError):
        raw.eval()
    with pytest.raises(NotImplementedError):
        nn.Sequential(raw).eval()
    m = tc.original.module()
    x = f64([[0.5, -1.5]])
    ref = m(x)
    for step in (lambda: m.to(device="cpu", dtype=torch.float64), m.eval, m.train, nn.Sequential(m).eval):
        step()
        assert torch.equal(bits(m(x)), bits(ref))


def test_training_dependent_source_keeps_its_export_time_branch():
    src = TrainDep().train()
    tc = compile_(src, box=BOX2, le=[1])
    x = f64([[0.5, -1.5]])
    with torch.no_grad():
        two_x = src.lin(x) * 2.0
    for model in (tc.original.module().eval(), td.computing_graph(tc).eval()):
        assert torch.equal(bits(model(x)), bits(two_x))
    e_prog = td.bounded_program(td.computing_graph(tc), tc.program, x)
    e_prog.eval()
    assert torch.equal(bits(e_prog(x)), bits(two_x))


def test_lifted_constant_moves_with_to():
    tc = compile_(Lifted(), box=BOX2)
    m = tc.original.module()
    assert "program.k" not in m.state_dict() and "k" in dict(m.program.named_buffers())
    m.to(dtype=torch.float32)
    assert m.program.k.dtype == torch.float32 and m.program.w.dtype == torch.float32
    assert not any(n.ori_name.endswith(".k") for n in tc.program.nodes_in)  # a lifted constant is a Constant node


def test_batched_rows_equal_single_calls_bitwise():
    tc = compile_(Poly(), box=BOX2, le=[1], eq=[2])
    m = tc.original.module()
    X = torch.from_numpy(td.candidates(*tc.box)).clone().requires_grad_(True)
    batched = m(X)
    rows = torch.cat([m(X[i:i + 1]) for i in range(X.shape[0])])
    assert torch.equal(bits(batched), bits(rows))
    g_b, = torch.autograd.grad(batched.sum(), X)
    g_r, = torch.autograd.grad(rows.sum(), X)
    assert torch.equal(bits(g_b), bits(g_r))


@pytest.mark.xfail(strict=True, reason="vmap of aten.linear rounds differently from [1, n] calls (<= 2 ulp); "
                                       "wp12a/REPORT.md anomaly 1. Batched rows carry no bitwise claim (Replay decides)")
def test_batched_linear_rows_equal_single_calls_bitwise():
    tc = compile_(MLP(), le=[1])
    m = tc.original.module()
    X = torch.from_numpy(td.candidates(*tc.box)).clone()
    assert torch.equal(bits(m(X)), bits(torch.cat([m(X[i:i + 1]) for i in range(X.shape[0])])))


# ---- the solver-facing layout -------------------------------------------------------------------------------------
def test_objective_wrapper_for_max_and_a_non_first_objective():
    tc = compile_(Poly(), box=BOX2, objective=1, le=[0, 2], sense=td.MAX)
    assert tc.objective_row.tolist() == [0.0, -1.0, 0.0]
    g = td.computing_graph(tc)
    x = f64([[0.5, 0.25]])
    y = Poly()(x)
    assert torch.equal(bits(g(x)), bits(torch.cat([-y[:, 1:2], y], dim=1)))
    assert tc.rowspec.num_outputs == 4 and tc.rowspec.indices.tolist() == [1, 3]


def test_structural_outputs_follow_the_option():
    lin = (f64([[1.0, 1.0], [1.0, -1.0]]), f64([1.0, 0.5]))
    on = compile_(Poly(), box=BOX2, le=[1], eq=[2], linear=lin, compile_keys={td.EXPAND_KEY: True})
    off = compile_(Poly(), box=BOX2, le=[1], eq=[2], linear=lin, compile_keys={td.EXPAND_KEY: False})
    none = compile_(Poly(), box=BOX2, le=[1], eq=[2], compile_keys={td.EXPAND_KEY: True})
    assert (on.structural_outputs, off.structural_outputs, none.structural_outputs) == (True, False, False)
    assert on.program_hash != off.program_hash and off.program_hash == none.program_hash
    assert {n.ori_name for n in on.program.nodes_in} >= {"constraint_A", "constraint_b"}
    assert on.rowspec.to_dense().tolist() == off.rowspec.to_dense().tolist()  # C/rhs before the runtime expansion
    with pytest.raises(ModelError, match="float64"):
        compile_(Poly(), box=BOX2, le=[1], eq=[2], linear=(lin[0].float(), lin[1]))
