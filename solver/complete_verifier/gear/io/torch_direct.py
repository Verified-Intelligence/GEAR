"""Direct torch path (G5 WP12a; DESIGN sections 1 rules O1-O3, 2 torch row T1-T4, Value check, TorchSource,
Ownership, Solver adapter; section 5 ``model_ori``).

``compile_torch`` turns an ``nn.Module`` or a callable into a ``TorchCompiled`` on the ``direct:torch`` path:

- O2 the box, T1 ``torch.export.export(copy, (x0,), strict=True)`` (no ``torch.jit``, no fallback), T2 the
  signature, O1 the output assignment, O3 the variable types, T3 every op in ``DIRECT_TABLE``
  (``gear.lower.emit``), T4 the bitwise value check at the midpoint and the first shifted-Halton points.
- The exported program, saved with ``torch.export.save``, is the only source of truth (``TorchSource``); every
  use materializes a fresh module from its bytes. The caller's object is read only at compile.

The solver-facing model (INTERFACE.md, ``wp12a``): the frozen solver minimizes output 0. Unless the objective is
output 0 under MIN, the source is wrapped in ``api._ObjectiveOutputWrapper`` with the row ``sense * e_objective``
(zeros for a feasibility problem), the frozen code's own minimize/maximize wrapper. With ``structural_outputs`` the
frozen code then wraps that in ``_StructuralConstraintOutputWrapper``; ``program`` describes the outermost model.
"""
from __future__ import annotations

import copy
import hashlib
import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.export.graph_signature import InputKind, OutputKind

from ..config import UNSET
from ..errors import InternalError, ModelError, OptionError, UnsupportedModelError, UnsupportedProblemError
from ..lower.emit import EmitRefused, NodeProgram, emit_program
from ..spec import RowSpec
from .program_manifest import program_hash

MIN, MAX = 1, -1
N_CANDIDATES = 64      # DESIGN section 2 Value check: fixed by the spec, not tuned
N_POINTS = 11
HALTON_SEED = 0
INT_LIMIT = 2 ** 53
CONTINUOUS, INTEGER, BINARY = 0, 1, 2
EXPAND_KEY = "solver.nlp.expand_input_constraints_as_outputs"
OPEN_KEY = "solver.nlp.open_bounds"
MINLP_KEY = "solver.nlp.minlp.enabled"
REBUILD_UNAVAILABLE = "rebuild:torch not available"


# ---- the source: exported bytes and the solver adapter --------------------------------------------------------
class ExportedSourceModule(nn.Module):
    """The solver adapter (DESIGN section 2, XREAD r2 H7). ``program`` is a fresh, prepared ``exported().module()``.

    ``[1, n]`` calls the program; ``[N, n]`` with N > 1 is ``vmap`` of the ``[1, n]`` call, row by row.
    """

    def __init__(self, program: nn.Module):
        super().__init__()
        self.program = program

    def forward(self, x):
        if x.dim() == 2 and x.shape[0] == 1:
            return self.program(x)
        return torch.func.vmap(lambda row: self.program(row.unsqueeze(0)).squeeze(0))(x)


def _owner(module: nn.Module, fqn: str):
    *path, attr = fqn.split(".")
    for part in path:
        module = getattr(module, part)
    return module, attr


def _prepare(program: nn.Module, constants: Mapping[str, Any]) -> nn.Module:
    """Adapter steps 1 and 2: drop the raising ``train``/``eval`` instance overrides; lifted tensor constants
    become non-persistent buffers (so ``.to()`` moves them; ``state_dict()`` is unchanged)."""
    for module in program.modules():
        for name in ("train", "eval"):
            if name in module.__dict__:
                delattr(module, name)
    for fqn, value in constants.items():
        if not isinstance(value, torch.Tensor):
            raise UnsupportedModelError(f"lifted constant {fqn!r} is not a tensor")
        owner, attr = _owner(program, fqn)
        if attr in owner._buffers:
            owner._non_persistent_buffers_set.add(attr)
            continue
        tensor = getattr(owner, attr)
        delattr(owner, attr)
        owner.register_buffer(attr, tensor, persistent=False)
    return program


@dataclass(frozen=True)
class TorchSource:
    export_bytes: bytes    # torch.export.save of the T1 ExportedProgram: graph, parameters, buffers, constants
    export_sha256: str

    def exported(self):
        """A fresh ``torch.export.load`` on every call."""
        return torch.export.load(io.BytesIO(self.export_bytes))

    def module(self) -> ExportedSourceModule:
        """A fresh ``exported().module()``, prepared and wrapped (the solver adapter)."""
        ep = self.exported()
        return ExportedSourceModule(_prepare(ep.module(), ep.constants))


@dataclass(frozen=True)
class OutputAssignment:  # rule O1; indices normalized into [0, k) and sorted
    objective: Optional[int]
    le: tuple[int, ...]
    eq: tuple[int, ...]


@dataclass(frozen=True)
class TorchCompiled:
    path: str
    original: TorchSource
    program: NodeProgram
    program_hash: str
    rowspec: RowSpec
    linear: Optional[tuple[np.ndarray, np.ndarray]]
    structural_outputs: bool
    box: tuple[np.ndarray, np.ndarray]
    vtype: Optional[np.ndarray]
    sense: int
    assignment: OutputAssignment
    objective_row: Optional[np.ndarray]
    num_inputs: int
    num_outputs: int
    reasons: tuple[str, ...]
    value_points: np.ndarray
    array_sha256: Mapping[str, str] = field(repr=False)


class _CallableModule(nn.Module):
    """A callable source, wrapped as an ``nn.Module`` whose ``forward`` calls it."""

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


# ---- rule O2: the box ------------------------------------------------------------------------------------------
def _endpoint(value, what: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float64:
            raise ModelError(f"box {what}: dtype {value.dtype}, float64 required (a box endpoint is never rounded)")
        array = value.detach().cpu().numpy().copy()
    elif isinstance(value, np.ndarray):
        if value.dtype != np.float64:
            raise ModelError(f"box {what}: dtype {value.dtype}, float64 required (a box endpoint is never rounded)")
        array = value.copy()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ModelError(f"box {what}: entries must be Python floats or ints, got {type(item).__name__}")
            if isinstance(item, int) and float(item) != item:
                raise ModelError(f"box {what}: the int {item} is not exactly a float64")
            items.append(float(item))
        array = np.asarray(items, dtype=np.float64)
    else:
        raise ModelError(f"box {what}: expected a float64 tensor or array or a sequence of numbers, "
                         f"got {type(value).__name__}")
    if array.ndim != 1 or array.size == 0:
        raise ModelError(f"box {what}: shape {list(array.shape)}, a non-empty [n] is required")
    return array


def check_box(box, compile_keys: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if box is None:
        raise ModelError("box is required: box=(lo, hi)")
    if not isinstance(box, (tuple, list)) or len(box) != 2:
        raise ModelError("box must be a pair (lo, hi)")
    lo, hi = _endpoint(box[0], "lo"), _endpoint(box[1], "hi")
    if lo.shape != hi.shape:
        raise ModelError(f"box: lo has shape {list(lo.shape)}, hi has {list(hi.shape)}")
    if np.isnan(lo).any() or np.isnan(hi).any():
        raise ModelError("box: NaN endpoint")
    if (lo > hi).any():
        raise ModelError(f"box: lo > hi at coordinates {np.flatnonzero(lo > hi).tolist()[:10]}")
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        # G3's open route has no in-memory entry yet, so every infinite torch endpoint raises in v1.
        why = "G3's open route has no in-memory entry" if compile_keys.get(OPEN_KEY) else \
            "solver.nlp.open_bounds is off"
        raise UnsupportedProblemError(f"box: infinite endpoint ({why})")
    for array in (lo, hi):
        array.setflags(write=False)
    return lo, hi


# ---- rule O1: the outputs --------------------------------------------------------------------------------------
def _indices(spec, k: int, what: str) -> list[int]:
    if isinstance(spec, slice):
        return list(range(k))[spec]
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, (bool, np.bool_)):
        items = [spec]
    elif isinstance(spec, (range, Sequence, np.ndarray)) and not isinstance(spec, (str, bytes)):
        items = list(spec)
    else:
        raise ModelError(f"{what}: expected an int, slice, range or sequence of ints, got {type(spec).__name__}")
    out = []
    for item in items:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise ModelError(f"{what}: index {item!r} is not an int")
        item = int(item)
        if not -k <= item < k:
            raise ModelError(f"{what}: index {item} is outside the {k} outputs")
        out.append(item % k)
    return out


def check_outputs(objective, le, eq, k: int) -> OutputAssignment:
    if objective is UNSET:
        raise ModelError("objective is required: an output index, or None for a feasibility problem")
    if objective is not None and (isinstance(objective, (bool, np.bool_))
                                  or not isinstance(objective, (int, np.integer))):
        raise ModelError("objective must be one int or None")
    obj = [] if objective is None else _indices(objective, k, "objective")
    le_i, eq_i = _indices(le, k, "le"), _indices(eq, k, "eq")
    repeated = sorted({i for group in (le_i, eq_i) for i in group if group.count(i) > 1})
    seen: dict[int, int] = {}
    for i in obj + le_i + eq_i:
        seen[i] = seen.get(i, 0) + 1
    doubly = sorted({i for i, c in seen.items() if c > 1} | set(repeated))
    unassigned = sorted(set(range(k)) - set(seen))
    if doubly or unassigned:
        raise ModelError(f"outputs: every one of the {k} outputs must be in exactly one of objective, le, eq; "
                         f"unassigned {unassigned}, doubly assigned {doubly}")
    return OutputAssignment(obj[0] if obj else None, tuple(sorted(le_i)), tuple(sorted(eq_i)))


# ---- rule O3: variable types -----------------------------------------------------------------------------------
def check_vtype(vtype, lo: np.ndarray, hi: np.ndarray, compile_keys: Mapping[str, Any]) -> None:
    if vtype is None:
        return None
    if isinstance(vtype, torch.Tensor):
        vtype = vtype.detach().cpu().numpy()
    if not isinstance(vtype, np.ndarray) or vtype.dtype != np.int8 or vtype.shape != lo.shape:
        raise ModelError(f"vtype must be an int8 array of shape [{lo.size}]")
    if not np.isin(vtype, (CONTINUOUS, INTEGER, BINARY)).all():
        raise ModelError("vtype: values must be VarType (0 continuous, 1 integer, 2 binary)")
    discrete = vtype != CONTINUOUS
    if not discrete.any():
        return None  # all continuous: the same problem as vtype=None
    ends = np.concatenate([lo[discrete], hi[discrete]])
    if not (np.isfinite(ends).all() and (np.floor(ends) == ends).all() and (np.abs(ends) <= INT_LIMIT).all()):
        raise UnsupportedProblemError("vtype: integer coordinates need finite integral endpoints with |.| <= 2^53")
    why = "G2's in-memory mask entry (minlp_meta) does not exist" if compile_keys.get(MINLP_KEY) else \
        "solver.nlp.minlp.enabled is off"
    raise UnsupportedProblemError(f"vtype: integer or binary coordinates have no route ({why})")


# ---- value-check points ----------------------------------------------------------------------------------------
def _primes(n: int) -> list[int]:
    out, c = [], 2
    while len(out) < n:
        if all(c % p for p in out if p * p <= c):
            out.append(c)
        c += 1
    return out


def _radical_inverse(i: int, base: int) -> float:
    num, den = 0, 1
    while i:
        i, digit = divmod(i, base)
        num, den = num * base + digit, den * base
    return num / den


def halton_u(n: int, count: int = N_CANDIDATES) -> np.ndarray:
    """u[i, j] = frac(phi_{p_j}(i) + r_j) for i = 1 .. count-1 (row i-1); r = default_rng(0).random(n)."""
    shift = np.random.default_rng(HALTON_SEED).random(n)
    primes = _primes(n)
    u = np.empty((count - 1, n), dtype=np.float64)
    for i in range(1, count):
        u[i - 1] = [_radical_inverse(i, p) for p in primes]
    return np.mod(u + shift, 1.0)


def candidates(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The 64 candidates: c_0 the midpoint, then c_i = lo + (hi - lo) * u_i clamped into [lo, hi]."""
    with np.errstate(over="ignore", invalid="ignore"):
        mid = np.clip((lo + hi) / 2, lo, hi)
        rest = np.clip(lo + (hi - lo) * halton_u(lo.size), lo, hi)
    return np.concatenate([mid[None, :], rest], axis=0)


def _row(c: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(c, dtype=np.float64, copy=True)).reshape(1, -1)


def _bitwise_equal(a, b) -> bool:
    return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype == torch.float64
            and a.shape == b.shape and torch.equal(a.contiguous().view(torch.int64), b.contiguous().view(torch.int64)))


@torch.no_grad()
def _call(module_or_fn, x: torch.Tensor):
    return module_or_fn(x)


def domain_valid(caller, cands: np.ndarray) -> list[int]:
    """Indices of the candidates where every output of the caller's forward is finite."""
    valid = []
    for i, c in enumerate(cands):
        try:
            y = _call(caller, _row(c))
        except Exception as exc:  # noqa: BLE001 - the caller's own error is the message
            raise UnsupportedModelError(f"the source raised at a value-check candidate: "
                                        f"{type(exc).__name__}: {exc}") from exc
        if not isinstance(y, torch.Tensor) or y.dtype != torch.float64 or y.dim() != 2 or y.shape[0] != 1:
            found = (y.dtype, list(y.shape)) if isinstance(y, torch.Tensor) else type(y).__name__
            raise UnsupportedModelError(f"signature (T2): forward returned {found}, a float64 [1, k] is required")
        if bool(torch.isfinite(y).all()):
            valid.append(i)
    return valid


# ---- T1, T2 ----------------------------------------------------------------------------------------------------
def _release_export_frames() -> None:
    """Drop the Dynamo cache entries that strict export leaves on the shared ``wrap_inline`` frame.

    torch 2.8 keeps one entry per strict export there; the 9th export of any module in a process then fails with
    ``FailOnRecompileLimitHit`` (measured, wp12a/REPORT.md). The entries are never reused. Only that one code
    object is reset; functions compiled with ``torch.compile`` keep their caches.
    """
    from torch._dynamo import external_utils
    from torch._dynamo.eval_frame import reset_code
    reset_code(external_utils.wrap_inline(len).__code__)


def export_source(private: nn.Module, x0: torch.Tensor):
    """T1: strict ``torch.export``; there is no ``torch.jit`` fallback."""
    try:
        return torch.export.export(private, (x0,), strict=True)
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedModelError(f"torch.export failed (T1): {type(exc).__name__}: {str(exc)[:2000]}") from exc
    finally:
        _release_export_frames()


def check_signature(ep, n: int) -> int:
    """T2: one user input [1, n] float64, one output [1, k] float64, no mutation, no random op. Returns k."""
    sig = ep.graph_signature
    users = [s for s in sig.input_specs if s.kind == InputKind.USER_INPUT]
    kinds = sorted({s.kind.name for s in sig.input_specs} - {"USER_INPUT", "PARAMETER", "BUFFER", "CONSTANT_TENSOR"})
    if len(users) != 1 or kinds:
        raise UnsupportedModelError(f"signature (T2): {len(users)} user inputs, other input kinds {kinds}; "
                                    "exactly one tensor input is required")
    placeholders = {node.name: node for node in ep.graph.nodes if node.op == "placeholder"}
    val = placeholders[users[0].arg.name].meta.get("val")
    if not isinstance(val, torch.Tensor) or val.dtype != torch.float64 or list(val.shape) != [1, n]:
        found = None if not isinstance(val, torch.Tensor) else (val.dtype, list(val.shape))
        raise UnsupportedModelError(f"signature (T2): input {found}, float64 [1, {n}] required")
    outs = sig.output_specs
    if len(outs) != 1 or outs[0].kind != OutputKind.USER_OUTPUT:
        raise UnsupportedModelError(f"signature (T2): outputs {[o.kind.name for o in outs]}; exactly one user "
                                    "output and no buffer or input mutation are required")
    out_node = next(node for node in ep.graph.nodes if node.op == "output")
    result = out_node.args[0]
    result = result[0] if isinstance(result, (list, tuple)) and len(result) == 1 else result
    oval = result.meta.get("val") if isinstance(result, torch.fx.Node) else None
    if (not isinstance(oval, torch.Tensor) or oval.dtype != torch.float64 or oval.dim() != 2
            or not all(isinstance(s, int) for s in oval.shape) or oval.shape[0] != 1 or oval.shape[1] < 1):
        found = None if not isinstance(oval, torch.Tensor) else (oval.dtype, list(oval.shape))
        raise UnsupportedModelError(f"signature (T2): output {found}, float64 [1, k] required")
    for node in ep.graph.nodes:
        if node.op == "call_function" and isinstance(node.target, torch._ops.OpOverload):
            if torch.Tag.nondeterministic_seeded in node.target.tags:
                raise UnsupportedModelError(f"signature (T2): random op {node.target}")
            if node.target._schema.is_mutable:  # the training IR keeps in-place ops (add_, copy_, ...)
                raise UnsupportedModelError(f"signature (T2): mutation by the in-place op {node.target}")
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor) and not all(isinstance(s, int) for s in val.shape):
            raise UnsupportedModelError(f"signature (T2): symbolic shape at {node.name}")
    return int(oval.shape[1])


# ---- the solver-facing model and its program -------------------------------------------------------------------
def _objective_row(assignment: OutputAssignment, sense: int, k: int) -> Optional[np.ndarray]:
    if assignment.objective == 0 and sense == MIN:
        return None
    row = np.zeros(k, dtype=np.float64)
    if assignment.objective is not None:
        row[assignment.objective] = float(sense)
    row.setflags(write=False)
    return row


def _wrappers():
    import api  # complete_verifier/api.py: the frozen wrappers
    return api._ObjectiveOutputWrapper, api._StructuralConstraintOutputWrapper


def _graph_model(source: TorchSource, objective_row) -> nn.Module:
    module = source.module()
    if objective_row is None:
        return module
    objective_wrapper, _ = _wrappers()
    return objective_wrapper(module, torch.from_numpy(np.array(objective_row)))


def _solver_model(source, objective_row, linear, structural: bool) -> nn.Module:
    model = _graph_model(source, objective_row)
    if structural:
        _, structural_wrapper = _wrappers()
        a, b = linear
        model = structural_wrapper(model, torch.from_numpy(np.array(a)), torch.from_numpy(np.array(b)))
    return model.to(dtype=torch.float64).eval()


def _emit_solver_program(model: nn.Module, x0: torch.Tensor) -> NodeProgram:
    """The program of the solver-facing model: the exported source inside the frozen wrappers, exported once more
    (non-strict, our own code) so that names are the state_dict keys ``LiRPANet`` sees."""
    try:
        ep = torch.export.export(model, (x0,), strict=False)
    except Exception as exc:  # noqa: BLE001
        raise InternalError(f"re-export of the solver-facing model failed: {type(exc).__name__}: {exc}") from exc
    return emit_program(ep, state_order=list(model.state_dict().keys()))


def _gear_program_callable(program: NodeProgram):
    def supply(model, inputs):
        nodes_op, nodes_in, nodes_out, template = copy.deepcopy(tuple(program))
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        ptb = getattr(x, "ptb", None)
        nodes_in = [n._replace(perturbation=ptb) if n.model_input_index == 0 else n for n in nodes_in]
        return list(nodes_op), nodes_in, list(nodes_out), template
    return supply


def attach_program(model: nn.Module, program: NodeProgram) -> nn.Module:
    """``model.__gear_program__`` = ``gear.lower.seam.ProgramSource(program)``: the WP6 seam gets ``program`` with
    every parameter and buffer bound to ``model``'s own tensor (bitwise checked), so ``.to(device)`` keeps Parameters
    as Parameters and the Bound graph's classes equal parse_module's on every device (A291; deep copies became
    BoundBuffers on CUDA)."""
    from ..lower.seam import ProgramSource
    model.__gear_program__ = ProgramSource(program)
    return model


E_PROG_BOUND_OPTS = {"optimize_graph": {"optimizer": None}, "hidden_square_rewrite": False,
                     "complete_square_rewrite": False}


def bounded_program(model: nn.Module, program: NodeProgram, x0: torch.Tensor):
    """E_prog: a CPU float64 ``BoundedModule`` built from ``program`` through the seam, optional rewrites off."""
    from auto_LiRPA import BoundedModule
    attach_program(model, program)
    return BoundedModule(model, (x0,), bound_opts=copy.deepcopy(E_PROG_BOUND_OPTS), device="cpu", verbose=False)


# ---- rowspec ---------------------------------------------------------------------------------------------------
def _rowspec(assignment: OutputAssignment, k: int, offset: int) -> RowSpec:
    cols, data, rhs = [], [], []
    for i in sorted(assignment.le + assignment.eq):
        cols.append(i + offset), data.append(1.0), rhs.append(0.0)
        if i in assignment.eq:  # (>= Y_i 0) read as -Y_i <= -0.0, as read_vnnlib does
            cols.append(i + offset), data.append(-1.0), rhs.append(-1 * 0.0)
    m = len(cols)
    spec = RowSpec(num_outputs=k + offset, indptr=np.arange(m + 1, dtype=np.int64),
                   indices=np.asarray(cols, dtype=np.int64), data=np.asarray(data, dtype=np.float64),
                   rhs=np.asarray(rhs, dtype=np.float64))
    for array in (spec.indptr, spec.indices, spec.data, spec.rhs):
        array.setflags(write=False)
    return spec


def _check_linear(linear, n: int):
    if linear is None:
        return None
    if not isinstance(linear, (tuple, list)) or len(linear) != 2:
        raise ModelError("linear must be a pair (A, b)")
    a, b = linear
    arrays = []
    for value, what in ((a, "A"), (b, "b")):
        if isinstance(value, torch.Tensor):
            if value.dtype != torch.float64:
                raise ModelError(f"linear {what}: dtype {value.dtype}, float64 required")
            value = value.detach().cpu()
            value = value.to_dense() if value.is_sparse else value
            arrays.append(value.numpy().copy())
        else:
            raise ModelError(f"linear {what}: a float64 torch tensor (dense or sparse) is required")
    a, b = arrays
    if a.ndim != 2 or a.shape[1] != n or b.shape != (a.shape[0],) or a.shape[0] == 0:
        raise ModelError(f"linear: A {list(a.shape)} and b {list(b.shape)}, need A [m, {n}], b [m], m >= 1")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ModelError("linear: non-finite coefficient")
    for array in (a, b):
        array.setflags(write=False)
    return a, b


def _digest(array: Optional[np.ndarray]) -> str:
    if array is None:
        return "none"
    array = np.ascontiguousarray(array)
    return f"{array.dtype}|{list(array.shape)}|{hashlib.sha256(array.tobytes()).hexdigest()}"


def _array_digests(box, linear, rowspec, objective_row) -> dict[str, str]:
    d = {"box.lo": _digest(box[0]), "box.hi": _digest(box[1]), "objective_row": _digest(objective_row),
         "linear.A": _digest(None if linear is None else linear[0]),
         "linear.b": _digest(None if linear is None else linear[1])}
    for name in ("indptr", "indices", "data", "rhs"):
        d[f"rowspec.{name}"] = _digest(getattr(rowspec, name))
    return d


# ---- compile ---------------------------------------------------------------------------------------------------
def check_cpu_source(src: nn.Module) -> None:
    """Every parameter, buffer and tensor attribute of the source is on the CPU (v1 interface decision, A294): the
    value check and the export read the module on the CPU, and solve places the program on ``Options.device``. A
    module on another device is refused, never copied silently."""
    for name, m in src.named_modules():
        tensors = list(m.named_parameters(recurse=False)) + list(m.named_buffers(recurse=False))
        tensors += [(k, v) for k, v in vars(m).items() if isinstance(v, torch.Tensor)]
        for key, t in tensors:
            if t.device.type != "cpu":
                raise ModelError(f"the source has tensors on {t.device} (first: {(name + '.' if name else '') + key!r}); "
                                 "gr.compile takes a CPU module: pass model.cpu() (Options.device places the compiled "
                                 "program at solve)")


def compile_torch(src, *, box=None, vtype=None, sense=MIN, objective=UNSET, le=(), eq=(), linear=None,
                  optimize=None, passes="default", constants=UNSET,
                  compile_keys: Optional[Mapping[str, Any]] = None) -> TorchCompiled:
    """Rules O2, T1, T2, O1, O3, T3, T4 in that order; every failure raises before any solve."""
    compile_keys = {} if compile_keys is None else compile_keys
    if isinstance(passes, tuple) and optimize is False:
        raise OptionError("a passes tuple implies optimize=True; it cannot be combined with optimize=False")
    if constants is not UNSET and constants != "dyadic":
        raise OptionError(f"constants={constants!r}: torch constants are float64 tensors, only 'dyadic' applies")
    if optimize is True or isinstance(passes, tuple):
        raise UnsupportedModelError(f"{REBUILD_UNAVAILABLE} (optimize=True or a passes tuple; WP12b)")
    if passes not in ("default", "none") and not isinstance(passes, tuple):
        raise OptionError(f"passes={passes!r}: 'default', 'none' or a tuple of pass names")
    if not isinstance(src, nn.Module):
        if not callable(src):
            raise UnsupportedModelError(f"source of type {type(src).__name__} is neither an nn.Module nor callable")
        src = _CallableModule(src)
    sense = int(sense)
    if sense not in (MIN, MAX):
        raise ModelError(f"sense must be MIN (1) or MAX (-1), got {sense}")

    lo, hi = check_box(box, compile_keys)                                          # O2
    n = lo.size
    check_cpu_source(src)                                                          # T1: compile reads a CPU module
    try:
        private = copy.deepcopy(src)
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedModelError(f"the source cannot be copied: {type(exc).__name__}: {exc}") from exc
    cands = candidates(lo, hi)
    valid = domain_valid(src, cands)
    if not valid:
        raise UnsupportedModelError(f"value check: none of the {N_CANDIDATES} candidates is domain-valid "
                                    "(the source is non-finite at all of them)")
    x0 = _row(cands[valid[0]])
    ep = export_source(private, x0)                                                # T1
    k = check_signature(ep, n)                                                     # T2
    assignment = check_outputs(objective, le, eq, k)                               # O1
    vt = check_vtype(vtype, lo, hi, compile_keys)                                  # O3
    try:
        emit_program(ep)                                                           # T3 on the source graph
    except EmitRefused as exc:
        if optimize is False:
            raise UnsupportedModelError(f"T3: {exc}; optimize=False forbids rebuild:torch") from exc
        if exc.in_import_table:
            raise UnsupportedModelError(f"{REBUILD_UNAVAILABLE} (T3: {exc}; WP12b)") from exc
        raise UnsupportedModelError(f"T3: {exc} (in neither DIRECT_TABLE nor IMPORT_TABLE)") from exc

    buf = io.BytesIO()
    torch.export.save(ep, buf)
    data = buf.getvalue()
    source = TorchSource(data, hashlib.sha256(data).hexdigest())

    lin = _check_linear(linear, n)
    structural = bool(lin is not None and compile_keys.get(EXPAND_KEY, False))
    objective_row = _objective_row(assignment, sense, k)
    offset = 0 if objective_row is None else 1
    rowspec = _rowspec(assignment, k, offset)

    # T4: the value check at the first N_POINTS domain-valid candidates.
    if len(valid) < N_POINTS:
        raise UnsupportedModelError(f"value check: m < {N_POINTS} domain-valid points ({len(valid)} of "
                                    f"{N_CANDIDATES})")
    points = np.stack([cands[i] for i in valid[:N_POINTS]])
    points.setflags(write=False)
    unwrapped = source.module().eval()
    for c in points:
        x = _row(c)
        e_ref = _call(unwrapped, x)
        if not (_bitwise_equal(e_ref, _call(src, x)) and bool(torch.isfinite(e_ref).all())):
            raise UnsupportedModelError("export changed the source's values (T4: E_ref != the caller's forward)")
    reference = _solver_model(source, objective_row, lin, structural)
    try:
        program = _emit_solver_program(_solver_model(source, objective_row, lin, structural), x0)
    except EmitRefused as exc:
        raise InternalError(f"the solver-facing wrappers left DIRECT_TABLE: {exc}") from exc
    width = k + offset + (lin[0].shape[0] if structural else 0)
    e_prog = bounded_program(_solver_model(source, objective_row, lin, structural), program, x0)
    for c in points:
        x = _row(c)
        e_ref = _call(reference, x)
        got = _call(e_prog, x)
        if tuple(e_ref.shape) != (1, width):
            raise InternalError(f"program output count {tuple(e_ref.shape)} != expanded C columns {width}")
        if not (_bitwise_equal(got, e_ref) and bool(torch.isfinite(got).all())):
            raise InternalError("value check (T4): E_prog != E_ref bitwise; this is a lowering bug")

    reasons = ["direct:torch: every op in DIRECT_TABLE; value check passed at "
               f"{N_POINTS} points (first valid candidate {valid[0]})"]
    if objective_row is not None:
        reasons.append(f"objective wrapper: output {assignment.objective} with sense {sense} moved to output 0")
    if structural:
        reasons.append("structural_outputs: the linear rows are model outputs")
    box_pair = (lo, hi)
    return TorchCompiled(
        path="direct:torch", original=source, program=program, program_hash=program_hash(*program),
        rowspec=rowspec, linear=lin, structural_outputs=structural, box=box_pair, vtype=vt, sense=sense,
        assignment=assignment, objective_row=objective_row, num_inputs=n, num_outputs=k, reasons=tuple(reasons),
        value_points=points, array_sha256=_array_digests(box_pair, lin, rowspec, objective_row))


# ---- helpers used by the facade at solve time (each returns fresh objects) --------------------------------------
def computing_graph(tc: TorchCompiled, device: str = "cpu") -> nn.Module:
    """The model handed to the frozen solver as ``computing_graph``: the adapter, inside the objective wrapper if
    ``tc.objective_row`` is set. The frozen code adds ``_StructuralConstraintOutputWrapper`` itself."""
    return _graph_model(tc.original, tc.objective_row).to(device=device, dtype=torch.float64)


def gear_program(tc: TorchCompiled):
    """A model-free ``__gear_program__`` supplier: deep copies of ``tc.program`` on every call. Its Parameters are
    copies, so on a non-CPU device they become BoundBuffers; build through ``attach_program`` for parse_module's
    classes (A291)."""
    return _gear_program_callable(tc.program)


def check_integrity(tc: TorchCompiled) -> None:
    """Raise ``ModelError`` if an array or a program tensor of ``tc`` changed after compile."""
    if program_hash(*tc.program) != tc.program_hash:
        raise ModelError("Compiled was mutated after compile (program)")
    now = _array_digests(tc.box, tc.linear, tc.rowspec, tc.objective_row)
    changed = sorted(k for k in now if now[k] != tc.array_sha256.get(k))
    if changed:
        raise ModelError(f"Compiled was mutated after compile ({', '.join(changed)})")
    if hashlib.sha256(tc.original.export_bytes).hexdigest() != tc.original.export_sha256:
        raise ModelError("Compiled was mutated after compile (export bytes)")


@torch.no_grad()
def evaluate(tc: TorchCompiled, x) -> np.ndarray:
    """The source outputs at one point: a fresh ``TorchSource.module()`` at ``[1, n]``, CPU float64 (Replay)."""
    y = tc.original.module().eval()(_row(np.asarray(x, dtype=np.float64)))
    return y.reshape(-1).numpy().copy()
