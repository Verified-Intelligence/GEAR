"""Direct emitter (G5 WP12a, DESIGN section 5 ``DIRECT_TABLE``): an exported program -> ``parse_module`` node tuples.

``emit_program(ep)`` walks the graph of a ``torch.export.ExportedProgram`` once and maps every ``call_function``
node 1:1 to the ONNX-named node that today's ``parse_module`` produces for the same ATen op (opset-17 symbolics as
used by ``auto_LiRPA/parse_graph.py``). The result is a ``NodeProgram`` of ``auto_LiRPA.parse_graph.Node`` tuples
that the ``BoundedModule._convert_nodes`` seam (``__gear_program__``) consumes unchanged.

No tracing and no ``torch.jit``: the exported graph is read, never re-run. An op outside ``DIRECT_TABLE``, or a
table op whose predicate fails, raises ``EmitRefused`` (rule T3); there is no fallback.

Deliberate differences from ``parse_module`` (values are identical, see the WP12a conformance tests):
- integer arguments of shape ops (Slice starts/ends/axes/steps, Expand shape) are emitted as 1-D int64
  ``onnx::Constant`` nodes instead of the exporter's scalar-Constant/Unsqueeze/Where plumbing;
- an identity ``slice``/``view``/``reshape`` (same shape in and out, full range, step 1) emits no node, since the
  exported graph is static at batch 1 (an identity reshape written as ``x.reshape(x.shape[0], -1)`` would otherwise
  pin the batch size);
- node names are ``/<k>`` in emission order.
"""
from __future__ import annotations

from collections import OrderedDict, namedtuple

import torch
from torch.export.graph_signature import InputKind, OutputKind

from auto_LiRPA.parse_graph import Node

NodeProgram = namedtuple("NodeProgram", "nodes_op nodes_in nodes_out template")

INT64_MAX = 2 ** 63 - 1
_F64, _I64 = torch.float64, torch.int64
_CAST_TO_DOUBLE = 11  # ONNX TensorProto.DOUBLE

_UNARY = {name: f"onnx::{name.capitalize()}" for name in
          ("exp", "log", "sqrt", "sin", "cos", "tan", "atan", "tanh", "abs", "relu", "sigmoid")}

# ATen op (``OpOverload.name()`` without the ``aten::`` prefix) -> Node.op.  Predicates are checked by the handlers.
# Removed after conformance (bitwise E_prog != E_ref, DESIGN section 2): ``div.Tensor`` (the always-on
# ``div_to_mul`` rewrite computes x * (1/y)); ``minimum``/``maximum`` with an operand that does not depend on the
# input (``minmax_to_relu``); ``pow.Tensor_Scalar`` with an exponent in ``POW_NOT_BITWISE``. Added: ``t`` (torch.export
# spells ``W.t()`` as aten.t; it is onnx::Transpose) and ``flatten.using_ints`` (nn.Flatten; onnx::Flatten).
DIRECT_TABLE = {
    "add.Tensor": "onnx::Add", "sub.Tensor": "onnx::Sub", "mul.Tensor": "onnx::Mul",
    "neg": "onnx::Neg", "reciprocal": "onnx::Reciprocal", "pow.Tensor_Scalar": "onnx::Pow",   # [exponent const]
    **_UNARY,
    "minimum": "onnx::Min", "maximum": "onnx::Max",                     # [both operands depend on the input]
    "clamp": "onnx::Clip",                                              # [bounds const]
    "mm": "onnx::Gemm", "addmm": "onnx::Gemm", "linear": "onnx::Gemm|onnx::MatMul", "matmul": "onnx::MatMul",
    "t": "onnx::Transpose",                                             # [weight const]
    "index_select": "onnx::Gather", "index.Tensor": "onnx::Gather", "select.int": "onnx::Gather",  # [index const]
    "index_add": "aten::ATen",                                          # [index const]
    "cat": "onnx::Concat", "view": "onnx::Reshape", "reshape": "onnx::Reshape", "_unsafe_view": "onnx::Reshape",
    "flatten.using_ints": "onnx::Flatten",                              # [end_dim = -1]
    "unsqueeze": "onnx::Unsqueeze", "squeeze.dim": "onnx::Squeeze", "slice.Tensor": "onnx::Slice",
    "expand": "onnx::Expand", "permute": "onnx::Transpose", "transpose.int": "onnx::Transpose",
    "sum.dim_IntList": "onnx::ReduceSum",
    "_to_copy": "onnx::Cast",                                           # [float64 -> float64 only]
}
# pow.Tensor_Scalar exponents removed after conformance: torch 2.8's CPU kernel computes x*x*x, 1/x, 1/(x*x),
# sqrt and rsqrt for them, BoundPow.forward computes pow(x, 0-d tensor) = std::pow (2 is kept: convert_sqr makes
# it BoundSqr, x*x, as torch does).
POW_NOT_BITWISE = frozenset({3.0, -1.0, -2.0, 0.5, -0.5})
# DESIGN IMPORT_TABLE minus DIRECT_TABLE: these go through the IR (WP12b) only.
IMPORT_ONLY = frozenset({"rsub.Scalar", "rsub.Tensor", "square", "sum", "mean.dim", "log1p", "expm1", "where.self",
                         "div.Tensor"})


class EmitRefused(Exception):
    """Rule T3 failed: ``op`` is outside ``DIRECT_TABLE`` or its predicate does not hold."""

    def __init__(self, op: str, reason: str):
        super().__init__(f"{op}: {reason}")
        self.op, self.reason = op, reason
        self.in_import_table = op in IMPORT_ONLY


def op_name(target) -> str:
    """``torch.ops.aten.add.Tensor`` -> ``"add.Tensor"``; a default overload drops its suffix (``"neg"``)."""
    name = target.name() if hasattr(target, "name") and callable(target.name) else str(target)
    ns, _, rest = name.partition("::")
    return rest if ns == "aten" and rest else name


def _static(val) -> bool:
    return isinstance(val, torch.Tensor) and all(isinstance(s, int) for s in val.shape)


class _Emitter:
    def __init__(self, ep, state_order=None):
        self.ep = ep
        self.sig = ep.graph_signature
        self.ops: list[Node] = []
        self.ins: list[Node] = []
        self.env: dict = {}
        self.k = 0
        self.state_order = list(state_order) if state_order is not None else None
        self.input_dep: set = set()

    # ---- naming and node creation ---------------------------------------------------------------------------
    def fresh(self) -> str:
        self.k += 1
        return f"#{self.k}"

    def emit(self, op: str, inputs, attr=None) -> str:
        name = self.fresh()
        self.ops.append(Node(name=name, op=op, inputs=list(inputs), attr=dict(attr or {}), output_index=0))
        return name

    def const(self, value: torch.Tensor) -> str:
        return self.emit("onnx::Constant", [], {"value": value.detach().clone()})

    def int_const(self, values) -> str:
        return self.const(torch.tensor(list(values), dtype=_I64))

    # ---- argument access -------------------------------------------------------------------------------------
    def tensor(self, arg, op, *, dtype=_F64, what="argument") -> str:
        if not isinstance(arg, torch.fx.Node):
            raise EmitRefused(op, f"{what} must be a tensor, got {type(arg).__name__}")
        val = arg.meta.get("val")
        if not _static(val):
            raise EmitRefused(op, f"{what} has no static tensor shape")
        if val.dtype != dtype:
            raise EmitRefused(op, f"{what} is {val.dtype}, the table requires {dtype}")
        return self.env[arg]

    def operand(self, arg, op) -> str:
        """A float64 tensor, or a Python number emitted as a float64 scalar Constant (as the tracer does)."""
        if isinstance(arg, torch.fx.Node):
            return self.tensor(arg, op)
        if isinstance(arg, bool) or not isinstance(arg, (int, float)):
            raise EmitRefused(op, f"scalar operand of type {type(arg).__name__}")
        return self.const(torch.tensor(float(arg), dtype=_F64))

    def constant_index(self, arg, op) -> str:
        """An int64 index tensor that does not depend on the input (a buffer or a lifted constant)."""
        if not isinstance(arg, torch.fx.Node) or arg.op != "placeholder" or arg in self.input_dep:
            raise EmitRefused(op, "index must be a constant tensor (buffer or constant), not computed")
        return self.tensor(arg, op, dtype=_I64, what="index")

    def weight(self, arg, op) -> str:
        if not isinstance(arg, torch.fx.Node) or arg in self.input_dep:
            raise EmitRefused(op, "weight must not depend on the input")
        return self.tensor(arg, op, what="weight")

    @staticmethod
    def shape(arg):
        return list(arg.meta["val"].shape)

    @staticmethod
    def ints(values, op, what):
        if not isinstance(values, (list, tuple)) or not all(isinstance(v, int) and not isinstance(v, bool)
                                                            for v in values):
            raise EmitRefused(op, f"{what} must be a list of ints")
        return [int(v) for v in values]

    @staticmethod
    def no_kwargs(node, op, allowed=()):
        for key, value in node.kwargs.items():
            if key not in allowed:
                raise EmitRefused(op, f"keyword argument {key}={value!r} is not supported")

    # ---- graph walk ------------------------------------------------------------------------------------------
    def run(self) -> NodeProgram:
        ep, sig = self.ep, self.sig
        specs = {spec.arg.name: spec for spec in sig.input_specs}
        state_keys = {}
        user_input = None
        for node in ep.graph.nodes:
            if node.op != "placeholder":
                break
            spec = specs[node.name]
            if spec.kind == InputKind.USER_INPUT:
                user_input = node
                name = self.fresh()
                self.ins.append(Node(name=name, ori_name="input_0", op="Parameter", inputs=[], attr="Tensor",
                                     param=None, model_input_index=0, perturbation=None))
                self.env[node] = name
                self.input_dep.add(node)
            elif spec.kind == InputKind.PARAMETER:
                p = ep.state_dict[spec.target]
                param = torch.nn.Parameter(p.detach().clone(), requires_grad=p.requires_grad)
                self.env[node] = self._state_input(spec.target, param, state_keys)
            elif spec.kind == InputKind.BUFFER and spec.persistent:
                self.env[node] = self._state_input(spec.target, ep.state_dict[spec.target].detach().clone(),
                                                   state_keys)
            elif spec.kind in (InputKind.BUFFER, InputKind.CONSTANT_TENSOR):
                value = ep.constants[spec.target]
                if not isinstance(value, torch.Tensor):
                    raise EmitRefused("placeholder", f"constant {spec.target!r} is not a tensor")
                self.env[node] = self.const(value)
            else:
                raise EmitRefused("placeholder", f"input kind {spec.kind.name} is not supported")
        if user_input is None:
            raise EmitRefused("placeholder", "no user input")

        out_node = None
        for node in ep.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "output":
                out_node = node
                break
            if node.op != "call_function":
                raise EmitRefused(node.op, f"graph node kind {node.op!r} is not supported")
            if any(a in self.input_dep for a in node.all_input_nodes):
                self.input_dep.add(node)
            op = op_name(node.target)
            handler = _HANDLERS.get(op)
            if handler is None:
                raise EmitRefused(op, "not in DIRECT_TABLE")
            val = node.meta.get("val")
            if not _static(val):
                raise EmitRefused(op, "output has no static tensor shape")
            if val.dtype != _F64:
                raise EmitRefused(op, f"output is {val.dtype}, the table requires float64")
            self.env[node] = handler(self, node, op)

        outs = [spec for spec in sig.output_specs if spec.kind == OutputKind.USER_OUTPUT]
        if len(outs) != 1 or len(sig.output_specs) != 1:
            raise EmitRefused("output", "exactly one user output and no mutated state are required")
        result = out_node.args[0]
        result = result[0] if isinstance(result, (list, tuple)) else result
        return self._finish(self.env[result], state_keys)

    def _state_input(self, key, value, state_keys) -> str:
        name = self.fresh()
        state_keys[name] = key
        self.ins.append(Node(name=name, ori_name=key, op="Parameter", inputs=[], attr="Tensor", param=value,
                             model_input_index=None, perturbation=None))
        return name

    def _finish(self, out: str, state_keys) -> NodeProgram:
        # Dead-code elimination (the tracer's DCE), then canonical names: inputs first, then ops.
        by_name = {n.name: n for n in self.ops}
        live, stack = set(), [out]
        while stack:
            name = stack.pop()
            if name in live:
                continue
            live.add(name)
            if name in by_name:
                stack.extend(by_name[name].inputs)
        ops = [n for n in self.ops if n.name in live]
        state = [n for n in self.ins if n.name in live and n.model_input_index is None]
        if self.state_order is not None:
            rank = {k: i for i, k in enumerate(self.state_order)}
            missing = [n.ori_name for n in state if n.ori_name not in rank]
            if missing:
                raise EmitRefused("placeholder", f"state keys not in the model's state_dict: {missing}")
            state.sort(key=lambda n: rank[n.ori_name])
        ins = [n for n in self.ins if n.model_input_index == 0] + state
        rename = {n.name: f"/{i}" for i, n in enumerate(ins + ops)}
        nodes_in = [n._replace(name=rename[n.name]) for n in ins]
        params = {n.name: n.param for n in nodes_in}
        nodes_op = []
        for n in ops:
            inputs = [rename[i] for i in n.inputs]
            param = OrderedDict((i, params[i]) for i in inputs if i in params)
            nodes_op.append(n._replace(name=rename[n.name], inputs=inputs, param=param))
        return NodeProgram(nodes_op, nodes_in, [rename[out]], None)


# ---- handlers: one per DIRECT_TABLE entry ----------------------------------------------------------------------
def _binary(onnx_op):
    def handler(e, node, op):
        e.no_kwargs(node, op, allowed=("alpha",))
        if node.kwargs.get("alpha", 1) != 1:
            raise EmitRefused(op, "alpha != 1")
        a, b = node.args[:2]
        if len(node.args) > 2:
            raise EmitRefused(op, "alpha != 1")
        return e.emit(onnx_op, [e.operand(a, op), e.operand(b, op)])
    return handler


def _unary(onnx_op):
    def handler(e, node, op):
        e.no_kwargs(node, op)
        return e.emit(onnx_op, [e.tensor(node.args[0], op)])
    return handler


def _minmax(onnx_op):
    def handler(e, node, op):
        e.no_kwargs(node, op)
        a, b = node.args
        if not (a in e.input_dep and b in e.input_dep):
            # The always-on minmax_to_relu rewrite turns max(x, c) into relu(x - c) + c, which is not bitwise max.
            raise EmitRefused(op, "an operand does not depend on the input")
        return e.emit(onnx_op, [e.tensor(a, op), e.tensor(b, op)])
    return handler


def _pow(e, node, op):
    e.no_kwargs(node, op)
    x, exponent = node.args
    if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
        raise EmitRefused(op, "exponent must be a constant number")
    if float(exponent) in POW_NOT_BITWISE:
        raise EmitRefused(op, f"exponent {exponent} is special-cased by torch's pow(Tensor, Scalar) kernel; BoundPow "
                              "computes std::pow, which differs in the last bit (conformance)")
    return e.emit("onnx::Pow", [e.tensor(x, op), e.const(torch.tensor(float(exponent), dtype=_F64))])


def _clamp(e, node, op):
    e.no_kwargs(node, op)
    args = list(node.args) + [None] * (3 - len(node.args))
    x, lo, hi = args[:3]
    for bound in (lo, hi):
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            raise EmitRefused(op, "both bounds must be constant numbers")
    xs = e.tensor(x, op)
    lo_c = e.emit("onnx::Cast", [e.const(torch.tensor(float(lo), dtype=_F64))], {"to": _CAST_TO_DOUBLE})
    hi_c = e.emit("onnx::Cast", [e.const(torch.tensor(float(hi), dtype=_F64))], {"to": _CAST_TO_DOUBLE})
    return e.emit("onnx::Clip", [xs, lo_c, hi_c])


def _gemm_rhs(e, arg, op):
    """(name, transB) for the right factor of a Gemm: ``t(W)`` of a constant 2-D W fuses into transB=1."""
    if (isinstance(arg, torch.fx.Node) and arg.op == "call_function" and op_name(arg.target) == "t"
            and arg not in e.input_dep and len(e.shape(arg.args[0])) == 2):
        return e.weight(arg.args[0], op), 1
    return e.weight(arg, op), 0


def _gemm_attrs(beta, trans_b):
    attr = {"alpha": 1.0, "beta": beta}
    if trans_b:
        attr["transB"] = 1
    return attr


def _mm(e, node, op):
    e.no_kwargs(node, op)
    a, b = node.args
    rhs, trans_b = _gemm_rhs(e, b, op)
    return e.emit("onnx::Gemm", [e.tensor(a, op), rhs], _gemm_attrs(0.0, trans_b))


def _addmm(e, node, op):
    e.no_kwargs(node, op, allowed=("beta", "alpha"))
    if node.kwargs.get("beta", 1) != 1 or node.kwargs.get("alpha", 1) != 1 or len(node.args) > 3:
        raise EmitRefused(op, "beta and alpha must be 1")
    c, a, b = node.args
    rhs, trans_b = _gemm_rhs(e, b, op)
    return e.emit("onnx::Gemm", [e.tensor(a, op), rhs, e.weight(c, op)], _gemm_attrs(1.0, trans_b))


def _linear(e, node, op):
    # torch.onnx symbolic_opset9.linear: rank-2 input with a bias -> addmm (Gemm, the transpose fused into transB);
    # otherwise MatMul(x, Transpose(W)) and then Add(bias, .).
    e.no_kwargs(node, op)
    args = list(node.args) + [None] * (3 - len(node.args))
    x, w, b = args[:3]
    xs, ws = e.tensor(x, op), e.weight(w, op)
    if len(e.shape(w)) != 2:
        raise EmitRefused(op, "weight must be 2-D")
    if b is not None and len(e.shape(x)) == 2:
        return e.emit("onnx::Gemm", [xs, ws, e.weight(b, op)], _gemm_attrs(1.0, 1))
    out = e.emit("onnx::MatMul", [xs, e.emit("onnx::Transpose", [ws], {"perm": [1, 0]})])
    if b is not None:
        out = e.emit("onnx::Add", [e.weight(b, op), out])
    return out


def _matmul(e, node, op):
    e.no_kwargs(node, op)
    a, b = node.args
    return e.emit("onnx::MatMul", [e.tensor(a, op), e.weight(b, op)])


def _t(e, node, op):
    e.no_kwargs(node, op)
    x = node.args[0]
    if len(e.shape(x)) < 2:
        return e.tensor(x, op)
    return e.emit("onnx::Transpose", [e.tensor(x, op)], {"perm": [1, 0]})


def _index_select(e, node, op):
    e.no_kwargs(node, op)
    x, dim, index = node.args
    return e.emit("onnx::Gather", [e.tensor(x, op), e.constant_index(index, op)], {"axis": int(dim)})


def _index(e, node, op):
    e.no_kwargs(node, op)
    x, indices = node.args
    given = [(i, idx) for i, idx in enumerate(indices) if idx is not None]
    if len(given) != 1:
        raise EmitRefused(op, "exactly one index tensor is supported")
    axis, index = given[0]
    if len(e.shape(index)) != 1:
        raise EmitRefused(op, "index must be 1-D")
    return e.emit("onnx::Gather", [e.tensor(x, op), e.constant_index(index, op)], {"axis": axis})


def _select(e, node, op):
    e.no_kwargs(node, op)
    x, dim, index = node.args
    return e.emit("onnx::Gather", [e.tensor(x, op), e.const(torch.tensor(int(index), dtype=_I64))],
                  {"axis": int(dim)})


def _index_add(e, node, op):
    e.no_kwargs(node, op, allowed=("alpha",))
    if node.kwargs.get("alpha", 1) != 1 or len(node.args) > 4:
        raise EmitRefused(op, "alpha must be 1")
    data, dim, index, source = node.args
    if len(e.shape(index)) != 1:
        raise EmitRefused(op, "index must be 1-D")
    return e.emit("aten::ATen", [e.tensor(data, op), e.const(torch.tensor(int(dim), dtype=_I64)),
                                 e.constant_index(index, op), e.tensor(source, op),
                                 e.const(torch.tensor(1, dtype=_I64))],
                  {"operator": "index_add", "overload_name": ""})


def _cat(e, node, op):
    e.no_kwargs(node, op)
    tensors = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else 0
    return e.emit("onnx::Concat", [e.tensor(t, op) for t in tensors], {"axis": int(dim)})


def _reshape(e, node, op):
    e.no_kwargs(node, op)
    x, shape = node.args
    shape = e.ints(shape, op, "shape")
    if e.shape(x) == e.shape(node):
        return e.tensor(x, op)  # identity: no node (module docstring)
    return e.emit("onnx::Reshape", [e.tensor(x, op), e.int_const(shape)], {"allowzero": 0})


def _flatten(e, node, op):
    # nn.Flatten; the exporter emits onnx::Flatten(axis=start_dim) for end_dim = -1.
    e.no_kwargs(node, op)
    args = list(node.args) + [None] * (3 - len(node.args))
    x, start, end = args[:3]
    start = 0 if start is None else int(start)
    end = -1 if end is None else int(end)
    rank = len(e.shape(x))
    if end % rank != rank - 1:
        raise EmitRefused(op, "end_dim must be the last dimension")
    return e.emit("onnx::Flatten", [e.tensor(x, op)], {"axis": start % rank})


def _unsqueeze(e, node, op):
    e.no_kwargs(node, op)
    x, dim = node.args
    return e.emit("onnx::Unsqueeze", [e.tensor(x, op), e.int_const([int(dim)])])


def _squeeze(e, node, op):
    e.no_kwargs(node, op)
    x, dim = node.args
    return e.emit("onnx::Squeeze", [e.tensor(x, op), e.int_const([int(dim)])])


def _slice(e, node, op):
    e.no_kwargs(node, op)
    args = list(node.args) + [None] * (5 - len(node.args))
    x, dim, start, end, step = args[:5]
    dim = 0 if dim is None else int(dim)
    start = 0 if start is None else int(start)
    end = INT64_MAX if end is None else int(end)
    step = 1 if step is None else int(step)
    if step != 1:
        raise EmitRefused(op, "step must be 1")
    if e.shape(x) == e.shape(node):
        return e.tensor(x, op)  # full range: no node
    return e.emit("onnx::Slice", [e.tensor(x, op), e.int_const([start]), e.int_const([end]), e.int_const([dim]),
                                  e.int_const([step])])


def _expand(e, node, op):
    e.no_kwargs(node, op)
    x, size = node.args[:2]
    size = e.ints(size, op, "size")
    # torch's -1 keeps the dimension; ONNX Expand broadcasts, where 1 keeps it (the exporter's Where(-1 -> 1)).
    return e.emit("onnx::Expand", [e.tensor(x, op), e.int_const([1 if s == -1 else s for s in size])])


def _permute(e, node, op):
    e.no_kwargs(node, op)
    x, dims = node.args
    return e.emit("onnx::Transpose", [e.tensor(x, op)], {"perm": e.ints(dims, op, "dims")})


def _transpose(e, node, op):
    e.no_kwargs(node, op)
    x, d0, d1 = node.args
    rank = len(e.shape(x))
    d0, d1 = int(d0) % rank, int(d1) % rank
    if d0 == d1:
        return e.tensor(x, op)
    perm = list(range(rank))
    perm[d0], perm[d1] = perm[d1], perm[d0]
    return e.emit("onnx::Transpose", [e.tensor(x, op)], {"perm": perm})


def _sum(e, node, op):
    e.no_kwargs(node, op, allowed=("dtype",))
    if node.kwargs.get("dtype") is not None:
        raise EmitRefused(op, "dtype must be None")
    args = list(node.args) + [False] * (3 - len(node.args))
    x, dims, keepdim = args[:3]
    dims = e.ints(dims, op, "dims") if dims is not None else []
    if not dims:
        raise EmitRefused(op, "a reduction over all dimensions is not in the table")
    return e.emit("onnx::ReduceSum", [e.tensor(x, op), e.int_const(dims)], {"keepdims": int(bool(keepdim))})


def _to_copy(e, node, op):
    e.no_kwargs(node, op, allowed=("dtype", "layout", "device", "memory_format"))
    kw = node.kwargs
    if kw.get("dtype", _F64) != _F64 or kw.get("layout") not in (None, torch.strided):
        raise EmitRefused(op, "only float64 -> float64 is supported")
    device = kw.get("device")
    if device is not None and torch.device(device).type != "cpu":
        raise EmitRefused(op, "a device change is not supported")
    return e.emit("onnx::Cast", [e.tensor(node.args[0], op)], {"to": _CAST_TO_DOUBLE})


_HANDLERS = {
    "add.Tensor": _binary("onnx::Add"), "sub.Tensor": _binary("onnx::Sub"), "mul.Tensor": _binary("onnx::Mul"),
    "neg": _unary("onnx::Neg"), "reciprocal": _unary("onnx::Reciprocal"), "pow.Tensor_Scalar": _pow,
    **{name: _unary(onnx_op) for name, onnx_op in _UNARY.items()},
    "minimum": _minmax("onnx::Min"), "maximum": _minmax("onnx::Max"), "clamp": _clamp,
    "mm": _mm, "addmm": _addmm, "linear": _linear, "matmul": _matmul, "t": _t,
    "index_select": _index_select, "index.Tensor": _index, "select.int": _select, "index_add": _index_add,
    "cat": _cat, "view": _reshape, "reshape": _reshape, "_unsafe_view": _reshape, "flatten.using_ints": _flatten,
    "unsqueeze": _unsqueeze, "squeeze.dim": _squeeze, "slice.Tensor": _slice, "expand": _expand,
    "permute": _permute, "transpose.int": _transpose, "sum.dim_IntList": _sum, "_to_copy": _to_copy,
}
assert set(_HANDLERS) == set(DIRECT_TABLE)


def emit_program(ep, state_order=None) -> NodeProgram:
    """The node program of ``ep``. ``state_order``: the state_dict keys of the module ``LiRPANet`` will receive;
    parameter and buffer nodes are named by them (``ori_name``) and ordered by them, as ``parse_module`` does."""
    return _Emitter(ep, state_order).run()
