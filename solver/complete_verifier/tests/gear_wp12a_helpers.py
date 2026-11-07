"""Shared helpers for the WP12a tests (direct torch path): the two evaluators and a names-free program comparison."""
from __future__ import annotations

import copy
import hashlib
import io

import numpy as np
import torch
from torch import nn

from gear.io import torch_direct as td


def f64(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def evaluators(module: nn.Module, x0: torch.Tensor):
    """(E_prog, E_ref, program) for ``module`` at the export example ``x0`` (no wrappers), as compile builds them."""
    ep = td.export_source(copy.deepcopy(module), x0)
    buf = io.BytesIO()
    torch.export.save(ep, buf)
    source = td.TorchSource(buf.getvalue(), hashlib.sha256(buf.getvalue()).hexdigest())
    program = td._emit_solver_program(td._solver_model(source, None, None, False), x0)
    e_prog = td.bounded_program(td._solver_model(source, None, None, False), program, x0)
    return e_prog, source.module().eval(), program


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().view(torch.int64)


# ---- names-free comparison of node programs (isomorphism) -------------------------------------------------------
_FOLD = {
    "onnx::Unsqueeze": lambda x, axes: _unsqueeze(x, axes),
    "onnx::Squeeze": lambda x, axes: _squeeze(x, axes),
    "onnx::Cast": None,  # handled with the attr
    "onnx::Shape": lambda x: torch.tensor(list(x.shape), dtype=torch.int64),
    "onnx::Mul": lambda a, b: a * b,
    "onnx::Equal": lambda a, b: a == b,
    "onnx::Where": lambda c, a, b: torch.where(c, a, b),
}


def _unsqueeze(x, axes):
    for a in sorted(int(v) for v in axes.reshape(-1)):
        x = x.unsqueeze(a)
    return x


def _squeeze(x, axes):
    for a in sorted((int(v) for v in axes.reshape(-1)), reverse=True):
        x = x.squeeze(a)
    return x


def _fold(node, values):
    if node.op == "onnx::Constant":
        return node.attr["value"]
    if any(v is None for v in values) or not values:
        return None
    if node.op == "onnx::Cast" and node.attr.get("to") == 11:
        return values[0].to(torch.float64)
    if node.op == "onnx::ConstantOfShape":
        return node.attr["value"].reshape(-1)[0].repeat(*[int(s) for s in values[0]]) if values[0].numel() else \
            node.attr["value"].reshape(())
    if node.op == "onnx::Concat" and all(v.dtype == torch.int64 for v in values):
        return torch.cat([v.reshape(-1) for v in values], dim=int(node.attr["axis"]))
    fn = _FOLD.get(node.op)
    return fn(*values) if fn is not None else None


def _tkey(t: torch.Tensor):
    t = t.detach().cpu().contiguous()
    return ("T", str(t.dtype), tuple(t.shape), hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes())
            .hexdigest())


def _akey(v):
    if isinstance(v, torch.Tensor):
        return _tkey(v)
    if isinstance(v, (list, tuple)):
        return tuple(_akey(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _akey(x)) for k, x in v.items()))
    return v


def canonical(program, intern=None):
    """An interned id per output node: two programs are isomorphic (same ops, attrs, inputs, constants and parameter
    values; names excluded) iff their ids are equal. Integer constant plumbing (Unsqueeze/Shape/Where/... of
    Constants) is folded to its value on both sides, so the exporter's scalar-Constant/Unsqueeze form of a Slice equals
    a 1-D Constant. Returns (ids of nodes_out, intern table)."""
    nodes_op, nodes_in, nodes_out = program[0], program[1], program[2]
    intern = {} if intern is None else intern
    ins = {n.name: n for n in nodes_in}
    ops = {n.name: n for n in nodes_op}
    ids, consts = {}, {}

    def ident(key):
        return intern.setdefault(key, len(intern))

    order = [n.name for n in nodes_in] + [n.name for n in nodes_op]
    for name in order:
        if name in ins:
            n = ins[name]
            key = ("X",) if n.model_input_index is not None else \
                ("P", isinstance(n.param, nn.Parameter), _tkey(n.param))
            ids[name] = ident(key)
            continue
        n = ops[name]
        value = _fold(n, [consts.get(i) for i in n.inputs])
        if value is not None:
            consts[name] = value
            ids[name] = ident(("C", _tkey(value)))
            continue
        attr = {k: v for k, v in n.attr.items() if k != "device"}
        ids[name] = ident((n.op, _akey(attr), tuple(ids[i] for i in n.inputs)))
    return [ids[o] for o in nodes_out], intern


def isomorphic(prog_a, prog_b) -> bool:
    intern = {}
    a, intern = canonical(prog_a, intern)
    b, _ = canonical(prog_b, intern)
    return a == b


def parse_program(module: nn.Module, x0: torch.Tensor):
    from auto_LiRPA.parse_graph import parse_module
    return parse_module(module, (x0,))


EDGE = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0, 1e-300, -1e-300, 1e300, -1e300, 5e-324, -5e-324,
        2.2250738585072014e-308, np.pi / 2, -np.pi / 2, np.pi, 1e-8, 7.25, 100.0, 709.0, 710.0, -745.0, 0.1, -0.7]
