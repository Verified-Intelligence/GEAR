"""The ``__gear_program__`` supplier for the WP6 seam in ``BoundedModule._convert_nodes`` (G5 DESIGN section 2).

``_convert_nodes`` moves the model with ``model.to(device)`` and then each node tensor with ``param.to(device)``;
an ``nn.Parameter`` stays a Parameter (``BoundParams``) only if it is the model's own tensor, which ``model.to`` moved
in place. ``parse_module`` binds the model's own tensors, so every supplier must too: a copied Parameter becomes a
plain tensor on another device, i.e. ``BoundBuffers`` (anomaly A291: CUDA bounds differed from parse_module's).
"""
from __future__ import annotations

import copy
from collections import OrderedDict

import torch

from ..errors import InternalError


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape or a.layout != b.layout:
        return False
    a, b = a.detach().cpu().contiguous().reshape(-1), b.detach().cpu().contiguous().reshape(-1)
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


class ProgramSource:
    """``__gear_program__``: the compiled node program, each parameter and buffer bound to the live model's own
    tensor (as ``parse_module`` binds them) after checking that the two are bitwise equal. Op attributes and the
    template are fresh copies per build; the program's own tensors are never handed out. A deep copy of the model
    shares this object. A perturbation carried by the first input is set on the input node, as parse_module does."""

    def __init__(self, program):
        self.program = program

    def __deepcopy__(self, memo):
        return self

    def __call__(self, model, inputs):
        nodes_op, nodes_in, nodes_out, template = self.program
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        ptb = getattr(x, "ptb", None)
        live = model.state_dict(keep_vars=True)
        bound_in = []
        for n in nodes_in:
            if n.param is not None:
                t = live.get(n.ori_name)
                if t is None or not same_bits(t, n.param):
                    raise InternalError(f"__gear_program__: {n.ori_name!r} of the live model differs from the compiled "
                                        "program")
                n = n._replace(param=t)
            elif ptb is not None and n.model_input_index == 0:
                n = n._replace(perturbation=ptb)
            bound_in.append(n)
        params = {n.name: n.param for n in bound_in}
        bound_op = [n._replace(attr=copy.deepcopy(n.attr), param=OrderedDict(
            (k, params[k] if k in params else copy.deepcopy(v)) for k, v in n.param.items())) for n in nodes_op]
        return bound_op, bound_in, list(nodes_out), copy.deepcopy(template)
