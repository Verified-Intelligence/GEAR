"""G5 v1c (CUDA kit run 2 follow-up, A291), CPU only.

A291: WP12a's attach_program handed the WP6 seam deep copies; ``_convert_nodes`` then turned each copied Parameter into
a plain tensor with ``param.to(device)`` on CUDA, so the Bound graph had BoundBuffers where parse_module has BoundParams
(and CROWN bounds differed by 4.65e-5 on V100). The device step of ``_convert_nodes`` (auto_LiRPA/bound_general.py:
``model.to(device)``; ``param.to(device)``; BoundParams iff ``nn.Parameter``) is replayed here with float32 standing in
for CUDA: ``Module.to`` moves Parameters in place for a CPU->CUDA move and for a dtype move alike (a shallow copy), so a
node tensor stays a Parameter exactly when it is the model's own. 'meta' cannot stand in: ``Module.to('meta')`` makes
new Parameter objects (no compatible shallow copy), and a BoundedModule cannot be built on meta.
"""
import copy

import numpy as np
import pytest
import torch
from torch import nn

import gear as gr
from auto_LiRPA.parse_graph import parse_module
from gear.io import torch_direct as td
from tests import test_gear_wp12a_nn_smoke as smoke
from tests import test_gear_wp12a_torch_direct as tdt


def _device_classes(model, nodes_in, target=torch.float32):
    """The leaf class _convert_nodes gives each node tensor after ``model.to(target)`` and ``param.to(target)``."""
    model.to(target)
    return [(tuple(n.param.shape), "BoundParams" if isinstance(n.param.to(target), nn.Parameter) else "BoundBuffers")
            for n in nodes_in if n.param is not None]


def _cases():
    return [("mlp", lambda: tdt.MLP(), tdt.BOX3, dict(le=[1])),
            ("lifted", lambda: tdt.Lifted(), tdt.BOX2, {}),
            ("gated_residual", lambda: smoke._make(smoke.GatedResidual), (np.zeros(20), np.ones(20)), dict(le=[1, 2, 3, 4]))]


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.mark.parametrize("name, ctor, box, kw", _cases(), ids=[c[0] for c in _cases()])
def test_attach_program_keeps_parse_module_classes_on_another_device(name, ctor, box, kw):
    src = ctor()
    tc = td.compile_torch(src, box=box, objective=0, **kw)
    x0 = torch.zeros(1, tc.num_inputs)
    ref_model = copy.deepcopy(src)
    _, ref_in, _, _ = parse_module(ref_model, (x0,))
    ref = _device_classes(ref_model, ref_in)
    g = td.attach_program(td.computing_graph(tc), tc.program)
    _, direct_in, _, _ = g.__gear_program__(g, (x0,))
    direct = _device_classes(g, direct_in)
    assert sorted(direct) == sorted(ref) and any(c == "BoundParams" for _, c in ref)
    # the program's own tensors are never handed out (ownership, DESIGN H3)
    own = {n.param.data_ptr() for n in tc.program.nodes_in if n.param is not None}
    assert not own & {n.param.data_ptr() for n in direct_in if n.param is not None}


def test_attach_program_refuses_a_model_that_differs_from_the_program():
    tc = td.compile_torch(tdt.MLP(), box=tdt.BOX3, objective=0, le=[1])
    g = td.attach_program(td.computing_graph(tc), tc.program)
    with torch.no_grad():
        next(p for p in g.parameters()).add_(1.0)
    with pytest.raises(gr.InternalError, match="differs from the compiled program"):
        g.__gear_program__(g, (torch.zeros(1, 3),))
