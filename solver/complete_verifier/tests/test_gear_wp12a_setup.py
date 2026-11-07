"""Commit mode "whole" in the gear setup (G5 WP12a, DESIGN section 4.1). CPU, float64."""
import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from auto_LiRPA import BoundedModule
from auto_LiRPA.op_budget import OpBudgetExhausted
from gear.io import torch_direct as td
from gear.lower.setup import build_bounded_module

OP_BUDGET = {"hidden_squares": 1 << 20, "complete_squares": 1 << 24, "prove_rows": 1 << 23, "commit": "partial"}
OPTIONS = {"domain_aware_ops": True, "compare_crown_with_ibp_intermediate": True,
           "undefined_relaxation_rows": True, "hidden_square_rewrite": True}


class HiddenSquare(nn.Module):
    """As tests/test_hidden_square_rewrite.py: u * v with v = 100 u hidden in the graph."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(100.0, dtype=torch.float64))

    def forward(self, x):
        x0, x1 = x[:, :1], x[:, 1:2]
        u = x0 * x0 - x1
        v = x0 * (x0 * self.scale) - x1 * self.scale
        return (u * v).sum(dim=-1, keepdim=True)


def signature(bm):
    h = hashlib.sha256()
    for n in bm.nodes():
        h.update(f"{n.name}|{type(n).__name__}|{[i.name for i in n.inputs]}".encode())
        v = getattr(n, "value", None)
        if torch.is_tensor(v):
            h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


def opts(**budget):
    return dict(OPTIONS, op_budget=dict(OP_BUDGET, **budget))


def build(bound_opts, model=None):
    model = HiddenSquare() if model is None else model
    return BoundedModule(model, torch.zeros(1, 2, dtype=torch.float64), bound_opts=bound_opts, device="cpu")


def test_whole_mode_rebuilds_an_exhausted_stage_with_the_stage_off():
    off = build(dict(opts(), hidden_square_rewrite=False))
    full = build(opts())
    assert signature(full) != signature(off)  # the rewrite applies with the full budget
    with pytest.raises(OpBudgetExhausted):
        build(opts(hidden_squares=1, commit="whole"))
    net = build_bounded_module(build, opts(hidden_squares=1, commit="whole"))
    assert signature(net) == signature(off)
    assert net.gear_anomalies == ("op_budget_exhausted:hidden_squares:1",)


def test_partial_mode_and_unexhausted_whole_mode_are_unchanged():
    assert signature(build_bounded_module(build, opts(hidden_squares=1))) == signature(build(opts(hidden_squares=1)))
    net = build_bounded_module(build, opts(commit="whole"))
    assert signature(net) == signature(build(opts())) and net.gear_anomalies == ()


def test_whole_mode_on_a_direct_torch_program():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        tc = td.compile_torch(HiddenSquare(), box=(np.array([-1.0, -1.0]), np.array([1.0, 1.0])), objective=0)
        x0 = torch.zeros(1, 2)

        def direct(bound_opts):
            return BoundedModule(td.attach_program(td.computing_graph(tc), tc.program), (x0,),
                                 bound_opts=bound_opts, device="cpu")
        off = direct(dict(opts(), hidden_square_rewrite=False))
        net = build_bounded_module(direct, opts(hidden_squares=1, commit="whole"))
        assert [type(n).__name__ for n in net.nodes()] == [type(n).__name__ for n in off.nodes()]
        assert net.gear_anomalies == ("op_budget_exhausted:hidden_squares:1",)
    finally:
        torch.set_default_dtype(old)
