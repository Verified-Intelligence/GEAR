# complete_verifier/tests/test_structure_walk.py
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from auto_LiRPA import BoundedModule
from nlp_solver.structure.forms import FormAlgebra
from nlp_solver.structure.walk import build_forms, output_forms


def forms_of(model, n, budget=None, lower=None):
    x = torch.full((1, n), 0.7, dtype=torch.float64)
    bm = BoundedModule(model, x, bound_opts={'pow_negative_exponent': True})
    algebra = FormAlgebra(n, lower) if budget is None else FormAlgebra(n, lower, budget)
    return output_forms(build_forms(bm, x, algebra)), model


def evaluate(form, point):
    total = 0.0
    for mono, coef in form.poly.items():
        term = float(coef)
        for var, power in mono:
            term *= float(point[var]) ** float(power)
        total += term
    return total


class Mixed(nn.Module):
    def forward(self, x):
        a, b = x[:, :1], x[:, 1:2]
        return torch.cat([a * b + a ** 2, 0.1 * a + b / 3.0, torch.exp(b), a / b], 1)


def test_polynomial_outputs_match_the_model_with_float_constants():
    outs, model = forms_of(Mixed(), 2)
    point = torch.tensor([[0.3, 1.7]], dtype=torch.float64)
    values = model(point).reshape(-1)
    for k in (0, 1, 3):
        assert outs[k].poly is not None
        assert abs(evaluate(outs[k], point[0]) - float(values[k])) <= 1e-9 * max(1, abs(float(values[k])))
    assert outs[2].poly is not None and outs[2].poly != {}   # exp(b): an atom


class ConcatConstant(nn.Module):
    def forward(self, x):
        const = torch.full((x.shape[0], 1), 2.5, dtype=x.dtype)
        return torch.cat([x[:, :1] * x[:, 1:2], const, x[:, 1:2]], 1)


def test_concat_with_a_constant_operand_keeps_every_row():
    outs, _ = forms_of(ConcatConstant(), 2)
    assert all(o.poly is not None for o in outs)
    assert outs[1].poly == {(): 2.5}


class IndexAdd(nn.Module):
    def forward(self, x):
        out = torch.zeros(x.shape[0], 3, dtype=x.dtype)
        return out.index_add(1, torch.tensor([0, 0, 2]), torch.cat([x, x[:, :1] ** 2], 1), alpha=0.5)


def test_index_add():
    outs, model = forms_of(IndexAdd(), 2)
    point = torch.tensor([[0.3, 1.7]], dtype=torch.float64)
    for k in range(3):
        assert abs(evaluate(outs[k], point[0]) - float(model(point)[0, k])) <= 1e-12


def test_budget_marks_forms_unknown_instead_of_raising():
    outs, _ = forms_of(Mixed(), 2, budget=1)
    assert any(o.poly is None and o.reason == 'budget' for o in outs)


class Overflow(nn.Module):
    def forward(self, x):
        a = x[:, :1]
        big = a * 0 + 1000.0        # input-dependent node whose form is the constant 1000
        return torch.cat([a * a + 1, torch.exp(big)], 1)


def test_constant_overflow_marks_the_node_unknown_and_the_walk_continues():
    outs, _ = forms_of(Overflow(), 2)
    assert outs[0].poly is not None
    point = torch.tensor([[0.3, 1.7]], dtype=torch.float64)
    assert abs(evaluate(outs[0], point[0]) - 1.09) <= 1e-12
    assert outs[1].poly is None and outs[1].reason == 'exc:OverflowError'


class RoundedZeroMatmul(nn.Module):
    # [exp(-1000), x0] @ [x1, x2]: the first entry rounds to 0 but is not 0
    def forward(self, x):
        tiny = torch.exp(-1000 + 0 * x[:, :1])
        return torch.cat([tiny, x[:, :1]], 1) @ x[:, 1:3].t()


def test_a_symbolic_matmul_keeps_the_taint_of_a_rounded_zero_entry():
    outs, _ = forms_of(RoundedZeroMatmul(), 3)
    assert outs[0].poly == {((0, 1), (2, 1)): 1} and outs[0].inexact
