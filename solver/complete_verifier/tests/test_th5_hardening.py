"""TH5: q >= 0 holds under _single_pass (on device, -inf on failure); the lane-compaction check raises;
the reported D includes min(D_M2, T)."""
import inspect
import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from nlp_solver.rigorous_dual import Rows, quadratic_lower_bound  # noqa: E402

F64 = torch.float64


def _args(q):
    B, n = q.shape
    G = torch.tensor([[[1.0, 1.0]]], dtype=F64).expand(B, 1, n).contiguous()
    return (Rows.from_dense(G), q, torch.zeros(B, n, dtype=F64), torch.zeros(B, dtype=F64),
            torch.ones(B, 1, dtype=F64), -torch.ones(B, n, dtype=F64), torch.ones(B, n, dtype=F64),
            torch.ones(B, 1, dtype=F64))


def test_negative_curvature_gives_minus_inf_under_single_pass():
    q = torch.tensor([[1.0, 2.0], [1.0, -1e-9], [float('nan'), 1.0]], dtype=F64)
    args = _args(q)
    args[0]._single_pass = True
    lb, _, _ = quadratic_lower_bound(*args)
    assert torch.isfinite(lb[0]) and lb[1] == -math.inf and lb[2] == -math.inf


def test_negative_curvature_still_raises_without_single_pass():
    with pytest.raises(ValueError):
        quadratic_lower_bound(*_args(torch.tensor([[1.0, -1.0]], dtype=F64)))


