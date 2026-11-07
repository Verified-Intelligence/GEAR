import os
import sys
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import arguments
from auto_LiRPA import BoundedModule
from input_split.branching_heuristics import input_split_heuristic_sb, _angle_first_choice
from nlp_solver.structure import build_structure


class Rows(nn.Module):
    """objective; x0^2 + a*x1^2 = 1 as two rows (a circle only when a == 1); a bilinear
    row; the circle x2^2 + x3^2 = 1 whose second row has its constant moved by k2 - 1
    (1e-5 is the models' own equality band); a circle with a linear term (not one)"""
    def __init__(self, a=1.0, k2=1.0):
        super().__init__()
        self.a, self.k2 = a, k2

    def forward(self, x):
        c = x[:, 0:1] ** 2 + self.a * x[:, 1:2] ** 2 - 1
        d = x[:, 2:3] ** 2 + x[:, 3:4] ** 2 - 1
        e = x[:, 4:5] ** 2 + x[:, 5:6] ** 2 + x[:, 4:5] - 1
        return torch.cat([x[:, 6:7], c, -c, x[:, 0:1] * x[:, 6:7], d, -d + self.k2 - 1, e, -e], 1)


def circles(a=1.0, k2=1.0):
    x = torch.zeros(1, 7, dtype=torch.float64)
    bm = BoundedModule(Rows(a, k2), x)
    m = 8
    lo = -torch.ones(1, 7, dtype=torch.float64)
    r = build_structure(bm, x, torch.eye(m, dtype=torch.float64).unsqueeze(0),
                        torch.zeros(1, m, dtype=torch.float64), lo, -lo)
    return [I.circle for I in r.rows]


def test_detection_is_exact():
    assert circles() == [None, (0, 1), (0, 1), None, (2, 3), (2, 3), None, None]
    # a perturbed coefficient: the two squares no longer have equal coefficients
    assert circles(a=1.0 + 1e-6)[1:3] == [None, None]
    # a band 1e-5 wide is still a circle equality, a wide band (other sense 0.5 away) is not
    assert circles(k2=1.0 + 1e-5)[4:6] == [(2, 3), (2, 3)]
    assert circles(k2=1.5)[4:6] == [None, None]


def _history(root=(2.0, 2.0, 2.0, 2.0)):
    return SimpleNamespace(angle_pairs=torch.tensor([[0, 1]]), root_width=torch.tensor(root))


def _run(ratio, lo, hi, idx, history):
    arguments.Config['bab']['branching']['input_split']['angle_first_ratio'] = ratio
    try:
        return _angle_first_choice(torch.tensor(idx), torch.tensor(lo), torch.tensor(hi), history)
    finally:
        arguments.Config['bab']['branching']['input_split'].pop('angle_first_ratio')


def test_override_while_wide_and_off_when_narrow():
    h = _history()
    lo = [[-1.0, -0.2, -1.0, -1.0], [-0.04, -0.04, -1.0, -1.0], [-1.0, -0.5, -1.0, -1.0]]
    hi = [[1.0, 0.2, 1.0, 1.0], [0.04, 0.04, 1.0, 1.0], [1.0, 0.5, 1.0, 1.0]]
    # wide: widest angle variable (0, or 1 when it is wider); narrow (< 0.05 of root): keep 3
    assert _run(0.05, lo, hi, [[3], [3], [3]], h).tolist() == [[0], [3], [0]]
    lo[2][0], hi[2][0] = -0.1, 0.1
    assert _run(0.05, lo, hi, [[3], [3], [3]], h).tolist()[2] == [1]
    assert _run(0.0, lo, hi, [[3], [3], [3]], h).tolist() == [[3], [3], [3]]       # off
    assert _run(0.05, lo, hi, [[3], [3], [3]], SimpleNamespace(root_width=h.root_width)).tolist() == [[3]] * 3


def test_non_finite_and_depth_keep_choice():
    h = _history()
    nan = float('nan')
    out = _run(0.05, [[nan, nan, -1.0, -1.0]], [[nan, nan, 1.0, 1.0]], [[3]], h)
    assert out.tolist() == [[3]]
    out = _run(0.05, [[-1.0, -1.0, -1.0, -1.0]], [[float('inf')] * 2 + [1.0, 1.0]], [[3]], h)
    assert out.tolist() == [[3]]
    assert _run(0.05, [[-1.0] * 4], [[1.0] * 4], [[3, 2]], h).tolist() == [[3, 2]]  # depth 2: untouched


def test_default_path_is_bit_identical():
    torch.manual_seed(0)
    lo = -torch.rand(5, 4)
    hi = torch.rand(5, 4)
    lA = torch.randn(5, 1, 4)
    args = (lo, hi, torch.zeros(5, 1), torch.ones(5, 1), lA)
    a = input_split_heuristic_sb(*args, history=None)
    b = input_split_heuristic_sb(*args, history=_history())    # pairs present, option at its default 0
    assert torch.equal(a, b)
