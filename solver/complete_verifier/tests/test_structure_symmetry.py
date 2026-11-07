import itertools
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from auto_LiRPA import BoundedModule
from input_split.batch_branch_and_bound import _add_symmetry_rows
from nlp_solver.structure import build_structure
from nlp_solver.structure.symmetry import ordering_pairs, sort_blocks, symmetry_chains


class Quartic(nn.Module):
    """-sum_{i<j} x_i^2 x_j^2 (x_i - x_j)^2 (schitt_s368) and the row sum x <= 2; `bias` perturbs one
    coefficient, `rhs_row` the right-hand side of a second row sum x^2 <= 3."""

    def __init__(self, n=4, odd=None):
        super().__init__()
        self.n, self.odd = n, odd

    def forward(self, x):
        o = 0
        for i, j in itertools.combinations(range(self.n), 2):
            o = o - x[:, i:i + 1] ** 2 * x[:, j:j + 1] ** 2 * (x[:, i:i + 1] - x[:, j:j + 1]) ** 2
        if self.odd is not None:
            o = o + (1 + self.odd) * x[:, 0:1] ** 2
        return torch.cat([o, x.sum(1, keepdim=True), (x ** 2).sum(1, keepdim=True)], 1)


class Points(nn.Module):
    """-sum_{i<j} log(|p_i - p_j|^2) for points p_i = (x[a_i], x[b_i]) with scrambled indices, and
    |p_i|^2 <= 1 for every point"""

    def __init__(self, perm):
        super().__init__()
        self.perm = perm

    def forward(self, x):
        m = len(self.perm) // 2
        p = [(x[:, self.perm[i]:self.perm[i] + 1], x[:, self.perm[m + i]:self.perm[m + i] + 1]) for i in range(m)]
        o = 0
        for i, j in itertools.combinations(range(m), 2):
            o = o - torch.log((p[i][0] - p[j][0]) ** 2 + (p[i][1] - p[j][1]) ** 2)
        return torch.cat([o] + [p[i][0] ** 2 + p[i][1] ** 2 for i in range(m)], 1)


def report(model, n, rhs=None, lower=-1., upper=1.):
    x = torch.zeros(1, n, dtype=torch.float64) + 0.3
    bm = BoundedModule(model, x)
    with torch.no_grad():
        m = model(x).shape[1]
    C = torch.eye(m, dtype=torch.float64).unsqueeze(0)
    rhs = torch.zeros(1, m, dtype=torch.float64) if rhs is None else rhs
    lo = torch.full((1, n), lower, dtype=torch.float64)
    hi = torch.full((1, n), upper, dtype=torch.float64)
    return build_structure(bm, x, C, rhs, lo, hi), rhs, lo, hi


def chains(model, n, rhs=None, **kw):
    r, rhs, lo, hi = report(model, n, rhs, **kw)
    return symmetry_chains(r, rhs[0, 1:].tolist(), lo[0].numpy(), hi[0].numpy())


def test_single_variables_are_one_chain():
    rhs = torch.tensor([[0., 2., 3.]], dtype=torch.float64)
    found = chains(Quartic(4), 4, rhs)
    assert [sorted(b[0] for b in c) for c in found] == [[0, 1, 2, 3]]
    assert ordering_pairs(found) == [(c[t][0], c[t + 1][0]) for c in found for t in range(3)]


def test_one_changed_coefficient_or_rhs_kills_the_variable():
    rhs = torch.tensor([[0., 2., 3.]], dtype=torch.float64)
    found = chains(Quartic(4, odd=1e-9), 4, rhs)         # x0 no longer interchangeable with the others
    assert [sorted(b[0] for b in c) for c in found] == [[1, 2, 3]]
    assert chains(Quartic(2, odd=1e-9), 2, rhs[:, :3]) == []
    box = Quartic(3)
    r, rhs3, lo, hi = report(box, 3, torch.tensor([[0., 2., 3.]], dtype=torch.float64))
    hi = hi.clone()
    hi[0, 2] = 0.5                                         # unequal boxes
    found = symmetry_chains(r, rhs3[0, 1:].tolist(), lo[0].numpy(), hi[0].numpy())
    assert [sorted(b[0] for b in c) for c in found] == [[0, 1]]


def test_rows_are_compared_with_their_right_hand_sides():
    r, rhs, lo, hi = report(Quartic(3), 3, torch.tensor([[0., 2., 3.]], dtype=torch.float64))
    # one more affine row on x0 only: x0 <= 0.5 breaks the symmetry of x0
    A, b = np.array([[1., 0., 0.]]), np.array([-0.5])
    found = symmetry_chains(r, rhs[0, 1:].tolist(), lo[0].numpy(), hi[0].numpy(), A, b)
    assert [sorted(c[0] for c in ch) for ch in found] == [[1, 2]]
    A, b = np.ones((1, 3)), np.array([-0.5])             # symmetric input row
    found = symmetry_chains(r, rhs[0, 1:].tolist(), lo[0].numpy(), hi[0].numpy(), A, b)
    assert [sorted(c[0] for c in ch) for ch in found] == [[0, 1, 2]]


class RoundedRow(Quartic):
    """Quartic plus the symmetric row sqrt(2) sum x + (sum x^2 + 1)^2: a rounded constant and a pow atom, whose
    expansion is a polynomial"""

    def forward(self, x):
        s = (x[:, :1] * 0 + 2.0) ** 0.5
        extra = s * x.sum(1, keepdim=True) + ((x ** 2).sum(1, keepdim=True) + 1) ** 2
        return torch.cat([super().forward(x), extra], 1)


def test_a_rounded_constant_hides_no_symmetry_behind_an_expansion():
    rhs = torch.tensor([[0., 2., 3., 5.]], dtype=torch.float64)
    assert [sorted(b[0] for b in c) for c in chains(Quartic(4), 4, rhs[:, :3])] == [[0, 1, 2, 3]]
    assert chains(RoundedRow(4), 4, rhs) == []


def test_non_finite_right_hand_sides_skip_the_detection():
    r, rhs, lo, hi = report(Quartic(3), 3, torch.tensor([[0., 2., 3.]], dtype=torch.float64))
    assert symmetry_chains(r, [2., math.inf], lo[0].numpy(), hi[0].numpy()) == []
    assert symmetry_chains(r, rhs[0, 1:].tolist(), lo[0].numpy(), hi[0].numpy(),
                           np.ones((1, 3)), np.array([math.nan])) == []


def test_blocks_with_scrambled_indices_and_atoms():
    perm = [5, 0, 7, 2, 9, 4, 1, 6, 3, 8]          # point i = (x[perm[i]], x[perm[5 + i]])
    found = chains(Points(perm), 10)
    assert len(found) == 1 and len(found[0]) == 5 and all(len(b) == 2 for b in found[0])
    points = {(perm[i], perm[5 + i]) for i in range(5)}
    assert {tuple(b) for b in found[0]} == points      # x with x, y with y, in every block
    assert all(b[0] in perm[:5] for b in found[0])


def test_minimum_is_unchanged_by_the_ordering_rows():
    n = 3
    r, rhs, lo, hi = report(Quartic(n), n, torch.tensor([[0., 2., 3.]], dtype=torch.float64))
    found = symmetry_chains(r, rhs[0, 1:].tolist(), lo[0].numpy(), hi[0].numpy())
    pairs = ordering_pairs(found)
    grid = torch.tensor(list(itertools.product(np.linspace(-1, 1, 21), repeat=n)), dtype=torch.float64)
    with torch.no_grad():
        out = Quartic(n)(grid)
    ok = (out[:, 1] <= 2) & (out[:, 2] <= 3)
    keep = ok.clone()
    for a, b in pairs:
        keep &= grid[:, a] <= grid[:, b]
    assert pairs and keep.sum() < ok.sum()
    assert out[ok, 0].min() == out[keep, 0].min()      # the grid is symmetric: exact equality


def test_sort_blocks_keeps_coordinates_together():
    chain = [[(0, 3), (1, 4), (2, 5)]]
    x = torch.tensor([0.9, 0.1, 0.5, 10., 20., 30.], dtype=torch.float64)
    y = sort_blocks(x, chain)
    assert y.tolist() == [0.1, 0.5, 0.9, 20., 30., 10.]
    assert torch.equal(sort_blocks(y, chain), y)


def test_add_symmetry_rows_extends_base_constraints_and_sorts_incumbents():
    r, rhs, lo, hi = report(Quartic(3), 3, torch.tensor([[0., 2., 3.]], dtype=torch.float64))
    x = SimpleNamespace(ptb=SimpleNamespace(x_L=lo, x_U=hi))
    domains = SimpleNamespace(base_constraint_A=None, base_constraint_b=None)
    point = torch.tensor([[0.7, -0.2, 0.1]], dtype=torch.float64)
    base, (sorted_point, none) = _add_symmetry_rows(domains, r, rhs, x, None, [point, None])
    A, b = base
    assert A.shape == (1, 2, 3) and torch.count_nonzero(b) == 0 and none is None
    assert (A @ sorted_point.reshape(3) + b[0] <= 0).all()      # the sorted incumbent satisfies the new rows
    assert domains.base_constraint_A.shape == (1, 2, 3)
    assert sorted(sorted_point.reshape(-1).tolist()) == sorted(point.reshape(-1).tolist())
    ordered = (A, b)                                   # an input chain x0 <= x1 <= x2 is itself not symmetric
    same, _ = _add_symmetry_rows(domains, r, rhs, x, ordered, [point])
    assert same is ordered
