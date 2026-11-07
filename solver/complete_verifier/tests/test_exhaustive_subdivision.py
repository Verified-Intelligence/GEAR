"""The exhaustive subdivision guard (input_split/branching_heuristics.py: _exhaustive_subdivision_guard, WP15-pre).

After as many splits as a box has open inputs have gone elsewhere, the box's widest input is split. So along any infinite
split path every input's relative width goes to 0, whatever the branching heuristics chose. The loop-level effect (E1's
iteration counts with the guard) is pinned in test_bab_loop.py."""
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import input_split.split as split  # noqa: E402
from input_split.branching_heuristics import _exhaustive_subdivision_guard  # noqa: E402
from input_split.split import input_split_and_repeat  # noqa: E402


@pytest.fixture(autouse=True)
def _one_split_per_step(monkeypatch):
    # the number of splits per step comes from the global config (batch size); the loop's one-column split_idx is one split
    monkeypatch.setattr(split, 'get_split_depth', lambda *a, **k: 1)


def _history(root):
    return SimpleNamespace(root_width=torch.as_tensor(root, dtype=torch.float64), repeat_for_split=lambda *a: None)


def _split(x_L, x_U, idx, counter, history):
    out = input_split_and_repeat(x_L, x_U, x_L.shape[0], x_L.shape, idx, 2, None, history=history, split_counter=counter)
    return out[0], out[1], out[-1]


def test_forced_after_as_many_non_widest_splits_as_there_are_open_inputs():
    """R counts the inputs of width > 0, the widest is measured relative to the root, and a widest choice resets the count"""
    root = [1.0, 1.0, 1.0, 1.0]
    width = torch.tensor([[1.0, 0.5, 0.25, 0.0]] * 5, dtype=torch.float64)
    x_L, x_U = torch.zeros_like(width), width
    choice = torch.tensor([[2], [2], [0], [1], [3]])
    count = torch.tensor([[2], [3], [3], [0], [3]], dtype=torch.int32)
    idx, new_count = _exhaustive_subdivision_guard(choice, count, x_L, x_U, _history(root))
    # 3 open inputs: count 2 keeps the choice, count 3 forces input 0; a choice of input 0 or a closed input (width 0)
    assert idx.flatten().tolist() == [2, 0, 0, 1, 0]
    assert new_count.flatten().tolist() == [2, -1, -1, 0, -1]
    assert new_count.dtype == count.dtype and new_count.shape == count.shape and choice.flatten().tolist() == [2, 2, 0, 1, 3]
    # the children add 1: a widest split restarts the count at 0, any other split extends it
    _, _, child = _split(x_L, x_U, idx, new_count, _history(root))
    assert child.flatten().tolist() == [3, 0, 0, 1, 0] * 2 and child.dtype == count.dtype


def test_the_widest_input_is_the_one_with_the_largest_width_relative_to_the_root():
    x_L = torch.zeros(2, 2, dtype=torch.float64)
    x_U = torch.tensor([[1.0, 0.5]] * 2, dtype=torch.float64)           # absolute: input 0 is wider; relative: input 1
    idx, count = _exhaustive_subdivision_guard(torch.tensor([[1], [0]]), torch.tensor([[0], [0]], dtype=torch.int32),
                                               x_L, x_U, _history([10.0, 1.0]))
    assert idx.flatten().tolist() == [1, 0] and count.flatten().tolist() == [-1, 0]
    idx, count = _exhaustive_subdivision_guard(torch.tensor([[1], [0]]), torch.tensor([[2], [2]], dtype=torch.int32),
                                               x_L, x_U, _history([10.0, 1.0]))
    assert idx.flatten().tolist() == [1, 1] and count.flatten().tolist() == [-1, -1]


def _walk(root, rule, steps, guard, seed=0, boxes=16):
    """`boxes` independent paths down the split tree (the child of every split is drawn at random); `rule` is the heuristic's
    choice, which the guard may override. Returns the largest relative width of every path after `steps` splits."""
    gen = torch.Generator().manual_seed(seed)
    root = torch.tensor(root, dtype=torch.float64)
    n = root.numel()
    history = _history(root)
    x_L, x_U = (-root / 2).repeat(boxes, 1), (root / 2).repeat(boxes, 1)
    count = torch.zeros(boxes, 1, dtype=torch.int32)
    sticky = torch.randint(n, (boxes, 1), generator=gen)
    for _ in range(steps):
        rel = (x_U - x_L) / root.clamp(min=1e-300)
        if rule == 'random':
            idx = torch.randint(n, (boxes, 1), generator=gen)
        elif rule == 'sticky':                                           # SB fixing on one input, now and then another
            redraw = torch.rand(boxes, 1, generator=gen) < 0.05
            sticky = torch.where(redraw, torch.randint(n, (boxes, 1), generator=gen), sticky)
            idx = sticky
        elif rule == 'narrowest':
            idx = rel.masked_fill(root == 0, float('inf')).argmin(1, keepdim=True)
        else:
            idx = torch.zeros(boxes, 1, dtype=torch.long)
        if guard:
            idx, count = _exhaustive_subdivision_guard(idx, count, x_L, x_U, history)
        new_L, new_U, new_count = _split(x_L, x_U, idx, count, history)
        pick = (torch.rand(boxes, generator=gen) < 0.5).long() * boxes + torch.arange(boxes)   # children: [first half; second half]
        x_L, x_U, count = new_L[pick], new_U[pick], new_count[pick]
    return ((x_U - x_L) / root.clamp(min=1e-300)).max(1).values


ROOTS = [[1.0], [1.0, 4.0], [1.0, 4.0, 0.5], [1.0, 4.0, 0.0, 2.0], [1.0, 4.0, 0.5, 2.0, 8.0]]   # 0.0: a closed input


@pytest.mark.parametrize('rule', ['random', 'sticky', 'narrowest', 'first'])
@pytest.mark.parametrize('root', ROOTS, ids=lambda r: f'n{len(r)}')
def test_every_relative_width_goes_to_zero_along_any_split_path(root, rule):
    """At most n widest splits halve the largest relative width and one comes within n + 1 splits (n = open inputs), so after
    k n (n + 1) splits it is at most 2^-k. All inputs are below it: this is the largest."""
    n, k = sum(w > 0 for w in root), 5
    assert _walk(root, rule, k * n * (n + 1), guard=True).max() <= 2.0 ** -k


@pytest.mark.parametrize('rule', ['narrowest', 'first'])
def test_the_walk_without_the_guard_does_not_get_there(rule):
    """The property has teeth: with the guard off, a heuristic that never picks the widest input leaves it at its root width"""
    root, k = ROOTS[-1], 5
    n = len(root)
    assert _walk(root, rule, k * n * (n + 1), guard=False).min() > 2.0 ** -k
