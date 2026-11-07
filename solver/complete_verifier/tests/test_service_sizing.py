"""Fixed launch cost must not trap controller chunks at a singleton."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from input_split.tree_slice import ServiceSizer, active, atom_work_cap
from input_split.root_controller import RootDeadline
from crown_spec_batching import compute_bounds_in_spec_batches


@pytest.mark.parametrize('fixed,per_item', [(0.1, 0.), (0., 0.0045), (0.1, 0.001)])
def test_growth_measures_larger_calls_and_recovers_from_overrun(fixed, per_item):
    sizer = ServiceSizer()
    counts = []
    for _ in range(20):
        count = sizer.size(408, .2)
        counts.append(count)
        sizer.account(count, fixed + per_item * count)
    assert max(counts) > 2
    if not per_item:
        assert counts[-1] == 408
    sizer.account(100, 1.)
    assert sizer.size(408, .2) == 20
    assert sizer.size(408, 0.) == 1


def test_real_spec_iterator_fixed_cost_calls_preserve_rows(monkeypatch):
    clock = [0.]
    monkeypatch.setattr('crown_spec_batching.time.monotonic', lambda: clock[0])
    calls = []
    class Net:
        input_name = ['input']
        def compute_bounds(self, C, **kwargs):
            clock[0] += .1
            calls.append(C.shape[1])
            return C[:, :, 0].clone(), None
    C = torch.arange(408.).reshape(1, 408, 1)
    result = compute_bounds_in_spec_batches(Net(), C=C, spec_batch_size=408,
                                           _sliced=True, _work_cap=lambda: .2)
    assert calls == [1, 2, 4, 8, 16, 32, 64, 128, 153]
    assert torch.equal(result[0], C[:, :, 0])


def test_deadline_still_checked_between_geometrically_grown_calls():
    import queue
    root = SimpleNamespace(time_left=lambda: .05, controller_tree_atom_quantum=.2,
                           controller_inbox=queue.Queue())
    with active(root):
        assert atom_work_cap() == .05
        root.time_left = lambda: 0.
        with pytest.raises(RootDeadline):
            from input_split.tree_slice import checkpoint
            checkpoint()
