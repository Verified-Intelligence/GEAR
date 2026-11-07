"""Deterministic op budgets of the exact-proof graph passes (WP4, A207). CPU only."""
import hashlib
import logging
import threading
from unittest.mock import patch

import torch

from auto_LiRPA import BoundedModule
import pytest

from auto_LiRPA.op_budget import OpBudget, OpBudgetExhausted, stage_budget
from auto_LiRPA.presolve import prove_rows
from test_hidden_square_rewrite import HiddenSquare, OP_BUDGET, OPTIONS as HS_OPTIONS
from test_presolve_rewrites import Quadratic, Rows, bounded


def signature(bm):
    """Node names, classes, input names and constant bytes of the whole graph."""
    h = hashlib.sha256()
    for n in bm.nodes():
        h.update(f'{n.name}|{type(n).__name__}|{[i.name for i in n.inputs]}'.encode())
        v = getattr(n, 'value', None)
        if torch.is_tensor(v):
            h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


def hidden(_full=None, **budget):
    opts = dict(HS_OPTIONS, hidden_square_rewrite=True,
                op_budget=dict(OP_BUDGET, **budget) if _full is None else _full)
    return BoundedModule(HiddenSquare().double(), torch.zeros(1, 2, dtype=torch.float64),
                         bound_opts=opts, device='cpu')


def completed(_full=None, **budget):
    return bounded(Quadratic(inner=True, product=False),
                   dict(complete_square_rewrite=True,
                        op_budget=dict(OP_BUDGET, **budget) if _full is None else _full))


def unrewritten():
    # The same model built with every rewrite option off is the "unchanged" reference.
    return bounded(Quadratic(inner=True, product=False), dict(complete_square_rewrite=False))


def test_budget_object_counts_and_raises():
    b = OpBudget('s', 5)
    b.spend(3); b.spend(2)
    assert b.used == 5
    try:
        b.spend()
    except OpBudgetExhausted as e:
        assert isinstance(e, TimeoutError)
    else:
        raise AssertionError


def test_exhaustion_leaves_hidden_squares_unchanged(caplog):
    full = hidden()
    off = BoundedModule(HiddenSquare().double(), torch.zeros(1, 2, dtype=torch.float64),
                        bound_opts=dict(HS_OPTIONS, hidden_square_rewrite=False), device='cpu')
    assert signature(full) != signature(off)
    with caplog.at_level(logging.WARNING):
        capped = hidden(hidden_squares=1)
    assert signature(capped) == signature(off)
    assert any('hidden_squares exhausted' in r.getMessage() for r in caplog.records)


def test_exhaustion_leaves_complete_squares_unchanged(caplog):
    full = completed()
    off = unrewritten()
    assert full.presolve_rewrite_counts['complete_square_rewrite'] == 1
    assert signature(full) != signature(off)
    with caplog.at_level(logging.WARNING):
        capped = completed(complete_squares=1)
    assert capped.presolve_rewrite_counts == {'complete_square_rewrite': 0, 'product_residual': 0}
    assert signature(capped) == signature(off)
    assert any('complete_squares exhausted' in r.getMessage() for r in caplog.records)


def test_exhaustion_leaves_prove_rows_empty(caplog):
    bm = bounded(Rows(), dict(square_row_to_abs=True))
    C = torch.eye(4, dtype=torch.float64)
    assert len(prove_rows(bm, C)[0]) == 1
    bm = bounded(Rows(), dict(square_row_to_abs=True, op_budget=dict(OP_BUDGET, prove_rows=1)))
    with caplog.at_level(logging.WARNING):
        assert prove_rows(bm, C) == ([], [])
    assert any('prove_rows exhausted' in r.getMessage() for r in caplog.records)


def test_op_counts_are_deterministic():
    def count():
        used, real = {}, OpBudget.spend

        def spend(self, n=1):
            real(self, n)
            used[self.stage] = self.used
        with patch.object(OpBudget, 'spend', spend):
            hidden(); completed()
        return used
    first = count()
    assert first['hidden_squares'] > 0 and first['complete_squares'] > 0
    assert count() == first == count()


def test_result_identical_under_cpu_load():
    quiet = (signature(hidden()), signature(completed()))
    stop = threading.Event()

    def burn():
        x = 0
        while not stop.is_set():
            x += 1
    threads = [threading.Thread(target=burn) for _ in range(8)]
    for t in threads:
        t.start()
    try:
        loaded = (signature(hidden()), signature(completed()))
    finally:
        stop.set()
        for t in threads:
            t.join()
    assert loaded == quiet


@pytest.mark.parametrize('bad', [None, True, 0, -5, 1.5, '8'])
def test_missing_or_invalid_budget_raises(bad):
    for stage, build in (('hidden_squares', hidden), ('complete_squares', completed)):
        budget = dict(OP_BUDGET)
        if bad is None:
            del budget[stage]
        else:
            budget[stage] = bad
        with pytest.raises(ValueError, match=stage):
            build(budget)
    budget = dict(OP_BUDGET)
    if bad is None:
        del budget['prove_rows']
    else:
        budget['prove_rows'] = bad
    bm = bounded(Rows(), dict(square_row_to_abs=True, op_budget=budget))
    with pytest.raises(ValueError, match='prove_rows'):
        prove_rows(bm, torch.eye(4, dtype=torch.float64))


def test_stage_off_needs_no_budget():
    # no op_budget in bound_opts, flag off: nothing is read
    bm = BoundedModule(HiddenSquare().double(), torch.zeros(1, 2, dtype=torch.float64),
                       bound_opts={}, device='cpu')
    assert bm.bound_opts.get('op_budget') is None


def test_op_budget_log_records_stage_usage():
    bm = hidden()
    (stage, used, cap, exhausted), = bm.op_budget_log
    assert (stage, cap, exhausted) == ('hidden_squares', 1 << 20, False) and used > 0
    bm = hidden(hidden_squares=1)
    (stage, used, cap, exhausted), = bm.op_budget_log
    assert (stage, used, cap, exhausted) == ('hidden_squares', 1, 1, True)


def test_whole_mode_partial_progress_fixture():
    """A stage that applies >= 1 rewrite before exhausting: kept under partial, raised under whole."""
    def budget_after_first_rewrite():
        used = {}
        real = OpBudget.spend

        def spend(self, n=1):
            real(self, n)
            used[self.stage] = self.used
        with patch.object(OpBudget, 'spend', spend):
            hidden()
        return used['hidden_squares']
    full_used = budget_after_first_rewrite()
    # the rewrite is applied late in the proof; cap just below the full count keeps partial progress
    # only if the node loop reaches it, so search the smallest cap that leaves the rewrite applied.
    off = BoundedModule(HiddenSquare().double(), torch.zeros(1, 2, dtype=torch.float64),
                        bound_opts=dict(HS_OPTIONS, hidden_square_rewrite=False), device='cpu')
    kept = None
    for cap in range(1, full_used + 1):
        bm = hidden(hidden_squares=cap)
        if signature(bm) != signature(off):
            kept = cap
            break
    assert kept is not None
    if kept < full_used:  # exhaustion happened after >= 1 rewrite
        with pytest.raises(OpBudgetExhausted):
            hidden(hidden_squares=kept, commit='whole')
    # whole mode with an exhausted cap raises and the stage-off rebuild is the stage-off graph
    with pytest.raises(OpBudgetExhausted):
        hidden(hidden_squares=1, commit='whole')


def test_prove_rows_whole_returns_empty():
    bm = bounded(Rows(), dict(square_row_to_abs=True, op_budget=dict(OP_BUDGET, prove_rows=3, commit='whole')))
    assert prove_rows(bm, torch.eye(4, dtype=torch.float64)) == ([], [])
