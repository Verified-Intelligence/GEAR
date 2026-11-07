"""A broad nearly exact variable must not starve a narrower violated product."""
import numpy as np
import pytest
import torch
from nlp_solver.lifted_lp import LiftedLP


@pytest.mark.parametrize('width, score, expected', [
    (.01, [1., 1e-6], 0),  # A40: the old mask discarded the only useful split.
    (.01, [1., .5], 1),    # Still penalize slivers when a broad split has useful error.
    (.1, [1., .5], 0),     # Comparable widths retain the original residual ranking.
    (1e-12, [1000., .5], 1),  # Retain the numerical-width safeguard.
])
def test_search_balances_width_without_starving_violation(width, score, expected):
    lp = object.__new__(LiftedLP)
    lp.n, lp.obj = 2, object()
    lp._variable_violation = lambda z: np.array(score)
    lo, hi = np.array([.4, 0.]), np.array([.4 + width, 1.])
    point = (lo + hi) / 2
    children = []

    def solve(l, u, *args, **kwargs):
        children.append((l.copy(), u.copy()))
        return 0., (l + u) / 2, l, u, None

    lp.solve_bab_node = solve
    resume = (lo[None], hi[None], np.array([0.]), point[None], [(point, None)])
    result = lp.branch_and_bound(torch.zeros(1, 2), torch.ones(1, 2),
                                torch.empty(1, 0), cutoff=10., resume=resume,
                                max_nodes=1, time_budget=float('inf'),
                                plateau_time=float('inf'))
    assert result[1] == 1 and len(children) == 2
    changed = np.flatnonzero(children[0][1] != hi)
    assert changed.tolist() == [expected]
    assert children[0][1][expected] == children[1][0][expected]
    assert np.array_equal(children[0][0], lo)
    assert np.array_equal(children[1][1], hi)
