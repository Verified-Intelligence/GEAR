"""Width reduction without certified objective progress must not earn probe credit."""
import numpy as np
from nlp_solver import lifted_lp as ll
from tests.test_lifted_lp_rlt import make_lp


def test_width_only_probes_stop_after_the_first_measurement(monkeypatch):
    lp, rh = make_lp([1, 1, 1], 1, [(0, 1), (1, 2)], [1, 1])
    point = np.r_[np.full(3, .5), np.zeros(2)]
    monkeypatch.setattr(lp, 'solve_node', lambda *a, **k: (1., point.copy()))
    calls = []
    def probe(c, *args, **kwargs):
        calls.append(c.copy())
        return .2 if c.max() > 0 else -.8
    monkeypatch.setattr(lp, 'solve_certified', probe)
    progress = [0.] * 4
    result = lp.solve_bab_node(np.zeros(3), np.ones(3), rh[0].numpy(), None, None,
                               10., progress=progress, inherited=0.)
    assert len(calls) == 2
    assert np.any(result[2] > 0) or np.any(result[3] < 1)
    assert result[0] == 1. and progress[0] == 0 and progress[1] > 0
    assert not ll._probe_pays(progress)


def test_running_comparison_uses_paid_seconds_and_allows_equal_rates():
    assert ll._probe_pays([0., 0., 1., 1.])  # no measured probe yet
    assert ll._probe_pays([2., 1., 4., 2.])
    assert not ll._probe_pays([2., 2., 4., 2.])
