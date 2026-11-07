"""IPOPT-in-BaB stops after consecutive calls that do not improve the incumbent."""
import os
import sys
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from input_split import nlp_in_bab  # noqa: E402


def test_calls_stop_after_consecutive_failures_and_reset_on_success():
    # pooling_adhya4tp: the first call found the optimum, the next ten did not and cost
    # most of the BaB budget.
    state = nlp_in_bab.init_alm_bab_state({})
    cfg = {"enable_ipopt_in_bab": True, "ipopt_bab_max_fails": 2}
    outcomes = iter([(torch.zeros(2), -5.0), (None, None), (None, None), (None, None)])
    calls = []

    def fake(net, batch, nlp_config, primal):
        calls.append(1)
        return next(outcomes)

    with mock.patch.object(nlp_in_bab, "run_ipopt_in_bab", side_effect=fake):
        results = [nlp_in_bab._scheduled_ipopt_in_bab(None, None, cfg, None, state)
                   for _ in range(5)]
        assert len(calls) == 3                 # success, fail, fail, then skipped
        assert results[0][1] == -5.0 and results[3] == (None, None)
        state.ipopt_bab_fails = 0              # re-armed: the next call runs again
        assert nlp_in_bab._scheduled_ipopt_in_bab(None, None, cfg, None, state) == (None, None)
        assert len(calls) == 4


def test_disabled_never_calls():
    state = nlp_in_bab.init_alm_bab_state({})
    with mock.patch.object(nlp_in_bab, "run_ipopt_in_bab") as run:
        assert nlp_in_bab._scheduled_ipopt_in_bab(
            None, None, {"enable_ipopt_in_bab": False}, None, state) == (None, None)
    run.assert_not_called()


def _picked(rhs0):
    rhs = torch.tensor([[rhs0, 0.0], [rhs0, 0.0]], dtype=torch.float64)
    return nlp_in_bab.PickedDomainBatch(
        dm_l=torch.zeros(2, 1, 3), dm_u=torch.ones(2, 1, 3), c=torch.eye(2).expand(2, 2, 2),
        rhs=rhs, spec_size=torch.tensor([2, 2]))


class _Net:
    model_ori = None


_CFG = {"enable_ipopt_in_bab": True, "ipopt_bab_topk": 1}


def test_improving_point_is_shifted_back_to_raw_units():
    # ipopt_falsification reports (objective row - rhs) and returns a point only when
    # that is negative. With rhs[:, 0] = incumbent 89.835, a point 2.6e-5 better comes
    # back as -2.6e-5; it must become 89.835 - 2.6e-5, not -2.6e-5 (wastewater04m2,
    # hs065, cute_hs074, elena_gold ... ended "verified" on the shifted value).
    x = torch.full((1, 3), 0.5)
    with mock.patch("nlp_solver.ipopt_solver.ipopt_falsification",
                    return_value=(True, x, -2.6e-5)):
        bx, obj = nlp_in_bab.run_ipopt_in_bab(_Net(), _picked(89.835), _CFG,
                                              torch.tensor([89.835], dtype=torch.float64))
    assert bx is x and abs(obj - (89.835 - 2.6e-5)) < 1e-9


def test_objective_without_a_point_is_not_an_incumbent():
    # A feasible but non-improving solve returns (False, None, shifted objective).
    with mock.patch("nlp_solver.ipopt_solver.ipopt_falsification",
                    return_value=(False, None, 2.0)):
        assert nlp_in_bab.run_ipopt_in_bab(_Net(), _picked(10.0), _CFG,
                                           torch.tensor([10.0], dtype=torch.float64)) == (None, None)
    with mock.patch("nlp_solver.ipopt_solver.ipopt_falsification",
                    return_value=(False, None, -3.0)):
        assert nlp_in_bab.run_ipopt_in_bab(_Net(), _picked(10.0), _CFG,
                                           torch.tensor([10.0], dtype=torch.float64)) == (None, None)


def test_no_incumbent_level_skips_the_domain():
    # rhs[:, 0] = +inf before any incumbent: objective - inf is no problem Ipopt can solve.
    with mock.patch("nlp_solver.ipopt_solver.ipopt_falsification") as fals:
        assert nlp_in_bab.run_ipopt_in_bab(_Net(), _picked(float('inf')), _CFG,
                                           torch.tensor([float('inf')])) == (None, None)
    fals.assert_not_called()
