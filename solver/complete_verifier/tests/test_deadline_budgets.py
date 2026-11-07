"""In-BaB ALM/IPOPT attack budgets are clipped to the time left before the global limit."""
import os
import sys
import time
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from input_split import nlp_in_bab  # noqa: E402


def test_budget_is_cut_to_the_time_left():
    cfg = {"bab_timeout": 10.0, "other": 1}
    out = nlp_in_bab._clip_to_deadline(cfg, "bab_timeout", 5.0, time.time() + 4.0)
    assert 3.0 < out["bab_timeout"] <= 4.0 and out["other"] == 1
    assert cfg["bab_timeout"] == 10.0                       # the caller's config is untouched


def test_budget_is_kept_when_time_is_plentiful():
    cfg = {"bab_timeout": 10.0}
    assert nlp_in_bab._clip_to_deadline(cfg, "bab_timeout", 5.0, time.time() + 100)["bab_timeout"] == 10.0
    assert nlp_in_bab._clip_to_deadline({}, "bab_timeout", 5.0, time.time() + 100)["bab_timeout"] == 5.0


def test_too_little_time_skips_the_attack():
    assert nlp_in_bab._clip_to_deadline({}, "bab_timeout", 5.0, time.time() + 0.5) is None
    assert nlp_in_bab._clip_to_deadline({}, "bab_timeout", 5.0, time.time() - 3.0) is None


def test_ipopt_in_bab_gets_the_clipped_timeout_and_is_skipped_at_the_limit():
    cfg = {"enable_ipopt_in_bab": True, "ipopt_bab_timeout": 18.0}
    state = nlp_in_bab.init_alm_bab_state({})
    state.deadline = time.time() + 5.0
    seen = []
    with mock.patch.object(nlp_in_bab, "run_ipopt_in_bab",
                           side_effect=lambda n, b, c, p: seen.append(c["ipopt_bab_timeout"])
                           or (torch.zeros(1), -1.0)):
        nlp_in_bab._scheduled_ipopt_in_bab(None, None, cfg, None, state)
        assert seen and seen[0] <= 5.0
        state.deadline = time.time() - 1.0
        assert nlp_in_bab._scheduled_ipopt_in_bab(None, None, cfg, None, state) == (None, None)
        assert len(seen) == 1


def test_default_deadline_is_unbounded():
    assert nlp_in_bab.init_alm_bab_state({}).deadline == float("inf")
