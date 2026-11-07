"""bab_alm_max_time_share delays scheduled in-BaB ALM calls; it never changes a call."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from input_split import nlp_in_bab  # noqa: E402


def _state(window, alm_time):
    state = nlp_in_bab.init_alm_bab_state({})
    state.share_start = time.time() - window
    state.share_alm_time = alm_time
    return state


def test_disabled_by_default():
    assert not nlp_in_bab._alm_over_time_share(_state(10.0, 9.0), {})
    assert not nlp_in_bab._alm_over_time_share(_state(10.0, 9.0), {"bab_alm_max_time_share": 0.0})


def test_skips_only_while_over_the_share():
    cfg = {"bab_alm_max_time_share": 0.25}
    assert nlp_in_bab._alm_over_time_share(_state(100.0, 30.0), cfg)       # 30 s > 25 s
    assert not nlp_in_bab._alm_over_time_share(_state(100.0, 20.0), cfg)   # 20 s < 25 s
    assert not nlp_in_bab._alm_over_time_share(_state(0.0, 0.0), cfg)      # fresh window


def test_init_opens_the_window():
    state = nlp_in_bab.init_alm_bab_state({})
    assert state.share_start is not None and state.share_alm_time == 0.0


def test_stop_after_failures_disabled_by_default():
    state = nlp_in_bab.init_alm_bab_state({})
    state.fail_streak = 100
    assert not nlp_in_bab._alm_given_up(state, {}, 10.0, [0.0])


def test_stop_after_failures_keeps_calling_near_certification():
    import math
    cfg = {"bab_alm_stop_after_failures": 4, "bab_alm_stop_gap": 0.01}
    state = nlp_in_bab.init_alm_bab_state({})
    state.fail_streak = 3
    assert not nlp_in_bab._alm_given_up(state, cfg, 10.0, [0.0])          # streak below K
    state.fail_streak = 4
    assert nlp_in_bab._alm_given_up(state, cfg, 10.0, [0.0, 5.0])         # gap 10/11
    assert not nlp_in_bab._alm_given_up(state, cfg, 10.0, [9.99, 10.5])   # gap 0.1%
    assert nlp_in_bab._alm_given_up(state, cfg, 10.0, [-math.inf])        # unbounded gap
    assert not nlp_in_bab._alm_given_up(state, cfg, math.inf, [0.0])      # no incumbent yet
    assert not nlp_in_bab._alm_given_up(state, cfg, 10.0, None)           # bounds unknown
