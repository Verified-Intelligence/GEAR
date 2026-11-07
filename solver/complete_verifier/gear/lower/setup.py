"""Commit mode "whole" of the legacy in-graph stages in the gear setup (G5 DESIGN section 4.1, change 8).

Under ``op_budget['commit'] == 'whole'`` a rewrite stage that exhausts its budget raises ``OpBudgetExhausted`` out of
the ``BoundedModule`` build (``auto_LiRPA/optimize_graph.py``, ``presolve.py``). ``build_bounded_module`` catches it
and rebuilds from the same program with that stage's flag False, so the stage commits all of its rewrites or none.
Exhaustion is deterministic (operation counts), so the rebuild is too. ``prove_rows`` needs no rebuild: under
"whole" it returns ``([], [])`` itself.
"""
from __future__ import annotations

import copy
from collections.abc import Callable

from auto_LiRPA.op_budget import OpBudgetExhausted

STAGE_FLAGS = {"hidden_squares": "hidden_square_rewrite", "complete_squares": "complete_square_rewrite"}


def build_bounded_module(build: Callable[[dict], object], bound_opts: dict):
    """``build(opts)`` -> BoundedModule, retried with an exhausted stage switched off (commit "whole" only).

    The returned module carries ``gear_anomalies``: one ``op_budget_exhausted:<stage>:<cap>`` per rebuilt stage.
    Under "partial" this is exactly ``build(bound_opts)``.
    """
    opts = copy.deepcopy(bound_opts)
    whole = (opts.get("op_budget") or {}).get("commit", "partial") == "whole"
    anomalies = []
    while True:
        try:
            net = build(opts)
        except OpBudgetExhausted as exc:
            stage = exc.args[0] if exc.args else None
            flag = STAGE_FLAGS.get(stage)
            if not whole or flag is None or not opts.get(flag):
                raise
            opts[flag] = False
            anomalies.append(f"op_budget_exhausted:{stage}:{opts['op_budget'][stage]}")
            continue
        net.gear_anomalies = tuple(anomalies)
        return net
