"""Deterministic operation budgets for the exact-proof graph passes.

A stage spends one unit per elementary step (a polynomial built, a term pair
multiplied, a node visited or a row proved). Exhaustion raises
`OpBudgetExhausted`, a `TimeoutError`, so each stage stops exactly where the
former wall-clock deadline stopped it: the unfinished part of the graph is left
unchanged. The result depends only on the model, never on machine speed or load.

The caps are not defined here. They come from `model.bound_opts['op_budget']`
(`{'hidden_squares': int, 'complete_squares': int, 'prove_rows': int,
'commit': 'partial' | 'whole'}`), which the caller fills from the config schema
(`model.op_budget.*`, complete_verifier/arguments.py).
"""
from .utils import logger

COMMIT_MODES = ('partial', 'whole')


class OpBudgetExhausted(TimeoutError):
    """The stage used up its operation budget."""


class OpBudget:
    def __init__(self, stage, cap, log=None):
        self.stage, self.cap, self.used, self.logged, self.log = stage, cap, 0, False, log

    def spend(self, n=1):
        self.used += n
        if self.used > self.cap:
            raise OpBudgetExhausted(self.stage)

    def log_cap(self):
        if self.logged:
            return
        self.logged = True
        logger.warning('[OpBudget] %s exhausted its budget of %d operations; '
                       'the unfinished part of the graph is left unchanged',
                       self.stage, self.cap)

    def finish(self):
        """Record (stage, used, cap, exhausted) in the model's op_budget_log."""
        if self.log is not None:
            self.log.append((self.stage, min(self.used, self.cap), self.cap, self.used > self.cap))


def stage_budget(model, stage):
    """The budget of `stage` from model.bound_opts['op_budget']; ValueError if absent or invalid."""
    opts = model.bound_opts.get('op_budget')
    cap = opts.get(stage) if isinstance(opts, dict) else None
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise ValueError(f"bound_opts['op_budget']['{stage}'] must be an int > 0, got {cap!r}")
    if opts.get('commit', 'partial') not in COMMIT_MODES:
        raise ValueError(f"bound_opts['op_budget']['commit'] must be one of {COMMIT_MODES}")
    if not hasattr(model, 'op_budget_log'):
        model.op_budget_log = []
    return OpBudget(stage, cap, model.op_budget_log)


def commit_whole(model):
    return model.bound_opts['op_budget'].get('commit', 'partial') == 'whole'
