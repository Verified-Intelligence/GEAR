"""One cooperative wall clock for the entire dynamics pipeline."""
import time
_now = time.monotonic


class BudgetExpired(BaseException):
    """Bypasses solver exception-to-error mapping; the cascade records expiry."""


class Budget:
    def __init__(self, seconds=None):
        self.start = _now()
        self.limit = seconds
        self.end = None if seconds is None else self.start + seconds
        self.phase = 'initialization'

    def remaining(self):
        return float('inf') if self.end is None else max(0.0, self.end - _now())

    def elapsed(self):
        return _now() - self.start

    def check(self, phase=None):
        if phase is not None:
            self.phase = phase
        if self.remaining() <= 0:
            raise BudgetExpired(self.phase)

    def solve_options(self, options):
        from ..config import resolve
        from ..solver import _time_limit
        self.check()
        if self.end is None:
            return options
        limit = min(self.remaining(), _time_limit(resolve(options).config))
        return options.replace(time_limit=limit, timing='solve')
