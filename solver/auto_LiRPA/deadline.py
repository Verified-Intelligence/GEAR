"""Cooperative cancellation of optional bounded work; no partial proof escapes."""
from contextlib import contextmanager
from contextvars import ContextVar
import time

_deadline = ContextVar('crown_deadline', default=None)


class BoundDeadline(BaseException):
    """Unwind an unfinished propagation without accepting its intermediate tensors."""


def check_deadline():
    deadline = _deadline.get()
    if deadline is not None and time.monotonic() >= deadline:
        raise BoundDeadline()


@contextmanager
def bound_deadline(wall_deadline, model=None):
    token = _deadline.set(time.monotonic() + max(0., wall_deadline - time.time()))
    handles = []
    try:
        check_deadline()
        if model is not None:
            handles.append(model.register_forward_pre_hook(lambda *args: check_deadline()))
            handles.append(model.register_forward_hook(lambda *args: check_deadline()))
        yield
    finally:
        for handle in handles:
            handle.remove()
        _deadline.reset(token)
