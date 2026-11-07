"""Controller-only work-sized tree atoms and cooperative deadline boundaries."""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import time

_ACTIVE = ContextVar('controller_tree_slice', default=None)
_OWNER = ContextVar('controller_owner', default=None)


def checkpoint():
    root = _ACTIVE.get() or _OWNER.get()
    if root is not None:
        from input_split.root_controller import RootClosed, RootDeadline, poll
        terminal = getattr(root, 'controller_terminal', None)
        if terminal is not None:
            raise terminal
        if root.time_left() <= 0:
            raise RootDeadline()
        if not root.controller_inbox.empty():
            poll(root, service=False)


@contextmanager
def active(root):
    token = _ACTIVE.set(root)
    try:
        checkpoint()
        yield
    finally:
        _ACTIVE.reset(token)


@contextmanager
def region(name, count):
    root = _ACTIVE.get()
    if root is None:
        yield
        return
    checkpoint()
    start = time.monotonic()
    cpu_start = time.process_time()
    try:
        yield
    finally:
        print(f'[TreeAtom] region={name} count={count} s={time.monotonic()-start:.9g} '
              f'cpu_s={time.process_time()-cpu_start:.9g} '
              f'left={root.time_left():.9g}', flush=True)
    checkpoint()


def atom_work_cap():
    root = _ACTIVE.get()
    quantum = getattr(root, 'controller_tree_atom_quantum', None)
    return min(quantum, root.time_left()) if quantum is not None else None


def service_clock(root):
    """Tree service excludes time spent suspended between deep bound yields."""
    started = getattr(root, 'controller_tree_dispatch', None)
    return getattr(root, 'controller_tree_paid', 0.) + (time.monotonic()-started if started is not None else 0.)


class ServiceSizer:
    """Explore larger chunks instead of extrapolating a singleton's cost.

    Doubling after an in-cap measurement exposes fixed call overhead. An
    over-cap measurement shrinks proportionally; it never licenses growth.
    The cap is a service target, not a hard bound on an unmeasured native call.
    """
    def __init__(self, initial=1):
        self.initial = max(1, initial)
        self.count = 0
        self.seconds = 0.0

    def size(self, limit, cap):
        if cap is not None and cap <= 0:
            return 1
        if not self.count:
            return min(limit, self.initial)
        if cap is None:
            cap = self.seconds
        count = (2 * self.count if self.seconds <= cap else
                 max(1, int(self.count * max(0., cap) / self.seconds)))
        return min(limit, count)

    def account(self, count, seconds):
        if count > 0:
            self.count, self.seconds = count, max(0., seconds)


class BatchCap:
    """Measured geometric growth within the existing child-memory cap."""
    def __init__(self, initial, max_children):
        self.sizer = ServiceSizer(initial)
        self.limit = max(1, max_children // 2)
        self.first = None

    def batch(self, requested, quantum, remaining):
        budget = self.first if quantum is None else quantum
        work = remaining if budget is None else min(budget, remaining)
        return self.sizer.size(min(requested, self.limit), work)

    def account(self, parents, seconds):
        if parents > 0:
            self.sizer.account(parents, seconds)
            if self.first is None:
                self.first = seconds


class OwnerInterrupts:
    """Wake the owner for native publication and the one absolute deadline.

    Handlers may run at any Python instruction, including cleanup. They publish
    or latch termination but never unwind the owner. Safe checkpoints perform
    the unwind, so a wakeup cannot interrupt finish, joins or interpreter exit.
    """
    def __init__(self, root):
        import threading
        self.root = root
        self.enabled = threading.current_thread() is threading.main_thread()
        self.finishing = False
        self.owner_token = _OWNER.set(root)
        if not self.enabled:
            return
        import os
        import signal
        from input_split.root_controller import RootClosed, RootDeadline, poll
        def notify(signum, frame):
            if (self.finishing or getattr(root, 'controller_terminal', None) is not None
                    or getattr(root, 'controller_committing', False)
                    or getattr(root.ledger, 'busy', False)):
                return
            if (_ACTIVE.get() is root and
                    getattr(getattr(root, 'domains', None), '_last_pickout', None) is None):
                return
            if root.time_left() <= 0:
                root.controller_terminal = RootDeadline()
                return
            import torch
            if torch._C._are_functorch_transforms_active():
                return
            try:
                poll(root, service=False)
                if root.closes(root.values()[1]):
                    root.controller_terminal = RootClosed('native_publication')
            except (RootClosed, RootDeadline) as terminal:
                root.controller_terminal = terminal
        self.old_notify = signal.signal(signal.SIGUSR1, notify)
        self.old_alarm = signal.signal(signal.SIGALRM, notify)
        self.old_timer = signal.setitimer(signal.ITIMER_REAL,
                                         max(0.0, root.time_left()) if math.isfinite(root.time_left()) else 0.)
        # A publisher may retain this callable across finish(). Serialize its
        # last enabled check and kill with disarming, then join all producers
        # before restoring the caller's handlers.
        self.notify_lock = threading.RLock()
        def wake():
            with self.notify_lock:
                if not self.finishing:
                    os.kill(os.getpid(), signal.SIGUSR1)
        root.controller_notify = wake

    def finish(self):
        if self.enabled:
            import signal
            # Ignore process-wide before changing any Python state. Even a
            # previously queued Python handler is non-throwing.
            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
            signal.signal(signal.SIGALRM, signal.SIG_IGN)
            signal.setitimer(signal.ITIMER_REAL, 0.)
            with self.notify_lock:
                self.finishing = True
                self.root.controller_notify = None
        else:
            self.finishing = True
            self.root.controller_notify = None

    def close(self):
        self.finish()
        _OWNER.reset(self.owner_token)
        if self.enabled:
            import signal
            signal.setitimer(signal.ITIMER_REAL, 0.)
            signal.signal(signal.SIGUSR1, self.old_notify)
            signal.signal(signal.SIGALRM, self.old_alarm)
            if self.old_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, *self.old_timer)
