"""Real signals at teardown instruction boundaries must not unwind cleanup."""
import os
import signal
import sys
import threading
from types import SimpleNamespace

import pytest

from input_split.tree_slice import OwnerInterrupts, checkpoint
from input_split.root_controller import RootClosed, RootDeadline


@pytest.mark.parametrize('terminal', ['closed', 'deadline'])
@pytest.mark.parametrize('where', ['entry', 'disarm', 'timer', 'join', 'restore'])
def test_signal_at_shutdown_boundaries(monkeypatch, terminal, where):
    import input_split.root_controller as ctl
    left = [100.]
    root = SimpleNamespace(time_left=lambda: left[0], ledger=SimpleNamespace(),
                           closes=lambda D: False, values=lambda: (1., 0.))
    def poll(*args, **kwargs):
        raise RootClosed('ipopt_publish')
    monkeypatch.setattr(ctl, 'poll', poll)
    interrupts = OwnerInterrupts(root)
    captured = root.controller_notify
    old_signal, old_timer = signal.signal, signal.setitimer
    events = []
    def inject():
        events.append(where)
        os.kill(os.getpid(), signal.SIGUSR1)
        os.kill(os.getpid(), signal.SIGALRM)
    if terminal == 'deadline':
        left[0] = -1.
    def trace(frame, event, arg):
        if event == 'line' and frame.f_code is OwnerInterrupts.finish.__code__:
            sys.settrace(None)
            inject()
        return trace
    def change_signal(sig, handler):
        if not events and (where == 'disarm' or (where == 'restore' and handler != signal.SIG_IGN)):
            inject()
        return old_signal(sig, handler)
    def change_timer(*args):
        if where == 'timer' and not events:
            inject()
        return old_timer(*args)
    monkeypatch.setattr(signal, 'signal', change_signal)
    monkeypatch.setattr(signal, 'setitimer', change_timer)
    try:
        if where == 'entry':
            sys.settrace(trace)
        interrupts.finish()
        if where == 'join':
            # Both a captured callback and direct late publication signals.
            worker = threading.Thread(target=lambda: (captured(), inject()))
            worker.start()
            worker.join()
        captured()
        interrupts.close()
        captured()  # retained publications must be inert after handler restoration too
        assert events
        assert root.controller_notify is None
    finally:
        sys.settrace(None)
        monkeypatch.setattr(signal, 'signal', old_signal)
        monkeypatch.setattr(signal, 'setitimer', old_timer)
        if not interrupts.finishing:
            interrupts.finish()


def test_native_publication_latches_closure_until_safe_boundary(monkeypatch):
    import input_split.root_controller as ctl
    root = SimpleNamespace(time_left=lambda: 100., ledger=SimpleNamespace())
    monkeypatch.setattr(ctl, 'poll', lambda *a, **k: (_ for _ in ()).throw(RootClosed('ipopt_publish')))
    interrupts = OwnerInterrupts(root)
    try:
        root.controller_notify()
        assert isinstance(root.controller_terminal, RootClosed)
        with pytest.raises(RootClosed):
            checkpoint()
    finally:
        interrupts.finish()
        interrupts.close()


def test_latched_closure_survives_a_later_alarm_and_timeout_fallback(monkeypatch):
    import input_split.root_controller as ctl
    left = [1.]
    root = SimpleNamespace(time_left=lambda: left[0], ledger=SimpleNamespace(),
                           values=lambda: (1., 1.), finish=lambda source, D: (source, D, 'safe'))
    calls = []
    def publish(*args, **kwargs):
        calls.append(1)
        raise RootClosed('ipopt_publish')
    monkeypatch.setattr(ctl, 'poll', publish)
    interrupts = OwnerInterrupts(root)
    try:
        root.controller_notify()
        terminal = root.controller_terminal
        left[0] = -1.
        signal.raise_signal(signal.SIGALRM)
        root.controller_notify()
        assert root.controller_terminal is terminal
        assert calls == [1]
        assert ctl.finish_timeout(root) == ('ipopt_publish', 1., 'safe')
    finally:
        interrupts.finish()
        interrupts.close()


def test_deadline_is_not_reopened_by_late_native_publication(monkeypatch):
    import input_split.root_controller as ctl
    root = SimpleNamespace(time_left=lambda: -1., ledger=SimpleNamespace())
    monkeypatch.setattr(ctl, 'poll', lambda *a, **k: pytest.fail('late publication after termination'))
    interrupts = OwnerInterrupts(root)
    try:
        signal.raise_signal(signal.SIGALRM)
        terminal = root.controller_terminal
        assert isinstance(terminal, RootDeadline)
        root.time_left = lambda: 1.
        root.controller_notify()
        assert root.controller_terminal is terminal
    finally:
        interrupts.finish()
        interrupts.close()
