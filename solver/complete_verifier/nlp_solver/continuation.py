"""Count-defined solver atoms and retained, owner-thread continuations."""
from contextvars import copy_context
from dataclasses import dataclass
from functools import wraps

import torch


@dataclass
class Atom:
    family: str
    point: object = None
    bound: object = None
    box: object = None
    wait_for: object = None
    settled: bool = False
    context: object = None
    progress: object = None


def candidate(x, phi, viol, tol):
    """Best raw-feasible candidate; the controller still checks the original model."""
    with torch.no_grad():
        phi, viol = phi.reshape(-1), viol.reshape(-1)
        valid = torch.isfinite(phi) & torch.isfinite(viol) & (viol <= tol)
        if not bool(valid.any()):
            return None
        idx = phi.masked_fill(~valid, float('inf')).argmin()
        return x.reshape(phi.numel(), -1)[idx].detach().clone()


def drain(iterator):
    while True:
        try:
            next(iterator)
        except StopIteration as done:
            return done.value


def retained(fn):
    """Keep the legacy result API; opt in to the underlying iterator explicitly."""
    @wraps(fn)
    def run(*args, _iterate=False, **kwargs):
        iterator = fn(*args, **kwargs)
        if _iterate:
            return iterator
        try:
            return drain(iterator)
        except RuntimeError as error:
            # PEP 479 converts a legacy callee's explicit StopIteration at a
            # generator boundary. Preserve that exception for the synchronous
            # result API; iterator callers retain normal generator semantics.
            if (str(error) == 'generator raised StopIteration'
                    and isinstance(error.__cause__, StopIteration)):
                raise error.__cause__ from None
            raise
    return run


class Continuation:
    """Retain the Python stack, optimizer, autograd context and private RNG.

    A suspended generator can be inside no_grad/enable_grad or eval_mode. Restore
    the caller's ambient state between atoms so another job cannot corrupt it.
    Closing uses the same context, and runs every generator's finally blocks.
    """
    def __init__(self, iterator, model=None):
        self.iterator, self.model = iterator, model
        self.context = copy_context()
        self.grad = torch.is_grad_enabled()
        self.rng = torch.get_rng_state()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        self.training = None if model is None else [m.training for m in model.modules()]
        self.done, self.result = False, None

    def _run(self, close=False):
        return self.context.run(self._advance, close)

    def _advance(self, close=False):
        grad, rng = torch.is_grad_enabled(), torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if self.cuda_rng is not None else None
        training = None if self.model is None else [m.training for m in self.model.modules()]
        torch.set_grad_enabled(self.grad)
        torch.set_rng_state(self.rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)
        if self.model is not None:
            for module, value in zip(self.model.modules(), self.training):
                module.training = value
        try:
            if close:
                self.iterator.close()
                self.done = True
            else:
                return next(self.iterator)
        except StopIteration as done:
            self.done, self.result = True, done.value
            return None
        finally:
            self.grad, self.rng = torch.is_grad_enabled(), torch.get_rng_state()
            if self.cuda_rng is not None:
                self.cuda_rng = torch.cuda.get_rng_state_all()
                torch.cuda.set_rng_state_all(cuda_rng)
            if self.model is not None:
                self.training = [m.training for m in self.model.modules()]
                for module, value in zip(self.model.modules(), training):
                    module.training = value
            torch.set_grad_enabled(grad)
            torch.set_rng_state(rng)

    def step(self):
        return None if self.done else self._run()

    def close(self):
        if not self.done:
            self._run(close=True)


def invoke(fn, *args, sliced=False, **kwargs):
    """Preserve ordinary calls (including overrides) outside the controller."""
    if sliced:
        return (yield from fn(*args, _iterate=True, **kwargs))
    return fn(*args, **kwargs)
