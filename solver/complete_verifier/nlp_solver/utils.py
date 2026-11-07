"""Utility functions for NLP solver module."""

from typing import Tuple, Callable, Generator
import threading
from contextlib import contextmanager
import torch
import time
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler


class DeadlineExceeded(TimeoutError):
    """Cancel an unfinished proposal without accepting its partial derivatives."""


def check_deadline(deadline):
    if deadline is not None and time.time() >= deadline:
        raise DeadlineExceeded

def _infer_device_dtype(
    model: Callable,
    fallback_device: torch.device,
    fallback_dtype: torch.dtype,
) -> Tuple[torch.device, torch.dtype]:
    """Follow model parameter dtype/device when possible; otherwise follow fallback."""
    if isinstance(model, nn.Module):
        first_tensor = None
        for tensor in model.parameters():
            if first_tensor is None:
                first_tensor = tensor
            if torch.is_floating_point(tensor) or torch.is_complex(tensor):
                return tensor.device, tensor.dtype
        for tensor in model.buffers():
            if first_tensor is None:
                first_tensor = tensor
            if torch.is_floating_point(tensor) or torch.is_complex(tensor):
                return tensor.device, tensor.dtype
        if first_tensor is not None:
            return first_tensor.device, first_tensor.dtype
    return fallback_device, fallback_dtype


@contextmanager
def _eval_mode(model: Callable) -> Generator[None, None, None]:
    """Force eval mode during optimization; restore original mode afterwards."""
    if not isinstance(model, nn.Module):
        yield
        return
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def _split_or_clauses(
    C_mat: torch.Tensor,
    rhs_mat: torch.Tensor,
    or_spec_size: torch.Tensor,
) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    """
    Yield (C_obj, rhs_obj, C_constr, rhs_constr) for each OR clause.

    Handles multiple formats:
    1. No or_spec_size: first row is objective, rest are constraints
    2. sum(or_spec_size) == num_spec: each clause has its own objective + constraints
    3. sum(or_spec_size) + 1 == num_spec: first row is shared objective, then constraints
    4. Uniform clause size: num_or * or_size == num_spec
    """
    num_spec = C_mat.shape[0]
    if or_spec_size is None or or_spec_size.numel() == 0:
        # Single clause: first row is objective, rest are constraints
        yield (C_mat[0:1], rhs_mat[0:1], C_mat[1:], rhs_mat[1:])
        return

    sizes = [int(v) for v in or_spec_size.flatten().tolist()]
    total_sizes = sum(sizes)

    # Case 1: sum(sizes) == num_spec - each clause is self-contained with obj+constr
    if total_sizes == num_spec:
        s = 0
        for sz in sizes:
            C = C_mat[s:s+sz]
            r = rhs_mat[s:s+sz]
            yield (C[0:1], r[0:1], C[1:], r[1:])
            s += sz
        return

    # Case 2: sum(sizes) + 1 == num_spec - first row is shared objective
    # This is the common vnnlib format where:
    #   - Row 0 is the objective
    #   - Rows 1 to total_sizes are constraints partitioned by or_spec_size
    if total_sizes + 1 == num_spec:
        C_obj = C_mat[0:1]
        rhs_obj = rhs_mat[0:1]
        s = 1  # Start after objective row
        for sz in sizes:
            C_constr = C_mat[s:s+sz]
            rhs_constr = rhs_mat[s:s+sz]
            yield (C_obj, rhs_obj, C_constr, rhs_constr)
            s += sz
        return

    # Case 3: Fallback - uniform clause size where num_or * or_size == num_spec
    or_size = sizes[0]
    num_or = len(sizes)
    if num_spec == num_or * or_size:
        for j in range(num_or):
            s = j * or_size
            C = C_mat[s:s+or_size]
            r = rhs_mat[s:s+or_size]
            yield (C[0:1], r[0:1], C[1:], r[1:])
        return

    # Last resort: treat as single clause with first row as objective
    # This handles edge cases gracefully
    yield (C_mat[0:1], rhs_mat[0:1], C_mat[1:], rhs_mat[1:])


class VectorLinearLR(_LRScheduler):
    """
    Linear LR scheduler for vector (per-param-group) learning rates.
    Used when each param_group has its own LR (e.g. per-domain LRs in BAB phase).
    Mirrors torch.optim.lr_scheduler.LinearLR semantics but avoids fill_() so
    vector/tensor LRs are updated with copy_().
    """
    def __init__(self, optimizer, start_factor=1.0, end_factor=0.1, total_iters=1, last_epoch=-1):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        # Clone tensor LRs so base_lrs stays independent of in-place copy_()
        # updates, preventing double-decay when get_lr() reads base_lrs.
        for group in optimizer.param_groups:
            if isinstance(group['lr'], torch.Tensor):
                group['initial_lr'] = group['lr'].clone()
        super(VectorLinearLR, self).__init__(optimizer, last_epoch)

    def _linear_factor(self):
        if self.last_epoch == 0:
            return self.start_factor
        progress = min(1.0, self.last_epoch / max(self.total_iters, 1))
        return self.start_factor + (self.end_factor - self.start_factor) * progress

    def get_lr(self):
        factor = self._linear_factor()
        return [base_lr * factor for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        factor = self._linear_factor()
        return [base_lr * factor for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """
        Override step() so we use copy_() for tensor LRs instead of fill_().
        """
        self._step_count += 1

        if epoch is None:
            self.last_epoch += 1
            values = self.get_lr()
        else:
            self.last_epoch = epoch
            if hasattr(self, "_get_closed_form_lr"):
                values = self._get_closed_form_lr()
            else:
                values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            if isinstance(lr, torch.Tensor):
                param_group["lr"].copy_(lr)
            else:
                param_group["lr"] = lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


# Forward-mode AD levels are process-global and must exit in LIFO order, so the
# concurrent IPOPT and ALM threads may not interleave torch.func.jvp calls.
_FORWARD_AD_LOCK = threading.RLock()


def locked_jvp(func, primals, tangents):
    with _FORWARD_AD_LOCK:
        return torch.func.jvp(func, primals, tangents)


def is_cuda_oom(error: BaseException) -> bool:
    """A CUDA out-of-memory error, also when TorchScript re-raised it as a plain
    RuntimeError ("The following operation failed in the TorchScript interpreter")."""
    from auto_LiRPA.cuda_memory import memory_error
    return memory_error(error)
