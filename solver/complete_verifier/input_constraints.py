"""Utilities for native input linear constraints A x <= b."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch


TensorPair = Tuple[torch.Tensor, torch.Tensor]


def _input_dim(input_shape: Sequence[int]) -> int:
    return int(np.prod(tuple(int(dim) for dim in input_shape))) if input_shape else 1


def normalize_input_linear_constraints(
    input_linear_constraints: Optional[Tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_size: int,
    input_shape: Sequence[int],
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Optional[TensorPair]:
    """Normalize user constraints to batched tensors encoding A x <= b."""
    if input_linear_constraints is None:
        return None
    if not isinstance(input_linear_constraints, (tuple, list)) or len(input_linear_constraints) != 2:
        raise ValueError("input_linear_constraints must be a pair (A, b).")

    A, b = input_linear_constraints
    A = torch.as_tensor(A).detach().clone()
    b = torch.as_tensor(b).detach().clone()

    if dtype is not None:
        A = A.to(dtype=dtype)
        b = b.to(dtype=dtype)
    else:
        A = A.float()
        b = b.float()
    if device is not None:
        A = A.to(device=device)
        b = b.to(device=device)

    if A.ndim == 2:
        A = A.unsqueeze(0)
    if b.ndim == 1:
        b = b.unsqueeze(0)
    if A.ndim != 3:
        raise ValueError("Constraint matrix A must have shape (m, d) or (B, m, d).")
    if b.ndim != 2:
        raise ValueError("Constraint rhs b must have shape (m,) or (B, m).")
    if A.shape[:2] != b.shape:
        raise ValueError("Constraint matrix A and rhs b must agree on batch and constraint dimensions.")

    expected_dim = _input_dim(input_shape)
    if A.shape[-1] != expected_dim:
        raise ValueError(
            f"Constraint matrix input dimension mismatch: expected {expected_dim}, got {A.shape[-1]}."
        )

    if A.shape[0] == 1 and batch_size > 1:
        A = A.expand(batch_size, -1, -1)
        b = b.expand(batch_size, -1)
    elif A.shape[0] != batch_size:
        raise ValueError(
            f"Constraint batch size mismatch: expected 1 or {batch_size}, got {A.shape[0]}."
        )

    return A.contiguous(), b.contiguous()


def input_constraints_to_standard_form(
    input_linear_constraints: Optional[TensorPair],
) -> Optional[TensorPair]:
    """Convert user constraints A x <= b into AutoLiRPA standard form A x + b_std <= 0."""
    if input_linear_constraints is None:
        return None
    A, b = input_linear_constraints
    return A, -b


def standard_form_to_input_constraints(
    standard_constraints: Optional[TensorPair],
) -> Optional[TensorPair]:
    """Convert AutoLiRPA standard form A x + b_std <= 0 back to A x <= b."""
    if standard_constraints is None:
        return None
    A, b_std = standard_constraints
    return A, -b_std


def normalize_standard_constraints(
    standard_constraints: Optional[TensorPair],
    *,
    batch_size: int,
    input_shape: Sequence[int],
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Optional[TensorPair]:
    """Normalize standard-form constraints A x + b <= 0 to batched tensors."""
    if standard_constraints is None:
        return None
    if not isinstance(standard_constraints, (tuple, list)) or len(standard_constraints) != 2:
        raise ValueError("standard_constraints must be a pair (A, b_std).")

    A, b = standard_constraints
    A = torch.as_tensor(A).detach().clone()
    b = torch.as_tensor(b).detach().clone()

    if dtype is not None:
        A = A.to(dtype=dtype)
        b = b.to(dtype=dtype)
    else:
        A = A.float()
        b = b.float()
    if device is not None:
        A = A.to(device=device)
        b = b.to(device=device)

    if A.ndim == 2:
        A = A.unsqueeze(0)
    if b.ndim == 1:
        b = b.unsqueeze(0)
    if A.ndim != 3:
        raise ValueError("Constraint matrix A must have shape (m, d) or (B, m, d).")
    if b.ndim != 2:
        raise ValueError("Constraint bias b must have shape (m,) or (B, m).")
    if A.shape[:2] != b.shape:
        raise ValueError("Constraint matrix A and bias b must agree on batch and constraint dimensions.")

    expected_dim = _input_dim(input_shape)
    if A.shape[-1] != expected_dim:
        raise ValueError(
            f"Constraint matrix input dimension mismatch: expected {expected_dim}, got {A.shape[-1]}."
        )

    if A.shape[0] == 1 and batch_size > 1:
        A = A.expand(batch_size, -1, -1)
        b = b.expand(batch_size, -1)
    elif A.shape[0] != batch_size:
        raise ValueError(
            f"Constraint batch size mismatch: expected 1 or {batch_size}, got {A.shape[0]}."
        )

    return A.contiguous(), b.contiguous()


def merge_standard_constraints(*constraints: Optional[TensorPair]) -> Optional[TensorPair]:
    """Concatenate multiple standard-form constraint sets along the constraint dimension."""
    active = []
    for item in constraints:
        if item is None:
            continue
        A, b = item
        if A.numel() == 0 or b.numel() == 0:
            continue
        active.append((A, b))
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    A = torch.cat([item[0] for item in active], dim=1)
    b = torch.cat([item[1] for item in active], dim=1)
    return A, b


def drop_prefix_constraints(
    constraints: Optional[TensorPair],
    prefix_count: int,
) -> Optional[TensorPair]:
    """Drop a fixed number of leading constraint rows from a standard-form tuple."""
    if constraints is None:
        return None
    if prefix_count <= 0:
        return constraints
    A, b = constraints
    if A.shape[1] <= prefix_count:
        return None
    return A[:, prefix_count:], b[:, prefix_count:]


def repeat_input_constraints(
    constraints: Optional[TensorPair],
    repeat_count: int,
) -> Optional[TensorPair]:
    """Repeat constraints along batch dimension."""
    if constraints is None:
        return None
    A, b = constraints
    repeat_shape_A = [1] * A.ndim
    repeat_shape_b = [1] * b.ndim
    repeat_shape_A[0] = repeat_count
    repeat_shape_b[0] = repeat_count
    return A.repeat(*repeat_shape_A), b.repeat(*repeat_shape_b)


def repeat_interleave_input_constraints(
    constraints: Optional[TensorPair],
    repeats: torch.Tensor,
) -> Optional[TensorPair]:
    """Repeat constraints according to per-input OR counts."""
    if constraints is None:
        return None
    A, b = constraints
    repeats = repeats.to(device=A.device, dtype=torch.long)
    return (
        A.repeat_interleave(repeats, dim=0),
        b.repeat_interleave(repeats, dim=0),
    )


def take_input_constraints_batch(
    constraints: Optional[TensorPair],
    batch_size: int,
    batch_idx: int,
) -> Optional[TensorPair]:
    """Slice a batched constraint tuple like utils.take_batch does for tensors."""
    if constraints is None:
        return None
    start = batch_idx * batch_size
    end = min(start + batch_size, constraints[0].shape[0])
    return constraints[0][start:end], constraints[1][start:end]


def affine_constraints_box_lower_bound(
    constraints: Optional[TensorPair],
    x_L: torch.Tensor,
    x_U: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Compute exact lower bounds of standard-form affine constraints over a box.

    Constraints use AutoLiRPA standard form `A x + b <= 0`, with `A` shaped
    `(m, d)` or `(B, m, d)` and `b` shaped `(m,)` or `(B, m)`. The returned
    tensor has shape `(B, m)` and contains `min_{x in [x_L, x_U]} A x + b`.
    """
    if constraints is None:
        return None

    A, b = constraints
    x_L_flat = x_L.flatten(1)
    x_U_flat = x_U.flatten(1)

    if A.ndim == 2:
        A = A.unsqueeze(0)
    if b.ndim == 1:
        b = b.unsqueeze(0)
    if A.ndim != 3 or b.ndim != 2:
        raise ValueError("constraints must be tensors with shapes (m, d)/(B, m, d) and (m,)/(B, m).")

    batch_size = x_L_flat.shape[0]
    if A.shape[0] == 1 and batch_size > 1:
        A = A.expand(batch_size, -1, -1)
        b = b.expand(batch_size, -1)
    elif A.shape[0] != batch_size:
        raise ValueError(f"Constraint batch size mismatch: expected 1 or {batch_size}, got {A.shape[0]}.")

    input_dim = x_L_flat.shape[1]
    if A.shape[-1] != input_dim:
        raise ValueError(
            f"Constraint input dimension mismatch: expected {input_dim}, got {A.shape[-1]}."
        )
    if x_U_flat.shape != x_L_flat.shape:
        raise ValueError("x_L and x_U must have the same flattened shape.")

    A = A.to(device=x_L_flat.device, dtype=x_L_flat.dtype)
    b = b.to(device=x_L_flat.device, dtype=x_L_flat.dtype)

    A_pos = A.clamp(min=0)
    A_neg = A.clamp(max=0)
    return (
        (A_pos * x_L_flat.unsqueeze(1)).sum(dim=-1)
        + (A_neg * x_U_flat.unsqueeze(1)).sum(dim=-1)
        + b
    )


def affine_constraints_domain_lower_bound(
    base_constraints: Optional[TensorPair],
    x_L: torch.Tensor,
    x_U: torch.Tensor,
    *,
    domain_constraints: Optional[TensorPair] = None,
    rearrange_constraints: bool = False,
    return_infeasible: bool = False,
) -> Optional[torch.Tensor]:
    """Lower-bound standard-form affine rows on the current feasible domain.

    When `domain_constraints` is provided, the result is computed over the
    intersection of the input box and `domain_constraints`; otherwise it falls
    back to the exact box bound.
    """
    if base_constraints is None:
        return (None, None) if return_infeasible else None

    batch_size = x_L.shape[0]
    input_shape = x_L.shape[1:]
    normalized_base = normalize_standard_constraints(
        base_constraints,
        batch_size=batch_size,
        input_shape=input_shape,
        device=x_L.device,
        dtype=x_L.dtype,
    )
    if normalized_base is None:
        return (None, None) if return_infeasible else None
    if domain_constraints is None:
        lower_bounds = affine_constraints_box_lower_bound(normalized_base, x_L, x_U)
        if return_infeasible:
            infeasible = torch.zeros(lower_bounds.shape[0], dtype=torch.bool, device=lower_bounds.device)
            return lower_bounds, infeasible
        return lower_bounds

    normalized_domain = normalize_standard_constraints(
        domain_constraints,
        batch_size=batch_size,
        input_shape=input_shape,
        device=x_L.device,
        dtype=x_L.dtype,
    )
    if normalized_domain is None or normalized_domain[0].numel() == 0:
        lower_bounds = affine_constraints_box_lower_bound(normalized_base, x_L, x_U)
        if return_infeasible:
            infeasible = torch.zeros(lower_bounds.shape[0], dtype=torch.bool, device=lower_bounds.device)
            return lower_bounds, infeasible
        return lower_bounds

    from auto_LiRPA.concretize_func import constraints_solving, sort_out_constr_batches

    processed_constraints, sorted_out_batches = sort_out_constr_batches(
        x_L,
        x_U,
        normalized_domain,
        rearrange_constraints=rearrange_constraints,
        no_return_inf=True,
        discard_abnormal=False,
    )
    if processed_constraints is None or processed_constraints[0].numel() == 0:
        lower_bounds = affine_constraints_box_lower_bound(normalized_base, x_L, x_U)
        if return_infeasible:
            infeasible = torch.zeros(lower_bounds.shape[0], dtype=torch.bool, device=lower_bounds.device)
            return lower_bounds, infeasible
        return lower_bounds

    base_A, base_b = normalized_base
    solved_bounds, infeasible_batches = constraints_solving(
        x_L,
        x_U,
        base_A,
        processed_constraints,
        sign=-1.0,
        sorted_out_batches=sorted_out_batches,
        constraints_enable=True,
        no_return_inf=True,
        use_grad=False,
    )
    lower_bounds = solved_bounds.squeeze(-1) + base_b
    if infeasible_batches is not None and infeasible_batches.any():
        lower_bounds = lower_bounds.clone()
        lower_bounds[infeasible_batches] = torch.inf
    if return_infeasible:
        if infeasible_batches is None:
            infeasible_batches = torch.zeros(
                lower_bounds.shape[0], dtype=torch.bool, device=lower_bounds.device
            )
        return lower_bounds, infeasible_batches
    return lower_bounds


def affine_constraints_domain_upper_bound(
    base_constraints: Optional[TensorPair],
    x_L: torch.Tensor,
    x_U: torch.Tensor,
    *,
    domain_constraints: Optional[TensorPair] = None,
    rearrange_constraints: bool = False,
    return_infeasible: bool = False,
) -> Optional[torch.Tensor]:
    """Upper-bound standard-form affine rows on the current feasible domain.

    This is the dual of `affine_constraints_domain_lower_bound`:
    `max(Ax+b) = -min(-(Ax+b))`.

    When `domain_constraints` is provided, the result is computed over the
    intersection of the input box and `domain_constraints`; otherwise it falls
    back to the exact box upper bound.
    """
    if base_constraints is None:
        return (None, None) if return_infeasible else None

    base_A, base_b = base_constraints
    upper = affine_constraints_domain_lower_bound(
        (-base_A, -base_b),
        x_L,
        x_U,
        domain_constraints=domain_constraints,
        rearrange_constraints=rearrange_constraints,
        return_infeasible=return_infeasible,
    )
    if return_infeasible:
        lower_neg, infeasible = upper
        return -lower_neg, infeasible
    return -upper
