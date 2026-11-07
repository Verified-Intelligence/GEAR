"""Verification of a candidate primal point against the real network."""

from typing import Tuple, Callable, Optional
import torch

from .utils import _infer_device_dtype



def verify_counterexample(
    model: Callable,
    x: torch.Tensor,
    C_obj: torch.Tensor,
    rhs_obj: torch.Tensor,
    C_constr: torch.Tensor,
    rhs_constr: torch.Tensor,
    input_constr_A: Optional[torch.Tensor] = None,
    input_constr_b: Optional[torch.Tensor] = None,
    tol: float = 1e-4,
) -> Tuple[bool, float, float]:
    """
    Verify that a point is a valid feasible solution.

    Valid = feasible (all constraints satisfied within tolerance).
    The caller decides what to do with the objective value (e.g., check obj < 0
    for counterexample detection).

    Uses model's device/dtype for computation to ensure consistency.

    Returns:
        (is_valid, objective_value, max_constraint_violation)
    """
    with torch.no_grad():
        x_input = x.unsqueeze(0)
        # Match model device/dtype for consistent computation.
        # Generated NLP benchmark models often keep their effective floating dtype
        # in buffers rather than parameters, so verification must follow the model
        # dtype first to avoid C_obj @ f(x) mismatches.
        dev, model_dt = _infer_device_dtype(model, x.device, x.dtype)
        dt = model_dt if torch.empty((), dtype=model_dt).is_floating_point() else (
            C_obj.dtype if torch.is_floating_point(C_obj) else x.dtype
        )
        x_input = x_input.to(device=dev, dtype=dt)
        C_obj_ = C_obj.to(device=dev, dtype=dt)
        rhs_obj_ = rhs_obj.to(device=dev, dtype=dt)
        C_constr_ = C_constr.to(device=dev, dtype=dt)
        rhs_constr_ = rhs_constr.to(device=dev, dtype=dt)
        x_flat = x.reshape(-1).to(device=dev, dtype=dt)

        out = model(x_input)

        obj = float(((C_obj_ @ out.T).squeeze() - rhs_obj_.squeeze()).item())

        if C_constr.numel() > 0:
            g = (C_constr_ @ out.T).squeeze(-1) - rhs_constr_
            output_viol = float(torch.max(torch.relu(g)).item())
        else:
            output_viol = 0.0

        input_viol = 0.0
        if input_constr_A is not None or input_constr_b is not None:
            if input_constr_A is None or input_constr_b is None:
                raise ValueError("input_constr_A and input_constr_b must be provided together.")

            if input_constr_A.ndim == 3:
                if input_constr_A.shape[0] != 1:
                    raise ValueError("verify_counterexample only supports a single input constraint batch.")
                input_constr_A = input_constr_A[0]
            elif input_constr_A.ndim != 2:
                raise ValueError("input_constr_A must have shape (n_constr, input_dim) or (1, n_constr, input_dim).")

            if input_constr_b.ndim == 2:
                if input_constr_b.shape[0] != 1:
                    raise ValueError("verify_counterexample only supports a single input constraint batch.")
                input_constr_b = input_constr_b[0]
            elif input_constr_b.ndim != 1:
                raise ValueError("input_constr_b must have shape (n_constr,) or (1, n_constr).")

            input_constr_A_ = input_constr_A.to(device=dev, dtype=dt)
            input_constr_b_ = input_constr_b.to(device=dev, dtype=dt)
            g_input = (input_constr_A_ @ x_flat) - input_constr_b_
            input_viol = float(torch.max(torch.relu(g_input)).item()) if g_input.numel() > 0 else 0.0

        viol = max(output_viol, input_viol)

    is_valid = viol <= tol
    return is_valid, obj, viol
