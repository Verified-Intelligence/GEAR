"""Problem definitions for NLP solver module."""

from dataclasses import dataclass
from typing import Optional, Callable
import torch
import numpy as np

from input_constraints import normalize_input_linear_constraints
from .utils import _infer_device_dtype


@dataclass
class NLPResult:
    """Result container for NLP solver."""
    success: bool                           # Whether counterexample found
    x_solution: Optional[np.ndarray]        # Input point (counterexample, numpy)
    objective_value: Optional[float]        # Objective at solution
    constraint_violation: Optional[float]   # Max constraint violation
    solve_time: float                       # Wall-clock time
    status: str                             # Solver status string
    num_iterations: int                     # Solver iterations


class NNProblemSingleDomain:
    """
    Neural network optimization problem for a single domain.

    Formulates the problem as:
        minimize    C[0] @ f(x) - rhs[0]       (objective from vnnlib)
        subject to  C[i] @ f(x) - rhs[i] <= 0  for i in 1..n  (constraints)
                    dm_l <= x <= dm_u          (input bounds for this domain)

    Where f(x) is the neural network output.
    """

    def __init__(
        self,
        model: Callable,           # Neural network (PyTorch module)
        dm_l: torch.Tensor,        # Input lower bounds [*input_shape]
        dm_u: torch.Tensor,        # Input upper bounds [*input_shape]
        C_obj: torch.Tensor,       # Objective row [1, output_dim]
        rhs_obj: torch.Tensor,     # Objective threshold [1]
        C_constr: torch.Tensor,    # Constraint matrix [n_constr, output_dim]
        rhs_constr: torch.Tensor,  # Constraint thresholds [n_constr]
        input_constr_A: Optional[torch.Tensor] = None,  # Input constraint matrix [n_in_constr, input_dim]
        input_constr_b: Optional[torch.Tensor] = None,  # Input constraint rhs [n_in_constr]
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        initialization_constr_A: Optional[torch.Tensor] = None,
        initialization_constr_b: Optional[torch.Tensor] = None,
    ):
        self.model = model
        # Infer device/dtype from model parameters when not specified
        inferred_device, inferred_dtype = _infer_device_dtype(model, dm_l.device, dm_l.dtype)
        self.device = device if device is not None else inferred_device
        self.dtype = dtype if dtype is not None else inferred_dtype

        # Infer batch size from C_obj FIRST
        if C_obj.ndim == 3:
            self.batch_size = C_obj.shape[0]
        else:
            self.batch_size = 1
            C_obj = C_obj.unsqueeze(0)

        # Ensure rhs_obj is (B, 1)
        if rhs_obj.numel() == self.batch_size:
            rhs_obj = rhs_obj.view(self.batch_size, 1)
        elif rhs_obj.numel() == 1:
            rhs_obj = rhs_obj.view(1, 1).expand(self.batch_size, 1)

        # Ensure C_constr/rhs_constr are (B, m, out) and (B, m)
        if C_constr.numel() > 0:
            if C_constr.ndim == 2:
                C_constr = C_constr.unsqueeze(0)
            if C_constr.shape[0] == 1 and self.batch_size > 1:
                C_constr = C_constr.expand(self.batch_size, *C_constr.shape[1:])
            
            if rhs_constr.ndim == 1:
                rhs_constr = rhs_constr.unsqueeze(0)
            if rhs_constr.shape[0] == 1 and self.batch_size > 1:
                rhs_constr = rhs_constr.expand(self.batch_size, *rhs_constr.shape[1:])

        # Determine input_shape (per-instance)
        # Always attempt to strip batch dimension if it matches batch_size
        if dm_l.ndim > 1 and dm_l.shape[0] == self.batch_size:
             self.input_shape = dm_l.shape[1:]
        else:
             self.input_shape = dm_l.shape
        
        self.n_vars = int(np.prod(self.input_shape))

        # Flatten bounds for scipy
        self.lb = dm_l.flatten().cpu().numpy().astype(np.float64)
        self.ub = dm_u.flatten().cpu().numpy().astype(np.float64)

        # Store specification with inferred device/dtype
        self.C_obj = C_obj.to(device=self.device, dtype=self.dtype)
        self.rhs_obj = rhs_obj.to(device=self.device, dtype=self.dtype)
        self.C_constr = C_constr.to(device=self.device, dtype=self.dtype)
        self.rhs_constr = rhs_constr.to(device=self.device, dtype=self.dtype)

        # Store original bounds for PGD (handle batch dim)
        self.dm_l = dm_l.to(device=self.device, dtype=self.dtype)
        self.dm_u = dm_u.to(device=self.device, dtype=self.dtype)
        
        self.output_dim = self.C_obj.shape[-1]

        # Ensure model handles batched inputs.  Some onnx2pytorch models
        # (e.g. those with Gather on axis 0) treat dim-0 as data, not batch,
        # silently collapsing the batch dimension.  Detect this once and wrap.
        self.model = self._ensure_batched_model(model)

        # Ensure bounds match batch size
        if self.dm_l.shape[0] != self.batch_size:
             # Broadcast if needed (e.g. single bound for multiple specs)
            if self.dm_l.shape[0] == 1:
                self.dm_l = self.dm_l.expand(self.batch_size, *self.dm_l.shape[1:])
                self.dm_u = self.dm_u.expand(self.batch_size, *self.dm_u.shape[1:])
            elif self.batch_size == 1:
                # If bounds have batch dim but C_obj doesn't (unlikely but possible)
                # We trust C_obj implies batch size.
                 pass
            else:
                 # Mismatched batch sizes
                 raise ValueError(f"Batch size mismatch: C_obj={self.batch_size}, dm_l={self.dm_l.shape[0]}")

        input_linear_constraints = None
        if input_constr_A is not None or input_constr_b is not None:
            if input_constr_A is None or input_constr_b is None:
                raise ValueError("input_constr_A and input_constr_b must be provided together.")
            input_linear_constraints = normalize_input_linear_constraints(
                (input_constr_A, input_constr_b),
                batch_size=self.batch_size,
                input_shape=self.input_shape,
                device=self.device,
                dtype=self.dtype,
            )

        if input_linear_constraints is None:
            self.input_constr_A = torch.empty(
                (self.batch_size, 0, self.n_vars),
                device=self.device,
                dtype=self.dtype,
            )
            self.input_constr_b = torch.empty(
                (self.batch_size, 0),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self.input_constr_A, self.input_constr_b = input_linear_constraints

        # Optional structural constraints used only to initialize heuristics.
        # They are separate from the authoritative residual list because
        # proof-time output expansion may already contain the same A x - b
        # rows; appending them again would duplicate the ALM penalty.
        initialization_linear_constraints = None
        if initialization_constr_A is not None or initialization_constr_b is not None:
            if initialization_constr_A is None or initialization_constr_b is None:
                raise ValueError(
                    "initialization_constr_A and initialization_constr_b "
                    "must be provided together."
                )
            initialization_linear_constraints = normalize_input_linear_constraints(
                (initialization_constr_A, initialization_constr_b),
                batch_size=self.batch_size,
                input_shape=self.input_shape,
                device=self.device,
                dtype=self.dtype,
            )
        elif input_linear_constraints is not None:
            initialization_linear_constraints = input_linear_constraints

        if initialization_linear_constraints is None:
            self.initialization_constr_A = torch.empty(
                (self.batch_size, 0, self.n_vars),
                device=self.device,
                dtype=self.dtype,
            )
            self.initialization_constr_b = torch.empty(
                (self.batch_size, 0),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            (
                self.initialization_constr_A,
                self.initialization_constr_b,
            ) = initialization_linear_constraints

        self.n_output_constraints = self.C_constr.shape[1] if self.C_constr.numel() > 0 else 0
        self.n_input_constraints = self.input_constr_A.shape[1]
        self.n_constraints = self.n_output_constraints + self.n_input_constraints

        # NOTE: we deliberately do NOT precompute a concatenated [C_obj; C_constr]
        # ("C_full"). For constraint-heavy specs (e.g. case57: 29493 rows x ~27400
        # outputs) that dense matrix is ~6.5GB, and torch.cat *duplicates* it,
        # doubling peak memory and OOMing. C_obj and C_constr already live
        # separately; compute_phi_and_output_residuals below matmuls on each and
        # concatenates only the small (N x m) result vectors.

    def compute_phi_and_output_residuals(self, out_BR, B, R):
        """Forward: returns (phi (N,), g_out (N, m_out)).

        g_out is empty (N, 0) if no output constraints exist. Computed from
        C_obj / C_constr separately to avoid materializing a concatenated dense
        C_full (see __init__ note).
        """
        N = B * R
        phi_BR = out_BR @ self.C_obj.transpose(1, 2) - self.rhs_obj.unsqueeze(1)  # (B,R,1)
        phi = phi_BR[..., 0].reshape(N)
        if self.n_output_constraints > 0:
            g_BR = out_BR @ self.C_constr.transpose(1, 2) - self.rhs_constr.unsqueeze(1)  # (B,R,m_out)
            g_out = g_BR.reshape(N, -1)
        else:
            g_out = phi.new_zeros((N, 0))
        return phi, g_out

    def compute_phi(self, x_flat, out_BR, B, R):
        """Compute objective: C_obj @ out - rhs_obj, returns (N,).

        This method is called by ALM to compute the objective value.
        Subclasses can override this to implement different objectives
        (e.g., bound tightening optimizes input coordinates instead of
        network output).

        Args:
            x_flat: Flattened input tensor (N, D) where N = B * R.
            out_BR: Network output reshaped to (B, R, out_dim).
            B: Batch size (number of problem instances).
            R: Number of restarts per instance.

        Returns:
            Tensor of shape (N,) with objective values.
        """
        N = B * R
        phi = (out_BR @ self.C_obj.transpose(1, 2)).squeeze(-1) - self.rhs_obj
        return phi.reshape(N)

    def compute_constraint_residuals(
        self,
        x: torch.Tensor,
        out_BR: torch.Tensor,
        B: int,
        R: int,
    ) -> torch.Tensor:
        """Compute all constraint residuals in fixed order [output, input]."""
        residuals = []

        if self.n_output_constraints > 0:
            g_out = (out_BR @ self.C_constr.transpose(1, 2)) - self.rhs_constr.unsqueeze(1)
            residuals.append(g_out)

        if self.n_input_constraints > 0:
            x_flat = x.reshape(B * R, -1).reshape(B, R, self.n_vars)
            g_in = (x_flat @ self.input_constr_A.transpose(1, 2)) - self.input_constr_b.unsqueeze(1)
            residuals.append(g_in)

        if not residuals:
            return torch.zeros(B * R, 0, device=x.device, dtype=x.dtype)
        return torch.cat(residuals, dim=2).reshape(B * R, -1)

    def _ensure_batched_model(self, model):
        """Return a model that correctly handles batched inputs.

        Some onnx2pytorch models (e.g. those with Gather on axis 0) treat
        dim-0 as the data axis rather than the batch axis, silently collapsing
        the batch dimension.  We probe the model with a 2-sample batch; if the
        output doesn't preserve the leading dimension, we wrap the model so
        that every call fans out to per-sample invocations and re-stacks the
        results.  The wrapper is an nn.Module so that parameter iteration,
        device transfer, and eval/train mode propagate transparently.
        """
        probe = torch.zeros(2, *self.input_shape,
                            device=self.device, dtype=self.dtype)
        try:
            with torch.no_grad():
                probe_out = model(probe)
            batch_safe = probe_out.shape[0] == 2
        except RuntimeError:
            # A model that concatenates batch-1 constants raises on a batch of 2
            # instead of collapsing it (st_e06); it needs the same per-sample wrapper.
            batch_safe = False
        if batch_safe:
            return model  # Already batch-safe

        class _BatchWrapper(torch.nn.Module):
            """Per-sample forward calls, vectorised with vmap when possible.

            The per-sample Python loop made one ALM inner step on st_e06 (20480
            restarts) take ~0.15 s, so the root ALM ran a single outer iteration
            in its 30 s budget.  vmap over the batch-1 call computes the same
            values ~300x faster; the loop remains the fallback.
            """
            def __init__(self, base):
                super().__init__()
                self.base = base
                self.use_vmap = True

            def _loop(self, x):
                return torch.cat([self.base(x[i:i+1]) for i in range(x.shape[0])],
                                 dim=0)

            def forward(self, x):
                if x.shape[0] == 1:
                    return self.base(x)
                if self.use_vmap:
                    try:
                        return torch.func.vmap(
                            lambda xi: self.base(xi.unsqueeze(0)).squeeze(0)
                        )(x)
                    except Exception:  # pylint: disable=broad-except
                        self.use_vmap = False
                return self._loop(x)

        return _BatchWrapper(model)

    def _to_torch(self, x: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor with correct shape."""
        x_torch = torch.from_numpy(x.astype(np.float64)).to(
            device=self.device, dtype=self.dtype
        ).view(1, *self.input_shape)
        return x_torch

    def objective(self, x: np.ndarray) -> float:
        """Compute objective value: C[0] @ f(x) - rhs[0]."""
        x_torch = self._to_torch(x)
        with torch.no_grad():
            out = self.model(x_torch)
        # Use view to handle batching correctly (B, 1) or scalar
        obj = (self.C_obj @ out.T).view(self.batch_size, -1) - self.rhs_obj.view(self.batch_size, -1)
        return float(obj.sum().item())

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of objective via PyTorch autograd."""
        x_torch = self._to_torch(x)
        x_torch = x_torch.clone().detach().requires_grad_(True)
        out = self.model(x_torch)
        obj = (self.C_obj @ out.T).view(self.batch_size, -1) - self.rhs_obj.view(self.batch_size, -1)
        obj.sum().backward()
        grad = x_torch.grad.flatten().cpu().numpy().astype(np.float64)
        return grad

    def constraints(self, x: np.ndarray) -> np.ndarray:
        """Compute constraint residuals in fixed order [output, input]."""
        if self.n_constraints == 0:
            return np.array([], dtype=np.float64)
        x_torch = self._to_torch(x)
        with torch.no_grad():
            out = self.model(x_torch)
        residuals = []
        if self.n_output_constraints > 0:
            residuals.append((self.C_constr @ out.T).squeeze(-1) - self.rhs_constr)
        if self.n_input_constraints > 0:
            x_flat = x_torch.reshape(self.batch_size, -1)
            residuals.append((self.input_constr_A @ x_flat.unsqueeze(-1)).squeeze(-1) - self.input_constr_b)
        constr = torch.cat(residuals, dim=1)
        return constr.flatten().cpu().numpy().astype(np.float64)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Compute Jacobian of constraints via PyTorch autograd."""
        if self.n_constraints == 0:
            return np.array([], dtype=np.float64).reshape(0, self.n_vars)

        x_torch = self._to_torch(x)
        x_torch = x_torch.clone().detach().requires_grad_(True)
        out = self.model(x_torch)
        jac_rows = []
        if self.n_output_constraints > 0:
            constr_out = (self.C_constr @ out.T).squeeze(-1) - self.rhs_constr
            for i in range(self.n_output_constraints):
                if x_torch.grad is not None:
                    x_torch.grad.zero_()
                keep_graph = i < self.n_output_constraints - 1 or self.n_input_constraints > 0
                constr_out[:, i].sum().backward(retain_graph=keep_graph)
                jac_rows.append(x_torch.grad.flatten().cpu().numpy().astype(np.float64).copy())

        if self.n_input_constraints > 0:
            input_jac = self.input_constr_A.reshape(-1, self.n_vars).cpu().numpy().astype(np.float64)
            jac_rows.extend([row.copy() for row in input_jac])

        return np.array(jac_rows)
