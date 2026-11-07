"""Global outer-approximation cuts for proved convex quadratic spec rows.

Rows have the solver's signed convention C[r] @ model(x) <= rhs[r]. A paired
equality therefore supplies a cut only for its convex side. Slight negative
curvature is allowed, but its lower eigenvalue bound always contributes QTB's
distance correction over the entire root box, including anchors outside it.
"""
from __future__ import annotations

import math

import torch

from nlp_solver.quadratic_tangent import QuadraticTangent, output_degrees


class ConvexRowCuts:
    """Cache constant row Hessians and retain at most four incumbent cut rounds."""

    def __init__(self, tangents, row_rhs, lower, upper, model_dtype):
        self.tangents = tangents
        self.row_rhs = row_rhs
        self.lower, self.upper = lower, upper
        self.model_dtype = model_dtype
        self.rounds = 0
        self.best_primal = math.inf

    def offer(self, x_hat, primal):
        """New standard-form rows A x + b <= 0, or None without an improvement.

        The stored rows already include the output feasibility tolerance, so
        clipping, concretization and base-row filtering can use them directly.
        Consumers which also relax base input rows only weaken these cuts further.
        """
        if x_hat is None or self.rounds >= 4 or not math.isfinite(float(primal)):
            return None
        if self.rounds and float(primal) >= self.best_primal:
            return None
        x_hat = x_hat.detach().reshape(-1).to(self.lower)
        if x_hat.numel() != self.lower.numel() or not torch.isfinite(x_hat).all():
            return None
        lower, upper = self.lower.double(), self.upper.double()
        max_abs = torch.maximum(lower.abs(), upper.abs())
        rows, biases = [], []
        for tangent, rhs in zip(self.tangents, self.row_rhs):
            if tangent.update(x_hat) is False:
                continue
            a, b = tangent.plane(lower, upper)
            # Include evaluation/gradient and affine accumulation rounding, and
            # enclose conversion of coefficients back to the solver's dtype.
            scale = (1. + abs(tangent.f) + (a.abs() * max_abs).sum()
                     + (tangent.g * tangent.x_hat).abs().sum() + rhs.abs())
            eps = max(torch.finfo(self.model_dtype).eps, torch.finfo(self.lower.dtype).eps)
            bias = b - rhs - 4 * (a.shape[-1] + 16) * eps * scale
            row = a.to(self.lower)
            bias = bias - ((row.double() - a).abs() * max_abs).sum(-1)
            bias = torch.nextafter(bias.to(self.lower), torch.full_like(bias.to(self.lower), -torch.inf))
            if torch.isfinite(row).all() and torch.isfinite(bias).all():
                rows.append(row)
                biases.append(bias)
        if not rows:
            return None
        self.rounds += 1
        self.best_primal = float(primal)
        return torch.stack(rows, dim=1), torch.stack(biases, dim=1)


def build_convex_row_cuts(bounded_module, model_ori, x_input, c, rhs, root_lower,
                          root_upper, x_hat, *, enabled=False, delta=1e-8,
                          max_vars=2000, max_rows=50, skip_rows=(), rowpool, allow_uncertified=False):
    """Prove row degrees, then admit only finite Hessians with lambda_min >= -delta.

    Degrees are checked for every output used by the signed row, rather than
    numerically guessing a quadratic. The root hull covers all initial domains;
    differing specifications cannot share problem-level cuts.

    The tangents built here carry no rounding certificate (f, grad f and the curvature are the computed values with
    a relative cushion), so a cancelling graph can give an invalid cut. Without `allow_uncertified` no cut is built;
    nlp_convex_tangent_rows (convex_tangent.build_convex_tangent_rows) is the certified path for these rows.
    """
    n = x_input[0].numel()
    if enabled and not allow_uncertified:
        print("[ConvexRowCuts] disabled: the row tangents have no rounding certificate (use convex_tangent_rows)")
        return None
    if not enabled or x_hat is None or n > max_vars or c.shape[1] < 2:
        return None
    if not (torch.equal(c, c[:1].expand_as(c))
            and torch.equal(rhs[:, 1:], rhs[:1, 1:].expand_as(rhs[:, 1:]))):
        return None
    lower = root_lower.flatten(1).amin(0, keepdim=True).detach()
    upper = root_upper.flatten(1).amax(0, keepdim=True).detach()
    if not (torch.isfinite(lower).all() and torch.isfinite(upper).all()):
        return None
    degrees = output_degrees(bounded_module, x_input[:1])
    if degrees.numel() != c.shape[-1]:
        return None
    # Same thresholds as the CROWN planes, pruning, and the node LP read.
    relaxed_rhs = rowpool.thresholds(rhs)[:1]
    candidates = []
    for r in range(1, c.shape[1]):  # spec row 0 is the objective, never a constraint
        if r in skip_rows:
            continue
        coeff = c[0, r].detach()
        support = coeff.detach().cpu() != 0
        if not torch.isfinite(coeff).all() or not torch.isfinite(relaxed_rhs[0, r]):
            continue
        if not (degrees[support] <= 2).all():
            continue
        if not (degrees[support] == 2).any():
            # An affine row is its own tangent: re-adding it as a base row only moved
            # every concretization onto the constrained path (cute_gouldqp2/3: 698
            # duplicated linear rows, root over budget, both lost).
            continue
        candidates.append((r, coeff))
    if len(candidates) > max_rows:
        # Each candidate costs a Hessian, and each cut row slows every concretization:
        # beyond a few dozen rows the cuts never paid off (kriging_peaks-full500 never
        # left the root; the gains, e.g. cute_airport, needed at most 42 rows).
        print(f"[ConvexRowCuts] skipped: {len(candidates)} quadratic rows exceed the cap {max_rows}")
        return None
    tangents, row_rhs = [], []
    for r, coeff in candidates:
        def row_fn(z, coeff=coeff):
            outputs = model_ori(z.reshape(x_input[:1].shape).to(x_input.dtype)).reshape(-1)
            # Ignore unused outputs: 0 * an undefined unrelated output is not a row.
            used = coeff != 0
            return (outputs[used] * coeff[used]).sum().double()

        try:
            tangent = QuadraticTangent(row_fn, n, x_hat.to(lower))
        except RuntimeError:
            # The cuts are optional: st_e11's GPU eigvalsh failed to converge and the
            # uncaught error ended the run (rc=1) that 51a2c25 verified in 33 s.
            continue
        # Classify using the eigenvalue estimate; the cut itself retains the
        # conservative lower bound, including its Hessian rounding allowance.
        if math.isfinite(tangent.lam_lb) and tangent.lam_min >= -delta:
            tangents.append(tangent)
            row_rhs.append(relaxed_rhs[0, r].detach().double())
    if not tangents:
        return None
    return ConvexRowCuts(tangents, row_rhs, lower, upper, x_input.dtype)
