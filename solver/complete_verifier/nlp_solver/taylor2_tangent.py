"""QTB-compatible Taylor planes with certified interval curvature on each box."""
from __future__ import annotations

import math

import torch

from .interval_hessian import IntervalHessian, outward, add, mul
from .quadratic_tangent import QuadraticTangent


class Taylor2Tangent(QuadraticTangent):
    def __init__(self, bounded_module, x_input, obj_index, objective_fn, x_hat,
                 root_lower=None, root_upper=None, max_domains=1024):
        self.objective_fn = objective_fn
        self.n = x_input.numel()
        self.max_domains = max(0, int(max_domains))
        self.linear_constraint_outputs = False
        self.lam_lb = self.lam_min = -float('inf')  # Never claim global convexity.
        self.polished = False
        xh = x_hat.detach().reshape(-1).double()
        self.fixed_mask = torch.zeros(self.n, dtype=torch.bool, device=xh.device)
        self.fixed_values = torch.zeros_like(xh)
        if root_lower is not None or root_upper is not None:
            if root_lower is None or root_upper is None:
                raise ValueError('Both root box endpoints are required')
            lo, hi = root_lower.reshape(-1).to(xh), root_upper.reshape(-1).to(xh)
            if lo.numel() != self.n or hi.numel() != self.n or (lo > hi).any():
                raise ValueError('Invalid root box for Taylor tangent')
            self.fixed_mask = (lo == hi) & torch.isfinite(lo)
            self.fixed_values = torch.where(self.fixed_mask, lo, 0.)
        self.evaluator = IntervalHessian(bounded_module, x_input, obj_index)
        largest = max(math.prod(shape) for shape in self.evaluator.shapes.values())
        # Bound peak AD storage independently of the solver's auto-enlarged batch.
        self.chunk_size = max(1, 2_000_000 // max(1, largest * self.n * self.n))
        self.update(xh)

    def update(self, x_hat):
        super().update(x_hat)
        # Point interval AD encloses value/gradient evaluation error accumulated
        # across the graph. This also checks C2 validity at the anchor itself.
        jet = self.evaluator.evaluate(self.x_hat[None], self.x_hat[None])
        self.anchor_valid = (bool(jet.valid.all()) and torch.isfinite(self.g).all().item()
                             and torch.isfinite(torch.tensor(self.f)).item())
        self.f_error = max(abs(self.f - float(t[0, 0])) for t in jet.value) + 1e-9 * (1. + abs(self.f))
        self.g_error = torch.maximum(*(abs(self.g - t[0, 0]) for t in jet.gradient)) + 1e-9 * (1. + self.g.abs())

    def plane(self, l, u, bound_scores=None):
        batch = l.shape[0]
        if not self.anchor_valid:
            return torch.zeros_like(l), torch.full_like(l[:, 0], -torch.inf)
        mu = torch.full((batch,), -torch.inf, dtype=torch.float64, device=l.device)
        if self.max_domains and batch:
            count = min(batch, self.max_domains)
            indices = (torch.arange(batch, device=l.device) if count == batch else
                       torch.argsort(bound_scores)[:count] if bound_scores is not None else
                       torch.arange(count, device=l.device))
            for chunk in indices.split(self.chunk_size):
                lo, hi = l[chunk].double(), u[chunk].double()
                xh = self.x_hat.to(lo)
                mu[chunk] = self.evaluator.lower_eigenvalue(
                    torch.minimum(lo, xh), torch.maximum(hi, xh), self.fixed_mask)
        # Use QTB's plane/max-distance code in double, including for float32 callers.
        a, b = super().plane(l.double(), u.double(), lam_lb=mu)
        displacement = add((l.double(), u.double()),
                           (-self.x_hat.to(l.device), -self.x_hat.to(l.device)))
        distance = torch.maximum(displacement[0].abs(), displacement[1].abs())
        gradient_error = self.g_error.to(distance)
        terms = mul((gradient_error, gradient_error), (distance, distance))[1]
        error = outward(terms.sum(-1) + self.f_error, terms.sum(-1) + self.f_error,
                        terms.sum(-1) + self.f_error, self.n)[1]
        b = outward(b - error, b - error, b.abs() + error, self.n)[0]
        # Enclose the coefficient conversion as well as the intercept conversion.
        a_out = a.to(l)
        delta = (a_out.double() - a).abs()
        magnitude = torch.maximum(l.double().abs(), u.double().abs())
        cast_terms = mul((delta, delta), (magnitude, magnitude))[1]
        cast_error = outward(cast_terms.sum(-1), cast_terms.sum(-1), cast_terms.sum(-1), self.n)[1]
        b_out = add((b, b), (-cast_error, -cast_error))[0]
        b_out = torch.nextafter(b_out.to(l), torch.full_like(l[:, 0], -torch.inf))
        return a_out, torch.where(torch.isfinite(b_out), b_out, -torch.inf)
