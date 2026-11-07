"""Quadratic tangent bound: a global lower plane for a provably quadratic objective.

If f is a polynomial of degree <= 2 with Hessian H, then for every x and any point x_hat

    f(x) = f(x_hat) + g^T (x - x_hat) + 1/2 (x - x_hat)^T H (x - x_hat),   g = grad f(x_hat)
         >= f(x_hat) + g^T (x - x_hat) + 1/2 min(lambda_min(H), 0) ||x - x_hat||^2,

so on a box [l, u] the plane g^T x + f(x_hat) - g^T x_hat + 1/2 min(lambda_min, 0) *
sum_j max((x_hat_j - l_j)^2, (u_j - x_hat_j)^2) is a valid lower bound. For a convex
objective (lambda_min >= 0) this is the tangent plane itself, and at a KKT point of a
linearly constrained problem its minimum over the feasible set equals f(x_hat): the
single CROWN plane of a convex quadratic is instead loose everywhere but at one vertex.

Degree <= 2 is proved on the bound graph (per output element, exact polynomial degree
propagation; any other perturbed operator counts as infinite degree), never guessed
numerically. The tangent point needs no feasibility: the plane is valid anywhere.
Coordinates fixed by the root box have zero displacement, so only the Hessian on
the free coordinates matters. Positive curvature can also be retained in a separable
quadratic lower model, minimized over the box with nonnegative row multipliers.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import torch
from torch import Tensor

from .rigorous_dual import ETA, U, Rows, dual_lower_bound, gamma, round_down

INF = float("inf")
NEG = -1e9      # "does not depend on the input at all" in max-plus products

_DATA_MOVEMENT = {
    "BoundGather", "BoundTranspose", "BoundReshape", "BoundConcat", "BoundUnsqueeze",
    "BoundSqueeze", "BoundSlice", "BoundExpand", "BoundFlatten", "BoundCast",
    "BoundIdentity", "BoundSplit",
}


def _const(node) -> Tensor:
    v = node.forward_value if node.forward_value is not None else getattr(node, "value", None)
    # Degree tensors live on the CPU; a constant left on the model's device would
    # make every op that mixes the two raise, i.e. count as infinite degree.
    return torch.as_tensor(v).detach().cpu()


def _is_const(node) -> bool:
    return not getattr(node, "perturbed", True)


def _degree_of(node, deg) -> Tensor:
    """Degree tensor (same shape as the node output) from input degree tensors."""
    name = type(node).__name__
    ins = node.inputs
    shape = node.forward_value.shape if node.forward_value is not None else None
    if name == "BoundInput":
        return torch.ones(shape, dtype=torch.float64)
    if _is_const(node):
        return torch.zeros(shape, dtype=torch.float64)
    d = [deg[i.name] for i in ins]
    if name in ("BoundAdd", "BoundSub"):
        return torch.maximum(d[0], d[1])
    if name == "BoundMul":
        return d[0] + d[1]
    if name == "BoundNeg":
        return d[0].clone()
    if name == "BoundSqr":
        return 2 * d[0]
    if name == "BoundPow":
        if _is_const(ins[1]):
            k = _const(ins[1]).reshape(-1)
            if k.numel() == 1 and float(k) >= 0 and float(k) == int(float(k)):
                return int(float(k)) * d[0]
        return torch.full(shape, INF, dtype=torch.float64)
    if name == "BoundDiv":
        if _is_const(ins[1]):
            return torch.broadcast_to(d[0], shape).clone()
        return torch.full(shape, INF, dtype=torch.float64)
    if name == "BoundMatMul":
        a, b = d[0], d[1]
        if _is_const(ins[1]):
            b = torch.where(_const(ins[1]) != 0, 0.0, NEG).to(torch.float64)
        if _is_const(ins[0]):
            a = torch.where(_const(ins[0]) != 0, 0.0, NEG).to(torch.float64)
        # max-plus product: out[..., i, j] = max_k a[..., i, k] + b[..., k, j]
        if b.ndim == 1:
            out = (a + b).amax(-1)
        else:
            out = (a.unsqueeze(-1) + b.unsqueeze(-3)).amax(-2)
        return out.clamp(min=0).reshape(shape)
    if name == "BoundLinear":
        # y = x @ w.t() (+ b) after the node's own transB/alpha/beta handling; the
        # api's structural-constraint outputs (F.linear of the input) take this form.
        if getattr(node, "transA", 0) or not all(_is_const(i) for i in ins[1:]):
            return torch.full(shape, INF, dtype=torch.float64)
        x_val = torch.zeros(d[0].shape, dtype=torch.float64)
        consts = [_const(i).to(torch.float64) for i in ins[1:]]
        _, w, _ = node._preprocess(x_val, *consts)
        wmask = torch.where(w != 0, 0.0, NEG).to(torch.float64)       # (out, in)
        out = (d[0].unsqueeze(-2) + wmask).amax(-1)
        return out.clamp(min=0).reshape(shape)
    if name == "BoundReduceSum":
        x = d[0]
        axis = getattr(node, "axis", None)
        keepdim = bool(getattr(node, "keepdim", False))
        if axis is None:
            return x.amax().reshape(shape)
        return x.amax(dim=tuple(axis) if isinstance(axis, (list, tuple)) else axis,
                      keepdim=keepdim).reshape(shape)
    if name == "BoundATenIndex_add":
        if not all(_is_const(ins[k]) for k in (1, 2)):
            return torch.full(shape, INF, dtype=torch.float64)
        axis = int(_const(ins[1]))
        data, src = d[0], d[3]
        if axis < 0:
            axis += data.ndim
        if data.ndim == src.ndim and data.shape != src.shape:
            tgt = [max(a, b) if i != axis else None for i, (a, b) in
                   enumerate(zip(data.shape, src.shape))]
            data = data.expand([t if t is not None else data.shape[i] for i, t in enumerate(tgt)])
            src = src.expand([t if t is not None else src.shape[i] for i, t in enumerate(tgt)])
        out = data.clone().index_reduce_(axis, _const(ins[2]).long(), src, "amax",
                                         include_self=True)
        return out.reshape(shape)
    if name == "BoundWhere":
        if not _is_const(ins[0]):
            return torch.full(shape, INF, dtype=torch.float64)
        return torch.where(_const(ins[0]).to(torch.bool), d[1], d[2]).reshape(shape)
    if name in _DATA_MOVEMENT:
        args = [deg[i.name] if not _is_const(i) else _const(i) for i in ins]
        out = node.forward(*args)
        return torch.as_tensor(out, dtype=torch.float64).reshape(shape)
    return torch.full(shape, INF, dtype=torch.float64)


@torch.no_grad()
def output_degrees(bounded_module, x: Tensor) -> Tensor:
    """Polynomial degree of every element of the final output (inf = not polynomial)."""
    bounded_module(x)                      # fills forward_value on every node
    deg = {}
    order = []
    seen = set()

    def visit(node):
        if node.name in seen:
            return
        seen.add(node.name)
        for i in node.inputs:
            visit(i)
        order.append(node)

    final = bounded_module[bounded_module.final_name]
    visit(final)
    for node in order:
        try:
            deg[node.name] = _degree_of(node, deg)
        except Exception:  # noqa: BLE001 - unknown shapes/ops: be conservative
            shape = node.forward_value.shape if node.forward_value is not None else ()
            deg[node.name] = torch.full(shape, INF, dtype=torch.float64)
    bounded_module(x)                      # restore any cache a data-movement op touched
    return deg[final.name].reshape(-1)


def _scaled_sqrt_sumsq(d: Tensor) -> float:
    """sqrt(sum d^2) with the squares scaled by an exact power of 2 (no underflow); the sum is the caller's to inflate."""
    m = float(d.max()) if d.numel() else 0.0
    if not (m > 0.0 and math.isfinite(m)):
        return m if m == 0.0 else INF
    s = 2.0 ** math.frexp(m)[1]
    return s * math.sqrt(float(((d / s) ** 2).sum()))


def _merge_sumsq(acc, d: Tensor):
    """(q, s) with q s^2 >= the running sum of squares plus sum d^2; rescales q to the larger power-of-2 scale."""
    q, s = acc
    m = float(d.max()) if d.numel() else 0.0
    if not math.isfinite(m):
        return (INF, s)
    if not m > 0.0:
        return acc
    t = 2.0 ** math.frexp(m)[1]
    new = max(s, t) if q > 0.0 else t
    return (q * (s / new) ** 2 + float(((d / new) ** 2).sum()), new)


def frobenius(M: Tensor) -> float:
    """||M||_F without underflow of the squares: scale by an exact power of 2 (the result is rounded like the unscaled
    norm up to the usual gamma terms; an underflowing result is below ETA, which callers add where it matters)."""
    m = float(M.abs().max()) if M.numel() else 0.0
    if not (m > 0.0 and math.isfinite(m)):
        return float(torch.linalg.matrix_norm(M, ord='fro')) if M.numel() else 0.0
    s = 2.0 ** math.frexp(m)[1]
    return s * float(torch.linalg.matrix_norm(M / s, ord='fro'))


class QuadraticTangent:
    """Tangent-plane lower bound for a quadratic objective (see module docstring)."""
    exact_eval = None            # x_hat -> (f, grad f) in exact rationals, small graphs only (_exact_evaluator)
    qtb_certificate = None       # subclasses that skip __init__ keep the uncertified plane
    f_qtb = g_qtb = None

    def __init__(self, objective_fn, n: int, x_hat: Tensor,
                 root_lower: Optional[Tensor] = None,
                 root_upper: Optional[Tensor] = None,
                 certificate=None, obj_index: int = 0, exact_eval=None,
                 exact_hessian=None, deadline: float = INF):
        self.objective_fn = objective_fn
        self.exact_hessian, self.deadline = exact_hessian, deadline      # (I, J, E) triplets; time.time() cutoff
        self.n = n
        # `certificate`: an IntervalHessian(first_order=True) of the same graph, whose point jets enclose the exact
        # f(x_hat), grad f(x_hat) (see _point_errors). Without it f_error is None and the plane keeps only its cushion.
        self.qtb_certificate, self.obj_index, self.exact_eval = certificate, obj_index, exact_eval
        # lam_lb is certified (_hessian_error) from probes of `certificate`'s first-order jets, or the exact evaluator;
        # without either, a certified tangent keeps the legacy 1e-9 margin.
        self.hess_err = None         # >= ||H_exact - H_hat||_F over the free coordinates (None: not certified)
        self.f_qtb, self.g_qtb = None, None
        self.linear_constraint_outputs = False
        xh = x_hat.detach().reshape(-1).to(torch.float64)
        self.fixed_mask = torch.zeros(n, dtype=torch.bool, device=xh.device)
        self.fixed_values = torch.zeros_like(xh)
        if root_lower is not None or root_upper is not None:
            if root_lower is None or root_upper is None:
                raise ValueError("Both root box endpoints are required")
            lower, upper = root_lower.reshape(-1).to(xh), root_upper.reshape(-1).to(xh)
            if lower.numel() != n or upper.numel() != n or (lower > upper).any():
                raise ValueError("Invalid root box for quadratic tangent")
            self.fixed_mask = (lower == upper) & torch.isfinite(lower)
            self.fixed_values = torch.where(self.fixed_mask, lower, 0.)
            xh = torch.where(self.fixed_mask, self.fixed_values, xh)
        H = torch.autograd.functional.hessian(lambda z: objective_fn(z), xh)
        H = (H + H.T) / 2
        # Descendant boxes only permit displacements in the free coordinates.
        free = ~self.fixed_mask
        H_free = H[free][:, free]
        if H_free.numel():
            lam = float(torch.linalg.eigvalsh(H_free).min())
            self.lam_min = lam
            norm = float(torch.linalg.matrix_norm(H_free))
            self.hess_err = self._hessian_error(H_free, xh) if (certificate is not None or exact_hessian is not None) \
                else None
            if self.hess_err is not None:
                # lambda_min(H) >= lambda_hat_min - ||H - H_hat||_2 (Weyl), and eigvalsh returns the eigenvalues of
                # H_hat + E with ||E||_2 <= p(n) u ||H_hat||_F, p(n) <= 16 n (LAPACK syevd: p(n) is a low-degree
                # polynomial; 16 n leaves a wide margin). Both subtractions are rounded down.
                backward = 16 * max(n, 1) * (U * frobenius(H_free) * (1 + 1e-9) + ETA)
                self.lam_lb = math.nextafter(math.nextafter(lam - self.hess_err, -INF) - backward, -INF)
            else:
                # eigvalsh backward error and the rounding of H are far below this (uncertified).
                self.lam_lb = lam - 1e-9 * (1.0 + norm)
        else:
            self.lam_min = 0.0
            self.lam_lb = 0.0  # Every permitted displacement is zero.
        self.H_free = H_free         # the matrix hess_err certifies; consumers must not recompute it
        self.x_hat = None
        self.polished = False
        self.update(xh)

    def _hessian_error(self, H_free: Tensor, xh: Tensor) -> Optional[float]:
        """A float >= ||H - H_hat||_F, H the exact (constant) Hessian of the graph on the free coordinates, H_hat the
        autograd matrix handed to eigvalsh; None when neither source applies, inf when both are unusable.

        Source 1 (_probe_error): f is quadratic, so H e_j delta = grad f(x_hat + delta e_j) - grad f(x_hat) exactly, and the
        first-order point jets enclose both gradients. O(n) jets, evaluated in chunks: memory O(chunk n size).
        Source 2 (n <= 64, only when the jet is looser than the legacy margin): the exact rational gradient at x_hat
        and at x_hat + e_j gives H e_j = (grad(x_hat + e_j) - grad(x_hat)) / delta_j exactly, because f is quadratic.
        Both bounds are rigorous, so the smaller is used. The norm itself is summed in floats: inflate by
        (1 + gamma_{n^2 + 8})."""
        free = (~self.fixed_mask).nonzero().flatten()
        k = free.numel()
        best = None
        if self.exact_hessian is not None:
            best = self._sparse_error(H_free, free)
            if best is not None:
                return best
        if self.qtb_certificate is not None:
            try:
                best = self._probe_error(H_free, xh, free)
            except Exception:  # noqa: BLE001 - unsupported jet: try the exact path
                best = INF
        threshold = 1e-9 * (1.0 + float(torch.linalg.matrix_norm(H_free)))
        if (best is None or best > threshold) and self.exact_eval is not None:
            from fractions import Fraction
            try:
                g0 = self.exact_eval(xh)
                total = Fraction(0)
                for a, j in enumerate(free.tolist()):
                    if time.time() >= self.deadline:
                        return INF
                    p = xh.clone()
                    p[j] = p[j] + 1.0
                    gj = self.exact_eval(p)
                    if g0 is None or gj is None:
                        raise ValueError
                    delta = Fraction(float(p[j])) - Fraction(float(xh[j]))
                    for b, i in enumerate(free.tolist()):
                        total += (Fraction(float(H_free[b, a])) - (gj[1][i] - g0[1][i]) / delta) ** 2
                # total may be below 2^-1075 (float() would give 0): scale by 4^e, sqrt, undo with the exact 2^-e
                e = 0 if total == 0 else -((total.numerator.bit_length() - total.denominator.bit_length()) // 2)
                exact = math.ldexp(math.sqrt(float(total * Fraction(4) ** e)), -e) * (1 + gamma(k * k + 8)) * (1 + 4 * U) * (1 + 4 * U)
                best = exact if best is None else min(best, exact)
            except (ValueError, OverflowError, ZeroDivisionError):
                pass
        return best

    def _sparse_error(self, H_free: Tensor, free: Tensor) -> Optional[float]:
        """O(nnz) bound >= ||H - H_hat||_F from the exact Hessian entries (exact_hessian = (I, J, E): the triplets of the
        upper triangle, E the correctly rounded binary64 of the exact rational entry e; None when there are none).

        Why each entry is known: f is a degree <= 2 polynomial with exact rational coefficients (the structure walk, no
        inexact atom), so H is the constant matrix with H_ij = c_ij for the monomial x_i x_j (i < j) and H_ii = 2 c_ii.
        Rounding: E = fl(e) has |E - e| <= U |E| (1 + 2U) + ETA (round to nearest; 2c is exact in floating point). With
        d = |E - hat| (1 + 4U) + 2 U |E| (1 + 4U) + 2 ETA, then (1 + 4U), |H_ij - hat_ij| <= d_ij: the first term covers
        the rounding of the subtraction, the second the rounding of E. Absent entries have e = 0 exactly and give |hat|.
        The Frobenius sum of the k^2 squares is inflated by (1 + gamma(k^2 + 8)) (1 + 4U) as in _probe_error, and
        ||.||_2 <= ||.||_F. An unproved or non-quadratic form gives no triplets and falls back to the jets."""
        I, J, E = self.exact_hessian
        k = free.numel()
        pos = torch.full((self.n,), -1, dtype=torch.long)
        pos[free.cpu()] = torch.arange(k)
        pi, pj = pos[I], pos[J]
        keep = (pi >= 0) & (pj >= 0)
        pi, pj, E = pi[keep], pj[keep], E[keep]
        M = torch.zeros(k, k, dtype=torch.float64)
        M[pi, pj] = E
        M[pj, pi] = E                       # the diagonal is written twice with one value
        hat = H_free.detach().cpu().to(torch.float64)
        d = (M - hat).abs() * (1 + 4 * U) + M.abs() * (2 * U * (1 + 4 * U)) + 2 * ETA
        d = d * (1 + 4 * U)
        out = _scaled_sqrt_sumsq(d) * (1 + gamma(k * k + 8)) * (1 + 4 * U)
        return out if math.isfinite(out) else None

    def _probe_error(self, H_free: Tensor, xh: Tensor, free: Tensor, chunk: int = 8) -> float:
        """||H - H_hat||_F bound (before the final norm inflation) from first-order jets at x_hat and x_hat + e_j.

        delta_j = fl(x_hat_j + 1) - x_hat_j is known to a relative 2u (a rounded difference of two floats), so
        H_ij * delta_j lies in (jet(P_j)_i - jet(x_hat)_i) and H_ij in that interval divided by [delta_lo, delta_hi]."""
        from .interval_hessian import add, neg
        k, n = self.obj_index, xh.numel()
        x0 = xh[None]
        j0 = self.qtb_certificate.evaluate(x0, x0)
        if not bool(j0.valid[0, k]):
            return INF
        g0 = tuple(t[0, k][free] for t in j0.gradient)
        total = (0.0, 1.0)             # (sum of squares / scale^2, scale): no underflow of the squares
        t_start = time.time()
        for start in range(0, free.numel(), chunk):
            if start:
                # After the first chunk, extrapolate: a projected finish past the deadline gives up (INF: no plane).
                done = start // chunk
                left = -(-(free.numel() - start) // chunk)
                now = time.time()
                if now + (now - t_start) / done * left >= self.deadline:
                    return INF
            if time.time() >= self.deadline:
                return INF
            cols = free[start:start + chunk]
            P = xh.repeat(len(cols), 1)
            P[torch.arange(len(cols)), cols] += 1.0
            d = P[torch.arange(len(cols)), cols] - xh[cols]
            if bool((d < 0.5).any()):
                return INF
            dlo, dhi = d * (1 - 4 * U), d * (1 + 4 * U)
            jet = self.qtb_certificate.evaluate(P, P)
            if not bool(jet.valid[:, k].all()):
                return INF
            lo, hi = add(tuple(t[:, k][:, free] for t in jet.gradient), neg(g0))        # (chunk, free) = H[:, cols] delta
            Hlo = torch.minimum(lo / dlo[:, None], lo / dhi[:, None])
            Hhi = torch.maximum(hi / dlo[:, None], hi / dhi[:, None])
            Hlo, Hhi = Hlo - 4 * U * Hlo.abs(), Hhi + 4 * U * Hhi.abs()                   # the divisions round to nearest
            hat = H_free[:, start:start + len(cols)].T                                    # (chunk, free)
            dev = torch.maximum((Hlo - hat).abs(), (Hhi - hat).abs())
            total = _merge_sumsq(total, dev)
        k_ = free.numel()
        out = math.sqrt(total[0]) * total[1] * (1 + gamma(k_ * k_ + 8)) * (1 + 4 * U)
        return out if math.isfinite(out) else INF

    def polish(self, x_kkt: Tensor) -> None:
        """Use a local KKT point as the tangent anchor.

        For a convex objective over that (linear) feasible set, g^T (x - x_kkt) >= 0 at
        every feasible x, so the plane bounds every sub-box by f(x_kkt) at once; at any
        other point the plane is tilted and loose across the set. Only lock this anchor
        when the objective is convex on the free subspace and all constraint outputs
        are affine; otherwise a local KKT point need not minimize the LP relaxation.
        """
        self.update(x_kkt)
        self.polished = self.linear_constraint_outputs and self.lam_lb >= 0

    def offer(self, x_hat: Tensor, margin: float) -> None:
        """Move the tangent to a new incumbent, unless the current point is polished and
        the incumbent does not beat it by more than `margin`."""
        if self.polished:
            xh = x_hat.detach().reshape(-1).to(torch.float64)
            xh = torch.where(self.fixed_mask.to(xh.device), self.fixed_values.to(xh), xh)
            f = float(self.objective_fn(xh))
            if not f < self.f - margin:
                return
            self.polished = False
        self.update(x_hat)

    def update(self, x_hat: Tensor) -> None:
        xh = x_hat.detach().reshape(-1).to(torch.float64)
        xh = torch.where(self.fixed_mask.to(xh.device), self.fixed_values.to(xh), xh)
        xh = xh.clone().requires_grad_(True)
        f = self.objective_fn(xh)
        g, = torch.autograd.grad(f, xh)
        self.x_hat = xh.detach()
        self.f = float(f.detach())
        self.g = g.detach()
        if self.qtb_certificate is not None:
            self._point_errors()

    def _point_errors(self):
        """f_error >= |f_computed - f(x_hat)|, g_error >= |g_computed - grad f(x_hat)| (exact reals, x_hat as stored).

        The point jet is outward-rounded interval AD of the same graph, so it encloses the exact value and gradient at the
        binary64 x_hat; the computed f and g lie within the enclosure's distance. A non-finite or invalid jet gives inf.
        """
        x = self.x_hat[None]
        jet = self.qtb_certificate.evaluate(x, x)
        k = self.obj_index
        ok = bool(jet.valid[:, k].all()) and math.isfinite(self.f) and bool(torch.isfinite(self.g).all())
        f_err = max(abs(self.f - float(t[0, k])) for t in jet.value) if ok else INF
        g_err = torch.maximum(*(abs(self.g - t[0, k]) for t in jet.gradient)) if ok else torch.full_like(self.g, INF)
        if not (math.isfinite(f_err) and bool(torch.isfinite(g_err).all())):
            f_err, g_err = INF, torch.full_like(self.g, INF)
        if self.exact_eval is not None and (f_err > 1e-9 * (1. + abs(self.f)) or bool((g_err > 1e-9 * (1. + self.g.abs())).any())):
            # Interval AD pays the dependency problem on long recursions (schitt_s332: 4e81 for a true error of 2e-15);
            # for a small proved-quadratic graph the exact rational value and gradient are cheap, and the error of the
            # computed f, g against them is then exact. Both bounds are rigorous: keep the smaller.
            from fractions import Fraction
            exact = self.exact_eval(self.x_hat)
            if exact is not None:
                up = 1. + 4 * 2.0 ** -52          # float() of a Fraction rounds to nearest: round the error up
                f_err = min(f_err, float(abs(Fraction(self.f) - exact[0])) * up)
                g_exact = torch.tensor([float(abs(Fraction(float(gj)) - ge)) * up for gj, ge in zip(self.g, exact[1])],
                                       dtype=self.g.dtype)
                g_err = torch.minimum(g_err, g_exact.to(g_err))
        self.f_qtb, self.g_qtb = f_err, g_err

    def plane(self, l: Tensor, u: Tensor, lam_lb=None):
        """(a (B, n), b (B,)) with a^T x + b <= f(x) for every x in [l, u]."""
        xh = self.x_hat.to(l)
        g = self.g.to(l)
        fixed = self.fixed_mask.to(l.device)
        values = self.fixed_values.to(l)
        if not ((l[:, fixed] == values[fixed]).all()
                and (u[:, fixed] == values[fixed]).all()):
            raise ValueError("Quadratic tangent used outside its fixed-coordinate subspace")
        a = g.unsqueeze(0).expand(l.shape[0], -1)
        const = self.f - float((g * xh).sum())
        cushion = 1e-9 * (1.0 + abs(self.f)
                          + (g.abs().unsqueeze(0) * torch.maximum(l.abs(), u.abs())).sum(-1)
                          + float((g * xh).abs().sum()))
        b = torch.full((l.shape[0],), const, dtype=l.dtype, device=l.device) - cushion
        if self.f_qtb is not None:
            # f(x) >= f(x_hat) + grad f(x_hat).d + lam_lb/2 |d|^2 with f, grad f exact; the computed f - f_error and
            # g -/+ g_error |d| are valid lower values. The relative cushion above now only covers the rounding of
            # const and of the consumer's a^T x + b (backward-stable summation), not an evaluation error of f.
            # lam_lb is certified by _hessian_error (interval jet or exact rationals) plus the eigvalsh backward error.
            from .interval_hessian import outward
            dist = torch.maximum((xh - l).abs(), (u - xh).abs())
            total = (self.g_qtb.to(l).unsqueeze(0) * dist).sum(-1) + self.f_qtb
            total = torch.where(torch.isnan(total), torch.full_like(total, INF), total)
            b = b - outward(total, total, total, self.n + 2)[1]
        if isinstance(lam_lb, Tensor):
            # Interval-Hessian Taylor bounds supply one curvature per domain.
            # Round the squared distance upwards before multiplying by mu <= 0.
            from .interval_hessian import outward, add, mul
            displacement = add((xh, xh), (-u, -l))
            distance = torch.maximum(displacement[0].abs(), displacement[1].abs())
            squared = mul((distance, distance), (distance, distance))[1][:, ~fixed]
            far = outward(squared.sum(-1), squared.sum(-1), squared.sum(-1), self.n)[1]
            mu = lam_lb.to(l).clamp(max=0)
            correction = mul((mu, mu), (far, far))[0] * 0.5
            b = add((b, b), (correction, correction))[0]
            b = torch.where(torch.isfinite(lam_lb) & torch.isfinite(b), b, -torch.inf)
        elif not self.lam_lb >= 0:     # a NaN curvature (kriging_peaks-full030) gives no plane
            far = torch.maximum((xh - l) ** 2, (u - xh) ** 2)[:, ~fixed].sum(-1)
            b = b + 0.5 * self.lam_lb * far
            b = torch.where(torch.isfinite(b), b, -torch.inf)
        return a, b

    @torch.no_grad()
    def quadratic_dual_bound(self, l: Tensor, u: Tensor, G: Tensor,
                             h: Tensor, y: Tensor) -> Optional[Tensor]:
        """Box minimum of the quadratic lower model plus y^T (G x - h).

        For lam_lb > 0, retaining curvature gives a separable box minimization.
        Weak duality holds for any nonnegative multipliers; also try y = 0.
        """
        if not math.isfinite(self.lam_lb) or self.lam_lb <= 0:
            return None
        dtype, f64 = l.dtype, torch.float64      # exact for fp32 data
        l, u, G, h, y = (t.to(f64) for t in (l, u, G, h, y))
        a, b = self.plane(l, u)
        xh = self.x_hat.to(l)
        lam = self.lam_lb
        active = torch.isfinite(h) & torch.isfinite(G).all(-1)
        G = torch.where(active.unsqueeze(-1), G, 0.)
        h = torch.where(active, h, 0.)
        y = torch.where(active & torch.isfinite(y), y.clamp(min=0), 0.)
        rows = Rows.from_dense(G)

        def evaluate(multipliers):
            q = a + torch.einsum('bmn,bm->bn', G, multipliers)
            point = torch.minimum(torch.maximum(xh - q / lam, l), u)
            # For every slope s, lam/2 (x - xh)^2 >= s (x - xh) - s^2 / (2 lam), with equality at xh + s / lam.
            # The slope at `point` makes the box minimum of this linear model exact; any s would be valid.
            s = lam * (point - xh)
            t_lin, t_sq = s * xh, s.square() / (2 * lam)
            c, c0 = a + s, b - (t_lin + t_sq).sum(-1)
            # the rounding of c and c0, in the evaluator's terms (rigorous_dual): one rounding per c_j, n + 4 per c0
            lb, _, _ = dual_lower_bound(rows, c, c0, h, l, u, multipliers, c_err=2 * U * c.abs(),
                                        c0_err=gamma(l.shape[-1] + 6) * (b.abs() + t_lin.abs().sum(-1) + t_sq.sum(-1)))
            return lb

        return round_down(torch.maximum(evaluate(y), evaluate(torch.zeros_like(y))), dtype)


def _exact_evaluator(bounded_module, x_input, obj_index, max_vars: int = 64, max_terms: int = 400_000):
    """x_hat -> (f(x_hat), grad f(x_hat)) in exact rationals through the exact affine/product forms of polynomial_graph;
    None when the graph is not fully folded, or larger than max_vars inputs / max_terms product-column terms."""
    n = x_input.numel()
    if n > max_vars:
        return None
    from fractions import Fraction as F
    from .polynomial_graph import _Fold
    try:
        with torch.no_grad():
            bounded_module(x_input)
            fold = _Fold(bounded_module)
        form = fold.forms[bounded_module.final_name].reshape(-1)[obj_index]
        if form.t is None or len(fold.cols) * n > max_terms:
            return None
    except Exception:  # noqa: BLE001 - unsupported operator: no exact evaluator
        return None

    def evaluate(aff, vals, grads):
        v, g = F(aff.k), [F(0)] * n
        for c, t in aff.t.items():
            v += t * vals[c]
            g = [gj + t * gc for gj, gc in zip(g, grads[c])]
        return v, g

    def exact(x_hat):
        x = [F(float(t)) for t in x_hat.reshape(-1)]
        vals, grads = [], []
        for c, col in enumerate(fold.cols):
            if col[0] == 'input':
                vals.append(x[col[1]])
                grads.append([F(int(i == col[1])) for i in range(n)])
                continue
            a, b = fold.operands.get(c, (None, None))
            if a is None or a.t is None or b.t is None:
                return None
            (va, ga), (vb, gb) = evaluate(a, vals, grads), evaluate(b, vals, grads)
            vals.append(va * vb)
            grads.append([p * vb + va * q for p, q in zip(ga, gb)])
        return evaluate(form, vals, grads)

    return exact


def _exact_hessian(model, x_input: Tensor, obj_index: int, root_lower: Optional[Tensor]):
    """(I, J, E) of the upper triangle of the exact constant Hessian of output obj_index, from the structure walk's exact
    polynomial; None unless the walk gives an exact (not inexact) polynomial in the inputs alone of degree <= 2.
    The walk's positivity rules use the root box's lower bound, so its form holds on the whole root box."""
    from fractions import Fraction
    n = int(x_input.numel())
    try:
        from .sparse_derivatives import output_walk
        lower = (root_lower.reshape(-1).double().cpu().numpy() if root_lower is not None
                 else __import__("numpy").full(n, -INF))
        forms, _ = output_walk(model, x_input.detach().reshape(-1).double().cpu().numpy(), lower, tuple(x_input.shape[1:]))
        form = forms[obj_index]
        if form.poly is None or form.inexact:
            return None
        I, J, E = [], [], []
        for mono, c in form.poly.items():
            if len(mono) == 2 and mono[0][1] == 1 and mono[1][1] == 1:
                i, j, h = mono[0][0], mono[1][0], Fraction(c)
            elif len(mono) == 1 and mono[0][1] == 2:
                i, j, h = mono[0][0], mono[0][0], 2 * Fraction(c)
            elif not mono or (len(mono) == 1 and mono[0][1] == 1 and mono[0][0] < n):
                continue                      # constant or linear term: no curvature
            else:
                return None                   # an atom or a higher degree: not a proved quadratic
            if not (0 <= i < n and 0 <= j < n):
                return None
            I.append(i), J.append(j), E.append(float(h))
        return (torch.tensor(I, dtype=torch.long), torch.tensor(J, dtype=torch.long),
                torch.tensor(E, dtype=torch.float64))
    except Exception:  # noqa: BLE001 - the walk cannot follow the graph: fall back to the jets
        return None


def build_quadratic_tangent(bounded_module, model_ori, x_input: Tensor, obj_index: int,
                            x_hat: Optional[Tensor], max_vars: int = 2000,
                            root_lower: Optional[Tensor] = None,
                            root_upper: Optional[Tensor] = None,
                            taylor2_bound: bool = False,
                            taylor2_max_vars: int = 40,
                            taylor2_max_domains: int = 1024,
                            quadratic_bound: bool = True,
                            deadline: float = INF,
                            ) -> Optional[QuadraticTangent]:
    """Build QTB, or the opt-in interval-Hessian tangent if the degree proof fails."""
    n = int(x_input.reshape(-1).numel())
    if x_hat is None or n > max_vars:
        return None
    try:
        degs = output_degrees(bounded_module, x_input)
    except Exception:  # noqa: BLE001
        if not taylor2_bound or n > taylor2_max_vars:
            return None
        # Failed structural proofs are not evidence of a quadratic objective.
        degs = torch.full((model_ori(x_input).numel(),), INF, dtype=torch.float64)
    if obj_index >= degs.numel() or not math.isfinite(float(degs[obj_index])) \
            or float(degs[obj_index]) > 2:
        if taylor2_bound and n <= taylor2_max_vars and obj_index < degs.numel():
            from auto_LiRPA.not_in_release import not_in_release
            not_in_release('solver.nlp.taylor2_bound (second-order Taylor tangent)')
        return None
    if not quadratic_bound:
        return None
    shape = x_input.shape

    # The rounding certificate bounds binary64 evaluation of the graph, so f and its gradient must be binary64 too.
    model_eval = model_ori if x_input.dtype == torch.float64 else __import__("copy").deepcopy(model_ori).double()

    def objective_fn(z):
        return model_eval(z.reshape(shape).to(torch.float64)).reshape(-1)[obj_index].to(torch.float64)

    from .convex_tangent import _certificate
    try:
        certificate = _certificate(bounded_module, x_input.double(), max_affine=40_000_000)
    except Exception as exc:  # noqa: BLE001 - unsupported operator: no certified plane
        print(f"[QTB] rounding certificate unavailable ({type(exc).__name__}: {exc}); QTB disabled")
        return None
    exact_eval = _exact_evaluator(bounded_module, x_input, obj_index)
    exact_hessian = _exact_hessian(model_ori, x_input, obj_index, root_lower)
    tangent = QuadraticTangent(objective_fn, n, x_hat, root_lower, root_upper, certificate, obj_index, exact_eval,
                               exact_hessian, deadline)
    constraint_degrees = torch.cat((degs[:obj_index], degs[obj_index + 1:]))
    tangent.linear_constraint_outputs = bool((constraint_degrees <= 1).all())
    return tangent
