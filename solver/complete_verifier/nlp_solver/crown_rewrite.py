"""Native CROWN bounds for the two structures GEAR used to bound with dedicated code (theory/crown_merge/REPORT.md).

Both rewrite a proved structure into a small graph whose convexity CROWN's own relaxations already exploit:

* Quadratic objective (QTB's degree proof, solver.nlp.qtb_crown_rewrite). At the anchor x^,
      f(x) = f(x^) + g.(x - x^) + 1/2 (x - x^)^T H (x - x^)   exactly, H = V diag(lam) V^T + E,
  so f = f(x^) + g.d + 1/2 sum_k lam_k z_k^2 + 1/2 d^T E d with z = V^T d. CROWN bounds each z_k^2 by its tangent
  (lam_k > 0; at z = 0 when the box holds x^, i.e. QTB's plane) or its chord (lam_k < 0), per box; the residual is
  bounded by 1/2 ||E||_2 max ||d||^2 on the box.
* Posynomials (the GP certificate, solver.nlp.gp_crown). In y = log x each monomial is exp(a.y + log c), convex; CROWN's
  exp tangents at a point give a global lower plane of every posynomial. CROWN proposes the slope; the intercept is
  re-derived with outward intervals (the certificate's own standard), so a rounded slope cannot invalidate it.
"""
from fractions import Fraction
import math

import numpy as np
import torch

from nlp_solver.rigorous_dual import U, gamma
from nlp_solver.quadratic_tangent import frobenius


class _Quadratic(torch.nn.Module):
    """f(x^) + g.(x - x^) + 1/2 sum_k lam_k (P (x - x^))_k^2, P = V^T on the free coordinates."""

    def __init__(self, xh, g, f, P, lam):
        super().__init__()
        self.z = torch.nn.Linear(P.shape[1], P.shape[0]).double()
        self.z.weight.data, self.z.bias.data = P, -(P @ xh)
        self.lin = torch.nn.Linear(P.shape[1], 1).double()
        self.lin.weight.data = g.reshape(1, -1).clone()
        self.lin.bias.data = torch.tensor([f - float(g @ xh)], dtype=torch.float64, device=xh.device)
        self.w = torch.nn.Linear(P.shape[0], 1, bias=False).double()
        self.w.weight.data = (0.5 * lam).reshape(1, -1).clone()

    def forward(self, x):
        return self.lin(x) + self.w(self.z(x) ** 2)


class QuadraticRewrite:
    """CROWN on the eigenbasis rewrite of a QuadraticTangent's objective, rebuilt when its anchor moves."""

    def __init__(self, qtb, device):
        from auto_LiRPA import BoundedModule
        self.qtb = qtb
        xh = qtb.x_hat.detach().to(device=device, dtype=torch.float64)
        self.anchor = qtb.x_hat.detach().clone()
        H = torch.autograd.functional.hessian(lambda z: qtb.objective_fn(z.to(qtb.x_hat)).to(torch.float64), xh)
        H = (H + H.T) / 2
        self.free = (~qtb.fixed_mask.to(device)).nonzero().flatten()
        Hf = H[self.free][:, self.free]
        if qtb.hess_err is not None and getattr(qtb, 'H_free', None) is not None \
                and qtb.H_free.shape == Hf.shape:
            Hf = qtb.H_free.to(Hf)       # hess_err certifies this matrix; a recomputed one may differ in rounding
        lam, V = torch.linalg.eigh(Hf)
        residual = float(torch.linalg.matrix_norm(Hf - V @ torch.diag(lam) @ V.T, ord=2))
        if qtb.hess_err is not None:
            # H = H_hat + (H - H_hat) and H_hat = V diag(lam) V^T + R, so ||H - V diag(lam) V^T||_2 <= ||R||_2 +
            # ||H - H_hat||_F (certified, QuadraticTangent._hessian_error). R and its norm are computed in floats:
            # R by two matrix products, |V||L||V|^T has 2-norm <= n ||L||, and the 2-norm (an SVD) has backward error
            # <= 16 n u ||R||_F-scale (as the eigvalsh bound in QuadraticTangent).
            n = max(len(self.free), 1)
            fro = frobenius(Hf)
            numerics = gamma(n + 4) * (fro + n * float(lam.abs().max())) + 16 * n * U * fro
            self.e_norm = (residual + qtb.hess_err + numerics) * (1 + 1e-9)
        else:
            # decomposition residual plus the Hessian's own rounding (QTB's cushion for the same Hessian), uncertified
            self.e_norm = residual + 1e-9 * (1.0 + float(torch.linalg.matrix_norm(Hf)))
        P = torch.zeros(len(self.free), xh.numel(), dtype=torch.float64, device=device)
        P[:, self.free] = V.T
        self.g, self.f, self.xh = qtb.g.detach().to(xh), float(qtb.f), xh
        self.lam = lam
        self.model = BoundedModule(_Quadratic(xh, self.g, self.f, P, lam), xh.reshape(1, -1),
                                   bound_opts={'conv_mode': 'matrix', 'pow': {'rounding_eps': True}}, device=device)
        print(f"[QTB] native CROWN eigenbasis rewrite: free={len(self.free)} lambda=[{float(lam.min()):.4g}, "
              f"{float(lam.max()):.4g}] negative={int((lam < 0).sum())} residual={self.e_norm:.3g} f(anchor)={self.f:.10g}")

    @torch.no_grad()
    def plane(self, l, u):
        """(a (B, n), b (B,)) with a^T x + b <= f(x) for every x in [l, u] (like QuadraticTangent.plane)."""
        from auto_LiRPA import BoundedTensor, PerturbationLpNorm
        dtype = l.dtype
        l64, u64 = l.to(self.xh), u.to(self.xh)
        x0 = torch.minimum(torch.maximum(self.xh.reshape(1, -1), l64), u64)
        x = BoundedTensor(x0, PerturbationLpNorm(x_L=l64, x_U=u64))
        out, inp = self.model.output_name[0], self.model.input_name[0]
        _, _, A = self.model.compute_bounds(x=(x,), method='CROWN', bound_upper=False, return_A=True,
                                            needed_A_dict={out: [inp]})
        a = A[out][inp]['lA'].reshape(l.shape[0], -1)
        b = A[out][inp]['lbias'].reshape(-1)
        d = torch.maximum((self.xh - l64).abs(), (u64 - self.xh).abs())[:, self.free]
        b = b - 0.5 * self.e_norm * d.square().sum(-1)
        # rounding of the plane's evaluation, as QuadraticTangent.plane
        b = b - 1e-9 * (1.0 + abs(self.f) + (self.g.abs() * torch.maximum(l64.abs(), u64.abs())).sum(-1)
                        + float((self.g * self.xh).abs().sum()))
        if self.qtb.f_qtb is not None:
            # certified evaluation error of f(x^), grad f(x^) (QuadraticTangent._point_errors), as QuadraticTangent.plane
            from nlp_solver.interval_hessian import outward
            dist = torch.maximum((self.xh - l64).abs(), (u64 - self.xh).abs())
            total = (self.qtb.g_qtb.to(l64).unsqueeze(0) * dist).sum(-1) + self.qtb.f_qtb
            total = torch.where(torch.isnan(total), torch.full_like(total, math.inf), total)
            b = b - outward(total, total, total, self.xh.numel() + 2)[1]
        b = torch.where(torch.isfinite(b) & torch.isfinite(a).all(-1), b, -torch.inf)
        a, b = a.to(dtype), b.to(dtype)
        return a, torch.nextafter(b, torch.full_like(b, -torch.inf))


def quadratic_rewrite_plane(history, qtb, l, u, max_vars):
    """CROWN-rewrite plane for a QuadraticTangent, cached on `history`; None when it does not apply."""
    from nlp_solver.quadratic_tangent import QuadraticTangent
    if type(qtb) is not QuadraticTangent or int((~qtb.fixed_mask).sum()) > max_vars:
        return None
    rw = getattr(history, 'qtb_rewrite', None)
    if rw is None or rw.qtb is not qtb or not torch.equal(rw.anchor, qtb.x_hat):
        rw = QuadraticRewrite(qtb, l.device)
        history.qtb_rewrite = rw
    return rw.plane(l, u)


class _Posynomials(torch.nn.Module):
    """sum_k exp(a_k . y + b_k) for each posynomial (one output per posynomial)."""

    def __init__(self, outputs):
        super().__init__()
        self.lin, self.sums = torch.nn.ModuleList(), torch.nn.ModuleList()
        for A, b in outputs:
            lin = torch.nn.Linear(A.shape[1], A.shape[0]).double()
            lin.weight.data, lin.bias.data = torch.tensor(A), torch.tensor(b)
            s = torch.nn.Linear(A.shape[0], 1, bias=False).double()
            s.weight.data = torch.ones(1, A.shape[0], dtype=torch.float64)
            self.lin.append(lin)
            self.sums.append(s)

    def forward(self, y):
        return torch.cat([s(torch.exp(lin(y))) for lin, s in zip(self.lin, self.sums)], dim=-1)


class PosynomialPlanes:
    """CROWN lower planes, in log coordinates, of a list of posynomials (gp_certificate Forms) on a log box."""

    def __init__(self, forms, lo, hi):
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
        from auto_LiRPA.operators import BoundExp
        from nlp_solver.gp_certificate import _log_data
        self.data = [_log_data(f) for f in forms]
        self.lo, self.hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        outputs = [(np.array([[float(a) for a in exps] for exps, _ in d], dtype=np.float64),
                    np.array([float(logc.mid) for _, logc in d], dtype=np.float64)) for d in self.data]
        y0 = torch.tensor((self.lo + self.hi) / 2).reshape(1, -1)
        self.model = BoundedModule(_Posynomials(outputs), y0,
                                   bound_opts={'conv_mode': 'matrix', 'exp': {'free_tangent': True}}, device='cpu')
        self.x = BoundedTensor(y0, PerturbationLpNorm(x_L=torch.tensor(self.lo).reshape(1, -1),
                                                      x_U=torch.tensor(self.hi).reshape(1, -1)))
        self.exps = [m for m in self.model.modules() if isinstance(m, BoundExp)]
        self.model.init_alpha((self.x,), share_alphas=False)

    @torch.no_grad()
    def planes(self, point):
        """[(slope, intercept)] per posynomial: a global lower plane of it on the log box, with CROWN's exp tangents
        at `point`; the intercept is certified with outward intervals for the rounded slope."""
        from nlp_solver.gp_certificate import _IV, _interval, _down
        self.model(torch.tensor(point).reshape(1, -1))      # each exp's input value at the point = its tangent point
        for e in self.exps:
            z = e.inputs[0].forward_value.detach().reshape(-1)
            for key in e.alpha:
                e.alpha[key].data.copy_(z.to(e.alpha[key]).expand_as(e.alpha[key].data))
        out, inp = self.model.output_name[0], self.model.input_name[0]
        _, _, A = self.model.compute_bounds(x=(self.x,), method='CROWN', bound_upper=False, return_A=True,
                                            needed_A_dict={out: [inp]}, reuse_alpha=True)
        slopes = A[out][inp]['lA'].reshape(len(self.data), -1).double().numpy()
        result = []
        box = [_IV.mpf([float(l), float(h)]) for l, h in zip(self.lo, self.hi)]
        for d, s in zip(self.data, slopes):
            # sum_k w_k (1 + a_k.y + log c_k - m_k), w_k = exp(m_k), m_k the tangent point clamped into z_k's range,
            # minus the rounded slope s over the box.
            intercept = _IV.mpf(0)
            exact = [_IV.mpf(0)] * len(s)
            for exps, logc in d:
                zr = logc + sum((_interval(Fraction(a)) * box[j] for j, a in enumerate(exps) if a), _IV.mpf(0))
                m = min(max(float(logc.mid) + float(np.dot([float(a) for a in exps], point)), float(zr.a)), float(zr.b))
                w = _IV.exp(_IV.mpf(m))
                intercept += w * (1 + logc - m)
                for j, a in enumerate(exps):
                    if a:
                        exact[j] = exact[j] + w * _interval(Fraction(a))
            for j in range(len(s)):
                intercept += (exact[j] - _interval(Fraction(float(s[j])))) * box[j]
            result.append((s, _down(intercept)))
        return result
