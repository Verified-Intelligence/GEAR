"""Batched SQP polish of ALM candidates: a GPU-native local optimality phase.

ALM-PGD reaches the feasible set late, at a large penalty, where its inner
Adam steps barely see the objective; it then stops at the first feasible
point.  Its restarts therefore end feasible (or nearly so) but not locally
optimal.  This module runs a batched SQP from a few distinct candidates,
every candidate an independent problem:

* derivatives: exact objective gradient and constraint Jacobian from one
  vectorized reverse pass; a damped-BFGS Hessian of the Lagrangian per
  candidate (always positive definite);
* step: the l1-elastic QP subproblem (always feasible) solved by a batched
  primal-dual interior point method whose Newton systems reduce to one n x n
  Cholesky per iteration;
* globalization: an l1 exact-penalty merit function with a steering penalty
  update, Armijo backtracking and one second-order correction;
* variables live in unit-box coordinates and every trial point is clamped to
  the box, so the box is never left.

Constraints are driven to ``g <= target`` with ``target = target_frac * tol``
(the primal may use the feasibility tolerance), and a point is only reported
when its raw violation ``max(g, 0)`` is ``<= tol``.  Pure torch; no CPU solver.
"""

import time
from typing import Optional, Tuple

import torch
from auto_LiRPA.cuda_memory import capture
from .utils import check_deadline, DeadlineExceeded
from .continuation import Atom, candidate, retained


def _is_shared(mat: torch.Tensor) -> bool:
    """True when every domain slice of a (B, ...) tensor is identical."""
    if mat.shape[0] <= 1 or mat.stride(0) == 0:
        return True
    return bool(torch.equal(mat, mat[:1].expand_as(mat)))


def _mv(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return torch.bmm(mat, vec.unsqueeze(-1)).squeeze(-1)


class CandidateEvaluator:
    """Objective and raw constraint residuals of K candidates, each in its own domain."""

    def __init__(self, problem, dom: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor):
        self.problem = problem
        self.K = int(dom.numel())
        self.lb = lb
        self.width = ub - lb
        self.c_obj = problem.C_obj[dom, 0]                      # (K, O)
        self.rhs_obj = problem.rhs_obj.reshape(problem.batch_size, -1)[dom, 0]
        self.n_out = int(problem.n_output_constraints)
        self.n_in = int(problem.n_input_constraints)
        if self.n_out > 0:
            c_constr = problem.C_constr
            if c_constr.ndim == 2:
                c_constr = c_constr.unsqueeze(0)
            self.c_shared = c_constr[0] if _is_shared(c_constr) else None
            self.c_gather = None if self.c_shared is not None else c_constr[dom]
            rhs = problem.rhs_constr
            if rhs.ndim == 1:
                rhs = rhs.unsqueeze(0)
            self.rhs_constr = rhs.expand(problem.batch_size, -1)[dom]
        if self.n_in > 0:
            a = problem.input_constr_A
            self.a_shared = a[0] if _is_shared(a) else None
            self.a_gather = None if self.a_shared is not None else a[dom]
            self.b_in = problem.input_constr_b[dom]

    def __call__(self, z: torch.Tensor):
        return self.at(self.lb + z * self.width)

    def at(self, x: torch.Tensor):
        """phi (K,), g (K, m) at explicit points x (K, n)."""
        out = self.problem.model(x.reshape(self.K, *self.problem.input_shape))
        out = out.reshape(self.K, -1)
        phi = (out * self.c_obj).sum(dim=1) - self.rhs_obj
        parts = []
        if self.n_out > 0:
            if self.c_shared is not None:
                g_out = out @ self.c_shared.T
            else:
                g_out = torch.bmm(self.c_gather, out.unsqueeze(-1)).squeeze(-1)
            parts.append(g_out - self.rhs_constr)
        if self.n_in > 0:
            if self.a_shared is not None:
                g_in = x @ self.a_shared.T
            else:
                g_in = torch.bmm(self.a_gather, x.unsqueeze(-1)).squeeze(-1)
            parts.append(g_in - self.b_in)
        g = torch.cat(parts, dim=1) if parts else phi.new_zeros((self.K, 0))
        return phi, g, x


@retained
def _value_and_jacobian(ev: CandidateEvaluator, z: torch.Tensor, max_rows: int = 8192,
                        deadline=None, sliced=False):
    """phi (K,), g (K, m), grad phi (K, n), Jacobian of g (K, m, n) at z.

    One reverse pass per output row, vectorized over rows (and over the
    independent candidates by summation), in chunks of ``max_rows`` K-rows.
    """
    k_cand = z.shape[0]
    check_deadline(deadline)
    with torch.enable_grad():
        z_var = z.detach().requires_grad_(True)
        phi, g, _ = ev(z_var)
        values = torch.cat([phi.unsqueeze(1), g], dim=1)       # (K, q)
        q = values.shape[1]
        chunk = max(1, int(max_rows) // max(k_cand, 1))
        blocks = []
        for start in range(0, q, chunk):
            check_deadline(deadline)
            stop = min(q, start + chunk)
            basis = torch.zeros(stop - start, k_cand, q, dtype=z.dtype, device=z.device)
            basis[torch.arange(stop - start), :, torch.arange(start, stop)] = 1.0
            try:
                (block,) = torch.autograd.grad(
                    values, z_var, grad_outputs=basis, retain_graph=stop < q,
                    is_grads_batched=True, allow_unused=True,
                )
            except RuntimeError:
                rows = []
                for i in range(start, stop):
                    check_deadline(deadline)
                    (row,) = torch.autograd.grad(
                        values[:, i].sum(), z_var, retain_graph=True, allow_unused=True,
                    )
                    rows.append(torch.zeros_like(z) if row is None else row)
                block = torch.stack(rows)
            if block is None:
                block = torch.zeros(stop - start, *z.shape, dtype=z.dtype, device=z.device)
            blocks.append(block)
            if sliced:
                yield Atom("sqp_jacobian")
        jac = torch.cat(blocks)
    jac = jac.transpose(0, 1)                                   # (K, q, n)
    return phi.detach(), g.detach(), jac[:, 0], jac[:, 1:]


def _chol_solve(factor, rhs):
    """Solve L L^T x = rhs by two triangular solves (graph-capturable, unlike
    the batched cholesky_solve, which allocates workspace on the GPU).

    On CUDA, a batched triangular solve of more than 8 systems larger than 512
    dispatches to MAGMA, whose raw cudaMalloc of a pointer array aborts the process
    when the caching allocator holds the memory (cute_gouldqp2/3, n = 699). Chunks
    of at most 8 systems stay on cuBLAS, which allocates nothing."""
    if factor.is_cuda and factor.shape[-1] > 512 and factor.shape[0] > 8:
        return torch.cat([_chol_solve(factor[i:i + 8], rhs[i:i + 8])
                          for i in range(0, factor.shape[0], 8)])
    y = torch.linalg.solve_triangular(factor, rhs.unsqueeze(-1), upper=False)
    return torch.linalg.solve_triangular(factor.transpose(1, 2), y, upper=True).squeeze(-1)


def _ipm_iterate(h, grad, cons, jac, z, nu, d, v, s, lam):
    """One Mehrotra predictor-corrector step of the elastic QP interior point.

    Pure tensor code without host synchronization, so it can be captured in a
    CUDA graph.  Returns the new (d, v, s, lam) and the (factor, sigma) of the
    Newton matrix used.
    """
    n = grad.shape[1]
    m = cons.shape[1]
    eye = torch.eye(n, dtype=grad.dtype, device=grad.device)
    nu_ = nu.unsqueeze(1)
    jt = jac.transpose(1, 2)
    b = torch.cat([-cons, torch.zeros_like(cons), z, 1.0 - z], dim=1)
    p = b.shape[1]
    i1, i2, i3 = m, 2 * m, 2 * m + n

    def gx(dd, dv):                      # G (d, v)
        return torch.cat([_mv(jac, dd) - dv, -dv, -dd, dd], dim=1)

    def gty(y):                          # G^T y, split into (d, v) parts
        return _mv(jt, y[:, :i1]) - y[:, i2:i3] + y[:, i3:], -y[:, :i1] - y[:, i1:i2]

    mu = (s * lam).sum(dim=1) / p
    gl_d, gl_v = gty(lam)
    rd_d = _mv(h, d) + grad + gl_d
    rd_v = nu_ + gl_v
    rp = gx(d, v) + s - b
    w = (lam / s).clamp(max=1e14)
    w1, w2 = w[:, :i1], w[:, i1:i2]
    sig = w1 * w2 / (w1 + w2)
    mat = h + torch.bmm(jt * sig.unsqueeze(1), jac) + torch.diag_embed(w[:, i2:i3] + w[:, i3:])
    mat = 0.5 * (mat + mat.transpose(1, 2))
    mat = mat + 1e-12 * mat.diagonal(dim1=1, dim2=2).abs().amax(dim=1).view(-1, 1, 1) * eye
    factor, _ = torch.linalg.cholesky_ex(mat)
    frac = w1 / (w1 + w2)

    def newton(rc):
        u = w * rp - rc / s
        u_d, u_v = gty(u)
        r1 = -rd_d - u_d
        r2 = -rd_v - u_v
        dd = _chol_solve(factor, r1 + _mv(jt, frac * r2))
        dv = (r2 + w1 * _mv(jac, dd)) / (w1 + w2)
        g_step = gx(dd, dv)
        return dd, dv, w * (g_step + rp) - rc / s, -rp - g_step

    def max_step(ds, dl):
        val = torch.cat([s, lam], dim=1)
        step = torch.cat([ds, dl], dim=1)
        ratio = torch.where(step < 0, -val / step, torch.full_like(val, float("inf")))
        return ratio.amin(dim=1).clamp(max=1.0)

    dd, dv, dl, ds = newton(s * lam)
    a_aff = max_step(ds, dl).unsqueeze(1)
    mu_aff = ((s + a_aff * ds) * (lam + a_aff * dl)).sum(dim=1) / p
    sigma = (mu_aff / mu.clamp(min=1e-300)).clamp(max=1.0) ** 3
    dd, dv, dl, ds = newton(s * lam + ds * dl - (sigma * mu).unsqueeze(1))
    alpha = (0.99 * max_step(ds, dl)).unsqueeze(1)
    return (d + alpha * dd, v + alpha * dv, (s + alpha * ds).clamp(min=1e-300),
            (lam + alpha * dl).clamp(min=1e-300), factor, sig)


def _ipm_start(cons, z, nu, lam0):
    """Interior starting point (d, v, s, lam) of the elastic QP."""
    nu_ = nu.unsqueeze(1)
    d = torch.zeros_like(z)
    v = torch.relu(cons)
    s = torch.cat([v - cons, v, z, 1.0 - z], dim=1).clamp(min=1e-2)
    l1 = torch.minimum(1.0 + lam0, 0.5 * nu_)
    lam = torch.cat([l1, nu_ - l1, torch.ones_like(z), torch.ones_like(z)], dim=1)
    return d, v, s, lam


@retained
def _qp_ipm(h, grad, cons, jac, z, nu, lam0, iters=20, deadline=None, sliced=False):
    """Batched primal-dual interior point for the l1-elastic SQP subproblem.

        min_d,v  grad^T d + 0.5 d^T H d + nu * sum(v)
        s.t.     cons + J d <= v,  v >= 0,  0 <= z + d <= 1

    written as G (d, v) <= b with the four row groups [J d - v <= -cons,
    -v <= 0, -d <= z, d <= 1 - z] stacked into one slack / multiplier vector.
    Always feasible (v absorbs an inconsistent linearization); H is positive
    definite.  The elastic block is eliminated analytically, so every Newton
    system is one n x n Cholesky (Mehrotra predictor-corrector; both solves
    share the factor).  Returns d, the constraint multipliers, (factor, sigma)
    for a cheap second-order correction, and the final complementarity mu.
    """
    m = cons.shape[1]
    d, v, s, lam = _ipm_start(cons, z, nu, lam0)
    p = s.shape[1]
    factor = sig = None
    for it in range(iters):
        check_deadline(deadline)
        if it > 0 and bool(((s * lam).sum(dim=1) / p <= 1e-10 * (1.0 + nu)).all()):
            break
        d, v, s, lam, factor, sig = _ipm_iterate(h, grad, cons, jac, z, nu, d, v, s, lam)
        if sliced:
            yield Atom("sqp_qp")
    clean = dict(nan=0.0, posinf=0.0, neginf=0.0)
    mu = (s * lam).sum(dim=1) / p
    return torch.nan_to_num(d, **clean), torch.nan_to_num(lam[:, :m], **clean), (factor, sig), mu


def _capture_qp_step(graph, step, device, stream):
    """Capture a deterministic QP step without exposing the owner's RNG state.

    PyTorch 2.8 capture_end can throw before the generator epilogue and before
    the allocator leaves its pool. Its graph context then also skips stream
    restoration. Keep an independent outer stream guard and a private generator
    state; restore the original state object even on BaseException. Byte RNG
    set_state only changes seed/offset and cannot clear a poisoned capture flag.
    """
    with torch.cuda.device(device):
        index = torch.cuda.current_device()
        generator = torch.cuda.default_generators[index]
        owner_state = generator.graphsafe_get_state()
        capture_state = generator.clone_state()
        pool = torch.cuda.graph_pool_handle()
        generator.graphsafe_set_state(capture_state)
        try:
            with torch.cuda.stream(stream):
                capture(graph, step, pool=pool, stream=stream)
        finally:
            generator.graphsafe_set_state(owner_state)


class _GraphedQP:
    """The same interior point with its iteration replayed from a CUDA graph.

    Shapes are fixed for a whole polish call, so one captured iteration serves
    every SQP step; this removes the per-kernel launch overhead that dominates
    small problems.  Inputs are copied into static buffers before each solve.
    """

    def __init__(self, k, n, m, dtype, device):
        z = dict(dtype=dtype, device=device)
        self.inp = dict(
            h=torch.zeros(k, n, n, **z), grad=torch.zeros(k, n, **z), cons=torch.zeros(k, m, **z),
            jac=torch.zeros(k, m, n, **z), z=torch.full((k, n), 0.5, **z), nu=torch.ones(k, **z),
        )
        self.inp["h"] += torch.eye(n, **z)
        p = 2 * m + 2 * n
        self.st = dict(d=torch.zeros(k, n, **z), v=torch.zeros(k, m, **z),
                       s=torch.ones(k, p, **z), lam=torch.ones(k, p, **z))
        self.factor = torch.zeros(k, n, n, **z)
        self.sig = torch.zeros(k, m, **z)
        self.m = m
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(2):
                self._step()
        torch.cuda.current_stream(device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        # The root runs IPOPT in a parallel thread: in the default "global" mode its
        # host-to-device copy during this capture aborts the solve (kall_congruentcircles_c62:
        # "operation not permitted when stream is capturing").
        _capture_qp_step(self.graph, self._step, device, side)

    def _step(self):
        i, st = self.inp, self.st
        d, v, s, lam, factor, sig = _ipm_iterate(
            i["h"], i["grad"], i["cons"], i["jac"], i["z"], i["nu"], st["d"], st["v"], st["s"], st["lam"])
        st["d"].copy_(d)
        st["v"].copy_(v)
        st["s"].copy_(s)
        st["lam"].copy_(lam)
        self.factor.copy_(factor)
        self.sig.copy_(sig)

    @retained
    def __call__(self, h, grad, cons, jac, z, nu, lam0, iters=20, check_every=4, deadline=None, sliced=False):
        for key, val in (("h", h), ("grad", grad), ("cons", cons), ("jac", jac), ("z", z), ("nu", nu)):
            self.inp[key].copy_(val)
        for key, val in zip(("d", "v", "s", "lam"), _ipm_start(cons, z, nu, lam0)):
            self.st[key].copy_(val)
        st = self.st
        p = st["s"].shape[1]
        for it in range(iters):
            check_deadline(deadline)
            if it > 0 and it % check_every == 0 and bool(
                ((st["s"] * st["lam"]).sum(dim=1) / p <= 1e-10 * (1.0 + nu)).all()
            ):
                break
            self.graph.replay()
            if sliced:
                yield Atom("sqp_qp")
        clean = dict(nan=0.0, posinf=0.0, neginf=0.0)
        mu = (st["s"] * st["lam"]).sum(dim=1) / p
        return (torch.nan_to_num(st["d"].clone(), **clean),
                torch.nan_to_num(st["lam"][:, :self.m].clone(), **clean),
                (self.factor.clone(), self.sig.clone()), mu)


def _make_qp_solver(k, n, m, dtype, device):
    """CUDA-graph interior point on the GPU, the eager one otherwise (or if capture fails)."""
    if device.type == "cuda":
        try:
            return _GraphedQP(k, n, m, dtype, device)
        except Exception as exc:  # pylint: disable=broad-except
            torch.cuda.synchronize(device)
            print(f"    [ALM-SQP] CUDA graph capture failed ({type(exc).__name__}: {exc}); eager QP")
    return _qp_ipm


def _bfgs_update(h, s, y, first):
    """Powell-damped BFGS update, batched; rows with a negligible step are kept."""
    hs = _mv(h, s)
    shs = (s * hs).sum(dim=1)
    sy = (s * y).sum(dim=1)
    ok = shs > 1e-300
    # Shanno-Phua initial scaling on the first update of a candidate.
    yy = (y * y).sum(dim=1)
    scale_now = first & ok & (sy > 1e-300)
    if bool(scale_now.any()):
        gamma = (yy / sy.clamp(min=1e-300)).clamp(min=1e-12, max=1e12)
        eye = torch.eye(h.shape[1], dtype=h.dtype, device=h.device)
        h = torch.where(scale_now.view(-1, 1, 1), gamma.view(-1, 1, 1) * eye, h)
        hs = _mv(h, s)
        shs = (s * hs).sum(dim=1)
    theta = torch.where(
        sy >= 0.2 * shs,
        torch.ones_like(sy),
        0.8 * shs / (shs - sy).clamp(min=1e-300),
    )
    r = theta.unsqueeze(1) * y + (1.0 - theta).unsqueeze(1) * hs
    sr = (s * r).sum(dim=1)
    ok = ok & (sr > 1e-300) & torch.isfinite(sr)
    upd = (
        -hs.unsqueeze(2) * hs.unsqueeze(1) / shs.clamp(min=1e-300).view(-1, 1, 1)
        + r.unsqueeze(2) * r.unsqueeze(1) / sr.clamp(min=1e-300).view(-1, 1, 1)
    )
    return torch.where(ok.view(-1, 1, 1), h + upd, h), ok


@retained
def sqp_polish(
    ev: CandidateEvaluator,
    z0: torch.Tensor,
    tol: float,
    target_frac: float = 0.9,
    max_iters: int = 50,
    deadline: Optional[float] = None,
    verbose: bool = False,
    sliced: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Polish K candidates (unit-box points ``z0``) with a batched SQP.

    Returns ``(best_z, best_phi, best_viol, found, iterations)`` where
    ``found[k]`` is True when candidate ``k`` reached a raw-feasible point
    (``max(g, 0) <= tol``); ``best_*`` hold its best such point by objective
    (or its least violated point when none was found).
    """
    k_cand, n = z0.shape
    dtype, device = z0.dtype, z0.device
    target = float(target_frac) * float(tol)
    eye = torch.eye(n, dtype=dtype, device=device)
    z = z0.detach().clone().clamp(0.0, 1.0)

    best_z = z.clone()
    best_phi = torch.full((k_cand,), float("inf"), dtype=dtype, device=device)
    best_viol = torch.full((k_cand,), float("inf"), dtype=dtype, device=device)
    found = torch.zeros(k_cand, dtype=torch.bool, device=device)

    def archive(z_eval, phi_eval, g_eval):
        viol = torch.relu(g_eval).amax(dim=1) if g_eval.shape[1] else torch.zeros_like(phi_eval)
        finite = torch.isfinite(phi_eval) & torch.isfinite(viol)
        feas = finite & (viol <= tol)
        better = feas & ((~found) | (phi_eval < best_phi))
        closer = finite & (~found) & (~feas) & (viol < best_viol)
        upd = better | closer
        if bool(upd.any()):
            best_z[upd] = z_eval[upd]
            best_phi[upd] = phi_eval[upd]
            best_viol[upd] = viol[upd]
        found.logical_or_(better)

    h = None
    first_update = torch.ones(k_cand, dtype=torch.bool, device=device)
    lam = None
    nu = torch.zeros(k_cand, dtype=dtype, device=device)
    running = torch.ones(k_cand, dtype=torch.bool, device=device)
    fails = torch.zeros(k_cand, dtype=torch.int64, device=device)
    stall = torch.zeros(k_cand, dtype=torch.int64, device=device)
    prev = None  # (z, grad, jac) at the previous iterate
    lin_prev = None  # linearized residuals of the previous QP step
    qp_solver = None
    iters = 0
    for it in range(max(int(max_iters), 0)):
        if sliced:
            yield Atom("sqp", candidate(ev.lb + best_z * ev.width, best_phi, best_viol, tol))

        if deadline is not None and time.time() >= deadline:
            break
        if not bool(running.any()):
            break
        try:
            phi, g, grad, jac = yield from _value_and_jacobian(
                ev, z, deadline=deadline, sliced=sliced, _iterate=True)
            check_deadline(deadline)
        except DeadlineExceeded:
            break
        iters = it + 1
        m = g.shape[1]
        archive(z, phi, g)
        running &= (
            torch.isfinite(phi)
            & torch.isfinite(g).all(dim=1)
            & torch.isfinite(grad).all(dim=1)
            & torch.isfinite(jac).flatten(1).all(dim=1)
        )
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        jac = torch.nan_to_num(jac, nan=0.0, posinf=0.0, neginf=0.0)
        cons = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) - target

        if lam is None:
            lam = torch.zeros(k_cand, m, dtype=dtype, device=device)
        grad_lag = grad + _mv(jac.transpose(1, 2), lam)
        if h is None:
            # Initial curvature: the first unconstrained step moves ~10% of the box.
            scale = grad_lag.abs().amax(dim=1).clamp(min=1e-8) / 0.1
            h = scale.view(-1, 1, 1) * eye.expand(k_cand, n, n).clone()
        elif prev is not None:
            z_old, grad_old, jac_old = prev
            s = z - z_old
            y = grad_lag - (grad_old + _mv(jac_old.transpose(1, 2), lam))
            moved = s.abs().amax(dim=1) > 1e-14
            h_new, ok = _bfgs_update(h, s, y, first_update & moved)
            ok = ok & moved
            h = torch.where(ok.view(-1, 1, 1), h_new, h)
            first_update &= ~ok

        # Elastic QP step.  Its penalty dominates the objective's gradient per
        # unit of constraint change, and the previous multipliers of rows the
        # previous QP could satisfy (the multiplier of a row left elastic sits
        # at the penalty; feeding it back would compound the penalty).
        row_norm = jac.norm(dim=2)
        scale = grad.norm(dim=1) / row_norm.clamp(min=1e-12).median(dim=1).values if m else grad.norm(dim=1)
        nu_qp = 1e3 * (1.0 + scale)
        if m and lin_prev is not None:
            trusted = (lin_prev <= 1e-3 * float(tol)) & settled_prev.unsqueeze(1)
            lam_ref = torch.where(trusted, lam, torch.zeros_like(lam)).amax(dim=1)
            nu_qp = torch.maximum(nu_qp, 10.0 * lam_ref)
        if qp_solver is None:
            qp_solver = _make_qp_solver(k_cand, n, m, dtype, device)
        try:
            d, lam_qp, (chol, sig), mu_qp = yield from qp_solver(
                h, grad, cons, jac, z, nu_qp, lam, deadline=deadline, sliced=sliced, _iterate=True)
        except DeadlineExceeded:
            break
        settled = mu_qp <= 1e-9 * (1.0 + nu_qp)
        lam_qp = lam_qp.clamp(min=0.0)
        d = torch.where(running.unsqueeze(1), d, torch.zeros_like(d))
        lin = cons + _mv(jac, d)

        # l1 merit on the tolerance-shifted constraints; the slope is the
        # directional derivative of its linear model along d.
        if m:
            nu = torch.where(settled | (nu <= 0), torch.maximum(nu, 2.0 * lam_qp.amax(dim=1)), nu)
        infeas0 = torch.relu(cons).sum(dim=1)
        # Steering: the penalty must make the step's predicted infeasibility
        # reduction outweigh its predicted objective change (Nocedal-Wright 18.36).
        predicted = infeas0 - torch.relu(lin).sum(dim=1)
        quad = (grad * d).sum(dim=1) + 0.5 * (d * _mv(h, d)).sum(dim=1)
        needed = torch.where(predicted > 0, quad / (0.9 * predicted.clamp(min=1e-300)), torch.zeros_like(quad))
        nu = torch.maximum(nu, torch.nan_to_num(needed, nan=0.0, posinf=0.0).clamp(min=0.0))
        merit0 = phi + nu * infeas0
        slope = (grad * d).sum(dim=1) + nu * (torch.relu(lin).sum(dim=1) - infeas0)
        step_ok = running & (d.abs().amax(dim=1) > 1e-12)
        alpha = torch.ones(k_cand, dtype=dtype, device=device)
        accepted = torch.zeros(k_cand, dtype=torch.bool, device=device)
        z_acc = z.clone()
        with torch.no_grad():
            for ls in range(10):
                if deadline is not None and time.time() >= deadline:
                    break
                trying = step_ok & ~accepted
                if not bool(trying.any()):
                    break
                z_try = (z + alpha.unsqueeze(1) * d).clamp(0.0, 1.0)
                phi_t, g_t, _ = ev(z_try)
                archive(z_try, phi_t, g_t)
                merit_t = phi_t + nu * torch.relu(g_t - target).sum(dim=1)
                ok = trying & torch.isfinite(merit_t) & (
                    merit_t <= merit0 + 1e-4 * alpha * slope.clamp(max=0.0)
                )
                if ls == 0 and m and bool((trying & ~ok).any()):
                    # Second-order correction against the curvature of the
                    # constraints (Maratos effect), full step only: one more
                    # solve with the last interior-point factor, weighted
                    # towards the rows the QP treats as active.
                    err = torch.nan_to_num(g_t - target, nan=0.0, posinf=0.0, neginf=0.0) - lin
                    corr = _chol_solve(chol, -_mv(jac.transpose(1, 2), sig * err))
                    z_soc = (z + d + torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)).clamp(0.0, 1.0)
                    phi_s, g_s, _ = ev(z_soc)
                    archive(z_soc, phi_s, g_s)
                    merit_s = phi_s + nu * torch.relu(g_s - target).sum(dim=1)
                    ok_s = trying & ~ok & torch.isfinite(merit_s) & (
                        merit_s <= merit0 + 1e-4 * slope.clamp(max=0.0)
                    )
                    z_acc = torch.where(ok_s.unsqueeze(1), z_soc, z_acc)
                    accepted |= ok_s
                z_acc = torch.where(ok.unsqueeze(1), z_try, z_acc)
                accepted |= ok
                alpha = torch.where(accepted, alpha, 0.5 * alpha)
        prev = (z, grad, jac)
        lam = lam_qp
        lin_prev = lin
        settled_prev = settled
        moved = (z_acc - z).abs().amax(dim=1)
        z = torch.where(accepted.unsqueeze(1), z_acc, z)
        fails = torch.where(accepted, torch.zeros_like(fails), fails + 1)
        # Converged or creeping: five steps below 1e-7 of the box in a row.
        stall = torch.where(accepted & (moved < 1e-7), stall + 1, torch.zeros_like(stall))
        rejected = step_ok & ~accepted
        if bool(rejected.any()):
            # Curvature that yields no acceptable step is discarded.
            diag_mean = torch.diagonal(h, dim1=1, dim2=2).mean(dim=1)
            h = torch.where(rejected.view(-1, 1, 1), diag_mean.view(-1, 1, 1) * eye, h)
            first_update |= rejected
        converged = running & (d.abs().amax(dim=1) <= 1e-10) & (infeas0 <= 1e-3 * float(tol))
        running &= ~converged & (fails < 3) & (stall < 5)
        if verbose and (it < 3 or it % 10 == 9):
            feas_phi = best_phi[found]
            print(
                f"    [ALM-SQP] it={it} running={int(running.sum())}/{k_cand} "
                f"accepted={int(accepted.sum())} feasible={int(found.sum())} "
                f"best={feas_phi.min().item() if feas_phi.numel() else float('nan'):.6g} "
                f"|d|={d.abs().amax().item():.2e}"
            )
    if deadline is None or time.time() < deadline:
        with torch.no_grad():
            archive(z, *ev(z)[:2])
    return best_z, best_phi, best_viol, found, iters


def select_candidates(
    x: torch.Tensor,
    phis: torch.Tensor,
    viols: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    count: int,
    tol: float,
    near_quantile: float = 0.05,
    grid: float = 1e-3,
):
    """Pick up to ``count`` distinct restart points to polish.

    Half of the slots go to the best raw-feasible points by objective, the rest
    to the lowest-objective points among the least violated ``near_quantile``
    share (near-feasible points in good basins).  Duplicates on a unit-box grid
    of spacing ``grid`` are dropped.  Returns flat indices into ``x``.
    """
    n_total = x.shape[0]
    flat = x.reshape(n_total, -1)
    z = (flat - lb) / (ub - lb).clamp(min=1e-300)
    key = torch.round(z / grid).to(torch.int64)
    finite = torch.isfinite(phis) & torch.isfinite(viols)
    if not bool(finite.any()):
        return torch.empty(0, dtype=torch.long, device=x.device)
    feas = finite & (viols <= tol)
    q = torch.quantile(viols[finite].double(), min(max(near_quantile, 0.0), 1.0)).to(viols.dtype)
    near = finite & (viols <= torch.clamp(q, min=float(tol)))
    chosen = []
    seen = set()

    def take(mask, limit):
        if limit <= 0 or not bool(mask.any()):
            return
        idx = torch.where(mask)[0]
        order = idx[torch.argsort(phis[idx])]
        # Scan a bounded prefix; dedup on the host (small).
        prefix = order[: max(limit * 32, 256)]
        keys = key[prefix].cpu()
        got = 0
        for pos, flat_idx in enumerate(prefix.tolist()):
            kk = tuple(keys[pos].tolist())
            if kk in seen:
                continue
            seen.add(kk)
            chosen.append(flat_idx)
            got += 1
            if got >= limit:
                break

    take(feas, (count + 1) // 2)
    take(near & ~feas, count - len(chosen))
    take(feas, count - len(chosen))
    return torch.as_tensor(chosen, dtype=torch.long, device=x.device)
