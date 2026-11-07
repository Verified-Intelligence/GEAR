"""Batched GPU feasibility restoration for ALM-PGD candidates.

The ALM outer loop locates promising basins with a first-order method.  Close to
a feasible point its progress is limited by the step-size continuation, and the
single-domain Gauss-Newton polish in ``alm.py`` handles one seed at a time.  This
module restores raw feasibility for many candidates at once (several seeds for
every domain of a BaB batch, or many seeds of the root problem):

* variables live in unit-box coordinates ``z = (x - l) / (u - l)`` of their domain;
* each iteration builds the exact residual Jacobian of every candidate with one
  batched reverse sweep;
* the step minimises ``0.5 ||[g + J d + s]_+||^2 + 0.5 lam ||d||^2`` over the box by
  a growing active set of inequality rows (rows that are violated, or would be
  violated after the step) and a projected free-variable solve for the box;
* a batched backtracking line search accepts decrease of the l2 merit
  ``sum relu(g)^2`` (or of the max violation) and keeps, per candidate, the best
  point by the raw rule (feasible with the smallest objective, else the smallest
  max violation).

Everything runs in the problem's dtype on its device.  Raw residuals and the
configured feasibility tolerance stay authoritative: callers accept a point only
if ``max relu(g) <= feas_tol`` on the original rows.
"""

import time
from typing import Callable, Optional, Tuple

import torch
from .utils import check_deadline, DeadlineExceeded
from .continuation import Atom, candidate, retained

Evaluator = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


def make_domain_evaluator(
    problem,
    domain_index: torch.Tensor,
    level: Optional[torch.Tensor] = None,
) -> Evaluator:
    """Build ``z (Ba, Q, n) -> (phi (Ba, Q), g (Ba, Q, m))`` in unit coordinates.

    ``domain_index`` (Ba,) selects the active domains of ``problem``; every
    candidate in row ``b`` lives in the box of domain ``domain_index[b]``.
    With ``level`` (Ba,), a search-only row ``phi - level_b <= 0`` is appended
    after the ``evaluate.n_raw`` original rows (objective level restoration);
    callers judge feasibility on the first ``n_raw`` rows only.
    """
    lb = problem.dm_l
    ub = problem.dm_u
    if lb.ndim == len(problem.input_shape):
        lb = lb.unsqueeze(0)
        ub = ub.unsqueeze(0)
    batch = int(domain_index.numel())
    lb_d = lb.index_select(0, domain_index).reshape(batch, 1, -1)
    width_d = (ub.index_select(0, domain_index).reshape(batch, 1, -1) - lb_d)
    c_obj = problem.C_obj.index_select(0, domain_index)
    rhs_obj = problem.rhs_obj.index_select(0, domain_index)
    c_con = rhs_con = a_in = b_in = None
    if problem.n_output_constraints > 0:
        c_con = problem.C_constr.index_select(0, domain_index)
        rhs_con = problem.rhs_constr.index_select(0, domain_index)
    if problem.n_input_constraints > 0:
        a_in = problem.input_constr_A.index_select(0, domain_index)
        b_in = problem.input_constr_b.index_select(0, domain_index)
    input_shape = tuple(problem.input_shape)

    def evaluate(z: torch.Tensor):
        rows, per = z.shape[0], z.shape[1]
        x = lb_d + z * width_d
        out = problem.model(x.reshape(rows * per, *input_shape))
        out = out.reshape(rows, per, -1)
        phi = (out @ c_obj.transpose(1, 2)).squeeze(-1) - rhs_obj
        pieces = []
        if c_con is not None:
            pieces.append(out @ c_con.transpose(1, 2) - rhs_con.unsqueeze(1))
        if a_in is not None:
            pieces.append(x @ a_in.transpose(1, 2) - b_in.unsqueeze(1))
        if level_d is not None:
            pieces.append((phi - level_d).unsqueeze(2))
        if pieces:
            g = torch.cat(pieces, dim=2)
        else:
            g = out.new_zeros((rows, per, 0))
        return phi, g

    level_d = None if level is None else level.reshape(batch, 1)
    evaluate.lb = lb_d
    evaluate.width = width_d
    evaluate.n_raw = int(problem.n_constraints)
    return evaluate


@retained
def residual_jacobian(
    evaluate: Evaluator,
    z: torch.Tensor,
    row_chunk: int = 64,
    deadline: Optional[float] = None,
    sliced: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``phi (Ba,Q), g (Ba,Q,m), J (Ba,Q,m,n)`` by batched reverse sweeps.

    Candidates are independent rows of the model batch, so the gradient of
    ``sum g[..., i]`` with respect to ``z`` is row ``i`` of every candidate's
    Jacobian.  Rows go in chunks through ``is_grads_batched``; a per-row loop is
    the fallback for models whose backward cannot be vmapped.
    """
    check_deadline(deadline)
    z = z.detach().requires_grad_(True)
    with torch.enable_grad():
        phi, g = evaluate(z)
    rows = g.shape[-1]
    check_deadline(deadline)
    jac = z.new_zeros(z.shape[:2] + (rows, z.shape[-1]))
    if rows == 0:
        return phi.detach(), g.detach(), jac
    count = int(z.shape[0] * z.shape[1])
    # Each chunk runs the model backward on chunk*count rows and materialises a
    # chunk*count*rows seed tensor: bound both.
    chunk = max(1, min(int(row_chunk), rows,
                       (1 << 16) // max(count, 1),
                       (1 << 25) // max(count * rows, 1)))
    for start in range(0, rows, chunk):
        check_deadline(deadline)
        stop = min(start + chunk, rows)
        size = stop - start
        selector = torch.zeros(size, rows, device=z.device, dtype=z.dtype)
        selector[torch.arange(size), torch.arange(start, stop)] = 1.0
        grad_outputs = selector.view(size, 1, 1, rows).expand(
            size, *g.shape
        )
        if not g.requires_grad:
            break  # no row depends on z: the Jacobian is zero
        try:
            (block,) = torch.autograd.grad(
                g, z, grad_outputs=grad_outputs,
                retain_graph=True, is_grads_batched=True, allow_unused=True,
            )
            if block is not None:
                jac[:, :, start:stop, :] = block.permute(1, 2, 0, 3)
        except RuntimeError:
            for row in range(start, stop):
                check_deadline(deadline)
                (grad_row,) = torch.autograd.grad(
                    g[..., row].sum(), z, retain_graph=True, allow_unused=True
                )
                if grad_row is not None:
                    jac[:, :, row, :] = grad_row
        if sliced:
            yield Atom("restore_jacobian")
    return phi.detach(), g.detach(), jac


def _masked_step(
    jac: torch.Tensor,
    target: torch.Tensor,
    rows: torch.Tensor,
    free: torch.Tensor,
    damping: torch.Tensor,
) -> torch.Tensor:
    """Solve ``min ||J_w d + t_w||^2 + lam ||d||^2`` with fixed coordinates.

    ``lam = damping * max diag(J_w^T J_w)`` (relative damping).
    Shapes: jac (K,m,n), target/rows (K,m), free (K,n), damping (K,).  ``rows``
    and ``free`` are 0/1 masks.  Fixed coordinates get a zero step.  The row-space
    (Gram) form is used when it is the smaller system.
    """
    jw = jac * rows.unsqueeze(2) * free.unsqueeze(1)
    tw = target * rows
    count, m, n = jw.shape
    # Levenberg-Marquardt damping relative to the curvature of the rows that
    # are actually in the system: strongly satisfied rows can have Jacobian
    # norms many orders larger than the active ones (1e7 vs 1 on ex3_1_1).
    scale = jw.square().sum(dim=1).amax(dim=1)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    damping = damping * scale
    if n <= m:
        normal = jw.transpose(1, 2) @ jw
        normal = normal + torch.diag_embed(damping.unsqueeze(1) + (1.0 - free))
        rhs = -(jw.transpose(1, 2) @ tw.unsqueeze(2))
        step = torch.linalg.solve(normal, rhs).squeeze(2)
    else:
        gram = jw @ jw.transpose(1, 2)
        gram = gram + torch.diag_embed(damping.unsqueeze(1).expand(count, m))
        dual = torch.linalg.solve(gram, -tw.unsqueeze(2))
        step = (jw.transpose(1, 2) @ dual).squeeze(2)
    return step * free


def row_norms(jac: torch.Tensor) -> torch.Tensor:
    """Per-row gradient norms (K, m), with rows that do not depend on z set to 1."""
    norms = torch.linalg.vector_norm(jac, dim=2)
    return torch.where(norms > 0, norms, torch.ones_like(norms))


@retained
def gauss_newton_step(
    z: torch.Tensor,
    g: torch.Tensor,
    jac: torch.Tensor,
    damping: torch.Tensor,
    shift: float,
    active_iters: int,
    row_norm: Optional[torch.Tensor] = None,
    lo=0.0,
    hi=1.0,
    deadline=None,
    sliced=False,
) -> torch.Tensor:
    """Box-projected, growing-active-set Gauss-Newton step for ``g(z) <= 0``.

    Shapes: z (K,n), g (K,m), jac (K,m,n), damping (K,).  Rows are driven to
    ``-shift``; the active set starts at the rows violating that target and grows
    by rows the linearised step would violate.  With ``row_norm`` every row is
    divided by its gradient norm first, so each row's residual reads as a
    distance in unit-box space; this leaves the solution set of the linearised
    rows unchanged but makes the damping act uniformly across rows whose raw
    scales differ by orders of magnitude.
    """
    target = g + shift
    if row_norm is not None:
        target = target / row_norm
        jac = jac / row_norm.unsqueeze(2)
    rows = (target > 0).to(z.dtype)
    grad = ((jac * rows.unsqueeze(2)).transpose(1, 2)
            @ (target * rows).unsqueeze(2)).squeeze(2)
    # A coordinate on a bound whose descent direction leaves the box is fixed.
    free = (~(((z <= lo) & (grad > 0.0)) | ((z >= hi) & (grad < 0.0)))).to(z.dtype)
    step = torch.zeros_like(z)
    for _ in range(max(int(active_iters), 1)):
        check_deadline(deadline)
        step = _masked_step(jac, target, rows, free, damping)
        # Coordinates that would leave the box stop at the bound; resolve the
        # remaining free coordinates with that move folded into the target.
        trial = z + step
        clipped = ((trial < lo) | (trial > hi)) & (free > 0)
        if bool(clipped.any().item()):
            bound_move = torch.where(
                clipped, torch.maximum(torch.minimum(trial, hi), lo) - z,
                torch.zeros_like(z)
            )
            free_after = free * (~clipped).to(z.dtype)
            folded = target + (jac @ bound_move.unsqueeze(2)).squeeze(2)
            step = _masked_step(jac, folded, rows, free_after, damping) + bound_move
        if sliced:
            yield Atom("restore_normal_solve")
        predicted = target + (jac @ step.unsqueeze(2)).squeeze(2)
        new_rows = torch.maximum(rows, (predicted > 0).to(z.dtype))
        if bool(torch.equal(new_rows, rows)):
            break
        rows = new_rows
    return step


@retained
def batched_gauss_newton_restoration(
    evaluate: Evaluator,
    z0: torch.Tensor,
    *,
    steps: int,
    feas_tol: float,
    damping: float = 1.0e-6,
    trust_radius: float = 1.0,
    active_iters: int = 3,
    target_frac: float = 0.5,
    row_chunk: int = 64,
    deadline: Optional[float] = None,
    verbose: bool = False,
    n_raw_rows: Optional[int] = None,
    interior=0.0,
    sliced: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Batched Gauss-Newton restoration from ``z0`` (Ba, P, n) in unit coords.

    Returns the per-candidate archived best ``z`` (Ba,P,n), objective (Ba,P) and
    raw max violation (Ba,P), plus the number of iterations run.  The step and
    merit use every row of ``evaluate``; the archive's feasibility uses the first
    ``n_raw_rows`` (default: all) rows only.
    """
    batch, per, dim = z0.shape
    # Interior margin in unit coordinates: a float, or a tensor broadcastable to
    # z0 (e.g. an absolute x-distance divided by each domain's widths).
    margin = torch.as_tensor(interior, device=z0.device, dtype=z0.dtype)
    margin = margin.clamp(0.0, 0.25).expand(batch, per, dim).reshape(batch * per, dim)
    lo = margin
    hi = 1.0 - margin
    z = torch.maximum(torch.minimum(z0.detach().reshape(batch * per, dim), hi), lo)
    z = z.reshape(batch, per, dim)
    count = batch * per
    dtype = z.dtype
    device = z.device
    tiny = torch.finfo(dtype).tiny
    shift = max(float(target_frac), 0.0) * float(feas_tol)
    alphas = torch.tensor([2.0 ** (-i) for i in range(8)], device=device, dtype=dtype)
    n_alpha = int(alphas.numel())
    idx = torch.arange(count, device=device)

    def max_viol(g_val: torch.Tensor, rows: Optional[int]) -> torch.Tensor:
        part = g_val if rows is None else g_val[..., :rows]
        if part.shape[-1] == 0:
            return part.new_zeros(part.shape[:-1])
        return torch.relu(part).amax(dim=-1)

    with torch.no_grad():
        phi, g = evaluate(z)
    viol = max_viol(g, None).reshape(count)
    merit = torch.relu(g).square().sum(dim=2).reshape(count)
    bad0 = ~(torch.isfinite(merit) & torch.isfinite(phi.reshape(count)))
    viol = torch.where(bad0, torch.full_like(viol, float("inf")), viol)
    best_z = z.reshape(count, dim).clone()
    best_phi = torch.where(bad0, torch.full_like(viol, float("inf")), phi.reshape(count))
    best_viol = torch.where(
        bad0, torch.full_like(viol, float("inf")), max_viol(g, n_raw_rows).reshape(count))
    z = z.reshape(count, dim)
    lam = torch.full((count,), float("nan"), device=device, dtype=dtype)
    radius = torch.full((count,), max(float(trust_radius), 1.0e-12), device=device, dtype=dtype)
    done = (viol <= shift) | bad0
    rel_damping = max(float(damping), 1.0e-14)
    iters = 0
    last_iter_time = 0.0
    for _ in range(max(int(steps), 0)):
        if sliced:
            points = evaluate.lb + best_z.reshape(batch, per, dim) * evaluate.width
            yield Atom("restore", candidate(points, best_phi, best_viol, feas_tol))

        now = time.time()
        # An iteration is not interruptible (one Jacobian sweep can take seconds
        # on large models): do not start one that would overrun the deadline.
        if deadline is not None and now + last_iter_time >= deadline:
            break
        live = ~done
        if not bool(live.any().item()):
            break
        iters += 1
        try:
            phi_cur, g_cur, jac = yield from residual_jacobian(
                evaluate, z.reshape(batch, per, dim), row_chunk, deadline=deadline,
                sliced=sliced, _iterate=True
            )
            check_deadline(deadline)
        except DeadlineExceeded:
            break
        g_cur = g_cur.reshape(count, -1)
        jac = jac.reshape(count, g_cur.shape[1], dim)
        with torch.no_grad():
            finite = torch.isfinite(jac).all(dim=2) & torch.isfinite(g_cur)
            # Rows with a non-finite value or derivative carry no usable model.
            jac = torch.where(finite.unsqueeze(2), jac, torch.zeros_like(jac))
            g_model = torch.where(finite, g_cur, torch.zeros_like(g_cur))
            lam = torch.where(torch.isnan(lam), torch.full_like(lam, rel_damping), lam)
            rn = row_norms(jac)
            try:
                step = yield from gauss_newton_step(
                    z, g_model, jac, lam, shift, active_iters, row_norm=rn, lo=lo, hi=hi, deadline=deadline,
                    sliced=sliced, _iterate=True)
            except DeadlineExceeded:
                break
            except RuntimeError:
                lam = lam * 10.0
                continue
            if deadline is not None and time.time() >= deadline:
                break
            step = torch.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
            step = step * live.unsqueeze(1).to(dtype)
            step_inf = step.abs().amax(dim=1).clamp(min=tiny)
            step = step * torch.clamp(radius / step_inf, max=1.0).unsqueeze(1)

            # Candidate-major trials keep every domain's points contiguous.
            trial = torch.maximum(torch.minimum(
                z.unsqueeze(1) + alphas.view(1, n_alpha, 1) * step.unsqueeze(1),
                hi.unsqueeze(1)), lo.unsqueeze(1))
            phi_t, g_t = evaluate(trial.reshape(batch, per * n_alpha, dim))
            phi_t = phi_t.reshape(count, n_alpha)
            g_t = g_t.reshape(count, n_alpha, -1)
            viol_t = max_viol(g_t, None)
            raw_t = max_viol(g_t, n_raw_rows)
            merit_t = (torch.relu(g_t) / rn.unsqueeze(1)).square().sum(dim=2)
            finite_t = torch.isfinite(merit_t) & torch.isfinite(phi_t)
            merit_cur = (torch.relu(g_cur) / rn).square().sum(dim=1)
            viol_cur = max_viol(g_cur, None)
            l2_ok = merit_t < merit_cur.unsqueeze(1) * (1.0 - 1.0e-8)
            linf_ok = viol_t < viol_cur.unsqueeze(1)
            ok = finite_t & (l2_ok | linf_ok) & live.unsqueeze(1)
            accepted = ok.any(dim=1)
            first = torch.argmax(ok.to(torch.int8), dim=1)
            z = torch.where(accepted.unsqueeze(1), trial[idx, first], z)

            # Archive: any trial point may be the best one seen.
            feas_t = finite_t & (raw_t <= feas_tol)
            best_feas = best_viol <= feas_tol
            cand_phi, cand_i = torch.where(
                feas_t, phi_t, torch.full_like(phi_t, float("inf"))).min(dim=1)
            cand_viol, cand_vi = torch.where(
                finite_t, raw_t, torch.full_like(raw_t, float("inf"))).min(dim=1)
            has_feas = torch.isfinite(cand_phi)
            take_feas = has_feas & ((~best_feas) | (cand_phi < best_phi))
            take_viol = (~has_feas) & (~best_feas) & (cand_viol < best_viol)
            pick = torch.where(take_feas, cand_i, cand_vi)
            take = take_feas | take_viol
            best_z = torch.where(take.unsqueeze(1), trial[idx, pick], best_z)
            best_phi = torch.where(take, phi_t[idx, pick], best_phi)
            best_viol = torch.where(take, raw_t[idx, pick], best_viol)

            full = accepted & (first == 0)
            lam = torch.where(full, lam / 3.0, lam)
            lam = torch.where(accepted, lam, lam * 8.0)
            lam = lam.clamp(min=1.0e-14, max=1.0e8)
            radius = torch.where(full, torch.clamp(radius * 2.0, max=float(trust_radius)), radius)
            radius = torch.where(accepted, radius, radius * 0.25)
            new_viol = torch.where(accepted, viol_t[idx, first], viol_cur)
            done = done | (new_viol <= shift) | (radius < 1.0e-10)
        last_iter_time = time.time() - now
        if verbose and (iters <= 2 or iters % 10 == 0):
            print(
                "    [ALM-FEAS] GN iter "
                f"{iters}: live={int(live.sum().item())}/{count} "
                f"accepted={int(accepted.sum().item())} "
                f"best_viol_min={best_viol.min().item():.3e} "
                f"feasible={int((best_viol <= feas_tol).sum().item())} "
                f"alpha_idx_med={first[live].float().median().item():.0f} "
                f"lam_rel_med={lam[live].median().item():.1e} "
                f"radius_med={radius[live].median().item():.1e}"
            )
    return (
        best_z.reshape(batch, per, dim),
        best_phi.reshape(batch, per),
        best_viol.reshape(batch, per),
        iters,
    )
