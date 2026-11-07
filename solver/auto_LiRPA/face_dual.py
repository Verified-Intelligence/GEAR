"""Face-dual CROWN (DC-CROWN): one backward of a Lagrangian spec in which every face of every relaxation carries
its own multiplier.

At a nonlinear node k with incoming lower-bound coefficient A (spec, batch, *shape), the operator's own relaxation
code gives lower faces z >= sum_p a_jp z_p + b_j and upper faces z <= sum_p a'_jp z_p + b'_j (p over its inputs).
With mu_j, nu_j >= 0 per face, spec and element,

    A_p   = sum_j mu_j a_jp - sum_j nu_j a'_jp
    bias += sum_j mu_j b_j - sum_j nu_j b'_j + min(r l_k, r u_k),      r = A - sum_j mu_j + sum_j nu_j,

where [l_k, u_k] is the node's own interval: its intermediate bounds intersected with IBP of the operator on its
input intervals. Any multipliers give a valid lower bound (weak duality). CROWN is the case r = 0 with the default
relaxation on each side (mode 'sign'); alpha-CROWN is mu = A^+ w, nu = A^- w' with w, w' on the simplex of the
faces (mode 'ratio'); mode 'param' takes mu, nu as given; mode 'basis' (Pivoting CROWN, pivot_crown.py) passes the
coefficient, of either sign, through one selected face per element.

Opt-in: a node takes part only while a FaceDual state is installed on it (`install`), and only in the lower bound
of the backward that starts at `state.start`. Without a state every operator runs its unchanged code.

The faces are built once per box from the fixed intermediate bounds (per_output) and then held fixed (as alpha-
CROWN's fix_interm_bounds): `optimize` installs them as a frozen cache, so no bound_relax / interpolated_relaxation
and no intermediate-bound pass runs during its steps. Each step is the final Lagrangian backward (spec 1) through the
cached faces, its concretization and its supergradient: with the faces fixed, the bound is a fixed piecewise-linear
function of the multipliers (a fixed linear map followed by the box minimisations), so a step can later be
captured as one CUDA graph.

`per_output` and `optimize` are the bound loop over (lambda, eta, mu, nu) for the problem
    min C[0] f(x)  s.t.  C[k] f(x) <= thr[k] (k >= 1, finite thr),  E x <= h,  x_L <= x <= x_U,
on frozen intermediate bounds: one backward of the spec C[0] + sum_k lambda_k C[k] per step, the rows E x <= h
priced by eta >= 0 in the input concretization (LagrangianBox), the best bound kept; the multipliers are moved by
Adam (as alpha-CROWN), a target-level Polyak step, or an accelerated gradient method on a smoothed bound.
"""
import math
import time
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .bounded_tensor import BoundedTensor
from .perturbations import PerturbationLpNorm

INF = float('inf')

# Faces of one node: lw, uw are tuples over the inputs of (F, batch, *shape) slopes; lb, ub (F, batch, *shape)
# offsets; dl, du the default relaxation as face weights (F, batch, *shape) summing to 1; l, u the node's interval;
# rl, ru each face's range over the boxes, (u - l) + sum_p |a_p| (u_p - l_p).
Faces = namedtuple('Faces', 'lw lb uw ub dl du l u rl ru')


class FaceDual:
    """Face-dual state of one node for the lower bound of the backward from the node named `start`."""

    def __init__(self, start, mode='sign', K=5, free_tangent=False):
        self.start, self.mode, self.K, self.free_tangent = start, mode, K, free_tangent
        self.mu = self.nu = None   # 'param': the multipliers; 'ratio': nonnegative face weights (unnormalised)
        self.faces = None          # (input-bound key, Faces)
        self.used = None           # (A, mu, nu) of the last backward, detached
        self.tau = None            # smoothing: the smoothed excess of the residual's pricing is kept
        self.r = None              # the residual coefficient of the last backward (for the explicit supergradient)
        self.sel = None            # 'basis': the selected face per element, one-hot (lower (F, batch, *shape), upper)
        self.inject = None         # 'basis': a coefficient (spec, batch, *shape) added to the incoming one


def active(node, last_lA, last_uA, start_node=None):
    """The node's face-dual state when it applies to this backward, else None."""
    state = vars(node).get('face_dual')
    if state is None or last_lA is None:
        return None
    name = start_node.name if start_node is not None else vars(node).get('_start')
    if name != state.start:
        return None
    if last_uA is not None or not isinstance(last_lA, Tensor):
        raise NotImplementedError('face-dual backward: lower bound with a dense coefficient only')
    return state


def seed(net, start, C):
    """Before a lower-bound backward from `start`: a zero coefficient on every node with a face-dual state for it.
    A node whose incoming coefficient is exactly zero is otherwise never visited (BoundConcat passes None for a zero
    slice), and its paired multipliers (mu = nu > 0 at A = 0) would not act. No state installed: nothing happens."""
    nodes = vars(net).get('face_dual_nodes')
    if not nodes or not isinstance(C, Tensor):
        return
    for node in nodes:
        state = vars(node).get('face_dual')
        if state is not None and state.start == start.name and node.lA is None:
            node.lA = torch.zeros(C.shape[0], C.shape[1], *node.output_shape[1:], dtype=C.dtype, device=C.device)


def _key(tensors):
    return tuple((t, t._version) if isinstance(t, Tensor) else (t, None) for t in tensors)


def _same(key, old):
    return len(key) == len(old) and all(a is b and va == vb for (a, va), (b, vb) in zip(key, old))


def _interval(node, inputs):
    """the node's intermediate bounds intersected with IBP of the operator on its input intervals"""
    from .operators.base import Interval
    given = {id(x): x for x in inputs}
    boxes = {}
    for inp in node.inputs:
        if id(inp) not in boxes:
            x = given.get(id(inp), inp)
            boxes[id(inp)] = (Interval(x.lower, x.upper) if inp.perturbed
                              else Interval(inp.forward_value, inp.forward_value))
    with torch.no_grad():
        ibp = node.interval_propagate(*[boxes[id(inp)] for inp in node.inputs])
    lo, up = ibp.lower, ibp.upper
    if isinstance(node.lower, Tensor):
        lo = torch.maximum(lo, node.lower)
    if isinstance(node.upper, Tensor):
        up = torch.minimum(up, node.upper)
    return lo, up


def _stack(lower, upper, dl, du, l, u, widths):
    """Faces from lists of (slopes per input, offset); exact duplicates of an earlier face on the same side are
    dropped (their default weight moves to the kept face). widths: the input intervals' widths, for the ranges."""
    def side(faces, d):
        kept, weights = [], []
        for j, (w, b) in enumerate(faces):
            for i, (w2, b2) in enumerate(kept):
                if torch.equal(b, b2) and all(torch.equal(p, q) for p, q in zip(w, w2)):
                    weights[i] = weights[i] + d[j]
                    break
            else:
                kept.append((w, b))
                weights.append(d[j])
        slopes = tuple(torch.stack([w[p] for w, _ in kept]) for p in range(len(kept[0][0])))
        return slopes, torch.stack([b for _, b in kept]), torch.stack(weights)
    lw, lb, dl = side(lower, dl)
    uw, ub, du = side(upper, du)
    shape = lb.shape[1:]
    l, u = l.expand(shape), u.expand(shape)
    rng = lambda slopes: (u - l) + sum(a.abs() * w for a, w in zip(slopes, widths))
    return Faces(lw, lb, uw, ub, dl, du, l, u, rng(lw), rng(uw))


def _simplex(w, default):
    """w (F, spec, batch, *shape) >= 0 normalised over the faces; the default weights where w sums to 0"""
    total = w.sum(0, keepdim=True)
    return torch.where(total > 0, w / total.clamp(min=torch.finfo(w.dtype).tiny), default.unsqueeze(1))


def _dot(w, b):
    """sum over faces and elements of w * b (spec, batch); an infinite offset counts only where its weight is > 0"""
    finite = torch.isfinite(b)
    out = (w * torch.where(finite, b, torch.zeros_like(b)).unsqueeze(1)).sum(0).flatten(2).sum(-1)
    if not bool(finite.all()):
        hit = torch.where((w > 0) & ~finite.unsqueeze(1), b.unsqueeze(1), torch.zeros_like(w))
        out = out + hit.sum(0).flatten(2).sum(-1)
    return out


def _pay(r, l, u):
    """sum over elements (dims >= 2) of min(r l, r u): the box [l, u] prices the residual coefficient r"""
    fl, fu = torch.isfinite(l), torch.isfinite(u)
    p = torch.minimum(r * torch.where(fl, l, torch.zeros_like(l)),
                      r * torch.where(fu, u, torch.zeros_like(u))).flatten(2).sum(-1)
    if not (bool(fl.all()) and bool(fu.all())):
        bad = (((r > 0) & ~fl) | ((r < 0) & ~fu)).flatten(2).any(-1)
        p = torch.where(bad, torch.full_like(p, -INF), p)
    return p


def _pay_smooth(r, l, u, tau):
    """sum over elements of min_{l <= z <= u} r z + tau/2 ((z - m) / w)^2, m the midpoint, w = u - l: smooth and
    concave in r (gradient z*, Lipschitz w^2 / tau), and at most tau / 8 per element above _pay (finite boxes)"""
    w, m = u - l, (l + u) / 2
    pos = w > 0
    ws = torch.where(pos, w, torch.ones_like(w))
    z = torch.where(pos, torch.minimum(torch.maximum(m - r * ws ** 2 / tau, l), u), m)
    q = torch.where(pos, (z - m) / ws, torch.zeros_like(z))
    return (r * z + tau / 2 * q ** 2).flatten(2).sum(-1)


def backward(state, node, A, inputs, build):
    """The face-dual step at `node` (lower bound): coefficients for `inputs` (spec, batch, *shape) and the bias
    (spec, batch). build() returns the operator's faces: (lower, upper, dl, du), see _stack."""
    key = _key([t for x in inputs for t in (x.lower, x.upper)] + [node.lower, node.upper])
    # a frozen cache (key None, installed from per_output) is never rebuilt
    if state.faces is None or state.faces[0] is not None and not _same(key, state.faces[0]):
        with torch.no_grad():
            lower, upper, dl, du = build()
            state.faces = (key, _stack(lower, upper, dl, du, *_interval(node, inputs),
                                       [x.upper - x.lower for x in inputs]))
    f = state.faces[1]
    if state.mode == 'sign':
        mu = A.clamp(min=0) * f.dl.unsqueeze(1)
        nu = (-A).clamp(min=0) * f.du.unsqueeze(1)
    elif state.mode == 'ratio':
        mu = A.clamp(min=0) * _simplex(state.mu, f.dl)
        nu = (-A).clamp(min=0) * _simplex(state.nu, f.du)
    elif state.mode == 'box':
        mu = A.new_zeros(f.lb.shape[:1] + A.shape)
        nu = A.new_zeros(f.ub.shape[:1] + A.shape)
    elif state.mode == 'basis':
        # a basis solve of Pivoting CROWN (pivot_crown.py): the injected coefficient is added, mu, nu (if given) act as
        # they are, and the rest, of either sign, passes through the element's selected face; an element without one
        # keeps it as its residual r
        if state.inject is not None:
            A = A + state.inject
        mu = A.new_zeros(f.lb.shape[:1] + A.shape) if state.mu is None else state.mu
        nu = A.new_zeros(f.ub.shape[:1] + A.shape) if state.nu is None else state.nu
        rest = A - mu.sum(0) + nu.sum(0)
        mu = mu + rest * state.sel[0].unsqueeze(1)
        nu = nu - rest * state.sel[1].unsqueeze(1)
    else:
        mu, nu = state.mu, state.nu
    state.used = (A.detach(), mu.detach(), nu.detach())
    A_in = tuple((mu * wl.unsqueeze(1)).sum(0) - (nu * wu.unsqueeze(1)).sum(0) for wl, wu in zip(f.lw, f.uw))
    r = A - mu.sum(0) + nu.sum(0)
    pay = _pay(r, f.l, f.u)
    state.r = r
    if state.tau is not None:
        state.smooth = _pay_smooth(r, f.l, f.u, state.tau) - pay
    return A_in, _dot(mu, f.lb) - _dot(nu, f.ub) + pay


def install(net, start, mode='sign', K=5, free_tangent=False, names=None, cache=None):
    """Install face-dual states on the nonlinear nodes of `net` (or those in `names`) for the backward from
    `start`; with `cache` ({name: Faces}, per_output's), the faces are frozen to it."""
    from .operators import BoundActivation, BoundMul, BoundConcat
    nodes = []
    for node in net.nodes():
        if isinstance(node, BoundConcat):
            # keep zero slices: a zero multiplier's gradient must reach its branch (BoundConcat.bound_backward)
            vars(node)['face_dual_keep_zero'] = True
        if names is not None and node.name not in names or cache is not None and node.name not in cache:
            continue
        if node.perturbed and (isinstance(node, BoundActivation) and not isinstance(node, BoundMul)
                               or isinstance(node, BoundMul) and not node.is_linear_op):
            state = FaceDual(start, mode, K, free_tangent)
            if cache is not None:
                state.faces = (None, cache[node.name])
            vars(node)['face_dual'] = state
            nodes.append(node)
    vars(net)['face_dual_nodes'] = nodes


def uninstall(net):
    vars(net).pop('face_dual_nodes', None)
    for node in net.nodes():
        vars(node).pop('face_dual', None)
        vars(node).pop('face_dual_keep_zero', None)


def states(net):
    """{name: state} of the nodes whose face-dual step ran in the last backward"""
    return {n.name: vars(n)['face_dual'] for n in net.nodes()
            if vars(n).get('face_dual') is not None and vars(n)['face_dual'].used is not None}


class LagrangianBox(PerturbationLpNorm):
    """The input box with the rows E x <= h priced by eta >= 0 in the concretization:
    min_{x_L <= x <= x_U} A x + eta (E x - h). eta (batch, spec, R) is set before each bound computation. The
    residual A + eta E is kept, and with tau set the smoothed minimisation's excess (see _pay_smooth) as well."""

    def __init__(self, x_L, x_U, E=None, h=None):
        super().__init__(x_L=x_L, x_U=x_U)
        self.E, self.h, self.eta, self.tau, self.r = E, h, None, None, None

    def concretize(self, x, A, sign=-1, constraints=None, aux=None):
        if A is None:
            return super().concretize(x, A, sign, constraints, aux)
        assert sign == -1 and isinstance(A, Tensor)
        A = A.reshape(A.shape[0], A.shape[1], -1)
        if self.eta is not None and self.E is not None:
            A = A + torch.einsum('bsr,rn->bsn', self.eta, self.E)
        bound = super().concretize(x, A, sign, constraints, aux)
        if self.eta is not None and self.E is not None:
            bound = bound - (self.eta * self.h).sum(-1)
        self.r = A
        if self.tau is not None:
            l, u = self.x_L.reshape(1, 1, -1), self.x_U.reshape(1, 1, -1)
            self.smooth = _pay_smooth(A, l, u, self.tau) - _pay(A, l, u)
        return bound


@dataclass
class Problem:
    """min C[0] f(x) s.t. C[k] f(x) <= thr[k] (k >= 1, finite thr), E x <= h, x_L <= x <= x_U.
    C (S, n_out); thr (S,), thr[0] unused; x_L, x_U (1, *input shape); E (R, n), h (R,); interm_bounds frozen."""
    x_L: Tensor
    x_U: Tensor
    C: Tensor
    thr: Tensor
    interm_bounds: dict
    E: Optional[Tensor] = None
    h: Optional[Tensor] = None

    @property
    def rows(self):
        return torch.nonzero(torch.isfinite(self.thr[1:])).reshape(-1) + 1


def _compute(net, prob, box, C, return_A=False):
    x = BoundedTensor(prob.x_L, box)
    final, inp = net.final_name, net.input_name[0]
    ret = net.compute_bounds(x=(x,), C=C, method='backward', bound_upper=False, interm_bounds=prob.interm_bounds,
                             return_A=return_A, needed_A_dict={final: [inp]} if return_A else None)
    return (ret[0].reshape(-1), ret[-1][final][inp]) if return_A else ret[0].reshape(-1)


# per_output: the planes, the face weights each row used, each row's affine-form width, and the face cache
PerOutput = namedtuple('PerOutput', 'lA lbias faces width cache')


def per_output(net, prob, K=5, free_tangent=False, cache=None):
    """CROWN (mode 'sign') on every spec row at once: the input planes lA (S, n), lbias (S,), per face node the face
    weights every row used {name: (mu, nu)} (spec S), and every row's width over the node boxes and the input box:
    sum |A_k| (u_k - l_k) + sum |A_x| (x_U - x_L) with A the coefficients of the row's exact affine form in the node
    outputs (mode 'box': no face is used, so the coefficient reaching node k is the form's own). The faces are built
    here from prob.interm_bounds (once per box) and returned as the frozen cache {name: Faces}; with `cache` given
    (e.g. a parent box's, valid on a child box), no face is built at all."""
    for m in net.get_enabled_opt_act():
        m.opt_stage = None
    S = prob.C.shape[0]
    install(net, net.final_name, 'sign', K, free_tangent, cache=cache)
    try:
        box = PerturbationLpNorm(x_L=prob.x_L, x_U=prob.x_U)
        with torch.no_grad():
            _, d = _compute(net, prob, box, prob.C.unsqueeze(0), return_A=True)
            faces = {name: (s.used[1], s.used[2]) for name, s in states(net).items()}
            for s in states(net).values():
                s.mode = 'box'
            _, dbox = _compute(net, prob, box, prob.C.unsqueeze(0), return_A=True)
            width = dbox['lA'][0].reshape(S, -1).abs() @ (prob.x_U - prob.x_L).reshape(-1)
            for s in states(net).values():
                f = s.faces[1]
                width = width + (s.used[0].abs() * (f.u - f.l)).flatten(1).sum(-1)
        cache = {name: s.faces[1] for name, s in states(net).items()}
        return PerOutput(d['lA'][0].reshape(S, -1), d['lbias'][0].reshape(S), faces, width, cache)
    finally:
        uninstall(net)


def _mean_nonzero(values, dtype):
    """the mean magnitude of the nonzero entries, None when there are none"""
    nz = torch.cat([v.abs().reshape(-1).to(dtype) for v in values] + [torch.zeros(0, dtype=dtype)])
    nz = nz[nz > 0]
    return nz.mean() if nz.numel() else None


def _row_floor(c, G):
    """the multiplier that makes each row of G as large as c (1-norms): ||c|| / ||G_r||; rows with a zero norm get
    the mean of the others (1 when no row has one)"""
    g = G.abs().sum(-1)
    rho = torch.where(g > 0, c.abs().sum() / g.clamp(min=torch.finfo(G.dtype).tiny), torch.zeros_like(g))
    fill = _mean_nonzero([rho], G.dtype)
    return torch.where(rho > 0, rho, torch.ones_like(rho) if fill is None else fill)


def _grad(out, xs, **kw):
    """d out / d xs, zeros for the variables out does not use (eta without base rows)"""
    g = torch.autograd.grad(out, xs, allow_unused=True, **kw)
    return [torch.zeros_like(x) if d is None else d for x, d in zip(xs, g)]


def supergradient(net, prob, box, nodes, tau=None):
    """The supergradient of the last face-dual evaluation without autograd. By Danskin it is the constraint values
    (face slacks, output rows minus thresholds, base rows) at the minimiser (x*, z*) of the Lagrangian over the
    boxes: x* and z* the box ends the reduced costs (box.r, the nodes' r) choose, the midpoint at a tie (autograd's
    choice too), or with tau the smoothed minimisers; the affine nodes follow from one forward pass of the graph with
    the face nodes held at z*. Returns [d lambda, d eta, d mu_1, d nu_1, ...] (nodes in the order of `nodes`)."""
    def argmin(r, l, u):
        if tau is None:
            return torch.where(r > 0, l, torch.where(r < 0, u, (l + u) / 2))
        w, m = u - l, (l + u) / 2
        return torch.where(w > 0, torch.minimum(torch.maximum(m - r * w ** 2 / tau, l), u), m)
    rows = prob.rows
    xs = argmin(box.r.reshape(-1), prob.x_L.reshape(-1), prob.x_U.reshape(-1)).reshape(prob.x_L.shape)
    zs = {n: argmin(s.r[0], s.faces[1].l, s.faces[1].u) for n, s in nodes.items()}
    with torch.no_grad():
        net.set_input(xs, clear_forward_only=True, reset_perturbed_nodes=False)
        for n, z in zs.items():
            net[n].forward_value = z
        out = net.get_forward_value(net[net.output_name[0]]).reshape(-1)
        g = [prob.C[rows] @ out - prob.thr[rows],
             torch.zeros(0, dtype=out.dtype) if prob.E is None else prob.E @ xs.reshape(-1) - prob.h]
        for n, s in nodes.items():
            f, z = s.faces[1], zs[n]
            zin = [net.get_forward_value(inp) for inp in net[n].inputs[:len(f.lw)]]
            g += [(sum(w * v for w, v in zip(f.lw, zin)) + f.lb - z).unsqueeze(1),
                  (z - sum(w * v for w, v in zip(f.uw, zin)) - f.ub).unsqueeze(1)]
    return g


def _inverse_range(R):
    """1 / R, and 1 where a constraint has a zero range"""
    return torch.where(R > 0, 1 / R.clamp(min=torch.finfo(R.dtype).tiny), torch.ones_like(R))


def optimize(net, prob, lam0, eta0, po, steps=1000, checkpoints=(), mode='param', method='adam', scaling='A',
             lr=0.01, lr_decay=0.98, target=None, level=True, window=10, cont_every=50, power_iters=50, K=5,
             free_tangent=False, warm=None, grad='autograd'):
    """Maximise the face-dual bound g over (lambda, eta, mu, nu) (mode 'param') or, the alpha-CROWN control, over
    (lambda, eta, face ratios) with mu = A^+ w, nu = A^- w' (mode 'ratio', method 'adam').
    Start: lam0 (len(prob.rows)), eta0 (R) the plane LP's multipliers on po = per_output(...); (a) mu = sum_i lam0_i
    mu^(i) (r = 0: the plane-LP value), (b) CROWN's own choice on the combined spec; the better one is used, unless
    `warm` ({'lam', 'eta', 'faces': {name: (mu, nu)}}, e.g. a parent box's result) gives the start (mode 'param').
    The faces are po.cache, frozen: none is rebuilt here.
    Every step is one bound evaluation and its supergradient; the best (hard) bound is kept. Variables y = s m:
      scaling 'A'      mu, nu at node k: s = max(|A_k|, mean nonzero |A_k| of the node); lambda_k:
                       max(lam0_k, ||lA_0|| / ||lA_k||); eta_r: max(eta0_r, ||lA_0|| / ||E_r||) (1-norms)
      scaling 'range'  s = 1 / (the constraint's range over the boxes): face j at node k (u_k - l_k) + sum_p |a_jp|
                       (u_p - l_p); lambda_i the width of output i's affine form (po.width); eta_r sum_n |E_rn|
                       (x_U - x_L)_n. Every supergradient component then lies in [-1, 1]. For 'adam', s is
                       multiplied by the mean nonzero scaled start value, so that lr is relative.
    Methods:
      'adam'    Adam on m (betas 0.9/0.999), ExponentialLR(lr_decay) per step (alpha-CROWN), m >= 0 by clamping
      'polyak'  m <- max(0, m + t s), t = (target_k - g) / ||s||^2, s the supergradient in m; target_k =
                min(target, best + delta) while best < target, else best + delta; delta = target - g_start, halved
                when the best has not risen by delta / 2 within `window` steps, else kept (Goffin-Kiwiel level rule);
                level=False never halves (target_k = target: a box whose bound is unbounded above)
      'smooth'  every box minimisation min_{l<=z<=u} r z (node residuals and input concretization) replaced by
                min r z + tau/2 ((z - m)/w)^2: g_tau >= g is smooth with an L = ||D^1/2 J||^2 / tau Lipschitz
                gradient (J: the residuals' linear map in m, D = box widths^2; power iteration at the start).
                Accelerated projected gradient (Tseng's variant: every evaluated point is a convex combination of
                feasible points), step 1/L; tau_0 = 8 (target - g_start) / (number of smoothed scalars), so that the
                smoothing excess is at most target - g_start; tau halved every cont_every steps, with a restart.
                The bound kept and reported is the hard g at the evaluated point.
    grad: 'autograd' (backward through the bound) or 'explicit' (`supergradient`: the evaluation runs without
    autograd, then one forward pass of the graph at the Lagrangian minimiser; mode 'param').
    Returns {'init_a', 'init_b', 'start', 'start_value', 'trace': [(step, best, current, seconds, lam, eta)],
    'best', 'lam', 'eta', 'faces' (the best point's mu, nu), 'seconds', 'steps', 'setup_evaluations',
    'setup_seconds'}."""
    if grad == 'explicit' and mode != 'param':
        raise ValueError("the explicit supergradient is for mode 'param'")
    for m in net.get_enabled_opt_act():
        m.opt_stage = None
    final = net.final_name
    rows, C, thr = prob.rows, prob.C, prob.thr
    box = LagrangianBox(prob.x_L, prob.x_U, prob.E, prob.h)
    dt = C.dtype
    lam0 = torch.as_tensor(lam0, dtype=dt).reshape(-1).clamp(min=0)
    eta0 = (torch.zeros(0, dtype=dt) if prob.E is None
            else torch.as_tensor(eta0, dtype=dt).reshape(-1).clamp(min=0))

    def value(lam, eta):
        box.eta = None if prob.E is None else eta.reshape(1, 1, -1)
        spec = (C[0] + lam @ C[rows]).reshape(1, 1, -1)
        return _compute(net, prob, box, spec)[0] - (lam * thr[rows]).sum()

    install(net, final, 'param', K, free_tangent, names=set(po.faces), cache=po.cache)
    try:
        nodes = {name: vars(net[name])['face_dual'] for name in po.faces}
        weights = torch.zeros(C.shape[0], dtype=dt)
        weights[0] = 1.0
        weights[rows] = lam0
        start_a = {name: tuple(torch.einsum('s,fs...->f...', weights, t).unsqueeze(1) for t in po.faces[name])
                   for name in po.faces}
        with torch.no_grad():
            for name, s in nodes.items():
                s.mu, s.nu = start_a[name]
            init_a = float(value(lam0, eta0))
            used_a = {name: s.used for name, s in nodes.items()}
            for s in nodes.values():
                s.mode = 'sign'
            init_b = float(value(lam0, eta0))
            used_b = {name: s.used for name, s in nodes.items()}
        which = 'a' if init_a >= init_b else 'b'
        lA = po.lA
        s_lam_A = torch.maximum(lam0, _row_floor(lA[0], lA[rows]))
        s_eta_A = eta0 if prob.E is None else torch.maximum(eta0, _row_floor(lA[0], prob.E))
        if mode == 'ratio':
            # alpha-CROWN's ratios from either start: start a's face weights normalised per side, or CROWN's default
            for s in nodes.values():
                s.mode = 'ratio'
            with torch.no_grad():
                cand = {}
                for tag in ('a', 'b'):
                    for name, s in nodes.items():
                        f = s.faces[1]
                        s.mu, s.nu = ((used_a[name][1], used_a[name][2]) if tag == 'a'
                                      else (f.dl.unsqueeze(1), f.du.unsqueeze(1)))
                    cand[tag] = (float(value(lam0, eta0)), {n: (s.mu.clone(), s.nu.clone()) for n, s in nodes.items()})
            which = max(cand, key=lambda t: cand[t][0])
            start_value = cand[which][0]
            ys = [lam0, eta0] + [t for n in nodes for t in cand[which][1][n]]
            scales = [s_lam_A, s_eta_A] + [torch.ones((), dtype=dt)] * (2 * len(nodes))
        else:
            for s in nodes.values():
                s.mode = 'param'
            used = used_a if which == 'a' else used_b
            start_value = init_a if which == 'a' else init_b
            ys = [lam0, eta0] + [t for n in nodes for t in used[n][1:]]
            if warm is not None:
                with torch.no_grad():
                    for n, st in nodes.items():
                        st.mu, st.nu = warm['faces'][n]
                    start_value = float(value(warm['lam'], warm['eta']))
                which = 'warm'
                used = {name: st.used for name, st in nodes.items()}
                ys = [warm['lam'], warm['eta']] + [t for n in nodes for t in warm['faces'][n]]
            if scaling == 'A':
                fall = _mean_nonzero([u[0] for u in used.values()], dt)
                scales = [s_lam_A, s_eta_A]
                for n in nodes:
                    fl = _mean_nonzero([used[n][0]], dt)
                    fl = fl if fl is not None else (fall if fall is not None else torch.ones((), dtype=dt))
                    sA = torch.maximum(used[n][0].abs(), fl).unsqueeze(0)
                    scales += [sA, sA]
            else:
                R = [po.width[rows], torch.zeros(0, dtype=dt) if prob.E is None
                     else prob.E.abs() @ (prob.x_U - prob.x_L).reshape(-1)]
                for s in nodes.values():
                    R += [s.faces[1].rl.unsqueeze(1), s.faces[1].ru.unsqueeze(1)]
                scales = [_inverse_range(r) for r in R]
                if method == 'adam':
                    c = _mean_nonzero([y / s for y, s in zip(ys, scales)], dt)
                    scales = [s * (c if c is not None else 1.0) for s in scales]
        ms = [(y / s).detach().clone().requires_grad_() for y, s in zip(ys, scales)]

        def evaluate(ms):
            with torch.set_grad_enabled(grad == 'autograd'):
                y = [s * m for s, m in zip(scales, ms)]
                for k, s in enumerate(nodes.values()):
                    s.mu, s.nu = y[2 + 2 * k], y[3 + 2 * k]
                return value(y[0], y[1]), y

        def supergrad(g, point, smooth=False):
            """d g (smooth: d g_tau) / d m at the point just evaluated"""
            if grad == 'explicit':
                return [s * d for s, d in zip(scales, supergradient(net, prob, box, nodes,
                                                                   box.tau if smooth else None))]
            return _grad(smoothed(g) if smooth else g, point)

        def smoothed(g):
            return g + sum(s.smooth.sum() for s in nodes.values()) + (box.smooth.sum() if box.tau else 0.0)

        checkpoints = set(checkpoints)
        best, best_y = -INF, [y.detach().clone() for y in ys]
        trace, setup, setup_seconds = [], 0, 0.0
        t0 = time.time()

        def keep(step, g, y):
            nonlocal best, best_y
            cur = float(g.detach())
            if cur > best:
                best, best_y = cur, [t.detach().clone() for t in y]
            if step in checkpoints:
                trace.append((step, best, cur, time.time() - t0, best_y[0].clone(), best_y[1].clone()))

        if method == 'adam':
            opt = torch.optim.Adam(ms, lr=lr)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)
            for step in range(steps + 1):
                g, y = evaluate(ms)
                keep(step, g, y)
                if step == steps:
                    break
                opt.zero_grad()
                if grad == 'explicit':
                    for m, d in zip(ms, supergrad(g, ms)):
                        m.grad = -d
                else:
                    (-g).backward()
                opt.step()
                scheduler.step()
                with torch.no_grad():
                    for m in ms:
                        m.clamp_(min=0)
        elif method == 'polyak':
            delta = target - start_value if target > start_value else 1e-3 * max(1.0, abs(start_value))
            ref, since = start_value, 0
            for step in range(steps + 1):
                g, y = evaluate(ms)
                keep(step, g, y)
                if step == steps:
                    break
                goal = min(target, best + delta) if best < target else best + delta
                grads = supergrad(g, ms)
                norm2 = sum(float((v ** 2).sum()) for v in grads)
                if norm2 > 0:
                    t = (goal - float(g.detach())) / norm2
                    with torch.no_grad():
                        for m, v in zip(ms, grads):
                            m.add_(t * v).clamp_(min=0)
                since += 1
                if best - ref >= delta / 2:
                    ref, since = best, 0
                elif since >= window and level:
                    delta, ref, since = delta / 2, best, 0
        elif method == 'smooth':
            n_smooth = sum(int(s.faces[1].l.numel()) for s in nodes.values()) + prob.x_L.numel()
            gap = target - start_value if target > start_value else 1e-3 * max(1.0, abs(start_value))
            tau = 8 * gap / n_smooth
            for s in nodes.values():
                s.tau = tau
            box.tau = tau
            widths = [s.faces[1].u - s.faces[1].l for s in nodes.values()] + [(prob.x_U - prob.x_L).reshape(1, 1, -1)]

            def residuals(point):
                y = [s * m for s, m in zip(scales, point)]
                for k, st in enumerate(nodes.values()):
                    st.mu, st.nu = y[2 + 2 * k], y[3 + 2 * k]
                value(y[0], y[1])
                return torch.cat([(s.r * w).reshape(-1) for s, w in zip(nodes.values(), widths)]
                                 + [(box.r * widths[-1]).reshape(-1)])

            # sigma^2 = ||D^1/2 J||^2 by power iteration; J v = R(m + v) - R(m) (R is affine in m)
            base = [m.detach().clone().requires_grad_() for m in ms]
            R0 = residuals(base)
            gen = torch.Generator().manual_seed(0)
            v = [torch.randn(m.shape, dtype=dt, generator=gen) for m in ms]
            sigma2 = 0.0
            for _ in range(power_iters):
                nv = math.sqrt(sum(float((a ** 2).sum()) for a in v))
                v = [a / nv for a in v]
                with torch.no_grad():
                    Jv = residuals([b.detach() + a for b, a in zip(base, v)]) - R0.detach()
                JtJv = _grad(R0, base, grad_outputs=Jv, retain_graph=True)
                sigma2 = math.sqrt(sum(float((a ** 2).sum()) for a in JtJv))
                v = [a.clone() for a in JtJv]
            setup = power_iters + 1
            setup_seconds = time.time() - t0
            L = sigma2 / tau
            y = [m.detach().clone() for m in ms]
            z = [m.detach().clone() for m in ms]
            theta = 1.0
            for step in range(steps + 1):
                point = [((1 - theta) * a + theta * b).requires_grad_(grad == 'autograd') for a, b in zip(y, z)]
                g, yv = evaluate(point)
                keep(step, g, yv)
                if step == steps:
                    break
                grads = supergrad(g, point, smooth=True)
                with torch.no_grad():
                    z = [(b + d / (theta * L)).clamp(min=0) for b, d in zip(z, grads)]
                    y = [(1 - theta) * a + theta * b for a, b in zip(y, z)]
                theta = (math.sqrt(theta ** 4 + 4 * theta ** 2) - theta ** 2) / 2
                if (step + 1) % cont_every == 0:
                    tau, L, theta = tau / 2, L * 2, 1.0
                    for s in nodes.values():
                        s.tau = tau
                    box.tau = tau
                    z = [a.clone() for a in y]
        else:
            raise ValueError(method)
        return {'init_a': init_a, 'init_b': init_b, 'start': which, 'start_value': start_value, 'trace': trace,
                'best': best, 'lam': best_y[0], 'eta': best_y[1],
                'faces': {n: (best_y[2 + 2 * k], best_y[3 + 2 * k]) for k, n in enumerate(nodes)},
                'seconds': time.time() - t0, 'steps': steps,
                'setup_evaluations': setup, 'setup_seconds': setup_seconds}
    finally:
        uninstall(net)
