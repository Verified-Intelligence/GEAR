"""At most 256 simultaneous support functions with a convex quadratic cutoff.

The extra cutoff multiplier mu is nonnegative. Every endpoint/ray uses
the outward value -h*y+mu*(c0-T)+sum min(mu*q*z^2+(s+G^T*y+mu*c)*z).
All steps are sparse O(nnz) per lane; fixed restarts accompany warm rounds.
"""
import numpy as np
import torch
import time
from .restarted_qp import product, _matrix, _pricing, _refresh_pricing, _capture_reset


def _workspace(data, target_input, capture, has_warm):
    from nlp_solver.rigorous_dual import quadratic_lower_bound, gamma, Rows
    created = time.perf_counter()
    original, *tensors = data
    rows = Rows(original.row, original.col, original.val.clone(), original.m, original.n)
    c, c0, h, l, u, q = [t.clone() for t in tensors]
    target = target_input.clone(); B, n = target.shape
    cutoff = c.new_zeros(())
    pricing = _pricing(rows, 1); matrix = _matrix(rows, 1)
    mv, tmv = matrix.right, matrix.left
    rows._single_pass = capture
    warm = (torch.zeros_like(target), c.new_zeros(B, rows.m), c.new_zeros(B)) if has_warm else None
    initial_v = c.new_tensor(np.random.default_rng(0).standard_normal(n))[None]
    mid = rad = Dr = Dc = hs = S = ac = qs = kc = sc = lw = uw = eta = X = Y = mu = Xs = Ys = ms = E = best = empty = since = iteration = restarts = None
    def cert(yy, mm, ray=False):
        physical_y, physical_mu = (Dr*yy).clamp_min(0), (sc*mm).clamp_min(0)
        qq = torch.nextafter(q*physical_mu[:, None], c.new_full((), -float('inf'))).clamp_min(0)
        obj = torch.zeros_like(target) if ray else target
        cc = obj+c*physical_mu[:, None]
        ce = gamma(4)*(obj.abs()+(c*physical_mu[:, None]).abs())
        off = physical_mu*(c0-cutoff)
        oe = gamma(5)*physical_mu*(c0.abs()+abs(cutoff))
        return quadratic_lower_bound(rows, qq, cc, off, h.expand(B, -1), l.expand(B, -1), u.expand(B, -1),
                                     physical_y, c_err=ce, c0_err=oe)[0]

    def step():
        grad = E+tmv(Y)+sc*ac*mu[:, None]
        Xn = torch.maximum(lw, torch.minimum(uw, (X/eta-grad)/(1/eta+2*sc*qs*mu[:, None])))
        extrap = 2*Xn-X
        Y.copy_((Y+eta*(mv(extrap)-hs)).clamp_min(0))
        mu.copy_((mu+eta*sc*(kc+(ac*extrap+qs*extrap*extrap).sum(1))).clamp_min(0))
        X.copy_(Xn); Xs.add_(X); Ys.add_(Y); ms.add_(mu)

    def chunk(count):
        for _ in range(count): step()
        since.add_(count); iteration.add_(count)
        best.copy_(torch.maximum(best, torch.maximum(cert(Y,mu),cert(Ys/since,ms/since))))
        empty.logical_or_(cert(Y,mu,ray=True)>0)
        restart = iteration.remainder(500)==0
        X.copy_(torch.where(restart,Xs/since,X)); Y.copy_(torch.where(restart,Ys/since,Y))
        mu.copy_(torch.where(restart,ms/since,mu))
        Xs.mul_(~restart); Ys.mul_(~restart); ms.mul_(~restart)
        since.mul_(~restart); restarts.add_(restart)

    def reset():
        nonlocal mid, rad, Dr, Dc, hs, S, ac, qs, kc, sc, lw, uw, eta, X, Y, mu, Xs, Ys, ms, E, best, empty, since, iteration, restarts
        _refresh_pricing(rows, pricing)
        mid = .5*(l+u); rad = torch.where(u > l, .5*(u-l), 1.)
        values = rows.val*rad[:, rows.col]
        Dr, Dc = torch.ones_like(h), torch.ones_like(c)
        for _ in range(10):
            rn = product(rows, values, torch.ones_like(c), absolute=True, maximum=True).sqrt()
            cn = product(rows, values, torch.ones_like(h), transpose=True, absolute=True, maximum=True).sqrt()
            rn, cn = torch.where(rn > 0, rn, 1.), torch.where(cn > 0, cn, 1.)
            Dr /= rn; Dc /= cn; values = values/rn[:, rows.row]/cn[:, rows.col]
        matrix.update(values)
        hs = (h-product(rows, rows.val, mid))*Dr; S = rad*Dc
        ac = (2*q*mid+c)*S; qs = q*S*S
        kc = (q*mid*mid+c*mid).sum(1)+c0-cutoff
        sc = 1/torch.maximum(torch.linalg.vector_norm(ac, dim=1),
                              torch.linalg.vector_norm(qs, dim=1)).clamp_min(1e-12)
        v = initial_v.clone()
        norm = lambda a: torch.linalg.vector_norm(a, dim=1)
        for _ in range(30): v = v/norm(v).clamp_min(1e-300)[:, None]; v = tmv(mv(v))
        lw, uw = (l-mid)/S, (u-mid)/S
        cutoff_gradient = norm(ac.abs()+2*qs*torch.maximum(lw.abs(), uw.abs()))
        eta = .9/(norm(v)+(sc*cutoff_gradient)**2).sqrt().clamp_min(1e-12)
        X = c.new_zeros(B, n); Y = c.new_zeros(B, rows.m); mu = c.new_zeros(B)
        if warm is not None and warm[0].shape == X.shape and warm[1].shape == Y.shape:
            X = torch.maximum(lw, torch.minimum(uw, (warm[0]-mid)/S))
            Y = warm[1].clamp_min(0)/Dr; mu = warm[2].clamp_min(0)/sc
        X = torch.maximum(lw, torch.minimum(uw, X))
        Xs, Ys, ms = torch.zeros_like(X), torch.zeros_like(Y), torch.zeros_like(mu)
        E = target*S
        best = c.new_full((B,), -float('inf')); empty = torch.zeros(B, dtype=torch.bool, device=c.device)
        since = torch.zeros((), dtype=torch.long, device=c.device)
        iteration = torch.zeros_like(since); restarts = torch.zeros_like(since)
        best.copy_(torch.maximum(best,cert(Y,mu)))

    reset_graph = step_graph = None
    if capture:
        reset_graph = _capture_reset(reset, c.device)
        step_graph = _capture_reset(lambda: chunk(50), c.device)
        reset_graph.replay()
        torch.cuda.synchronize(c.device)
    preparation_seconds = time.perf_counter()-created

    def run(data, new_target, new_cutoff, budget, new_warm, deadline, cold):
        started = time.perf_counter()
        original, *source = data
        rows.val.copy_(original.val)
        for dst, src in zip((c, c0, h, l, u, q), source): dst.copy_(src)
        target.copy_(new_target); cutoff.fill_(new_cutoff)
        if warm is not None:
            for dst, src in zip(warm, new_warm): dst.copy_(src)
        rows._single_pass = capture
        if capture: reset_graph.replay()
        else: reset()
        if c.is_cuda: torch.cuda.synchronize(c.device)
        setup_seconds = time.perf_counter()-started
        tick = time.perf_counter(); it = 0
        while it < budget:
            if deadline is not None and time.time() >= deadline: break
            count = min(50, budget-it)
            if capture and count == 50: step_graph.replay()
            else: chunk(count)
            it += count
        if capture:
            rows._single_pass = False
            best.copy_(torch.maximum(best,torch.maximum(cert(Y,mu),cert(Ys/since.clamp_min(1),ms/since.clamp_min(1)))))
            empty.logical_or_(cert(Y,mu,ray=True)>0)
        if c.is_cuda: torch.cuda.synchronize(c.device)
        seconds = time.perf_counter()-tick
        metrics = dict(iterations=it,lanes=B,capture=capture,cache_hit=not cold,
                       setup_seconds=setup_seconds+(preparation_seconds if cold and not capture else 0.),
                       capture_seconds=preparation_seconds if cold and capture else 0.,
                       iteration_seconds=seconds,ms_per_iteration=1000*seconds/max(1,it))
        return (best.clone(), ((mid+S*X).clone(), (Dr*Y).clone(), (sc*mu).clone()),
                empty.clone(), int(restarts)), metrics
    run.graphs = reset_graph, step_graph
    return run


@torch.no_grad()
def support(data, target, cutoff, budget=1000, warm=None, capture=True, deadline=None, cache=None):
    """Refresh a fixed support workspace and restart its averaging each call."""
    rows, c, c0, h, l, u, q = data
    B, n = target.shape
    if not 0 < B <= 256 or not 0 <= budget <= 10000 or len(c) != 1:
        raise ValueError('QCQP support capacity exceeded')
    if bool((q < 0).any()): raise ValueError('negative cutoff curvature')
    if warm is not None and (warm[0].shape != target.shape or warm[1].shape != (B, rows.m)):
        warm = None
    capture = bool(capture and c.is_cuda and budget >= 50)
    cache = {} if cache is None or not capture else cache
    key = ('support', rows.row.data_ptr(), rows.col.data_ptr(), rows.m, rows.n,
           c.device, tuple(target.shape), warm is not None)
    cold = key not in cache
    if cold: cache[key] = _workspace(data, target, capture, warm is not None)
    result, support.last_metrics = cache[key](data, target, cutoff, budget, warm, deadline, cold)
    return result
