"""Fixed-capacity batched restarted PDHG with a closed diagonal QP prox.

Lanes can share one table or carry independent boxes. Every step
uses two sparse products and vector operations: O(B*(nnz+m+n)), B <= 256.
No basis, factorisation or direct LP oracle. Search
scaling is heuristic; only the supplied outward certificate is trusted.
"""
from dataclasses import dataclass
import numpy as np
import torch
import time
from .cuda_memory import capture


@dataclass
class QPResult:
    lower: torch.Tensor
    empty: torch.Tensor
    z: torch.Tensor
    y: torch.Tensor
    best_y: torch.Tensor
    restarts: torch.Tensor
    trace: list
    metrics: dict


def product(rows, values, x, transpose=False, absolute=False, maximum=False):
    src, dst, size = (rows.row, rows.col, rows.n) if transpose else (rows.col, rows.row, rows.m)
    v = values.abs() if absolute else values
    terms = v*x[:, src]
    out = x.new_zeros(terms.shape[0], size)
    if maximum:
        return out.scatter_reduce(1, dst[None].expand_as(terms), terms,
                                  reduce='amax', include_self=True)
    return out.index_add(1, dst, terms)


class _MutableMat:
    """Sliced-ELL SpMM on CUDA, CSR SpMM on shared CPU tables.

    A singleton value axis broadcasts over the dense lane block. CPU uses
    native CSR to avoid materializing gather products at every iteration;
    certificate pricing keeps the established ELL reduction order.
    """
    def __init__(self, rows, lanes, cpu_spmm=False):
        from .compiled_crown import Mat
        self.m, self.n = rows.m, rows.n
        self.cpu_spmm = cpu_spmm and not rows.val.is_cuda
        self.indices = torch.stack((rows.row,rows.col)) if self.cpu_spmm else None
        self.csr = None
        self.maps = []
        self.mat = None
        if rows.row.numel():
            # _Reduce's topology builder runs on the host only on a cache miss.
            marker = torch.arange(1, rows.row.numel()+1, dtype=torch.float64)
            self.mat = Mat(coo=(rows.row.cpu(), rows.col.cpu(), marker),
                           shape=(rows.m, rows.n), sparse=True)
            for red in (self.mat.cl, self.mat.cr):
                buckets = []
                for idx, slot in red.buckets:
                    ids = (slot.long()-1).clamp_min(0).to(rows.val.device)
                    mask = (slot != 0).to(rows.val.device)
                    val = rows.val.new_zeros(lanes, *slot.shape)
                    self.maps.append((ids, mask, val))
                    buckets.append((idx.to(rows.val.device), val))
                red.buckets = buckets
                if red.inv is not None: red.inv = red.inv.to(rows.val.device)

    def update(self, values):
        if self.cpu_spmm and self.indices.numel():
            a = torch.sparse_coo_tensor(self.indices,values[0],(self.m,self.n)).coalesce()
            self.csr = a.to_sparse_csr(),a.transpose(0,1).coalesce().to_sparse_csr()
        for ids, mask, val in self.maps:
            val.copy_(torch.where(mask, values[:, ids], 0.))

    def left(self, x):
        if self.csr is not None: return torch.sparse.mm(self.csr[1],x.T).T.contiguous()
        return self.mat.left(x) if self.mat is not None else x.new_zeros(len(x), self.n)

    def right(self, x):
        if self.csr is not None: return torch.sparse.mm(self.csr[0],x.T).T.contiguous()
        return self.mat.right(x) if self.mat is not None else x.new_zeros(len(x), self.m)


class _SharedMat:
    """One CSR per orientation times a dense lane block; no lane-by-nnz buffer."""
    def __init__(self, rows):
        self.m,self.n=rows.m,rows.n
        self.maps=[];self.transforms=[];self.csr=None
        self.mat=self if rows.row.numel() else None
        if self.mat is None:return
        r,c=rows.row.cpu(),rows.col.cpu();device=rows.val.device
        matrices=[]
        for rr,cc,m,n in ((r,c,self.m,self.n),(c,r,self.n,self.m)):
            order=torch.argsort(rr*max(1,n)+cc,stable=True)
            ptr=torch.cat((rr.new_zeros(1),torch.bincount(rr,minlength=m).cumsum(0)))
            values=rows.val.new_zeros(len(order))
            matrices.append(torch.sparse_csr_tensor(ptr.to(device),cc[order].to(device),values,
                                                   size=(m,n),device=device))
            self.transforms.append((order.to(device),values))
        self.csr=tuple(matrices)

    def update(self, values):
        for order,buffer in self.transforms:buffer.copy_(values[0,order])

    def left(self, x):
        return (torch.sparse.mm(self.csr[1],x.T.contiguous()).T.contiguous()
                if self.csr is not None else x.new_zeros(len(x),self.n))

    def right(self, x):
        return (torch.sparse.mm(self.csr[0],x.T.contiguous()).T.contiguous()
                if self.csr is not None else x.new_zeros(len(x),self.m))


class _BatchedMat:
    """Independent lane matrices as one block-diagonal CSR SpMM per product.

    Fixed COO topology is expanded once outside capture. Values refresh in
    place; dense lane vectors flatten to a single column without a Python
    lane loop. Unlike a batch of dense tables, storage remains O(B*nnz).
    """
    def __init__(self, rows, lanes):
        self.m,self.n,self.lanes=rows.m,rows.n,lanes
        self.maps=[];self.transforms=[];self.csr=None
        self.mat=self if rows.row.numel() else None
        if self.mat is None:return
        r,c=rows.row.cpu(),rows.col.cpu();device=rows.val.device
        matrices=[];nz=len(r)
        for rr,cc,m,n in ((r,c,self.m,self.n),(c,r,self.n,self.m)):
            order=torch.argsort(rr*max(1,n)+cc,stable=True)
            counts=torch.bincount(rr,minlength=m).repeat(lanes)
            ptr=torch.cat((r.new_zeros(1),counts.cumsum(0)))
            cols=(cc[order][None]+torch.arange(lanes)[:,None]*n).flatten()
            values=rows.val.new_zeros(lanes*nz)
            matrices.append(torch.sparse_csr_tensor(ptr.to(device),cols.to(device),values,
                            size=(lanes*m,lanes*n),device=device))
            self.transforms.append((order.to(device),values))
        self.csr=tuple(matrices)

    def update(self, values):
        for order,buffer in self.transforms:
            buffer.copy_(values.expand(self.lanes,-1)[:,order].reshape(-1))

    def left(self, x):
        return (torch.sparse.mm(self.csr[1],x.reshape(-1,1)).reshape(self.lanes,self.n)
                if self.csr is not None else x.new_zeros(len(x),self.n))

    def right(self, x):
        return (torch.sparse.mm(self.csr[0],x.reshape(-1,1)).reshape(self.lanes,self.m)
                if self.csr is not None else x.new_zeros(len(x),self.m))


def _matrix(rows, lanes, batched_csr=False):
    if lanes==1:
        # Certificates retain individual coefficients. A duplicate COO
        # coordinate uses the shared ELL reduction instead of pre-summing it.
        keys=rows.row.cpu()*max(1,rows.n)+rows.col.cpu()
        if keys.unique().numel()==keys.numel():return _SharedMat(rows)
    if batched_csr and lanes>1:return _BatchedMat(rows,lanes)
    return _MutableMat(rows,lanes)


def _matrix_bytes(rows, mats):
    size=rows.val.numel()*rows.val.element_size()
    for mat in mats:
        if getattr(mat,'csr',None) is not None:
            size+=sum(a.values().numel()*a.element_size() for a in mat.csr)
        size+=sum(v.numel()*v.element_size() for _,_,v in mat.maps)
    return size


def _pricing(rows, lanes):
    mats = _matrix(rows, lanes), _matrix(rows, lanes)
    if rows.row.numel():
        rows._pricing = tuple(m.mat for m in mats)
        rows._pricing_k = rows.val.new_zeros(lanes, rows.n)
    # Optional double-double refinement runs outside capture, with cached slots.
    rows.slots()
    return mats


def _refresh_pricing(rows, mats):
    mats[0].update(rows.val); mats[1].update(rows.val.abs())
    if rows.row.numel():
        rows._pricing_k.zero_().index_add_(1, rows.col, (rows.val != 0).to(rows.val.dtype).expand(len(rows._pricing_k), -1))


def _capture_reset(reset, device):
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream): reset()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    capture(graph, reset)
    return graph


def _qp_workspace(data, checkpoint, capture, certify=None, certify_ray=None, has_convergence=False, iteration_scheme="average", *, persistent=False, dual_mask=None, valid_mask=None, omega_multipliers=None):
    """Persistent input, reset and iteration graphs for one immutable COO pattern."""
    from nlp_solver.rigorous_dual import Rows, certify_infeasible
    halpern = iteration_scheme == "halpern"
    from .lockstep_dual import CompiledCertifier
    created = time.perf_counter()
    native = None
    if not persistent and not data[1].is_cuda and certify is None and certify_ray is None:
        from .cpu_pdhg import kernel,uniform
        if uniform(data):native=kernel()
    original, *tensors = data
    rows = Rows(original.row, original.col, original.val.clone(), original.m, original.n)
    c, c0, h, lower, upper, q = [t.clone() for t in tensors]
    B, n = c.shape; m = rows.m
    shared = all(t.shape[0] == 1 for t in (rows.val,h,lower,upper,q))
    matrix_lanes = 1 if shared else B
    pricing = _pricing(rows, matrix_lanes); matrix = _matrix(rows, matrix_lanes, batched_csr=halpern)
    mv, tmv = matrix.right, matrix.left
    norm = lambda x: torch.linalg.vector_norm(x, dim=1)
    rows._single_pass = True
    warm = (torch.zeros_like(c), h.new_zeros(B,m))
    use_warm = c.new_zeros((),dtype=torch.bool)
    tolerance = c.new_full((B,),-1.)
    initial_v = c.new_tensor(np.random.default_rng(0).standard_normal(n))[None].expand(matrix_lanes, -1).clone()
    if certify is None:
        certify = lambda yy: CompiledCertifier.diagonal_qp(rows, q.expand(B,-1), c, c0, h.expand(B,-1), lower.expand(B,-1), upper.expand(B,-1), yy)
    if certify_ray is None:
        certify_ray = lambda yy: certify_infeasible(rows, h.expand(B,-1), lower.expand(B,-1), upper.expand(B,-1), yy)
    mid = rad = Dr = Dc = hs = S = qs = off = cs = lw = uw = eta = omega = w = y = ws = ys = wl = yl = since = restarts = best_y = best = empty = elast = eprev = iteration = live = lane_iterations = baseline = error = proposal = None
    def plain(yy):
        yp = (Dr*yy).clamp_min(0)
        r = c+(pricing[0].left(yp) if shared else product(rows, rows.val, yp, transpose=True))
        a = torch.maximum(lower, torch.minimum(upper, -r/torch.where(q > 0, 2*q, 1.)))
        terms = torch.where(q > 0, q*a*a+r*a, torch.minimum(r*lower, r*upper))
        return c0-(h*yp).sum(1)+terms.sum(1)

    def kkt(ww, yy, physical=False):
        if physical:
            physical=torch.maximum(lower,torch.minimum(upper,mid+S*ww))
            residual=(pricing[0].right(physical)-h).clamp_min(0)*Dr
            primal=c0+(c*physical+q*physical*physical).sum(1)
        else:
            residual=(mv(ww)-hs).clamp_min(0)
            primal=off+(cs*ww+qs*ww*ww).sum(1)
        if dual_mask is not None: residual[:,-1].mul_(dual_mask)
        return torch.hypot(omega*norm(residual),(primal-plain(yy)).abs())

    def step():
        tp = (eta/omega)[:, None]
        wn = torch.maximum(lw, torch.minimum(uw, (w/tp-(cs+tmv(y)))/(1/tp+2*qs)))
        yn = (y+(eta*omega)[:, None]*(mv(2*wn-w)-hs)).clamp_min(0)
        if dual_mask is not None: yn[:,-1].mul_(dual_mask)
        if halpern:
            # ws/ys hold T(z), wl/yl are the restart anchor. This replaces
            # averaging using exactly the same persistent vector storage.
            k = since.to(w.dtype)
            a1 = ((k+1)/(k+2))[:, None]
            a0 = (1./(k+2))[:, None]
            wp = a1*(2*wn-w)+a0*wl
            yp = a1*(2*yn-y)+a0*yl
            ws.copy_(torch.where(live[:,None],wn,ws))
            ys.copy_(torch.where(live[:,None],yn,ys))
            w.copy_(torch.where(live[:,None],wp,w))
            y.copy_(torch.where(live[:,None],yp,y))
        elif has_convergence or valid_mask is not None:
            y.copy_(torch.where(live[:,None],yn,y))
            w.copy_(torch.where(live[:,None],wn,w))
            ws.add_(torch.where(live[:,None],w,0.)); ys.add_(torch.where(live[:,None],y,0.))
        else:
            y.copy_(yn); w.copy_(wn); ws.add_(w); ys.add_(y)
        since.add_(live); lane_iterations.add_(live)

    def report_device():
        avg = ys/since.clamp_min(1)[:, None]
        for candidate in ((ys,) if halpern else (y, avg)):
            physical = (Dr*candidate).clamp_min(0)
            value = certify(physical)
            use = value > best
            best.copy_(torch.maximum(best, value))
            best_y.copy_(torch.where(use[:, None], physical, best_y))
        current = ys if halpern else y
        for ray in (Dr*current, Dr*(current-yl).clamp_min(0)):
            ray = ray.clamp_min(0)
            ray = ray/ray.abs().amax(1, keepdim=True).clamp_min(1.) if m else ray
            empty.logical_or_(certify_ray(ray))

    def restart_device():
        wa, ya = ws/since.clamp_min(1)[:, None], ys/since.clamp_min(1)[:, None]
        if halpern:
            wc, yc = ws, ys
            ec = kkt(wc, yc)
        else:
            ec, ea = kkt(w, y), kkt(wa, ya)
            use = ea < ec
            wc, yc = torch.where(use[:, None], wa, w), torch.where(use[:, None], ya, y)
            ec = torch.minimum(ec, ea)
        restart = live & ((ec <= .2*elast) | ((ec <= .8*elast) & (ec > eprev)) | (since >= .36*(lane_iterations if persistent else iteration)))
        eprev.copy_(ec)
        dw, dy = norm(wc-wl), norm(yc-yl)
        upd = restart & (dw > 1e-10) & (dy > 1e-10)
        new_omega = torch.exp(.5*torch.log(dy.clamp_min(1e-300)/dw.clamp_min(1e-300))+.5*omega.log())
        omega.copy_(torch.where(upd, new_omega, omega))
        w.copy_(torch.where(restart[:, None], wc, w)); y.copy_(torch.where(restart[:, None], yc, y))
        wl.copy_(torch.where(restart[:, None], w, wl)); yl.copy_(torch.where(restart[:, None], y, yl))
        enew = kkt(ws, ys) if halpern else kkt(w, y)
        error.copy_(kkt(ws if halpern else w,ys if halpern else y,physical=True) if persistent else enew)
        if persistent:
            current=ws if halpern else w
            pe,we=kkt(proposal,best_y/Dr,physical=True),kkt(current,best_y/Dr,physical=True)
            proposal.copy_(torch.where((we<pe)[:,None],current,proposal))
            if not halpern:
                ae=kkt(wa,best_y/Dr,physical=True)
                proposal.copy_(torch.where((ae<torch.minimum(pe,we))[:,None],wa,proposal))
        elast.copy_(torch.where(restart, enew, elast)); eprev.copy_(torch.where(restart, enew, eprev))
        if not halpern:
            ws.mul_((~restart)[:, None]); ys.mul_((~restart)[:, None])
        since.mul_(~restart)
        restarts.add_(restart)

    def chunk(count):
        for _ in range(count): step()
        iteration.add_(count)
        report_device()
        restart_device()
        if has_convergence:
            # Resolve KKT error relative to tightening already obtained.
            # Zero gain requires zero KKT error: a box-width floor can stop
            # an uninformative lane while it is still physically infeasible.
            gain = (best-baseline).clamp_min(0)
            done = empty | (torch.isfinite(error) & (error <= tolerance*gain))
            live.logical_and_(~((tolerance >= 0) & done))

    def reset():
        nonlocal mid, rad, Dr, Dc, hs, S, qs, off, cs, lw, uw, eta, omega, w, y, ws, ys, wl, yl, since, restarts, best_y, best, empty, elast, eprev, iteration, live, lane_iterations, baseline, error, proposal
        _refresh_pricing(rows, pricing)
        mid = .5*(lower+upper)
        rad = torch.where(upper > lower, .5*(upper-lower), 1.)
        values = rows.val*rad[:, rows.col]
        Dr, Dc = h.new_ones(matrix_lanes, m), lower.new_ones(matrix_lanes, n)
        for i in range(11):
            ones_c, ones_r = torch.ones_like(Dc), torch.ones_like(Dr)
            rn = product(rows, values, ones_c, absolute=True, maximum=i < 10).sqrt()
            cn = product(rows, values, ones_r, transpose=True, absolute=True, maximum=i < 10).sqrt()
            rn, cn = torch.where(rn > 0, rn, 1.), torch.where(cn > 0, cn, 1.)
            Dr /= rn; Dc /= cn
            values = values/rn[:, rows.row]/cn[:, rows.col]
        # Segmented sparse reductions avoid fp64 atomic contention on columns
        # with many ancestor rows. _Reduce groups padding within 4*nnz.
        matrix.update(values)
        hs = (h-product(rows, rows.val, mid))*Dr
        S = rad*Dc
        qs = q*S*S; off = c0+(c*mid+q*mid*mid).sum(1)
        cs = (c+2*q*mid)*S
        lw, uw = (lower-mid)/S, (upper-mid)/S
        # Same deterministic vector as the CPU alignment reference; no GPU RNG.
        v = initial_v.clone()
        for _ in range(40):
            v = v/norm(v).clamp_min(1e-300)[:, None]; v = tmv(mv(v))
        eta = .95/norm(v).sqrt().clamp_min(1e-12)
        hn, cn = norm(hs), norm(cs)
        if dual_mask is not None:
            without = norm(hs[:,:-1])
            hn = torch.where(dual_mask,hn,without)
        omega = torch.where(hn > 1e-10, cn.clamp_min(1e-10)/hn.clamp_min(1e-10), 1.)
        if omega_multipliers is not None:omega = omega*omega_multipliers
        w = torch.maximum(lw, torch.minimum(uw, torch.zeros_like(c)))
        y = h.new_zeros(B,m)
        w = torch.where(use_warm, torch.maximum(lw, torch.minimum(uw, (warm[0]-mid)/S)), w)
        y = torch.where(use_warm, warm[1].clamp_min(0)/Dr, y)
        if dual_mask is not None: y[:,-1].mul_(dual_mask)
        ws, ys = (w.clone(), y.clone()) if halpern else (torch.zeros_like(w), torch.zeros_like(y))
        wl, yl = w.clone(), y.clone()
        since = torch.zeros(B, dtype=torch.long, device=c.device)
        restarts = torch.zeros_like(since)
        best_y = (Dr*y).clone(); best = certify(best_y)
        empty = torch.zeros(B, dtype=torch.bool, device=c.device)
        elast, eprev = kkt(w, y), kkt(w, y)
        iteration = torch.zeros((), dtype=torch.long, device=c.device)
        live = torch.ones(B,dtype=torch.bool,device=c.device)
        if valid_mask is not None:live.copy_(valid_mask)
        lane_iterations = torch.zeros(B,dtype=torch.long,device=c.device)
        baseline = plain(torch.zeros_like(y)) if has_convergence or persistent else c0.new_zeros(B)
        error = kkt(w,y,physical=True) if persistent else elast.clone()
        proposal = w.clone() if persistent else None
        report_device()

    reset_graph = step_graph = None
    if capture:
        reset_graph = _capture_reset(reset, c.device)
        step_graph = _capture_reset(lambda: chunk(checkpoint), c.device)
        reset_graph.replay()
        torch.cuda.synchronize(c.device)
    preparation_seconds = time.perf_counter()-created

    def run(data, budget, stop, warm_input, deadline, cold=False, convergence=None, record_search=False, *, resume=False):
        started = time.perf_counter()
        setup_seconds = 0.
        if not resume:
            original, *source = data
            rows.val.copy_(original.val)
            for dst, src in zip((c, c0, h, lower, upper, q), source): dst.copy_(src)
            tolerance.fill_(-1. if convergence is None else convergence)
            use_warm.fill_(warm_input is not None)
            if warm_input is not None:
                for dst, src in zip(warm, warm_input): dst.copy_(src)
            rows._single_pass = True
            if capture: reset_graph.replay()
            else: reset()
            if c.is_cuda: torch.cuda.synchronize(c.device)
            setup_seconds = time.perf_counter()-started
            if native is not None:
                from .cpu_pdhg import run as cpu_run
                matrix_bytes = _matrix_bytes(rows,(*pricing,matrix))
                metrics=dict(lanes=B,capture=False,checkpoint=checkpoint,shared_table=shared,
                             matrix_lanes=matrix_lanes,matrix_bytes=matrix_bytes,cache_hit=not cold,
                             setup_seconds=setup_seconds+(preparation_seconds if cold else 0.),capture_seconds=0.)
                return cpu_run(native,(rows,c,c0,h,lower,upper,q),budget,checkpoint,
                               None if warm_input is None else warm,deadline,convergence,metrics,iteration_scheme,record_search)
        start_steps=lane_iterations.clone() if persistent else None
        trace = []
        def record(it):
            trace.append(dict(iteration=it, best=best.clone(), empty=empty.clone(), restarts=restarts.clone()))
        record(0)
        tick = time.perf_counter(); it = 0
        # Time device steps independently of the final certificate extraction.
        events=None
        if persistent and c.is_cuda:
            events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            events[0].record()
        while it < budget:
            if deadline is not None and time.time() >= deadline: break
            count = min(checkpoint, budget-it)
            if capture and count == checkpoint: step_graph.replay()
            else: chunk(count)
            it += count; record(it)
            if stop is not None and bool(((best >= stop) | empty).all()): break
            if convergence is not None and (not bool(live.any()) or (shared and bool(empty.any()))): break
        if events is not None:events[1].record()
        rows._single_pass = False
        report_device()
        best.copy_(torch.maximum(best, certify(best_y)))
        trace[-1].update(best=best.clone(), empty=empty.clone(), restarts=restarts.clone())
        if c.is_cuda: torch.cuda.synchronize(c.device)
        seconds = time.perf_counter()-tick
        matrix_bytes = _matrix_bytes(rows,(*pricing,matrix))
        used_steps=lane_iterations-start_steps if persistent else lane_iterations
        metrics = dict(iterations=it, lanes=B, capture=capture, checkpoint=checkpoint,
                       shared_table=shared, matrix_lanes=matrix_lanes, matrix_bytes=matrix_bytes, cpu_fused=False,
                       lane_iterations=used_steps.tolist(), active_lane_steps=int(used_steps.sum()),
                       lane_steps=B*it, converged=int(((~live) if valid_mask is None else ((~live)&valid_mask)).sum()),
                       cache_hit=not cold, setup_seconds=setup_seconds+(preparation_seconds if cold and not capture else 0.),
                       capture_seconds=preparation_seconds if cold and capture else 0., iteration_seconds=seconds,
                       ms_per_iteration=1000*seconds/max(1,it))
        if events is not None:metrics['step_seconds']=events[0].elapsed_time(events[1])/1000
        result = QPResult(best.clone(), empty.clone(), (mid+S*(ws if halpern else w)).clone(), (Dr*(ys if halpern else y)).clamp_min(0).clone(),
                          best_y.clone(), restarts.clone(), trace, metrics)
        if persistent:
            result.error, result.baseline = torch.minimum(error,kkt(ws if halpern else w,best_y/Dr,physical=True)).clone(), baseline.clone()
            # For a unit support, its dual value proposes the target endpoint.
            # Retire only if this *whole* primal/dual pair also passes KKT;
            # proposals never create bounds or alter active search iterates.
            target=c.abs().argmax(1)
            sign=c.gather(1,target[:,None]).squeeze(1)
            support=((c!=0).sum(1)==1) & (sign.abs()==1) & (c0==0)
            if dual_mask is not None:support &= dual_mask
            point=sign*plain(best_y/Dr)
            position=(point-mid.expand(B,-1).gather(1,target[:,None]).squeeze(1))/S.expand(B,-1).gather(1,target[:,None]).squeeze(1)
            wc=proposal.clone();wc.scatter_(1,target[:,None],position[:,None])
            wc=torch.maximum(lw,torch.minimum(uw,wc))
            pe=kkt(proposal,best_y/Dr,physical=True)
            ce=kkt(wc,best_y/Dr,physical=True)
            use=support & (ce<pe)
            result.z=torch.maximum(lower,torch.minimum(upper,result.z))
            result.candidate_z=torch.maximum(lower,torch.minimum(upper,mid+S*torch.where(use[:,None],wc,proposal)))
            result.candidate_error=torch.where(use,ce,pe)
            from nlp_solver.rigorous_dual import gamma
            reach=torch.maximum(lower.abs(),upper.abs())
            mag=(pricing[1].right(reach)+h.abs())*Dr
            gapmag=c0.abs()+(c.abs()*reach+q.abs()*reach*reach).sum(1)
            result.error_rounding=gamma(2*(len(rows.row)+m+n)+16)*(omega*norm(mag)+gapmag)
        return result
    if persistent:
        def snapshot():
            return dict(omega=omega,w=w,y=y,ws=ws,ys=ys,wl=wl,yl=yl,since=since,
                        restarts=restarts,best_y=best_y,best=best,empty=empty,elast=elast,
                        eprev=eprev,iteration=iteration,live=live,lane_iterations=lane_iterations,
                        baseline=baseline,error=error,proposal=proposal)
        def restore(state, indices):
            for name,dst in snapshot().items():
                src=state[name]
                if src.ndim==0:dst.copy_(src)
                else:dst[:len(indices)].copy_(src[indices])
        run.snapshot,run.restore=snapshot,restore
    # Keep both graphs alive: replayed chunks read reset graph's output allocations.
    run.graphs = reset_graph, step_graph
    return run


@torch.no_grad()
def solve(rows, c, c0, h, lower, upper, q, budget=1000, checkpoint=25,
          certify=None, certify_ray=None, capture=True, stop=None, warm=None, deadline=None, cache=None,
          convergence=None, iteration_scheme="average", implied_box=False, record_search=False):
    """Restarted solve; a caller-owned cache reuses graphs, never dual iterates.

    One cache belongs to one immutable COO table. Values, box bounds and
    objectives are refreshed on every call. Optional warm states are only
    for restarted root rounds. Returned tensors own their storage.
    record_search retains untrusted CPU checkpoint values only for offline
    diagnostics; it is off by default and never supplies a certificate.
    """
    if iteration_scheme not in ("average", "halpern"):
        raise ValueError("iteration_scheme must be average or halpern")
    if convergence is not None and not 0 <= convergence < 1:
        raise ValueError('convergence must be a relative tolerance in [0,1)')
    # Any input (including either warm vector) can supply the lane axis.
    # Keep singleton tables/boxes intact so the shared scaling path survives.
    batches = (rows.val, c, c0, h, lower, upper, q) + (() if warm is None else tuple(warm))
    B = max(len(t) for t in batches)
    if any(len(t) not in (1, B) for t in batches):
        raise ValueError('QP input batches must be singleton or match the lane count')
    c, c0 = c.expand(B, -1), c0.expand(B)
    box_info = None
    if implied_box:
        if certify is not None or certify_ray is not None:
            raise ValueError('implied box requires the current-table certifiers')
        from .implied_box import implied_box as tighten_box
        lower, upper, box_info = tighten_box(rows, h, lower, upper)
    capture = bool(capture and c.is_cuda and budget >= checkpoint)
    if not 0 < len(c) <= 256 or budget < 0 or checkpoint < 1:
        raise ValueError('QP work exceeds fixed lane/iteration capacity')
    if any(t.dtype != torch.float64 for t in (c, c0, h, lower, upper, q, rows.val)):
        raise TypeError('QP search/certificates require binary64')
    if bool((q < 0).any()): raise ValueError('negative QP curvature')
    cache = {} if cache is None or not capture else cache
    # The caller owns one cache per immutable table. Pointer identities prevent
    # accidental reuse for a different sparsity pattern of the same dimensions.
    key = ('qp', rows.row.data_ptr(), rows.col.data_ptr(), rows.m, rows.n,
           c.device, tuple(c.shape), tuple(rows.val.shape), tuple(h.shape), tuple(lower.shape), tuple(upper.shape), tuple(q.shape), checkpoint, capture, certify, certify_ray, convergence is not None, iteration_scheme)
    data = rows, c, c0, h, lower, upper, q
    cold = key not in cache
    if cold: cache[key] = _qp_workspace(data, checkpoint, capture, certify, certify_ray, convergence is not None, iteration_scheme)
    result = cache[key](data, budget, stop, warm, deadline, cold, convergence, record_search)
    if box_info is not None:
        result.metrics.update(implied_box_passes=box_info['passes'],
                              implied_box_empty_hints=box_info['empty_hint'].tolist())
    return result


def persistent_qp(data, *, checkpoint=25, capture=True, warm=None, objective_cutoff=False, iteration_scheme="average", implied_box=False, omega_multipliers=None, objective_lanes=1):
    """One warm shared-table batch. Checkpoints do not reset search state.

    Refreshes reset scaling, omega and restart history; physical z/y survive.
    Existing solve callers retain their original behavior.
    """
    return PersistentQP(data, checkpoint=checkpoint, capture=capture, warm=warm, objective_cutoff=objective_cutoff, iteration_scheme=iteration_scheme, implied_box=implied_box, omega_multipliers=omega_multipliers, objective_lanes=objective_lanes)


def _check_shared_lp(data, checkpoint, objective_cutoff):
    r,c,c0,h,l,u,q=data
    if not len(c) or checkpoint<1 or (objective_cutoff and r.m<1):
        raise ValueError('persistent LP requires lanes, a checkpoint, and a cutoff row when requested')
    if any(t.dtype!=torch.float64 for t in (r.val,c,c0,h,l,u,q)):
        raise TypeError('persistent LP certificates require binary64')
    if any(len(t)!=1 for t in (r.val,h,l,u,q)) or bool(q.any()):
        raise ValueError('persistent lanes require one shared linear LP table')


class PersistentQP:
    def __init__(self, data, *, checkpoint=25, capture=True, warm=None, objective_cutoff=False, iteration_scheme="average", implied_box=False, omega_multipliers=None, objective_lanes=1):
        self.checkpoint, self.capture = checkpoint, bool(capture and data[1].is_cuda)
        self.serial_steps = 0
        if iteration_scheme not in ("average","halpern"):raise ValueError("invalid iteration scheme")
        self.iteration_scheme=iteration_scheme
        self.implied_box=implied_box
        self.objective_cutoff=objective_cutoff
        self.objective_lanes=objective_lanes
        self.omega_multipliers=omega_multipliers
        if omega_multipliers is not None:
            self.omega_multipliers=torch.as_tensor(omega_multipliers,device=data[1].device,dtype=torch.float64).expand(len(data[1])).clone()
            if not bool((torch.isfinite(self.omega_multipliers)&(self.omega_multipliers>0)).all()):
                raise ValueError('omega multipliers must be finite and positive')
        self.full_omega=self.omega_multipliers
        self.mask=None
        self.run = None
        self.workspaces={}
        self.full_count=len(data[1])
        self.price_key=None
        self.refresh(data, warm=warm)

    def _workspace(self, data, warm=None):
        r,c,c0,h,l,u,q=data
        count=len(c)
        key=(count,r.row.data_ptr(),r.col.data_ptr(),r.m,r.n,r.val.device,self.capture)
        if not self.capture:self.workspaces.clear()
        elif key not in self.workspaces:
            # Keep only full-table replay and the current compact batch.
            # Exact lane counts avoid arithmetic on retired or padding slots.
            self.workspaces={k:v for k,v in self.workspaces.items() if k[0]==self.full_count}
        if key not in self.workspaces:
            mask=None if self.mask is None else self.mask.clone()
            run=_qp_workspace(data,self.checkpoint,self.capture,persistent=True,dual_mask=mask,iteration_scheme=self.iteration_scheme,omega_multipliers=self.omega_multipliers)
            self.workspaces[key]=run,mask
        run,mask=self.workspaces[key]
        if mask is not None:mask.copy_(self.mask)
        self.run=run
        self.run_data=data
        self.capacity=count
        return self._slice(run(data,0,None,warm,None))

    def _slice(self, result):
        count=len(self.data[1]);capacity=self.capacity
        for name,tensor in vars(result).items():
            if torch.is_tensor(tensor) and tensor.ndim and len(tensor)==capacity:
                setattr(result,name,tensor[:count].clone())
        for record in result.trace:
            for name,tensor in record.items():
                if torch.is_tensor(tensor) and tensor.ndim and len(tensor)==capacity:
                    record[name]=tensor[:count].clone()
        result.metrics.update(lanes=count,capacity=capacity)
        result.metrics['lane_iterations']=result.metrics['lane_iterations'][:count]
        return result

    def refresh(self, data, *, warm=None):
        _check_shared_lp(data,self.checkpoint,self.objective_cutoff)
        if warm is None and hasattr(self, 'result'):
            warm=self.result.z,self.result.y
        if warm is not None:
            if warm[0].shape!=data[1].shape or warm[1].shape!=(len(data[1]),data[0].m):
                raise ValueError('warm direction/row shapes changed; remap identities and zero-extend new rows')
            warm=tuple(t.to(device=data[1].device,dtype=torch.float64) for t in warm)
        if self.full_omega is not None and len(data[1])==len(self.full_omega):
            self.omega_multipliers=self.full_omega
        self.box_info=None
        if self.implied_box:
            from .implied_box import implied_box as tighten_box
            r,c,c0,h,l,u,q=data
            # One box is shared by every direction. Lane 0 ignores the last
            # (cutoff) row, so only the intersection of their valid row sets
            # may tighten this box. Other rows are constraints for all lanes.
            box_h=h
            if self.objective_cutoff:
                box_h=h.clone();box_h[:,-1]=torch.inf
            l,u,self.box_info=tighten_box(r,box_h,l,u)
            data=r,c,c0,h,l,u,q
        previous=getattr(self,'data',None)
        self.data=data
        self.mask=None
        if self.objective_cutoff:
            self.mask=torch.ones(len(data[1]),dtype=torch.bool,device=data[1].device)
            self.mask[:self.objective_lanes]=False
        rows=data[0]
        pattern=(rows.row.data_ptr(),rows.col.data_ptr(),rows.m,rows.n,rows.val.device)
        if pattern!=self.price_key:
            self.pricing=_pricing(rows,1)
            self.pricing_k=getattr(rows,'_pricing_k',None)
            self.workspaces.clear()
            self.price_key=pattern
        elif rows.row.numel():
            rows._pricing=tuple(m.mat for m in self.pricing)
            rows._pricing_k=self.pricing_k
            if previous is not None:rows._slots=previous[0]._slots
        _refresh_pricing(rows,self.pricing)
        rows._single_pass=True
        self.native=None
        import shutil
        if not data[1].is_cuda and shutil.which('g++') is not None:
            from .cpu_pdhg_session import Session
            self.native=Session(data,warm,self.mask,self.iteration_scheme)
            if self.omega_multipliers is not None:
                _replacement_library().m2_omega(self.native.ptr,self.omega_multipliers.contiguous().data_ptr())
            self.result=self._native_result(0)
        else:self.result=self._workspace(data,warm)
        self.result.metrics['serial_steps']=self.serial_steps
        if self.native is None:self._rounding()
        if self.box_info is not None:
            self.result.metrics.update(implied_box_passes=self.box_info['passes'],
                                       implied_box_empty_hints=self.box_info['empty_hint'].tolist())
        return self.result

    def _rounding(self):
        from nlp_solver.rigorous_dual import dual_lower_bound
        r,c,c0,h,l,u,q=self.data
        lb,E,_=dual_lower_bound(r,c,c0,h.expand(len(c),-1),l.expand(len(c),-1),u.expand(len(c),-1),self.result.best_y)
        self.result.lower=torch.maximum(self.result.lower,lb)
        self.result.rounding=E

    def compact(self, indices):
        count=len(self.data[1])
        if not len(indices):raise ValueError('compact requires at least one active lane')
        if len(indices)==count and torch.equal(indices,torch.arange(count,device=indices.device)):return
        import copy
        old=self.result
        r,c,c0,h,l,u,q=self.data
        data=r,c[indices].contiguous(),c0[indices].contiguous(),h,l,u,q
        if self.mask is not None:self.mask=self.mask[indices]
        if self.omega_multipliers is not None:self.omega_multipliers=self.omega_multipliers[indices]
        if self.native is not None:self.native.compact(indices,data)
        else:
            state={name:t.clone() for name,t in self.run.snapshot().items()}
            self.data=data
            self._workspace(data)
            self.run.restore(state,indices)
        self.data=data
        self.result=copy.copy(old)
        for name,tensor in vars(old).items():
            if torch.is_tensor(tensor) and tensor.ndim and len(tensor)==count:
                setattr(self.result,name,tensor[indices].clone())
        self.result.metrics=dict(old.metrics,lanes=len(indices),iterations=0,lane_steps=0,active_lane_steps=0)

    def replace(self, slots, c, *, warm=None):
        """Replace direction identities, resetting only those physical slots.

        Surviving iterates, weights, anchors and iteration counters are kept.
        A new direction defaults to the objective witness and zero duals;
        callers may supply a column-keyed physical warm memo instead.
        """
        slots=slots.to(device=self.data[1].device,dtype=torch.long)
        if not len(slots):return self.result
        if c.shape!=(len(slots),self.data[0].n):raise ValueError('replacement cost shape')
        r,cc,c0,h,l,u,q=self.data
        cc=cc.clone();cc[slots]=c
        c0=c0.clone();c0[slots]=0.
        z,y=self.result.z.clone(),self.result.y.clone()
        z[slots]=self.result.z[:1] if warm is None else warm[0]
        y[slots]=0. if warm is None else warm[1]
        data=r,cc,c0,h,l,u,q
        if self.native is not None:
            self.native.inputs[3:5]=[cc.contiguous(),c0.contiguous()]
            self.data=data
            extension=_replacement_library()
            zz,yy=z[slots].contiguous(),y[slots].contiguous()
            extension.m2_replace(self.native.ptr,len(slots),slots.contiguous().data_ptr(),
                cc.data_ptr(),c0.data_ptr(),zz.data_ptr(),yy.data_ptr())
            if self.omega_multipliers is not None:
                extension.m2_replace_omega(self.native.ptr,len(slots),slots.contiguous().data_ptr(),self.omega_multipliers.contiguous().data_ptr())
            self.result=self._native_result(0)
        else:
            state={name:t.clone() for name,t in self.run.snapshot().items()}
            self.data=data
            self._workspace(data,(z,y))
            fresh=self.run.snapshot()
            for name,tensor in fresh.items():
                if tensor.ndim:state[name][slots]=tensor[slots]
            self.run.restore(state,torch.arange(len(cc),device=cc.device))
            self.result=self._slice(self.run(self.run_data,0,None,None,None,resume=True))
            self._rounding()
        self.result.metrics['serial_steps']=self.serial_steps
        return self.result

    def _native_result(self, count):
        from nlp_solver.rigorous_dual import dual_lower_bound, certify_infeasible
        tick = time.perf_counter()
        work_tick=time.process_time()
        z,y,by,ray,candidate,stats = self.native.advance(count,self.checkpoint)
        step_seconds=time.process_time()-work_tick
        rows,c,c0,h,l,u,q = self.data
        lower,E,_ = dual_lower_bound(rows,c,c0,h.expand(len(c),-1),l.expand(len(c),-1),u.expand(len(c),-1),by)
        empty = certify_infeasible(rows,h.expand(len(c),-1),l.expand(len(c),-1),u.expand(len(c),-1),ray)
        metrics = dict(iterations=count, lanes=len(c), lane_steps=count*len(c), active_lane_steps=count*len(c),
                       setup_seconds=0.,capture_seconds=0.,iteration_seconds=time.perf_counter()-tick,step_seconds=step_seconds,cpu_fused=True)
        out = QPResult(lower,empty,z.clone(),y.clone(),by.clone(),stats[:,2].long(),
                       [dict(iteration=count,best=lower.clone())],metrics)
        out.rounding=E
        out.error,out.baseline = stats[:,0].clone(),stats[:,1].clone()
        out.error_rounding = stats[:,4].clone()
        out.candidate_z,out.candidate_error=candidate.clone(),stats[:,5].clone()
        return out

    def advance(self, count=None, *, deadline=None):
        count = self.checkpoint if count is None else count
        if count<0:raise ValueError('negative persistent LP step count')
        if deadline is not None and time.time() >= deadline: count = 0
        self.result = (self._native_result(count) if self.native is not None else
                       self._slice(self.run(self.run_data,count,None,None,deadline,resume=True)))
        self.serial_steps += self.result.metrics['iterations']
        self.result.metrics['serial_steps'] = self.serial_steps
        if self.native is None:self._rounding()
        if self.box_info is not None:
            self.result.metrics.update(implied_box_passes=self.box_info['passes'],
                                       implied_box_empty_hints=self.box_info['empty_hint'].tolist())
        return self.result


def _replacement_library():
    """Optional native direction reset; the original CPU backend is untouched."""
    import ctypes
    import hashlib
    import subprocess
    import tempfile
    import fcntl
    from pathlib import Path
    from .cpu_pdhg_session import library
    cached=getattr(library,'m2_replacement',None)
    if cached is not None:return cached
    source=Path(__file__).with_name('cpu_pdhg_session.cpp')
    code=r'''#include "cpu_pdhg_session.cpp"
extern "C" void m2_replace(void* ptr,int B,const int64_t* ids,const double* obj,const double* off,const double* z,const double* y){
 auto& s=*static_cast<Session*>(ptr);auto& t=s.table;s.obj=obj;s.c0=off;
 for(int b=0;b<B;b++){int j=ids[b];bool ignore=s.lanes[j].ignore_cut,halp=s.lanes[j].halpern;
  s.lanes[j]=Lane(t,obj+j*t.n,off[j],z+b*t.n,y+b*t.m,ignore,halp);}
}
static void multiply_omega(Session& s,int j,double factor){
 auto& v=s.lanes[j];auto& t=s.table;v.omega*=factor;
 v.error=v.kkt(t,s.obj+j*t.n,s.c0[j],v.w,v.y);
 v.elast=v.eprev=v.search_kkt(t,s.obj+j*t.n,s.c0[j],v.w,v.y);
}
extern "C" void m2_omega(void* ptr,const double* factors){
 auto& s=*static_cast<Session*>(ptr);for(size_t j=0;j<s.lanes.size();j++)multiply_omega(s,j,factors[j]);
}
extern "C" void m2_replace_omega(void* ptr,int B,const int64_t* ids,const double* factors){
 auto& s=*static_cast<Session*>(ptr);for(int b=0;b<B;b++)multiply_omega(s,ids[b],factors[ids[b]]);
}
'''
    digest=hashlib.sha256(code.encode()+source.read_bytes()+source.with_name('cpu_pdhg.cpp').read_bytes()).hexdigest()[:20]
    base=Path(tempfile.gettempdir())/('crown_m2_'+digest)
    with base.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not base.with_suffix('.so').exists():
            base.with_suffix('.cpp').write_text(code)
            subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC','-ffp-contract=off',
                            '-I',str(source.parent),str(base.with_suffix('.cpp')),'-o',str(base.with_suffix('.so'))],check=True,capture_output=True)
    lib=ctypes.CDLL(str(base.with_suffix('.so')))
    lib.m2_replace.argtypes=[ctypes.c_void_p,ctypes.c_int]+[ctypes.c_void_p]*5
    lib.m2_omega.argtypes=[ctypes.c_void_p]*2
    lib.m2_replace_omega.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    library.m2_replacement=lib
    return lib
