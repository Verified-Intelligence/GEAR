"""Fixed-capacity PDHG node queue. Tensor masks retire/refill individual slots.

Only reset and checkpoint graphs are captured, once per power-of-two capacity
and queue bucket. No graph, Python lane loop or host decision on a refill.
The scaling and restart constants are identical to restarted_qp.
"""
import time
import numpy as np
import torch
from .restarted_qp import product, _matrix, _pricing, _refresh_pricing, _capture_reset, QPResult, _SharedMat
from .lockstep_dual import CompiledCertifier


def item_tables(tables, items, ids):
    """Per-item (rows,c,c0,h,l,u,q) of items `ids` of a box-table queue. Transient: callers chunk it.

    Item i is (owner, j, sign): j<0 is the owner's node lane, j>=0 the signed direction sign*e_j.
    """
    from nlp_solver.rigorous_dual import Rows
    r,c,c0,h,l,u,q=tables;o,j,s=(v[ids] for v in items);o=o.long()
    g=(torch.arange(r.n,device=c.device)[None]==j.long()[:,None]).to(c.dtype)*s.to(c.dtype)[:,None]
    return (Rows(r.row,r.col,r.val[o],r.m,r.n),torch.where((j>=0)[:,None],g,c[o]),torch.where(j>=0,0.,c0[o]),h[o],l[o],u[o],q[o])


def _workspace(data, capacity, checkpoint, capture, iteration_scheme="average", pt=None, dual_accuracy=None, support=None, shared_support=False, probe_kind=None, probe_account=False, queue=None):
    """Capturable slot queue.

    `data` is one table per BOX (G rows). Items index it through `owner`. Without `queue` every box is
    one item (owner=arange, the classic layout). With `queue`=N the items are int32 triples
    (owner, j, sign): j<0 is the box's own node lane, j>=0 the signed direction sign*e_j, whose
    objective is implicit and is gathered into the slot buffers at refill. Only the slots are heavy.
    """
    halpern=iteration_scheme=="halpern"
    from nlp_solver.rigorous_dual import Rows, certify_infeasible
    original,*source=data
    qr=Rows(original.row,original.col,original.val.clone(),original.m,original.n)
    c,c0,h,l,u,q=[v.clone() for v in source]
    G,n=c.shape;m=qr.m;B=capacity
    tables=queue is not None;N=G if queue is None else queue
    dev=c.device
    if tables and probe_kind is None:probe_kind=torch.zeros(N,dtype=torch.bool,device=dev)
    if probe_kind is not None:probe_kind=probe_kind.clone()  # static buffer: run() refills it, so one graph serves every kind pattern
    if pt is not None:
        cutoff_row,*pt_source=pt
        objective=tuple(v.clone() for v in pt_source)  # per box: (c,c0,q,T)
        row_terms=int(torch.bincount(qr.row.cpu(),minlength=m).max())
    dual_control=None if dual_accuracy is None else tuple(v.clone() for v in dual_accuracy)
    support_tol=None if support is None else support.clone()
    if support is not None:
        row_terms=int(torch.bincount(qr.row.cpu(),minlength=m).max()) if m else 0
    # Box tables exist once per box; the queue holds three ints per item. Warm
    # states and outputs are per item.
    owner=torch.arange(N,device=dev,dtype=torch.int32)%G;jj=owner.new_full((N,),-1);sg=owner.new_zeros(N)
    z0=c.new_zeros(N,n);y0=h.new_zeros(N,m);target=c0.new_zeros(N);accuracy=c0.new_zeros(N);valid=c0.new_zeros(N,dtype=torch.bool)
    seed=c.new_tensor(np.random.default_rng(0).standard_normal(n))[None].expand(G,-1)
    ar=torch.arange(B,device=dev);arn=torch.arange(n,device=dev)
    rows=Rows(qr.row,qr.col,qr.val[:1].expand(B,-1).clone(),m,n)
    price=_pricing(rows,B)
    # One CSR per slot: measured faster than a grouped block CSR for G>=4 boxes (A158).
    matrix=_SharedMat(rows) if shared_support else _matrix(rows,B,batched_csr=halpern)
    rows._single_pass=True;qr._single_pass=True
    norm=lambda x:torch.linalg.vector_norm(x,dim=1)
    cert=lambda r,cc,off,hh,ll,uu,qq,yy:CompiledCertifier.diagonal_qp(r,qq,cc,off,hh,ll,uu,yy)
    # Stable output allocations are essential: both graphs refer to them.
    best=c0.new_zeros(N);best_y=y0.clone();zout=z0.clone();yout=y0.clone()
    empty=c0.new_zeros(N,dtype=torch.bool);steps=c0.new_zeros(N,dtype=torch.long)
    restarts=steps.clone();reason=steps.clone()
    cursor=steps.new_zeros(());ids=ar.clone();bi=ar.clone();live=valid.new_zeros(B)
    w=c.new_zeros(B,n);y=h.new_zeros(B,m);ws=w.clone();ys=y.clone();wl=w.clone();yl=y.clone()
    since=steps.new_zeros(B);it=since.clone();nr=since.clone()
    omega=c0.new_zeros(B);elast=omega.clone();eprev=omega.clone();previous_w=w.clone();previous_y=y.clone()
    # Slot buffers: the item's own objective (gathered from its owner or implicit) and the scalings
    # that depend on it.
    cI=w.clone();c0I=omega.clone();cs=w.clone();off=omega.clone()
    if pt is not None:slot_pt=(w.clone(),omega.clone(),w.clone(),omega.clone())
    mid=rad=Dr=S=hs=lw=uw=scaled=qs=eta=None
    serial_probe=steps.new_zeros(())
    half_slots=steps.new_full((),B//2)  # device constants: no H2D copy in capture
    competitive_probe=steps.new_zeros(());competitive_node=steps.new_zeros(())
    probe_slots=steps.new_zeros(());node_slots=steps.new_zeros(())

    def refresh():
        rows.val.copy_(qr.val[bi]);_refresh_pricing(rows,price);matrix.update(scaled[bi])

    def plain(ww,yy):
        physical=(Dr[bi]*yy).clamp_min(0)
        r=cI+product(rows,rows.val,physical,transpose=True)
        qb=q[bi]
        zz=torch.maximum(l[bi],torch.minimum(u[bi],-r/torch.where(qb>0,2*qb,1.)))
        terms=torch.where(qb>0,qb*zz*zz+r*zz,torch.minimum(r*l[bi],r*u[bi]))
        dual=c0I-(h[bi]*physical).sum(1)+terms.sum(1)
        primal=off+(cs*ww+qs[bi]*ww*ww).sum(1)
        pres=norm((matrix.right(ww)-hs[bi]).clamp_min(0))
        return torch.hypot(omega*pres,(primal-dual).abs())

    def gather_item():
        # Objective of the slot's item, from its owner's table or implicit sign*e_j; then its scalings.
        ip=jj[ids]>=0
        if tables:
            g=(arn[None]==jj[ids].long()[:,None]).to(c.dtype)*sg[ids].to(c.dtype)[:,None]
            cI.copy_(torch.where(ip[:,None],g,c[bi]));c0I.copy_(torch.where(ip,0.,c0[bi]))
        else:
            cI.copy_(c[bi]);c0I.copy_(c0[bi])
        if pt is not None:
            for dst,src in zip(slot_pt,objective):dst.copy_(src[bi])
            if tables:
                slot_pt[0].copy_(torch.where(ip[:,None],g,slot_pt[0]));slot_pt[1].copy_(torch.where(ip,0.,slot_pt[1]))
                slot_pt[2].copy_(torch.where(ip[:,None],q[bi],slot_pt[2]))
        cs.copy_((cI+2*q[bi]*mid[bi])*S[bi])
        off.copy_(c0I+(cI*mid[bi]+q[bi]*mid[bi]*mid[bi]).sum(1))
        hn,cn=norm(hs[bi]),norm(cs)
        return torch.where(hn>1e-10,cn.clamp_min(1e-10)/hn.clamp_min(1e-10),1.)

    def refill(mask):
        # Every retiring slot consumes one queue item; masked/padded boxes have
        # zero budget and are drained without doing a numerical step.
        rank=mask.long().cumsum(0)-1
        proposed=cursor+rank
        newids=proposed.clamp(0,N-1)
        take=mask & (proposed<N)
        ids.copy_(torch.where(take,newids,ids))
        bi.copy_(owner[ids].long())
        cursor.add_(mask.sum())
        live.copy_(torch.where(mask,take & valid[ids],live))
        omega0=gather_item()
        wn=torch.maximum(lw[bi],torch.minimum(uw[bi],(z0[ids]-mid[bi])/S[bi]))
        yn=y0[ids].clamp_min(0)/Dr[bi]
        for dst,src in ((w,wn),(y,yn),(wl,wn),(yl,yn)):
            dst.copy_(torch.where(mask[:,None],src,dst))
        if halpern:
            ws.copy_(torch.where(mask[:,None],wn,ws));ys.copy_(torch.where(mask[:,None],yn,ys))
        else:
            for dst in (ws,ys):dst.copy_(torch.where(mask[:,None],0.,dst))  # not mul_: NaN*0 keeps a stale NaN
        for dst in (since,it,nr):dst.mul_(~mask)
        omega.copy_(torch.where(mask,omega0,omega))
        refresh()
        error=plain(ws if halpern else w,ys if halpern else y)
        elast.copy_(torch.where(mask,error,elast));eprev.copy_(torch.where(mask,error,eprev))

    def reset():
        nonlocal mid,rad,Dr,S,hs,lw,uw,scaled,qs,eta
        mid=.5*(l+u);rad=torch.where(u>l,.5*(u-l),1.)
        scaled=qr.val*rad[:,qr.col]
        Dr=torch.ones_like(h);Dc=torch.ones_like(l)
        for k in range(11):
            rn=product(qr,scaled,torch.ones_like(l),absolute=True,maximum=k<10).sqrt()
            cn=product(qr,scaled,torch.ones_like(h),transpose=True,absolute=True,maximum=k<10).sqrt()
            rn=torch.where(rn>0,rn,1.);cn=torch.where(cn>0,cn,1.)
            Dr/=rn;Dc/=cn;scaled=scaled/rn[:,qr.row]/cn[:,qr.col]
        hs=(h-product(qr,qr.val,mid))*Dr;S=rad*Dc
        qs=q*S*S
        lw=(l-mid)/S;uw=(u-mid)/S
        v=seed.clone()
        for _ in range(40):
            v=v/norm(v).clamp_min(1e-300)[:,None]
            v=product(qr,scaled,product(qr,scaled,v),transpose=True)
        eta=.95/norm(v).sqrt().clamp_min(1e-12)
        empty.zero_();steps.zero_();restarts.zero_();reason.zero_()
        for metric in (competitive_probe,competitive_node,probe_slots,node_slots,serial_probe):metric.zero_()
        cursor.zero_();live.zero_();refill(torch.ones_like(live))

    def checkpoint_step(budget):
        # Close warm certificates BEFORE the first iteration. All other slots
        # continue, including when another slot is replaced in this graph.
        initial=live & (best[ids]>=target[ids])
        lane_budget=budget if probe_kind is None else torch.where(probe_kind[ids],torch.minimum(budget,probe_cap),budget)
        if dual_control is not None:
            cutoff,tau,width=dual_control
            cluster=best[ids]>=cutoff[ids]-4*tau[ids]
            # Ski rental: a lane at the cutoff rents twice the ordinary budget.
            lane_budget=torch.where(cluster & (budget>0),2*budget,lane_budget)
        previous_w.copy_(w);previous_y.copy_(y)
        run=live & ~initial
        if probe_kind is not None:
            is_probe=probe_kind[ids]
            # Node items precede support items in the FIFO. Thus once the
            # last resident node retires no unstarted node can remain.
            nodes=(run&~is_probe).sum()
            rank=(run&is_probe).long().cumsum(0)
            allowed=~is_probe|(nodes==0)|(rank<=torch.minimum(nodes,half_slots))
            run &= allowed
        for _ in range(checkpoint):
            active=run & (it<lane_budget)
            if probe_account:
                serial_probe.add_((active & (torch.ones_like(active) if probe_kind is None else is_probe)).any())
            if probe_kind is not None:
                np=(active&is_probe).sum();nn=(active&~is_probe).sum()
                competitive_probe.add_(torch.where(nn>0,np,0));competitive_node.add_(nn)
                probe_slots.add_(np);node_slots.add_(nn)
            tp=(eta[bi]/omega)[:,None]
            wn=torch.maximum(lw[bi],torch.minimum(uw[bi],(w/tp-(cs+matrix.left(y)))/(1/tp+2*qs[bi])))
            yn=(y+(eta[bi]*omega)[:,None]*(matrix.right(2*wn-w)-hs[bi])).clamp_min(0)
            if halpern:
                k=since.to(w.dtype);a1=((k+1)/(k+2))[:,None];a0=(1./(k+2))[:,None]
                wp=a1*(2*wn-w)+a0*wl;yp=a1*(2*yn-y)+a0*yl
                ws.copy_(torch.where(active[:,None],wn,ws));ys.copy_(torch.where(active[:,None],yn,ys))
                w.copy_(torch.where(active[:,None],wp,w));y.copy_(torch.where(active[:,None],yp,y))
            else:
                w.copy_(torch.where(active[:,None],wn,w));y.copy_(torch.where(active[:,None],yn,y))
                ws.add_(torch.where(active[:,None],w,0.));ys.add_(torch.where(active[:,None],y,0.))
            since.add_(active);it.add_(active)
        for candidate in ((ys,) if halpern else (y,ys/since.clamp_min(1)[:,None])):
            physical=(Dr[bi]*candidate).clamp_min(0)
            val=cert(rows,cI,c0I,h[bi],l[bi],u[bi],q[bi],physical)
            use=live & (val>best[ids])
            best_y.index_copy_(0,ids,torch.where(use[:,None],physical,best_y[ids]))
            best.index_copy_(0,ids,torch.where(live,torch.maximum(best[ids],val),best[ids]))
        ray=(Dr[bi]*(ys if halpern else y)).clamp_min(0)
        ray=ray/ray.abs().amax(1,keepdim=True).clamp_min(1.) if m else ray
        infeasible=live & certify_infeasible(rows,h[bi],l[bi],u[bi],ray)
        empty.index_copy_(0,ids,empty[ids]|infeasible)
        if pt is not None:
            # The objective is identically zero: a strictly positive outward
            # dual value is itself a Farkas certificate for this table.
            objective_node=torch.ones_like(live) if probe_kind is None else ~probe_kind[ids]
            empty.index_copy_(0,ids,empty[ids]|((best[ids]>0)&objective_node))
        wa=ws/since.clamp_min(1)[:,None];ya=ys/since.clamp_min(1)[:,None]
        if halpern:
            wc,yc=ws,ys;ec=plain(wc,yc)
        else:
            ec,ea=plain(w,y),plain(wa,ya);use=ea<ec
            wc,yc=torch.where(use[:,None],wa,w),torch.where(use[:,None],ya,y)
            ec=torch.minimum(ec,ea)
        restart=run & ((ec<=.2*elast)|((ec<=.8*elast)&(ec>eprev))|(since>=.36*it))
        dw,dy=norm(wc-wl),norm(yc-yl)
        upd=restart & (dw>1e-10)&(dy>1e-10)
        newomega=torch.exp(.5*torch.log(dy.clamp_min(1e-300)/dw.clamp_min(1e-300))+.5*omega.log())
        omega.copy_(torch.where(upd,newomega,omega));eprev.copy_(ec)
        for dst,src in ((w,wc),(y,yc),(wl,wc),(yl,yc)):
            dst.copy_(torch.where(restart[:,None],src,dst))
        error=plain(ws if halpern else w,ys if halpern else y)
        elast.copy_(torch.where(restart,error,elast));eprev.copy_(torch.where(restart,error,eprev))
        if not halpern:
            ws.copy_(torch.where(restart[:,None],0.,ws));ys.copy_(torch.where(restart[:,None],0.,ys))
        since.mul_(~restart);nr.add_(restart)
        pruned=(best[ids]>=target[ids])|empty[ids]
        converged=torch.isfinite(error)&(error<=accuracy[ids])
        if dual_control is not None:
            point=torch.maximum(l[bi],torch.minimum(u[bi],mid[bi]+S[bi]*wc))
            r=cI+2*q[bi]*point+product(rows,rows.val,best_y[ids],transpose=True)
            column=(r.abs()*(u[bi]-l[bi])).sum(1)
            # C2: w*|r|=O(w^2). The tolerance tightens quadratically in
            # relative diameter, hence the multiplier residual scales as w.
            tolerance=tau[ids]*width[ids].square()/4
            # A lane can enter the cluster on this very checkpoint, including
            # its last ordinary-budget checkpoint. Do not retire it using the
            # preceding checkpoint's cluster mask.
            cluster=best[ids]>=cutoff[ids]-4*tau[ids]
            lane_budget=torch.where(cluster & (budget>0),2*budget,lane_budget)
            converged=torch.isfinite(error)&(error<=tolerance)&(~cluster|(column<=tolerance))
        if support_tol is not None:
            from .implied_box import point_test
            point=torch.maximum(l[bi],torch.minimum(u[bi],mid[bi]+S[bi]*wc))
            support_objective=(cI,c0I,q[bi],target[ids])
            relative,_=point_test(rows,h[bi],point,support_objective,m-1,row_terms)
            gap=(c0I+(cI*point+q[bi]*point*point).sum(1)-best[ids]).abs()
            support_done=(relative<=1e-8)&torch.isfinite(gap)&(gap<=support_tol[ids])
            converged=support_done if probe_kind is None else torch.where(probe_kind[ids],support_done,converged)
        branch=torch.zeros_like(live)
        if pt is not None:
            from .implied_box import point_test
            point=torch.maximum(l[bi],torch.minimum(u[bi],mid[bi]+S[bi]*wc))
            relative,strict=point_test(rows,h[bi],point,slot_pt,cutoff_row,row_terms)
            branch=(relative<=1e-8)&strict
            if probe_kind is None:converged=torch.zeros_like(live)
            else:
                branch &= ~probe_kind[ids]
                converged &= probe_kind[ids]
        # Numerical stall means no representable primal OR dual movement over
        # a checkpoint, not an arbitrary number of fruitless bound reports.
        stalled=(w==previous_w).all(1)&(y==previous_y).all(1)
        if pt is not None:stalled=torch.zeros_like(stalled)
        done=live & (pruned|branch|converged|(stalled&run)|(it>=lane_budget))
        reason.index_copy_(0,ids,torch.where(done,torch.where(pruned,1,torch.where(branch,5,torch.where(converged,2,torch.where(stalled,3,4)))),reason[ids]))
        for dst,src in ((zout,mid[bi]+S[bi]*(ws if halpern else w)),(yout,(Dr[bi]*(ys if halpern else y)).clamp_min(0))):
            dst.index_copy_(0,ids,torch.where(live[:,None],src,dst[ids]))
        steps.index_copy_(0,ids,torch.where(live,it,steps[ids]));restarts.index_copy_(0,ids,torch.where(live,nr,restarts[ids]))
        refill(done|~live)

    # The budget is a scalar device input, so changing K does not recapture.
    limit=steps.new_zeros(());probe_cap=steps.new_zeros(())
    reset_graph=step_graph=None
    if capture:
        reset_graph=_capture_reset(reset,dev)
        step_graph=_capture_reset(lambda:checkpoint_step(limit),dev)

    def fit(dst,src):
        # Rows beyond the caller's count repeat row 0: valid inputs that no item ever reads.
        dst[:len(src)].copy_(src)
        if len(src)<len(dst):dst[len(src):].copy_(src[:1].expand(len(dst)-len(src),*src.shape[1:]))

    def prologue(count):
        # Step-zero certificates and outputs of EVERY item, also those that never obtain a slot.
        for s in range(0,N,B):
            idx=torch.arange(s,min(N,s+B),device=dev)
            b=owner[idx].long()
            if tables:
                ip=jj[idx]>=0
                g=(arn[None]==jj[idx].long()[:,None]).to(c.dtype)*sg[idx].to(c.dtype)[:,None]
                ci=torch.where(ip[:,None],g,c[b]);c0i=torch.where(ip,0.,c0[b])
            else:ci,c0i=c[b],c0[b]
            r=Rows(qr.row,qr.col,qr.val[b],m,n)
            best_y[idx]=y0[idx].clamp_min(0)
            best[idx]=cert(r,ci,c0i,h[b],l[b],u[b],q[b],best_y[idx])
            zout[idx]=torch.maximum(l[b],torch.minimum(u[b],z0[idx]));yout[idx]=y0[idx]

    def run(data,budget,stop,warm,tol,deadline,count,pt_data=None,dual_data=None,support_data=None,cap=None,kind_data=None,items=None):
        original,*values=data
        fit(qr.val,original.val)
        for dst,src in zip((c,c0,h,l,u,q),values):fit(dst,src)
        if tables:
            for dst,src in zip((owner,jj,sg),items):fit(dst,src)
            probe_kind.copy_(jj>=0)
        elif kind_data is not None:probe_kind.copy_(kind_data)
        if pt is not None:
            for dst,src in zip(objective,pt_data[1:]):fit(dst,src)
        if dual_control is not None:
            for dst,src in zip(dual_control,dual_data):fit(dst,src)
        if support_tol is not None:fit(support_tol,support_data)
        fit(z0,((l+u)/2)[owner.long()] if warm is None else warm[0]);y0.zero_() if warm is None else fit(y0,warm[1])
        fit(target,stop);fit(accuracy,tol);valid.copy_(torch.arange(N,device=dev)<count)
        limit.fill_(budget);probe_cap.fill_(budget if cap is None else cap)
        prologue(count)
        if capture:reset_graph.replay()
        else:reset()
        initial=best.clone();tick=time.perf_counter();chunks=0
        while bool(live.any()) or int(cursor)<count:
            if deadline is not None and time.time()>=deadline:break
            if capture:step_graph.replay()
            else:checkpoint_step(limit)
            chunks+=1
        # Returned state owns storage. A missed deadline retains the valid
        # step-zero certificate even for boxes that never obtained a slot.
        metrics=dict(iterations=int(steps.max()),lane_iterations=steps[:count].tolist(),
            active_lane_steps=int(steps[:count].sum()),lane_steps=B*chunks*checkpoint,
            capacity=B,nodes=count,refills=max(0,count-B),decided=int((reason[:count]==1).sum()),
            converged=int((reason[:count]==2).sum()),stalled=int((reason[:count]==3).sum()),
            occupancy=float(steps[:count].sum())/max(1,B*chunks*checkpoint),
            steps_per_decided_node=float(steps[:count].sum())/max(1,int((reason[:count]==1).sum())),
            iteration_seconds=time.perf_counter()-tick,capture=capture)
        if pt is not None:
            metrics.update(pt_lane=True,reasons=reason[:count].tolist(),branched=int((reason[:count]==5).sum()))
            metrics['pruned']=metrics['decided']
            metrics['decided']+=metrics['branched']
            metrics['steps_per_decided_node']=float(steps[:count].sum())/max(1,metrics['decided'])
        if dual_control is not None:
            metrics.update(dual_accuracy=True,reasons=reason[:count].tolist(),
                           accuracy=(dual_control[1]*dual_control[2].square()/4)[:count].tolist())
        if support_tol is not None:
            metrics.update(support_accuracy=True,reasons=reason[:count].tolist())
        if probe_kind is not None:
            metrics.update(competitive_probe_steps=int(competitive_probe),competitive_node_steps=int(competitive_node),
                           probe_slot_steps=int(probe_slots),node_slot_steps=int(node_slots))
        if probe_account:
            mask=torch.ones(count,dtype=torch.bool,device=dev) if probe_kind is None else probe_kind[:count]
            metrics.update(probe_serial_steps=int(serial_probe),probe_lane_steps=int(steps[:count][mask].sum()))
        return QPResult(best[:count].clone(),empty[:count].clone(),zout[:count].clone(),yout[:count].clone(),
            best_y[:count].clone(),restarts[:count].clone(),[dict(iteration=0,best=initial[:count]),
            dict(iteration=metrics['iterations'],best=best[:count].clone())],metrics)
    run.graphs=reset_graph,step_graph
    return run


def _cpu_nodes(data, capacity, budget, stop, warm, accuracy, checkpoint, deadline, iteration_scheme="average"):
    from .cpu_pdhg import kernel
    from nlp_solver.rigorous_dual import dual_lower_bound, certify_infeasible
    import ctypes
    root=kernel()
    if root is None:return None
    fn=root.library.solve_node_batch
    ptr=ctypes.c_void_p
    # solve_node_batch has a Halpern flag, but no trace.
    fn.argtypes=[ctypes.c_int]*4+[ptr]*9+[ctypes.c_int]*4+[ptr]*2+[ctypes.c_double]*2+[ptr]*7;fn.restype=None
    r,c,c0,h,l,u,q=data;B,n=c.shape;m=r.m
    expanded=[t.expand(B,*t.shape[1:]).contiguous() for t in (r.val,c,c0,h,l,u)]
    av,cc,off,hh,ll,uu=expanded
    z=c.new_zeros(B,n);yy=h.new_zeros(B,m);ray=yy.clone();by=yy.clone();stats=c.new_zeros(B,10)
    z0=(l+u).expand(B,-1).contiguous()/2 if warm is None else warm[0].contiguous()
    y0=yy.clone() if warm is None else warm[1].contiguous()
    tol=accuracy.expand(B).contiguous() if torch.is_tensor(accuracy) else c0.new_full((B,),accuracy)
    target=stop.contiguous()
    seed=c.new_tensor(np.random.default_rng(0).standard_normal(n))
    args=[r.row.contiguous(),r.col.contiguous(),av,cc,off,hh,ll,uu,seed]
    started=time.perf_counter()
    initial=dual_lower_bound(r,c,c0,hh,ll,uu,y0)[0]
    fn(B,n,m,len(r.row),*(t.data_ptr() for t in args),budget,checkpoint,1,int(iteration_scheme=="halpern"),z0.data_ptr(),y0.data_ptr(),
       -1.,-1. if deadline is None else deadline,*(t.data_ptr() for t in (z,by,ray,yy,stats,target,tol)))
    lower=dual_lower_bound(r,c,c0,hh,ll,uu,by)[0]
    empty=certify_infeasible(r,hh,ll,uu,ray)
    steps=stats[:,8].long();decided=(lower>=stop)|empty
    # Occupancy is active numerical work divided by capacity times makespan.
    # CPU arithmetic is serial; this queue simulation reports the corresponding
    # fixed-capacity schedule without claiming CPU parallel execution.
    schedule=root.library.node_makespan
    schedule.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p];schedule.restype=ctypes.c_double
    makespan=schedule(B,capacity,stats.data_ptr())
    metrics=dict(iterations=int(steps.max()),lane_iterations=steps.tolist(),active_lane_steps=int(steps.sum()),
        lane_steps=capacity*makespan,capacity=capacity,nodes=B,refills=max(0,B-capacity),decided=int(decided.sum()),
        occupancy=float(steps.sum())/max(1,capacity*makespan),
        steps_per_decided_node=float(steps.sum())/max(1,int(decided.sum())),
        cpu_fused=True,capture=False,setup_seconds=0.,capture_seconds=0.,iteration_seconds=time.perf_counter()-started)
    return QPResult(lower,empty,z,yy,by,stats[:,1].long(),[dict(iteration=0,best=initial),
        dict(iteration=int(steps.max()),best=lower.clone())],metrics)


def _anytime_support(data, capacity, budget, stop, warm, support, checkpoint,
                     deadline, iteration_scheme, capture, probe_work, options, items=None):
    """Ski rental per direction: pay at most the two-child split allowance.

    A direction rents iterations until its two-sided support tolerance is
    met or the cost of buying a split is reached. Checks double from the
    existing checkpoint quantum; compaction preserves direction identities.
    """
    from nlp_solver.rigorous_dual import Rows
    from .implied_box import point_test
    from .cpu_pdhg import uniform
    from .restarted_qp import persistent_qp
    r,c,c0,h,l,u,q=data;N=len(c) if items is None else len(items[0])
    limit=min(budget,max(0,int(probe_work)))
    cap=1<<(min(capacity,N)-1).bit_length()
    by=h.new_zeros(N,r.m) if warm is None else warm[1].clamp_min(0).clone()
    if items is not None:z=((l+u)/2)[items[0].long()] if warm is None else warm[0].clone()
    else:z=(l+u).expand_as(c).clone()/2 if warm is None else warm[0].clone()
    y=by.clone()
    if items is None:best=CompiledCertifier.diagonal_qp(r,q,c,c0,h,l,u,by)
    else:  # the per-item tables are transient: one slot-width chunk at a time
        best=c0.new_zeros(N)
        for s in range(0,N,cap):
            idx=torch.arange(s,min(N,s+cap),device=c.device);rr,cc,off,hh,ll,uu,qq=item_tables(data,items,idx)
            best[idx]=CompiledCertifier.diagonal_qp(rr,qq,cc,off,hh,ll,uu,by[idx])
    initial=best.clone();empty=c0.new_zeros(N,dtype=torch.bool)
    steps=torch.zeros(N,dtype=torch.long,device=c.device);restarts=steps.clone();reason=steps.clone()
    terms=int(torch.bincount(r.row.cpu(),minlength=r.m).max()) if r.m else 0
    kind=options.get('probe_kind') if items is None else items[1]>=0
    is_probe=torch.ones(N,dtype=torch.bool,device=c.device) if kind is None else kind
    def sliced(ids):
        take=lambda v:v.expand(N,*v.shape[1:])[ids]
        return (Rows(r.row,r.col,take(r.val),r.m,r.n),*(take(v) for v in (c,c0,h,l,u,q)))
    def done(ids,point):
        if items is not None and len(ids)>cap:
            return torch.cat([done(ids[s:s+cap],point[s:s+cap]) for s in range(0,len(ids),cap)])
        rr,cc,off,hh,ll,uu,qq=sliced(ids) if items is None else item_tables(data,items,ids)
        point=torch.maximum(ll,torch.minimum(uu,point))
        relative,_=point_test(rr,hh,point,(cc,off,qq,stop[ids]),max(0,r.m-1),terms)
        gap=(off+(cc*point+qq*point*point).sum(1)-best[ids]).abs()
        return (relative<=1e-8)&torch.isfinite(gap)&(gap<=support[ids])&is_probe[ids]
    ids=torch.arange(N,device=c.device)
    solved=done(ids,z);pruned=best>=stop
    reason=torch.where(pruned,1,torch.where(solved,2,reason))
    ids=ids[~(solved|pruned)]
    counts=[len(ids)];epoch=checkpoint;spent=0;serial=slot_steps=0;tick=time.perf_counter()
    competitive_probe=competitive_node=0
    session=None
    # The existing persistent CPU backend shares one table and retains all
    # PDHG state across checks/compactions. Heterogeneous/GPU buckets use the
    # production queue in compact batches, never a per-node Python solve.
    if len(ids) and limit and items is None and not c.is_cuda and uniform(data) and kind is None:
        shared=(Rows(r.row,r.col,r.val[:1],r.m,r.n),c[ids],c0[ids],h[:1],l[:1],u[:1],q[:1])
        session=persistent_qp(shared,warm=(z[ids],y[ids]),capture=False,checkpoint=checkpoint,iteration_scheme=iteration_scheme)
    while len(ids) and spent<limit and (deadline is None or time.time()<deadline):
        # The queue retires lanes at support accuracy and refills them, so the
        # tensor path is ONE call: probes rent at most `limit` each (probe_cap),
        # node lanes keep the caller's budget.
        work=min(epoch,limit-spent) if session is not None else limit-spent
        if session is not None:
            o=session.advance(work,deadline=deadline)
            point=o.candidate_z
            took=torch.full_like(ids,o.metrics['iterations'])
        else:
            kw=dict(options)
            if items is not None:kw['items']=tuple(v[ids] for v in items)  # tables and per-box pt stay whole
            else:
                if kw.get('probe_kind') is not None:kw['probe_kind']=kw['probe_kind'][ids]
                if kw.get('pt') is not None:kw['pt']=(kw['pt'][0],*(v[ids] for v in kw['pt'][1:]))
            o=solve_pool(data if items is not None else sliced(ids),capacity=min(cap,len(ids)),budget=min(budget,limit) if kind is not None else work,
                probe_cap=work if kind is not None else None,stop=stop[ids],
                warm=(z[ids],y[ids]),support=support[ids],checkpoint=checkpoint,
                deadline=deadline,iteration_scheme=iteration_scheme,capture=capture,**kw)
            point=o.z
            took=torch.tensor(o.metrics['lane_iterations'],device=c.device)
        use=o.lower>best[ids]
        by[ids]=torch.where(use[:,None],o.best_y,by[ids]);best[ids]=torch.maximum(best[ids],o.lower)
        z[ids]=o.z;y[ids]=o.y;empty[ids]|=o.empty;steps[ids]+=took
        restarts[ids]=o.restarts
        # Synchronous compact epochs: ceil(active/capacity) waves; each
        # wave runs at most 'work' serial iterations (CPU executes serially).
        waves=(len(ids)+cap-1)//cap
        serial+=waves*o.metrics['iterations'] if session is not None else o.metrics.get('probe_serial_steps',waves*work)
        slot_steps+=cap*waves*(o.metrics['iterations'] if session is not None else work)
        competitive_probe+=o.metrics.get('competitive_probe_steps',0)
        competitive_node+=o.metrics.get('competitive_node_steps',0)
        spent+=work
        solved=done(ids,point);pruned=(best[ids]>=stop[ids])|empty[ids]
        why=torch.tensor(o.metrics.get('reasons',[0]*len(ids)),device=c.device)
        node_done=~is_probe[ids]&((why==1)|(why==2)|(why==5))
        retire=solved|pruned|node_done|(spent>=limit)
        reason[ids]=torch.where(pruned,1,torch.where(solved,2,torch.where(node_done,why,torch.where(retire,6,0))))
        keep=(~retire).nonzero().flatten()
        ids=ids[keep];counts.append(len(ids))
        if session is not None and len(ids):session.compact(keep)
        epoch*=2
    metrics=dict(iterations=int(steps.max()),lane_iterations=steps.tolist(),active_lane_steps=int(steps.sum()),
        lane_steps=slot_steps,capacity=cap,nodes=N,refills=max(0,N-cap),decided=int((reason==1).sum()),
        converged=int((reason==2).sum()),stalled=0,occupancy=float(steps.sum())/max(1,slot_steps),
        iteration_seconds=time.perf_counter()-tick,capture=False,support_accuracy=True,reasons=reason.tolist(),
        probe_active_counts=counts,probe_work_limit=limit)
    if options.get('probe_account',False):
        metrics.update(probe_serial_steps=serial,probe_lane_steps=int(steps[is_probe].sum()))
    if kind is not None:
        metrics.update(probe_slot_steps=int(steps[is_probe].sum()),node_slot_steps=int(steps[~is_probe].sum()),
            competitive_probe_steps=competitive_probe,competitive_node_steps=competitive_node)
    return QPResult(best,empty,z,y,by,restarts,[dict(iteration=0,best=initial),dict(iteration=int(steps.max()),best=best.clone())],metrics)


@torch.no_grad()
def solve_pool(data, *, capacity, budget, stop, warm=None, accuracy=0., checkpoint=25,
               capture=True, cache=None, deadline=None, iteration_scheme="average", implied_box=False, pt=None,
               dual_accuracy=None, support=None, probe_kind=None, owners=None, probe_work=None, probe_account=False, probe_cap=None,
               items=None):
    """Process a best-first queue with a power-of-two active lane bucket.

    `owners` (the box of each lane) is accepted for callers but unused: every
    slot owns its CSR, which beats a grouped block CSR from G>=4 boxes (A158).

    `items=(owner, j, sign)` (int32 [Q]) switches to a box-table queue: `data`, and a per-box `pt`, hold
    one row per BOX and item i is box owner[i]'s node lane (j<0) or the direction sign[i]*e_j. `stop`,
    `warm`, `support` are per item. Results are per item, bit-identical to the same items expanded.
    """
    from nlp_solver.rigorous_dual import Rows
    if iteration_scheme not in ("average","halpern"):raise ValueError("invalid iteration scheme")
    rows,c,c0,h,l,u,q=data;count=len(c)
    if count<1 or capacity<1 or budget<0:raise ValueError('invalid node capacity/budget')
    if any(t.dtype!=torch.float64 for t in (rows.val,c,c0,h,l,u,q)):
        raise TypeError('node lanes require binary64 search and certificates')
    if bool((q<0).any()):raise ValueError('negative node curvature')
    box_info=None
    if implied_box:
        from .implied_box import implied_box as tighten_box
        l,u,box_info=tighten_box(rows,h,l,u)
        data=rows,c,c0,h,l,u,q
    def box_metrics(result):
        if probe_account and 'probe_serial_steps' not in result.metrics:
            result.metrics.update(probe_serial_steps=int(result.metrics['lane_steps']/result.metrics['capacity']),
                                  probe_lane_steps=result.metrics['active_lane_steps'])
        if box_info is not None:
            result.metrics.update(implied_box_passes=box_info['passes'],
                                  implied_box_empty_hints=box_info['empty_hint'].tolist())
        return result
    cache={} if cache is None else cache
    if probe_work is not None:
        if support is None:raise ValueError('anytime probes require support tolerances')
        return box_metrics(_anytime_support(data,capacity,budget,stop,warm,support,checkpoint,
            deadline,iteration_scheme,capture,probe_work,dict(pt=pt,probe_kind=probe_kind,probe_account=probe_account,cache=cache),items))
    if items is not None:
        Q=len(items[0]);G=len(c);Gc=1<<(G-1).bit_length();Qc=1<<(Q-1).bit_length()
        capacity=1<<(min(capacity,Q)-1).bit_length()
        tile=lambda t,k:t[:1].expand(k,*t.shape[1:])  # build templates only: run() fills the real rows
        template=(Rows(rows.row,rows.col,tile(rows.val,Gc),rows.m,rows.n),*(tile(t,Gc) for t in (c,c0,h,l,u,q)))
        pt_template=None if pt is None else (pt[0],*(tile(t,Gc) for t in pt[1:]))
        support_template=None if support is None else tile(support,Qc)
        capture=capture and c.is_cuda
        shared_support=support is not None and G==1
        key=(rows.row.data_ptr(),rows.col.data_ptr(),Gc,Qc,capacity,checkpoint,capture,iteration_scheme,'items',shared_support)
        if probe_account:key+=('probe_account',)
        if pt is not None:key+=('pt',pt[0])
        cold=key not in cache;tick=time.perf_counter()
        if cold:cache[key]=_workspace(template,capacity,checkpoint,capture,iteration_scheme,pt_template,None,support_template,
                                      shared_support,None,probe_account,queue=Qc)
        setup=time.perf_counter()-tick
        result=cache[key](data,budget,stop,warm,c0.new_full((Q,),accuracy),deadline,Q,pt,None,support,probe_cap,None,items)
        result.metrics.update(cache_hit=not cold,setup_seconds=setup,capture_seconds=setup if cold and capture else 0.)
        return box_metrics(result)
    capacity=1<<(min(capacity,count)-1).bit_length()
    # Controlled lanes certify BEFORE retirement. The legacy fused CPU ABI
    # stops on a plain value (A144), and cannot implement a two-sided test.
    if pt is None and dual_accuracy is None and support is None and probe_kind is None and not c.is_cuda and not bool(q.any()):
        result=_cpu_nodes(data,capacity,budget,stop,warm,accuracy,checkpoint,deadline,iteration_scheme)
        if result is not None:return box_metrics(result)
    if (support is not None and pt is None and dual_accuracy is None and probe_kind is None
            and not c.is_cuda and not bool(q.any()) and bool(stop.isinf().all())):
        # Offline CPU support gates can use the unchanged batch ABI: all
        # active targets are +inf, so its plain-value stop cannot accept an
        # uncertified endpoint. First try the endpoint gap, then spend the
        # remaining allowance only on directions that fail the full guard.
        first=min(budget,20000)
        result=_cpu_nodes(data,capacity,first,stop,warm,support,checkpoint,deadline,iteration_scheme)
        if result is not None:
            from .implied_box import point_test
            terms=int(torch.bincount(rows.row.cpu(),minlength=rows.m).max()) if rows.m else 0
            def qualify(o):
                point=torch.maximum(l,torch.minimum(u,o.z))
                relative,_=point_test(rows,h,point,(c,c0,q,stop),max(0,rows.m-1),terms)
                gap=(c0+(c*point+q*point*point).sum(1)-o.lower).abs()
                pruned=o.empty | (o.lower>=stop)
                return pruned,~pruned & (relative<=1e-8) & torch.isfinite(gap) & (gap<=support)
            pruned,converged=qualify(result)
            remaining=~pruned & ~converged
            if budget>first and bool(remaining.any()) and (deadline is None or time.time()<deadline):
                target=torch.where(remaining,stop,-torch.inf)
                more=_cpu_nodes(data,capacity,budget-first,target,(result.z,result.y),0.,checkpoint,deadline,iteration_scheme)
                use=more.lower>result.lower
                result.lower=torch.maximum(result.lower,more.lower)
                result.best_y=torch.where(use[:,None],more.best_y,result.best_y)
                result.empty |= more.empty & remaining
                result.z=torch.where(remaining[:,None],more.z,result.z)
                result.y=torch.where(remaining[:,None],more.y,result.y)
                result.restarts+=more.restarts
                steps=torch.tensor(result.metrics['lane_iterations'])+torch.tensor(more.metrics['lane_iterations'])
                for key in ('lane_steps','iteration_seconds'):result.metrics[key]+=more.metrics[key]
                result.metrics.update(iterations=int(steps.max()),lane_iterations=steps.tolist(),active_lane_steps=int(steps.sum()))
                result.metrics['occupancy']=float(steps.sum())/max(1,result.metrics['lane_steps'])
                result.trace[-1]=dict(iteration=int(steps.max()),best=result.lower.clone())
                pruned,converged=qualify(result)
            reasons=torch.where(pruned,1,torch.where(converged,2,4))
            result.metrics.update(support_accuracy=True,reasons=reasons.tolist(),converged=int(converged.sum()))
            return box_metrics(result)
    total=1<<(count-1).bit_length()
    def pad(t):
        t=t.expand(count,*t.shape[1:])
        return torch.cat((t,t[:1].expand(total-count,*t.shape[1:]))) if total>count else t
    padded=(Rows(rows.row,rows.col,pad(rows.val),rows.m,rows.n),*(pad(t) for t in (c,c0,h,l,u,q)))
    stop=pad(stop);tol=pad(c0.new_full((count,),accuracy))
    warm=None if warm is None else tuple(pad(t) for t in warm)
    pt=None if pt is None else (pt[0],*(pad(t) for t in pt[1:]))
    dual_accuracy=None if dual_accuracy is None else tuple(pad(t) for t in dual_accuracy)
    support=None if support is None else pad(support)
    probe_kind=None if probe_kind is None else pad(probe_kind)
    capture=capture and c.is_cuda
    key=(rows.row.data_ptr(),rows.col.data_ptr(),total,capacity,checkpoint,capture,iteration_scheme)
    if probe_account:key+=('probe_account',)
    if pt is not None:key+=('pt',pt[0])
    if dual_accuracy is not None:key+=('dual_accuracy',)
    if probe_kind is not None:key+=('mixed',)  # the kinds are a static device input of the workspace
    shared_support=False
    if support is not None:
        # Directions of one box have identical scaled constraint matrices.
        # One CSR times a dense direction block avoids duplicating every row
        # in a block-diagonal CSR. Check physical bounds as well as values;
        # heterogeneous box buckets retain the independent-matrix path.
        shared_support=all(torch.equal(t,t[:1].expand_as(t)) for t in (rows.val,l,u))
        key+=('support',shared_support)
    cold=key not in cache;tick=time.perf_counter()
    if cold:cache[key]=_workspace(padded,capacity,checkpoint,capture,iteration_scheme,pt,dual_accuracy,support,shared_support,probe_kind,probe_account)
    setup=time.perf_counter()-tick
    result=cache[key](padded,budget,stop,warm,tol,deadline,count,pt,dual_accuracy,support,probe_cap,probe_kind)
    result.metrics.update(cache_hit=not cold,setup_seconds=setup,capture_seconds=setup if cold and capture else 0.)
    return box_metrics(result)
