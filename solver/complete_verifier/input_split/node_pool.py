"""Batched node lanes beside native GEAR; only duals certify bounds.

Physical iterates are optional search memory. A parent certificate may be
inherited only inside its input box and at an equal or tighter cutoff.
"""
from contextlib import nullcontext
import time
import json
import itertools
import numpy as np
import torch
from auto_LiRPA.lockstep_dual import CompiledCertifier


def certificate(data, y):
    rows,c,c0,h,l,u,q=data
    return CompiledCertifier.diagonal_qp(rows,q,c,c0,h,l,u,y.clamp_min(0))


def crown_seed(e, d, data, *, with_certificate=False, multipliers=None):
    """Map compiled CROWN's face masses through GraphQP's immutable row map.

    Eliminated private squares have an exact convex prox. Epigraph link rows
    receive their objective weight; all unmapped/new rows start with zero mass.
    """
    parts=e.cg.sign_y(d)
    if (torch.is_tensor(multipliers) and e.Sg>1
            and multipliers.shape[-1]>=e.Sg-1):
        # GEAR appends input/cut multipliers after the output rows. Only the
        # unchanged output identities map here; dynamic native cuts stay native.
        lam=multipliers.to(d.lo).reshape(-1,multipliers.shape[-1])[:,:e.Sg-1].expand(len(d.lo),-1).clamp_min(0)
        lam=torch.where(torch.isfinite(lam),lam,0.)
        specs=e.grow[1:]
        weights=d.lo.new_zeros(len(d.lo),e.cg.S)
        weights.scatter_add_(1,specs.clamp_min(0)[None].expand_as(lam),torch.where(specs[None]>=0,lam,0.))
        effective=e.cg.CM0+e.cg.Lcm.left(weights)
        keep=[]
        views=[e.cg._view(d,k) for k in range(len(e.cg.levels))]
        e.cg.crown(views,effective[:,None],len(views)-1,keep)
        A=torch.cat([v.expand(len(d.lo),1,v.shape[-1]) for v in keep],-1)
        parts=(A.clamp_min(0).unsqueeze(2)*d.dl[:,None],(-A).clamp_min(0).unsqueeze(2)*d.du[:,None],
               weights[:,None,e.cg.rows],parts[3])
    flat=torch.cat([v.flatten(2) for v in parts],2)[:,0]
    base=e.qp.base
    mapped=flat[:,base.refs.clamp_min(0)]
    y=data[3].new_zeros(len(d.lo),data[0].m)
    y[:,:base.m]=torch.where(base.refs[None]>=0,mapped,0.).clamp_min(0)
    for t,term in enumerate(e.qp.epigraph.terms):
        y[:,base.m+t]=data[1][:,base.n+t]
    if with_certificate:
        # CROWN's backward Plan already encloses this same sign dual in the
        # original factored faces. Scalar COO expansion has additional error;
        # retain both dual certificate channels instead of weakening CROWN.
        source=getattr(d,'node_crown',None)
        if source is None:source=e.cg.output_bound(d)[0][:,0]
        return y,source
    return y


class NodeTables:
    """One capturable CROWN/FBBT/GraphQP refresh for a whole box bucket."""
    def __init__(self, engine, *, capture=True):
        self.engine=engine;self.capture=capture;self.cache={};self.metrics={}
        from input_split.node_faces import compiler
        self.cg=compiler(engine)
        # _hb may call a QTB/convex tangent plane: fixed-coordinate validation
        # and scalar reductions there synchronize with the host. Never begin a
        # speculative capture: an invalidated CUDA capture can poison its pool.
        self.capture = capture and self.cg.capture_safe() and engine.tangent is None

    @torch.no_grad()
    def build(self, lo, hi, cutoff, caps=None):
        from auto_LiRPA.restarted_qp import _capture_reset
        from auto_LiRPA.obbt_unary import OutwardCompiledGraph
        from auto_LiRPA.compiled_crown import Data
        from auto_LiRPA.epigraph import Epigraph
        from nlp_solver.rigorous_dual import Rows
        e=self.engine;count=len(lo);capacity=1<<(count-1).bit_length()
        def pad(t):
            return torch.cat((t,t[:1].expand(capacity-count,*t.shape[1:]))) if capacity>count else t
        source=tuple(pad(t) for t in (lo,hi,cutoff,*(() if caps is None else caps)))
        key=(capacity,caps is not None)
        cold=key not in self.cache;started=time.perf_counter()
        if cold:
            inputs=tuple(t.clone() for t in source);result=[]
            invalid=lo.new_zeros(capacity,dtype=torch.bool)
            def compute():
                ll,uu,T=inputs[:3]
                inherited_caps=tuple(inputs[3:]) if len(inputs)>3 else None
                invalid.zero_()
                deferred = self.capture and lo.is_cuda
                if deferred:
                    e.cg._capture_invalid=invalid;self.cg._capture_invalid=invalid
                try:
                    # Call the uncached compiler INSIDE this graph. All FBBT
                    # masks and strict-domain flags remain device tensors.
                    d,out=OutwardCompiledGraph.bound(self.cg,ll,uu,e._thr(ll,uu,T),e._hb(ll,uu,T),caps=inherited_caps)
                    empty=torch.zeros_like(invalid)
                    if e.fbbt is not None:
                        fl,fu,empty=e._fbbt_box(d,T)
                        nl=torch.where(empty[:,None],ll,torch.maximum(ll,fl[:,:e.n]))
                        nu=torch.where(empty[:,None],uu,torch.minimum(uu,fu[:,:e.n]))
                        caps=(torch.where(empty[:,None],-torch.inf,fl[:,e.n:]),
                              torch.where(empty[:,None],torch.inf,fu[:,e.n:]))
                        d,out=OutwardCompiledGraph.bound(self.cg,nl,nu,e._thr(nl,nu,T),e._hb(nl,nu,T),caps=caps)
                    d.node_crown=out[:,0]
                    data=Epigraph.clean(e.qp.build(d))
                finally:
                    if deferred:
                        del e.cg._capture_invalid;del self.cg._capture_invalid
                result[:]=[data,d,empty]
            graph=_capture_reset(compute,lo.device) if self.capture and lo.is_cuda else None
            self.cache[key]=(inputs,result,invalid,graph,compute)
        inputs,result,invalid,graph,compute=self.cache[key]
        for dst,src in zip(inputs,source):dst.copy_(src)
        if graph is None:compute()
        else:graph.replay()
        # One boundary check, never a host synchronization inside the build.
        if bool(invalid[:count].any()):raise ValueError('strict node table is undefined')
        def active(t):
            if torch.is_tensor(t):return t[:count].clone()
            if isinstance(t,tuple):return tuple(active(v) for v in t)
            return t
        data,d,empty=result;r,*ts=data
        out=(Rows(r.row,r.col,active(r.val),r.m,r.n),*(active(t) for t in ts))
        dd=Data(**{k:active(v) for k,v in vars(d).items()})
        self.metrics=dict(boxes=count,capacity=capacity,cache_hit=not cold,seconds=time.perf_counter()-started,
                          capture=graph is not None)
        return out,dd,active(empty)


class ParentBank:
    """Capacity-bounded, tensor-only ancestry lookup; no per-box Python map."""
    def __init__(self, capacity, *, prefer_converged=False):
        self.capacity=capacity
        self.data=None
        self.caps=None
        self.prefer_converged=prefer_converged
        self.converged=None
        self.labels=None

    def caps_for(self, lo, hi, cutoff):
        if self.data is None or self.caps is None:return None
        pl,pu,_,_,_,pt=self.data
        if not len(pl):return None
        scope=((lo[:,None]>=pl[None]) & (hi[:,None]<=pu[None])).all(2) & (cutoff[:,None]<=pt[None])
        rank=torch.where(scope,(pu-pl).abs().sum(1)[None],torch.inf)
        ids=rank.argmin(1);found=scope.any(1)
        return (torch.where(found[:,None],self.caps[0][ids],-torch.inf),
                torch.where(found[:,None],self.caps[1][ids],torch.inf))

    def seed(self, lo, hi, default, cutoff, labels=None, nearest=False):
        z,y=default
        if self.data is None:
            return z,y,lo.new_full((len(lo),),-torch.inf),lo.new_zeros(len(lo),dtype=torch.bool)
        pl,pu,pz,py,pd,pt=self.data
        if not len(pl):return z,y,lo.new_full((len(lo),),-torch.inf),lo.new_zeros(len(lo),dtype=torch.bool)
        if labels is not None:
            # Factor the repeated box geometry out of the direction lookup.
            # The old N-direction by N-parent by n-coordinate comparison
            # becomes box-by-box geometry plus direction-by-parent-box rank.
            first,index,qualified=self.support_index(nearest=nearest)
            boxes,inverse=torch.unique_consecutive(torch.cat((lo,hi),1),dim=0,return_inverse=True)
            nested=((boxes[:,None,:lo.shape[1]]>=pl[first][None]) &
                    (boxes[:,None,lo.shape[1]:]<=pu[first][None])).all(2)[inverse]
            candidates=index[:,labels].T
            nested &= candidates>=0
            if self.prefer_converged and not nearest:
                good=nested&qualified[:,labels].T
                nested=torch.where(good.any(1)[:,None],good,nested)
            rank=torch.where(nested,(pu[first]-pl[first]).abs().sum(1)[None],torch.inf)
            group=rank.argmin(1);found=nested.any(1)
            ids=candidates[torch.arange(len(lo),device=lo.device),group].clamp_min(0)
            zz=torch.maximum(lo,torch.minimum(hi,pz[ids])) if pz.shape[1]==lo.shape[1] else pz[ids]
            return (torch.where(found[:,None],zz,z),torch.where(found[:,None],py[ids],y),
                    torch.where(found&(cutoff<=pt[ids]),pd[ids],-torch.inf),found)
        nested=((lo[:,None]>=pl[None]) & (hi[:,None]<=pu[None])).all(2)
        # The smallest containing parent is the closest available ancestor.
        size=(pu-pl).abs().sum(1)
        if self.prefer_converged and self.converged is not None:
            qualified=nested & self.converged[None]
            nested=torch.where(qualified.any(1)[:,None],qualified,nested)
        rank=torch.where(nested,size[None],torch.inf)
        ids=rank.argmin(1);found=nested.any(1)
        zz=torch.maximum(lo,torch.minimum(hi,pz[ids])) if pz.shape[1]==lo.shape[1] else pz[ids]
        zz=torch.where(found[:,None],zz,z)
        yy=torch.where(found[:,None],py[ids],y)
        floor=torch.where(found & (cutoff<=pt[ids]),pd[ids],-torch.inf)
        return zz,yy,floor,found

    def support_index(self, nearest=False):
        """Box/label index for support warm starts; scalar values stay hints."""
        cached=None if nearest else getattr(self,'_support_index',None)
        if cached is not None:return cached
        pl,pu,*_=self.data
        _,group=torch.unique_consecutive(torch.cat((pl,pu),1),dim=0,return_inverse=True)
        G=int(group.max())+1;K=2*pl.shape[1];N=len(pl)
        ar=torch.arange(N,device=pl.device)
        first=torch.full((G,),N,dtype=torch.long,device=pl.device).scatter_reduce_(0,group,ar,reduce='amin')
        key=group*K+self.labels
        index=torch.full((G*K,),N,dtype=torch.long,device=pl.device).scatter_reduce_(0,key,ar,reduce='amin')
        good=index.clone().fill_(N)
        if self.prefer_converged and not nearest and self.converged is not None:
            good.scatter_reduce_(0,key,torch.where(self.converged,ar,N),reduce='amin')
            index=torch.where(good<N,good,index)
        index=torch.where(index<N,index,-1).reshape(G,K)
        result=(first,index,(good<N).reshape(G,K))
        if not nearest:self._support_index=result
        return result

    def save(self, lo, hi, z, y, bound, cutoff, caps=None, converged=None, labels=None):
        values=(lo,hi,z,y,bound,cutoff)
        if not len(lo):return
        if labels is not None:
            if self.labels is not None:labels=torch.cat((self.labels,labels))
            self.labels=labels[-self.capacity:].detach().clone()
        if self.prefer_converged:
            flags=lo.new_zeros(len(lo),dtype=torch.bool) if converged is None else converged
            if self.data is not None:
                old=self.converged
                if old is None:old=lo.new_zeros(len(self.data[0]),dtype=torch.bool)
                flags=torch.cat((old,flags))
            self.converged=flags[-self.capacity:].detach().clone()
        if caps is not None:
            if self.data is not None:
                old=self.caps
                if old is None:
                    old=(caps[0].new_full((len(self.data[0]),caps[0].shape[1]),-torch.inf),
                         caps[1].new_full((len(self.data[0]),caps[1].shape[1]),torch.inf))
                caps=tuple(torch.cat((a,b)) for a,b in zip(old,caps))
            self.caps=tuple(v[-self.capacity:].detach().clone() for v in caps)
        else:self.caps=None
        if self.data is not None:
            values=tuple(torch.cat((a,b)) for a,b in zip(self.data,values))
        # Only lane candidates reside here. Old search memory may be evicted;
        # GEAR retains each node's certified scalar independently of this bank.
        self.data=tuple(v[-self.capacity:].detach().clone() for v in values)
        if labels is not None:self._support_index=None


def probe_bits(lo, hi, ll, uu, floor):
    """Certified bisections' worth of volume removed from [lo,hi] by [ll,uu], per box (scheduling only).

    A coordinate counts only if it narrowed by more than the float rounding of its two endpoints
    (machine epsilon times their magnitude): volume ratios at float resolution are 0. A coordinate
    narrowed below `floor` (the feasibility tolerance times the root width) is fixed, so its bits are
    capped at log2(old width / floor); one already at the floor adds none.
    """
    eps=torch.finfo(lo.dtype).eps
    old=hi-lo;new=(uu-ll).clamp_min(0)
    resolution=eps*torch.stack((lo.abs(),hi.abs(),ll.abs(),uu.abs())).amax(0)
    # Native bisection cannot resolve below the float resolution either (opus_review_n9 B), and a
    # narrowing smaller than the floor is not a shrink anyone needs: both earn 0 bits.
    floor=torch.maximum(torch.as_tensor(floor,dtype=lo.dtype,device=lo.device),2*resolution)
    real=(old-new>floor)&(old>floor)
    return torch.where(real,torch.log2(old/torch.maximum(new,floor)),0.).sum(1)


def slot_workspace_bytes(items, n, m, nnz, width=None, boxes=None):
    """Peak bytes of one solve_pool queue of `items` run through `width` slots (A158, A159, nodes8a).

    Everything is rounded up to a power of two (solve_pool's buckets), float64 words.
    Per slot (C=pow2(width); every slot owns its vectors and its CSR):
      nnz-sized (16): rows.val, scaled (2); the two pricing matrices' ELL values in both orientations
        (4); the batched CSR's values and int64 columns, both orientations (4, halpern; the shared
        CSR has none); reset's rn/cn rescaling and product() terms transients (2); a checkpoint's
        gathered x[:,src] and v*x terms (2), the certificates' |val| and per-entry terms (2).
      n-sized (30+6): w,ws,wl,previous_w (4); reset's mid,rad,lw,uw,cs,qs,S,Dc,v (9); pt's c,q (3); a
        checkpoint's gathered c,l,u,q,mid,S,lw,uw,cs,qs (10) and point/residual temporaries (4); the
        slot's own objective, scaling and pt gather (6, box tables only).
      m-sized (12) fits inside the same factor, as do ~25 per-slot scalars (+10 per item).
    `boxes=None` is the per-item layout: every item owns a table copy, per queue item
    (nnz-sized (2): padded values and the qr.val clone; n-sized (10): padded and cloned c,l,u,q, z0, zout;
    m-sized (5): padded and cloned h, y0, yout, best_y).
    With `boxes` (G tables, items are int32 triples): per box 6 nnz-sized words (values, scaled, reset
    transients), 16 n-sized (c,l,u,q, the scalings, transients), 7 m-sized (h and its scalings); per item
    12 bytes of triple plus its warm and output vectors (2 n-sized, 3 m-sized) and ~7 scalars. One call
    queues at most one wave, so the per-item vectors are bounded by the lane budget.
    """
    pow2=lambda k:1<<(max(1,k)-1).bit_length()
    Q=pow2(items);C=min(Q,pow2(items if width is None else width))
    if boxes is None:
        return 8*(Q*(10*(n+m)+2*nnz)+C*(30*(n+m)+16*nnz))
    return 8*(pow2(boxes)*(6*nnz+16*n+7*m+4)+Q*(2*n+3*m+7)+C*(36*n+30*m+16*nnz))+12*Q


class LruCache(dict):
    """solve_pool's workspace cache; every get or set stamps the entry from a clock shared by a pool's caches."""
    def __init__(self, clock):
        super().__init__();self.clock=clock;self.stamp={}
    def __getitem__(self, key):
        self.stamp[key]=next(self.clock);return super().__getitem__(key)
    def __setitem__(self, key, value):
        self.stamp[key]=next(self.clock);super().__setitem__(key,value)
    def pop(self, key, *default):
        self.stamp.pop(key,None);return super().pop(key,*default)
    def clear(self):
        self.stamp.clear();super().clear()


class NodePool:
    def __init__(self, engine):
        self.engine=engine
        self.parents=ParentBank(engine.top,prefer_converged=engine.cfg.get('crown_engine_node_warm',False))
        self._tables=None
        self.clock=itertools.count()
        self.cache=LruCache(self.clock)
        self.metrics={}
        self.branch_cache=LruCache(self.clock)
        self.cutoff_pattern=None
        self.direction=engine.c.new_tensor(np.random.default_rng(0).standard_normal(engine.n))
        self.input_caps=None
        self.probe_parents=None
        self.probe_cache=LruCache(self.clock)
        self.models={}  # measured workspace bytes per shape class
        self.lanes_per_box=None  # measured lanes (node + due directions) of a selected box, sizes the next wave
        self.start_widths={}
        self.last_bits=None;self.last_ran=False;self.last_paused=False;self.last_lanes_used=0

    @property
    def tables(self):
        if self._tables is None:
            e=self.engine
            with e.memory.allocations() if e.memory is not None else nullcontext():
                if e.memory is not None:e.memory.require(e.scratch_bytes)
                self._tables=NodeTables(e)
        return self._tables

    def _caches(self):
        return (self.cache,self.branch_cache,self.probe_cache)

    def evict_one(self):
        """Drop the least recently used workspace of the three caches; False if none is left. Cached
        workspaces are optional residency: admission counts them as reclaimable, never as a cost."""
        held=[(c.stamp[k],c,k) for c in self._caches() for k in c]
        if not held:return False
        _,c,k=min(held,key=lambda t:t[0])
        c.pop(k)
        from auto_LiRPA.cuda_memory import release_cuda
        release_cuda(self.engine.dev)
        return True

    def reclaimable(self):
        """Bytes held by the cached workspaces."""
        memory=self.engine.memory
        if not getattr(memory,'enabled',False):return 0
        from auto_LiRPA.cuda_memory import owned_cuda_bytes
        return owned_cuda_bytes(self._caches(),[s for s in torch.cuda.memory_snapshot() if s['device']==memory.device.index])

    def workspace_bytes(self, items, n, m, nnz, width, boxes=None):
        """Peak bytes of a solve_pool call. On CUDA: the per-slot and per-box bytes measured on this device
        for this shape class (see measure_workspace), else the analytic slot_workspace_bytes."""
        model=self.models.get((n,m,nnz,self.engine.cfg.get('crown_engine_halpern',False)))
        if model is None or boxes is None:return slot_workspace_bytes(items,n,m,nnz,width,boxes)
        pow2=lambda k:1<<(max(1,k)-1).bit_length()
        Q=pow2(items);C=min(Q,pow2(items if width is None else width))
        base,slot,box,item=model
        return int(base+slot*C+box*pow2(boxes)+item*Q)

    @torch.no_grad()
    def measure_workspace(self, rows):
        """Measure, once per shape class, the peak reserved bytes of a cold captured probe call (graph pool,
        reset scaling and all) at S/4 and S slots and at 2 and 2+S/4 boxes; the model is linear in each."""
        from auto_LiRPA.node_lanes import solve_pool
        from nlp_solver.rigorous_dual import Rows
        from input_split.m1_qp import solver_options
        e=self.engine;dev=e.dev;n,m,nnz=rows.n,rows.m,len(rows.row)
        key=(n,m,nnz,e.cfg.get('crown_engine_halpern',False))
        if key in self.models or dev.type!='cuda' or e.memory is None:return
        S=256  # wide enough that the allocator's block rounding does not dominate the slope
        while S>2 and slot_workspace_bytes(S,n,m,nnz,S,S)>e.memory.available()//3:S//=2
        def peak(C,G):
            val=rows.val[:1].expand(G,-1).contiguous()
            table=(Rows(rows.row,rows.col,val,m,n),val.new_zeros(G,n),val.new_zeros(G),val.new_zeros(G,m),
                   val.new_full((G,n),-1.),val.new_ones(G,n),val.new_zeros(G,n))
            items=(torch.arange(C,device=dev,dtype=torch.int32)%G,torch.arange(C,device=dev,dtype=torch.int32)%n,
                   torch.ones(C,device=dev,dtype=torch.int32))
            cache={}
            torch.cuda.synchronize(dev);torch.cuda.empty_cache()
            before=torch.cuda.memory_reserved(dev);torch.cuda.reset_peak_memory_stats(dev)
            solve_pool(table,capacity=C,budget=1,stop=val.new_full((C,),torch.inf),support=val.new_zeros(C),
                       capture=True,cache=cache,items=items,**solver_options(e))
            torch.cuda.synchronize(dev)
            used=torch.cuda.max_memory_reserved(dev)-before
            del cache,table,items
            torch.cuda.empty_cache()
            return used
        # G=2 is the smallest production layout: one CSR per slot (a single box shares one matrix)
        item=8*(2*n+3*m+7)+12
        p1,p2,pg=peak(S//4,2),peak(S,2),peak(S//4,2+S//4)
        slot=max(0.,(p2-p1)/(S-S//4)-item);box=max(0.,(pg-p1)/(S//4))
        self.models[key]=(max(0.,p1-(S//4)*(slot+item)-2*box),slot,box,item)
        print('[node-pool] workspace model '+json.dumps(dict(n=n,m=m,nnz=nnz,base=self.models[key][0],slot=slot,box=box,item=item)),flush=True)

    def _width(self, items, rows, boxes=None):
        """Lane width for a queue of `items`: the whole queue, else the widest power of two memory admits.

        solve_pool serves a longer queue by FIFO refill, so a narrower width serializes and never drops.
        Cached workspaces are reclaimable: LRU entries are evicted before a width is given up.
        If even one slot does not fit, require() raises and the caller's pause path runs.
        """
        e=self.engine
        if e.memory is None:return items
        self.measure_workspace(rows)
        need=lambda w:self.workspace_bytes(items,rows.n,rows.m,len(rows.row),w,boxes)
        width=1<<(items-1).bit_length()
        while True:
            avail=e.memory.available()
            if need(width)<=avail or not self.evict_one():break
        while width>1 and need(width)>avail:width>>=1
        e.memory.require(need(width))
        return items if width>=items else width

    def _shape(self):
        """The root box's cutoff table, once: it fixes (n,m,nnz) of this shape class."""
        e=self.engine;dev=e.dev
        lo,hi=e.root_lo.flatten()[None].to(dev,torch.float64),e.root_hi.flatten()[None].to(dev,torch.float64)
        T=lo.new_full((1,),torch.inf)
        shape=(dev.type,dev.index,e.cfg.get('crown_engine_halpern',False))
        if shape not in self.start_widths:
            data,_,_=self.tables.build(lo,hi,T)
            r,c,c0,h,l,u,q=pt_table(data,T,pattern=self.cutoff_pattern)
            self.cutoff_pattern=(r.row,r.col)
            self.start_widths[shape]=[(r.n,r.m,len(r.row)),None,(r,c,c0,h,l,u,q),r]
        return shape,lo

    def _wide(self, n, m, nnz, rows=None):
        """The widest power-of-two slot count (<= every signed direction of every resident box) memory admits."""
        e=self.engine
        top=max(1,e.top)*2*e.n
        if e.memory is None:return top
        if rows is not None:self.measure_workspace(rows)
        avail=e.memory.available()+self.reclaimable()  # cached workspaces are evicted on demand by _width
        wide=1
        while wide<top and self.workspace_bytes(2*wide,n,m,nnz,2*wide,1)<=avail:wide*=2
        return wide

    @torch.no_grad()
    def admitted_lanes(self, lower):
        """Memory-admitted lane width, in lanes (a box brings its node lane and its due directions)."""
        try:
            shape,_=self._shape()
            (n,m,nnz),_,_,rows=self.start_widths[shape]
            return self._wide(n,m,nnz,rows)
        except Exception as error:
            from auto_LiRPA.cuda_memory import memory_error
            if not (memory_error(error) or isinstance(error,ValueError) and str(error).startswith('strict ')):raise
            return 1

    @torch.no_grad()
    def lane_start(self, lower):
        """Start width W*: where one step takes twice the width-1 time (t(W)=a+bW, so W*=1+t(1)/b).

        Timed once per table shape class on this device on the production path (G>=2 boxes, one CSR per
        slot; 25 steps at width 1 and at the memory-admitted width) and capped by the memory-admitted
        width. Guarded like bound(): an OOM pauses M1 (returns 0, last_ran False) instead of raising.
        """
        from auto_LiRPA.cuda_memory import memory_error
        from input_split.m1_qp import memory_ready
        e=self.engine
        self.last_ran=False;self.last_paused=False
        if getattr(self,'strict_root',False) or not memory_ready(e,'m1'):return 0
        try:
            with e.memory.allocations() if e.memory is not None else nullcontext():
                return self._lane_start(lower)
        except Exception as error:
            if isinstance(error,ValueError) and str(error).startswith('strict '):
                self.strict_root=True  # the root table is fixed: M1 node lanes never start (opus_review_n9 A1)
                return 0
            if not memory_error(error):raise
            reason=f'{type(error).__name__}: {str(error)[:200]}'
        self.pause(reason)
        return 0

    def _lane_start(self, lower):
        from auto_LiRPA.node_lanes import solve_pool
        from input_split.m1_qp import solver_options
        e=self.engine;dev=e.dev
        shape,lo=self._shape()
        (n,m,nnz),timing,table,rows=self.start_widths[shape]
        wide=self._wide(n,m,nnz,rows)
        if timing is None:
            r,c,c0,h,l,u,q=table
            G=2  # production waves hold several boxes: per-slot CSR, not the shared single-box matrix
            from nlp_solver.rigorous_dual import Rows
            rep=lambda t:t[:1].expand(G,*t.shape[1:]).contiguous()
            tab=(Rows(r.row,r.col,rep(r.val),r.m,r.n),rep(c*0),rep(c0*0),rep(h),rep(l),rep(u),rep(q))
            def clock(width):
                items=(torch.arange(width,dtype=torch.int32,device=dev)%G,torch.arange(width,dtype=torch.int32,device=dev)%n,
                       torch.ones(width,dtype=torch.int32,device=dev))
                cache={};seconds=[]
                for _ in range(2):  # the first call captures
                    o=solve_pool(tab,capacity=width,budget=25,stop=lo.new_full((width,),torch.inf),
                                 support=lo.new_zeros(width),capture=dev.type=='cuda',cache=cache,items=items,**solver_options(e))
                    seconds.append(o.metrics['iteration_seconds']/max(1,o.metrics['lane_steps']/(o.metrics['capacity'])))
                return seconds[-1]
            # Widen by 16x from width 1 until a step takes twice the width-1 time (W* is then inside the last
            # interval) or the admitted width is reached: the crossing is all that is needed, so the timing
            # never has to hold the whole admitted width.
            t1=tw=clock(1);w=1
            while w<wide and tw<2*t1:
                w=min(wide,16*w);tw=clock(w)
            # t(W)=a+bW with t(1)=t1: the width where t=2*t1 is 1+t1/b. Memory only caps it, per call.
            timing=self.start_widths[shape][1]=(t1,(tw-t1)/max(1,w-1))
            self.start_widths[shape][2]=None
        t1,slope=timing
        return wide if slope<=0 else max(1,min(wide,int(1+t1/slope)))

    def pause(self, reason=None):
        """Transient allocator pressure: drop the optional residency, record the pause. Never removes a box or a bound."""
        e=self.engine
        self.parents.data=None;self.parents.caps=None;self.cache.clear();self.branch_cache.clear();self._tables=None
        if e.cfg.get('crown_engine_node_probe2',False):
            self.input_caps=None;self.probe_parents=None;self.probe_cache.clear()
            self.parents.probe_schedule=None
        from input_split.m1_qp import memory_pause
        memory_pause(e,'m1',1)
        e.stats['pool_memory_pauses']=e.stats.get('pool_memory_pauses',0)+1
        self.last_ran=False;self.last_paused=True
        if reason is not None:print(f'[node-pool] memory pause: {reason}',flush=True)

    @torch.no_grad()
    def bound(self, lower, upper, native, thresholds, deadline, lanes, history=None):
        e=self.engine
        self.last_bits=lower.new_zeros(len(lower),dtype=torch.float64);self.last_ran=False;self.last_paused=False;self.last_lanes_used=0
        from auto_LiRPA.cuda_memory import memory_error
        from input_split.m1_qp import memory_ready
        if deadline is not None and time.time()>=deadline:return native
        if not memory_ready(e,'m1'):
            return native  # a wait, not a pause: L is not halved (opus_review_n9 A3)
        try:
            with e.memory.allocations() if e.memory is not None else nullcontext():
                return self._bound(lower,upper,native,thresholds,deadline,lanes,history)
        except Exception as error:
            if isinstance(error,ValueError) and str(error).startswith('strict '):
                e.stats['undefined_fallbacks']=e.stats.get('undefined_fallbacks',0)+1
                if history is not None:
                    history.engine_witness=None
                    history.engine_valid=torch.zeros(len(lower),dtype=torch.bool,device=lower.device)
                return native
            if not memory_error(error):raise
            reason=f'{type(error).__name__}: {str(error)[:200]}'
        self.pause(reason)
        return native

    def _bound(self, lower, upper, native, thresholds, deadline, lanes, history):
        from auto_LiRPA.node_lanes import solve_pool
        from input_split.m1_qp import solver_options
        from nlp_solver.rigorous_dual import round_down
        e=self.engine;started=time.perf_counter()
        open_=~((native>thresholds).any(1) | (native[:,0]>=thresholds[:,0]))
        order=torch.argsort(torch.where(open_,native[:,0],torch.inf),stable=True)
        ids=order[open_[order]]
        if not len(ids):return native
        # One wave per call: `lanes` is the lane budget L. Take the lowest-bound boxes that L can hold at
        # the lanes per box measured on the previous wave (a box has at most its node lane and 2n
        # directions); _probe2 then fills L exactly from the directions actually due.
        probe2=e.cfg.get('crown_engine_node_probe2',False)
        if probe2:
            free=(upper[ids]>lower[ids]).flatten(1).sum(1)
            per_box=free.new_full(free.shape,self.lanes_per_box,dtype=torch.float64) if self.lanes_per_box else 1+2*free.double()
            ids=ids[:max(1,int((per_box.cumsum(0)<=lanes).sum()))]
        else:ids=ids[:max(1,int(lanes))]
        if e.memory is not None:
            parents=0 if self.parents.data is None else len(self.parents.data[0])
            schedule=getattr(self.parents,'probe_schedule',None)
            scheduled=0 if schedule is None or schedule.data is None else len(schedule.data[0])
            # Peak ancestry lookup: two comparisons plus their conjunction,
            # then one bool scope and one binary64 rank per stored parent.
            per_box=e.scratch_bytes+(parents+scheduled)*(3*e.n+9)
            room=int(e.memory.available()//per_box)
            if room<1:e.memory.require(per_box)
            limit=1<<max(0,room.bit_length()-1)
            ids=ids[:limit]
            e.memory.require(len(ids)*per_box)
            self.parents.capacity=max(self.parents.capacity,len(ids))
        lo=lower[ids].flatten(1).to(e.dev,torch.float64).clone()
        hi=upper[ids].flatten(1).to(lo).clone();T=thresholds[ids,0].to(lo)
        scope_lo,scope_hi=lo.clone(),hi.clone()
        inherited_empty=torch.zeros_like(T,dtype=torch.bool)
        if e.cfg.get('crown_engine_node_probe2',False) and self.input_caps is not None:
            caps=self.input_caps.caps_for(lo,hi,T)
            if caps is not None:
                ll,uu=torch.maximum(lo,caps[0]),torch.minimum(hi,caps[1])
                crossed=ll>uu
                inherited_empty=crossed.any(1)
                # These are endpoints certified on a containing cutoff set,
                # not an arbitrary empty-box hint. Keep other contractions
                # but give numerical kernels a nonempty placeholder column.
                lo=torch.where(crossed,lo,ll);hi=torch.where(crossed,hi,uu)
        data,d,empty=self.tables.build(lo,hi,T,caps=self.parents.caps_for(lo,hi,T))
        empty |= inherited_empty
        pt=e.cfg.get('crown_engine_node_pt',False)
        seed,source=crown_seed(e,d,data,with_certificate=True,multipliers=getattr(history,'dual_lam_star',None))
        if pt:seed=torch.cat((seed,seed.new_zeros(len(seed),1)),1)
        z,y,floor,found=self.parents.seed(scope_lo,scope_hi,((data[4]+data[5])/2,seed),T)
        z=torch.maximum(data[4],torch.minimum(data[5],z))
        tick=time.perf_counter()
        step0=torch.fmax(certificate(data,y[:,:data[0].m] if pt else y),source)
        if pt:
            hd=pt_table(data,T,pattern=self.cutoff_pattern);r,c,c0,h,l,u,q=hd
            self.cutoff_pattern=(r.row,r.col)
            # Ski rental: buy a split after its measured two-child cost.
            # Until a warm table-build/iteration sample exists, T2's 1.5K
            # envelope preserves its late Farkas detections.
            budget=max(0,int(e.cfg.get('crown_engine_m1_steps',1000)))
            rent=0
            if self.tables.metrics['cache_hit'] and self.metrics.get('active_lane_steps',0):
                per_step=self.metrics['iteration_seconds']/self.metrics['active_lane_steps']
                rent=int(2*(self.tables.metrics['seconds']+time.perf_counter()-tick)/len(ids)/max(per_step,1e-300))
            budget=max((3*budget+1)//2,rent)
            o=solve_pool((r,torch.zeros_like(c),torch.zeros_like(c0),h,l,u,q),capacity=self._width(len(ids),r),budget=budget,
                stop=torch.where((step0>=T)|empty,-torch.inf,T.new_full(T.shape,1e-300)),warm=(z,y),
                capture=e.dev.type=='cuda',cache=self.cache,deadline=deadline,**solver_options(e),
                pt=(r.m-1,data[1],data[2],data[6],T))
            # The c=0 lower is a Farkas value, never an objective bound.
            bound=torch.minimum(torch.fmax(step0,floor),T)
        else:
            control={}
            if e.cfg.get('crown_engine_node_warm',False):
                width=((hi-lo)/(e.root_hi-e.root_lo).clamp_min(1e-300)).amax(1).clamp(max=1.)
                scale=torch.where(torch.isfinite(T),T.abs(),step0.abs()).clamp_min(1.)
                tau=torch.maximum(scale*e.rel_tol,T.new_full(T.shape,e.abs_tol))
                control['dual_accuracy']=(T,tau,width)
            o=solve_pool(data,capacity=self._width(len(ids),data[0]),budget=max(0,int(e.cfg.get('crown_engine_m1_steps',1000))),
                stop=torch.where((step0>=T)|empty,-torch.inf,T),warm=(z,y),accuracy=e.abs_tol,capture=e.dev.type=='cuda',cache=self.cache,deadline=deadline,**solver_options(e),**control)
            bound=torch.minimum(torch.fmax(torch.fmax(o.lower,step0),floor),T)
        zero=certificate(data,torch.zeros_like(seed[:,:data[0].m])) if pt else certificate(data,torch.zeros_like(y))
        hybrid=torch.zeros_like(empty) if pt else ((certificate(data,o.best_y)<=zero) | (o.best_y==0).all(1)) & ~empty & ~o.empty & (bound<T)
        witness=o.z.clone()
        hybrid_steps=0
        if bool(hybrid.any()):
            # One generic direction per uninformative node, on the cutoff
            # relaxation R. This is search only; it never supplies a bound.
            hd=cutoff_table(data,T,o.z,pattern=self.cutoff_pattern);r,c,c0,h,l,u,q=hd
            self.cutoff_pattern=(r.row,r.col)
            chosen=hybrid.nonzero().flatten()
            from nlp_solver.rigorous_dual import Rows
            r=Rows(r.row,r.col,r.val[chosen],r.m,r.n)
            c,c0,h,l,u,q=(v[chosen] for v in (c,c0,h,l,u,q))
            g=torch.zeros_like(c);g[:,:e.n]=self.direction*(hi-lo)[chosen]
            probe=(r,g,torch.zeros_like(c0),h,l,u,q)
            direction=solve_pool(probe,capacity=self._width(len(chosen),r),budget=max(0,int(e.cfg.get('crown_engine_m1_steps',1000))),
                stop=c0.new_full(c0.shape,torch.inf),accuracy=e.abs_tol,capture=e.dev.type=='cuda',
                cache=self.branch_cache,deadline=deadline,**solver_options(e))
            witness[chosen]=direction.z
            hybrid_steps=direction.metrics['active_lane_steps']
        # FBBT used the cutoff; its exclusion of improving points proves T.
        bound=torch.where(empty|o.empty,T,bound)
        probe_metrics={}
        schedule=None
        witness0=witness.clone()  # each box's own node point, before any probe refreshes the table
        selected=torch.ones_like(T,dtype=torch.bool);attempted=None
        if e.cfg.get('crown_engine_node_probe2',False):
            schedule=getattr(self.parents,'probe_schedule',None)
            if schedule is None:
                schedule=self.parents.probe_schedule=ProbeSchedule()
            state=schedule.lookup(scope_lo,scope_hi,T)
            eligible,stalled,backoff=schedule.due(T,bound,state)
            data,d,lo,hi,bound,witness,probe_metrics=self._probe2(
                data,d,lo,hi,T,bound,witness,scope_lo,scope_hi,lanes,deadline,eligible)
            selected=torch.tensor(probe_metrics.pop('probe2_selected'),device=T.device)
            for key in ('active_lane_steps','lane_steps','iteration_seconds'):
                o.metrics[key]+=probe_metrics.pop('probe2_node_'+key,0)
            if probe_metrics['probe2_directions']:
                # Rebuilt faces carry the gain from OBBT. Reusing only the
                # pre-probe scalar would leave the current node under-solved.
                old=o
                _,source2=crown_seed(e,d,data,with_certificate=True)
                dual=old.best_y[:,:data[0].m]
                step2=torch.fmax(certificate(data,dual),source2)
                budget2=max(0,int(e.cfg.get('crown_engine_m1_steps',1000)))
                if pt:
                    hd=pt_table(data,T,pattern=self.cutoff_pattern);r,c,c0,h,l,u,q=hd
                    self.cutoff_pattern=(r.row,r.col)
                    o=solve_pool((r,c*0,c0*0,h,l,u,q),capacity=self._width(len(ids),r),budget=(3*budget2+1)//2,
                        stop=torch.where((bound>=T)|~selected,-torch.inf,T.new_full(T.shape,1e-300)),
                        warm=(witness,old.y),capture=e.dev.type=='cuda',cache=self.cache,
                        deadline=deadline,pt=(r.m-1,data[1],data[2],data[6],T),**solver_options(e))
                    bound=torch.maximum(bound,step2)
                else:
                    o=solve_pool(data,capacity=self._width(len(ids),data[0]),budget=budget2,
                        stop=torch.where((bound>=T)|~selected,-torch.inf,T),warm=(witness,dual),accuracy=e.abs_tol,
                        capture=e.dev.type=='cuda',cache=self.cache,deadline=deadline,
                        **solver_options(e),**control)
                    bound=torch.maximum(bound,torch.fmax(o.lower,step2))
                bound=torch.where(o.empty,T,torch.minimum(bound,T))
                witness=o.z.clone()
                for key in ('active_lane_steps','lane_steps','iteration_seconds'):
                    o.metrics[key]+=old.metrics[key]
                # Residency is about either solve having done useful work.
                o.metrics['lane_iterations']=(torch.tensor(old.metrics['lane_iterations'])+torch.tensor(o.metrics['lane_iterations'])).tolist()
            # Narrow dtypes are converted outwards. These are the very same
            # certified input caps used to rebuild the next GEAR node table.
            lower[ids]=round_down(lo,lower.dtype).reshape_as(lower[ids])
            upper[ids]=(-round_down(-hi,upper.dtype)).reshape_as(upper[ids])
            attempted=torch.tensor(probe_metrics.pop('probe2_attempted',[False]*len(T)),device=T.device)
            effective=torch.tensor(probe_metrics.pop('probe2_effective',[False]*len(T)),device=T.device)
            if e.memory is not None:e.memory.require(8*len(T)*(2*e.n+7))
            # A box outside the wave keeps its schedule entry: it stays due.
            path=state[0].clone()
            path[selected]=schedule.save(scope_lo[selected],scope_hi[selected],T[selected],bound[selected],
                tuple(v[selected] for v in state),attempted[selected],effective[selected],stalled[selected])
            probe_metrics.update(probe2_attempts=attempted.tolist(),probe2_effectiveness=(attempted&(effective|(bound>=T))).tolist(),
                                 probe2_stall=stalled.tolist(),probe2_backoff=backoff.tolist(),
                                 probe2_path=path.tolist())
        result=native.clone()
        taken=ids[selected]  # boxes outside the wave return their baseline
        result[taken,0]=torch.fmax(native[taken,0],round_down(bound[selected],native.dtype))
        if e.memory is not None:e.memory.require(8*len(ids)*(2*e.n+e.qp.n+e.qp.m+2+2*(e.ncol-e.n)))
        resident=(bound<T) & torch.tensor(o.metrics['lane_iterations'],device=T.device).gt(0) & selected
        from input_split.lane_primal import offer
        rank=torch.maximum(bound,native[ids,0].to(bound))
        offer(e,o.z[~empty & ~o.empty],rank[~empty & ~o.empty])
        qualified=None
        if e.cfg.get('crown_engine_node_warm',False):
            qualified=torch.tensor(o.metrics.get('reasons',[0]*len(ids)),device=T.device)==(5 if pt else 2)
            # The best certificate and its dual travel together. Unconverged
            # iterates remain fallback search memory, never qualified ancestors.
        self.parents.save(scope_lo[resident],scope_hi[resident],o.z[resident],(o.y if pt or qualified is None else o.best_y)[resident],bound[resident],T[resident],caps=(d.lo[resident,e.n:],d.hi[resident,e.n:]),converged=None if qualified is None else qualified[resident])
        self.metrics=dict(o.metrics,step0=step0.tolist(),warm=int(found.sum()),nodes=len(ids),hybrid_lane_steps=hybrid_steps,
                          seconds=time.perf_counter()-started,table_build_seconds=self.tables.metrics['seconds'],**probe_metrics)
        e.calls+=1
        # What the controller reads: certified bits per input box, only where a wave probed; lanes the wave held.
        bits=probe_metrics.get('probe2_bits')
        if bits is not None:self.last_bits[ids]=torch.tensor(bits,dtype=torch.float64,device=lower.device)*selected.to(lower.device)
        self.last_ran=True;self.last_lanes_used=int(selected.sum())+probe_metrics.get('probe2_directions',0)
        if probe2:self.lanes_per_box=self.last_lanes_used/max(1,int(selected.sum()))
        for key,value in (('node_lane_steps',o.metrics['active_lane_steps']+probe_metrics.get('probe_lane_steps',0)),
                          ('node_decided',o.metrics['decided']),('node_nodes',int(selected.sum())),
                          ('node_slot_steps',o.metrics['lane_steps']+probe_metrics.get('probe_slot_steps',0)),
                          ('node_build_seconds',self.tables.metrics['seconds'])):
            e.stats[key]=e.stats.get(key,0)+value
        e.stats['node_occupancy']=e.stats['node_lane_steps']/max(1,e.stats['node_slot_steps'])
        print('[node-pool] '+json.dumps({k:v for k,v in self.metrics.items() if k!='step0'}),flush=True)
        if history is not None:
            point=lower.new_full((len(lower),e.ncol),torch.nan)
            # W0=2: a probed box splits at its refreshed point, every other box at its own node point.
            point[ids]=e.qp.expand(witness0 if attempted is None else torch.where(attempted[:,None],witness,witness0)).to(point)
            history.engine_witness=point
            history.engine_valid=torch.isfinite(point).all(1)
            node_branch(e,lower,upper,history)
        return result

    def _probe2(self, data, d, lo, hi, T, bound, witness, scope_lo, scope_hi, lanes, deadline, eligible):
        """Batched +/- input supports on the production cutoff table, as ONE wave under the lane budget.

        Boxes arrive lowest bound first. Whole boxes fill `lanes`: a box counts its node lane plus the
        score>0 signed directions that rule R makes due (a decided box counts none). At least one box is
        taken; the others keep their native bound and stay due. Only current-table outward dual endpoints
        change a box. The primal gap and point-relative residual control search accuracy, never soundness.
        The queue keeps one table per selected box and an (owner, j, sign) triple per item.
        """
        from auto_LiRPA.node_lanes import solve_pool
        from nlp_solver.rigorous_dual import Rows
        from input_split.m1_qp import solver_options
        e=self.engine;B=len(lo)
        score=node_score(e,e.qp.expand(witness),lo,hi)*(hi-lo)
        due=eligible[:,None]&(score>0)
        cost=torch.where(bound<T,1+2*due.sum(1),0)
        keep=torch.cumsum(cost,0)<=lanes
        keep[0]=True
        eligible=eligible&keep
        owner,var=(eligible[:,None]&(score>0)).nonzero(as_tuple=True)
        owner=owner.repeat_interleave(2);var=var.repeat_interleave(2)
        N=len(owner)
        extra=dict(probe2_selected=keep.tolist())
        if not N:
            costs=dict(probe_serial_steps=0) if e.cfg.get('crown_engine_node_probe_cost',False) else {}
            return data,d,lo,hi,bound,witness,dict(probe2_directions=0,probe_lane_steps=0,**costs,**extra)
        r,c,c0,h,l,u,q=pt_table(data,T,pattern=self.cutoff_pattern)
        self.cutoff_pattern=(r.row,r.col)
        if e.memory is not None:
            old=0 if self.probe_parents is None or self.probe_parents.data is None else len(self.probe_parents.data[0])
            groups=0 if not old else len(self.probe_parents.support_index()[0])
            e.memory.require(B*groups*(3*e.n+9)+24*N*groups)
        sign=lo.new_tensor([1.,-1.]).repeat(N//2)
        labels=2*var+(sign<0).long()
        sl,su,tt=scope_lo[owner],scope_hi[owner],T[owner]
        if self.probe_parents is None:self.probe_parents=ParentBank(max(1,e.top)*2*e.n,prefer_converged=True)
        z,y,_,found=self.probe_parents.seed(sl,su,(witness[owner],h.new_zeros(N,r.m)),tt,labels=labels,
            nearest=e.cfg.get('crown_engine_node_probe_warm',False))
        widths=(hi-lo)[owner,var]
        # O1's 2e4 condition-number budget, with T1 O5's fourfold probe
        # allowance. The 1%-width stopping mask retires solved directions.
        # Non-waiting nodes and support lanes use one queue and one table per
        # box. A checkpoint reserves at least as many active node slots as
        # probe slots until the node queue drains. All-waiting frontiers may
        # use the entire bucket for useful support work.
        nodes=(keep&~eligible&(bound<T)).nonzero().flatten()
        is_pt=e.cfg.get('crown_engine_node_pt',False)
        taken=torch.cat((nodes,owner.unique()))
        place=torch.full((B,),-1,dtype=torch.long,device=lo.device);place[taken]=torch.arange(len(taken),device=lo.device)
        # The table of each box in the queue. Node lanes minimise the real objective (c, or the
        # cutoff-row test of the pt lane); probes minimise the implicit sign*e_j.
        tables=(Rows(r.row,r.col,r.val[taken],r.m,r.n),(c*0 if is_pt else c)[taken],(c0*0 if is_pt else c0)[taken],
                h[taken],l[taken],u[taken],q[taken])
        items=(torch.cat((nodes,owner)),torch.cat((nodes.new_full((len(nodes),),-1),var)),torch.cat((sign.new_zeros(len(nodes)),sign)))
        items=(place[items[0]].int(),items[1].int(),items[2].int())
        pt=None
        if is_pt and len(nodes):pt=(r.m-1,c[taken],c0[taken],data[6][taken],T[taken])
        z=torch.cat((witness[nodes],z));y=torch.cat((h[nodes]*0,y))
        target=torch.cat((T[nodes]*0+1e-300 if is_pt else T[nodes],tt.new_full(tt.shape,torch.inf)))
        tol=torch.cat((T[nodes]*0,widths/100))
        extra_args={}
        if e.cfg.get('crown_engine_node_probe_cost',False):extra_args['probe_account']=True
        if e.cfg.get('crown_engine_node_probe_anytime',False):
            # A split buys two child solves at the configured node allowance.
            extra_args['probe_work']=2*max(0,int(e.cfg.get('crown_engine_m1_steps',1000)))
        queue=len(items[0])
        o=solve_pool(tables,capacity=self._width(queue,r,boxes=len(taken)),budget=4*max(20000,int(e.cfg.get('crown_engine_m1_steps',1000))),
            stop=target,warm=(z,y),capture=e.dev.type=='cuda',cache=self.probe_cache,
            deadline=deadline,support=tol,items=items,pt=pt,**extra_args,**solver_options(e))
        if len(nodes):
            if not is_pt:bound[nodes]=torch.maximum(bound[nodes],o.lower[:len(nodes)])
            bound[nodes]=torch.where(o.empty[:len(nodes)],T[nodes],bound[nodes])
            witness[nodes]=o.z[:len(nodes)]
            steps=o.metrics['lane_iterations']
            extra.update(probe2_node_active_lane_steps=sum(steps[:len(nodes)]),
                       probe2_node_lane_steps=o.metrics.get('node_slot_steps',sum(steps[:len(nodes)])),
                       probe2_node_iteration_seconds=0.)
            o.metrics['active_lane_steps']=sum(steps[len(nodes):])
            o.metrics['lane_steps']=o.metrics.get('probe_slot_steps',o.metrics['lane_steps'])
            for key in ('lower','empty','z','y','best_y'):setattr(o,key,getattr(o,key)[len(nodes):])
        from auto_LiRPA.implied_box import outward_support_box
        # solve_pool certifies best_y against this call's rows, RHS, box and
        # signed objective, including at step zero and on a deadline. The
        # parent's saved scalar was deliberately discarded by seed above.
        ends=o.lower
        ll,uu,crossed_columns=outward_support_box(lo,hi,owner,var,sign,ends)
        crossed=crossed_columns.any(1)
        certified_caps=(ll,uu)
        # Both endpoints enclose the SAME current cutoff table. Their
        # intersection being empty is a proof, unlike FBBT's generic hint.
        # Never feed crossed columns to a solver; preserve all other caps.
        ll=torch.where(crossed_columns,lo,ll);uu=torch.where(crossed_columns,hi,uu)
        infeasible=torch.zeros_like(T,dtype=torch.uint8).scatter_reduce(0,owner,o.empty.to(torch.uint8),reduce='amax').bool()  # CUDA has no Bool scatter_reduce
        bound=torch.where(infeasible|crossed,T,bound)
        reason=torch.tensor(o.metrics['reasons'][len(nodes):],device=lo.device)
        save=~o.empty
        self.probe_parents.save(sl[save],su[save],o.z[save],o.best_y[save],o.lower[save],tt[save],
                                converged=(reason[save]==2),labels=labels[save])
        if self.input_caps is None:self.input_caps=ParentBank(e.top)
        self.input_caps.save(scope_lo,scope_hi,ll,ll,bound,T,caps=certified_caps)
        # Refresh faces as well as input columns. A stale parent face cannot
        # replace the tighter relaxation on the probed box.
        data,d,empty=self.tables.build(ll,uu,T,caps=(d.lo[:,e.n:],d.hi[:,e.n:]))
        seed,source=crown_seed(e,d,data,with_certificate=True)
        refreshed=torch.maximum(certificate(data,seed),source)
        bound=torch.where(empty,T,torch.minimum(T,torch.maximum(bound,refreshed)))
        witness=torch.maximum(data[4],torch.minimum(data[5],witness))
        attempted=torch.zeros_like(T,dtype=torch.bool).scatter_(0,owner,True)
        # A bit is one bisection's volume reduction; fixed coordinates do
        # not contribute. This is a scheduling measurement, not a proof.
        floor=float(e.cfg.get('bab_feas_tol',1e-4))*(e.root_hi-e.root_lo).flatten().to(lo)
        bits=probe_bits(lo,hi,ll,uu,floor)
        effective=attempted&((bound>=T)|(bits>=1))
        if e.cfg.get('crown_engine_node_probe_cost',False):
            extra.update(probe_serial_steps=o.metrics['probe_serial_steps'],
                         probe_capacity=o.metrics['capacity'])
        return data,d,ll,uu,bound,witness,dict(probe2_directions=N,
            probe_lane_steps=o.metrics['active_lane_steps'],probe_slot_steps=o.metrics['lane_steps'],
            probe2_competitive_probe_steps=o.metrics.get('competitive_probe_steps',0),
            probe2_competitive_node_steps=o.metrics.get('competitive_node_steps',0),
            probe2_converged=int((reason==2).sum()),probe2_reasons=reason.tolist(),
            probe2_lane_iterations=o.metrics['lane_iterations'][len(nodes):],probe2_attempted=attempted.tolist(),
            probe2_effective=effective.tolist(),probe2_bits=bits.tolist(),
            probe2_warm=int(found.sum()),probe2_crossed_prunes=int(crossed.sum()),**extra)


class ProbeSchedule:
    """Path-local rule R state; only search decisions depend on this bank."""
    def __init__(self):
        self.data=None

    def lookup(self, lo, hi, T):
        # Columns: m, s, depth, ineffective, non-stall ineffective (root
        # excluded). Keep this bank with ParentBank so checkpoints preserve it.
        state=torch.zeros(len(lo),5,dtype=torch.long,device=lo.device)
        parent=T.new_full(T.shape,-torch.inf)
        root=torch.ones_like(T,dtype=torch.bool)
        if self.data is not None:
            pl,pu,pt,pb,ps=self.data
            scope=((lo[:,None]>=pl[None])&(hi[:,None]<=pu[None])).all(2)&(T[:,None]<=pt[None])
            rank=torch.where(scope,(pu-pl).abs().sum(1)[None],torch.inf)
            best=rank.amin(1)
            # Latest entry wins ties, including a deadline retry of one box.
            ids=torch.where(scope&(rank==best[:,None]),torch.arange(len(pl),device=lo.device),-1).amax(1).clamp_min(0)
            root=~scope.any(1)
            child=((lo!=pl[ids])|(hi!=pu[ids])).any(1)&~root
            state=torch.where(root[:,None],state,ps[ids]).clone()
            state[:,1]+=child;state[:,2]+=child
            parent=torch.where(root,parent,pb[ids])
        return state,parent,root

    @staticmethod
    def due(T, own, state):
        path,parent,root=state
        stalled=(T-own)>(T-parent)/2
        backoff=path[:,1].to(T)>=torch.exp2(path[:,0].to(T))-1
        return (root|stalled|backoff)&(own<T),stalled,backoff

    def save(self, lo, hi, T, bound, state, attempted, effective, stalled):
        path,_,root=state;path=path.clone()
        effective=attempted&(effective|(bound>=T))
        ineffective=attempted&~effective
        path[:,0]=torch.where(attempted,torch.where(effective,0,path[:,0]+1),path[:,0])
        path[:,1]=torch.where(attempted,0,path[:,1])
        path[:,3]+=ineffective
        path[:,4]+=ineffective&~stalled&~root
        values=(lo,hi,T,bound,path)
        if self.data is not None:values=tuple(torch.cat((a,b)) for a,b in zip(self.data,values))
        # Unlike warm-start duals, path state cannot be evicted: resetting m
        # for an old open branch would defeat the geometric backoff bound.
        # NodePool accounts for the ancestry lookup and storage in admission.
        self.data=tuple(v.detach().clone() for v in values)
        return path


def cutoff_table(data, cutoff, point=None, *, pattern=None):
    """One conservative objective-cutoff face, with all arithmetic enclosed."""
    from nlp_solver.rigorous_dual import Rows, gamma
    rows,c,c0,h,l,u,q=data
    z=(l+u)/2 if point is None else torch.maximum(l,torch.minimum(u,point))
    coeff=c+2*q*z
    rhs=cutoff-c0+(q*z*z).sum(1)
    magnitude=(c.abs()+2*q.abs()*z.abs())*torch.maximum(l.abs(),u.abs())
    error=gamma(rows.n+16)*(cutoff.abs()+c0.abs()+(q*z*z).abs().sum(1)+magnitude.sum(1))
    rhs=torch.nextafter(rhs+error,torch.full_like(rhs,torch.inf))
    ids=torch.arange(rows.n,device=c.device)
    if pattern is None:
        pattern=(torch.cat((rows.row,ids.new_full((rows.n,),rows.m))),torch.cat((rows.col,ids)))
    rr=Rows(*pattern,torch.cat((rows.val,coeff),1),rows.m+1,rows.n)
    return rr,c,c0,torch.cat((h,rhs[:,None]),1),l,u,torch.zeros_like(q)


def pt_table(data, cutoff, *, pattern=None):
    """Production cutoff relaxation; an infinite cutoff is a vacuous row."""
    finite=torch.isfinite(cutoff)
    r,c,c0,h,l,u,q=cutoff_table(data,torch.where(finite,cutoff,0.),pattern=pattern)
    r.val[:,-r.n:]=torch.where(finite[:,None],r.val[:,-r.n:],0.)
    h[:,-1]=torch.where(finite,h[:,-1],1.)
    return r,c,c0,h,l,u,q


def node_score(e, point, lo, hi):
    from input_split.m1_qp import violation_score, nonlinear_residual
    if not hasattr(e,'absorb_incidence'):violation_score(e,point,lo,hi)
    rr,cc=e.absorb_incidence
    score=lo.new_zeros(len(lo),e.n)
    return score.index_add(1,cc,nonlinear_residual(e,point)[:,rr])*(hi>lo)


def node_branch(e, lower, upper, history):
    lo,hi=lower.flatten(1),upper.flatten(1)
    point=history.engine_witness.to(lo)
    score=node_score(e,point,lo,hi)
    valid=getattr(history,'engine_valid',torch.isfinite(point).all(1))
    good=valid & torch.isfinite(point).all(1) & (score.amax(1)>0)
    # Keep sol_node's explicit contraction: every child retains at most 90%
    # of its parent's selected width. This bounds repeated endpoint splitting
    # geometrically and is independent of objective scale or instance identity.
    margin=(hi-lo)/10
    split=torch.maximum(lo+margin,torch.minimum(hi-margin,point[:,:e.n]))
    history.lifted_branch_score=torch.where(good[:,None],score,torch.nan)
    history.lifted_split_point=torch.where(good[:,None],split,torch.nan)
    history.node_pool=True
    return torch.where(good,score.argmax(1),(hi-lo).argmax(1))[:,None]
