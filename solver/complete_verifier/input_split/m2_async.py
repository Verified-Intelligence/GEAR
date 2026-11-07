"""One persistent warm objective/support batch on a shared root LP table."""
import json
import time
import torch
from auto_LiRPA.cap_log import cap_log
from auto_LiRPA.epigraph import EpigraphQP
from auto_LiRPA.restarted_qp import persistent_qp
from .m1_qp import solver_options
from nlp_solver.rigorous_dual import Rows, round_down


def clock(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        return time.perf_counter()
    # CPU work is single-threaded. Scheduler suspension is not device work.
    return time.process_time()


class RefreshCadence:
    """Ski rental: rent steps until they cost the last measured rebuild."""
    def rebuilt(self, seconds):
        self.rebuild_seconds, self.step_seconds = seconds, 0.

    def stepped(self, seconds):
        self.step_seconds += seconds

    def due(self, improved):
        return improved and self.step_seconds >= self.rebuild_seconds


class LaneRetirement:
    """KKT convergence plus no resolvable gain at the certificate refresh."""
    def __init__(self, value, slack):
        self.value,self.slack=value.clone(),slack.clone()
        self.active=torch.ones_like(value,dtype=torch.bool)

    def observe(self, value, slack, converged, *, progress_mask=None):
        progress=value-self.value>slack+self.slack
        if progress_mask is not None:progress &= progress_mask
        self.active &= ~(converged & ~progress)
        # Slot 0 publishes the root bound now. Support endpoints keep their
        # reference until the next publish or rebuild.
        self.value[:1],self.slack[:1]=value[:1],slack[:1]

    def replay(self, value, slack, *, floor=None, progress_mask=None):
        progress=value-(self.value if floor is None else floor)>slack+self.slack
        if progress_mask is not None:progress &= progress_mask
        self.active |= progress
        published=torch.maximum(self.value,value) if floor is None else floor.clone()
        published[:1]=torch.maximum(self.value[:1],value[:1])
        self.value,self.slack=published,slack.clone()


class TargetCycle:
    """Price order within a sweep; every column is served before any repeat."""
    def __init__(self, columns):
        self.columns=columns.clone()
        self.seen=torch.zeros_like(columns,dtype=torch.bool)

    def take(self, score, count):
        order=torch.argsort(score[self.columns],descending=True,stable=True)
        available=order[~self.seen[order]][:count]
        self.seen[available]=True
        return self.columns[available]

    def restart(self):
        self.seen.zero_()


def lane_capacity(e, qp):
    """Budget physical direction states plus a column-keyed warm memo.

    Use the device budget or the host/scope memory available to CPU work.
    Reserve the exact memo tensors, then use the existing conservative QP
    scratch formula per resident lane. No input/aux prefix or lane cap.
    """
    from pathlib import Path
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    memory=getattr(e,'memory',None)
    if memory is not None:available=min(available,memory.available())
    if e.dev.type=='cpu':
        # CPU scopes supply their actual host memory limit, independently of
        # the GPU engine's configured byte cap. Ancestor limits also apply.
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            if line.startswith('0::'):
                group=Path('/sys/fs/cgroup')/line[3:].lstrip('/')
                for parent in (group,*group.parents):
                    limit=parent/'memory.max'
                    if limit.is_file() and limit.read_text().strip()!='max':
                        available=min(available,int(limit.read_text()))
    memo=2*qp.n*(qp.n+2*(qp.m+1))*8+2*qp.n*8+3*qp.n
    scratch=8*(16*(qp.n+qp.m+1)+8*(len(qp.row)+qp.n))
    width=3 if e.cfg.get('m2_omega_portfolio',False) else 1
    return min(width*(1+2*qp.n),max(width,int(max(0,available-memo)//scratch)))


class OmegaPortfolio:
    """Three physical members per logical direction; trust their dual maximum."""
    def __init__(self, data, **options):
        self.options=options
        self.inner=None
        self.refresh(data,warm=options.pop('warm',None))

    def refresh(self, data, *, warm=None):
        self.data=data
        r,c,c0,h,l,u,q=data
        physical=r,c.repeat_interleave(3,0),c0.repeat_interleave(3),h,l,u,q
        seed=None if warm is None else tuple(t.repeat_interleave(3,0) for t in warm)
        if self.inner is None or len(physical[1])!=len(self.inner.full_omega):
            factors=c.new_tensor([1.,.1,10.]).repeat(len(c))
            self.inner=persistent_qp(physical,omega_multipliers=factors,objective_lanes=3,
                                     warm=seed,**self.options)
        else:self.inner.refresh(physical,warm=seed)
        return self.aggregate()

    @property
    def pricing(self):return self.inner.pricing

    def aggregate(self):
        import copy
        src=self.inner.result
        out=copy.copy(src);count=len(self.data[1])
        values=src.lower.reshape(count,3)
        best=values.argmax(1)
        error=torch.minimum(src.error,src.candidate_error).reshape(count,3).argmin(1)
        candidate=src.candidate_error.reshape(count,3).argmin(1)
        base=3*torch.arange(count,device=values.device)
        for name,value in vars(src).items():
            if not (torch.is_tensor(value) and value.ndim and len(value)==3*count):continue
            # The reference warm-chain protocol carries the central member.
            # Mixing a small-omega KKT winner into all three restart seeds
            # would repeatedly erase the central search at table refreshes.
            pick=(best if name in ('lower','best_y','rounding') else
                  candidate if name in ('candidate_z','candidate_error') else
                  error if name in ('error','error_rounding') else torch.zeros_like(best))
            setattr(out,name,value[base+pick].clone())
        out.empty=src.empty.reshape(count,3).any(1)
        out.metrics=dict(src.metrics,portfolio_width=3,physical_lanes=3*count)
        self.result=out
        return out

    def advance(self, count=None, *, deadline=None):
        self.inner.advance(count,deadline=deadline)
        return self.aggregate()

    def compact(self, indices):
        physical=(3*indices[:,None]+torch.arange(3,device=indices.device)).flatten()
        self.inner.compact(physical)
        r,c,c0,h,l,u,q=self.data
        self.data=r,c[indices],c0[indices],h,l,u,q
        self.aggregate()

    def replace(self, slots, c, *, warm=None):
        physical=(3*slots[:,None]+torch.arange(3,device=slots.device)).flatten()
        seed=None if warm is None else tuple(t.repeat_interleave(3,0) for t in warm)
        self.inner.replace(physical,c.repeat_interleave(3,0),warm=seed)
        r,cc,c0,h,l,u,q=self.data
        cc=cc.clone();cc[slots]=c
        c0=c0.clone();c0[slots]=0.
        self.data=r,cc,c0,h,l,u,q
        return self.aggregate()


class RootBatch:
    def __init__(self,e,lo,hi,cutoff,*,columns=None):
        self.e=e
        # Retain square atom columns and their strict faces in the shared LP.
        self.qp=EpigraphQP(e.cg,square_elements=[])
        self.lo,self.hi,self.cutoff=lo.clone(),hi.clone(),cutoff
        self.caps=None
        self.portfolio=getattr(e,'cfg',{}).get('m2_omega_portfolio',False)
        self.rotate=getattr(e,'cfg',{}).get('m2_rotate',False)
        self.column_pool=columns
        self.columns=columns
        self.bound=-float('inf')
        self.steps=0
        self.refreshes=0
        self.cadence=RefreshCadence()
        self.dirty=False
        self.rebuild()
        if not self.closed and not self.undefined:
            self.full=self.solver.result
            self.endpoint_slack=self.full.rounding.new_zeros(len(self.full.lower)-1)
            self.retirement=LaneRetirement(self.replay_floor(),self.resolution(self.full))
            self.active_ids=torch.arange(len(self.full.lower),device=e.dev)

    def remember(self):
        """Memo tensors are indexed by column id, then lower/upper direction."""
        ids=self.columns
        self.memo_z[ids]=self.full.z[1:].reshape(len(ids),2,self.data[0].n)
        self.memo_y[ids]=self.full.y[1:].reshape(len(ids),2,self.data[0].m)
        self.memo_best_y[ids]=self.full.best_y[1:].reshape(len(ids),2,self.data[0].m)
        self.memo_slack[ids]=self.endpoint_slack.reshape(-1,2)
        self.memo_active[ids]=self.retirement.active[1:].reshape(-1,2)
        self.memo_valid[ids]=True

    def select_columns(self, r, c, l, u):
        from auto_LiRPA.restarted_qp import product
        n=r.n
        if not hasattr(self,'cycle'):
            capacity=lane_capacity(self.e,self.qp)
            self.rotation_capacity=min(capacity,getattr(self.e,'m2_lanes',capacity))
            pool=torch.arange(len(self.qp.keep),device=self.e.dev) if self.column_pool is None else self.column_pool
            self.cycle=TargetCycle(pool)
            self.memo_z=c.new_zeros(n,2,n)
            self.memo_y=c.new_zeros(n,2,r.m+1)
            self.memo_best_y=torch.zeros_like(self.memo_y)
            self.memo_slack=c.new_zeros(n,2)
            self.memo_valid=torch.zeros(n,dtype=torch.bool,device=c.device)
            self.memo_active=torch.ones(n,2,dtype=torch.bool,device=c.device)
        else:self.remember()
        y=c.new_zeros(1,r.m) if not hasattr(self,'full') else self.full.best_y[:1,:r.m]
        self.prices=c[0].abs()+product(r,r.val.abs(),y,transpose=True)[0]
        self.target_scores=torch.nan_to_num(self.prices*(u-l)[0],nan=0.,posinf=float('inf'))
        # Reuse the admitted residency; a retained memo is not allocated again.
        # Honor the interleave's smaller admission after an allocation failure.
        capacity=min(self.rotation_capacity,getattr(self.e,'m2_lanes',self.rotation_capacity))
        count=max(0,(capacity//(3 if self.portfolio else 1)-1)//2)
        ids=self.cycle.take(self.target_scores,count)
        if not len(ids) and count:
            self.cycle.restart();ids=self.cycle.take(self.target_scores,count)
        self.columns=ids
        self.endpoint_slack=self.memo_slack[ids].flatten().clone()
        objective=(l+u)/2 if not hasattr(self,'full') else self.full.z[:1]
        z=objective.expand(1+2*len(ids),-1).clone()
        yy=c.new_zeros(len(z),r.m+1)
        by=yy.clone()
        if hasattr(self,'full'):yy[:1]=self.full.y[:1];by[:1]=self.full.best_y[:1]
        valid=self.memo_valid[ids].repeat_interleave(2)
        z[1:]=torch.where(valid[:,None],self.memo_z[ids].flatten(0,1),z[1:])
        yy[1:]=self.memo_y[ids].flatten(0,1)
        by[1:]=self.memo_best_y[ids].flatten(0,1)
        self.rotation_warm=(z,yy)
        self.rotation_best=by

    def replace_retired(self):
        """Refill retired direction pairs on the same table before compaction."""
        pairs=(~self.retirement.active[1:]).reshape(-1,2).all(1).nonzero().flatten()
        columns=self.cycle.take(self.target_scores,len(pairs))
        if not len(columns):return
        pairs=pairs[:len(columns)]
        # Publish the outgoing endpoint proofs before their identities change.
        self.endpoints(self.full)
        self.remember()
        slots=(1+2*pairs[:,None]+torch.arange(2,device=pairs.device)).flatten()
        cost=self.data[1].new_zeros(len(slots),self.data[0].n)
        cost[torch.arange(len(slots),device=slots.device),columns.repeat_interleave(2)]=cost.new_tensor([1.,-1.]).repeat(len(columns))
        valid=self.memo_valid[columns].repeat_interleave(2)
        z=torch.where(valid[:,None],self.memo_z[columns].flatten(0,1),self.full.z[:1])
        y=self.memo_y[columns].flatten(0,1)
        fresh=self.solver.replace(slots,cost,warm=(z,y))
        self.columns[pairs]=columns
        self.target_list=self.columns.tolist()
        self.data=self.solver.data
        for name,value in vars(fresh).items():
            if torch.is_tensor(value) and value.ndim and len(value)==len(self.full.lower):
                getattr(self.full,name)[slots]=value[slots]
        self.endpoint_slack[slots-1]=self.memo_slack[columns].flatten()
        floor=self.replay_floor()
        self.retirement.value[slots]=floor[slots]
        self.retirement.slack[slots]=self.resolution(self.full)[slots]
        self.retirement.active[slots]=True

    def resolution(self, out):
        """A boxed endpoint retains the rounding slack of its source dual.

        Replaying y=0 on that box does not create a more precise source
        certificate. Only a resolved, genuinely stronger endpoint replaces it.
        """
        return out.rounding+torch.cat((out.rounding.new_zeros(1),self.endpoint_slack))

    def replay_floor(self):
        ids=self.columns;inp=ids<self.e.n
        original=self.qp.keep[ids]
        lo,hi=self.d.lo[0,original].clone(),self.d.hi[0,original].clone()
        lo[inp],hi[inp]=self.lo[0,ids[inp]],self.hi[0,ids[inp]]
        if self.caps is not None:
            aux=original[~inp]-self.e.n
            lo[~inp]=torch.maximum(lo[~inp],self.caps[0][0,aux])
            hi[~inp]=torch.minimum(hi[~inp],self.caps[1][0,aux])
        return torch.cat((self.full.lower[:1],torch.stack((lo,-hi),1).flatten()))

    def influence(self, out, floor):
        # Triangle bound on fixed-table Lagrangian box sensitivity. Keeping
        # unsigned prices avoids cancellation of an interior reduced cost.
        from nlp_solver.rigorous_dual import gamma
        r,c=self.data[:2]
        price=c[0].abs()+self.solver.pricing[1].left(out.best_y[:1])[0]
        n=2*(len(r.row)+r.n)+4
        tiny=torch.nextafter(price.new_tensor(0.),price.new_tensor(float('inf')))
        up=lambda x:torch.nextafter(x,x.new_tensor(float('inf')))
        price=up((price+n*tiny)/(1-gamma(n)))
        price=torch.nan_to_num(price,nan=float('inf'),posinf=float('inf'))
        delta=up((out.lower[1:]-floor[1:]).clamp_min(0))
        effect=up(price[self.columns].repeat_interleave(2)*delta)
        count=len(effect)+2
        total=up((effect.sum()+count*tiny)/(1-gamma(count)))
        scale=max(1.,abs(self.cutoff));tau=max(self.e.abs_tol,self.e.rel_tol*scale)
        gap=torch.nextafter(out.lower.new_tensor(self.cutoff-self.bound),out.lower.new_tensor(-float('inf')))
        self.worthwhile=bool((total>tau)|((gap>0)&(total>=gap)))
        # One batch shares one objective-accuracy budget; small improvements
        # are kept together when their combined upper impact is resolvable.
        mask=torch.full_like(effect,self.worthwhile,dtype=torch.bool)
        return torch.cat((mask.new_ones(1),mask))

    def publish(self, lo, hi, caps):
        self.lo,self.hi,self.caps=lo,hi,caps
        self.endpoint_slack=self.pending_slack
        self.dirty=True
        if hasattr(self,'retirement'):self.retirement.value[1:],self.retirement.slack[1:]=self.replay_floor()[1:],self.endpoint_slack

    def rebuild(self):
        tick=clock(self.e.dev)
        self.rebuilding=True  # stays set if this rebuild fails; the next resume retries it
        # A later meaningful refresh also consumes previously deferred proofs.
        if hasattr(self,'pending_slack'):
            pl,pu,pc=self.pending
            caps=pc if self.caps is None else (torch.maximum(self.caps[0],pc[0]),torch.minimum(self.caps[1],pc[1]))
            self.publish(torch.maximum(self.lo,pl),torch.minimum(self.hi,pu),caps)
        e=self.e
        d,_,empty,_=e._bound_fbbt(self.lo,self.hi,slice(0,1),self.lo.new_tensor([self.cutoff]),caps_in=self.caps)
        self.closed=bool(empty.any()) and not bool(d.undefined.any())
        self.close_reason='fbbt_empty'
        self.undefined=bool(d.undefined.any())
        if self.closed or self.undefined:
            self.dirty=self.rebuilding=False
            return
        data=self.qp.build(d)
        ring=self.qp.epigraph
        if ring.terms:
            if hasattr(self,'raw_data'):
                mask=data[0].row>=ring.first
                data[0].val[:,mask]=self.raw_data[0].val[:,mask]
                data[3][:,ring.first:]=self.raw_data[3][:,ring.first:]
                data[0].epigraph_used=self.raw_data[0].epigraph_used
            point=(data[4]+data[5])/2 if not hasattr(self,'full') else self.full.z[:1]
            ring.add(data,point)
        self.raw_data=data
        self.d=d
        r,c,c0,h,l,u,q=self.qp.epigraph.clean(data)
        if self.rotate:self.select_columns(r,c,l,u)
        if self.columns is None:
            if e.n>128:cap_log('m2_targets','inputs',128,e.n,'first 128 inputs only')
            if len(self.qp.keep)-e.n>32:cap_log('m2_targets','aux',32,len(self.qp.keep)-e.n,'first 32 aux only')
            self.columns=torch.cat((torch.arange(min(128,e.n),device=e.dev),
                                    torch.arange(e.n,min(e.n+32,len(self.qp.keep)),device=e.dev)))
            if self.portfolio:self.columns=self.columns[:max(0,(lane_capacity(e,self.qp)//3-1)//2)]
        ids=self.columns
        self.target_list=ids.tolist()
        lanes=1+2*len(ids)
        cost=c.new_zeros(lanes,r.n);cost[0]=c[0]
        cost[torch.arange(1,lanes,device=e.dev),ids.repeat_interleave(2)]=c.new_tensor([1.,-1.]).repeat(len(ids))
        offset=c0.new_zeros(lanes);offset[0]=c0[0]
        self.cut_indices=(torch.cat((r.row,r.row.new_full((r.n,),r.m))),
                          torch.cat((r.col,torch.arange(r.n,device=e.dev)))) if not hasattr(self,'cut_indices') else self.cut_indices
        rr=Rows(*self.cut_indices,torch.cat((r.val,c),1),r.m+1,r.n)
        rhs=torch.nextafter(c0.new_tensor(self.cutoff)-c0,c0.new_tensor(float('inf')))
        self.data=rr,cost,offset,torch.cat((h,rhs[:,None]),1),l,u,q
        if getattr(self,'solver',None) is not None:
            warm=self.rotation_warm if self.rotate else ((self.full.z,self.full.y) if hasattr(self,'full') else None)
            self.solver.refresh(self.data,warm=warm)
        else:self.solver=(OmegaPortfolio if self.portfolio else persistent_qp)(self.data,capture=e.dev.type=='cuda',objective_cutoff=True,**solver_options(self.e),
                                      warm=self.rotation_warm if self.rotate else ((self.full.z,self.full.y) if hasattr(self,'full') else None))
        self.refreshes+=1
        self.pending=(self.lo.clone(),self.hi.clone(),self.caps)
        self.bound=max(self.bound,float(self.solver.result.lower[0]))
        self.closed=bool(self.solver.result.empty.any()) or self.bound>=self.cutoff
        self.close_reason='closed'
        if self.rotate and hasattr(self,'retirement'):
            from nlp_solver.rigorous_dual import dual_lower_bound
            replay=self.solver.result
            rr,cc,off,hh,ll,uu,qq=self.solver.data
            value,slack,_=dual_lower_bound(rr,cc,off,hh.expand(len(cc),-1),ll.expand(len(cc),-1),uu.expand(len(cc),-1),self.rotation_best)
            use=value>replay.lower
            replay.lower=torch.maximum(replay.lower,value)
            replay.best_y=torch.where(use[:,None],self.rotation_best,replay.best_y)
            replay.rounding=torch.where(use,slack,replay.rounding)
            self.full=replay
            self.bound=max(self.bound,float(replay.lower[0]))
            self.closed |= self.bound>=self.cutoff
            floor=self.replay_floor()
            self.retirement=LaneRetirement(floor,self.resolution(replay))
            self.retirement.active[1:]=self.memo_active[self.columns].flatten()
            self.retirement.replay(replay.lower,self.resolution(replay),floor=floor,progress_mask=self.influence(replay,floor))
            scale=max(1.,abs(self.cutoff));rho=max(e.abs_tol,e.rel_tol*scale)/scale
            error=torch.minimum(replay.error,replay.candidate_error)
            tolerance=rho*(replay.lower-replay.baseline).clamp_min(0)+self.resolution(replay)+replay.error_rounding
            numeric=self.resolution(replay)+replay.error_rounding
            unresolved=(replay.lower[0]+error[0]+self.resolution(replay)[0]>=self.cutoff)&(error[0]>numeric[0])
            self.retirement.active[0]=(error[0]>tolerance[0])|unresolved|self.retirement.active[1:].any()
            if ring.terms and self.raw_data[0].epigraph_used<ring.capacity:self.retirement.active[0]=True
            self.active_ids=(torch.arange(len(cc),device=e.dev) if self.retirement.active.any()
                             else torch.empty(0,dtype=torch.long,device=e.dev))
        elif hasattr(self,'retirement'):
            replay=self.solver.result
            saved=self.full
            from nlp_solver.rigorous_dual import dual_lower_bound
            rr,cc,off,hh,ll,uu,qq=self.solver.data
            value,slack,_=dual_lower_bound(rr,cc,off,hh.expand(len(cc),-1),ll.expand(len(cc),-1),uu.expand(len(cc),-1),saved.best_y)
            use=value>replay.lower
            replay.lower=torch.maximum(replay.lower,value)
            replay.best_y=torch.where(use[:,None],saved.best_y,replay.best_y)
            replay.rounding=torch.where(use,slack,replay.rounding)
            self.bound=max(self.bound,float(replay.lower[0]))
            self.closed |= self.bound>=self.cutoff
            floor=self.replay_floor()
            worth=self.influence(replay,floor)
            self.retirement.replay(replay.lower,self.resolution(replay),floor=floor,progress_mask=worth)
            scale=max(1.,abs(self.cutoff));rho=max(e.abs_tol,e.rel_tol*scale)/scale
            tolerance=rho*(replay.lower-replay.baseline).clamp_min(0)+self.resolution(replay)+replay.error_rounding
            error=torch.minimum(replay.error,replay.candidate_error)
            # Fresh rows can help the root bound even when direct box prices
            # are zero: validate the objective's KKT pair on every refresh.
            numeric=self.resolution(replay)+replay.error_rounding
            unresolved=(replay.lower[0]+error[0]+self.resolution(replay)[0]>=self.cutoff)&(error[0]>numeric[0])
            self.retirement.active[0] |= (error[0]>tolerance[0])|unresolved
            self.full=replay
            self.retirement.active[0] |= self.retirement.active[1:].any()
            if ring.terms and self.raw_data[0].epigraph_used<ring.capacity:self.retirement.active[0]=True
            self.active_ids=self.retirement.active.nonzero().flatten()
            if len(self.active_ids) and not self.closed:self.solver.compact(self.active_ids)
        self.dirty=self.rebuilding=False
        self.rebuild_seconds=clock(e.dev)-tick
        self.since_rebuild=0.
        self.cadence.rebuilt(self.rebuild_seconds)

    def restore(self):
        """Restore M2 residency from its frozen table, without calling FBBT."""
        self.solver=(OmegaPortfolio if self.portfolio else persistent_qp)(self.data,capture=self.e.dev.type=='cuda',objective_cutoff=True,**solver_options(self.e),
                                  warm=(self.full.z,self.full.y))
        if len(self.active_ids) and not self.rotate:self.solver.compact(self.active_ids)

    def change_cutoff(self, cutoff):
        """M2-R changes one RHS. Direction identities and physical z/y survive."""
        self.cutoff=cutoff
        offset=self.data[2][0]
        self.data[3][:,-1]=torch.nextafter(offset.new_tensor(cutoff)-offset,offset.new_tensor(float('inf')))
        saved=self.full
        if self.solver is None:
            self.solver=(OmegaPortfolio if self.portfolio else persistent_qp)(self.data,capture=self.e.dev.type=='cuda',objective_cutoff=True,**solver_options(self.e),
                                      warm=(saved.z,saved.y))
            replay=self.solver.result
        else:replay=self.solver.refresh(self.data,warm=(saved.z,saved.y))
        from nlp_solver.rigorous_dual import dual_lower_bound
        r,c,c0,h,l,u,q=self.solver.data
        value,slack,_=dual_lower_bound(r,c,c0,h.expand(len(c),-1),l.expand(len(c),-1),u.expand(len(c),-1),saved.best_y)
        use=value>replay.lower
        replay.lower=torch.maximum(replay.lower,value)
        replay.best_y=torch.where(use[:,None],saved.best_y,replay.best_y)
        replay.rounding=torch.where(use,slack,replay.rounding)
        self.full=replay
        self.bound=max(self.bound,float(replay.lower[0]))
        self.retirement.replay(replay.lower,self.resolution(replay),floor=self.replay_floor(),progress_mask=self.influence(replay,self.replay_floor()))
        self.retirement.active.fill_(True)
        self.active_ids=self.retirement.active.nonzero().flatten()
        if len(self.active_ids) and not self.rotate:self.solver.compact(self.active_ids)
        self.closed=bool(replay.empty.any()) or self.bound>=cutoff

    def checkpoint(self,deadline):
        tick=clock(self.e.dev)
        part=self.solver.advance(deadline=deadline)
        elapsed=clock(self.e.dev)-tick
        self.since_rebuild+=elapsed
        self.cadence.stepped(part.metrics.get('step_seconds',part.metrics['iteration_seconds']))
        self.steps+=part.metrics['iterations']
        out=self.full
        ids=self.active_ids
        use=part.lower>out.lower[ids]
        out.lower[ids]=torch.maximum(out.lower[ids],part.lower)
        out.best_y[ids]=torch.where(use[:,None],part.best_y,out.best_y[ids])
        out.rounding[ids]=torch.where(use,part.rounding,out.rounding[ids])
        out.empty[ids] |= part.empty
        for name in ('z','y','error','baseline','error_rounding','restarts','candidate_z','candidate_error'):
            getattr(out,name)[ids]=getattr(part,name)
        out.metrics=part.metrics
        self.bound=max(self.bound,float(out.lower[0]))
        self.closed=bool(out.empty.any()) or self.bound>=self.cutoff
        scale=max(1.,abs(self.cutoff))
        rho=max(self.e.abs_tol,self.e.rel_tol*scale)/scale
        tolerance=rho*(out.lower-out.baseline).clamp_min(0)+self.resolution(out)+out.error_rounding
        recovered=torch.isfinite(out.candidate_error)&torch.isfinite(tolerance)&(out.candidate_error<=tolerance)
        converged=(torch.isfinite(out.error)&torch.isfinite(tolerance)&(out.error<=tolerance))|recovered
        error=torch.minimum(out.error,out.candidate_error)
        numeric=self.resolution(out)+out.error_rounding
        unresolved=(out.lower[0]+error[0]+self.resolution(out)[0]>=self.cutoff)&(error[0]>numeric[0])
        converged[0] &= ~unresolved
        worth=self.influence(out,self.retirement.value)
        self.retirement.observe(out.lower,self.resolution(out),converged,progress_mask=worth)
        # Keeping one objective monitor costs width, never extra serial depth.
        self.retirement.active[0] |= self.retirement.active[1:].any()
        if self.qp.epigraph.terms and self.raw_data[0].epigraph_used<self.qp.epigraph.capacity:
            self.retirement.active[0]=True
        retired=~self.retirement.active & recovered
        picked=retired[self.active_ids]
        slots=self.active_ids[picked]
        out.z[slots]=out.candidate_z[slots]
        out.y[slots]=part.best_y[picked]
        if self.rotate:
            self.replace_retired()
            self.retirement.active[0] |= self.retirement.active[1:].any()
            self.active_ids=(torch.arange(len(out.lower),device=self.e.dev) if self.retirement.active.any()
                             else self.active_ids[:0])
            return out
        ids=self.retirement.active.nonzero().flatten()
        if len(ids) and not torch.equal(ids,self.active_ids):
            # Surviving scaled iterates, omega and restart state are gathered,
            # not reset; retired lanes disappear from subsequent SpMMs.
            if not bool(torch.isin(ids,self.active_ids).all()):
                raise RuntimeError('survivors must be a subset of active lanes')
            local=torch.searchsorted(self.active_ids,ids)
            self.solver.compact(local)
        self.active_ids=ids
        return out

    def endpoints(self,out):
        vals=out.lower[1:]
        ids=self.columns;inp=ids<self.e.n
        base=self.replay_floor()[1:]
        resolved=vals-base>out.rounding[1:]+self.endpoint_slack
        vals=torch.where(resolved,vals,base)
        copied=~out.best_y[1:].ne(0).any(1)
        slack=torch.where(copied,torch.maximum(self.endpoint_slack,out.rounding[1:]),out.rounding[1:])
        self.pending_slack=torch.where(resolved,slack,self.endpoint_slack)
        lows=torch.nextafter(vals[::2],vals.new_tensor(-float('inf')))
        highs=torch.nextafter(-vals[1::2],vals.new_tensor(float('inf')))
        ids=self.columns;inp=ids<self.e.n
        lo,hi=self.pending[0].clone(),self.pending[1].clone()
        lo[0,ids[inp]]=torch.maximum(lo[0,ids[inp]],lows[inp])
        hi[0,ids[inp]]=torch.minimum(hi[0,ids[inp]],highs[inp])
        cl,cu=self.d.lo[:,self.e.n:].clone(),self.d.hi[:,self.e.n:].clone()
        if self.pending[2] is not None:cl,cu=torch.maximum(cl,self.pending[2][0]),torch.minimum(cu,self.pending[2][1])
        aux=self.qp.keep[ids[~inp]]-self.e.n
        cl[0,aux]=torch.maximum(cl[0,aux],lows[~inp]);cu[0,aux]=torch.minimum(cu[0,aux],highs[~inp])
        if getattr(self.e,'cfg',{}).get('m2_rcbt',False):
            from auto_LiRPA.implied_box import rc_tighten
            r,c,c0,h,l,u,q=self.solver.data
            rl,ru,info=rc_tighten(r,c[:1],c0[:1],h,l,u,out.best_y[:1],self.cutoff)
            keep=self.qp.keep
            oldl=torch.cat((lo,cl),1)[:,keep]
            oldu=torch.cat((hi,cu),1)[:,keep]
            nl=torch.maximum(oldl,rl[:,:len(keep)])
            nu=torch.minimum(oldu,ru[:,:len(keep)])
            # RCBT's empty hint is never a new closure certificate. If the
            # intersection crosses, retain the last nonempty pending box.
            valid=~((nl>nu).any(1)|info['empty_hint'])
            nl=torch.where(valid[:,None],nl,oldl)
            nu=torch.where(valid[:,None],nu,oldu)
            inp=keep<self.e.n;aux=keep[~inp]-self.e.n
            lo[:,keep[inp]]=nl[:,inp];hi[:,keep[inp]]=nu[:,inp]
            cl[:,aux]=nl[:,~inp];cu[:,aux]=nu[:,~inp]
            # Apply the existing shared objective-accuracy budget to RCBT
            # gains too, including columns with no resident support lane.
            gain=(nl-oldl).clamp_min(0)+(oldu-nu).clamp_min(0)
            price=c[0].abs()+self.solver.pricing[1].left(out.best_y[:1])[0]
            from nlp_solver.rigorous_dual import gamma
            count=2*(len(r.row)+r.n)+4
            up=lambda x:torch.nextafter(x,x.new_tensor(float('inf')))
            price=up((price+count*2.**-1074)/(1-gamma(count)))
            effect=up(price[:len(keep)]*gain[0])
            total=up((effect.sum()+(len(effect)+2)*2.**-1074)/(1-gamma(len(effect)+2)))
            tau=max(self.e.abs_tol,self.e.rel_tol*max(1.,abs(self.cutoff)))
            gap=self.cutoff-self.bound
            self.worthwhile=getattr(self,'worthwhile',False) or bool((total>tau)|((gap>0)&(total>=gap)))
        changed=bool((lo>self.lo).any() or (hi<self.hi).any())
        reflo,refhi=self.d.lo[:,self.e.n:],self.d.hi[:,self.e.n:]
        if self.caps is not None:
            reflo,refhi=torch.maximum(reflo,self.caps[0]),torch.minimum(refhi,self.caps[1])
        changed |= bool(((cl>reflo)|(cu<refhi)).any())
        self.closed |= bool((lo>hi).any() or (cl>cu).any())
        self.pending=lo,hi,(cl,cu)
        return lo,hi,(cl,cu),changed


def release(e):
    """Evict only M2 graphs/search residency, retaining its box and duals."""
    from auto_LiRPA.cuda_memory import release_cuda
    resident=bool(getattr(e,'support_cache',None))
    if resident:e.support_cache.clear()
    batch=(getattr(e,'m2_state',None) or {}).get('batch')
    if batch is not None and getattr(batch,'solver',None) is not None:
        batch.solver=None
        resident=True
    if resident:release_cuda(e.dev)
    return resident


@torch.no_grad()
def root_obbt(e,lower,upper,cutoff,deadline,*,round_limit=None,refresh_tol=None,columns=None):
    lo=lower.reshape(1,-1).to(e.dev,torch.float64)
    hi=upper.reshape(1,-1).to(e.dev,torch.float64)
    if not torch.isfinite(lo.new_tensor(cutoff)):return lower,upper
    state=getattr(e,'m2_state',None)
    if state is not None:
        if cutoff>state['cutoff'] or bool((lo<state['lo']).any() or (hi>state['hi']).any()):
            raise ValueError('M2 resume requires a sub-box and nonincreasing cutoff')
        if state['stop'] in ('closed','fbbt_empty','undefined','global_deadline'):return lower,upper
    if getattr(e,'local_yield',False):
        e.local_yield=False
        e.stats['m2_stop']='yield'
        if state is None:e.m2_state=dict(stop='yield',lo=lo.clone(),hi=hi.clone(),cutoff=cutoff,anchor_cutoff=cutoff,scope=(lo.clone(),hi.clone()),rounds=0)
        if not hasattr(e,'m2_trace'):e.m2_trace=[]
        return lower,upper
    if state is None:
        state=dict(scope=(lo.clone(),hi.clone()),anchor_cutoff=cutoff,rounds=0,stop='yield',cutoff=cutoff,lo=lo.clone(),hi=hi.clone())
        e.m2_state=state;e.m2_trace=[];e.m2_closed=False
    batch=state.get('batch')
    if batch is None:
        batch=RootBatch(e,lo,hi,cutoff,columns=columns);state['batch']=batch
    else:
        tau=max(e.abs_tol,e.rel_tol*max(1.,abs(cutoff))) if refresh_tol is None else refresh_tol
        if getattr(batch,'rebuilding',False) or bool((lo>batch.lo).any() or (hi<batch.hi).any()):
            batch.lo,batch.hi=torch.maximum(batch.lo,lo),torch.minimum(batch.hi,hi)
            batch.rebuild()
        if not (batch.closed or batch.undefined) and state['anchor_cutoff']-cutoff>tau:
            batch.change_cutoff(cutoff)
            state['anchor_cutoff']=cutoff
            e.stats['m2_refreshes']=e.stats.get('m2_refreshes',0)+1
        elif not (batch.closed or batch.undefined) and batch.solver is None and len(batch.active_ids):batch.restore()
    if batch.closed or batch.undefined:
        reason='undefined' if batch.undefined else batch.close_reason
    elif not len(batch.active_ids):
        reason='retired'
    else:
        reason='yield'
        remaining=round_limit
        while remaining is None or remaining>0:
            if remaining is not None:remaining-=1
            if deadline is not None and time.time()>=deadline:reason='global_deadline';break
            oldlo,oldhi=batch.lo.clone(),batch.hi.clone()
            out=batch.checkpoint(deadline)
            if not batch.closed:
                queue=getattr(e,'lane_points',None)
                if queue is not None:
                    queue.offer(batch.qp.expand(out.z[~out.empty]),out.lower.new_tensor(batch.bound))
                    ref=getattr(e,'lane_local',None)
                    lane=None if ref is None else ref()
                    if lane is not None:lane.proposal_ready()
            nl,nu,caps,changed=batch.endpoints(out)
            ring=batch.qp.epigraph
            cuts=bool(ring.terms) and batch.raw_data[0].epigraph_used<ring.capacity
            pub=changed and batch.worthwhile
            if pub:batch.publish(nl,nu,caps)
            due=batch.cadence.due(batch.dirty or cuts)
            ok=(oldhi>oldlo)&(nu>nl)
            logw=float(torch.log((oldhi-oldlo)/(nu-nl))[ok].sum()) if pub else 0.
            rec=dict(active_lanes=len(batch.active_ids),round=len(e.m2_trace)+1,lower=batch.bound,cutoff=cutoff,steps=batch.steps,
                     lanes=len(out.lower),targets=batch.target_list,qp_iterations=0,
                     support_iterations=out.metrics['iterations'],support_lane_steps=out.metrics['lane_steps'],
                     active_support_lane_steps=out.metrics['active_lane_steps'],
                     tightened=int(((nl>oldlo)|(nu<oldhi)).sum()),log_width=logw,
                     rebuilds=batch.refreshes,rebuild_seconds=batch.rebuild_seconds,
                     step_seconds=batch.cadence.step_seconds,writeback=pub,closed=batch.closed)
            e.m2_trace.append(rec)
            previous=e.m2_trace[-2] if len(e.m2_trace)>1 else None
            if previous is None or due or pub or batch.closed or rec['active_lanes']!=previous.get('active_lanes') or rec['lower']>previous.get('lower',-float('inf')):
                print('[m2-round] '+json.dumps(rec),flush=True)
            if batch.closed:reason='closed';break
            retiring=not len(batch.active_ids)
            if retiring and not (batch.dirty or cuts):
                reason='retired';break
            if retiring:
                if changed and not pub:batch.publish(nl,nu,caps)
                due=True
            if due:
                batch.rebuild()
                if batch.closed or batch.undefined:
                    reason='undefined' if batch.undefined else batch.close_reason
                    break
                if not len(batch.active_ids):
                    reason='retired';break
    if not e.m2_trace:
        e.m2_trace.append(dict(round=1,lower=batch.bound if batch.bound>-float('inf') else None,steps=batch.steps,active_lanes=0,closed=reason in ('closed','fbbt_empty')))
    if reason in ('closed','fbbt_empty'):
        e.m2_closed=True;e.m2_certificate=(*state['scope'],cutoff)
    e.stats.update(m2_stop=reason,m2_rounds=len(e.m2_trace),m2_steps=batch.steps)
    low=round_down(batch.lo.reshape_as(lower),lower.dtype).to(lower.device)
    high=-round_down(-batch.hi.reshape_as(upper),upper.dtype).to(upper.device)
    state.update(cutoff=cutoff,lo=low.to(lo),hi=high.to(hi),rounds=len(e.m2_trace),stop=reason)
    return low,high
