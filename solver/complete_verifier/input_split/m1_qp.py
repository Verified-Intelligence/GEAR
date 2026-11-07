"""Opt-in first-order graph bounds. No lockstep/simplex search is constructed."""
import atexit
import json
import time
import torch
from auto_LiRPA.obbt_unary import OutwardCompiledGraph as AbsorbCompiledGraph


class M1CompiledGraph(AbsorbCompiledGraph):
    """Strict unary faces in the compiled (faces, lanes, elements) layout."""
    def bound(self, x_L, x_U, thr=None, h=None, caps=None):
        if not getattr(self, '_cache_bounds', False):
            return super().bound(x_L, x_U, thr, h, caps)
        from auto_LiRPA.compiled_crown import Data
        from auto_LiRPA.restarted_qp import _capture_reset
        # Pad repeated valid boxes, then slice every result back to active lanes.
        # No padded box can add a certificate to an active lane.
        B = len(x_L); capacity = 1 << (B-1).bit_length()
        def pad(t):
            return torch.cat((t, t[:1].expand(capacity-B, *t.shape[1:]))) if capacity>B else t
        source = [pad(x_L), pad(x_U), pad(thr), pad(h.expand(B,-1) if h is not None else x_L.new_zeros(B,0))]
        if caps is not None: source += [pad(t) for t in caps]
        key = (capacity, caps is not None)
        cold = key not in self._bound_cache
        if cold:
            tick = time.perf_counter()
            inputs = [t.clone() for t in source]
            result = []
            invalid = torch.zeros(capacity,dtype=torch.bool,device=x_L.device)
            def compute():
                invalid.zero_(); self._capture_invalid = invalid
                try:
                    d, out = super(M1CompiledGraph,self).bound(*inputs[:4],
                        caps=tuple(inputs[4:]) if caps is not None else None)
                finally: del self._capture_invalid
                result[:] = [d,out]
            graph = _capture_reset(compute,x_L.device)
            self._bound_cache[key] = inputs,graph,result,invalid
            self._bound_capture_seconds += time.perf_counter()-tick
        inputs,graph,result,invalid = self._bound_cache[key]
        for dst,src in zip(inputs,source):dst.copy_(src)
        graph.replay()
        if bool(invalid[:B].any()):
            raise ValueError('strict OBBT unary domain is unsupported or undefined')
        def active(value):
            if torch.is_tensor(value):return value[:B].clone()
            if isinstance(value,tuple):return tuple(active(t) for t in value)
            return value
        d,out=result
        return Data(**{k:active(v) for k,v in vars(d).items()}),out[:B].clone()



def solver_options(e):
    return dict(iteration_scheme='halpern' if getattr(e,'cfg',{}).get('crown_engine_halpern',False) else 'average',
                implied_box=getattr(e,'cfg',{}).get('crown_engine_implied_box',False))


def configure(cfg):
    # Private compiler options only. Never mutate GEAR's r4 configuration.
    cfg.update(crown_engine_sparse='true', crown_engine_squares='true',
               crown_engine_mixed_squares='true', crown_engine_ibp='true')


def initialize(e, started, report):
    from auto_LiRPA.epigraph import EpigraphQP
    e.qp = EpigraphQP(e.cg)
    e.cg._cache_bounds = e.dev.type == 'cuda' and e.cg.capture_safe()
    e.cg._bound_cache = {}; e.cg._bound_capture_seconds = 0.
    e.c, e.c0 = e.cg.CM0.clone(), float(e.cg.Cc[0])
    e.top = max(1, int(e.cfg.get('crown_engine_lanes', 256)))
    # The domain list retains scalars; the bounded node pool owns physical warm states.
    e.width, e.node_caps, e.propagate = 1, False, False
    e.root_caps, e.root_seed, e.lb = None, None, None
    e.branching, e.stay = 'mcc', False
    e.obbt_targets = 32 if e.cfg.get('crown_engine_m2', False) else 0
    e.calls = 0; e._share = True
    e.qp_cache = {}; e.support_cache = {}; e.call_metrics = []
    e.m2_lanes = min(256,2*(min(128,e.n)+min(32,len(e.qp.keep)-e.n)))
    if e.cfg.get('m2_rotate',False) or e.cfg.get('m2_omega_portfolio',False):
        from .m2_async import lane_capacity
        e.m2_lanes = lane_capacity(e,e.qp)
    e.tan_row = e.m-1 if e.tangent is not None else None
    e.sq = torch.zeros(e.Et, dtype=torch.bool, device=e.dev)
    e.recip = torch.zeros_like(e.sq)
    for nd in e.cg.nodes:
        name = type(e.pg.net[nd.name]).__name__
        if name == 'BoundSqr': e.sq[nd.s:nd.s+nd.E] = True
        if name == 'BoundReciprocal': e.recip[nd.s:nd.s+nd.E] = True
    e.stats = dict(lanes=0, bound=0, farkas=0, seconds=0., cpu_lp=0, iterations=0,
                   seconds_fbbt=0., fbbt_tightened=0, fbbt_empty=0)
    e.scratch_bytes = 8*(16*(e.qp.n+e.qp.m)+8*len(e.qp.row))
    e.info = dict(n=e.n, ncol=e.ncol, qp_columns=e.qp.n, qp_rows=e.qp.m,
                  qp_nnz=len(e.qp.row), squares=len(e.qp.chosen),
                  operand_columns=len(e.qp.virtual), top=e.top, seconds_setup=time.time()-started)
    print('[crown-m1] ready '+str(e.info), flush=True)
    if report: atexit.register(e.report)


def record_work(e, result, phase):
    for key in ('iterations','setup_seconds','capture_seconds','iteration_seconds'):
        name=phase+'_'+key
        e.stats[name]=e.stats.get(name,0)+result[key]
    for key in ('lane_steps','active_lane_steps'):
        if key in result:
            name=phase+'_'+key;e.stats[name]=e.stats.get(name,0)+result[key]
    key=phase+'_solves';e.stats[key]=e.stats.get(key,0)+1
    e.stats['compiled_capture_seconds'] = e.cg._bound_capture_seconds


def release_optional(e, *, phase=None):
    """Keep certificates and the target cycle; evict optional search residency."""
    if phase == 'm2':
        from .m2_async import release
        return release(e)
    from auto_LiRPA.cuda_memory import release_cuda
    resident = False
    for owner, name in ((e,'support_cache'),(e,'qp_cache'),(getattr(e,'cg',None),'_bound_cache')):
        cache = getattr(owner,name,None)
        if cache:
            resident = True
            cache.clear()
    state = getattr(e,'m2_state',None)
    if state is not None:
        resident |= state.get('warm') is not None or state.get('weights') is not None
        state.update(warm=None,weights=None)
    if hasattr(e,'m2_cut_rows'):
        del e.m2_cut_rows
        resident = True
    if resident: release_cuda(e.dev)
    return resident


def memory_pause(e, phase, lanes):
    """Record demand after eviction; only the configured cap is permanent.

    Current driver pressure (including another process) can always change at
    a later service. A failed single lane is not evidence of permanent OOM.
    """
    from auto_LiRPA.cuda_memory import CudaMemoryBudget, release_cuda
    release_optional(e,phase=phase)
    release_cuda(e.dev)
    memory = getattr(e,'memory',None)
    need = lanes*getattr(e,'scratch_bytes',0)
    disabled = False
    if isinstance(memory,CudaMemoryBudget):
        need = memory.retry_need(need)
        if memory.enabled:
            own = memory._measure()[0]
            cold = memory.failed_owned <= own
            disabled = lanes == 1 and cold and own+need > memory.limit
            if not cold:
                # Prove single-wave exhaustion with a cold attempt first.
                need = min(need,max(lanes*getattr(e,'scratch_bytes',0),memory.limit-own))
    setattr(e,phase+'_memory_need',need)
    key = phase+'_memory_pauses'
    e.stats[key] = e.stats.get(key,0)+1
    setattr(e,phase+'_memory_disabled',disabled)
    e.stats[phase+'_stop'] = 'memory' if disabled else 'memory_pause'
    return disabled


def memory_ready(e, phase):
    if getattr(e,phase+'_memory_disabled',False): return False
    need = getattr(e,phase+'_memory_need',None)
    if need is None: return True
    memory = getattr(e,'memory',None)
    if memory is not None and hasattr(memory,'available') and memory.available() < need: return False
    setattr(e,phase+'_memory_need',None)
    key = phase+'_memory_resumes'
    e.stats[key] = e.stats.get(key,0)+1
    e.stats[phase+'_stop'] = (getattr(e,'m2_state',{}).get('stop','yield')
                             if phase == 'm2' else 'yield')
    return True


def bounded_nodes(e, x_L, x_U, dm_lb, thresholds, history, deadline):
    """Admit smaller fixed-capacity waves; preserve the cover on exhaustion."""
    from contextlib import nullcontext
    from auto_LiRPA.cuda_memory import memory_error, release_cuda
    while (deadline is None or time.time() < deadline) and memory_ready(e,'m1'):
        try:
            with e.memory.allocations() if e.memory is not None else nullcontext():
                if e.memory is not None:
                    lanes = min(len(x_L),e.top)
                    if e.dev.type == 'cuda': lanes = 1 << (lanes-1).bit_length()
                    e.memory.require(lanes*e.scratch_bytes)
                return bound_nodes(e, x_L, x_U, dm_lb, thresholds, history, deadline)
        except Exception as error:
            if isinstance(error, ValueError) and str(error).startswith(('strict OBBT unary domain', 'strict QP affine-fold')):
                e.stats['undefined_fallbacks'] = e.stats.get('undefined_fallbacks', 0)+1
                break
            if not memory_error(error): raise
            e.stats['memory_backoffs'] = e.stats.get('memory_backoffs', 0)+1
        # Leave the exception scope before releasing tensors held by its traceback.
        if e.top > 1:
            release_optional(e); release_cuda(e.dev)
            e.top = max(1, e.top//2)
        else:
            memory_pause(e,'m1',1)
            break
    # Unfinished optional work never removes a domain. Its inherited native
    # certificate remains valid on the original box, including at a deadline.
    D = len(x_L)
    low = (thresholds.new_full((D, e.Sg), -float('inf')) if dm_lb is None else dm_lb.clone())
    history.engine_state = e.empty_states(D)
    history.engine_state[:, 0] = e._f32(low[:, 0].to(e.dev, torch.float64), False)
    history.engine_boxes = (x_L, x_U)
    history.engine_witness = None
    history.engine_valid = torch.zeros(D, dtype=torch.bool, device=x_L.device)
    history.engine_stay = history.engine_pending = None
    return (low, x_L.new_zeros(()).expand(D, e.Sg, *x_L.shape[1:]),
            thresholds.new_full((), max(-1e300,torch.finfo(thresholds.dtype).min)).expand(D, e.Sg))


@torch.no_grad()
def bound_nodes(e, x_L, x_U, dm_lb, thresholds, history, deadline):
    """Compatibility adapter: all node callers use the shared warm pool."""
    from input_split.node_pool import NodePool
    if not hasattr(e,'node_pool'):e.node_pool=NodePool(e)
    D=len(x_L)
    native=(thresholds.new_full((D,e.Sg),-float('inf')) if dm_lb is None else dm_lb)
    low=e.node_pool.bound(x_L,x_U,native,thresholds,deadline,e.top,history)
    low=torch.where(torch.isnan(low),-torch.inf,low)
    history.engine_state=e.empty_states(D)
    history.engine_state[:,0]=e._f32(low[:,0].to(e.dev,torch.float64),False)
    history.engine_boxes=(x_L,x_U)
    history.engine_stay=history.engine_pending=None
    if not hasattr(history,'engine_witness'):history.engine_witness=None
    if not hasattr(history,'engine_valid'):
        history.engine_valid=torch.zeros(D,dtype=torch.bool,device=x_L.device)
    # These constant planes are the node's certified scalar, valid on its
    # scope. Native GEAR still supplies its own planes for clipping.
    return low,low.new_zeros(D,e.Sg,*x_L.shape[1:]),torch.where(low.isneginf(),max(-1e300,torch.finfo(low.dtype).min),low)


class SupportWarm:
    """Physical iterates keyed by (lifted column identity, sign), not lane slot.

    A changed row/column mapping discards the pool. Values and box endpoints
    may change freely: solve clips z and rescales nonnegative physical y.
    No primal weight, averages or restart history is retained.
    """
    def __init__(self):
        self.structure = None
        self.lanes = {}

    def seed(self, data, keep, keys):
        rows, c, _, h, l, u, q = data
        structure = (rows.row, rows.col, keep)
        shape = (rows.m, rows.n, bool(q.any()))
        if (self.structure is None or self.shape != shape
                or any(not torch.equal(a,b) for a,b in zip(self.structure,structure))):
            self.structure = tuple(t.clone() for t in structure)
            self.shape = shape
            self.lanes.clear()
        if not any(key in self.lanes for key in keys): return None
        z = (.5*(l+u)).expand(len(keys),-1).clone()
        y = h.new_zeros(len(keys),rows.m+(not self.shape[2]))
        mu = c.new_zeros(len(keys)) if self.shape[2] else None
        for j,key in enumerate(keys):
            if key in self.lanes:
                zz,yy,mm = self.lanes[key]
                if zz.shape != z[j].shape or yy.shape != y[j].shape: continue
                z[j],y[j] = zz,yy
                if mu is not None: mu[j] = mm
        return z,y,mu

    def save(self, keys, warm):
        for j,key in enumerate(keys):
            self.lanes[key] = tuple(None if t is None else t[j].clone() for t in warm)


@torch.no_grad()
def m2_support(e, data, target, cutoff, steps, warm, deadline):
    """Use the bound solver for linear tables, including the cutoff as a row.

    A quadratic cutoff retains its convex-QCQP certificate; it cannot be
    inserted as a linear row without losing the square term.
    """
    from auto_LiRPA.restarted_qp import solve
    from auto_LiRPA.restarted_qcqp import support
    from nlp_solver.rigorous_dual import Rows
    lanes = getattr(e,'m2_lanes',256)
    capacity = lanes if e.dev.type == 'cuda' or bool(data[-1].any()) else min(16,lanes)
    if len(target) > capacity:
        # CUDA waves reuse one captured workspace of m2_lanes lanes (halved
        # after a memory pause). CPU uses at most 16 lanes per CSR block to
        # keep its active vectors in cache. Sequential CPU blocks retain only
        # their returned vectors, not their matrices. CUDA padding repeats
        # valid directions in a short final wave; slice it off before
        # returning any endpoints or state.
        pieces, metrics = [], []
        for start in range(0,len(target),capacity):
            tt = target[start:start+capacity]; count = len(tt)
            pad = lambda t: torch.cat((t,t[:1].expand(capacity-count,*t.shape[1:]))) if count < capacity and e.dev.type == 'cuda' else t
            seed = None if warm is None else tuple(None if t is None else pad(t[start:start+capacity]) for t in warm)
            p = m2_support(e,data,pad(tt),cutoff,steps,seed,deadline)
            pieces.append((p[0][:count],tuple(None if t is None else t[:count] for t in p[1]),p[2][:count],p[3]))
            metrics.append(m2_support.last_metrics)
        m2_support.last_metrics = dict(metrics[0])
        for name in ('setup_seconds','capture_seconds','iteration_seconds'):
            m2_support.last_metrics[name] = sum(m[name] for m in metrics)
        m2_support.last_metrics['iterations'] = max(m['iterations'] for m in metrics)
        for name in ('lane_steps','active_lane_steps','converged'):
            if name in metrics[0]: m2_support.last_metrics[name] = sum(m[name] for m in metrics)
        m2_support.last_metrics['active_lanes'] = len(target)
        if 'lane_iterations' in metrics[0]:
            m2_support.last_metrics['lane_iterations'] = [i for m in metrics for i in m['lane_iterations']]
        return (torch.cat([p[0] for p in pieces]),
                tuple(None if pieces[0][1][j] is None else torch.cat([p[1][j] for p in pieces]) for j in range(3)),
                torch.cat([p[2] for p in pieces]),sum(p[3] for p in pieces))
    rows, c, c0, h, l, u, q = data
    if bool((q != 0).any()):
        result = support(data,target,cutoff,min(steps,10000),warm,e.dev.type == 'cuda',deadline,cache=e.support_cache)
        m2_support.last_metrics = support.last_metrics
        return result
    # Immutable COO indices keep the solver's captured workspace reusable.
    if (not hasattr(e, 'm2_cut_rows') or e.m2_cut_shape != (rows.m,rows.n)
            or not torch.equal(e.m2_cut_rows[0][:-rows.n],rows.row)
            or not torch.equal(e.m2_cut_rows[1][:-rows.n],rows.col)):
        e.m2_cut_rows = (torch.cat((rows.row, torch.full((rows.n,),rows.m,device=e.dev,dtype=torch.long))),
                         torch.cat((rows.col, torch.arange(rows.n,device=e.dev))))
        e.m2_cut_shape = rows.m,rows.n
        e.support_cache.clear()
    B = len(target)
    r, j = e.m2_cut_rows
    rr = Rows(r,j,torch.cat((rows.val,c),1),rows.m+1,rows.n)
    rhs = torch.nextafter(c0.new_tensor(cutoff)-c0,c0.new_full((),float('inf')))
    hh = torch.cat((h,rhs[:,None]),1)
    convergence = None
    if getattr(e,'cfg',{}).get('crown_engine_m2_convergence',True):
        scale = max(1.,abs(cutoff))
        convergence = max(getattr(e,'abs_tol',1e-5),getattr(e,'rel_tol',1e-4)*scale)/scale
    output = solve(rr,target,c0.new_zeros(B),hh,l,u,q,
                   budget=steps,capture=e.dev.type == 'cuda',warm=None if warm is None else warm[:2],
                   deadline=deadline,cache=e.support_cache,convergence=convergence, **solver_options(e))
    m2_support.last_metrics = output.metrics
    return output.lower, (output.z,output.y,None), output.empty, int(output.restarts.sum())


def m2_refresh(state, cutoff, tau):
    return state is not None and state['anchor_cutoff']-cutoff > tau


def root_obbt(e, lower, upper, cutoff, deadline, *, round_limit=None, refresh_tol=None, columns=None):
    from .m2_async import root_obbt as run
    return run(e, lower, upper, cutoff, deadline, round_limit=round_limit, refresh_tol=refresh_tol, columns=columns)


# Spatial scheduling for the standalone measurement harness only.
# The GEAR add-on never exports these scores or witnesses to the native tree.
def nonlinear_residual(e, point):
    gap = e._violation(point)
    if not hasattr(e, 'absorb_exp_elements'):
        e.absorb_exp_elements = [j for nd in e.cg.nodes
            if type(e.pg.net[nd.name]).__name__ == 'BoundExp' for j in range(nd.s, nd.s+nd.E)]
    if e.absorb_exp_elements:
        op = e.cg.Lop.right(torch.nan_to_num(point)).unflatten(-1, (2, e.Et))+e.cg.cop[None]
        ids = e.absorb_exp_elements
        gap[:, ids] = (point[:, e.n:][:, ids]-op[:, 0, ids].exp()).abs().nan_to_num(0., 0., 0.)
    return gap


def violation_score(e, point, lo, hi):
    """Sum nonlinear residuals over each input's graph incidence, as spatial BaB.

    Incidence follows graph ancestors, including affine and product operands
    at earlier levels. It does not depend on a particular problem's names.
    """
    if not hasattr(e, 'absorb_incidence'):
        rows, cols, vals = (v.detach().cpu() for v in e.cg._op())
        parents = [set() for _ in range(e.Et)]
        for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist()):
            if v:
                parents[r % e.Et].add(c)
        deps = [{j} for j in range(e.n)]
        rr, cc = [], []
        nonlinear = (e.mul | e.sq | e.recip).cpu().tolist()
        # Products introduced solely for RLT rows do not belong to the
        # original model's spatial violation score (LiftedLP.branch_K).
        original = torch.zeros(e.Et, dtype=torch.bool)
        for node in e.cg.nodes:
            source = e.pg.carry.get(node.name, (node.name,))[0]
            if source in e.pg.fold.faces:
                original[node.s:node.s+node.E] = True
            if type(e.pg.net[node.name]).__name__ == 'BoundExp':
                for j in range(node.s, node.s+node.E):
                    nonlinear[j] = True
        for k, inputs in enumerate(parents):
            dep = set().union(*(deps[j] for j in inputs))
            deps.append(dep)
            if nonlinear[k] and original[k]:
                for j in sorted(dep):
                    rr.append(k); cc.append(j)
        e.absorb_incidence = (torch.tensor(rr, device=e.dev, dtype=torch.long),
                              torch.tensor(cc, device=e.dev, dtype=torch.long))
    rr, cc = e.absorb_incidence
    score = lo.new_zeros(len(lo), e.n)
    score.index_add_(1, cc, nonlinear_residual(e, point)[:, rr])
    rel = (hi-lo)/(e.root_hi-e.root_lo).clamp(min=1e-300)
    # Same soft sliver penalty as LiftedLP.branch_and_bound (ratio .05).
    weight = (rel/(.05*rel.amax(1, keepdim=True)).clamp(min=1e-300)).clamp(max=1)
    return score * (rel > 1e-9) * weight


def branch(e, lower, upper, history):
    lo, hi = lower.flatten(1).to(e.root_lo), upper.flatten(1).to(e.root_hi)
    point = getattr(history, 'engine_witness', None)
    if not torch.is_tensor(point) or len(point) != len(lo):
        return ((hi-lo)/(e.root_hi-e.root_lo).clamp(min=1e-300)).argmax(1)[:, None].to(lower.device)
    from input_split.node_pool import node_branch
    return node_branch(e,lower,upper,history)
