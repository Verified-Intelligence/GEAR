"""Certified Jacobi FBBT over the *current* binary64 COO table A z <= h.

No row/element Python loop. Each pass uses five segmented reductions: row
sum, absolute row sum, unusable-row flag, column min, column max. The cap
spends at most the sparse reduction count of solver scaling (11 two-sided
Ruiz/PC passes + 40 two-product power steps), hence 102//5 = 20 passes.

Products, exclusion subtraction and division are directed with nextafter.
For an arbitrarily ordered sum of k binary64 terms, gamma_(k-1) times the
exact absolute sum encloses normal rounding; k*minsub/(1-k*u) additionally
covers gradual underflow. All arithmetic constructing that envelope is
rounded outwards too. Nonfinite sums/coefficients are simply unusable.
Duplicate COO entries remain separate: excluding any one occurrence is
valid even when other occurrences refer to the same variable.
"""
import torch

SCALING_REDUCTIONS = 2*11 + 2*40
REDUCTIONS_PER_PASS = 5
DEFAULT_PASSES = SCALING_REDUCTIONS // REDUCTIONS_PER_PASS


@torch.no_grad()
def implied_box(rows, h, lower, upper, max_passes=DEFAULT_PASSES):
    if any(t.dtype != torch.float64 for t in (rows.val,h,lower,upper)):
        raise TypeError('implied box requires binary64')
    if not 0 <= max_passes <= DEFAULT_PASSES:
        raise ValueError('FBBT pass cap exceeds scaling work budget')
    B=max(len(t) for t in (rows.val,h,lower,upper))
    l=lower.expand(B,-1).clone();u=upper.expand(B,-1).clone()
    empty=torch.zeros(B,dtype=torch.bool,device=l.device)
    info=dict(passes=0,empty_hint=empty)
    if not rows.row.numel() or not max_passes:return l,u,info
    a=rows.val.expand(B,-1);rr=rows.row;cc=rows.col
    inf=l.new_tensor(float('inf'));unit=2.**-53
    up=lambda x:torch.nextafter(x,inf)
    down=lambda x:torch.nextafter(x,-inf)
    count=torch.bincount(rr,minlength=rows.m).to(l.dtype)[None]
    ku=count*unit  # exact: integer count times a power of two
    den=down(1-ku)
    gamma=up(ku/den)  # gamma_k is conservative for k-1 additions
    # Exact smallest subnormal; scaling it by the integer count is exact.
    under=up((count*2.**-1074)/den)
    rowids=rr[None].expand(B,-1);colids=cc[None].expand(B,-1)
    # Run the capped batch without a host decision between passes. A lane
    # freezes at its first representable fixed point (or empty hint); one
    # terminal read reports the effective pass count for the whole batch.
    live=torch.ones(B,dtype=torch.bool,device=l.device)
    changes=torch.zeros(max_passes,dtype=torch.bool,device=l.device)
    for it in range(max_passes):
        endpoint=torch.where(a>0,l[:,cc],u[:,cc])
        term=torch.where(a==0,0.,down(a*endpoint))
        finite=torch.isfinite(term)&torch.isfinite(a)
        terms=torch.where(finite,term,0.)
        total=l.new_zeros(B,rows.m).scatter_add_(1,rowids,terms)
        magnitude=l.new_zeros(B,rows.m).scatter_add_(1,rowids,terms.abs())
        bad=l.new_zeros(B,rows.m).scatter_reduce_(1,rowids,(~finite).to(l.dtype),reduce='amax')
        # exact sum |terms| <= fl(sum |terms|)/(1-gamma_k), with
        # the additive underflow enclosure included before division.
        absolute=up(up(magnitude+under)/down(1-gamma))
        error=up(up(gamma*absolute)+under)
        rowlo=down(total-error)
        rest=down(rowlo[:,rr]-terms)
        rhs=h.expand(B,-1)[:,rr]
        numerator=up(rhs-rest)
        # Upper bound on numerator/positive a; for a<0 the inequality
        # reverses, so round the quotient down.
        quotient=numerator/a
        bound=torch.where(a>0,up(quotient),down(quotient))
        valid=(bad[:,rr]==0)&torch.isfinite(rhs)&torch.isfinite(rowlo[:,rr])&torch.isfinite(bound)&(a!=0)
        ul=u.clone().scatter_reduce_(1,colids,torch.where(valid&(a>0),bound,inf),reduce='amin')
        ll=l.clone().scatter_reduce_(1,colids,torch.where(valid&(a<0),bound,-inf),reduce='amax')
        # Never pass an empty box to the solver/certifier or promote this
        # hint to a proof. Keep the preceding nonempty enclosure for a lane
        # whose simultaneous proposals cross; existing Farkas must decide.
        empty.logical_or_((ll>ul).any(1))
        ll=torch.where((live & ~empty)[:,None],ll,l);ul=torch.where((live & ~empty)[:,None],ul,u)
        changed=(ll!=l).any(1)|(ul!=u).any(1)
        live.logical_and_(changed)
        changes[it]=changed.any()
        l,u=ll,ul
    info['passes']=min(max_passes,1+int(changes.sum()))
    return l,u,info


@torch.no_grad()
def rc_tighten(rows, c, c0, h, lower, upper, y, cutoff):
    """Enclose {z in [lower, upper]: A z <= h, c z+c0 <= cutoff}.

    For y >= 0, objective >= D(y) + |r_j| distance_j, where
    r=c+A^T y and distance is measured from its minimizing box side.
    Use a current-table rigorous lower bound for D, a lower enclosure of
    |r_j|, and upward gap/division. The reduced-cost envelope covers both
    products and arbitrarily ordered reductions (including subnormals).
    Work/storage is O(B*(nnz+n+m)); no LP or per-column Python loop.
    Nonfinite/unresolved costs do nothing. Crossing proposals are only an
    empty hint: retain the preceding box for the existing Farkas certifier.
    """
    from nlp_solver.rigorous_dual import dual_lower_bound, _reduced_float
    if any(t.dtype != torch.float64 for t in (rows.val,c,c0,h,lower,upper,y)):
        raise TypeError('reduced-cost tightening requires binary64')
    T=torch.as_tensor(cutoff,dtype=c.dtype,device=c.device).reshape(-1)
    B=max(len(t) for t in (rows.val,c,c0,h,lower,upper,y,T))
    c,c0,h,y=c.expand(B,-1),c0.expand(B),h.expand(B,-1),y.expand(B,-1).clamp_min(0)
    l,u=lower.expand(B,-1).clone(),upper.expand(B,-1).clone()
    D,_,_=dual_lower_bound(rows,c,c0,h,l,u,y)
    r,rho,k=_reduced_float(rows,c,y)
    inf=c.new_tensor(float('inf'))
    up=lambda x:torch.nextafter(x,inf)
    down=lambda x:torch.nextafter(x,-inf)
    count=2*k+2  # products, column additions, and the final addition of c
    ku=count*2.**-53
    den=down(1-ku)
    g=up(ku/den)
    under=up((count*2.**-1074)/den)
    absolute=up(up(rho+under)/down(1-g))
    eps=up(up(g*absolute)+under)
    positive=down(r-eps);negative=down(-up(r+eps))
    gap=up(T.expand(B)-D)
    valid=(torch.isfinite(D)&torch.isfinite(gap)&(gap>=0)&torch.isfinite(y).all(1))[:,None]
    valid=valid & torch.isfinite(r)&torch.isfinite(eps)&(den>0)
    du=up(gap[:,None]/torch.where(positive>0,positive,1.))
    dl=up(gap[:,None]/torch.where(negative>0,negative,1.))
    uu=torch.where(valid&(positive>0)&torch.isfinite(l),torch.minimum(u,up(l+du)),u)
    ll=torch.where(valid&(negative>0)&torch.isfinite(u),torch.maximum(l,down(u-dl)),l)
    empty=(ll>uu).any(1)|(D>T)
    ll=torch.where(empty[:,None],l,ll);uu=torch.where(empty[:,None],u,uu)
    return ll,uu,dict(empty_hint=empty,lower=D,reduced_lower=positive,reduced_upper=-negative)


def point_test(rows, h, z, objective, cutoff_row, row_terms):
    """Outward point-relative residual and strict objective/cutoff checks.

    This is a scheduling witness only. It never supplies a bound or emptiness
    certificate. Vacuous +inf rows are ignored; other nonfinite data fail.
    row_terms is the maximum COO row length, computed outside graph capture.
    """
    from .restarted_qp import product
    from nlp_solver.rigorous_dual import gamma
    inf=z.new_full((),torch.inf)  # fill kernel, capture-safe (new_tensor is an H2D copy)
    up=lambda x:torch.nextafter(x,inf)
    down=lambda x:torch.nextafter(x,-inf)
    mag=product(rows,rows.val.abs(),z.abs())
    activity=product(rows,rows.val,z)
    error=up(gamma(row_terms+4)*up(mag+h.abs())+(row_terms+4)*2.**-1074)
    residual=up(up(activity-h)+error)
    scale=down(down(1+h.abs())+down(mag-error)).clamp_min(1.)
    relative=up(residual.clamp_min(0)/scale)
    relative=torch.where(h.isposinf(),0.,relative)
    relative=torch.where(torch.isfinite(relative),relative,inf)
    relative=relative.amax(1) if rows.m else z.new_zeros(len(z))
    c,c0,q,T=objective
    terms=c*z+q*z*z
    p=c0+terms.sum(1)
    pe=up(gamma(4*rows.n+8)*(c0.abs()+(c*z).abs().sum(1)+(q*z*z).abs().sum(1))
          +(4*rows.n+8)*2.**-1074)
    strict=(up(p+pe)<T)&torch.isfinite(z).all(1)
    if rows.m:strict &= residual[:,cutoff_row]<0
    return relative,strict


def outward_support_box(lower, upper, owner, variable, sign, endpoint):
    """Intersect input supports certified on ONE current cutoff table/box.

    endpoint must be an outward lower certificate for sign*e_variable on
    that owner's current table, whose cutoff encloses the node's cutoff.
    A primal/plain value, a parent-table value, or a scheduling residual is
    not an endpoint. Warm duals must be re-evaluated on the current table.
    Under this contract a crossing proves the cutoff set empty; other
    coordinates retain their certified contractions. Callers must replace
    crossed columns by numerical placeholders before rebuilding faces.
    """
    index=owner*lower.shape[1]+variable
    ll=lower.flatten().scatter_reduce(0,index,torch.where(sign>0,endpoint,-torch.inf),reduce='amax').reshape_as(lower)
    uu=upper.flatten().scatter_reduce(0,index,torch.where(sign<0,-endpoint,torch.inf),reduce='amin').reshape_as(upper)
    return ll,uu,ll>uu
