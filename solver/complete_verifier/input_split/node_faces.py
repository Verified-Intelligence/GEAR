"""Nested unary face anchors for the optional node compiler only."""
import copy
import torch
from auto_LiRPA.obbt_unary import OutwardCompiledGraph
from input_split.m1_qp import M1CompiledGraph


class NestedFaces(M1CompiledGraph):
    def _unary_slopes(self, nd, lo, hi, lw, uw):
        kind=type(self.net[nd.name]).__name__
        key=(nd.name,nd.E)
        if key not in self.anchors:
            self.anchors[key]=(lw[:,:1].clone(),uw[:,:1].clone())
            return lw,uw
        root_l,root_u=self.anchors[key];a,b=lo[:,0],hi[:,0]
        # The endpoint anchors project to child endpoints. Interior root
        # anchors are retained while inside the child. For a convex atom,
        # an endpoint tangent dominates any ancestor tangent outside the
        # child; the child secant dominates the ancestor secant. No resampling.
        power=float(getattr(self.net[nd.name],'exponent',0.)) if kind=='BoundPow' else None
        if kind=='BoundPow' and power in (0.,1.):return lw,uw
        def projected(k):
            if kind=='BoundSqr':p=k/2
            elif kind=='BoundExp':p=k.clamp_min(torch.finfo(k.dtype).tiny).log()
            elif kind=='BoundLog':p=1/k
            elif kind=='BoundPow':p=(k/power).clamp_min(torch.finfo(k.dtype).tiny).pow(1/(power-1))
            elif kind=='BoundReciprocal':p=torch.rsqrt((-k).clamp_min(torch.finfo(k.dtype).tiny))*torch.sign(a)
            else:return k.expand(-1,len(lo),-1)
            p=torch.maximum(a,torch.minimum(b,p))
            if kind=='BoundSqr':return 2*p
            if kind=='BoundExp':return p.exp()
            if kind=='BoundLog':return 1/p
            if kind=='BoundPow':return power*p.pow(power-1)
            return -1/(p*p)
        convex=(kind in ('BoundSqr','BoundExp')) or (kind=='BoundPow' and (power<0 or power>1))
        concave=(kind=='BoundLog') or (kind=='BoundPow' and 0<power<1)
        if kind=='BoundReciprocal':
            # Mixed sign elements have independent convex/concave sides.
            cv=a>0
            lk=projected(root_l[:,:,0]);uk=projected(root_u[:,:,0])
            lw=lw.clone();uw=uw.clone()
            lw[:,:,0]=torch.where(cv,lk,lw[:,:,0]);uw[:,:,0]=torch.where(~cv,uk,uw[:,:,0])
        elif convex:
            lw=lw.clone();lw[:,:,0]=projected(root_l[:,:,0])
        elif concave:
            uw=uw.clone();uw[:,:,0]=projected(root_u[:,:,0])
        else:
            # For nonconvex atoms, preserve the root slopes and re-enclose
            # their residual over the smaller interval (fixed affine family).
            lw=root_l.expand(-1,len(lo),-1,-1);uw=root_u.expand(-1,len(lo),-1,-1)
        return lw,uw


def compiler(engine):
    cg=copy.copy(engine.cg);cg.__class__=NestedFaces
    cg.anchors={};cg._cache_bounds=False;cg._bound_cache={}
    lo,hi=engine.root_lo,engine.root_hi
    T=lo.new_full((len(lo),),torch.inf)
    OutwardCompiledGraph.bound(cg,lo,hi,engine._thr(lo,hi,T),engine._hb(lo,hi,T))
    return cg
