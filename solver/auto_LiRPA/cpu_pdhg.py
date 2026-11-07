"""Optional CPU LP backend matching the scalar reference arithmetic.

One table/scaling serves all direction columns. This is search code only:
all bounds and Farkas rays are rechecked by rigorous_dual after the call.
CUDA and diagonal QPs use the tensor backend.
"""
import ctypes
import functools
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import numpy as np
import torch


@functools.lru_cache(None)
def kernel():
    compiler=shutil.which('g++');source=Path(__file__).with_suffix('.cpp')
    if compiler is None or os.name!='posix' or not source.is_file():return None
    import fcntl
    flags=['-O3','-std=c++17','-shared','-fPIC','-ffp-contract=off']
    digest=hashlib.sha256(source.read_bytes()+repr((compiler,flags)).encode()).hexdigest()[:20]
    root=Path(tempfile.gettempdir())/'crown_cpu_pdhg';root.mkdir(exist_ok=True)
    library=root/(digest+'.so')
    with (root/(digest+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not library.exists():
            temporary=root/(digest+f'.{os.getpid()}.so')
            subprocess.run([compiler,*flags,str(source),'-o',str(temporary)],check=True,capture_output=True)
            temporary.replace(library)
    lib=ctypes.CDLL(str(library));fn=lib.solve_lp;ptr=ctypes.c_void_p
    fn.argtypes=[ctypes.c_int]*4+[ptr]*9+[ctypes.c_int]*4+[ptr]*2+[ctypes.c_double]*2+[ptr]*6
    fn.restype=None;fn.library=lib
    return fn


def uniform(data):
    r,c,c0,h,l,u,q=data
    return (not c.is_cuda and not bool(q.any()) and all(
        len(t)==1 or t.stride(0)==0 or torch.equal(t,t[:1].expand_as(t)) for t in (r.val,h,l,u)))


def run(fn,data,budget,checkpoint,warm,deadline,convergence,metrics,iteration_scheme="average",record_search=False):
    from .restarted_qp import QPResult
    from nlp_solver.rigorous_dual import Rows,dual_lower_bound,certify_infeasible
    r,c,c0,h,l,u,q=data;B,n=c.shape;m=r.m
    c0=c0.expand(B).contiguous()
    # Preserve individual COO entries, including duplicates, in certificates.
    rows=Rows(r.row,r.col,r.val[:1],m,n)
    hh,ll,uu=(t[:1].expand(B,-1) for t in (h,l,u))
    z=c.new_zeros(B,n);best_y=c.new_zeros(B,m);ray=torch.zeros_like(best_y);y=torch.zeros_like(best_y)
    z0=(.5*(l[:1]+u[:1])).expand(B,-1).contiguous() if warm is None else warm[0].contiguous()
    y0=c.new_zeros(B,m) if warm is None else warm[1].contiguous()
    seed=torch.from_numpy(np.random.default_rng(0).standard_normal(n))
    inputs=[r.row.contiguous(),r.col.contiguous(),r.val[:1].contiguous(),c.contiguous(),c0.contiguous(),
            h[:1].contiguous(),l[:1].contiguous(),u[:1].contiguous(),seed]
    stats=c.new_zeros(B,10)
    search_trace=c.new_zeros(B,1+(budget+checkpoint-1)//checkpoint) if record_search else None
    started=time.perf_counter()
    initial=dual_lower_bound(rows,c,c0,hh,ll,uu,y0.clamp_min(0))[0]
    fn(B,n,m,len(r.row),*(t.data_ptr() for t in inputs),budget,checkpoint,int(warm is not None),int(iteration_scheme=="halpern"),
       z0.data_ptr(),y0.data_ptr(),-1. if convergence is None else convergence,
       -1. if deadline is None else deadline,*(t.data_ptr() for t in (z,best_y,ray,y,stats)),
       0 if search_trace is None else search_trace.data_ptr())
    lower=dual_lower_bound(rows,c,c0,hh,ll,uu,best_y)[0]
    empty=torch.zeros(B,dtype=torch.bool)
    nonzero=ray.ne(0).any(1)
    if bool(nonzero.any()):
        empty[nonzero]=certify_infeasible(rows,hh[nonzero],ll[nonzero],uu[nonzero],ray[nonzero])
    steps=stats[:,8].to(torch.long);restarts=stats[:,1].to(torch.long);it=int(steps.max())
    metrics=dict(metrics,iterations=it,cpu_fused=True,lane_iterations=steps.tolist(),
                 lane_steps=int(steps.sum()),active_lane_steps=int(steps.sum()),block_steps=B*it,
                 converged=int(stats[:,9].sum()),iteration_seconds=time.perf_counter()-started)
    if record_search:metrics['search_trace']=search_trace.tolist()
    metrics['ms_per_iteration']=1000*metrics['iteration_seconds']/max(1,it)
    trace=[dict(iteration=0,best=initial,empty=torch.zeros_like(empty),restarts=torch.zeros_like(restarts)),
           dict(iteration=it,best=lower.clone(),empty=empty.clone(),restarts=restarts.clone())]
    return QPResult(lower,empty,z,y,best_y,restarts,trace,metrics)
