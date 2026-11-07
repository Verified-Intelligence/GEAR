"""CPU persistent search backend. Bounds are always re-certified on the table."""
import ctypes
import functools
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import torch


@functools.lru_cache(None)
def library():
    source=Path(__file__).with_suffix('.cpp')
    digest=hashlib.sha256(source.read_bytes()+source.with_name('cpu_pdhg.cpp').read_bytes()).hexdigest()[:20]
    path=Path(tempfile.gettempdir())/('crown_session_'+digest+'.so')
    if not path.exists():
        import fcntl
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not path.exists():
                subprocess.run([shutil.which('g++'),'-O3','-std=c++17','-shared','-fPIC','-ffp-contract=off',str(source),'-o',str(path)],check=True,capture_output=True)
    lib=ctypes.CDLL(str(path));ptr=ctypes.c_void_p
    lib.session_new.argtypes=[ctypes.c_int]*5+[ptr]*12;lib.session_new.restype=ptr
    lib.session_delete.argtypes=[ptr]
    lib.session_step.argtypes=[ptr,ctypes.c_int,ctypes.c_int]+[ptr]*6
    lib.session_compact.argtypes=[ptr,ctypes.c_int,ptr,ptr,ptr]
    return lib


class Session:
    def __init__(self,data,warm=None,mask=None,iteration_scheme="average"):
        r,c,c0,h,l,u,q=data;B,n=c.shape
        assert not bool(q.any())
        z=(.5*(l+u)).expand(B,-1).contiguous() if warm is None else warm[0].contiguous()
        y=c.new_zeros(B,r.m) if warm is None else warm[1].contiguous()
        self.inputs=[r.row.contiguous(),r.col.contiguous(),r.val.contiguous(),c.contiguous(),c0.contiguous(),h.contiguous(),l.contiguous(),u.contiguous(),torch.from_numpy(np.random.default_rng(0).standard_normal(n)),z,y]
        self.mask=torch.ones(B,dtype=torch.bool) if mask is None else mask.contiguous()
        self.lib=library()
        self.ptr=self.lib.session_new(B,n,r.m,len(r.row),int(iteration_scheme=="halpern"),*(t.data_ptr() for t in self.inputs),self.mask.data_ptr())
        self.outputs=[c.new_zeros(B,n),c.new_zeros(B,r.m),c.new_zeros(B,r.m),c.new_zeros(B,r.m),c.new_zeros(B,n),c.new_zeros(B,6)]

    def advance(self,count,checkpoint):
        self.lib.session_step(self.ptr,count,checkpoint,*(t.data_ptr() for t in self.outputs))
        return self.outputs

    def compact(self,indices,data):
        self.inputs[3],self.inputs[4]=data[1].contiguous(),data[2].contiguous()
        self.lib.session_compact(self.ptr,len(indices),indices.contiguous().data_ptr(),
                                 self.inputs[3].data_ptr(),self.inputs[4].data_ptr())
        c=data[1];m=data[0].m
        self.outputs=[c.new_zeros(len(c),c.shape[1]),c.new_zeros(len(c),m),c.new_zeros(len(c),m),c.new_zeros(len(c),m),c.new_zeros(len(c),c.shape[1]),c.new_zeros(len(c),6)]

    def __del__(self):
        if getattr(self,'ptr',None):self.lib.session_delete(self.ptr)
