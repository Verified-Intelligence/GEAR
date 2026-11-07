"""Batched inverses of small square matrices (fp64 or fp32, s <= 64), one CUDA thread block per matrix, the matrix in shared
memory: Gauss-Jordan with partial pivoting (the first largest |entry| of the column, as LAPACK's idamax) on the
leading k x k block, k per matrix (the caller guarantees the rest is identity padding with zero coupling), fixed
shapes, no host synchronisation, capturable in a CUDA graph, an opaque custom op under torch.compile.

lockstep_dual uses it for its k x k Schur kernels and q x q exchange kernels as the alternative (kernel='gj') to
cuBLAS's batched getrf (torch.linalg.lu_factor_ex under preferred_linalg_library('cusolver')), whose latency on
64 x 64 fp64 batches grows to milliseconds (lockstep/out/lu_micro.*). The work of a matrix is proportional to its own
k. The kernel is compiled at first use by NVRTC (the CUDA 12 runtime compiler shipped with PyTorch's wheels) and
launched by the driver API on the current stream. Returns the inverse and ok (no exactly zero or non-finite pivot,
LAPACK's singularity criterion). The search uses it; certificates never depend on it."""
import ctypes
import glob
import os

import torch

_SRC = r'''
template <typename F>
__device__ void gj(const F* __restrict__ K, F* __restrict__ X, const long long* __restrict__ kk,
                   unsigned char* __restrict__ ok, int S)
{
    extern __shared__ __align__(8) unsigned char smem[];
    F* a = reinterpret_cast<F*>(smem);       // S x S, then the column and the pivot row (S each)
    F* col = a + S * S;
    F* prow = col + S;
    __shared__ int s_p, s_bad, ipiv[64];
    __shared__ F s_pinv;
    const long long b = blockIdx.x;
    const F* Kb = K + b * S * S;
    F* Xb = X + b * S * S;
    const int k = (int)kk[b];
    const int tid = threadIdx.x, nt = blockDim.x;
    const int tx = tid % S, ty = tid / S, ny = nt / S;     // a thread: one column, rows ty, ty + ny, ...
    for (int e = tid; e < S * S; e += nt) a[e] = Kb[e];
    if (tid == 0) s_bad = 0;
    __syncthreads();
    for (int j = 0; j < k; ++j) {
        if (tid < 32) {
            F best = -1; int bi = j;
            for (int i = j + tid; i < k; i += 32) {
                F v = fabs(a[i * S + j]);
                if (v > best) { best = v; bi = i; }
            }
            for (int off = 16; off > 0; off >>= 1) {
                F ob = __shfl_down_sync(0xffffffffu, best, off);
                int oi = __shfl_down_sync(0xffffffffu, bi, off);
                if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; }
            }
            if (tid == 0) { s_p = bi; ipiv[j] = bi; }
        }
        __syncthreads();
        const int p = s_p;
        if (tid < k && p != j) { F t = a[j * S + tid]; a[j * S + tid] = a[p * S + tid]; a[p * S + tid] = t; }
        __syncthreads();
        if (tid < k) col[tid] = (tid == j) ? F(0) : a[tid * S + j];
        if (tid == 0) {
            F piv = a[j * S + j];
            if (piv == F(0) || !isfinite(piv)) { s_bad = 1; s_pinv = F(0); } else s_pinv = F(1) / piv;
        }
        __syncthreads();
        if (tid < k) { F v = (tid == j ? F(1) : a[j * S + tid]) * s_pinv; prow[tid] = v; a[j * S + tid] = v; }
        __syncthreads();
        if (tx < k) {
            const F rj = prow[tx];
            for (int i = ty; i < k; i += ny)
                if (i != j) a[i * S + tx] = (tx == j ? F(0) : a[i * S + tx]) - col[i] * rj;
        }
        __syncthreads();
    }
    for (int j = k - 1; j >= 0; --j) {           // undo the row interchanges on the columns
        const int p = ipiv[j];
        if (p != j && tid < k) { F t = a[tid * S + j]; a[tid * S + j] = a[tid * S + p]; a[tid * S + p] = t; }
        __syncthreads();
    }
    for (int e = tid; e < S * S; e += nt) Xb[e] = a[e];
    if (tid == 0) ok[b] = (unsigned char)(s_bad == 0);
}

extern "C" __global__ void gj_inverse_f64(const double* K, double* X, const long long* kk, unsigned char* ok, int S)
{ gj<double>(K, X, kk, ok, S); }
extern "C" __global__ void gj_inverse_f32(const float* K, float* X, const long long* kk, unsigned char* ok, int S)
{ gj<float>(K, X, kk, ok, S); }
'''

_mods = {}


def _libs():
    import nvidia.cuda_nvrtc as m
    d = os.path.join(os.path.dirname(m.__file__), 'lib')
    nvrtc = ctypes.CDLL(sorted(glob.glob(os.path.join(d, 'libnvrtc.so.*')))[0])
    cuda = ctypes.CDLL('libcuda.so.1')
    return nvrtc, cuda


def _check(r, what):
    if r != 0:
        raise RuntimeError(f'{what} failed ({r})')


def _function(device):
    """the compiled kernel for `device` (NVRTC -> cubin for its compute capability, loaded in the current context)"""
    idx = torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()
    if idx in _mods:
        return _mods[idx]
    nvrtc, cuda = _libs()
    major, minor = torch.cuda.get_device_capability(idx)
    prog = ctypes.c_void_p()
    _check(nvrtc.nvrtcCreateProgram(ctypes.byref(prog), _SRC.encode(), b'gj_inverse.cu', 0, None, None),
           'nvrtcCreateProgram')
    opts = [f'--gpu-architecture=sm_{major}{minor}'.encode(), b'--std=c++14', b'-fmad=false']
    arr = (ctypes.c_char_p * len(opts))(*opts)
    r = nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
    if r != 0:
        n = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        log = ctypes.create_string_buffer(n.value)
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError('nvrtc: ' + log.value.decode())
    n = ctypes.c_size_t()
    _check(nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(n)), 'nvrtcGetCUBINSize')
    cubin = ctypes.create_string_buffer(n.value)
    _check(nvrtc.nvrtcGetCUBIN(prog, cubin), 'nvrtcGetCUBIN')
    torch.empty(1, device=f'cuda:{idx}')                       # torch's primary context current on this thread
    mod = ctypes.c_void_p()
    _check(cuda.cuModuleLoadData(ctypes.byref(mod), cubin), 'cuModuleLoadData')
    fns = {}
    for dt, name in ((torch.float64, b'gj_inverse_f64'), (torch.float32, b'gj_inverse_f32')):
        fn = ctypes.c_void_p()
        _check(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, name), 'cuModuleGetFunction')
        fns[dt] = fn
    _mods[idx] = (cuda, mod, fns)
    return _mods[idx]


def _launch(K, X, kk, ok):
    cuda, _, fns = _function(K.device)
    fn = fns[K.dtype]
    B, S = K.shape[0], K.shape[1]
    nt = max(32, S * (256 // S))                               # a multiple of S (S <= 64), >= a warp
    shared = (S * S + 2 * S) * K.element_size()
    pK, pX, pk, po = (ctypes.c_uint64(t.data_ptr()) for t in (K, X, kk, ok))
    pS = ctypes.c_int(S)
    args = (ctypes.c_void_p * 5)(*(ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in (pK, pX, pk, po, pS)))
    stream = ctypes.c_void_p(torch.cuda.current_stream(K.device).cuda_stream)
    _check(cuda.cuLaunchKernel(fn, B, 1, 1, nt, 1, 1, shared, stream, args, None), 'cuLaunchKernel')


@torch.library.custom_op('lockstep::gj_inverse', mutates_args=())
def _gj(K: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    K = K.contiguous()
    X = torch.empty_like(K)
    ok = torch.empty(K.shape[0], dtype=torch.uint8, device=K.device)
    if K.shape[0]:
        _launch(K, X, k.to(torch.int64).contiguous(), ok)
    return X, ok.bool()


@_gj.register_fake
def _(K, k):
    return torch.empty_like(K), torch.empty(K.shape[0], dtype=torch.bool, device=K.device)


def inverse(K, k=None):
    """K (B, s, s) fp64 or fp32 (s <= 64), its leading k x k block (k (B,), default s) the matrix and the rest identity ->
    (K^-1 (B, s, s), ok (B,) bool)"""
    B, s, _ = K.shape
    assert s <= 64 and K.dtype in (torch.float64, torch.float32) and K.is_cuda
    kk = torch.full((B,), s, dtype=torch.int64, device=K.device) if k is None else k
    return _gj(K, kk)
