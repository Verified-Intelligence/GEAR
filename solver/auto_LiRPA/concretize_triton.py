#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2026 The α,β-CROWN Team                        ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Fused GPU kernel for the sequential dual sweep of `constraints_solving`.

`constraints_solving` bounds  min_x c^T x  over a box intersected with linear rows by
visiting the rows one at a time: for row k it picks the multiplier beta_k >= 0 that
maximizes the one-row Lagrangian of the current objective (`_solve_dual_var`: sort the
breakpoints -c_i / a_i, cumulative-sum the weights |a_i eps_i|, take the first
breakpoint where the slope changes sign), then folds the row in (c += beta_k a_k).

In PyTorch every row costs ~15 kernels over a (B, H, X) tensor, including a radix sort
and a scan that are slow for short rows (X ~ 20-200): on wastewater04m2 the sweep was
half of all GPU time. This kernel keeps each objective row in registers for the whole
sweep, sorts the breakpoints with a bitonic network, and writes one number per
objective.

It computes the same multipliers and the same bound up to floating-point rounding. Only
three details differ from the reference: the breakpoints are sorted on keys whose lowest
log2(XP) bits carry the element index (so two breakpoints closer than ~2^-44 relative
may be visited in either order), a breakpoint is c_i * (-1 / a_i) rather than
-(c_i / a_i) (one rounding apart), and sums are reduced in a different order. None can
affect validity: the Lagrangian bound is valid for every beta >= 0, and the final value
is evaluated from the betas actually used, exactly as in the reference.

Production launches use only artifacts installed by tools/precompile_triton.py.
The fixed ABI prevents scalar/alignment specializations; a missing artifact
selects the existing PyTorch sweep. The JIT entry point is for installation and
GPU validation only.
"""
import torch
import time
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sysconfig

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover - triton missing
    HAS_TRITON = False

# Largest padded row length handled by the kernel; longer rows use the PyTorch sweep.
MAX_XP = 1024


def sweep_bucket(H, X):
    """The original launch geometry, independent of B, K and tensor strides."""
    if H <= 0 or not 0 < X <= MAX_XP:
        raise ValueError("sweep requires H > 0 and 0 < X <= 1024")
    xp = max(16, 1 << (X - 1).bit_length())
    block_h = max(1, min(1 << (H - 1).bit_length(), 256 // xp))
    return xp, block_h, 1 if xp <= 32 else (2 if xp <= 128 else 4)


def sweep_buckets():
    """All 17 geometries (34 binaries including the magnitude flag)."""
    return sorted({sweep_bucket(h, x) for x in (16, 32, 64, 128, 256, 512, 1024)
                   for h in (1, 2, 4, 8, 16)})


def _artifact_root():
    return Path(os.environ.get("AUTOLIRPA_TRITON_CACHE",
                               "~/.cache/auto_lirpa/triton")).expanduser()


def _artifact_identity(device):
    # The native launcher ABI below is deliberately limited to the inspected
    # Triton release. Other versions safely use PyTorch until supported.
    if not HAS_TRITON or triton.__version__ != "3.4.0":
        return None
    arch = torch.cuda.get_device_capability(device)
    data = dict(schema=1, triton=triton.__version__, arch=list(arch),
                source=hashlib.sha256(_sweep_kernel.src.encode()).hexdigest(),
                abi=sysconfig.get_config_var("SOABI"))
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _artifact_dir(device, bucket, return_magnitude):
    identity = _artifact_identity(device)
    if identity is None:
        return None
    xp, bh, nw = bucket
    return _artifact_root() / identity / f"x{xp}_h{bh}_w{nw}_m{int(return_magnitude)}"


def _load_native(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=256)
def _load_artifact(directory, device_index):
    """Load native code only. Never call JIT, compile(), ptxas or a C compiler.

    The install script copies both the cubin and its native launcher here. A
    missing/invalid receipt is a normal cache miss, not a compiler failure.
    Misses are cached for this process; run precompilation before starting it.
    """
    try:
        receipt = json.loads((directory / "ready.json").read_text())
        for name in ("kernel.json", "kernel.cubin", "launcher.so", "cuda_utils.so"):
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != receipt["sha256"][name]:
                return None
        metadata = json.loads((directory / "kernel.json").read_text())
        if (metadata["global_scratch_size"] or metadata["launch_cooperative_grid"]
                or metadata["launch_pdl"] or metadata["num_ctas"] != 1):
            return None
        # Copy the driver shim too: initializing Triton's lazy driver can invoke
        # a host C compiler when ~/.triton/cache has been cleared.
        utils = _load_native("cuda_utils", directory / "cuda_utils.so")
        shared = metadata["shared"]
        props = torch.cuda.get_device_properties(device_index)
        if shared > props.shared_memory_per_block_optin:
            return None
        module, function, _, _, max_threads = utils.load_binary(
            metadata["name"], (directory / "kernel.cubin").read_bytes(), shared, device_index)
        if metadata["num_warps"] * 32 > max_threads:
            return None
        launcher = _load_native("__triton_launcher", directory / "launcher.so")
        packed = (metadata["num_warps"], 1, shared, *metadata["cluster_dims"])
        # Retain the loaded CUDA module for the lifetime of the callable.
        def launch(grid, args):
            stream = torch.cuda.current_stream(device_index).cuda_stream
            launcher.launch(*grid, stream, function, False, False, None,
                            packed, None, None, None, *args)
        launch.module = module
        return launch
    except torch.OutOfMemoryError:
        raise
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError):
        return None


def _precompiled_sweep(device, H, X, return_magnitude):
    directory = _artifact_dir(device, sweep_bucket(H, X), return_magnitude)
    if directory is None:
        return None
    with torch.cuda.device(device):
        return _load_artifact(directory, torch.cuda.current_device())


if HAS_TRITON:
    @triton.jit
    def _sweep_kernel(C, A, D, E, OUT, MAG, H, X, K,
                      s_cb, s_ch, s_ab, s_ak, s_db, s_eb, s_ob,
                      BLOCK_H: tl.constexpr, XP: tl.constexpr, LOGXP: tl.constexpr,
                      RETURN_MAGNITUDE: tl.constexpr):
        b = tl.program_id(0).to(tl.int64)
        hb = tl.program_id(1)
        offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_x = tl.arange(0, XP)
        mh = offs_h < H
        mx = offs_x < X
        m2 = mh[:, None] & mx[None, :]
        c = tl.load(C + b * s_cb + offs_h[:, None].to(tl.int64) * s_ch + offs_x[None, :],
                    mask=m2, other=0.0)
        e = tl.load(E + b * s_eb + offs_x, mask=mx, other=0.0)
        dp = tl.zeros([BLOCK_H], dtype=tl.float64)
        if RETURN_MAGNITUDE:
            cm = tl.abs(c)
            dm = tl.zeros([BLOCK_H], dtype=tl.float64)
        col = tl.broadcast_to(offs_x[None, :], [BLOCK_H, XP])
        col64 = col.to(tl.int64)
        inf = float("inf")
        for k in range(K):
            a = tl.load(A + b * s_ab + k * s_ak + offs_x, mask=mx, other=0.0)
            d = tl.load(D + b * s_db + k)
            w = tl.abs(a * e)
            # Breakpoints -c_i / a_i of the one-row Lagrangian (one reciprocal per column:
            # fp64 division is the kernel's most expensive instruction); padding sorts
            # last with weight 0. a_i = 0 gives -inf * c_i, as -(c_i / 0) does.
            ra = -1.0 / a
            q = c * ra[None, :]
            q = tl.where(mx[None, :], q, inf)
            bits = q.to(tl.int64, bitcast=True)
            key = bits ^ ((bits >> 63) & 0x7FFFFFFFFFFFFFFF)     # float order as int64
            key = ((key >> LOGXP) << LOGXP) | col64              # carry the index
            key = tl.sort(key, dim=1)
            sidx = (key & (XP - 1)).to(tl.int32)
            ws = tl.gather(tl.broadcast_to(w[None, :], [BLOCK_H, XP]), sidx, axis=1)
            qs = tl.gather(q, sidx, axis=1)
            cs = tl.cumsum(ws, axis=1)
            total = tl.sum(tl.where(col == XP - 1, cs, 0.0), axis=1)
            g0 = -total - d
            g = (cs - total[:, None]) + cs - d
            first = tl.min(tl.where(g >= 0, col, XP), axis=1)
            beta = tl.min(tl.where(col == first[:, None], qs, inf), axis=1)
            beta = tl.where(g0 >= 0, 0.0, beta)
            beta = tl.maximum(beta, 0.0, propagate_nan=tl.PropagateNan.ALL)
            c = c + beta[:, None] * a[None, :]
            dp = dp + beta * d
            if RETURN_MAGNITUDE:
                update = tl.abs(beta[:, None] * a[None, :])
                cm += tl.where(update < inf, update, 0.0)
                update_d = tl.abs(beta * d)
                dm += tl.where(update_d < inf, update_d, 0.0)
        out = dp - tl.sum(tl.abs(c) * e[None, :], axis=1)
        tl.store(OUT + b * s_ob + offs_h, out, mask=mh)
        if RETURN_MAGNITUDE:
            magnitude = dm + tl.sum((cm + tl.abs(c)) * tl.abs(e[None, :]), axis=1)
            tl.store(MAG + b * s_ob + offs_h, magnitude, mask=mh)


def fused_sweep_supported(obj_mat, constr_A, eps_mat, return_magnitude=False):
    """Whether `fused_dual_sweep` can replace the PyTorch sweep for these tensors."""
    if not HAS_TRITON or obj_mat.device.type != "cuda":
        return False
    if obj_mat.dtype != torch.float64 or constr_A.dtype != torch.float64 or eps_mat.dtype != torch.float64:
        return False
    if torch.is_grad_enabled() and (obj_mat.requires_grad or constr_A.requires_grad):
        return False
    X = obj_mat.shape[-1]
    H = obj_mat.shape[1]
    return (0 < X <= MAX_XP and 0 < H < 2**31 and constr_A.shape[1] < 2**31
            and _precompiled_sweep(obj_mat.device, H, X, return_magnitude) is not None)


def fused_dual_sweep(obj_mat, constr_A, constr_d, eps_mat, return_magnitude=False,
                     deadline=None):
    """Sequential dual sweep over all rows, fused.

    Args:
        obj_mat: (B, H, X) objective rows, already negated for an upper bound.
        constr_A: (B, K, X) constraint rows a_k (a_k^T (x - x0) + d_k <= 0).
        constr_d: (B, K) centred offsets d_k.
        eps_mat: (B, X, 1) or (B, X) box radii.
    Returns:
        (B, H): sum_k beta_k d_k - |c + sum_k beta_k a_k|^T eps, the reference's
        `dual_objective_part` after the final epsilon term.
        When return_magnitude is True, also return the sum of absolute bias
        and coefficient-update contributions for the rounding enclosure.
    """
    B, H, X = obj_mat.shape
    if obj_mat.device.type != "cuda" or any(
            t.dtype != torch.float64 or t.device != obj_mat.device
            for t in (obj_mat, constr_A, constr_d, eps_mat)):
        raise ValueError("Precompiled sweep requires float64 tensors on the same CUDA device")
    if deadline is not None and time.time() >= deadline:
        radius = (obj_mat.abs() * eps_mat.reshape(B, 1, X)).sum(-1)
        return (-radius, 2 * radius) if return_magnitude else -radius
    K = constr_A.shape[1]
    kernel = _precompiled_sweep(obj_mat.device, H, X, return_magnitude)
    if kernel is None:
        raise RuntimeError("Sweep is not precompiled; run tools/precompile_triton.py before solving")
    C = obj_mat.contiguous()
    A = constr_A.contiguous()
    D = constr_d.contiguous()
    E = eps_mat.reshape(B, X).contiguous()
    out = torch.empty(B, H, device=C.device, dtype=C.dtype)
    magnitude = torch.empty_like(out) if return_magnitude else out
    if B == 0:
        return (out, magnitude) if return_magnitude else out
    # ~256 breakpoints per program on one or two warps: the best of a (BLOCK_H,
    # num_warps) sweep on B x H x X = 4096x36x55, 4096x125x55, 131072x21x21,
    # 16384x20x26 and 256x200x60 (RTX 5090), 12-37% faster than 16 rows x 4 warps.
    _, BLOCK_H, _ = sweep_bucket(H, X)
    with torch.cuda.device(C.device):
        kernel((B, triton.cdiv(H, BLOCK_H), 1),
               (C, A, D, E, out, magnitude, H, X, K,
                C.stride(0), C.stride(1), A.stride(0), A.stride(1), D.stride(0),
                E.stride(0), out.stride(0)))
    return (out, magnitude) if return_magnitude else out
