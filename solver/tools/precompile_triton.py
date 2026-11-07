#!/usr/bin/env python3
"""Install every fused sweep variant on each GPU host, before running solves.

Uses a fixed ABI: six fp64 pointers, i32 H/X/K, seven i64 strides. No inferred
scalar, alignment or equals-one specializations. Arithmetic and geometry are
unchanged. Run again after changing this kernel, Python, Triton or GPU arch.
AUTOLIRPA_TRITON_CACHE selects the persistent artifact directory. Keep it local
and trusted, like any directory containing native executable code.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_LiRPA import concretize_triton as ct


def sweep_source(bucket, magnitude):
    from triton.compiler import ASTSource
    xp, bh, _ = bucket
    signature = dict(zip(ct._sweep_kernel.arg_names[:16],
                         ["*fp64"] * 6 + ["i32"] * 3 + ["i64"] * 7))
    return ASTSource(ct._sweep_kernel, signature, constexprs={
        "BLOCK_H": bh, "XP": xp, "LOGXP": xp.bit_length() - 1,
        "RETURN_MAGNITUDE": magnitude}, attrs={})


def compile_variant(bucket, magnitude):
    import triton
    return triton.compile(sweep_source(bucket, magnitude), options={
        "num_warps": bucket[2], "num_stages": 3, "enable_fp_fusion": True,
        "debug": False})


def install_variant(bucket, magnitude, device):
    """Publish a receipt only after all native code is compiled and loadable."""
    from triton.runtime.driver import driver
    destination = ct._artifact_dir(device, bucket, magnitude)
    start = time.perf_counter()
    kernel = compile_variant(bucket, magnitude)
    kernel._init_handles()  # also compile the native launcher and check resources
    cold_or_cached = time.perf_counter() - start
    start = time.perf_counter()
    compile_variant(bucket, magnitude)._init_handles()
    warm = time.perf_counter() - start
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        staging = Path(tmp)
        (staging / "kernel.cubin").write_bytes(kernel.asm["cubin"])
        (staging / "kernel.json").write_text(json.dumps(kernel.metadata._asdict(), default=vars))
        shutil.copyfile(kernel.run.launch.__self__.__file__, staging / "launcher.so")
        shutil.copyfile(driver.active.utils.load_binary.__self__.__file__, staging / "cuda_utils.so")
        names = ("kernel.cubin", "kernel.json", "launcher.so", "cuda_utils.so")
        receipt = dict(sha256={n: hashlib.sha256((staging / n).read_bytes()).hexdigest() for n in names})
        # Readers require ready.json. Reinstalling invalidates it before replacing
        # files, and publishes it last; an interrupted install safely falls back.
        destination.mkdir(exist_ok=True)
        (destination / "ready.json").unlink(missing_ok=True)
        for name in names:
            os.replace(staging / name, destination / name)
        (staging / "ready.json").write_text(json.dumps(receipt, sort_keys=True))
        os.replace(staging / "ready.json", destination / "ready.json")
    return dict(bucket=bucket, magnitude=magnitude,
                first_compile_or_cache_seconds=cold_or_cached, warm_cache_seconds=warm,
                artifact=str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="visible CUDA device index")
    parser.add_argument("--output", type=Path, help="write timing records as JSON")
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available() or ct._artifact_identity(args.device) is None:
        parser.error("requires CUDA and Triton 3.4.0; other environments use PyTorch")
    records = []
    with torch.cuda.device(args.device):
        for bucket in ct.sweep_buckets():
            for magnitude in (False, True):
                record = install_variant(bucket, magnitude, args.device)
                print(json.dumps(record), flush=True)
                records.append(record)
    if args.output:
        args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
