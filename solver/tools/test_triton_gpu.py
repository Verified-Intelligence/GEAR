#!/usr/bin/env python3
"""GPU acceptance gate for A9; main runs this on both GPU hosts.

Requires either --gould-capture (real constraints_solving kwargs saved with
torch.save) or --config and --benchmark-root to capture cute_gouldqp2 directly.
Uses a fresh temporary compiler cache; never deletes the host's Triton cache.
Writes per-variant cold/warm compilation times and correctness receipts.
"""
import argparse
from contextlib import ExitStack
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def legacy_sweep(c, a, d, e, return_magnitude=False, deadline=None):
    """The original launch site, using the unchanged kernel body and JIT rules."""
    import torch
    from auto_LiRPA import concretize_triton as ct
    b, h, x = c.shape
    c, a, d, e = c.contiguous(), a.contiguous(), d.contiguous(), e.reshape(b, x).contiguous()
    out = torch.empty((b, h), device=c.device, dtype=c.dtype)
    mag = torch.empty_like(out) if return_magnitude else out
    xp, bh, nw = ct.sweep_bucket(h, x)
    ct._sweep_kernel[(b, (h + bh - 1) // bh)](
        c, a, d, e, out, mag, h, x, a.shape[1],
        c.stride(0), c.stride(1), a.stride(0), a.stride(1), d.stride(0),
        e.stride(0), out.stride(0), BLOCK_H=bh, XP=xp, LOGXP=xp.bit_length() - 1,
        RETURN_MAGNITUDE=return_magnitude, num_warps=nw)
    return (out, mag) if return_magnitude else out


def no_compile():
    """Fail if the production dispatch reaches any GPU or host compiler."""
    import triton
    import triton.compiler.compiler as compiler
    import triton.runtime.build as build
    from auto_LiRPA import concretize_triton as ct
    def forbidden(*args, **kwargs):
        raise AssertionError("compiler reached from solve path")
    stack = ExitStack()
    stack.enter_context(patch.object(triton, "compile", forbidden))
    stack.enter_context(patch.object(compiler, "compile", forbidden))
    stack.enter_context(patch.object(build, "_build", forbidden))
    stack.enter_context(patch.object(ct._sweep_kernel, "run", forbidden))
    return stack


def bitwise(a, b):
    import torch
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for av, bv in zip(a, b):
            bitwise(av, bv)
    elif a is None:
        assert b is None
    else:
        assert a.dtype == b.dtype and a.shape == b.shape
        av = a.contiguous().view(torch.int64) if a.dtype == torch.float64 else a
        bv = b.contiguous().view(torch.int64) if b.dtype == torch.float64 else b
        assert torch.equal(av, bv), "precompiled and original fused output differ in bits"


def random_gate():
    import torch
    from auto_LiRPA import concretize_func as cf, concretize_triton as ct
    g = torch.Generator(device="cuda").manual_seed(20261019)
    records = []
    for xp, bh, nw in ct.sweep_buckets():
        # Cover exact/padded widths, scalar equals-one/alignment classes, a
        # noncontiguous input, zero constraints, and runtime row/batch counts.
        shapes = [(1, bh, xp, 0), (3, bh, xp - 1, 1), (2, bh, xp, 7)]
        if bh == max(1, 256 // xp):
            shapes.append((3, 2 * bh + 1, max(1, xp // 2 + 1), 16))
        if xp == 16:
            shapes.append((2, bh, 1, 3))
        for b, h, x, k in shapes:
            c = torch.randn(b, h, x * 2, device="cuda", dtype=torch.float64, generator=g)[..., ::2]
            a = torch.randn(b, k, x, device="cuda", dtype=torch.float64, generator=g)
            d = -torch.rand(b, k, device="cuda", dtype=torch.float64, generator=g) - .1
            e = torch.rand(b, x, 1, device="cuda", dtype=torch.float64, generator=g) + .1
            for magnitude in (False, True):
                ref = legacy_sweep(c, a, d, e, magnitude)
                # Clear loaded handles to exercise native disk loading too.
                ct._load_artifact.cache_clear()
                with no_compile():
                    assert ct.fused_sweep_supported(c, a, e, magnitude)
                    start = time.perf_counter()
                    result = ct.fused_dual_sweep(c, a, d, e, magnitude)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                bitwise(ref, result)
                coeff, value = c.clone(), torch.zeros(b, h, device="cuda", dtype=c.dtype)
                for row in range(k):
                    beta = cf._solve_dual_var(a[:, row], coeff, d[:, row], e)
                    coeff += beta[..., None] * a[:, row, None]
                    value += beta * d[:, row, None]
                value -= (coeff.abs() * e.reshape(b, 1, x)).sum(-1)
                fused_value = result[0] if magnitude else result
                torch.testing.assert_close(fused_value, value, atol=1e-9, rtol=1e-9)
                records.append(dict(shape=[b, h, x, k], bucket=[xp, bh, nw],
                                    magnitude=magnitude, seconds=elapsed, bitwise=True))
    # An absent artifact must cause selection of PyTorch, without any compiler.
    with tempfile.TemporaryDirectory() as absent, patch.dict(os.environ, {"AUTOLIRPA_TRITON_CACHE": absent}):
        ct._load_artifact.cache_clear()
        with no_compile():
            assert not ct.fused_sweep_supported(c, a, e, True)
            lower = -e.squeeze(-1)
            upper = -lower
            constraints, batches = cf.sort_out_constr_batches(lower, upper, (a, d), no_return_inf=True)
            def bounds():
                return cf.constraints_solving(lower, upper, c, constraints, -1.,
                                             sorted_out_batches=batches, no_return_inf=True,
                                             use_grad=False)
            with patch.object(cf, "FUSED_SWEEP", False):
                fallback = bounds()
            with patch.object(cf, "FUSED_SWEEP", True):
                miss = bounds()
            bitwise(fallback, miss)
    return records


def tree_map(value, fn):
    import torch
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict):
        return {k: tree_map(v, fn) for k, v in value.items()}
    if isinstance(value, tuple) and not hasattr(value, "_fields"):
        return tuple(tree_map(v, fn) for v in value)
    return value


def gould_gate(args):
    import torch
    from auto_LiRPA import concretize_func as cf, concretize_triton as ct
    original = cf.constraints_solving
    native = ct.fused_dual_sweep
    records = []
    def replay(values):
        values = tree_map(dict(values, timer=None, use_grad=False), lambda t: t.to("cuda"))
        used = []
        def measured(*a, **kw):
            used.append(list(a[0].shape))
            return native(*a, **kw)
        with patch.object(cf, "FUSED_SWEEP", True), patch.object(ct, "fused_dual_sweep", measured):
            with no_compile():
                result = original(**values)
        assert used, "Gould capture did not exercise the precompiled sweep"
        with patch.object(cf, "FUSED_SWEEP", True), patch.object(ct, "fused_dual_sweep", legacy_sweep):
            baseline = original(**values)
        bitwise(baseline, result)
        records.append(dict(shapes=used, bitwise=True))
        torch.save(tree_map(result, lambda t: t.cpu()), args.output / "gould_output.pt")
    if args.gould_capture:
        replay(torch.load(args.gould_capture, map_location="cpu", weights_only=False))
    else:
        import auto_LiRPA.perturbations as perturb
        signature = inspect.signature(original)
        class Captured(BaseException):
            pass
        def capture(*a, **kw):
            values = dict(signature.bind(*a, **kw).arguments)
            active = (values.get("sorted_out_batches") or {}).get("active_indices")
            if (values["x_L"].shape[0] >= 2 and values.get("constraints") is not None
                    and (active is None or active.numel() >= 2)):
                saved = tree_map(dict(values, timer=None), lambda t: t.detach().cpu())
                torch.save(saved, args.output / "gould_inputs.pt")
                replay(saved)
                raise Captured()
            return original(*a, **kw)
        old_cwd = Path.cwd()
        try:
            os.chdir(ROOT / "complete_verifier")
            sys.path.insert(0, str(Path.cwd()))
            with patch.object(cf, "constraints_solving", capture), patch.object(perturb, "constraints_solving", capture):
                sys.argv = ["nlp_example.py", "--problem_name", "cute_gouldqp2", "--benchmark", "gams_world",
                            "--benchmark_root", str(args.benchmark_root), "--device", "cuda", "--use_beta", "False",
                            "--override_timeout", "180", "--batch_size", "1", "--config", str(args.config)]
                runpy.run_path("nlp_example.py", run_name="__main__")
        except Captured:
            pass
        finally:
            os.chdir(old_cwd)
        assert records, "No active Gould concretization captured; provide a saved capture"
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gould-capture", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--benchmark-root", type=Path)
    args = p.parse_args()
    if not args.gould_capture and not (args.config and args.benchmark_root):
        p.error("provide --gould-capture or both --config and --benchmark-root")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # Use a separate artifact tree as well: tests cannot replace an installation
    # concurrently used by benchmark workers.
    os.environ["AUTOLIRPA_TRITON_CACHE"] = str(args.output / "artifacts")
    with tempfile.TemporaryDirectory(prefix="a9-cold-triton-") as cold:
        env = dict(os.environ, TRITON_CACHE_DIR=cold)
        subprocess.run([sys.executable, str(ROOT / "tools/precompile_triton.py"),
                        "--output", str(args.output / "compile_times.json")], env=env, check=True)
        # Prove disk-cache reuse in a fresh Python process too.
        subprocess.run([sys.executable, str(ROOT / "tools/precompile_triton.py"),
                        "--output", str(args.output / "warm_process_times.json")], env=env, check=True)
    import torch
    torch.set_num_threads(min(6, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    from auto_LiRPA import concretize_triton as ct
    ct._load_artifact.cache_clear()
    random_results = random_gate()
    gould_results = gould_gate(args)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_concretize_fused.py"],
                   cwd=ROOT, check=True)
    receipt = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                   random=random_results, cute_gouldqp2=gould_results)
    (args.output / "correctness.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("PASS: no compile in solve; original fused bits and fallback tolerance; Gould bits unchanged")


if __name__ == "__main__":
    main()
