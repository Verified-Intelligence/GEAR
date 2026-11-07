"""CPU regressions for the finite sweep geometry and cache-only dispatch."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from auto_LiRPA import concretize_triton as ct


@pytest.mark.parametrize("x", range(1, 1025))
def test_bucket_matches_original_geometry(x):
    for h in (1, 2, 3, 4, 7, 8, 15, 16, 17, 36, 125, 200, 1025):
        xp = max(16, 1 << (x - 1).bit_length())
        bh = max(1, min(1 << (h - 1).bit_length(), 256 // xp))
        nw = 1 if xp <= 32 else (2 if xp <= 128 else 4)
        assert ct.sweep_bucket(h, x) == (xp, bh, nw)
        assert ct.sweep_bucket(h, x) in ct.sweep_buckets()


def test_all_buckets_are_reachable():
    buckets = ct.sweep_buckets()
    assert len(buckets) == 17
    assert {xp for xp, _, _ in buckets} == {16, 32, 64, 128, 256, 512, 1024}
    for xp, bh, nw in buckets:
        assert ct.sweep_bucket(bh, xp) == (xp, bh, nw)


@pytest.mark.parametrize("h,x", [(0, 20), (-1, 20), (1, 0), (1, -1), (1, 1025)])
def test_unsupported_bucket(h, x):
    with pytest.raises(ValueError):
        ct.sweep_bucket(h, x)


def test_incomplete_corrupt_or_missing_artifact_does_not_compile(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a cache miss must not initialize native code or compile")
    monkeypatch.setattr(ct, "_load_native", forbidden)
    ct._load_artifact.cache_clear()
    assert ct._load_artifact(tmp_path, 0) is None
    (tmp_path / "ready.json").write_text('{"sha256": {}}')
    ct._load_artifact.cache_clear()
    assert ct._load_artifact(tmp_path, 0) is None
    names = ("kernel.json", "kernel.cubin", "launcher.so", "cuda_utils.so")
    for name in names:
        (tmp_path / name).write_bytes(b"invalid")
    hashes = {name: hashlib.sha256(b"expected").hexdigest() for name in names}
    (tmp_path / "ready.json").write_text(json.dumps(dict(sha256=hashes)))
    ct._load_artifact.cache_clear()
    assert ct._load_artifact(tmp_path, 0) is None


def test_supported_requires_ready_variant_and_magnitude(monkeypatch):
    monkeypatch.setattr(ct, "HAS_TRITON", True)
    c = SimpleNamespace(device=SimpleNamespace(type="cuda"), dtype=torch.float64,
                        shape=(3, 5, 33), requires_grad=False)
    a = SimpleNamespace(dtype=torch.float64, requires_grad=False, shape=(3, 7, 33))
    e = SimpleNamespace(dtype=torch.float64)
    calls = []
    def lookup(device, h, x, magnitude):
        calls.append((h, x, magnitude))
        return object() if magnitude else None
    monkeypatch.setattr(ct, "_precompiled_sweep", lookup)
    assert not ct.fused_sweep_supported(c, a, e)
    assert ct.fused_sweep_supported(c, a, e, return_magnitude=True)
    assert calls == [(5, 33, False), (5, 33, True)]
    c.shape = (3, 5, 1025)
    assert not ct.fused_sweep_supported(c, a, e, return_magnitude=True)
    assert len(calls) == 2


def test_native_loader_abi_without_cuda(tmp_path, monkeypatch):
    metadata = dict(global_scratch_size=0, launch_cooperative_grid=False,
                    launch_pdl=False, num_ctas=1, shared=32, name="_sweep_kernel",
                    num_warps=2, cluster_dims=[1, 1, 1])
    (tmp_path / "kernel.json").write_text(json.dumps(metadata))
    for name in ("kernel.cubin", "launcher.so", "cuda_utils.so"):
        (tmp_path / name).write_bytes(b"native code")
    names = ("kernel.json", "kernel.cubin", "launcher.so", "cuda_utils.so")
    hashes = {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() for n in names}
    (tmp_path / "ready.json").write_text(json.dumps(dict(sha256=hashes)))
    calls = []
    def load_binary(name, cubin, shared, device):
        assert (name, cubin, shared, device) == ("_sweep_kernel", b"native code", 32, 2)
        return 123, 456, 10, 0, 128
    utils = SimpleNamespace(load_binary=load_binary)
    launcher = SimpleNamespace(launch=lambda *args: calls.append(args))
    monkeypatch.setattr(ct, "_load_native", lambda name, path:
                        utils if name == "cuda_utils" else launcher)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device:
                        SimpleNamespace(shared_memory_per_block_optin=1024))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=789))
    ct._load_artifact.cache_clear()
    kernel = ct._load_artifact(tmp_path, 2)
    assert kernel is not None and kernel.module == 123
    kernel((3, 5, 1), tuple(range(16)))
    assert calls == [(3, 5, 1, 789, 456, False, False, None,
                      (2, 1, 32, 1, 1, 1), None, None, None, *range(16))]
    ct._load_artifact.cache_clear()


def test_compile_signature_has_no_inferred_specializations():
    if not ct.HAS_TRITON:
        pytest.skip("Triton is optional")
    from tools.precompile_triton import sweep_source
    for bucket in ct.sweep_buckets():
        for magnitude in (False, True):
            source = sweep_source(bucket, magnitude)
            assert list(source.signature.values()) == ["*fp64"] * 6 + ["i32"] * 3 + ["i64"] * 7
            assert source.attrs == {}
            assert source.constants == {(16,): bucket[1], (17,): bucket[0],
                                        (18,): bucket[0].bit_length() - 1, (19,): magnitude}


def test_artifact_identity_rejects_stale_kernel_or_version(monkeypatch):
    if not ct.HAS_TRITON:
        pytest.skip("Triton is optional")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    identity = ct._artifact_identity(0)
    monkeypatch.setattr(ct, "_sweep_kernel", SimpleNamespace(src="different arithmetic"))
    assert ct._artifact_identity(0) != identity
    monkeypatch.setattr(ct.triton, "__version__", "unknown ABI")
    assert ct._artifact_identity(0) is None


def test_cpu_fused_request_uses_unchanged_fallback(monkeypatch):
    from auto_LiRPA import concretize_func as cf
    xl = torch.full((2, 5), -1., dtype=torch.float64)
    xu = -xl
    g = torch.Generator().manual_seed(1)
    objective = torch.randn(2, 3, 5, dtype=torch.float64, generator=g)
    a = torch.randn(2, 4, 5, dtype=torch.float64, generator=g)
    d = torch.full((2, 4), -0.5, dtype=torch.float64)
    processed, batches = cf.sort_out_constr_batches(xl, xu, (a, d), no_return_inf=True)
    def bounds():
        return cf.constraints_solving(xl, xu, objective, processed, -1.,
                                     sorted_out_batches=batches, no_return_inf=True,
                                     use_grad=False)
    monkeypatch.setattr(cf, "FUSED_SWEEP", False)
    ref = bounds()
    monkeypatch.setattr(cf, "FUSED_SWEEP", True)
    def forbidden(*args, **kwargs):
        pytest.fail("a CPU or cache-miss call must not launch the fused kernel")
    monkeypatch.setattr(ct, "fused_dual_sweep", forbidden)
    new = bounds()
    assert torch.equal(ref[0], new[0])
    assert ref[1] is new[1] is None
