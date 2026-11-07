"""The fused dual-sweep kernel (concretize_triton.py) against the PyTorch sweep."""
import pytest
import torch

import auto_LiRPA.concretize_func as cf
from auto_LiRPA.concretize_triton import HAS_TRITON

pytestmark = pytest.mark.skipif(not (HAS_TRITON and torch.cuda.is_available()),
                                reason="needs triton and CUDA")
DEV = "cuda"
DT = torch.float64


def _problem(B, H, X, NC, seed, zero_frac=0.0, ties=False, redundant_rows=0,
             infeasible_batch=None):
    g = torch.Generator(device=DEV).manual_seed(seed)
    xl = torch.rand(B, X, device=DEV, dtype=DT, generator=g) - 1
    xu = xl + torch.rand(B, X, device=DEV, dtype=DT, generator=g) * 0.5 + 0.01
    obj = torch.randn(B, H, X, device=DEV, dtype=DT, generator=g)
    CA = torch.randn(B, NC, X, device=DEV, dtype=DT, generator=g)
    if zero_frac > 0:
        CA = CA * (torch.rand(B, NC, X, device=DEV, dtype=DT, generator=g) > zero_frac)
        obj = obj * (torch.rand(B, H, X, device=DEV, dtype=DT, generator=g) > zero_frac)
    if ties:
        # Integer data: many identical breakpoints -c_i / a_i.
        CA = torch.randint(-2, 3, (B, NC, X), device=DEV, generator=g).to(DT)
        obj = torch.randint(-3, 4, (B, H, X), device=DEV, generator=g).to(DT)
        xl = torch.zeros(B, X, device=DEV, dtype=DT)
        xu = torch.ones(B, X, device=DEV, dtype=DT)
    # Rows vanishing at a random point of the box, so that they bind.
    xp = xl + torch.rand(B, X, device=DEV, dtype=DT, generator=g) * (xu - xl)
    Cb = -torch.einsum("bkx,bx->bk", CA, xp) + 0.1 * torch.randn(B, NC, device=DEV, dtype=DT, generator=g)
    if redundant_rows:
        Cb[:, :redundant_rows] = -1e3
    if infeasible_batch is not None:
        Cb[infeasible_batch, -1] = 1e3
    return xl, xu, obj, (CA, Cb)


def _bounds(xl, xu, obj, cons, sign, fused, max_rows=16):
    cf.FUSED_SWEEP = fused
    cf.MAX_CONSTRAINT_ROWS = max_rows
    try:
        processed, sob = cf.sort_out_constr_batches(xl, xu, cons, no_return_inf=True)
        return cf.constraints_solving(xl, xu, obj, processed, sign, sorted_out_batches=sob,
                                      no_return_inf=True, use_grad=False)
    finally:
        cf.FUSED_SWEEP = False
        cf.MAX_CONSTRAINT_ROWS = None


@pytest.mark.parametrize("B,H,X,NC,kw", [
    (64, 36, 55, 40, {}),
    (33, 7, 26, 57, {}),                     # H, X not powers of two
    (8, 3, 5, 4, {}),                        # tiny rows, padded to 16
    (16, 20, 130, 12, {}),                   # X > 128
    (40, 10, 20, 20, {"zero_frac": 0.4}),    # a_i = 0 and c_i = 0 entries
    (40, 10, 20, 20, {"ties": True}),        # repeated breakpoints
    (40, 10, 20, 20, {"redundant_rows": 5}),
    (40, 10, 20, 20, {"infeasible_batch": 3}),
])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_fused_matches_pytorch_sweep(B, H, X, NC, kw, sign):
    xl, xu, obj, cons = _problem(B, H, X, NC, seed=B * 1000 + X, **kw)
    ref, inf_ref = _bounds(xl, xu, obj, cons, sign, fused=False)
    new, inf_new = _bounds(xl, xu, obj, cons, sign, fused=True)
    assert (inf_ref is None) == (inf_new is None)
    if inf_ref is not None:
        assert torch.equal(inf_ref, inf_new)
    fin = torch.isfinite(ref)
    assert torch.equal(fin, torch.isfinite(new))
    scale = 1.0 + ref[fin].abs()
    assert ((ref[fin] - new[fin]).abs() <= 1e-9 * scale).all(), \
        float(((ref[fin] - new[fin]).abs() / scale).max())


def test_kernel_failure_falls_back_to_pytorch_sweep(monkeypatch):
    import auto_LiRPA.concretize_triton as ct
    xl, xu, obj, cons = _problem(16, 5, 12, 6, seed=3)
    ref, _ = _bounds(xl, xu, obj, cons, -1.0, fused=False)

    def broken(*a, **k):
        raise RuntimeError("no kernel for this shape")
    monkeypatch.setattr(ct, "fused_dual_sweep", broken)
    with pytest.warns(UserWarning, match="Fused constrained-concretization kernel failed"):
        new, _ = _bounds(xl, xu, obj, cons, -1.0, fused=True)
    assert torch.equal(ref, new)


def test_fused_bound_is_valid_on_sampled_feasible_points():
    # Independent check: the bound never exceeds the objective at feasible points.
    B, H, X, NC = 16, 5, 6, 3
    xl, xu, obj, cons = _problem(B, H, X, NC, seed=7)
    lb, _ = _bounds(xl, xu, obj, cons, -1.0, fused=True, max_rows=None)
    CA, Cb = cons
    g = torch.Generator(device=DEV).manual_seed(1)
    pts = xl[:, None] + torch.rand(B, 20000, X, device=DEV, dtype=DT, generator=g) * (xu - xl)[:, None]
    feas = (torch.einsum("bkx,bnx->bnk", CA, pts) + Cb[:, None] <= 0).all(-1)
    vals = torch.einsum("bhx,bnx->bhn", obj, pts)
    vals = torch.where(feas[:, None], vals, torch.full_like(vals, float("inf")))
    assert (lb.squeeze(-1) <= vals.min(-1).values + 1e-9).all()
