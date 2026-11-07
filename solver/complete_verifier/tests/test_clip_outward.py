"""TH3: clip.py rounds outward. No exactly feasible point is removed (Fraction fuzz), and the clipped volume
is within 1e-12 (relative) of the plain floating-point clip. Run: python -m pytest tests/test_clip_outward.py"""
import os
import sys
from fractions import Fraction as Fr

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from input_split.clip import clip_domains, concretize_bounds  # noqa: E402

F64 = torch.float64


def plain_clip(x_L, x_U, lA, lbias, thr):
    """The previous single-iteration clip (no margins, no directed rounding), num_spec rows, is_lower."""
    xhat, eps = (x_U + x_L) / 2, (x_U - x_L) / 2
    dm = concretize_bounds(xhat, eps, lA, lbias, True).unsqueeze(-1)
    cm1 = dm - lA * xhat.unsqueeze(1) + lA.abs() * eps.unsqueeze(1)
    cx = (thr.unsqueeze(-1) - cm1) / lA
    xu = torch.where(lA > 0, cx, torch.inf).amin(1)
    xl = torch.where(lA < 0, cx, -torch.inf).amax(1)
    return torch.max(xl, x_L), torch.min(xu, x_U)


def exact_min(lA, lbias, lo, hi, thr_unused=None):
    v = Fr(float(lbias))
    for a, l, u in zip(lA, lo, hi):
        a, l, u = Fr(float(a)), Fr(float(l)), Fr(float(u))
        v += min(a * l, a * u)
    return v


def test_no_exactly_feasible_point_removed_and_volume_close():
    torch.manual_seed(0)
    bad = removed_nonempty = 0
    worst_rel = 0.0
    for trial in range(300):
        n = int(torch.randint(1, 5, ()))
        scale = 10 ** float(torch.randint(-3, 4, ()))
        lo = torch.randn(1, n, dtype=F64) - 1
        hi = lo + torch.rand(1, n, dtype=F64) * 2 + 0.1
        lA = (torch.randn(1, 1, n, dtype=F64) * scale)
        lbias = torch.randn(1, 1, dtype=F64) * scale
        mid = exact_min(lA[0, 0], lbias[0, 0], lo[0], hi[0])
        top = -exact_min(-lA[0, 0], -lbias[0, 0], lo[0], hi[0])
        thr = torch.tensor([[float(mid + (top - mid) * Fr(int(torch.randint(5, 95, ())), 100))]], dtype=F64)
        new_L, new_U = clip_domains(lo.clone(), hi.clone(), thr, lA, lbias, num_iters=1)
        new_L, new_U = new_L.reshape(1, n), new_U.reshape(1, n)
        if (new_L > new_U).any():          # emptied: must be exactly infeasible
            if Fr(float(thr[0, 0])) >= mid:
                bad += 1
            continue
        # points exactly on the plane = thr (boundary of the feasible set) and strictly inside it
        for j in range(n):
            if lA[0, 0, j] == 0:
                continue
            for _ in range(5):
                x = [Fr(float(lo[0, k])) + (Fr(float(hi[0, k])) - Fr(float(lo[0, k]))) * Fr(int(torch.randint(0, 1001, ())), 1000)
                     for k in range(n)]
                rest = Fr(float(lbias[0, 0])) + sum(Fr(float(lA[0, 0, k])) * x[k] for k in range(n) if k != j)
                x[j] = (Fr(float(thr[0, 0])) - rest) / Fr(float(lA[0, 0, j]))
                if not (Fr(float(lo[0, j])) <= x[j] <= Fr(float(hi[0, j]))):
                    continue
                if not all(Fr(float(new_L[0, k])) <= x[k] <= Fr(float(new_U[0, k])) for k in range(n)):
                    bad += 1
        ref_L, ref_U = plain_clip(lo, hi, lA, lbias, thr)
        wn, wo = (new_U - new_L).clamp_min(0), (ref_U - ref_L).clamp_min(0)
        if (wo > 0).all():
            rel = abs(float(torch.log(wn).sum() - torch.log(wo).sum()))
            worst_rel = max(worst_rel, rel)
    assert bad == 0
    assert worst_rel < 1e-12, worst_rel


def test_certified_mask_needs_margin():
    # lb is the plane's min over the box rounded to nearest; a threshold one ulp below it is not certified
    lo, hi = torch.tensor([[0.0, 0.0]], dtype=F64), torch.tensor([[1.0, 1.0]], dtype=F64)
    lA = torch.tensor([[[1.0, 1.0]]], dtype=F64)
    lbias = torch.tensor([[0.1]], dtype=F64)
    lb = concretize_bounds((hi + lo) / 2, (hi - lo) / 2, lA, lbias, True)
    thr = torch.nextafter(lb, torch.full_like(lb, -torch.inf))
    new_L, new_U = clip_domains(lo.clone(), hi.clone(), thr, lA, lbias, num_iters=1)
    assert bool((new_L <= new_U).all()), "domain at rounding distance from the threshold must not be emptied"
    new_L, new_U = clip_domains(lo.clone(), hi.clone(), lb - 1e-6, lA, lbias, num_iters=1)
    assert bool((new_L > new_U).any())
