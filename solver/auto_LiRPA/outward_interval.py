"""FP64 tensor intervals, including elementary functions without libm assumptions.

Each basic IEEE operation is widened by nextafter (also covering subnormals).
No compiler fusion is used. Log uses the atanh series after exact frexp reduction;
exp and sin/cos use Taylor polynomials with explicit absolute remainder bounds.
Invalid domains produce NaNs and the caller rejects the certificate.
"""
import math
from fractions import Fraction
import torch


def down(x):
    return torch.nextafter(x, torch.full_like(x, -math.inf))


def up(x):
    return torch.nextafter(x, torch.full_like(x, math.inf))


def exact(q, like):
    q = Fraction(q)
    try:
        v = float(q)
    except OverflowError:
        # Exact affine folding can exceed binary64's representable range.
        # An unbounded enclosure makes the caller decline this certificate.
        return torch.full_like(like, -math.inf), torch.full_like(like, math.inf)
    t = torch.full_like(like, v)
    return (down(t) if Fraction(v) > q else t,
            up(t) if Fraction(v) < q else t)


def neg(a):
    return -a[1], -a[0]


def add(a, b):
    return down(a[0]+b[0]), up(a[1]+b[1])


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    p = torch.stack([a[i]*b[j] for i in (0, 1) for j in (0, 1)])
    return down(p.amin(0)), up(p.amax(0))


def div(a, b):
    valid = (b[0] > 0) | (b[1] < 0)
    r = down(1/b[1]), up(1/b[0])
    v = mul(a, r)
    return tuple(torch.where(valid, x, torch.nan) for x in v)


def square(a):
    l, u = a
    return (torch.where((l <= 0) & (u >= 0), 0., down(torch.minimum(l*l, u*u))),
            up(torch.maximum(l*l, u*u)))


def isum(a, dim=-1):
    # Pairwise addition, with outward rounding at every level; no reduction
    # accuracy assumption and no dependence on CUDA atomic summation order.
    l, u = (v.movedim(dim, -1) for v in a)
    while l.shape[-1] > 1:
        if l.shape[-1] % 2:
            l = torch.cat([l, torch.zeros_like(l[..., :1])], -1)
            u = torch.cat([u, torch.zeros_like(u[..., :1])], -1)
        l, u = down(l[..., ::2]+l[..., 1::2]), up(u[..., ::2]+u[..., 1::2])
    return l.squeeze(-1), u.squeeze(-1)


def _log_unit(x):
    # 1 <= x <= 2, z in [0,1/3]. After 24 terms the tail is bounded by
    # 2*(1/3)^49/(49*(1-1/9)) < 2e-25.
    one = exact(1, x)
    z = div(sub((x, x), one), add((x, x), one))
    z2 = square(z)
    term, total = z, z
    for k in range(1, 24):
        term = mul(term, z2)
        total = add(total, div(term, exact(2*k+1, x)))
    v = mul(exact(2, x), total)
    return down(v[0]-2e-25), up(v[1]+2e-25)


_LN2 = {}


def _ln2(like):
    key = (like.device, like.dtype)
    if key not in _LN2:
        _LN2[key] = _log_unit(like.new_tensor(2.))
    return _LN2[key]


def _log_point(x):
    m, k = torch.frexp(x)
    m, k = m*2, k-1  # exact for every positive binary64 input
    v = add(_log_unit(m), mul((k.double(), k.double()), _ln2(x)))
    return tuple(torch.where((x > 0) & torch.isfinite(x), y, torch.nan) for y in v)


def log(a):
    return _log_point(a[0])[0], _log_point(a[1])[1]


def _exp_point(x):
    """Positive Taylor series, 32 terms; geometric bound on the tail.

    Outside [-708,708], monotonicity gives a finite inner endpoint and an
    unbounded outer endpoint. No libm exp or underflow accuracy is assumed.
    """
    clipped = x.clamp(min=-708., max=708.)
    l2 = _ln2(x)
    k = torch.floor(clipped/((l2[0]+l2[1])/2))-1
    rlo, rhi = sub((clipped, clipped), mul((k, k), l2))
    valid = (rlo >= 0) & (rhi < 2)
    rlo = torch.where(valid, rlo, 0.)
    rhi = torch.where(valid, rhi, 1.)
    tlo = thi = torch.ones_like(x)
    slo = shi = torch.ones_like(x)
    for j in range(1, 33):
        tlo = down(down(tlo*rlo)/j).clamp(min=0)
        thi = up(up(thi*rhi)/j)
        slo = down(slo+tlo)
        shi = up(shi+thi)
    # t_33/(1-rhi/34) bounds every remaining positive term.
    next_hi = up(up(thi*rhi)/33)
    tail = up(next_hi/down(1-up(rhi/34)))
    lower = down(torch.ldexp(slo, k.to(torch.int32))).clamp(min=0)
    upper = up(torch.ldexp(up(shi+tail), k.to(torch.int32)))
    return (torch.where(torch.isnan(x), torch.nan, torch.where(valid & (x >= -708), lower, 0.)),
            torch.where(torch.isnan(x), torch.nan, torch.where(valid & (x <= 708), upper, math.inf)))


def sqrt(a):
    # IEEE binary64 sqrt is correctly rounded (unlike libm transcendentals).
    l = torch.sqrt(a[0])
    u = l if a[0] is a[1] else torch.sqrt(a[1])
    return down(l).clamp_min(0), up(u)


def exp(a):
    return _exp_point(a[0])[0], _exp_point(a[1])[1]


# Decimal bounds are rational and bracket pi; binary conversions are enclosed.
PI_LO = Fraction('3.14159265358979323846264338327950288419716939937510')
PI_HI = Fraction('3.14159265358979323846264338327950288419716939937511')


def pi(like):
    return exact(PI_LO, like)[0], exact(PI_HI, like)[1]


def trig(a, cosine=False):
    # Point/range evaluation after an arbitrary integral 2*pi translation.
    # Taylor's Lagrange remainder for degree 49/48 at |r|<=4 is < 1e-32.
    mid = a[0]/2+a[1]/2
    k = torch.round(mid/(2*math.pi))
    r = sub(a, mul((2*k, 2*k), pi(mid)))
    r2 = square(r)
    term = exact(1, mid) if cosine else r
    total = term
    for j in range(1, 25):
        d = (2*j-1)*(2*j) if cosine else (2*j)*(2*j+1)
        term = neg(div(mul(term, r2), exact(d, mid)))
        total = add(total, term)
    l, u = down(total[0]-1e-32), up(total[1]+1e-32)
    # A wide argument has the trivial interval; no inexact range reduction.
    ok = (r[0] >= -4) & (r[1] <= 4) & (k.abs() <= 2**40)
    return torch.where(ok, l.clamp_min(-1), -1.), torch.where(ok, u.clamp_max(1), 1.)


def power(a, p):
    p = Fraction(p)
    if p == 0:
        return exact(1, a[0])
    if p == 1:
        return a
    if p == 2:
        return square(a)
    return exp(mul(exact(p, a[0]), log(a)))
