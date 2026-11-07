"""Opt-in outward unary faces for column OBBT (IEEE binary64 arithmetic).

Square and reciprocal use elementary inequalities. Real powers on x>0 use
log/exp interval series with explicit remainder bounds; libm pow is not trusted.
This does not modify the default CompiledGraph or any operator's search faces.
"""
import torch
from .compiled_crown import CompiledGraph


# Shared with S1/S2: every transcendental result has a series remainder.
from .outward_interval import (down, up, mul, add, neg, div, square,
                               _log_point as log_point, _exp_point as exp_point,
                               trig, pi, sqrt)


def real_power(base, exponent):
    """Positive base interval raised to a real exponent interval."""
    good = (base[0] > 0)&torch.isfinite(base[1])&torch.isfinite(exponent[0])&torch.isfinite(exponent[1])
    base = tuple(torch.where(good, x, 1.) for x in base)
    exponent = tuple(torch.where(good, x, 0.) for x in exponent)
    logb = log_point(base[0])[0], log_point(base[1])[1]
    a, b = mul(logb, exponent)
    return torch.where(good, exp_point(a)[0], 0.), torch.where(good, exp_point(b)[1], float('inf'))


def scalar_power(x, p):
    exponent = x.new_tensor(p)
    return real_power((x, x), (exponent, exponent))


def endpoint_residual(value, k, x):
    return add(value, neg(mul((k, k), (x, x))))


def tan_domain(lo, hi):
    # The period is an untrusted integer proposal; certify that the entire
    # translated interval is strictly between two certified pole bounds.
    period = pi(lo)
    mid = lo/2+hi/2
    n = torch.round(mid/((period[0]+period[1])/2))
    r = add((lo, hi), neg(mul((n, n), period)))
    half = mul((lo.new_tensor(.5),)*2, period)
    return (n.abs() <= 2**40)&(r[0] > -half[0])&(r[1] < half[0])


def unary_point(kind, x):
    if kind == 'BoundExp':
        return exp_point(x)
    if kind == 'BoundLog':
        return log_point(x)
    if kind == 'BoundSqrt':
        return sqrt((x, x))
    if kind == 'BoundAbs':
        v = x.abs()
        return v, v
    if kind == 'BoundTan':
        value = div(trig((x, x)), trig((x, x), cosine=True))
        good = tan_domain(x, x)
        return tuple(torch.where(good, v, torch.nan) for v in value)
    return trig((x, x), cosine=kind == 'BoundCos')


def unary_value(kind, a):
    """Certified image enclosure; invalid domains are NaN, never placeholders."""
    if kind in ('BoundCos', 'BoundSin'):
        return trig(a, cosine=kind == 'BoundCos')
    if kind == 'BoundAbs':
        return (torch.where((a[0] <= 0)&(a[1] >= 0), 0.,
                            torch.minimum(a[0].abs(), a[1].abs())),
                torch.maximum(a[0].abs(), a[1].abs()))
    value = unary_point(kind, torch.stack(a))
    l, u = value[0][0], value[1][1]
    if kind == 'BoundTan':
        good = tan_domain(*a)
        l, u = torch.where(good, l, torch.nan), torch.where(good, u, torch.nan)
    return l, u


def trig_support(kind, k, lo, hi, lower, upper):
    """Endpoints plus the first/last root in each periodic stationary family.

    cos(alpha)=k, alpha in [0,pi]. acos is ONLY an untrusted proposal: two
    certified cosine inequalities bracket the root, otherwise use [0,pi].
    A sqrt(eps)-scale bracket lets a mean-value residual enclosure recover
    O(eps) accuracy, without assuming anything about libm's acos accuracy.
    First/last indices use interval division by certified 2*pi. Extra roots
    are safe; no loop depends on interval width or the number of periods.
    Large arguments fall back to [-1,1]-k*[lo,hi].
    """
    k, lo, hi = torch.broadcast_tensors(k, lo, hi)
    take = k.abs() < 1
    kk = k.clamp(-1, 1)
    pp = pi(k)
    proposal = torch.acos(kk)
    # This is a search radius, not an accuracy assertion about acos.
    radius = (1+proposal.abs())*torch.finfo(k.dtype).eps**0.5
    al = down(proposal-radius).clamp_min(0)
    au = torch.minimum(up(proposal+radius), pp[1])
    c = trig((torch.stack((al, au)), torch.stack((al, au))), cosine=True)
    certified = (torch.isfinite(proposal)&(al <= au)&(al <= pp[0])
                 &(c[0][0] >= kk)&(c[1][1] <= kk))
    al = torch.where(certified, al, 0.)
    au = torch.where(certified, au, pp[1])
    alpha = (torch.stack((al, -au)), torch.stack((au, -al)))
    if kind == 'BoundCos':
        half_pi = mul((k.new_tensor(.5),)*2, pp)
        alpha = add(alpha, neg(half_pi))
    period = mul((k.new_tensor(2.),)*2, pp)
    qlo = div(add((lo, lo), neg(alpha)), period)
    qhi = div(add((hi, hi), neg(alpha)), period)
    first, last = torch.ceil(qlo[0]), torch.floor(qhi[1])
    # Above 2**40, pi uncertainty and integer arithmetic are not used.
    good = (first.abs() <= 2**40)&(last.abs() <= 2**40)
    ns = torch.stack((first, last))
    ns = torch.where(good[None], ns, 0.)
    roots = add(alpha, mul((ns, ns), period))
    centre = roots[0]/2+roots[1]/2
    rad = up(torch.maximum(up(centre-roots[0]), up(roots[1]-centre)))
    v = endpoint_residual(unary_point(kind, centre), k, centre)
    derivative = trig(roots, cosine=kind == 'BoundSin')
    if kind == 'BoundCos':
        derivative = neg(derivative)
    d = add(derivative, (-k, -k))
    error = up(torch.maximum(d[0].abs(), d[1].abs())*rad)
    v = add(v, (-error, error))
    has_root = (first <= last)&take
    low = torch.where(has_root[None], v[0], float('inf')).amin((0, 1))
    high = torch.where(has_root[None], v[1], -float('inf')).amax((0, 1))
    lower, upper = torch.minimum(lower, low), torch.maximum(upper, high)
    fallback = endpoint_residual((-torch.ones_like(lo), torch.ones_like(hi)),
                                 k, torch.where(k >= 0, hi, lo))[0]
    fallback_hi = endpoint_residual((-torch.ones_like(lo), torch.ones_like(hi)),
                                    k, torch.where(k >= 0, lo, hi))[1]
    usable = good.all(0) | ~take
    return torch.where(usable, lower, fallback), torch.where(usable, upper, fallback_hi)


def elementary_support(kind, k, lo, hi):
    value = unary_point(kind, torch.stack((lo, hi)))
    vl = endpoint_residual((value[0][0], value[1][0]), k, lo)
    vu = endpoint_residual((value[0][1], value[1][1]), k, hi)
    lower, upper = torch.minimum(vl[0], vu[0]), torch.maximum(vl[1], vu[1])
    if kind == 'BoundTan':
        # tan is increasing between poles. Its image minus k*[lo,hi] is
        # always a support; where sec^2-k has constant sign, endpoints are
        # exact extrema. This coarse stationary enclosure avoids any atan
        # accuracy assumption, including arbitrarily close to a pole.
        good = tan_domain(lo, hi)
        image = (torch.where(good, value[0][0], torch.nan),
                 torch.where(good, value[1][1], torch.nan))
        natural = add(image, neg(mul((k, k), (lo, hi))))
        derivative = add(square(image), (torch.ones_like(k),)*2)
        monotone = (derivative[0] >= k)|(derivative[1] <= k)
        return (torch.where(monotone, lower, natural[0]),
                torch.where(monotone, upper, natural[1]))
    if kind in ('BoundCos', 'BoundSin'):
        return trig_support(kind, k, lo, hi, lower, upper)
    if kind == 'BoundAbs':
        # A piecewise affine convex residual: endpoints and the kink.
        lower = torch.where((lo <= 0)&(hi >= 0), torch.minimum(lower, k*0), lower)
    else:
        take = k > 0
        safe = torch.where(take, k, 1.)
        if kind == 'BoundExp':
            root = log_point(safe)
            intercept = mul((safe, safe), add((torch.ones_like(k),)*2, neg(root)))
        elif kind == 'BoundLog':
            root = down(1/safe), up(1/safe)
            intercept = add(neg(log_point(safe)), (-torch.ones_like(k),)*2)
        else:  # sqrt: x*=1/(4*k^2), sup(sqrt(x)-kx)=1/(4*k)
            half = down(.5/safe).clamp_min(0), up(.5/safe)
            root = square(half)
            intercept = down(.25/safe), up(.25/safe)
        take = take & (root[1] >= lo) & (root[0] <= hi)
        if kind == 'BoundExp':
            lower = torch.where(take, torch.minimum(lower, intercept[0]), lower)
        else:
            upper = torch.where(take, torch.maximum(upper, intercept[1]), upper)
    return lower, upper


def support(kind, k, lo, hi, exponent=None):
    """Enclose inf and sup of phi(x)-k*x over the operand interval.

    Global tangent intercepts are sufficient lower/upper supports even when
    their stationary point lies outside the interval. No approximate root is
    treated as exact. The finite endpoint hull supplies the opposite side.
    """
    if kind in ('BoundCos', 'BoundSin', 'BoundLog', 'BoundSqrt', 'BoundAbs', 'BoundExp', 'BoundTan'):
        return elementary_support(kind, k, lo, hi)
    if kind == 'BoundSqr':
        fl, fu = (down(lo*lo), up(lo*lo)), (down(hi*hi), up(hi*hi))
        vl, vu = endpoint_residual(fl, k, lo), endpoint_residual(fu, k, hi)
        tangent = -up(up(k*k)/4)
        return tangent, torch.maximum(vl[1], vu[1])
    if kind == 'BoundReciprocal':
        fl, fu = (down(1/lo), up(1/lo)), (down(1/hi), up(1/hi))
        vl, vu = endpoint_residual(fl, k, lo), endpoint_residual(fu, k, hi)
        # IEEE sqrt is correctly rounded; 1/x+a*x >=2 sqrt(a) on x>0.
        root = torch.sqrt((-k).clamp(min=0))
        small, large = down(2*down(root)), up(2*up(root))
        lower = torch.minimum(vl[0], vu[0])
        upper = torch.maximum(vl[1], vu[1])
        lower = torch.where((lo > 0)&(k < 0), small, lower)
        upper = torch.where((hi < 0)&(k < 0), -small, upper)
        return lower, upper
    if kind != 'BoundPow':
        raise NotImplementedError('strict OBBT unary faces: '+kind)
    p = float(exponent)
    fl, fu = scalar_power(lo, p), scalar_power(hi, p)
    vl, vu = endpoint_residual(fl, k, lo), endpoint_residual(fu, k, hi)
    lower, upper = torch.minimum(vl[0], vu[0]), torch.maximum(vl[1], vu[1])
    if p in (0., 1.):
        return lower, upper
    # For slope k with the derivative's sign, stationary intercept is
    # (1-p)*(k/p)^(p/(p-1)); compute both exponent and base as intervals.
    take = (k < 0) if p < 0 else (k > 0)
    pp = k.new_tensor(p)
    pm1 = down(pp-1), up(pp-1)
    q = div((pp, pp), pm1)
    safe_k = torch.where(take, k, pp)
    base = div((safe_k, safe_k), (pp, pp))
    intercept = mul((down(1-pp), up(1-pp)), real_power(base, q))
    if p < 0 or p > 1:  # convex: stationary global minimum
        lower = torch.where(take, intercept[0], lower)
    else:  # 0<p<1: concave: stationary global maximum
        upper = torch.where(take, intercept[1], upper)
    return lower, upper


class OutwardCompiledGraph(CompiledGraph):
    """Only for the opt-in OBBT prototype; unsupported unary kinds fail closed."""
    def capture_safe(self):
        """Whether every reachable face/IBP proposal has fixed-shape device code.

        This is deliberately an allowlist: new operators must be audited and
        tested under capture before opting in. Domain checks of these operators
        are deferred to the caller's device flag, checked outside the graph.
        Sin/Cos and odd powers use S-shaped proposals with host decisions;
        Log/Sqrt/Tan also branch on their input domain in operator code.
        """
        return all(type(self.net[nd.name]).__name__ in
                   ('BoundSqr', 'BoundReciprocal', 'BoundMul', 'BoundExp', 'BoundAbs')
                   for nd in self.nodes)

    def _node_faces(self, nd, lo, hi):
        if lo.dtype != torch.float64:
            raise ValueError('outward unary construction requires binary64')
        kind = type(self.net[nd.name]).__name__
        if kind == 'BoundMul':
            return super()._node_faces(nd, lo, hi)  # enclosed by _mul_valid
        if kind not in ('BoundSqr','BoundReciprocal','BoundPow',
                        'BoundCos','BoundSin','BoundLog','BoundSqrt','BoundAbs','BoundExp','BoundTan'):
            raise NotImplementedError('strict OBBT unary faces: '+kind)
        a, b = lo[:, 0], hi[:, 0]
        good = torch.isfinite(a)&torch.isfinite(b)&(a <= b)
        if kind == 'BoundReciprocal': good &= (a > 0)|(b < 0)
        if kind in ('BoundPow', 'BoundLog'): good &= a > 0
        if kind == 'BoundSqrt': good &= a >= 0
        if kind == 'BoundTan': good &= tan_domain(a, b)
        # Reject domains BEFORE asking the search operator for proposals.
        # During capture the flag is checked before any table can be used;
        # raw invalid bounds are never replaced by a finite surrogate.
        invalid = (~good).reshape(len(lo),-1).any(1)
        if hasattr(self, '_capture_invalid'):
            self._capture_invalid.logical_or_(invalid)
        elif bool(invalid.any()):
            raise ValueError('strict OBBT unary domain is unsupported or undefined')
        faces, box, undefined = super()._node_faces(nd, lo, hi)
        if hasattr(self, '_capture_invalid'):
            self._capture_invalid.logical_or_(undefined)
        elif bool(undefined.any()):
            raise ValueError('strict OBBT unary domain is unsupported or undefined')
        lw, lb, uw, ub, dl, du = faces
        # Slopes are proposals, too. Nonfinite operator slopes (e.g. sqrt at
        # zero) become constant faces whose intercepts are certified below.
        lw = torch.where(torch.isfinite(lw), lw, 0.)
        uw = torch.where(torch.isfinite(uw), uw, 0.)
        if hasattr(self, '_unary_slopes'):
            lw,uw=self._unary_slopes(nd,lo,hi,lw,uw)
        p = getattr(self.net[nd.name], 'exponent', None)
        lower = support(kind, lw[:, :, 0], a, b, p)[0]
        upper = support(kind, uw[:, :, 0], a, b, p)[1]
        lb, ub = lower, upper
        if kind == 'BoundSqr':
            l = torch.where((a <= 0)&(b >= 0), 0., down(torch.minimum(a*a,b*b))).clamp(min=0)
            u = up(torch.maximum(a*a,b*b))
        elif kind == 'BoundReciprocal':
            l, u = down(1/b), up(1/a)
        elif kind != 'BoundPow':
            l, u = unary_value(kind, (a, b))
        else:
            pa, pb = scalar_power(a, float(p)), scalar_power(b, float(p))
            l, u = torch.minimum(pa[0],pb[0]), torch.maximum(pa[1],pb[1])
        # Failed transcendental division/reduction must reject the table as
        # undefined, including a tan endpoint too close to certify its cosine.
        failed = (torch.isnan(l)|torch.isnan(u)).reshape(len(lo), -1).any(1)
        for intercept in (lb, ub):
            failed |= torch.isnan(intercept).movedim(1, 0).reshape(len(lo), -1).any(1)
        undefined = undefined | failed
        if hasattr(self, '_capture_invalid'):
            self._capture_invalid.logical_or_(failed)
        elif bool(failed.any()):
            raise ValueError('strict OBBT unary domain is unsupported or undefined')
        return (lw,lb,uw,ub,dl,du), (l,u), undefined
