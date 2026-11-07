"""Graph/box-only Young and affine-trigonometric root certificates.

Recognition and coefficient aggregation use exact rationals of binary graph
constants. Evaluation uses FP64 tensor intervals on the box's device. No LP,
reference bounds, adaptive atom points, primal points, or instance names enter.
"""
import math
import time
from collections import defaultdict
from fractions import Fraction as Q
import torch
from . import s12_interval as I


class Rejected(ValueError):
    pass


def const(q):
    return ('c', Q(q))


def plus(a, b):
    if a == const(0): return b
    if b == const(0): return a
    if a[0] == b[0] == 'c': return const(a[1]+b[1])
    return ('add', a, b)


def times(a, b):
    if a == const(0) or b == const(0): return const(0)
    if a == const(1): return b
    if b == const(1): return a
    if a[0] == b[0] == 'c': return const(a[1]*b[1])
    return ('mul', a, b)


def unary(op, a, p=None):
    # Exact exp(q*log(v)) identity, retaining the binary rational q.
    if op == 'exp':
        if a[0] == 'log': return ('pow', a[1], Q(1))
        if a[0] == 'mul':
            for c, l in ((a[1], a[2]), (a[2], a[1])):
                if c[0] == 'c' and l[0] == 'log': return ('pow', l[1], c[1])
    return (op, a, Q(p)) if op == 'pow' else (op, a)


def graph_forms(net, C):
    """Read the exact graph through _Fold; unknown operations reject, never probe."""
    import numpy as np
    from nlp_solver.polynomial_graph import _Fold, Aff, _ex, _MOVE, UNKNOWN

    class CheckedFold(_Fold):
        def _node(self, node):
            value = super()._node(node)
            if not (torch.is_tensor(value) and value.is_floating_point() and node.inputs):
                return value
            kind = type(node).__name__
            if kind in _MOVE or kind in ('BoundConstantOfShape', 'BoundWhere'):
                return value
            # _Fold caches input-independent computations as floats. A rounded
            # constant arithmetic expression is not an exact graph identity.
            # Accept its cached value only after a rational equality check;
            # unsupported constant transcendental computations stay unknown.
            args = [self.array(i) for i in node.inputs]
            if kind in ('BoundAdd', 'BoundSub', 'BoundMul', 'BoundDiv'):
                a, b = args[:2]
                try:
                    exact = {'BoundAdd': lambda: a+b, 'BoundSub': lambda: a-b,
                             'BoundMul': lambda: a*b, 'BoundDiv': lambda: a/b}[kind]()
                except ZeroDivisionError:
                    exact = None
            elif kind == 'BoundNeg': exact = -args[0]
            elif kind == 'BoundSqr': exact = args[0]*args[0]
            else: exact = None
            if exact is not None:
                exact = np.asarray(exact, dtype=object).reshape(-1)
                actual = value.reshape(-1).tolist()
                if len(exact) == len(actual) and all(
                        e.t == {} and math.isfinite(v) and e.k == _ex(v) for e, v in zip(exact, actual)):
                    return value
            return np.full(tuple(value.shape), UNKNOWN, dtype=object)

    f = CheckedFold(net)
    columns = {j: ('var', j) for j in range(f.n)}

    def expression(a):
        if a.t is None: raise Rejected('unknown affine operation')
        e = const(a.k)
        for j, q in sorted(a.t.items()):
            if j not in columns: raise Rejected('unknown nonlinear operation')
            e = plus(e, times(const(q), columns[j]))
        return e

    for j, (name, pos) in enumerate(f.cols[f.n:], f.n):
        node = net[name]
        if j in f.operands:
            a, b = f.operands[j]
            columns[j] = times(expression(a), expression(b))
            continue
        op = {'BoundLog': 'log', 'BoundExp': 'exp', 'BoundPow': 'pow',
              'BoundSin': 'sin', 'BoundCos': 'cos'}.get(type(node).__name__)
        if op is None: continue
        a = np.broadcast_to(f.array(node.inputs[0]), tuple(node.forward_value.shape)).reshape(-1)[pos]
        p = None
        if op == 'pow':
            exponent = f.array(node.inputs[1]).reshape(-1)
            ep = exponent[0 if len(exponent) == 1 else pos]
            if ep.t != {}: continue
            p = ep.k
        try:
            columns[j] = unary(op, expression(a), p)
        except Rejected:
            continue
    out = f.array(net[net.final_name]).reshape(-1)
    rows = []
    for row in C.detach().cpu().double():
        a = sum((out[k]*_ex(float(row[k])) for k in row.nonzero().flatten().tolist()), Aff({}, 0))
        rows.append(a)
    # Keep unsupported constraints as unknown; the certificates may safely
    # ignore them. An unsupported objective always rejects both recognisers.
    return f.n, expression(rows[0]), rows, columns


def dependencies(e):
    if e[0] == 'var': return {e[1]}
    d = set()
    for a in e[1:]:
        if isinstance(a, tuple): d |= dependencies(a)
    return d


def univariate(e):
    if e[0] == 'var': return ('x',)
    return tuple(univariate(a) if isinstance(a, tuple) else a for a in e)


def monomial(e):
    if e[0] == 'var': return {e[1]: Q(1)}
    if e[0] == 'pow': return {j: q*e[2] for j, q in monomial(e[1]).items()}
    if e[0] == 'mul':
        a, b = monomial(e[1]), monomial(e[2])
        for j, q in b.items(): a[j] = a.get(j, Q(0))+q
        return a
    raise Rejected('coupling is not a positive monomial')


def terms(e, q=Q(1)):
    if e[0] == 'add': return terms(e[1], q)+terms(e[2], q)
    if e[0] == 'mul':
        if e[1][0] == 'c': return terms(e[2], q*e[1][1])
        if e[2][0] == 'c': return terms(e[1], q*e[2][1])
    return [(q, e)] if q else []


def evaluate(e, x, cache=None):
    cache = {} if cache is None else cache
    if e in cache: return cache[e]
    op = e[0]
    zero, one = I.exact(0, x[0]), I.exact(1, x[0])
    if op == 'x': v = x, one
    elif op == 'c': v = I.exact(e[1], x[0]), zero
    elif op in ('add', 'mul'):
        a, da = evaluate(e[1], x, cache); b, db = evaluate(e[2], x, cache)
        v = (I.add(a, b), I.add(da, db)) if op == 'add' else (I.mul(a, b), I.add(I.mul(da, b), I.mul(a, db)))
    else:
        a, da = evaluate(e[1], x, cache)
        if op == 'log': v = I.log(a), I.div(da, a)
        elif op == 'exp':
            y = I.exp(a); v = y, I.mul(y, da)
        elif op == 'pow':
            p = e[2]
            v = I.power(a, p), I.mul(I.mul(I.exact(p, x[0]), I.power(a, p-1)), da)
        else: raise Rejected('unsupported scalar operation')
    cache[e] = v
    return v


class Young:
    kind = 'S1'

    def __init__(self, n, objective, lower):
        self.n, self.couplings = n, 0
        self.positive_inputs = set()
        coords = [const(0) for _ in range(n)]
        for q, e in terms(objective):
            dep = dependencies(e)
            if len(dep) <= 1:
                j = next(iter(dep), 0)
                coords[j] = plus(coords[j], times(const(q), univariate(e)))
                continue
            if q >= 0: raise Rejected('positive coupled term')
            powers = monomial(e)
            if any(p <= 0 or lower[j] <= 0 for j, p in powers.items()):
                raise Rejected('Young requires positive powers and inputs')
            p = sum(powers.values())
            self.couplings += 1
            self.positive_inputs.update(powers)
            for j, a in powers.items():
                coords[j] = plus(coords[j], times(const(q*a/p), ('pow', ('x',), p)))
        if not self.couplings: raise Rejected('no Young coupling')
        self.groups = defaultdict(list)
        for j, e in enumerate(coords): self.groups[e].append(j)
        # Reject scalar functions outside the interval evaluator's grammar.
        def check(e):
            if e[0] not in ('c', 'x', 'add', 'mul', 'pow', 'log', 'exp'):
                raise Rejected('unsupported scalar operation')
            for a in e[1:]:
                if isinstance(a, tuple): check(a)
        for e in self.groups: check(e)

    def bound(self, lower, upper):
        lower, upper = lower.double().flatten(1), upper.double().flatten(1)
        B = len(lower)
        answer = torch.zeros_like(lower)
        work = dict(round_cap=24, cell_cap=256, evaluations=0, peak_cells=0)
        for e, js in self.groups.items():
            l, u = lower[:, js].flatten(), upper[:, js].flatten()
            N = len(l); owner = torch.arange(N, device=l.device)
            best = torch.full_like(l, math.inf)
            for rd in range(25):
                work['evaluations'] += len(l)
                count = torch.bincount(owner, minlength=N)
                work['peak_cells'] = max(work['peak_cells'], int(count.max()))
                v, d = evaluate(e, (l, u))
                mono = (d[0] >= 0) | (d[1] <= 0)
                endpoint = torch.where(d[0] >= 0, l, u)
                ev = evaluate(e, (endpoint, endpoint))[0][0]
                # Half-sums can underflow outside a subnormal point box.
                # The Lipschitz argument requires the evaluation point inside.
                m = torch.maximum(l, torch.minimum(u, l/2+u/2))
                fm = evaluate(e, (m, m))[0]
                radius = I.up(torch.maximum(m-l, u-m))
                mag = torch.maximum(d[0].abs(), d[1].abs())
                lip = I.sub(fm, I.mul((mag, mag), (radius, radius)))[0]
                lb = torch.where(mono, ev, torch.maximum(v[0], lip))
                # Fail closed, including any domain invalid in the derivative.
                valid = torch.isfinite(lb) & torch.isfinite(d[0]) & torch.isfinite(d[1])
                lb = torch.where(valid, lb, -math.inf)
                finish = mono | ~valid | (count[owner]*2 > 256) | (rd == 24) | (m <= l) | (m >= u)
                best.scatter_reduce_(0, owner[finish], lb[finish], 'amin', include_self=True)
                active = ~finish
                if not bool(active.any()): break
                a, b, c, o = l[active], u[active], m[active], owner[active]
                l, u, owner = torch.cat([a, c]), torch.cat([c, b]), o.repeat(2)
            answer[:, js] = best.reshape(B, len(js))
        self.work = work
        value = I.isum((answer, answer))[0]
        valid = (lower[:, sorted(self.positive_inputs)] > 0).all(1)
        return torch.where(valid & torch.isfinite(value), value, -math.inf)


def affine(e):
    q, k = {}, Q(0)
    for c, t in terms(e):
        if t[0] == 'c': k += c*t[1]
        elif t[0] == 'var': q[t[1]] = q.get(t[1], Q(0))+c
        else: raise Rejected('trig argument is not affine in inputs')
    return {j: v for j, v in q.items() if v}, k


def key(q):
    a = sorted((j, v) for j, v in q.items() if v)
    if not a: return None, None
    scale = abs(a[0][1])
    return tuple((j, v/scale) for j, v in a), scale


def affine_interval(q, k, lower, upper):
    v = I.exact(k, lower[:, 0])
    for j, c in q.items(): v = I.add(v, I.mul(I.exact(c, lower[:, 0]), (lower[:, j], upper[:, j])))
    return v


def sine_support(m, a):
    """max(sin(t)-m*t): endpoints and first/last maximum stationary point.

    acos is a proposal only. Certified cos endpoint inequalities bracket it.
    The integer candidates include neighbours of both first/last proposals;
    completeness is checked using interval inequalities, otherwise -inf is
    returned by the caller. Large arguments are deliberately unsupported.
    """
    lo, hi = a
    mm = (m, m)
    theta = torch.acos(m)
    rad = torch.full_like(m, 2e-12)
    al, au = theta-rad, theta+rad
    cp, cq = I.trig((al, al), True), I.trig((au, au), True)
    valid = (m.abs() < 1) & (al > 0) & (au < math.pi) & (cp[0] >= m) & (cq[1] <= m)
    angle = al, au
    p = I.pi(m)
    two_pi = I.mul(I.exact(2, m), p)
    first = torch.floor((lo-theta)/(2*math.pi))
    last = torch.floor((hi-theta)/(2*math.pi))
    h = torch.maximum(I.sub(I.trig((lo, lo)), I.mul(mm, (lo, lo)))[1],
                      I.sub(I.trig((hi, hi)), I.mul(mm, (hi, hi)))[1])
    endpoints = h
    # Check each proposed integer is within one of the exact floor. This
    # includes boundary stationary points even when the floor proposal rounds.
    for x, k in ((lo, first), (hi, last)):
        left = I.add(angle, I.mul((k-1, k-1), two_pi))
        right = I.add(angle, I.mul((k+2, k+2), two_pi))
        valid &= (left[1] <= x) & (right[0] >= x) & (k.abs() <= 2**40)
    for base in (first, last):
        for offset in (-1, 0, 1, 2):
            k = base+offset
            t = I.add(angle, I.mul((k, k), two_pi))
            # Evaluating the full bracket is safe even if it overlaps a bound.
            v = I.sub(I.trig(t), I.mul(mm, t))[1]
            overlap = (t[1] >= lo) & (t[0] <= hi)
            h = torch.maximum(h, torch.where(overlap, v, -math.inf))
    # At +/-1 the function is monotone, so endpoints suffice.
    h = torch.where(m.abs() == 1, endpoints, torch.where(valid, h, math.inf))
    return torch.where(lo <= hi, h, math.inf)


class Trig:
    kind = 'S2'

    def __init__(self, n, objective, rows, thresholds):
        self.n, self.atoms, self.linear, self.constant = n, [], {}, Q(0)
        self.direction, self.offset = {}, Q(0)
        for c, e in terms(objective):
            if e[0] == 'c': self.constant += c*e[1]
            elif e[0] == 'var': self.linear[e[1]] = self.linear.get(e[1], Q(0))+c
            elif e[0] in ('sin', 'cos'):
                q, k = affine(e[1])
                if not q: raise Rejected('constant trig atom')
                self.atoms.append((c, q, k, e[0] == 'cos'))
                for j, v in q.items(): self.direction[j] = self.direction.get(j, Q(0))+c*v
                self.offset += c*k
            else: raise Rejected('objective is not an affine sum of trig atoms')
        if not self.atoms: raise Rejected('no affine trig atoms')
        self.direction = {j: v for j, v in self.direction.items() if v}
        self.rowmap = {}
        for row, t in zip(rows[1:], thresholds[1:]):
            if row.t is None or not math.isfinite(float(t)) or any(j >= n for j in row.t): continue
            kk, scale = key(row.t)
            if kk is None: continue
            b = (Q(float(t))-row.k)/scale
            self.rowmap[kk] = min(b, self.rowmap.get(kk, b))

    def bound(self, lower, upper):
        lower, upper = lower.double().flatten(1), upper.double().flatten(1)
        like = lower[:, 0]
        # Merge the exact signed weighted directions before taking box ranges.
        total = affine_interval({j: -v for j, v in self.direction.items()}, -self.offset, lower, upper)
        phases = I.exact(0, like)
        weight = sum(abs(c) for c, _, _, _ in self.atoms)
        for c, _, _, cosine in self.atoms:
            if cosine: phases = I.add(phases, I.mul(I.exact(-c/2, like), I.pi(like)))
        total = I.add(total, phases)
        mean = I.div(total, I.exact(weight, like))
        # The slope is arbitrary: proposal libm accuracy does not affect proof.
        m = torch.cos(mean[0]/2+mean[1]/2)
        # All atom supports are evaluated together, across atoms and boxes.
        ranges, weights = [], []
        matches = 0
        phase_weight = Q(0)
        for c, q, k, cosine in self.atoms:
            al, au = affine_interval(q, k, lower, upper)
            for sign in (1, -1):
                kk, scale = key({j: sign*v for j, v in q.items()})
                if kk in self.rowmap:
                    v = I.exact(sign*k+self.rowmap[kk]*scale, like)
                    if sign > 0: au = torch.minimum(au, v[1])
                    else: al = torch.maximum(al, -v[1])
                    matches += 1
            arg = al, au
            if cosine:
                arg = I.add(arg, I.mul(I.exact(Q(1, 2), like), I.pi(like)))
                phase_weight += c/2
            if c > 0: arg = I.neg(arg)
            ranges.append(arg)
            weights.append(I.exact(-abs(c), like))
        args = tuple(torch.stack([v[i] for v in ranges], -1) for i in (0, 1))
        ws = tuple(torch.stack([v[i] for v in weights], -1) for i in (0, 1))
        h = sine_support(m[:, None].expand_as(args[0]), args)
        offset = I.add(I.exact(self.offset, like), I.mul(I.exact(phase_weight, like), I.pi(like)))
        constant = I.add(I.exact(self.constant, like), I.mul((m, m), offset))
        constant = I.add(constant, I.isum(I.mul(ws, (h, h))))
        # c*sin(arg) >= c*m*arg - |c|*H, signed arguments above.
        value = constant
        for j in sorted(set(self.linear) | set(self.direction)):
            coeff = I.add(I.exact(self.linear.get(j, 0), like),
                          I.mul((m, m), I.exact(self.direction.get(j, 0), like)))
            value = I.add(value, I.mul(coeff, (lower[:, j], upper[:, j])))
        self.work = dict(atoms=len(self.atoms), direct_matches=matches, stationary_candidates=8)
        return torch.where(torch.isfinite(value[0]), value[0], -math.inf)


def recognise(net, C, thresholds, lower):
    try:
        n, objective, rows, _ = graph_forms(net, C)
    except (Rejected, ArithmeticError, RecursionError) as exc:
        return [], {'graph': str(exc)}
    certs, reasons = [], {}
    for cls in (Young, Trig):
        try:
            c = cls(n, objective, lower.detach().cpu().flatten().tolist()) if cls is Young else cls(n, objective, rows, thresholds.detach().cpu().flatten().tolist())
            certs.append(c)
        except (Rejected, ArithmeticError, RecursionError) as exc: reasons[cls.kind] = str(exc)
    return certs, reasons


def root_structural_addon(root, deadline):
    """One root call, alongside M1/M2; preserve GEAR by a coordinatewise max."""
    if deadline is None or time.time() >= deadline: return
    if not torch.equal(root.c, root.c[:1].expand_as(root.c)) or not torch.equal(
            root.rhs[:, 1:], root.rhs[:1, 1:].expand_as(root.rhs[:, 1:])):
        return
    thr = root.rowpool.thresholds(root.rhs[:1].double()).reshape(-1)
    started = time.perf_counter()
    certs, _ = recognise(root.net.net, root.c[0].double(), thr, root.dm_l[:1])
    if not certs: return
    for cert in certs:
        if time.time() >= deadline: break
        candidate = cert.bound(root.dm_l, root.dm_u)
        # A dtype conversion must never round a certified lower bound upward.
        candidate = I.down(candidate.to(root.global_lb))
        old = root.global_lb[:, 0].clone()
        wins = candidate > old
        root.global_lb = root.global_lb.clone()
        root.global_lb[:, 0] = torch.maximum(old, candidate)
        if hasattr(root, 'ret'):
            root.ret = dict(root.ret)
            root.ret['global_lb'] = root.global_lb
        if bool(wins.any()):
            # A scalar certificate is a constant support, not a raised bias
            # under the previous CROWN slope. Clipping may run if the root
            # remains open; update both the saved and live plane copies.
            if hasattr(root, 'ret'):
                root.ret['lA'] = {name: None if A is None else A.clone()
                                  for name, A in (root.ret.get('lA') or {}).items()}
                for A in root.ret['lA'].values():
                    if A is not None: A[wins, 0] = 0
            if getattr(root, 'lA', None) is not None:
                root.lA = root.lA.clone()
                root.lA[wins, 0] = 0
            if getattr(root, 'root_lbias', None) is not None:
                root.root_lbias = root.root_lbias.clone()
                root.root_lbias[wins, 0] = root.global_lb[wins, 0]
        print(f'[s12-root] kind={cert.kind} lower={candidate.tolist()} raised={int(wins.sum())} '
              f'seconds={time.perf_counter()-started:.6f} work={cert.work}', flush=True)
    return root.closes(root.global_lb[:, 0].amin())
