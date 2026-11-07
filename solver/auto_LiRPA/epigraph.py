"""Fixed COO slots for graph-derived convex objective chains and Kelley rows.

Recognition uses exact affine graph forms, never an LP range or problem name.
Each epigraph retains its original atom face column as a lower support. All
additional cuts enclose value, derivative, affine-fold and product rounding.
Only data changes between solves; the primal/dual reset graph is reused.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
import time
import torch
from .obbt_unary import add, mul, div, neg, down, up, exp_point, log_point, real_power


def enclosure(q):
    v = float(q)
    return (v if Fraction(v) <= q else math.nextafter(v, -math.inf),
            v if Fraction(v) >= q else math.nextafter(v, math.inf))


def square(v):
    a, b = v
    return (torch.where((a <= 0) & (b >= 0), 0., down(torch.minimum(a*a, b*b))),
            up(torch.maximum(a*a, b*b)))


def evaluate(chain, interval):
    """Interval AD of the exact unary chain, including its rational constants."""
    v = interval; one = torch.ones_like(v[0]); zero = torch.zeros_like(one)
    first, second = (one, one), (zero, zero)
    good = torch.isfinite(v[0]) & torch.isfinite(v[1])
    power, positive_power_chain = Fraction(1), True
    for kind, arg, shift in chain:
        if kind == 'affine':
            aa = tuple(one*x for x in enclosure(arg)); bb = tuple(one*x for x in enclosure(shift))
            v, first, second = add(mul(v, aa), bb), mul(first, aa), mul(second, aa)
            positive_power_chain &= arg > 0 and shift == 0
            continue
        if kind == 'exp':
            result = (exp_point(v[0])[0], exp_point(v[1])[1]); df = dd = result
            positive_power_chain = False
        elif kind == 'log':
            good &= v[0] > 0
            v = tuple(torch.where(good, x, 1.) for x in v)
            result = log_point(v[0])[0], log_point(v[1])[1]
            df = div((one, one), v); dd = neg(div((one, one), square(v)))
            positive_power_chain = False
        elif kind == 'pow':
            p = arg
            if p == -1:
                good &= (v[0] > 0) | (v[1] < 0)
                v = tuple(torch.where(good, x, 1.) for x in v)
                result = div((one, one), v)
                df = neg(square(result)); dd = mul((2*one, 2*one), mul(square(result), result))
            elif p == 2:
                result = square(v); df = mul((2*one, 2*one), v); dd = 2*one, 2*one
            else:
                good &= v[0] > 0
                v = tuple(torch.where(good, x, 1.) for x in v)
                pp = tuple(one*x for x in enclosure(p))
                result = real_power(v, pp)
                df = div(mul(pp, result), v)
                dd = div(mul(tuple(one*x for x in enclosure(p*(p-1))), result), square(v))
            power *= p
        else:
            raise ValueError(kind)
        second = add(mul(dd, square(first)), mul(df, second))
        first = mul(df, first); v = result
    convex = second[0] >= 0
    if positive_power_chain and (power <= 0 or power >= 1):
        convex |= interval[0] > 0
    return v, first, good & convex


@dataclass
class Term:
    leaf: int
    sign: int
    columns: list
    coefficients: list
    constant: Fraction
    chain: tuple


def recognize(cg, base, *, include_squares=False):
    from nlp_solver.polynomial_graph import _Fold
    fold = _Fold(cg.net)
    loc = {('input', j): j for j in range(cg.n)}
    loc.update({(nd.name, k): cg.n+nd.s+k for nd in cg.nodes for k in range(nd.E)})
    index = {int(j): k for k, j in enumerate(base.keep.cpu().tolist())}
    atoms = {}
    for nd in cg.nodes:
        node = cg.net[nd.name]; kind = type(node).__name__
        op = {'BoundExp': ('exp', 0), 'BoundLog': ('log', 0),
              'BoundReciprocal': ('pow', -1), 'BoundPow': ('pow', getattr(node, 'exponent', 1)),
              'BoundSqr': ('pow', 2)}.get(kind)
        if op is None: continue
        args = fold.array(node.inputs[0]).reshape(-1)
        for k in range(nd.E):
            col = fold.col_index.get((nd.name, k))
            if col is not None:
                atoms[col] = args[k if len(args) > 1 else 0], (op[0], Fraction(float(op[1])), Fraction(0))
    def trace(col, depth=0):
        if depth >= 8: return None
        operand, step = atoms[col]
        if operand.t is None: return None
        if len(operand.t) == 1:
            parent, scale = next(iter(operand.t.items()))
            if parent in atoms:
                inner = trace(parent, depth+1)
                if inner is not None:
                    aff, chain = inner
                    return aff, chain+(('affine', Fraction(scale), Fraction(operand.k)), step)
        return operand, (step,)
    terms = []
    for col, (_, step) in atoms.items():
        leaf = loc.get(fold.cols[col]); weight = float(cg.CM0[leaf]) if leaf is not None else 0.
        if not weight or leaf not in index or (not include_squares and step[:2] == ('pow', Fraction(2))): continue
        result = trace(col)
        if result is None: continue
        aff, chain = result
        if any(loc.get(fold.cols[j]) not in index for j in aff.t): continue
        if not all(math.isfinite(float(q)) for q in (*aff.t.values(), aff.k)): continue
        sign = 1 if weight > 0 else -1
        if sign < 0: chain += (('affine', Fraction(-1), Fraction(0)),)
        terms.append(Term(index[leaf], sign, [index[loc[fold.cols[j]]] for j in aff.t],
                          list(map(Fraction, aff.t.values())), Fraction(aff.k), chain))
    return terms


class Epigraph:
    capacity = 32

    def __init__(self, cg, base, *, include_squares=False):
        self.base = base
        self.terms = recognize(cg, base, include_squares=include_squares)
        self.first = base.m+len(self.terms)
        self.columns = list(range(base.n, base.n+len(self.terms)))
        r, j = base.row.tolist(), base.col.tolist()
        self.slices = []
        for t, term in enumerate(self.terms):
            # sign*leaf <= epigraph: retains the original atom relaxation.
            r += [base.m+t]*2; j += [term.leaf, base.n+t]
        for t, term in enumerate(self.terms):
            start = len(r)
            for slot in range(self.capacity):
                r += [self.first+t*self.capacity+slot]*(len(term.columns)+1)
                j += term.columns+[base.n+t]
            self.slices.append(slice(start, len(r)))
        self.row = base.row.new_tensor(r); self.col = base.col.new_tensor(j)
        self.m = self.first+len(self.terms)*self.capacity
        self.n = base.n+len(self.terms)
        from .curvature_qp import check_capacity
        check_capacity('epigraph', self.m, self.n, len(r), device=base.values.device)

    @staticmethod
    def clean(data):
        from nlp_solver.rigorous_dual import Rows
        rows, c, c0, h, l, u, q = data
        active = torch.isfinite(h)
        rr = Rows(rows.row, rows.col, torch.where(active[:, rows.row], rows.val, 0.), rows.m, rows.n)
        return rr, c, c0, torch.where(active, h, 0.), l, u, q

    def build(self, data):
        from nlp_solver.rigorous_dual import Rows
        rows, c, c0, h, l, u, q = data
        B, T = len(c), len(self.terms)
        if not T: return data
        values = c.new_zeros(B, len(self.row)); values[:, :len(rows.row)] = rows.val
        cc, qq = torch.cat((c.clone(), c.new_zeros(B, T)), 1), torch.cat((q, q.new_zeros(B, T)), 1)
        ll, uu = [torch.cat((v, v.new_zeros(B, T)), 1) for v in (l, u)]
        hh = torch.cat((h, h.new_zeros(B, T), h.new_full((B, T*self.capacity), math.inf)), 1)
        for t, term in enumerate(self.terms):
            # |c|*t >= c*leaf only when c agrees with the row's sign. The built c
            # can differ from the graph's CM0 sign (a chosen square adds its
            # expanded linear term to the same column); such lanes keep c.
            agree = term.sign*c[:, term.leaf] >= 0
            cc[:, self.base.n+t] = torch.where(agree, c[:, term.leaf].abs(), 0.)
            cc[:, term.leaf] = torch.where(agree, 0., c[:, term.leaf])
            a, b = term.sign*l[:, term.leaf], term.sign*u[:, term.leaf]
            ll[:, self.base.n+t], uu[:, self.base.n+t] = torch.minimum(a, b), torch.maximum(a, b)
            pos = len(rows.row)+2*t
            values[:, pos], values[:, pos+1] = term.sign, -1.
        rr = Rows(self.row, self.col, values, self.m, self.n)
        rr.epigraph_used = 0
        return rr, cc, c0, hh, ll, uu, qq

    def add(self, data, z):
        """One fresh point per term/lane; invalid cuts leave their slots empty."""
        rows, _, _, h, l, u, _ = data
        slot = rows.epigraph_used
        if slot >= self.capacity: return []
        arguments, groups = [], {}
        for t, term in enumerate(self.terms):
            ids = term.columns
            coeff = z.new_tensor([float(q) for q in term.coefficients])
            err = z.new_tensor([float(abs(Fraction(float(q))-q)) for q in term.coefficients])
            err = up(err)
            b = z.new_tensor(float(term.constant)); be = up(z.new_tensor(float(abs(Fraction(float(term.constant))-term.constant))))
            # Directed sums avoid a fitted relative margin (including subnormals).
            qlo = down(b-be); qhi = up(b+be); error = be.expand(len(z)).clone()
            point = b.expand(len(z)).clone()
            for k, j in enumerate(ids):
                value = mul((coeff[k], coeff[k]), (l[:, j], u[:, j]))
                reach = torch.maximum(l[:, j].abs(), u[:, j].abs())
                ee = up(err[k]*reach)
                qlo = down(qlo+down(value[0]-ee)); qhi = up(qhi+up(value[1]+ee))
                error = up(error+ee)
                point += coeff[k]*z[:, j]
            point = torch.maximum(qlo, torch.minimum(qhi, point))
            arguments.append((coeff, b, error, qlo, qhi, point))
            groups.setdefault(term.chain, []).append(t)
        evaluated = {}
        # Share elementary interval series over both domain/point and all
        # terms of the same chain. Every tensor also carries the box batch.
        for chain, ids in groups.items():
            qlo, qhi, point = [torch.stack([arguments[t][k] for t in ids], 1) for k in (3, 4, 5)]
            value, derivative, good = evaluate(chain, (torch.stack((qlo, point)), torch.stack((qhi, point))))
            for k, t in enumerate(ids):
                evaluated[t] = value[0][1, :, k], (derivative[0][1, :, k], derivative[1][1, :, k]), good[0, :, k]
        changed = []
        for t, term in enumerate(self.terms):
            ids = term.columns
            coeff, b, error, qlo, qhi, point = arguments[t]
            value, derivative, valid = evaluated[t]
            slope = .5*derivative[0]+.5*derivative[1]
            delta = up(torch.maximum((slope-derivative[0]).abs(), (slope-derivative[1]).abs()))
            reach = up(torch.maximum(up(qhi-point).abs(), down(qlo-point).abs()))
            # Exact cut s*(a*z+b) - t <= s*p - phi(p); every
            # rounded multiplication/sum is enclosed with interval arithmetic.
            rhs = add(mul((slope, slope), (point, point)), neg((value, value)))[1]
            rhs = add((rhs, rhs), neg(mul((slope, slope), (b, b))))[1]
            rhs = up(up(rhs+up(delta*reach))+up(slope.abs()*error))
            vals = slope[:, None]*coeff[None]
            for k, j in enumerate(ids):
                exact = mul((slope, slope), (coeff[k], coeff[k]))
                ce = up(torch.maximum((vals[:, k]-exact[0]).abs(), (vals[:, k]-exact[1]).abs()))
                rhs = up(rhs+up(ce*torch.maximum(l[:, j].abs(), u[:, j].abs())))
            valid &= torch.isfinite(rhs) & torch.isfinite(vals).all(1) & (qlo <= qhi)
            row = self.first+t*self.capacity+slot
            start = self.slices[t].start+slot*(len(ids)+1)
            rows.val[:, start:start+len(ids)] = torch.where(valid[:, None], vals, 0.)
            rows.val[:, start+len(ids)] = torch.where(valid, -1., 0.)
            h[:, row] = torch.where(valid, rhs, math.inf)
            changed.append(row)
        rows.epigraph_used += 1
        return changed


class EpigraphQP:
    """Same GraphQP interface, with immutable extra columns and row indices."""
    def __init__(self, cg, *, square_elements=None):
        from .curvature_qp import GraphQP
        self.base = GraphQP(cg, square_elements=square_elements)
        self.epigraph = Epigraph(cg, self.base, include_squares=square_elements is not None and len(square_elements)==0)
        for name in ('keep', 'chosen', 'virtual', 'cg', 'destination', 'scales', 'shifts'):
            setattr(self, name, getattr(self.base, name))
        self.n, self.m = self.epigraph.n, self.epigraph.m
        self.row, self.col = self.epigraph.row, self.epigraph.col

    def build(self, d):
        return self.epigraph.build(self.base.build(d))

    def expand(self, z):
        return self.base.expand(z[:, :self.base.n])


def objective_solve(qp, data, *, tau, **kwargs):
    """Bounded Kelley loop; retain physical (z,y), reset omega/averages."""
    from .restarted_qp import solve
    ring = qp.epigraph
    if not ring.terms: return solve(*ring.clean(data), **kwargs)
    kwargs = dict(kwargs)
    warm = ((data[4]+data[5])/2, torch.zeros_like(data[3]))
    trace, metrics, best, best_y = [], None, None, None
    for rd in range(ring.capacity+1):
        result = solve(*ring.clean(data), warm=warm, **kwargs)
        if best is None:
            best, best_y = result.lower.clone(), result.best_y.clone()
            metrics = dict(result.metrics)
        else:
            use = result.lower > best
            best_y = torch.where(use[:, None], result.best_y, best_y)
            for key in ('iterations', 'setup_seconds', 'capture_seconds', 'iteration_seconds'):
                metrics[key] += result.metrics[key]
        gain = result.lower-best
        best = torch.maximum(best, result.lower)
        trace.append(dict(ring=rd, lower=best.tolist(), gain=gain.tolist()))
        stop = kwargs.get('stop'); deadline = kwargs.get('deadline')
        if (rd == ring.capacity or (rd > 0 and bool((gain < tau).all()))
                or (stop is not None and bool(((best >= stop) | result.empty).all()))
                or (deadline is not None and time.time() >= deadline)):
            break
        changed = ring.add(data, result.z)
        warm = result.z, result.y.clone()
        warm[1][:, changed] = 0.
    metrics['kelley_rings'] = len(trace)-1
    metrics['ms_per_iteration'] = 1000*metrics['iteration_seconds']/max(1, metrics['iterations'])
    result.lower, result.best_y, result.metrics = best, best_y, metrics
    result.kelley_trace = trace
    result.trace[-1].update(iteration=metrics['iterations'], best=best.clone())
    return result
