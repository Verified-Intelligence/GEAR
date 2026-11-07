"""Offline oracle for the row facts of nlp_solver.structure.build_structure on every benchmark instance: evaluate
the model at seeded random box points (the box corners included) and check that each claimed fact holds at every
sample: box containment, redundant, infeasible, duplicate/opposite (row = scale * row_j + offset), alias
(row = alpha * objective + beta) and lambda_min (the midpoint identity of a quadratic).

Usage: python tests/structure_audit.py --list sweep_order.tsv --out audit.json [--samples N] [--limit N]
       [--only name,name] [--config yaml] [--benchmark-root dir]
Run from complete_verifier (CPU only). A sampled fact can only be refuted, never proved: 0 violations is
necessary, not sufficient. Not a pytest module.
"""
import argparse
import json
import os
import signal
import sys
import traceback
import warnings

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import structure_instances as si  # noqa: E402
import torch  # noqa: E402

REL_TOL = 1e-9
INSTANCE_LIMIT = 300


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


signal.signal(signal.SIGALRM, _alarm)


def sample_points(d, count, seed=1):
    """count box points: the lower and upper corners twice each, then uniform; a dimension without finite
    bounds stays at the root center. count must be even (the points are used in pairs)."""
    L, U, x0, n = d['L'], d['U'], d['x0'], d['n']
    finite = torch.isfinite(L) & torch.isfinite(U)
    g = torch.Generator().manual_seed(seed)
    pts = torch.where(finite, L + (U - L) * torch.rand(count, n, dtype=torch.float64, generator=g), x0.expand(count, n))
    pts[:2] = torch.where(finite, L, x0[0])
    pts[2:4] = torch.where(finite, U, x0[0])
    return pts


def evaluate(model, pts):
    """model rows at each point; a model that does not broadcast a constant output over the batch is evaluated
    point by point"""
    with torch.no_grad():
        try:
            return model(pts).numpy()
        except RuntimeError:
            return torch.cat([model(p[None]) for p in pts]).numpy()


def violations(rep, Y, Ymid, A, D2, thr):
    """messages for every claimed fact of the report that a sample refutes. Y: row values at the points,
    Ymid: row values at the midpoints of consecutive point pairs, A: (2, pairs, rows) values at the pair endpoints,
    D2: (pairs, n) squared coordinate differences of each pair"""
    out = []
    obj = Y[:, 0]
    for r, I in enumerate(rep.rows):
        v = Y[:, r]
        v = v[np.isfinite(v)]                 # a row undefined at a sample (log of a negative) says nothing there
        if not len(v):
            continue
        tol = REL_TOL * (1 + np.abs(v).max())
        if I.box is not None and (v.min() < I.box[0] - tol or v.max() > I.box[1] + tol):
            out.append(f'box row {r} {I.box} but samples span [{v.min()}, {v.max()}]')
        if I.redundant and (v > thr[r] + tol).any():
            out.append(f'redundant row {r}: value {v.max()} > rhs {thr[r]}')
        if I.infeasible and (v <= thr[r] - tol).any():
            out.append(f'infeasible row {r}: value {v.min()} <= rhs {thr[r]}')
        for kind, link in (('duplicate', I.duplicate_of), ('opposite', I.opposite_of)):
            if link is not None:
                j, scale, offset = link
                fit = float(scale) * Y[:, j] + float(offset)
                err = np.abs(Y[:, r] - fit)
                if np.nanmax(err) > REL_TOL * (1 + np.nanmax(np.abs(fit))):
                    out.append(f'{kind} row {r} of row {j}: scale {scale} offset {offset} off by {np.nanmax(err)}')
        if I.alias is not None:
            alpha, beta = float(I.alias[0]), float(I.alias[1])
            fit = alpha * obj + beta
            if np.nanmax(np.abs(Y[:, r] - fit)) > REL_TOL * (1 + np.nanmax(np.abs(fit))):
                out.append(f'alias row {r}: {I.alias} off by {np.nanmax(np.abs(Y[:, r] - fit))}')
        if I.lambda_min is not None:
            # a quadratic: (f(a) + f(b)) / 2 - f(mid) = (b - a)' H (b - a) / 8 >= lambda_min * |b - a|^2 / 8,
            # where H and the distance live on the row's support
            B = D2[:, sorted(I.support)].sum(1)
            gap = (A[0, :, r] + A[1, :, r]) / 2 - Ymid[:, r]
            slack = gap - I.lambda_min * B / 8
            tol_mid = REL_TOL * (1 + np.nanmax(np.abs(A[:, :, r])) + abs(I.lambda_min) * np.nanmax(B))
            if np.nanmin(slack) < -tol_mid:
                out.append(f'lambda_min row {r}: {I.lambda_min} refuted by {np.nanmin(slack)}')
            if I.curvature == 'concave' and np.nanmax(gap) > tol_mid:
                out.append(f'concave row {r} refuted by {np.nanmax(gap)}')
    return out


def audit_one(bench, name, bound_opts, root, samples):
    from nlp_solver.structure import build_structure
    d = si.load(bench, name, bound_opts, root)
    bm = d['bm']
    bm(d['x0'])
    rep = build_structure(bm, d['x0'], d['C'], d['rhs'], d['L'].reshape(1, -1), d['U'].reshape(1, -1))
    if rep.skipped:
        return dict(skipped=rep.skipped)
    pts = sample_points(d, samples)
    a, b = pts[::2], pts[1::2]
    Y, Ymid = evaluate(d['model'], pts), evaluate(d['model'], (a + b) / 2)
    A = np.stack([Y[::2], Y[1::2]])
    D2 = ((a - b) ** 2).numpy()
    thr = rep.pool.thresholds(d['rhs'])[0].numpy().reshape(-1)
    facts = dict(boxes=sum(I.box is not None for I in rep.rows), redundant=sum(I.redundant for I in rep.rows),
                 infeasible=sum(I.infeasible for I in rep.rows),
                 duplicates=sum(I.duplicate_of is not None for I in rep.rows),
                 opposites=sum(I.opposite_of is not None for I in rep.rows),
                 aliases=sum(I.alias is not None for I in rep.rows),
                 lambda_min=sum(I.lambda_min is not None for I in rep.rows))
    return dict(rows=len(rep.rows), facts=facts, violations=violations(rep, Y, Ymid, A, D2, thr))


def summary(res):
    ok = {k: v for k, v in res.items() if 'error' not in v and not v.get('skipped')}
    bad = {k: v for k, v in ok.items() if v['violations']}
    total = {}
    for v in ok.values():
        for key, c in v['facts'].items():
            total[key] = total.get(key, 0) + c
    lines = [f'instances: {len(res)}  audited: {len(ok)}  errors: {sum("error" in v for v in res.values())}  '
             f'skipped: {sum(bool(v.get("skipped")) for v in res.values())}',
             'facts checked: ' + ', '.join(f'{k} {c}' for k, c in total.items()),
             f'violations: {sum(len(v["violations"]) for v in ok.values())} in {len(bad)} instances']
    lines += [f'  {k}: {m}' for k, v in bad.items() for m in v['violations'][:5]]
    lines += [f'  ERROR {k}: {v["error"][:150]}' for k, v in res.items() if 'error' in v]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    si.add_arguments(ap)
    ap.add_argument('--out', required=True)
    ap.add_argument('--samples', type=int, default=200)
    a = ap.parse_args()
    bound_opts, inst = si.read_instances(a)
    res = {}
    for bench, name in inst:
        signal.alarm(INSTANCE_LIMIT)
        try:
            res[name] = audit_one(bench, name, bound_opts, a.benchmark_root, a.samples)
        except Exception as e:
            res[name] = dict(error=type(e).__name__ + ': ' + str(e)[:200] + traceback.format_exc()[-300:])
        finally:
            signal.alarm(0)
        s = res[name]
        print(bench, name, s.get('error', '')[:100] if 'error' in s else s.get('skipped') or
              ('OK' if not s['violations'] else f'{len(s["violations"])} VIOLATIONS'), flush=True)
        with open(a.out, 'w') as f:
            json.dump(res, f, default=str)
    print('\n==== SUMMARY ====')
    print(summary(res))
    sys.exit(1 if any(v.get('violations') for v in res.values()) else 0)


if __name__ == '__main__':
    main()
