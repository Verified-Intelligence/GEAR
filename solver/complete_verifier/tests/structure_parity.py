"""Offline parity of nlp_solver.structure.build_structure with the existing detectors on every benchmark instance.

Usage: python tests/structure_parity.py --list sweep_order.tsv --out full.json [--limit N] [--only name,name]
[--config yaml] [--benchmark-root dir]
Run from complete_verifier (CPU only). Not a pytest module.
"""
import argparse
import collections
import json
import os
import signal
import sys
import time
import traceback
import warnings

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import structure_instances as si  # noqa: E402
import torch  # noqa: E402

INF = float('inf')
DETECTOR_LIMIT = 60


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


signal.signal(signal.SIGALRM, _alarm)


def guarded(fn):
    """(result, seconds, error name); an exception or timeout counts as a decline."""
    signal.alarm(DETECTOR_LIMIT)
    t = time.perf_counter()
    try:
        return fn(), time.perf_counter() - t, None
    except Timeout:
        return None, time.perf_counter() - t, 'Timeout'
    except Exception as e:
        return None, time.perf_counter() - t, type(e).__name__ + ': ' + str(e)[:100]
    finally:
        signal.alarm(0)


def run_one(bench, name, bound_opts, root):
    from nlp_solver.structure import build_structure
    d = si.load(bench, name, bound_opts, root)
    bm, model, x0, C, rhs, n = d['bm'], d['model'], d['x0'], d['C'], d['rhs'], d['n']
    lo, hi = d['L'], d['U']
    lo_t, hi_t = lo.reshape(1, -1), hi.reshape(1, -1)
    bm(x0)
    t = time.perf_counter()
    rep = build_structure(bm, x0, C, rhs, lo_t, hi_t)
    secs = time.perf_counter() - t
    rows, obj = rep.rows, rep.objective
    out = dict(bench=bench, n=n, nrows=len(rows), seconds=secs, spent=rep.spent, skipped=rep.skipped,
               obj=dict(cls=obj.cls, tags=sorted(obj.tags), reason=obj.reason, inexact=obj.inexact),
               budget_rows=sum(1 for r in rows if r.reason == 'budget'),
               reasons=dict(collections.Counter(r.reason for r in rows if r.reason)))
    if rep.skipped:
        return out
    # 1. degrees
    from nlp_solver.quadratic_tangent import output_degrees
    deg, t, err = guarded(lambda: output_degrees(bm, x0))
    o = dict(t=t, err=err)
    if deg is not None:
        deg = deg.numpy()
        mine = np.array([r.degree if r.cls in ('const', 'affine', 'quad', 'poly') else INF for r in rows])
        th, mi = deg <= 2, mine <= 2
        both = th & mi
        o.update(n_theirs_le2=int(th.sum()), n_mine_le2=int(mi.sum()), n_both=int(both.sum()),
                 n_differ=int((both & (deg != mine)).sum()), differ=[int(i) for i in np.flatnonzero(both & (deg != mine))][:10],
                 theirs_only=[int(i) for i in np.flatnonzero(th & ~mi)][:10], n_theirs_only=int((th & ~mi).sum()),
                 theirs_only_why=dict(collections.Counter(f'{rows[i].cls}|{rows[i].reason}|{rows[i].inexact}' for i in np.flatnonzero(th & ~mi))),
                 n_mine_only=int((~th & mi).sum()), mine_only=[int(i) for i in np.flatnonzero(~th & mi)][:10])
    out['degrees'] = o
    # 2. GP
    from nlp_solver.gp_certificate import build_gp_certificate
    diag = {}
    res, t, err = guarded(lambda: build_gp_certificate(bm, model, x0, C, rhs, lo_t, hi_t, diag, rowpool=rep.pool))
    pos = bool((lo > 0).all() and torch.isfinite(hi).all())
    pred = bool(pos and (obj.gp or obj.cls == 'const'))
    out['gp'] = dict(t=t, err=err, accepted=res is not None, reason=diag.get('reason'), pred=pred,
                     rows_not_gp=sum(1 for r in rows[1:] if r.cls not in ('const',) and not r.gp))
    # 3. exp
    from nlp_solver.exp_certificate import build_exp_certificate
    diag = {}
    res, t, err = guarded(lambda: build_exp_certificate(bm, model, x0, C, rhs, lo_t, hi_t, diag, rowpool=rep.pool))
    out['exp'] = dict(t=t, err=err, accepted=res is not None, reason=diag.get('reason'), pred='exp' in obj.tags)
    # 4. convex objective
    from nlp_solver.convex_objective import recognize_convex_objective
    res, t, err = guarded(lambda: recognize_convex_objective(bm, x0, C, lo_t, hi_t))
    pred = bool(('sepatoms' in obj.tags and obj.atom_kinds <= {'exp', 'log', 'pow'} and obj.atom_deg is not None
                 and obj.atom_deg <= 2) or (obj.sepuni and obj.cls in ('poly', 'laurent', 'frac')))
    out['convobj'] = dict(t=t, err=err, accepted=res is not None, pred=pred)
    # 5. rational rows
    from nlp_solver.rational_rows import extract_rational_rows
    pts = lo_t + (hi_t - lo_t) * torch.rand(4, n, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    rr, t, err = guarded(lambda: extract_rational_rows(bm, model, x0, C, pts))
    o = dict(t=t, err=err)
    if rr is not None:
        th = np.array([f.terms is not None for f in rr])
        mi = np.array([r.cls in ('laurent', 'frac', 'poly', 'quad', 'affine', 'const') for r in rows])
        o.update(n_theirs=int(th.sum()), n_mine=int(mi.sum()), n_theirs_only=int((th & ~mi).sum()),
                 theirs_only=[int(i) for i in np.flatnonzero(th & ~mi)][:10], n_mine_only=int((~th & mi).sum()))
    out['rational'] = o
    # 6. objective aliases
    from auto_LiRPA.presolve import prove_rows
    pr, t, err = guarded(lambda: prove_rows(bm, C[0]))
    o = dict(t=t, err=err)
    if pr is not None:
        theirs = sorted(i for i, _, _ in pr[1])
        mine = sorted(i for i, r in enumerate(rows) if r.alias is not None)
        o.update(theirs=theirs[:20], mine=mine[:20], theirs_only=sorted(set(theirs) - set(mine))[:20],
                 mine_only=sorted(set(mine) - set(theirs))[:20])
    out['alias'] = o
    return out


def disagreements(res):
    """(instance, detector, kind, detail) for every disagreement."""
    out = []
    for k, v in res.items():
        if 'error' in v:
            out.append((k, 'build_structure', 'exception', v['error']))
            continue
        if v.get('skipped'):
            out.append((k, 'build_structure', 'skipped', v['skipped']))
            continue
        for key in ('gp', 'exp', 'convobj'):
            a, p = v[key]['accepted'], v[key]['pred']
            if a and not p:
                out.append((k, key, 'report declines what the detector accepts', v[key].get('reason')))
            elif p and not a:
                out.append((k, key, 'report predicts accept, detector declines (informational)', v[key].get('err') or v[key].get('reason')))
        d = v['degrees']
        if d.get('n_differ'):
            out.append((k, 'degrees', 'degree differs', d['differ']))
        if d.get('n_theirs_only'):
            out.append((k, 'degrees', 'detector has degree<=2, report does not', d['theirs_only_why']))
        if d.get('n_mine_only'):
            out.append((k, 'degrees', 'report has degree<=2, detector does not', d['mine_only']))
        r = v['rational']
        if r.get('n_theirs_only'):
            out.append((k, 'rational', 'detector has terms, report does not', r['theirs_only']))
        a = v['alias']
        if a.get('theirs_only'):
            out.append((k, 'alias', 'detector alias missing in report', a['theirs_only']))
        if a.get('mine_only'):
            out.append((k, 'alias', 'extra alias in report (informational)', a['mine_only']))
        if v['budget_rows'] and not v['reasons'].get('budget'):
            out.append((k, 'budget', 'inconsistent', v['reasons']))
    return out


def summary(res):
    ok = {k: v for k, v in res.items() if 'error' not in v and not v.get('skipped')}
    L = []
    L.append(f'instances: {len(res)}  analysed: {len(ok)}  errors: {sum(1 for v in res.values() if "error" in v)}  skipped: {sum(1 for v in res.values() if v.get("skipped"))}')
    deg = [v['degrees'] for v in ok.values() if 'n_both' in v['degrees']]
    L.append(f'degrees: rows compared (both <=2) {sum(d["n_both"] for d in deg)}, differ {sum(d["n_differ"] for d in deg)}, '
             f'detector-only {sum(d["n_theirs_only"] for d in deg)}, report-only {sum(d["n_mine_only"] for d in deg)}, '
             f'detector errors {sum(1 for v in ok.values() if v["degrees"].get("err"))}')
    for key in ('gp', 'exp', 'convobj'):
        c = collections.Counter((v[key]['accepted'], v[key]['pred']) for v in ok.values())
        L.append(f'{key}: detector accepts {c[(True, True)] + c[(True, False)]}, of which report predicts {c[(True, True)]}, '
                 f'report declines {c[(True, False)]}; report predicts but detector declines {c[(False, True)]}; '
                 f'detector errors {sum(1 for v in ok.values() if v[key]["err"])}')
    r = [v['rational'] for v in ok.values() if 'n_theirs' in v['rational']]
    L.append(f'rational rows: detector {sum(x["n_theirs"] for x in r)}, report {sum(x["n_mine"] for x in r)}, '
             f'detector-only {sum(x["n_theirs_only"] for x in r)}, report-only {sum(x["n_mine_only"] for x in r)}')
    a = [v['alias'] for v in ok.values() if 'theirs' in v['alias']]
    L.append(f'objective aliases: detector-only {sum(len(x["theirs_only"]) for x in a)}, '
             f'report-only {sum(len(x["mine_only"]) for x in a)} in {sum(1 for x in a if x["mine_only"])} instances')
    t = np.array([v['seconds'] for v in ok.values()]) if ok else np.zeros(1)
    mx = max(ok, key=lambda k: ok[k]['seconds']) if ok else None
    L.append('analysis seconds: median %.4f p90 %.4f p99 %.4f max %.3f (%s)' % (np.median(t), np.percentile(t, 90), np.percentile(t, 99), t.max(), mx))
    L.append(f'operation units spent: max {max((v["spent"] for v in ok.values()), default=0)}; '
             f'instances with budget rows {sum(1 for v in ok.values() if v["budget_rows"])}')
    dis = disagreements(res)
    hard = [x for x in dis if 'informational' not in x[2]]
    L.append(f'disagreements: {len(hard)} (plus {len(dis) - len(hard)} informational)')
    for x in dis:
        L.append('  %s | %s | %s | %s' % (x[0], x[1], x[2], str(x[3])[:150]))
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    si.add_arguments(ap)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    bound_opts, inst = si.read_instances(a)
    res = {}
    for bench, name in inst:
        signal.alarm(900)
        try:
            res[name] = run_one(bench, name, bound_opts, a.benchmark_root)
        except Exception as e:
            res[name] = dict(bench=bench, error=type(e).__name__ + ': ' + str(e)[:200] + traceback.format_exc()[-300:])
        finally:
            signal.alarm(0)
        s = res[name]
        print(bench, name, s.get('error', '')[:100] if 'error' in s else '%.3fs' % s['seconds'], flush=True)
        with open(a.out, 'w') as f:
            json.dump(res, f, default=str)
    print('\n==== SUMMARY ====')
    print(summary(res))


if __name__ == '__main__':
    main()
