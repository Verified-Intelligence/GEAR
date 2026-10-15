"""Final baseline table (16 cores, 180 s, 1e-6 feasibility, gap rule), strict local crediting (a),
and the global union. Writes results/final_table.json."""
import bl_eval as ev, collections, json
from collections import defaultdict
final = {s: ev.eval_run(f"final_{s}", s) for s in ev.GLOBAL + ev.LOCAL if (ev.RUNS / f"final_{s}" / "logs" / s).exists()}
pool = []
for run in ev.RUNS.glob("tune_*"):
    sv = next(iter((run / "logs").glob("*")), None)
    if sv: pool += ev.eval_run(run.name, sv.name)
for rows in final.values(): pool += rows
best = ev.best_known(pool)
cert_by = defaultdict(set); Dref = defaultdict(list); counts = {}
for s in ev.GLOBAL:
    ev.mark_sound(final[s], best)
    counts[s] = collections.Counter(r['family'] for r in final[s] if r['cert_b'])
    for r in final[s]:
        if r['cert_b']:
            k = (r['family'], r['instance']); cert_by[k].add(s)
            if r.get('D_solver') is not None: Dref[k].append(r['D_solver'])
local_by = defaultdict(set)
for s in ev.LOCAL:
    c = collections.Counter()
    for r in final[s]:
        if not r['feas6'] or r['P_ampl'] is None: continue
        k = (r['family'], r['instance']); P = r['P_ampl']; sg = ev.sign(r['sense'])
        Ds = list(Dref.get(k, []))
        if Ds:
            D = max(Ds) if sg > 0 else min(Ds)
            if -ev.tol_b(P, D) <= sg * (P - D) <= ev.tol_b(P, D):
                c[r['family']] += 1; local_by[k].add(s)
    counts[s] = c
U_glob = set(cert_by); U_all = U_glob | set(local_by)
res = {s: {'gams': c.get('gams_world', 0), 'minlp': c.get('minlplib_linear', 0)} for s, c in counts.items()}
for s in res: res[s]['total'] = res[s]['gams'] + res[s]['minlp']
summary = {'counts': res, 'U_global': len(U_glob), 'U_global_by_family': dict(collections.Counter(f for f, _ in U_glob)),
           'U_all': len(U_all)}
json.dump(summary, open(ev.ROOT / 'results' / 'final_table.json', 'w'), indent=1)
for s, v in res.items(): print(f"{s:12s} {v['gams']:3d} {v['minlp']:3d} {v['total']:3d} {100*v['total']/505:.1f}")
print({k: v for k, v in summary.items() if k != 'counts'})
