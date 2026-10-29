#!/usr/bin/env python3
"""Build per-instance tables, unions and results/report.md."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import bl_eval as ev

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
FAMS = ["gams_world", "minlplib_linear"]
NFAM = {"gams_world": 265, "minlplib_linear": 240}


def main():
    all_rows, final = [], {}
    for s in ev.GLOBAL + ev.LOCAL:
        if (ev.RUNS / ev.final_run(s) / "logs" / s).exists():
            final[s] = ev.eval_run(ev.final_run(s), s)
    pool = []
    for run in ev.RUNS.glob("*"):
        if run.name.startswith(("tune_",)):
            sv = next(iter((run / "logs").glob("*")), None)
            if sv:
                pool += ev.eval_run(run.name, sv.name)
    for rows in final.values():
        pool += rows
    for run, s in ev.REPLACED:
        if (ev.RUNS / run / "logs" / s).exists():
            pool += ev.eval_run(run, s)
    best = ev.best_known(pool)
    for s in ev.GLOBAL:
        if s in final:
            ev.mark_sound(final[s], best)
    infeas = ev.credit_infeasible(final, pool)

    # ---- certified references per instance and criterion
    ref = {"a": defaultdict(list), "b": defaultdict(list)}
    dref = defaultdict(list)          # certified dual bounds (strict local crediting)
    cert = {"a": defaultdict(set), "b": defaultdict(set)}
    for s in ev.GLOBAL:
        for r in final.get(s, []):
            k = (r["family"], r["instance"])
            for lvl in "ab":
                if r[f"cert_{lvl}"]:
                    cert[lvl][k].add(s)
                    ref[lvl][k].append(r["P_ampl"])
                    if lvl == "b" and r.get("D_solver") is not None:
                        dref[k].append(r["D_solver"])
    # ---- local matching
    match = {"a": defaultdict(set), "b": defaultdict(set)}
    for s in ev.LOCAL:
        for r in final.get(s, []):
            k = (r["family"], r["instance"])
            sg = ev.sign(r["sense"])
            for lvl, feas, tf in (("a", "feas8", ev.tol_a), ("b", "feas6", ev.tol_b)):
                if not r[feas] or r["P_ampl"] is None:
                    continue
                if lvl == "b":
                    # strict variant (a): -tol <= sign*(P - D) <= tol against the tightest certified dual
                    Ds = list(dref.get(k, []))
                    if Ds:
                        D = max(Ds) if sg > 0 else min(Ds)
                        if -tf(r["P_ampl"], D) <= sg * (r["P_ampl"] - D) <= tf(r["P_ampl"], D):
                            match[lvl][k].add(s)
                    continue
                refs = list(ref[lvl].get(k, []))
                if any(abs(r["P_ampl"] - x) <= tf(r["P_ampl"], x) for x in refs):
                    match[lvl][k].add(s)

    insts = ev.load_manifest()
    keys = [(i.family, i.name) for i in insts]
    # ---- per-instance outputs
    with (RES / "per_instance_long.tsv").open("w") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["solver", "kind", "family", "instance", "criterion_a", "criterion_b",
                    "a_raw", "b_raw", "local_match_a", "local_match_b", "wall_time",
                    "P_ampl", "P_solver", "D_solver", "max_violation", "feas8", "feas6",
                    "solve_result", "runner_status"])
        for s, rows in final.items():
            kind = "global" if s in ev.GLOBAL else "local"
            for r in rows:
                k = (r["family"], r["instance"])
                w.writerow([s, kind, *k,
                            int(bool(r.get("cert_a"))), int(bool(r.get("cert_b"))),
                            int(r["cert_a_raw"]), int(r["cert_b_raw"]),
                            int(s in match["a"].get(k, ())), int(s in match["b"].get(k, ())),
                            r["wall_time"], r["P_ampl"], r["P_solver"], r["D_solver"],
                            r["max_violation"], int(r["feas8"]), int(r["feas6"]),
                            r["solve_result"], r["runner_status"]])

    ug = {l: {k for k in keys if cert[l].get(k)} for l in "ab"}
    ul = {l: ug[l] | {k for k in keys if match[l].get(k)} for l in "ab"}
    with (RES / "per_instance_wide.tsv").open("w") as fh:
        w = csv.writer(fh, delimiter="\t")
        sv = list(final)
        w.writerow(["family", "instance"] + [f"{s}_{l}" for s in sv for l in "ab"]
                   + ["union_global_a", "union_global_b", "union_all_a", "union_all_b"])
        for k in keys:
            row = []
            for s in sv:
                for l in "ab":
                    if s in ev.GLOBAL:
                        v = next((r.get(f"cert_{l}") for r in final[s] if (r["family"], r["instance"]) == k), 0)
                    else:
                        v = s in match[l].get(k, ())
                    row.append(int(bool(v)))
            w.writerow([*k] + row + [int(k in ug["a"]), int(k in ug["b"]),
                                     int(k in ul["a"]), int(k in ul["b"])])

    out = []
    P = out.append
    P("# Baseline campaign: competitor solvers on the 505-instance set (180 s, 16 logical CPUs)\n")
    P("Single credited standard: 1e-6 audited feasibility + solver's own P/D with relative gap "
      "(P-D)/max(1,|P|,|D|) <= 1e-4. Counts are *sound* (certificates contradicted by any "
      "run's point at the same feasibility level are removed). Local solvers and LGO are never "
      "credited with proofs.\n")
    for lvl, name in (("b", "uniform standard (1e-6 feasibility, gap rule)"),):
        P(f"\n## Solved@180s, global solvers, criterion {name}\n")
        P("| Solver | gams_world | minlplib | Solved@180s | Solved % |\n|---|---|---|---|---|")
        for s in ev.GLOBAL:
            if s not in final:
                P(f"| {s} | n/a | n/a | n/a | n/a |")
                continue
            c = Counter(r["family"] for r in final[s] if r[f"cert_{lvl}"])
            n = c["gams_world"] + c["minlplib_linear"]
            P(f"| {s} | {c['gams_world']} | {c['minlplib_linear']} | {n} | {100 * n / 505:.1f} |")
        P(f"\n### Local/heuristic solvers matching a sound certified objective ({name})\n")
        P("| Solver | gams_world | minlplib | Matched@180s | Matched % |\n|---|---|---|---|---|")
        for s in ev.LOCAL:
            if s not in final:
                P(f"| {s} | n/a | n/a | n/a | n/a |")
                continue
            c = Counter(k[0] for k, v in match[lvl].items() if s in v)
            n = c["gams_world"] + c["minlplib_linear"]
            P(f"| {s} | {c['gams_world']} | {c['minlplib_linear']} | {n} | {100 * n / 505:.1f} |")
    P("\n## Union sizes\n")
    P("| Set | size | gams_world | minlplib |\n|---|---|---|---|")
    for nm, st in (("all global solvers", ug["b"]), ("all global + local-matched", ul["b"])):
        P(f"| {nm} | {len(st)} | {sum(k[0] == 'gams_world' for k in st)} | {sum(k[0] == 'minlplib_linear' for k in st)} |")
    P(f"| (extra, not credited) strict 1e-8/1e-6-gap global | {len(ug['a'])} | | |")
    raw = {l: sum(1 for k in keys if any(r[f"cert_{l}_raw"] for s in ev.GLOBAL
                   for r in final.get(s, []) if (r["family"], r["instance"]) == k)) for l in "ab"}
    P(f"| (diagnostic) raw unfiltered global union | {raw['b']} | | |")

    P("\n## Chosen configurations and tuning evidence\n")
    P("Subset: 40 instances (configs/subset40.txt); score = sound (a) proofs, ties by (b), then wall.\n")
    for f in sorted((ROOT / "configs" / "chosen").glob("*.json")):
        if f.name.endswith("_cfg.json"):
            continue
        c = json.loads(f.read_text())
        P(f"\n### {c['solver']}: **{c['name']}** (layout {c['jobs']} jobs x {c['threads']} threads)\n")
        P(f"options: `{c['options']}`  native: `{c['native'].strip()!r}`  notes: {c['notes']}\n")
        if c["solver"] in ev.FINAL_RUN:
            fc = json.loads((ROOT / "configs" / "chosen" / f"{c['solver']}_cfg.json").read_text())
            P(f"final run `{ev.FINAL_RUN[c['solver']]}` uses **{fc['name']}** (`configs/chosen/{c['solver']}_cfg.json`): {fc['notes']}; "
              "found on the full benchmark after tuning (harness/bl_eval.py, FINAL_RUN)\n")
        t = c.get("tuned", {})
        if "all" in t:
            P("| config | a_sound | b_sound | a_raw | b_raw | contra_a | contra_b |\n|---|---|---|---|---|---|---|")
            for a in t["all"]:
                P(f"| {a['config']} | {a['a_sound']} | {a['b_sound']} | {a['a_raw']} | {a['b_raw']} | {a['contra_a']} | {a['contra_b']} |")
    if (RES / "tuning_log.csv").exists():
        P("\n### Full tuning log (every run)\n\nSee results/tuning_log.csv.\n")
    (RES / "report.md").write_text("\n".join(out) + "\n")
    print("\n".join(out[:60]))


if __name__ == "__main__":
    main()
