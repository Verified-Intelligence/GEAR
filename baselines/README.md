# AMPL baseline solvers on 505 global-optimization instances

12 AMPL solvers on 505 continuous instances: 265 `gams_world` and 240 `minlplib_linear`.

## Contents

| Path | Content |
|---|---|
| `benchmark/` | the 505 AMPL models and `manifest.tsv` (family, instance, path, sha256, size, sense) |
| `configs/candidates/` | candidate configurations evaluated during tuning |
| `configs/chosen/` | the configuration used for each solver, with its tuning scores |
| `configs/subset40.txt` | the 40-instance tuning subset |
| `commands.md` | exact AMPL options, thread layout and command per solver |
| `harness/` | runner, tuning campaign and scoring scripts |
| `results/` | per-instance results, final table, tuning log |
| `../logs/` | solver logs of the final runs (see the top-level README) |

## Setup

- 180 s wall-clock limit per instance.
- 16 logical CPUs per solver.
  - Gurobi, Xpress and SCIP solve one instance at a time with 16 threads; SCIP runs in concurrent mode.
  - All other solvers run 16 instances in parallel with one thread each.
- Solver tolerances: feasibility 1e-6, relative gap 1e-4, absolute gap 1e-5.
- One configuration per solver, chosen on `configs/subset40.txt`. SCIP's final runs use `s2_obbt_pow1`, the tuned `s2_obbt` with powers of sums kept
  unexpanded (`expr/pow/expandmaxexponent = 1`): this setting was found on the full benchmark after tuning, certifies 8 more instances and
  loses none, so it favours SCIP (`harness/bl_eval.py`, `FINAL_RUN`; the earlier `s2_obbt` run stays in the pool of contradicting points).

## Solved

- Feasibility is checked by re-evaluating the returned point in AMPL. The maximum violation over all constraints and variable bounds must be at most 1e-6.
- **Global solvers** (BARON, Couenne, SCIP, LindoGlobal, Gurobi, Xpress): solved if the point is feasible and the solver's primal and dual bounds have relative gap (P - D) / max(1, |P|, |D|) <= 1e-4.
  - Convex runs that finish without a branch-and-bound log are certified by their final objective: Gurobi `Optimal objective` (no `Solving non-convex`), Xpress `Final objective` or `Final MIQP objective (global optimum)`, and BARON problems solved during preprocessing (objective from the driver line).
  - A global solver's `infeasible` verdict is credited as solved when the instance is confirmed infeasible: no run of any solver returned a point feasible at 1e-6, and at least two global solvers report infeasible within 180 s (only `gams_world/mosel_start_folioqpgraph`). Local solvers get no credit for infeasibility.
  - The certificate is not counted if the solver returns AMPL status `solved?` (for example BARON's "Globality is therefore not guaranteed"), or if any run found a feasible point better than the certified bound.
- **Local solvers** (LGO, Knitro, IPOPT, SNOPT, MINOS, CONOPT): solved if the point is feasible and its objective P is within the same tolerance of a dual bound D certified by a global solver.

## Results

| Solver | gams_world | minlplib_linear | Total | % |
|---|---|---|---|---|
| SCIP | 209 | 155 | 364 | 72.1 |
| Gurobi | 227 | 174 | 401 | 79.4 |
| BARON | 168 | 129 | 297 | 58.8 |
| Couenne | 182 | 100 | 282 | 55.8 |
| LindoGlobal | 164 | 110 | 274 | 54.3 |
| Xpress | 176 | 120 | 296 | 58.6 |
| Knitro | 231 | 170 | 401 | 79.4 |
| IPOPT | 203 | 117 | 320 | 63.4 |
| CONOPT | 205 | 91 | 296 | 58.6 |
| LGO | 206 | 78 | 284 | 56.2 |
| MINOS | 193 | 53 | 246 | 48.7 |
| SNOPT | 182 | 51 | 233 | 46.1 |

427 instances (243 gams_world, 184 minlplib_linear) are solved by at least one global solver.

## Versions

| Solver | Version |
|---|---|
| BARON | 25.12.10 |
| Couenne | 0.5.8 |
| SCIP | 10.0.2 |
| LindoGlobal | 16.0.7099.148 |
| Gurobi | 13.0.3 |
| Xpress | 9.8.1 |
| LGO | 2015-01-17 |
| Knitro | 15.1.0 |
| IPOPT | 3.14.19 |
| SNOPT | 7.5-1.2 |
| MINOS | 5.51 |
| CONOPT | 4.38.1 |

## Reproduce

```bash
pip install amplpy
python -m amplpy.modules install base coin baron knitro lgo lindoglobal minos scip snopt gurobi xpress conopt
python -m amplpy.modules activate <license-uuid>

cd baselines
bash harness/run_campaign.sh          # tuning on subset40, then final runs for all solvers
python harness/bl.py drive --config configs/candidates/scip/s2_obbt_pow1.json --run final_scip_pow1 --time-limit 180   # SCIP's final run
python harness/final_table.py         # results/final_table.json
python harness/report.py              # results/per_instance_*.tsv, results/report.md
```

`GEAR_BASELINE_RUNS` sets the output directory (default `baselines/runs`).
