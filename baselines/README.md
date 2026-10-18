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

## Setup

- 180 s wall-clock limit per instance.
- 16 logical CPUs per solver.
  - Gurobi, Xpress and SCIP solve one instance at a time with 16 threads; SCIP runs in concurrent mode.
  - All other solvers run 16 instances in parallel with one thread each.
- Solver tolerances: feasibility 1e-6, relative gap 1e-4, absolute gap 1e-5.
- One configuration per solver, chosen on `configs/subset40.txt`.

## Solved

- Feasibility is checked by re-evaluating the returned point in AMPL. The maximum violation over all constraints and variable bounds must be at most 1e-6.
- **Global solvers** (BARON, Couenne, SCIP, LindoGlobal, Gurobi, Xpress): solved if the point is feasible and the solver's primal and dual bounds satisfy P - D <= max(1e-5, 1e-4 * max(1, |P|, |D|)).
  - Convex runs that finish without a branch-and-bound log are certified by their final objective: Gurobi `Optimal objective` (no `Solving non-convex`), Xpress `Final objective` or `Final MIQP objective (global optimum)`, and BARON problems solved during preprocessing (objective from the driver line).
  - A global solver's `infeasible` verdict is credited as solved when the instance is confirmed infeasible: no run of any solver returned a point feasible at 1e-6, and at least two global solvers report infeasible within 180 s (only `gams_world/mosel_start_folioqpgraph`). Local solvers get no credit for infeasibility.
  - The certificate is not counted if the solver returns AMPL status `solved?` (for example BARON's "Globality is therefore not guaranteed"), or if any run found a feasible point better than the certified bound.
- **Local solvers** (LGO, Knitro, IPOPT, SNOPT, MINOS, CONOPT): solved if the point is feasible and its objective P is within the same tolerance of a dual bound D certified by a global solver.

## Results

| Solver | gams_world | minlplib_linear | Total | % |
|---|---|---|---|---|
| SCIP | 201 | 155 | 356 | 70.5 |
| Gurobi | 227 | 174 | 401 | 79.4 |
| BARON | 168 | 129 | 297 | 58.8 |
| Couenne | 182 | 100 | 282 | 55.8 |
| LindoGlobal | 164 | 110 | 274 | 54.3 |
| Xpress | 176 | 120 | 296 | 58.6 |
| Knitro | 230 | 170 | 400 | 79.2 |
| IPOPT | 203 | 117 | 320 | 63.4 |
| CONOPT | 204 | 91 | 295 | 58.4 |
| LGO | 206 | 78 | 284 | 56.2 |
| MINOS | 193 | 53 | 246 | 48.7 |
| SNOPT | 182 | 51 | 233 | 46.1 |

426 instances (242 gams_world, 184 minlplib_linear) are solved by at least one global solver.

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
python harness/bl.py drive --config configs/chosen/scip_cfg.json --run final_scip --time-limit 180   # one solver
python harness/final_table.py         # results/final_table.json
python harness/report.py              # results/per_instance_*.tsv, results/report.md
```

`GEAR_BASELINE_RUNS` sets the output directory (default `baselines/runs`).
