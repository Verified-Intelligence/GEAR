# Baseline campaign: competitor solvers on the 505-instance set (180 s, 16 logical CPUs)

Single credited standard: 1e-6 audited feasibility + solver's own P/D with P-D <= max(1e-5, 1e-4*max(1,|P|,|D|)). Counts are *sound* (certificates contradicted by any run's point at the same feasibility level are removed). Local solvers and LGO are never credited with proofs.


## Solved@180s, global solvers, criterion uniform standard (1e-6 feasibility, gap rule)

| Solver | gams_world | minlplib | Solved@180s | Solved % |
|---|---|---|---|---|
| baron | 168 | 129 | 297 | 58.8 |
| couenne | 182 | 100 | 282 | 55.8 |
| scip | 201 | 155 | 356 | 70.5 |
| lindoglobal | 164 | 110 | 274 | 54.3 |
| gurobi | 227 | 174 | 401 | 79.4 |
| xpress | 176 | 120 | 296 | 58.6 |

### Local/heuristic solvers matching a sound certified objective (uniform standard (1e-6 feasibility, gap rule))

| Solver | gams_world | minlplib | Matched@180s | Matched % |
|---|---|---|---|---|
| lgo | 206 | 78 | 284 | 56.2 |
| knitro | 230 | 170 | 400 | 79.2 |
| ipopt | 203 | 117 | 320 | 63.4 |
| snopt | 182 | 51 | 233 | 46.1 |
| minos | 193 | 53 | 246 | 48.7 |
| conopt | 204 | 91 | 295 | 58.4 |

## Union sizes

| Set | size | gams_world | minlplib |
|---|---|---|---|
| all global solvers | 426 | 242 | 184 |
| all global + local-matched | 426 | 242 | 184 |
| (extra, not credited) strict 1e-8/1e-6-gap global | 349 | | |
| (diagnostic) raw unfiltered global union | 432 | | |

## Chosen configurations and tuning evidence

Subset: 40 instances (configs/subset40.txt); score = sound (a) proofs, ties by (b), then wall.


### baron: **b3_nobt** (layout 16 jobs x 1 threads)

options: `maxtime={time} threads=1 outlev=1 epsr=1e-4 epsa=1e-5`  native: `'AbsConFeasTol 1e-6\nNumLoc 20\nOBTTDo 0\nLBTTDo 0'`  notes: b1 + LP-based/optimality-based tightening off (intensity down)

| config | a_sound | b_sound | a_raw | b_raw | contra_a | contra_b |
|---|---|---|---|---|---|---|
| b0_std | 15 | 18 | 15 | 18 | 0 | 0 |
| b1_loc20 | 15 | 18 | 15 | 18 | 0 | 0 |
| b2_probe | 17 | 19 | 17 | 19 | 0 | 0 |
| b3_nobt | 16 | 20 | 16 | 20 | 0 | 0 |
| b4_nodesel | 17 | 19 | 17 | 19 | 0 | 0 |

### couenne: **c0_std** (layout 16 jobs x 1 threads)

options: `bonmin.time_limit={time} outlev=1`  native: `'feas_tolerance 1e-6\nbonmin.allowable_fraction_gap 1e-4\nbonmin.allowable_gap 1e-5'`  notes: standard tolerances (paper rule: OBBT/aggr FBBT off above 100 vars)

| config | a_sound | b_sound | a_raw | b_raw | contra_a | contra_b |
|---|---|---|---|---|---|---|
| c0_std | 7 | 11 | 7 | 11 | 0 | 0 |
| c1_lo4 | 7 | 10 | 7 | 10 | 0 | 0 |
| c2_bt_off | 8 | 10 | 8 | 10 | 0 | 0 |
| c3_bt_on | 7 | 10 | 7 | 10 | 0 | 0 |
| c4_branch | 7 | 10 | 7 | 10 | 0 | 0 |

### gurobi: **g0_std** (layout 1 jobs x 16 threads)

options: `lim:time={time} tech:threads={threads} tech:outlev=1 mip:gap=1e-4 mip:gapabs=1e-5 alg:feastol=1e-6`  native: `''`  notes: standard tolerances, automatic function handling, 16 threads

| config | a_sound | b_sound | a_raw | b_raw | contra_a | contra_b |
|---|---|---|---|---|---|---|
| g0_std | 9 | 22 | 10 | 23 | 1 | 1 |
| g1_nonlin | 9 | 22 | 10 | 23 | 1 | 1 |
| g2_focus3 | 9 | 22 | 10 | 23 | 1 | 1 |
| g3_pre_cuts | 9 | 21 | 10 | 22 | 1 | 1 |
| g4_obbt | 8 | 22 | 9 | 23 | 1 | 1 |
| g5_heur | 8 | 22 | 9 | 23 | 1 | 1 |
| g0_std_L2x8 | 8 | 21 | 9 | 22 | 1 | 1 |
| g0_std_L4x4 | 7 | 20 | 8 | 21 | 1 | 1 |

### knitro: **knitro_final** (layout 16 jobs x 1 threads)

options: `maxtime={time} numthreads={threads} nlp_algorithm=4 ms_enable=1 ms_numthreads={threads} ms_deterministic=0 ms_maxsolves=200 ms_terminate=0 concurrent_evals=1 soltype=1 outlev=2 timing=1 feastol=1e-6 feastol_abs=1e-6`  native: `''`  notes: prior tuned (SQP + multistart), feastol 1e-6; 16 serial jobs


### lindoglobal: **l0_std** (layout 16 jobs x 1 threads)

options: `maxtime={time} threads={threads} outlev=1 epsr=1e-4 epsa=1e-5`  native: `''`  notes: standard gaps, single thread (driver exposes no other tuning options)

| config | a_sound | b_sound | a_raw | b_raw | contra_a | contra_b |
|---|---|---|---|---|---|---|
| l0_std | 9 | 13 | 10 | 13 | 1 | 0 |

### scip: **s2_obbt** (layout 1 jobs x 16 threads)

options: `lim:time={time} lp:threads=1 tech:outlev=1 tech:timing=2 alg:concurrent=1 par:minnthreads={threads} par:maxnthreads={threads} par:mode=0`  native: `'numerics/feastol = 1e-6\nlimits/gap = 1e-4\nlimits/absgap = 1e-5\nseparating/rlt/freq = 10\nseparating/rlt/detecthidden = TRUE\nseparating/rlt/onlyoriginal = FALSE\npropagating/obbt/onlynonconvexvars = FALSE\npropagating/obbt/freq = 10'`  notes: s1 + OBBT in the tree (bound tightening up)

| config | a_sound | b_sound | a_raw | b_raw | contra_a | contra_b |
|---|---|---|---|---|---|---|
| s0_std | 2 | 19 | 2 | 19 | 0 | 0 |
| s1_rlt | 4 | 19 | 5 | 19 | 1 | 0 |
| s2_obbt | 4 | 19 | 4 | 19 | 0 | 0 |
| s3_heur | 3 | 17 | 3 | 17 | 0 | 0 |
| s4_branch | 1 | 19 | 1 | 19 | 0 | 0 |
| s5_portfolio | 2 | 19 | 2 | 19 | 0 | 0 |
| s2_obbt_L2x8 | 2 | 19 | 2 | 19 | 0 | 0 |
| s2_obbt_L4x4 | 3 | 19 | 3 | 19 | 0 | 0 |

### xpress: **x3_cuts** (layout 1 jobs x 16 threads)

options: `lim:time={time} tech:threads={threads} tech:outlev=1 mip:gap=1e-4 mip:gapabs=1e-5 alg:feastol=1e-6 alg:nlpsolver=2 cut:rltcuts=1 global:xprs_globalspatialbranchcuttingeffort=0.5`  native: `''`  notes: x1 + RLT cuts and more branching cuts

| config | a_sound | b_sound | a_raw | b_raw | contra_a | contra_b |
|---|---|---|---|---|---|---|
| x0_std | 7 | 13 | 7 | 15 | 0 | 2 |
| x1_global | 7 | 13 | 7 | 15 | 0 | 2 |
| x2_obbt3 | 6 | 12 | 6 | 14 | 0 | 2 |
| x3_cuts | 7 | 13 | 7 | 15 | 0 | 2 |
| x4_heur | 6 | 12 | 6 | 13 | 0 | 1 |
| x5_nobt | 7 | 13 | 7 | 15 | 0 | 2 |
| x3_cuts_L2x8 | 7 | 12 | 7 | 14 | 0 | 2 |
| x3_cuts_L4x4 | 6 | 13 | 6 | 15 | 0 | 2 |

### Full tuning log (every run)

See results/tuning_log.csv.

