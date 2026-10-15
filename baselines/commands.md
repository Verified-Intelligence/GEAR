# Exact per-solver commands

All arms: 180 s wall clock, 16 logical CPUs, no other jobs running. `{time}` = 180 and `{threads}` = the configured thread count are substituted by the harness. Each solver is invoked through its AMPL driver (`ampl-module-<solver>`); the harness writes the option string and (where present) native settings file into `<run>/config/`. Final configurations: `configs/chosen/<solver>_cfg.json` (tuning evidence: `configs/chosen/<solver>.json`, `results/tuning_log.csv`).

Runner (any solver S): `python harness/bl.py drive --config configs/chosen/S_cfg.json --run final_S --time-limit 180` (add `GEAR_BASELINE_RUNS=<dir>` to choose the output directory). The full pipeline (tuning + final run) is `python harness/campaign.py S`.

## baron

- config: `b3_nobt`; layout: 1 thread per solve, 16 instances concurrently (jobs=16, threads=1)
- AMPL option string (`option baron_options`): `maxtime={time} threads=1 outlev=1 epsr=1e-4 epsa=1e-5`
- native settings file (read through `optfile=<run>/config/native.txt`):
```
AbsConFeasTol 1e-6
NumLoc 20
OBTTDo 0
LBTTDo 0
```
- process environment: `{"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}`
- command: `python harness/bl.py drive --config configs/chosen/baron_cfg.json --run final_baron --time-limit 180`

## couenne

- config: `c0_std`; layout: 1 thread per solve, 16 instances concurrently (jobs=16, threads=1)
- AMPL option string (`option couenne_options`): `bonmin.time_limit={time} outlev=1`
- native settings file (appended to the Couenne option file):
```
feas_tolerance 1e-6
bonmin.allowable_fraction_gap 1e-4
bonmin.allowable_gap 1e-5
```
- process environment: `{"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}`
- command: `python harness/bl.py drive --config configs/chosen/couenne_cfg.json --run final_couenne --time-limit 180`

## scip

- config: `s2_obbt`; layout: 1 instance at a time, 16 threads, concurrent mode (`alg:concurrent=1`, 16-way opportunistic solve) (jobs=1, threads=16)
- AMPL option string (`option scip_options`): `lim:time={time} lp:threads=1 tech:outlev=1 tech:timing=2 alg:concurrent=1 par:minnthreads={threads} par:maxnthreads={threads} par:mode=0`
- native settings file (read through `tech:param:read=<run>/config/native.txt`):
```
numerics/feastol = 1e-6
limits/gap = 1e-4
limits/absgap = 1e-5
separating/rlt/freq = 10
separating/rlt/detecthidden = TRUE
separating/rlt/onlyoriginal = FALSE
propagating/obbt/onlynonconvexvars = FALSE
propagating/obbt/freq = 10
```
- process environment: `{"OMP_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "16", "MKL_NUM_THREADS": "16", "BLIS_NUM_THREADS": "16", "NUMEXPR_NUM_THREADS": "16"}`
- command: `python harness/bl.py drive --config configs/chosen/scip_cfg.json --run final_scip --time-limit 180`

## lindoglobal

- config: `l0_std`; layout: 1 thread per solve, 16 instances concurrently (jobs=16, threads=1)
- AMPL option string (`option lindoglobal_options`): `maxtime={time} threads={threads} outlev=1 epsr=1e-4 epsa=1e-5`
- process environment: `{"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}`
- command: `python harness/bl.py drive --config configs/chosen/lindoglobal_cfg.json --run final_lindoglobal --time-limit 180`

## gurobi

- config: `g0_std`; layout: 1 instance at a time, 16 threads (jobs=1, threads=16)
- AMPL option string (`option gurobi_options`): `lim:time={time} tech:threads={threads} tech:outlev=1 mip:gap=1e-4 mip:gapabs=1e-5 alg:feastol=1e-6`
- process environment: `{"OMP_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "16", "MKL_NUM_THREADS": "16", "BLIS_NUM_THREADS": "16", "NUMEXPR_NUM_THREADS": "16"}`
- command: `python harness/bl.py drive --config configs/chosen/gurobi_cfg.json --run final_gurobi --time-limit 180`

## xpress

- config: `x3_cuts`; layout: 1 instance at a time, 16 threads (jobs=1, threads=16)
- AMPL option string (`option xpress_options`): `lim:time={time} tech:threads={threads} tech:outlev=1 mip:gap=1e-4 mip:gapabs=1e-5 alg:feastol=1e-6 alg:nlpsolver=2 cut:rltcuts=1 global:xprs_globalspatialbranchcuttingeffort=0.5`
- process environment: `{"OMP_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "16", "MKL_NUM_THREADS": "16", "BLIS_NUM_THREADS": "16", "NUMEXPR_NUM_THREADS": "16"}`
- command: `python harness/bl.py drive --config configs/chosen/xpress_cfg.json --run final_xpress --time-limit 180`

## lgo

- config: `lgo_final`; layout: 1 thread per solve, 16 instances concurrently (jobs=16, threads=1)
- AMPL option string (`option lgo_options`): `timelim={time} outlev=1 timing=1 g_maxfct=10000000 maxnosuc=10000000 penmult=100 con_tol=1e-6`
- process environment: `{"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}`
- command: `python harness/bl.py drive --config configs/chosen/lgo_cfg.json --run final_lgo --time-limit 180`

## knitro

- config: `knitro_final`; layout: 1 thread per solve, 16 instances concurrently (jobs=16, threads=1)
- AMPL option string (`option knitro_options`): `maxtime={time} numthreads={threads} nlp_algorithm=4 ms_enable=1 ms_numthreads={threads} ms_deterministic=0 ms_maxsolves=200 ms_terminate=0 concurrent_evals=1 soltype=1 outlev=2 timing=1 feastol=1e-6 feastol_abs=1e-6`
- process environment: `{"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}`
- command: `python harness/bl.py drive --config configs/chosen/knitro_cfg.json --run final_knitro --time-limit 180`

## ipopt

- config: `ipopt_final`; layout: 1 thread per solve, 16 instances concurrently (jobs=16, threads=1)
- AMPL option string (`option ipopt_options`): `max_wall_time={time} print_level=5 timing_statistics=yes honor_original_bounds=yes constr_viol_tol=1e-6 acceptable_constr_viol_tol=1e-6 bound_relax_factor=0 max_iter=100000`
- process environment: `{"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}`
- command: `python harness/bl.py drive --config configs/chosen/ipopt_cfg.json --run final_ipopt --time-limit 180`

## snopt

- config: `snopt_final`; layout: 1 thread per solve, 16 instances concurrently (jobs=16, threads=1)
- AMPL option string (`option snopt_options`): `outlev=1 timing=1 major_feasibility_tolerance=1e-6 minor_feasibility_tolerance=1e-6`
- process environment: `{"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}`
- command: `python harness/bl.py drive --config configs/chosen/snopt_cfg.json --run final_snopt --time-limit 180`

## minos

- config: `minos_final`; layout: 1 thread per solve, 16 instances concurrently (jobs=16, threads=1)
- AMPL option string (`option minos_options`): `timelim={time} outlev=1 timing=1 feasibility_tolerance=1e-6 row_tolerance=1e-6 major_iterations=1000 minor_iterations=1000 iterations_limit=10000000 superbasics_limit=10000 scale_option=0 completion=full`
- process environment: `{"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}`
- command: `python harness/bl.py drive --config configs/chosen/minos_cfg.json --run final_minos --time-limit 180`

## conopt

- config: `conopt_default`; layout: 1 thread per solve, 16 instances concurrently (jobs=16, threads=1)
- AMPL option string (`option conopt_options`): `maxftime={time}`
- process environment: `{"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}`
- command: `python harness/bl.py drive --config configs/chosen/conopt_cfg.json --run final_conopt --time-limit 180`
